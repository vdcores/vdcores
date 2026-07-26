from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dae.instructions import CommunicationInstruction, PoolSliceExchange
from dae.pool_slice import (
    POOL_SLICE_CONFIG_BYTES,
    POOL_SLICE_MAX_LOCAL_READERS,
    POOL_SLICE_PROFILE_DONE,
    POOL_SLICE_PROFILE_DATA_PUBLISHED,
    POOL_SLICE_PROFILE_COMPUTE_READY,
    POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED,
    POOL_SLICE_PROFILE_FIRST_PAYLOAD,
    POOL_SLICE_PROFILE_GATHER_READY,
    POOL_SLICE_PROFILE_METADATA_CLOSED,
    POOL_SLICE_PROFILE_PAYLOAD_DONE,
    POOL_SLICE_PROFILE_RETURN_PAYLOAD_DONE,
    POOL_SLICE_PROFILE_RETURN_SIGNALS_CLOSED,
    POOL_SLICE_PROFILE_SCATTER_DONE,
    POOL_SLICE_PROFILE_START,
    POOL_SLICE_PUBLISH_BYTES,
    POOL_SLICE_RECEIVE_BYTES,
    PoolSliceBatchFlags,
    PoolSliceConfig,
    PoolSliceProgram,
    PoolSlicePublishBatch,
    PoolSliceReceiveBatch,
    group_routes_by_reader,
    select_pool_slice_pack_warps,
)
from dae.runtime import comm_opcode


ROOT = Path(__file__).resolve().parents[1]


def _config(**updates) -> PoolSliceConfig:
    values = dict(
        source_address=1,
        token_pool_address=2,
        delivery_pool_address=3,
        expert_input_address=4,
        expert_output_address=5,
        return_inbox_address=6,
        returned_address=7,
        send_offsets_address=8,
        send_rows_address=9,
        send_origin_rows_address=10,
        send_batches_address=11,
        receive_batches_address=12,
        receive_routes_address=13,
        sequence_address=14,
        group_ready_address=15,
        control_address=16,
        row_bytes=1024,
        source_stride=1024,
        pool_stride=1024,
        delivery_stride=1024,
        expert_row_stride=1024,
        return_stride=1024,
        expert_stride=16 * 1024,
        active_rows=8,
        token_capacity=8,
        route_capacity=8,
        expert_capacity_rows=16,
        local_readers=2,
        num_pes=2,
        my_pe=0,
        signal_base=0,
        signal_count=2,
        return_capacity_rows=8,
        pack_warps=2,
        write_chunks=1,
        write_chunk_rows=8,
    )
    values.update(updates)
    return PoolSliceConfig(**values)


def _fields(instruction) -> list[int]:
    return instruction.tensor().view(torch.uint16).tolist()


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    segment = ast.get_source_segment(source, function)
    assert segment is not None
    return segment


def test_pool_slice_batch_abis_round_trip():
    publish = PoolSlicePublishBatch(
        sequence=9,
        source_pe=1,
        target_pe=3,
        active_rows=17,
        route_begin=4,
        route_end=11,
        reader_counts=(2, 5, 0, 0, 0, 0, 0, 0),
        flags=PoolSliceBatchFlags.ERROR,
    )
    receive = PoolSliceReceiveBatch(
        sequence=9,
        base_row=11,
        source_begin=7,
        row_count=4,
        source_pe=1,
        local_reader=2,
        flags=PoolSliceBatchFlags.ERROR,
    )

    assert len(publish.pack()) == POOL_SLICE_PUBLISH_BYTES == 64
    assert len(receive.pack()) == POOL_SLICE_RECEIVE_BYTES == 48
    assert PoolSlicePublishBatch.unpack(publish.pack()) == publish
    assert PoolSliceReceiveBatch.unpack(receive.pack()) == receive


def test_pack_warp_policy_preserves_receive_concurrency():
    assert select_pool_slice_pack_warps(
        num_pes=2, route_capacity=32, row_bytes=8192
    ) == 4
    assert select_pool_slice_pack_warps(
        num_pes=2, route_capacity=128, row_bytes=8192
    ) == 6
    assert select_pool_slice_pack_warps(
        num_pes=4, route_capacity=128, row_bytes=8192
    ) == 4
    assert select_pool_slice_pack_warps(
        num_pes=8, route_capacity=128, row_bytes=8192
    ) == 4
    assert select_pool_slice_pack_warps(
        num_pes=8, route_capacity=128, row_bytes=8192, requested=3
    ) == 3


def test_pool_slice_config_abi_and_ranges():
    assert len(_config().pack()) == POOL_SLICE_CONFIG_BYTES == 208

    with pytest.raises(ValueError, match="signal range"):
        _config(signal_base=1).pack()
    with pytest.raises(ValueError, match="row_bytes"):
        _config(row_bytes=1008).pack()
    with pytest.raises(ValueError, match="delivery rows"):
        _config(delivery_stride=2048).pack()
    with pytest.raises(ValueError, match="expert_stride"):
        _config(expert_stride=15 * 1024).pack()
    with pytest.raises(ValueError, match="route_capacity"):
        _config(active_rows=9).pack()
    with pytest.raises(ValueError, match="PE range"):
        _config(my_pe=2).pack()
    with pytest.raises(ValueError, match="local_readers"):
        _config(local_readers=POOL_SLICE_MAX_LOCAL_READERS + 1).pack()
    with pytest.raises(ValueError, match="receive worker"):
        _config(pack_warps=8).pack()
    with pytest.raises(ValueError, match="write_chunks"):
        _config(write_chunks=2).pack()


def test_group_routes_is_stable_and_slice_offsets_are_composable():
    offsets, rows, origins = group_routes_by_reader(
        [3, 0, 2, 1, 0, 3],
        num_readers=4,
        source_rows=[10, 11, 12, 13, 14, 15],
        origin_rows=[20, 21, 22, 23, 24, 25],
    )

    assert offsets.dtype == torch.uint32
    assert offsets.tolist() == [0, 2, 3, 4, 6]
    assert rows.tolist() == [11, 14, 13, 12, 10, 15]
    assert origins.tolist() == [21, 24, 23, 22, 20, 25]
    assert offsets[:3].tolist() == [0, 2, 3]
    assert offsets[2:].tolist() == [3, 4, 6]


def test_group_routes_supports_topk_and_explicit_zero_route_readers():
    offsets, rows, origins = group_routes_by_reader(
        [0, 2, 0, 2],
        num_readers=4,
        source_rows=[0, 0, 1, 1],
        origin_rows=[0, 1, 2, 3],
    )
    assert offsets.tolist() == [0, 2, 2, 4, 4]
    assert rows.tolist() == [0, 1, 0, 1]
    assert origins.tolist() == [0, 2, 1, 3]

    empty_offsets, empty_rows, empty_origins = group_routes_by_reader(
        [], num_readers=4
    )
    assert empty_offsets.tolist() == [0, 0, 0, 0, 0]
    assert empty_rows.numel() == 0
    assert empty_origins.numel() == 0


def test_pool_slice_exchange_is_a_macro_communication_operator():
    address = 0x123456789ABCDEF0
    exchange = PoolSliceExchange(
        address,
        write_barrier=7,
        dispatch_barrier_base=11,
        compute_barrier_base=19,
    )
    assert _fields(exchange)[:4] == [
        comm_opcode.COMM_POOL_SLICE_EXCHANGE,
        7,
        11,
        19,
    ]
    assert isinstance(exchange, CommunicationInstruction)
    assert exchange.requires_signal_array


def test_pool_slice_timing_uses_only_vdcores_internal_events():
    profile = torch.zeros((3, 128), dtype=torch.uint64)
    profile[1, POOL_SLICE_PROFILE_START] = 100
    profile[1, POOL_SLICE_PROFILE_GATHER_READY] = 170
    profile[1, POOL_SLICE_PROFILE_DONE] = 260
    program = PoolSliceProgram(
        launcher=SimpleNamespace(profile=profile),
        write_barrier=0,
        dispatch_barriers=(),
        compute_barriers=(),
        chunk_rows=1,
        communication_block=1,
    )

    assert program.timing_ns() == (70, 90, 160)
    profile[1, POOL_SLICE_PROFILE_DATA_PUBLISHED] = 120
    profile[1, POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED] = 110
    profile[1, POOL_SLICE_PROFILE_FIRST_PAYLOAD] = 130
    profile[1, POOL_SLICE_PROFILE_METADATA_CLOSED] = 145
    profile[1, POOL_SLICE_PROFILE_PAYLOAD_DONE] = 165
    profile[1, POOL_SLICE_PROFILE_COMPUTE_READY] = 180
    profile[1, POOL_SLICE_PROFILE_RETURN_PAYLOAD_DONE] = 210
    profile[1, POOL_SLICE_PROFILE_RETURN_SIGNALS_CLOSED] = 230
    profile[1, POOL_SLICE_PROFILE_SCATTER_DONE] = 250
    assert program.overlap_timing_ns() == {
        "first_data_published": 10,
        "data_published": 20,
        "first_payload": 30,
        "metadata_closed": 45,
        "payload_done": 65,
        "gather_ready": 70,
        "compute_ready": 80,
        "return_payload_done": 110,
        "return_signals_closed": 130,
        "scatter_done": 150,
    }


def test_pool_program_is_only_vdcores_ops_and_uses_an_isolated_comm_block():
    source = _function_source(
        ROOT / "python" / "dae" / "pool_slice.py",
        "build_pool_slice_copy_program",
    )
    assert "communication_builder.add_communication(" in source
    assert "PoolSliceExchange(" in source
    assert "writer_builder.add_memory(TmaLoad1D" in source
    assert "launcher.builder[local_reader + 2]" in source
    assert "torch.cuda.Stream" not in source
    assert "torch.cuda.Event" not in source


def test_pool_mailbox_scan_is_lane_parallel_and_uses_one_merged_word():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "__ballot_sync(0xffffffffU, observed >= metadata_value)" in source
    assert "__ballot_sync(0xffffffffU, observed >= data_value)" in source
    assert "config.signal_base + lane" in source
    assert "POOL_SLICE_SIGNAL_RETURN" in source


def test_macro_operator_batches_payload_and_uses_cooperative_quiet_per_direction():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "config.delivery_pool_address" in source
    assert "route.row_count) * config.row_bytes" in source
    assert "shared_dispatch_issued" in source
    assert "shared_return_issued" in source
    assert source.count("pool_slice_quiet(config.num_pes, thread_id);") == 2
    assert "if (num_pes <= 2)" in source
    assert "nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>();" in source


def test_runtime_specialization_does_not_change_the_ordinary_role_split():
    source = (ROOT / "include" / "dae" / "dae2.cuh").read_text()
    specialized = source.index("comminsts[0].opcode == COMM_POOL_SLICE_EXCHANGE")
    ordinary = source.index("// start memory and computation execution")
    assert specialized < ordinary
    assert "pool_slice_exchange(" in source[specialized:ordinary]
    context = (ROOT / "include" / "dae" / "context.cuh").read_text()
    assert "static constexpr int numCommunicationWarps = 1;" in context
