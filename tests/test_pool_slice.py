from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dae.instructions import (
    CommunicationInstruction,
    PoolSliceGather,
    PoolSlicePublish,
    PoolSliceReturn,
)
from dae.pool_slice import (
    POOL_SLICE_CONFIG_BYTES,
    POOL_SLICE_PROFILE_DONE,
    POOL_SLICE_PROFILE_DATA_PUBLISHED,
    POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED,
    POOL_SLICE_PROFILE_FIRST_PAYLOAD,
    POOL_SLICE_PROFILE_GATHER_READY,
    POOL_SLICE_PROFILE_METADATA_CLOSED,
    POOL_SLICE_PROFILE_PAYLOAD_DONE,
    POOL_SLICE_PROFILE_START,
    POOL_SLICE_PUBLISH_BYTES,
    POOL_SLICE_RECEIVE_BYTES,
    PoolSliceBatchFlags,
    PoolSliceConfig,
    PoolSliceFlags,
    PoolSliceProgram,
    PoolSlicePublishBatch,
    PoolSliceReceiveBatch,
    group_routes_by_reader,
)
from dae.runtime import comm_opcode


ROOT = Path(__file__).resolve().parents[1]


def _config(**updates) -> PoolSliceConfig:
    values = dict(
        source_address=1,
        token_pool_address=2,
        expert_input_address=3,
        expert_output_address=4,
        return_inbox_address=5,
        returned_address=6,
        send_offsets_address=7,
        send_rows_address=8,
        send_origin_rows_address=9,
        send_batches_address=10,
        receive_batches_address=11,
        offsets_inbox_address=12,
        rows_inbox_address=13,
        receive_routes_address=15,
        reader_tails_address=16,
        sequence_address=17,
        group_ready_address=18,
        control_address=19,
        row_bytes=1024,
        source_stride=1024,
        pool_stride=1024,
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
        queue_signal_base=0,
        data_signal_base=2,
        return_signal_base=4,
        signal_count=6,
        return_capacity_rows=8,
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

    assert len(publish.pack()) == POOL_SLICE_PUBLISH_BYTES
    assert len(receive.pack()) == POOL_SLICE_RECEIVE_BYTES
    assert PoolSlicePublishBatch.unpack(publish.pack()) == publish
    assert PoolSliceReceiveBatch.unpack(receive.pack()) == receive


def test_pool_slice_config_abi_and_ranges():
    assert len(_config().pack()) == POOL_SLICE_CONFIG_BYTES

    with pytest.raises(ValueError, match="data signal range"):
        _config(data_signal_base=5).pack()
    with pytest.raises(ValueError, match="signal ranges overlap"):
        _config(data_signal_base=1).pack()
    with pytest.raises(ValueError, match="row_bytes"):
        _config(row_bytes=1008).pack()
    with pytest.raises(ValueError, match="expert_stride"):
        _config(expert_stride=15 * 1024).pack()
    with pytest.raises(ValueError, match="route_capacity"):
        _config(active_rows=9).pack()
    with pytest.raises(ValueError, match="PE range"):
        _config(my_pe=2).pack()
    assert len(_config(flags=PoolSliceFlags.STREAMING_GATHER).pack()) == (
        POOL_SLICE_CONFIG_BYTES
    )
    assert len(
        _config(
            flags=PoolSliceFlags.STREAMING_GATHER,
            data_stages=2,
            early_ready_rows=4,
        ).pack()
    ) == POOL_SLICE_CONFIG_BYTES
    with pytest.raises(ValueError, match="strict row prefix"):
        _config(
            flags=PoolSliceFlags.STREAMING_GATHER,
            data_stages=2,
        ).pack()
    with pytest.raises(ValueError, match="unsupported"):
        _config(flags=2).pack()


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
    # With two readers per PE, either pool slice can fetch one contiguous
    # offsets span and then only its own route-row subspan.
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


def test_pool_slice_instructions_are_communication_domain_operators():
    address = 0x123456789ABCDEF0
    publish = PoolSlicePublish(address)
    gather = PoolSliceGather(
        address, write_barrier=7, dispatch_barrier_base=11
    )
    returned = PoolSliceReturn(address, compute_barrier_base=19)

    assert _fields(publish)[:4] == [
        comm_opcode.COMM_POOL_SLICE_PUBLISH,
        0,
        0,
        0,
    ]
    assert _fields(gather)[:4] == [
        comm_opcode.COMM_POOL_SLICE_GATHER,
        7,
        11,
        0,
    ]
    assert _fields(returned)[:4] == [
        comm_opcode.COMM_POOL_SLICE_RETURN,
        19,
        0,
        0,
    ]
    for instruction in (publish, gather, returned):
        assert isinstance(instruction, CommunicationInstruction)
        assert instruction.requires_signal_array


def test_pool_slice_timing_uses_only_vdcores_internal_events():
    profile = torch.zeros((3, 128), dtype=torch.uint64)
    profile[0, POOL_SLICE_PROFILE_START] = 100
    profile[0, POOL_SLICE_PROFILE_GATHER_READY] = 170
    profile[0, POOL_SLICE_PROFILE_DONE] = 260
    program = PoolSliceProgram(
        launcher=SimpleNamespace(profile=profile),
        write_barrier=0,
        dispatch_barriers=(),
        compute_barriers=(),
        chunk_rows=1,
        streaming_gather=True,
    )

    assert program.timing_ns() == (70, 90, 160)
    profile[0, POOL_SLICE_PROFILE_DATA_PUBLISHED] = 120
    profile[0, POOL_SLICE_PROFILE_FIRST_DATA_PUBLISHED] = 110
    profile[0, POOL_SLICE_PROFILE_FIRST_PAYLOAD] = 130
    profile[0, POOL_SLICE_PROFILE_METADATA_CLOSED] = 145
    profile[0, POOL_SLICE_PROFILE_PAYLOAD_DONE] = 165
    assert program.overlap_timing_ns() == {
        "first_data_published": 10,
        "data_published": 20,
        "first_payload": 30,
        "metadata_closed": 45,
        "payload_done": 65,
        "gather_ready": 70,
    }


def test_pool_program_keeps_all_communication_on_the_pool_block():
    source = _function_source(
        ROOT / "python" / "dae" / "pool_slice.py",
        "build_pool_slice_copy_program",
    )
    assert "pool_builder.add_communication(PoolSlicePublish" in source
    assert "pool_builder.add_communication(\n        PoolSliceGather" in source
    assert "pool_builder.add_communication(\n        PoolSliceReturn" in source
    assert "launcher.builder[1].add_communication(PoolSlicePublish" not in source


def test_pool_mailbox_wait_is_warp_parallel():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "__ballot_sync(0xffffffffU, ready)" in source
    assert "config.queue_signal_base" in source
    assert "lane >= count" in source


def test_streaming_gather_intersects_independent_metadata_and_data_readiness():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "queue_seen_mask" in source
    assert "data_seen_mask" in source
    assert (
        "metadata_ready_mask & data_seen_mask & ~payload_issued_mask" in source
    )
    assert "poolSliceProfileFirstPayload" in source
    assert "peak_inflight_sources" in source
