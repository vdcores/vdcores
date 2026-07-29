from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dae.instructions import (
    PoolInstruction,
    PoolSliceExchange,
    PoolSliceHostWeightedExchange,
    PoolSliceWeightedExchange,
)
from dae.pool_slice import (
    POOL_SLICE_CONFIG_BYTES,
    POOL_SLICE_HOST_CONFIG_BYTES,
    POOL_SLICE_MAX_LOCAL_READERS,
    POOL_SLICE_MAX_POOL_BLOCKS,
    POOL_SLICE_MAX_STREAM_QUEUES,
    POOL_SLICE_QUEUE_ENTRY_BYTES,
    POOL_SLICE_STREAM_QUEUE_DEPTH,
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
    PoolSliceHostConfig,
    PoolSliceProgram,
    PoolSlicePublishBatch,
    PoolSliceReceiveBatch,
    group_routes_by_reader,
)
from dae.runtime import pool_opcode


ROOT = Path(__file__).resolve().parents[1]


def _config(**updates) -> PoolSliceConfig:
    values = dict(
        combine_rows_address=1,
        token_pool_address=2,
        delivery_pool_address=3,
        expert_input_address=4,
        expert_output_address=5,
        return_inbox_address=6,
        returned_address=7,
        send_offsets_address=8,
        send_rows_address=9,
        send_origin_rows_address=10,
        send_token_rows_address=11,
        send_token_counts_address=12,
        send_batches_address=13,
        receive_batches_address=14,
        receive_routes_address=15,
        sequence_address=16,
        control_address=17,
        row_bytes=1024,
        active_rows=8,
        token_capacity=8,
        route_capacity=8,
        expert_capacity_rows=16,
        local_readers=2,
        num_pes=2,
        my_pe=0,
        signal_base=0,
        group_limit=2,
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
    assert len(receive.pack()) == POOL_SLICE_RECEIVE_BYTES == 32
    assert PoolSlicePublishBatch.unpack(publish.pack()) == publish
    assert PoolSliceReceiveBatch.unpack(receive.pack()) == receive


def test_pool_gather_is_the_only_dispatch_protocol():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "POOL_SLICE_DISPATCH_DIRECT_PUT" not in source
    assert "POOL_SLICE_DISPATCH_BATCHED_PUT" not in source
    assert "pool_slice_direct_put_group(" not in source
    assert "pool_slice_batched_put_group(" not in source


def test_pool_slice_config_abi_and_ranges():
    assert len(_config().pack()) == POOL_SLICE_CONFIG_BYTES == 192

    abi = (ROOT / "include" / "dae" / "pool_slice_abi.cuh").read_text()
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "sizeof(PoolSliceReceiveBatch) == 32" in abi
    assert "sizeof(PoolSliceConfig) == 192" in abi
    assert "pool_slice_valid_config(" not in source
    for removed_field in (
        "pool_stride",
        "delivery_stride",
        "expert_row_stride",
        "return_stride",
        "expert_stride",
        "signal_count",
        "return_capacity_rows",
    ):
        assert removed_field not in abi

    with pytest.raises(ValueError, match="row_bytes"):
        _config(row_bytes=1008).pack()
    with pytest.raises(ValueError, match="route_capacity"):
        _config(active_rows=9).pack()
    with pytest.raises(ValueError, match="PE range"):
        _config(my_pe=2).pack()
    with pytest.raises(ValueError, match="local_readers"):
        _config(local_readers=POOL_SLICE_MAX_LOCAL_READERS + 1).pack()
    assert len(_config(group_limit=8).pack()) == POOL_SLICE_CONFIG_BYTES
    assert len(
        _config(
            group_limit=POOL_SLICE_MAX_POOL_BLOCKS,
            pool_count=2,
            pool_rank=1,
        ).pack()
    ) == POOL_SLICE_CONFIG_BYTES
    with pytest.raises(ValueError, match="protocol limit"):
        _config(
            group_limit=POOL_SLICE_MAX_POOL_BLOCKS + 1,
            pool_count=2,
            pool_rank=1,
        ).pack()
    with pytest.raises(ValueError, match="write_chunks"):
        _config(write_chunks=2).pack()
    with pytest.raises(ValueError, match="pool_rank"):
        _config(pool_rank=1).pack()
    with pytest.raises(ValueError, match="one token-capacity segment"):
        _config(expert_capacity_rows=15).pack()
    assert len(
        _config(
            pool_count=2,
            pool_rank=1,
        ).pack()
    ) == POOL_SLICE_CONFIG_BYTES


def test_pool_slice_host_config_only_extends_the_data_plane():
    host = PoolSliceHostConfig(
        pool=_config(),
        peers_address=0x1000,
        producer_generations_address=0x2000,
        local_lkey=17,
    )
    packed = host.pack()
    assert len(packed) == POOL_SLICE_HOST_CONFIG_BYTES == 224
    assert packed[:POOL_SLICE_CONFIG_BYTES] == _config().pack()

    abi = (ROOT / "include" / "dae" / "pool_slice_abi.cuh").read_text()
    assert "sizeof(PoolSliceHostPeer) == 40" in abi
    assert "sizeof(PoolSliceHostConfig) == 224" in abi


def test_group_routes_is_stable_and_slice_offsets_are_composable():
    offsets, rows, origins, weights = group_routes_by_reader(
        [3, 0, 2, 1, 0, 3],
        num_readers=4,
        source_rows=[10, 11, 12, 13, 14, 15],
        origin_rows=[20, 21, 22, 23, 24, 25],
        route_weights=[1, 2, 3, 4, 5, 6],
    )

    assert offsets.dtype == torch.uint32
    assert offsets.tolist() == [0, 2, 3, 4, 6]
    assert rows.tolist() == [11, 14, 13, 12, 10, 15]
    assert origins.tolist() == [21, 24, 23, 22, 20, 25]
    assert weights.float().tolist() == [2, 5, 4, 3, 1, 6]
    assert offsets[:3].tolist() == [0, 2, 3]
    assert offsets[2:].tolist() == [3, 4, 6]


def test_group_routes_supports_topk_and_explicit_zero_route_readers():
    offsets, rows, origins, weights = group_routes_by_reader(
        [0, 2, 0, 2],
        num_readers=4,
        source_rows=[0, 0, 1, 1],
        origin_rows=[0, 1, 2, 3],
    )
    assert offsets.tolist() == [0, 2, 2, 4, 4]
    assert rows.tolist() == [0, 1, 0, 1]
    assert origins.tolist() == [0, 2, 1, 3]
    assert weights.float().tolist() == [1, 1, 1, 1]

    empty_offsets, empty_rows, empty_origins, empty_weights = group_routes_by_reader(
        [], num_readers=4
    )
    assert empty_offsets.tolist() == [0, 0, 0, 0, 0]
    assert empty_rows.numel() == 0
    assert empty_origins.numel() == 0
    assert empty_weights.numel() == 0


def test_pool_slice_exchange_is_a_separate_macro_pool_instruction():
    address = 0x123456789ABCDEF0
    exchange = PoolSliceExchange(
        address,
        write_barrier=7,
        dispatch_barrier_base=11,
        compute_barrier_base=19,
    )
    assert _fields(exchange)[:4] == [
        pool_opcode.POOL_SLICE_EXCHANGE,
        7,
        11,
        19,
    ]
    assert isinstance(exchange, PoolInstruction)
    assert exchange.requires_signal_array

    weighted = PoolSliceWeightedExchange(
        address,
        write_barrier=7,
        dispatch_barrier_base=11,
        compute_barrier_base=19,
    )
    assert _fields(weighted)[:4] == [
        pool_opcode.POOL_SLICE_WEIGHTED_EXCHANGE,
        7,
        11,
        19,
    ]

    host_weighted = PoolSliceHostWeightedExchange(
        address,
        write_barrier=7,
        dispatch_barrier_base=11,
        compute_barrier_base=19,
    )
    assert _fields(host_weighted)[:4] == [
        pool_opcode.POOL_SLICE_HOST_WEIGHTED_EXCHANGE,
        7,
        11,
        19,
    ]
    assert isinstance(host_weighted, PoolInstruction)
    assert host_weighted.requires_signal_array

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


def test_pool_program_is_only_vdcores_ops_and_uses_an_isolated_pool_block():
    source = _function_source(
        ROOT / "python" / "dae" / "pool_slice.py",
        "build_pool_slice_copy_program",
    )
    assert "pool_builder.add_pool(" in source
    assert "PoolSliceWeightedExchange" in source
    assert "pool_instruction = PoolSliceExchange" in source
    assert "writer_builder.add_memory(TmaLoad1D" in source
    assert "PoolTmaStore1D(" in source
    assert "PoolWaitSignal(" in source
    assert "reader_base = 1 + buffers.pool_count" in source
    assert "torch.cuda.Stream" not in source
    assert "torch.cuda.Event" not in source
    assert "source_preloaded" in source
    assert "launcher.new_bar(0 if source_preloaded else 1)" in source


def test_pool_mailbox_scan_is_lane_parallel_and_uses_one_merged_word():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "__ballot_sync(0xffffffffU, metadata_ready)" in source
    assert "config.signal_base + lane" in source
    assert "poolSliceControlStreamMetadataTransportReady + lane" in source


def test_macro_operator_batches_payload_and_uses_cooperative_quiet_per_direction():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "config.delivery_pool_address" in source
    assert "route.row_count) * config.row_bytes" in source
    assert source.count("pool_slice_quiet_block();") >= 2
    assert "nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>();" in source


def test_pool_gather_is_multi_poolinst_and_uses_generation_slots():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    abi = (ROOT / "include" / "dae" / "pool_slice_abi.cuh").read_text()
    python = (ROOT / "python" / "dae" / "pool_slice.py").read_text()
    assert "POOL_SLICE_DISPATCH_POOL_GATHER" not in source
    assert "pool_slice_replicate_target_shard(" not in source
    assert "token_count > config.token_capacity" in source
    assert "source_row >= config.token_capacity" in source
    assert "pool_slice_stream_gather_rows(" in source
    assert "pool_slice_return_scatter_pipelined(" not in source
    assert "nvshmemx_putmem_signal_nbi_warp(" in source
    assert "poolSliceControlDispatchGeneration" in source
    assert "poolSliceControlReturnGeneration" in source
    assert "poolSliceControlReturnReady" in source
    assert "poolSliceControlScatterGeneration" in source
    assert "poolSliceControlReaderRowCount" in source
    assert "dae_atomic_store_release_gpu(" in source
    assert "dae_atomic_load_acquire_gpu(" in source
    assert "poolSliceMaxPoolBlocks = 32" in abi
    assert "for pool_rank in range(buffers.pool_count)" in python
    assert "POOL_SLICE_CONTROL_DISPATCH_READY = 102" in python


def test_weighted_scatter_bypasses_reduction_for_one_pool_contributor():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    scatter = source.split("pool_slice_weighted_scatter_token(", 1)[1].split(
        "pool_slice_weighted_source_shards(", 1
    )[0]
    assert "contributor_mask = __ballot_sync(" in scatter
    assert "__popc(contributor_mask) == 1" in scatter
    assert "pool_slice_copy_warp_shard(" in scatter
    assert "pool_slice_add_bf16_warp_shard(" in scatter
    assert "__hadd2(" in source
    assert "float2 sums[4][4]" not in scatter
    weighted_return = source.split("pool_slice_return_weighted(", 1)[1].split(
        "pool_slice_stream_publish_metadata(", 1
    )[0]
    assert "if (warp == 0 && active_shard)" in weighted_return
    assert "pool_slice_quiet_block();" not in weighted_return
    assert "nvshmemx_putmem_signal_nbi_warp(" in weighted_return
    assert "pool_slice_weighted_source_shards(" in weighted_return
    assert "pool_slice_weighted_return_group_count(" in weighted_return
    assert "dae_atomic_fetch_add_acq_rel_gpu(" in weighted_return


def test_streaming_pool_gather_decouples_metadata_and_dynamic_data_groups():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    abi = (ROOT / "include" / "dae" / "pool_slice_abi.cuh").read_text()
    host_abi = (ROOT / "include" / "dae" / "pool_host_abi.h").read_text()
    python = (ROOT / "python" / "dae" / "pool_slice.py").read_text()
    assert POOL_SLICE_MAX_STREAM_QUEUES == 2
    assert POOL_SLICE_QUEUE_ENTRY_BYTES == 32
    assert POOL_SLICE_STREAM_QUEUE_DEPTH == POOL_SLICE_MAX_POOL_BLOCKS + 2
    assert "POOL_SLICE_FLAGS_STREAMING_DISPATCH" not in abi
    assert "pool_slice_stream_group_count(" in source
    assert "pool_slice_stream_send_group(" in source
    assert "pool_slice_stream_gather_rows(" in source
    assert "pool_slice_stream_build_queues(" in source
    assert "pool_slice_stream_claim_queue_head(" in source
    assert "pool_slice_stream_drain_queue_control(" in source
    metadata_publisher = source.split(
        "pool_slice_stream_publish_metadata_target(", 1
    )[1].split("pool_slice_stream_accept_metadata(", 1)[0]
    assert "nvshmemx_putmem_signal_nbi_warp(" in metadata_publisher
    assert "nvshmem_putmem_signal_nbi(" not in metadata_publisher
    assert metadata_publisher.count("nvshmemx_putmem_signal_nbi_warp(") == 1
    assert "NVSHMEM_SIGNAL_SET" in metadata_publisher
    assert "poolSliceControlStreamMetadataTransportReady" in metadata_publisher
    assert "pool_slice_stream_route_words(" in metadata_publisher
    assert "sizeof(uint32_t)" in metadata_publisher
    assert "sizeof(uint64_t)" not in metadata_publisher
    assert "route_count) * sizeof(uint32_t) + 15" in metadata_publisher
    assert "~15ULL" in metadata_publisher
    assert "poolSliceControlStreamMetadataParts" not in source
    assert "pool_slice_quiet_warp" not in source
    assert "nvshmemx_putmem_signal_warp(" not in metadata_publisher
    assert "nvshmem_putmem_signal(" not in metadata_publisher
    assert "index = warp" in metadata_publisher
    assert "index += TotalWarps" in metadata_publisher
    metadata_call = source.index("pool_slice_stream_publish_metadata<TotalWarps>(")
    assert "if (config.pool_rank == 0)" in source[metadata_call - 512 : metadata_call]
    assert source.count("pool_slice_stream_publish_metadata<TotalWarps>(") == 1
    assert "weight_bits << 16" in python
    assert "send_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)" in python
    assert "route_capacity * 4 + 15" in python
    assert "token_capacity > 1 << 16" in python
    assert "POOL_SLICE_QUEUE_RESERVE_ROUTES" in abi
    assert "POOL_SLICE_QUEUE_COPY_ROWS" in abi
    assert "POOL_SLICE_QUEUE_END" in abi
    assert "poolSliceMaxStreamQueues = 2" in abi
    assert "PoolSliceMetadataEnvelope" in abi
    assert "pool_slice_stream_route_lower_bound(" in source
    assert "poolSliceControlStreamMetadataReady" in source
    assert "metadata_parts_expected" not in source
    assert "poolSliceControlStreamRouteReady" in source
    assert "poolSliceControlStreamMetadataIssued" not in source
    assert "poolSliceControlStreamMetadataPosted" not in source
    assert "poolSliceControlStreamDataReady" in source
    assert "poolSliceControlStreamQueueHead" in source
    assert "poolSliceControlStreamQueueClaim" in source
    assert "retired & (1ULL << queue_index)" in source
    assert "poolSliceControlStreamExpectedGroups" not in source
    assert "pool_slice_quiet_block();" in source
    assert "pool_slice_remote_first_pe(" in source
    assert "nvshmemx_signal_op(" in source
    assert "pool_ibgda_sg_put_signal_warp(" not in source
    assert "target_group_bytes = 512ULL * 1024" in source
    assert "target_group_rows = 32" in source
    assert "hostSglRingMaxRows = 512" in host_abi
    assert "POOL_HOST_RING_MAX_ROWS = 512" in python
    assert "token_capacity > POOL_HOST_RING_MAX_ROWS" in python
    assert "dae_atomic_fetch_or_acq_rel_gpu(" in source
    assert "index < 3; index += blockDim.x" in source
    assert "index < 4; index += blockDim.x" not in source
    assert "streaming_dispatch" not in python


def test_payload_publication_separates_visibility_from_dependency_tracking():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    atomics = (ROOT / "include" / "dae" / "scoped_atomic.cuh").read_text()
    assert "dae_atomic_add_release_sys(" not in source
    assert "dae_atomic_load_acquire_sys(" not in source
    assert ".sys.global" not in atomics
    assert "atom.acq_rel.gpu.global.or.b64" in atomics
    assert "atom.acq_rel.gpu.global.add.u32" in atomics
    assert "atom.acquire.gpu.global.cas.b64" in atomics
    assert "cuda::atomic" not in source
    assert "pool_slice_publish_counter_release(" not in source
    assert "pool_slice_publish_counter_ready(" not in source
    assert "__threadfence_system();" not in source
    assert "nvshmem_fence();" not in source
    assert "nvshmemx_putmem_signal_nbi_warp(" in source


def test_pool_protocol_has_no_explicit_system_fence():
    sources = "\n".join(
        path.read_text()
        for path in (ROOT / "include" / "dae").rglob("*.cuh")
    )
    assert "__threadfence_system" not in sources


def test_pool_local_dependencies_use_isolated_release_acquire_signals():
    helper = (ROOT / "include" / "dae" / "pool_signal.cuh").read_text()
    assert "st.release.gpu.global.u32" in helper
    assert "ld.acquire.gpu.global.u32" in helper
    store = (ROOT / "include" / "dae" / "pipeline" / "stwarp.cuh").read_text()
    assert "OP_ALLOC_WB_POOL_TMA_STORE_1D" in store
    assert "pool_signal_release(" in store
    allocator = (
        ROOT / "include" / "dae" / "pipeline" / "allocwarp.cuh"
    ).read_text()
    assert "OP_POOL_WAIT_SIGNAL" in allocator
    assert "pool_signal_ready(" in allocator
    assert "atomicSub(&bars[inst.bar()], 1);" in store


def test_pool_inst_has_its_own_compile_time_warp_type():
    source = (ROOT / "include" / "dae" / "dae2.cuh").read_text()
    assert "typename PoolInstExecuteWarp" in source
    assert "PoolInstExecuteWarp::execute(" in source
    assert "pool_instructions[sm_id * numPoolInsts]" in source
    assert "communicationwarp_execute(" in source
    pool_inst = (ROOT / "include" / "dae" / "pipeline" / "poolinst.cuh").read_text()
    assert "struct PoolSliceExchangeExecuteWarp" in pool_inst
    assert "struct PoolSliceWeightedExchangeExecuteWarp" in pool_inst
    assert "struct PoolSliceHostWeightedExchangeExecuteWarp" in pool_inst
    assert "pool_slice_exchange<false, num_warps>(" in pool_inst
    assert "pool_slice_exchange<true, num_warps>(" in pool_inst
    assert "pool_slice_host_weighted_exchange<num_warps>(" in pool_inst
    assert "(void)physical_warps;" in pool_inst
    assert "switch (inst.opcode)" not in pool_inst
    registry = (ROOT / "include" / "dae" / "pool_opcode.cuh.inc").read_text()
    assert "PoolSliceExchangeExecuteWarp" in registry
    assert "PoolSliceWeightedExchangeExecuteWarp" in registry
    assert "PoolSliceHostWeightedExchangeExecuteWarp" in registry
    context = (ROOT / "include" / "dae" / "context.cuh").read_text()
    assert "struct alignas(16) PoolInst" in context
    assert "struct alignas(16) CommInst" in context
