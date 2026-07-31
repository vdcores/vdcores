from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dae.instructions import (
    PoolInstruction,
    PoolSliceDynamicReadCopy,
    PoolSliceDynamicReadReduceAdd,
    PoolSliceExchange,
    PoolSliceHostWeightedExchange,
    PoolSliceWeightedExchange,
    RawAddress,
)
from dae.pool_slice import (
    POOL_SLICE_COMPLETION_SLOTS,
    POOL_SLICE_CONFIG_BYTES,
    POOL_SLICE_CONTROL_COMBINE_FIRST_READY,
    POOL_SLICE_CONTROL_COMBINE_PLAN,
    POOL_SLICE_CONTROL_EXECUTOR_CONSUMER,
    POOL_SLICE_CONTROL_EXECUTOR_INITIALIZED,
    POOL_SLICE_CONTROL_EXECUTOR_PHASE_SEQUENCE,
    POOL_SLICE_CONTROL_EXECUTOR_PRODUCER,
    POOL_SLICE_CONTROL_EXECUTOR_RING,
    POOL_SLICE_DYNAMIC_READ_PLAN_WORDS,
    POOL_SLICE_EXECUTOR_RING_DEPTH,
    POOL_SLICE_EXECUTOR_SLOT_WORDS,
    POOL_SLICE_HOST_CONFIG_BYTES,
    POOL_SLICE_MAX_DATA_GROUPS,
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
    POOL_SLICE_PAYLOAD_WARPS,
    POOL_SLICE_WARP_QP_COMPLETION,
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
            group_limit=POOL_SLICE_MAX_DATA_GROUPS,
            pool_count=2,
            pool_rank=1,
        ).pack()
    ) == POOL_SLICE_CONFIG_BYTES
    with pytest.raises(ValueError, match="protocol limit"):
        _config(
            group_limit=POOL_SLICE_MAX_DATA_GROUPS + 1,
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


def test_pool_slice_dynamic_reads_are_pool_instructions():
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

    copy = PoolSliceDynamicReadCopy(
        address,
        local_reader=3,
        write_barrier=7,
        dispatch_barrier_base=11,
    )
    assert _fields(copy)[:4] == [0x100, 3, 7, 11]
    assert copy.selects_pool_execute_warp is False

    reduce_add = PoolSliceDynamicReadReduceAdd(
        address,
        plan_rank=5,
        compute_barrier_base=19,
    )
    assert _fields(reduce_add)[:4] == [0x101, 5, 19, 0]
    assert reduce_add.selects_pool_execute_warp is False
    assert exchange.selects_pool_execute_warp is True

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
        num_pes=3,
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
    assert "store = TmaStore1D(" in source
    assert "builder.add_memory(IssueBarrier(" in source
    assert "PoolTmaStore1D" not in source
    assert "PoolWaitSignal" not in source
    assert "writer_blocks = 0 if source_preloaded else 1" in source
    assert "pool_base = writer_blocks" in source
    assert "reader_base = pool_base + buffers.pool_count" in source
    assert "torch.cuda.Stream" not in source
    assert "torch.cuda.Event" not in source
    assert "source_preloaded" in source
    assert "launcher.new_bar(0 if source_preloaded else 1)" in source
    assert "launcher.disable_cache_window()" in source


def test_pool_program_has_no_model_specific_rms_operator():
    sources = "\n".join(
        (ROOT / relative).read_text()
        for relative in (
            "benchmarks/pool_slice_nccl_compare.py",
            "include/dae/compute_dispatch.cuh",
            "include/dae/opcode.cuh.inc",
            "include/task/rms_norm.cuh",
            "python/dae/instructions.py",
            "python/dae/pool_slice.py",
        )
    )
    for removed_name in (
        "OP_POOL_RMS_NORM",
        "POOL_RMS_NORM",
        "task_pool_rms_norm",
        "reader_rms",
        "--reader-op",
    ):
        assert removed_name not in sources


def test_pool_mailbox_scan_is_lane_parallel_and_uses_one_source_word():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "__ballot_sync(0xffffffffU, metadata_ready)" in source
    assert "base < total_queue_count" in source
    assert "base + lane" in source
    assert "config.signal_base + lane" in source
    assert "poolSliceControlStreamMetadataTransportReady + lane" in source
    metadata = source.split(
        "pool_slice_stream_publish_metadata_target(", 1
    )[1].split("pool_slice_stream_accept_metadata(", 1)[0]
    assert "nvshmemx_putmem_signal_nbi_warp(" in metadata
    assert "NVSHMEM_SIGNAL_ADD" in metadata
    assert "signal_delta" in metadata
    assert "payload_coupled" not in metadata
    assert "inline_generation" not in metadata
    assert "nvshmemx_putmem_nbi_warp(" not in metadata
    assert "poolSliceControlStreamMetadataSourceSequence" in source


def test_raw_sgl_progress_reuses_one_group_signal_and_one_reader_task():
    abi = (ROOT / "include" / "dae" / "pool_slice_abi.cuh").read_text()
    raw = (ROOT / "include" / "dae" / "pool_ibgda_sgl.cuh").read_text()
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()

    assert "poolSliceRawSglProgressStride = 1ULL << 17" in abi
    assert "total_wqebbs += pool_ibgda_sgl_wqebbs(count) + 1" in raw
    assert "const uint32_t reserved_wqebbs = total_wqebbs" in raw
    assert "pool_ibgda_put_contiguous_signal_warp(" in raw
    assert (
        "static __device__ __noinline__ void pool_ibgda_sgl_put_rows_warp("
        in raw
    )
    assert (
        "static __device__ __noinline__ void\n"
        "pool_ibgda_put_contiguous_signal_warp(" in raw
    )
    assert "return false;" not in raw
    assert "constexpr uint32_t wqebbs = 2" in raw
    assert "sequence * poolSliceRawSglProgressStride + wqe + 1" in raw
    assert "wqe + 1 == wqe_count ? MLX5_WQE_CTRL_CQ_UPDATE : 0" in raw
    assert raw.count("ibgda_submit_requests<true>(") == 2
    assert "base_wqe + total_wqebbs" not in raw
    assert "shared_sgl_sent" not in source

    assert "pool_slice_stream_data_progress(" in source
    assert "pool_slice_stream_data_segments(" in source
    assert "pool_slice_stream_gather_rows<HostDataPlane, TotalWarps>" in source
    assert "message.ready_slot |" in source
    assert "static_cast<uint32_t>(instruction.size) << 16" in source
    assert "ready_slot_and_reader & 0xffffU" in source
    assert "segment * poolSliceRawSglWidth" in source
    assert "reader-cta-progress" not in source
    # Progress changes only the meaning of the existing ready word: it does
    # not create another queue, group, claim state, or destination task.
    assert "poolSliceMaxStreamQueues = 2" in abi
    assert "poolSliceCompletionSlots" in abi


def test_device_dispatch_uses_exact_static_qp_scope_generations():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "config.delivery_pool_address" in source
    assert "route.row_count) * config.row_bytes" in source
    send_group = source.split("pool_slice_stream_send_group(", 1)[1].split(
        "pool_slice_stream_gather_rows(", 1
    )[0]
    public_put = source.split(
        "pool_slice_stream_put_rows_public(", 1
    )[1].split("#undef DAE_POOL_SLICE_PUBLIC_FALLBACK_QUALIFIER", 1)[0]
    assert "pool_slice_put_nbi_warp(" in public_put
    assert "nvshmemx_putmem_nbi_warp(" in source
    assert "pool_slice_peer_ptr(destination, target_pe)" in source
    assert "pool_slice_stream_put_rows_public(" in send_group
    assert "if constexpr (poolSliceWarpQpCompletion)" in send_group
    assert "if constexpr (!poolSliceWarpQpCompletion)" in send_group
    assert "nvshmemx_putmem_signal_nbi_warp(" not in send_group
    assert "nvshmemx_signal_op(" not in send_group
    assert "nvshmem_uint64_p(" in send_group
    assert "pool_slice_stream_reserve_data_signal_delta(" not in send_group
    assert "NVSHMEM_SIGNAL_ADD" not in send_group
    assert "NVSHMEM_SIGNAL_SET" not in send_group
    assert "pool_slice_quiet_block();" not in send_group
    assert "message->ready_slot" in source
    assert "payload_warp < TotalWarps" in source
    assert "poolSlicePayloadWarps" in source
    assert "poolSliceCompletionSlots" in source
    assert "poolSliceControlStreamDataSourceSequence" not in source
    # The generic unweighted return still needs a cooperative completion;
    # weighted EP return already uses payload-coupled generations.
    assert source.count("pool_slice_quiet_block();") == 1
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
    assert "nvshmemx_putmem_nbi_warp(" in source
    assert "nvshmem_uint64_p(" in source
    assert "poolSliceControlDispatchGeneration" in source
    assert "poolSliceControlReturnGeneration" in source
    assert "poolSliceControlReturnReady" in source
    assert "poolSliceControlScatterGeneration" in source
    assert "poolSliceControlReaderRowCount" in source
    assert "dae_atomic_store_release_gpu(" in source
    assert "dae_atomic_load_acquire_gpu(" in source
    assert "poolSliceMaxPoolBlocks = 132" in abi
    assert "poolSliceMaxDataGroups = 32" in abi
    assert "poolSlicePayloadWarps = DAE_POOL_SLICE_WARPS" in abi
    assert "poolSliceCompletionSlots" in abi
    assert "for pool_rank in range(buffers.pool_count)" in python
    assert (
        "POOL_SLICE_CONTROL_DISPATCH_READY = POOL_SLICE_CONTROL_START + 1"
        in python
    )


def test_reduce_add_scatter_bypasses_reduction_for_one_pool_contributor():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    scatter = source.split("pool_slice_weighted_scatter_token(", 1)[1].split(
        "pool_slice_reduce_add_source_shards(", 1
    )[0]
    assert "contributor_mask = __ballot_sync(" in scatter
    assert "__popc(contributor_mask) == 1" in scatter
    assert "pool_slice_copy_warp_shard(" in scatter
    assert "pool_slice_add_bf16_warp_shard(" in scatter
    assert "__hadd2(" in source
    assert "float2 sums[4][4]" not in scatter
    reduce_add = source.split(
        "pool_slice_dynamic_read_reduce_add_local(", 1
    )[1].split(
        "pool_slice_dynamic_read_reduce_add_finish(", 1
    )[0]
    reduce_add_grouping = source.split(
        "pool_slice_reduce_add_group_count(\n"
        "    uint32_t rows, uint32_t active_shards, uint32_t row_bytes) {",
        1,
    )[1].split("pool_slice_reduce_add_shard_range(\n", 1)[0]
    assert "if (warp == transport_warp && active_shard)" in reduce_add
    assert "transport_warp = group % TotalWarps" in reduce_add
    assert "pool_slice_quiet_block();" not in reduce_add
    assert "nvshmemx_putmem_nbi_warp(" in reduce_add
    assert "nvshmem_uint64_p(" in reduce_add
    assert "nvshmemx_putmem_signal_nbi_warp(" not in reduce_add
    assert "pool_ibgda_put_contiguous_signal_warp(" in reduce_add
    assert "pool_slice_reduce_add_source_shards(" in reduce_add
    assert "pool_slice_reduce_add_return_group_count(" in reduce_add
    assert "pool_slice_reduce_add_group_count(" in reduce_add
    assert "target_group_bytes = 256ULL * 1024" in reduce_add_grouping
    assert "dae_atomic_fetch_add_acq_rel_gpu(" in reduce_add


def test_dynamic_reads_are_prebuilt_poolinsts_with_shared_workers():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    abi = (ROOT / "include" / "dae" / "pool_slice_abi.cuh").read_text()
    python = (ROOT / "python" / "dae" / "pool_slice.py").read_text()

    assert POOL_SLICE_DYNAMIC_READ_PLAN_WORDS == 4
    assert POOL_SLICE_CONTROL_COMBINE_FIRST_READY < (
        POOL_SLICE_CONTROL_COMBINE_PLAN
    )
    assert "struct alignas(16) PoolSliceDynamicReadPlan" in abi
    assert "sizeof(PoolSliceDynamicReadPlan) ==" in abi
    assert "POOL_SLICE_QUEUE_DATA = 2" in abi
    assert "POOL_SLICE_QUEUE_COPY_ROWS" not in abi
    assert "POOL_SLICE_DYNAMIC_READ_PLAN_WORDS = 4" in python
    assert "enum PoolSliceDynamicReadOpcode" in abi
    assert "POOL_SLICE_DYNAMIC_READ_COPY" in abi
    assert "POOL_SLICE_DYNAMIC_READ_REDUCE_ADD" in abi
    assert "POOL_SLICE_DYNAMIC_READ_COPY = 0x100" in abi
    assert "POOL_SLICE_DYNAMIC_READ_REDUCE_ADD = 0x101" in abi
    assert "struct PoolSliceDynamicReadWorker" in source
    assert source.count("PoolSliceDynamicReadWorker::") >= 3
    assert "PoolSliceDynamicReadExecutor" not in source
    assert "switch (instruction.opcode)" in source
    assert "PoolSliceDynamicReadCopy" in python
    assert "PoolSliceDynamicReadReduceAdd" in python
    assert "switch (transform)" not in source
    assert POOL_SLICE_CONTROL_EXECUTOR_INITIALIZED > (
        POOL_SLICE_CONTROL_COMBINE_PLAN
    )
    assert POOL_SLICE_CONTROL_EXECUTOR_PRODUCER == (
        POOL_SLICE_CONTROL_EXECUTOR_INITIALIZED + 1
    )
    assert POOL_SLICE_CONTROL_EXECUTOR_CONSUMER == (
        POOL_SLICE_CONTROL_EXECUTOR_PRODUCER + 1
    )
    assert POOL_SLICE_CONTROL_EXECUTOR_PHASE_SEQUENCE == (
        POOL_SLICE_CONTROL_EXECUTOR_CONSUMER + 1
    )
    assert POOL_SLICE_CONTROL_EXECUTOR_RING == (
        POOL_SLICE_CONTROL_EXECUTOR_PHASE_SEQUENCE + 1
    )
    assert POOL_SLICE_CONTROL_EXECUTOR_RING % 2 == 0
    assert POOL_SLICE_EXECUTOR_RING_DEPTH == 512
    assert POOL_SLICE_EXECUTOR_RING_DEPTH & (
        POOL_SLICE_EXECUTOR_RING_DEPTH - 1
    ) == 0
    assert POOL_SLICE_EXECUTOR_SLOT_WORDS == 4
    assert "struct alignas(16) PoolSliceExecutorSlot" in abi
    assert "sizeof(PoolSliceExecutorSlot) == 32" in abi
    assert "uint32_t message_count" in abi

    builder = source.split(
        "pool_slice_build_reduce_add_plans_source(", 1
    )[1].split(
        "pool_slice_reduce_add_return_ready(", 1
    )[0]
    assert "uint32_t source_pe" in builder
    assert "plan.dependency_mask" in builder
    assert "batch.reader_counts[reader] == rows" in builder
    assert "reader_rows[row] != UINT64_MAX" in builder
    assert "pool_slice_dynamic_read_plan(control, pool_rank) = plan" in builder
    assert "nvshmem" not in builder

    reserve = source.split(
        "pool_slice_stream_execute_reserve_routes(", 1
    )[1].split("pool_slice_stream_execute_queue_control(", 1)[0]
    assert "combine_rows[" in reserve
    assert "route.base_row + relative" in reserve
    assert "pool_slice_build_reduce_add_plans_source(" in reserve
    assert reserve.index("pool_slice_build_reduce_add_plans_source(") < (
        reserve.index("poolSliceControlStreamRouteReady + source_pe")
    )
    gather = source.split("pool_slice_stream_gather_rows(", 1)[1].split(
        "pool_slice_route_weight(", 1
    )[0]
    assert "combine_rows" not in gather
    assert "poolSliceControlReaderDataDone + local_reader" in gather
    assert "dae_atomic_add_release_gpu(" in gather
    assert "atomicOr(" not in gather

    completion = source.split(
        "control + poolSliceControlDispatchGeneration + config.pool_rank",
        1,
    )[1]
    assert "pool_slice_build_reduce_add_plans" not in completion
    coordinator = source.split(
        "pool_slice_scheduler_dispatch_warp(", 1
    )[1].split("pool_slice_executor_loop(", 1)[0]
    assert "pool_slice_stream_reader_data_groups(" in coordinator
    assert "poolSliceControlReaderDataDone + lane" in coordinator
    assert "shared->bars + shared->dispatch_barrier_base + lane" in coordinator
    coordinator_start = source.index(
        "bool metadata_ready = lane >= config.num_pes"
    )
    assert source.index(
        "control + poolSliceControlDispatchReady", coordinator_start
    ) < source.index(
        "control + poolSliceControlDispatchGeneration + config.pool_rank"
    )

    reduce_add = source.split(
        "pool_slice_dynamic_read_reduce_add_local(", 1
    )[1].split("pool_slice_dynamic_read_reduce_add_finish(", 1)[0]
    assert "const uint32_t dependencies = plan.dependency_mask" in reduce_add
    assert "dependencies & (1U << lane)" in reduce_add
    assert "bars + compute_barrier_base + lane" in reduce_add
    assert "pool_slice_dynamic_read_plan(control, plan_rank)" in reduce_add
    assert "poolSliceControlReturnGeneration + plan_rank" in reduce_add
    assert "lane >= config.local_readers;" not in reduce_add
    finish = source.split(
        "pool_slice_dynamic_read_reduce_add_finish(", 1
    )[1].split("pool_slice_stream_publish_metadata_target(", 1)[0]
    assert "poolSliceControlReturnGeneration" in reduce_add
    assert "pool_slice_wait_generation_warp(" in finish
    assert "poolSliceControlReturnGeneration" in finish
    scheduler = source.split(
        "pool_slice_scheduler_dispatch_warp(", 1
    )[1].split("pool_slice_executor_loop(", 1)[0]
    executor = source.split("pool_slice_executor_loop(", 1)[1].split(
        "pool_slice_return_unweighted(", 1
    )[0]
    assert "shared_scheduler_heads" not in executor
    assert "shared->heads[queue_index]" in scheduler
    assert "shared->expected[queue_index]" in scheduler
    assert "pool_slice_executor_publish_batch_warp(" in scheduler
    assert "config.local_readers," in source
    assert "POOL_SLICE_EXECUTOR_DYNAMIC_READ" in source
    assert "POOL_SLICE_EXECUTOR_RESERVE_ROUTES" not in source
    assert "shared_direct_route_source" in source
    assert "POOL_SLICE_EXECUTOR_SEND" in scheduler
    assert "direct_send_count" in source
    assert "shared_direct_send_owns_metadata" not in source
    assert "free_worker_count" not in source
    assert "preferred_direct_count" not in source
    assert "worker_index < direct_send_count" in source
    assert "candidate_group < groups" in source
    assert "POOL_SLICE_EXECUTOR_STOP" in scheduler
    assert "pool_slice_stream_claim_queue_head(" not in scheduler
    assert "pool_slice_dynamic_read_finish_data_head(" not in scheduler
    assert "poolSliceControlExecutorConsumer" in executor
    assert "atomicAdd(" in executor
    assert "shared->ticket += shared->ticket_stride" not in executor
    assert "shared->phase_start" in executor
    assert "shared->ticket_index" not in executor
    assert "poolSliceControlExecutorPhaseSequence" in executor
    assert "poolSliceControlExecutorPhaseBase" not in executor
    assert "shared->direct_send_count" in scheduler
    assert "shared->send_count" in scheduler
    assert "pool_slice_stream_decode_send_job(" not in source
    assert "pool_slice_executor_send_task<" in source
    assert "initial_send_published = shared->direct_send_count" in scheduler
    assert "message_count + static_cast<uint32_t>(route_pending)" in source
    assert "poolSliceControlStreamMetadataSubmittedMask" not in source
    assert "PoolSliceDynamicReadWorker::" in executor
    assert "dae_atomic_or_release_gpu(" in executor
    assert "pool_slice_stream_queue_head(" not in executor
    assert "shared_dynamic_read_instructions[" in source
    assert "dynamic_read_instructions[instruction]" in source
    assert "shared->batch_offset < shared->batch_count" in executor
    assert "shared->task.message_index + shared->batch_offset" in executor
    assert "while (!pool_slice_scheduler_queue_message_ready" in executor
    assert "pool_slice_executor_publish_routes_warp(" not in scheduler
    assert "pool_slice_scheduler_prepost_source_reads(" in scheduler
    assert "route_mask & ~shared->reads_preposted_mask" in scheduler
    assert "shared->instruction" in executor
    assert "POOL_SLICE_EXECUTOR_DYNAMIC_READ_STOP" not in source
    assert "pool_slice_scheduler_publish_reduce_warp" not in source
    assert "pool_slice_scheduler_prepost_source_reductions(" in scheduler
    assert "shared->reductions_preposted_mask" in scheduler
    assert "config.local_readers + source_pe +" in source
    assert "(base + lane) * config.num_pes" in source
    assert "ReduceAdd tickets precede all STOP tickets" in scheduler
    assert "WeightedReturn && config.pool_rank == 0" in source
    assert "config.pool_rank <= executor_count" in source
    assert "if (config.pool_count == 1)" in source
    assert "pool_slice_executor_loop<" in source


def test_streaming_pool_gather_decouples_metadata_and_dynamic_data_groups():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    abi = (ROOT / "include" / "dae" / "pool_slice_abi.cuh").read_text()
    host_abi = (ROOT / "include" / "dae" / "pool_host_abi.h").read_text()
    python = (ROOT / "python" / "dae" / "pool_slice.py").read_text()
    dispatch_grouping = source.split(
        "pool_slice_stream_group_count(", 1
    )[1].split("pool_slice_stream_dispatch_worker_count(", 1)[0]
    assert POOL_SLICE_MAX_STREAM_QUEUES == 2
    assert POOL_SLICE_MAX_POOL_BLOCKS == 132
    assert POOL_SLICE_MAX_DATA_GROUPS == 32
    assert POOL_SLICE_PAYLOAD_WARPS == 8
    assert POOL_SLICE_WARP_QP_COMPLETION is False
    assert POOL_SLICE_COMPLETION_SLOTS == 1
    assert POOL_SLICE_QUEUE_ENTRY_BYTES == 32
    assert POOL_SLICE_STREAM_QUEUE_DEPTH == 2 + (
        POOL_SLICE_MAX_DATA_GROUPS + 1
    ) // 2
    assert "POOL_SLICE_FLAGS_STREAMING_DISPATCH" not in abi
    assert "pool_slice_stream_group_count(" in source
    assert "pool_slice_stream_send_group(" in source
    assert "pool_slice_stream_gather_rows(" in source
    assert "pool_slice_stream_build_queues(" in source
    assert "pool_slice_scheduler_queue_message_ready(" in source
    assert "pool_slice_scheduler_dispatch_warp(" in source
    assert "pool_slice_executor_publish_batch_warp(" in source
    assert "pool_slice_executor_loop(" in source
    assert "offset = base + lane" in source
    assert "One rank-zero warp owns every destination queue cursor" in source
    assert "Persistent executor CTAs dynamically claim" in source
    assert "candidate_group < groups" in source
    assert "signal_delta,\n          config.pool_rank," in source
    assert "neither submission gates the other" in source
    assert '"central-shared-head-executor-ring"' in (
        ROOT / "benchmarks" / "pool_slice_nccl_compare.py"
    ).read_text()
    assert '"target-major-dynamic-group-cta-stripe"' in (
        ROOT / "benchmarks" / "pool_slice_nccl_compare.py"
    ).read_text()
    assert "pool_slice_stream_drain_queue_control(" in source
    assert "poolSliceControlStreamDataSourceSequence" not in abi
    assert "POOL_SLICE_CONTROL_STREAM_DATA_SOURCE_SEQUENCE" not in python
    metadata_publisher = source.split(
        "pool_slice_stream_publish_metadata_target(", 1
    )[1].split("pool_slice_stream_accept_metadata(", 1)[0]
    assert "nvshmemx_putmem_signal_nbi_warp(" in metadata_publisher
    assert "nvshmemx_putmem_nbi_warp(" not in metadata_publisher
    assert "nvshmem_putmem_signal_nbi(" not in metadata_publisher
    assert metadata_publisher.count("nvshmemx_putmem_signal_nbi_warp(") == 1
    assert "NVSHMEM_SIGNAL_ADD" in metadata_publisher
    assert "signal_delta" in metadata_publisher
    assert "payload_coupled" not in metadata_publisher
    assert "inline_generation" not in metadata_publisher
    assert "nvshmem_uint64_p(transport_ready, sequence, target_pe)" not in (
        metadata_publisher
    )
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
    metadata_call = source.index(
        "pool_slice_stream_publish_metadata<HostDataPlane, TotalWarps>("
    )
    assert source.count(
        "pool_slice_stream_publish_metadata<HostDataPlane, TotalWarps>("
    ) == 1
    assert "config.pool_count >= config.num_pes" in (
        metadata_publisher
    )
    assert "config.pool_rank < config.num_pes && warp == 0" in metadata_publisher
    assert "if (config.pool_rank != 0)" in metadata_publisher
    assert "smaller generic assembly falls back" in metadata_publisher
    coordinator = source[metadata_call:]
    assert "poolSliceControlStreamMetadataTransportReady + lane" in coordinator
    assert "pool_slice_stream_data_ready(control, lane, 0, 0)" not in coordinator
    assert "poolSliceControlStreamMetadataSignalDelta" in source
    assert "weight_bits << 16" in python
    assert "send_rows = nvshmem.zeros(route_capacity, dtype=torch.uint32)" in python
    assert "route_capacity * 4 + 15" in python
    assert "token_capacity > 1 << 16" in python
    assert "POOL_SLICE_QUEUE_RESERVE_ROUTES" in abi
    assert "POOL_SLICE_QUEUE_DATA" in abi
    assert "POOL_SLICE_QUEUE_END" in abi
    assert "poolSliceMaxStreamQueues = 2" in abi
    assert "PoolSliceMetadataEnvelope" in abi
    assert "pool_slice_stream_route_lower_bound(" in source
    assert "poolSliceControlStreamMetadataReady" in source
    assert "metadata_parts_expected" not in source
    assert "poolSliceControlStreamRouteReady" in source
    assert "poolSliceControlReaderDataDone" in abi
    assert "POOL_SLICE_CONTROL_READER_DATA_DONE" in python
    assert "poolSliceControlStreamMetadataIssued" not in source
    assert "poolSliceControlStreamDataReady" in source
    assert "poolSliceControlStreamQueueHead" in source
    assert "poolSliceControlStreamQueueClaim" in source
    assert "retired & (1ULL << queue_index)" in source
    assert "poolSliceExecutorRingDepth" in source
    assert "ticket & (poolSliceExecutorRingDepth - 1)" in source
    assert "ticket + poolSliceExecutorRingDepth" in source
    assert "ticket + 1" in source
    assert "poolSliceControlExecutorInitialized" in source
    assert "shared_executor_initialize" in source
    assert "poolSliceControlStreamExpectedGroups" not in source
    assert "pool_slice_stream_data_ready(" in source
    assert "payload_warp" in source
    assert "atomicCAS(copy_claim, state, desired)" in source
    assert "gather_work = control[4] *" in source
    assert "dae_atomic_add_release_gpu(" in source
    assert "pool_slice_remote_first_pe(" in source
    assert "nvshmemx_signal_op(" in source
    assert "pool_ibgda_sg_put_signal_warp(" not in source
    assert "target_group_bytes_low_pe = 256ULL * 1024" in dispatch_grouping
    assert "target_group_bytes_high_pe = 512ULL * 1024" in dispatch_grouping
    assert "num_pes >= 8" in dispatch_grouping
    assert "pilot_group_bytes" not in source
    assert "target_group_rows = 32" in source
    assert "active_rows) * group / group_count" in source
    assert "active_rows) * (group + 1) / group_count" in source
    assert "#if DAE_POOL_SLICE_RAW_SGL" not in metadata_publisher
    assert "hostSglRingMaxRows = 512" in host_abi
    assert "POOL_HOST_RING_MAX_ROWS = 512" in python
    assert "token_capacity > POOL_HOST_RING_MAX_ROWS" in python
    assert "dae_atomic_fetch_or_acq_rel_gpu(" in source
    scratch_reset = source.split(
        "Only these three words are invocation-local scratch", 1
    )[1].split("if constexpr (WeightedReturn)", 1)[0]
    assert "index < 3; index += blockDim.x" in scratch_reset
    assert "index < 4; index += blockDim.x" not in scratch_reset
    assert "streaming_dispatch" not in python


def test_payload_publication_separates_visibility_from_dependency_tracking():
    source = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    atomics = (ROOT / "include" / "dae" / "scoped_atomic.cuh").read_text()
    assert "dae_atomic_add_release_sys(" not in source
    assert "dae_atomic_load_acquire_sys(" not in source
    assert ".sys.global" not in atomics
    assert "atom.acq_rel.gpu.global.or.b64" in atomics
    assert "red.release.gpu.global.or.b64" in atomics
    assert "atom.acq_rel.gpu.global.add.u32" in atomics
    assert "atom.acquire.gpu.global.cas.b64" in atomics
    assert "cuda::atomic" not in source
    assert "pool_slice_publish_counter_release(" not in source
    assert "pool_slice_publish_counter_ready(" not in source
    assert "__threadfence_system();" not in source
    assert "nvshmem_fence();" not in source
    assert "nvshmemx_putmem_nbi_warp(" in source
    assert "nvshmem_uint64_p(" in source


def test_pool_protocol_has_no_explicit_system_fence():
    sources = "\n".join(
        path.read_text()
        for path in (ROOT / "include" / "dae").rglob("*.cuh")
    )
    assert "__threadfence_system" not in sources


def test_pool_local_dependencies_reuse_normal_countdown_barriers():
    assert not (ROOT / "include" / "dae" / "pool_signal.cuh").exists()
    opcodes = (ROOT / "include" / "dae" / "opcode.cuh.inc").read_text()
    assert "OP_POOL_WAIT_SIGNAL" not in opcodes
    assert "OP_ALLOC_WB_POOL_TMA_STORE_1D" not in opcodes
    assert "OP_ALLOC_WB_POOL_RAW_ADDRESS" not in opcodes

    pool = (ROOT / "include" / "dae" / "pool_slice.cuh").read_text()
    assert "pool_slice_barrier_ready(" in pool
    assert "volatile const int*" in pool
    assert "atomicSub(bars + dispatch_barrier_base + reader, 1);" in pool

    store = (ROOT / "include" / "dae" / "pipeline" / "stwarp.cuh").read_text()
    assert "case op(OP_ALLOC_WB_RAW_ADDRESS):" in store
    assert "atomicSub(&bars[inst.bar()], 1);" in store
    assert "const auto slot = extract(slot_mask);" in store
    assert "special_completion" not in store

    virtual_core = (ROOT / "include" / "dae" / "virtualcore.cuh").read_text()
    assert "SLOT_SPECIAL_COMPLETION" not in virtual_core
    assert "special_slot_completion" not in virtual_core

    direct_output_tasks = "\n".join(
        (ROOT / relative).read_text()
        for relative in ("include/task/rms_norm.cuh", "include/task/silu.cuh")
    )
    assert "special_slot_completion" not in direct_output_tasks
    assert direct_output_tasks.count("1U <<") >= 2

    allocator = (
        ROOT / "include" / "dae" / "pipeline" / "allocwarp.cuh"
    ).read_text()
    assert "case op(OP_ISSUE_BARRIER):" in allocator
    assert "OP_POOL_WAIT_SIGNAL" not in allocator


def test_raw_address_writeback_rejects_unrepresentable_c2m_slots():
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="cuda"),
        data_ptr=lambda: 0,
    )
    assert RawAddress(tensor, 30).writeback().opcode & 2
    with pytest.raises(ValueError, match="slot_id <= 30"):
        RawAddress(tensor, 31).writeback()
    with pytest.raises(ValueError, match="slot_id <= 30"):
        RawAddress(tensor, 32).bar(7).writeback()


def test_pool_inst_has_its_own_compile_time_warp_type():
    source = (ROOT / "include" / "dae" / "dae2.cuh").read_text()
    assert "typename PoolInstExecuteWarp" in source
    assert "PoolInstExecuteWarp::execute(" in source
    assert "pool_instructions + sm_id * numPoolInsts" in source
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
    assert "const PoolInst* instructions" in pool_inst
    registry = (ROOT / "include" / "dae" / "pool_opcode.cuh.inc").read_text()
    assert "PoolSliceExchangeExecuteWarp" in registry
    assert "PoolSliceWeightedExchangeExecuteWarp" in registry
    assert "PoolSliceHostWeightedExchangeExecuteWarp" in registry
    context = (ROOT / "include" / "dae" / "context.cuh").read_text()
    assert "struct alignas(16) PoolInst" in context
    assert "struct alignas(16) CommInst" in context
    assert "numPoolInsts = 1 + 8 + 132" in context
