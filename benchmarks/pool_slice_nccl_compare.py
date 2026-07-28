"""External pool-slice dynamic-read versus dense NCCL ring EP benchmark.

The pool samples come only from communication-warp global-timer events.  The
NCCL helper uses CUDA events and lives in this benchmark directory so neither
mechanism leaks into the VDCores runtime or application sources.
"""

from __future__ import annotations

import argparse

import torch
from mpi4py import MPI

import dae.nvshmem as nvshmem
from dae.pool_slice import (
    POOL_SLICE_PUBLISH_BYTES,
    POOL_SLICE_QUEUE_ENTRY_BYTES,
    POOL_SLICE_STREAM_QUEUE_DEPTH,
    POOL_SLICE_RETURN_GROUPS_PER_SOURCE,
    PoolSliceStatus,
    allocate_pool_slice,
    build_pool_slice_copy_program,
)
from nccl_ep_reference import (
    Timing,
    initialize_nccl,
    print_result,
    rank_max,
    run_nccl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--experts-per-pe", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--route-placement",
        choices=("clustered", "source-local", "remote-clustered", "spread"),
        default="clustered",
        help=(
            "cluster each token's routes on one balanced pool slice, force "
            "that slice local/remote to the source, or spread routes "
            "round-robin across all pool slices"
        ),
    )
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("pool", "nccl", "both"), default="both")
    parser.add_argument("--symmetric-size", default="1G")
    parser.add_argument(
        "--data-groups",
        dest="group_limit",
        type=int,
        default=0,
        help="maximum runtime data groups; zero selects the PE/CTA-aware policy",
    )
    parser.add_argument("--pool-blocks", type=int, default=32)
    parser.add_argument(
        "--weighted-return",
        action="store_true",
        help="pool-local weighted partial reduction plus token-major source sum",
    )
    parser.add_argument(
        "--reader-op",
        choices=("copy", "rms"),
        default="copy",
        help="ordinary VDCores work executed after each dynamic read",
    )
    parser.add_argument("--in-place-identity", action="store_true")
    parser.add_argument(
        "--print-pool-ctas",
        action="store_true",
        help="print the final per-PoolInst CTA weighted-return timestamps",
    )
    parser.add_argument(
        "--source-preloaded",
        action="store_true",
        help="benchmark transport from already-produced pool slots",
    )
    return parser.parse_args()


def _source_grouped_return_rmas(
    rows_per_source: list[int], local_pe: int, pool_blocks: int
) -> int:
    """Count source-owned return shards that carry a remote payload."""

    num_pes = len(rows_per_source)
    rmas = 0
    for source_pe, rows in enumerate(rows_per_source):
        if source_pe == local_pe or source_pe >= pool_blocks:
            continue
        available = 1 + (pool_blocks - 1 - source_pe) // num_pes
        rmas += min(
            int(rows), available, POOL_SLICE_RETURN_GROUPS_PER_SOURCE
        )
    return rmas


def _stream_group_counts(
    rows: torch.Tensor, *, row_bytes: int, group_limit: int
) -> torch.Tensor:
    """Mirror PoolInst's runtime-sized group policy for the cost model."""

    byte_groups = (
        rows * row_bytes + (512 * 1024 - 1)
    ).div(512 * 1024, rounding_mode="floor")
    row_groups = (rows + 31).div(32, rounding_mode="floor")
    desired = torch.maximum(byte_groups, row_groups)
    return torch.minimum(
        rows,
        torch.minimum(desired, torch.full_like(rows, group_limit)),
    )


def _balanced_expert_ids(
    global_rows: torch.Tensor,
    *,
    top_k: int,
    num_pes: int,
    experts_per_pe: int,
    placement: str,
) -> torch.Tensor:
    """Return deterministic balanced routes with controlled PE placement."""

    route = torch.arange(top_k, dtype=torch.int64)
    if placement == "clustered":
        return (
            global_rows[:, None] * top_k + route[None, :]
        ).remainder(num_pes * experts_per_pe)

    source_pe = global_rows.div(global_rows.numel(), rounding_mode="floor")
    if placement in {"source-local", "remote-clustered"}:
        target_pe = source_pe
        if placement == "remote-clustered":
            target_pe = (target_pe + 1).remainder(num_pes)
        local_expert = (
            global_rows[:, None] * top_k + route[None, :]
        ).remainder(experts_per_pe)
        return target_pe[:, None] * experts_per_pe + local_expert

    target_pe = (global_rows[:, None] + route[None, :]).remainder(num_pes)
    local_expert = (
        global_rows[:, None] * top_k + route[None, :]
    ).div(num_pes, rounding_mode="floor").remainder(experts_per_pe)
    return target_pe * experts_per_pe + local_expert


def run_pool(
    args: argparse.Namespace,
    runtime,
    comm: MPI.Comm,
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[Timing, dict[str, int | float | str]]:
    num_readers = runtime.num_pes * args.experts_per_pe
    local_routes = args.tokens_per_pe * args.top_k
    expert_capacity_rows = runtime.num_pes * args.tokens_per_pe
    signals = nvshmem.init_signal_space(runtime.num_pes)
    buffers = allocate_pool_slice(
        signals,
        num_pes=runtime.num_pes,
        my_pe=runtime.pe,
        local_readers=args.experts_per_pe,
        token_capacity=args.tokens_per_pe,
        route_capacity=local_routes,
        expert_capacity_rows=expert_capacity_rows,
        hidden_size=args.hidden_size,
        dtype=tokens.dtype,
        group_limit=args.group_limit,
        pool_blocks=args.pool_blocks,
        in_place_expert_output=args.in_place_identity,
        weighted_return=args.weighted_return,
    )
    returned_rows = args.tokens_per_pe if args.weighted_return else local_routes
    returned = nvshmem.zeros((returned_rows, args.hidden_size), dtype=tokens.dtype)
    local_rows = torch.arange(args.tokens_per_pe, dtype=torch.int64)
    source_rows = local_rows.repeat_interleave(args.top_k)
    origin_rows = torch.arange(local_routes, dtype=torch.int64)
    buffers.write_routes(
        expert_ids,
        source_rows=source_rows,
        origin_rows=source_rows if args.weighted_return else origin_rows,
        route_weights=(
            torch.full((local_routes,), 1.0 / args.top_k)
            if args.weighted_return
            else None
        ),
    )
    pool_source = tokens
    if args.source_preloaded:
        buffers.token_pool.copy_(tokens)
        pool_source = buffers.token_pool
    buffers.prepare(pool_source, returned)
    rms_weights = None
    if args.reader_op == "rms":
        if args.dtype != "bfloat16" or args.hidden_size != 4096:
            raise ValueError("reader RMS requires bfloat16 hidden-size 4096")
        rms_weights = torch.ones(
            args.hidden_size,
            dtype=tokens.dtype,
            device=tokens.device,
        )
    program = build_pool_slice_copy_program(
        buffers,
        benchmark_barrier=nvshmem.benchmark_barrier,
        in_place_identity=args.in_place_identity,
        source_preloaded=args.source_preloaded,
        reader_rms_weights=rms_weights,
    )

    gather_samples: list[float] = []
    tail_samples: list[float] = []
    total_samples: list[float] = []
    overlap_samples: dict[str, list[float]] = {
        "first_data_published": [],
        "data_published": [],
        "first_payload": [],
        "metadata_closed": [],
        "payload_done": [],
        "compute_ready": [],
        "return_payload_done": [],
        "return_signals_closed": [],
        "scatter_done": [],
    }
    overlap_samples["first_gather"] = []
    overlap_samples["stream_gather_done"] = []
    if args.weighted_return:
        overlap_samples.update(
            {
                "return_reduce_start": [],
                "return_reduce_done": [],
                "first_return_put": [],
                "return_cta_done": [],
            }
        )
    rounds = args.warmup + args.iterations
    for iteration in range(rounds):
        buffers.set_sequence(iteration + 1)
        torch.cuda.synchronize(runtime.device)
        comm.Barrier()
        program.launch()
        torch.cuda.synchronize(runtime.device)
        gather_ns, tail_ns, total_ns = program.timing_ns()
        overlap = program.overlap_timing_ns()
        if args.weighted_return:
            overlap.update(program.weighted_return_timing_ns())
        gather_ms = rank_max(comm, gather_ns / 1.0e6)
        tail_ms = rank_max(comm, tail_ns / 1.0e6)
        total_ms = rank_max(comm, total_ns / 1.0e6)
        if iteration >= args.warmup:
            gather_samples.append(gather_ms)
            tail_samples.append(tail_ms)
            total_samples.append(total_ms)
            for name in overlap_samples:
                value = overlap[name]
                present = rank_max(comm, float(value is not None))
                if present:
                    overlap_samples[name].append(
                        rank_max(
                            comm,
                            -1.0 if value is None else value / 1.0e6,
                        )
                    )

    expected_tokens = tokens
    if args.reader_op == "rms":
        input_f32 = tokens.float()
        expected_tokens = (
            input_f32
            * torch.rsqrt(input_f32.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        ).to(tokens.dtype)
    expected_returned = (
        expected_tokens
        if args.weighted_return
        else expected_tokens.index_select(0, source_rows.to(tokens.device))
    )
    torch.testing.assert_close(
        returned,
        expected_returned,
        rtol=1.0e-2 if args.reader_op == "rms" else 0,
        atol=1.0e-2 if args.reader_op == "rms" else 0,
    )
    status, senders, received_rows, returned_slices, dispatch_ready = (
        buffers.control_state()
    )
    assert status == PoolSliceStatus.OK
    assert senders == runtime.num_pes
    assert returned_slices == runtime.num_pes
    assert dispatch_ready == rounds

    counts = torch.bincount(expert_ids, minlength=num_readers)
    target_pes = torch.arange(num_readers).div(
        args.experts_per_pe, rounding_mode="floor"
    )
    remote_routes = int((counts * (target_pes != runtime.pe)).sum().item())
    local_target_routes = int(
        (counts * (target_pes == runtime.pe)).sum().item()
    )
    remote_readers = int(
        ((counts != 0) & (target_pes != runtime.pe)).sum().item()
    )
    remote_pes = runtime.num_pes - 1
    row_bytes = args.hidden_size * tokens.element_size()
    unique_target_rows = buffers.send_token_counts.cpu().to(torch.int64)
    compact_remote_rows = int(
        unique_target_rows[
            torch.arange(runtime.num_pes) != runtime.pe
        ].sum().item()
    )
    compact_total_rows = int(unique_target_rows.sum().item())
    remote_target_rows = unique_target_rows[
        torch.arange(runtime.num_pes) != runtime.pe
    ]
    nonempty_remote_targets = int((remote_target_rows != 0).sum().item())
    dynamic_groups = _stream_group_counts(
        remote_target_rows,
        row_bytes=row_bytes,
        group_limit=buffers.group_limit,
    )
    metadata_slot_rounds = 2 + (dynamic_groups + 1).div(
        2, rounding_mode="floor"
    )
    dispatch_data_rmas = int(dynamic_groups.sum().item())
    stream_queues_per_source = 2
    model = {
        "protocol": "pool-gather-streaming",
        "pool_blocks": buffers.pool_count,
        "weighted_return": args.weighted_return,
        "weighted_return_sharding": "source-grouped-4",
        "weighted_reduce": "fp32-ilp4",
        "reader_op": args.reader_op,
        "data_group_limit": buffers.group_limit,
        "stream_queues_per_source": stream_queues_per_source,
        "stream_queue_metadata_bytes_per_pe": int(
            (
                metadata_slot_rounds
                * stream_queues_per_source
                * POOL_SLICE_QUEUE_ENTRY_BYTES
            ).sum().item()
        ),
        "remote_routes_per_pe": remote_routes,
        "dispatch_payload_bytes_per_pe": compact_remote_rows * row_bytes,
        "return_payload_bytes_per_pe": (
            compact_remote_rows * row_bytes
            if args.weighted_return
            else remote_routes * row_bytes
        ),
        "local_return_inbox_copy_bytes_per_pe": (
            0
            if args.weighted_return
            else 2 * local_target_routes * row_bytes
        ),
        "descriptor_bytes_per_pe": remote_pes * POOL_SLICE_PUBLISH_BYTES,
        "offset_metadata_bytes_received_per_pe": 0,
        "route_metadata_bytes_sent_remote_per_pe": remote_routes * 8,
        "dispatch_data_rmas_per_pe_current": dispatch_data_rmas,
        "return_data_rmas_per_pe": (
            _source_grouped_return_rmas(
                unique_target_rows.tolist(),
                runtime.pe,
                buffers.pool_count,
            )
            if args.weighted_return
            else remote_readers
        ),
        "merged_signal_words_per_pe": runtime.num_pes,
        "return_batch_signal_words_per_pe": 0,
        "return_fused_signal_updates_per_pe": (
            _source_grouped_return_rmas(
                unique_target_rows.tolist(),
                runtime.pe,
                buffers.pool_count,
            )
            if args.weighted_return
            else 0
        ),
        "metadata_phase_updates_per_pe": remote_pes,
        "data_phase_updates_per_pe": int(dynamic_groups.sum().item()),
        "return_phase_updates_per_pe": (
            0 if args.weighted_return else nonempty_remote_targets
        ),
        "local_token_pool_copy_bytes_per_pe": (
            0 if args.source_preloaded else 2 * args.tokens_per_pe * row_bytes
        ),
        "local_delivery_pack_bytes_per_pe": 0,
        "local_pool_gather_bytes_per_pe": 2 * received_rows * row_bytes,
        "local_reader_copy_bytes_per_pe": (
            0
            if args.in_place_identity
            else 2
            * (
                received_rows
                if args.reader_op == "rms"
                else args.experts_per_pe * expert_capacity_rows
            )
            * row_bytes
        ),
        "received_rows": received_rows,
    }
    if overlap_samples["payload_done"]:
        import statistics

        for name, samples in overlap_samples.items():
            if samples:
                model[f"median_{name}_ms"] = statistics.median(samples)
        model["active_groups"] = buffers.active_group_count()
    if args.print_pool_ctas and args.weighted_return:
        model["pool_cta_return_timing_ns"] = (
            program.weighted_return_cta_timing_ns()
        )
    return Timing(gather_samples, tail_samples, total_samples), model


def main() -> None:
    args = parse_args()
    if min(
        args.tokens_per_pe,
        args.hidden_size,
        args.experts_per_pe,
        args.top_k,
        args.pool_blocks,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        raise ValueError("sizes and iterations must be positive; warmup may be zero")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    comm = MPI.COMM_WORLD
    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    nccl_initialized = False
    try:
        num_readers = runtime.num_pes * args.experts_per_pe
        if args.top_k > num_readers:
            raise ValueError("top-k cannot exceed the global expert count")
        if (
            args.route_placement in {"source-local", "remote-clustered"}
            and args.top_k > args.experts_per_pe
        ):
            raise ValueError(
                "single-slice placements require top-k <= experts-per-pe"
            )
        tokens = nvshmem.empty(
            (args.tokens_per_pe, args.hidden_size), dtype=dtype
        )
        token_values = torch.arange(
            runtime.pe * args.tokens_per_pe * args.hidden_size,
            (runtime.pe + 1) * args.tokens_per_pe * args.hidden_size,
            dtype=torch.float32,
            device=tokens.device,
        ).view_as(tokens)
        tokens.copy_((token_values.remainder(97) - 48).to(dtype))
        global_ids = runtime.pe * args.tokens_per_pe + torch.arange(
            args.tokens_per_pe, dtype=torch.int64
        )
        expert_ids = _balanced_expert_ids(
            global_ids,
            top_k=args.top_k,
            num_pes=runtime.num_pes,
            experts_per_pe=args.experts_per_pe,
            placement=args.route_placement,
        ).reshape(-1)

        pool_result = None
        nccl_result = None
        if args.mode in {"pool", "both"}:
            pool_result = run_pool(args, runtime, comm, tokens, expert_ids)
        if args.mode in {"nccl", "both"}:
            if args.top_k != 1:
                raise ValueError(
                    "the dense NCCL surrogate currently supports only top-k=1"
                )
            initialize_nccl(runtime, comm)
            nccl_initialized = True
            nccl_result = run_nccl(args, runtime, comm, tokens, expert_ids)

        pool_cta_profiles = None
        if args.print_pool_ctas and pool_result is not None:
            local_profile = pool_result[1].pop(
                "pool_cta_return_timing_ns", None
            )
            pool_cta_profiles = comm.gather(local_profile, root=0)

        comm.Barrier()
        if runtime.rank == 0:
            if pool_cta_profiles is not None:
                pool_result[1]["pool_cta_return_timing_ns_by_rank"] = {
                    rank: profile
                    for rank, profile in enumerate(pool_cta_profiles)
                }
            print(
                f"configuration: pes={runtime.num_pes} "
                f"tokens/pe={args.tokens_per_pe} hidden={args.hidden_size} "
                f"experts/pe={args.experts_per_pe} dtype={args.dtype} "
                f"top_k={args.top_k} "
                f"route_placement={args.route_placement} "
                f"protocol={pool_result[1]['protocol'] if pool_result else 'pool-gather-streaming'} "
                f"pool_blocks={args.pool_blocks} in_place_identity="
                f"{args.in_place_identity} source_preloaded="
                f"{args.source_preloaded} weighted_return="
                f"{args.weighted_return} "
                f"reader_op={args.reader_op} "
                f"data_group_limit="
                f"{pool_result[1]['data_group_limit'] if pool_result else args.group_limit} "
                f"queues/source="
                f"{pool_result[1]['stream_queues_per_source'] if pool_result else 2} "
                f"warmup={args.warmup} iterations={args.iterations}"
            )
            if pool_result is not None:
                print_result("pool-slice", *pool_result)
            if nccl_result is not None:
                print_result("nccl-ring", *nccl_result)
            if pool_result is not None and nccl_result is not None:
                pool_ms = pool_result[0].summary()["end_to_end_ms"]
                nccl_ms = nccl_result[0].summary()["end_to_end_ms"]
                print(
                    "latency ratio pool-slice/nccl-ring: "
                    f"{pool_ms / nccl_ms:.3f}x"
                )
    finally:
        if nccl_initialized:
            import torch.distributed as dist

            dist.destroy_process_group()
        nvshmem.finalize()


if __name__ == "__main__":
    main()
