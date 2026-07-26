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
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("pool", "nccl", "both"), default="both")
    parser.add_argument("--symmetric-size", default="1G")
    parser.add_argument(
        "--pack-warps",
        type=int,
        default=0,
        help="pack warps; zero selects the PE/payload-aware policy",
    )
    return parser.parse_args()


def run_pool(
    args: argparse.Namespace,
    runtime,
    comm: MPI.Comm,
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[Timing, dict[str, int | float | str]]:
    num_readers = runtime.num_pes * args.experts_per_pe
    global_tokens = runtime.num_pes * args.tokens_per_pe
    expert_capacity_rows = (
        global_tokens + num_readers - 1
    ) // num_readers
    signals = nvshmem.init_signal_space(runtime.num_pes)
    buffers = allocate_pool_slice(
        signals,
        num_pes=runtime.num_pes,
        my_pe=runtime.pe,
        local_readers=args.experts_per_pe,
        token_capacity=args.tokens_per_pe,
        expert_capacity_rows=expert_capacity_rows,
        hidden_size=args.hidden_size,
        dtype=tokens.dtype,
        pack_warps=args.pack_warps,
    )
    returned = nvshmem.zeros(tokens.shape, dtype=tokens.dtype)
    local_rows = torch.arange(args.tokens_per_pe, dtype=torch.int64)
    buffers.write_routes(
        expert_ids,
        source_rows=local_rows,
        origin_rows=local_rows,
    )
    buffers.prepare(tokens, returned)
    program = build_pool_slice_copy_program(
        buffers,
        benchmark_barrier=nvshmem.benchmark_barrier,
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
    rounds = args.warmup + args.iterations
    for iteration in range(rounds):
        buffers.set_sequence(iteration + 1)
        torch.cuda.synchronize(runtime.device)
        comm.Barrier()
        program.launch()
        torch.cuda.synchronize(runtime.device)
        gather_ns, tail_ns, total_ns = program.timing_ns()
        overlap = program.overlap_timing_ns()
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

    torch.testing.assert_close(returned, tokens, rtol=0, atol=0)
    status, senders, received_rows, returned_slices, observed, group_ready = (
        buffers.control_state()
    )
    assert status == PoolSliceStatus.OK
    assert senders == runtime.num_pes
    assert returned_slices == runtime.num_pes
    assert observed == rounds
    assert group_ready == rounds

    counts = torch.bincount(expert_ids, minlength=num_readers)
    target_pes = torch.arange(num_readers).div(
        args.experts_per_pe, rounding_mode="floor"
    )
    remote_routes = int((counts * (target_pes != runtime.pe)).sum().item())
    remote_readers = int(
        ((counts != 0) & (target_pes != runtime.pe)).sum().item()
    )
    remote_pes = runtime.num_pes - 1
    row_bytes = args.hidden_size * tokens.element_size()
    model = {
        "protocol": "batched-macro",
        "pack_warps": buffers.pack_warps,
        "remote_routes_per_pe": remote_routes,
        "payload_bytes_per_direction_per_pe": remote_routes * row_bytes,
        "descriptor_bytes_per_pe": remote_pes * POOL_SLICE_PUBLISH_BYTES,
        "offset_metadata_bytes_received_per_pe": 0,
        "route_metadata_bytes_sent_remote_per_pe": 0,
        "dispatch_data_rmas_per_pe_current": remote_readers,
        "return_data_rmas_per_pe": remote_readers,
        "merged_signal_words_per_pe": runtime.num_pes,
        "metadata_phase_updates_per_pe": remote_pes,
        "data_phase_updates_per_pe": remote_pes,
        "return_phase_updates_per_pe": remote_pes,
        "local_token_pool_copy_bytes_per_pe": 2 * args.tokens_per_pe * row_bytes,
        "local_delivery_pack_bytes_per_pe": 2 * buffers.active_rows * row_bytes,
        "local_reader_copy_bytes_per_pe": (
            2 * args.experts_per_pe * expert_capacity_rows * row_bytes
        ),
        "received_rows": received_rows,
    }
    if overlap_samples["payload_done"]:
        import statistics

        for name, samples in overlap_samples.items():
            if samples:
                model[f"median_{name}_ms"] = statistics.median(samples)
        payload_sources, dispatch_batches, active_pack_warps = (
            buffers.performance_state()
        )
        model["payload_sources"] = payload_sources
        model["dispatch_batches"] = dispatch_batches
        model["active_pack_warps"] = active_pack_warps
    return Timing(gather_samples, tail_samples, total_samples), model


def main() -> None:
    args = parse_args()
    if min(
        args.tokens_per_pe,
        args.hidden_size,
        args.experts_per_pe,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        raise ValueError("sizes and iterations must be positive; warmup may be zero")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    comm = MPI.COMM_WORLD
    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    nccl_initialized = False
    try:
        num_readers = runtime.num_pes * args.experts_per_pe
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
        expert_ids = global_ids.remainder(num_readers)

        pool_result = None
        nccl_result = None
        if args.mode in {"pool", "both"}:
            pool_result = run_pool(args, runtime, comm, tokens, expert_ids)
        if args.mode in {"nccl", "both"}:
            initialize_nccl(runtime, comm)
            nccl_initialized = True
            nccl_result = run_nccl(args, runtime, comm, tokens, expert_ids)

        comm.Barrier()
        if runtime.rank == 0:
            print(
                f"configuration: pes={runtime.num_pes} "
                f"tokens/pe={args.tokens_per_pe} hidden={args.hidden_size} "
                f"experts/pe={args.experts_per_pe} dtype={args.dtype} "
                f"protocol=batched-macro pack_warps="
                f"{pool_result[1]['pack_warps'] if pool_result else args.pack_warps} "
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
