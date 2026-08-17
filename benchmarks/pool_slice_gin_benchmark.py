"""Weighted PoolInst EP over the compile-time NCCL GIN/GDAKI backend.

This benchmark owns only host setup, route generation, validation, and result
aggregation. The measured dispatch/combine path is one VDCores PoolInst
program, and timing comes exclusively from its internal global-timer events.
"""

from __future__ import annotations

import argparse
import json
import statistics

import torch
from mpi4py import MPI

from dae.gin import GinRuntime, allocate_pool_slice_gin
from dae.pool_slice import (
    POOL_SLICE_RAW_SGL,
    POOL_SLICE_RAW_SGL_WIDTH,
    PoolSliceStatus,
    build_pool_slice_copy_program,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--experts-per-pe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--route-placement",
        choices=("clustered", "source-local", "remote-clustered", "spread"),
        default="clustered",
    )
    parser.add_argument("--pool-blocks", type=int, default=64)
    parser.add_argument("--data-groups", type=int, default=0)
    parser.add_argument("--gin-contexts", type=int, default=16)
    parser.add_argument("--gin-queue-depth", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    return parser.parse_args()


def balanced_expert_ids(
    global_rows: torch.Tensor,
    *,
    tokens_per_pe: int,
    top_k: int,
    num_pes: int,
    experts_per_pe: int,
    placement: str,
) -> torch.Tensor:
    route = torch.arange(top_k, dtype=torch.int64)
    if placement == "clustered":
        return (
            global_rows[:, None] * top_k + route[None, :]
        ).remainder(num_pes * experts_per_pe)

    source_pe = global_rows.div(tokens_per_pe, rounding_mode="floor")
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


def weighted_identity_reference(
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
    *,
    num_pes: int,
    experts_per_pe: int,
    top_k: int,
) -> torch.Tensor:
    target_pes = expert_ids.view(tokens.shape[0], top_k).div(
        experts_per_pe, rounding_mode="floor"
    ).to(tokens.device)
    values = tokens.float()
    weight = torch.tensor(
        1.0 / top_k, dtype=tokens.dtype, device=tokens.device
    ).float()
    total = torch.zeros_like(values)
    for target_pe in range(num_pes):
        partial = torch.zeros_like(values)
        for route in range(top_k):
            active = (target_pes[:, route] == target_pe).unsqueeze(1)
            partial += torch.where(active, values * weight, 0.0)
        total += partial.to(tokens.dtype).float()
    return total.to(tokens.dtype)


def rank_max(comm: MPI.Comm, value: float) -> float:
    return float(comm.allreduce(float(value), op=MPI.MAX))


def main() -> None:
    args = parse_args()
    if min(
        args.tokens_per_pe,
        args.hidden_size,
        args.experts_per_pe,
        args.top_k,
        args.pool_blocks,
        args.gin_contexts,
        args.gin_queue_depth,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        raise ValueError("sizes and iterations must be positive")

    transport = GinRuntime.init()
    try:
        num_readers = transport.num_pes * args.experts_per_pe
        if args.top_k > num_readers:
            raise ValueError("top-k exceeds the global expert count")
        if (
            args.route_placement in {"source-local", "remote-clustered"}
            and args.top_k > args.experts_per_pe
        ):
            raise ValueError("single-slice placement exceeds local experts")
        routes = args.tokens_per_pe * args.top_k
        allocation = allocate_pool_slice_gin(
            transport,
            local_readers=args.experts_per_pe,
            token_capacity=args.tokens_per_pe,
            route_capacity=routes,
            expert_capacity_rows=transport.num_pes * args.tokens_per_pe,
            hidden_size=args.hidden_size,
            group_limit=args.data_groups,
            pool_blocks=args.pool_blocks,
            in_place_expert_output=True,
            context_count=args.gin_contexts,
            queue_depth=args.gin_queue_depth,
        )
        buffers = allocation.buffers
        tokens = buffers.token_pool
        returned = allocation.returned
        values = torch.arange(
            transport.pe * args.tokens_per_pe * args.hidden_size,
            (transport.pe + 1) * args.tokens_per_pe * args.hidden_size,
            dtype=torch.float32,
            device=transport.device,
        ).view_as(tokens)
        tokens.copy_((values.remainder(97) - 48).to(torch.bfloat16))
        global_rows = transport.pe * args.tokens_per_pe + torch.arange(
            args.tokens_per_pe, dtype=torch.int64
        )
        expert_ids = balanced_expert_ids(
            global_rows,
            tokens_per_pe=args.tokens_per_pe,
            top_k=args.top_k,
            num_pes=transport.num_pes,
            experts_per_pe=args.experts_per_pe,
            placement=args.route_placement,
        ).reshape(-1)
        source_rows = torch.arange(
            args.tokens_per_pe, dtype=torch.int64
        ).repeat_interleave(args.top_k)
        buffers.write_routes(
            expert_ids,
            source_rows=source_rows,
            origin_rows=source_rows,
            route_weights=torch.full((routes,), 1.0 / args.top_k),
        )
        buffers.prepare(tokens, returned)
        program = build_pool_slice_copy_program(
            buffers,
            benchmark_barrier=transport.benchmark_barrier,
            in_place_identity=True,
            source_preloaded=True,
        )

        samples: list[float] = []
        gather_samples: list[float] = []
        tail_samples: list[float] = []
        phase_samples: dict[str, list[float]] = {}
        rounds = args.warmup + args.iterations
        for iteration in range(rounds):
            buffers.set_sequence(iteration + 1)
            transport.benchmark_barrier()
            program.launch()
            torch.cuda.synchronize(transport.device)
            gather_ns, tail_ns, total_ns = program.timing_ns()
            overlap = program.overlap_timing_ns()
            overlap.update(program.weighted_return_timing_ns())
            total_ms = rank_max(transport.mpi, total_ns / 1.0e6)
            gather_ms = rank_max(transport.mpi, gather_ns / 1.0e6)
            tail_ms = rank_max(transport.mpi, tail_ns / 1.0e6)
            if iteration >= args.warmup:
                samples.append(total_ms)
                gather_samples.append(gather_ms)
                tail_samples.append(tail_ms)
                for name, value in overlap.items():
                    if value is not None:
                        phase_samples.setdefault(name, []).append(
                            rank_max(transport.mpi, value / 1.0e6)
                        )

        expected = weighted_identity_reference(
            tokens,
            expert_ids,
            num_pes=transport.num_pes,
            experts_per_pe=args.experts_per_pe,
            top_k=args.top_k,
        )
        torch.testing.assert_close(returned, expected, rtol=0, atol=0)
        status, senders, _, returned_slices, dispatch_ready = (
            buffers.control_state()
        )
        if (
            status != PoolSliceStatus.OK
            or senders != transport.num_pes
            or returned_slices != transport.num_pes
            or dispatch_ready != rounds
        ):
            raise AssertionError("PoolInst control state did not retire cleanly")

        result = {
            "transport": (
                "nccl-gin-gdaki-raw-sgl"
                if POOL_SLICE_RAW_SGL
                else "nccl-gin-gdaki-aggregated"
            ),
            "sgl_width": (
                POOL_SLICE_RAW_SGL_WIDTH if POOL_SLICE_RAW_SGL else 1
            ),
            "pes": transport.num_pes,
            "tokens_per_pe": args.tokens_per_pe,
            "hidden_size": args.hidden_size,
            "experts_per_pe": args.experts_per_pe,
            "top_k": args.top_k,
            "route_placement": args.route_placement,
            "pool_blocks": args.pool_blocks,
            "data_group_limit": buffers.group_limit,
            "requested_contexts": args.gin_contexts,
            "actual_contexts": allocation.actual_contexts,
            "queue_depth": args.gin_queue_depth,
            "arena_mib": allocation.arena.numel() / (1024 * 1024),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "median_ms": statistics.median(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "median_gather_ms": statistics.median(gather_samples),
            "median_tail_ms": statistics.median(tail_samples),
            "phase_median_ms": {
                name: statistics.median(values)
                for name, values in phase_samples.items()
            },
            "correct": True,
        }
        transport.mpi.Barrier()
        if transport.rank == 0:
            print("GIN_RESULT " + json.dumps(result, sort_keys=True))
    finally:
        transport.close()


if __name__ == "__main__":
    main()
