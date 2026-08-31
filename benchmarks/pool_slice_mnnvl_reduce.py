"""Rank-per-GPU PoolInst reduction over one CUDA Fabric/MNNVL domain."""

from __future__ import annotations

import argparse
import json
import socket
import statistics

from mpi4py import MPI
import torch

from dae.mnnvl_pool import allocate_mnnvl_pool_slice
from dae.pool_slice import PoolSliceStatus, build_pool_slice_copy_program
from ep_baseline_common import balanced_topk, route_digest


BF16_IDENTITY_RTOL = 2.0e-2
BF16_IDENTITY_ATOL = 2.0e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--experts-per-pe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--route-placement",
        choices=("random", "clustered", "spread"),
        default="random",
        help="random global top-k is the production comparison contract",
    )
    parser.add_argument("--route-seed", type=int, default=20260802)
    parser.add_argument(
        "--pool-blocks",
        type=int,
        default=0,
        help="PoolInst CTA count; zero selects the profiled GB300 policy",
    )
    parser.add_argument(
        "--group-limit",
        type=int,
        default=0,
        help="maximum streaming data groups; zero selects the local policy",
    )
    parser.add_argument(
        "--reduction-backend",
        choices=("multimem", "forward", "source_gather"),
        default="multimem",
        help=(
            "valid multi-destination reduction/return implementation; "
            "peer-direct overwrite is intentionally unavailable"
        ),
    )
    parser.add_argument(
        "--static-routes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse the fixed random top-k route handle across launches, "
            "matching NCCL-EP handle creation outside the timed loop"
        ),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--vary-input-every-iteration",
        action="store_true",
        help=(
            "rewrite source values and poison the return buffer before every "
            "launch, then require exact output on every iteration"
        ),
    )
    parser.add_argument(
        "--vdcores-workers",
        action="store_true",
        help=(
            "keep rank zero as PoolInst scheduler and execute every other "
            "pool worker through a cooperative normal compute opcode"
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace, world_size: int) -> None:
    for name in (
        "tokens_per_pe",
        "hidden_size",
        "experts_per_pe",
        "top_k",
        "iterations",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.route_seed < 0:
        raise ValueError("route-seed must be non-negative")
    if args.top_k > world_size * args.experts_per_pe:
        raise ValueError("top-k exceeds the global expert count")
    if args.route_placement == "clustered" and args.top_k > args.experts_per_pe:
        raise ValueError("clustered routing requires top-k <= experts-per-pe")


def routes(
    args: argparse.Namespace, rank: int, world_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = torch.arange(args.tokens_per_pe, dtype=torch.int64)
    topk_idx = balanced_topk(
        rank=rank,
        tokens_per_pe=args.tokens_per_pe,
        num_experts=world_size * args.experts_per_pe,
        experts_per_pe=args.experts_per_pe,
        top_k=args.top_k,
        placement=args.route_placement,
        device=torch.device("cpu"),
        dtype=torch.int64,
        seed=args.route_seed,
    )
    destinations = topk_idx.div(args.experts_per_pe, rounding_mode="floor")
    reader_ids = topk_idx.reshape(-1)
    source_rows = rows.repeat_interleave(args.top_k)
    return reader_ids, source_rows, destinations, topk_idx


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    validate_args(args, world_size)

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    if local_comm.Get_size() > torch.cuda.device_count():
        raise RuntimeError("local MPI ranks exceed visible CUDA devices")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    pool = allocate_mnnvl_pool_slice(
        comm=comm,
        device=local_rank,
        local_readers=args.experts_per_pe,
        token_capacity=args.tokens_per_pe,
        route_capacity=args.tokens_per_pe * args.top_k,
        expert_capacity_rows=world_size * args.tokens_per_pe,
        hidden_size=args.hidden_size,
        pool_blocks=args.pool_blocks or None,
        group_limit=args.group_limit,
        reduction_backend=args.reduction_backend,
        static_routes=args.static_routes,
        in_place_expert_output=True,
    )

    reader_ids, source_rows, destinations, topk_idx = routes(
        args, rank, world_size
    )
    global_route_digest = route_digest(comm, topk_idx)
    route_weights = torch.full(
        (args.top_k,), 1.0 / args.top_k, dtype=torch.float32
    )
    pool.write_routes(
        reader_ids,
        source_rows=source_rows,
        origin_rows=source_rows,
        route_weights=route_weights.repeat(args.tokens_per_pe),
    )

    with torch.cuda.device(device):
        value_begin = rank * args.tokens_per_pe * args.hidden_size
        values = torch.arange(
            value_begin,
            value_begin + args.tokens_per_pe * args.hidden_size,
            dtype=torch.float32,
            device=device,
        ).reshape(args.tokens_per_pe, args.hidden_size)
        def load_input(pattern: int) -> None:
            pool.token_pool.copy_(
                (((values + pattern * 17).remainder(97) - 48) / 48).to(
                    torch.bfloat16
                )
            )

        load_input(0)

        expected = torch.zeros_like(pool.token_pool)
        bf16_weights = route_weights.to(torch.bfloat16).float()

        def refresh_expected() -> None:
            expected.zero_()
            for destination_pe in range(world_size):
                mask = destinations == destination_pe
                if mask.any():
                    scale = (
                        mask.to(torch.float32) * bf16_weights[None, :]
                    ).sum(dim=1).to(device)[:, None]
                    partial = (pool.token_pool.float() * scale).to(torch.bfloat16)
                    expected.add_(partial)

        refresh_expected()

        output = (
            pool.local_reduction_output
            if args.reduction_backend == "multimem"
            else torch.empty_like(pool.token_pool)
        )
        pool.prepare(pool.token_pool, output)
        program = build_pool_slice_copy_program(
            pool,
            in_place_identity=True,
            source_preloaded=True,
            vdcores_workers=args.vdcores_workers,
        )
        torch.cuda.synchronize(device)
    comm.Barrier()

    gather_samples: list[float] = []
    return_samples: list[float] = []
    total_samples: list[float] = []
    for iteration in range(args.warmup + args.iterations):
        if args.vary_input_every_iteration:
            with torch.cuda.device(device):
                load_input(iteration + 1)
                refresh_expected()
                output.fill_(32.0 + iteration)
                torch.cuda.synchronize(device)
        pool.set_sequence(iteration + 1)
        comm.Barrier()
        program.launch()
        torch.cuda.synchronize(device)
        gather_ns, return_ns, total_ns = program.timing_ns()
        gather_ms = comm.allreduce(gather_ns / 1e6, op=MPI.MAX)
        return_ms = comm.allreduce(return_ns / 1e6, op=MPI.MAX)
        total_ms = comm.allreduce(total_ns / 1e6, op=MPI.MAX)
        if rank == 0 and iteration >= args.warmup:
            gather_samples.append(float(gather_ms))
            return_samples.append(float(return_ms))
            total_samples.append(float(total_ms))
        if args.vary_input_every_iteration:
            local_iteration_close = torch.allclose(
                output.float(),
                expected.float(),
                rtol=BF16_IDENTITY_RTOL,
                atol=BF16_IDENTITY_ATOL,
            )
            all_iteration_close = bool(
                comm.allreduce(int(local_iteration_close), op=MPI.MIN)
            )
            if not all_iteration_close:
                local_iteration_max_abs = float(
                    (output.float() - expected.float()).abs().max().item()
                )
                iteration_max_abs = float(
                    comm.allreduce(local_iteration_max_abs, op=MPI.MAX)
                )
                raise AssertionError(
                    "changing-input BF16 correctness failed at iteration "
                    f"{iteration}: max_abs={iteration_max_abs}"
                )

    raw_profile = program.launcher.profile.cpu().to(torch.int64)
    profile_start = int(
        raw_profile[program.communication_block, 5].item()
    )
    local_boundaries: dict[str, int | None] = {}
    for name, index in (
        ("gather_ready", 6),
        ("data_published", 8),
        ("first_payload", 9),
        ("metadata_closed", 10),
        ("dispatch_payload_done", 11),
        ("first_data", 12),
        ("compute_ready", 13),
        ("return_payload_done", 14),
        ("return_signals_closed", 15),
        ("scatter_done", 16),
        ("first_gather", 17),
        ("stream_gather_done", 18),
        ("plan_ready", 23),
        ("first_reader_ready", 24),
        ("all_readers_ready", 25),
    ):
        value = int(raw_profile[program.communication_block, index].item())
        local_boundaries[name] = (
            value - profile_start if value >= profile_start else None
        )
    boundary_records = comm.gather(local_boundaries, root=0)

    local_block_events: dict[str, dict[str, int | None]] = {}
    for name, index in (
        ("return_reduce_start", 19),
        ("return_reduce_done", 20),
        ("first_return_put", 21),
        ("return_cta_done", 22),
    ):
        values = [
            int(value) - profile_start
            for value in raw_profile[:, index].tolist()
            if int(value) >= profile_start
        ]
        local_block_events[name] = {
            "min_ns": min(values) if values else None,
            "max_ns": max(values) if values else None,
            "count": len(values),
        }
    block_event_records = comm.gather(local_block_events, root=0)

    status = pool.control_state()[0]
    local_status_ok = status == PoolSliceStatus.OK
    local_exact = torch.equal(output, expected)
    local_max_abs = float(
        (output.float() - expected.float()).abs().max().item()
    )
    local_close = torch.allclose(
        output.float(),
        expected.float(),
        rtol=BF16_IDENTITY_RTOL,
        atol=BF16_IDENTITY_ATOL,
    )
    all_status_ok = bool(comm.allreduce(int(local_status_ok), op=MPI.MIN))
    all_exact = bool(comm.allreduce(int(local_exact), op=MPI.MIN))
    all_close = bool(comm.allreduce(int(local_close), op=MPI.MIN))
    max_abs = float(comm.allreduce(local_max_abs, op=MPI.MAX))
    if not all_status_ok or not all_close:
        raise AssertionError(
            f"MNNVL PoolSlice failed: local_status={status.name}, "
            f"all_status_ok={all_status_ok}, max_abs={max_abs}"
        )

    hostnames = comm.gather(socket.gethostname(), root=0)
    if rank == 0:
        profile_rank_max_ns = {}
        for name in local_boundaries:
            values = [
                record[name]
                for record in boundary_records
                if record[name] is not None
            ]
            profile_rank_max_ns[name] = max(values) if values else None
        profile_block_global_ns = {}
        for name in local_block_events:
            minimums = [
                record[name]["min_ns"]
                for record in block_event_records
                if record[name]["min_ns"] is not None
            ]
            maximums = [
                record[name]["max_ns"]
                for record in block_event_records
                if record[name]["max_ns"] is not None
            ]
            counts = [record[name]["count"] for record in block_event_records]
            profile_block_global_ns[name] = {
                "min_ns": min(minimums) if minimums else None,
                "max_ns": max(maximums) if maximums else None,
                "count_min": min(counts),
                "count_max": max(counts),
            }
        result = {
            "implementation": f"vdcores-pool-fabric-{args.reduction_backend}",
            "precision": "bfloat16",
            "activation": (
                "bf16-deepseek-v3-width-7168"
                if args.hidden_size == 7168
                else "custom"
            ),
            "top_k": args.top_k,
            "experts_per_pe": args.experts_per_pe,
            "tokens_per_pe": args.tokens_per_pe,
            "gpus": world_size,
            "hosts": list(dict.fromkeys(hostnames)),
            "route_placement": args.route_placement,
            "route_seed": args.route_seed,
            "route_digest": global_route_digest,
            "route_weights": "uniform-1/top-k",
            "pool_blocks": pool.pool_count,
            "group_limit": pool.group_limit,
            "reduction_backend": args.reduction_backend,
            "static_routes": args.static_routes,
            "vdcores_workers": args.vdcores_workers,
            "dispatch_ms": statistics.median(gather_samples),
            "combine_or_tail_ms": statistics.median(return_samples),
            "end_to_end_ms": statistics.median(total_samples),
            "end_to_end_min_ms": min(total_samples),
            "end_to_end_max_ms": max(total_samples),
            "timing": "PoolInst-internal-events-rank-max-median",
            "payload_transport": (
                "CUDA-Fabric-VMM-MNNVL-multimem"
                if args.reduction_backend == "multimem"
                else (
                    "CUDA-Fabric-VMM-MNNVL-source-gather"
                    if args.reduction_backend == "source_gather"
                    else "CUDA-Fabric-VMM-MNNVL-peer-forward"
                )
            ),
            "bootstrap_transport": "MPI-TCP-handle-exchange-only",
            "correctness": "exact" if all_exact else "bf16-close",
            "correctness_rtol": BF16_IDENTITY_RTOL,
            "correctness_atol": BF16_IDENTITY_ATOL,
            "max_abs": max_abs,
            "arena_mapped_bytes": pool._mnnvl_arena_mapped_bytes,
            "multicast_mapped_bytes": pool._mnnvl_multicast_mapped_bytes,
            "profile_rank_max_ns": profile_rank_max_ns,
            "profile_block_global_ns": profile_block_global_ns,
        }
        print(
            f"backend=fabric-{args.reduction_backend} gpus={world_size} "
            f"tokens={args.tokens_per_pe} hidden={args.hidden_size} "
            f"top_k={args.top_k} routing={args.route_placement} "
            f"route_seed={args.route_seed} "
            f"static_routes={args.static_routes} "
            f"pool_blocks={pool.pool_count} "
            f"median_ms=(gather={result['dispatch_ms']:.6f}, "
            f"return={result['combine_or_tail_ms']:.6f}, "
            f"total={result['end_to_end_ms']:.6f}) "
            f"correctness={result['correctness']} max_abs={max_abs:.6f}",
            flush=True,
        )
        print("mnnvl-pool-json: " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
