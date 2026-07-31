"""Repeatable NVSHMEM-free PoolInst reduction benchmark on local NVLink GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import statistics
import threading

import torch

from dae.local_pool import allocate_local_pool_slices
from dae.pool_slice import PoolSliceStatus, build_pool_slice_copy_program
from dae import runtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="1,2")
    parser.add_argument("--backend", choices=("forward", "multimem"), required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--readers-per-pe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--routing", choices=("clustered", "striped"), default="clustered")
    parser.add_argument("--weighted", action="store_true")
    parser.add_argument(
        "--pool-blocks", type=int, default=0,
        help="PoolInst CTA count; zero selects the profiled local-GB300 policy",
    )
    parser.add_argument(
        "--group-limit", type=int, default=0,
        help="maximum number of streaming data groups; zero selects auto",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--external-output",
        action="store_true",
        help="copy multimem results to a separate tensor instead of consuming them in-pool",
    )
    parser.add_argument(
        "--launch",
        choices=("concurrent", "sequential"),
        default="concurrent",
        help="host enqueue policy; concurrent matches one process/rank per GPU",
    )
    parser.add_argument(
        "--launch-cpus",
        default="",
        help="optional comma-separated CPU pin, one CPU per GPU launch thread",
    )
    args = parser.parse_args()

    devices = [int(value) for value in args.devices.split(",")]
    num_pes = len(devices)
    launch_cpus = (
        [int(value) for value in args.launch_cpus.split(",")]
        if args.launch_cpus
        else []
    )
    if launch_cpus and len(launch_cpus) != num_pes:
        raise ValueError("--launch-cpus must contain one CPU per device")
    if args.routing == "clustered" and args.top_k > args.readers_per_pe:
        raise ValueError("clustered routing requires top-k <= readers-per-pe")
    if args.top_k > num_pes * args.readers_per_pe:
        raise ValueError("top-k exceeds the global reader count")
    buffers = allocate_local_pool_slices(
        devices=devices,
        local_readers=args.readers_per_pe,
        token_capacity=args.tokens,
        route_capacity=args.tokens * args.top_k,
        expert_capacity_rows=num_pes * args.tokens,
        hidden_size=args.hidden,
        pool_blocks=args.pool_blocks or None,
        group_limit=args.group_limit,
        reduction_backend=args.backend,
        in_place_expert_output=True,
    )

    programs = []
    returned = []
    expected_outputs = []
    for pe, (device, pool) in enumerate(zip(devices, buffers)):
        rows = torch.arange(args.tokens, dtype=torch.int64)
        # Keep all top-k experts for a token on one destination PE, matching
        # the production DeepEP comparison shape.
        route_rank = torch.arange(args.top_k, dtype=torch.int64)
        if args.routing == "clustered":
            destination = (rows + pe * args.tokens).remainder(num_pes)
            reader_ids = (
                destination[:, None] * args.readers_per_pe
                + route_rank[None, :]
            ).reshape(-1)
            route_destinations = destination[:, None].expand(-1, args.top_k)
        else:
            route_destinations = route_rank.remainder(num_pes)[None, :].expand(
                args.tokens, -1
            )
            local_reader = route_rank.div(num_pes, rounding_mode="floor")
            reader_ids = (
                route_destinations * args.readers_per_pe + local_reader[None, :]
            ).reshape(-1)
        source_rows = rows.repeat_interleave(args.top_k)
        route_weights = (
            (route_rank.to(torch.float32) + 1) / (args.top_k + 1)
            if args.weighted
            else torch.ones(args.top_k, dtype=torch.float32)
        )
        pool.write_routes(
            reader_ids,
            source_rows=source_rows,
            origin_rows=source_rows,
            route_weights=route_weights.repeat(args.tokens),
        )
        with torch.cuda.device(device):
            values = torch.arange(
                args.tokens * args.hidden, dtype=torch.float32, device=device
            ).reshape(args.tokens, args.hidden)
            pool.token_pool.copy_((values.remainder(251) - 125).to(torch.bfloat16))
            # Match the command semantics: each destination accumulates its
            # reader weights in FP32 and rounds its partial to BF16; multimem
            # then adds destination partials in BF16.
            expected = torch.zeros_like(pool.token_pool)
            bf16_weights = route_weights.to(torch.bfloat16).float()
            for destination_pe in range(num_pes):
                mask = route_destinations == destination_pe
                if mask.any():
                    scale = (
                        mask.to(torch.float32) * bf16_weights[None, :]
                    ).sum(dim=1).to(device)[:, None]
                    partial = (
                        pool.token_pool.float() * scale
                    ).to(torch.bfloat16)
                    expected.add_(partial)
            output = (
                torch.zeros_like(pool.token_pool)
                if args.external_output or args.backend != "multimem"
                else pool.local_reduction_output
            )
            pool.prepare(pool.token_pool, output)
            program = build_pool_slice_copy_program(
                pool, in_place_identity=True, source_preloaded=True
            )
        returned.append(output)
        expected_outputs.append(expected)
        programs.append(program)

    gather_samples: list[float] = []
    return_samples: list[float] = []
    total_samples: list[float] = []
    exact = True
    max_abs_error = 0.0
    launch_barrier = threading.Barrier(num_pes)
    native_launch_barrier = (
        args.launch == "concurrent"
        and hasattr(runtime, "configure_local_launch_barrier")
    )
    if native_launch_barrier:
        runtime.configure_local_launch_barrier(num_pes)

    def launch_one(device: int, program) -> None:
        if launch_cpus:
            os.sched_setaffinity(0, {launch_cpus[devices.index(device)]})
        with torch.cuda.device(device):
            if args.launch == "concurrent" and not native_launch_barrier:
                launch_barrier.wait()
            program.launch()

    executor = (
        ThreadPoolExecutor(max_workers=num_pes)
        if args.launch == "concurrent"
        else None
    )
    try:
        for iteration in range(args.warmup + args.iterations):
            for pool in buffers:
                pool.set_sequence(iteration + 1)
            # launch_dae is asynchronous in the local-pool build. Enqueue
            # every peer before synchronizing so device-side waits progress.
            if executor is None:
                for device, program in zip(devices, programs):
                    launch_one(device, program)
            else:
                futures = [
                    executor.submit(launch_one, device, program)
                    for device, program in zip(devices, programs)
                ]
                for future in futures:
                    future.result()
            for device in devices:
                torch.cuda.synchronize(device)
            timings = [program.timing_ns() for program in programs]
            if iteration >= args.warmup:
                gather_samples.append(max(value[0] for value in timings) / 1e6)
                return_samples.append(max(value[1] for value in timings) / 1e6)
                total_samples.append(max(value[2] for value in timings) / 1e6)
    finally:
        if executor is not None:
            executor.shutdown()
        if native_launch_barrier:
            runtime.configure_local_launch_barrier(0)

    for pe, (pool, output, expected) in enumerate(
        zip(buffers, returned, expected_outputs)
    ):
        status = pool.control_state()[0]
        if status is not PoolSliceStatus.OK:
            raise AssertionError(f"PE {pe} status={status.name}")
        if not torch.equal(output, expected):
            error = (output.float() - expected.float()).abs().max().item()
            exact = False
            max_abs_error = max(max_abs_error, error)
            if not torch.allclose(
                output.float(), expected.float(), rtol=1e-2, atol=1.0
            ):
                raise AssertionError(
                    f"PE {pe} reduction mismatch, max_abs={error}, "
                    f"output0={output[0, :8].float().tolist()}, "
                    f"expected0={expected[0, :8].float().tolist()}"
                )

    print(
        f"backend={args.backend} devices={devices} tokens={args.tokens} "
        f"hidden={args.hidden} top_k={args.top_k} routing={args.routing} "
        f"weighted={args.weighted} pool_blocks={buffers[0].pool_count} "
        f"output={'external' if args.external_output or args.backend != 'multimem' else 'pool'} "
        f"median_ms=(gather={statistics.median(gather_samples):.6f}, "
        f"return={statistics.median(return_samples):.6f}, "
        f"total={statistics.median(total_samples):.6f}) "
        f"correctness={'exact' if exact else 'bf16-close'} "
        f"max_abs={max_abs_error:.6f}"
    )
    if args.profile:
        for pe, program in enumerate(programs):
            raw = program.launcher.profile.cpu().to(torch.int64)
            start = int(raw[program.communication_block, 5].item())
            boundaries = {
                name: (
                    int(raw[program.communication_block, index].item()) - start
                    if int(raw[program.communication_block, index].item()) >= start
                    else None
                )
                for name, index in (
                    ("first_data", 12),
                    ("data_published", 8),
                    ("first_payload", 9),
                    ("metadata_closed", 10),
                    ("first_gather", 17),
                    ("plan_ready", 23),
                    ("first_reader_ready", 24),
                    ("all_readers_ready", 25),
                    ("stream_gather_done", 18),
                    ("dispatch_payload_done", 11),
                    ("gather_ready", 6),
                    ("return_payload_done", 14),
                    ("scatter_done", 16),
                )
            }
            print(
                f"PE {pe} return_profile_ns="
                f"{program.weighted_return_timing_ns()} boundaries_ns={boundaries}"
            )


if __name__ == "__main__":
    main()
