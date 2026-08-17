"""External Triton-distributed low-latency EP dispatch/combine reference.

The optimized layer performs online BF16-to-FP8 dispatch and BF16 weighted
combine.  Untimed local quantization plus global expert counts validate data
and metadata routing without adding a second distributed transport.  Timings
use CUDA events only and remain separate from VDCores' profiling space.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import sys

from mpi4py import MPI
import torch

from ep_baseline_common import (
    add_common_arguments,
    balanced_topk,
    configure_torchrun_environment,
    emit_result,
    rank_max,
    remote_route_count,
    validate_common_arguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triton-distributed low-latency FP8 EP baseline"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--triton-dist-root",
        default=os.environ.get("TRITON_DISTRIBUTED_ROOT", ""),
        help="optional official checkout; its python directory is prepended",
    )
    parser.add_argument("--master-port", type=int, default=29691)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.master_port <= 0:
        raise ValueError("master-port must be positive")

    comm = MPI.COMM_WORLD
    rank, world_size, local_rank, local_world_size = (
        configure_torchrun_environment(comm, args.master_port)
    )
    validate_common_arguments(args, world_size)
    if args.hidden_size % 128 != 0:
        raise ValueError("Triton low-latency FP8 requires hidden-size % 128 == 0")
    if local_world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"{local_world_size} local ranks exceed {torch.cuda.device_count()} GPUs"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if args.triton_dist_root:
        python_root = Path(args.triton_dist_root).resolve() / "python"
        if not (python_root / "triton_dist" / "__init__.py").is_file():
            raise RuntimeError(f"triton_dist sources not found under {python_root}")
        sys.path.insert(0, str(python_root))

    try:
        import triton
        from triton_dist.layers.nvidia import EPLowLatencyAllToAllLayer
        from triton_dist.test.nvidia.ep_a2a_utils import (
            dequant_fp8_bf16,
            quant_bf16_fp8,
        )
        from triton_dist.utils import (
            finalize_distributed,
            initialize_distributed,
        )
    except ImportError as error:
        raise RuntimeError(
            "Triton-distributed is not importable; install the pinned external "
            "checkout and its NVSHMEM4py dependency in an isolated environment"
        ) from error
    if args.triton_dist_root:
        triton_file = Path(triton.__file__).absolute()
        if python_root not in triton_file.parents:
            raise RuntimeError(
                f"Triton resolved to {triton_file}, not the pinned checkout "
                f"under {python_root}; prepend that directory to PYTHONPATH"
            )

    ep_group = None
    layer = None
    try:
        ep_group = initialize_distributed()
        num_experts = world_size * args.experts_per_pe
        tokens = torch.empty(
            (args.tokens_per_pe, args.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        values = torch.arange(
            rank * args.tokens_per_pe * args.hidden_size,
            (rank + 1) * args.tokens_per_pe * args.hidden_size,
            dtype=torch.float32,
            device=device,
        ).view_as(tokens)
        tokens.copy_(((values.remainder(97) - 48) / 48).to(torch.bfloat16))
        topk_idx = balanced_topk(
            rank=rank,
            tokens_per_pe=args.tokens_per_pe,
            num_experts=num_experts,
            experts_per_pe=args.experts_per_pe,
            top_k=args.top_k,
            placement=args.route_placement,
            device=device,
            dtype=torch.int32,
        )
        topk_weights = torch.full(
            (args.tokens_per_pe, args.top_k),
            1.0 / args.top_k,
            dtype=torch.float32,
            device=device,
        )
        remote_routes = remote_route_count(
            topk_idx, rank=rank, experts_per_pe=args.experts_per_pe
        )

        layer = EPLowLatencyAllToAllLayer(
            max_m=args.tokens_per_pe,
            hidden=args.hidden_size,
            topk=args.top_k,
            online_quant_fp8=True,
            rank=rank,
            num_experts=num_experts,
            local_world_size=local_world_size,
            world_size=world_size,
            fp8_gsize=128,
            dtype=torch.bfloat16,
            enable_profiling=False,
        )

        # One untimed pass checks both metadata counts and identity-expert data.
        expected_q, expected_scales = quant_bf16_fp8(tokens)
        expected = dequant_fp8_bf16(expected_q, expected_scales)
        recv_x, recv_scales, expert_recv_count, dispatch_meta = layer.dispatch(
            tokens, None, topk_idx
        )
        expert_template = dequant_fp8_bf16(recv_x, recv_scales)
        combined = layer.combine(
            expert_template,
            topk_idx,
            topk_weights,
            dispatch_meta,
        )
        torch.cuda.synchronize(device)
        gathered_routes = comm.allgather(topk_idx.cpu())
        expected_counts = torch.bincount(
            torch.cat(gathered_routes).view(-1).to(torch.int64),
            minlength=num_experts,
        )[rank * args.experts_per_pe : (rank + 1) * args.experts_per_pe]
        torch.testing.assert_close(
            expert_recv_count.cpu().to(torch.int64),
            expected_counts,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(combined, expected, rtol=0, atol=0)

        dispatch_samples: list[float] = []
        combine_samples: list[float] = []
        total_samples: list[float] = []
        for iteration in range(args.warmup + args.iterations):
            torch.cuda.synchronize(device)
            comm.Barrier()
            start = torch.cuda.Event(enable_timing=True)
            dispatched = torch.cuda.Event(enable_timing=True)
            expert_ready = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)

            start.record()
            recv_x, recv_scales, _, dispatch_meta = layer.dispatch(
                tokens, None, topk_idx
            )
            dispatched.record()
            expert_output = dequant_fp8_bf16(recv_x, recv_scales)
            expert_ready.record()
            combined = layer.combine(
                expert_output,
                topk_idx,
                topk_weights,
                dispatch_meta,
            )
            done.record()
            done.synchronize()

            dispatch_ms = rank_max(comm, start.elapsed_time(dispatched))
            combine_ms = rank_max(comm, expert_ready.elapsed_time(done))
            total_ms = rank_max(
                comm,
                start.elapsed_time(dispatched)
                + expert_ready.elapsed_time(done),
            )
            if iteration >= args.warmup:
                dispatch_samples.append(dispatch_ms)
                combine_samples.append(combine_ms)
                total_samples.append(total_ms)

        torch.testing.assert_close(combined, expected, rtol=0, atol=0)
        dispatch_ms = statistics.median(dispatch_samples)
        combine_ms = statistics.median(combine_samples)
        total_ms = statistics.median(total_samples)
        route_rows = args.tokens_per_pe * args.top_k
        dispatch_row_bytes = (
            args.hidden_size + args.hidden_size // 128 * 4 + 16
        )
        combine_row_bytes = args.hidden_size * 2
        if rank == 0:
            print(
                "triton-distributed-low-latency: "
                f"pes={world_size} tokens/pe={args.tokens_per_pe} "
                f"hidden={args.hidden_size} experts/pe={args.experts_per_pe} "
                f"top_k={args.top_k} dispatch=float8 "
                f"route_placement={args.route_placement} "
                f"warmup={args.warmup} iterations={args.iterations}"
            )
            print(
                "triton-distributed-low-latency timing: "
                f"dispatch={dispatch_ms:.4f} ms combine={combine_ms:.4f} ms "
                f"end-to-end={total_ms:.4f} ms"
            )
            print(
                "triton-distributed-low-latency cost-model: "
                f"logical_dispatch_B/pe={route_rows * dispatch_row_bytes} "
                f"logical_combine_B/pe={route_rows * combine_row_bytes} "
                f"estimated_remote_dispatch_B/pe="
                f"{remote_routes * dispatch_row_bytes} "
                f"estimated_remote_combine_B/pe="
                f"{remote_routes * combine_row_bytes}"
            )
            emit_result(
                {
                    "implementation": "triton-distributed-low-latency",
                    "pes": world_size,
                    "tokens_per_pe": args.tokens_per_pe,
                    "hidden_size": args.hidden_size,
                    "experts_per_pe": args.experts_per_pe,
                    "top_k": args.top_k,
                    "route_placement": args.route_placement,
                    "dispatch_dtype": "float8",
                    "dispatch_ms": dispatch_ms,
                    "combine_ms": combine_ms,
                    "end_to_end_ms": total_ms,
                }
            )
    finally:
        if layer is not None:
            layer.finalize()
        if ep_group is not None:
            finalize_distributed()
        comm.Barrier()


if __name__ == "__main__":
    main()
