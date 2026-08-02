"""External UCCL-EP low-latency dispatch/combine reference.

UCCL and its CPU RDMA proxy threads remain external to VDCores.  Launch one
MPI rank per GPU/node.  CUDA events time only UCCL dispatch and combine; they
never enter the VDCores internal profiling space.
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import statistics
import sys

from mpi4py import MPI
import torch
import torch.distributed as dist

from ep_baseline_common import (
    add_common_arguments,
    balanced_topk,
    configure_torchrun_environment,
    emit_result,
    rank_max,
    remote_route_count,
    route_digest,
    validate_common_arguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UCCL-EP low-latency dispatch/combine baseline"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--dispatch-dtype",
        choices=("bfloat16", "float8"),
        default="bfloat16",
        help="UCCL dispatch wire format; combine consumes BF16",
    )
    parser.add_argument(
        "--uccl-root",
        default=os.environ.get("UCCL_EP_ROOT", ""),
        help="official UCCL checkout containing ep/deep_ep_wrapper",
    )
    parser.add_argument("--master-port", type=int, default=29681)
    return parser.parse_args()


def _import_uccl_wrapper(uccl_root: str):
    if not uccl_root:
        raise RuntimeError("set UCCL_EP_ROOT or pass --uccl-root")
    wrapper_root = Path(uccl_root).resolve() / "ep" / "deep_ep_wrapper"
    if not (wrapper_root / "deep_ep" / "__init__.py").is_file():
        raise RuntimeError(f"UCCL DeepEP wrapper not found under {wrapper_root}")
    sys.path.insert(0, str(wrapper_root))
    try:
        module = importlib.import_module("deep_ep")
    except ImportError as error:
        raise RuntimeError(
            "UCCL-EP is not importable; build/install uccl.ep externally, "
            "then point --uccl-root at the matching official checkout"
        ) from error
    module_file = Path(module.__file__).resolve()
    if wrapper_root not in module_file.parents:
        raise RuntimeError(
            f"import resolved to {module_file}, not UCCL wrapper {wrapper_root}"
        )
    utils = importlib.import_module("deep_ep.utils")
    return module, utils.per_token_cast_back


def _dequantize_dispatch(recv_x, hidden: int, cast_back):
    values, scales = recv_x
    return cast_back(
        values.view(-1, hidden),
        scales.contiguous().view(-1, hidden // 128),
    ).view(values.shape)


def main() -> None:
    args = parse_args()
    if args.master_port <= 0:
        raise ValueError("master-port must be positive")

    comm = MPI.COMM_WORLD
    rank, world_size, local_rank, local_world_size = (
        configure_torchrun_environment(comm, args.master_port)
    )
    validate_common_arguments(args, world_size)
    if local_world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"{local_world_size} local ranks exceed {torch.cuda.device_count()} GPUs"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    buffer = None
    try:
        uccl_deep_ep, per_token_cast_back = _import_uccl_wrapper(args.uccl_root)
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
            seed=args.route_seed,
        )
        global_route_digest = route_digest(comm, topk_idx)
        topk_weights = torch.full(
            (args.tokens_per_pe, args.top_k),
            1.0 / args.top_k,
            dtype=torch.float32,
            device=device,
        )
        remote_routes = remote_route_count(
            topk_idx, rank=rank, experts_per_pe=args.experts_per_pe
        )

        rdma_bytes = uccl_deep_ep.Buffer.get_low_latency_rdma_size_hint(
            args.tokens_per_pe,
            args.hidden_size,
            world_size,
            num_experts,
        )
        buffer = uccl_deep_ep.Buffer(
            group=dist.group.WORLD,
            num_rdma_bytes=rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=args.experts_per_pe,
            allow_nvlink_for_low_latency_mode=False,
            explicitly_destroy=True,
            is_intranode=(world_size == local_world_size),
        )

        use_fp8 = args.dispatch_dtype == "float8"

        # Establish correctness and create the identity-expert output once.
        recv_x, _, handle, _, _ = buffer.low_latency_dispatch(
            tokens,
            topk_idx,
            args.tokens_per_pe,
            num_experts,
            use_fp8=use_fp8,
            async_finish=False,
            return_recv_hook=False,
        )
        expert_template = (
            _dequantize_dispatch(
                recv_x, args.hidden_size, per_token_cast_back
            )
            if use_fp8
            else recv_x
        )
        combined, _, _ = buffer.low_latency_combine(
            expert_template,
            topk_idx,
            topk_weights,
            handle,
            async_finish=False,
            return_recv_hook=False,
        )
        torch.cuda.synchronize(device)
        torch.testing.assert_close(combined, tokens, rtol=3.0e-2, atol=3.0e-2)

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
            recv_x, _, handle, _, _ = buffer.low_latency_dispatch(
                tokens,
                topk_idx,
                args.tokens_per_pe,
                num_experts,
                use_fp8=use_fp8,
                async_finish=False,
                return_recv_hook=False,
            )
            dispatched.record()
            expert_output = (
                _dequantize_dispatch(
                    recv_x, args.hidden_size, per_token_cast_back
                )
                if use_fp8
                else recv_x
            )
            expert_ready.record()
            combined, _, _ = buffer.low_latency_combine(
                expert_output,
                topk_idx,
                topk_weights,
                handle,
                async_finish=False,
                return_recv_hook=False,
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

        torch.testing.assert_close(combined, tokens, rtol=3.0e-2, atol=3.0e-2)
        dispatch_ms = statistics.median(dispatch_samples)
        combine_ms = statistics.median(combine_samples)
        total_ms = statistics.median(total_samples)
        route_rows = args.tokens_per_pe * args.top_k
        dispatch_row_bytes = args.hidden_size * (1 if use_fp8 else 2) + 16
        if use_fp8:
            dispatch_row_bytes += args.hidden_size // 128 * 4
        combine_row_bytes = args.hidden_size * 2
        if rank == 0:
            print(
                "uccl-ep-low-latency: "
                f"pes={world_size} tokens/pe={args.tokens_per_pe} "
                f"hidden={args.hidden_size} experts/pe={args.experts_per_pe} "
                f"top_k={args.top_k} dispatch={args.dispatch_dtype} "
                f"route_placement={args.route_placement} "
                f"route_seed={args.route_seed} "
                f"warmup={args.warmup} iterations={args.iterations}"
            )
            print(
                "uccl-ep-low-latency timing: "
                f"dispatch={dispatch_ms:.4f} ms combine={combine_ms:.4f} ms "
                f"end-to-end={total_ms:.4f} ms"
            )
            print(
                "uccl-ep-low-latency cost-model: "
                f"logical_dispatch_B/pe={route_rows * dispatch_row_bytes} "
                f"logical_combine_B/pe={route_rows * combine_row_bytes} "
                f"estimated_remote_dispatch_B/pe="
                f"{remote_routes * dispatch_row_bytes} "
                f"estimated_remote_combine_B/pe="
                f"{remote_routes * combine_row_bytes} "
                f"rdma_buffer_B/pe={rdma_bytes}"
            )
            emit_result(
                {
                    "implementation": "uccl-ep-low-latency",
                    "pes": world_size,
                    "tokens_per_pe": args.tokens_per_pe,
                    "hidden_size": args.hidden_size,
                    "experts_per_pe": args.experts_per_pe,
                    "top_k": args.top_k,
                    "route_placement": args.route_placement,
                    "route_seed": args.route_seed,
                    "route_digest": global_route_digest,
                    "dispatch_dtype": args.dispatch_dtype,
                    "dispatch_ms": dispatch_ms,
                    "combine_ms": combine_ms,
                    "end_to_end_ms": total_ms,
                }
            )
    finally:
        if buffer is not None:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
        comm.Barrier()


if __name__ == "__main__":
    main()
