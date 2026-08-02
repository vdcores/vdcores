"""External DeepEP V1 low-latency dispatch/combine reference.

This benchmark intentionally does not import or modify the VDCores runtime.
Install DeepEP V1 in an external environment, place it on ``PYTHONPATH``, and
launch one MPI rank per GPU.  Timings use CUDA events and an MPI rank-maximum;
VDCores measurements continue to use only their internal ``g_events`` space.
"""

from __future__ import annotations

import argparse
import os
import statistics

from mpi4py import MPI
import torch

from ep_baseline_common import balanced_topk, route_digest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--experts-per-pe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--route-placement",
        choices=(
            "random",
            "clustered",
            "source-local",
            "remote-clustered",
            "spread",
        ),
        default="random",
        help=(
            "cluster each token's routes on one balanced PE, force that PE "
            "local/remote to the source, or spread routes round-robin "
            "across PEs"
        ),
    )
    parser.add_argument("--route-seed", type=int, default=20260802)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--dispatch-dtype",
        choices=("bfloat16", "float8"),
        default="bfloat16",
        help="DeepEP dispatch wire format; combine always consumes BF16",
    )
    parser.add_argument(
        "--qps-per-rank",
        type=int,
        default=0,
        help="IBGDA RC QPs per rank; zero uses experts-per-pe",
    )
    parser.add_argument(
        "--qp-depth",
        type=int,
        default=1024,
        help="NVSHMEM QP depth configured before DeepEP initializes",
    )
    parser.add_argument(
        "--allow-nvlink",
        action="store_true",
        help="allow DeepEP's low-latency NVLink path (off for one-GPU nodes)",
    )
    parser.add_argument(
        "--allow-mnnvl",
        action="store_true",
        help="allow DeepEP's multi-node NVLink path on one MNNVL fabric",
    )
    return parser.parse_args()


def _rank_max(comm: MPI.Comm, value: float) -> float:
    return float(comm.allreduce(float(value), op=MPI.MAX))


def _balanced_topk(
    *,
    rank: int,
    tokens_per_pe: int,
    num_experts: int,
    experts_per_pe: int,
    top_k: int,
    placement: str,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    return balanced_topk(
        rank=rank,
        tokens_per_pe=tokens_per_pe,
        num_experts=num_experts,
        experts_per_pe=experts_per_pe,
        top_k=top_k,
        placement=placement,
        device=device,
        seed=seed,
    )


def main() -> None:
    args = parse_args()
    for name in (
        "tokens_per_pe",
        "hidden_size",
        "experts_per_pe",
        "top_k",
        "iterations",
        "qp_depth",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if (
        args.route_placement in {"source-local", "remote-clustered"}
        and args.top_k > args.experts_per_pe
    ):
        raise ValueError(
            "single-PE placements require top-k <= experts-per-pe"
        )
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    if args.route_seed < 0:
        raise ValueError("route-seed must be non-negative")

    comm = MPI.COMM_WORLD
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_size = local_comm.Get_size()
    if local_size > torch.cuda.device_count():
        raise RuntimeError(
            f"{local_size} local MPI ranks exceed {torch.cuda.device_count()} GPUs"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Import only after selecting the device and after the launcher has staged
    # the external DeepEP package on PYTHONPATH.
    try:
        import deep_ep
    except ImportError as error:
        raise RuntimeError(
            "DeepEP V1 is not importable; build it externally and add its "
            "source plus extension directory to PYTHONPATH"
        ) from error

    world_size = comm.Get_size()
    rank = comm.Get_rank()
    num_experts = world_size * args.experts_per_pe
    if args.top_k > num_experts:
        raise ValueError("top-k cannot exceed the global expert count")
    qps_per_rank = args.qps_per_rank or args.experts_per_pe
    if qps_per_rank <= 0:
        raise ValueError("qps-per-rank must be positive")
    if qps_per_rank < args.experts_per_pe:
        raise ValueError(
            "DeepEP V1 low-latency mode requires at least one RC QP per "
            "local expert"
        )
    os.environ["NVSHMEM_QP_DEPTH"] = str(args.qp_depth)

    tokens = torch.empty(
        (args.tokens_per_pe, args.hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )
    token_values = torch.arange(
        rank * args.tokens_per_pe * args.hidden_size,
        (rank + 1) * args.tokens_per_pe * args.hidden_size,
        dtype=torch.float32,
        device=device,
    ).view_as(tokens)
    tokens.copy_((token_values.remainder(97) - 48).to(torch.bfloat16))
    topk_idx = _balanced_topk(
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

    rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
        args.tokens_per_pe,
        args.hidden_size,
        world_size,
        num_experts,
    )
    buffer = deep_ep.Buffer(
        group=None,
        comm=comm,
        num_rdma_bytes=rdma_bytes,
        low_latency_mode=True,
        num_qps_per_rank=qps_per_rank,
        allow_nvlink_for_low_latency_mode=args.allow_nvlink,
        allow_mnnvl=args.allow_mnnvl,
        explicitly_destroy=True,
    )
    buffer.clean_low_latency_buffer(
        args.tokens_per_pe, args.hidden_size, num_experts
    )

    use_fp8 = args.dispatch_dtype == "float8"
    fp8_expert_output = (
        torch.zeros(
            (
                args.experts_per_pe,
                args.tokens_per_pe * world_size,
                args.hidden_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        if use_fp8
        else None
    )
    dispatch_samples: list[float] = []
    combine_samples: list[float] = []
    total_samples: list[float] = []
    combined = None
    try:
        rounds = args.warmup + args.iterations
        for iteration in range(rounds):
            torch.cuda.synchronize(device)
            comm.Barrier()
            start = torch.cuda.Event(enable_timing=True)
            dispatched = torch.cuda.Event(enable_timing=True)
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

            if use_fp8:
                # A production expert GEMM consumes FP8 dispatch and emits
                # BF16.  Keep that compute outside this communication-only
                # reference while preserving DeepEP's actual combine shape.
                expert_output = fp8_expert_output
            else:
                # Identity is the communication-only stand-in for expert
                # compute, matching the VDCores transport harness.
                expert_output = recv_x

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

            dispatch_ms = _rank_max(comm, start.elapsed_time(dispatched))
            combine_ms = _rank_max(comm, dispatched.elapsed_time(done))
            total_ms = _rank_max(comm, start.elapsed_time(done))
            if iteration >= args.warmup:
                dispatch_samples.append(dispatch_ms)
                combine_samples.append(combine_ms)
                total_samples.append(total_ms)

        if not use_fp8:
            torch.testing.assert_close(
                combined,
                tokens,
                rtol=2.0e-2,
                atol=2.0e-1,
            )

        route_rows = args.tokens_per_pe * args.top_k
        dispatch_row_bytes = args.hidden_size * (1 if use_fp8 else 2)
        if use_fp8:
            dispatch_row_bytes += args.hidden_size // 128 * 4 + 16
        combine_row_bytes = args.hidden_size * 2
        remote_routes = int(
            (
                topk_idx.div(args.experts_per_pe, rounding_mode="floor")
                != rank
            ).sum().item()
        )
        if rank == 0:
            print(
                "deepep-v1-low-latency: "
                f"pes={world_size} tokens/pe={args.tokens_per_pe} "
                f"hidden={args.hidden_size} experts/pe={args.experts_per_pe} "
                f"top_k={args.top_k} dispatch={args.dispatch_dtype} "
                f"route_placement={args.route_placement} "
                f"route_seed={args.route_seed} "
                f"qps/rank={qps_per_rank} qp_depth={args.qp_depth} "
                f"warmup={args.warmup} iterations={args.iterations}"
            )
            print(
                "deepep-v1-low-latency timing: "
                f"dispatch={statistics.median(dispatch_samples):.4f} ms "
                f"combine={statistics.median(combine_samples):.4f} ms "
                f"end-to-end={statistics.median(total_samples):.4f} ms"
            )
            print(
                "deepep-v1-low-latency cost-model: "
                f"logical_dispatch_B/pe={route_rows * dispatch_row_bytes} "
                f"logical_combine_B/pe={route_rows * combine_row_bytes} "
                f"estimated_remote_dispatch_B/pe="
                f"{remote_routes * dispatch_row_bytes} "
                f"estimated_remote_combine_B/pe="
                f"{remote_routes * combine_row_bytes} "
                f"rdma_buffer_B/pe={rdma_bytes}"
            )
            print(f"deepep-v1 route-digest: {global_route_digest}")
    finally:
        buffer.destroy()
        comm.Barrier()


if __name__ == "__main__":
    main()
