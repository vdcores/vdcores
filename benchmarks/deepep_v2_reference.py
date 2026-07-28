"""External DeepEP V2 cached-decode dispatch/combine reference.

DeepEP and its NCCL Gin dependency are intentionally external to VDCores.
Launch one MPI rank per GPU/node and add the DeepEP source/build directories
to ``PYTHONPATH`` plus its NCCL wheel ``lib`` directory to ``LD_LIBRARY_PATH``.
"""

from __future__ import annotations

import argparse
import socket
import statistics

from mpi4py import MPI
import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--experts-per-pe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument("--num-qps", type=int, default=0)
    parser.add_argument("--prefer-overlap-with-compute", action="store_true")
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="construct and destroy ElasticBuffer without launching EP kernels",
    )
    parser.add_argument("--master-port", type=int, default=29671)
    return parser.parse_args()


def rank_max(comm: MPI.Comm, value: float) -> float:
    return float(comm.allreduce(float(value), op=MPI.MAX))


def main() -> None:
    args = parse_args()
    if min(
        args.tokens_per_pe,
        args.hidden_size,
        args.experts_per_pe,
        args.top_k,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        raise ValueError("sizes must be positive and warmup non-negative")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    if local_rank >= torch.cuda.device_count():
        raise RuntimeError("one MPI rank requires one local CUDA device")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    master = comm.bcast(socket.gethostname() if rank == 0 else None, root=0)
    if rank == 0:
        print("deepep-v2 bootstrap: initializing torch NCCL communicator", flush=True)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master}:{args.master_port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        if rank == 0:
            print("deepep-v2 bootstrap: importing external extension", flush=True)
        import deep_ep
    except BaseException:
        dist.destroy_process_group()
        raise

    num_experts = world_size * args.experts_per_pe
    if args.top_k > num_experts:
        raise ValueError("top-k cannot exceed the global expert count")
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
    tokens.copy_((values.remainder(97) - 48).to(torch.bfloat16))
    global_rows = rank * args.tokens_per_pe + torch.arange(
        args.tokens_per_pe, dtype=torch.int64, device=device
    )
    route = torch.arange(args.top_k, dtype=torch.int64, device=device)
    topk_idx = (
        global_rows[:, None] * args.top_k + route[None, :]
    ).remainder(num_experts).to(deep_ep.topk_idx_t)
    topk_weights = torch.full(
        (args.tokens_per_pe, args.top_k),
        1.0 / args.top_k,
        dtype=torch.float32,
        device=device,
    )

    try:
        if rank == 0:
            print("deepep-v2 bootstrap: constructing ElasticBuffer", flush=True)
        buffer = deep_ep.ElasticBuffer(
            dist.group.WORLD,
            num_max_tokens_per_rank=args.tokens_per_pe,
            hidden=args.hidden_size,
            num_topk=args.top_k,
            use_fp8_dispatch=False,
            deterministic=False,
            allow_hybrid_mode=False,
            allow_multiple_reduction=False,
            prefer_overlap_with_compute=args.prefer_overlap_with_compute,
            num_allocated_qps=max(17, args.num_qps),
            explicitly_destroy=True,
        )
    except BaseException:
        dist.destroy_process_group()
        raise
    num_sms = args.num_sms or buffer.get_theoretical_num_sms(
        num_experts, args.top_k
    )
    num_qps = args.num_qps or buffer.get_theoretical_num_qps(num_sms)
    if rank == 0:
        print(
            f"deepep-v2 bootstrap: ready theoretical_sms={num_sms} "
            f"theoretical_qps={num_qps}",
            flush=True,
        )
    dispatch_samples: list[float] = []
    combine_samples: list[float] = []
    total_samples: list[float] = []
    combined = None
    try:
        if args.bootstrap_only:
            comm.Barrier()
            return
        _, _, recv_weights, handle, _ = buffer.dispatch(
            tokens,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=args.tokens_per_pe,
            expert_alignment=1,
            num_sms=num_sms,
            num_qps=num_qps,
            do_handle_copy=True,
            do_cpu_sync=True,
        )
        assert recv_weights is not None
        torch.cuda.synchronize(device)
        comm.Barrier()

        for iteration in range(args.warmup + args.iterations):
            torch.cuda.synchronize(device)
            comm.Barrier()
            start = torch.cuda.Event(enable_timing=True)
            dispatched = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)
            start.record()
            recv_x, _, _, _, _ = buffer.dispatch(
                tokens,
                handle=handle,
                num_sms=num_sms,
                num_qps=num_qps,
                do_handle_copy=False,
            )
            dispatched.record()
            combined, _, _ = buffer.combine(
                recv_x,
                handle=handle,
                topk_weights=recv_weights,
                num_sms=num_sms,
                num_qps=num_qps,
            )
            done.record()
            done.synchronize()
            if iteration >= args.warmup:
                dispatch_samples.append(
                    rank_max(comm, start.elapsed_time(dispatched))
                )
                combine_samples.append(
                    rank_max(comm, dispatched.elapsed_time(done))
                )
                total_samples.append(rank_max(comm, start.elapsed_time(done)))

        torch.testing.assert_close(combined, tokens, rtol=2e-2, atol=2e-1)
        if rank == 0:
            print(
                "deepep-v2-cached: "
                f"pes={world_size} tokens/pe={args.tokens_per_pe} "
                f"hidden={args.hidden_size} experts/pe={args.experts_per_pe} "
                f"top_k={args.top_k} sms={num_sms} qps={num_qps} "
                f"prefer_overlap={args.prefer_overlap_with_compute}"
            )
            print(
                "deepep-v2-cached timing: "
                f"dispatch={statistics.median(dispatch_samples):.4f} ms "
                f"combine={statistics.median(combine_samples):.4f} ms "
                f"end-to-end={statistics.median(total_samples):.4f} ms"
            )
    finally:
        buffer.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
