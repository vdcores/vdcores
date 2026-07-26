"""External two-node sharded-pool versus NCCL ring-all-reduce benchmark.

The NCCL path is intentionally a dense collective reference: one ring
all-reduce materializes expert-major dispatch storage and a second ring
all-reduce returns token-major results.  The pool path moves only routed rows.
In-kernel global-timer events measure the pool and CUDA events measure NCCL;
an MPI max reduces each sample so reported latency is the slowest PE, as
required by a distributed critical path. This file intentionally lives outside
the VDCores runtime and application source trees.
"""

from __future__ import annotations

import argparse
import os
import socket
import statistics
from dataclasses import dataclass

import torch
from mpi4py import MPI

import dae.nvshmem as nvshmem
from dae.ep_pool import (
    EP_BATCH_BYTES,
    ExpertPoolStatus,
    allocate_expert_pool,
    build_expert_pool_copy_program,
)


@dataclass(frozen=True)
class Timing:
    dispatch_ready_ms: list[float]
    tail_ms: list[float]
    end_to_end_ms: list[float]

    def summary(self) -> dict[str, float]:
        def median(values: list[float]) -> float:
            return float(statistics.median(values))

        return {
            "dispatch_ready_ms": median(self.dispatch_ready_ms),
            "tail_ms": median(self.tail_ms),
            "end_to_end_ms": median(self.end_to_end_ms),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--experts-per-pe", type=int, default=1)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mode", choices=("pool", "nccl", "both"), default="both")
    parser.add_argument("--symmetric-size", default="1G")
    return parser.parse_args()


def rank_max(comm: MPI.Comm, value: float) -> float:
    return float(comm.allreduce(float(value), op=MPI.MAX))


def event_quadruple() -> tuple[torch.cuda.Event, ...]:
    return tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))


def timed_values(
    comm: MPI.Comm,
    events: tuple[torch.cuda.Event, ...],
) -> tuple[float, float, float]:
    events[-1].synchronize()
    dispatch = rank_max(comm, events[0].elapsed_time(events[1]))
    tail = rank_max(comm, events[1].elapsed_time(events[3]))
    end_to_end = rank_max(comm, events[0].elapsed_time(events[3]))
    return dispatch, tail, end_to_end


def run_pool(
    args: argparse.Namespace,
    runtime,
    comm: MPI.Comm,
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[Timing, dict[str, int]]:
    num_experts = runtime.num_pes * args.experts_per_pe
    global_tokens = runtime.num_pes * args.tokens_per_pe
    expert_capacity_rows = (global_tokens + num_experts - 1) // num_experts
    signal_count = num_experts * runtime.num_pes + num_experts + runtime.num_pes
    signals = nvshmem.init_signal_space(signal_count)
    buffers = allocate_expert_pool(
        signals,
        num_pes=runtime.num_pes,
        my_pe=runtime.pe,
        experts_per_pe=args.experts_per_pe,
        token_capacity=args.tokens_per_pe,
        expert_capacity_rows=expert_capacity_rows,
        hidden_size=args.hidden_size,
        dtype=tokens.dtype,
    )
    returned = nvshmem.zeros(tokens.shape, dtype=tokens.dtype)
    local_rows = torch.arange(args.tokens_per_pe, dtype=torch.int64)
    buffers.write_routes(
        expert_ids,
        source_rows=local_rows,
        origin_rows=local_rows,
    )
    buffers.prepare(tokens, returned)
    program = build_expert_pool_copy_program(
        buffers,
        benchmark_barrier=nvshmem.benchmark_barrier,
    )

    dispatch_ready_samples: list[float] = []
    tail_samples: list[float] = []
    end_to_end_samples: list[float] = []
    total_iterations = args.warmup + args.iterations
    for index in range(total_iterations):
        sequence = index + 1
        buffers.reset_dispatch(sequence)
        torch.cuda.synchronize(runtime.device)
        comm.Barrier()

        program.launch()
        torch.cuda.synchronize(runtime.device)
        dispatch_ns, tail_ns, end_to_end_ns = program.timing_ns()
        dispatch_ms = rank_max(comm, dispatch_ns / 1.0e6)
        tail_ms = rank_max(comm, tail_ns / 1.0e6)
        end_to_end_ms = rank_max(comm, end_to_end_ns / 1.0e6)
        if index >= args.warmup:
            dispatch_ready_samples.append(dispatch_ms)
            tail_samples.append(tail_ms)
            end_to_end_samples.append(end_to_end_ms)

    torch.testing.assert_close(returned, tokens, rtol=0, atol=0)
    status, metadata_batches, received_rows, returned_experts, observed_sequence = (
        buffers.control_state()
    )
    assert status == ExpertPoolStatus.OK
    assert metadata_batches == runtime.num_pes * args.experts_per_pe
    assert returned_experts == num_experts
    assert observed_sequence == total_iterations

    counts = torch.bincount(expert_ids, minlength=num_experts)
    target_pes = torch.arange(num_experts).div(
        args.experts_per_pe, rounding_mode="floor"
    )
    remote_mask = target_pes != runtime.pe
    remote_routes = int(counts[remote_mask].sum().item())
    nonempty_remote_batches = int((counts[remote_mask] != 0).sum().item())
    remote_target_pes = runtime.num_pes - 1
    row_bytes = args.hidden_size * tokens.element_size()
    model = {
        "remote_routes_per_pe": remote_routes,
        "payload_bytes_per_direction_per_pe": remote_routes * row_bytes,
        "descriptor_bytes_per_pe": (
            remote_target_pes * args.experts_per_pe * EP_BATCH_BYTES
        ),
        "remote_atomics_per_pe": nonempty_remote_batches,
        "dispatch_data_rmas_per_pe": nonempty_remote_batches,
        "return_data_rmas_per_pe": nonempty_remote_batches,
        "dispatch_signals_per_pe": remote_target_pes * args.experts_per_pe,
        "return_signals_per_pe": remote_target_pes * args.experts_per_pe,
        "reset_signals_per_pe": remote_target_pes,
        "local_pack_bytes_per_pe": args.tokens_per_pe * row_bytes * 2,
        "local_return_scatter_bytes_per_pe": args.tokens_per_pe * row_bytes * 2,
        "identity_expert_copy_bytes_per_pe": (
            2 * args.experts_per_pe * expert_capacity_rows * row_bytes
        ),
        "received_rows": received_rows,
    }
    return (
        Timing(dispatch_ready_samples, tail_samples, end_to_end_samples),
        model,
    )


def initialize_nccl(runtime, comm: MPI.Comm) -> None:
    # Set before process-group construction so the comparison is specifically
    # NCCL's ring implementation rather than an automatically selected tree.
    os.environ["NCCL_ALGO"] = "Ring"
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    if runtime.rank == 0:
        master_address = socket.gethostname()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("", 0))
            master_port = listener.getsockname()[1]
    else:
        master_address = None
        master_port = None
    master_address = comm.bcast(master_address, root=0)
    master_port = comm.bcast(master_port, root=0)
    os.environ["MASTER_ADDR"] = str(master_address)
    os.environ["MASTER_PORT"] = str(master_port)

    import torch.distributed as dist

    dist.init_process_group(
        backend="nccl",
        rank=runtime.rank,
        world_size=runtime.world_size,
    )


def run_nccl(
    args: argparse.Namespace,
    runtime,
    comm: MPI.Comm,
    tokens: torch.Tensor,
    expert_ids: torch.Tensor,
) -> tuple[Timing, dict[str, float]]:
    import torch.distributed as dist

    num_experts = runtime.num_pes * args.experts_per_pe
    global_tokens = runtime.num_pes * args.tokens_per_pe
    row_bytes = args.hidden_size * tokens.element_size()
    global_ids = torch.arange(
        runtime.pe * args.tokens_per_pe,
        (runtime.pe + 1) * args.tokens_per_pe,
        device=tokens.device,
        dtype=torch.long,
    )
    expert_ids_device = expert_ids.to(tokens.device, dtype=torch.long)
    dispatch_template = torch.zeros(
        (num_experts, global_tokens, args.hidden_size),
        dtype=tokens.dtype,
        device=tokens.device,
    )
    dispatch_template[expert_ids_device, global_ids] = tokens
    dispatch_buffer = torch.empty_like(dispatch_template)
    return_buffer = torch.zeros(
        (global_tokens, args.hidden_size),
        dtype=tokens.dtype,
        device=tokens.device,
    )

    owned_token_ids: list[tuple[int, torch.Tensor]] = []
    all_global_ids = torch.arange(global_tokens, device=tokens.device)
    for local_expert in range(args.experts_per_pe):
        global_expert = runtime.pe * args.experts_per_pe + local_expert
        ids = all_global_ids[all_global_ids.remainder(num_experts) == global_expert]
        owned_token_ids.append((global_expert, ids))

    dispatch_ready_samples: list[float] = []
    tail_samples: list[float] = []
    end_to_end_samples: list[float] = []
    total_iterations = args.warmup + args.iterations
    for index in range(total_iterations):
        dispatch_buffer.copy_(dispatch_template)
        return_buffer.zero_()
        torch.cuda.synchronize(runtime.device)
        comm.Barrier()

        events = event_quadruple()
        events[0].record()
        dist.all_reduce(dispatch_buffer, op=dist.ReduceOp.SUM)
        events[1].record()
        for global_expert, ids in owned_token_ids:
            return_buffer[ids] = dispatch_buffer[global_expert, ids]
        events[2].record()
        dist.all_reduce(return_buffer, op=dist.ReduceOp.SUM)
        events[3].record()
        dispatch_ms, return_ms, end_to_end_ms = timed_values(comm, events)
        if index >= args.warmup:
            dispatch_ready_samples.append(dispatch_ms)
            tail_samples.append(return_ms)
            end_to_end_samples.append(end_to_end_ms)

    local_return = return_buffer[
        runtime.pe * args.tokens_per_pe : (runtime.pe + 1) * args.tokens_per_pe
    ]
    torch.testing.assert_close(local_return, tokens, rtol=0, atol=0)

    dispatch_tensor_bytes = num_experts * global_tokens * row_bytes
    return_tensor_bytes = global_tokens * row_bytes
    ring_factor = 2.0 * (runtime.num_pes - 1) / runtime.num_pes
    model = {
        "dispatch_tensor_bytes": dispatch_tensor_bytes,
        "return_tensor_bytes": return_tensor_bytes,
        "ring_network_bytes_per_pe": ring_factor
        * (dispatch_tensor_bytes + return_tensor_bytes),
        "collectives": 2,
    }
    return (
        Timing(dispatch_ready_samples, tail_samples, end_to_end_samples),
        model,
    )


def print_result(name: str, timing: Timing, model: dict[str, int | float]) -> None:
    summary = timing.summary()
    print(
        f"{name}: dispatch-ready={summary['dispatch_ready_ms']:.3f} ms "
        f"tail={summary['tail_ms']:.3f} ms "
        f"end-to-end={summary['end_to_end_ms']:.3f} ms"
    )
    print(f"{name} cost-model: {model}")


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
        num_experts = runtime.num_pes * args.experts_per_pe
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
        expert_ids = global_ids.remainder(num_experts)

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
                f"configuration: pes={runtime.num_pes} tokens/pe={args.tokens_per_pe} "
                f"hidden={args.hidden_size} experts/pe={args.experts_per_pe} "
                f"dtype={args.dtype} warmup={args.warmup} iterations={args.iterations}"
            )
            if pool_result is not None:
                print_result("pool", *pool_result)
            if nccl_result is not None:
                print_result("nccl-ring", *nccl_result)
            if pool_result is not None and nccl_result is not None:
                pool_ms = pool_result[0].summary()["end_to_end_ms"]
                nccl_ms = nccl_result[0].summary()["end_to_end_ms"]
                print(f"latency ratio pool/nccl-ring: {pool_ms / nccl_ms:.3f}x")
    finally:
        if nccl_initialized:
            import torch.distributed as dist

            dist.destroy_process_group()
        nvshmem.finalize()


if __name__ == "__main__":
    main()
