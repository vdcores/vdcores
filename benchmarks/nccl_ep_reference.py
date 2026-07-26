"""External NCCL ring reference shared by pool benchmarks.

This module intentionally remains outside the VDCores runtime and application
source trees. Pool timing uses internal VDCores profile events; NCCL timing
uses CUDA events and MPI rank-max reduction.
"""

from __future__ import annotations

import os
import socket
import statistics
from dataclasses import dataclass

import torch
from mpi4py import MPI


@dataclass(frozen=True)
class Timing:
    dispatch_ready_ms: list[float]
    tail_ms: list[float]
    end_to_end_ms: list[float]

    def summary(self) -> dict[str, float]:
        return {
            "dispatch_ready_ms": float(statistics.median(self.dispatch_ready_ms)),
            "tail_ms": float(statistics.median(self.tail_ms)),
            "end_to_end_ms": float(statistics.median(self.end_to_end_ms)),
        }


def rank_max(comm: MPI.Comm, value: float) -> float:
    return float(comm.allreduce(float(value), op=MPI.MAX))


def _event_quadruple() -> tuple[torch.cuda.Event, ...]:
    return tuple(torch.cuda.Event(enable_timing=True) for _ in range(4))


def _timed_values(
    comm: MPI.Comm,
    events: tuple[torch.cuda.Event, ...],
) -> tuple[float, float, float]:
    events[-1].synchronize()
    dispatch = rank_max(comm, events[0].elapsed_time(events[1]))
    tail = rank_max(comm, events[1].elapsed_time(events[3]))
    end_to_end = rank_max(comm, events[0].elapsed_time(events[3]))
    return dispatch, tail, end_to_end


def initialize_nccl(runtime, comm: MPI.Comm) -> None:
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
    args,
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

        events = _event_quadruple()
        events[0].record()
        dist.all_reduce(dispatch_buffer, op=dist.ReduceOp.SUM)
        events[1].record()
        for global_expert, ids in owned_token_ids:
            return_buffer[ids] = dispatch_buffer[global_expert, ids]
        events[2].record()
        dist.all_reduce(return_buffer, op=dist.ReduceOp.SUM)
        events[3].record()
        dispatch_ms, return_ms, end_to_end_ms = _timed_values(comm, events)
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


def print_result(name: str, timing: Timing, model: dict[str, object]) -> None:
    summary = timing.summary()
    print(
        f"{name}: dispatch-ready={summary['dispatch_ready_ms']:.3f} ms "
        f"tail={summary['tail_ms']:.3f} ms "
        f"end-to-end={summary['end_to_end_ms']:.3f} ms"
    )
    print(f"{name} cost-model: {model}")


__all__ = [
    "Timing",
    "initialize_nccl",
    "print_result",
    "rank_max",
    "run_nccl",
]
