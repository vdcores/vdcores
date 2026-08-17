"""Shared shape, routing, and reporting helpers for external EP baselines."""

from __future__ import annotations

import argparse
import json
import os
import socket
from typing import Any

import torch


ROUTE_PLACEMENTS = (
    "clustered",
    "source-local",
    "remote-clustered",
    "spread",
)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tokens-per-pe", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--experts-per-pe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--route-placement",
        choices=ROUTE_PLACEMENTS,
        default="clustered",
        help=(
            "cluster each token's routes on one balanced PE, force that PE "
            "local/remote to the source, or spread routes across PEs"
        ),
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=50)


def validate_common_arguments(args: argparse.Namespace, world_size: int) -> None:
    for name in (
        "tokens_per_pe",
        "hidden_size",
        "experts_per_pe",
        "top_k",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    num_experts = world_size * args.experts_per_pe
    if args.top_k > num_experts:
        raise ValueError("top-k cannot exceed the global expert count")
    if (
        args.route_placement in {"source-local", "remote-clustered"}
        and args.top_k > args.experts_per_pe
    ):
        raise ValueError(
            "single-PE placements require top-k <= experts-per-pe"
        )


def configure_torchrun_environment(
    comm: Any, master_port: int
) -> tuple[int, int, int, int]:
    """Translate an MPI one-rank-per-GPU launch into torchrun variables."""

    from mpi4py import MPI

    rank = comm.Get_rank()
    world_size = comm.Get_size()
    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_world_size = local_comm.Get_size()
    master_addr = comm.bcast(socket.gethostname() if rank == 0 else None, root=0)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(local_world_size)
    return rank, world_size, local_rank, local_world_size


def balanced_topk(
    *,
    rank: int,
    tokens_per_pe: int,
    num_experts: int,
    experts_per_pe: int,
    top_k: int,
    placement: str,
    device: torch.device,
    dtype: torch.dtype = torch.int64,
) -> torch.Tensor:
    """Build distinct deterministic routes with balanced global ownership."""

    global_rows = rank * tokens_per_pe + torch.arange(
        tokens_per_pe, dtype=torch.int64, device=device
    )
    route = torch.arange(top_k, dtype=torch.int64, device=device)
    if placement == "clustered":
        result = (global_rows[:, None] * top_k + route[None, :]).remainder(
            num_experts
        )
    else:
        num_pes = num_experts // experts_per_pe
        source_pe = global_rows.div(tokens_per_pe, rounding_mode="floor")
        if placement in {"source-local", "remote-clustered"}:
            target_pe = source_pe
            if placement == "remote-clustered":
                target_pe = (target_pe + 1).remainder(num_pes)
            local_expert = (
                global_rows[:, None] * top_k + route[None, :]
            ).remainder(experts_per_pe)
            result = target_pe[:, None] * experts_per_pe + local_expert
        else:
            target_pe = (global_rows[:, None] + route[None, :]).remainder(
                num_pes
            )
            local_expert = (
                global_rows[:, None] * top_k + route[None, :]
            ).div(num_pes, rounding_mode="floor").remainder(experts_per_pe)
            result = target_pe * experts_per_pe + local_expert
    return result.to(dtype=dtype)


def remote_route_count(
    topk_idx: torch.Tensor, *, rank: int, experts_per_pe: int
) -> int:
    owners = topk_idx.to(torch.int64).div(
        experts_per_pe, rounding_mode="floor"
    )
    return int((owners != rank).sum().item())


def rank_max(comm: Any, value: float) -> float:
    # mpi4py exposes MAX on the module, while the communicator owns allreduce.
    from mpi4py import MPI

    return float(comm.allreduce(float(value), op=MPI.MAX))


def emit_result(result: dict[str, object]) -> None:
    """Emit one stable machine-readable line in addition to human output."""

    print("ep-baseline-json: " + json.dumps(result, sort_keys=True), flush=True)
