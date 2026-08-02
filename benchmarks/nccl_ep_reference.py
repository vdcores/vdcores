"""NVIDIA NCCL EP low-latency dispatch/combine baseline.

This adapter uses the real ``ncclEpDispatch`` and ``ncclEpCombine`` APIs from
an externally installed ``nccl4py``/``libnccl_ep`` package. It never imports
or modifies the VDCores runtime. CUDA events time only NCCL EP communication;
MPI rank-maximum reduction makes each sample represent distributed completion.

Launch one MPI rank per GPU. NCCL EP, NCCL, and their Python bindings must be
installed outside this repository and selected before the Python process
starts (notably through ``LD_LIBRARY_PATH`` when multiple NCCL builds exist).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import statistics
import subprocess

from mpi4py import MPI
import torch

from ep_baseline_common import (
    add_common_arguments,
    balanced_topk,
    emit_result,
    rank_max,
    remote_route_count,
    route_digest,
    validate_common_arguments,
)


NCCL_EP_LL_HIDDEN_SIZES = (2048, 2560, 4096, 5120, 6144, 7168, 8192)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NVIDIA NCCL EP low-latency BF16 baseline"
    )
    add_common_arguments(parser)
    parser.add_argument(
        "--nccl-ep-root",
        default=os.environ.get("NCCL_EP_ROOT", ""),
        help="optional official NVIDIA/nccl checkout used for provenance",
    )
    parser.add_argument(
        "--num-qps-per-rank",
        type=int,
        default=0,
        help="NCCL EP QPs per rank; zero selects experts-per-pe",
    )
    parser.add_argument(
        "--max-num-sms",
        type=int,
        default=0,
        help="maximum NCCL EP kernel SMs; zero selects library auto-tuning",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=0,
        help="NCCL EP channels per rank; zero selects library auto-tuning",
    )
    return parser.parse_args()


def _source_commit(root: Path | None) -> str:
    if root is None:
        return "unknown"
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _unique_remote_destination_rows(
    topk_idx: torch.Tensor, *, rank: int, experts_per_pe: int
) -> int:
    """Count token/destination-rank payloads after NCCL EP dispatch dedup."""

    owners = topk_idx.to(torch.int64).div(
        experts_per_pe, rounding_mode="floor"
    ).cpu()
    total = 0
    for token_owners in owners:
        total += len({int(owner) for owner in token_owners if int(owner) != rank})
    return total


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    validate_common_arguments(args, world_size)
    if world_size not in (2, 4) and world_size % 8 != 0:
        raise ValueError("NCCL EP LL supports 2, 4, or a multiple of 8 ranks")
    if args.top_k > args.experts_per_pe:
        raise ValueError(
            "NCCL EP LL expert-major requires top-k <= experts-per-pe"
        )
    if args.hidden_size not in NCCL_EP_LL_HIDDEN_SIZES:
        raise ValueError(
            "NCCL EP LL hidden-size must be one of "
            f"{NCCL_EP_LL_HIDDEN_SIZES}"
        )
    if args.max_num_sms < 0 or args.num_channels < 0:
        raise ValueError("max-num-sms and num-channels must be non-negative")

    qps_per_rank = args.num_qps_per_rank or args.experts_per_pe
    if qps_per_rank < args.experts_per_pe:
        raise ValueError(
            "NCCL EP LL requires at least experts-per-pe QPs per rank"
        )

    source_root = (
        Path(args.nccl_ep_root).resolve() if args.nccl_ep_root else None
    )
    if source_root is not None and not (
        source_root / "contrib" / "nccl_ep" / "README.md"
    ).is_file():
        raise RuntimeError(
            f"official NCCL EP sources not found under {source_root}"
        )

    local_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()
    local_world_size = local_comm.Get_size()
    if local_world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"{local_world_size} local ranks exceed {torch.cuda.device_count()} GPUs"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    stream = torch.cuda.current_stream(device)

    # NVIDIA recommends GDAKI for multi-node NCCL EP. Respect an explicit
    # caller choice while making the production multi-node path the default.
    os.environ.setdefault("NCCL_GIN_TYPE", "3")
    try:
        import nccl
        import nccl.core as nccl_core
        import nccl.ep as nccl_ep
    except (ImportError, RuntimeError) as error:
        raise RuntimeError(
            "NVIDIA NCCL EP is not importable; install nccl4py[cu13] in an "
            "external environment and select its matching libnccl.so before "
            "launching this benchmark"
        ) from error

    unique_id = nccl_core.get_unique_id() if rank == 0 else None
    unique_id = comm.bcast(unique_id, root=0)
    nccl_comm = nccl_core.Communicator.init(
        nranks=world_size, rank=rank, unique_id=unique_id
    )

    ep_group = None
    ep_handle = None
    try:
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
            dtype=torch.int64,
            seed=args.route_seed,
        ).contiguous()
        global_route_digest = route_digest(comm, topk_idx)
        topk_weights = torch.full(
            (args.tokens_per_pe, args.top_k),
            1.0 / args.top_k,
            dtype=torch.float32,
            device=device,
        )

        config = nccl_ep.GroupConfig(
            algorithm=nccl_ep.Algorithm.LOW_LATENCY,
            num_experts=num_experts,
            max_dispatch_tokens_per_rank=args.tokens_per_pe,
            max_recv_tokens_per_rank=args.tokens_per_pe * world_size,
            max_token_bytes=args.hidden_size * tokens.element_size(),
            num_qp_per_rank=qps_per_rank,
            num_channels=args.num_channels,
            max_num_sms=args.max_num_sms,
        )
        ep_group = nccl_ep.Group.create(nccl_comm, config)

        token_desc = nccl_ep.Tensor(tokens)
        topk_desc = nccl_ep.Tensor(topk_idx)
        weight_desc = nccl_ep.Tensor(topk_weights)
        ep_handle = ep_group.create_handle(
            nccl_ep.Layout.EXPERT_MAJOR,
            topk_desc,
            config=nccl_ep.HandleConfig(),
            stream=stream,
        )

        recv_tokens = torch.empty(
            (
                args.experts_per_pe,
                args.tokens_per_pe * world_size,
                args.hidden_size,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        recv_counts = torch.empty(
            (args.experts_per_pe,), dtype=torch.int32, device=device
        )
        combined = torch.empty_like(tokens)
        recv_desc = nccl_ep.Tensor(recv_tokens)
        count_desc = nccl_ep.Tensor(recv_counts)
        combined_desc = nccl_ep.Tensor(combined)

        dispatch_inputs = nccl_ep.DispatchInputs(tokens=token_desc)
        dispatch_outputs = nccl_ep.DispatchOutputs(tokens=recv_desc)
        dispatch_layout = nccl_ep.LayoutInfo(expert_counters=count_desc)
        dispatch_config = nccl_ep.DispatchConfig(send_only=0)
        combine_inputs = nccl_ep.CombineInputs(tokens=recv_desc)
        combine_outputs = nccl_ep.CombineOutputs(
            tokens=combined_desc,
            topk_weights=weight_desc,
        )
        combine_config = nccl_ep.CombineConfig(send_only=0)

        torch.cuda.synchronize(device)
        comm.Barrier()

        # Validate metadata and weighted identity semantics before timing.
        ep_handle.dispatch(
            dispatch_inputs,
            dispatch_outputs,
            layout_info=dispatch_layout,
            config=dispatch_config,
            stream=stream,
        )
        ep_handle.combine(
            combine_inputs,
            combine_outputs,
            config=combine_config,
            stream=stream,
        )
        torch.cuda.synchronize(device)
        gathered_routes = comm.allgather(topk_idx.cpu())
        expected_counts = torch.bincount(
            torch.cat(gathered_routes).reshape(-1).to(torch.int64),
            minlength=num_experts,
        )[rank * args.experts_per_pe : (rank + 1) * args.experts_per_pe]
        torch.testing.assert_close(
            recv_counts.cpu().to(torch.int64), expected_counts, rtol=0, atol=0
        )
        torch.testing.assert_close(combined, tokens, rtol=2.0e-2, atol=2.0e-1)

        dispatch_samples: list[float] = []
        combine_samples: list[float] = []
        total_samples: list[float] = []
        for iteration in range(args.warmup + args.iterations):
            torch.cuda.synchronize(device)
            comm.Barrier()
            start = torch.cuda.Event(enable_timing=True)
            dispatched = torch.cuda.Event(enable_timing=True)
            done = torch.cuda.Event(enable_timing=True)

            start.record(stream)
            ep_handle.dispatch(
                dispatch_inputs,
                dispatch_outputs,
                layout_info=dispatch_layout,
                config=dispatch_config,
                stream=stream,
            )
            dispatched.record(stream)
            # Identity expert work: NCCL EP's expert-major dispatch output is
            # consumed directly, with no computation included in either timer.
            ep_handle.combine(
                combine_inputs,
                combine_outputs,
                config=combine_config,
                stream=stream,
            )
            done.record(stream)
            done.synchronize()

            dispatch_ms = rank_max(comm, start.elapsed_time(dispatched))
            combine_ms = rank_max(comm, dispatched.elapsed_time(done))
            total_ms = rank_max(comm, start.elapsed_time(done))
            if iteration >= args.warmup:
                dispatch_samples.append(dispatch_ms)
                combine_samples.append(combine_ms)
                total_samples.append(total_ms)

        torch.testing.assert_close(combined, tokens, rtol=2.0e-2, atol=2.0e-1)
        dispatch_ms = statistics.median(dispatch_samples)
        combine_ms = statistics.median(combine_samples)
        total_ms = statistics.median(total_samples)
        remote_routes = remote_route_count(
            topk_idx, rank=rank, experts_per_pe=args.experts_per_pe
        )
        unique_remote_rows = _unique_remote_destination_rows(
            topk_idx, rank=rank, experts_per_pe=args.experts_per_pe
        )
        row_bytes = args.hidden_size * tokens.element_size()
        source_commit = _source_commit(source_root)
        nccl_ep_version = str(nccl_ep.get_lib_version())
        nccl_version = str(nccl_core.get_lib_version())
        nccl4py_version = getattr(nccl, "__version__", "unknown")
        nccl_ep_library = str(nccl_ep.get_lib_path())
        nccl_library = str(nccl_core.get_lib_path())

        if rank == 0:
            print(
                "nccl-ep-low-latency: "
                f"pes={world_size} tokens/pe={args.tokens_per_pe} "
                f"hidden={args.hidden_size} experts/pe={args.experts_per_pe} "
                f"top_k={args.top_k} dispatch=bfloat16 "
                f"layout=expert-major route_placement={args.route_placement} "
                f"route_seed={args.route_seed} "
                f"qps/rank={qps_per_rank} max_sms={args.max_num_sms or 'auto'} "
                f"channels={args.num_channels or 'auto'} "
                f"warmup={args.warmup} iterations={args.iterations}"
            )
            print(
                "nccl-ep-low-latency timing: "
                f"dispatch={dispatch_ms:.4f} ms combine={combine_ms:.4f} ms "
                f"end-to-end={total_ms:.4f} ms"
            )
            print(
                "nccl-ep-low-latency cost-model: "
                f"logical_route_rows/pe={args.tokens_per_pe * args.top_k} "
                f"remote_route_rows/pe={remote_routes} "
                f"deduplicated_remote_dispatch_rows/pe={unique_remote_rows} "
                f"estimated_remote_dispatch_payload_B/pe="
                f"{unique_remote_rows * row_bytes} "
                f"estimated_remote_combine_payload_B/pe={remote_routes * row_bytes}"
            )
            print(
                "nccl-ep-low-latency provenance: "
                f"nccl4py={nccl4py_version} nccl_ep={nccl_ep_version} "
                f"nccl={nccl_version} source_commit={source_commit} "
                f"libnccl_ep={nccl_ep_library} libnccl={nccl_library}"
            )
            emit_result(
                {
                    "implementation": "nvidia-nccl-ep-low-latency",
                    "pes": world_size,
                    "tokens_per_pe": args.tokens_per_pe,
                    "hidden_size": args.hidden_size,
                    "experts_per_pe": args.experts_per_pe,
                    "top_k": args.top_k,
                    "route_placement": args.route_placement,
                    "route_seed": args.route_seed,
                    "route_digest": global_route_digest,
                    "layout": "expert-major",
                    "dispatch_dtype": "bfloat16",
                    "num_qps_per_rank": qps_per_rank,
                    "max_num_sms": args.max_num_sms,
                    "num_channels": args.num_channels,
                    "dispatch_ms": dispatch_ms,
                    "combine_ms": combine_ms,
                    "end_to_end_ms": total_ms,
                    "remote_route_rows_per_pe": remote_routes,
                    "deduplicated_remote_dispatch_rows_per_pe": unique_remote_rows,
                    "nccl4py_version": nccl4py_version,
                    "nccl_ep_version": nccl_ep_version,
                    "nccl_version": nccl_version,
                    "nccl_ep_source_commit": source_commit,
                    "nccl_ep_library": nccl_ep_library,
                    "nccl_library": nccl_library,
                }
            )
    finally:
        if ep_handle is not None:
            ep_handle.destroy()
        if ep_group is not None:
            ep_group.destroy()
        nccl_comm.destroy()
        comm.Barrier()


if __name__ == "__main__":
    main()
