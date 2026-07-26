"""Pool-owned top-k dynamic-read correctness and internal-timing harness.

Every PE owns one logical pool slice.  The application writes only source
tokens and a stable-grouped route table; the VDCores pool block publishes
metadata, resolves the shared sender-set dependency, gathers reader rows,
runs the identity stand-in for expert compute, and returns route-major rows.
"""

from __future__ import annotations

import argparse
import statistics

import torch

import dae.nvshmem as nvshmem
from dae.pool_slice import (
    POOL_SLICE_PUBLISH_BYTES,
    PoolSliceStatus,
    allocate_pool_slice,
    build_pool_slice_copy_program,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--readers-per-pe", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--symmetric-size", default="512M")
    parser.add_argument(
        "--gather-mode",
        choices=("streaming", "phased"),
        default="streaming",
    )
    parser.add_argument("--activation-stages", type=int, choices=(1, 2), default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "tokens_per_pe",
        "hidden_size",
        "readers_per_pe",
        "top_k",
        "iterations",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    try:
        num_readers = runtime.num_pes * args.readers_per_pe
        if args.top_k > num_readers:
            raise ValueError("top-k cannot exceed the global reader count")

        local_routes = args.tokens_per_pe * args.top_k
        global_routes = runtime.num_pes * local_routes
        expert_capacity_rows = (
            global_routes + num_readers - 1
        ) // num_readers
        signals = nvshmem.init_signal_space(3 * runtime.num_pes)
        buffers = allocate_pool_slice(
            signals,
            num_pes=runtime.num_pes,
            my_pe=runtime.pe,
            local_readers=args.readers_per_pe,
            token_capacity=args.tokens_per_pe,
            route_capacity=local_routes,
            expert_capacity_rows=expert_capacity_rows,
            hidden_size=args.hidden_size,
            dtype=dtype,
            streaming_gather=args.gather_mode == "streaming",
            activation_stages=args.activation_stages,
        )
        tokens = nvshmem.empty(
            (args.tokens_per_pe, args.hidden_size), dtype=dtype
        )
        # Top-k results are returned route-major.  A later pool reduction may
        # combine routes sharing a token; this harness isolates dynamic read.
        returned = nvshmem.zeros((local_routes, args.hidden_size), dtype=dtype)

        local_rows = torch.arange(args.tokens_per_pe, dtype=torch.int64)
        global_rows = runtime.pe * args.tokens_per_pe + local_rows
        route_rank = torch.arange(args.top_k, dtype=torch.int64)
        reader_ids = (
            global_rows[:, None] + route_rank[None, :]
        ).remainder(num_readers).reshape(-1)
        source_rows = local_rows.repeat_interleave(args.top_k)
        origin_rows = torch.arange(local_routes, dtype=torch.int64)
        buffers.write_routes(
            reader_ids,
            source_rows=source_rows,
            origin_rows=origin_rows,
        )
        buffers.prepare(tokens, returned)
        program = build_pool_slice_copy_program(
            buffers,
            benchmark_barrier=nvshmem.benchmark_barrier,
        )

        token_values = torch.arange(
            runtime.pe * args.tokens_per_pe * args.hidden_size,
            (runtime.pe + 1) * args.tokens_per_pe * args.hidden_size,
            dtype=torch.float32,
            device=tokens.device,
        ).view_as(tokens)
        tokens.copy_((token_values.remainder(251) - 125).to(dtype))

        gather_samples: list[float] = []
        return_samples: list[float] = []
        total_samples: list[float] = []
        first_payload_samples: list[float] = []
        rounds = args.warmup + args.iterations
        for iteration in range(rounds):
            sequence = iteration + 1
            buffers.set_sequence(sequence)
            torch.cuda.synchronize(runtime.device)
            nvshmem.barrier()
            program.launch()
            torch.cuda.synchronize(runtime.device)
            gather_ns, return_ns, total_ns = program.timing_ns()
            overlap = (
                program.overlap_timing_ns()
                if args.gather_mode == "streaming"
                else None
            )
            if iteration >= args.warmup:
                gather_samples.append(gather_ns / 1.0e6)
                return_samples.append(return_ns / 1.0e6)
                total_samples.append(total_ns / 1.0e6)
                if overlap is not None and overlap["first_payload"] is not None:
                    first_payload_samples.append(
                        overlap["first_payload"] / 1.0e6
                    )

        status, senders, received_rows, returned_slices, observed, group_ready = (
            buffers.control_state()
        )
        assert status == PoolSliceStatus.OK
        assert senders == runtime.num_pes
        assert returned_slices == runtime.num_pes
        assert observed == rounds
        assert group_ready == rounds
        assert received_rows == int(buffers.reader_tails.sum().item())
        metadata_waves, payload_sources, peak_inflight = (
            buffers.streaming_state()
        )

        expected = tokens.index_select(0, source_rows.to(tokens.device))
        if not torch.equal(returned, expected):
            snapshots = {
                "tokens": tokens[0, :8].float().cpu().tolist(),
                "token_pool": buffers.token_pool[0, :8].float().cpu().tolist(),
                "expert_input": buffers.expert_input[0, 0, :8]
                .float()
                .cpu()
                .tolist(),
                "expert_output": buffers.expert_output[0, 0, :8]
                .float()
                .cpu()
                .tolist(),
                "return_inbox": buffers.return_inbox[0, :8]
                .float()
                .cpu()
                .tolist(),
                "returned": returned[0, :8].float().cpu().tolist(),
                "send_offsets": buffers.send_offsets.cpu().tolist(),
                "send_rows": buffers.send_rows[: buffers.active_rows]
                .cpu()
                .tolist(),
                "offsets_inbox": buffers.offsets_inbox.cpu().tolist(),
                "receive_batches": buffers.receive_batches.cpu().tolist(),
                "reader_tails": buffers.reader_tails.cpu().tolist(),
            }
            raise AssertionError(
                f"pool-slice data path mismatch; control="
                f"{(status, senders, received_rows, returned_slices, observed, group_ready)} "
                f"snapshots={snapshots}"
            )

        row_bytes = args.hidden_size * tokens.element_size()
        remote_slices = runtime.num_pes - 1
        overlap_summary = (
            f"first_payload_ms={statistics.median(first_payload_samples):.4f}, "
            f"metadata_waves={metadata_waves}, "
            f"payload_sources={payload_sources}, "
            f"peak_inflight={peak_inflight}"
            if first_payload_samples
            else "disabled"
        )
        nvshmem.barrier()
        print(
            f"PE {runtime.pe}/{runtime.num_pes}: pool-slice dynamic-read PASS "
            f"tokens={args.tokens_per_pe} hidden={args.hidden_size} "
            f"readers={num_readers} top_k={args.top_k} "
            f"gather_mode={args.gather_mode} "
            f"activation_stages={args.activation_stages} "
            f"received={received_rows} launches={rounds} "
            f"median_ms=(gather={statistics.median(gather_samples):.4f}, "
            f"return={statistics.median(return_samples):.4f}, "
            f"total={statistics.median(total_samples):.4f}) "
            f"overlap=({overlap_summary}) "
            f"model=(descriptor_B={remote_slices * POOL_SLICE_PUBLISH_BYTES}, "
            f"route_payload_B={local_routes * row_bytes})"
        )
    finally:
        nvshmem.finalize()


if __name__ == "__main__":
    main()
