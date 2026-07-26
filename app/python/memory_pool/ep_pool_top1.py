"""Top-k EP correctness as one mixed-domain VDCores program."""

from __future__ import annotations

import argparse

import torch

import dae.nvshmem as nvshmem
from dae.ep_pool import (
    ExpertPoolStatus,
    allocate_expert_pool,
    build_expert_pool_copy_program,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--experts-per-pe", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--symmetric-size", default="512M")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.tokens_per_pe <= 0
        or args.hidden_size <= 0
        or args.experts_per_pe <= 0
        or args.top_k <= 0
    ):
        raise ValueError("tokens, hidden size, experts per PE, and top-k must be positive")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    try:
        num_experts = runtime.num_pes * args.experts_per_pe
        if args.top_k > num_experts:
            raise ValueError("top-k cannot exceed the global expert count")
        total_tokens = runtime.num_pes * args.tokens_per_pe
        local_routes = args.tokens_per_pe * args.top_k
        total_routes = total_tokens * args.top_k
        expert_capacity_rows = (total_routes + num_experts - 1) // num_experts
        signal_count = (
            num_experts * runtime.num_pes
            + num_experts
            + runtime.num_pes
        )
        signals = nvshmem.init_signal_space(signal_count)
        buffers = allocate_expert_pool(
            signals,
            num_pes=runtime.num_pes,
            my_pe=runtime.pe,
            experts_per_pe=args.experts_per_pe,
            token_capacity=args.tokens_per_pe,
            route_capacity=local_routes,
            expert_capacity_rows=expert_capacity_rows,
            hidden_size=args.hidden_size,
            dtype=dtype,
        )
        tokens = nvshmem.empty(
            (args.tokens_per_pe, args.hidden_size), dtype=dtype
        )
        returned = nvshmem.zeros(
            (local_routes, args.hidden_size), dtype=dtype
        )

        local_rows = torch.arange(args.tokens_per_pe, dtype=torch.int64)
        global_ids = runtime.pe * args.tokens_per_pe + local_rows
        route_rank = torch.arange(args.top_k, dtype=torch.int64)
        expert_ids = (
            global_ids[:, None] + route_rank[None, :]
        ).remainder(num_experts).reshape(-1)
        source_rows = local_rows.repeat_interleave(args.top_k)
        origin_rows = torch.arange(local_routes, dtype=torch.int64)
        buffers.write_routes(
            expert_ids,
            source_rows=source_rows,
            origin_rows=origin_rows,
        )
        buffers.prepare(tokens, returned)
        program = build_expert_pool_copy_program(
            buffers,
            benchmark_barrier=nvshmem.benchmark_barrier,
        )

        token_values = torch.arange(
            runtime.pe * args.tokens_per_pe * args.hidden_size,
            (runtime.pe + 1) * args.tokens_per_pe * args.hidden_size,
            dtype=torch.float32,
            device=tokens.device,
        ).view_as(tokens)
        tokens.copy_((token_values.remainder(97) - 48).to(dtype))

        sequence = 1
        buffers.reset_dispatch(sequence)
        torch.cuda.synchronize(runtime.device)
        nvshmem.barrier()

        program.launch()
        torch.cuda.synchronize(runtime.device)

        source_rows_device = source_rows.to(device=tokens.device)
        expected = tokens.index_select(0, source_rows_device)
        torch.testing.assert_close(returned, expected, rtol=0, atol=0)

        status, received_batches, received_rows, returned_experts, observed_sequence = (
            buffers.control_state()
        )
        assert status == ExpertPoolStatus.OK
        assert received_batches == runtime.num_pes * args.experts_per_pe
        assert returned_experts == num_experts
        assert observed_sequence == sequence
        assert received_rows == int(buffers.expert_tails.sum().item())

        nvshmem.barrier()
        print(
            f"PE {runtime.pe}/{runtime.num_pes}: sharded EP PASS "
            f"tokens={args.tokens_per_pe} hidden={args.hidden_size} "
            f"experts={num_experts} received={received_rows} top_k={args.top_k} "
            f"launches=1"
        )
    finally:
        nvshmem.finalize()


if __name__ == "__main__":
    main()
