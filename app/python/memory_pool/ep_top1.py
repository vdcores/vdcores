"""Top-1 expert-parallel scatter/compute/gather over the VDCores pool."""

from __future__ import annotations

import argparse

import torch

import dae.nvshmem as nvshmem
from dae.instructions import Copy, TerminateC, TerminateM, TmaLoad1D, TmaStore1D
from dae.launcher import Launcher
from dae.memory_pool import (
    MemoryPoolRequest,
    MemoryPoolStatus,
    allocate_memory_pool,
    make_phase_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--pool-pe", type=int, default=0)
    parser.add_argument("--symmetric-size", default="512M")
    return parser.parse_args()


def run_phase(
    runtime,
    buffers,
    active_mailboxes,
    pool_pe,
    expected_requests,
    *,
    vdcores_copy=None,
):
    base_schedule = make_phase_schedule(
        buffers,
        active_mailboxes,
        current_pe=runtime.pe,
        pool_pe=pool_pe,
        expected_requests=expected_requests,
    )
    memory_schedule = base_schedule
    compute_schedule = None
    if vdcores_copy is not None:
        copy_mailbox, source, destination = vdcores_copy
        copy_index = active_mailboxes.index(copy_mailbox)
        copy_sm = 1 + copy_index
        copy_bytes = source.nbytes
        if destination.nbytes != copy_bytes:
            raise ValueError("VDCores expert copy tensors must have equal byte sizes")
        if copy_bytes > 0xFFFF:
            raise ValueError("VDCores expert copy harness is limited to 65535 bytes")

        def memory_schedule(sm):
            instructions = list(base_schedule(sm))
            if sm == copy_sm:
                instructions.extend(
                    (TmaLoad1D(source), TmaStore1D(destination))
                )
            return instructions

        def compute_schedule(sm):
            return [Copy(1, copy_bytes)] if sm == copy_sm else []

    launcher = Launcher(
        num_sms=base_schedule.num_sms,
        device=torch.device("cuda", runtime.device),
        signal_array=buffers.signals,
        benchmark_barrier=nvshmem.benchmark_barrier,
    )
    launcher.i(memory_schedule, compute_schedule, TerminateM(), TerminateC())
    launcher.launch()


def main() -> None:
    args = parse_args()
    if args.tokens_per_pe <= 0 or args.hidden_size <= 0:
        raise ValueError("tokens-per-pe and hidden-size must be positive")

    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    try:
        if not 0 <= args.pool_pe < runtime.num_pes:
            raise ValueError("pool-pe is outside the NVSHMEM PE range")

        tokens_per_pe = args.tokens_per_pe
        total_tokens = runtime.num_pes * tokens_per_pe
        row_bytes = args.hidden_size * torch.float32.itemsize
        pool_rows = runtime.num_pes * total_tokens
        mailbox_count = runtime.num_pes * 2

        signals = nvshmem.init_signal_space(mailbox_count * 2)
        buffers = allocate_memory_pool(
            signals,
            mailbox_count=mailbox_count,
            pool_bytes=pool_rows * row_bytes,
            data_scratch_bytes=row_bytes,
            route_capacity=total_tokens,
            dependency_count=2,
        )
        tokens = nvshmem.empty(
            (tokens_per_pe, args.hidden_size), dtype=torch.float32
        )
        expert_input = nvshmem.zeros(
            (total_tokens, args.hidden_size), dtype=torch.float32
        )
        expert_output = nvshmem.zeros(
            (total_tokens, args.hidden_size), dtype=torch.float32
        )
        final_output = nvshmem.zeros(
            (tokens_per_pe, args.hidden_size), dtype=torch.float32
        )

        local_global_ids = [
            runtime.pe * tokens_per_pe + token for token in range(tokens_per_pe)
        ]
        token_values = torch.arange(
            runtime.pe * tokens_per_pe * args.hidden_size,
            (runtime.pe + 1) * tokens_per_pe * args.hidden_size,
            dtype=torch.float32,
            device=tokens.device,
        ).view_as(tokens)
        tokens.copy_(token_values / 100.0)

        original_pool_rows = [
            (global_id % runtime.num_pes) * total_tokens + global_id
            for global_id in local_global_ids
        ]
        expert_global_ids = [
            global_id
            for global_id in range(total_tokens)
            if global_id % runtime.num_pes == runtime.pe
        ]
        expert_pool_rows = [runtime.pe * total_tokens + global_id for global_id in expert_global_ids]
        expert_token_count = len(expert_pool_rows)

        scatter_mailbox = runtime.pe * 2
        gather_mailbox = scatter_mailbox + 1

        original_routes = buffers.write_routes(scatter_mailbox, original_pool_rows)
        expert_routes = buffers.write_routes(gather_mailbox, expert_pool_rows)
        buffers.write_request(
            scatter_mailbox,
            MemoryPoolRequest.scatter(
                sequence=1,
                source=tokens,
                routes=original_routes,
                source_pe=runtime.pe,
                pool_offset=0,
                row_count=tokens_per_pe,
                row_bytes=row_bytes,
                completion_pe=runtime.pe,
                completion_signal=buffers.completion_signal(scatter_mailbox),
                signal_slot=0,
                signal_delta=1,
                user_tag=100 + runtime.pe,
            ),
        )
        buffers.write_request(
            gather_mailbox,
            MemoryPoolRequest.gather(
                sequence=1,
                destination=expert_input,
                routes=expert_routes,
                target_pe=runtime.pe,
                pool_offset=0,
                row_count=expert_token_count,
                row_bytes=row_bytes,
                completion_signal=buffers.completion_signal(gather_mailbox),
                wait_slot=0,
                wait_value=runtime.num_pes,
                user_tag=200 + runtime.pe,
            ),
        )

        torch.cuda.synchronize(runtime.device)
        nvshmem.barrier()
        run_phase(
            runtime,
            buffers,
            [scatter_mailbox, gather_mailbox],
            args.pool_pe,
            mailbox_count,
            vdcores_copy=(
                gather_mailbox,
                expert_input[:expert_token_count],
                expert_output[:expert_token_count],
            ),
        )

        # Expert-specific local compute. Communication on both sides of this
        # function is performed only by VDCores memory-pool operators, and the
        # input/output handoff above is an ordinary VDCores load/Copy/store.
        torch.testing.assert_close(
            expert_output[:expert_token_count],
            expert_input[:expert_token_count],
            rtol=0,
            atol=0,
        )
        expert_output[:expert_token_count].mul_(runtime.pe + 1).add_(
            runtime.pe * 0.25
        )

        expert_routes = buffers.write_routes(scatter_mailbox, expert_pool_rows)
        original_routes = buffers.write_routes(gather_mailbox, original_pool_rows)
        buffers.write_request(
            scatter_mailbox,
            MemoryPoolRequest.scatter(
                sequence=2,
                source=expert_output,
                routes=expert_routes,
                source_pe=runtime.pe,
                pool_offset=0,
                row_count=expert_token_count,
                row_bytes=row_bytes,
                completion_pe=runtime.pe,
                completion_signal=buffers.completion_signal(scatter_mailbox),
                signal_slot=1,
                signal_delta=1,
                user_tag=300 + runtime.pe,
            ),
        )
        buffers.write_request(
            gather_mailbox,
            MemoryPoolRequest.gather(
                sequence=2,
                destination=final_output,
                routes=original_routes,
                target_pe=runtime.pe,
                pool_offset=0,
                row_count=tokens_per_pe,
                row_bytes=row_bytes,
                completion_signal=buffers.completion_signal(gather_mailbox),
                wait_slot=1,
                wait_value=runtime.num_pes,
                user_tag=400 + runtime.pe,
            ),
        )
        torch.cuda.synchronize(runtime.device)
        run_phase(
            runtime,
            buffers,
            [scatter_mailbox, gather_mailbox],
            args.pool_pe,
            mailbox_count,
        )

        expert_ids = torch.tensor(
            [global_id % runtime.num_pes for global_id in local_global_ids],
            dtype=torch.float32,
            device=tokens.device,
        ).unsqueeze(1)
        expected = tokens * (expert_ids + 1) + expert_ids * 0.25
        torch.testing.assert_close(final_output, expected, rtol=0, atol=0)

        if runtime.pe == args.pool_pe:
            status, completed, _, _ = buffers.control_state()
            assert status == MemoryPoolStatus.OK
            assert completed == mailbox_count
            assert buffers.dependencies[:2].cpu().tolist() == [
                runtime.num_pes,
                runtime.num_pes,
            ]
        nvshmem.barrier()
        print(
            f"PE {runtime.pe}/{runtime.num_pes}: top-1 EP PASS "
            f"tokens={tokens_per_pe} hidden={args.hidden_size}"
        )
    finally:
        nvshmem.finalize()


if __name__ == "__main__":
    main()
