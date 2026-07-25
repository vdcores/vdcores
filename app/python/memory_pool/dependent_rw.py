"""Dependent write/reduce/read test for the VDCores HBM memory pool.

Run directly for a one-PE, multi-SM smoke test or with ``ibrun`` for the real
multi-PE path.  Every write contributes one float32 tensor to the pool, and
every PE's read is submitted concurrently but cannot fire until all write
tickets have arrived.
"""

from __future__ import annotations

import argparse

import torch

import dae.nvshmem as nvshmem
from dae.instructions import TerminateC, TerminateM
from dae.launcher import Launcher
from dae.memory_pool import (
    MemoryPoolRequest,
    MemoryPoolStatus,
    allocate_memory_pool,
    make_phase_schedule,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--writes-per-pe", type=int, default=8)
    parser.add_argument("--elements", type=int, default=256)
    parser.add_argument("--pool-pe", type=int, default=0)
    parser.add_argument("--symmetric-size", default="512M")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.writes_per_pe <= 0 or args.elements <= 0:
        raise ValueError("writes-per-pe and elements must be positive")

    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    try:
        if not 0 <= args.pool_pe < runtime.num_pes:
            raise ValueError("pool-pe is outside the NVSHMEM PE range")

        requests_per_pe = args.writes_per_pe + 1
        mailbox_count = runtime.num_pes * requests_per_pe
        signal_count = mailbox_count * 2
        total_writes = runtime.num_pes * args.writes_per_pe

        signals = nvshmem.init_signal_space(signal_count)
        buffers = allocate_memory_pool(
            signals,
            mailbox_count=mailbox_count,
            pool_bytes=args.elements * torch.float32.itemsize,
            data_scratch_bytes=args.elements * torch.float32.itemsize,
            route_capacity=1,
            dependency_count=1,
        )
        partials = nvshmem.empty(
            (args.writes_per_pe, args.elements), dtype=torch.float32
        )
        output = nvshmem.zeros(args.elements, dtype=torch.float32)

        base_value = runtime.pe * args.writes_per_pe
        for local_write in range(args.writes_per_pe):
            partials[local_write].fill_(base_value + local_write + 1)

        base_mailbox = runtime.pe * requests_per_pe
        read_mailbox = base_mailbox
        read_request = MemoryPoolRequest.read(
            sequence=1,
            destination=output,
            target_pe=runtime.pe,
            pool_offset=0,
            nbytes=output.nbytes,
            completion_signal=buffers.completion_signal(read_mailbox),
            wait_slot=0,
            wait_value=total_writes,
            user_tag=10_000 + runtime.pe,
        )
        buffers.write_request(read_mailbox, read_request)

        active_mailboxes = [read_mailbox]
        for local_write in range(args.writes_per_pe):
            mailbox = base_mailbox + 1 + local_write
            request = MemoryPoolRequest.write(
                sequence=1,
                source=partials[local_write],
                source_pe=runtime.pe,
                pool_offset=0,
                nbytes=partials[local_write].nbytes,
                completion_pe=runtime.pe,
                completion_signal=buffers.completion_signal(mailbox),
                signal_slot=0,
                signal_delta=1,
                reduce_sum_f32=True,
                user_tag=base_value + local_write + 1,
            )
            buffers.write_request(mailbox, request)
            active_mailboxes.append(mailbox)

        torch.cuda.synchronize(runtime.device)
        nvshmem.barrier()

        schedule = make_phase_schedule(
            buffers,
            active_mailboxes,
            current_pe=runtime.pe,
            pool_pe=args.pool_pe,
            expected_requests=mailbox_count,
        )
        launcher = Launcher(
            num_sms=schedule.num_sms,
            device=torch.device("cuda", runtime.device),
            signal_array=signals,
            benchmark_barrier=nvshmem.benchmark_barrier,
        )
        launcher.i(schedule, TerminateM(), TerminateC())
        launcher.launch()

        expected_value = total_writes * (total_writes + 1) / 2
        expected = torch.full_like(output, expected_value)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if runtime.pe == args.pool_pe:
            status, completed, _, _ = buffers.control_state()
            assert status == MemoryPoolStatus.OK
            assert completed == mailbox_count
            assert buffers.dependencies[0].item() == total_writes

        nvshmem.barrier()
        print(
            f"PE {runtime.pe}/{runtime.num_pes}: dependent RW PASS "
            f"writes={total_writes} value={expected_value:.0f}"
        )
    finally:
        nvshmem.finalize()


if __name__ == "__main__":
    main()
