import torch

import dae.nvshmem as nvshmem
from dae.instructions import (
    Copy,
    IssueBarrier,
    NvshmemPut,
    NvshmemWait,
    TerminateC,
    TerminateM,
    TmaLoad1D,
    TmaStore1D,
)
from dae.launcher import Launcher

NUM_ELEMENTS = 2048
SIGNAL_ID = 0
STORE_BAR_ID = 0


def main() -> None:
    runtime = nvshmem.init(symmetric_size="512M")

    try:
        if runtime.num_pes != 2:
            raise RuntimeError(
                f"This test requires exactly 2 PEs, got {runtime.num_pes}"
            )

        # Symmetric storage for signals.
        signals = nvshmem.init_signal_space(runtime.num_pes)

        # Both PEs must make this symmetric allocation in the same order.
        symmetric_buffer = nvshmem.zeros(
            NUM_ELEMENTS,
            dtype=torch.float32,
        )

        # Local output used by PE 1 for verification.
        received_output = torch.full_like(
            symmetric_buffer,
            -1,
        )

        # PE 0's local input. DAE loads this and stores it into the
        # symmetric buffer before issuing the NVSHMEM PUT.
        source_data = None

        if runtime.pe == 0:
            source_data = torch.arange(
                NUM_ELEMENTS,
                device=symmetric_buffer.device,
                dtype=symmetric_buffer.dtype,
            )

        # Ensure allocation and initialization are complete before either
        # PE launches its DAE schedule.
        torch.cuda.synchronize(runtime.device)
        nvshmem.barrier()

        # Each Python process creates a launcher for its assigned GPU:
        # PE 0 -> PUT schedule
        # PE 1 -> WAIT and receive schedule
        launcher = Launcher(
            num_sms=1,
            device=torch.device("cuda", runtime.device),
            signal_array=signals,
            benchmark_barrier=nvshmem.benchmark_barrier,
        )

        if runtime.pe == 0:
            assert source_data is not None

            # In terms of the order: here is one compute task, and here are the memory operations that feed and drain it.
            launcher.i(
                # Compute instruction used by the load/store copy path.
                Copy(1, symmetric_buffer.nbytes),

                # Local source -> shared-memory slot.
                TmaLoad1D(source_data),

                # Shared-memory slot -> symmetric HBM buffer.
                # Record completion using STORE_BAR_ID.
                TmaStore1D(symmetric_buffer).bar(STORE_BAR_ID),

                # Wait until the barred store has completed.
                IssueBarrier(STORE_BAR_ID),

                # Transfer the buffer and update signal_array[SIGNAL_ID].
                NvshmemPut(
                    address=symmetric_buffer.data_ptr(),
                    nbytes=symmetric_buffer.nbytes,
                    target_pe=1,
                    signal_id=SIGNAL_ID,
                ),

                TerminateM(),
                TerminateC(),
            )

        else:
            launcher.i(
                # Compute instruction used by the load/store copy path.
                Copy(1, symmetric_buffer.nbytes),

                # Wait for PE 0's PUT and signal update.
                NvshmemWait(signal_id=SIGNAL_ID),

                # Read the transferred symmetric buffer.
                TmaLoad1D(symmetric_buffer),

                # Store it into the verification tensor.
                TmaStore1D(received_output),

                TerminateM(),
                TerminateC(),
            )

        launcher.launch()
        torch.cuda.synchronize(runtime.device)

        if runtime.pe == 1:
            expected = torch.arange(
                NUM_ELEMENTS,
                device=received_output.device,
                dtype=received_output.dtype,
            )

            if not torch.equal(received_output, expected):
                max_diff = (
                    received_output - expected
                ).abs().max().item()

                print(
                    "[PE 1] First received values:",
                    received_output[:8].tolist(),
                )

                raise AssertionError(
                    "Remote PUT test failed; "
                    f"max difference={max_diff}"
                )

            print(
                "[PE 1] PASS: received PE 0's data through "
                "VDCores PUT/WAIT."
            )

        nvshmem.barrier()

    finally:
        nvshmem.finalize()


if __name__ == "__main__":
    main()
