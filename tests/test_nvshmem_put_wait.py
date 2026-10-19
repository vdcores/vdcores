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

DTYPE = torch.float32
NUM_ELEMENTS = 2048
SIGNAL_ID = 0
STORE_BAR_ID = 0


def generate_data(pe_id: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(NUM_ELEMENTS, device=device, dtype=DTYPE)
    return base + float(pe_id * NUM_ELEMENTS)


def main() -> None:
    runtime = nvshmem.init(symmetric_size="512M")

    try:
        num_pes = runtime.num_pes
        if num_pes < 2:
            raise RuntimeError(f"This test requires at least 2 PEs, got {num_pes}")

        pe = runtime.pe
        next_pe = (pe + 1) % num_pes
        prev_pe = (pe - 1) % num_pes

        signals = nvshmem.init_signal_space(num_pes)

        # One slot per PE. Each PE only writes its own slot locally and only
        # receives into its predecessor's slot, so the shared source/destination
        # address of the PUT never collides with a neighbour's traffic.
        symmetric_buffer = nvshmem.zeros(num_pes * NUM_ELEMENTS, dtype=DTYPE)
        send_slot = symmetric_buffer[pe * NUM_ELEMENTS:(pe + 1) * NUM_ELEMENTS]
        recv_slot = symmetric_buffer[prev_pe * NUM_ELEMENTS:(prev_pe + 1) * NUM_ELEMENTS]

        device = symmetric_buffer.device
        source_data = generate_data(pe, device)
        received_output = torch.full_like(source_data, -1)

        torch.cuda.synchronize(runtime.device)
        nvshmem.barrier()

        # Every PE runs the same schedule: stage its payload, send it to the
        # next PE, wait for the previous PE, and read the result back.
        launcher = Launcher(
            num_sms=1,
            device=torch.device("cuda", runtime.device),
            signal_array=signals,
            benchmark_barrier=nvshmem.benchmark_barrier,
        )

        launcher.i(
            Copy(2, send_slot.nbytes),

            TmaLoad1D(source_data),
            TmaStore1D(send_slot).bar(STORE_BAR_ID),
            IssueBarrier(STORE_BAR_ID),

            NvshmemPut(
                address=send_slot.data_ptr(),
                nbytes=send_slot.nbytes,
                target_pe=next_pe,
                signal_id=SIGNAL_ID,
            ),

            NvshmemWait(signal_id=SIGNAL_ID),

            TmaLoad1D(recv_slot),
            TmaStore1D(received_output),

            TerminateM(),
            TerminateC(),
        )

        launcher.launch()
        torch.cuda.synchronize(runtime.device)

        expected = generate_data(prev_pe, received_output.device)

        if not torch.equal(received_output, expected):
            max_diff = (received_output - expected).abs().max().item()

            print(f"[PE {pe}] First received values:", received_output[:8].tolist())

            raise AssertionError(
                f"Ring PUT test failed on PE {pe} (sender {prev_pe}); "
                f"max difference={max_diff}"
            )

        print(f"[PE {pe}] PASS: received PE {prev_pe}'s data through VDCores PUT/WAIT.")

        nvshmem.barrier()

    finally:
        nvshmem.finalize()


if __name__ == "__main__":
    main()