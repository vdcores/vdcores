"""Minimal DAE/NVSHMEM example for TACC ``ibrun``."""

import torch

import dae.nvshmem as nvshmem
from dae.instructions import Copy, TerminateC, TerminateM, TmaLoad1D, TmaStore1D
from dae.nvshmem_launcher import Launcher


def main() -> None:
    launcher = Launcher(num_sms=1, symmetric_size="512M")
    signals = launcher.init_signal_space(launcher.num_pes)
    symmetric_input = launcher.empty(2048, dtype=torch.float32)
    symmetric_output = launcher.zeros(2048, dtype=torch.float32)
    symmetric_input.copy_(
        torch.arange(2048, device=symmetric_input.device) + launcher.pe * 4096
    )

    assert signals.dtype == torch.uint64
    assert nvshmem.is_symmetric_tensor(signals)
    assert nvshmem.is_symmetric_tensor(symmetric_input)
    assert nvshmem.is_symmetric_tensor(symmetric_output)

    launcher.i(
        Copy(1, symmetric_input.nbytes),
        TmaLoad1D(symmetric_input),
        TmaStore1D(symmetric_output),
        TerminateM(),
        TerminateC(),
    )
    launcher.launch()
    assert torch.equal(symmetric_output, symmetric_input)

    next_pe = (launcher.pe + 1) % launcher.num_pes
    previous_pe = (launcher.pe - 1 + launcher.num_pes) % launcher.num_pes
    launcher.signal(launcher.pe, launcher.pe + 1, next_pe)
    launcher.wait_signal(previous_pe, previous_pe + 1)
    torch.cuda.synchronize()
    launcher.barrier()

    print(
        f"PE {launcher.pe}/{launcher.num_pes} device={launcher.nvshmem_info.device} "
        f"copy=[{symmetric_output[0].item():.0f}, {symmetric_output[-1].item():.0f}] "
        f"received_signal={signals[previous_pe].item()}"
    )

    # finalize() is collective and invalidates every symmetric tensor.
    launcher.signal_space = None
    del signals, symmetric_input, symmetric_output
    launcher.finalize()


if __name__ == "__main__":
    main()
