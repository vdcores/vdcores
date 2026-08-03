"""Minimal DAE/NVSHMEM example for TACC ``ibrun``."""

import torch

import dae.nvshmem as nvshmem
from dae.instructions import Copy, TerminateC, TerminateM, TmaLoad1D, TmaStore1D
from dae.launcher import Launcher


def main() -> None:
    runtime = nvshmem.init(symmetric_size="512M")
    try:
        signals = nvshmem.init_signal_space(runtime.num_pes)
        symmetric_input = nvshmem.empty(2048, dtype=torch.float32)
        symmetric_output = nvshmem.zeros(2048, dtype=torch.float32)
        symmetric_input.copy_(
            torch.arange(2048, device=symmetric_input.device) + runtime.pe * 4096
        )

        launcher = Launcher(
            num_sms=1,
            device=torch.device("cuda", runtime.device),
            signal_array=signals,
            benchmark_barrier=nvshmem.benchmark_barrier,
        )
        launcher.i(
            Copy(1, symmetric_input.nbytes),
            TmaLoad1D(symmetric_input),
            TmaStore1D(symmetric_output),
            TerminateM(),
            TerminateC(),
        )
        launcher.launch()
        assert torch.equal(symmetric_output, symmetric_input)

        next_pe = (runtime.pe + 1) % runtime.num_pes
        previous_pe = (runtime.pe - 1) % runtime.num_pes
        nvshmem.signal(runtime.pe, runtime.pe + 1, next_pe)
        nvshmem.wait_signal(previous_pe, previous_pe + 1)
        torch.cuda.synchronize(runtime.device)
        nvshmem.barrier()

        print(
            f"PE {runtime.pe}/{runtime.num_pes} device={runtime.device} "
            f"copy=[{symmetric_output[0].item():.0f}, "
            f"{symmetric_output[-1].item():.0f}] "
            f"received_signal={signals[previous_pe].item()}"
        )
    finally:
        # Collective finalization releases module state and symmetric tensors.
        nvshmem.finalize()


if __name__ == "__main__":
    main()
