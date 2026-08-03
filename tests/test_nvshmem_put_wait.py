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


SIZE = 2048


def test_nvshmem_put_wait():
    runtime = nvshmem.init(symmetric_size="512M")

    values = None
    if runtime.pe == 0:
        values = torch.arange(
            SIZE,
            dtype=torch.float32,
            device="cuda",
        )

    try:
        if runtime.num_pes != 2:
            raise RuntimeError(
                f"Expected 2 PEs, got {runtime.num_pes}"
            )

        signal = nvshmem.init_signal_space(runtime.num_pes)
        buffer = nvshmem.zeros(
            SIZE,
            dtype=torch.float32
        )

        nvshmem.barrier()

        result = torch.full_like(buffer, -1)

        nvshmem.barrier()

        put_inst = NvshmemPut(
            address=buffer.data_ptr(),
            nbytes=buffer.nbytes,
            target_pe=1,
            signal_id=0,
        )
        wait_inst = NvshmemWait(signal_id=0)

        assert put_inst.num_slots == 0
        assert wait_inst.num_slots == 0

        launcher = Launcher(
            num_sms=1,
            device=torch.device("cuda", runtime.device),
            signal_array=signal,
            benchmark_barrier=nvshmem.benchmark_barrier,
        )

        if runtime.pe == 0:
            assert values is not None

            launcher.i(
                Copy(1, buffer.nbytes),
                TmaLoad1D(values),
                TmaStore1D(buffer).bar(0),
                IssueBarrier(0),
                put_inst,
                TerminateM(),
                TerminateC(),
            )
        else:
            launcher.i(
                Copy(1, buffer.nbytes),
                wait_inst,
                TmaLoad1D(buffer),
                TmaStore1D(result),
                TerminateM(),
                TerminateC(),
            )

        launcher.launch()
        torch.cuda.synchronize(runtime.device)

        if runtime.pe == 1:
            expected = torch.arange(
                SIZE,
                device=result.device,
                dtype=torch.float32,
            )

            assert torch.equal(result, expected), (
                "NVSHMEM PUT/WAIT result mismatch"
            )

        if runtime.pe == 1:
            print("NVSHMEM PUT/WAIT test passed", flush=True)

        nvshmem.barrier()

    finally:
        try:
            nvshmem.finalize()
        except Exception as e:
            print("Finalize error:", e, flush=True)


if __name__ == "__main__":
    test_nvshmem_put_wait()