import os

import torch

from dae.instructions import Copy, RepeatM, TerminateC, TerminateM, TmaLoad1D, TmaStore1D
from dae.launcher import Launcher


def main() -> None:
    num_sms = int(os.environ.get("DAE_SMOKE_SMS", "1"))
    num_copies = int(os.environ.get("DAE_SMOKE_COPIES", "8"))
    copy_bytes = int(os.environ.get("DAE_SMOKE_BYTES", str(8 * 1024)))
    if copy_bytes % 4 != 0:
        raise ValueError("DAE_SMOKE_BYTES must be divisible by four")

    device = torch.device("cuda")
    source = torch.arange(
        num_sms * num_copies * (copy_bytes // 4),
        dtype=torch.int32,
        device=device,
    ).reshape(num_sms, num_copies, copy_bytes // 4)
    destination = torch.zeros_like(source)

    launcher = Launcher(num_sms, device=device)

    def memory_tasks(sm: int):
        return RepeatM.on(
            num_copies,
            (TmaLoad1D(source[sm, 0]), copy_bytes),
            (TmaStore1D(destination[sm, 0]), copy_bytes),
        )

    launcher.i(
        Copy(num_copies, copy_bytes),
        memory_tasks,
        TerminateM(),
        TerminateC(),
    )
    launcher.launch()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)

    destination.zero_()
    launcher.launch(synchronize=False, reset_bars=False)
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)

    destination.zero_()
    launcher.launch_sequence(
        [launcher.loop_counters.copy(), launcher.loop_counters.copy()],
        synchronize=False,
        reset_bars=True,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(destination, source, rtol=0, atol=0)

    props = torch.cuda.get_device_properties(device)
    print(
        "blackwell runtime smoke passed:",
        f"device={props.name}",
        f"cc={props.major}.{props.minor}",
        f"sms_launched={num_sms}",
        f"copies_per_sm={num_copies}",
        f"bytes_per_copy={copy_bytes}",
        "sequence_launches=2",
    )


if __name__ == "__main__":
    main()
