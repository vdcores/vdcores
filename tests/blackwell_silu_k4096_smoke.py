#!/usr/bin/env python3
"""Hardware regression for the existing 4096-wide interleaved SwiGLU op."""

import torch

from dae.launcher import Launcher
from dae.schedule import SchedSmemSiLUInterleaved


def main() -> None:
    device = torch.device("cuda")
    gate = torch.linspace(-4.0, 4.0, 4096, dtype=torch.float32, device=device)[
        None
    ].to(torch.bfloat16)
    up = torch.linspace(1.0, -1.0, 4096, dtype=torch.float32, device=device)[
        None
    ].to(torch.bfloat16)
    output = torch.zeros_like(gate)

    launcher = Launcher(1, device=device)
    launcher.s(SchedSmemSiLUInterleaved(1, gate, up, output).place(1))
    launcher.launch()

    expected = (torch.nn.functional.silu(gate.float()) * up.float()).to(
        torch.bfloat16
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    print("BLACKWELL_SILU_K4096 status=PASS elements=4096", flush=True)


if __name__ == "__main__":
    main()
