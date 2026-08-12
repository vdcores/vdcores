#!/usr/bin/env python3
"""Hardware smoke for an in-queue, output-only zero fill."""

import argparse

import torch

from dae.launcher import Launcher
from dae.schedule import SchedDsv4ZeroFill


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=32768)
    parser.add_argument("--sms", type=int, default=152)
    args = parser.parse_args()
    if args.elements <= 0 or args.elements % 8:
        parser.error("elements must be a positive multiple of eight BF16 values")

    device = torch.device("cuda")
    output = torch.ones((args.elements,), dtype=torch.bfloat16, device=device)
    gate = torch.zeros((1,), dtype=torch.uint32, device=device)
    launcher = Launcher(args.sms, device=device)
    launcher.s(SchedDsv4ZeroFill(gate, output).place(args.sms))
    launcher.launch()
    torch.cuda.synchronize(device)
    nonzero = int(torch.count_nonzero(output).item())
    if nonzero:
        raise AssertionError(f"zero-fill left {nonzero} nonzero elements")
    print(
        "BLACKWELL_ZERO_FILL status=PASS "
        f"elements={args.elements} sms={args.sms}",
        flush=True,
    )


if __name__ == "__main__":
    main()
