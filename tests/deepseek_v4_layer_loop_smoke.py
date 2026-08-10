#!/usr/bin/env python3
"""Compact two-layer loop: indirect LDU source plus counter-strided STU output."""

from __future__ import annotations

import torch

from dae.instructions import (
    CounterOffsetMemoryInstruction,
    Dsv4Rope128_64,
    TmaLoad1D,
    TmaStore1D,
)
from dae.launcher import Launcher
from dae.schedule import LayeredSchedule, Schedule
from dae.sequential import LoopedSequentialProgram, SequentialBlock, SequentialStage


class IdentityRope(Schedule):
    def __init__(self, source, table, destination, output_stride):
        super().__init__()
        self.source = source
        self.table = table
        self.destination = destination
        self.output_stride = output_stride

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("identity RoPE smoke uses one SM")

    def schedule(self, sm):
        if sm != 0:
            return []
        return [
            Dsv4Rope128_64(1, False),
            TmaLoad1D(self.source),
            TmaLoad1D(self.table),
            CounterOffsetMemoryInstruction(
                0,
                TmaStore1D(self.destination).bar(self._bar("output")),
                self.output_stride,
            ),
        ]

    def bar_release_count(self, role: str):
        return self._bar_release_if_present(role, 1) if role == "output" else 0


def main() -> None:
    device = torch.device("cuda")
    source = torch.arange(256, dtype=torch.float32, device=device).to(
        torch.bfloat16
    ).reshape(2, 128)
    destination = torch.zeros_like(source)
    table = torch.zeros((32, 2), dtype=torch.float32, device=device)
    table[:, 0] = 1.0
    row_bytes = source.shape[1] * source.element_size()

    base = IdentityRope(source[0], table, destination[0], row_bytes)
    layered = LayeredSchedule(
        base,
        ((source[0], (source[0], source[1])),),
        counter_strides=((0, 1),),
    )
    launcher = Launcher(1, device=device)
    program = LoopedSequentialProgram(
        launcher,
        (
            SequentialBlock(
                "two_layers",
                (SequentialStage("identity", layered, 1),),
                repeat=2,
                barrier_banks=2,
            ),
        ),
    )
    launcher.s(program)
    launcher.launch()

    torch.testing.assert_close(destination, source, rtol=0, atol=0)
    print(
        "DSV4_LAYER_LOOP status=PASS launches=1 layers=2 barrier_banks=2 "
        f"compute_insts={program.max_compute_instructions} "
        f"memory_insts={program.max_memory_instructions}",
        flush=True,
    )


if __name__ == "__main__":
    main()
