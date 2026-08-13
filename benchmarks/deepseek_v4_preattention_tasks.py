#!/usr/bin/env python3
"""Focused same-shape probes for clean-room pre-attention projection tasks."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Bf16Gemv,
    SchedDsv4Bf16GemvGroup4SplitK,
)
from dae.tma_utils import Major


def time_launcher(
    launcher: Launcher,
    *,
    output: torch.Tensor,
    reduce_output: bool,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        if reduce_output:
            output.zero_()
        launcher.launch()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iterations):
        if reduce_output:
            output.zero_()
        torch.cuda.synchronize()
        start.record()
        launcher.launch(synchronize=False)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1.0e3)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, choices=(512, 1024, 2048), default=2048)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260813)
    k = 4096
    weight = torch.randn(
        (args.rows, k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.01
    source = torch.randn(
        (k,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.01
    reference = weight.float() @ source.float()

    scalar_output = torch.empty((args.rows,), dtype=torch.float32, device=device)
    scalar_launcher = Launcher(152, device=device)
    scalar_launcher.s(
        SchedDsv4Bf16Gemv(weight, source, scalar_output).place(152)
    )
    scalar_us = time_launcher(
        scalar_launcher,
        output=scalar_output,
        reduce_output=False,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    torch.testing.assert_close(
        scalar_output, reference, rtol=1.0e-5, atol=1.0e-5
    )
    print(
        f"DSV4_PREATTENTION_TASK task=bf16_scalar rows={args.rows} k={k} "
        f"sms=152 median_us={scalar_us:.3f}",
        flush=True,
    )

    for split_k in (1, 2, 4, 8):
        work_items = args.rows // 512 * split_k
        grouped_output = torch.zeros(
            (args.rows // 128, 128), dtype=torch.float32, device=device
        )
        launcher = Launcher(152, device=device)
        weight_tma = TmaTensor(launcher, weight).wgmma_load(
            128, 128, Major.K
        )
        output_tma = TmaTensor(launcher, grouped_output).rowmajor_2d(
            "reduce", 4, 128
        )
        launcher.s(
            SchedDsv4Bf16GemvGroup4SplitK(
                weight,
                weight_tma,
                source,
                output_tma,
                split_k,
            ).place(work_items)
        )
        grouped_us = time_launcher(
            launcher,
            output=grouped_output,
            reduce_output=True,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        delta = grouped_output.reshape(-1) - reference
        torch.testing.assert_close(
            grouped_output.reshape(-1), reference, rtol=2.0e-3, atol=2.0e-3
        )
        print(
            "DSV4_PREATTENTION_TASK "
            f"task=bf16_group4_splitk rows={args.rows} k={k} "
            f"split_k={split_k} sms={work_items} median_us={grouped_us:.3f} "
            f"max_abs={delta.abs().max().item():.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
