#!/usr/bin/env python3
"""Correctness and latency benchmark for the DeepSeek-V4 FP8 GEMV task."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from dae.launcher import Launcher
from dae.schedule import SchedFp8Block128Gemv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trace-stages", action="store_true")
    args = parser.parse_args()

    def stage(name: str) -> None:
        if args.trace_stages:
            torch.cuda.synchronize()
            print(f"DSV4_FP8_STAGE {name}", flush=True)

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810)
    weight_source = torch.randn(
        (args.m, args.k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.05
    input_source = torch.randn(
        (args.k,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.1
    stage("sources_ready")
    weight, weight_scale = quantize_fp8_block128(weight_source)
    activation, activation_scale = quantize_fp8_block128(input_source)
    output = torch.empty((args.m,), dtype=torch.bfloat16, device=device)
    stage("quantized")

    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    num_sms = args.sms or min(args.m, device_sms)
    launcher = Launcher(num_sms, device=device)
    launcher.s(
        SchedFp8Block128Gemv(
            weight, weight_scale, activation, activation_scale, output
        ).place(num_sms)
    )
    stage("launcher_ready")
    launcher.launch()
    torch.cuda.synchronize()
    stage("first_launch_complete")

    reference = (
        dequantize_fp8_block128(weight, weight_scale)
        @ dequantize_fp8_block128(activation, activation_scale)
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=5e-2)
    max_abs = (output.float() - reference.float()).abs().max().item()
    mean_rel = (
        (output.float() - reference.float()).abs().mean()
        / reference.float().abs().mean().clamp_min(1.0e-8)
    ).item()

    for _ in range(args.warmup):
        launcher.launch()
    torch.cuda.synchronize()
    timings = []
    for _ in range(args.iterations):
        launcher.launch()
        profile = launcher.profile[:, :2].cpu().numpy()
        timings.append((profile[:, 1].max() - profile[:, 0].min()) / 1.0e3)

    print(
        "DSV4_FP8_GEMV_RESULT "
        f"shape={args.m}x1x{args.k} sms={num_sms} "
        f"min_us={min(timings):.6f} median_us={statistics.median(timings):.6f} "
        f"max_us={max(timings):.6f} max_abs={max_abs:.6f} "
        f"mean_relative={mean_rel:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
