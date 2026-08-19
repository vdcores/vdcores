#!/usr/bin/env python3
"""Focused device-counter checks from the full production resident image."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4 import bounded_swiglu, hc_head_reference
from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4HcHeadRms,
    SchedFp8Block128GateUpSwiGlu,
)


def device_envelope_us(launcher: Launcher) -> float:
    profile = launcher.profile[:, :2].cpu().numpy()
    return float(profile[:, 1].max() - profile[:, 0].min()) / 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--op",
        choices=(
            "shared-gate-up-swiglu",
            "hc-head-rms",
        ),
        default="shared-gate-up-swiglu",
    )
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--sms", type=int, default=56)
    parser.add_argument("--limit", type=float, default=10.0)
    parser.add_argument("--epsilon", type=float, default=1.0e-6)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.limit <= 0 or args.epsilon <= 0:
        parser.error("limit and epsilon must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("iterations must be positive and warmup non-negative")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260819)
    if args.op == "shared-gate-up-swiglu":
        args.width = 2048
        k = 4096
        if not 0 < args.sms <= args.width:
            parser.error("shared gate/up SwiGLU sms must be in [1,2048]")
        launcher = Launcher(args.sms, device=device)
        gate_source = torch.randn(
            (args.width, k), generator=generator,
            dtype=torch.bfloat16, device=device,
        ) * 0.05
        up_source = torch.randn(
            (args.width, k), generator=generator,
            dtype=torch.bfloat16, device=device,
        ) * 0.05
        activation_source = torch.randn(
            (k,), generator=generator,
            dtype=torch.bfloat16, device=device,
        ) * 0.1
        gate, gate_scale = quantize_fp8_block128(gate_source)
        up, up_scale = quantize_fp8_block128(up_source)
        activation, activation_scale = quantize_fp8_block128(
            activation_source
        )
        output = torch.empty((args.width,), dtype=torch.bfloat16, device=device)
        schedule = SchedFp8Block128GateUpSwiGlu(
            gate,
            gate_scale,
            up,
            up_scale,
            activation,
            activation_scale,
            output,
            swiglu_limit=args.limit,
        ).place(args.sms)
        activation_reference = dequantize_fp8_block128(
            activation, activation_scale
        )
        gate_reference = torch.mv(
            dequantize_fp8_block128(gate, gate_scale), activation_reference
        ).to(torch.bfloat16)
        up_reference = torch.mv(
            dequantize_fp8_block128(up, up_scale), activation_reference
        ).to(torch.bfloat16)
        expected = bounded_swiglu(
            gate_reference, up_reference, limit=args.limit
        )
        rtol, atol = 3.0e-2, 1.0e-1
    else:
        launcher = Launcher(1, device=device)
        args.width = 4096
        residual = torch.randn(
            (4, args.width), generator=generator,
            dtype=torch.bfloat16, device=device,
        ) * 0.125
        mixes = torch.randn(
            (4,), generator=generator, dtype=torch.float32, device=device,
        ) * 0.1
        scale = torch.tensor([0.625], dtype=torch.float32, device=device)
        base = torch.randn(
            (4,), generator=generator, dtype=torch.float32, device=device,
        ) * 0.1
        weight = (
            torch.randn(
                (args.width,), generator=generator,
                dtype=torch.bfloat16, device=device,
            )
            * 0.05
            + 1.0
        )
        output = torch.empty(
            (args.width,), dtype=torch.bfloat16, device=device
        )
        schedule = SchedDsv4HcHeadRms(
            residual,
            mixes,
            scale,
            base,
            weight,
            output,
            rms_epsilon=args.epsilon,
        ).place(1)
        expected_head = hc_head_reference(residual, mixes, scale, base)
        expected = (
            expected_head.float()
            * torch.rsqrt(
                expected_head.float().square().mean() + args.epsilon
            )
            * weight.float()
        ).to(torch.bfloat16)
        rtol, atol = 3.0e-2, 3.0e-2
    launcher.s(schedule)

    launcher.launch()
    torch.cuda.synchronize(device)
    cold_us = device_envelope_us(launcher)
    torch.testing.assert_close(output, expected, rtol=rtol, atol=atol)

    for _ in range(args.warmup):
        launcher.launch()
    torch.cuda.synchronize(device)
    timings = []
    for _ in range(args.iterations):
        launcher.launch()
        timings.append(device_envelope_us(launcher))

    error = (output.float() - expected.float()).abs()
    print(
        "DSV4_VECTOR_OP_RESULT "
        f"op={args.op.replace('-', '_')} "
        f"width={args.width} cold_device_us={cold_us:.6f} "
        f"hot_min_us={min(timings):.6f} "
        f"hot_median_us={statistics.median(timings):.6f} "
        f"hot_max_us={max(timings):.6f} "
        f"max_abs={error.max().item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
