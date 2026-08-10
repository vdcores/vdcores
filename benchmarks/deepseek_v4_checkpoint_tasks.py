#!/usr/bin/env python3
"""Run representative VDCores tasks against raw DeepSeek-V4 checkpoint tensors."""

from __future__ import annotations

import argparse

import torch

from dae.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from dae.deepseek_v4_quant import dequantize_fp8_block128, dequantize_nvfp4
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Fp8Quant128,
    SchedDsv4Nvfp4Quant16,
    SchedFp8Block128Gemv,
    SchedNvfp4Gemv,
)


def _run_fp8(
    checkpoint: DeepSeekV4Checkpoint,
    prefix: str,
    device: torch.device,
    sms: int,
) -> float:
    linear = checkpoint.load_fp8_linear(prefix, device=str(device))
    rows, k = linear.weight.shape
    source = torch.linspace(-0.03125, 0.03125, k, dtype=torch.bfloat16, device=device)
    activation = torch.empty((k,), dtype=torch.float8_e4m3fn, device=device)
    activation_scale = torch.empty(
        (k // 128,), dtype=torch.float8_e8m0fnu, device=device
    )
    output = torch.empty((rows,), dtype=torch.bfloat16, device=device)

    quant_sms = min(k // 128, sms)
    quant = Launcher(quant_sms, device=device)
    quant.s(
        SchedDsv4Fp8Quant128(source, activation, activation_scale).place(quant_sms)
    )
    gemv_sms = min(rows, sms)
    gemv = Launcher(gemv_sms, device=device)
    gemv.s(
        SchedFp8Block128Gemv(
            linear.weight,
            linear.scale,
            activation,
            activation_scale,
            output,
        ).place(gemv_sms)
    )
    quant.launch(synchronize=False)
    gemv.launch()
    torch.cuda.synchronize()

    reference = (
        dequantize_fp8_block128(linear.weight, linear.scale)
        @ dequantize_fp8_block128(activation, activation_scale)
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=5e-2)
    return float((output.float() - reference.float()).abs().max().item())


def _run_nvfp4(
    checkpoint: DeepSeekV4Checkpoint,
    prefix: str,
    device: torch.device,
    sms: int,
) -> float:
    linear = checkpoint.load_nvfp4_linear(prefix, device=str(device))
    rows, packed_k = linear.weight.shape
    k = packed_k * 2
    source = torch.linspace(-0.015625, 0.015625, k, dtype=torch.bfloat16, device=device)
    activation = torch.empty((packed_k,), dtype=torch.uint8, device=device)
    activation_scale = torch.empty(
        (k // 16,), dtype=torch.float8_e4m3fn, device=device
    )
    output = torch.empty((rows,), dtype=torch.bfloat16, device=device)

    quant_sms = min(k // 16, sms)
    quant = Launcher(quant_sms, device=device)
    quant.s(
        SchedDsv4Nvfp4Quant16(
            source,
            linear.input_scale.reshape(1),
            activation,
            activation_scale,
        ).place(quant_sms)
    )
    gemv_sms = min(rows, sms)
    gemv = Launcher(gemv_sms, device=device)
    gemv.s(
        SchedNvfp4Gemv(
            linear.weight,
            linear.weight_scale,
            activation,
            activation_scale,
            linear.alpha,
            output,
        ).place(gemv_sms)
    )
    quant.launch(synchronize=False)
    gemv.launch()
    torch.cuda.synchronize()

    reference = (
        dequantize_nvfp4(
            linear.weight,
            linear.weight_scale,
            linear.weight_scale_2,
        )
        @ dequantize_nvfp4(
            activation,
            activation_scale,
            linear.input_scale,
        )
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=5e-2)
    return float((output.float() - reference.float()).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp8-prefix", default="layers.2.attn.wq_a")
    parser.add_argument(
        "--nvfp4-prefix", default="layers.2.ffn.experts.0.w1"
    )
    parser.add_argument("--sms", type=int, default=152)
    args = parser.parse_args()
    if args.sms <= 0:
        parser.error("sms must be positive")

    device = torch.device("cuda")
    sms = min(args.sms, torch.cuda.get_device_properties(device).multi_processor_count)
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint)
    fp8_max_abs = _run_fp8(checkpoint, args.fp8_prefix, device, sms)
    nvfp4_max_abs = _run_nvfp4(checkpoint, args.nvfp4_prefix, device, sms)
    print(
        "DSV4_CHECKPOINT_TASKS status=PASS "
        f"fp8={args.fp8_prefix} fp8_max_abs={fp8_max_abs:.6f} "
        f"nvfp4={args.nvfp4_prefix} nvfp4_max_abs={nvfp4_max_abs:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
