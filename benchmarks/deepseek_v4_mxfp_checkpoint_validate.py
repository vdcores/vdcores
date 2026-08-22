#!/usr/bin/env python3
"""Validate one real checkpoint matrix after offline MXFP4 packing."""

from __future__ import annotations

import argparse

import torch

from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from dae.deepseek_v4_mxfp_checkpoint import (
    pack_mxfp4_data,
    pack_mxfp4_scales,
)
from dae.deepseek_v4_quant import dequantize_fp8_block128, dequantize_nvfp4
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Mxfp8QuantFfnInput,
    SchedMxfp4Mxfp8GemvUmmaK512,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=-1)
    parser.add_argument("--tag", choices=("w1", "w3"), default="w1")
    args = parser.parse_args()
    config = DeepSeekV4FlashConfig()
    if not 0 <= args.layer < config.num_layers:
        parser.error("layer is outside the model")
    if not -1 <= args.expert < config.num_experts:
        parser.error("expert must be -1 (shared) or a routed expert id")

    import flashinfer
    from flashinfer import SfLayout

    device = torch.device("cuda")
    checkpoint = DeepSeekV4Checkpoint(args.checkpoint, config)
    if args.expert < 0:
        prefix = f"layers.{args.layer}.ffn.shared_experts.{args.tag}"
        linear = checkpoint.load_fp8_linear(prefix, device=str(device))
        source_weight = dequantize_fp8_block128(
            linear.weight, linear.scale
        )
    else:
        prefix = f"layers.{args.layer}.ffn.experts.{args.expert}.{args.tag}"
        linear = checkpoint.load_nvfp4_linear(prefix, device=str(device))
        source_weight = dequantize_nvfp4(
            linear.weight, linear.weight_scale, linear.weight_scale_2
        )
    source_weight = source_weight.to(torch.bfloat16)
    packed, scales = flashinfer.mxfp4_quantize(
        source_weight, sfLayout=SfLayout.layout_128x4
    )
    weight_data = pack_mxfp4_data(packed, tile_k=512)
    weight_scale = pack_mxfp4_scales(
        scales,
        rows=source_weight.shape[0],
        k=source_weight.shape[1],
        tile_k=512,
    )

    generator = torch.Generator(device=device).manual_seed(20260822)
    hidden = torch.randn(
        (config.hidden_size,),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).mul_(0.1)
    activation_records = torch.empty(
        (8, SchedDsv4Mxfp8QuantFfnInput.RECORD_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    quant_launcher = Launcher(8, device=device)
    quant_launcher.s(
        SchedDsv4Mxfp8QuantFfnInput(hidden, activation_records).place(8)
    )
    quant_launcher.launch()
    torch.cuda.synchronize()
    # The resident FFN consumes the interleaved records directly.  This
    # standalone GEMV validator has separate contiguous data/SFB operands, so
    # materialize those two views after the setup-only quantization launch.
    activation_data = activation_records[:, :4096].contiguous()
    activation_scale = activation_records[:, 4096:].contiguous()
    output = torch.empty(
        (source_weight.shape[0],), dtype=torch.float32, device=device
    )
    workers = source_weight.shape[0] // 128
    launcher = Launcher(workers, device=device)
    weight_tma = TmaTensor(launcher, weight_data).mxfp4_load(512)
    schedule = SchedMxfp4Mxfp8GemvUmmaK512(
        weight_data,
        weight_scale,
        activation_data,
        activation_scale,
        output,
        weight_tma,
        scale_mode="tma",
    )
    launcher.s(schedule.place(workers))
    launcher.launch()
    torch.cuda.synchronize()

    reference = source_weight.float() @ hidden.float()
    error = (output - reference).abs()
    mean_relative = (
        error.mean() / reference.abs().mean().clamp_min(1.0e-8)
    ).item()
    cosine = torch.nn.functional.cosine_similarity(
        output, reference, dim=0
    ).item()
    if not torch.isfinite(output).all() or mean_relative > 0.20 or cosine < 0.98:
        raise AssertionError(
            "offline MXFP task image failed quantized validation: "
            f"mean_relative={mean_relative:.6f} cosine={cosine:.6f} "
            f"max_abs={error.max().item():.6f}"
        )
    print(
        "DSV4_MXFP_CHECKPOINT_VALIDATE status=PASS "
        f"prefix={prefix} mean_relative={mean_relative:.6f} "
        f"cosine={cosine:.6f} max_abs={error.max().item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
