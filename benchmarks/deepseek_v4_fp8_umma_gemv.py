#!/usr/bin/env python3
"""Correctness and latency benchmark for native SM100 MXF8 projections."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from dae.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from dae.instructions import ProfileEvent, TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Fp8QuantUmmaB,
    SchedFp8GemvUmmaStream,
    SchedFp8GemvUmmaSplitK,
    SchedFp8UmmaPrepack,
)
from dae.tma_utils import Major


def profile_span_us(launcher: Launcher, begin: int, end: int) -> float:
    profile = launcher.profile[:, : end + 1].cpu().numpy()
    return (profile[:, end].max() - profile[:, begin].min()) / 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prefix")
    parser.add_argument("--sms", type=int, default=0)
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument(
        "--reduction-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--zero-input", action="store_true")
    args = parser.parse_args()
    if bool(args.checkpoint) != bool(args.prefix):
        parser.error("--checkpoint and --prefix must be supplied together")
    if args.m <= 0 or args.k <= 0 or args.m % 128 or args.k % 128:
        parser.error("M and K must be positive multiples of 128")
    if args.k > 8192:
        parser.error("the native MXF8 path supports K <= 8192")
    if min(args.warmup, args.iterations) <= 0:
        parser.error("timing counts must be positive")

    m_tiles = args.m // SchedFp8UmmaPrepack.TILE_M
    k_tiles = args.k // SchedFp8UmmaPrepack.TILE_K
    if args.split_k <= 0 or k_tiles % args.split_k:
        parser.error("split-k must be positive and divide K/128")
    if args.split_k == 1 and args.reduction_dtype != "fp32":
        parser.error("reduction dtype applies only to split-K")
    default_sms = min(m_tiles * args.split_k, 152)
    num_sms = args.sms or default_sms
    max_sms = min(152, m_tiles * args.split_k)
    if not 0 < num_sms <= max_sms:
        parser.error(f"sms must be in [1,{max_sms}]")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260811)
    if args.checkpoint:
        linear = DeepSeekV4Checkpoint(args.checkpoint).load_fp8_linear(
            args.prefix, device=str(device)
        )
        if tuple(linear.weight.shape) != (args.m, args.k):
            parser.error(
                f"checkpoint linear has shape {tuple(linear.weight.shape)}, "
                f"expected {(args.m, args.k)}"
            )
        weight = linear.weight
        weight_scale = linear.scale
    else:
        weight_source = torch.randn(
            (args.m, args.k),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ) * 0.05
        weight, weight_scale = quantize_fp8_block128(weight_source)
    input_source = torch.randn(
        (args.k,),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.1
    if args.zero_input:
        input_source.zero_()
    activation, activation_scale = quantize_fp8_block128(input_source)

    packed_weight = torch.empty(
        (m_tiles, k_tiles, SchedFp8UmmaPrepack.WEIGHT_TILE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    packed_activation_oracle = torch.empty(
        (k_tiles, SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES),
        dtype=torch.uint8,
        device=device,
    )

    prepack_sms = min(m_tiles, 152)
    weight_prepack_launcher = Launcher(prepack_sms, device=device)
    weight_tma = TmaTensor(
        weight_prepack_launcher, weight.view(torch.uint8)
    ).wgmma_load(128, 128, Major.K)
    weight_prepack_launcher.s(
        SchedFp8UmmaPrepack(
            SchedFp8UmmaPrepack.WEIGHT,
            weight,
            weight_scale,
            packed_weight,
            weight_tma,
        ).place(prepack_sms)
    )
    weight_prepack_launcher.launch()

    activation_rows = activation.reshape(1, -1).expand(8, -1).contiguous()
    activation_prepack_launcher = Launcher(1, device=device)
    activation_tma = TmaTensor(
        activation_prepack_launcher, activation_rows.view(torch.uint8)
    ).wgmma_load(8, 128, Major.K)
    activation_prepack_launcher.s(
        SchedFp8UmmaPrepack(
            SchedFp8UmmaPrepack.ACTIVATION,
            activation_rows,
            activation_scale,
            packed_activation_oracle,
            activation_tma,
        ).place(1)
    )
    activation_prepack_launcher.launch()

    packed_activation = torch.empty_like(packed_activation_oracle)
    quant_launcher = Launcher(k_tiles, device=device)
    quant_launcher.s(
        ProfileEvent(2),
        SchedDsv4Fp8QuantUmmaB(
            input_source, packed_activation
        ).place(k_tiles),
        ProfileEvent(3),
    )
    quant_launcher.launch()
    torch.cuda.synchronize(device)
    if args.diagnostic:
        logical_weight_chunks = (
            weight.view(torch.uint8)
            .reshape(m_tiles, 128, k_tiles, 128)
            .permute(0, 2, 1, 3)
            .reshape(m_tiles, k_tiles, 128, 8, 16)
        )
        expected_weight_data = torch.empty_like(logical_weight_chunks)
        for row in range(128):
            for source_chunk in range(8):
                destination_chunk = source_chunk ^ (row & 7)
                expected_weight_data[:, :, row, destination_chunk].copy_(
                    logical_weight_chunks[:, :, row, source_chunk]
                )
        actual_weight_data = packed_weight[
            :, :, : 128 * 128
        ].reshape_as(expected_weight_data)
        weight_mismatch = (
            actual_weight_data != expected_weight_data
        ).reshape(m_tiles, k_tiles, -1).sum(dim=2)
        actual_weight_scale = packed_weight[:, :, 128 * 128 :]
        expected_weight_scale = weight_scale.view(torch.uint8).unsqueeze(2)
        scale_mismatch = (
            actual_weight_scale != expected_weight_scale
        ).sum(dim=2)
        print(
            "DSV4_FP8_UMMA_WEIGHT_LAYOUT "
            f"mismatch_per_tile={weight_mismatch.cpu().tolist()} "
            f"scale_mismatch_per_tile={scale_mismatch.cpu().tolist()}",
            flush=True,
        )
    torch.testing.assert_close(
        packed_activation,
        packed_activation_oracle,
        rtol=0,
        atol=0,
    )

    output = torch.empty((args.m,), dtype=torch.bfloat16, device=device)
    gemv_launcher = Launcher(num_sms, device=device)
    accumulator = None
    if args.split_k > 1:
        reduction_dtype = (
            torch.float32
            if args.reduction_dtype == "fp32"
            else torch.bfloat16
        )
        accumulator = torch.zeros(
            (SchedFp8GemvUmmaSplitK.OUTPUT_ROWS, args.m),
            dtype=reduction_dtype,
            device=device,
        )
        output_reduce = TmaTensor(
            gemv_launcher, accumulator
        ).rowmajor_2d("reduce", SchedFp8GemvUmmaSplitK.OUTPUT_ROWS, 128)
        gemv_schedule = SchedFp8GemvUmmaSplitK(
            packed_weight,
            packed_activation,
            output_reduce,
            args.split_k,
        )
    else:
        gemv_schedule = SchedFp8GemvUmmaStream(
            packed_weight, packed_activation, output
        )
    gemv_launcher.s(
        ProfileEvent(2),
        gemv_schedule.place(num_sms),
        ProfileEvent(3),
    )
    gemv_launcher.launch()
    torch.cuda.synchronize(device)

    weight_dequant = dequantize_fp8_block128(weight, weight_scale)
    activation_dequant = dequantize_fp8_block128(
        activation, activation_scale
    )
    reference_float = weight_dequant @ activation_dequant
    reference = reference_float.to(torch.bfloat16)
    result = accumulator[0] if accumulator is not None else output
    if args.diagnostic:
        contributions = torch.stack(
            [
                weight_dequant[
                    :, tile * 128 : (tile + 1) * 128
                ]
                @ activation_dequant[tile * 128 : (tile + 1) * 128]
                for tile in range(k_tiles)
            ]
        )
        single_tile_errors = (
            contributions - result.float().unsqueeze(0)
        ).abs().amax(dim=1)
        cross_errors = []
        for weight_tile in range(k_tiles):
            cross_errors.append([])
            for activation_tile in range(k_tiles):
                cross = weight_dequant[
                    :, weight_tile * 128 : (weight_tile + 1) * 128
                ] @ activation_dequant[
                    activation_tile * 128 : (activation_tile + 1) * 128
                ]
                cross_errors[-1].append(
                    (cross - result.float()).abs().max().item()
                )
        print(
            "DSV4_FP8_UMMA_DIAGNOSTIC "
            f"output_norm={result.float().norm().item():.6f} "
            f"reference_norm={reference.float().norm().item():.6f} "
            f"cosine={torch.nn.functional.cosine_similarity(result.float(), reference_float, dim=0).item():.8f} "
            f"single_tile_max_abs={single_tile_errors.cpu().tolist()} "
            f"cross_tile_max_abs={cross_errors} "
            f"output_head={result[:8].float().cpu().tolist()} "
            f"reference_head={reference[:8].float().cpu().tolist()}",
            flush=True,
        )
    expected = (
        reference_float
        if accumulator is not None and accumulator.dtype == torch.float32
        else reference
    )
    torch.testing.assert_close(result, expected, rtol=3.0e-2, atol=1.0e-1)
    max_abs = (result.float() - expected.float()).abs().max().item()

    for _ in range(args.warmup):
        quant_launcher.launch()
        if accumulator is not None:
            accumulator.zero_()
        gemv_launcher.launch()
    torch.cuda.synchronize(device)
    quant_timings = []
    task_timings = []
    kernel_timings = []
    for _ in range(args.iterations):
        quant_launcher.launch()
        quant_timings.append(profile_span_us(quant_launcher, 2, 3))
        if accumulator is not None:
            accumulator.zero_()
        gemv_launcher.launch()
        task_timings.append(profile_span_us(gemv_launcher, 2, 3))
        kernel_timings.append(profile_span_us(gemv_launcher, 0, 1))

    print(
        "DSV4_FP8_UMMA_RESULT "
        f"shape={args.m}x1x{args.k} sms={num_sms} split_k={args.split_k} "
        f"reduction_dtype={args.reduction_dtype} "
        f"reset_in_span={str(args.split_k == 1).lower()} "
        f"quant_median_us={statistics.median(quant_timings):.6f} "
        f"task_min_us={min(task_timings):.6f} "
        f"task_median_us={statistics.median(task_timings):.6f} "
        f"task_max_us={max(task_timings):.6f} "
        f"kernel_median_us={statistics.median(kernel_timings):.6f} "
        f"max_abs={max_abs:.6f} layout_exact=true",
        flush=True,
    )


if __name__ == "__main__":
    main()
