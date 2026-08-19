#!/usr/bin/env python3
"""Correctness and latency benchmark for the common coupled MXFP8 task."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    quantize_fp8_block128,
)
from dae.instructions import ProfileEvent, TmaTensor
from dae.launcher import Launcher
from dae.schedule import (
    SchedDsv4Fp8QuantUmmaB,
    SchedFp8GemvUmmaCoupled,
)
from dae import runtime


def profile_span_us(launcher: Launcher, begin: int, end: int) -> float:
    profile = launcher.profile[:, : end + 1].cpu().numpy()
    return (profile[:, end].max() - profile[:, begin].min()) / 1.0e3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sms", type=int, default=0)
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument("--balanced-k", action="store_true")
    parser.add_argument(
        "--reduction-dtype", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--validate-router", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if args.m <= 0 or args.k <= 0 or args.m % 256 or args.k % 256:
        parser.error("coupled MXFP8 requires positive M256/K256 shapes")
    if args.batch <= 0:
        parser.error("projection batch must be positive")
    if args.split_k <= 0 or (args.k // 256) % args.split_k:
        parser.error("split-k must divide K/256")
    if args.balanced_k and args.split_k != 1:
        parser.error("balanced-k and uniform split-k are mutually exclusive")
    if (
        args.split_k == 1
        and not args.balanced_k
        and args.reduction_dtype != "fp32"
    ):
        parser.error("reduction dtype applies only to split-K")
    if min(args.warmup, args.iterations) <= 0:
        parser.error("timing counts must be positive")

    work_tiles = args.batch * (args.m // 256) * (
        args.k // 256 if args.balanced_k else args.split_k
    )
    num_sms = args.sms or min(work_tiles, 152)
    if not 0 < num_sms <= work_tiles:
        parser.error(f"sms must be in [1,{work_tiles}]")

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260811)
    weight_source = torch.randn(
        (args.batch * args.m, args.k),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.05
    input_source = torch.randn(
        (args.batch, args.k),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    ) * 0.1
    weight, weight_scale = quantize_fp8_block128(weight_source)
    activation_and_scale = [
        quantize_fp8_block128(input_source[batch])
        for batch in range(args.batch)
    ]
    activation = torch.stack(
        [item[0] for item in activation_and_scale], dim=0
    )
    activation_scale = torch.stack(
        [item[1] for item in activation_and_scale], dim=0
    )

    m_tiles = args.m // 128
    k_tiles = args.k // 128
    packed_weight_flat = torch.empty(
        (args.batch * m_tiles, k_tiles, 16896),
        dtype=torch.uint8,
        device=device,
    )
    runtime.prepack_fp8_checkpoint(
        weight, weight_scale, packed_weight_flat, 2
    )
    packed_weight = packed_weight_flat.view(
        args.batch, m_tiles, k_tiles, 16896
    )
    packed_activation = torch.empty(
        (args.batch, k_tiles, 2048), dtype=torch.uint8, device=device
    )
    quant_launcher = Launcher(k_tiles // 2, device=device)
    quant_items = [ProfileEvent(2)]
    for batch in range(args.batch):
        quant_items.append(
            SchedDsv4Fp8QuantUmmaB(
                input_source[batch], packed_activation[batch], 2
            ).place(k_tiles // 2)
        )
    quant_items.append(ProfileEvent(3))
    quant_launcher.s(*quant_items)

    launcher = Launcher(num_sms, device=device)
    accumulator = None
    if args.split_k == 1 and not args.balanced_k:
        output = torch.empty(
            (args.batch, args.m), dtype=torch.bfloat16, device=device
        )
        schedule_output = output
    else:
        reduction_dtype = (
            torch.float32
            if args.reduction_dtype == "fp32"
            else torch.bfloat16
        )
        accumulator = torch.zeros(
            (args.batch * args.m // 128, 128),
            dtype=reduction_dtype,
            device=device,
        )
        schedule_output = TmaTensor(
            launcher, accumulator
        ).rowmajor_2d("reduce", 1, 128)
    schedule = SchedFp8GemvUmmaCoupled(
        packed_weight,
        packed_activation,
        schedule_output,
        split_k=args.split_k,
        balanced_k=args.balanced_k,
    )
    placed_schedule = schedule.place(num_sms)
    launcher.s(ProfileEvent(2), placed_schedule, ProfileEvent(3))

    quant_launcher.launch()
    torch.cuda.synchronize(device)
    if args.diagnostic:
        weight_scale_stream = placed_schedule.weight_stream[
            ...,
            placed_schedule.WEIGHT_DATA_BYTES
            * placed_schedule.OUTPUT_TILES
            * placed_schedule.SCALE_PACK :,
        ]
        activation_scale_stream = packed_activation[
            :, :: placed_schedule.SCALE_PACK,
            placed_schedule.ACTIVATION_DATA_BYTES :,
        ]
        print(
            "DSV4_FP8_COUPLED_STREAM_DIAGNOSTIC "
            f"weight_scale_min={weight_scale_stream.min().item()} "
            f"weight_scale_max={weight_scale_stream.max().item()} "
            f"weight_scale_nonzero={torch.count_nonzero(weight_scale_stream).item()} "
            f"activation_scale_min={activation_scale_stream.min().item()} "
            f"activation_scale_max={activation_scale_stream.max().item()} "
            f"activation_scale_nonzero={torch.count_nonzero(activation_scale_stream).item()}",
            flush=True,
        )

    if accumulator is not None:
        accumulator.zero_()
    launcher.launch()
    torch.cuda.synchronize(device)

    weight_dequant = dequantize_fp8_block128(
        weight, weight_scale
    ).reshape(args.batch, args.m, args.k)
    activation_dequant = torch.stack(
        [
            dequantize_fp8_block128(
                activation[batch], activation_scale[batch]
            )
            for batch in range(args.batch)
        ],
        dim=0,
    )
    reference_float = torch.einsum(
        "bmk,bk->bm", weight_dequant, activation_dequant
    )
    result = (
        accumulator.view(args.batch, args.m)
        if accumulator is not None
        else output
    )
    expected = (
        reference_float
        if accumulator is not None and accumulator.dtype == torch.float32
        else reference_float.to(torch.bfloat16)
    )
    if args.diagnostic:
        error = (result.float() - expected.float()).abs()
        group_error = error.reshape(-1, 128)
        group_result = result.float().reshape(-1, 128)
        group_expected = expected.float().reshape(-1, 128)
        print(
            "DSV4_FP8_COUPLED_DIAGNOSTIC "
            f"output_norm={result.float().norm().item():.6f} "
            f"reference_norm={expected.float().norm().item():.6f} "
            f"cosine={torch.nn.functional.cosine_similarity(result.float().reshape(-1), expected.float().reshape(-1), dim=0).item():.8f} "
            f"group_max_abs={group_error.amax(dim=1).cpu().tolist()} "
            f"group_mean_abs={group_error.mean(dim=1).cpu().tolist()} "
            f"group_result_norm={group_result.norm(dim=1).cpu().tolist()} "
            f"group_reference_norm={group_expected.norm(dim=1).cpu().tolist()} "
            f"output_head={result.reshape(-1)[:16].float().cpu().tolist()} "
            f"reference_head={expected.reshape(-1)[:16].float().cpu().tolist()}",
            flush=True,
        )
    torch.testing.assert_close(result, expected, rtol=3.0e-2, atol=1.0e-1)
    if args.validate_router:
        if args.m != 256 or args.batch != 1:
            parser.error("router validation requires batch=1 and M=256")
        actual_ids = torch.topk(result[0].float(), 6).indices
        expected_ids = torch.topk(reference_float[0], 6).indices
        torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
        actual_weights = torch.softmax(
            result[0].float()[actual_ids], dim=0
        )
        expected_weights = torch.softmax(
            reference_float[0, expected_ids], dim=0
        )
        torch.testing.assert_close(
            actual_weights, expected_weights, rtol=3.0e-2, atol=2.0e-3
        )

    for _ in range(args.warmup):
        quant_launcher.launch()
        if accumulator is not None:
            accumulator.zero_()
        launcher.launch()
    torch.cuda.synchronize(device)

    quant_timings = []
    task_timings = []
    kernel_timings = []
    for _ in range(args.iterations):
        quant_launcher.launch()
        quant_timings.append(profile_span_us(quant_launcher, 2, 3))
        if accumulator is not None:
            accumulator.zero_()
        launcher.launch()
        task_timings.append(profile_span_us(launcher, 2, 3))
        kernel_timings.append(profile_span_us(launcher, 0, 1))

    max_abs = (result.float() - expected.float()).abs().max().item()
    print(
        "DSV4_FP8_COUPLED_RESULT "
        f"shape={args.m}x{args.batch}x{args.k} sms={num_sms} "
        f"split_k={args.split_k} balanced_k={str(args.balanced_k).lower()} "
        f"reduction_dtype={args.reduction_dtype} "
        f"quant_median_us={statistics.median(quant_timings):.6f} "
        f"task_min_us={min(task_timings):.6f} "
        f"task_median_us={statistics.median(task_timings):.6f} "
        f"task_max_us={max(task_timings):.6f} "
        f"kernel_median_us={statistics.median(kernel_timings):.6f} "
        f"max_abs={max_abs:.6f} prepack=python_setup",
        flush=True,
    )


if __name__ == "__main__":
    main()
