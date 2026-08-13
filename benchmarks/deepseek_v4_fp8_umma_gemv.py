#!/usr/bin/env python3
"""Correctness and latency benchmark for native SM100 MXF8 projections."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae import runtime
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
)


TILE_K = 128


def profile_span_us(launcher: Launcher, begin: int, end: int) -> float:
    profile = launcher.profile[:, : end + 1].cpu().numpy()
    return (profile[:, end].max() - profile[:, begin].min()) / 1.0e3


def report_track_profile(launcher: Launcher) -> None:
    # NumPy preserves the underlying uint64 values.  Converting an
    # uninitialized non-tracked torch scalar through int64 can overflow before
    # the magic-value guard has a chance to return.
    profile = launcher.profile.cpu().numpy()
    track_magic = 0x4454524B50524631
    if not all(int(value) == track_magic for value in profile[:, 127]):
        return
    counter_base = runtime.config.track_profile_event_base
    internal_span = max(int(value) for value in profile[:, 1]) - min(
        int(value) for value in profile[:, 0]
    )
    grid_envelope = internal_span * profile.shape[0]

    def counter_sum(offset: int) -> int:
        return sum(int(value) for value in profile[:, counter_base + offset])

    def grid_percent(offset: int) -> float:
        if grid_envelope <= 0:
            return 0.0
        return 100.0 * counter_sum(offset) / grid_envelope

    print(
        "DSV4_FP8_UMMA_COUNTERS "
        f"internal_span_us={internal_span / 1.0e3:.6f} "
        f"compute_m2c_wait_grid_pct={grid_percent(0):.3f} "
        f"allocator_slot_stall_grid_pct={grid_percent(3):.3f} "
        f"ldu0_queue_wait_grid_pct={grid_percent(9):.3f} "
        f"ldu0_dependency_wait_grid_pct={grid_percent(11):.3f} "
        f"ldu1_queue_wait_grid_pct={grid_percent(14):.3f} "
        f"ldu1_dependency_wait_grid_pct={grid_percent(16):.3f} "
        f"store_queue_wait_grid_pct={grid_percent(19):.3f} "
        f"allocator_instructions={counter_sum(8)} "
        f"allocator_slot_stall_events={counter_sum(4)} "
        f"allocator_slot_retries={counter_sum(5)} "
        f"ldu0_commands={counter_sum(13)} "
        f"ldu1_commands={counter_sum(18)} "
        f"store_commands={counter_sum(23)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--prefix")
    parser.add_argument("--sms", type=int, default=0)
    parser.add_argument("--split-k", type=int, default=1)
    parser.add_argument(
        "--scale-pack", type=int, choices=(1, 2, 4), default=2
    )
    parser.add_argument(
        "--output-group-size", type=int, choices=(1, 2), default=2
    )
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

    m_tiles = args.m // SchedFp8GemvUmmaStream.TILE_M
    k_tiles = args.k // TILE_K
    if args.split_k <= 0 or k_tiles % args.split_k:
        parser.error("split-k must be positive and divide K/128")
    if k_tiles % args.scale_pack:
        parser.error("scale-pack must divide K/128")
    if (k_tiles // args.split_k) % args.scale_pack:
        parser.error("scale-pack must divide every split-K shard")
    if args.output_group_size > 1 and args.scale_pack == 1:
        parser.error("grouped output tasks require packed scales")
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
        (m_tiles, k_tiles, SchedFp8GemvUmmaStream.WEIGHT_TILE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    # Immutable checkpoint weights are converted once during Python setup,
    # matching the resident checkpoint loader.  Keep setup-only layout work
    # out of the VDCores operator image under test.
    runtime.prepack_fp8_checkpoint(
        weight, weight_scale, packed_weight, args.scale_pack
    )

    packed_activation = torch.empty(
        (k_tiles, SchedFp8GemvUmmaStream.ACTIVATION_TILE_BYTES),
        dtype=torch.uint8,
        device=device,
    )
    scale_groups = k_tiles // args.scale_pack
    quant_launcher = Launcher(scale_groups, device=device)
    quant_launcher.s(
        ProfileEvent(2),
        SchedDsv4Fp8QuantUmmaB(
            input_source, packed_activation, args.scale_pack
        ).place(scale_groups),
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
        expected_weight_scale = torch.zeros_like(actual_weight_scale)
        scale_bytes = weight_scale.view(torch.uint8)
        for group_start in range(0, k_tiles, args.scale_pack):
            for sf in range(args.scale_pack):
                expected_weight_scale[
                    :, group_start, sf * 128 : (sf + 1) * 128
                ] = scale_bytes[:, group_start + sf].unsqueeze(1)
        scale_mismatch = (
            actual_weight_scale.sort(dim=2).values
            != expected_weight_scale.sort(dim=2).values
        ).sum(dim=2)
        print(
            "DSV4_FP8_UMMA_WEIGHT_LAYOUT "
            f"mismatch_per_tile={weight_mismatch.cpu().tolist()} "
            f"scale_mismatch_per_tile={scale_mismatch.cpu().tolist()}",
            flush=True,
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
            args.scale_pack,
            args.output_group_size,
        )
    else:
        gemv_schedule = SchedFp8GemvUmmaStream(
            packed_weight,
            packed_activation,
            output,
            args.scale_pack,
            args.output_group_size,
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
        f"scale_pack={args.scale_pack} "
        f"output_group_size={args.output_group_size} "
        f"reduction_dtype={args.reduction_dtype} "
        f"reset_in_span={str(args.split_k == 1).lower()} "
        f"quant_median_us={statistics.median(quant_timings):.6f} "
        f"task_min_us={min(task_timings):.6f} "
        f"task_median_us={statistics.median(task_timings):.6f} "
        f"task_max_us={max(task_timings):.6f} "
        f"kernel_median_us={statistics.median(kernel_timings):.6f} "
        f"max_abs={max_abs:.6f} prepack=python_setup",
        flush=True,
    )
    report_track_profile(gemv_launcher)


if __name__ == "__main__":
    main()
