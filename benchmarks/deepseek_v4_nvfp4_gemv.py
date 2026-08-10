#!/usr/bin/env python3
"""Correctness and latency benchmark for the DeepSeek-V4 NVFP4 GEMV task."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4_quant import dequantize_nvfp4, quantize_nvfp4
from dae.instructions import ProfileEvent
from dae.launcher import Launcher
from dae.schedule import SchedNvfp4Gemv, SchedNvfp4GemvUmma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=0)
    parser.add_argument(
        "--implementation", choices=("cuda", "umma"), default="cuda"
    )
    parser.add_argument(
        "--sms-list",
        default="",
        help="comma-separated SM counts; reuses one quantized input for a sweep",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--trace-stages", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="print the first launch's vectors and error distribution",
    )
    parser.add_argument(
        "--unit-scales",
        action="store_true",
        help="use exact E2M1 sources with every NVFP4 block scale equal to one",
    )
    parser.add_argument(
        "--dump-columns",
        action="store_true",
        help="drain and verify all eight native UMMA output columns",
    )
    parser.add_argument(
        "--identity-weight",
        action="store_true",
        help="use an exact packed FP4 identity matrix (requires M=K)",
    )
    parser.add_argument(
        "--indexed-activation",
        action="store_true",
        help="use packed codes k mod 16 with unit block/global scales",
    )
    parser.add_argument(
        "--block-indexed-activation",
        action="store_true",
        help="use one distinct FP4 code per 16-value scale block",
    )
    parser.add_argument(
        "--indexed-scales",
        action="store_true",
        help="use constant +1 FP4 data and one distinct E4M3 scale per block",
    )
    args = parser.parse_args()

    def stage(name: str) -> None:
        if args.trace_stages:
            torch.cuda.synchronize()
            print(f"DSV4_NVFP4_STAGE {name}", flush=True)

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810)
    if args.unit_scales:
        codebook = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            dtype=torch.bfloat16,
            device=device,
        )
        weight_source = codebook[
            torch.randint(
                0, 16, (args.m, args.k), generator=generator, device=device
            )
        ]
        # Constant activation data and scales isolate A-layout/output mapping;
        # every logical B row staged for the N=8 UMMA tile is identical.
        input_source = torch.ones(
            (args.k,), dtype=torch.bfloat16, device=device
        )
        weight_source.reshape(args.m, -1, 16)[:, :, 0] = 6.0
        forced_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        weight_source = torch.randn(
            (args.m, args.k), generator=generator,
            dtype=torch.bfloat16, device=device
        ) * 0.05
        input_source = torch.randn(
            (args.k,), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.1
        forced_scale = None
    stage("sources_ready")
    weight, weight_sf, weight_scale2 = quantize_nvfp4(
        weight_source, forced_scale
    )
    if args.identity_weight:
        if args.m != args.k:
            raise ValueError("--identity-weight requires M=K")
        weight = torch.zeros(
            (args.m, args.k // 2), dtype=torch.uint8, device=device
        )
        diagonal = torch.arange(args.m, device=device)
        identity_codes = torch.where(
            (diagonal & 1) == 0,
            torch.full_like(diagonal, 2),
            torch.full_like(diagonal, 2 << 4),
        ).to(torch.uint8)
        weight[diagonal, diagonal // 2] = identity_codes
        weight_sf = torch.ones(
            (args.m, args.k // 16),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        weight_scale2 = torch.ones((), dtype=torch.float32, device=device)
    stage("weight_quantized")
    activation, activation_sf, input_scale = quantize_nvfp4(
        input_source, forced_scale
    )
    if args.indexed_activation or args.block_indexed_activation:
        activation_indices = torch.arange(
            args.k, device=device, dtype=torch.int64
        )
        if args.block_indexed_activation:
            activation_codes = ((activation_indices // 16) % 16).to(
                torch.uint8
            )
        else:
            activation_codes = (activation_indices % 16).to(torch.uint8)
        activation = (
            activation_codes[0::2] | (activation_codes[1::2] << 4)
        ).contiguous()
        activation_sf = torch.ones(
            (args.k // 16,), dtype=torch.float8_e4m3fn, device=device
        )
        input_scale = torch.ones((), dtype=torch.float32, device=device)
    if args.indexed_scales:
        scale_count = args.k // 16
        if scale_count > 32:
            raise ValueError("--indexed-scales supports at most 32 blocks")
        activation = torch.full(
            (args.k // 2,), 0x22, dtype=torch.uint8, device=device
        )
        activation_sf = torch.arange(
            48, 48 + scale_count, dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn)
        input_scale = torch.ones((), dtype=torch.float32, device=device)
    if args.unit_scales:
        assert bool((weight_sf.float() == 1.0).all().item())
        assert activation_sf.float().unique().numel() == 1
    stage("activation_quantized")
    alpha = (weight_scale2 * input_scale).reshape(1)
    reference = (
        dequantize_nvfp4(weight, weight_sf, weight_scale2)
        @ dequantize_nvfp4(activation, activation_sf, input_scale)
    ).to(torch.bfloat16)
    stage("reference_ready")

    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if args.sms_list:
        sms_values = [int(value) for value in args.sms_list.split(",")]
    else:
        default_sms = (
            min(args.m, device_sms)
            if args.implementation == "cuda"
            else (args.m + 127) // 128
        )
        sms_values = [args.sms or default_sms]
    if any(value <= 0 or value > device_sms for value in sms_values):
        raise ValueError(f"SM counts must be in [1, {device_sms}]")

    for num_sms in sms_values:
        output_columns = 8 if args.dump_columns else 1
        if args.dump_columns and args.implementation != "umma":
            raise ValueError("--dump-columns requires --implementation umma")
        output = torch.empty(
            (args.m * output_columns,), dtype=torch.bfloat16, device=device
        )
        launcher = Launcher(num_sms, device=device)
        schedule_cls = (
            SchedNvfp4Gemv
            if args.implementation == "cuda"
            else SchedNvfp4GemvUmma
        )
        launcher.s(
            ProfileEvent(2),
            schedule_cls(
                weight, weight_sf, activation, activation_sf, alpha, output,
                **({"output_columns": output_columns}
                   if args.implementation == "umma" else {})
            ).place(num_sms),
            ProfileEvent(3),
        )
        stage(f"launcher_ready_{num_sms}")
        launcher.launch()
        torch.cuda.synchronize()
        stage(f"first_launch_complete_{num_sms}")

        output_matrix = output.reshape(args.m, output_columns)
        comparison = output_matrix[:, 0]
        expected = reference.reshape(args.m, 1).expand_as(output_matrix)
        max_abs = (comparison.float() - reference.float()).abs().max().item()
        mean_rel = (
            (comparison.float() - reference.float()).abs().mean()
            / reference.float().abs().mean().clamp_min(1.0e-8)
        ).item()
        if args.diagnostic:
            difference = (comparison.float() - reference.float()).abs()
            print(f"output={comparison.float().cpu().tolist()}", flush=True)
            print(f"reference={reference.float().cpu().tolist()}", flush=True)
            print(
                "diagnostic "
                f"bad_indices={(difference > 5e-2).nonzero().flatten().cpu().tolist()} "
                f"cosine={torch.nn.functional.cosine_similarity(comparison.float(), reference.float(), dim=0).item():.8f}",
                flush=True,
            )
        if args.dump_columns:
            all_errors = (output_matrix.float() - expected.float()).abs()
            column_errors = all_errors.amax(dim=0)
            tolerance = 5e-2 + 2e-2 * expected.float().abs()
            good = all_errors <= tolerance
            good_masks = sum(
                good[:, column].to(torch.int32) << column
                for column in range(output_columns)
            )
            unique_masks, mask_counts = torch.unique(
                good_masks, return_counts=True
            )
            mask_histogram = {
                int(mask): int(count)
                for mask, count in zip(
                    unique_masks.cpu().tolist(), mask_counts.cpu().tolist()
                )
            }
            print(
                f"column_max_abs={column_errors.cpu().tolist()} "
                f"row_good_mask_histogram={mask_histogram}", flush=True
            )
        torch.testing.assert_close(
            output_matrix, expected, rtol=2e-2, atol=5e-2
        )

        for _ in range(args.warmup):
            launcher.launch()
        torch.cuda.synchronize()
        kernel_timings = []
        task_timings = []
        for _ in range(args.iterations):
            launcher.launch()
            profile = launcher.profile[:, :4].cpu().numpy()
            kernel_timings.append(
                (profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
            )
            task_timings.append(
                (profile[:, 3].max() - profile[:, 2].min()) / 1.0e3
            )

        print(
            "DSV4_NVFP4_GEMV_RESULT "
            f"implementation={args.implementation} "
            f"shape={args.m}x1x{args.k} sms={num_sms} "
            f"task_min_us={min(task_timings):.6f} "
            f"task_median_us={statistics.median(task_timings):.6f} "
            f"task_max_us={max(task_timings):.6f} "
            f"kernel_median_us={statistics.median(kernel_timings):.6f} "
            f"max_abs={max_abs:.6f} "
            f"mean_relative={mean_rel:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
