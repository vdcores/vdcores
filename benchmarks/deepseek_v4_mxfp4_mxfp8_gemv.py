#!/usr/bin/env python3
"""Native MXFP4-weight/MXFP8-activation fixed-shape UMMA benchmark.

All token-time operands already use their final HBM layouts.  Constant +1
payloads make the native data swizzles invariant, while every byte in both
native UE8M0 scale layouts is the encoding of 1.0.  No conversion or repacking
is part of the measured launcher.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from dae import runtime
from dae.instructions import ProfileEvent, TmaTensor
from dae.launcher import Launcher
from dae.schedule import SchedMxfp4Mxfp8GemvUmmaK512


def report_track_profile(launcher: Launcher, mode: str) -> None:
    profile = launcher.profile.cpu().numpy()
    if not all(int(value) == 0x4454524B50524631 for value in profile[:, 127]):
        return
    base = runtime.config.track_profile_event_base
    span = max(int(value) for value in profile[:, 1]) - min(
        int(value) for value in profile[:, 0]
    )
    envelope = span * profile.shape[0]

    def total(offset: int) -> int:
        return sum(int(value) for value in profile[:, base + offset])

    def percent(offset: int) -> float:
        return 100.0 * total(offset) / envelope if envelope > 0 else 0.0

    print(
        "DSV4_MXFP4_MXFP8_COUNTERS "
        f"scale_mode={mode} internal_span_us={span / 1.0e3:.6f} "
        f"compute_m2c_wait_pct={percent(0):.3f} "
        f"allocator_slot_stall_pct={percent(3):.3f} "
        f"ldu0_queue_wait_pct={percent(9):.3f} "
        f"ldu0_dependency_wait_pct={percent(11):.3f} "
        f"ldu1_queue_wait_pct={percent(14):.3f} "
        f"ldu1_dependency_wait_pct={percent(16):.3f} "
        f"allocator_instructions={total(8)} "
        f"allocator_slot_stall_events={total(4)} "
        f"ldu0_commands={total(13)} ldu1_commands={total(18)}",
        flush=True,
    )


def build_metadata(
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
) -> torch.Tensor:
    records = torch.zeros(
        (weight_scale.shape[0], 16), dtype=torch.int64, device="cpu"
    )
    for m_tile in range(weight_scale.shape[0]):
        records[m_tile, 2] = weight_scale[m_tile, 0].data_ptr()
        records[m_tile, 3] = activation_scale[0].data_ptr()
    return records.view(torch.uint8).to(weight_scale.device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument(
        "--scale-mode",
        choices=("tma", "metadata", "both"),
        default="metadata",
    )
    parser.add_argument(
        "--activation-tiles",
        default="2",
        help="comma-separated K512 activation tiles per allocator load",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--weight-byte", type=lambda value: int(value, 0), default=0x66)
    parser.add_argument("--weight-scale", type=int, default=125)
    parser.add_argument("--activation-byte", type=lambda value: int(value, 0), default=0x78)
    parser.add_argument("--activation-scale", type=int, default=119)
    parser.add_argument("--expected", type=float, default=4096.0)
    args = parser.parse_args()
    if args.m <= 0 or args.m % 128:
        parser.error("--m must be a positive multiple of 128")
    activation_tile_values = [
        int(value) for value in args.activation_tiles.split(",")
    ]
    if any(value not in (1, 2, 4, 8) for value in activation_tile_values):
        parser.error("activation tiles must be selected from 1,2,4,8")

    device = torch.device("cuda")
    m_tiles = args.m // 128
    k512_tiles = SchedMxfp4Mxfp8GemvUmmaK512.K512_TILES
    weight_data = torch.full(
        (
            m_tiles,
            k512_tiles,
            SchedMxfp4Mxfp8GemvUmmaK512.WEIGHT_K128_TILES,
            SchedMxfp4Mxfp8GemvUmmaK512.TILE_M,
            SchedMxfp4Mxfp8GemvUmmaK512.WEIGHT_PACKED_K128_BYTES,
        ),
        args.weight_byte,
        dtype=torch.uint8,
        device=device,
    )
    weight_scale = torch.full(
        (
            m_tiles,
            k512_tiles,
            SchedMxfp4Mxfp8GemvUmmaK512.WEIGHT_SCALE_BYTES,
        ),
        args.weight_scale,
        dtype=torch.uint8,
        device=device,
    )
    activation_data = torch.full(
        (
            k512_tiles,
            SchedMxfp4Mxfp8GemvUmmaK512.ACTIVATION_DATA_BYTES,
        ),
        args.activation_byte,
        dtype=torch.uint8,
        device=device,
    )
    activation_scale = torch.full(
        (
            k512_tiles,
            SchedMxfp4Mxfp8GemvUmmaK512.ACTIVATION_SCALE_BYTES,
        ),
        args.activation_scale,
        dtype=torch.uint8,
        device=device,
    )
    metadata = build_metadata(weight_scale, activation_scale)
    expected = torch.full(
        (args.m,), args.expected, dtype=torch.float32, device=device
    )

    modes = ("tma", "metadata") if args.scale_mode == "both" else (
        args.scale_mode,
    )
    for activation_tiles in activation_tile_values:
        for mode in modes:
            output = torch.full(
                (args.m,), float("nan"), dtype=torch.float32, device=device
            )
            launcher = Launcher(m_tiles, device=device)
            weight_tma = TmaTensor(launcher, weight_data).mxfp4_k512_load()
            schedule = SchedMxfp4Mxfp8GemvUmmaK512(
                weight_data,
                weight_scale,
                activation_data,
                activation_scale,
                output,
                weight_tma,
                scale_mode=mode,
                metadata=metadata if mode == "metadata" else None,
                activation_tiles_per_load=activation_tiles,
            ).place(m_tiles)
            launcher.s(ProfileEvent(2), schedule, ProfileEvent(3))

            launcher.launch()
            torch.cuda.synchronize()
            try:
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
            except AssertionError:
                finite = output[torch.isfinite(output)]
                unique, counts = torch.unique(finite, return_counts=True)
                print(
                    "DSV4_MXFP4_MXFP8_GEMV_CORRECTNESS_FAILURE "
                    f"nan_count={torch.isnan(output).sum().item()} "
                    f"nan_indices={torch.isnan(output).nonzero().flatten().tolist()} "
                    f"sample_head={output[:8].tolist()} "
                    f"sample_tail={output[-8:].tolist()} "
                    f"finite_histogram={list(zip(unique.tolist(), counts.tolist()))}",
                    flush=True,
                )
                raise

            for _ in range(args.warmup):
                launcher.launch()
            torch.cuda.synchronize()
            task_timings = []
            kernel_timings = []
            for _ in range(args.iterations):
                launcher.launch()
                profile = launcher.profile[:, :4].cpu().numpy()
                task_timings.append(
                    (profile[:, 3].max() - profile[:, 2].min()) / 1.0e3
                )
                kernel_timings.append(
                    (profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
                )

            print(
                "DSV4_MXFP4_MXFP8_GEMV_RESULT "
                f"shape={args.m}x1x4096 scale_mode={mode} "
                f"activation_tiles_per_load={activation_tiles} "
                f"task_min_us={min(task_timings):.6f} "
                f"task_median_us={statistics.median(task_timings):.6f} "
                f"task_max_us={max(task_timings):.6f} "
                f"kernel_median_us={statistics.median(kernel_timings):.6f} "
                "output_exact=true",
                flush=True,
            )
            report_track_profile(launcher, mode)


if __name__ == "__main__":
    main()
