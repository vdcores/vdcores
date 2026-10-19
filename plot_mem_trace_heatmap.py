#!/usr/bin/env python3
"""Compact bandwidth timeline; top: Mirage, bottom: VDE."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from plot_mem_trace_compare import (
    compute_effective_bw_series,
    load_baseline_mem_trace_npz,
    load_vdc_bw_json,
    moving_average,
    resample_step_series,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_trace", type=Path)
    parser.add_argument("vdc_bw", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/pdf/effective_bw_heatmap.pdf"))
    parser.add_argument("--bin-us", type=float, default=1.0)
    parser.add_argument("--vdc-smooth-window-ns", type=float, default=5000.0,
                        help="Match comparison plot smoothing; 0 disables it")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Shared upper color limit in GB/s; defaults to observed peak")
    parser.add_argument("--cmap", default="Blues")
    parser.add_argument("--gap-in", type=float, default=0.5,
                        help="Blank space between heatmap strips in inches")
    args = parser.parse_args()
    if args.bin_us <= 0 or args.vdc_smooth_window_ns < 0:
        parser.error("bin width must be positive and smoothing window nonnegative")
    if not np.isfinite(args.gap_in) or args.gap_in < 0:
        parser.error("gap must be finite and nonnegative")

    trace = load_baseline_mem_trace_npz(str(args.baseline_trace))
    _, baseline_bw = compute_effective_bw_series(trace, bin_us=args.bin_us)
    if baseline_bw.size == 0:
        parser.error("baseline trace contains no transfers")
    baseline_end = float(trace["end"].max() - trace["start"].min()) / 1000.0
    baseline_edges = np.arange(baseline_bw.size + 1, dtype=float) * args.bin_us
    baseline_edges[-1] = baseline_end

    vdc_us, vdc_bw = load_vdc_bw_json(str(args.vdc_bw))
    if not np.all(np.isfinite(vdc_us)) or not np.all(np.diff(vdc_us) > 0):
        parser.error("VDE timestamps must be finite and strictly increasing")
    if not np.all(np.isfinite(vdc_bw)):
        parser.error("VDE bandwidth values must be finite")
    vdc_end = float(vdc_us[-1])
    vdc_edges = vdc_us
    if args.vdc_smooth_window_ns > 0:
        step_ns = 256.0
        smooth_us, smooth_bw = resample_step_series(vdc_us * 1000, vdc_bw, step_ns)
        window = max(1, round(args.vdc_smooth_window_ns / step_ns))
        smooth_bw = moving_average(smooth_bw, window)
        smooth_bw[-1] = 0.0
        smooth_us /= 1000
        # Use the original trace endpoint, without extending its duration.
        vdc_edges = np.append(smooth_us[smooth_us < vdc_end], vdc_end)
        centers = (vdc_edges[:-1] + vdc_edges[1:]) / 2
        vdc_bw = np.interp(centers, smooth_us, smooth_bw)
    else:
        # The last sample marks completion; each preceding value spans an interval.
        vdc_bw = vdc_bw[:-1]

    vmax = args.vmax if args.vmax is not None else max(baseline_bw.max(), vdc_bw.max())
    if not np.isfinite(vmax) or vmax <= 0:
        parser.error("color limit must be finite and positive")
    norm = Normalize(vmin=0, vmax=vmax, clip=True)
    # Keep strip thickness fixed when making more room for manual annotations.
    strip_in, bottom_in, top_in = 0.22, 0.42, 0.1
    height_in = bottom_in + 2 * strip_in + args.gap_in + top_in
    fig = plt.figure(figsize=(6, height_in))
    lower = fig.add_axes([0.025, bottom_in / height_in, 0.79, strip_in / height_in])
    upper = fig.add_axes([0.025, (bottom_in + strip_in + args.gap_in) / height_in,
                          0.79, strip_in / height_in], sharex=lower)
    axes = (upper, lower)
    for ax, edges, bw in zip(axes, (baseline_edges, vdc_edges), (baseline_bw, vdc_bw)):
        mesh = ax.pcolormesh(edges, [0, 1], np.maximum(bw, 0)[None, :],
                      cmap=args.cmap, norm=norm, shading="flat", edgecolors="none",
                      antialiased=False, rasterized=True)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="x", labelsize=8, length=2, pad=2)
    axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
    axes[1].set_xlim(0, max(baseline_end, vdc_end))
    axes[1].set_xlabel("Time (µs)", fontsize=9, labelpad=2)
    color_ax = fig.add_axes([0.845, bottom_in / height_in, 0.018,
                             (2 * strip_in + args.gap_in) / height_in])
    colorbar = fig.colorbar(mesh, cax=color_ax)
    colorbar.set_label("BW (GB/s)", fontsize=8, labelpad=5)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    colorbar.outline.set_visible(False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", pad_inches=0.025)
    fig.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")
    print(f"Top: Mirage; bottom: VDE. Shared color scale: 0–{vmax:.1f} GB/s.")
    print(f"Duration: Mirage {baseline_end:.3f} µs; VDE {vdc_end:.3f} µs.")


if __name__ == "__main__":
    main()
