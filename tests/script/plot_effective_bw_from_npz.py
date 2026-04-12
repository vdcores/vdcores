#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def resample_step_series(xs: np.ndarray, ys: np.ndarray, step_ns: float) -> tuple[np.ndarray, np.ndarray]:
    if xs.size == 0:
        return xs, ys
    if step_ns <= 0:
        raise ValueError("step_ns must be positive")

    start = float(xs[0])
    end = float(xs[-1])
    if end <= start:
        return xs, ys

    grid = np.arange(start, end + step_ns, step_ns, dtype=np.float64)
    idx = np.searchsorted(xs, grid, side="right") - 1
    idx = np.clip(idx, 0, len(ys) - 1)
    return grid, ys[idx]


def moving_average(ys: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return ys
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    pad_left = window_size // 2
    pad_right = window_size - 1 - pad_left
    padded = np.pad(ys, (pad_left, pad_right), mode="constant", constant_values=0.0)
    return np.convolve(padded, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="Plot aggregated effective bandwidth from saved NPZ data")
    parser.add_argument("input", type=Path, help="NPZ file with timestamp_ns and effective_bw_gbs")
    parser.add_argument("--output", type=Path, default=Path("effective_bw_from_npz.png"), help="output PNG path")
    parser.add_argument("--title", type=str, default="Effective Bandwidth Over Time", help="plot title")
    parser.add_argument(
        "--resample-step-ns",
        type=float,
        default=256.0,
        help="uniform resample step in ns for optional smoothing; default: 256",
    )
    parser.add_argument(
        "--smooth-window-ns",
        type=float,
        default=0.0,
        help="moving-average smoothing window in ns; 0 disables smoothing",
    )
    args = parser.parse_args()

    payload = np.load(args.input)
    xs = np.asarray(payload["timestamp_ns"], dtype=np.float64)
    ys = np.asarray(payload["effective_bw_gbs"], dtype=np.float64)
    if xs.size == 0 or ys.size == 0 or xs.size != ys.size:
        raise ValueError("NPZ must contain non-empty timestamp_ns and effective_bw_gbs arrays of equal length")

    plot_xs = xs
    plot_ys = ys
    resample_step_ns = args.resample_step_ns
    smooth_window_ns = args.smooth_window_ns
    if smooth_window_ns > 0:
        plot_xs, plot_ys = resample_step_series(xs, ys, resample_step_ns)
        window_size = max(1, int(round(smooth_window_ns / resample_step_ns)))
        plot_ys = moving_average(plot_ys, window_size)
        if plot_ys.size:
            plot_ys[-1] = 0.0

    fig, ax = plt.subplots(figsize=(14, 7))
    if smooth_window_ns > 0:
        ax.plot(plot_xs / 1e3, plot_ys, color="black", linewidth=1.5)
    else:
        ax.step(plot_xs / 1e3, plot_ys, where="post", color="black", linewidth=1.5)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Estimated Effective Bandwidth (GB/s)")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Wrote plot to {args.output}")


if __name__ == "__main__":
    main()
