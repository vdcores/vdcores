import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from dae.util import (
    average_effective_bw_heatmap,
    compute_effective_bw_heatmap,
    load_mem_trace_npz,
    load_mem_trace_runs_npz,
)


def load_heatmap(trace_path: str, bin_us: float, is_bench: bool):
    if is_bench:
        trace_runs = load_mem_trace_runs_npz(trace_path)
        num_sms = 0
        for trace in trace_runs:
            if trace.size > 0:
                num_sms = max(num_sms, int(trace["sm_id"].max()) + 1)
        return average_effective_bw_heatmap(trace_runs, num_sms=num_sms, bin_us=bin_us)

    trace_records = load_mem_trace_npz(trace_path)
    num_sms = int(trace_records["sm_id"].max()) + 1 if trace_records.size > 0 else 0
    return compute_effective_bw_heatmap(trace_records, num_sms=num_sms, bin_us=bin_us)


def render_panel(ax, times_us, heatmap, title: str, bin_us: float, vmax: float):
    time_extent = times_us[-1] + 0.5 * bin_us if times_us.size > 0 else bin_us
    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0.0, time_extent, -0.5, heatmap.shape[0] - 0.5],
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("SM ID")
    ax.set_title(title)
    return im


def main():
    parser = argparse.ArgumentParser(description="Compare two saved VDCores memory-trace heatmaps side by side")
    parser.add_argument("--left-trace", type=str, required=True, help="Left trace .npz")
    parser.add_argument("--left-title", type=str, required=True, help="Left panel title")
    parser.add_argument("--left-bin-us", type=float, default=1.0, help="Left trace bin width in microseconds")
    parser.add_argument("--left-bench", action="store_true", help="Interpret the left trace as a benchmark trace archive")
    parser.add_argument("--right-trace", type=str, required=True, help="Right trace .npz")
    parser.add_argument("--right-title", type=str, required=True, help="Right panel title")
    parser.add_argument("--right-bin-us", type=float, default=1.0, help="Right trace bin width in microseconds")
    parser.add_argument("--right-bench", action="store_true", help="Interpret the right trace as a benchmark trace archive")
    parser.add_argument("--output", type=str, default=None, help="Output path for the side-by-side heatmap figure")
    args = parser.parse_args()

    left_times_us, left_heatmap = load_heatmap(args.left_trace, args.left_bin_us, args.left_bench)
    right_times_us, right_heatmap = load_heatmap(args.right_trace, args.right_bin_us, args.right_bench)

    if left_heatmap.size == 0 or right_heatmap.size == 0:
        print("[mem-trace] one or both traces did not produce valid heatmap data")
        return

    shared_vmax = float(max(np.max(left_heatmap), np.max(right_heatmap)))
    if shared_vmax <= 0.0:
        shared_vmax = 1.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=200, constrained_layout=True)
    left_im = render_panel(axes[0], left_times_us, left_heatmap, args.left_title, args.left_bin_us, shared_vmax)
    render_panel(axes[1], right_times_us, right_heatmap, args.right_title, args.right_bin_us, shared_vmax)
    fig.colorbar(left_im, ax=axes, label="Effective BW (GB/s)")

    if args.output is None:
        args.output = f"effective_bw_heatmap_compare_{int(time.time())}.png"
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote side-by-side heatmap comparison to {args.output}")


if __name__ == "__main__":
    main()
