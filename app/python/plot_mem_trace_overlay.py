import argparse
import time

import matplotlib.pyplot as plt

from dae.util import (
    average_effective_bw_series,
    compute_effective_bw_series,
    load_mem_trace_npz,
    load_mem_trace_runs_npz,
)


def add_series(ax, trace_path: str, label: str, bin_us: float, is_bench: bool):
    if is_bench:
        trace_runs = load_mem_trace_runs_npz(trace_path)
        times_us, bw_gbps = average_effective_bw_series(trace_runs, bin_us=bin_us)
    else:
        trace_records = load_mem_trace_npz(trace_path)
        times_us, bw_gbps = compute_effective_bw_series(trace_records, bin_us=bin_us)

    if bw_gbps.size == 0:
        print(f"[mem-trace] no valid samples for {label} from {trace_path}")
        return False

    ax.plot(times_us, bw_gbps, linewidth=1.5, label=label)
    return True


def main():
    parser = argparse.ArgumentParser(description="Overlay BW-over-time curves from saved VDCores memory traces")
    parser.add_argument(
        "--series",
        action="append",
        nargs=3,
        metavar=("TRACE", "LABEL", "BIN_US"),
        default=[],
        help="Add a single-run trace series with its legend label and bin width",
    )
    parser.add_argument(
        "--bench-series",
        action="append",
        nargs=3,
        metavar=("TRACE", "LABEL", "BIN_US"),
        default=[],
        help="Add a benchmark trace archive series with its legend label and bin width",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for the overlaid BW plot")
    parser.add_argument("--title", type=str, default="Effective Memory Bandwidth Over Time", help="Plot title")
    args = parser.parse_args()

    if not args.series and not args.bench_series:
        parser.error("provide at least one --series or --bench-series entry")

    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
    plotted_any = False

    for trace_path, label, bin_us in args.series:
        plotted_any |= add_series(ax, trace_path, label, float(bin_us), is_bench=False)

    for trace_path, label, bin_us in args.bench_series:
        plotted_any |= add_series(ax, trace_path, label, float(bin_us), is_bench=True)

    if not plotted_any:
        print("[mem-trace] no valid series to plot")
        plt.close(fig)
        return

    if args.output is None:
        args.output = f"effective_bw_overlay_{int(time.time())}.png"

    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Effective BW (GB/s)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote overlaid effective bandwidth plot to {args.output}")


if __name__ == "__main__":
    main()
