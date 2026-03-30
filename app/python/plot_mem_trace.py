import argparse

from dae.util import (
    load_mem_trace_npz,
    load_mem_trace_runs_npz,
    save_effective_bw_heatmap_from_runs,
    save_effective_bw_heatmap_from_trace,
    save_effective_bw_plot_from_runs,
    save_effective_bw_plot_from_trace,
)


def main():
    parser = argparse.ArgumentParser(description="Plot saved VDCores memory traces")
    parser.add_argument("trace", type=str, help="Path to a saved .npz memory trace")
    parser.add_argument("--bench", action="store_true", help="Interpret the trace file as a benchmark trace-runs archive")
    parser.add_argument("--bin-us", type=float, default=1.0, help="Time bin width in microseconds")
    parser.add_argument("--bw-plot", type=str, default=None, help="Output path for the aggregate BW-over-time plot")
    parser.add_argument("--heatmap", type=str, default=None, help="Output path for the per-SM BW heatmap")
    args = parser.parse_args()

    if args.bench:
        trace_runs = load_mem_trace_runs_npz(args.trace)
        if not trace_runs:
            print("[mem-trace] no runs found in trace archive")
            return
        num_sms = 0
        for trace in trace_runs:
            if trace.size > 0:
                num_sms = max(num_sms, int(trace["sm_id"].max()) + 1)
        if args.bw_plot is not None:
            save_effective_bw_plot_from_runs(trace_runs, path=args.bw_plot, bin_us=args.bin_us)
        if args.heatmap is not None:
            save_effective_bw_heatmap_from_runs(trace_runs, num_sms=num_sms, path=args.heatmap, bin_us=args.bin_us)
        if args.bw_plot is None and args.heatmap is None:
            save_effective_bw_plot_from_runs(trace_runs, bin_us=args.bin_us)
            save_effective_bw_heatmap_from_runs(trace_runs, num_sms=num_sms, bin_us=args.bin_us)
        return

    trace_records = load_mem_trace_npz(args.trace)
    num_sms = int(trace_records["sm_id"].max()) + 1 if trace_records.size > 0 else 0
    if args.bw_plot is not None:
        save_effective_bw_plot_from_trace(trace_records, path=args.bw_plot, bin_us=args.bin_us)
    if args.heatmap is not None:
        save_effective_bw_heatmap_from_trace(trace_records, num_sms=num_sms, path=args.heatmap, bin_us=args.bin_us)
    if args.bw_plot is None and args.heatmap is None:
        save_effective_bw_plot_from_trace(trace_records, bin_us=args.bin_us)
        save_effective_bw_heatmap_from_trace(trace_records, num_sms=num_sms, bin_us=args.bin_us)


if __name__ == "__main__":
    main()
