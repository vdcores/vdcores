import argparse
import time

import matplotlib.pyplot as plt
import numpy as np


def build_mem_trace_records(sm_id, start, end, size, opcode):
    return np.rec.fromarrays(
        [sm_id, start, end, size, opcode],
        names=["sm_id", "start", "end", "size", "opcode"],
    )


def load_mem_trace_npz(path: str):
    with np.load(path) as data:
        required = {"sm_id", "start", "end", "size", "opcode"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing required trace arrays in {path}: {sorted(missing)}")

        return build_mem_trace_records(
            data["sm_id"].astype(np.int32, copy=False),
            data["start"].astype(np.uint64, copy=False),
            data["end"].astype(np.uint64, copy=False),
            data["size"].astype(np.uint32, copy=False),
            data["opcode"].astype(np.uint16, copy=False),
        )


def compute_effective_bw_series(trace_records, bin_us: float = 1.0):
    if trace_records.size == 0:
        return np.array([]), np.array([])

    bin_ns = max(bin_us * 1e3, 1.0)
    start_ns = float(trace_records["start"].min())
    end_ns = float(trace_records["end"].max())
    num_bins = max(1, int(np.ceil((end_ns - start_ns) / bin_ns)))
    bytes_per_bin = np.zeros(num_bins, dtype=np.float64)

    for record in trace_records:
        rec_start = float(record["start"])
        rec_end = float(record["end"])
        if rec_end <= rec_start:
            continue

        total_bytes = float(record["size"])
        left = int((rec_start - start_ns) // bin_ns)
        right = int(np.ceil((rec_end - start_ns) / bin_ns))
        for bin_idx in range(max(0, left), min(num_bins, right)):
            bin_start = start_ns + bin_idx * bin_ns
            bin_end = bin_start + bin_ns
            overlap = max(0.0, min(rec_end, bin_end) - max(rec_start, bin_start))
            if overlap <= 0:
                continue
            bytes_per_bin[bin_idx] += total_bytes * overlap / (rec_end - rec_start)

    bw_gbps = bytes_per_bin / (bin_ns * 1e-9) / 1e9
    times_us = (start_ns + (np.arange(num_bins) + 0.5) * bin_ns - start_ns) / 1e3
    return times_us, bw_gbps


def main():
    parser = argparse.ArgumentParser(description="Standalone BW-over-time plot for a saved memory trace")
    parser.add_argument("trace", type=str, help="Path to a single-run memory trace .npz")
    parser.add_argument("--bin-us", type=float, default=1.0, help="Time bin width in microseconds")
    parser.add_argument("--output", type=str, default=None, help="Output path for the BW plot")
    parser.add_argument("--title", type=str, default="Effective Memory Bandwidth Over Time", help="Plot title")
    args = parser.parse_args()

    trace_records = load_mem_trace_npz(args.trace)
    times_us, bw_gbps = compute_effective_bw_series(trace_records, bin_us=args.bin_us)
    if bw_gbps.size == 0:
        print("[mem-trace] no valid memory trace samples captured")
        return

    if args.output is None:
        args.output = f"effective_bw_{int(time.time())}.png"

    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
    ax.plot(times_us, bw_gbps, linewidth=1.5)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Effective BW (GB/s)")
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    plt.close(fig)

    print(f"[mem-trace] wrote effective bandwidth plot to {args.output}")
    print(f"[mem-trace] average effective bandwidth: {bw_gbps.mean():.3f} GB/s")
    print(f"[mem-trace] peak effective bandwidth: {bw_gbps.max():.3f} GB/s")


if __name__ == "__main__":
    main()
