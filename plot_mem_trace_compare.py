#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


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


def parse_float_list(text: str) -> list[float]:
    items = []
    for piece in text.split(","):
        piece = piece.strip()
        if not piece:
            continue
        items.append(float(piece))
    return items


GUIDE_LINES = [
    (6.7, "gray"),
    (19.5, "red"),
    (151.0, "red"),
    (160.0, "red"),
]


def build_mem_trace_records(sm_id, start, end, size, opcode):
    return np.rec.fromarrays(
        [sm_id, start, end, size, opcode],
        names=["sm_id", "start", "end", "size", "opcode"],
    )


def load_baseline_mem_trace_npz(path: str):
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
    times_us = (np.arange(num_bins, dtype=np.float64) + 0.5) * bin_us
    return times_us, bw_gbps


def load_vdc_bw_json(path: str):
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("VDC JSON must be an object with timestamp_ns and effective_bw_gbs")
    if "timestamp_ns" not in payload or "effective_bw_gbs" not in payload:
        raise ValueError("VDC JSON must contain timestamp_ns and effective_bw_gbs")
    xs_ns = np.asarray(payload["timestamp_ns"], dtype=np.float64)
    ys_gbs = np.asarray(payload["effective_bw_gbs"], dtype=np.float64)
    if xs_ns.size == 0 or ys_gbs.size == 0 or xs_ns.size != ys_gbs.size:
        raise ValueError("VDC JSON must contain non-empty timestamp_ns and effective_bw_gbs arrays of equal length")
    xs_us = (xs_ns - xs_ns[0]) / 1e3
    return xs_us, ys_gbs


def main():
    parser = argparse.ArgumentParser(description="Overlay baseline and VDC effective bandwidth on the same plot")
    parser.add_argument("baseline_trace", type=Path, help="Baseline mem-trace NPZ with sm_id/start/end/size/opcode")
    parser.add_argument("vdc_bw", type=Path, help="VDC aggregated JSON with timestamp_ns/effective_bw_gbs")
    parser.add_argument("--bin-us", type=float, default=1.0, help="Time bin width in microseconds for the baseline series")
    parser.add_argument("--output", type=Path, default=Path("effective_bw_compare.png"), help="Output PNG path")
    parser.add_argument("--title", type=str, default="Effective Bandwidth Comparison", help="Plot title")
    parser.add_argument("--baseline-label", type=str, default="baseline", help="Legend label for baseline")
    parser.add_argument("--vdc-label", type=str, default="vdc", help="Legend label for VDC")
    parser.add_argument(
        "--vdc-resample-step-ns",
        type=float,
        default=256.0,
        help="uniform resample step in ns for optional VDC smoothing; default: 256",
    )
    parser.add_argument(
        "--vdc-smooth-window-ns",
        type=float,
        default=0.0,
        help="moving-average smoothing window in ns for VDC; 0 disables smoothing",
    )
    args = parser.parse_args()

    baseline_trace = load_baseline_mem_trace_npz(str(args.baseline_trace))
    base_xs, base_ys = compute_effective_bw_series(baseline_trace, bin_us=args.bin_us)
    vdc_xs, vdc_ys = load_vdc_bw_json(str(args.vdc_bw))
    vdc_smoothed = False
    if args.vdc_smooth_window_ns > 0:
        vdc_xs_ns = vdc_xs * 1e3
        vdc_xs_ns, vdc_ys = resample_step_series(vdc_xs_ns, vdc_ys, args.vdc_resample_step_ns)
        window_size = max(1, int(round(args.vdc_smooth_window_ns / args.vdc_resample_step_ns)))
        vdc_ys = moving_average(vdc_ys, window_size)
        if vdc_ys.size:
            vdc_ys[-1] = 0.0
        vdc_xs = vdc_xs_ns / 1e3
        vdc_smoothed = True

    if base_xs.size == 0:
        raise ValueError("baseline trace produced no valid samples")

    fig, ax = plt.subplots(figsize=(8, 3), dpi=160)
    ax.plot(base_xs, base_ys, linewidth=1.5, label=args.baseline_label, color="#76B7B2")
    if vdc_smoothed:
        ax.plot(vdc_xs, vdc_ys, linewidth=1.5, label=args.vdc_label, color="#F28E2B")
    else:
        ax.step(vdc_xs, vdc_ys, where="post", linewidth=1.5, label=args.vdc_label, color="#F28E2B")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Effective BW (GB/s)")
    ax.grid(alpha=0.3)
    kernel_boundary_handle = Line2D([0], [0], color="red", linestyle="--", linewidth=1.0, label="Kernel Boundary")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [kernel_boundary_handle],
        labels + ["Kernel Boundary"],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False,
    )

    curve_max = max(float(base_ys.max()), float(vdc_ys.max()) if vdc_ys.size else 0.0)
    for x, color in GUIDE_LINES:
        ax.vlines(x, 0.0, curve_max, colors=color, linestyles="--", linewidth=1.0, alpha=0.8)
    ax.set_ylim(bottom=0.0)
    fig.canvas.draw()
    plot_ymax = ax.get_ylim()[1]
    for coll in ax.collections[-len(GUIDE_LINES):]:
        segments = coll.get_segments()
        for seg in segments:
            seg[0, 1] = 0.0
            seg[1, 1] = plot_ymax
        coll.set_segments(segments)

    ax.annotate(
        "",
        xy=(19.5, 900.0),
        xytext=(6.7, 900.0),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2, linestyle="--"),
    )
    ax.text(
        19.5 + 0.8,
        900.0,
        "Delayed Loading",
        ha="left",
        va="center",
        color="black",
    )
    ax.annotate(
        "",
        xy=(161.0, 1500.0),
        xytext=(150.0, 1500.0),
        arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2, linestyle="--"),
    )
    ax.text(
        160.0 + 1.5,
        1500.0,
        "Gmem Round-trip",
        ha="left",
        va="center",
        color="black",
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    print(f"[compare] wrote comparison plot to {args.output}")
    print(f"[compare] {args.baseline_label} avg: {base_ys.mean():.3f} GB/s")
    print(f"[compare] {args.baseline_label} peak: {base_ys.max():.3f} GB/s")
    print(f"[compare] {args.vdc_label} avg: {vdc_ys.mean():.3f} GB/s")
    print(f"[compare] {args.vdc_label} peak: {vdc_ys.max():.3f} GB/s")


if __name__ == "__main__":
    main()
