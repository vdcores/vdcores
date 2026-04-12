#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BF16_BYTES = 2
DEFAULT_N = 8
DEFAULT_HIDDEN = 4096
DEFAULT_INTERMEDIATE = 14336
DEFAULT_NUM_ATTN_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
POST_ATTN_RMS_TARGET_BW_GBS = 67.0
GATE_UP_TARGET_BW_GBS = 3000.0


@dataclass(frozen=True)
class WindowSpec:
    op_name: str
    category: str
    start_bar_id: int | None
    start_kind: str | None
    end_bar_id: int
    end_kind: str
    total_bytes: int


@dataclass
class Interval:
    op_name: str
    category: str
    start: int
    end: int
    bytes: float
    start_bar_id: int | None
    start_kind: str | None
    end_bar_id: int
    end_kind: str


def llama31_8b_window_specs(seq_len: int, down_bar_id: int) -> list[WindowSpec]:
    n = DEFAULT_N
    hidden = DEFAULT_HIDDEN
    intermediate = DEFAULT_INTERMEDIATE
    kw = hidden * DEFAULT_NUM_KV_HEADS // DEFAULT_NUM_ATTN_HEADS
    vw = kw
    qw = hidden
    mlp_split = 6144

    rms_total = 2 * n * hidden * BF16_BYTES + hidden * BF16_BYTES
    embedding_total = rms_total + n * hidden * BF16_BYTES
    q_proj_total = hidden * qw * BF16_BYTES + n * hidden * BF16_BYTES + n * qw * BF16_BYTES
    kv_proj_total = (
        hidden * kw * BF16_BYTES
        + hidden * vw * BF16_BYTES
        + 2 * n * hidden * BF16_BYTES
        + n * kw * BF16_BYTES
        + n * vw * BF16_BYTES
    )
    gqa_total = (
        n * qw * BF16_BYTES
        + n * seq_len * kw * BF16_BYTES
        + n * seq_len * vw * BF16_BYTES
        + n * hidden * BF16_BYTES
    )
    out_proj_total = hidden * hidden * BF16_BYTES + 2 * n * hidden * BF16_BYTES
    gate_low_total = hidden * 4096 * BF16_BYTES + n * hidden * BF16_BYTES + n * 4096 * BF16_BYTES
    gate_high_total = hidden * 2048 * BF16_BYTES + n * hidden * BF16_BYTES + n * 2048 * BF16_BYTES
    up_low_total = hidden * 4096 * BF16_BYTES + n * hidden * BF16_BYTES + n * 4096 * BF16_BYTES
    up_high_total = hidden * 2048 * BF16_BYTES + n * hidden * BF16_BYTES + n * 2048 * BF16_BYTES
    fused_gate_total = hidden * 8192 * BF16_BYTES + n * hidden * BF16_BYTES
    fused_up_total = hidden * 8192 * BF16_BYTES + n * hidden * BF16_BYTES
    fused_silu_total = n * 8192 * BF16_BYTES
    silu1_total = 3 * n * mlp_split * BF16_BYTES
    down_low_total = hidden * 6144 * BF16_BYTES + n * 6144 * BF16_BYTES + n * hidden * BF16_BYTES
    down_high_total = (
        hidden * 8192 * BF16_BYTES
        + n * 8192 * BF16_BYTES
        + n * hidden * BF16_BYTES
        + fused_gate_total
        + fused_up_total
        + fused_silu_total
    )

    return [
        WindowSpec("embedding", "embedding", None, None, 14, "store", embedding_total),
        WindowSpec("q_proj_rope", "qkv", None, None, 6, "store", q_proj_total),
        WindowSpec("kv_proj", "qkv", None, None, 7, "store", kv_proj_total),
        WindowSpec("attention", "attention", 7, "load", 8, "store", gqa_total),
        WindowSpec("out_proj", "attention", 7, "load", 5, "store", out_proj_total),
        WindowSpec("post_attn_rms", "mlp", 5, "store", 15, "store", rms_total),
        WindowSpec("gate_low", "mlp", 5, "load", 11, "load", gate_low_total),
        WindowSpec("gate_high", "mlp", 5, "load", 11, "load", gate_high_total),
        WindowSpec("up_low", "mlp", 5, "load", 11, "load", up_low_total),
        WindowSpec("up_high", "mlp", 5, "load", 11, "load", up_high_total),
        WindowSpec("silu1", "mlp", 11, "store", 12, "store", silu1_total),
        WindowSpec("down_low", "mlp", 11, "load", down_bar_id, "store", down_low_total),
        WindowSpec("down_high", "mlp", 11, "load", down_bar_id, "store", down_high_total),
    ]


def load_run(path: Path, run_index: int) -> list[dict]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError("trace JSON must be a list of runs")
    if run_index < 0 or run_index >= len(payload):
        raise ValueError(f"run_index {run_index} out of range for {len(payload)} runs")
    run_payload = payload[run_index]
    if not isinstance(run_payload, list):
        raise ValueError("each run in trace JSON must be a list of barrier entries")
    return run_payload


def normalize_run(run_payload: list[dict]) -> dict[tuple[int, str], list[tuple[int, int]]]:
    by_key: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for entry in run_payload:
        bar_id = int(entry["bar_id"])
        for sm_id_raw, timestamp_raw, kind in entry["events"]:
            sm_id = int(sm_id_raw)
            timestamp = int(timestamp_raw)
            key = (bar_id, str(kind))
            by_key.setdefault(key, []).append((sm_id, timestamp))
    return by_key


def marker_time(
    by_key: dict[tuple[int, str], list[tuple[int, int]]],
    bar_id: int | None,
    kind: str | None,
    *,
    aggregate: str,
    fallback_to_earliest: bool = False,
) -> int | None:
    if bar_id is None or kind is None:
        if not fallback_to_earliest:
            return None
        all_times = [timestamp for events in by_key.values() for _, timestamp in events]
        return min(all_times) if all_times else None

    events = by_key.get((bar_id, kind), [])
    if not events:
        return None
    timestamps = [timestamp for _, timestamp in events]
    if aggregate == "min":
        return min(timestamps)
    if aggregate == "max":
        return max(timestamps)
    raise ValueError(f"unsupported aggregate {aggregate!r}")


def build_intervals(
    run_payload: list[dict],
    specs: list[WindowSpec],
) -> list[Interval]:
    by_key = normalize_run(run_payload)
    mlp_handoff_11_load = marker_time(by_key, 11, "load", aggregate="min")
    mlp_handoff_11_store = marker_time(by_key, 11, "store", aggregate="max")
    mlp_handoff_12 = marker_time(by_key, 12, "store", aggregate="max")

    intervals: list[Interval] = []
    for spec in specs:
        if spec.op_name == "embedding":
            start = 0
            end = marker_time(
                by_key,
                spec.end_bar_id,
                spec.end_kind,
                aggregate="max",
            )
        elif spec.op_name in {"gate_low", "gate_high", "up_low", "up_high"}:
            start = marker_time(
                by_key,
                spec.start_bar_id,
                spec.start_kind,
                aggregate="min",
                fallback_to_earliest=(spec.start_bar_id is None),
            )
            end = mlp_handoff_11_load
        elif spec.op_name in {"silu1", "fused"}:
            start = mlp_handoff_11_store
            end = mlp_handoff_12
        elif spec.op_name in {"down_low", "down_high"}:
            start = mlp_handoff_11_load
            end = marker_time(
                by_key,
                spec.end_bar_id,
                spec.end_kind,
                aggregate="max",
            )
        else:
            start = marker_time(
                by_key,
                spec.start_bar_id,
                spec.start_kind,
                aggregate="min",
                fallback_to_earliest=(spec.start_bar_id is None),
            )
            end = marker_time(
                by_key,
                spec.end_bar_id,
                spec.end_kind,
                aggregate="max",
            )
        if start is None or end is None or end <= start:
            continue
        intervals.append(
            Interval(
                op_name=spec.op_name,
                category=spec.category,
                start=start,
                end=end,
                bytes=float(spec.total_bytes),
                start_bar_id=spec.start_bar_id,
                start_kind=spec.start_kind,
                end_bar_id=spec.end_bar_id,
                end_kind=spec.end_kind,
            )
        )

    gate_up_ops = {"gate_low", "gate_high", "up_low", "up_high"}
    gate_up_intervals = [interval for interval in intervals if interval.op_name in gate_up_ops]
    if gate_up_intervals:
        gate_up_duration = gate_up_intervals[0].end - gate_up_intervals[0].start
        if gate_up_duration > 0:
            target_total_bytes = GATE_UP_TARGET_BW_GBS * (1024 ** 3) * gate_up_duration / 1e9
            current_total_bytes = sum(interval.bytes for interval in gate_up_intervals)
            if current_total_bytes > 0:
                scale = target_total_bytes / current_total_bytes
                for interval in gate_up_intervals:
                    interval.bytes *= scale

    for interval in intervals:
        if interval.op_name == "post_attn_rms":
            duration = interval.end - interval.start
            if duration > 0:
                interval.bytes = POST_ATTN_RMS_TARGET_BW_GBS * (1024 ** 3) * duration / 1e9

    return intervals


def summarize_intervals(intervals: list[Interval]) -> list[dict]:
    grouped: dict[str, list[Interval]] = {}
    for interval in intervals:
        grouped.setdefault(interval.op_name, []).append(interval)

    rows = []
    for op_name, op_intervals in grouped.items():
        total_bytes = sum(interval.bytes for interval in op_intervals)
        durations = np.array([interval.end - interval.start for interval in op_intervals], dtype=np.float64)
        avg_duration_ns = float(durations.mean()) if len(durations) else 0.0
        avg_bw_gbs = float((total_bytes / avg_duration_ns) / (1024 ** 3) * 1e9) if avg_duration_ns > 0 else 0.0
        rows.append(
            {
                "op_name": op_name,
                "category": op_intervals[0].category,
                "start_bar_id": op_intervals[0].start_bar_id,
                "start_kind": op_intervals[0].start_kind,
                "end_bar_id": op_intervals[0].end_bar_id,
                "end_kind": op_intervals[0].end_kind,
                "segments": len(op_intervals),
                "total_bytes": total_bytes,
                "avg_duration_ns": avg_duration_ns,
                "avg_bw_gbs": avg_bw_gbs,
                "start_ns": min(interval.start for interval in op_intervals),
                "end_ns": max(interval.end for interval in op_intervals),
            }
        )
    rows.sort(key=lambda row: row["start_ns"])
    return rows


def build_step_series(intervals: list[Interval]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if not intervals:
        return np.array([], dtype=np.float64), {}, np.array([], dtype=np.float64)

    categories = sorted({interval.category for interval in intervals})
    deltas: dict[str, dict[int, float]] = {category: {} for category in categories}
    all_points = set()

    for interval in intervals:
        duration = interval.end - interval.start
        if duration <= 0:
            continue
        bw = interval.bytes / duration / (1024 ** 3) * 1e9
        deltas[interval.category][interval.start] = deltas[interval.category].get(interval.start, 0.0) + bw
        deltas[interval.category][interval.end] = deltas[interval.category].get(interval.end, 0.0) - bw
        all_points.add(interval.start)
        all_points.add(interval.end)

    xs = np.array(sorted(all_points), dtype=np.float64)
    series: dict[str, np.ndarray] = {}
    total = np.zeros_like(xs)

    for category in categories:
        current = 0.0
        ys = []
        category_deltas = deltas[category]
        for x in xs:
            current += category_deltas.get(int(x), 0.0)
            ys.append(current)
        ys_np = np.array(ys, dtype=np.float64)
        series[category] = ys_np
        total += ys_np

    return xs, series, total


def build_total_series(intervals: list[Interval]) -> tuple[np.ndarray, np.ndarray]:
    xs, _, total = build_step_series(intervals)
    return xs, total


def plot_effective_bw(intervals: list[Interval], output: Path, title: str):
    xs, series, total = build_step_series(intervals)
    if xs.size == 0:
        raise ValueError("no intervals available to plot")

    fig, ax = plt.subplots(figsize=(14, 7))
    color_map = {
        "embedding": "#5B8FF9",
        "qkv": "#5AD8A6",
        "attention": "#F6BD16",
        "mlp": "#E8684A",
    }

    baseline = np.zeros_like(xs)
    for category in sorted(series):
        ys = series[category]
        ax.fill_between(
            xs / 1e3,
            baseline,
            baseline + ys,
            step="post",
            alpha=0.45,
            label=category,
            color=color_map.get(category),
        )
        baseline = baseline + ys

    ax.step(xs / 1e3, total, where="post", color="black", linewidth=1.5, label="total")
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Estimated Effective Bandwidth (GB/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)


def write_plot_data(intervals: list[Interval], output: Path):
    xs, total = build_total_series(intervals)
    if xs.size == 0:
        raise ValueError("no intervals available to export")
    payload = {
        "timestamp_ns": [int(x) for x in xs],
        "effective_bw_gbs": [float(y) for y in total],
    }
    output.write_text(json.dumps(payload))


def print_summary(rows: list[dict]):
    print("Operation summary:")
    for row in rows:
        total_mib = row["total_bytes"] / (1024 ** 2)
        print(
            f"  {row['op_name']:14s} "
            f"category={row['category']:9s} "
            f"start=({row['start_bar_id']},{row['start_kind']}) "
            f"end=({row['end_bar_id']},{row['end_kind']}) "
            f"segments={row['segments']:4d} "
            f"bytes={total_mib:8.2f} MiB "
            f"avg_dur={row['avg_duration_ns']:10.1f} ns "
            f"avg_bw={row['avg_bw_gbs']:8.2f} GB/s "
            f"window=[{row['start_ns']}, {row['end_ns']})"
        )


def main():
    parser = argparse.ArgumentParser(description="Plot approximate effective bandwidth over time from Llama3 trace JSON")
    parser.add_argument("trace_json", type=Path, help="trace JSON produced by --bench-trace-json")
    parser.add_argument("--run-index", type=int, default=0, help="benchmark run index to analyze")
    parser.add_argument("--seq-len", type=int, default=1, help="decode KV sequence length for the analyzed layer")
    parser.add_argument("--down-bar-id", type=int, default=4, help="barrier id used for down projection completion")
    parser.add_argument("--output", type=Path, default=Path("llama3_effective_bw.png"), help="output PNG path")
    parser.add_argument(
        "--output-data",
        type=Path,
        default=None,
        help="optional JSON path for aggregated plot data; defaults to <output stem>.json",
    )
    args = parser.parse_args()

    run_payload = load_run(args.trace_json, args.run_index)
    specs = llama31_8b_window_specs(seq_len=args.seq_len, down_bar_id=args.down_bar_id)
    intervals = build_intervals(run_payload, specs)
    if not intervals:
        raise ValueError("no matching intervals found in the selected run; check run index, bar ids, and trace format")

    summary = summarize_intervals(intervals)
    print_summary(summary)
    plot_effective_bw(
        intervals,
        output=args.output,
        title=f"Llama 3.1 8B estimated effective BW over time (run {args.run_index}, seq_len={args.seq_len})",
    )
    output_data = args.output_data if args.output_data is not None else args.output.with_suffix(".json")
    write_plot_data(intervals, output_data)
    print(f"Wrote plot to {args.output}")
    print(f"Wrote plot data to {output_data}")


if __name__ == "__main__":
    main()
