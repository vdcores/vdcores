#!/usr/bin/env python3
import argparse
import ast
import json
import re
import runpy
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.transforms import blended_transform_factory


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
    # (6.7, "gray"),
    # (19.5, "red"),
    # (151.0, "red"),
    # (160.0, "red"),
]

SYSTEM_KEYS = ("baseline", "vdc")
STAGE_PALETTE = list(plt.get_cmap("tab20").colors)
X_TICK_FONTSIZE = 8
Y_TICK_FONTSIZE = 8
ENABLE_BOX_HATCH = False


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


def normalize_stage_name(stage_name: str) -> str:
    return re.sub(r"-\d+$", "", stage_name)


def warn(message: str) -> None:
    print(f"[compare] warning: {message}", file=sys.stderr)


def detect_duplicate_stage_keys(path: Path) -> list[str]:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    duplicates: list[str] = []

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "stage_annotations" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for key_node, value_node in zip(node.value.keys, node.value.values):
            if not isinstance(key_node, ast.Constant) or key_node.value != "stages":
                continue
            if not isinstance(value_node, ast.Dict):
                continue
            seen: set[str] = set()
            for stage_key in value_node.keys:
                if not isinstance(stage_key, ast.Constant) or not isinstance(stage_key.value, str):
                    continue
                if stage_key.value in seen:
                    duplicates.append(stage_key.value)
                else:
                    seen.add(stage_key.value)
            return duplicates
    return duplicates


def load_stage_annotations(path: Path) -> dict | None:
    if not path.exists():
        warn(f"annotation file not found: {path}; skipping stage spans")
        return None

    duplicates = detect_duplicate_stage_keys(path)
    if duplicates:
        raise ValueError(
            f"Duplicate stage keys in {path}: {sorted(set(duplicates))}. "
            "Use unique names such as rms-1 and rms-2."
        )

    payload = runpy.run_path(str(path))
    annotations = payload.get("stage_annotations")
    if not isinstance(annotations, dict):
        raise ValueError(f"{path} must define a top-level stage_annotations dict")
    if annotations.get("unit") != "us":
        raise ValueError(f"{path} stage_annotations['unit'] must be 'us'")
    if annotations.get("timebase") != "relative_from_plot_start":
        raise ValueError(f"{path} stage_annotations['timebase'] must be 'relative_from_plot_start'")
    stages = annotations.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"{path} stage_annotations['stages'] must be a dict")
    return annotations


def validate_stage_range(stage_name: str, system_name: str, stage_range: object) -> tuple[float, float, str | None] | None:
    if not isinstance(stage_range, dict):
        warn(f"skipping {stage_name}.{system_name}: range must be an object")
        return None
    if "start_us" not in stage_range or "end_us" not in stage_range:
        warn(f"skipping {stage_name}.{system_name}: missing start_us/end_us")
        return None

    start_us = float(stage_range["start_us"])
    end_us = float(stage_range["end_us"])
    if end_us < start_us:
        warn(f"skipping {stage_name}.{system_name}: end_us ({end_us}) < start_us ({start_us})")
        return None
    if bool(stage_range.get("fused", False)):
        mode = "xx////" if stage_name == "rms-1" else "////"
    elif bool(stage_range.get("overlap", False)):
        mode = "xx"
    else:
        mode = None
    return start_us, end_us, mode


def build_stage_color_map(stages: dict) -> dict[str, tuple[float, float, float]]:
    color_map: dict[str, tuple[float, float, float]] = {}
    for stage_name in stages:
        color_key = normalize_stage_name(stage_name)
        if color_key not in color_map:
            color_map[color_key] = STAGE_PALETTE[len(color_map) % len(STAGE_PALETTE)]
    return color_map


def compute_overlap_regions(
    intervals: list[tuple[float, float, str, tuple[float, float, float], str | None]]
) -> list[tuple[float, float, str]]:
    events: list[tuple[float, int, int, float, str | None]] = []
    for idx, (start_us, end_us, _, _, mode) in enumerate(intervals):
        if end_us <= start_us:
            continue
        events.append((start_us, 1, idx, start_us, mode))
        events.append((end_us, -1, idx, start_us, mode))

    events.sort(key=lambda item: (item[0], -item[1], item[2]))
    overlaps: list[tuple[float, float, str]] = []
    active = 0
    active_modes: dict[int, tuple[float, str | None]] = {}
    overlap_start: float | None = None
    overlap_hatch = "xx"

    def choose_hatch() -> str:
        explicit_modes = [value for value in active_modes.values() if value[1] is not None]
        if not explicit_modes:
            return "xx"
        _, hatch = max(explicit_modes, key=lambda item: item[0])
        assert hatch is not None
        return hatch

    for time_us, delta, interval_id, start_us, mode in events:
        prev_active = active
        prev_hatch = choose_hatch()
        active += delta
        if delta > 0:
            active_modes[interval_id] = (start_us, mode)
        else:
            active_modes.pop(interval_id, None)
        if prev_active < 2 and active >= 2:
            overlap_start = time_us
            overlap_hatch = choose_hatch()
        elif prev_active >= 2 and active >= 2 and overlap_start is not None:
            next_hatch = choose_hatch()
            if next_hatch != prev_hatch and time_us > overlap_start:
                overlaps.append((overlap_start, time_us, prev_hatch))
                overlap_start = time_us
                overlap_hatch = next_hatch
        elif prev_active >= 2 and active < 2 and overlap_start is not None and time_us > overlap_start:
            overlaps.append((overlap_start, time_us, overlap_hatch))
            overlap_start = None

    return overlaps


def add_curve_overlap_hatching(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    overlap_regions: list[tuple[float, float, str]],
    *,
    step_where: str | None = None,
) -> None:
    if xs.size == 0 or ys.size == 0:
        return

    def sample_curve_y(query_x: float) -> float:
        if step_where == "post":
            idx = np.searchsorted(xs, query_x, side="right") - 1
            idx = int(np.clip(idx, 0, len(ys) - 1))
            return float(ys[idx])
        return float(np.interp(query_x, xs, ys))

    for overlap_start, overlap_end, hatch in overlap_regions:
        where = (xs >= overlap_start) & (xs <= overlap_end)
        if not np.any(where):
            continue
        fill_kwargs = dict(
            where=where,
            interpolate=True,
            facecolor="none",
            edgecolor="#222222",
            hatch=hatch,
            linewidth=0.0,
            zorder=1.1,
        )
        if step_where is not None:
            fill_kwargs["step"] = step_where
        ax.fill_between(xs, 0.0, ys, **fill_kwargs)
        boundary_ys = [sample_curve_y(overlap_start), sample_curve_y(overlap_end)]
        ax.vlines(
            [overlap_start, overlap_end],
            0.0,
            boundary_ys,
            colors="#222222",
            linewidth=0.8,
            zorder=1.15,
        )


def add_stage_spans(base_ax, vdc_ax, annotations: dict | None) -> tuple[list[Patch], dict[str, list[tuple[float, float, str]]]]:
    if annotations is None:
        return [], {"baseline": [], "vdc": []}

    stages = annotations["stages"]
    color_map = build_stage_color_map(stages)
    legend_handles: list[Patch] = []
    seen_legend_keys: set[str] = set()
    axis_by_system = {"baseline": base_ax, "vdc": vdc_ax}
    system_intervals: dict[str, list[tuple[float, float, str, tuple[float, float, float], str | None]]] = {
        "baseline": [],
        "vdc": [],
    }
    overlap_regions_by_system: dict[str, list[tuple[float, float, str]]] = {"baseline": [], "vdc": []}

    for stage_name, stage_spec in stages.items():
        if not isinstance(stage_spec, dict):
            warn(f"skipping {stage_name}: stage entry must be an object")
            continue

        color_key = normalize_stage_name(stage_name)
        color = color_map[color_key]
        if color_key not in seen_legend_keys:
            legend_handles.append(Patch(facecolor=color, edgecolor="#333333", linewidth=0.9, alpha=0.35, label=color_key))
            seen_legend_keys.add(color_key)

        for system_name in SYSTEM_KEYS:
            if system_name not in stage_spec:
                continue
            validated = validate_stage_range(stage_name, system_name, stage_spec[system_name])
            if validated is None:
                continue
            start_us, end_us, mode = validated
            system_intervals[system_name].append((start_us, end_us, stage_name, color, mode))

    for system_name, intervals in system_intervals.items():
        if not intervals:
            continue

        band_top = 1.16
        band_bottom = 1.04
        axis = axis_by_system[system_name]
        transform = blended_transform_factory(axis.transData, axis.transAxes)
        for start_us, end_us, stage_name, color, mode in intervals:
            axis.add_patch(
                Rectangle(
                    (start_us, band_bottom),
                    end_us - start_us,
                    band_top - band_bottom,
                    transform=transform,
                    facecolor=color,
                    edgecolor="#333333",
                    linewidth=0.9,
                    alpha=0.35,
                    clip_on=False,
                    zorder=0.5,
                )
            )
        for overlap_start, overlap_end, hatch in compute_overlap_regions(intervals):
            overlap_regions_by_system[system_name].append((overlap_start, overlap_end, hatch))
            if ENABLE_BOX_HATCH:
                axis.add_patch(
                    Rectangle(
                        (overlap_start, band_bottom),
                        overlap_end - overlap_start,
                        band_top - band_bottom,
                        transform=transform,
                        facecolor="none",
                        edgecolor="#222222",
                        hatch=hatch,
                        linewidth=0.0,
                        clip_on=False,
                        zorder=0.7,
                    )
                )
                axis.vlines(
                    [overlap_start, overlap_end],
                    band_bottom,
                    band_top,
                    colors="#222222",
                    linewidth=0.8,
                    transform=transform,
                    clip_on=False,
                    zorder=0.75,
                )

    return legend_handles, overlap_regions_by_system


def build_hatch_legend_handles() -> list[Patch]:
    return [
        Patch(facecolor="white", edgecolor="#222222", hatch="xx", linewidth=0.8, label="Overlap"),
        Patch(facecolor="white", edgecolor="#222222", hatch="////", linewidth=0.8, label="Fusion"),
    ]


def build_curve_legend_handles(baseline_label: str, vdc_label: str) -> list[Line2D]:
    return [
        Line2D([0], [0], color="#76B7B2", linewidth=1.5, label=baseline_label),
        Line2D([0], [0], color="#F28E2B", linewidth=1.5, label=vdc_label),
    ]


def main():
    parser = argparse.ArgumentParser(description="Plot baseline and VDC effective bandwidth on stacked shared-x subplots")
    parser.add_argument("baseline_trace", type=Path, help="Baseline mem-trace NPZ with sm_id/start/end/size/opcode")
    parser.add_argument("vdc_bw", type=Path, help="VDC aggregated JSON with timestamp_ns/effective_bw_gbs")
    parser.add_argument("--bin-us", type=float, default=1.0, help="Time bin width in microseconds for the baseline series")
    parser.add_argument("--output", type=Path, default=Path("effective_bw_compare.png"), help="Output PNG path")
    parser.add_argument("--title", type=str, default="Effective Bandwidth Comparison", help="Plot title")
    parser.add_argument("--baseline-label", type=str, default="Mirage", help="Legend label for baseline")
    parser.add_argument("--vdc-label", type=str, default="VDE", help="Legend label for VDC")
    parser.add_argument(
        "--events",
        type=Path,
        default=Path("event.py"),
        help="Python file defining stage_annotations for stage color spans; default: ./event.py",
    )
    parser.add_argument(
        "--vdc-resample-step-ns",
        type=float,
        default=256.0,
        help="uniform resample step in ns for optional VDC smoothing; default: 256",
    )
    parser.add_argument(
        "--vdc-smooth-window-ns",
        type=float,
        default=5000.0,
        help="moving-average smoothing window in ns for VDC; 0 disables smoothing",
    )
    args = parser.parse_args()

    baseline_trace = load_baseline_mem_trace_npz(str(args.baseline_trace))
    base_xs, base_ys = compute_effective_bw_series(baseline_trace, bin_us=args.bin_us)
    vdc_xs, vdc_ys = load_vdc_bw_json(str(args.vdc_bw))
    stage_annotations = load_stage_annotations(args.events)
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

    fig, (base_ax, vdc_ax) = plt.subplots(
        2,
        1,
        figsize=(6, 4.5),
        dpi=160,
        sharex=True,
        gridspec_kw={"hspace": 0.42},
    )
    stage_handles, overlap_regions_by_system = add_stage_spans(base_ax, vdc_ax, stage_annotations)
    curve_handles = build_curve_legend_handles(args.baseline_label, args.vdc_label)
    hatch_handles = build_hatch_legend_handles() if stage_handles else []
    base_ax.plot(base_xs, base_ys, linewidth=1.5, label=args.baseline_label, color="#76B7B2")
    if vdc_smoothed:
        vdc_ax.plot(vdc_xs, vdc_ys, linewidth=1.5, label=args.vdc_label, color="#F28E2B")
    else:
        vdc_ax.step(vdc_xs, vdc_ys, where="post", linewidth=1.5, label=args.vdc_label, color="#F28E2B")
    add_curve_overlap_hatching(base_ax, base_xs, base_ys, overlap_regions_by_system["baseline"])
    add_curve_overlap_hatching(
        vdc_ax,
        vdc_xs,
        vdc_ys,
        overlap_regions_by_system["vdc"],
        step_where=None if vdc_smoothed else "post",
    )

    # base_ax.set_title(args.title)
    base_ax.set_ylabel("Effective BW (GB/s)", fontsize=10)
    vdc_ax.set_ylabel("Effective BW (GB/s)", fontsize=10)
    vdc_ax.set_xlabel("Time (us)", fontsize=10)
    if X_TICK_FONTSIZE is not None:
        base_ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
        vdc_ax.tick_params(axis="x", labelsize=X_TICK_FONTSIZE)
    if Y_TICK_FONTSIZE is not None:
        base_ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
        vdc_ax.tick_params(axis="y", labelsize=Y_TICK_FONTSIZE)
    base_ax.grid(alpha=0.3)
    vdc_ax.grid(alpha=0.3)
    if stage_handles:
        fig.legend(
            handles=curve_handles + stage_handles + hatch_handles,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            ncol=1,
            frameon=False,
            fontsize=8,
            # title="Stage Colors",
        )
    else:
        fig.legend(
            handles=curve_handles,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            ncol=1,
            frameon=False,
            fontsize=8,
        )

    for x, color in GUIDE_LINES:
        base_ax.axvline(x, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
        vdc_ax.axvline(x, color=color, linestyle="--", linewidth=1.0, alpha=0.8)
    base_ax.set_ylim(bottom=0.0)
    vdc_ax.set_ylim(bottom=0.0)

    fig.subplots_adjust(
        left=0.1,
        right=0.82 if stage_handles else 0.97,
        top=0.88 if stage_handles else 0.92,
        bottom=0.10,
    )
    fig.savefig(args.output, dpi=160)
    plt.close(fig)

    print(f"[compare] wrote comparison plot to {args.output}")
    print(f"[compare] {args.baseline_label} avg: {base_ys.mean():.3f} GB/s")
    print(f"[compare] {args.baseline_label} peak: {base_ys.max():.3f} GB/s")
    print(f"[compare] {args.vdc_label} avg: {vdc_ys.mean():.3f} GB/s")
    print(f"[compare] {args.vdc_label} peak: {vdc_ys.max():.3f} GB/s")


if __name__ == "__main__":
    main()
