"""Render camera task-graph data with trial spread and explicit scope labels."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from dae.camera_control import generated_frames, frame_identity

METHODS = ("cpu", "serial-gpu", "dag-gpu", "adaptive")
LABELS = ("CPU only", "Serial GPU", "Ready-task GPU", "Adaptive")
COLORS = ("#64748b", "#5378b5", "#078b88", "#db8b2b")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "figure.facecolor": "#fafbfc",
                     "axes.facecolor": "#fafbfc", "svg.hashsalt": "camera-task-graphs-v1"})


def save(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=170, bbox_inches="tight")
    path = directory / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def trials(data, method, metric):
    return [metric(r) for r in data["runs"] if r["method"] == method]


def bars(ax, names, datasets, metric, ylabel):
    x = np.arange(len(names))
    for index, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        values = [trials(d, method, metric) for d in datasets]
        center = np.array([statistics.median(v) for v in values])
        low, high = np.array([min(v) for v in values]), np.array([max(v) for v in values])
        ax.bar(x + (index - 1.5) * .19, center, .18, color=color, label=label,
               yerr=np.array([center - low, high - center]), capsize=2,
               error_kw={"linewidth": .8})
    ax.set_xticks(x, names)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/camera_control/a100-taskgraph-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "results/camera_control/figures")
    args = parser.parse_args()
    names = ("small-1", "small-4", "large-1", "large-4", "large-4-cpu-load", "large-4-gpu-load",
             "large-4-period10ms", "large-4-period2ms", "large-4-tail")
    data = {}
    for name in names:
        with gzip.open(args.data / f"{name}.json.gz", "rt") as stream:
            data[name] = json.load(stream)
    args.output.mkdir(parents=True, exist_ok=True)
    # Archive the exact generated RGB inputs, not merely a seed or a preview.
    # Check against the measured dataset identity before calling them identical.
    fixture_directory = args.output.parent / "fixtures"
    fixture_directory.mkdir(exist_ok=True)
    for h, w in {(d["input"]["shape"][1], d["input"]["shape"][2]) for d in data.values()}:
        frames = generated_frames(height=h, width=w)
        identity = frame_identity(frames, source="generated")
        expected = {d["input"]["sha256"] for d in data.values() if d["input"]["shape"][1:3] == [h, w]}
        if expected != {identity["sha256"]}:
            raise ValueError("generated fixture does not match measured image hashes")
        np.savez_compressed(fixture_directory / f"generated-{h}x{w}.npz", frames=frames)
    summary = [{"dataset": name, "method": r["method"], "trial": r["repeat"],
                "correctness": r["correctness"], "measurements": r["measurements"]}
               for name, dataset in data.items() for r in dataset["runs"]]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    base = [data[name] for name in names[:4]]
    labels = ["128×192\n1 loop", "128×192\n4 loops", "480×640\n1 loop", "480×640\n4 loops"]
    bars(axes[0], labels, base, lambda r: r["measurements"]["latency"]["median_ns"] / 1e6, "Frame latency (ms)")
    bars(axes[1], labels, base, lambda r: r["measurements"]["throughput_frames_per_second"], "Aggregate frames / second")
    axes[0].legend(ncol=4, frameon=False, loc="upper left")
    axes[0].set_title("End-to-end camera processing: device choice depends on image size", loc="left", pad=18)
    fig.supxlabel("Bars: median of three trial summaries · whiskers: trial min–max · generated scenes, A100 host", fontsize=9)
    save(fig, args.output, "method_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    load = [data[n] for n in ("large-4", "large-4-cpu-load", "large-4-gpu-load")]
    for ax, field, label in zip(axes, ("median_ns", "p99_ns"), ("Median response (ms)", "p99 response (ms)")):
        bars(ax, ["Idle host", "2 competing\nCPU threads", "Competing\nGPU pulses"], load,
             lambda r, field=field: r["measurements"]["latency"][field] / 1e6, label)
    axes[0].legend(ncol=2, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.suptitle("Sharing the machine: background load changes the trade-offs", weight="bold")
    fig.supxlabel("480×640 · four feedback loops · identical operations · whiskers show trial spread, not confidence intervals", fontsize=9)
    save(fig, args.output, "background_load")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for column, (key, label) in enumerate((("large-4-period10ms", "10 ms arrivals"), ("large-4-period2ms", "2 ms arrivals"))):
        dataset = data[key]
        for index, (method, method_label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
            response = trials(dataset, method, lambda r: r["measurements"]["latency"]["median_ns"] / 1e6)
            active = trials(dataset, method, lambda r: r["measurements"]["active_latency"]["median_ns"] / 1e6)
            axes[0, column].scatter([index - .12] * len(response), response, s=30, color=color,
                                    label="Scheduled arrival → output" if index == 0 else None)
            axes[0, column].scatter([index + .12] * len(active), active, s=32, facecolors="none", edgecolors=color,
                                    marker="D", label="Admission → output" if index == 0 else None)
            method_runs = [r for r in dataset["runs"] if r["method"] == method]
            miss_fraction = (sum(r["measurements"]["deadline_misses"] for r in method_runs) /
                             sum(r["measurements"]["latency"]["count"] for r in method_runs))
            axes[1, column].bar(index, miss_fraction, color=color)
            axes[1, column].text(index, miss_fraction + .015, f"{miss_fraction:.1%}", ha="center", fontsize=9)
        for ax in axes[:, column]:
            ax.set_xticks(range(4), LABELS, rotation=18, ha="right")
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
        axes[0, column].set_title(label + " per loop")
        axes[0, column].set_ylabel("Trial median latency (ms, log scale)")
        axes[0, column].set_yscale("log")
        axes[0, column].axhline(10, color="#b54444", linestyle="--", linewidth=1)
        axes[1, column].set_ylabel("All measured frames missing 10 ms deadline")
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, column].set_ylim(0, 1.05)
    fig.suptitle("Arrival backlog counts: no frame drops hidden", weight="bold")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.supxlabel("Filled circles: response including backlog · hollow diamonds: active processing · dashed line: 10 ms deadline", fontsize=9)
    save(fig, args.output, "periodic_deadlines")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    dataset = data["large-4-tail"]
    for method, label, color in zip(METHODS, LABELS, COLORS):
        raw = [row["end_ns"] - row["release_ns"] for r in dataset["runs"] if r["method"] == method
               for row in r["frames"] if row["frame"] >= dataset["configuration"]["warmup"]]
        ordered = np.sort(np.asarray(raw) / 1e6)
        survivor = (len(ordered) - np.arange(len(ordered))) / len(ordered)
        axes[0].step(ordered, survivor, where="post", label=label, color=color)
        for r in dataset["runs"]:
            if r["method"] == method:
                axes[1].scatter([row["lane"] + METHODS.index(method) * .11 for row in r["measurements"]["per_lane"]],
                                [row["latency"]["p99_ns"] / 1e6 for row in r["measurements"]["per_lane"]],
                                s=20, color=color, alpha=.65)
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xticks([5, 10, 20, 50, 100], ["5", "10", "20", "50", "100"])
    axes[0].set_xlabel("Arrival → control output (ms)")
    axes[0].set_ylabel("Fraction at or above latency")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=.15)
    axes[1].set_xticks(range(4), ["Loop 0", "Loop 1", "Loop 2", "Loop 3"])
    axes[1].set_ylabel("Per-loop p99 response (ms)")
    axes[1].grid(axis="y", alpha=.15)
    fig.suptitle("Longer runs preserve slow frames and per-loop differences", weight="bold")
    fig.supxlabel("1,000 measured frames/loop/trial · three trials · pooled survival curves; dots are individual loop/trial p99", fontsize=9)
    save(fig, args.output, "tail_latency")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), constrained_layout=True)
    run = next(r for r in data["large-4"]["runs"] if r["method"] == "dag-gpu" and r["repeat"] == 0)
    for lane in range(4):
        rows = sorted([r for r in run["frames"] if r["lane"] == lane and r["frame"] < 48], key=lambda r: r["frame"])
        for ax, index in zip(axes, (0, 1)):
            ax.plot([r["frame"] for r in rows], [r["output"][index] for r in rows], label=f"Loop {lane}", linewidth=1.5)
    axes[0].set_ylabel("Steering command")
    axes[1].set_ylabel("Speed command")
    axes[1].set_xlabel("Feedback frame")
    axes[0].legend(ncol=4, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.suptitle("Commands change with the replayed scene and previous control state", weight="bold")
    for ax in axes:
        ax.grid(alpha=.15)
    fig.supxlabel("Generated scene sequence · normalized commands, not calibrated physical steering or velocity", fontsize=9)
    save(fig, args.output, "feedback_commands")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, task in zip(axes, ("lane", "obstacle")):
        fractions = []
        for dataset in base:
            runs = [r for r in dataset["runs"] if r["method"] == "adaptive"]
            cpu = sum(r["measurements"]["placement_counts"][task]["cpu"] for r in runs)
            gpu = sum(r["measurements"]["placement_counts"][task]["gpu"] for r in runs)
            fractions.append(cpu / (cpu + gpu))
        ax.bar(range(4), fractions, color=COLORS[0], label="CPU")
        ax.bar(range(4), 1 - np.asarray(fractions), bottom=fractions, color=COLORS[2], label="GPU")
        ax.set_xticks(range(4), labels)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(task.capitalize() + " perception")
    axes[0].legend(frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.02))
    fig.suptitle("What adaptive execution actually selected", weight="bold")
    fig.supxlabel("Measured frames pooled over three trials · placement is a cost prediction, not an optimality guarantee", fontsize=9)
    save(fig, args.output, "adaptive_decisions")

    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1, 1.8))
    image_ax, graph_ax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    image_ax.imshow(generated_frames()[5])
    image_ax.axis("off")
    image_ax.set_title("Generated camera fixture")
    graph_ax.set_xlim(0, 10)
    graph_ax.set_ylim(0, 6)
    graph_ax.axis("off")
    boxes = [(1.4, 3, "CPU\nROI + normalize"), (5, 4.5, "CPU or GPU\nLane perception"),
             (5, 1.5, "CPU or GPU\nObstacle perception"), (8.5, 3, "CPU\nSteering + speed")]
    for x, y, label in boxes:
        graph_ax.text(x, y, label, ha="center", va="center", fontsize=11,
                      bbox={"boxstyle": "round,pad=.7", "fc": "#e8f3f2", "ec": COLORS[2]})
    for start, end in [((2.6, 3.4), (3.6, 4.3)), ((2.6, 2.6), (3.6, 1.7)),
                       ((6.3, 4.3), (7.4, 3.4)), ((6.3, 1.7), (7.4, 2.6))]:
        graph_ax.annotate("", end, start, arrowprops={"arrowstyle": "->", "color": "#475569", "lw": 1.5})
    graph_ax.annotate("", (1.4, 2.3), (8.5, 2.3),
                      arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=-.5", "color": COLORS[3], "lw": 1.8})
    graph_ax.text(5, .15, "Previous command changes the next frame's processing", ha="center", fontsize=10)
    graph_ax.set_title("One reusable graph per feedback loop")
    fig.suptitle("Camera → perception → command: the implemented end-to-end path", weight="bold")
    fig.supxlabel("Python coordinates ordinary Torch tasks · no physical robot, trained detector, or original VDCores VM in these runs", fontsize=9)
    save(fig, args.output, "camera_pipeline")
    captions = [
        ("camera_pipeline", "The implemented path and generated scene fixture; this is software-in-the-loop, not real-world autonomy."),
        ("method_comparison", "All four idle-host configurations. Bars are medians of three trial summaries; whiskers span trials."),
        ("background_load", "Matched 480×640/four-loop comparisons with explicit CPU or GPU competing work. p99 from 200-frame trials is coarse."),
        ("periodic_deadlines", "Scheduled arrival latency includes backlog; active processing latency is separate. Dots show individual trial medians; miss fractions pool all measured frames. Every frame is retained."),
        ("tail_latency", "12,000 measured frames per method across three trials. Pooled curves and individual per-loop/trial p99 preserve slow observations."),
        ("feedback_commands", "Stateful commands from actual recorded experiment outputs, not an illustrative trajectory or physical robot trace."),
        ("adaptive_decisions", "Actual CPU/GPU task decisions. A high GPU selection rate does not by itself establish a performance benefit."),
    ]
    content = "# Camera-control task graph figures\n\nGenerated-scene A100 experiments; ordinary Torch execution, not the original VDCores GPU VM.\n\n"
    for name, caption in captions:
        content += f"## {name.replace('_', ' ').title()}\n\n{caption}\n\n![{caption}]({name}.png)\n\n[Editable SVG]({name}.svg)\n\n"
    content += "[Trial summary data](summary.json). Regenerate with `python tools/plot_camera_control.py`.\n"
    (args.output / "README.md").write_text(content)


if __name__ == "__main__":
    main()
