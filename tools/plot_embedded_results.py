"""Produce reproducible scientific figures from recorded embedded-loop JSON."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import statistics as stats
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"events": "#2476B7", "blocking": "#DD8B25", "native_cpu_workers": "#26836C",
          "adaptive": "#9555AD", "cpu_reference": "#26836C",
          "graph_events": "#0D887D", "graph_blocking": "#AE6378"}
LABELS = {"events": "GPU events", "blocking": "GPU blocking", "native_cpu_workers": "Native CPU",
          "adaptive": "Adaptive", "cpu_reference": "CPU reference",
          "graph_events": "Graph + events", "graph_blocking": "Graph + blocking"}


def load(path):
    with (gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)) as file:
        return json.load(file)


def paths(directory):
    found = {}
    for path in sorted(directory.glob("*.json*")):
        if path.name.endswith((".json", ".json.gz")):
            found[path.name.removesuffix(".gz")] = path
    return list(found.values())


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 12, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#A8B3BC", "axes.titlepad": 12,
                         "text.color": "#233448", "axes.labelcolor": "#233448",
                         "xtick.color": "#526375", "ytick.color": "#526375",
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "grid.color": "#DDE4EB", "grid.alpha": 0.8,
                         "svg.fonttype": "none"})


def save(fig, directory, name, caption):
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    svg = directory / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)
    return {"name": name, "caption": caption}


def trial_values(data, mode, metric="median_ns"):
    return [r["measurements"]["stages"]["end_to_end_ns"][metric] / 1000
            for r in data["runs"] if r["mode"] == mode]


def event_sweep(root, out):
    data = [load(p) for p in paths(root / "a100-initial-sweep")]
    if not data:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    fig.suptitle("Initial A100 sweep: sharing and polling affect response time", fontsize=17, x=0.02, ha="left")
    for row, lanes in enumerate((1, 4)):
        for col, metric in enumerate(("median_ns", "p95_ns")):
            ax = axes[row, col]
            for poll in (0, 50):
                selected = sorted([d for d in data if d["configuration"]["lanes"] == lanes
                                   and d["configuration"]["poll_us"] == poll],
                                  key=lambda d: d["configuration"]["work_items"])
                for mode in ("events", "blocking"):
                    x = [d["configuration"]["work_items"] for d in selected]
                    values = [trial_values(d, mode, metric) for d in selected]
                    center = [stats.median(v) for v in values]
                    ax.plot(x, center, marker="o", color=COLORS[mode],
                            ls="-" if poll == 0 else "--", label=f"{LABELS[mode]}, poll {poll} µs")
                    ax.fill_between(x, [min(v) for v in values], [max(v) for v in values],
                                    color=COLORS[mode], alpha=0.07)
            ax.set(xscale="log", xlabel="Input values (float64)", ylabel="Per-frame latency (µs)",
                   title=f"{lanes} independent loop{'s' if lanes > 1 else ''} · {'median' if col == 0 else 'p95'}")
            ax.grid(axis="y")
    axes[0, 0].legend(fontsize=9, loc="best", frameon=False)
    return [save(fig, out, "a100_event_sweep", "Initial implementation, before the zero-time condition-variable poll fix. Median of three trial summaries; shading is their min/max range, not a confidence interval. Each trial has 300 measured frames per loop. Different curves show requested polling intervals.")]


def stage_breakdown(root, out):
    selected = []
    for path in paths(root / "a100-initial-sweep"):
        d = load(path)
        if d["configuration"]["work_items"] == 4096 and d["configuration"]["poll_us"] == 0:
            for mode in ("events", "blocking"):
                selected.append((d["configuration"]["lanes"], mode, d))
    if not selected:
        return []
    selected.sort(key=lambda x: (x[0], x[1]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fig.suptitle("Initial stage breakdown: host coordination dominated", fontsize=17, x=0.02, ha="left")
    spans = [("CPU queue", "ready_ns", "prep_start_ns", "#91B8A7"),
             ("Preprocess", "prep_start_ns", "prep_end_ns", "#26836C"),
             ("External service queue", "prep_end_ns", "taken_ns", "#C5CFD9"),
             ("Submit / GPU / observe", "taken_ns", "external_end_ns", "#2476B7"),
             ("Controller queue + work", "external_end_ns", "end_ns", "#9555AD")]
    labels, bottom = [], np.zeros(len(selected))
    for name, start, end, color in spans:
        means = []
        for lanes, mode, d in selected:
            traces = [t for r in d["runs"] if r["mode"] == mode for t in r["traces"]
                      if t["frame"] >= d["configuration"]["warmup"]]
            means.append(stats.mean(t[end] - t[start] for t in traces) / 1000)
        axes[0].bar(range(len(selected)), means, bottom=bottom, color=color, label=name, width=0.65)
        bottom += means
    bottom = np.zeros(len(selected))
    for key, label, color in (("h2d_ns", "Host → GPU", "#2476B7"),
                              ("compute_ns", "Reduction", "#26836C"),
                              ("d2h_ns", "GPU → host", "#DD8B25")):
        means = [stats.mean(t[key] for r in d["runs"] if r["mode"] == mode
                            for t in r["device_timings"] if t["frame"] >= d["configuration"]["warmup"]) / 1000
                 for _, mode, d in selected]
        axes[1].bar(range(len(selected)), means, bottom=bottom, color=color, label=label, width=0.65)
        bottom += means
    labels = [f"{lanes} loop{'s' if lanes > 1 else ''}\n{mode}" for lanes, mode, _ in selected]
    for ax, title in zip(axes, ("Host-clock frame spans", "CUDA-event intervals (different scale)")):
        ax.set(xticks=range(len(selected)), xticklabels=labels, ylabel="Mean elapsed time (µs)", title=title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        if ax is axes[1]:
            ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        else:
            ax.legend(fontsize=9, frameon=False, loc="upper left")
    return [save(fig, out, "a100_stage_breakdown", "Initial implementation before the zero-time poll fix and graph replay: 4,096 values, poll 0 µs, three trials. Means of individual frame spans add to the mean total. GPU event intervals include work captured between events and are shown on a separate scale; host overhead is not a kernel-latency measurement.")]


def placement(root, out):
    candidates = sorted(root.glob("a100-placement-calibration-final-v2.json*")) or sorted(root.glob("a100-placement-calibration-final.json*"))
    if not candidates:
        return []
    d = load(candidates[-1])
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    fig.suptitle("Measured placement cost: CPU wins small, GPU wins large", fontsize=17, x=0.02, ha="left")
    for device, color, label in (("cpu", COLORS["native_cpu_workers"], "Persistent native CPU workers"),
                                 ("gpu", COLORS["events"], "A100 with copies and event completion")):
        x = [r["work_items"] for r in d["rows"]]
        values = [r[device]["external_observed_ns"]["trial_medians_ns"] for r in d["rows"]]
        med = [stats.median(v) / 1000 for v in values]
        ax.plot(x, med, "o-", color=color, lw=2, label=label)
        ax.fill_between(x, [min(v) / 1000 for v in values], [max(v) / 1000 for v in values], alpha=0.12, color=color)
    ax.set(xscale="log", yscale="log", xlabel="Input values (float64)",
           ylabel="Remaining feature-stage latency (µs)")
    ax.grid(which="major")
    ax.legend(frameon=False)
    return [save(fig, out, "a100_placement_calibration", "Same-host single-loop calibration, three trials per device and size. Cost includes dispatch, queueing, transfers and completion observation. This is not raw GPU kernel time; curves connect measurements and do not establish an exact crossover.")]


def adaptive(root, out):
    directory = root / "a100-final-paired"
    if not directory.exists():
        directory = root / "a100-paired-comparison"
    if not directory.exists():
        directory = root / "a100-final-adaptive"
    data = [load(p) for p in paths(directory)]
    if not data:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fig.suptitle("Adaptive placement versus both fixed-device baselines", fontsize=17, x=0.02, ha="left")
    for ax, lanes in zip(axes, (1, 4)):
        for mode in ("native_cpu_workers", "events", "adaptive"):
            selected = sorted([d for d in data if d["configuration"]["lanes"] == lanes
                               and any(r["mode"] == mode for r in d["runs"])],
                              key=lambda d: d["configuration"]["work_items"])
            x = [d["configuration"]["work_items"] for d in selected]
            med = [stats.median(trial_values(d, mode)) / 1000 for d in selected]
            ax.plot(x, med, "o-", color=COLORS[mode], label=LABELS[mode], lw=2)
            for d, xpos in zip(selected, x):
                ax.scatter([xpos] * len(trial_values(d, mode)), np.array(trial_values(d, mode)) / 1000,
                           color=COLORS[mode], s=12, alpha=0.3)
        ax.set(xscale="log", xlabel="Input values (float64)", ylabel="Median frame latency (ms)",
               title=f"{lanes} independent loop{'s' if lanes > 1 else ''}")
        ax.grid(axis="y")
    axes[0].legend(frameon=False)
    for ax in axes:
        sizes = sorted({d["configuration"]["work_items"] for d in data})
        ax.set_xticks(sizes, [f"{size:,}" for size in sizes])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return [save(fig, out, "a100_adaptive_placement", "Three independent trials; points are trial medians and lines show their median. 131,072 and 524,288 values are held out from the calibration grid; 1,048,576 is a calibrated endpoint. All methods execute the same feedback workload. Adaptive logging/selection overhead remains included.")]


def tail(root, out):
    directory = root / "a100-final-tail"
    if not directory.exists():
        directory = root / "a100-tail-clean"
    if not directory.exists():
        directory = root / "a100-tail-validation"
    data = [load(p) for p in paths(directory)]
    if not data:
        return []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fig.suptitle("Longer runs expose the tail and per-loop response time", fontsize=17, x=0.02, ha="left")
    modes = [mode for mode in ("events", "blocking", "graph_events", "graph_blocking")
             if any(r["mode"] == mode for d in data for r in d["runs"])]
    for index, mode in enumerate(modes):
        runs = [r for d in data for r in d["runs"] if r["mode"] == mode]
        values = np.sort([ (t["end_ns"] - t["ready_ns"]) / 1e6 for d in data
                          for r in d["runs"] if r["mode"] == mode
                          for t in r["traces"] if t["frame"] >= d["configuration"]["warmup"]])
        axes[0].plot(values, (len(values) - np.arange(len(values))) / len(values),
                     color=COLORS[mode], label=LABELS[mode], lw=2)
        for lane in range(4):
            val = [r["measurements"]["per_lane"][lane]["latency"]["p99_ns"] / 1e6 for r in runs]
            axes[1].scatter([lane + (index-(len(modes)-1)/2)*.10] * len(val), val,
                            color=COLORS[mode], alpha=0.8)
    axes[0].set(xscale="log", yscale="log", xlabel="Frame latency (ms, logarithmic scale)", ylabel="Fraction of frames at least this slow")
    axes[0].set_xticks([0.3, 0.5, 1, 2, 5, 10, 40], ["0.3", "0.5", "1", "2", "5", "10", "40"])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].legend(frameon=False)
    axes[1].set(xticks=range(4), xticklabels=[f"Loop {i+1}" for i in range(4)], ylabel="Per-loop p99 latency (ms)")
    for ax in axes:
        ax.grid(axis="y")
    return [save(fig, out, "a100_tail_latency", "65,536 values and four loops, 10,000 measured frames per loop in each of three trials per mode. Left pools raw latency samples for the empirical survival curve; right shows each trial's per-loop p99. Pooling is descriptive and not a confidence interval.")]


def local_and_feedback(root, out):
    data = [load(p) for p in paths(root) if "macos-cpu-lanes" in p.name]
    figures = []
    if data:
        data.sort(key=lambda d: d["configuration"]["lanes"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
        fig.suptitle("Local CPU validation: more throughput, longer response time", fontsize=16, x=0.02, ha="left")
        for ax, metric, ylabel in ((axes[0], "latency", "Median frame latency (µs)"),
                                    (axes[1], "rate", "Aggregate frames / second")):
            for i, d in enumerate(data):
                vals = (trial_values(d, "cpu_reference") if metric == "latency" else
                        [r["measurements"]["throughput_frames_per_second"] for r in d["runs"]])
                ax.bar(i, stats.median(vals), color="#26836C", alpha=0.3, width=0.55)
                ax.scatter(np.linspace(i-.08, i+.08, len(vals)), vals, color="#26836C", s=30)
            ax.set(xticks=range(len(data)), xticklabels=[f"{d['configuration']['lanes']} loop(s)" for d in data], ylabel=ylabel)
            ax.grid(axis="y")
            ax.set_axisbelow(True)
        figures.append(save(fig, out, "local_cpu_validation", "Apple M4 Pro, real CPU reference execution, no CUDA. Three trials; 4,096 values, two native workers. Dots show individual trials. These numbers are not a GPU comparison."))
        traces = data[0]["runs"][0]["traces"][:60]
        fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
        ax.plot([t["frame"] for t in traces], [t["action"] for t in traces], color="#9555AD", lw=2, label="Controller action")
        ax.plot([t["frame"] for t in traces], [t["feature"] for t in traces], color="#2476B7", lw=2, label="Measured feature")
        ax.set(title="The previous action changes the next synthetic sensor frame", xlabel="Frame", ylabel="Value")
        ax.grid(axis="y")
        ax.legend(frameon=False)
        figures.append(save(fig, out, "feedback_trace", "First 60 actual frames of one local loop, including warmup. Every feature/action is checked against an independent reference, rather than merely checking that a counter advances."))
    return figures


def graph_comparison(root, out):
    directory = root / "a100-final-graph"
    if not directory.exists():
        directory = root / "a100-graph-comparison"
    data = [load(p) for p in paths(directory)]
    if not data:
        return []
    data.sort(key=lambda d: (d["configuration"]["work_items"], d["configuration"]["lanes"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained")
    fig.suptitle("CUDA Graph replay reduces repeated submission work", fontsize=17, x=0.02, ha="left")
    modes = ("events", "blocking", "graph_events", "graph_blocking")
    for ax, metric, title in zip(axes, ("median_ns", "p95_ns"), ("Median frame latency", "p95 frame latency")):
        for i, mode in enumerate(modes):
            med = [stats.median(trial_values(d, mode, metric)) for d in data]
            xpos = np.arange(len(data)) + (i-1.5)*.19
            ax.bar(xpos, med, width=.18, color=COLORS[mode], label=LABELS[mode])
            for x, d in zip(xpos, data):
                vals = trial_values(d, mode, metric)
                ax.scatter([x]*len(vals), vals, color="#233448", s=8, alpha=.5)
        ax.set(xticks=range(len(data)), xticklabels=[f"{d['configuration']['work_items']:,} values\n{d['configuration']['lanes']} loop(s)" for d in data],
               ylabel="Frame latency (µs)", title=title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)
    return [save(fig, out, "a100_cuda_graph_replay", "Same native graph, buffers, reductions, and copies. Four transports are run in alternating trial order. Bars are medians of trial summaries; dots are the individual trials. Graph replay has one whole-replay timing interval, so it is not assigned fictitious H2D/kernel/D2H substage timings.")]


def interpolation_diagnostic(root, out):
    candidates = sorted(root.glob("a100-placement-calibration.json*"))
    observed = [p for p in paths(root / "a100-adaptive") if "n262144-lanes1" in p.name]
    if not candidates or not observed:
        return []
    sys.path.insert(0, str(ROOT / "python"))
    from dae.placement import MeasuredCrossoverPolicy
    data = load(candidates[-1])
    predictions = [MeasuredCrossoverPolicy.from_benchmark(data, metric="external_observed_ns",
                   switch_margin=.05, interpolation=method).choose(262144) for method in ("log2", "linear")]
    measured = {}
    for path in observed:
        d = load(path)
        for mode in ("native_cpu_workers", "events"):
            vals = [r["measurements"]["stages"]["external_observed_ns"]["median_ns"]
                    for r in d["runs"] if r["mode"] == mode]
            if vals:
                measured[mode] = stats.median(vals)
    if len(measured) != 2:
        return []
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    fig.suptitle("Held-out validation caught an early GPU decision", fontsize=17, x=.02, ha="left")
    for i, (field, mode, label) in enumerate((("cpu_estimate_ns", "native_cpu_workers", "CPU cost"),
                                            ("gpu_estimate_ns", "events", "GPU cost"))):
        values = [measured[mode], *(getattr(p, field) for p in predictions)]
        ax.bar(np.arange(3)+(i-.5)*.3, np.array(values)/1000, width=.28, color=COLORS[mode], label=label)
    ax.set(xticks=range(3), xticklabels=["Held-out measurement", f"Log-size model\nchooses {predictions[0].device.upper()}",
                                       f"Linear-size model\nchooses {predictions[1].device.upper()}"],
           ylabel="Remaining feature-stage cost (µs)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    return [save(fig, out, "placement_model_validation", "262,144 values, one loop. Both predictions use the same original four calibration sizes, excluding this held-out size. Bars show the measured stage cost or its estimate. Linear interpolation corrects this CPU/GPU decision, but neither model perfectly predicts both costs; later calibration adds denser samples.")]


def historical(out):
    handoff = load(ROOT / "results/cpu_gpu_handoff/2026-09-21-gh200.json")
    chain = load(ROOT / "results/cpu_gpu_handoff/2026-09-22-attention-chain.json")["final_trial"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    fig.suptitle("Earlier GH200 evidence: separate hardware and workload", fontsize=17, x=0.02, ha="left")
    for j, (key, label, color) in enumerate((("gpu_to_cpu", "GPU → CPU", "#2476B7"),
                                           ("round_trip", "Round trip", "#9555AD"))):
        for i, field in enumerate(("system_trials", "pinned_trials")):
            vals = [t[key]["median_ns"] / 1000 for t in handoff[field]]
            axes[0].scatter([i + (j-.5)*.18]*len(vals), vals, s=32, alpha=0.6, color=color,
                            label=label if i == 0 else None)
            axes[0].hlines(stats.median(vals), i+(j-.5)*.18-.05, i+(j-.5)*.18+.05, color=color, lw=3)
    axes[0].set(xticks=[0, 1], xticklabels=["Ordinary CPU memory", "Pinned memory"],
                ylabel="Per-trial median (µs)", title="Basic handoff includes fresh launches")
    axes[0].legend(frameon=False)
    for i, (mode, color) in enumerate((("blocking", "#DD8B25"), ("handoff", "#2476B7"))):
        for j, metric in enumerate(("median_ns", "p95_ns")):
            value = chain[mode]["end_to_end_ns"][metric] / 1e6
            axes[1].bar(j + (i-.5)*.32, value, width=.3, color=color, label=mode if j == 0 else None)
            axes[1].text(j+(i-.5)*.32, value+.035, f"{value:.3f}", ha="center", fontsize=10)
    axes[1].set(xticks=[0, 1], xticklabels=["Median", "p95"], ylabel="Full-chain latency (ms)",
                title="Attention → CPU → prequeued GPU")
    axes[1].legend(frameon=False)
    axes[1].set_ylim(0, 4.2)
    for ax in axes:
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    return [save(fig, out, "historical_gh200", "September 21–22 archived GH200 results. The first plot retains the transient outlier; the second uses the final 1,000-iteration attention-chain run. These launch-inclusive coherent-handoff results cannot be directly compared with the new A100 synthetic loop.")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/embedded_loop")
    parser.add_argument("--output", type=Path, default=ROOT / "results/embedded_loop/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup()
    figures = []
    for function in (event_sweep, stage_breakdown, placement, adaptive, tail, local_and_feedback,
                     graph_comparison, interpolation_diagnostic):
        figures.extend(function(args.results, args.output))
    figures.extend(historical(args.output))
    rows = []
    for path in sorted(args.results.rglob("*.json*")):
        if args.output in path.parents:
            continue
        if path.name.endswith((".json", ".json.gz")):
            d = load(path)
            if not isinstance(d, dict):
                continue
            for mode in sorted({r["mode"] for r in d.get("runs", [])}):
                runs = [r for r in d["runs"] if r["mode"] == mode]
                rows.append({"file": str(path.relative_to(args.results)), "mode": mode,
                             "configuration": d["configuration"],
                             "checked_frames": sum(r["correctness"]["checked_frames"] for r in runs),
                             "median_us": stats.median(trial_values(d, mode)),
                             "p95_us": stats.median(trial_values(d, mode, "p95_ns")),
                             "p99_us": stats.median(trial_values(d, mode, "p99_ns")),
                             "throughput_fps": stats.median(r["measurements"]["throughput_frames_per_second"] for r in runs),
                             "deadline_misses": sum(r["measurements"]["deadline_misses"] or 0 for r in runs),
                             "placements": {device: sum(t["device"] == device for r in runs
                                                         for t in r.get("placement_decisions", []))
                                            for device in ("cpu", "gpu")}})
    (args.output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    gallery = ["# VDCores experiment figures", "", "Generated from recorded measurements by `tools/plot_embedded_results.py`. PNG and editable SVG files are included. Trial spread, sample counts, hardware, and limitations are stated below each plot.", ""]
    for figure in figures:
        gallery += [f"## {figure['name'].replace('_', ' ').title()}", "",
                    f"![{figure['name']}]({figure['name']}.png)", "", figure["caption"], ""]
    (args.output / "README.md").write_text("\n".join(gallery))
    print(f"Generated {len(figures)} figures and summaries in {args.output}")


if __name__ == "__main__":
    main()
