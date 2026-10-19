"""Measure same-host native-worker / CUDA-event remaining-stage placement costs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from dae.embedded_backends import NativeCpuBackend, CudaEventBackend
from dae.embedded_experiment import run_loop, measurements, verify_feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-items", default="64,4096,65536,1048576")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--interpolation", choices=("linear", "log2"), default="linear")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sizes = sorted(int(n) for n in args.work_items.split(","))
    if (not sizes or len(set(sizes)) != len(sizes) or min(sizes) <= 0 or max(sizes) > 16777216
            or not 1 <= args.iterations <= 1_000_000 or args.warmup < 0
            or not 1 <= args.repeats <= 100 or not 1 <= args.workers <= 256):
        parser.error("invalid calibration sizes or run counts")
    if args.output.exists():
        parser.error("output already exists")
    import torch
    if not torch.cuda.is_available():
        parser.error("real CUDA calibration requires a GPU")
    result = {
        "schema_version": 1, "experiment": "vdcores_embedded_placement_calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "workload": "synthetic_float64_energy_feedback", "gpu_transport": "torch_cuda_events",
        "cpu_executor": "persistent_native_workers",
        "metric": "external_observed_ns", "poll_us": 0,
        "interpolation": args.interpolation,
        "hardware": {"gpu_name": torch.cuda.get_device_name(args.device), "machine": platform.node(),
                     "workers": args.workers, "torch_version": torch.__version__,
                     "torch_cuda": torch.version.cuda, "platform": platform.platform(),
                     "cpu": platform.processor()},
        "native_source_sha256": hashlib.sha256((ROOT / "src/embedded_loop.cpp").read_bytes()).hexdigest(),
        "configuration": {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}, "rows": [], "trials": [],
    }
    for count in sizes:
        medians = {"cpu": [], "gpu": []}
        cpu_costs = {"preprocess_ns": [], "control_ns": []}
        for repeat in range(args.repeats):
            order = ("cpu", "gpu") if repeat % 2 == 0 else ("gpu", "cpu")
            for device in order:
                backend = (NativeCpuBackend(count, 1) if device == "cpu" else
                           CudaEventBackend(count, 1, device=args.device))
                run = run_loop(backend, work_items=count, frames=args.iterations + args.warmup,
                               lanes=1, workers=args.workers, poll_us=0, timeout_s=300,
                               library=args.library)
                run["correctness"] = verify_feedback(run["traces"], work_items=count,
                                                     frames=args.iterations + args.warmup, lanes=1)
                run["measurements"] = measurements(run, warmup=args.warmup, lanes=1, deadline_ns=None)
                run.update(device=device, repeat=repeat, work_items=count)
                result["trials"].append(run)
                medians[device].append(run["measurements"]["stages"]["external_observed_ns"]["median_ns"])
                if device == "cpu":
                    for name in cpu_costs:
                        cpu_costs[name].append(run["measurements"]["stages"][name]["median_ns"])
        row = {"work_items": count, **{device: {"external_observed_ns": {
            "median_ns": round(statistics.median(values)), "trial_medians_ns": values}}
            for device, values in medians.items()}}
        row["cpu_stage_costs_ns"] = {name: round(statistics.median(values))
                                     for name, values in cpu_costs.items()}
        result["rows"].append(row)
        print(f"n={count}: CPU={row['cpu']['external_observed_ns']['median_ns']/1000:.1f} us, "
              f"GPU={row['gpu']['external_observed_ns']['median_ns']/1000:.1f} us", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
