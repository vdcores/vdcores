"""Reproducible CPU correctness / discrete-GPU event-transport experiment."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from dae.embedded import positive
from dae.embedded_backends import CpuBackend, CudaEventBackend, NativeCpuBackend, AdaptiveBackend, CudaGraphBackend
from dae.embedded_experiment import measurements, run_loop, verify_feedback


def command(*args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "native-cpu", "cuda", "adaptive", "compare", "cuda-graph", "graph-compare"), default="cpu")
    parser.add_argument("--policy", type=Path, help="real calibration JSON, required for adaptive")
    parser.add_argument("--mode", choices=("events", "blocking", "both"), default="both")
    parser.add_argument("--work-items", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-us", type=float, default=50)
    parser.add_argument("--timeout-s", type=float, default=60)
    parser.add_argument("--deadline-ms", type=float)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    positive(args.iterations, "iterations", 1_000_000)
    positive(args.repeats, "repeats", 100)
    positive(args.work_items, "work_items", 16_777_216)
    positive(args.lanes, "lanes", 256)
    positive(args.workers, "workers", 256)
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    if (args.iterations + args.warmup) * args.lanes > 1_000_000:
        parser.error("at most 1,000,000 frame records per run")
    if args.work_items * args.lanes > 33_554_432:
        parser.error("input buffers exceed 256 MiB")
    if not math.isfinite(args.poll_us) or args.poll_us < 0:
        parser.error("poll-us must be finite and nonnegative")
    if not math.isfinite(args.timeout_s) or args.timeout_s <= 0:
        parser.error("timeout-s must be finite and positive")
    if args.deadline_ms is not None and (not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0):
        parser.error("deadline-ms must be finite and positive")
    # Preserve previous experiment files rather than replacing them accidentally.
    if args.output.exists():
        parser.error("output already exists; choose a fresh result path")
    if args.backend in ("adaptive", "compare") and args.policy is None:
        parser.error("adaptive execution requires --policy")
    environment = {"python": sys.version, "platform": platform.platform(),
                   "cpu_model": command("sysctl", "-n", "machdep.cpu.brand_string")
                   if sys.platform == "darwin" else platform.processor(),
                   "compiler": command("c++", "--version"),
                   "cpu_count": os.cpu_count(), "git_revision": command("git", "rev-parse", "HEAD"),
                   "git_status": command("git", "status", "--short"),
                   "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,pci.bus_id,power.limit", "--format=csv")
                   if args.backend in ("cuda", "adaptive", "compare", "cuda-graph", "graph-compare") else None}
    if args.backend in ("cuda", "adaptive", "compare", "cuda-graph", "graph-compare"):
        import torch
        if not torch.cuda.is_available():
            parser.error("CUDA is unavailable; CPU results cannot substitute for CUDA measurements")
        props = torch.cuda.get_device_properties(args.device)
        environment.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                           gpu_name=props.name, compute_capability=[props.major, props.minor],
                           gpu_memory_bytes=props.total_memory)
    result = {"schema_version": 1, "experiment": "vdcores_embedded_fixed_feedback_loop",
              "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment,
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "transport": {"cuda": "cuda_events_with_explicit_copies", "cpu": "native_cpu_reference",
                            "native-cpu": "native_cpu_workers", "adaptive": "adaptive_cpu_cuda_events",
                            "compare": "paired_native_cpu_gpu_adaptive", "cuda-graph": "cuda_graph_explicit_copies",
                            "graph-compare": "paired_cuda_submission_transports"}[args.backend],
              "workload": "synthetic_float64_energy_feedback_not_neural_inference",
              "scope": "fixed_graph_prototype_not_hopper_vm_or_coherent_atomic_measurement",
              "runs": []}
    modes = ["events", "blocking"] if args.mode == "both" else [args.mode]
    if args.backend == "cpu":
        modes = ["cpu_reference"]  # No invented CUDA baseline on a CPU-only host.
    elif args.backend == "native-cpu":
        modes = ["native_cpu_workers"]
    elif args.backend == "adaptive":
        modes = ["adaptive"]
    elif args.backend == "compare":
        modes = ["native_cpu_workers", "events", "adaptive"]
    elif args.backend == "cuda-graph":
        modes = ["graph_events", "graph_blocking"] if args.mode == "both" else [f"graph_{args.mode}"]
    elif args.backend == "graph-compare":
        modes = ["events", "blocking", "graph_events", "graph_blocking"]
    for repeat in range(args.repeats):
        for mode in modes if repeat % 2 == 0 else list(reversed(modes)):
            if args.backend == "cpu":
                backend = CpuBackend(args.work_items, args.lanes)
            elif args.backend == "native-cpu" or (args.backend == "compare" and mode == "native_cpu_workers"):
                backend = NativeCpuBackend(args.work_items, args.lanes)
            elif args.backend == "adaptive" or (args.backend == "compare" and mode == "adaptive"):
                backend = AdaptiveBackend(args.work_items, args.lanes, policy_path=args.policy,
                                          workers=args.workers, device=args.device)
            elif mode.startswith("graph_"):
                backend = CudaGraphBackend(args.work_items, args.lanes,
                                           blocking=mode == "graph_blocking", device=args.device)
            else:
                backend = CudaEventBackend(args.work_items, args.lanes, blocking=mode == "blocking", device=args.device)
            run = run_loop(backend, work_items=args.work_items, frames=args.warmup + args.iterations,
                           lanes=args.lanes, workers=args.workers, timeout_s=args.timeout_s,
                           poll_us=args.poll_us, library=args.library)
            run.update(repeat=repeat, mode=mode)
            run["correctness"] = verify_feedback(run["traces"], work_items=args.work_items,
                                                 frames=args.warmup + args.iterations, lanes=args.lanes)
            run["measurements"] = measurements(run, warmup=args.warmup, lanes=args.lanes,
                                              deadline_ns=round(args.deadline_ms * 1e6) if args.deadline_ms else None)
            result["runs"].append(run)
            stats = run["measurements"]["stages"]["end_to_end_ns"]
            print(f"repeat={repeat} mode={mode}: median={stats['median_ns']/1000:.1f} us, "
                  f"p95={stats['p95_ns']/1000:.1f} us, correctness=PASS")
    # Native code may differ from HEAD while developing. Record exact source hash.
    source = ROOT / "src" / "embedded_loop.cpp"
    result["native_source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [source, ROOT / "python/dae/embedded.py",
                                            ROOT / "python/dae/embedded_backends.py",
                                            ROOT / "python/dae/embedded_experiment.py",
                                            ROOT / "python/dae/placement.py",
                                            Path(__file__).resolve()]}
    library = args.library or ROOT / "build" / ("libembedded_loop.dylib" if sys.platform == "darwin" else "libembedded_loop.so")
    result["native_library_sha256"] = hashlib.sha256(library.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "xt", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
