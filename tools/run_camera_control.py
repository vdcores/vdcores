"""Sequential, auditable camera-control graph calibration and evaluation."""
from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from dae.camera_control import (camera_graph, frame_identity, generated_frames, input_factory,
                                load_frames, preprocess, verify_camera)
from dae.embedded_experiment import summarize
from dae.graph_execution import CpuTaskBackend, TorchCudaTaskBackend, execute_graph
from dae.placement import MeasuredCrossoverPolicy, PlacementSample


def command(*args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def environment():
    sources = ["python/dae/graph_execution.py", "python/dae/camera_control.py",
               "python/dae/heterogeneous.py", "python/dae/placement.py", "tools/run_camera_control.py"]
    return {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
            "cpu": command("sysctl", "-n", "machdep.cpu.brand_string") if sys.platform == "darwin"
                   else command("lscpu"), "git_revision": command("git", "rev-parse", "HEAD"),
            "git_status": command("git", "status", "--short"),
            "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources},
            "torch_cuda": torch.version.cuda,
            "nvidia_smi": command("nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv")
                          if torch.cuda.is_available() else None}


def calibrate(graph, images, *, repeats=3, samples=20, workers=2, gpu_slots=4):
    """Measure per-task observed service costs on separate calibration frames."""
    wake = threading.Event()
    cpu = CpuTaskBackend(workers, notify=wake)
    gpu = TorchCudaTaskBackend(gpu_slots)
    rows = []
    try:
        for name, task in graph.tasks.items():
            if len(task.operation.devices) == 1:
                continue
            for trial in range(repeats):
                for device in (("cpu", "gpu") if trial % 2 == 0 else ("gpu", "cpu")):
                    backend = cpu if device == "cpu" else gpu
                    values = []
                    for i in range(samples + 5):
                        frame = torch.from_numpy(images[i % len(images)])
                        arguments = (preprocess(frame, torch.zeros(3), device="cpu"),)
                        start = time.perf_counter_ns()
                        wake.clear()
                        future = backend.submit(task, arguments)
                        while not future.done():
                            wake.wait(.01 if device == "cpu" else .00005)
                        output, _ = future.result()
                        observed = time.perf_counter_ns() - start
                        if not torch.isfinite(output).all():
                            raise AssertionError("nonfinite calibration output")
                        if i >= 5:
                            values.append(observed)
                    rows.append({"task": name, "device": device, "trial": trial,
                                 "samples_ns": values, "summary": summarize(values)})
    finally:
        try:
            cpu.close()
        finally:
            gpu.close()
    costs = {}
    policies = {}
    for name, task in graph.tasks.items():
        if len(task.operation.devices) == 1:
            continue
        costs[name] = {device: round(statistics.median(
            row["summary"]["median_ns"] for row in rows if row["task"] == name and row["device"] == device))
                       for device in ("cpu", "gpu")}
        policies[name] = MeasuredCrossoverPolicy([
            PlacementSample(task.operation.work_items, costs[name]["cpu"], costs[name]["gpu"])],
            switch_margin=.05, interpolation="linear")
    return {"rows": rows, "costs_ns": costs,
            "scope": "isolated task service including host dispatch, copies, and observed completion; exact resolution only",
            "input": frame_identity(images, source="held-out calibration subset")}, policies


class BackgroundLoad(AbstractContextManager):
    """Explicit controllable competing work, never silently mixed into baselines."""
    def __init__(self, cpu_workers=0, gpu=False):
        self.cpu_workers, self.gpu = cpu_workers, gpu
        self.stop = threading.Event()
        self.threads, self.errors, self.counts = [], [], []

    def __enter__(self):
        kinds = ["cpu"] * self.cpu_workers + (["gpu"] if self.gpu else [])
        for kind in kinds:
            ready = threading.Event()
            index = len(self.counts)
            self.counts.append(0)
            def work(kind=kind, ready=ready, index=index):
                try:
                    torch.set_num_threads(1)
                    if kind == "gpu":
                        stream = torch.cuda.Stream()
                        with torch.cuda.stream(stream):
                            a = torch.ones((2048, 2048), device="cuda")
                            b = torch.ones_like(a)
                            output = torch.empty_like(a)
                        stream.synchronize()
                    else:
                        a = torch.ones((512, 512))
                        b, output = torch.ones_like(a), torch.empty_like(a)
                    ready.set()
                    while not self.stop.is_set():
                        if kind == "gpu":
                            with torch.cuda.stream(stream):
                                torch.mm(a, b, out=output)
                            stream.synchronize()
                            self.stop.wait(.001)  # Explicit 1 ms pause per GPU pulse.
                        else:
                            torch.mm(a, b, out=output)
                        self.counts[index] += 1
                except BaseException as error:
                    self.errors.append(error)
                    ready.set()
            thread = threading.Thread(target=work, name=f"background-{kind}", daemon=True)
            self.threads.append(thread)
            thread.start()
            if not ready.wait(30):
                self.__exit__(None, None, None)
                raise TimeoutError("background load setup timed out")
        if self.errors:
            self.__exit__(None, None, None)
        return self

    def __exit__(self, *args):
        self.stop.set()
        for thread in self.threads:
            thread.join()
        if self.errors:
            raise RuntimeError("background load failed") from self.errors[0]

    def metadata(self):
        return {"cpu_workers": self.cpu_workers, "cpu_matrix": [512, 512],
                "gpu_enabled": self.gpu, "gpu_matrix": [2048, 2048],
                "gpu_pause_ms": 1, "completed_pulses": list(self.counts)}


def measurements(result, *, warmup, lanes, deadline_ns):
    rows = [r for r in result["frames"] if r["frame"] >= warmup]
    def latency(row):
        return row["end_ns"] - row["release_ns"]
    start = min(r["release_ns"] for r in rows)
    end = max(r["end_ns"] for r in rows)
    return {"latency": summarize([latency(r) for r in rows]),
            "active_latency": summarize([r["end_ns"] - r["admitted_ns"] for r in rows]),
            "arrival_lateness": summarize([max(0, r["admitted_ns"] - r["release_ns"]) for r in rows]),
            "deadline_ns": deadline_ns, "deadline_misses": sum(latency(r) > deadline_ns for r in rows),
            "throughput_frames_per_second": len(rows) * 1e9 / max(1, end - start),
            "cpu_core_equivalents_including_warmup_and_background": result["process_cpu_ns"] / max(1, result["wall_ns"]),
            "per_lane": [{"lane": lane, "latency": summarize([latency(r) for r in rows if r["lane"] == lane])}
                         for lane in range(lanes)],
            "task_service": {name: summarize([r["completed_ns"] - r["submitted_ns"] for r in result["tasks"]
                                               if r["task"] == name and r["frame"] >= warmup])
                             for name in ("preprocess", "lane", "obstacle", "control")},
            "placement_counts": {name: {device: sum(t["task"] == name and t["device"] == device
                                                     and t["frame"] >= warmup for t in result["tasks"])
                                         for device in ("cpu", "gpu")} for name in ("lane", "obstacle")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="NPZ with uint8 frames [N,H,W,3]; otherwise generated fixture")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu-slots", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", default="cpu,serial-gpu,dag-gpu,adaptive")
    parser.add_argument("--period-ms", type=float, default=0)
    parser.add_argument("--deadline-ms", type=float, default=10)
    parser.add_argument("--timeout-s", type=float, default=180)
    parser.add_argument("--calibration-samples", type=int, default=20)
    parser.add_argument("--cpu-load-workers", type=int, default=0)
    parser.add_argument("--gpu-load", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path")
    if args.input is None and (min(args.height, args.width) < 16 or
                              24 * args.height * args.width * 3 > 512 * 1024 * 1024):
        parser.error("generated fixture dimensions must be >=16 and fit 512 MiB")
    if any(n <= 0 for n in (args.iterations, args.repeats, args.lanes, args.workers,
                             args.gpu_slots, args.calibration_samples)) or args.warmup < 0:
        parser.error("counts must be positive and warmup nonnegative")
    if args.lanes > 32 or args.workers > 32 or args.gpu_slots > 64 or args.cpu_load_workers not in range(17):
        parser.error("resource count exceeds experiment bounds")
    if args.lanes * (args.iterations + args.warmup) * args.repeats > 200_000:
        parser.error("too many frame records")
    if (not math.isfinite(args.period_ms) or args.period_ms < 0 or
        not math.isfinite(args.deadline_ms) or args.deadline_ms <= 0 or
        not math.isfinite(args.timeout_s) or args.timeout_s <= 0):
        parser.error("invalid arrival period, deadline, or timeout")
    methods = args.methods.split(",")
    if not set(methods) <= {"cpu", "serial-gpu", "dag-gpu", "adaptive"} or len(set(methods)) != len(methods):
        parser.error("unknown or repeated method")
    needs_gpu = any(method != "cpu" for method in methods) or args.gpu_load
    if needs_gpu and not torch.cuda.is_available():
        parser.error("real CUDA required for GPU methods/load")
    torch.set_num_threads(1)  # Avoid nested Torch pools dominating CPU comparisons.
    images = load_frames(args.input) if args.input else generated_frames(height=args.height, width=args.width)
    if len(images) < 4:
        parser.error("need >=4 distinct frames to separate calibration and evaluation")
    split = max(1, len(images) // 3)
    calibration_images, evaluation_images = images[:split], images[split:]
    graph = camera_graph(*images.shape[1:3])
    result = {"schema_version": 1, "experiment": "python_camera_control_task_graph",
              "created_at": datetime.now(timezone.utc).isoformat(), "environment": environment(),
              "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "scope": "ordinary Torch CPU/CUDA task graph; original VDCores VM NOT executed",
              "workload": "classical RGB lane/obstacle processing and stateful control; not learned inference or physical robot",
              "input": frame_identity(images, source="recorded NPZ" if args.input else "generated deterministic scenes, seed 7"),
              "evaluation_input": frame_identity(evaluation_images, source="held-out evaluation subset"),
              "arrival_mode": "periodic lossless backlog" if args.period_ms else "closed-loop saturation",
              "runs": []}
    policies = {}
    if "adaptive" in methods:
        result["calibration"], policies = calibrate(graph, calibration_images,
            samples=args.calibration_samples, workers=args.workers, gpu_slots=args.gpu_slots)
    # Warm up CPU operators/CUDA libraries separately before the first timed run.
    for method in methods:
        execute_graph(graph, input_factory(calibration_images), frames=3, lanes=1,
                      workers=args.workers, initial=torch.zeros(3), placement="cpu" if method == "cpu" else "gpu",
                      gpu_backend=TorchCudaTaskBackend(args.gpu_slots) if method != "cpu" else None)
    for repeat in range(args.repeats):
        order = methods[repeat % len(methods):] + methods[:repeat % len(methods)]
        if repeat % 2:
            order = list(reversed(order))
        for method in order:
            gpu = TorchCudaTaskBackend(args.gpu_slots) if method != "cpu" else None
            with BackgroundLoad(args.cpu_load_workers, args.gpu_load) as load:
                run = execute_graph(graph, input_factory(evaluation_images), frames=args.iterations + args.warmup,
                                    lanes=args.lanes, workers=args.workers, initial=torch.zeros(3),
                                    placement="cpu" if method == "cpu" else "adaptive" if method == "adaptive" else "gpu",
                                    policies=policies, gpu_backend=gpu, serial=method == "serial-gpu",
                                    period_ns=round(args.period_ms * 1e6), timeout_s=args.timeout_s)
            run["background_load"] = load.metadata()
            run.update(method=method, repeat=repeat)
            run["correctness"] = verify_camera(run, evaluation_images,
                frames=args.iterations + args.warmup, lanes=args.lanes)
            run["measurements"] = measurements(run, warmup=args.warmup, lanes=args.lanes,
                                               deadline_ns=round(args.deadline_ms * 1e6))
            for row in run["frames"]:
                row["output"] = row["output"].tolist()
            result["runs"].append(run)
            stats = run["measurements"]
            print(f'{method} trial={repeat} median={stats["latency"]["median_ns"]/1e6:.3f}ms '
                  f'p99={stats["latency"]["p99_ns"]/1e6:.3f}ms '
                  f'fps={stats["throughput_frames_per_second"]:.1f}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "xt", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    main()
