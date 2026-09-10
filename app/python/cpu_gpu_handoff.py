"""Measure a system-scope atomic notification from the GPU to the CPU."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from dae import handoff_runtime as runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--pinned",
        action="store_true",
        help="use CUDA page-locked host memory instead of an ordinary CPU allocation",
    )
    parser.add_argument(
        "--require-hardware-coherence",
        action="store_true",
        help="fail unless CUDA reports host-page-table hardware coherence",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def run_once(counter: torch.Tensor, stream: int, timeout_ms: int) -> tuple[int, int]:
    runtime.cpu_atomic_store(counter, 0)
    start_ns = time.perf_counter_ns()
    runtime.gpu_atomic_add(counter, 1, stream)
    runtime.gpu_atomic_wait_add(counter, 2, 1, timeout_ms, stream)
    observed = runtime.cpu_atomic_wait(counter, 1, timeout_ms)
    gpu_to_cpu_ns = time.perf_counter_ns() - start_ns
    if observed != 1:
        raise RuntimeError(f"expected GPU signal value 1, observed {observed}")

    previous = runtime.cpu_atomic_add(counter, 1)
    if previous != 1:
        raise RuntimeError("CPU atomic acknowledgement failed")
    observed = runtime.cpu_atomic_wait(counter, 3, timeout_ms)
    round_trip_ns = time.perf_counter_ns() - start_ns
    if observed != 3 or runtime.cpu_atomic_load(counter) != 3:
        raise RuntimeError("GPU did not observe the CPU acknowledgement")
    return gpu_to_cpu_ns, round_trip_ns


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required")

    capabilities = dict(runtime.handoff_capabilities())
    if args.require_hardware_coherence and not capabilities["uses_host_page_tables"]:
        raise RuntimeError(
            "this run requires hardware-coherent host page tables, as provided by "
            "Grace Hopper and Grace Blackwell systems"
        )

    counter = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=args.pinned)
    stream = torch.cuda.current_stream().cuda_stream
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        run_once(counter, stream, args.timeout_ms)
    torch.cuda.synchronize()

    measurements = [
        run_once(counter, stream, args.timeout_ms) for _ in range(args.iterations)
    ]
    torch.cuda.synchronize()
    gpu_to_cpu_ns = [measurement[0] for measurement in measurements]
    round_trip_ns = [measurement[1] for measurement in measurements]

    result = {
        "device": capabilities["device_name"],
        "allocation": "pinned" if args.pinned else "system",
        "iterations": args.iterations,
        "gpu_to_cpu": {
            "min_ns": min(gpu_to_cpu_ns),
            "median_ns": int(statistics.median(gpu_to_cpu_ns)),
            "p95_ns": percentile(gpu_to_cpu_ns, 0.95),
            "max_ns": max(gpu_to_cpu_ns),
        },
        "round_trip": {
            "min_ns": min(round_trip_ns),
            "median_ns": int(statistics.median(round_trip_ns)),
            "p95_ns": percentile(round_trip_ns, 0.95),
            "max_ns": max(round_trip_ns),
        },
        "capabilities": capabilities,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
        return

    print(f"Device: {result['device']}")
    print(f"CPU allocation: {result['allocation']}")
    print(f"CUDA handoff capabilities: {capabilities}")
    for label, key in [
        ("GPU-to-CPU notification", "gpu_to_cpu"),
        ("GPU-to-CPU-to-GPU round trip", "round_trip"),
    ]:
        values = result[key]
        print(
            f"{label} latency (ns): min={values['min_ns']} "
            f"median={values['median_ns']} p95={values['p95_ns']} "
            f"max={values['max_ns']}"
        )


if __name__ == "__main__":
    main()
