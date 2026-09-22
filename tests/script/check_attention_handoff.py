"""Benchmark a fixed VDCores -> CPU -> GPU dependency chain."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import statistics
import sys
import time
from pathlib import Path

import torch

from dae import handoff_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app" / "python"
TARGET = APP_DIR / "attention_split_kv.py"
WORK_ITEMS = 1024
EXPECTED_DOT = (WORK_ITEMS - 1) * WORK_ITEMS * (2 * WORK_ITEMS - 1) // 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--mean-error-limit", type=float, default=0.05)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: list[int]) -> dict[str, int]:
    return {
        "min_ns": min(values),
        "median_ns": int(statistics.median(values)),
        "p95_ns": percentile(values, 0.95),
        "max_ns": max(values),
    }


def run_blocking_once(dae, cpu_input, gpu_input, gpu_output) -> dict[str, int]:
    start_ns = time.perf_counter_ns()
    dae.launch()
    attention_done_ns = time.perf_counter_ns()

    cpu_result = int(torch.dot(cpu_input, cpu_input).item())
    cpu_done_ns = time.perf_counter_ns()

    torch.dot(gpu_input, gpu_input, out=gpu_output)
    torch.cuda.synchronize()
    end_ns = time.perf_counter_ns()

    if cpu_result != EXPECTED_DOT or int(gpu_output.item()) != EXPECTED_DOT:
        raise AssertionError("blocking baseline produced an incorrect dependent result")

    return {
        "attention_to_cpu_ns": attention_done_ns - start_ns,
        "cpu_task_ns": cpu_done_ns - attention_done_ns,
        "gpu_task_ns": end_ns - cpu_done_ns,
        "end_to_end_ns": end_ns - start_ns,
    }


def run_handoff_once(
    dae,
    counter,
    cpu_input,
    gpu_input,
    gpu_output,
    timeout_ms: int,
) -> dict[str, int]:
    handoff_runtime.cpu_atomic_store(counter, 0)
    start_ns = time.perf_counter_ns()
    stream = dae.launch_async()
    try:
        # 0 -> 1: VDCores attention has completed on this stream.
        handoff_runtime.gpu_atomic_add(counter, 1, stream.cuda_stream)
        # Wait for the CPU's 1 -> 2 acknowledgement, then advance 2 -> 3.
        handoff_runtime.gpu_atomic_wait_add(
            counter, 2, 1, timeout_ms, stream.cuda_stream
        )
        # A real GPU consumer follows the acknowledgement on the same stream.
        with torch.cuda.stream(stream):
            torch.dot(gpu_input, gpu_input, out=gpu_output)
        # 3 -> 4 signals that the GPU consumer itself has completed.
        handoff_runtime.gpu_atomic_add(counter, 1, stream.cuda_stream)

        observed = handoff_runtime.cpu_atomic_wait(counter, 1, timeout_ms)
        attention_observed_ns = time.perf_counter_ns()
        if observed != 1:
            raise AssertionError(f"expected attention signal 1, observed {observed}")

        cpu_result = int(torch.dot(cpu_input, cpu_input).item())
        cpu_done_ns = time.perf_counter_ns()
        previous = handoff_runtime.cpu_atomic_add(counter, 1)
        acknowledged_ns = time.perf_counter_ns()
        if previous != 1:
            raise AssertionError(f"expected counter 1 before CPU acknowledgement, got {previous}")

        observed = handoff_runtime.cpu_atomic_wait(counter, 4, timeout_ms)
        end_ns = time.perf_counter_ns()
        if observed != 4:
            raise AssertionError(f"expected GPU completion signal 4, observed {observed}")
    finally:
        dae.synchronize()

    if cpu_result != EXPECTED_DOT or int(gpu_output.item()) != EXPECTED_DOT:
        raise AssertionError("handoff chain produced an incorrect dependent result")
    if handoff_runtime.cpu_atomic_load(counter) != 4:
        raise AssertionError("handoff counter did not finish at 4")

    return {
        "attention_to_cpu_ns": attention_observed_ns - start_ns,
        "cpu_task_ns": cpu_done_ns - attention_observed_ns,
        "cpu_ack_ns": acknowledged_ns - cpu_done_ns,
        "cpu_to_gpu_completion_ns": end_ns - acknowledged_ns,
        "end_to_end_ns": end_ns - start_ns,
    }


def aggregate(measurements: list[dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        key: summarize([measurement[key] for measurement in measurements])
        for key in measurements[0]
    }


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")

    capabilities = dict(handoff_runtime.handoff_capabilities())
    if not capabilities["uses_host_page_tables"]:
        raise RuntimeError(
            "the attention handoff test requires a hardware-coherent GH200/GB200 system"
        )

    sys.path.insert(0, str(APP_DIR))
    sys.argv = [str(TARGET)]
    state = runpy.run_path(str(TARGET), run_name="__main__")

    dae = state["dae"]
    counter = torch.zeros(1, dtype=torch.int32, device="cpu")
    cpu_input = torch.arange(WORK_ITEMS, dtype=torch.int64, device="cpu")
    gpu_input = torch.arange(WORK_ITEMS, dtype=torch.float64, device="cuda")
    gpu_output = torch.empty((), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        run_blocking_once(dae, cpu_input, gpu_input, gpu_output)
        run_handoff_once(
            dae, counter, cpu_input, gpu_input, gpu_output, args.timeout_ms
        )

    blocking_measurements = []
    handoff_measurements = []
    for iteration in range(args.iterations):
        if iteration % 2 == 0:
            handoff_measurements.append(
                run_handoff_once(
                    dae, counter, cpu_input, gpu_input, gpu_output, args.timeout_ms
                )
            )
            blocking_measurements.append(
                run_blocking_once(dae, cpu_input, gpu_input, gpu_output)
            )
        else:
            blocking_measurements.append(
                run_blocking_once(dae, cpu_input, gpu_input, gpu_output)
            )
            handoff_measurements.append(
                run_handoff_once(
                    dae, counter, cpu_input, gpu_input, gpu_output, args.timeout_ms
                )
            )

    _, expected = state["gqa_ref"]()
    actual = state["matO_attn_view"]
    expected = expected.reshape(actual.shape)
    mean_error = (actual.float() - expected.float()).abs().mean().item()
    reference_scale = expected.float().abs().mean().item()
    normalized_mean_error = mean_error / max(reference_scale, 1.0e-12)
    if (
        not math.isfinite(normalized_mean_error)
        or normalized_mean_error > args.mean_error_limit
    ):
        raise AssertionError(
            f"split-KV normalized mean error {normalized_mean_error:.6f} "
            f"exceeds limit {args.mean_error_limit:.6f}"
        )

    result = {
        "device": capabilities["device_name"],
        "iterations": args.iterations,
        "work_items": WORK_ITEMS,
        "capabilities": capabilities,
        "blocking": aggregate(blocking_measurements),
        "handoff": aggregate(handoff_measurements),
        "split_kv_normalized_mean_error": normalized_mean_error,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
        return

    print(f"Device: {result['device']}")
    print(f"Split-KV normalized mean error: {normalized_mean_error:.6f}")
    print(
        "Blocking end-to-end median (ns): "
        f"{result['blocking']['end_to_end_ns']['median_ns']}"
    )
    print(
        "Handoff end-to-end median (ns): "
        f"{result['handoff']['end_to_end_ns']['median_ns']}"
    )
    print(
        "CPU acknowledgement to GPU-consumer completion median (ns): "
        f"{result['handoff']['cpu_to_gpu_completion_ns']['median_ns']}"
    )


if __name__ == "__main__":
    main()
