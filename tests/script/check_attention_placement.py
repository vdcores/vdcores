"""Measure the CPU/GPU crossover for a post-attention operation.

Both placements execute the same float64 dot product after VDCores split-KV
attention and before a tiny prequeued GPU successor. The CPU path crosses the
coherent counter in both directions; the GPU path stays on the queued CUDA
stream. The resulting JSON can be loaded by ``MeasuredCrossoverPolicy``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import runpy
import statistics
import sys
import time

import torch

from dae import handoff_runtime
from dae.placement import MeasuredCrossoverPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app" / "python"
TARGET = APP_DIR / "attention_split_kv.py"
DEFAULT_WORK_ITEMS = (64, 256, 1024, 4096, 16384, 65536)


def parse_work_items(value: str) -> tuple[int, ...]:
    try:
        items = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "work sizes must be comma-separated integers"
        ) from error
    if not items or any(item <= 0 for item in items):
        raise argparse.ArgumentTypeError("all work sizes must be positive")
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("work sizes must be unique")
    return tuple(sorted(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-items",
        type=parse_work_items,
        default=DEFAULT_WORK_ITEMS,
        help="comma-separated dot-product sizes",
    )
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--mean-error-limit", type=float, default=0.05)
    parser.add_argument(
        "--switch-margin",
        type=float,
        default=0.05,
        help="minimum predicted improvement required to switch away from CPU",
    )
    parser.add_argument("--output", type=Path)
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


def expected_dot(work_items: int) -> float:
    return float((work_items - 1) * work_items * (2 * work_items - 1) // 6)


def check_result(label: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-6):
        raise AssertionError(f"{label} dot product was {actual}, expected {expected}")


def run_cpu_once(
    dae,
    counter: torch.Tensor,
    cpu_input: torch.Tensor,
    timeout_ms: int,
) -> dict[str, int]:
    handoff_runtime.cpu_atomic_store(counter, 0)
    start_ns = time.perf_counter_ns()
    stream = dae.launch_async()
    try:
        # Attention completion: 0 -> 1. The final GPU successor is already
        # queued and waits for the CPU placement to acknowledge 1 -> 2.
        handoff_runtime.gpu_atomic_add(counter, 1, stream.cuda_stream)
        handoff_runtime.gpu_atomic_wait_add(
            counter, 2, 1, timeout_ms, stream.cuda_stream
        )

        observed = handoff_runtime.cpu_atomic_wait(counter, 1, timeout_ms)
        attention_observed_ns = time.perf_counter_ns()
        if observed != 1:
            raise AssertionError(f"expected attention signal 1, observed {observed}")

        result = float(torch.dot(cpu_input, cpu_input).item())
        stage_done_ns = time.perf_counter_ns()
        previous = handoff_runtime.cpu_atomic_add(counter, 1)
        if previous != 1:
            raise AssertionError(f"expected counter 1 before CPU ack, got {previous}")

        observed = handoff_runtime.cpu_atomic_wait(counter, 3, timeout_ms)
        end_ns = time.perf_counter_ns()
        if observed != 3:
            raise AssertionError(f"expected successor signal 3, observed {observed}")
    finally:
        dae.synchronize()

    check_result("CPU", result, expected_dot(cpu_input.numel()))
    return {
        "attention_to_cpu_ns": attention_observed_ns - start_ns,
        "stage_ns": stage_done_ns - attention_observed_ns,
        "end_to_end_ns": end_ns - start_ns,
    }


def run_gpu_once(
    dae,
    counter: torch.Tensor,
    gpu_input: torch.Tensor,
    gpu_output: torch.Tensor,
    timeout_ms: int,
) -> dict[str, int]:
    handoff_runtime.cpu_atomic_store(counter, 0)
    start_ns = time.perf_counter_ns()
    stream = dae.launch_async()
    try:
        # Attention completion: 0 -> 1. The candidate GPU operation waits for
        # 1, writes its result, and advances to 2. A queued successor then
        # advances 2 -> 3, matching the CPU path's dependency graph.
        handoff_runtime.gpu_atomic_add(counter, 1, stream.cuda_stream)
        handoff_runtime.gpu_atomic_wait_dot_add(
            counter,
            1,
            gpu_input,
            gpu_output,
            1,
            timeout_ms,
            stream.cuda_stream,
        )
        handoff_runtime.gpu_atomic_wait_add(
            counter, 2, 1, timeout_ms, stream.cuda_stream
        )

        observed = handoff_runtime.cpu_atomic_wait(counter, 3, timeout_ms)
        end_ns = time.perf_counter_ns()
        if observed != 3:
            raise AssertionError(f"expected successor signal 3, observed {observed}")
    finally:
        dae.synchronize()

    check_result("GPU", float(gpu_output.item()), expected_dot(gpu_input.numel()))
    return {"end_to_end_ns": end_ns - start_ns}


def aggregate(measurements: list[dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        key: summarize([measurement[key] for measurement in measurements])
        for key in measurements[0]
    }


def measure_size(
    dae,
    counter: torch.Tensor,
    work_items: int,
    iterations: int,
    warmup: int,
    timeout_ms: int,
) -> dict[str, object]:
    cpu_input = torch.arange(work_items, dtype=torch.float64, device="cpu")
    gpu_input = torch.arange(work_items, dtype=torch.float64, device="cuda")
    gpu_output = torch.empty((), dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()

    for _ in range(warmup):
        run_cpu_once(dae, counter, cpu_input, timeout_ms)
        run_gpu_once(dae, counter, gpu_input, gpu_output, timeout_ms)

    cpu_measurements = []
    gpu_measurements = []
    for iteration in range(iterations):
        if iteration % 2 == 0:
            gpu_measurements.append(
                run_gpu_once(dae, counter, gpu_input, gpu_output, timeout_ms)
            )
            cpu_measurements.append(
                run_cpu_once(dae, counter, cpu_input, timeout_ms)
            )
        else:
            cpu_measurements.append(
                run_cpu_once(dae, counter, cpu_input, timeout_ms)
            )
            gpu_measurements.append(
                run_gpu_once(dae, counter, gpu_input, gpu_output, timeout_ms)
            )

    return {
        "work_items": work_items,
        "cpu": aggregate(cpu_measurements),
        "gpu": aggregate(gpu_measurements),
    }


def split_kv_error(state: dict[str, object], limit: float) -> float:
    _, expected = state["gqa_ref"]()
    actual = state["matO_attn_view"]
    expected = expected.reshape(actual.shape)
    mean_error = (actual.float() - expected.float()).abs().mean().item()
    reference_scale = expected.float().abs().mean().item()
    normalized = mean_error / max(reference_scale, 1.0e-12)
    if not math.isfinite(normalized) or normalized > limit:
        raise AssertionError(
            f"split-KV normalized mean error {normalized:.6f} exceeds {limit:.6f}"
        )
    return normalized


def main() -> None:
    args = parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")
    if not 0.0 <= args.switch_margin < 1.0:
        raise ValueError("--switch-margin must be in [0, 1)")

    capabilities = dict(handoff_runtime.handoff_capabilities())
    if not capabilities["uses_host_page_tables"]:
        raise RuntimeError(
            "the placement experiment requires a coherent GH200/GB200 system"
        )

    sys.path.insert(0, str(APP_DIR))
    sys.argv = [str(TARGET)]
    state = runpy.run_path(str(TARGET), run_name="__main__")
    dae = state["dae"]
    counter = torch.zeros(1, dtype=torch.int32, device="cpu")

    rows = [
        measure_size(
            dae,
            counter,
            work_items,
            args.iterations,
            args.warmup,
            args.timeout_ms,
        )
        for work_items in args.work_items
    ]

    result = {
        "schema_version": 1,
        "experiment": "vdcores_attention_cpu_gpu_placement",
        "device": capabilities["device_name"],
        "iterations_per_placement": args.iterations,
        "warmup_per_placement": args.warmup,
        "capabilities": capabilities,
        "split_kv_normalized_mean_error": split_kv_error(
            state, args.mean_error_limit
        ),
        "rows": rows,
    }
    policy = MeasuredCrossoverPolicy.from_benchmark(
        result, preferred_device="cpu", switch_margin=args.switch_margin
    )
    for row in rows:
        decision = policy.choose(int(row["work_items"]))
        row["selected_device"] = decision.device
        row["selection_reason"] = decision.reason

    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")

    if args.json:
        print(encoded)
        return

    print(f"Device: {result['device']}")
    print(
        "Split-KV normalized mean error: "
        f"{result['split_kv_normalized_mean_error']:.6f}"
    )
    for row in rows:
        cpu_ns = row["cpu"]["end_to_end_ns"]["median_ns"]
        gpu_ns = row["gpu"]["end_to_end_ns"]["median_ns"]
        print(
            f"work_items={row['work_items']}: cpu={cpu_ns / 1e6:.3f} ms "
            f"gpu={gpu_ns / 1e6:.3f} ms -> {row['selected_device']}"
        )
    if args.output is not None:
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
