"""Run split-KV attention and notify the CPU when VDCores completes."""

from __future__ import annotations

import argparse
import math
import runpy
import sys
import time
from pathlib import Path

import torch

from dae import handoff_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app" / "python"
TARGET = APP_DIR / "attention_split_kv.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--mean-error-limit", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    cpu_input = torch.arange(1024, dtype=torch.int64, device="cpu")
    expected_cpu_result = 1023 * 1024 * 2047 // 6
    handoff_runtime.cpu_atomic_store(counter, 0)

    start_ns = time.perf_counter_ns()
    stream = dae.launch_async()
    try:
        handoff_runtime.gpu_atomic_add(counter, 1, stream.cuda_stream)
        observed = handoff_runtime.cpu_atomic_wait(counter, 1, args.timeout_ms)
        elapsed_ns = time.perf_counter_ns() - start_ns

        cpu_start_ns = time.perf_counter_ns()
        cpu_result = int(torch.dot(cpu_input, cpu_input).item())
        cpu_elapsed_ns = time.perf_counter_ns() - cpu_start_ns
        previous = handoff_runtime.cpu_atomic_add(counter, 1)
    finally:
        dae.synchronize()

    if observed != 1:
        raise AssertionError(f"expected completion counter 1, observed {observed}")
    if cpu_result != expected_cpu_result:
        raise AssertionError(
            f"CPU task returned {cpu_result}, expected {expected_cpu_result}"
        )
    if previous != 1 or handoff_runtime.cpu_atomic_load(counter) != 2:
        raise AssertionError("CPU completion acknowledgement failed")

    _, expected = state["gqa_ref"]()
    actual = state["matO_attn_view"]
    expected = expected.reshape(actual.shape)
    mean_error = (actual.float() - expected.float()).abs().mean().item()
    reference_scale = expected.float().abs().mean().item()
    normalized_mean_error = mean_error / max(reference_scale, 1.0e-12)

    if not math.isfinite(normalized_mean_error) or normalized_mean_error > args.mean_error_limit:
        raise AssertionError(
            f"split-KV normalized mean error {normalized_mean_error:.6f} "
            f"exceeds limit {args.mean_error_limit:.6f}"
        )

    print(f"Device: {capabilities['device_name']}")
    print(f"VDCores completion observed by CPU after {elapsed_ns} ns")
    print(f"Dependent CPU task completed in {cpu_elapsed_ns} ns")
    print(f"Split-KV normalized mean error: {normalized_mean_error:.6f}")


if __name__ == "__main__":
    main()
