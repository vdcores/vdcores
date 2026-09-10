"""Inspect support for a system-scope CPU/GPU atomic handoff."""

from __future__ import annotations

import argparse
import json

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
    runtime.cpu_atomic_store(counter, 0)
    result = {
        "device": capabilities["device_name"],
        "allocation": "pinned" if args.pinned else "system",
        "capabilities": capabilities,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True))
        return

    print(f"Device: {result['device']}")
    print(f"CPU allocation: {result['allocation']}")
    print(f"CUDA handoff capabilities: {capabilities}")


if __name__ == "__main__":
    main()
