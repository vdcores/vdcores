"""Choose and run one measured CPU/GPU placement for the attention chain."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
APP_DIR = REPO_ROOT / "app" / "python"
TARGET = APP_DIR / "attention_split_kv.py"
sys.path.insert(0, str(PYTHON_DIR))

from dae.heterogeneous import HeterogeneousScheduler, OperationSpec  # noqa: E402
from dae.placement import MeasuredCrossoverPolicy  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--work-items", type=int, required=True)
    parser.add_argument("--cpu-queue-us", type=float, default=0.0)
    parser.add_argument("--gpu-queue-us", type=float, default=0.0)
    parser.add_argument("--deadline-us", type=float)
    parser.add_argument("--switch-margin", type=float, default=0.05)
    parser.add_argument("--preferred-device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--allow-outside-range", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timeout-ms", type=int, default=30000)
    parser.add_argument("--mean-error-limit", type=float, default=0.05)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="choose a placement without importing CUDA or executing the chain",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def microseconds_to_nanoseconds(value: float | None, label: str) -> int | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"{label} cannot be negative")
    return round(value * 1000.0)


def load_policy(args: argparse.Namespace) -> tuple[MeasuredCrossoverPolicy, dict]:
    result = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = MeasuredCrossoverPolicy.from_benchmark(
        result,
        preferred_device=args.preferred_device,
        switch_margin=args.switch_margin,
    )
    return policy, result


def plan_chain(
    policy: MeasuredCrossoverPolicy,
    *,
    work_items: int,
    cpu_queue_ns: int,
    gpu_queue_ns: int,
    deadline_ns: int | None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    scheduler = HeterogeneousScheduler(placement_policy=policy)
    scheduler.add(OperationSpec("attention", 32768, devices=("gpu",)))
    scheduler.add(
        OperationSpec(
            "flex_dot",
            work_items,
            dependencies=("attention",),
            deadline_ns=deadline_ns,
        )
    )
    scheduler.add(
        OperationSpec(
            "gpu_successor",
            1,
            dependencies=("flex_dot",),
            devices=("gpu",),
        )
    )
    scheduler.seal()

    records = []
    attention = scheduler.dispatch("attention")
    records.append(asdict(attention))
    scheduler.complete("attention")

    middle = scheduler.dispatch(
        "flex_dot", cpu_queue_ns=cpu_queue_ns, gpu_queue_ns=gpu_queue_ns
    )
    records.append(asdict(middle))
    scheduler.complete("flex_dot")

    successor = scheduler.dispatch("gpu_successor")
    records.append(asdict(successor))
    scheduler.complete("gpu_successor")
    if not scheduler.is_complete:
        raise AssertionError("adaptive attention plan did not complete")
    if middle.placement is None:
        raise AssertionError("flexible operation did not record a placement decision")
    return records, asdict(middle.placement)


def write_or_print(result: dict[str, object], args: argparse.Namespace) -> None:
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    if args.json or args.plan_only:
        print(encoded)
    else:
        decision = result["decision"]
        print(
            f"Selected {decision['device']} for {decision['work_items']} work items: "
            f"{decision['reason']}"
        )
        measured = result["measured"]
        print(
            "Measured selected-path end-to-end median: "
            f"{measured['end_to_end_ns']['median_ns'] / 1e6:.3f} ms"
        )
        if args.output is not None:
            print(f"Wrote {args.output}")


def run_selected_path(
    args: argparse.Namespace, selected_device: str
) -> dict[str, object]:
    # Keep CUDA imports out of plan-only mode so placement logic remains
    # testable on machines without a built VDCores extension.
    import torch

    from dae import handoff_runtime

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import check_attention_placement as benchmark

    capabilities = dict(handoff_runtime.handoff_capabilities())
    if not capabilities["uses_host_page_tables"]:
        raise RuntimeError(
            "adaptive execution requires a hardware-coherent GH200/GB200 system"
        )

    sys.path.insert(0, str(APP_DIR))
    sys.argv = [str(TARGET)]
    state = runpy.run_path(str(TARGET), run_name="__main__")
    dae = state["dae"]
    counter = torch.zeros(1, dtype=torch.int32, device="cpu")

    if selected_device == "cpu":
        cpu_input = torch.arange(
            args.work_items, dtype=torch.float64, device="cpu"
        )

        def run_once():
            return benchmark.run_cpu_once(
                dae, counter, cpu_input, args.timeout_ms
            )

    else:
        gpu_input = torch.arange(
            args.work_items, dtype=torch.float64, device="cuda"
        )
        gpu_output = torch.empty((), dtype=torch.float64, device="cuda")
        torch.cuda.synchronize()

        def run_once():
            return benchmark.run_gpu_once(
                dae, counter, gpu_input, gpu_output, args.timeout_ms
            )

    for _ in range(args.warmup):
        run_once()
    measurements = [run_once() for _ in range(args.iterations)]
    return {
        "device": capabilities["device_name"],
        "capabilities": capabilities,
        "split_kv_normalized_mean_error": benchmark.split_kv_error(
            state, args.mean_error_limit
        ),
        "measured": benchmark.aggregate(measurements),
    }


def main() -> None:
    args = parse_args()
    if args.work_items <= 0:
        raise ValueError("--work-items must be positive")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative")
    if args.timeout_ms <= 0:
        raise ValueError("--timeout-ms must be positive")

    cpu_queue_ns = microseconds_to_nanoseconds(args.cpu_queue_us, "CPU queue")
    gpu_queue_ns = microseconds_to_nanoseconds(args.gpu_queue_us, "GPU queue")
    deadline_ns = microseconds_to_nanoseconds(args.deadline_us, "deadline")
    if deadline_ns == 0:
        raise ValueError("--deadline-us must be positive")

    policy, calibration = load_policy(args)
    plan, decision = plan_chain(
        policy,
        work_items=args.work_items,
        cpu_queue_ns=cpu_queue_ns,
        gpu_queue_ns=gpu_queue_ns,
        deadline_ns=deadline_ns,
    )
    if decision["clamped_to_measured_range"] and not args.allow_outside_range:
        low, high = policy.measured_range
        raise ValueError(
            f"work size {args.work_items} is outside measured range [{low}, {high}]; "
            "pass --allow-outside-range to use the nearest endpoint"
        )

    result: dict[str, object] = {
        "schema_version": 1,
        "experiment": "vdcores_adaptive_attention_handoff",
        "mode": "plan_only" if args.plan_only else "execute",
        "policy_source": str(args.policy),
        "calibration_experiment": calibration.get("experiment"),
        "plan": plan,
        "decision": decision,
    }
    if not args.plan_only:
        result.update(run_selected_path(args, str(decision["device"])))
    write_or_print(result, args)


if __name__ == "__main__":
    main()
