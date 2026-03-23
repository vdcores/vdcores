import argparse
from dataclasses import dataclass

import torch

from dae.launcher import *
from dae.schedule import SchedGemv
from dae.util import tensor_diff


torch.manual_seed(0)

gpu = torch.device("cuda")
dtype = torch.bfloat16

Atom = Gemv_M64N8
TileM, TileN, TileK = Atom.MNK

HIDDEN = 4096
FEAT_OUT = 4096
LORA_RANK = 64
TOTAL_SMS = 128

SCENARIOS = {
    "uniform": [16, 16, 16, 16],
    "skewed": [40, 16, 8, 8],
}


@dataclass
class GroupPlan:
    adapter_id: int
    token_start: int
    token_count: int
    num_sms: int
    base_sm: int
    split_factor: int
    mode: str


def split_factor_for_sms(num_sms: int) -> int:
    assert 64 % num_sms == 0, f"num_sms={num_sms} must evenly divide the 64 output tiles"
    split_factor = 64 // num_sms
    assert split_factor in (1, 2, 4, 8), f"Unsupported split factor {split_factor} for num_sms={num_sms}"
    return split_factor


def baseline_allocations(num_groups: int) -> list[int]:
    assert TOTAL_SMS % num_groups == 0, "Baseline equal split requires TOTAL_SMS divisible by number of groups"
    per_group = TOTAL_SMS // num_groups
    split_factor_for_sms(per_group)
    return [per_group] * num_groups


def adaptive_allocations(group_sizes: list[int]) -> list[int]:
    allocations = []
    for size in group_sizes:
        if size >= 32:
            allocations.append(64)
        elif size >= 16:
            allocations.append(32)
        else:
            allocations.append(16)

    assert sum(allocations) == TOTAL_SMS, (
        f"Adaptive allocations must consume {TOTAL_SMS} SMs, got {allocations} -> {sum(allocations)}"
    )
    return allocations


def plan_groups(group_sizes: list[int], schedule_kind: str) -> list[GroupPlan]:
    if schedule_kind == "baseline":
        allocations = baseline_allocations(len(group_sizes))
    elif schedule_kind == "adaptive":
        allocations = adaptive_allocations(group_sizes)
    else:
        raise ValueError(f"Unknown schedule kind: {schedule_kind}")

    plans = []
    token_start = 0
    base_sm = 0
    for adapter_id, (token_count, num_sms) in enumerate(zip(group_sizes, allocations)):
        split_factor = split_factor_for_sms(num_sms)
        mode = "full-M" if split_factor == 1 else f"split-Mx{split_factor}"
        plans.append(
            GroupPlan(
                adapter_id=adapter_id,
                token_start=token_start,
                token_count=token_count,
                num_sms=num_sms,
                base_sm=base_sm,
                split_factor=split_factor,
                mode=mode,
            )
        )
        token_start += token_count
        base_sm += num_sms

    assert base_sm == TOTAL_SMS, f"Expected to place exactly {TOTAL_SMS} SMs, got {base_sm}"
    return plans


def make_delta_weights(num_adapters: int) -> torch.Tensor:
    lora_a = torch.rand(num_adapters, LORA_RANK, HIDDEN, dtype=dtype, device=gpu) - 0.5
    lora_b = torch.rand(num_adapters, FEAT_OUT, LORA_RANK, dtype=dtype, device=gpu) - 0.5

    # Precompose each adapter's AB update so the demo can stay within the current
    # GEMV atoms while still modeling the LoRA-only update path xAB.
    delta = torch.matmul(lora_b.float(), lora_a.float()) / LORA_RANK
    return delta.to(dtype)


def build_reference(mat_x: torch.Tensor, group_sizes: list[int], delta_weights: torch.Tensor) -> torch.Tensor:
    ref_y = torch.zeros(mat_x.shape[0], FEAT_OUT, dtype=dtype, device=gpu)
    token_start = 0
    for adapter_id, token_count in enumerate(group_sizes):
        token_end = token_start + token_count
        ref_y[token_start:token_end] = mat_x[token_start:token_end].float() @ delta_weights[adapter_id].t().float()
        token_start = token_end
    return ref_y.to(dtype)


def make_workload(group_sizes: list[int], seed: int):
    torch.manual_seed(seed)
    delta_weights = make_delta_weights(len(group_sizes))
    total_tokens = sum(group_sizes)
    mat_x = torch.rand(total_tokens, HIDDEN, dtype=dtype, device=gpu) - 0.5
    ref_y = build_reference(mat_x, group_sizes, delta_weights)
    return mat_x, delta_weights, ref_y


def make_group_schedule(
    dae: Launcher,
    mat_x: torch.Tensor,
    mat_y: torch.Tensor,
    delta_weights: torch.Tensor,
    plan: GroupPlan,
):
    x_group = mat_x[plan.token_start:plan.token_start + plan.token_count]
    y_group = mat_y[plan.token_start:plan.token_start + plan.token_count]
    delta = delta_weights[plan.adapter_id]

    load_delta = TmaTensor(dae, delta).wgmma_load(TileM, TileK, Major.K)
    load_x = TmaTensor(dae, x_group).wgmma_load(plan.token_count, TileK * Atom.n_batch, Major.K)
    store_y = TmaTensor(dae, y_group).wgmma_store(plan.token_count, TileM, Major.MN)

    sched = SchedGemv(
        Atom,
        MNK=(FEAT_OUT, plan.token_count, HIDDEN),
        tmas=(load_delta, load_x, store_y),
    )
    if plan.split_factor > 1:
        sched = sched.split_M(plan.split_factor)
    return sched.place(plan.num_sms, base_sm=plan.base_sm)


def build_demo(group_sizes: list[int], schedule_kind: str, mat_x: torch.Tensor, delta_weights: torch.Tensor, ref_y: torch.Tensor):
    mat_y = torch.zeros(mat_x.shape[0], FEAT_OUT, dtype=dtype, device=gpu)
    plans = plan_groups(group_sizes, schedule_kind)

    dae = Launcher(TOTAL_SMS, device=gpu)
    schedules = [make_group_schedule(dae, mat_x, mat_y, delta_weights, plan) for plan in plans]
    dae.i(*schedules, TerminateC(), TerminateM())
    return dae, mat_y, ref_y, plans


def mean_execution_time_ns(dae: Launcher, iterations: int) -> float:
    torch.cuda.synchronize()
    execution_ns = 0.0
    for _ in range(iterations):
        dae.launch()
        torch.cuda.synchronize()
        profile = dae.profile[:, 0:2].cpu()
        execution_ns += (profile[:, 1].max() - profile[:, 0].min()).item()
    return execution_ns / iterations


def print_plan_summary(scenario_name: str, schedule_kind: str, group_sizes: list[int], plans: list[GroupPlan]):
    print(f"\n[{scenario_name}] {schedule_kind}")
    print(f"group sizes: {group_sizes}")
    for plan in plans:
        token_end = plan.token_start + plan.token_count
        print(
            f"  adapter={plan.adapter_id} tokens=[{plan.token_start}:{token_end}) "
            f"sms={plan.num_sms} base_sm={plan.base_sm} mode={plan.mode}"
        )


def run_case(
    scenario_name: str,
    group_sizes: list[int],
    schedule_kind: str,
    iterations: int,
    mat_x: torch.Tensor,
    delta_weights: torch.Tensor,
    ref_y: torch.Tensor,
):
    dae, mat_y, ref_y, plans = build_demo(group_sizes, schedule_kind, mat_x, delta_weights, ref_y)
    print_plan_summary(scenario_name, schedule_kind, group_sizes, plans)
    print(f"avg instructions per SM (compute, memory): {dae.num_insts()}")

    exec_ns = mean_execution_time_ns(dae, iterations)
    exec_us = exec_ns / 1e3
    print(f"average execution time: {exec_us:.3f} us over {iterations} iteration(s)")
    tensor_diff(f"{scenario_name}/{schedule_kind}", ref_y, mat_y)
    return exec_us


def parse_args():
    parser = argparse.ArgumentParser(description="Fixed-rank LoRA schedule demo")
    parser.add_argument(
        "--scenario",
        choices=["uniform", "skewed", "all"],
        default="all",
        help="Which workload mix to run",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "adaptive", "all"],
        default="all",
        help="Which schedule family to run",
    )
    parser.add_argument(
        "-b",
        "--bench",
        type=int,
        default=1,
        help="Number of launch iterations to average",
    )
    return parser.parse_args()


def selected_items(option: str, choices: dict[str, list[int]] | tuple[str, ...]):
    if option == "all":
        if isinstance(choices, dict):
            return list(choices.items())
        return [(item, item) for item in choices]
    if isinstance(choices, dict):
        return [(option, choices[option])]
    return [(option, option)]


def main():
    args = parse_args()
    schedule_modes = ("baseline", "adaptive")
    results = {}

    for scenario_idx, (scenario_name, group_sizes) in enumerate(selected_items(args.scenario, SCENARIOS)):
        mat_x, delta_weights, ref_y = make_workload(group_sizes, seed=1234 + scenario_idx)
        scenario_results = {}
        for _, schedule_kind in selected_items(args.mode, schedule_modes):
            exec_us = run_case(
                scenario_name,
                group_sizes,
                schedule_kind,
                args.bench,
                mat_x,
                delta_weights,
                ref_y,
            )
            scenario_results[schedule_kind] = exec_us
        results[scenario_name] = scenario_results

    if args.mode == "all":
        print("\nSummary:")
        for scenario_name, scenario_results in results.items():
            if "baseline" not in scenario_results or "adaptive" not in scenario_results:
                continue
            baseline = scenario_results["baseline"]
            adaptive = scenario_results["adaptive"]
            speedup = baseline / adaptive if adaptive > 0 else float("inf")
            print(
                f"  {scenario_name}: baseline={baseline:.3f} us, "
                f"adaptive={adaptive:.3f} us, speedup={speedup:.3f}x"
            )


if __name__ == "__main__":
    main()
