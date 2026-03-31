import json
from pathlib import Path

import torch

from dae.launcher import *
from dae.schedule import SchedGemv
from dae.util import dae_app, read_compute_durations, tensor_diff


torch.manual_seed(0)

gpu = torch.device("cuda")
dtype = torch.bfloat16

HIDDEN = 4096
LORA_RANK = 64
GROUP_SIZES = [128] + [64] * 4 + [8] * 11
NUM_SMS = 128

EXPAND_GEMV = globals().get("Gemv_M64N8K64")
if EXPAND_GEMV is None:
    raise RuntimeError("Gemv_M64N8K64 must be added before running app/python/lora_baseline.py")


def make_group_tensors():
    xs = []
    a_weights = []
    b_weights = []
    shrink_outs = []
    expand_outs = []

    for token_count in GROUP_SIZES:
        xs.append(torch.rand(token_count, HIDDEN, dtype=dtype, device=gpu) - 0.5)
        a_weights.append(torch.rand(LORA_RANK, HIDDEN, dtype=dtype, device=gpu) - 0.5)
        b_weights.append(torch.rand(HIDDEN, LORA_RANK, dtype=dtype, device=gpu) - 0.5)
        shrink_outs.append(torch.zeros(token_count, LORA_RANK, dtype=dtype, device=gpu))
        expand_outs.append(torch.zeros(token_count, HIDDEN, dtype=dtype, device=gpu))

    return xs, a_weights, b_weights, shrink_outs, expand_outs


def build_reference(xs, a_weights, b_weights):
    shrink_refs = []
    expand_refs = []
    for x, a_weight, b_weight in zip(xs, a_weights, b_weights):
        shrink_ref = x.float() @ a_weight.t().float()
        expand_ref = shrink_ref @ b_weight.t().float()
        shrink_refs.append(shrink_ref.to(dtype))
        expand_refs.append(expand_ref.to(dtype))
    return shrink_refs, expand_refs


matX, matA, matB, matShrink, matOut = make_group_tensors()
refShrink, refOut = build_reference(matX, matA, matB)

dae = Launcher(NUM_SMS, device=gpu)
bars = [None] * (len(GROUP_SIZES) + 1)

shrink_insts = []
expand_insts = []
plan_entries = []

SHRINK_ALPHA = 0.9
EXPAND_ALPHA = 0.5


def estimate_schedule_proxy_latency(schedule):
    tile_m, tile_n, _ = schedule.Atom.MNK
    return tile_m * tile_n * schedule.k_per_fold


def count_compute_ops_for_sm(schedule, sm_id):
    return sum(1 for inst in schedule(sm_id) if isinstance(inst, ComputeInstruction))


def append_plan_entry(key, group_id, token_count, stage_name, num_sms, base_sm, schedule):
    plan_entries.append({
        "key": key,
        "group_id": group_id,
        "token_count": token_count,
        "stage_name": stage_name,
        "num_sms": num_sms,
        "base_sm": base_sm,
        "proxy_latency": estimate_schedule_proxy_latency(schedule),
        "schedule": schedule,
    })


def build_predicted_segments(plan):
    segments = []
    sm_available = [0 for _ in range(NUM_SMS)]
    global_shrink_finish = 0

    for entry in [entry for entry in plan if entry["stage_name"] == "shrink"]:
        for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
            duration = entry["proxy_latency"] * max(1, count_compute_ops_for_sm(entry["schedule"], sm_id))
            start = sm_available[sm_id]
            end = start + duration
            segments.append({
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": start,
                "end_time": end,
            })
            sm_available[sm_id] = end
            global_shrink_finish = max(global_shrink_finish, end)

    for entry in [entry for entry in plan if entry["stage_name"] == "expand"]:
        for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
            duration = entry["proxy_latency"] * max(1, count_compute_ops_for_sm(entry["schedule"], sm_id))
            start = max(global_shrink_finish, sm_available[sm_id])
            end = start + duration
            segments.append({
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": start,
                "end_time": end,
            })
            sm_available[sm_id] = end

    return segments


def build_measured_segments(plan, durations_per_sm):
    segments = []
    sm_offsets = [0 for _ in range(NUM_SMS)]
    sm_available = [0 for _ in range(NUM_SMS)]
    global_shrink_finish = 0

    for entry in [entry for entry in plan if entry["stage_name"] == "shrink"]:
        record_end_times = []
        for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
            op_count = max(1, count_compute_ops_for_sm(entry["schedule"], sm_id))
            sm_durations = durations_per_sm[sm_id]
            next_offset = min(sm_offsets[sm_id] + op_count, len(sm_durations))
            measured = sm_durations[sm_offsets[sm_id]:next_offset]
            sm_offsets[sm_id] = next_offset
            if measured.size == 0:
                continue
            start = sm_available[sm_id]
            end = start + int(measured.sum())
            segments.append({
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": start,
                "end_time": end,
            })
            sm_available[sm_id] = end
            record_end_times.append(end)
        if record_end_times:
            global_shrink_finish = max(global_shrink_finish, max(record_end_times))

    for entry in [entry for entry in plan if entry["stage_name"] == "expand"]:
        for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
            op_count = max(1, count_compute_ops_for_sm(entry["schedule"], sm_id))
            sm_durations = durations_per_sm[sm_id]
            next_offset = min(sm_offsets[sm_id] + op_count, len(sm_durations))
            measured = sm_durations[sm_offsets[sm_id]:next_offset]
            sm_offsets[sm_id] = next_offset
            if measured.size == 0:
                continue
            start = max(global_shrink_finish, sm_available[sm_id])
            end = start + int(measured.sum())
            segments.append({
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": start,
                "end_time": end,
            })
            sm_available[sm_id] = end

    return segments


def _draw_schedule_axis(ax, entries, title, x_label, y_label=True):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cmap = plt.get_cmap("tab20")
    for entry in entries:
        rect = Rectangle(
            (entry["start_time"], entry["sm_id"]),
            entry["end_time"] - entry["start_time"],
            1.0,
            facecolor=cmap(entry["group_id"] % cmap.N),
            edgecolor="black",
            linewidth=0.4,
            alpha=SHRINK_ALPHA if entry["stage_name"] == "shrink" else EXPAND_ALPHA,
        )
        ax.add_patch(rect)

    ax.set_xlim(0, max(entry["end_time"] for entry in entries) * 1.02)
    ax.set_ylim(0, NUM_SMS)
    ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel("SM ID")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)


def visualize_schedule_comparison(predicted_segments, measured_segments, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping baseline schedule visualization")
        return

    if not predicted_segments or not measured_segments:
        print("missing predicted or measured baseline schedule data, skipping visualization")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, 7), sharey=True)
    fig.suptitle("LoRA Baseline Schedule Comparison\nWorkload: g0=128, g1-g4=64, g5-g15=8", fontsize=14)
    _draw_schedule_axis(ax_left, predicted_segments, "Predicted Schedule", "Proxy Time", y_label=True)
    _draw_schedule_axis(ax_right, measured_segments, "Measured Schedule", "Measured Compute Time (ns)", y_label=False)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved baseline comparison visualization to {output_path}")


def dump_schedule_replay(plan, durations_per_sm, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "group_sizes": GROUP_SIZES,
        "num_sms": NUM_SMS,
        "plan": [
            {key: value for key, value in entry.items() if key != "schedule"}
            for entry in plan
        ],
        "durations_per_sm_ns": [[int(v) for v in durations.tolist()] for durations in durations_per_sm],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved baseline replay artifact to {output_path}")


def base_shrink_sched():
    shrink_base_sm = 0

    def split_n(base_sm, num_sm, token_count, atom):
        tile_m, tile_n, tile_k = atom.MNK
        insts = []
        loadA = TmaTensor(dae, matA[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        for i in range(token_count // tile_n):
            loadB = TmaTensor(dae, matX[group_id][i * tile_n:(i + 1) * tile_n]).wgmma_load(
                tile_n, tile_k * atom.n_batch, Major.K
            )
            reduceC = TmaTensor(dae, matShrink[group_id][i * tile_n:(i + 1) * tile_n]).wgmma(
                "reduce", tile_n, tile_m, Major.MN
            )

            inst = SchedGemv(
                atom,
                MNK=(LORA_RANK, tile_n, HIDDEN),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sm, base_sm).bar("store", bars[-1])
            insts.append(inst)
            append_plan_entry(f"shrink:g{group_id}:n{i}", group_id, tile_n, "shrink", num_sm, base_sm, inst)
            base_sm = (base_sm + num_sm) % NUM_SMS
        return insts, base_sm

    bar_cnt = sum(token_count for token_count in GROUP_SIZES) // 8
    bars[-1] = dae.new_bar(bar_cnt)
    for group_id, token_count in enumerate(GROUP_SIZES):
        insts, shrink_base_sm = split_n(shrink_base_sm, 1, token_count, Gemv_M64N8)
        shrink_insts.extend(insts)


def base_expand_sched():
    expand_base_sm = 0

    def split_n_m(base_sm, num_sm, token_count, atom):
        tile_m, tile_n, tile_k = atom.MNK
        insts = []
        loadA = TmaTensor(dae, matB[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        for i in range(token_count // tile_n):
            loadB = TmaTensor(dae, matShrink[group_id][i * tile_n:(i + 1) * tile_n]).wgmma_load(
                tile_n, tile_k * atom.n_batch, Major.K
            )
            reduceC = TmaTensor(dae, matOut[group_id][i * tile_n:(i + 1) * tile_n]).wgmma(
                "reduce", tile_n, tile_m, Major.MN
            )

            inst = SchedGemv(
                atom,
                MNK=(HIDDEN, tile_n, LORA_RANK),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sm, base_sm)
            insts.append(inst)
            append_plan_entry(f"expand:g{group_id}:n{i}", group_id, tile_n, "expand", num_sm, base_sm, inst)
            base_sm = (base_sm + num_sm) % NUM_SMS
        return insts, base_sm

    for group_id, token_count in enumerate(GROUP_SIZES):
        num_sms = HIDDEN // 64
        insts, expand_base_sm = split_n_m(expand_base_sm, num_sms, token_count, Gemv_M64N8K64)
        expand_insts.extend(insts)


base_shrink_sched()
base_expand_sched()

dae.i(
    shrink_insts,
    expand_insts,
    TerminateC(),
    TerminateM(),
)

print("LoRA fixed-rank baseline")
print(f"group sizes: {GROUP_SIZES}, sms: {NUM_SMS}")

dae_app(dae)

durations_per_sm = read_compute_durations(dae)
if any(len(durations) > 0 for durations in durations_per_sm):
    predicted_segments = build_predicted_segments(plan_entries)
    measured_segments = build_measured_segments(plan_entries, durations_per_sm)
    visualize_schedule_comparison(
        predicted_segments,
        measured_segments,
        Path("build") / "plots" / "lora_baseline_schedule_comparison.png",
    )
    dump_schedule_replay(
        plan_entries,
        durations_per_sm,
        Path("build") / "plots" / "lora_baseline_schedule_replay.json",
    )

# for group_id, token_count in enumerate(GROUP_SIZES):
#     tensor_diff(f"group{group_id}_shrink_{token_count}", refShrink[group_id], matShrink[group_id])
#     tensor_diff(f"group{group_id}_expand_{token_count}", refOut[group_id], matOut[group_id])
