import argparse
import json
from pathlib import Path


SHRINK_ALPHA = 0.9
EXPAND_ALPHA = 0.5
NUM_SMS_DEFAULT = 128


def infer_op_count(entry):
    token_count = int(entry.get("token_count", 0))
    num_sms = int(entry.get("num_sms", 1))
    stage_name = entry.get("stage_name", "")

    if token_count == 8:
        return 1
    if token_count == 64 and stage_name == "shrink":
        return 1
    if token_count == 64 and stage_name == "expand":
        return max(1, (64 + num_sms - 1) // num_sms)
    return 1


def expand_plan_segments(plan):
    if not plan:
        return []
    if "start_time" in plan[0] and "end_time" in plan[0]:
        return [
            {
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": entry["start_time"],
                "end_time": entry["end_time"],
            }
            for entry in plan
            for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"])
        ]

    segments = []
    sm_available = [0 for _ in range(NUM_SMS_DEFAULT)]
    shrink_finish = 0
    for entry in [entry for entry in plan if entry["stage_name"] == "shrink"]:
        for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
            start = sm_available[sm_id]
            end = start + entry["proxy_latency"]
            segments.append({
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": start,
                "end_time": end,
            })
            sm_available[sm_id] = end
            shrink_finish = max(shrink_finish, end)
    for entry in [entry for entry in plan if entry["stage_name"] == "expand"]:
        for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
            start = max(shrink_finish, sm_available[sm_id])
            end = start + entry["proxy_latency"]
            segments.append({
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": start,
                "end_time": end,
            })
            sm_available[sm_id] = end
    return segments


def reconstruct_measured_segments(payload):
    plan = payload.get("plan", [])
    durations_per_sm = payload.get("durations_per_sm_ns", [])
    num_sms = int(payload.get("num_sms", NUM_SMS_DEFAULT))
    if not plan or not durations_per_sm:
        return []

    sm_offsets = [0 for _ in range(num_sms)]
    sm_available = [0 for _ in range(num_sms)]
    group_shrink_finish = {}
    segments = []

    has_start_time = bool(plan and "start_time" in plan[0])
    ordered_plan = sorted(plan, key=lambda entry: (entry.get("start_time", 0), entry["group_id"], entry["stage_name"]))

    if not has_start_time:
        global_shrink_finish = 0
        for entry in [entry for entry in ordered_plan if entry["stage_name"] == "shrink"]:
            record_end_times = []
            for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
                op_count = infer_op_count(entry)
                sm_durations = durations_per_sm[sm_id]
                next_offset = min(sm_offsets[sm_id] + op_count, len(sm_durations))
                measured = sm_durations[sm_offsets[sm_id]:next_offset]
                sm_offsets[sm_id] = next_offset
                if not measured:
                    continue
                start = sm_available[sm_id]
                end = start + sum(measured)
                segments.append({"group_id": entry["group_id"], "stage_name": entry["stage_name"], "sm_id": sm_id, "start_time": start, "end_time": end})
                sm_available[sm_id] = end
                record_end_times.append(end)
            if record_end_times:
                global_shrink_finish = max(global_shrink_finish, max(record_end_times))
        for entry in [entry for entry in ordered_plan if entry["stage_name"] == "expand"]:
            for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
                op_count = infer_op_count(entry)
                sm_durations = durations_per_sm[sm_id]
                next_offset = min(sm_offsets[sm_id] + op_count, len(sm_durations))
                measured = sm_durations[sm_offsets[sm_id]:next_offset]
                sm_offsets[sm_id] = next_offset
                if not measured:
                    continue
                start = max(global_shrink_finish, sm_available[sm_id])
                end = start + sum(measured)
                segments.append({"group_id": entry["group_id"], "stage_name": entry["stage_name"], "sm_id": sm_id, "start_time": start, "end_time": end})
                sm_available[sm_id] = end
        return segments

    for entry in ordered_plan:
        pred_ready = 0
        if entry["stage_name"] == "expand":
            pred_ready = group_shrink_finish.get(entry["group_id"], 0)
        record_end_times = []
        for sm_id in range(entry["base_sm"], entry["base_sm"] + entry["num_sms"]):
            op_count = infer_op_count(entry)
            sm_durations = durations_per_sm[sm_id]
            next_offset = min(sm_offsets[sm_id] + op_count, len(sm_durations))
            measured = sm_durations[sm_offsets[sm_id]:next_offset]
            sm_offsets[sm_id] = next_offset
            if not measured:
                continue
            start = max(pred_ready, sm_available[sm_id])
            end = start + sum(measured)
            segments.append({
                "group_id": entry["group_id"],
                "stage_name": entry["stage_name"],
                "sm_id": sm_id,
                "start_time": start,
                "end_time": end,
            })
            sm_available[sm_id] = end
            record_end_times.append(end)
        if entry["stage_name"] == "shrink" and record_end_times:
            group_shrink_finish[entry["group_id"]] = max(group_shrink_finish.get(entry["group_id"], 0), max(record_end_times))
    return segments


def load_replay(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"group_sizes", "num_sms"}
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Replay {path} missing fields: {sorted(missing)}")
    if "predicted_segments" not in payload:
        if "plan" not in payload:
            raise ValueError(f"Replay {path} missing fields: ['predicted_segments', 'plan']")
        payload["predicted_segments"] = expand_plan_segments(payload["plan"])
        print(f"Replay {path} missing predicted_segments, reconstructed them from plan")
    if "measured_segments" not in payload:
        payload["measured_segments"] = reconstruct_measured_segments(payload)
        print(f"Replay {path} missing measured_segments, reconstructed them best-effort from plan and durations")
    return payload


def workload_text(group_sizes):
    if group_sizes == [128] + [64] * 4 + [8] * 11:
        return "Workload: g0=128, g1-g4=64, g5-g15=8"
    return f"Group sizes: {group_sizes}"


def draw_schedule_axis(ax, entries, num_sms, title, x_label, y_label=True, label_entries=True, height_getter=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cmap = plt.get_cmap("tab20")
    height_getter = height_getter or (lambda entry: entry["num_sms"] if "num_sms" in entry else 1.0)
    for entry in entries:
        y_base = entry["base_sm"] if "base_sm" in entry else entry["sm_id"]
        height = height_getter(entry)
        rect = Rectangle(
            (entry["start_time"], y_base),
            entry["end_time"] - entry["start_time"],
            height,
            facecolor=cmap(entry["group_id"] % cmap.N),
            edgecolor="black",
            linewidth=0.8 if "base_sm" in entry else 0.4,
            alpha=SHRINK_ALPHA if entry["stage_name"] == "shrink" else EXPAND_ALPHA,
        )
        ax.add_patch(rect)
        if label_entries and (entry["end_time"] - entry["start_time"]) > 0:
            ax.text(
                entry["start_time"] + (entry["end_time"] - entry["start_time"]) / 2,
                y_base + height / 2,
                f"g{entry['group_id']} {entry['stage_name'][0].upper()}",
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )

    ax.set_xlim(0, max(entry["end_time"] for entry in entries) * 1.02 if entries else 1)
    ax.set_ylim(0, num_sms)
    ax.set_xlabel(x_label, fontsize=14)
    ax.tick_params(axis='both', labelsize=14)
    if y_label:
        ax.set_ylabel("SM ID", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description="Plot LoRA scheduler replay comparison figure")
    parser.add_argument(
        "--replay",
        type=str,
        default="build/plots/lora_fixed_rank_schedule_replay.json",
        help="Path to scheduler replay JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="build/plots/lora_replay_comparison.png",
        help="Output image path",
    )
    args = parser.parse_args()

    replay = load_replay(args.replay)
    predicted_entries = replay.get("plan", replay["predicted_segments"])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, cannot plot replay comparison")
        return

    fig, axes = plt.subplots(1, 2, figsize=(22, 7), sharey=True)
    draw_schedule_axis(
        axes[0],
        predicted_entries,
        replay["num_sms"],
        "Predicted Schedule",
        "Proxy Time Unit",
        y_label=True,
        label_entries=True,
    )
    draw_schedule_axis(
        axes[1],
        replay["measured_segments"],
        replay["num_sms"],
        "Execution Timeline",
        "Measured Operation Duration (ns)",
        y_label=False,
        label_entries=False,
        height_getter=lambda _entry: 1.0,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved replay comparison figure to {output_path}")


if __name__ == "__main__":
    main()
