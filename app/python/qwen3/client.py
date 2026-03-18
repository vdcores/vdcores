from layer import *
from dae.util import *

input_after_embedding = uniform_rand_scaled(NUM_REQ, Q_SEQ_LEN, HID_DIM, dtype=torch.float16, device=gpu, scale=0.1)

model = QwenModel()
refOut = model.reference_forward(input_after_embedding)

schedule, final_output = model.get_schedule_plan(num_sms, input_after_embedding)
print(sorted(Module._bar_id_2_cnt.items()))
dae.i(
    schedule,

    # WriteBarrier(),
    TerminateC(),
    TerminateM(),
)

dump_insts(dae, 0)

print("Launching Qwen3 DAE...")
dae.launch()
profile_data = dae.profile.cpu().numpy()
duration_ns = torch.zeros(num_sms, dtype=torch.uint64)
duration_ns += (profile_data[:,1] - profile_data[:,0])
avg_duration_ns = duration_ns.double().mean()
print(f"Average duration: {avg_duration_ns/1e3:.3f} us over {num_sms} SMs")

bar_of_interest = [bar_id for bar_id, cnt in Module._bar_id_2_cnt.items() if cnt > 0]
profile_idx_of_interest = [bar_id if bar_id < config.max_tmas else (bar_id + 13) % 16 for bar_id in bar_of_interest]
print(profile_idx_of_interest)

bar_ts = profile_data[:,profile_idx_of_interest]
bar_ts = bar_ts[bar_ts > 0] - profile_data[:,0:1]

# from matplotlib import pyplot as plt
# # create len(profile idx of interest) subplots
# fig, axs = plt.subplots(len(profile_idx_of_interest), 1, figsize=(8, 4 * len(profile_idx_of_interest)))
# for i, idx in enumerate(profile_idx_of_interest):
#     axs[i].hist(bar_ts[:, i] / 1e3, bins=50)
#     axs[i].set_title(f"Barrier {bar_of_interest[i]} timing distribution")
#     axs[i].set_xlabel("Time (us)")
#     axs[i].set_ylabel("Frequency")
# plt.tight_layout()
# plt.savefig("barrier_timing_distribution.png")

bar_ts = np.mean(bar_ts, axis=0)

last_ts = 0
for bar_id, ts in zip(bar_of_interest, bar_ts):
    print(f"  Bar {bar_id}: {ts/1e3:.3f} us, dur = {(ts - last_ts)/1e3:.3f} us")
    last_ts = ts

output = final_output
print("DAE Output shape:", output.shape)
print("Reference Output shape:", refOut.shape)

print("DAE output: ", output[:16])
print("Ref output: ", refOut[:16])
tensor_diff("DAE", refOut[:16], output[:16])

dae.bench()