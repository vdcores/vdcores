# Qwen3 1.7B Scheduling Notes

- Entry point: `app/python/qwen3_1p7b/sched.py`
- Model target: `Qwen/Qwen3-1.7B`

## Confirmed Geometry

- `hidden_size=2048`
- `intermediate_size=6144`
- `num_hidden_layers=28`
- `num_attention_heads=16`
- `num_key_value_heads=8`
- `head_dim=128`
- `vocab_size=151936`
- `rope_theta=1000000` via `rope_scaling["rope_theta"]`

## Scheduling Choices

- The app keeps the existing Qwen fused decode-attention path because `head_dim=128` matches the current fused runtime path.
- The launch baseline stays at `REQ=8`, `N=8`, `KVBlockSize=64`, `num_sms=128`, `full_sms=132`.
- `python/dae/model.py:tma_gqa_load_q(...)` had to be generalized from the old implicit `HEAD_GROUP_SIZE=4` layout to `q_tile_repeat = 64 // HEAD_GROUP_SIZE`; without that, the 1.7B path described only half of the Q tile and hung after launch.
- Q/K/V placement is reduced for the 2048-wide hidden path:
  - `QProj.place(64)`
  - `KProj.place(32, base_sm=64)`
  - `VProj.place(32, base_sm=96)`
- Once Q/K/V stop covering all `128` compute SMs, they need explicit `bar_pre_attn_rms` load barriers instead of relying on implicit ordering from the wider 8B placement.

## MLP Split

- The 1.7B path replaces the 8B `4096 + 8192` fused-tail schedule with a `4096 + 2048` schedule.
- The `[4096:6144)` slices of `matGateOut` and `matInterm` are zeroed before the reduce-backed high GEMVs.
- Gate and up projections run as:
  - store-backed low half on `[0:4096)`
  - reduce-backed high half on `[4096:6144)`
- SiLU runs once over the full `6144` intermediate width, and down projection runs once over `K=6144`.
- With the current `Gemv_M64N8` fold rules, that single `K=6144` down projection cannot be placed on `128` SMs because the implied `k_per_fold=1536` is not a valid multiple of `1024`; the nearest legal wide placement is `96` SMs.

## Verification Snapshot

- `python tests/script/run_with_launch_timeout.py --post-launch-timeout 60 --post-launch-idle-timeout 20 -- python app/python/qwen3_1p7b/sched.py --correctness` passed on 2026-03-23.
- Fresh-process `python app/python/qwen3_1p7b/sched.py -b 1` measured about `2.02 ms` execution time on the current machine on 2026-03-23.
