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

## Schedule Sweep Notes

- `app/python/qwen3_1p7b/sched.py` now exposes placement and prefetch tuning knobs through environment variables:
  - `QWEN1P7B_QPROJ_SMS`
  - `QWEN1P7B_KPROJ_SMS`
  - `QWEN1P7B_VPROJ_SMS`
  - `QWEN1P7B_OUTPROJ_SMS`
  - `QWEN1P7B_GATE_LOW_SMS`
  - `QWEN1P7B_GATE_HIGH_SMS`
  - `QWEN1P7B_UP_LOW_SMS`
  - `QWEN1P7B_UP_HIGH_SMS`
  - `QWEN1P7B_DOWNPROJ_SMS`
  - `QWEN1P7B_SILU_SMS`
  - `QWEN1P7B_LOGITS_SPLIT_M`
  - `QWEN1P7B_NO_PREFETCH`
- The current default schedule remains the original placement/prefetch configuration; the knobs are for exploration, not a baked-in alternate preset.
- Measured on 2026-03-23 with fresh-process `-b 1` runs:
  - baseline:
    - `N=1`: about `2.018 ms`
    - `N=8`: about `16.58 ms` total, `2.07 ms/token`
  - `QWEN1P7B_DOWNPROJ_SMS=64`:
    - `N=1`: about `2.059 ms`
  - compact placement (`Q=32`, `K=16`, `V=16`, `Out=32`, `GateHigh=32`, `UpHigh=32`, `Down=64`):
    - `N=1`: about `160.32 ms`
    - strong evidence that the current path does not tolerate under-provisioning these stages
  - `QWEN1P7B_NO_PREFETCH=logits`:
    - `N=1`: around `2.01 ms`
    - `N=2`: around `4.09 ms`
    - `N=8`: observed between about `15.77 ms` and `16.44 ms`
    - directionally promising for multi-token, but not stable enough in one-shot measurements to make default yet
  - `QWEN1P7B_NO_PREFETCH=q_proj,k_proj,v_proj,out_proj,gate_low,gate_high,up_low,up_high,down_proj`:
    - `N=1`: about `2.12 ms`
- Current conclusion:
  - no tested legal schedule got close to the `~1 ms` single-token target
  - the best direction found so far is logits-specific no-prefetch tuning, which may help multi-token more than single-token
  - the remaining gap is likely in the logits path and/or larger schedule structure rather than a simple SMS placement flip
