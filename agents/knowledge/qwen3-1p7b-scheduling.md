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

## 2026-03-31 Perf Debug Update

- `HF_TOKEN` is optional for this path. `Qwen/Qwen3-1.7B` loads unauthenticated on this machine, and the app now skips the token argument when the env var is unset instead of emitting an empty `Bearer` header.
- `app/python/qwen3_1p7b/sched.py` now supports coarse profiling/debug controls:
  - `--dry-build`
  - `--debug-num-layers`
  - `--debug-stop-after {final_rms,logits,argmax,restore,full}`
- Keep the full path on the original bind/issue ordering. A temporary refactor that routed `full` through the same coarse stage-gating code changed results after layer 0; the kept version only uses the gated path for explicit debug stops.
- For quick build-time introspection without loading model weights, use:

```bash
python app/python/qwen3_1p7b/sched.py --dry-build -w /tmp/qwen3_1p7b.ops
```

- That dry-build currently emits only 9 required compute operators for Qwen 1.7B:
  - `OP_GEMV_WGMMA__M_64__N_8__K_256__BLOAD_4__RESIDUAL_0`
  - `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim`
  - `OP_RMS_NORM_F16_K_2048_SMEM`
  - `OP_SILU_MUL_SHARED_BF16_K_4096_INTER`
  - `OP_LOOPC`
  - `OP_ARGMAX_PARTIAL_bf16_1152_50688_132`
  - `OP_ARGMAX_REDUCE_bf16_1152_132`
  - `OP_TERMINATEC`
  - `OP_COPY`
- The dedicated build file for that exact set is `dae_compute_ops.qwen3_1p7b.fast.build`. Build with:

```bash
PATH=/root/miniconda3/bin:$PATH \
PYTHON=/root/miniconda3/bin/python \
DAE_COMPUTE_OPS_FILE=dae_compute_ops.qwen3_1p7b.fast.build \
make pyext
```

- On 2026-03-31, that specialized build reduced the monolithic runtime kernel substantially:
  - `runtime.o` SASS lines: about `28993 -> 10369`
  - registers: `192 -> 164`
  - shared memory: `15648 -> 14624` bytes
- One same-session 5-iteration stage sweep on the specialized build reported:
  - `final_rms_l1`: about `0.125 ms`
  - `final_rms_l4`: about `0.516 ms`
  - `final_rms_l8`: about `1.053 ms`
  - `final_rms_l28`: about `3.675 ms`
  - `full_l28`: about `3.896 ms`
- Practical conclusion from that sweep:
  - the dominant hotspot is the repeated per-layer body, not an isolated logits tail
  - the current remaining gap to `~1 ms` is therefore larger than a simple logits split or small SM placement tweak
- Full single-token timings on this host were still noisy after the slim build. On 2026-03-31, fresh-process `-b 20` full runs ranged from about `2.29 ms` in a warmed rerun to about `3.90 ms` in the colder stage-breakdown sweep.
- `QWEN1P7B_ENABLE_CACHE_HINTS` is now an opt-in knob rather than a baked-in assumption. Cache/prefetch experiments on this host were too noisy to justify another default change yet.

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
- Additional perf toggle:
  - `QWEN1P7B_ENABLE_CACHE_HINTS`
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
  - the new 2026-03-31 stage sweep shifts suspicion away from the logits tail and toward the repeated layer stack
  - the specialized compute-op build is a clear codegen win and should be the first build to use for future Qwen perf work
  - cache/prefetch changes remained too noisy on this host to promote as a stable default
