# Llama Scheduling Notes

## Llama 3.2 1B Baseline

- The isolated 1B app path lives in `app/python/llama32_1b/sched.py`.
- The current baseline keeps the existing decode width and placement pattern:
  `N=8`, `REQ=8`, `KVBlockSize=64`, `rms_sms=8`, `num_sms=128`, `full_sms=132`.
- The intended 1B geometry is:
  `hidden_size=2048`, `intermediate_size=8192`, `num_layers=16`, `num_attention_heads=32`, `num_key_value_heads=8`, `head_dim=64`.

## MLP Split Rationale

- The 1B baseline keeps a two-phase MLP schedule with a `6144 + 2048` split.
- Phase A computes gate/up for `[0:4096)` and `[4096:6144)`, then runs the shared-memory SiLU stage on `[0:6144)`.
- Phase B computes gate/up for `[6144:8192)` into registers, then runs the fused register-backed SiLU stage for the tail.
- Down projection starts on the `[0:6144)` slice as soon as `bar_silu_out1` is released and finishes after the fused tail reaches `bar_silu_out2`.
- This preserves the existing overlap pattern and avoids adding a new shared-memory SiLU opcode just for the 1B path.

## 1B GEMV Placement Rule

- `Gemv_M64N8` consumes `TileK * n_batch = 256 * 4 = 1024` K elements per repeat, so any placed schedule must keep `k_per_fold >= 1024`.
- For the 1B geometry with `K=2048`, any fold-2 GEMV is the maximum safe fold. Higher folding produces a zero-repeat schedule and should be rejected during `SchedGemv.validate()`.
- The 1B path therefore uses smaller placements for `QProj`, `KProj`, `VProj`, `OutProj`, and the fused MLP tail than the 8B path.
- Any GEMV stage whose output is consumed through `RegStore` and then immediately by `SchedRegSiLUFused` must remain fold-1 on 1B; otherwise the register-backed tail only receives a partial result.

## Shared Python Parameterization

- `python/dae/tma_utils.py` now lets rope-table TMA loading scale with `head_dim`, and `ToRopeTableCordAdapter` now accepts an explicit rope-tile repeat count.
- `python/dae/model.py` now derives GQA Q-load TMA metadata from the tensor shape instead of assuming `head_dim=128` and `num_kv_head=4`.
- `python/dae/schedule.py` now derives RMS per-token byte stride from the scheduled hidden size and routes kernel selection through helper selectors in `python/dae/instructions.py`.

## Deadlock Debugging Lessons

- If a schedule prints `[launch]` and then stalls, treat it as a likely barrier or data-dependency bug before treating it as a kernel crash.
- The most effective narrowing path on the 1B schedule was: one layer first, then one operator boundary at a time using `--debug-num-layers 1` and `--debug-stop-after`.
- Splitting operators across disjoint SM ranges can remove implicit ordering that previously came “for free” when wide stages occupied the same SM set. On the 1B path, `KProj` and `VProj` needed an explicit `bar_pre_attn_rms` load barrier once `QProj` stopped covering all 128 SMs.
- The head-dim-64 rope path also needed a rope-table TMA fix: the loader must still build a full `64 x 8` rope tile even when the model head dimension is only `64`.

## Implemented Low-Level Support

- The runtime now has a dedicated `RMS_NORM_F16_K_2048_SMEM` path for the 1B hidden size.
- `CC0` now carries the embedding row stride as a shift width, so power-of-two row sizes like `4096` bytes (`2048` bf16) and `8192` bytes (`4096` bf16) use the same fast path.
- The runtime now has a dedicated attention decode opcode/instruction path for `head_dim=64`, and the Python attention schedulers now select the decode instruction from `head_dim`.
- The isolated 1B path now verifies end to end against `unsloth/Llama-3.2-1B-Instruct` for single-token correctness.

## Performance Debugging Notes

- Process hygiene matters for this app. Before collecting timings, clear leftover decode jobs with `killall python || true`; stale Python workers can make the benchmark look dramatically worse than the clean baseline.
- The timeout wrapper is useful for separating deadlocks from slow schedules:
  `python tests/script/run_with_launch_timeout.py --post-launch-timeout 20 --post-launch-idle-timeout 10 -- python app/python/llama32_1b/sched.py ...`
- On 2026-03-21, clean sequential benchmark measurements from the current branch were:
  - `N=1`: about `1.22 ms`
  - `N=2`: about `6.42 ms` total, `3.21 ms/token`
  - `N=4`: about `9.94 ms` total, `2.49 ms/token`
  - `N=8`: about `26.37 ms` total, `3.30 ms/token`
  - `N=16`: about `65.45 ms` total, `4.09 ms/token`
- For longer multi-token runs on the current branch, fresh one-shot `-b 1` launches were more trustworthy than repeated `-b 3` averages; repeated launches in one process showed unstable timings and likely need separate reset-path debugging.
- The current one-token path is already close to the target; the larger remaining gap is multi-token scaling.
- The full multi-token path launched successfully under the timeout wrapper for `N=2`, so the current main issue is not a full-path deadlock.
- The partial multi-token debug harness is still incomplete: `--debug-stop-after final_rms` timed out after launch for `N=2` with both `7` and `8` layers, so stage-by-stage timing past that point should not yet be trusted on the multi-token path.

## Multi-Token Compute Looping

- `LoopC` now carries an explicit loop-register index in `python/dae/instructions.py`, matching the runtime `count[4]` loop-counter array in `include/dae/dae2.cuh`.
- The Llama3 decode path in `app/python/llama3/sched.py` uses register `0` for the existing per-layer compute loop and register `1` for the outer repeated-token compute loop.
- The repeated-token refactor keeps per-token memory instructions linear, but emits the repeated compute body once and loops it with `LoopC` after `bar_token_finish`.
- Non-split decode attention in `include/dae/compute_dispatch.cuh` now interprets its encoded KV sequence length as a base value and adds the outer compute loop counter from register `1` at runtime.
