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

## Llama3 Runtime Interaction

- `app/python/llama3/sched.py` programs the runtime by composing `Schedule` objects, then placing each schedule onto an SM range with `.place(num_sms, base_sm=...)`.
- `python/dae/schedule.py` maps placed SM ids to logical workers; each schedule's `schedule(sm)` method returns the compute instruction plus the memory instructions for that worker.
- `SchedGemv` partitions work over output `M` tiles first and folds over `K`. After placement, `sm_per_fold = M / TileM` and `k_per_fold = K / fold`, so one SM computes one output tile `m` and one `K` fold. With fold greater than `1`, the output path must use a reduce-mode store.
- `SchedAttentionDecoding` uses one SM per `(request, kv_head)` pair. Each worker loads one Q tile, streams all KV blocks for that head, and stores one O tile.
- Memory movement in the Llama3 path uses three patterns:
- 2D/3D TMA load-store or reduce descriptors for normal tensor traffic such as GEMV weights, activations, KV cache, and residual accumulation.
- raw-address operands for kernels that consume or produce GMEM pointers directly, such as argmax and the raw-address SiLU path in the generic scheduler library.
- register handoff through `RegStore` and `RegLoad` when the next op can consume the producer output without writing that intermediate to GMEM first.
- In the current Llama3 decode path, Q and K use register handoff into a separate rope stage before the rotated result is reduced into GMEM, while V writes straight to KV cache. The fused MLP tail similarly keeps gate and up outputs in registers until `SchedRegSiLUFused` writes the final SiLU result to GMEM.
- `LoopM.toNext(..., resource_group=layerg)` and `LoopC.toNext(...)` advance memory and compute instruction streams across the repeated `layer` resource group, so a single scheduled layer body iterates through all transformer layers with shifted TMA and barrier ids.

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
