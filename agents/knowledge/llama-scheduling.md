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

## 2026-03-31 1B One-Token Debug

- On this branch, a fresh rebuild with the selected compute-ops file mattered before profiling:
  `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base && DAE_COMPUTE_OPS_FILE=dae_compute_ops.vdcore.build make pyext`
- A useful one-token narrowing path was:
  `-N1 --debug-num-layers 1 --debug-stop-after down_high`
  vs.
  `-N1 --debug-stop-after final_rms`
  vs.
  `-N1 --debug-stop-after logits`
- With the rebuilt runtime, the layer body through `final_rms` was much smaller than the full pass, so the dominant remaining hotspot was the logits tail rather than `OP_RMS_NORM_F16_K_2048_SMEM` or `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64`.
- The more durable issue was cache-policy plumbing, not the new RMS or attention opcodes. CUDA only exposes one active `cudaStreamAttributeAccessPolicyWindow` per stream, so repeated `runtime.set_cache_policy(...)` calls do not stack; the last call wins.
- `python/dae/launcher.py` was overwriting the stream APW with internal metadata buffers at launch time, which also masked any schedule-side tensor hinting. For the 1B path, disabling those internal launcher cache windows and leaving the launch APW empty cut the 30-iteration `-N1 -b` average from about `9.5 ms` to about `1.4 ms` on this machine.
- A single explicit launch-time APW can still be useful, but only when chosen intentionally. On this path, `LLAMA32_1B_LAUNCH_CACHE_WINDOW=rms_hidden_persist` stayed near target at about `1.8 ms`, while `lm_head_streaming` regressed back toward `11 ms`.
- Disabling prefetch across the split logits GEMV list remained helpful, and `logits_fold` is effectively fixed at `8` for this app today because the argmax partial opcode is hard-wired to `I_STRIDE=65536`.
- Fresh-launch timings on this machine were still noisy, so prefer repeated in-process benchmarks for comparison after the APW fix and treat isolated `-b 1` wins as directional only.
- A later pass added a second argmax partial opcode for `logits_slice=32768` and generalized the 1B schedule to tune `LLAMA32_1B_LOGITS_SPLIT_M`, `LLAMA32_1B_LOGITS_WAVE_DIV`, and `LLAMA32_1B_ARGMAX_SMS`.
- The most reliable shared-build improvement was simpler than the experimental schedule search: for `app/python/llama32_1b/sched.py`, defaulting `DAE_PERSISTING_L2_BYTES=0` brought the normal fold-8 path back to about `1.41 ms` while preserving the existing correctness check.
- There is still an experimental fold-4 lane. Build with `DAE_COMPUTE_OPS_FILE=dae_compute_ops.llama32_1b.fast.build`, then run `DAE_PERSISTING_L2_BYTES=0 LLAMA32_1B_LOGITS_SPLIT_M=4 python app/python/llama32_1b/sched.py -N1 -b ...`. On this machine that reached about `1.40 ms`, but the exact HF-reference argmax token was not stable enough to make it the default path.
- A later tuning pass found a more structural logits-schedule issue: using `SchedGemv.split_M(...)` on the large logits tensor pushes the generated TMA coordinates into large base offsets, and that path both blocks larger logical slices (`131072` hits the current uint16 cord limit) and performs poorly on this machine.
- The kept schedule change was to replace logits `split_M(...)` with explicit tensor subviews per logits wave. Each wave now launches a normal GEMV on a real `[8192, 2048]` weight slice and writes into a matching `matLogits[:, start:end]` view, so the logical logits slice can stay large while each wave still uses small coordinates.
- Within-session A/B runs after that refactor consistently showed `LLAMA32_1B_LOGITS_NO_PREFETCH=1` outperforming the prefetched logits path by a large margin on `--debug-stop-after logits`, while `LLAMA32_1B_LOGITS_SPLIT_M=8` remained better than both `4` and `16`.
- Absolute one-token timings on this machine became too unstable to trust across separate launches. `nvidia-smi` showed the H100 often parked at low or mid clocks, and clock locking was not permitted for this user, so treat local absolute timing claims as directional unless they come from the same warmed session.

## 2026-03-31 Layer-Growth Follow-Up

- On the current branch state, a fresh `final_rms` layer-count sweep showed the main increase is over layer depth, not logits tail:
  - `1` layer: about `0.126 ms`
  - `4` layers: about `1.38 ms`
  - `8` layers: about `3.99 ms`
  - `16` layers through `final_rms`: about `11.43 ms`
  - `16` layers through `logits`: about `11.76 ms`
- For the current tree, treat the logits tail as secondary until the layer body is back under control.
- `app/python/llama32_1b/sched.py` now has broader GEMV tuning hooks:
  - `LLAMA32_1B_NO_PREFETCH` accepts a comma-separated stage list or `all`; `none` reenables prefetch everywhere.
  - `LLAMA32_1B_GEMV_ATOM` accepts `m64n8` or `b2`.
- On this machine, `LLAMA32_1B_NO_PREFETCH=all` consistently improved both `--debug-stop-after final_rms` and the full `-N1 -b` run relative to the prefetched path from earlier passes, but keep it opt-in until the exact-token correctness mismatch on this host is explained.
- `LLAMA32_1B_GEMV_ATOM=b2` was not a winning operator change for the 1B path. It made the layer body substantially slower in repeated tests, so keep it only as an experiment knob.
- The runtime polling backoff is now tunable through compile-time macros exposed in `include/dae/context.cuh` and passed through `Makefile` via `EXTRA_NVCC_FLAGS`.
- Rebuilding with `-DDAE_ALLOC_RETRY_SLEEP_CYCLES=0 -DDAE_BARRIER_POLL_SLEEP_CYCLES=0 -DDAE_QUEUE_POLL_SLEEP_CYCLES=0` reduced the current `final_rms` timing enough to treat barrier or queue waiting as a real contributor on this tree.
- `DAE_TMA_L2_PROMOTION_BYTES` remains a useful hint knob but not a stable default on this host. Endpoint re-checks with `0` and `256` bytes changed order across fresh-process runs, so only trust that knob when comparing within the same warmed session.
- A deeper repeated-profile pass showed the dominant wait is specifically the LD read-barrier loop, not allocwarp issue barriers. In the added profile slots, alloc wait counters stayed at zero while `ld0` accumulated almost all barrier wait cycles and `ld1` stayed idle on the one-token `final_rms` path.
- Static larger sleeps did not convincingly improve time. Moving the barrier sleep from `16` to `64` reduced raw poll count but left total wait time near-flat, which is more consistent with a late producer or long barrier residency than with barrier metadata read latency being the main issue.
- The runtime now supports adaptive barrier polling: `DAE_BARRIER_POLL_SLEEP_CYCLES` is the starting sleep, `DAE_BARRIER_POLL_MAX_SLEEP_CYCLES` is the cap, and the wait loops double the sleep every 8 unchanged polls. This is meant to reduce cache pressure when a barrier is clearly not progressing without paying the full latency of a coarse fixed sleep.
- The metadata-only launcher cache path is now stronger: when `DAE_LAUNCHER_INTERNAL_CACHE_MODE=metadata`, the packed bars+TMA window is kept persistent and the instruction buffers are explicitly marked streaming so instruction fetch does not claim the same APW budget as reusable metadata.
- The launcher now has placement knobs for metadata experiments without changing model schedules:
  - `DAE_LAUNCHER_BAR_ID_STRIDE`
  - `DAE_LAUNCHER_TMA_ID_STRIDE`
  - `DAE_LAUNCHER_METADATA_ALIGN_BYTES`
  - `DAE_LAUNCHER_METADATA_FRONT_PAD_BYTES`
  - `DAE_LAUNCHER_METADATA_GAP_BYTES`
  - `DAE_LAUNCHER_METADATA_ORDER`
- For `app/python/llama32_1b/sched.py`, sparse placement is feasible within current encoding limits because the dry build uses only about `214` barriers and `391` TMA descriptors. A tested `bar_stride=4`, `tma_stride=2` layout stayed within the `1024`-entry runtime limits.
- On this host, sparse barrier/TMA placement plus padded metadata packing did not produce a clear stable improvement on the one-token `final_rms` path. Good runs stayed near the same `~2.03 ms` band as the baseline metadata layout, so keep these knobs for exploration rather than as defaults.
- A descriptor-free `TmaLoad1D` GEMV weight path is still blocked on an offline swizzle/packing step. The current GEMV kernels consume `GMMA::Layout_K_SW128_Atom` tiles from shared memory, so a TMA1D weight load only works if the global source bytes are already packed in that exact tile layout.
