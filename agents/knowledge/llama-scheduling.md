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

- For attention-only comparisons on `app/python/llama32_1b/sched.py`, the cleanest metric is the prefix delta `attn - v_proj`, not the raw `attn` prefix. `attn` still includes `QProj`, `QRope`, `KProj`, `KRope`, and `VProj`.
- On 2026-04-10, with `-N 1 --debug-num-layers 1` and the current grouped-query decode path:
  - `--debug-stop-after v_proj -b 10`: about `11.59 us`
  - `--debug-stop-after attn -b 10`: about `19.80 us`
  - the pure GQA decode delta was therefore about `8.20 us`
- The 1B grouped-query decode kernel is still structurally mismatched to the workload:
  - `matO_attn_view` is shaped `[N, NUM_KV_HEAD, HEAD_GROUP_SIZE, HEAD_DIM]`
  - for Llama 3.2 1B, `HEAD_GROUP_SIZE = 4` and `HEAD_DIM = 64`
  - but `SchedAttentionDecoding` still dispatches the standard `M64N64K16` decode attention kernel
  - `tma_gqa_load_q()` in `python/dae/model.py` explicitly duplicates the 4 active Q rows to fill a 64-row tile
  - so the decode path is still paying for a 64-row Q tile even though only 4 rows are logically live
- A first attempt at a dedicated small-Q decode opcode on 2026-04-10 confirmed the likely optimization direction but was not stable enough to keep:
  - the idea was a warp-per-query-row online-softmax kernel for the `head_dim=64`, `num_active_q=4`, `KVBlockSize=64` path
  - after integrating it as a new compute opcode, the monolithic `dae2` launcher became unstable even on earlier prefixes
  - the experimental opcode was reverted, so the current tree is back on the original attention-decode kernel
- Durable takeaway: the next serious attention optimization should still target a dedicated small-Q decode path, but it needs a safer integration strategy than simply dropping a new opcode into the current monolithic kernel.

- Process hygiene matters for this app. Before collecting timings, clear leftover decode jobs with `killall python || true`; stale Python workers can make the benchmark look dramatically worse than the clean baseline.
- The timeout wrapper is useful for separating deadlocks from slow schedules:
  `python tests/script/run_with_launch_timeout.py --post-launch-timeout 20 --post-launch-idle-timeout 10 -- python app/python/llama32_1b/sched.py ...`
- The most reliable one-layer timing method on the 1B path is still prefix subtraction with fresh processes:
  - run `--debug-num-layers 1`
  - benchmark `--debug-stop-after <stage>` in a fresh process
  - subtract adjacent cumulative prefixes
  - prefer medians over means when a stage occasionally spikes
- For end-to-end stage comparisons against an external baseline, remember that the current 1B path is still built around `Gemv_M64N8`-style `N=8` WGMMA GEMV tiles, so comparisons against a true batch-1 matvec-style latency path are not apples-to-apples for `o_proj` and `down_proj`.
- In the current branch, the most important runtime-geometry lesson is that `SchedGemv` legality depends on atom granularity:
  - `Gemv_M64N8` requires `k_per_fold` to be a multiple of `256 * 4 = 1024`
  - `Gemv_M64N8B2` lowers that to `256 * 2 = 512`
  - this is why full `128`-SM one-wave down projection was illegal with `Gemv_M64N8` but legal with `Gemv_M64N8B2`
- The current best down-proj experiment on the 1B path is:
  - collapse split down projection into one full `MNK=(HIDDEN, N, INTERMIDIATE)` GEMV
  - use `Gemv_M64N8B2`
  - place it on `128` SM
  - wait for full SiLU output on `bar_silu_out2`
  - keep the completion signal on `bar_layer`
- Measured down-proj variants on 2026-04-09:
  - original split path (`down_low + down_high`): about `13.578 us`
  - full single-stage `Gemv_M64N8B2`: about `12.288 us`
  - full single-stage `Gemv_M64N8B2` without the load-side barrier: about `12.128 us`
  - full single-stage `Gemv_M64N8K64`: about `34.784 us`
- The load-side barrier on the full `Gemv_M64N8B2` down projection contributed only about `0.16 us`, so the down-proj bottleneck there is the actual GEMV work, not synchronization.
- On 2026-04-09, the isolated `OutProj` load barrier on the 1B path was much more expensive:
  - `--debug-stop-after attn`: about `22.9-23.6 us`
  - `--debug-stop-after out` with `OutProj` load barrier disabled: about `29.3-30.1 us`
  - `--debug-stop-after out` with `OutProj.bar("load", layerg["bar_attn_out"])`: about `31.9 us`
  - the load-side barrier therefore added about `2.6-2.8 us`
- `layer.bar_attn_out` is still bound to `64` even when the `OutProj` load barrier is disabled, because attention output stores still release it. The extra `OutProj` cost is therefore consumer-side waiting on the barrier, not newly introduced producer-side barrier arrivals.
- Per-SM whole-kernel durations for SM `0..63` stayed fairly tight in both cases, so the current best hypothesis is not severe producer imbalance. The more likely issue is the runtime's global barrier wait path itself:
  - producer stores decrement one shared global counter
  - `OutProj` load warps poll that same counter until it reaches zero
  - the polling/wakeup behavior appears to cost multiple microseconds on this path
- Later on 2026-04-09, direct instrumentation of the load-warp wait path confirmed that hypothesis:
  - with the `OutProj` load barrier disabled, filtered wait cycles for `layer.bar_attn_out` were exactly zero
  - with `OutProj.bar("load", layerg["bar_attn_out"])` enabled, each of the `64` `OutProj` SMs incurred one matched wait on that barrier
  - the matched wait averaged about `15.98 us` per SM, with a max of about `16.13 us`
  - the kernel wall-time delta remained smaller than that because the wait overlaps with other work, but the barrier is still causing a substantial real stall on every participating SM
- A follow-up experiment skipped the actual matched `bar_attn_out` load after waiting but still handed a slot to compute. That did not make the barrier-enabled `OutProj` path faster, which points away from the `attnO` TMA load itself and toward the synchronization path as the real source of the latency penalty.
- Another follow-up made the load-barrier polling backoff runtime-configurable. On the barrier-enabled `OutProj` path:
  - polling sleep `0` cycles was best at about `28.7 us`
  - polling sleep `16` cycles was about `30.0 us`
  - polling sleep `64` cycles was about `31.4 us`
  - polling sleep `256` cycles was about `30.5 us`
- The runtime therefore now defaults the new load-barrier polling knob to `0` cycles. On this path, the old `__nanosleep(16)` backoff was itself contributing around `1+ us` of additional latency.
- `Gemv_M64N8K64` was a useful negative result: deeper per-SM pipelining did not help this full down projection and instead made it much slower than both the split path and the `B2` path.
- After collapsing down projection into one full stage, `debug_utils.py` still keeps the old `down_low` / `down_high` names. In that experimental shape:
  - `down_low` means the real full down projection
  - `down_high` is effectively an empty checkpoint and should be treated as subtraction noise
- The biggest remaining latency gap versus the Megakernels baseline is likely in `upgate_silu`, not down projection:
  - Megakernels fuses RMS + up matvec + gate matvec + SiLU + writeback in one latency op
  - the current 1B path still separates `post_attn_rms`, gate/up GEMVs, intermediate global writes, and two different SiLU stages
  - this path is therefore much more fragmented and likely pays extra global-memory traffic and barrier overhead
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
