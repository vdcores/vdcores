# Llama Scheduling Notes

## Llama 3.1 8B MLP Standalone

- `app/python/llama3/mlp_sched.py` is the clean MLP-only workload that mirrors the default `app/python/llama3/sched.py` MLP schedule without ablation controls.
- Its H20-oriented MLP split is `2048 + 4096 + 4096 + 4096`: gate/up A use 32 SMs each, gate/up B/C/D use 64 SMs each, SiLU chunks use 8 side SMs at base SM 64, and two down GEMVs consume `[0:6144)` and `[6144:14336)`.
- The clean MLP schedule stores all gate/up chunks through TMA/global memory; it does not use the register-backed fused tail.
- `app/python/llama3/mlp_standalone.py` is the ablation-oriented MLP harness and may include experimental controls such as serialized barriers or TMA tail storage.

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

## Llama 3.1 8B Control-Flow Decode

- `app/python/llama3/sched.py` defaults to control-flow decode; use `--no-control-flow` for the older unrolled scheduling path.
- The simplest launch form is now `python app/python/llama3/sched.py "prompt text"`. A positional prompt auto-launches when no explicit `-l`, `-b`, instruction dump, or compute-op write mode is supplied.
- `--prompt "..."` tokenizes a user prompt with the HF tokenizer. All prompt tokens except the last one are prefetched by PyTorch; the final prompt token becomes the first VDCores decode input.
- `--message "..."` applies the tokenizer chat template when one is available, treats each flag as a user message, and prints `[output] generated_text:` after launch or benchmark.
- After VDCores execution, the app prints `[perf]` from `dae.profile` timestamps, including total VDCores decode time, `TBT_ms`, and decode tokens/s.
- On the default control-flow path, `-N/--num-generates` is treated as a decode budget, and when omitted `--max-decode-steps` provides the budget with a default of `128`. The scheduler chooses the largest supported decode count at or below that budget. For example, `prefill_tokens=0` with budget `128` schedules `128` decode steps, `prefill_tokens=70` with budget `128` schedules `122`, and `prefill_tokens=16` with `-N 256` schedules `240` because appended decode after the current KV block must be a multiple of `64`.
- PyTorch prefill uses `transformers.cache_utils.StaticCache` when `StaticKVCache` is not present in the installed Transformers version. The returned cache tensors are shaped `[batch, kv_heads, seq, head_dim]`.
- VDCores still consumes flattened KV buffers shaped `[REQ, MAX_SEQ_LEN, kv_heads * head_dim]`; prefilled keys are permuted from HF half-rotary layout into the interleaved RoPE layout before copying, while values only need the `[seq, kv_heads, head_dim]` to flat reshape.
- The Llama3 app seeds the prefetched prompt KV rows into all active request lanes because the schedule still executes the fixed `N=8` decode tile even when only lane `0` is checked.
- Prompt prefill can seed full KV blocks through the PyTorch prefill path, then VDCores starts from the final prompt token as the first decode input.
- Unaligned prompt prefill beyond the first KV block is covered by the same path. On 2026-04-21, `prefill_tokens=70` passed control-flow correctness both through the current block (`58` steps) and with one appended full block (`-N 122`).
- The control-flow path emits one decode-token body for the current KV block and repeats it with top-level `LOOPC`/`LOOPM` on counter register/lane `1`.
- When decode extends past that current block, the Llama3 schedule appends a second full-block body. That body has an inner token loop on register/lane `1` for exactly `KVBlockSize` steps and an outer full-block loop on register/lane `2`.
- The existing per-layer loop continues to use compute counter `0` and memory lane `0`.
- After the initial/current block, the control-flow path only accepts appended decode lengths that are a multiple of `KVBlockSize`.
- Dynamic memory offsets are applied with `RepeatM.offsetByCounter(...)` or `RepeatM.offsetByCounters(...)` for embedding token reads, RoPE position loads, K/V cache writeback, and final argmax token writeback.
- Decode attention uses dynamic `last_kv_active_token_len` from token counter `1`; the appended full-block body also adds block counter `2` to the attention `num_kv_blocks` and to the memory repeat count for loading previous full K/V blocks.
- Multi-block decode relies on consecutive token/block `OP_REPEAT` seeds remaining independent. On 2026-04-21, active repeat address accumulation made the full-block body write KV rows at `128 + 2*c` and leave alternating holes after the first block. The fix keeps the fast address-shift path unchanged, makes accumulator `OP_REPEAT`s extend the active repeat seed without resetting `loop_counter`, and has `RepeatM.offsetByCounters(...)` accumulate into the final consumer lane.
- RoPE table offsets must be absolute-position indexed. The Llama rope table TMA coords are `[0, 0, tile, position]`, so token iteration advances by `[0, 0, 0, 1]`; do not also slice the table to the prompt start position, or prompt position `p` will load position `2p`.
- The current-Q buffer is reduce-add backed and reused across tokens, so the repeated body clears each layer's Q buffer after that layer has consumed it. The clear runs on spare SMs `128..131` and waits on the grouped per-layer `bar_attn_out` before zeroing Q, so it cannot race ahead of GQA's Q load.
- Control-flow correctness replays greedy HF decode after the same prompt prefix and compares generated `matTokens`.
- On 2026-04-21, the no-prefill default prompt path failed exact-token correctness for longer `-N` values: `-N 48` passed, while `-N 57`, `58`, `64`, `128`, and `256` failed. A real repeated-body Q lifetime race was fixed by making `clear_q` wait on the grouped per-layer `bar_attn_out` before zeroing the reusable Q buffer. After that fix, the first no-prefill mismatch at `-N 57` remained, but final-token V/K/hidden/logit diagnostics were within thresholds and the argmax mismatch was a near-tie ordering difference (`6603` versus `7528`).

## Performance Debugging Notes

- Process hygiene matters for this app. Before collecting timings, clear leftover decode jobs with `killall python || true`; stale Python workers can make the benchmark look dramatically worse than the clean baseline.
- The timeout wrapper is useful for separating deadlocks from slow schedules:
  `python tests/script/run_with_launch_timeout.py --post-launch-timeout 20 --post-launch-idle-timeout 10 -- python app/python/llama32_1b/sched.py ...`
- For the 8B control-flow path with `prefill_tokens=70`, valid decode counts below 256 are `58`, `122`, `186`, and `250` because appended decode after the current block must be a multiple of `KVBlockSize`. On 2026-04-21, fresh-process `-b 1` measurements were `4.588`, `4.675`, `4.746`, and `4.759 ms/token` respectively.
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
