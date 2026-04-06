# Task: Compiled Mode Support

- Status: in_progress
- Created: 2026-04-05
- Updated: 2026-04-06
- Slug: compiled-mode-support

## Description

Track the multi-conversation effort to make compiled mode usable for real schedules: export stable compiled specs, generate efficient role code, keep correctness aligned with interpreted mode, and recover performance for memory-heavy paths without regressing debuggability.

## Current State

Compiled mode is working end-to-end for the current supported subset and has been verified on several standalone apps, including `gemv_out`, `gemv_mma_out`, `argmax`, `rmsnorm`, `tmacopy`, repeat-form `tma1d`, and `gemv_logits`. The compiled async `OP_ALLOC_TMA_LOAD_1D` path now uses inline PTX `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` instead of `cuda::device::memcpy_async_tx(...)`, which clears the preserved barrier repro hang and restores the async path for `gemv_mlp_mixed`. Compiled alloc code now skips `st_insts[slot]` materialization for ordinary shared-memory slots and only writes `st_insts` for `OP_ALLOC_WB_RAW_ADDRESS`, which is the only current compiled-mode path that still needs slot-to-global-pointer metadata. The generator now also lowers per-SM program/live lookup helpers by table shape instead of always emitting giant `switch (sm_id)` trees: small piecewise-constant tables become uniform range checks, small piecewise-affine tables become arithmetic, and only irregular dense tables fall back to `__device__ __constant__` lookup arrays. The latest profiling pass shows `gemv_mlp_mixed` compiled mode is no longer losing inside the kernel itself after loop folding and launcher-side caching. A follow-up live-value compaction now packs coordinate payloads into one 64-bit value instead of four separate scalars, cutting the mixed-MLP payload size by about `74.9%`, but the first `gemv_mlp_mixed` measurements show that footprint win is roughly performance-neutral and slightly negative in-kernel because the generated code must unpack coords before tensor ops.

## Progress

- 2026-04-04: added the initial opt-in compiled-mode export, build, and launch flow across Python export, generated includes, runtime entry points, and Torch extension wiring.
- 2026-04-04: moved compiled memory/compute state toward a compact structural-spec plus per-SM payload model so structurally identical programs deduplicate instead of exploding per SM.
- 2026-04-04: expanded compiled support to more compute ops and memory forms, including barrier-tagged alloc ops, `RegLoad`, `RegStore`, `RawAddress`, and split LDU generation by load port.
- 2026-04-05: fixed the minimal barriered producer/store/load repro and `gemv_mlp_mixed` launch path by removing pure writeback queue no-ops and using a blocking 1D load fallback for correctness.
- 2026-04-05: preserved a self-contained async 1D barrier repro bundle under `build/generated/` to debug the original non-blocking path without rebuilding the reproducer from scratch.
- 2026-04-05: cleaned up compiled codegen to make compute `pc` debug-only, lower memory instructions through direct field locals, and stop routing synthetic end tokens through LDU queues.
- 2026-04-05: rechecked `gemv_out` and repeat-form `tma1d`; `gemv_out` still shows a modest compiled win and repeat-form `tma1d` improved versus the earlier blocking-path regression.
- 2026-04-05: silenced the new generator-side unused-field warning source by marking emitted per-step field locals `[[maybe_unused]]` until field emission is specialized per consumer context.
- 2026-04-05: replaced compiled `OP_ALLOC_TMA_LOAD_1D` lowering with inline PTX `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes`, verified the preserved debug barrier repro completes, and confirmed `app/python/gemv_mlp_mixed.py --mode compiled -l` launches and finishes again on the async path.
- 2026-04-05: changed compiled alloc codegen so only `OP_ALLOC_WB_RAW_ADDRESS` writes `st_insts[slot]`; normal shared-memory loads/stores and reg-carrier steps now skip that dead metadata store.
- 2026-04-05: reran the support-set interpreter-vs-compiled benches after the slot-write cleanup:
  - repeat-form `tma1d`: `442347.48 ns` interpreted vs `272461.68 ns` compiled
  - `tmacopy`: `898424.46 ns` interpreted vs `661277.70 ns` compiled
  - `gemv_out`: `6569.76 ns` interpreted vs `5777.76 ns` compiled
  - `gemv_mma_out`: `18635.82 ns` interpreted vs `17931.53 ns` compiled
  - `rmsnorm`: `2523.60 ns` interpreted vs `1824.80 ns` compiled
  - `argmax`: `6545.26 ns` interpreted vs `5423.55 ns` compiled
- 2026-04-05: measured Hopper codegen for the large per-SM lookup switches versus dense table lookups, then updated the generator to choose compact lookup helpers by table shape: piecewise-constant runs become range checks, piecewise-affine live-offset tables become arithmetic, and only irregular cases fall back to `__device__ __constant__` tables.
- 2026-04-05: persisted the compiled-mode `st_insts[]` semantics into `agents/knowledge/runtime/vdcores-vm-model.md` so future ops can tell when slot metadata is semantically required versus just an interpreter-side implementation detail.
- 2026-04-05: reran the support-set interpreter-vs-compiled benches after the per-SM lookup optimization:
  - repeat-form `tma1d`: `442147.09 ns` interpreted vs `272464.27 ns` compiled
  - `tmacopy`: `905226.75 ns` interpreted vs `653863.03 ns` compiled
  - `gemv_out`: `6534.51 ns` interpreted vs `5815.72 ns` compiled
  - `gemv_mma_out`: `18828.35 ns` interpreted vs `17849.05 ns` compiled
  - `rmsnorm`: `2536.80 ns` interpreted vs `1825.40 ns` compiled
  - `argmax`: `6584.64 ns` interpreted vs `4686.40 ns` compiled
- 2026-04-05: reran `app/python/gemv_mlp_mixed.py` with `-b 20` after the lookup/codegen cleanups:
  - interpreted: `81114.21 ns` average duration, `86684.80 ns` average execution time
  - compiled: `89239.02 ns` average duration, `91590.40 ns` average execution time
  - compiled is still about `10%` slower than interpreted on this larger mixed schedule, but much better than the earlier pre-inline-PTX regression where compiled mode was far slower.
  - both benchmark runs printed an `Ave Diff out` near `1897%` while `Ave Diff silu1` stayed near `0.0968%`, so that specific bench-path `out` mismatch is not currently compiled-only.
- 2026-04-05: repeated the same mixed-MLP benchmark strictly sequentially to rule out overlap between interpreted and compiled timing runs:
  - interpreted: `81188.52 ns` average duration, `86947.20 ns` average execution time
  - compiled: `88903.02 ns` average duration, `91160.00 ns` average execution time
  - compiled remains about `9.5%` slower than interpreted, so the earlier rerun was not an artifact of concurrent benchmark execution.
  - the large final-`out` diff still appears in both modes (`1897.14%` interpreted, `1897.25%` compiled), while `silu1` remains aligned at `0.0968%`.
- 2026-04-06: updated `app/python/gemv_logits.py` to use `dae_app(...)`, added `--mode` / `--write-compiled-spec` support, and added a direct reference diff/checksum print for executed runs.
- 2026-04-06: exported `build/generated/gemv_logits_compiled_spec.json`, rebuilt `dae.runtime` against it, and benchmarked `gemv_logits.py -b 20`:
  - interpreted: `322707.43 ns` average duration, `335904.00 ns` average execution time, `3826.78 GB/s`
  - compiled: `325879.03 ns` average duration, `334539.20 ns` average execution time, `3789.54 GB/s`
  - compiled is only about `0.98%` slower than interpreted on average duration, so this harness is now compile-mode runnable with near-parity performance.
  - both modes reported the same large reference diff (`98.89147%`) and the same output checksum (`108.45755`), so the mismatch is not compile-mode-specific.
- 2026-04-06: profiled `gemv_logits` with `ncu` before optimizing compiled codegen:
  - compiled basic profile: `346.75 us`, `89.73%` memory throughput, `14.83%` compute throughput, `36` registers/thread
  - interpreted basic profile: `367.62 us`, `84.83%` memory throughput, `16.16%` compute throughput, `38` registers/thread
  - compiled scheduler/instruction profile: `85.08%` no-eligible cycles and `36,099,128` executed instructions
  - takeaway: the harness is primarily memory-bound, so codegen cleanup is likely to deliver only modest wins unless it also changes memory behavior.
- 2026-04-06: added generator-side loop folding in `tools/generate_compiled_program.py`:
  - consecutive identical compiled compute ops now emit a `for (__ci ...)` loop instead of repeated cloned bodies
  - consecutive repeat-block memory steps with the same lowered behavior and contiguous payload blocks now emit a `for (__step ...)` loop
  - the common GEMV `RepeatM` pattern now folds the four affine `OP_ALLOC_TMA_LOAD_4D` steps into one loop body
- 2026-04-06: rebuilt `gemv_logits` after loop folding and re-profiled/re-benched it:
  - `ptxas` compiled-kernel register usage dropped from `36` to `34`
  - compiled basic `ncu` profile moved to `351.14 us`, `89.43%` memory throughput, `15.32%` compute throughput, `34` registers/thread
  - compiled scheduler/instruction profile moved to `83.91%` no-eligible cycles and `38,114,062` executed instructions
  - benchmark: interpreted `322786.74 ns`, compiled `324617.83 ns`
  - compiled improved by about `0.39%` versus the earlier compiled run, but the harness remains slightly slower than interpreted because it is still memory-bound.
- 2026-04-06: rebuilt against `build/generated/gemv_mlp_mixed_compiled_spec.json` and reran the larger mixed benchmark after loop folding:
  - interpreted: `81543.20 ns` average duration, `86784.00 ns` average execution time
  - compiled: `87483.13 ns` average duration, `90217.60 ns` average execution time
  - compiled improved by about `1.60%` versus the earlier `88903.02 ns` compiled baseline
  - the compiled-vs-interpreted gap on this mixed schedule narrowed from about `9.5%` to about `7.3%`
  - the large bench-path final-`out` diff still remains in both modes (`1897.20%` interpreted, `1897.22%` compiled), while `silu1` stays aligned at `0.09682%`
- 2026-04-06: profiled `gemv_mlp_mixed` with `ncu` after loop folding to locate the remaining gap:
  - compiled basic profile: `100.51 us`, `75.73%` memory throughput, `13.19%` compute throughput, `40` registers/thread, `1.55 KB` static shared memory, `12.14%` achieved occupancy
  - interpreted basic profile: `99.52 us`, `77.21%` memory throughput, `16.21%` compute throughput, `40` registers/thread, `14.37 KB` static shared memory, `12.36%` achieved occupancy
  - compiled scheduler/instruction profile: `85.68%` no-eligible cycles and `9,666,282` executed instructions
  - interpreted scheduler/instruction profile: `82.35%` no-eligible cycles and `11,817,477` executed instructions
  - takeaway: compiled already executes fewer instructions than interpreted, so the remaining slowdown is not primarily due to excess generated control flow.
- 2026-04-06: tried staging compiled `live_values` into shared memory inside `dae2_compiled`, then reverted it after measurement:
  - compiled bench regressed to `88521.27 ns` average duration and `91539.20 ns` average execution time
  - compiled `ncu` duration regressed to `104.38 us`, with static shared memory increasing from `1.55 KB` to `2.99 KB`
  - takeaway: this schedule does not benefit from copying compiled payload state into shared memory up front.
- 2026-04-06: cached the compiled runtime bundle and device `live_values` tensor in `python/dae/launcher.py`, invalidating the cache only when new instructions are appended:
  - this removes repeated `build_compiled_runtime_bundle(...)` work and avoids recreating/uploading the same compiled live-value tensor on every launch/bench iteration
  - rebuilt `dae.runtime` against `build/generated/gemv_mlp_mixed_compiled_spec.json` and reran the mixed benchmark:
    - interpreted: `82155.48 ns` average duration, `87478.40 ns` average execution time
    - compiled: `86914.69 ns` average duration, `89593.60 ns` average execution time
    - compiled improved by about `0.65%` versus the earlier loop-folded compiled run (`87483.13 ns`)
    - the compiled-vs-interpreted benchmark gap narrowed further from about `7.3%` to about `5.8%`
  - on the same rebuilt binary, `ncu` basic profiles now show:
    - compiled: `98.24 us`, `77.46%` memory throughput, `13.61%` compute throughput, `40` registers/thread, `1.55 KB` static shared memory, `12.15%` achieved occupancy
    - interpreted: `99.20 us`, `77.44%` memory throughput, `16.26%` compute throughput, `40` registers/thread, `14.37 KB` static shared memory, `12.36%` achieved occupancy
  - takeaway: after caching, compiled is slightly ahead on kernel duration under `ncu`, so the residual benchmark-only gap is most likely launch/setup/cache-state overhead outside the generated kernel body.
- 2026-04-06: changed compiled live-value serialization so coord-based memory ops store one packed `uint64` payload entry per step instead of four separate coord scalars, and the generated code now unpacks coords only where tensor ops actually need them:
  - `build/generated/gemv_mlp_mixed_compiled_spec.json` moved from spec version `3` to `4`
  - `gemv_mlp_mixed` live values dropped from `23064` to `5784` entries, a `74.92%` reduction
  - the rebuilt mixed benchmark remained essentially flat:
    - interpreted: `81356.96 ns` average duration, `86379.20 ns` average execution time
    - compiled: `87066.57 ns` average duration, `89993.60 ns` average execution time
  - compiled basic `ncu` on this packed-coord build measured `99.84 us`, `76.22%` memory throughput, `13.36%` compute throughput, `40` registers/thread, `1.55 KB` static shared memory
  - takeaway: packing coords is a strong payload-footprint optimization, but on the cached `gemv_mlp_mixed` path it does not improve runtime and may slightly hurt kernel execution because coord unpack now happens on device.
- 2026-04-06: kept the packed live-value layout and tightened generated field emission so stages only unpack coords when they actually consume them:
  - alloc-side generated code no longer unpacks coords for ordinary tensor loads/stores because alloc only needs queue metadata
  - ld-side generated code now unpacks coords only for real tensor load ops, not for writeback/raw-address bookkeeping paths
  - st-side generated code now unpacks only the coord arity each opcode actually uses, for example 2 coords for 2D ops and 3 coords for 3D ops
  - on `gemv_mlp_mixed`, this cleanup was performance-neutral in the first check:
    - compiled benchmark: `87069.83 ns` average duration, `90056.00 ns` average execution time
    - compiled basic `ncu`: `101.38 us`, `75.05%` memory throughput, `13.13%` compute throughput, `40` registers/thread, `1.55 KB` static shared memory
  - takeaway: the consumer-aware unpack cleanup is worth keeping for code quality and to avoid dead generated work, but it does not materially change the current mixed-MLP performance.

## TODO

- Replace the broad `[[maybe_unused]]` suppression with narrower field emission once the alloc/LDU/STU consumers are stable enough to specialize cleanly.
- Extend the refreshed bench/correctness sweep beyond the current support set if needed, especially `gemv_mlp_mixed` and other larger compiled schedules.
- Investigate the large `gemv_logits` reference mismatch now that both interpreter and compiled paths are benchmarkable and produce the same output checksum.
- Investigate the remaining compiled-vs-interpreted `out` mismatch in `gemv_mlp_mixed` after the 1D-load hang fix.
- Investigate why `gemv_mlp_mixed.py -b` now reports a very large final-`out` diff in both interpreted and compiled modes even though the earlier launch-style checks were much smaller.
- Use the new `ncu` direction to target the next compiled-mode optimization beyond loop folding, now focusing on launch/setup overhead in the compiled path before chasing more generated-kernel rewrites.
- If packed coord payloads stay, look for a lower-overhead unpack strategy or a way to consume packed coords directly in generated tensor ops so the payload-size win can translate into runtime.
- If packed coord payloads stay, the next performance step is likely direct packed-coord consumption or launch-path work rather than more local dead-code pruning.
- Keep tracked workflow/knowledge docs in sync as the compiled-mode support surface or build procedure changes.

## Blockers / Assumptions

- The current local build flow depends on `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate` plus `DAE_ALLOW_UNSUPPORTED_COMPILER=1` before `make pyext`.
- Compiled rebuilds are single-checkout, single-spec operations because `make pyext` rewrites shared generated includes under `build/generated/dae/`.
- The preserved async barrier repro remains the quickest focused signal for future regressions in compiled 1D-load lowering.
- In this tree, compiled `OP_ALLOC_TMA_LOAD_1D` should use inline PTX rather than `cuda::device::memcpy_async_tx(...)`; the latter reproduces the preserved barrier hang while the inline PTX form completes.
- In the current compiled support set, only `OP_ALLOC_WB_RAW_ADDRESS` still needs `st_insts[slot]` metadata. Normal shared-memory slot producers do not need compiled alloc to materialize a full `MInst` in `st_insts`.
- The current per-SM lookup strategy assumes `sm_id` is a dense hardware index with modest cardinality; for this shape on Hopper, uniform range checks / affine helpers / constant-memory table loads are preferable to large generated `switch (sm_id)` trees.
- `app/python/tma1d.py` in its current linear form exceeds the interpreted memory-instruction budget, so repeat-form scheduling is the right apples-to-apples check for interpreted vs compiled verification.

## Key Files

- `/home1/11362/depctg/vdcores/tools/generate_compiled_program.py`
- `/home1/11362/depctg/vdcores/include/dae/compiled_program.cuh`
- `/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh`
- `/home1/11362/depctg/vdcores/include/dae/dae2.cuh`
- `/home1/11362/depctg/vdcores/python/dae/compiled_mode.py`
- `/home1/11362/depctg/vdcores/python/dae/launcher.py`
- `/home1/11362/depctg/vdcores/Makefile`

## Commands / Verification

- `python -m py_compile tools/generate_compiled_program.py`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/debug_copy_barrier_compiled_spec.json make pyext`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate && python tests/script/run_with_launch_timeout.py --launch-pattern "[compiled alloc]" --post-launch-timeout 20 --post-launch-idle-timeout 10 -- python build/generated/debug_manual_copy_barrier_async.py > build/generated/debug_manual_copy_barrier_async.inline-ptx.log 2>&1`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_mlp_mixed_compiled_spec.json make pyext`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate && timeout 90s python app/python/gemv_mlp_mixed.py --mode compiled -l > build/generated/gemv_mlp_mixed.inline-ptx.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --write-compiled-spec build/generated/gemv_mlp_mixed_compiled_spec.json`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --mode interpreted -b 20`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_mlp_mixed.py --mode compiled -b 20`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --mode interpreted -b 20 > .agentlog/tmp/gemv_mlp_mixed_seq.interpreted.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_mlp_mixed.py --mode compiled -b 20 > .agentlog/tmp/gemv_mlp_mixed_seq.compiled.log 2>&1`
- `python .agentlog/tmp/tma1d_repeat_bench.py --write-compiled-spec build/generated/tma1d_repeat_compiled_spec.json`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/<case>_compiled_spec.json make pyext`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python <script> --mode interpreted -b 20`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python <script> --mode compiled -b 20`
- `python .agentlog/tmp/compiled_support_bench.py`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_logits.py --write-compiled-spec build/generated/gemv_logits_compiled_spec.json`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_logits_compiled_spec.json make pyext`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_logits.py --mode interpreted -b 20 > .agentlog/tmp/gemv_logits.interpreted.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_logits.py --mode compiled -b 20 > .agentlog/tmp/gemv_logits.compiled.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name regex:.*dae2_compiled.* --launch-count 1 python app/python/gemv_logits.py --mode compiled -l > .agentlog/ncu/gemv_logits.compiled.basic.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2.*' --launch-count 1 python app/python/gemv_logits.py --mode interpreted -l > .agentlog/ncu/gemv_logits.interpreted.basic.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2_compiled.*' --launch-count 1 --section LaunchStats --section Occupancy --section InstructionStats --section SchedulerStats python app/python/gemv_logits.py --mode compiled -l > .agentlog/ncu/gemv_logits.compiled.instr_sched.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2\\(.*' --launch-count 1 --section LaunchStats --section Occupancy --section InstructionStats --section SchedulerStats python app/python/gemv_logits.py --mode interpreted -l > .agentlog/ncu/gemv_logits.interpreted.instr_sched.txt 2>&1`
- `python -m py_compile tools/generate_compiled_program.py`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_logits_compiled_spec.json make pyext`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_logits.py --mode interpreted -b 20 > .agentlog/tmp/gemv_logits.loopfold.interpreted.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_logits.py --mode compiled -b 20 > .agentlog/tmp/gemv_logits.loopfold.compiled.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name regex:.*dae2_compiled.* --launch-count 1 python app/python/gemv_logits.py --mode compiled -l > .agentlog/ncu/gemv_logits.compiled.basic.loopfold.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2_compiled.*' --launch-count 1 --section LaunchStats --section Occupancy --section InstructionStats --section SchedulerStats python app/python/gemv_logits.py --mode compiled -l > .agentlog/ncu/gemv_logits.compiled.instr_sched.loopfold.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --write-compiled-spec build/generated/gemv_mlp_mixed_compiled_spec.json > .agentlog/tmp/gemv_mlp_mixed.loopfold.spec.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_mlp_mixed_compiled_spec.json make pyext > .agentlog/tmp/gemv_mlp_mixed.loopfold.build.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --mode interpreted -b 20 > .agentlog/tmp/gemv_mlp_mixed.loopfold.interpreted.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_mlp_mixed.py --mode compiled -b 20 > .agentlog/tmp/gemv_mlp_mixed.loopfold.compiled.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2_compiled.*' --launch-count 1 python app/python/gemv_mlp_mixed.py --mode compiled -l > .agentlog/ncu/gemv_mlp_mixed.compiled.basic.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2\\(.*' --launch-count 1 python app/python/gemv_mlp_mixed.py --mode interpreted -l > .agentlog/ncu/gemv_mlp_mixed.interpreted.basic.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2_compiled.*' --launch-count 1 --section LaunchStats --section Occupancy --section InstructionStats --section SchedulerStats python app/python/gemv_mlp_mixed.py --mode compiled -l > .agentlog/ncu/gemv_mlp_mixed.compiled.instr_sched.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2\\(.*' --launch-count 1 --section LaunchStats --section Occupancy --section InstructionStats --section SchedulerStats python app/python/gemv_mlp_mixed.py --mode interpreted -l > .agentlog/ncu/gemv_mlp_mixed.interpreted.instr_sched.txt 2>&1`
- `python -m py_compile python/dae/launcher.py`
- `python -m py_compile python/dae/compiled_mode.py tools/generate_compiled_program.py python/dae/launcher.py`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_mlp_mixed_compiled_spec.json make pyext > .agentlog/tmp/gemv_mlp_mixed.finalopt.build.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --mode interpreted -b 20 > .agentlog/tmp/gemv_mlp_mixed.finalopt.interpreted.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_mlp_mixed.py --mode compiled -b 20 > .agentlog/tmp/gemv_mlp_mixed.finalopt.compiled.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2_compiled.*' --launch-count 1 python app/python/gemv_mlp_mixed.py --mode compiled -l > .agentlog/ncu/gemv_mlp_mixed.compiled.basic.finalopt.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2\\(.*' --launch-count 1 python app/python/gemv_mlp_mixed.py --mode interpreted -l > .agentlog/ncu/gemv_mlp_mixed.interpreted.basic.finalopt.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --write-compiled-spec build/generated/gemv_mlp_mixed_compiled_spec.json > .agentlog/tmp/gemv_mlp_mixed.packedcoords.spec.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_mlp_mixed_compiled_spec.json make pyext > .agentlog/tmp/gemv_mlp_mixed.packedcoords.build.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --mode interpreted -b 20 > .agentlog/tmp/gemv_mlp_mixed.packedcoords.interpreted.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_mlp_mixed.py --mode compiled -b 20 > .agentlog/tmp/gemv_mlp_mixed.packedcoords.compiled.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2_compiled.*' --launch-count 1 python app/python/gemv_mlp_mixed.py --mode compiled -l > .agentlog/ncu/gemv_mlp_mixed.compiled.basic.packedcoords.txt 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_mlp_mixed_compiled_spec.json make pyext > .agentlog/tmp/gemv_mlp_mixed.packedcoords-pruned.build.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && timeout 180s python app/python/gemv_mlp_mixed.py --mode compiled -b 20 > .agentlog/tmp/gemv_mlp_mixed.packedcoords-pruned.compiled.log 2>&1`
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate >/dev/null 2>&1 || true && conda activate && ncu --set basic --target-processes all --kernel-name-base demangled --kernel-name 'regex:.*dae2_compiled.*' --launch-count 1 python app/python/gemv_mlp_mixed.py --mode compiled -l > .agentlog/ncu/gemv_mlp_mixed.compiled.basic.packedcoords-pruned.txt 2>&1`

## Artifacts

- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.py`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.inc`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.log`
- `/home1/11362/depctg/vdcores/build/generated/tma1d_perf.inc`
- `/home1/11362/depctg/vdcores/build/generated/test_tma1d_compiled_enabled.inc`
- `/home1/11362/depctg/vdcores/build/generated/debug_copy_barrier_compiled_spec.json`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.inline-ptx.log`
- `/home1/11362/depctg/vdcores/build/generated/gemv_mlp_mixed.inline-ptx.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_bench.spec.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_bench.build.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_bench.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_bench.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_seq.spec.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_seq.build.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_seq.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed_seq.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/tma1d_repeat_bench.py`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/bench_slotinst_opt/`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/compiled_support_bench.py`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/bench_sm_lookup_opt/`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/sm_lookup_codegen_probe.cu`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/sm_lookup_codegen_probe.ptx`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/sm_lookup_codegen_probe.cubin`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/sm_lookup_codegen_probe.sass`
- `/home1/11362/depctg/vdcores/build/generated/gemv_logits_compiled_spec.json`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_logits.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_logits.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_logits.compiled.basic.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_logits.interpreted.basic.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_logits.compiled.instr_sched.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_logits.interpreted.instr_sched.txt`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_logits.loopfold.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_logits.loopfold.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_logits.compiled.basic.loopfold.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_logits.compiled.instr_sched.loopfold.txt`
- `/home1/11362/depctg/vdcores/build/generated/gemv_mlp_mixed_compiled_spec.json`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.loopfold.spec.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.loopfold.build.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.loopfold.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.loopfold.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.compiled.basic.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.interpreted.basic.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.compiled.instr_sched.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.interpreted.instr_sched.txt`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.smemlive.build.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.smemlive.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.smemlive.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.compiled.basic.smemlive.txt`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.cached.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.cached.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.finalopt.build.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.finalopt.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.finalopt.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.compiled.basic.finalopt.txt`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.interpreted.basic.finalopt.txt`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.packedcoords.spec.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.packedcoords.build.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.packedcoords.interpreted.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.packedcoords.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.compiled.basic.packedcoords.txt`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.packedcoords-pruned.build.log`
- `/home1/11362/depctg/vdcores/.agentlog/tmp/gemv_mlp_mixed.packedcoords-pruned.compiled.log`
- `/home1/11362/depctg/vdcores/.agentlog/ncu/gemv_mlp_mixed.compiled.basic.packedcoords-pruned.txt`

## Next Step

Decide whether to keep the packed-coord live-value layout as a footprint optimization or refine/revert it for `gemv_mlp_mixed`, then continue targeting the next win in the compiled launch path while separately deciding when to stop and fix the large non-compiled-specific bench-path output mismatches in `gemv_logits.py` and `gemv_mlp_mixed.py -b`.
