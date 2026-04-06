# Task: Compiled Mode Support

- Status: in_progress
- Created: 2026-04-05
- Updated: 2026-04-05
- Slug: compiled-mode-support

## Description

Track the multi-conversation effort to make compiled mode usable for real schedules: export stable compiled specs, generate efficient role code, keep correctness aligned with interpreted mode, and recover performance for memory-heavy paths without regressing debuggability.

## Current State

Compiled mode is working end-to-end for the current supported subset and has been verified on several standalone apps, including `gemv_out`, `gemv_mma_out`, `argmax`, `rmsnorm`, `tmacopy`, and repeat-form `tma1d`. The compiled async `OP_ALLOC_TMA_LOAD_1D` path now uses inline PTX `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` instead of `cuda::device::memcpy_async_tx(...)`, which clears the preserved barrier repro hang and restores the async path for `gemv_mlp_mixed`. Compiled alloc code now skips `st_insts[slot]` materialization for ordinary shared-memory slots and only writes `st_insts` for `OP_ALLOC_WB_RAW_ADDRESS`, which is the only current compiled-mode path that still needs slot-to-global-pointer metadata. The generator now also lowers per-SM program/live lookup helpers by table shape instead of always emitting giant `switch (sm_id)` trees: small piecewise-constant tables become uniform range checks, small piecewise-affine tables become arithmetic, and only irregular dense tables fall back to `__device__ __constant__` lookup arrays. The latest support-set bench pass still shows compiled mode ahead on all six currently tracked standalone harnesses, while the larger `gemv_mlp_mixed` benchmark is now runnable again but still slightly slower in compiled mode.

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

## TODO

- Replace the broad `[[maybe_unused]]` suppression with narrower field emission once the alloc/LDU/STU consumers are stable enough to specialize cleanly.
- Extend the refreshed bench/correctness sweep beyond the current support set if needed, especially `gemv_mlp_mixed` and other larger compiled schedules.
- Investigate the remaining compiled-vs-interpreted `out` mismatch in `gemv_mlp_mixed` after the 1D-load hang fix.
- Investigate why `gemv_mlp_mixed.py -b` now reports a very large final-`out` diff in both interpreted and compiled modes even though the earlier launch-style checks were much smaller.
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

## Next Step

Investigate the large `gemv_mlp_mixed.py -b` final-`out` mismatch that now appears in both modes, then decide whether there is still a worthwhile compiled-mode optimization path for that larger mixed schedule beyond the current support-set wins.
