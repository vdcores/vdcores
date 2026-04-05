# Task: Compiled Mode Support

- Status: in_progress
- Created: 2026-04-05
- Updated: 2026-04-05
- Slug: compiled-mode-support

## Description

Track the multi-conversation effort to make compiled mode usable for real schedules: export stable compiled specs, generate efficient role code, keep correctness aligned with interpreted mode, and recover performance for memory-heavy paths without regressing debuggability.

## Current State

Compiled mode is working end-to-end for the current supported subset and has been verified on several standalone apps, including `gemv_out`, `gemv_mma_out`, `argmax`, `rmsnorm`, `tmacopy`, and repeat-form `tma1d`. Recent cleanup removed compiled-only control-flow overhead, made compute `pc` debug-only, stopped pushing synthetic LDU end tokens, and lowered memory ops through direct field locals. The main open issue is the async `OP_ALLOC_TMA_LOAD_1D` path: the blocking fallback is correct but causes severe regressions on memory-heavy schedules, while the async path still hangs on the preserved barrier repro.

## Progress

- 2026-04-04: added the initial opt-in compiled-mode export, build, and launch flow across Python export, generated includes, runtime entry points, and Torch extension wiring.
- 2026-04-04: moved compiled memory/compute state toward a compact structural-spec plus per-SM payload model so structurally identical programs deduplicate instead of exploding per SM.
- 2026-04-04: expanded compiled support to more compute ops and memory forms, including barrier-tagged alloc ops, `RegLoad`, `RegStore`, `RawAddress`, and split LDU generation by load port.
- 2026-04-05: fixed the minimal barriered producer/store/load repro and `gemv_mlp_mixed` launch path by removing pure writeback queue no-ops and using a blocking 1D load fallback for correctness.
- 2026-04-05: preserved a self-contained async 1D barrier repro bundle under `build/generated/` to debug the original non-blocking path without rebuilding the reproducer from scratch.
- 2026-04-05: cleaned up compiled codegen to make compute `pc` debug-only, lower memory instructions through direct field locals, and stop routing synthetic end tokens through LDU queues.
- 2026-04-05: rechecked `gemv_out` and repeat-form `tma1d`; `gemv_out` still shows a modest compiled win and repeat-form `tma1d` improved versus the earlier blocking-path regression.
- 2026-04-05: silenced the new generator-side unused-field warning source by marking emitted per-step field locals `[[maybe_unused]]` until field emission is specialized per consumer context.

## TODO

- Restore a correct async lowering for compiled `OP_ALLOC_TMA_LOAD_1D` so `tma1d`, `tmacopy`, `rmsnorm`, and mixed GEMV paths recover performance.
- Debug the preserved async barrier repro and identify why the second barriered load never reaches the later compiled LDU arrival point.
- Replace the broad `[[maybe_unused]]` suppression with narrower field emission once the alloc/LDU/STU consumers are stable enough to specialize cleanly.
- Re-run correctness and performance checks for `tma1d`, `gemv_out`, `tmacopy`, `rmsnorm`, `argmax`, and `gemv_mlp_mixed` after the async 1D path is fixed.
- Keep tracked workflow/knowledge docs in sync as the compiled-mode support surface or build procedure changes.

## Blockers / Assumptions

- The current local build flow depends on `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate` plus `DAE_ALLOW_UNSUPPORTED_COMPILER=1` before `make pyext`.
- Compiled rebuilds are single-checkout, single-spec operations because `make pyext` rewrites shared generated includes under `build/generated/dae/`.
- The preserved async barrier repro is still the best focused signal for the remaining 1D-load hang.
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

## Artifacts

- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.py`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.inc`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.log`
- `/home1/11362/depctg/vdcores/build/generated/tma1d_perf.inc`
- `/home1/11362/depctg/vdcores/build/generated/test_tma1d_compiled_enabled.inc`
- `/home1/11362/depctg/vdcores/build/generated/debug_copy_barrier_compiled_spec.json`

## Next Step

Continue from the preserved async 1D barrier repro, then replace the blocking fallback with a correct async compiled load path and re-benchmark the memory-heavy apps.
