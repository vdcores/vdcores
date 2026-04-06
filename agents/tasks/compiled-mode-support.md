# Task: Compiled Mode Support

- Status: in_progress
- Created: 2026-04-05
- Updated: 2026-04-05
- Slug: compiled-mode-support

## Description

Track the multi-conversation effort to make compiled mode usable for real schedules: export stable compiled specs, generate efficient role code, keep correctness aligned with interpreted mode, and recover performance for memory-heavy paths without regressing debuggability.

## Current State

Compiled mode is working end-to-end for the current supported subset and has been verified on several standalone apps, including `gemv_out`, `gemv_mma_out`, `argmax`, `rmsnorm`, `tmacopy`, and repeat-form `tma1d`. The compiled async `OP_ALLOC_TMA_LOAD_1D` path now uses inline PTX `cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes` instead of `cuda::device::memcpy_async_tx(...)`, which clears the preserved barrier repro hang and restores the async path for `gemv_mlp_mixed`. The main remaining work is broader correctness/performance rechecks on the 1D-load-heavy examples and the residual compiled-vs-interpreted `out` mismatch in mixed MLP.

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

## TODO

- Replace the broad `[[maybe_unused]]` suppression with narrower field emission once the alloc/LDU/STU consumers are stable enough to specialize cleanly.
- Re-run correctness and performance checks for `tma1d`, `gemv_out`, `tmacopy`, `rmsnorm`, `argmax`, and `gemv_mlp_mixed` now that the async 1D path is fixed with inline PTX.
- Investigate the remaining compiled-vs-interpreted `out` mismatch in `gemv_mlp_mixed` after the 1D-load hang fix.
- Keep tracked workflow/knowledge docs in sync as the compiled-mode support surface or build procedure changes.

## Blockers / Assumptions

- The current local build flow depends on `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate` plus `DAE_ALLOW_UNSUPPORTED_COMPILER=1` before `make pyext`.
- Compiled rebuilds are single-checkout, single-spec operations because `make pyext` rewrites shared generated includes under `build/generated/dae/`.
- The preserved async barrier repro remains the quickest focused signal for future regressions in compiled 1D-load lowering.
- In this tree, compiled `OP_ALLOC_TMA_LOAD_1D` should use inline PTX rather than `cuda::device::memcpy_async_tx(...)`; the latter reproduces the preserved barrier hang while the inline PTX form completes.
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

## Artifacts

- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.py`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.inc`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.log`
- `/home1/11362/depctg/vdcores/build/generated/tma1d_perf.inc`
- `/home1/11362/depctg/vdcores/build/generated/test_tma1d_compiled_enabled.inc`
- `/home1/11362/depctg/vdcores/build/generated/debug_copy_barrier_compiled_spec.json`
- `/home1/11362/depctg/vdcores/build/generated/debug_manual_copy_barrier_async.inline-ptx.log`
- `/home1/11362/depctg/vdcores/build/generated/gemv_mlp_mixed.inline-ptx.log`

## Next Step

Re-run the broader 1D-load-heavy compiled apps (`tma1d`, `tmacopy`, `rmsnorm`, `gemv_out`) on the inline-PTX path, then investigate the remaining `gemv_mlp_mixed` final-`out` mismatch now that the launch hang is gone.
