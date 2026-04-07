# Task: Compiled Mode Support

- Status: in_progress
- Created: 2026-04-05
- Updated: 2026-04-06
- Slug: compiled-mode-support

## Description

Make compiled mode usable for real schedules by exporting stable compiled specs, generating efficient runtime code, keeping correctness aligned with interpreted mode, and recovering performance on larger mixed workloads.

## Current State

Compiled mode works end-to-end for the current supported subset and has been exercised on `gemv_out`, `gemv_mma_out`, `argmax`, `rmsnorm`, `tmacopy`, repeat-form `tma1d`, `gemv_logits`, and `gemv_mlp_mixed`.

The current support surface includes the key correctness and codegen fixes that were blocking broader use:

- compiled `OP_ALLOC_TMA_LOAD_1D` uses inline PTX `cp.async.bulk...complete_tx::bytes`, which avoids the earlier preserved barrier hang;
- compiled alloc lowering only materializes `st_insts[slot]` for `OP_ALLOC_WB_RAW_ADDRESS`, not ordinary shared-memory slots;
- per-SM lookup lowering now prefers range/arithmetic helpers over large generated `switch (sm_id)` trees when table structure permits it;
- live-value lowering now keeps only root values and affine `sm_id`-derived schedule coords, which cut the mixed MLP spec to version `5`, reduced live values from `24` to `4`, and collapsed the mixed schedule from `129` compiled programs to `3`;
- loop-fold and accumulator lowering can keep repeated affine coord/address updates in registers instead of rebuilding full expressions each iteration.

On the latest mixed-MLP work, compiled mode is no longer clearly losing in the hot kernel body. The remaining work is mostly around correctness investigation for known output diffs and deciding whether any further optimization should target launch/setup overhead instead of more local generated-kernel pruning.

The split compiled launch path exists for experimentation, but it is currently slower and larger than the default monolithic launch and should remain experimental.

## Progress Summary

- Added the compiled export, build, and launch flow across Python export, generated includes, runtime entry points, and Torch extension wiring.
- Moved compiled state toward structural spec plus payload dedup so structurally identical programs do not explode per SM.
- Expanded compiled support across the current memory/compute subset, including barrier-tagged alloc ops, register load/store handling, raw-address cases, and more load/store lowering coverage.
- Fixed the async 1D barriered load path by switching compiled lowering to inline PTX and kept the focused repro artifact for future regressions.
- Simplified generated code by pruning dead slot metadata writes, narrowing emitted locals, and lowering per-SM tables and loop-varying values more compactly.
- Added launcher-side profiling hooks and refreshed mixed-workload profiling; current evidence points to near-parity in kernel time on `gemv_mlp_mixed`.
- Added an opt-in split compiled launch mode for experiments, but it is not a default candidate yet.

## TODO

- Investigate the remaining `gemv_mlp_mixed` output mismatch, especially the large final-`out` diff that appears in benchmark-style runs.
- Investigate the `gemv_logits` reference mismatch now that interpreted and compiled runs share the same checksum path.
- Extend verification beyond the current support set when new schedules or operators are added.
- Replace broad `[[maybe_unused]]` suppression with narrower field emission once alloc/LDU/STU consumers settle.
- Use `ncu` plus `--profile-sm-times` to decide whether the next optimization target is launch/setup overhead, latency hiding, or another codegen pass.
- Keep workflow and runtime knowledge docs aligned with any future compiled-mode semantic changes.

## Blockers / Assumptions

- Rebuilds are effectively single-spec because `make pyext` rewrites shared generated includes under `build/generated/dae/`.
- Local rebuilds depend on the conda `base` environment and `DAE_ALLOW_UNSUPPORTED_COMPILER=1`.
- The preserved async barrier repro remains the fastest targeted check for compiled 1D-load regressions.
- In the current support set, only `OP_ALLOC_WB_RAW_ADDRESS` still semantically requires `st_insts[slot]`.
- The generic launcher `profile[:,0:2]` timestamps are useful for comparative per-SM skew analysis, not calibrated wall-clock timing.

## Key Files

- `/home1/11362/depctg/vdcores/tools/generate_compiled_program.py`
- `/home1/11362/depctg/vdcores/python/dae/compiled_mode.py`
- `/home1/11362/depctg/vdcores/python/dae/launcher.py`
- `/home1/11362/depctg/vdcores/include/dae/compiled_program.cuh`
- `/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh`
- `/home1/11362/depctg/vdcores/include/dae/dae2.cuh`
- `/home1/11362/depctg/vdcores/src/runtime.cu`

## Commands / Verification

- `python -m py_compile tools/generate_compiled_program.py python/dae/compiled_mode.py python/dae/launcher.py`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --write-compiled-spec build/generated/gemv_mlp_mixed_compiled_spec.json`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate >/dev/null 2>&1 || true && conda activate && DAE_ALLOW_UNSUPPORTED_COMPILER=1 DAE_COMPILED_SPEC_FILE=build/generated/gemv_mlp_mixed_compiled_spec.json make pyext`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --mode compiled -l`
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate >/dev/null 2>&1 || true && conda activate && python app/python/gemv_mlp_mixed.py --mode compiled -b 20`

## Artifacts

- `/home1/11362/depctg/vdcores/build/generated/gemv_mlp_mixed_compiled_spec.json`
- `/home1/11362/depctg/vdcores/build/generated/dae/compiled_program.inc`
- `/home1/11362/depctg/vdcores/build/generated/dae/compute_opcode_order.inc`
- `/home1/11362/depctg/vdcores/build/generated/dae/dynamic_compute_handlers.inc`
- `/home1/11362/depctg/vdcores/build/generated/debug_copy_barrier_compiled_spec.json`

## Next Step

Use the current mixed-MLP build to resolve the remaining correctness mismatch first, then decide whether any further compiled-mode optimization should focus on launch/setup overhead or on another generated-memory/codegen pass.
