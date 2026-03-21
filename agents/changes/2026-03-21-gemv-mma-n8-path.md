# Purpose

Add a separate `N=8` MMA-based GEMV operator path without changing the existing `Gemv_M64N8` WGMMA path, then rebuild and verify the new kernel on GH200 hardware.

## Files Touched

- `include/dae/opcode.cuh.inc`
- `include/dae/dae2.cuh`
- `include/task/gemv.cuh`
- `python/dae/launcher.py`
- `app/python/gemv_mma_out.py`
- `agents/changes/2026-03-21-gemv-mma-n8-path.md`
- `agents/workflows/development-and-test.md`
- `agents/knowledge/project-map.md`

## What Changed

- Added a new compute opcode `OP_GEMV_M64N8_MMA`.
- Added a new Python instruction wrapper `Gemv_M64N8_MMA` with fixed `MNK=(64, 8, 256)` and `n_batch=1`.
- Added a new dispatch case in `include/dae/dae2.cuh` that routes the new opcode to `task_gemv_mma<__nv_bfloat16, 64, 8, 256>(...)`.
- Fixed `task_gemv_mma` to use the queue APIs in the same way as the rest of the runtime:
  - use `m2c.template pop<0>()`
  - use `c2m.push(tid, ...)`
  - use `c2m.template push<0, true>(tid, ...)`
- Switched the MMA kernel shared-memory layouts to the same swizzled K-major / MN-swizzled layouts used by the existing GEMV path so the new kernel matches the existing TMA loaders and store path.
- Added `app/python/gemv_mma_out.py` as a dedicated harness for the new operator. It uses one SM per `64x8` output tile, defaults to `M=4096`, `K=4096`, and allows `GEMV_M`, `GEMV_K`, and `GEMV_SMS` overrides for smaller checks.
- Left the existing `app/python/gemv_out.py` path unchanged.

## Verification

- `python -m py_compile app/python/gemv_mma_out.py python/dae/launcher.py`
- `make pyext`
- `GEMV_M=64 GEMV_K=256 GEMV_SMS=1 python app/python/gemv_mma_out.py -l`
- `python app/python/gemv_mma_out.py -l`
- `python app/python/gemv_out.py -l`

## Notes

- A direct `make pyext` from the starting shell failed because `nvcc 12.5` rejected the active Conda GCC 14 host compiler.
- Running `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate && make pyext` succeeded in this environment and rebuilt the extension.
- New-kernel verification results:
  - `M=64, N=8, K=256`: `Ave Diff GEMV MMA M64N8: 0.0 %`
  - `M=4096, N=8, K=4096`: `Ave Diff GEMV MMA M64N8: 0.0 %`
- Existing GEMV regression check after the new path still launched successfully:
  - `app/python/gemv_out.py -l`: `Ave Diff GEMV Layer: out_proj: 0.16383243138063858 %`
