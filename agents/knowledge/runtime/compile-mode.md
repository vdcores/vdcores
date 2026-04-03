# Compile Mode Notes

These notes summarize the new compile-mode scaffolding added on the Python side.

## Entry Points

- Compiler IR and source emitter: [python/dae/compiler.py](/home1/11362/depctg/vdcores/python/dae/compiler.py)
- Launcher mode plumbing: [python/dae/launcher.py](/home1/11362/depctg/vdcores/python/dae/launcher.py)
- CLI exposure: [python/dae/util.py](/home1/11362/depctg/vdcores/python/dae/util.py)
- Compiler tests: [tests/test_compiler.py](/home1/11362/depctg/vdcores/tests/test_compiler.py)

## Current Modes

- `interp`:
  - existing behavior
  - builder instructions encode directly to `CInst` and `MInst`
- `compile_ir`:
  - builds `ProgramIR`
  - normalizes repeat structure
  - validates semantics
  - re-emits the same logical instruction ISA
  - then uses the existing interpreter runtime
- `compile_cuda`:
  - builds the same normalized IR
  - lowers to split-unit metadata first
  - emits direct split-loop CUDA source with per-SM ALLOC, LDU, STU, and compute loops
  - writes `build/generated/dae/generated_compiled_runtime.cuh` for the CUDA extension build
  - launches through `runtime.launch_compiled_dae(...)` after `make pyext`
  - currently launches one generated kernel per logical SM program on separate CUDA streams

## IR Shape

- Per-SM program container:
  - `ProgramIR`
  - `SMProgramIR`
- Compute nodes:
  - `ComputeOpIR`
  - `LoopCIR`
  - `TerminateComputeIR`
- Memory nodes:
  - `MemoryOpIR`
  - `RepeatControlIR`
  - `RepeatRegionIR`
  - `LoopMIR`
  - `BarrierIssueIR`
  - `TerminateMemoryIR`
- Split-unit lowering:
  - `SplitUnitProgramIR`
  - `SMSplitUnitIR`
  - `SplitAllocOpIR`
  - `SplitMemOpIR`
  - `SplitComputeOpIR`
  - `SplitLoopSpanIR`

## Important Semantics Captured

- `RepeatM` is recovered as structured `RepeatRegionIR` from the linear memory stream.
- Re-emission preserves the existing ISA and flags, including:
  - group
  - jump
  - barrier attachment
  - port selection
  - writeback vs load role
- `LoopM` keeps:
  - target PC
  - lane/register owner
  - bar shift
  - TMA shift
- `compile_cuda` treats ALLOC ordering as separate from LDU/STU address generation.
- Generated CUDA keeps no instruction tables in the direct path:
  - repeat regions become CUDA `for` loops
  - address deltas become direct address or coord expressions
  - unused LDU ports emit no-op bodies instead of dead local command state

## Current Limitations

- Compile modes currently assume a fresh launcher state and do not support incremental compile after prior instruction builds.
- `compile_cuda` is currently validated for:
  - `GEMV_WGMMA` compute-family emission
  - `Dummy` and `Copy` compute ops
  - TMA load/store/reduce memory ops
- `compile_cuda` now executes for the supported subset after rebuilding `dae.runtime`, but unsupported control-heavy programs still fail loudly during compilation.
- For `app/python/gemv_out.py`, the printed layer diff is against the PyTorch reference path. A direct `interp` vs `compile_cuda` parity harness is the correct way to check compiler correctness, and that parity is currently exact on the supported GEMV path.
