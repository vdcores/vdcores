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
  - emits split-loop CUDA source
  - does not yet launch an executable generated kernel
  - fails loudly after source emission if used through `Launcher.launch(...)`

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

## Current Limitations

- Compile modes currently assume a fresh launcher state and do not support incremental compile after prior instruction builds.
- `compile_cuda` is only validated for the early GEMV/TMA subset.
- Full generated-kernel execution is still pending; current V2 work is source generation plus split-unit metadata.
