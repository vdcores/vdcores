# Compile Mode Notes

These notes summarize the new compile-mode scaffolding added on the Python side.

## Entry Points

- Compiler IR and source emitter: [python/dae/compiler.py](/home1/11362/depctg/vdcores/python/dae/compiler.py)
- Launcher mode plumbing: [python/dae/launcher.py](/home1/11362/depctg/vdcores/python/dae/launcher.py)
- CLI exposure: [python/dae/util.py](/home1/11362/depctg/vdcores/python/dae/util.py)
- Synthetic single-token dry-build path: [app/python/llama32_1b/sched.py](/home1/11362/depctg/vdcores/app/python/llama32_1b/sched.py)
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
  - now launches one generated kernel grid per shared template on the caller stream, with `blockIdx.x` selecting the template member / logical SM

## CLI And Debugging

- `dae_app(...)` now supports:
  - `--bench-compare`
  - `--compare-modes`
  - `--print-optimized ir|cuda|both`
  - `--emit-cuda`
- The synthetic single-token llama32 dry-build path now emits real instructions and forwards dae CLI flags, so it can be used for:
  - IR dumps
  - CUDA source emission
  - `interp` / `compile_ir` execution
- The model-level parsers in:
  - [app/python/llama32_1b/sched.py](/home1/11362/depctg/vdcores/app/python/llama32_1b/sched.py)
  - [app/python/llama3/sched.py](/home1/11362/depctg/vdcores/app/python/llama3/sched.py)
  - [app/python/qwen3/cli.py](/home1/11362/depctg/vdcores/app/python/qwen3/cli.py)
  now use `allow_abbrev=False`, which avoids `--mode` being misparsed as `--model-name`.

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
- Shared-template emission groups structurally equivalent SM programs together:
  - one generated template body per shape
  - per-SM scalar params in generated device arrays
  - one kernel grid launch per shared template, with one block per matching logical SM

## Current Limitations

- Compile modes currently assume a fresh launcher state and do not support incremental compile after prior instruction builds.
- `compile_cuda` is currently validated for:
  - `GEMV_WGMMA` compute-family emission
  - `Dummy` and `Copy` compute ops
  - TMA load/store/reduce memory ops
- `compile_cuda` now executes for the supported subset after rebuilding `dae.runtime`, but unsupported control-heavy programs still fail loudly during compilation.
- The compiler test suite must not write to the real generated runtime header. `compile_builders(..., mode="compile_cuda")` accepts a dedicated `runtime_header_path`, and [tests/test_compiler.py](/home1/11362/depctg/vdcores/tests/test_compiler.py) uses temporary headers so unit tests do not clobber `build/generated/dae/generated_compiled_runtime.cuh`.
- Single-token llama `compile_cuda` is currently blocked by absolute-address lowering:
  - generated source still bakes process-local pointer values for `RawAddress`, `tensor1d`, `CC0`, and similar address-based memory ops
  - those raw addresses change across fresh Python processes, so the generated runtime tag changes even when the schedule shape is otherwise identical
  - fixing this requires a real runtime address-table / symbolic-address lowering step, not more loop recovery work
- For `app/python/gemv_out.py`, the printed layer diff is against the PyTorch reference path. A direct `interp` vs `compile_cuda` parity harness is the correct way to check compiler correctness, and that parity is currently exact on the supported GEMV path.

## Benchmarks

- GEMV app benchmark on `app/python/gemv_out.py`, 10 iterations on 2026-04-04 after the launch/timing fix:
  - `interp`: `exec_ns=7334.40`
  - `compile_ir`: `exec_ns=6860.80`
  - `compile_cuda`: `exec_ns=5968.00`
- The large earlier `compile_cuda` regressions on GEMV were mostly a measurement artifact:
  - the generated runtime launched one kernel per logical SM on separate streams
  - `exec_ns` used `max(end) - min(start)`, so cross-stream launch skew showed up as fake device execution time
  - moving to one grid launch per shared template removed that skew from the benchmark path for GEMV
- The generated kernel now records its start timestamp after block-local runtime initialization, which avoids charging compile mode for setup work that interpreter mode had already excluded from its event window.
- Synthetic one-layer llama32 dry-build benchmark, 3 iterations:
  - `interp`: `avg_execution_time_ns=238069.33`
  - `compile_ir`: `avg_execution_time_ns=236533.33`
- Host-visible wall time for `compile_cuda` is still slower than interpreter on GEMV, so the remaining optimization work is in the launch/runtime path rather than the measured device execution path.

## Code Size Reduction

- Current code-size optimization path:
  - allocator emission is now table-driven inside the generated ALLOC unit
  - repeated alloc-side runs become small local arrays plus a loop over `compiled_run_alloc_op(...)`
  - large generated ALLOC/LDU/STU template helpers are emitted `__noinline__`
  - small generated ALLOC/LDU/STU helpers are emitted `__forceinline__` so short kernels such as `gemv_out` do not pay avoidable call / stack overhead
  - generated compute helpers stay inline, because making WGMMA compute noinline caused PTXAS function-boundary warnings and a worse GEMV result
- Measured generated-source reduction versus the previous shared-template emitter:
  - GEMV compiled source: `149482 -> 143069` bytes
  - llama32 one-layer dry-build compiled source: `2510050 -> 2176179` bytes
- The launch/timing fixes did not materially change GEMV generated-source size:
  - `build/generated/gemv_out_compiled.cu`: `143069` bytes on 2026-04-04
- Separate-compilation idea:
  - splitting by logical SM is probably too fine-grained now that shared-template grouping already collapses many SMs onto the same code body
  - if compile parallelism is needed later, splitting by shared template is the better next experiment than one translation unit per SM
