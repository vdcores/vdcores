# Allocwarp GPR Control Ops

- `include/dae/pipeline/allocwarp.cuh` uses `MemoryVirtualCore::gpr` for repeat address deltas (`MVC_GPR_DELTA`) and accumulators (`MVC_GPR_ACC`).
- `OP_ALLOC_REG_LOAD` is handled in `include/dae/pipeline/ldwarp.cuh` and only writes the LD warp `regFile`; it does not update allocwarp GPR state.
- `OP_LOAD_REGISTER` is the allocwarp-side control op for seeding a 64-bit GPR immediate across a contiguous lane range. Python exposes it as `LoadRegisterM` in `python/dae/instructions.py`.
- `RepeatM` now exposes allocwarp's base-register field through `arg` low bits. To preserve previously loaded GPR state, emit the repeat with an empty init range (`reg_end == reg`) so allocwarp skips the delta/acc reset and keeps the old value alive.
- `python/dae/tma_utils.py` now lets `ToRepeatedCordAdapter(..., persistent=True)` emit that zero-init-free guard directly around a single TMA-backed instruction.
- `CC0` and `CC0_ROW_BYTES` arm the following allocation instruction as a one-step repeat target by setting `MVC_GPR32_LOOP_COUNTER = 1`, `MVC_GPR32_LOOP_START_PC = pc + 1`, `MVC_GPR32_BASE_REG = 0`, `MVC_GPR_DELTA = 0`, and `MVC_GPR_ACC` to the token row offset on lane `0`.
- For token-varying embedding and copy loads, emit `RepeatM(token_delta) -> CC0 -> load.jump()`. The pre-`CC0` repeat advances the token source pointer across outer iterations, and `CC0` itself provides the one-step repeat context for the following load.
- `LoopM(..., reg=...)` already distributes loop control over different lanes because each allocwarp lane owns its own `gpr_32` state. Distinct loop registers therefore do not require an expanded shared jump-counter array in the runtime.
