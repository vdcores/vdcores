# Allocwarp Lane Predicate

The loop-carried accumulator update in [`include/dae/pipeline/allocwarp.cuh`](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh) uses:

```c++
__builtin_assume(di.gpr_32[MVC_GPR32_BASE_REG] >= 0);
__builtin_assume(di.gpr_32[MVC_GPR32_BASE_REG] < 32);
__builtin_assume(reg_offset >= 0);
__builtin_assume(reg_offset < 32);
if (lane_id >= di.gpr_32[MVC_GPR32_BASE_REG] && lane_id <= reg_offset)
  di.gpr[MVC_GPR_ACC] += di.gpr[MVC_GPR_DELTA];
```

On the project’s CUDA 13.0 `nvcc` toolchain, a minimized repro of this pattern showed:

- the assumptions changed the compare domain from signed to unsigned in PTX and SASS;
- PTX changed from `setp.ge.s32` / `setp.le.s32` to `setp.ge.u32` / `setp.le.u32`;
- Hopper SASS changed from generic `ISETP.GE/GT.AND` forms to `ISETP.GE/GT.U32.AND`;
- instruction count and structure did not improve beyond that.

Current implementation:

```c++
const int base_reg = di.gpr_32[MVC_GPR32_BASE_REG];
__builtin_assume(base_reg >= 0);
__builtin_assume(base_reg < 32);
__builtin_assume(reg_offset >= base_reg);
__builtin_assume(reg_offset < 32);
if ((unsigned)(lane_id - base_reg) <= (unsigned)(reg_offset - base_reg))
  di.gpr[MVC_GPR_ACC] += di.gpr[MVC_GPR_DELTA];
```

Practical takeaway:

- `__builtin_assume()` can help the compiler pick unsigned range reasoning here, but it did not collapse the predicate into a cheaper lane-mask sequence in the tested build.
- Rewriting the range test as a single unsigned interval compare did improve the minimized PTX/SASS sequence compared with the original `lane >= base && lane <= end` form.
- This rewrite depends on the invariant `reg_offset >= base_reg`; in `allocwarp` that follows from `reg_offset = pc - loop_start_pc + base_reg` on the repeat/jump path.
- An explicit synthesized bitmask was worse in the minimized Hopper test and is not preferred unless the mask is reused elsewhere.
- The same idea also helps half-open lane ranges. In the minimized CUDA 13.0 Hopper test, `lane >= start && lane < end` lowered better when rewritten as `((unsigned)(lane - start) < (unsigned)(end - start))`.
- The pipeline code now uses shared helpers from [`virtualcore.cuh`](/home1/11362/depctg/vdcores/include/dae/virtualcore.cuh) for both closed and half-open lane-range predicates so future pipeline code can reuse the same codegen-friendly form.
