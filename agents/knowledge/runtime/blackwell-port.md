# Blackwell Runtime Port

## Verified Hardware

- The current cluster workers identify as NVIDIA GB200, compute capability `10.0` (`SM100`), despite the cluster inventory's B300 label.
- Each GPU exposes `152` SMs, `64K` 32-bit registers per SM, `228 KiB` shared memory per SM, and a `227 KiB` per-block opt-in shared-memory limit.
- B300 is compute capability `10.3`; architecture-accelerated `sm_100a` and `sm_103a` cubins are not interchangeable. Validate an SM103 build on physical B300 before claiming support.

## Build Target

- `Makefile` and `setup.py` use `DAE_CUDA_ARCH`, defaulting to `100a`.
- Examples:
  - GB200/B200: `make DAE_CUDA_ARCH=100a pyext`
  - B300: `make DAE_CUDA_ARCH=103a pyext`
  - Hopper regression: `make DAE_CUDA_ARCH=90a pyext`
- Keep the virtual and real targets architecture-accelerated (`compute_100a` and `sm_100a`) when using Blackwell UMMA/TMEM instructions.

## Runtime Validation

- `src/torch_runtime.cu` validates launch width against the active device's `multiProcessorCount` instead of the former fixed `132`-SM limit.
- Dynamic shared-memory requests are checked against `sharedMemPerBlockOptin`; failed `cudaFuncSetAttribute` calls now propagate to Python instead of returning a false success.
- `tests/blackwell_runtime_smoke.py` exercises the actual VDCores memory protocol: 1D async load, compute-side copy, writeback, and exact global-memory verification.

Verified on 2026-08-04 through the cooperative GPU launcher:

```bash
gpu-cluster/scripts/mpi-run -n 1 -- \
  /home/azhpcuser/miniconda3/bin/python tests/blackwell_runtime_smoke.py

gpu-cluster/scripts/mpi-run -n 1 \
  --env DAE_SMOKE_SMS=152 --env DAE_SMOKE_COPIES=2 -- \
  /home/azhpcuser/miniconda3/bin/python tests/blackwell_runtime_smoke.py
```

Both the one-SM and all-152-SM cases passed exact comparison on GB200.

## Remaining Compute-Port Constraint

- A full SM100 compile of the pre-port task set reaches `255` registers and spills.
- Hopper `SM90_*` WGMMA atoms are not a Blackwell task implementation. Blackwell tensor-core tasks must use SM100 UMMA/tcgen05 with TMEM accumulation and an explicit TMEM-to-register/shared-memory epilogue.
