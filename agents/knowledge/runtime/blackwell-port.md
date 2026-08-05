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

## Llama-8B Framework Task Baselines

The Blackwell single-token comparison uses the exact installed framework paths,
not generic stand-ins. vLLM 0.23.0 and SGLang 0.5.12.post1 both use
unquantized `F.linear` with fused QKV and fused gate/up projections. Their
default Llama MHA decode paths both call FlashInfer TRTLLM batch decode; vLLM
uses FlashInfer 0.6.12 with page size 16 and the actual maximum sequence,
whereas SGLang uses FlashInfer 0.6.11.post1 with page size 64 and the configured
131072-token model maximum.

At BF16 batch 8 on GB200, the production-shaped task measurements show:

- VDCores KV GEMV is 4.352 us, about 18% faster than either framework's
  shape-matched component probe.
- VDCores M64 Q/O and down GEMVs are 7.216 and 22.768 us, 22-23% and 18-20%
  behind the framework component probes.
- The exact tile-packed, two-epoch M64 LM head is 174.432 us versus about
  149.7 us for either framework. M128 reaches 161.472 us in isolation but
  requires fold-2 additive output plus a synchronized 2 MiB per-token clear,
  so it is not the retained minimal repeated-decode path.
- Aligned 128-bit shared-memory packs and register reuse reduce VDCores
  RMSNorm to 2.496 us at B8, matching vLLM's 2.490 us but trailing SGLang's
  2.100 us. Three-way 2048-element sharding reduces the materialized
  6144-wide SwiGLU prefix from 3.904 to 2.560 us, ahead of vLLM's 2.682 us
  and SGLang's 2.919 us.
- VDCores argmax is 7.360 us, 36-37% faster than vLLM/SGLang.
- VDCores B8 decode attention is 4.032 us at S128 and 8.960 us at S512,
  versus 4.679/5.579 us for vLLM and 5.556/5.656 us for SGLang. The retained
  short-context kernel leads both frameworks; split reduction remains the
  long-context bottleneck.

Do not sum these isolated values to explain TBT. The VDCores Llama path fuses
K/V cache writes, residual reductions, and the register-forwarded MLP tail and
overlaps auxiliary-SM down-projection work inside one persistent megakernel.
Use `benchmarks/blackwell_framework_tasks.py` and
`benchmarks/blackwell_vdcores_tasks.py` for the exact comparison methodology.

The elementwise task search rejected direct global-memory SwiGLU/RMS paths,
port-1 weight or activation loads, and a two-SM RMS reduction because their
queue/global synchronization cost exceeded TMA staging. The retained exact
Llama build remains spill-free. Its 24-SM sharded SwiGLU placement lowers the
128-step median from 401.380 to 393.859 ms (3.077 ms TBT) while preserving
four-token exact greedy correctness.

## SM100 Decode Attention Pipeline

- The retained head-dim-128 path drains the four live GQA score rows from TMEM
  into compact FP32 shared storage with one warp, uses all four compute warps
  for row-parallel online softmax, overwrites the same special-slot region
  with swizzled BF16 probabilities, and consumes them with shared/shared UMMA
  PV. It supports both one and multiple KV tiles and rescales prior TMEM output
  when the online maximum changes.
- The direct epilogue converts TMEM output into aligned BF16x4 global stores.
  Raw-address writeback must publish `1U << slot_id` to C2M because M2C carries
  a special-slot index while the store queue consumes a one-hot slot mask.
- The exact 11-operator Llama image compiles at 200 registers, 9 barriers, a
  96-byte stack frame, and zero spills. Its 128-step median is 383.258 ms,
  or 2.994 ms TBT, versus 3.335 ms for the stricter vLLM decode estimate and
  3.820 ms reported by SGLang.
- Final isolated attention medians are 3.424 us for B1/S64 and 3.904, 3.936,
  3.936, and 4.032 us for B1/B2/B4/B8 at S128. All lead the exact installed
  vLLM/SGLang paths. At S512 the retained split choices measure 6.112, 6.208,
  6.688, and 8.960 us; they improve the prior VDCores path but still trail the
  framework kernels.
- Correct but rejected experiments include CUDA-core non-UMMA SDPA (7.488 us
  at B1/S64), a cross-CTA atomic fused reducer (6.208 us at B1/S512), and a
  direct-global reducer epilogue (no median gain). Keep these out of the final
  task implementation unless the synchronization design changes materially.
