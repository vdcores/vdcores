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
- Four-output M128 Q/O and down GEMVs are 5.792 and 18.704 us. Their packed
  rank-4 reduction epilogue puts them 1.2-2.2% and 1.0-2.9% ahead of the
  vLLM/SGLang component probes.
- The retained two-epoch LM head assigns four disjoint M128 output tiles to
  each of 128 SMs, reuses each B tile four times, and drains four F32 TMEM
  accumulators directly to BF16 logits. It measures 147.840 us versus 149.703
  us in vLLM and 149.781 us in SGLang, with exact isolated BF16 agreement.
- A 128-thread RMS row uses all four compute warps, caches aligned 128-bit input
  and weight packs, reduces four warp partials through shared memory once, and
  broadcasts the final scalar within each warp. One row per SM measures
  1.920/1.952/1.984/2.080 us at B1/B2/B4/B8, ahead of vLLM throughout and
  ahead of SGLang at B2/B4; B8 is within 0.6% of SGLang. The task retains
  TMA/shared-memory memory ops rather than raw global addressing. Three-way
  2048-element sharding reduces the materialized
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

## Grouped LM-Head Pipeline

- Raw-address descriptors bypass the shared-slot allocator and may be issued
  ahead of compute. Consecutive direct-output tasks therefore need distinct
  special slots, and those slots must also be distinct from following tasks.
  The two logits epochs use slots 30/31; argmax uses slots 24--29.
- C2M completion is a one-hot mask. Slot 31 sets the sign bit of the queue's
  `int`, so only `-1` is an invalid-allocation sentinel; a generic `val < 0`
  check incorrectly drops a valid slot-31 completion and deadlocks its barrier.
- The final exact Llama image is spill-free at 202 registers, 9 barriers, and
  a 96-byte stack. Four-token greedy output matches Hugging Face exactly.
- Cooperative job `20260805T082334Z-4118216` measured 382.133 ms median for
  128 decode steps, or 2.985 ms TBT and 334.96 token-steps/s.

## Grouped Projection Reduction

- Four M128 output tiles reuse each staged B tile and accumulate in separate
  TMEM column ranges. The four BF16 epilogues total exactly 8 KiB, so one
  shared slot carries all of them.
- A rank-4 tensor map represents the four non-contiguous output quarters and
  one `cp.reduce.async.bulk.tensor.4d` publishes the complete slot. This avoids
  four allocator/store-queue transactions per worker task.
- At BF16 B8, the retained K4096 path is 5.792 us with 0.46% mean-relative
  error, versus 5.863/5.922 us for vLLM/SGLang. K14336 is 18.704 us with 0.46%
  error, versus 19.264/18.893 us.
- Two-output reduction (5.792/19.968 us), four separate output stores
  (6.896/18.656 us), 112-SM fold-14 (19.840 us), and 64-SM fold-8 (28.672 us)
  were explored and removed or superseded.
- Keep the grouped down GEMV as a standalone task, not in the production Llama
  schedule. Two fold-16 BF16 reduction epochs accumulated 47.5% hidden-state
  drift over 32 layers. A single B7 task avoided the second reduction but its
  29-load window deadlocked after the existing MLP pipeline. A phased B3/B4
  TMEM accumulator removed that window; group-4 still drifted 14.3%, while
  group-2 reduced drift to 3.32% but measured 21.888 us and changed a
  control-flow argmax. All phased variants were removed.
- After restoring the overlapped M64 down schedule, four-token greedy output
  again matches Hugging Face exactly. Cooperative job
  `20260805T094633Z-354598` measured 382.217 ms / 128 steps, or 2.986 ms TBT.

The elementwise task search rejected direct global-memory SwiGLU/RMS paths,
port-1 weight or activation loads, and a two-SM RMS reduction because their
queue/global synchronization cost exceeded TMA staging. The retained exact
Llama build remains spill-free. Its 24-SM sharded SwiGLU placement lowers the
128-step median from 401.380 to 393.859 ms (3.077 ms TBT) while preserving
four-token exact greedy correctness.

The final 128-thread RMS follow-up keeps the memory-op path and assigns one row
to each SM in the B1-B8 decode regime. In the minimal RMS+terminate image it
uses 68 registers, 9 barriers, and no spills; repeated B1/B2/B4/B8 medians are
1.920/1.952/1.984/2.080 us. The exact 12-op Llama image remains spill-free at
128 registers, 9 barriers, and a 96-byte stack. Four greedy tokens match the
Hugging Face reference, and job `20260805T145137Z-1748528` measures 377.306 ms
for 128 steps, or 2.948 ms TBT and 339.25 token-steps/s.

For the streaming deployment comparison, charge VDCores only for its internal
cross-SM megakernel span and retain launch/scheduler overhead for vLLM/SGLang,
which dispatch each decode step. A direct fixed-context B8 sweep measures
VDCores/vLLM/SGLang medians of 2.914/2.760/3.340 ms at S64,
2.907/2.769/3.356 ms at S128, and 2.945/3.513/3.645 ms at S512. The frameworks
use `C - 1` input tokens and the first-to-second output interval, so the timed
decode sees exactly `C` KV tokens without prefill. VDCores trails vLLM by
5.6%/5.0% at S64/S128, then leads it by 16.2% at S512; it leads SGLang by
12.8-19.2% throughout. Keep the timing-scope difference explicit rather than
relabeling framework token intervals as kernel-internal counters. Reproduce
the framework sweep with `benchmarks/blackwell_fixed_context_decode.py`.

The embedding RMS stage deliberately remains two operators. RMSNorm on SMs
0-7 overlaps an 8 KiB residual copy on SMs 64-71. A dual-output RMS prototype
reused its cached BF16 input and removed the duplicate load, but serialized a
second shared-memory write and writeback. In a same-process alternating B8
test, the overlapped pair measured 2.464 us versus 2.688 us for dual output;
both outputs were correct. The 9.1% regression means the prototype should not
be restored unless the writeback pipeline changes materially.

## Projection-to-RoPE Handoff

- `RegStore` followed by `RegLoad` is an on-SM shared-memory handoff, not a
  global-memory round trip. GEMV publishes its swizzled M64N8 epilogue slot,
  RoPE consumes that exact slot, and only RoPE requests the final TMA
  reduction/store.
- A fused SM100 M64N8 alternative rotates the GEMV epilogue after TMEM drains
  to shared memory and before the final TMA. Its raw table address uses special
  slot 32; `SchedGemvRope` applies the fixed-position byte offset to that raw
  address and the compute instruction selects the M64 half of the 128-element
  rotary row.
- At batch eight, the production Q fold-2 and K fold-4 shapes total 12.544 us
  with the two-operator handoff and 12.352 us fused. The 1.5% fusion gain is
  not large enough to displace the simpler, already-qualified two-operator
  schedule, so both tasks remain available and the two-operator path stays the
  default.
- The two-operator total is below component-matched vLLM/SGLang sums
  (14.095/12.738 us). These sums use separate Q/K projection probes plus the
  joint Q+K RoPE probe and must not be confused with the frameworks' fused-QKV
  scope.
- Keep the output tensor-map coordinates in `(N, M)` order. For M64 workers,
  `storeC.cord(0, m)` is correct; `storeC.cord(m, 0)` silently leaves every
  nonzero M tile unwritten.
- Computing the position-dependent table offset inside the fused task lowered
  its selective register count from 40 to 32 but regressed the measured Q/K
  spans by 3-4%. Keep fixed-position offsetting in the memory instruction;
  a future dynamic fused schedule should use the memory VM's counter-offset
  mechanism rather than adding scalar address work to every compute task.
- Adding the optional fused handler does not perturb the selected production
  image: it remains spill-free at 128 registers, nine barriers, and a 96-byte
  stack. Four-token greedy output stays exact and the qualified 128-step
  median remains 377.288 ms (2.948 ms TBT).

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
