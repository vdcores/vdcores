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
- The two-epoch LM-head projection assigns four disjoint M128 output tiles to
  each of 128 SMs and reuses each B tile four times. Its diagnostic
  materialized-output path measures 147.840 us versus 149.703 us in vLLM and
  149.781 us in SGLang, with exact isolated BF16 agreement. Production keeps
  the same TMEM accumulation but reduces the epilogue directly to compact
  argmax records.
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

- The earlier materialized-output diagnostic uses raw-address descriptors,
  which bypass the shared-slot allocator and may be issued ahead of compute.
  Consecutive direct-output tasks therefore need distinct special slots, and
  those slots must also be distinct from following tasks.
- C2M completion is a one-hot mask. Slot 31 sets the sign bit of the queue's
  `int`, so only `-1` is an invalid-allocation sentinel; a generic `val < 0`
  check incorrectly drops a valid slot-31 completion and deadlocks its barrier.
- Production emits one 16-byte value/absolute-index record per LM-head
  task/token, joins only the 128 compute threads before C2M publication, and
  reduces all 256 records on one SM per token. It no longer allocates or
  rereads the 2 MiB padded logits tensor.
- The exact 11-op fused Llama image is spill-free at 128 registers, 9 barriers,
  and a 96-byte stack. Build it with
  `benchmarks/blackwell_llama8b_fused_argmax.ops`; four-token greedy output
  matches Hugging Face exactly.
- A fused/materialized/fused 500-step S128 sandwich averages 2.899672 ms fused
  versus 2.911456 ms materialized, an 11.784 us reduction.

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
which dispatch each decode step. Configure one framework engine per context;
sharing an S512-capacity engine changes the S64/S128 result. The strict B8
VDCores/vLLM/SGLang medians are 2.906/2.842/3.381 ms at S64,
2.900/2.816/3.312 ms at S128, and 2.931/3.499/3.683 ms at S512. The frameworks
use `C - 1` input tokens and the first-to-second output interval, so the timed
decode sees exactly `C` KV tokens without prefill. VDCores trails vLLM by
2.2%/3.0% at S64/S128, then leads it by 16.2% at S512; it leads SGLang by
12.5-20.4% throughout. Keep the timing-scope difference explicit rather than
relabeling framework token intervals as kernel-internal counters. The benchmark
rejects multi-context invocations so engine capacity cannot leak across rows.

The S128 bottleneck audit separates kernel and schedule effects. Production
M64 output/down GEMVs measure 7.456/21.760 us for exact BF16 B8 shapes, versus
5.949/19.367 us in vLLM and 5.949/18.405 us in SGLang. M128 improves them to
6.688/20.480 us but does not close the gap. In the full VDCores layer, down
compute is 21.000 us, reduction completion plus the next RMS is 2.000 us, and
Q clear is 0.250 us. Thus reduction/RMS pipelining is already effective and
clear is already negligible on auxiliary SMs; projection execution is the
actionable task bottleneck. A strict vLLM Nsight trace contains 385 graph nodes
plus 11 sampler nodes, with 3437.791 us summed kernel work in a 3332.255 us
span. Treat the 105.536 us overlap as graph-topology evidence, not an absolute
untraced timing, because Nsight inflates kernel durations.

The spare-SM follow-up tested schedule changes with both task timers and the
full internal counter. "Group2 M64" is a two-output M64 CTA: 32 output-row
pairs times four K partitions form 128 projection tasks. The count is a clean
factorization, not a hardware limit; the remaining 24 of 152 SMs are available
for auxiliary work. Group2 improves isolated K4096 from 7.552 to 6.752 us but
is neutral in a prefix and either violates the 32-layer logits threshold or
regresses the passing 1536-row subset to about 2.951 ms, so its new opcode was
removed. An all-152-SM LM head also measures about 2.951 ms, balanced MLP
placement measures 3.230 ms, and early Q clear loses in a paired A/B. Only the
fused LM-head argmax produces a repeatable end-to-end win.

Profiling has four complementary scopes: resident cross-SM `globaltimer` for
VDCores task probes, CUDA-event graph replay for like-shaped framework tasks,
temporary megakernel frontier/per-SM markers for critical-path overlap, and
Nsight for framework graph topology. Parameter changes must be justified by
the first three and then qualified by exact-token end-to-end inference; Nsight
durations are not substituted for uninstrumented latency.

The embedding RMS stage deliberately remains two operators. RMSNorm on SMs
0-7 overlaps an 8 KiB residual copy on SMs 64-71. A dual-output RMS prototype
reused its cached BF16 input and removed the duplicate load, but serialized a
second shared-memory write and writeback. In a same-process alternating B8
test, the overlapped pair measured 2.464 us versus 2.688 us for dual output;
both outputs were correct. The 9.1% regression means the prototype should not
be restored unless the writeback pipeline changes materially.

## Fine-Grained MLP Readiness

The retained Blackwell Llama schedule no longer puts the entire 6144-wide MLP
prefix behind one producer barrier. Gate/up output, SwiGLU, and the low-K down
projection use three 2048-wide readiness chains. Each chain releases after its
32 gate and 32 up tiles, lets eight token-local SwiGLU tasks run, and then
releases the matching K2048 down-projection folds. The output tiles, K folds,
SM placement, compute opcodes, and BF16 reduction work are unchanged; only the
frontier granularity and ready-shard issue order differ. The coarse schedule
remains available with `VDCORES_FINE_MLP_BARRIERS=0` for paired comparisons.

Internal-counter medians on the same exact 11-op image were:

| B8 context | Coarse MLP barriers | Three shard barriers | Delta |
| ---: | ---: | ---: | ---: |
| 64 | 2.808928 ms | 2.766432 ms | -42.496 us (-1.51%) |
| 128 | 2.811984 ms | 2.769376 ms | -42.608 us (-1.52%) |
| 512 | 2.844864 ms | 2.797472 ms | -47.392 us (-1.67%) |

The 501-iteration S128 A/B/A sandwich is job
`20260806T212444Z-2221138`; the S64/S512 sweep is job
`20260806T212616Z-2225221`. A detailed single-token tensor comparison passed
all thresholds in `20260806T212343Z-2218735`, and four control-flow greedy
tokens exactly matched Hugging Face in `20260806T212301Z-2216505`. Because
this is a Python schedule change, the selective runtime remains at 128
registers, nine barriers, a 96-byte stack, and zero spills. This is the useful
VDCores analogue of dependent-kernel overlap: publish a coarse-enough shard to
another compute group without paying a command transition for each epilogue.

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

## TMEM Overlap Feasibility

The 2026-08-06 idea-verification round tested whether an outer,
memory-VCore-like warp can own explicit TMEM copy/drain work while the four
compute warps continue independently. The low-level mechanism is viable. In a
correctness-checked UTCCP/TMEM drain probe, a disjoint helper reduced the
internal-counter span by 6-10%; making the producer warp drain its own pipeline
did not overlap and regressed the 4 KiB case.

| TMEM payload | Serial dedicated | One helper | Two helpers |
| ---: | ---: | ---: | ---: |
| 4 KiB | 468.813 ns | 429.750 ns (-8.34%) | 428.750 ns (-8.53%) |
| 8 KiB | 820.812 ns | 745.500 ns (-9.29%) | 752.562 ns (-8.31%) |
| 16 KiB | 1426.125 ns | 1303.687 ns (-8.60%) | 1276.625 ns (-10.41%) |
| 32 KiB | 2820.937 ns | 2636.812 ns (-6.56%) | 2542.813 ns (-9.88%) |

All 24 size/schedule cases had zero hash and final-output mismatches in job
`20260806T172836Z-1324881`. Use one helper for payloads through 8 KiB and only
consider two for 16 KiB or larger. This is a mechanism result, not permission
to add a helper to every task: the helper must be disjoint from the producer
and there must be independent work covering its synchronization cost.

Reducing GEMV UMMA completion waits was beneficial inside an isolated compute
window but harmful to the real VDCores load/slot pipeline. For M4096/N8,
batching two or four K256 commits improved representative K2048/K4096/K8192
probes by roughly 2-4%, but commit-4 regressed K14336 from 22.336 to 26.048 us.
On production-shaped tasks, commit-2 changed down projection from 22.304 to
22.912 us, a wide projection from 24.992 to 25.696 us, and a three-epoch
prefix from 20.064 to 20.320 us. A two-completion-barrier pipeline changed
7.552 to 7.936 us. Holding shared-memory operand slots across extra K tiles
prevents the memory VM from refilling them; early slot release is more valuable
than removing these local waits. All GEMV variants were removed.

Attention exposed the same boundary. An isolated split-P probe, modeled after
FlashAttention-4's early publication of the first three quarters of P, reduced
1.362400 to 1.306675 us (-4.09%) with identical output in job
`20260806T172903Z-1327697`. On the actual four-warp attention task, however,
the same-warp variant raised B8 latency from 3.328/4.672/7.136 us to
3.424/4.768/7.424 us at S128/S256/S512. A real fifth tensor sidecar warp was
then added experimentally, with a double-buffered command mailbox, named
barrier, UMMA mbarrier, and TMEM proxy fences. It was exact and spill-free, but
the repeated internal-counter results were:

| B8 context | Four-warp baseline | Command sidecar | Delta |
| ---: | ---: | ---: | ---: |
| 128 | 3.360 us | 3.488 us | +0.128 us |
| 256 | 4.640 us | 4.768 us | +0.128 us |
| 512 | 7.104 us | 7.328 us | +0.224 us |
| 1024 | 11.360 us | 11.712 us | +0.352 us |
| 2048 | 19.872 us | 20.544 us | +0.672 us |

The approximately 42 ns per-KV-block command transition is on the critical
path. The retained attention kernel therefore remains unchanged. A worthwhile
next prototype is a persistent, coarse staged state machine: give the outer
tensor VCore ownership of a stage ring and let softmax/correction warps signal
stage readiness directly, rather than submitting QK, PV-head, and PV-tail as
three commands per block. It should preserve the split-P dependency cut and
use separate TMEM accumulator regions so a helper can drain one stage while
UMMA fills another.

### Two-bank UMMA/TMEM pipeline

Multiple UMMA groups can be in flight inside one compute task when they target
disjoint TMEM columns and have independent completion barriers. An exact
M128N8K64 probe seeded banks 0 and 1, waited for bank `n`, started its 4 KiB
TMEM-to-register drain, and submitted group `n+2` into the freed stage. The
drain therefore overlaps UMMA filling the other bank; it does not try to read
the bank still owned by UMMA. Job `20260806T183731Z-1618074` measured:

| UMMA operations per group | Serial | Two-bank overlap | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 444.719 ns | 385.688 ns | -13.27% |
| 2 | 529.625 ns | 385.563 ns | -27.20% |
| 4 | 733.687 ns | 540.937 ns | -26.27% |
| 8 | 1154.313 ns | 938.781 ns | -18.67% |

All final values, pair ordering checks, and repeat checks were exact over
10,001 iterations. A phase-accurate LM-head probe covering 1,002 M128 output
tiles and nine rounds reduced the internal span from 91.216 to 69.869 us
(-23.4%) with exact CPU and 1,001-repeat checks in job
`20260806T184153Z-1636614`. Paired projection probes also confirmed a smaller
but real opportunity when the whole schedule exposes independent work: with
384 threads, gate/up improved from 4.984 to 4.657 us (-6.55%) and down from
5.713 to 5.298 us (-7.26%) in job `20260806T183835Z-1621647`. A 352-thread
form remained exact but reduced those gains to 4.26% and 3.86%; extra sidecar
warps are not automatically better.

The fine-grained integration into the retained grouped M128 GEMV did not win.
That task can only drain an output bank after all of its K tiles have
accumulated, so the prototype overlapped only the four final epilogues while
adding two completion stages and compute-group transitions to every task. A
same-image internal-counter sweep in job `20260806T185434Z-1686708` was:

| K | Retained grouped task | Final-epilogue pipeline | Delta |
| ---: | ---: | ---: | ---: |
| 512 | 9.344 us | 9.568 us | +2.40% |
| 1024 | 22.272 us | 22.528 us | +1.15% |
| 2048 | 40.384 us | 40.448 us | +0.16% |
| 4096 | 75.968 us | 76.128 us | +0.21% |

Moving the two stage barriers to persistent runtime state, removing task-local
initialization, and skipping the two producer-free tail barriers lowered the
selective image from 48 to 45 registers with no spills. It still measured
9.600 versus 9.376 us at K512 and 76.128 versus 75.968 us at K4096 in job
`20260806T190102Z-1708233`. The experimental opcode, schedule reorder, and
extra runtime barriers were therefore removed.

The restored direct4+terminate image uses 32 registers, one barrier, an
80-byte stack, and no spills. Its exact K4096 validation in job
`20260806T190514Z-1726220` measured a 76.000 us median over 501 iterations.

The useful design boundary is coarse ping-pong, not a final-epilogue patch:
assign a persistent CTA successive independent output tiles, let UMMA reduce
tile `n+1` into one TMEM bank while a disjoint role drains and writes tile `n`,
and amortize each stage transition over a complete K reduction. This best fits
LM head or another output-tile-parallel phase with spare SMs. It must be
compared against the retained grouped task's B-tile reuse, because duplicating
or retaining B traffic can erase the 4-7% task-level opportunity even though
the underlying overlap is valid.

After removing every experimental opcode/runtime hook, the selective
production image returned to 63 registers, 9 barriers, a 96-byte stack, and no
spills. Fresh B8 internal medians were 3.264/4.672/7.200 us for unsplit
S128/S256/S512 and 6.080 us for S512 split-2; mean-relative error remained at
or below 0.277%. The corresponding cooperative jobs are
`20260806T181310Z-1514927`, `20260806T181341Z-1516257`,
`20260806T181415Z-1518677`, and `20260806T181443Z-1520101`.

TMEM synchronization must follow the tcgen05 proxy protocol. `wait::ld` and
`wait::st` complete accesses for their issuing thread; a cross-warp ownership
handoff additionally needs `tcgen05.fence::before_thread_sync`, a real named
barrier or mbarrier transition, then `tcgen05.fence::after_thread_sync` in the
consumer. Do not substitute `threadfence` or a full-CTA barrier that the memory
warps do not join. This follows the NVIDIA PTX
[fifth-generation TensorCore memory model](https://docs.nvidia.com/cuda/parallel-thread-execution/#memory-consistency-model-for-5th-generation-of-tensorcore-operations)
and CUTLASS's staged
[`PipelineUmmaAsync`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/pipeline.html)
model. FlashAttention-4's current SM100 implementation is the reference for
the persistent role split and early-P publication; see its
[SM100 forward kernel](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py).
It is not evidence that a per-phase VDCores mailbox is free.
