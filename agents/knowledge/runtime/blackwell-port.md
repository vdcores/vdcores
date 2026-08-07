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

### Second compute role for LM-head epilogues

The follow-up tested compute-compute overlap separately from the retained
fine-grained schedule barriers. Two consecutive 128-SM LM-head epochs used
TMEM columns 0--31 and 32--63 as a ping-pong pair. The original four compute
warps accumulated epoch `n+1` while a persistent sidecar drained epoch `n`,
performed the BF16 argmax epilogue, and published the normal C2M completion.
The memory warps never joined the handoff. Ownership used
`tcgen05.fence::before_thread_sync`, compute-only named barriers, and
`tcgen05.fence::after_thread_sync`; no thread fence or whole-CTA task barrier
was substituted.

A single 32-thread "light compute" warp is not a valid replacement for the
four-warp epilogue role. Replaying the four logical CUTE TMEM slices from one
physical warp produced 1,534 wrong partial indices out of 2,048 at K512 in job
`20260806T215320Z-2302814`. The physical warp cannot impersonate the other
three TMEM datapath owners. The correct form therefore needs one complete
additional warpgroup, making the experimental CTA 384 threads.

The viable prototype kept its two-command mailbox after the 16 KiB attention
scratch region in the tail of the existing 212 KiB dynamic allocation. Its
sidecar device function was no-inline so the main interpreter did not inherit
the epilogue instruction footprint. Following the PTX `setmaxnreg` contract,
all four sidecar warps released their register tail to 64 registers while
dormant and reacquired 128 only after the first bank became ready. The exact
11-op images were spill-free: the retained 256-thread entry remained at 128
registers, 9 barriers, and a 96-byte stack; the 384-thread entry used 128
registers, 16 barriers, and the same stack.

The isolated two-epoch task did overlap successfully:

| Two LM-head epochs, 128 SMs | Retained epilogue | Four-warp sidecar | Delta |
| --- | ---: | ---: | ---: |
| K512 | 24.480 us | **23.584 us** | -0.896 us (-3.66%) |
| K4096 | 148.864 us | **148.352 us** | -0.512 us (-0.34%) |

Both variants produced identical partial indices. The 501-iteration jobs are
`20260806T224419Z-2428936` and `20260806T224447Z-2430036`. A fresh full-model
run also passed all tensor thresholds and the exact Hugging Face token in job
`20260806T222502Z-2378848`.

The full unprofiled S128 schedule did not retain that task-level gain. A
same-process A/B/A compared each sidecar pass with the mean of its two retained
neighbors:

| Sidecar pacing before LM head/reducer | Retained mean | Sidecar | Delta |
| ---: | ---: | ---: | ---: |
| 0 cycles | **2.768432 ms** | 2.799520 ms | +31.088 us (+1.12%) |
| 32 cycles | **2.755728 ms** | 2.761792 ms | +6.064 us (+0.22%) |
| 64 cycles | **2.757760 ms** | 2.761568 ms | +3.808 us (+0.14%) |
| 128 cycles | **2.768112 ms** | 2.791744 ms | +23.632 us (+0.85%) |

These are jobs `20260806T224531Z-2431915`,
`20260806T224902Z-2439720`, `20260806T225113Z-2442944`, and
`20260806T225326Z-2448617`. Small producer delays reduce the regression by
letting the already-running memory core lead the next M2C join, but no
unprofiled setting wins.

Temporary LM-head frontier stores selected a faster timing phase and made the
instrumented sidecar appear 7.776 us faster end to end. That result is not a
qualification measurement because the profiling writes change the schedule.
Its useful breakdown is diagnostic: pre-LM work changed from 2604.512 to
2592.256 us, the paired LM head from 148.800 to 147.200 us, and the reducer/
termination tail from 10.672 to 17.376 us in job
`20260806T224146Z-2422398`. The longer tail and the phase sensitivity explain
why the isolated drain saving does not transfer reliably.

The sidecar entries, mailbox, pacing hooks, and temporary benchmarks were
removed. The retained lesson is narrower: coarse two-bank TMEM overlap is
real, but adding a dormant warpgroup to the whole resident VDCores CTA is not
free. A future compute-compute role split must amortize that CTA-shape cost
across multiple stages, or repurpose an existing resident role, rather than
attach four warps only for the final LM-head epilogue. This also follows the
[PTX `setmaxnreg` warpgroup rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)
and the coarse specialized roles in
[FlashAttention-4 SM100](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py).
The restored unprofiled image passed the runtime smoke test in job
`20260806T225825Z-2457622` and measured a 2.765856 ms S128 median over 501
iterations in job `20260806T225857Z-2458535`.

### Whole-token compute-frontier profiling

An opt-in `VDCORES_STAGE_PROFILE=1` mode records compute-warpgroup arrival
times in the runtime's existing per-SM event buffer. `OP_PROFILE_EVENT`
synchronizes only the four compute warps before thread 0 reads `globaltimer`;
the independent memory warps never join that barrier. The marker opcode is
included only by `benchmarks/blackwell_llama8b_stage_profile.ops`, so the
11-op production image and its dispatch path are unchanged. Marker runs are
diagnostic rather than qualification measurements: detailed markers changed
the S128 median from 2.765440 ms unprofiled to 2.793120 ms.

S64, S128, and S512 traces showed the same final-layer topology. The main
tail is deterministic placement, not a random slow block:

- Gate/up prefix work on SMs 0--135 reaches a 40--50 us frontier, while the
  second shard-2 up-projection chunk serializes on SMs 136--151 and reaches
  66.7 us. Shard-local barriers let useful down-projection work overlap this
  tail, so the spread is not itself the full critical-path cost.
- The six low-K down-projection placements reach 77.0 us. The following
  high-K ranges converge at about 83 us; the late physical SM IDs reflect the
  deliberate balance between early shard work and later output ranges.
- The two LM-head epochs reach about 234 us. Their max-minus-median frontier
  spread is only about 2--3 us and the tail SM IDs move between runs, so there
  is no evidence yet that a dynamic work queue would repay its atomic and
  interpreter costs.

The profiling jobs are `20260806T231633Z-2498393` (S128 stage),
`20260806T231820Z-2501831` (S64), `20260806T231859Z-2503092` (S512), and
`20260806T232129Z-2507589` (detailed S128 task parts). The profiling image
remained spill-free at 128 registers, nine barriers, and a 96-byte stack, and
the full-model correctness run `20260806T231739Z-2500732` matched every
existing tensor threshold and the exact reference token.

Use frontier timestamps, not per-SM marker-to-marker duration alone, to find
the critical block. A compute warpgroup that arrives early at a downstream
barrier reports a long local duration while waiting, but is not the tail that
releases the stage. Prefer static dependency and placement changes while the
tail is repeatable; prototype dynamic task dispatch only if a material,
run-varying straggler remains after those changes.

### Cross-stage down-projection interleave

The retained whole-token optimization uses the existing shard-local MLP
barriers to schedule independent work across a stage boundary. After down
shards 0 and 1 have started, 48 SMs that would otherwise wait for shard 2
compute the first 768 output rows of the high-K `(6144, 8192)` slice. Shard-2
low-K work then runs, followed by the remaining disjoint high-K output rows.
The factorization still contains exactly 152 high-K reduction tasks and emits
the same `bar_layer` count; it changes only task order and physical placement,
not a compute or memory opcode. `VDCORES_INTERLEAVE_DOWN_HIGH=0` retains the
old ordering for A/B measurement.

The final-layer profile moved the next-RMS frontier from about 82.4 us to
81.3 us. The exact, profiling-free 11-op image measured:

| Fixed context | Shard-first order | Interleaved order | Delta |
| ---: | ---: | ---: | ---: |
| 64 | 2.767072 ms | **2.741056 ms** | -26.016 us (-0.94%) |
| 128 | 2.769824 ms | **2.742272 ms** | -27.552 us (-0.99%) |
| 256 | 2.764256 ms | **2.738432 ms** | -25.824 us (-0.93%) |
| 512 | 2.810528 ms | **2.781408 ms** | -29.120 us (-1.04%) |

S128 uses 1,001 internal-counter samples; the other rows use 501, all after
five warmups. Baseline/retained jobs are respectively
`20260806T235208Z-2574218`/`20260806T235246Z-2575347` (S64),
`20260806T235038Z-2571740`/`20260806T235115Z-2572444` (S128),
`20260806T235323Z-2576419`/`20260806T235401Z-2577975` (S256), and
`20260806T235438Z-2579290`/`20260806T235516Z-2580502` (S512).

A width sweep showed that 48 early tasks is a schedule boundary rather than a
generic "more is better" knob. In the diagnostic image, 40/44/48/52/56 early
tasks measured 2.773088/2.768896/2.751040/2.758592/2.758560 ms at S128. The
48-task range maps exactly to the large shard-2 placement on SMs 96--143;
other widths either leave part of that range idle or reorder the auxiliary
small-row tasks.

A separate attention-to-output proof of concept split eight KV heads into two
four-head barriers and factorized output projection into two K2048 halves,
without changing its 152-task total. It passed full correctness, but the
profiled output frontier remained 29.6 us versus about 29.8 us for the coarse
barrier. With the retained down interleave it improved S128 by only 2.208 us
(2.751680 to 2.749472 ms) in the diagnostic image, within the schedule's phase
sensitivity. The extra barriers, head-offset adapters, and split placements
were removed.

The complementary Q/K/V-to-attention experiment kept every compute task and
tensor coordinate unchanged while replacing the all-head dependency with
2-way, 4-way, and per-head readiness. Profiling-free S128 internal medians
were 2.737056 ms for the coarse production barrier, 2.738272 ms for two head
groups, 2.736576 ms for four groups, and 2.741216 ms for a correct per-head
variant. Separate Q and KV counters per head overflow the VM's 10-bit encoded
barrier-ID space after 32-layer resource expansion, so the legal per-head
probe used one combined Q+K+V counter per head. Both independent TMA load
ports must wait on that counter; waiting only on Q produced an invalid
2.721760 ms result because K/V raced their stores, which full tensor
correctness caught even though the final token happened to match. The valid
four-way delta was only 0.480 us and the other granularities regressed, so all
prototype code was removed. Jobs are recorded in
`.agentlog/2026-08-07-shared-barrier-candidates.md`.

Projection-to-RMS shard readiness was also rejected. The proof of concept
used eight 512-row counters, 64 shard sum-of-squares tasks, and eight RMS
finalizers per layer. A diagnostic caught an important dependency rule: both
the partial-vector load and the independently issued full hidden-row load must
wait for producer completion; gating only the partial vector lets the row load
capture a mixture of new and stale projection output. After correcting that
race, the 13-op image still regressed the profiling-free S128 internal median
from 2.743520 ms to 2.846784 ms at the output boundary and 2.855040 ms at the
down boundary. The added work cost 103.264/111.520 us per token, so a finer
eight-load finalizer could not plausibly recover the deficit. The full-model
output-boundary run also narrowly missed the existing tensor thresholds even
though its final token matched. All prototype code was removed; exact jobs and
errors are recorded in `.agentlog/2026-08-07-shared-barrier-candidates.md`.

RMS-to-projection K streaming was rejected after a two- and four-shard study.
Splitting the normalized row into two K2048 stores cost 0.352 us in the
standalone B8 RMS task, while four K1024 stores cost 1.184 us. A non-blocking
post-attention consumer schedule appeared 11.760 us faster at S128 and about
18 us faster at S64/S256, but that result was invalid: single-step tensor
correctness passed, whereas the second token diverged after layer 0. The RMS
operator remained correct across repeated standalone launches, barrier values
restored correctly, and stock coarse consumers passed with the same two-store
producer. The unsafe part was the split per-K consumer wait encoding. Replacing
it with allocator-side shard waits restored repeated-token correctness but
regressed S128 by about 189.5 us because it blocked memory issuance. Async
proxy fencing and gating both load ports did not rescue the fast path. All PoC
code was removed. The durable qualification rule is that a new resident shared
frontier must pass repeated-token correctness; one clean decode step can hide
a reuse race. Exact jobs are in
`.agentlog/2026-08-07-shared-barrier-candidates.md`.

LM-head epoch-to-reducer streaming was also rejected.  A spill-free prototype
gave the two 128-task projection epochs separate counters and let one reducer
task retain epoch-0 records in registers while its memory core waited for
epoch 1; it added neither reducer tasks nor a second block reduction.  Both
single-step tensor checks and four-token resident-loop correctness passed.
However, three same-image S128 coarse controls averaged 2.746752 ms, while two
staged runs on the original reducer SMs averaged 2.750416 ms (+3.664 us).
Spare-SM placements were slower even after moving them away from the SM128
barrier-restore task.  The LM-head tail available to hide was only 2--3 us,
less than the added pointer handoff and scheduling cost.  The prototype was
removed; exact jobs are in the shared-barrier agent log.  The restored exact
11-op production image passed four-token resident correctness and measured a
fresh 2.740256 ms S128 median over 501 profiling-free internal samples.

The production image remains at 128 registers, nine barriers, a 96-byte
stack, and zero spills. Runtime smoke job `20260806T234925Z-2569187` passed;
full-model job `20260806T234956Z-2570323` passed every tensor threshold and
matched reference token 24748 exactly. A two-step control-flow regression also
matched `[24748, 24748]` exactly in job `20260806T235703Z-2583774`. Dynamic
dispatch was not prototyped: the only run-varying tail was the roughly 2--3 us
LM-head spread, while the
material tail responded to deterministic static ordering. A queue would add
atomics and VM dispatch to a problem that did not exhibit queue-worthy
stochastic imbalance.

## Observer-owned memory-to-compute handoffs

Blackwell M2C handoffs no longer require all 128 compute threads to arrive on
every loaded-operand mbarrier. The load VCore is the sole participant and
publishes the phase after its TMA transaction completes; compute threads use
an acquire `mbarrier.try_wait.parity` to observe that phase. Queue-local parity
flips when the 32-entry ring wraps. This preserves the asynchronous-proxy
visibility guarantee without adding a compute-side arrival or named-barrier
broadcast.

The change matters because one Llama token executes about 4,530 M2C handoffs.
A same-source profiling-free S128 comparison improved from 2.734976 to
2.683328 ms, or 51.648 us / 1.89%. Full tensor correctness and exact
four-token resident-loop correctness passed, and the exact selective image
remains spill-free at 126 registers, nine barriers, and a 96-byte stack. A
legacy build remains available with `make m2c_legacy=1` for regression tests.

This optimization is deliberately limited to the 128-thread consumer. The
same parity polling on the single-lane allocator-to-load queues regressed, as
did moving TMA transaction publication into the allocator. Combining a
port-1 activation load and a port-0 weight load into one transaction phase was
also correct but 115.904 us slower than its paired observer-owned control.
The load warps therefore continue to own TMA transaction setup and completion;
only the compute warpgroup changed from participant to observer.

## Cross-task gate/up UMMA fusion

A paired M64 gate/up task tested whether sharing the activation load and
eliminating the register-slot SwiGLU handoff could expose useful cross-task
overlap. Gate and up accumulated in disjoint TMEM columns and the task emitted
the BF16-rounded SwiGLU result directly. Reusing one UMMA completion phase for
both accumulators requires all participating compute threads to observe each
phase before it can be reused; omitting that rendezvous deadlocked repeated
96/128-SM probes.

A warp-specialized follow-up made compute warp 0 the sole UMMA issuer and
parked the other three compute warps for the reduction. The parked warps
advanced their private M2C cursors over the 36 consumed operand messages and
joined only for the four-warp TMEM epilogue. This replaced 32 per-tile
rendezvous with three task-level compute barriers and passed repeated 128-SM
correctness. It still regressed the profiling-free S128 median from the
2.687360 ms mean of neighboring controls to 2.697600 ms (+10.240 us, +0.38%).
Sharing the activation was less valuable than preserving independent gate/up
projection progress, so the prototype was removed.

This experiment also exposed a compiler boundary constraint. Marking the
large task no-inline made the resident interpreter non-leaf and changed the
whole selective image from 126 registers/96-byte stack to 74 registers/176-
byte stack, corrupting even runs that did not dispatch the new opcode.
Force-inlining restored the spill-free control image. Future experimental
task handlers must check entry-function resources and disabled-path
correctness, not only the new handler's local resource report.

## Retained up-UMMA/SwiGLU overlap

The production MLP tail keeps gate and up as independent projections. Gate
ends in the existing `RegStore` special slot; the following up task queues all
of its operand loads before requesting that gate slot. After submitting the
final up UMMA group, all four compute warps consume the gate tile from shared
memory and evaluate its FP32 sigmoid while the tensor core owns the independent
up accumulator. The epilogue rounds up to BF16, multiplies by the retained
FP32 gate activation, and writes SwiGLU directly. There is no intermediate TMA
round trip, extra compute warp, or cross-SM dependency.

Placing the CUDA work under the final group is material. An initial first-group
form measured 25.088 us for the isolated 128-SM tail versus 25.024 us for the
three-task control because it delayed early A/B slot release and prefetch. The
late gate handoff measured 24.736 versus 24.896 us. In a 1,001-sample S128
sandwich, late-overlap runs measured 2.673408 and 2.674592 ms around a
2.679456 ms control, a 5.456 us / 0.20% gain. The exact 11-op production image
then measured 2.678304 ms and remains spill-free at 128 registers, nine
barriers, and a 96-byte stack.

Full S128 tensor validation passed all thresholds and token 24748 exactly in
job `20260807T084358Z-3316527`. Four resident tokens reused the late handoff
correctly in `20260807T084435Z-3317529`, and the minimal final image repeated
that result in `20260807T084806Z-3321002`. The retained schedule has exactly
two tail compute tasks: gate GEMV and up GEMV with overlapped SwiGLU.
