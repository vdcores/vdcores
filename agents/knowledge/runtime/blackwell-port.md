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

The later packed-ownership follow-up keeps those exact three counters and 24
tasks but removes a physical-placement hazard in the round-robin consumer
mapping.  SMs 128--135 now execute all shard-0 and shard-1 token tasks after
their shard-1 projection tail; SMs 144--151 execute shard 2.  Early
down-projection owners no longer encounter a local shard-2 wait before their
shard-0/1 work.  A same-image control/packed/control sandwich measured
2.617888/2.594016/2.618592 ms, a 24.224 us gain, while the restored exact
11-op production image measured 2.586304 ms at S128.  Single-step tensor
validation and four resident-token reuse both pass exactly.  The change adds
no opcode, hardware barrier, register, stack, or spill cost.

Moving either eight-token RMS stage to otherwise different physical owners
does not add useful overlap.  Post-attention RMS at bases 64/128 measured
2.587264/2.594400 ms, and next-layer RMS at bases 64/128 measured
2.584672/2.589696 ms, versus the fresh 2.586304 ms production result.  The
1.632 us best delta is noise-sized and the auxiliary placement can regress by
8.096 us, so RMS remains on SMs 0--7 and no placement selector is retained.

Likewise, separating packed SwiGLU shard 0 from shard 1 onto SMs 136--143
measured 2.588480 versus 2.589312 ms in a 501-sample pair, only 0.832 us.  A
main-path SM120--127 placement regressed by 80.384 us.  The current auxiliary
serialization is hidden behind the gate/up tail; no shard-0 placement selector
is retained.

A grouped two-phase version of the register-forwarded 8,192-element MLP tail
was also rejected.  It kept each fold-4 down task intact and selected one of
two K4096 readiness counters inside `SchedGemvPhasedActivation`; sharing the
already-KV-dominated head-7 Q/KV counter kept the static barrier IDs within
the VM's 10-bit field.  Full tensor and four-token resident correctness passed,
and the diagnostic image moved the next-RMS frontier about 2 us earlier, but
the exact 11-op control/phase/control medians were
2.588832/2.588128/2.590848 ms.  The 1.712 us gain is too small for the added
frontier, so the complete proof was removed.  A coarse tail-first reorder was
much worse at 2.881152 ms because it delayed the prefix gate producers.

Extending only selected elementwise tasks with parked helper warps also did
not improve the resident frontier.  One- and two-warp paired SwiGLU, a
192-thread late-shard SwiGLU, and a balanced 192-thread RMS all passed exact
S128 validation, but their same-image deltas were respectively +1.472 us,
-1.456 us (noise-sized), +0.624 us, and +0.480 us.  The helper slept on a
named barrier instead of polling, so this result is not the earlier full
eight-compute-warp interpreter penalty.  Per-stage markers showed that the
earlier paired work was hidden behind shard 2, while the wide tasks paid a
six-warp join for too little CUDA work.  Keep the 128-thread task variants and
do not add a light-compute role unless it can own a longer independent
epilogue chain without a recurring all-participant rendezvous.  After removal,
the exact 11-op production image returned to 96 registers, nine barriers, a
96-byte stack, and zero spills; full S128 correctness passed and the fresh
501-sample internal median was 2.588416 ms.

The retained cross-layer ownership follow-up moves Q fold 1 for heads 5--7
from late down-projection owners SM104--127 onto SM128--151.  Their K tasks
remain on SM104--127 and explicitly acquire the next-layer RMS barrier, so Q
and K overlap without changing task shapes or per-head reduction barriers.
The exact 11-op control/variant/control medians were
2.589344/2.576576/2.588384 ms, a 12.288 us gain.  Full S128 tensor checks and
four resident tokens pass exactly; the image remains 96-register,
nine-barrier, stack-96, and spill-free.  Keep the explicit K acquire whenever
Q is not colocated: the prior physical placement was also carrying a semantic
dependency.  Use `VDCORES_Q_FOLD1_AUX_TAIL=0` only for control measurements.

The retained companion placement moves V heads 3, 6, and 7 from their long-Q
owners onto the three K-only groups at SM104--127.  An explicit per-V RMS
acquire is required; omitting it produces an invalid apparent gain with stale
layer-1 V.  The valid exact-image control/variant/control medians were
2.581376/2.574240/2.582592 ms, a further 7.744 us.  Four resident tokens pass
exactly, no task or barrier count changes, and `VDCORES_V_K_TAIL=0` restores
the earlier map.

Interleaving the two LM-head weight epochs across complementary 64-SM groups
was exact but neutral: mixed/ordinary 501-sample medians were
2.574368/2.573696 ms.  Reversing which 512 MiB half each CTA sees first does
not eliminate a task, transfer, or frontier, so the split schedules were
removed.  Do not revisit epoch ordering without a mechanism that also removes
an epilogue or overlaps an independent consumer.
The restored exact schedule measured 2.568928 ms over 501 internal samples.

Stage-detail profiling must compare timestamps to one global layer origin,
not to each CTA's local `layer_start`.  The latter answers how long a CTA has
been active, but it can make a late-entering owner look artificially fast.
`VDCORES_STAGE_PROFILE_DETAIL` therefore reports both `frontier_us` (local)
and `absolute_us` (relative to the earliest layer start).  On the retained
QKV map, absolute Q completion is balanced at roughly 11.4--12.6 us; K heads
5--7 complete near 9.4 us while K heads 0--4 complete near 15 us.

That correction motivated a valid heads-5/6/7 V-placement check.  With the
same explicit RMS acquires as production, full S128 correctness passed, but a
same-image 301-sample control/variant/control sequence measured
2.578112/2.578880/2.579136 ms.  The variant lies inside the control drift and
is rejected.  Keep V heads 3/6/7; an earlier component timestamp alone does
not advance a converged per-head attention wave.

Absolute MLP timing also explains why moving the auxiliary prefix tail onto
apparently idle main CTAs is invalid optimization accounting.  The balanced
map reduced maximum prefix completion from roughly 62 to 57 us, but those
extra up GEMVs then ran before the same CTAs' register-forwarded gate/up tail.
The layer frontier moved from about 84 to 97 us and the profiling-disabled
301-sample median regressed to 2.835872 ms.  The proof was correct and was
removed; the present 192-prefix-task plus 256-tail-task map already minimizes
the maximum number of projection tasks per CTA.

The second load VCore does not create another LM-head consumption pipeline.
Sending only the activation to port 1 measured 2.573504 versus 2.573536 ms.
Splitting the last two of four weight groups across port 1 passed full S128
correctness, but a 501-sample control/variant/control comparison measured
2.573920/2.575264/2.571872 ms, a 2.368 us regression against the control
mean.  Both paths were removed.  The ordered M2C stream, slot retirement, and
one UMMA consumer remain the limiting path even with two TMA issuer warps.

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
- At batch eight, current 5,001-iteration Q fold-2 and K fold-4 probes total
  12.352 us with the two-operator handoff and 11.872 us fused. The fused
  epilogue is the Llama default; the two-operator schedule remains available
  with `VDCORES_FUSED_QK_ROPE=0` for diagnostics.
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
- The fused handler submits the final UMMA group before loading one RoPE
  coefficient pair per compute thread. The independent loads execute while
  UMMA owns the accumulator, and the pair stays live for both batch groups in
  the shared-memory epilogue. Moving those loads back after the completion
  wait changed isolated Q/K medians from 7.328/4.544 to 7.392/4.672 us, so
  this final-group overlap contributes 0.192 us/layer.
- A same-image 1,001-sample S128 two-op/fused/two-op sandwich measured
  2.684960/2.643360/2.686016 ms. The retained path saves 42.128 us (1.57%)
  against the control mean, materially more than the 15.36 us/token implied
  by isolated spans; removing the compute-task boundary also shortens the
  cross-layer critical path. Full S128 tensor validation matched token 24748,
  and four resident tokens produced `hello` four times across the KV128
  boundary.
- The final minimal 11-op image omits standalone RoPE, uses 96 registers,
  nine barriers, a 96-byte stack, and zero spills. Its 1,001-sample S128
  internal median is 2.642304 ms.

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

### Rejected same-accumulator K pipelines

A follow-up tested whether ordinary M64 GEMVs could hide their per-K256 UMMA
wait without introducing a second output tile. Combining two ordered K tiles
under one completion event improved an isolated M4096/K4096 fold-2 projection
from 6.944 to 6.688 us, but extended the lifetime of both shared operands.
That ownership cost reversed at K14336 (21.568 to 22.240 us) and at real
producer/consumer boundaries. Against a 2.646720 ms same-image S128 control,
V-only was neutral at 2.647520 ms, output projection regressed to 2.662080 ms,
MLP projections regressed to 2.693760 ms, and all K4096 scopes reached
2.711360 ms. Full tensor correctness still passed in
`20260807T094538Z-3372345`.

A true depth-two version attached alternating K tiles to two persistent UMMA
completion barriers, submitted tile `n+1` before waiting for tile `n`, and
released `n`'s operands while `n+1` remained in flight. It was also exact but
regressed the isolated K4096 projection from 7.008 to 7.456 us and raised the
selective image from 96 to 128 registers. Jobs were
`20260807T095448Z-3379841` and `20260807T095417Z-3379361`.

The ordinary GEMV critical path is producer/slot paced, not a bare UMMA wait.
Same-accumulator depth either holds operands too long or pays an extra phase
and live-state cost. Multiple-inflight integration should therefore use
disjoint TMEM/output work whose operands can retire independently, consistent
with the successful two-bank microprobe above. Both prototypes and the second
persistent barrier were removed. The restored 11-op image returned to 96
registers with zero spills and measured 2.634560 ms at S128.

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

The Q/K/V-to-attention frontier was later revisited after a resource-lifetime
audit found four barriers that are dead in the retained configuration: two
legacy RMS barriers and the two coarse SwiGLU barriers superseded by the three
shard-local pairs. Removing those allocations keeps separate Q and KV
counters for every KV head inside the VM's 10-bit encoded barrier-ID space
after 32-layer expansion. This fixes the main limitation of the earlier
prototype, which had to combine Q+K+V readiness into one counter per head and
therefore could not expose useful overlap.

The retained schedule preserves every projection tile and tensor coordinate.
For each KV head, its 16 Q contributors are split across matching low/high
eight-SM groups, while its eight V contributors run on the low group and its
eight K contributors run on the high group. Attention uses a head-major
placement, so the low group can consume that head as soon as its independent
16-count Q and 16-count K+V barriers reach zero; it no longer waits for the
other seven heads. Weight and activation loads remain on the existing
independent VCore ports, and both K and V are covered by the KV counter.

A same-image 1,001-sample coarse/per-head/coarse sandwich measured
2.635520/2.625248/2.636096 ms in jobs `20260807T102605Z-3405400`,
`20260807T102644Z-3406015`, and `20260807T102723Z-3406616`. Per-head readiness
saves 10.560 us / 0.40% against the control mean. Grouping two adjacent heads
measured 2.639360 ms (`20260807T102837Z-3407381`) and was rejected. Full S128
tensor correctness and token 24748 passed in `20260807T103049Z-3409224`; four
resident tokens crossed the KV128 boundary and produced `hello` four times in
`20260807T103132Z-3410039`. The exact 11-op image remains spill-free at 96
registers, nine barriers, and a 96-byte stack, and its final S128 median is
2.626368 ms (`20260807T103337Z-3411736`).

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

## Rejected stagger-aware tail/down pipeline

The post-overlap profile exposed a deterministic imbalance worth testing:
auxiliary SMs 136--151 finished the third MLP-prefix wave at about 65.0 us and
prefix SwiGLU at 67.9 us, while the main register-forwarded tail finished at
62.3 us. A proof of concept moved half of the final prefix-up wave to eight
main SMs and gave the four 2,048-element tail ranges separate counters. Every
high-K down fold was then represented as four K2048 schedules so unaffected
shards could start before the delayed main range. The tile set and all 448
down contributors were unchanged.

The arithmetic was correct, but the dependency representation was not cheap.
A same-image S128 comparison regressed from 2.675520 to 2.902272 ms (+226.752
us, +8.47%), and the profiled per-layer frontier moved from 81.376 to 89.824
us. Each factored K2048 schedule made the memory VM perform a separate
barrier-gated readiness acquisition; that repeated queue/issue cost dwarfed
the 5.6 us producer stagger. The prototype was removed. If tail K-shard
readiness is revisited, one grouped down command must retain its existing
operand pipeline and consume staged phases internally; Python-level task
factorization is the wrong boundary.

## Rejected direct8 LM-head epoch fusion

The two 65,536-row LM-head epochs were merged into one 131,072-row epoch so
each of 128 SMs held eight independent M128 output groups. The prototype used
a two-tile B-load interval to stay within the 32-step memory-program mask,
loaded the activation only once, removed one epoch frontier, and reduced 128
local argmax records instead of 256. Full S128 correctness passed in job
`20260807T100617Z-3388663`, and the selectable image remained spill-free at
100 registers, nine barriers, and a 96-byte stack.

The larger live TMEM/output state outweighed those savings. In one selectable
image, neighboring direct4 controls measured 2.641696 and 2.634016 ms while
direct8 measured 2.671424 ms over 1,001 internal-timer samples (jobs
`20260807T100655Z-3389198`, `20260807T100843Z-3390920`, and
`20260807T100734Z-3389738`). Direct8 therefore regressed by 33.568 us against
the control mean. The opcode, reducer, schedule switch, and manifest entries
were removed. Cross-task fusion should not increase the number of independent
TMEM accumulators held by one compute warpgroup unless a measured consumer is
available to overlap with them.

## Rejected LM-head epoch handoffs

Three two-operator handoffs attempted to compact epoch zero's four local
maxima and merge them with epoch one, replacing 256 global records with 128.
All kept the existing direct4 computation shape, unlike the rejected direct8
fusion.

The first version passed a 128-byte record through the dynamic register queue.
It was correct in job `20260807T104336Z-3420100`, but occupied one complete
dynamic slot throughout epoch one. Same-image control/handoff/control medians
were 2.628736/2.648256/2.624128 ms in jobs
`20260807T104413Z-3420430`, `20260807T104453Z-3421145`, and
`20260807T104532Z-3421723`, a 21.824 us regression. A static shared scratch
record removed that queue pressure but held four maxima in registers across
the boundary, raised the image to 100 registers, and measured 2.673952 ms
against a 2.626848 ms control (+47.104 us). Correctness passed in
`20260807T104932Z-3424780`; timing jobs were
`20260807T105015Z-3425122` and `20260807T105055Z-3425633`.

The final version left epoch zero in TMEM groups 0--3, filled epoch one into
groups 4--7, and drained all eight groups once. Full S128 correctness passed
in `20260807T105405Z-3428344`. Same-image control/TMEM/control medians were
2.622400/2.630944/2.624288 ms in jobs `20260807T105613Z-3430142`,
`20260807T105645Z-3430793`, and `20260807T105718Z-3431358`, a 7.600 us
regression. All three variants were removed. Extending resource lifetime
across a vocabulary epoch costs more than one compact publication and the
256-way reducer traffic, even when the handoff avoids both registers and a
dynamic queue slot.

## Retained phased attention-output consumption

The output projection now consumes attention output through three shared
frontiers: KV head 0, KV head 1, and heads 2--7. Its existing K512 activation
repeat is exactly one KV head, so each projection remains one compute task
while its memory program gates only the corresponding activation segment.
Independent weight loads continue on the other memory port. No attention or
GEMV compute opcode, tile, or thread count changed.

Physical ownership is part of the dependency cut. All K<2048 contributors are
mapped to SMs 64--139, outside the 64-SM attention placement, while late-K
contributors use the complementary SM set after attention. The factorization
still contains exactly 152 reduction tasks. Screening 4/4, 2/2/4, 1/1/6, and
1/3/4 barriers showed that releasing the first two heads individually and
then staying coarse best amortizes memory-program waits; all other grouping
selectors were removed.

A production-image coarse/phased/coarse sandwich measured
2.627072/2.620544/2.627168 ms over 1,001 internal-timer samples in jobs
`20260807T112850Z-3457366`, `20260807T112922Z-3457657`, and
`20260807T112954Z-3457967`. The retained schedule saves 6.576 us / 0.25%
against the control mean. Full S128 tensor correctness and token 24748 passed
in `20260807T113120Z-3459483`; two resident tokens crossed KV128 and produced
`hello hello` in `20260807T113322Z-3461205`. Four Python-unrolled tokens now
exceed the fixed 4,096 memory-instruction buffer, so that larger unrolled
qualification remains deferred with the multi-token milestone rather than
expanding the runtime in this task-focused change.

All 33 host schedule tests pass. The exact 11-op image remains at 96
registers, nine barriers, a 96-byte stack, and zero spills. Its final S128
median is 2.620128 ms in `20260807T113402Z-3461684`, 6.96% below the strict
2.816003 ms vLLM result and 85.725 us above the requested 10%-lead target.

## Rejected independent-accumulator output GEMV pipeline

An output-projection proof of concept alternated K256 groups between two
independent TMEM accumulators and two persistent UMMA completion events. It
submitted group n+1 before waiting for group n, retained the early operand
release, and drained both banks only after all K groups before adding the two
FP32 partials. This directly tested whether UMMA filling bank n+1 could hide
the TMEM/completion tail of bank n without changing the M64 tile or projection
factorization.

Full S128 correctness and token 24748 passed in
`20260807T114015Z-3467590`. The selectable 12-op image used 126 registers,
nine barriers, a 112-byte stack, and zero spills. In the same image, a
control/dual-bank/control sandwich measured 2.619648/2.664224/2.620416 ms over
501 internal-timer samples in jobs `20260807T114056Z-3468155`,
`20260807T114127Z-3468635`, and `20260807T114157Z-3468970`. The dual-bank path
therefore regressed by 44.192 us against the control mean. A second live
accumulator, two TMEM drains, and the FP32 merge cost substantially more than
the completion wait they could hide, so the opcode, second completion event,
schedule selector, and tests were removed.

## Rejected early and static-zero Q cleanup

The updated final-layer frontier showed Q-buffer clearing on auxiliary SM 147
ending at 79.392 us, after next-layer RMS had reached 77.760 us. Moving the
existing load/copy/store task immediately behind attention output and placing
it on main SMs tested whether that cleanup could fit under their MLP slack.
Instead, a same-image control/early/control sandwich measured
2.618752/2.630464/2.619168 ms in jobs `20260807T115559Z-3480216`,
`20260807T115644Z-3480904`, and `20260807T115723Z-3481264`: +11.504 us. The
cleanup left the final frontier but displaced useful MLP/down work.

A stronger VCore prototype reserved a read-only zero tile outside attention
scratch. A direct allocator-warp TMA store was not a valid SM100 execution
path and trapped, so the safe form kept normal ownership: the load VCore
published only a dynamically allocated store descriptor, a lightweight
compute task forwarded that descriptor without touching payload bytes, and
the store VCore sourced the fixed zero tile. This removed the zero TMA load
and 1 KiB CUDA copy. Full S128 correctness passed in
`20260807T121649Z-3497984`, and two tokens crossed KV128 with `hello hello` in
`20260807T121729Z-3498441`.

The reduced handoff still did not shorten the critical path. Same-image
control/static/control medians were 2.624384/2.623712/2.621824 ms in jobs
`20260807T121809Z-3498880`, `20260807T121850Z-3499564`, and
`20260807T121931Z-3500282`, making the late static path 0.608 us slower than
the control mean. Moving that lighter form early measured 2.631008 ms in
`20260807T122014Z-3500679`, a 7.904 us regression. The zero tile, opcodes,
runtime cases, selectors, and manifest entries were removed. Q cleanup is not
a profitable boundary to move while every form still consumes the global
store and queue ordering; future cleanup work needs deletion or safe lifetime
reuse rather than a cheaper copy mechanism.

## Rejected full-K Q ownership and Q/K/V repartition

A no-clear projection topology gave every Q M64 tile one full-K task, applied
fused RoPE once, and used a normal store instead of two BF16 reduction
contributors. Q occupied SMs 64--127 while K then V occupied SMs 0--63, with
the intent to overlap Q with the combined KV path and delete all 64 Q-clear
stores per layer.

The experiment exposed two schedule dependencies that had previously been
implicit in physical placement. V did not carry its own pre-attention-RMS
wait because the low-half Q fold always waited first; moving K to that half
had the same issue. K RoPE and attention also both used raw descriptor slot 24
only because they formerly ran on disjoint CTAs. Explicit RMS waits and a
separate K raw slot made the new topology correct. Full S128 validation passed
all thresholds and token 24748 in `20260807T123155Z-3510844`.

The topology was nevertheless slower. Same-image split-Q/full-Q/split-Q
medians were 2.620864/2.639040/2.620960 ms in jobs
`20260807T123233Z-3511607`, `20260807T123311Z-3512003`, and
`20260807T123353Z-3512451`, a 18.128 us regression against the control mean.
The longer full-K Q task plus serialized K and V work cost more than removing
the Q reduction contributors and clear. The selector and all schedule changes
were removed. Future placement changes must audit both explicit counters and
same-CTA ordering/raw-slot assumptions before performance screening.

## Rejected one-wave heterogeneous LM head

A one-wave LM-head topology used every physical SM and removed the boundary
between the two 128-task vocabulary epochs. SMs 0--111 each owned seven M128
tiles and SMs 112--151 each owned six, covering all 1,024 padded vocabulary
tiles exactly. Both rectangles waited directly on final RMS and jointly
released one 152-record global-argmax frontier, so no dynamic dispatch or
intermediate logits were introduced.

Full S128 correctness and token 24748 passed in
`20260807T124111Z-3518668`. The selectable image grew from 96 to 100 registers
without spills. Same-image two-epoch/one-wave/two-epoch medians were
2.626144/2.666912/2.631136 ms in jobs `20260807T124153Z-3519030`,
`20260807T124230Z-3519650`, and `20260807T124312Z-3520327`. The one-wave path
therefore regressed by 38.272 us against the control mean. Removing an epoch
frontier does not compensate for extending each SM's critical path from four
live output groups to six or seven. The grouped opcodes, 152-way reducer,
selector, and manifest entries were removed, restoring the minimal
96-register production image.

## Rejected cross-task down-projection TMEM pipeline

A combined compute instruction preserved every existing K2048 down-projection
memory program, readiness barrier, TMA reduction, and per-UMMA operand release,
but joined each CTA's two or three tasks into one ping-pong pipeline. After
task n completed in one TMEM bank, task n+1 submitted its first independent
UMMA group into the other bank before the same four compute warps drained and
published task n. This tested compute/compute overlap without another warp,
an accumulator merge, or a global handoff.

Full S128 correctness and token 24748 passed in
`20260807T125547Z-3530042`. The selectable image used 126 registers, nine
barriers, a 96-byte stack, and no spills. Same-image control/all-CTA/control
medians were 2.632672/2.644128/2.626944 ms in jobs
`20260807T125626Z-3530468`, `20260807T125708Z-3531167`, and
`20260807T125749Z-3531674`, a 14.320 us regression. Per-SM profiling in jobs
`20260807T130000Z-3533457` and `20260807T130044Z-3534406` showed that the
final-layer down frontier could appear about 1 us earlier, but the following
RMS frontier was identical at 77.984 us and the gain did not survive 32
layers.

Restricting the pipeline to the 104 main CTAs retained correctness in
`20260807T130218Z-3535452` and reduced the damage, but its 2.638784 ms median
in `20260807T130259Z-3536135` was still 8.976 us slower than the same-image
control mean. One hidden first UMMA does not amortize the extra live state,
control path, and burstier memory demand. The task, opcode, schedule wrappers,
selectors, and manifest entries were removed, returning to the 96-register
image.

## Rejected full-K direct down ownership

Eight persistent output owners replaced the 448 K-sharded down-projection
tasks and their cross-CTA TMA reductions. Each owner computed four M128 output
groups across the complete K14336 dimension in FP32 TMEM, consumed the three
early SwiGLU shards and the final tail through phase-specific readiness
barriers, and wrote the residual result directly to the hidden-state buffer.
This tested whether deleting all reduction traffic and global shard frontiers
could outweigh the much longer owner lifetime.

The exact selectable image still used 96 registers, nine barriers, a 96-byte
stack, and no spills. One diagnostic S128 launch completed with token 24748 in
`20260807T131411Z-3545913`, but changing seven independently rounded BF16
partials into one FP32 accumulation exceeded tensor-equivalence thresholds:
layer-31 gate/up errors were 9.19/7.95%, SwiGLU was 14.43%, and final hidden
and RMS errors were 14.54/14.60%. More importantly, the four-output form hung
nondeterministically in jobs `20260807T131551Z-3547354`,
`20260807T131910Z-3550152`, and `20260807T132550Z-3556147`. Reducing the task
to two output groups and only nine memory messages still hung in
`20260807T132928Z-3558768`, confirming that long phased owners can exhaust the
31-slot live descriptor window or form a producer/consumer cycle. Only the
verified orphan processes from these launches were terminated.

The 2.627168 ms control from `20260807T131509Z-3546665` is recorded only as a
health check; the unsafe variant never produced a valid timing. All opcodes,
packing, TMA descriptors, schedule classes, selectors, and manifest entries
were removed. Full-K output ownership is incompatible with the current
bounded VCore queue and also changes the intended BF16 reduction arithmetic;
future work should shorten live ranges rather than aggregate more K work into
one owner.

## Production hardware-stall profile and rejected LM UMMA grouping

Nsight Compute profiled the unchanged 96-register S128 production
megakernel in `20260807T135224Z-3577336`. Four replay passes reported only
0.38% CTA-barrier stalls and 0.00% memory-barrier stalls. The dominant sampled
reasons were long scoreboard at 37.77%, fixed-latency wait at 20.63%, short
scoreboard at 5.30%, and no-instruction at 3.70%. The profiler replay time is
not a TBT measurement, but the stall mix rules out whole-CTA barriers as the
primary remaining bubble and points instead to operand delivery and
completion pacing.

Following the SM100 canonical pattern of issuing several `tcgen05` operations
before one `umma_arrive`, a selectable LM-head task grouped the four disjoint
M128 output UMMAs for each K128 step under one completion event and retired
their shared-memory slots together. It preserved the direct4 weight/activation
reuse, TMEM layout, and fused argmax, and the combined image remained at 96
registers, nine barriers, a 96-byte stack, and zero spills. Full S128
correctness and token 24748 passed in `20260807T140103Z-3584460`, but repeated
benchmark execution hung in `20260807T140223Z-3585633`; the verified orphan
PID from only that job was terminated.

A bounded two-UMMA completion group was repeat-safe and passed full S128
correctness/token 24748 in `20260807T140643Z-3589031`. Five launches completed
in `20260807T140723Z-3589937`, but the 501-sample median was 2.634240 ms in
`20260807T140801Z-3590358`, versus 2.623968/2.623936 ms same-image controls in
`20260807T140143Z-3585196` and `20260807T140838Z-3590935`: a 10.288 us
regression against the control mean. Coalescing completion delays operand
retirement and makes the producer stream burstier; at depth four that can form
a liveness cycle, and at depth two its cost exceeds the removed completion
wait. Both opcodes, instruction classes, selectors, manifest entries, and
template branches were removed.

## Rejected explicit M2C suspension window

The observer-owned M2C wait was rebuilt with the PTX
`mbarrier.try_wait.parity` suspension-time operand set to 128 ns. This tested
whether the hardware profile's 20.63% fixed-latency wait component came from
hot polling that displaced useful memory/UMMA issue. The exact image remained
at 96 registers, nine barriers, a 96-byte stack, and zero spills, but its
501-sample S128 median was 2.627360 ms in
`20260807T141517Z-3595917`, about 3.4 us slower than the neighboring default
controls. The default implementation-defined suspension policy is already
better; the compile selector and alternate wait path were removed without a
hint sweep.

## Rejected light-warp gate/up TMEM handoff

A combined gate/up instruction tested true cross-task compute overlap rather
than another same-accumulator completion variant. Gate accumulated in TMEM
bank 0 while up used bank 1. At the boundary, a disjoint physical warp drained
one gate epilogue slice while compute warp 0 began the up UMMA stream; the
other three compute warps retained their original logical slices. Ownership
used `tcgen05.fence::before_thread_sync`, compute-role named barriers, and
`tcgen05.fence::after_thread_sync`. The existing RegStore/RegLoad and C2M slot
protocol remained in place.

A ninth physical warp proved the mechanism valid. The 288-thread selectable
image used 96 registers, 12 barriers, a 96-byte stack, and no spills. Full
S128 correctness and token 24748 passed in `20260807T142924Z-3607769`.
Same-image control/helper medians were 2.643232/2.643584 ms in jobs
`20260807T143212Z-3610100` and `20260807T143253Z-3610627`. The overlap paid
back essentially all of the added CTA-shape cost, but did not improve TBT and
remained slower than the 256-thread production image.

Borrowing an existing memory VCore exposed physical TMEM datapath constraints.
Store warp 5 cannot replace logical slice 0; that form corrupted the next
layer in `20260807T143705Z-3613663`. Matching it to slice 1 while moving UMMA
issue to compute warp 1 reduced the damage but still failed layer-1 tensor
correctness in `20260807T144053Z-3616436`. The unchanged schedule on the same
nonblocking C2M runtime was correct in `20260807T143752Z-3614353`, isolating
the failure to the TMEM/issuer role substitution rather than queue polling.

Allocator warp 4 has the required datapath rank 0 and produced exact results.
Generic mailbox polling was too expensive: polling on every memory instruction
made an unused-helper control 2.920096 ms in `20260807T144532Z-3620653`, and
polling only during allocation retry still measured 2.716096 ms in
`20260807T144827Z-3624515`. An explicit zero-slot memory command removed that
hot-path tax. Stopping after a complete B4 operand seed could exhaust the
24-slot pool and hung job `20260807T145210Z-3628562`; only its verified orphan
process was terminated. Seeds of one through three individual A tiles were
repeat-safe, and seed 1 passed full correctness/token 24748 in
`20260807T145411Z-3630116`.

The bounded seed sweep still did not win. Against the 2.635040 ms same-image
control in `20260807T145453Z-3630838`, seed 1 measured 2.647360 ms over 501
samples in `20260807T145531Z-3631515`, seed 2 screened at 2.644416 ms over 101
samples in `20260807T145614Z-3632098`, and the best seed-3 form measured
2.638720 ms over 501 samples in `20260807T145730Z-3633124`. More seeded UMMA
work amortizes the rendezvous, but the largest safe seed remains 3.680 us
slower than its matched control and 18.592 us slower than the retained
2.620128 ms production median.

The architectural boundary is now explicit: a light warp can overlap a legal
TMEM slice only when its physical warpgroup rank matches that slice, but
borrowing the allocator also removes the producer needed to sustain the next
UMMA stream. Adding a ninth warp pays a resident-CTA cost; borrowing warp 4
pays a producer pause. A future light-compute role must be persistent across
multiple stages or operate on work that does not depend on the borrowed
memory producer. The helper opcodes, mailbox, command, schedule fusion, and
experimental manifests were removed.

## Rejected paired-M128 gate/up ownership

A mixed M128 task packed one M64 gate tile and the matching M64 up tile into a
single UMMA A tile, loaded the normalized activation once, rounded both FP32
TMEM halves to BF16, and emitted SwiGLU directly. This differed from the
earlier paired-M64 experiment: one native M128 UMMA owned both projections,
so 96 tasks covered the 6,144-row prefix instead of distributing 192 M64 tasks
over a three-wave auxiliary-SM tail. The selectable Llama image remained at
96 registers, nine barriers, a 96-byte stack, and zero spills.

The task mechanism was exact in isolation. A 96-SM M6144-pair/K4096 probe
measured 20.928 us and zero error over 101 launches in
`20260807T152036Z-3651364`; four back-to-back epochs split over three physical
shards also remained exact in `20260807T152736Z-3656540`. Full-model
integration exposed another implicit placement dependency: the retained gate
tail on SM96--127 had inherited its post-attention RMS wait from the old
prefix. Once the paired prefix occupied only SM0--95, those CTAs could read a
stale normalized row. Adding the explicit wait restored layer-1 and final
tensor agreement (0.68% V, 2.19% SwiGLU, 2.10% hidden, 2.04% RMS) and token
24748 in `20260807T153007Z-3658676`.

The topology still did not qualify. The fine prefix faulted nondeterministically
after several complete kernel replays in `20260807T153153Z-3660498` and
`20260807T153447Z-3663175`, despite isolated repeat safety. Collapsing it to
the exact standalone 96-SM command plus one coarse publication frontier was
live for ten launches, but its best sample was 3.251232 ms and its median was
3.800304 ms in `20260807T153605Z-3664068`, far above the 2.619488 ms control
in `20260807T153121Z-3660036`. Tail-only fusion passed every 32-layer tensor
check and token 24748 in `20260807T152634Z-3655865`, but measured 2.683424 ms
in `20260807T153250Z-3661481`, a 63.936 us regression.

Native M128 ownership reduces command count and activation traffic but
serializes gate/up progress that the resident M64 placement already overlaps.
Its shorter schedule is therefore not a shorter critical path, and the fine
form also fails the repeated-launch lifetime requirement. All paired weights,
TMA descriptors, opcode/task/schedule code, benchmark switches, and manifests
were removed. The rebuilt 11-op production image measured 2.627648 ms over
501 internal-timer samples in `20260807T154150Z-3669016`.

## Rejected balanced two-wave 152-SM LM head

A schedule audit first ruled out a memory-only wrapper around the staged down
tasks. `Launcher` already flattens every schedule into independent continuous
compute and memory instruction streams; the allocator and load VCore can run
past a compute-task boundary without an added handoff. The remaining down
waits are the actual shard-readiness dependencies. A new wrapper with the same
instructions would therefore add representation without exposing work.

The next experiment used the 24 SMs left idle by the 128-SM LM head without
repeating the rejected six/seven-accumulator owner. Each of two waves contained
56 four-output tasks and 96 three-output tasks. The long rectangles swapped
physical ranges between waves, so 112 SMs processed seven M128 vocabulary
tiles and 40 processed six; every task retained at most the production four
live FP32 TMEM accumulators. The complete vocabulary still covered 1,024
tiles, but the partial frontier grew from 256 to 304 records.

Full S128 tensor validation and exact token 24748 passed in
`20260807T155923Z-3683419`. The selectable unprofiled image used 100 registers,
nine barriers, a 96-byte stack, and zero spills. A same-image
control/experiment/control sandwich measured 2.618720/2.621888/2.620384 ms in
jobs `20260807T160006Z-3683960`, `20260807T160046Z-3684652`, and
`20260807T160128Z-3685457`. The balanced form was 2.336 us slower than the
2.619552 ms control mean.

Diagnostic frontier jobs `20260807T160525Z-3688685` and
`20260807T160605Z-3689079` showed why. The last LM compute marker moved from
230.624 us to 234.624 us relative to the final-layer start. The main reducer
SMs spent less visible time waiting at their local LM marker, but the extra 48
activation/task/partial streams extended the distributed projection tail.
Using more SMs did not increase the bandwidth-bound LM throughput. The
group-three opcode, 304-record reducer, weight partitions, selector, and both
manifest additions were removed. The rebuilt 11-op production image returned
to 96 registers with no spills and measured 2.619680 ms over 501 internal
samples in `20260807T161018Z-3692811`.

## Rejected CTA-cluster activation multicast

A cluster-size-two prototype tested whether paired projection CTAs should
replace their duplicate activation TMA loads with one
`cp.async.bulk.shared::cluster.global...multicast::cluster` transaction. It
kept weight ownership and UMMA execution local to each CTA. Explicit cluster
launch was correct through the full S128 model and exact token 24748 in
`20260807T161726Z-3698409`, but the otherwise unchanged kernel measured
2.626912 ms in `20260807T161804Z-3699272`, 7.232 us slower than the neighboring
2.619680 ms non-cluster production result. This is an internal-timer delta,
so it reflects cluster placement/lifecycle constraints rather than host launch
latency.

A 152-CTA standalone probe separated transfer savings from synchronization.
At 8 KiB for 256 waves, independent unicast measured 104.192 us
(`20260807T162408Z-3704310`). Adding one complete `cluster.sync()` per wave
raised matched unicast to 242.464 us (`20260807T162432Z-3704580`); multicast
reduced that to 236.640 us (`20260807T162456Z-3704761`). Halving the activation
transactions therefore saved 5.824 us, but the full-CTA lifecycle rendezvous
cost over 130 us and could not be placed on the production path.

An observer-style replacement used two shared-memory stages. Only the load
leader recycled a stage after one remote arrival from each CTA's completed
consumer; the load VCore, not the compute leader, rearmed the TMA mbarrier.
This was repeat-correct. Against independent unicast over 256 waves, the
leader-owned multicast form was still slower at every tested operand size:
122.944 versus 128.544 us for 4 KiB, 127.744 versus 132.704 us for 8 KiB, and
135.872 versus 140.960 us for 16 KiB. The jobs were
`20260807T163033Z-3710233`/`20260807T163058Z-3710742`,
`20260807T162845Z-3708478`/`20260807T162910Z-3708884`, and
`20260807T163120Z-3711128`/`20260807T163143Z-3711549` respectively.

A final producer/consumer probe split the load leader from a 128-thread
compute VCore and let the two TMA stages run ahead. With no synthetic compute,
pipelined unicast/multicast measured 60.576/68.768 us. A 200 ns compute window
hid part of the lifecycle cost but still measured 92.864/96.160 us. At 500 ns
the paths tied at 166.880/167.008 us; multicast never became faster. Adding a
unique 8 KiB weight stream per CTA, which models the projection's simultaneous
non-shareable operand traffic, exposed the cost again at 190.080/198.208 us in
`20260807T164416Z-3722263` and `20260807T164439Z-3722734`.

The shared activation is already cache-resident enough that removing one TMA
request does not repay remote stage ownership. Compute can hide the extra
barrier, but then there is no positive transfer delta to offset the measured
7.232 us production cluster penalty. Cluster launch support, multicast
instructions, the standalone probe, and all selectors were removed. A future
cluster design needs an operation that intrinsically requires two-CTA UMMA or
shares substantially more than one cached activation tile; it should not be
introduced only to deduplicate projection B loads. The restored non-cluster
11-op image compiled with 96 registers, nine barriers, a 96-byte stack, and no
spills; fresh 501-sample medians were 2.627904 and 2.624640 ms in
`20260807T164758Z-3725282` and `20260807T164840Z-3725741`.

## Rejected projection weight prefetch and narrower B cadence

The next operand-delivery audit separated tile shape, activation cadence, and
cache lead. An exact fold-2 M4096/N8/K4096 projection measured 6.784 us with
the retained M64 tile on 128 SMs (`20260807T165550Z-3731265`). A native M128
tile must use only 64 SMs for the same K fold and measured 10.688 us
(`20260807T165621Z-3731593`). The earlier M128 wins came from changing the K
factorization, not from a faster like-for-like tile. Loading B every two A
tiles instead of every four reduced the live input footprint but raised the
same M64 probe to 7.488 us (`20260807T165854Z-3733994`); the extra activation
commands cost more than the shorter slot lifetime.

A raw 152-CTA stream then established that Blackwell's non-allocating
`cp.async.bulk.prefetch.L2.global` mechanism can hide a genuinely cold next
tile. Across 64 unique 32 KiB tiles per CTA, no hint measured 59.904 us and a
one-tile lead measured 46.176 us in `20260807T170244Z-3736883` and
`20260807T170308Z-3737355`. Two tiles were neutral to one at 46.208 us, while
four regressed to 50.560 us. A rank-5 tensor-map form was therefore added
temporarily to the allocator VCore so it consumed neither a shared slot nor an
M2C handoff. Prefetching the current group was harmful. Moving the hint between
the first and second K2048 groups did improve an eight-epoch isolated
projection from 49.952 to 47.904 us (`20260807T171614Z-3747317`).

That isolated cache gain did not survive the full model's concurrent weight
stream. Against the same selectable-image 2.631680 ms control
(`20260807T171248Z-3744678`), one-group-ahead output prefetch measured 2.637248
ms (`20260807T171913Z-3749752`), down prefetch measured 2.665120 ms
(`20260807T171952Z-3750322`), and enabling both measured 2.673120 ms
(`20260807T172029Z-3750678`). These 201-sample internal medians regress by
5.568, 33.440, and 41.440 us respectively. The model is already issuing useful
TMA traffic during compute; speculative hints add request/cache pressure rather
than expose a new overlap window. The tensor-prefetch opcode, scheduler hooks,
standalone probe, B2 selector, and all Llama selectors were removed. Future
operand work should pipeline completion and slot retirement of existing
traffic, not inject duplicate weight reads.

## Rejected store-VCore completion pipeline

The store VCore currently commits one asynchronous TMA writeback and waits for
that group before releasing its shared slot or publishing a global counter.
A depth-two prototype tested whether a ready successor could be issued before
retiring the older group. The safe form preserved the existing 129-party C2M
barrier, probed a copy of the store warp's arrival token, and retained at most
one older slot. If the successor was not already published, it completed and
released the current store before blocking, so a dependent compute task could
not form a store/barrier cycle.

The mechanism works for a homogeneous projection stream. Eight consecutive
M4096/N8/K4096 fold-2 reductions measured 48.992 us over 501 internal samples
in `20260807T175637Z-3781446`, versus the neighboring 49.952 us serial control
in `20260807T170809Z-3741087`: a 0.960 us / 1.92% reduction. Merely arriving
at the next C2M phase during the current store, without issuing another store,
was repeat-safe and spill-free but regressed full S128 from 2.616352 ms
(`20260807T173115Z-3760343`) to 2.624000 ms
(`20260807T175100Z-3776579`).

Applying depth two to every writeback exposed invalid queue assumptions and
hung full-model launches. In particular, making the store an observer by
changing the global C2M participation count lets special-slot producers lap
the 32-entry ring; C2M cannot adopt the M2C producer-owned protocol wholesale.
The exact failed jobs were stopped only after verifying their owned worker
PIDs. Restricting depth two to consecutive rank-2 projection reductions made
the full model repeat-live (`20260807T175937Z-3784305`) while leaving all
attention, raw-address, and ordinary stores on the original path.

The restricted production test still lost: its 501-sample S128 median was
2.631552 ms in `20260807T180110Z-3786923`, 15.200 us slower than the matched
serial build. The valid 0.960 us micro gain is outweighed by C2M readiness
probing, an extra live output slot, and burstier global reduction publication.
All queue, runtime, and build selectors were removed. The next overlap design
should eliminate a materialized stage or handoff rather than retain more
writebacks in the already slot-paced pipeline. The rebuilt production image
returned to 96 registers, nine barriers, a 96-byte stack, and zero spills; its
fresh 501-sample S128 median was 2.623648 ms in
`20260807T180747Z-3793689`.

## Rejected post-attention RMS gamma folding

Folding each post-attention RMS gamma vector into the gate and up weights
removes the gamma TMA load and elementwise multiply from the RMS task, but the
resulting per-token inverse RMS must then reach every projection owner. An
isolated B8/K4096 task confirmed that the local opportunity is real: ordinary
shared RMS measured 2.144 us while scale-only RMS measured 1.664 us in
`20260807T181228Z-3798360`, a 0.480 us / 22.4% saving. Folding the actual model
weights produced only BF16 rounding differences (roughly 0.30--0.32% mean
relative error across q, gate, up, and LM projections) in
`20260807T181532Z-3801038`.

The first delivery design co-loaded the dynamic scale beside every repeated B
activation tile. Full-model validation passed every tensor threshold and exact
token 24748, but its 501-sample internal median was 2.658848 ms in
`20260807T184023Z-3823696`, 35.200 us slower than the retained 2.623648 ms
production result. A second design materialized the scale once and published
it through a separate stage-wide shared-scratch M2C handoff to all 152 CTAs.
It was exact-token correct but measured 2.795936 ms in
`20260807T184514Z-3827034`; the extra 152-way handoff and rendezvous cost about
4.3 us per layer.

The final design let each SM's first projection seed a local scale scratch and
reused it in later projection epilogues, avoiding the stage-wide handoff. Its
best full-model median was still 2.647488 ms in
`20260807T185947Z-3839451`, 23.840 us slower than production, and accumulated
roughly 5% gate/SiLU relative error in later layers. Isolated scaled projection
tests likewise showed no hidden epilogue win: M4096/K4096 measured 6.944 us
versus 6.880 us, M2048 measured 4.256 versus 4.192 us, and M6144 measured
11.456 versus 11.424 us.

The scale-only RMS task therefore saves work, but distributing one dynamic
scalar through the persistent CTA graph costs more in load-program traffic,
queue state, and synchronization than it removes. All folded weights, scale
scratch, opcodes, task variants, schedule paths, and benchmark switches were
removed. Revisit this only if the scale can remain inside the same physical
owner across RMS and both MLP projections; another broadcast representation
is not promising. The rebuilt 11-op production image retained 96 registers,
nine barriers, a 96-byte stack, and zero spills. Full S128 validation passed
every tensor threshold and exact token 24748 in `20260807T190549Z-3844602`;
its fresh 501-sample internal median was 2.621760 ms (2.608160 ms minimum) in
`20260807T190630Z-3845286`.

## Refreshed stage frontier and rejected auxiliary QKV overlap

The post-RMS production schedule was reprofiled with the isolated 13-op marker
image in `20260807T190928Z-3847568`. Markers raised the launch to 2.677 ms and
remain diagnostic only. In the final layer, Q reached a 9.536 us frontier,
attention reached 16.448 us, post-attention RMS reached 22.080 us, the MLP
prefix tail reached 62.976 us, next-layer RMS reached 77.824 us, and the
slowest layer-loop arrival reached 80.032 us. The two LM-head epochs then took
about 81.8 and 72.9 us, followed by a roughly 6 us reducer. The remaining
repeatable gap is distributed across each layer; it is not a single random
straggler or launch tail.

A first schedule-only proof expressed Q as four K1024 folds and balanced the
combined Q/K/V work over all 152 SMs. Layer 0 was numerically sound, but the
shortened Q producers exposed that Q clear is an independent store-VCore
operation: layer 1 could be overwritten after the compute stream had advanced.
Adding a separate carried clear frontier exceeded the memory instruction's
10-bit barrier-ID range. Folding clear into the existing next-RMS barrier did
not produce a live phase protocol and was stopped after only the owned job was
verified. The topology was removed without timing an invalid path.

A narrower proof retained the K2048 Q tasks and moved both Q folds for head 0
to auxiliary SMs, allowing its original K/V owners to overlap with Q. Keeping
the new Q owners away from the CTAs that clear head 0 fixed the immediate reuse
failure: layer-1 V and K returned to 0.69% and 0.40% relative error. The early
head frontier nevertheless changed the phased output reduction order enough
to compound to about 37% hidden/RMS error by layer 31. A 101-sample mechanism
screen measured 2.603456 ms versus a 2.616256 ms same-image control in
`20260807T192422Z-3860709` and `20260807T192458Z-3861296`, only a 12.800 us
gain. Moving only head-0 K/V to auxiliary SMs left Q ownership unchanged but
reproduced the same late-layer drift in `20260807T192615Z-3862139`, isolating
the problem to the materially earlier head/output reduction order rather than
Q clear alone.

The potential gain is too small to justify another barrier or a change in the
32-layer BF16 reduction semantics. All Q-fold, placement, explicit-wait, and
selector code was removed. Spare-SM QKV work should next be considered only
with a numerically stable accumulation boundary; merely advancing one phased
attention head is not a valid production optimization. The restored 11-op
image remained at 96 registers, nine barriers, a 96-byte stack, and zero
spills. Full S128 validation passed every tensor threshold and exact token
24748 in `20260807T192906Z-3864350`; its fresh 501-sample internal median was
2.620640 ms in `20260807T192944Z-3865232`.

## Parked full-warpgroup base cost

The sidecar experiment kept the existing four-warp task ABI and queue
participation exactly intact, but added an opt-in second four-warp compute
group. The auxiliary group blocked on named
barrier 15 for an ordinary schedule, never interprets instructions, and never
joins M2C or C2M. This is deliberately different from the old global T256
runtime, where eight compute warps affected every queue phase. The default
build still launches 256 threads and compiles at 96 registers, nine hardware
barriers, a 96-byte stack, and zero spills. The opt-in 384-thread skeleton
uses the same registers/stack/spill count; the named-barrier ID makes ptxas
report 16 hardware barriers.

S128 correctness passed every tensor threshold and exact token 24748 in
`20260808T021323Z-18791`. On the exact 11-op image, a 501-sample
default/aux/default sequence measured 2.569824/2.574720/2.569120 ms in
`20260808T021121Z-17328`, `20260808T021400Z-19378`, and
`20260808T021603Z-20972`. The parked 128 threads therefore cost 5.248 us
against the control mean, or 0.20% of full S128. This established the fixed
budget for the projection-chain experiments below.

Three coarse paired-projection organizations were then correctness checked
with disjoint TMEM banks and independent completion ownership. First, both
warpgroups issued independent UMMA streams. Two K2048 projection epochs were
neutral at 12.480 versus 12.512 us, while two K4096 epochs regressed to 24.736
versus 24.512 us in jobs `20260808T022721Z-30461`--
`20260808T023027Z-33083`. The second organization kept UMMA issue on the
original warpgroup and handed only a completed bank to the sidecar epilogue.
Two-task K2048/K4096 results were again neutral at 12.384/24.640 versus
12.416/24.672 us, but a four-task chain regressed to 26.880 from 26.464 us in
`20260808T024706Z-47041` and `20260808T024736Z-47582`. Narrowing the repeated
handoff to the 32-thread issuer warp plus the 128-thread epilogue group did
not help: two and four tasks measured 12.672 and 26.656 us, regressions of
0.256 and 0.192 us in `20260808T025053Z-50099` and
`20260808T025024Z-49965`.

The dependent gate/up form used one TMEM-ready edge and a second edge only
where up consumed the gate tile. It passed every S128 tensor threshold and
exact token 24748 in `20260808T024156Z-42543`. On the same selectable image,
pair/control medians were 2.579200/2.579616 ms in
`20260808T024234Z-43225` and `20260808T024311Z-43699`, a sub-drift 0.416 us
apparent gain. That image used 100 registers and a 112-byte stack and remained
about 9.7 us slower than the neighboring 256-thread production controls.

The result is structural: the load VCore and one ordered UMMA stream already
pace these M64 tasks, while every cross-warpgroup TMEM ownership transition
costs about as much as the epilogue it can hide. Longer chains accumulate the
transition cost. All mailbox, paired op, task splits, schedule wrappers,
extra barriers, and the wider launch selector were removed; the minimal
four-compute-warp runtime remains production. The restored exact image uses
96 registers, nine barriers, a 96-byte stack, and zero spills. All 20 runtime
tests passed; job `20260808T025743Z-55875` matched four repeated greedy tokens,
and job `20260808T025824Z-56647` measured a 2.569056 ms S128 internal median.

## Rejected distributed atomic LM-head reduction

The refreshed stage profile in `20260808T030449Z-62995` placed the distributed
LM-head projection frontier at roughly 229.5 us after the final layer start and
the token reducer at roughly 245.3 us. Three proof-of-concept paths tested
whether compact maxima could eliminate that apparent 15.8 us tail without
materializing the existing 256 partial records.

The first path packed each CTA's BF16 maximum and inverse vocabulary index into
one ordered 64-bit key and performed eight compute-side `atomicMax` operations.
It passed exact single- and four-token resident correctness in
`20260808T031213Z-68948` and `20260808T031258Z-69592`. A
control/atomic/control sequence measured 2.576192/2.574464/2.575520 ms, only a
1.392 us apparent gain against the control mean. Sharding the key across 16
independent destinations also passed repeated correctness but measured
2.565248/2.570688/2.572288 ms, a 1.920 us regression against its control mean.

The second organization sent eight packed keys through the ordinary shared
slot and C2M path, letting the store VCore perform the atomics and publish the
completion barrier. Its first form was neutral at
2.570080/2.570240/2.570816 ms. A tighter form retained epoch 0's keys in the
store VCore's registers, merged epoch 1, and issued only eight atomics plus 128
partial-barrier releases for the whole LM head. It passed exact four-step
resident decode in `20260808T033523Z-86948`, but the decisive
control/variant/control medians were 2.569920/2.572448/2.570528 ms in
`20260808T033611Z-87446`, `20260808T033653Z-88423`, and
`20260808T033730Z-88677`: 2.224 us slower than the control mean.

The store form also raised the selectable image from 96 to 128 registers. The
profiled tail therefore was not a serial reducer-only interval: much of the
reducer already overlaps the uneven LM-head completion frontier, while atomic
publication adds contention and whole-image register pressure. All atomic
opcodes, key buffers, store-VCore state, finalizer, schedule variants, and
selectors were removed. A future LM-head change should reduce projection work
or keep a non-atomic reduction wholly inside an existing owner; another global
publication representation is not promising. The restored 11-op image passed
all 20 runtime tests, retained 96 registers/nine barriers/zero spills, and
matched four repeated greedy tokens in `20260808T034403Z-94356`. Its fresh
501-sample S128 internal median was 2.570208 ms in
`20260808T034444Z-95073`.

## Rejected same-owner auxiliary up pairing

The current absolute profile in `20260808T034925Z-99210` confirmed that the
layer boundary is already overlapped: down owners, next RMS, and auxiliary Q
clear converge around 83.6--84.3 us, while loop progression adds only about
0.4 us. The remaining deterministic spread inside MLP is the two- and
three-task auxiliary up chain. A targeted paired M64 task therefore combined
only two outputs already serialized on the same CTA and feeding the same
2,048-row readiness shard. It retained independent M64 TMEM accumulators,
immediate A-slot retirement, two ordinary TMA stores, and two barrier
releases, but reused each four-tile normalized-activation B transaction.

The 12-op selectable image stayed at 96 registers, nine barriers, a 96-byte
stack, and zero spills. Four resident decode steps matched reference token
24748 exactly in `20260808T035827Z-106651`. The full shard-1+2 pairing still
lost: control/paired/control medians were
2.568832/2.573824/2.572832 ms in `20260808T035909Z-107054`,
`20260808T035948Z-107663`, and `20260808T040031Z-108397`, a 2.992 us
regression against the control mean.

The diagnostic profile in `20260808T040245Z-110233` showed that the task-local
mechanism did work. The slow auxiliary prefix moved roughly 0.5--2.0 us
earlier and the shard-SiLU absolute frontier improved by about 1.2 us versus
`20260808T034925Z-99210`. The final layer frontier nevertheless changed from
84.288 to 84.416 us. Pairing only critical shard 2 then measured 2.572832 ms
in `20260808T040610Z-112732` versus bracketing 2.572832/2.569152 ms controls,
a 1.840 us regression against their mean.

This isolates the cost to operand scheduling rather than compute or TMEM
drain: retaining B while issuing the second weight stream makes the local CTA
earlier but perturbs concurrent main-tail/down traffic. An extra compute warp
cannot repair that load/slot interference. The paired task, schedule wrapper,
opcode, selector, and manifest entry were removed. Future cross-stage work
must preserve the fine-grained producer order instead of making one CTA's
weight burst denser. The restored 11-op image passed all 20 runtime tests and
four exact resident steps in `20260808T041134Z-116929`; its fresh 501-sample
S128 internal median was 2.569664 ms in `20260808T041219Z-117575`.

## Cross-track overlap requires a shared timeline

A two-phase next-layer RMS proof established an important placement limit.
Down projection published low/high hidden-row readiness, and one RMS task
cached the low K2048 half through the ordinary memory-op path before waiting
for the high half. Moving that task from SMs 0--7 to SMs 136--143 advanced its
absolute completion by about 1.7 us, but unprofiled S128 timing was neutral to
worse and the auxiliary clear/loop frontier stayed near 84 us. The proof was
correct and spill-free, then removed.

Do not infer token overlap from a compute marker alone. For a candidate edge,
correlate at least: compute opcode issue/completion, UMMA/TMEM drain, both LDU
streams, allocator/shared-slot occupancy, writeback completion, and the exact
barrier primitive/participant scope. Also inspect local instruction order: a
globally ready operand cannot start a task that is still behind unrelated work
in that CTA's compute stream, while moving the task may only expose a clear or
memory-track successor. Prefer a mechanism screen and absolute per-SM
frontiers before adding another task variant.
The restored 11-op image retained 96 registers, nine barriers, a 96-byte
stack, and zero spills; four resident steps were exact in
`20260808T044341Z-144054`, and the fresh S128 internal median was 2.568672 ms
in `20260808T044420Z-144879`.

## Opt-in multi-track profiling

Build the selective runtime with `make track_profile=1` and run Llama with
`VDCORES_TRACK_PROFILE=1` to collect per-SM aggregate time for slot-allocation
retries, each LDU's command/dependency waits, compute M2C operand waits, and
store queue/service.  The counters occupy profile events 96--120 and event
127 carries a required image sentinel.  Schedule `OP_PROFILE_EVENT` markers
remain in the lower range, so the stage and role views can share one
`globaltimer` timeline.  Do not interpret store queue wait as a store stall:
it is time for which the store VCore has no command; store service is reported
separately.  The profiler is compile-time-only and the production image has no
additional timer instructions or counter state.

On the exact S128 schedule, full correctness and token 24748 passed in
`20260808T045508Z-154231`.  A representative trace
(`20260808T045415Z-153452`) measured median per-SM compute M2C wait of 985.248
us (37.14% of the compute-thread span), allocator slot-exhaustion time of
668.320 us, LDU0/LDU1 dependency waits of 8.224/652.608 us, and store service
of 106.080 us.  A simultaneous stage trace
(`20260808T045710Z-155849`) identified Q clear on SM136--143 as the final-layer
tail: next RMS finished near 84.9 us, that clear cohort finished at
85.4--85.9 us, and the slowest layer-loop arrival was 86.240 us.  Other clear
cohorts finished near 80--81 us, so evaluate Q lifetime/placement before
optimizing clear copy bandwidth.

The diagnostic exact image uses 100 registers, nine hardware barriers, a
144-byte stack, and zero spills.  Rebuilding without `track_profile=1`
restores 96 registers, nine barriers, a 96-byte stack, and zero spills; the
fresh profiling-free S128 median was 2.575904 ms in
`20260808T050051Z-159185`.

## Late cleanup is pacing as well as work

Treat deletion of a cleanup stream as a queue-scheduling change. On the
current single-token image, deleting Q clear increased allocator-slot and
LDU1 dependency stalls and regressed by about 5.9 us even though the stores
themselves were small. A one-group pipelined TMA store was +1.856 us, early
barrier-phased clear was about +125 us, and replacing clear with a
barrier-correct fold-0 overwrite/fold-1 reduction was +104.992 us. The last
form lost both parallel Q writeback and Q/K overlap. These mechanisms were
removed; a versioned/no-clear design must restore the pacing explicitly and
must budget any additional per-layer barrier IDs.

The useful minimal change spreads the unchanged 64 Q-zero stores over
SM88--151, one tile per CTA, instead of leaving two or three stores on each of
24 late auxiliary CTAs. A 1,001-sample late24/late64/late24 sandwich measured
2.580096/2.579488/2.582400 ms, or -1.760 us versus the control mean. The
retained 11-op image stays at 96 registers, nine hardware barriers, a 96-byte
stack, and zero spills. Full S128 correctness is
`20260808T060649Z-214439`, four exact resident tokens are
`20260808T060732Z-215233`, and the final profiling-free median is 2.572704 ms
in `20260808T060813Z-215760`.

## Requalify dependency proofs after the bottleneck moves

Spreading Q cleanup moved the converged frontier from clear to the
down-to-next-RMS edge, but rebuilding the earlier staged-RMS proof did not
make it retainable. The exact two-phase path passed full S128 correctness in
`20260808T064051Z-243028`; four phases passed in
`20260808T065037Z-250626`. The latter added more load publications and
compute-group slot-release barriers and measured 2.579136 ms versus
2.573440/2.573984 ms controls.

More importantly, the apparent initial two-phase gain failed a fresh
1,001-sample sandwich: control/staged/control was
2.577472/2.577088/2.574880 ms, making staged 0.912 us slower than the control
mean (`20260808T070043Z-259638`, `20260808T070127Z-260163`,
`20260808T070202Z-260745`). Do not retain a structural dependency change on
one sandwich whose delta is inside run-to-run drift. Increasing phase count
also consumes scarce per-layer counter IDs; if a future design needs more
frontiers, expand the barrier model explicitly and require a moved whole-layer
boundary, not only an earlier local marker. All staged-RMS machinery was
removed.

## Q storage is already layer-private; cleanup is still schedule state

The Llama schedule already owns one Q allocation per layer. A token-end batch
proof therefore removed every per-layer clear and had 24 auxiliary CTAs clean
all 32 buffers while the main CTAs ran the LM head. A dedicated system barrier
counted all 2,048 completed TMA stores before reuse; this is the correct way to
expand synchronization rather than overloading a per-layer counter. The proof
passed S128 correctness in `20260808T071429Z-270421`, but measured
2.586240 ms versus 2.572544/2.572512 ms controls, +13.712 us. The barrier made
the tail visible, while deleting periodic clear changed allocator/LDU pacing.

A one-layer-delay proof retained one cleanup pulse per layer. Its descriptor
order was rotated so layer L cleared L-1, current pre-attention RMS readiness
provided the safe dependency, and SM88--151 executed cleanup concurrently with
attention. It passed full S128 correctness and four resident-token reuse in
`20260808T071853Z-274287` and `20260808T072416Z-278446`. Narrowing cleanup to
24 auxiliary CTAs or moving it to SM64--127 was worse. The best placement
initially appeared about 2.2 us faster, but a clean final
control/delayed/control run was 2.571104/2.571904/2.572416 ms, making delayed
0.144 us slower than the control mean. Cleanup location affects several
tracks, and deltas below drift must not be retained. Both proofs were removed.

## Preserve early down weights on LDU0

High allocator-slot and LDU1 dependency times do not imply that weights should
wait behind activation readiness. An allocator `IssueBarrier` before low-K or
high-K down work measured 2.587328/2.615104 ms versus a 2.573536 ms control.
The existing LDU0 weight stream is hiding the SiLU producer edge; a strict
same-track order exposes 13.8--41.6 us instead of removing a bubble.

Using more concurrency did not help either. Sending weight tiles 2--3 of each
four-tile group to LDU1 measured 2.573952 ms for low-K, 2.574400 ms for
high-K, and 2.578240 ms for both. Load completion still enters one ordered M2C
sequence, while LDU1 carries the activation dependency. Keep activations on
LDU1 and weights on LDU0 for down projection; all experimental hooks were
removed.

## Compact swapped-attention scratch first unlocked 26 allocator slots

The exact Llama swapped-attention task needs 2 KiB for its swizzled BF16 P
tile and 2 KiB for an FP32 score transpose, but those representations do not
have overlapping lifetimes. The first qualified `aux_slots=1` build aliased them in
one 2-KiB region. All four compute warps first copy their score rows into
registers, join through the existing compute-only barrier primitive, and only
then overwrite the region with BF16 probabilities. Scratch can therefore live
above a 26-by-8-KiB allocator arena inside the unchanged 212-KiB dynamic
shared-memory allocation. This avoids a schedule-specific attention-owner
mask and gives every CTA two additional shared slots.

Use the option only with the exact swapped-attention selective image. The
current option includes the 27th-slot sizing described below:

```bash
DAE_COMPUTE_OPS_FILE=benchmarks/blackwell_llama8b_fused_argmax.ops \
make aux_slots=1 pyext
```

`setup.py` receives the same build defines as `runtime.o`, and direct attention
outputs derive their raw special-slot ID from `runtime.config.num_slots`, so
the host encoder and resident kernel agree on the shifted special-slot range.
Do not enable this option for a generic FA4 shared-P image without separately
budgeting its larger scratch region.

On fixed B8/S128, the 24-slot control measured 2.573248 ms. Two retained
1,001-sample profiling-free internal medians were 2.543392 and 2.543360 ms,
averaging 2.543376 ms. That is 29.872 us faster than control and 9.681% faster
than strict vLLM's 2.816003 ms; the 10% target is 2.534403 ms. The exact image
uses 96 registers, nine hardware barriers, a 96-byte stack, and zero spills.
Full tensor correctness passed in `20260808T083254Z-337814`, four resident
tokens passed in `20260808T083331Z-338417`, and all 34 schedule/attention host
tests passed. A fresh profiling-free rebuild measured 2.543680 ms in
`20260808T085133Z-353947`.

The retained diagnostic run `20260808T084729Z-350511` confirms the mechanism:
relative to the matched 24-slot trace, median allocator stall fell by 54.496
us, compute M2C wait by 35.424 us, LDU0 queue wait by 38.048 us, and store
queue wait by 29.472 us. LDU1 dependency wait increased by 24.128 us, so the
extra depth is converting allocator starvation into useful prefetch while
moving some pressure onto the activation stream.

The allocator change requires the cleanup requalification below. Any further
versioned/no-clear proof must retain useful issuance pacing and model buffer
lifetime explicitly. Add dedicated barrier IDs if required; do not overload a
QKV, RMS, or MLP frontier merely to avoid expanding the barrier set.

## One-layer-deep Q cleanup on the 26-slot pipeline

Requalifying cleanup after the allocator expansion changed the result of the
earlier one-layer-delay proof. The retained schedule rotates only the grouped
Q-clear descriptors: layer L clears the layer-L-1 private Q buffer. It waits
on layer L's pre-attention RMS frontier, is issued behind current Q/K/V, and
runs one unchanged tile per CTA on SM88--151 while the attention CTAs execute.
The pre-RMS counter is the actual lifetime edge, so no QKV/MLP barrier is
overloaded and no new operator or runtime path is needed.

An exact-image control/delayed/control sandwich measured
2.541344/2.536320/2.543712 ms over 1,001 internal-timer samples in jobs
`20260808T091126Z-371554`, `20260808T091212Z-372148`, and
`20260808T091249Z-372830`: a 6.208 us gain against the control mean. Two final
default-path runs measured 2.535968 and 2.543072 ms in
`20260808T092447Z-384212` and `20260808T092523Z-384493`. The median of the
three retained runs is 2.536320 ms, 9.932% below strict vLLM's 2.816003 ms and
1.917 us above the exact 10% target.

The simultaneous stage/track diagnostic comparison
(`20260808T091952Z-379549` versus `20260808T092031Z-380191`) shortened the
diagnostic token from 2.684000 to 2.674720 ms. The clear pulse completed near
15--18 us absolute under attention/output work instead of remaining at the
layer tail; the layer-loop frontier maximum fell from 79.232 to 78.624 us.
Median compute M2C wait fell from 724.000 to 716.480 us, while median store
barrier service fell from 111.680 to 101.056 us. Allocator stall was roughly
flat (646.816 versus 648.544 us), confirming that the gain is moved work and
shorter synchronization exposure rather than deleted cleanup traffic.

The stronger non-periodic proof removed all in-loop clears, cleaned all 32
independent layer buffers on the 24 auxiliary CTAs during LM head, and used a
dedicated system barrier for all 2,048 completed stores before token finish.
It passed full and four-token correctness in `20260808T091515Z-374872` and
`20260808T091550Z-375550`, but measured 2.552448 ms in
`20260808T091627Z-376424`, 9.920 us slower than the nearby control mean. The
store burst competed with LM-head bandwidth and removing periodic cleanup
again changed useful allocator/LDU pacing, so the barrier and batch machinery
were removed.

A final barrier-elision proof relied on the RMS-gated Q/K/V task already
present on every SM88--151 owner and the later compute-to-memory return to
protect the clear store. It passed full and four-token correctness
(`20260808T092655Z-385786`, `20260808T092730Z-386051`) but measured
2.538880 ms in `20260808T092804Z-386585`, with no material improvement over
the explicit-barrier result. Keep the explicit pre-RMS wait as the clearer
lifetime proof.

A separate spare-SM LM-head proof split epoch one across 104 main and 24
auxiliary CTAs and held auxiliary allocation until all epoch-zero records were
stored. It was exact in `20260808T090006Z-361651`, but measured 2.589728 ms
versus 2.540416 ms control (`20260808T090122Z-362816` and
`20260808T090047Z-362540`). The full frontier prevents HBM contention but also
discards the resident schedule's useful cross-epoch weight prefetch; all
partition and barrier code was removed.

## Right-size the exact instruction image for the 27th slot

The exact S128 program uses at most 18 compute and 162 memory instructions per
SM, so the generic 512-entry shared instruction caches were dead capacity.
The retained exact-image build uses 192 entries, a 30-instruction margin over
the measured maximum. This saves 7,680 bytes of static shared memory and lets
`aux_slots=1` use a 27-by-8-KiB allocator arena, the packed 2-KiB attention
scratch, and up to 1 KiB of base alignment inside a 219-KiB dynamic request.
Ptxas reports 7,024 bytes of static shared memory, so the block occupies
231,280 of the GB200's 232,448-byte opt-in limit. The image remains one CTA per
SM and retains 96 registers, nine hardware barriers, a 96-byte stack, and zero
spills.

The dynamic request is now a compiled runtime constant exported through
`runtime.config`, rather than a Python-only 212-KiB literal. `runtime.o`, the
Python extension, instruction tensors, allocator slot IDs, and launch shared
memory therefore all derive from the same opt-in build. The generic defaults
remain 24 slots, 512 instructions, and 212 KiB; the 27-slot form is still
restricted to the exact swapped-attention manifest.

A rebuilt 26-slot control measured 2.537888 ms in
`20260808T094320Z-398578`. Five profiling-free 27-slot internal medians were
2.524704, 2.525824, 2.525312, 2.526656, and 2.524800 ms in jobs
`20260808T094115Z-396940`, `20260808T094530Z-400421`,
`20260808T094611Z-400916`, `20260808T101914Z-429988`, and
`20260808T101954Z-430693`. Their median is 2.525312 ms, 12.576 us below the
matched 26-slot control, 10.323% below strict vLLM's 2.816003 ms, and 9.091 us
past the exact 10% target of 2.534403 ms.

The diagnostic 27-slot trace `20260808T095032Z-404858` measured 2.660256 ms,
versus 2.674720 ms for the matched retained 26-slot diagnostic
`20260808T092031Z-380191`. Median allocator exhaustion fell from 648.544 to
599.680 us and compute M2C wait from 716.480 to 707.616 us. This is a deeper
useful prefetch window, not an instruction-dispatch saving: shrinking the
cache makes room for one more live shared operand and moves the allocator
frontier by 48.864 us.

Final full S128 correctness passed in `20260808T101758Z-428833`; four resident
tokens were exactly `[24748, 24748, 24748, 24748]` in
`20260808T101838Z-429605`; all 34 schedule/attention host tests passed.

## Rejected deeper and non-periodic cleanup on 27 slots

Rotating the same private Q-buffer descriptors by 1, 2, 4, or 8 layers
measured 2.528000, 2.526176, 2.527104, and 2.525920 ms in one sequential sweep
(`20260808T095333Z-407402`). An eight-layer rotation also passed four resident
tokens in `20260808T095513Z-409001`. The 2.080-us spread is run drift: all four
forms issue identical traffic behind the same current-layer pre-RMS lifetime
barrier. Keep the one-layer rotation because it is the shortest explicit
lifetime proof.

The non-periodic proof was requalified with the extra allocator slot. It used
all 24 otherwise-free SM128--151 CTAs during LM head, advanced matching
compute and memory VCore loops across the 32 layer-private buffers, and used a
dedicated system barrier for exactly 2,048 completed stores. The barrier
reached zero and four resident tokens passed in `20260808T101318Z-424975`.
Its 2.542688-ms internal median (`20260808T101357Z-425528`) was 17.376 us
slower than the retained five-run median: the concentrated store burst still
competes with LM-head HBM traffic. The batch loop, barrier, debug counters, and
depth selector were removed.

This proof also established the synchronization rule for future deep cleanup:
looping only the memory VCore leaves later load/store pairs without a compute
consumer; looping only one track produced 65 of 128 expected arrivals in the
two-layer diagnostic. Compute and memory loops must advance together, and the
global completion wait must occur after every layer iteration has issued, not
inside the first 64-store iteration. A separate attempt to move the retained
clear dependency from the allocator to the zero-load LDU passed correctness
but measured 2.541760 ms in `20260808T093213Z-389821`; blocking LDU0 delayed
following weights by about 5.4 us. No additional cleanup barrier or runtime
operator is retained.

## Rejected attention-snapshot Q cleanup barrier

A finer lifetime proof replaced the retained next-layer RMS dependency with a
dedicated 64-count barrier per layer. Each attention CTA published after its
2-KiB Q TMA snapshot completed, and SM88--151 cleared the current layer's Q
buffer while QK/softmax/PV continued. The proof used independent counters; it
did not overload a QKV, RMS, or MLP frontier. Compacting only the unused tail
instances of the repeated barrier group kept every referenced barrier in the
VM's 10-bit field. The exact image remained at 96 registers, nine hardware
barriers, a 96-byte stack, and zero spills.

The first ownership form made the Q load VCore execute `arrive_and_wait()` on
its own M2C mbarrier before decrementing the global counter. Compute remained
an observer, so there was no invalid whole-CTA fence or memory-warp join. Full
S128 correctness and exact token 24748 passed in
`20260808T104306Z-451647`. A control/snapshot/control internal-timer sandwich
measured 2.526592/2.535264/2.530080 ms in
`20260808T104352Z-452458`, a 6.928-us regression against the control mean.
The repeated LDU completion waits delayed later loads.

The second form removed those waits. Lane 0 published through a small
out-of-line helper immediately after the compute-side acquire observed the Q
snapshot; the barrier address traveled in otherwise-zero Q-descriptor fields.
An inline pointer-carrying version had first raised the image to 128 registers
and was rejected before GPU timing. The out-of-line form restored 96 registers
and passed full S128 correctness/token 24748 in
`20260808T104909Z-456622`, but the matched control/snapshot/control run was
2.526144/2.529408/2.524608 ms in `20260808T104948Z-457191`: still 4.032 us
slower than the control mean.

This establishes a publication floor for this design: all 64 disjoint Q
snapshots still require 64 cross-SM updates per layer, or 2,048 per token.
Splitting the wait into per-head barriers would move individual clear tiles
earlier but would retain those updates, so expanding the barrier set around
this losing primitive is not justified. All snapshot opcodes, descriptor
packing, barrier compaction, schedule selectors, and helper code were removed.
Keep the one-layer-delayed RMS-gated cleanup. A future no-clear or deeper
buffer-lifetime design should eliminate publication traffic rather than only
subdivide it.

## Rejected GEMV input-slot recycling

An allocator-pressure proof delayed an ordinary M64N8 GEMV's TMA output
descriptor and retained one 8-KiB physical slot from the final four-slot A
tile after its UMMA completion. The first form made compute lane 0 bind the
shifted descriptor to that slot. It preserved the exact image's 96 registers,
nine hardware barriers, 96-byte stack, and zero spills, and eight consecutive
M4096/K4096 GEMVs passed in job `20260808T112002Z-482657`. It was nevertheless
slower in isolated 1,001-sample sandwiches: 7.072 us versus 7.008/7.008 us at
K4096 (`20260808T112042Z-483417`) and 20.736 us versus 20.608/20.608 us at
K14336 (`20260808T112125Z-483774`). A full S128 correctness run passed and
returned token 24748 in `20260808T112317Z-485180`, but the exact-image
control/variant/control medians were 2.523680/2.526624/2.527968 ms in
`20260808T112400Z-485890`.

The stronger form moved the 16-byte descriptor bind to LDU0. The deferred
command follows the final A load on the same ordered track, so LDU0 can issue
that TMA, overwrite its now-dead slot descriptor, and publish the normal M2C
phase without changing the compute barrier or epilogue. This recovered the
standalone cost: K4096 tied exactly at 6.976 us and K14336 measured 20.736 us
versus 20.768/20.768 us controls in jobs `20260808T112824Z-489979` and
`20260808T112901Z-490262`. Keeping the descriptor lookup inside only the rare
deferred switch case was essential; putting its predicate on every LDU command
slowed the whole image.

Same-image schedule sweeps showed why the mechanism initially looked useful.
On S128, none/out/down/all/none measured
2.531744/2.533504/2.525376/2.525120/2.534400 ms in
`20260808T113549Z-495903`. Down-only therefore gained 7.696 us against the
control mean, while attention-output-only regressed 0.432 us. A reversed
down/control/down run measured 2.527072/2.533728/2.526336 ms
(`20260808T113738Z-497289`), again favoring down-only by 7.024 us. Down sits
behind staged SiLU readiness and can use the earlier slot release for competing
weight prefetch; the already-phased attention output does not.

The decisive comparison used detached commit `078be70` and the experimental
tree as separate compiled images under one GPU lock. The original 11-op image
measured 2.524832/2.525120 ms around a 2.526400-ms down-only variant in
`20260808T114112Z-500427`: the experiment was 1.424 us slower than the old
binary mean. Reusing the existing issuer-only compute opcode instead of adding
a twelfth dispatch case did not rescue it; old/new/old measured
2.524992/2.529120/2.524160 ms in `20260808T114650Z-505068`, a 4.544-us
regression. The 32-entry shifted-descriptor ring added 512 bytes of static
shared memory (7,536 versus 7,024 bytes), and the runtime/handler footprint
cost more globally than down scheduling recovered. All recycling opcodes,
descriptor state, scheduler selectors, tests, and benchmark hooks were
removed. No additional barrier is justified for this path; the existing
LDU-owned M2C phase was already sufficient for correctness.

The next cleanup study should instead compare versioned/no-clear Q buffers
with a deeper periodic clear pipeline. Account for descriptor storage and
barrier IDs explicitly: expand the dedicated barrier set if a buffer version
has a real lifetime frontier, and do not place a new predicate on every LDU or
allocator command merely to support a rare cleanup event.

The restored exact 11-op image rebuilt at 96 registers/nine barriers with no
spills, and all 34 host tests passed. Independent 1,001-sample S128 restoration
runs measured 2.523680 ms in `20260808T105450Z-461557` before the proof and
2.520288 ms in `20260808T115311Z-510386` after all experimental code was
removed; the latter also returned the correct one-token output.

## Rejected no-clear and two-stage periodic Q cleanup

The 27-slot image requalified cleanup deletion before attempting a versioned
buffer implementation. Omitting all 64 per-layer Q-clear tasks is correct for
one fresh Q generation: the complete S128 tensor check and token 24748 passed
in `20260808T115803Z-514114`. It is not a reusable inference design because
the two Q projection folds reduce into the same destination. A matched
clear/no-clear/clear timing upper bound nevertheless measured
2.521664/2.534912/2.524352 ms over 1,001 internal-timer samples in
`20260808T115852Z-515205`. Deleting the pulse regressed 11.904 us against the
control mean, so independent or versioned buffers cannot win merely by
removing the store traffic.

The diagnostic image explains the loss. With clear enabled versus disabled,
median allocator-slot stall was 545.984 versus 574.400 us, compute M2C wait
was 922.944 versus 937.280 us, LDU0 dependency wait was 7.520 versus
17.600 us, and LDU1 dependency wait was 640.096 versus 673.184 us in
`20260808T120249Z-518607`. Thus the early clear wave deliberately occupies a
small amount of allocator/compute/store capacity and prevents a larger burst
of speculative loads from reaching both LDU dependency frontiers. A future
no-clear design must replace that useful pacing with a cheaper explicit
mechanism; simply adding storage removes it.

A deeper periodic proof retained every clear but split the 64-tile wave into
two 32-tile cohorts with the original tile-to-SM ownership. The first half
stayed behind current Q/K/V and the existing pre-attention-RMS lifetime edge;
the second half moved after output projection, post-attention RMS, SiLU, or
down projection. Controls at the beginning and end averaged 2.520432 ms. The
four variants measured 2.522240, 2.520800, 2.527456, and 2.524256 ms,
respectively, over 501 samples in `20260808T120737Z-521830`. Post-RMS is a
0.368-us tie inside drift, while every later split loses the early pacing.
All split/no-clear selectors and generalized clear-slice code were removed.

No extra synchronization frontier was necessary for the split proof: both
halves become safe at the same current-layer pre-RMS barrier, and delaying an
already-safe store does not justify a new counter. The barrier budget is also
tight. The default and system groups consume eight global counters, while 30
layer barrier names times 33 instances consume 990; alignment makes the total
1,000 of the VM's 1,024 counters. Naively adding one layer barrier would use
1,032 counters after alignment and overflow. Only `bar_pre_attn_rms` uses the
post-layer tail instance, so targeted tail-instance compaction could make room
for exactly one additional per-layer frontier if a later buffer-generation
design demonstrates that it is semantically required. Do not spend it on
queue timing alone.

The exact production source and 11-op, 96-register, nine-hardware-barrier
image were restored. A final 1,001-sample S128 internal median was 2.521024 ms
in `20260808T121039Z-524374`. Retain the one-layer-delayed 64-SM clear pulse.

## Retained critical-path down-tail offload

A fresh 12-op marker trace exposed path imbalance that uniform down-task
counts hid. In the final layer, SM96--103 reached `layers_done` at roughly
79.7--80.1 us absolute, while auxiliary SM128--135 reached it near
74.8--75.0 us (`20260808T121420Z-527159`). The retained schedule moves only
the final two M64 tiles of the high-K down projection, four K folds per tile,
from SM96--103 to SM128--135. The remaining 1,536 output rows keep their 96
contributors on SM0--95. All 448 down contributors, output ranges, reduction
stores, and `bar_layer` releases are unchanged.

The first same-image screen measured control/SM128/SM144/control at
2.525312/2.519808/2.572672/2.521696 ms over 501 internal-timer samples in
`20260808T121659Z-529599`. SM128 gained 3.696 us against the control mean;
SM144 lost 49.168 us because its apparent idle time was already compensating
for a later producer path. On the exact 11-op image, control/SM128/control was
2.524672/2.520192/2.524320 ms in `20260808T121938Z-531410`, a 4.304-us gain.
The reversed SM128/control/SM128 order measured
2.520608/2.522880/2.520608 ms in `20260808T122049Z-532592`, a 2.272-us gain.
Across those strict runs, the control and offload medians are 2.524320 and
2.520608 ms.

The matched offload marker trace confirms the transfer rather than a task
deletion: SM96--103 moved to about 75.6--76.1 us absolute, while SM128--135
absorbed the eight tasks and moved to about 80.0--80.6 us
(`20260808T122427Z-535831`). Those auxiliary CTAs do not own the LM-head wave,
and their reduction stores still precede the final RMS frontier. The result is
cross-stage path balancing, not a lower standalone GEMV time. It needs no new
barrier, descriptor, compute operator, or runtime case.

The post-retention multi-track audit also identifies the affected resource
tail. Against the matched pre-offload diagnostic in
`20260808T120249Z-518607`, maximum compute M2C wait fell from 1,158.880 to
1,040.608 us, maximum allocator-slot stall from 806.848 to 682.624 us, and
maximum LDU1 dependency wait from 956.896 to 817.088 us in
`20260808T123230Z-542417`. Median values moved only slightly and sometimes in
the opposite direction. The gain therefore comes from compressing pathological
CTA tails across compute, shared-slot allocation, and the activation-load
track, not from increasing average per-SM throughput.

Full S128 tensor validation and token 24748 passed, and four resident decode
steps exactly produced `[24748, 24748, 24748, 24748]`, in
`20260808T122205Z-533609`. All 34 host schedule/attention tests pass. The final
exact image remains at 96 registers, nine hardware barriers, a 96-byte stack,
zero spills, and 7,024 bytes of static shared memory. Its fresh 1,001-sample
profiling-free S128 internal median is 2.520352 ms in
`20260808T122750Z-538407`: 10.499% faster than strict vLLM's 2.816003 ms and
14.051 us beyond the 2.534403-ms 10%-lead target.

## Rejected numerically safe grouped-down reduction

The fast grouped M128 down proof could not be retained with its BF16 fold
reduction because the error accumulated across 32 layers. A safe variant gave
each of the 16 K folds a distinct FP32 partial record and reduced those records
before adding the BF16 residual. The packed FP32 producer remained correct
(0.141173% mean relative error, 0.031174 maximum) and cost only 0.272 us over
the grouped-BF16 producer: the matched medians were 19.488/19.776/19.520 us in
`20260808T130335Z-567570`, `20260808T130357Z-567898`, and
`20260808T130418Z-568424`. Splitting the partial into four stores increased
the median to 20.128 us in `20260808T131439Z-576402`.

The reducer used both VDCores LDUs concurrently: each loaded one 32-KiB half
of the FP32 record, after which the compute group summed 16 folds and emitted
BF16. Its full internal time was 2.848 us in
`20260808T125158Z-558255`; the same 32-SM resident image had a 0.384-us empty
floor, so the net work was about 2.464 us. Eight serial 8-KiB loads took
5.024 us (`20260808T125058Z-557442`). Direct scalar and explicit `float4`
global-load sidecars took 6.272 and 4.512 us
(`20260808T130853Z-571894`, `20260808T131058Z-573399`), confirming that
the dual-LDU path is the right mechanism.

A complete 152-SM proof assigned 24 reducers to auxiliary SM128--151 and
reused eight main CTAs only after their producer barrier. With packed stores,
its strict 1,001-sample completion-to-completion medians were 24.448 and
24.512 us for main-CTA reuse bases 0 and 120
(`20260808T133235Z-591802`, `20260808T133257Z-592072`). The result was
correct at 0.140358% mean relative error. The fair retained M64 path, measured
to the same output-completion frontier, was 23.296 us in
`20260808T132614Z-586707`. Thus the best safe grouped path loses 1.152 us
(4.95%) even though its producer is faster.

The remaining loss is the grouped producer's TMA-store completion frontier,
not FP32 arithmetic. A compute-owned TMEM-to-register-to-global producer used
compute-group barriers around a single-thread fence (the memory warp did not
join) and then published through C2M. It was correct but took 26.496 us
one-shot versus 26.272 us for the packed TMA producer
(`20260808T133028Z-590183`, `20260808T132332Z-584074`). Fusing only RMS
sum-of-squares cannot recover the gap: the prior shared-RMS versus scale-only
bound is about 0.480 us before accounting for compact reduction/finalization.
All grouped-safe operators, descriptor builders, and benchmark-only schedule
paths were removed. Retain the balanced M64 down implementation.

## Rejected no-clear pacing substitutes

The later no-clear revisit separated useful queue pacing from zero-store
traffic. A schedule-only selector replaced the 64-SM Q-clear wave with one
issue-stage wait on the first phased-attention output needed by each physical
SM. Waiting on the exact group, or uniformly on groups 0, 1, or 2, measured
2.573312/2.573408/2.573568/2.575744 ms versus 2.521088 and 2.516288 ms clear
controls over 301 samples in `20260808T134313Z-600550`. No-clear was
2.531616 ms. An issue barrier therefore creates global head-of-line blocking;
it is not a cheaper form of the clear's asynchronous back-pressure.

Redirecting the same Copy pipeline to harmless scratch isolated the payload.
For 128/256/512/1,024-byte load/store copies, medians were
2.515168/2.516928/2.521472/2.513472 ms, bracketed by 2.516960 and 2.514752 ms
clear controls in `20260808T134624Z-603152`. The slot size and two
memory-to-compute operands stayed unchanged. A second proof compiled the
ordinary Dummy release operator and replaced Copy plus writeback with one
load slot and one M2C/C2M handoff. Payloads of 16, 128, and 1,024 bytes were
again indistinguishable in the short screen
(`20260808T135049Z-606092`), but strict 1,001-sample controls exposed a
real cadence loss: two clear medians averaged 2.525792 ms while two 16-byte
load-only pulses averaged 2.531488 ms
(`20260808T135316Z-608128`), a 5.696-us regression.

Finally, a two-slot 16-byte scratch Copy retained the full handoff topology
while making global traffic negligible. Two clear medians averaged
2.524800 ms and two scratch-pulse medians averaged 2.526224 ms in
`20260808T135447Z-609512`, only a 1.424-us loss. This proves that the
1-KiB Q-zero payload is already hidden; the benefit is the allocator and
two-slot Copy/M2C/C2M phase pulse. Independent Q storage cannot improve the
current schedule merely by deleting the stores, and replacing them with
synthetic work is not a useful final design.

Keep the one-layer-delayed clear. It already uses the existing
`bar_pre_attn_rms` lifetime edge. The issue-wait proof reused existing
phased-attention barriers and was much worse, so there is no evidence for
spending the one potentially recoverable per-layer barrier counter on
cleanup. Reconsider versioned buffers or a deeper clear pipeline only if a
future schedule removes the need for this queue-phase pulse or introduces a
new buffer-reuse frontier. All selectors, scratch tensors, the Dummy opcode
addition, and experimental schedules were removed. The restored 11-op image
returned to 96 registers, nine hardware barriers, a 96-byte stack, and zero
spills; its fresh 1,001-sample S128 internal median was 2.522592 ms in
`20260808T135833Z-612404`.

## Rejected RMS ownership and LDU-port moves after down-tail offload

A fresh combined stage/track diagnostic in `20260808T140126Z-615264`
showed why compute-only placement markers are insufficient. After the
retained final down-tail offload, SM96--103 reached `layers_done` at about
77.9--78.0 us absolute, while the current SM0--7 RMS owners completed the
next-RMS frontier at 84.1--84.3 us and loop progression ended near
84.5--84.6 us. However, SM96--103 also had the largest LDU0 queue wait
(up to 1,870.208 us over the token), whereas SM0--7 owned the largest LDU0
dependency wait (up to 351.488 us). The apparently idle compute cohort was
not an idle load cohort.

Moving the unchanged eight RMS tasks confirmed that distinction. With
2.516512/2.517184-ms base-0 controls, bases 64, 96, 104, 128, and 144 measured
2.521088, 2.526080, 2.529280, 2.528768, and 2.524352 ms over 301 internal
samples in `20260808T140441Z-618195`. The newly freed SM96--103 cohort
regressed 9.232 us against the control mean despite its early compute marker.

A second proof independently routed the RMS weight prefetch and
barrier-gated hidden load to LDU1. On SM0--7, port pairs 01/10/11 measured
2.519392/2.517728/2.515776 ms versus 2.517216/2.517088-ms controls; on
SM96--103 they measured 2.528416/2.524640/2.528896 ms
(`20260808T140803Z-621252`). Only the all-LDU1 current-owner form screened
positive. Its strict 1,001-sample qualification did not justify retention:
two controls averaged 2.519776 ms and two all-LDU1 runs averaged 2.518752 ms
in `20260808T141032Z-623381`, a 1.024-us / 0.04% gain below control drift
and the threshold for changing a global load-track assignment.

No opcode, barrier, or task arithmetic changed in either proof. All ownership
and port selectors were removed. Use simultaneous compute and LDU timelines
before moving work to an apparently spare SM; a completed compute program
does not imply that its asynchronous memory queues are empty.

## Rejected fused Up/SwiGLU gate-slot reuse

The fused tail already keeps the gate projection in a RegStore slot, reloads
it into compute registers, overlaps SiLU with the final independent up UMMA
group, and allocates a second shared slot for the output TMA store. Three
proofs tested eliminating that final allocator round trip by overwriting the
now-dead gate payload.

The first proof issued the output store through a special descriptor slot and
copied its 16-byte `MInst` into the gate slot. Moving the required compute
rendezvous immediately behind the gate load kept it under the outstanding
UMMA group and restored the exact 96-register image. A compute-group barrier
measured 2.523040 ms; narrowing it to the disjoint warp-owned tile partitions
measured 2.522592 ms. Both lose to the fresh 2.519104-ms control. A direct
queue encoding removed the metadata copy but was unsafe: the special
descriptor could be overwritten after compute consumed its readiness token
and before the store warp serviced it. Job `20260808T163532Z-716775` exceeded
the tensor threshold at 5.112% final-hidden and 5.161% final-RMS error.

The lifetime-correct proof made the memory control warp remember the gate
RegStore's physical mask and issue the later TMA descriptor directly against
that occupied slot. Full tensor/token correctness passed, but both an
allocation-loop form and a narrowed control-op form measured
2.541248/2.542016 ms. Matched track profiles explain the regression. Relative
to control, median allocator stall fell 553.312 to 494.080 us, while compute
M2C wait rose 926.624 to 944.992 us and LDU0/LDU1 queue-idle time rose
1,785.504/1,855.520 to 1,811.200/1,891.328 us. The profiled kernel span grew
2.606 to 2.630 ms (`20260808T164553Z-722556` versus
`20260808T164755Z-723547`).

The ordinary output allocation is useful admission pacing: deleting it lets
the control/load streams advance into work the consumer cannot yet use. All
reuse opcodes, queue formats, selectors, and schedule changes were removed.
Future slot-lifetime work must retain the current publication cadence or move
the corresponding consumer earlier at the same time.

## Rejected cross-task gate/up activation retention

The gate and up tail projections consume the same four K1024 groups of the
RMS-normalized activation, so a cross-task proof kept gate's eight physical
8-KiB B slots resident and let the same CTA reuse them in the following
up/SwiGLU task.  This preserved the separate projection operators and their
compute order; only the shared-memory lifetime crossed the task boundary.
Full S128 tensor validation and exact token 24748 passed in
`20260808T170003Z-730493` and `20260808T171454Z-738493`.

The traffic reduction was real.  A matched track image removed 128 LDU0
commands and 128 M2C publications per CTA over the resident token, and median
compute M2C wait fell from 924.352 to 900.736 us.  However, holding eight
slots across the boundary raised median allocator stall from 544.512 to
630.400 us.  Splitting the memory program to retain only the last one, two,
or three groups also inserted a second RepeatM phase: 301-sample medians for
zero through four retained groups were 2.525088, 2.529952, 2.539520,
2.550656, and 2.524864 ms (`20260808T170937Z-735405` through
`20260808T171136Z-736706`).

An exact 11-op, 96-register image removed the diagnostic dispatch-footprint
confound.  Full four-group retention measured 2.525760 ms in
`20260808T171531Z-739081`, versus 2.522976 ms for the same binary with normal
reloads in `20260808T171616Z-739425` and 2.517504 ms for fresh production in
`20260808T165134Z-725676`.  The activation reloads are already overlapped and
their slot release supplies useful allocator cadence; eliminating them moves
pressure to the shared arena rather than shortening the token frontier.  All
retention tasks, schedule forms, and build selectors were removed.

## Retained 152-SM M128 output projection

The output projection is the first stage retiled around the GB200's full 152
SM topology rather than inherited M64 ownership.  Its weights are packed as
M128K128 tiles and use the generic M128 UMMA family.  Six M128 rows use eight
K512 folds and the other 26 use four K1024 folds, producing 48 + 104 = 152
independent tasks.  K<2048 consumers map to SM64--139 outside the attention
owners; the late folds map to SM0--63 and SM140--151.  This preserves the
arithmetic, number of reduction records, and one-task-per-SM topology while
halving each task's B-activation footprint and widening its TMEM/register
epilogue extent.

The 301-sample internal task-count sweep measured 2.519712, 2.506848,
2.508512, and 2.506432 ms for 128, 136, 144, and 152 tasks.  A strict
1,001-sample same-image qualification measured 2.511072 ms for the 136-task
form, 2.508832 ms for the 152-task form, and 2.521664 ms for the original M64
schedule (`20260808T174050Z-753404`, `20260808T174133Z-753675`, and
`20260808T174215Z-754048`).  Keeping all physical owners is therefore faster
than reducing task count and wins 12.832 us over the matched M64 control.

Matched track profiles in `20260808T174449Z-755477` and
`20260808T174532Z-755968` show median compute M2C wait falling from 715.136 to
696.352 us, store service from 115.296 to 113.504 us, and contended compute
calls from 40 to 30.  Allocator stall stayed neutral at 607.392/608.352 us;
LDU1 dependency wait rose from 650.304 to 656.000 us.  The retile improves
the activation-to-UMMA/epilogue track without pretending that the whole CTA
is compute-idle or spending another shared slot.

The minimal default passed full S128 tensors and exact token 24748 in
`20260808T175059Z-758790`; four resident decode steps exactly matched
`[24748, 24748, 24748, 24748]` across the KV128 boundary in
`20260808T175136Z-759221`.  All 34 schedule/runtime host tests pass.  The
fresh 1,001-sample profiling-free median is 2.507136 ms in
`20260808T175211Z-759358`, 10.968% faster than strict vLLM's 2.816003-ms S128
baseline and 27.267 us past the 2.534403-ms 10%-lead threshold.  Adding the
M128 family makes the selective manifest 12 operators, but the resident
kernel remains at 96 registers, nine barriers, a 96-byte stack, 7,024 bytes
of static shared memory, and zero spills.

## Rejected M128 materialized-MLP prefix

The separate 6,144-row gate and up prefixes were retiled from 96 M64 full-K
tasks to 48 M128 tiles with two K2048 folds, still 96 tasks per projection.
This preserved physical owners, arithmetic, and weight bytes per task while
halving each task's B activation.  Since the two folds use reduction stores,
16 auxiliary CTAs cleared one contiguous 12-KiB token row of each output
before attention and contributed to the post-attention-RMS readiness
barrier.  Static clear addresses are valid across layers, but their barrier
field must be group-relative; omitting the group flag made every later clear
arrive on layer 0 and deadlocked the loop at layer 1.

The corrected form passed full S128 tensors and token 24748 in
`20260808T181150Z-771165`.  Moving the 16 clear CTAs to bases 104, 120, 128,
and 136 produced 301-sample medians of 2.514752, 2.512672, 2.510784, and
2.508192 ms.  Strict 1,001-sample candidate runs averaged 2.513216 ms
(`20260808T181453Z-773045` and `20260808T181614Z-773686`), 2.368 us slower
than the intervening 2.510848-ms M64 control in
`20260808T181535Z-773247`.  A no-clear fold-1 form used only 48 tasks per
projection and regressed sharply to 3.306528 ms
(`20260808T181810Z-774738`), demonstrating that 16 simultaneous owners per
2,048-row shard do not expose enough system bandwidth.

Matched track images show that clear traffic is not the deciding cost.
M128 lowered median LDU0/LDU1 queue wait from 1,764.512/1,843.008 to
1,741.664/1,815.296 us, while compute M2C wait rose 941.408 to 989.408 us,
allocator stall 550.656 to 561.216 us, store service 118.464 to 125.664 us,
and store-barrier service 104.864 to 117.632 us
(`20260808T182032Z-776155` and `20260808T182111Z-776347`).  A deliberately
invalid no-clear timing proof left M2C wait at 987.776 us and the diagnostic
kernel span unchanged (`20260808T182210Z-776910`).  Thus independent
per-layer buffers cannot recover the loss: splitting each output into two
publications moves the bubble from LDU queues to M2C/store synchronization.
All experimental buffers, descriptors, selectors, and schedules were
removed; retain the M64 prefix.

## Retained M128 fused Q projection

Q is a favorable M128 retile even though the materialized MLP prefix is not.
Use four K1024 folds of four M128 tasks per query head instead of two K2048
folds of eight M64 tasks.  This preserves 128 total contributors, 16 releases
per head, and the proven SM0--103 plus SM128--151 owner set, while halving the
RMS-activation bytes requested by each task.  Q weights need their own
M128K128 tile-major packing and activation/output TMA descriptors; K and V
must retain their M64K256 resources.

The 2,001-sample isolated M64/M128/M64 medians are 7.424/6.880/7.392 us.
Full S128 internal medians are 2.498880/2.480064/2.498816 ms for the matched
control/candidate/control.  Stage profiles move Q p50/max from
9.920/12.640 to 8.256/11.808 us and attention max from 17.632 to 15.168 us,
showing that the smaller activation requests advance downstream queues.  The
default passes full token/tensor validation and four-token KV128 crossing;
the 13-op image stays at 96 registers with no spills.  Its final selector-free
1,001-sample internal median is 2.474624 ms, 12.123% faster than the strict
2.816003-ms vLLM S128 baseline.
