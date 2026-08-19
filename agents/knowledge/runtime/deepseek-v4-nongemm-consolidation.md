# DeepSeek-V4 non-GEMM consolidation

## Goal

Minimize the production resident image's compute-operator count and generated
code while preserving the ordinary VDCores compute-task plus memory-task
execution model.  Performance decisions must be made in the full resident
image.  Device-side per-operator time must be at least 10% below the matched
vLLM/FlashInfer primitive, except that attention may use a relaxed gate.

The current `benchmarks/deepseek_v4_resident.ops` file is a diagnostic
superset.  It contains mutually exclusive scalar/native, BF16/FP32,
pooling, attention, and quantization variants.  It is not the desired
production image.

## Normal MXFP8 contract

- One contiguous group of 128 activation values has one UE8M0 scale.
- Every native activation K128 tile contains the FP8 payload plus the packed,
  replicated SFB representation required by UMMA.
- `ScalePack=2` groups two independent K128 tiles for transport and pipeline
  efficiency.  It does not change the one-scale-per-K128 quantization rule.
- Projection weights and their scales are converted to the final native UMMA
  tile layout outside decode.  Runtime weight conversion is not a compute op.
- A producer that owns a complete K128 group should write the next consumer's
  native layout directly.  The physical destination may be HBM or a retained
  shared-memory lease; the logical layout is the same.

## Conversion placement

Prefer producer writeout:

- mHC pre plus RMS writes the shared hidden MXFP8 activation once.
- Q-rank's global RMS finalizer writes native Q-b/index-Q input directly.
- attention reduction applies inverse RoPE and writes native O-a input.
- each O-a output partition writes native O-b input.
- paired W1/W3 epilogues apply bounded SwiGLU and write native W2 input.
- head mHC plus RMS writes native vocabulary input.

Consumer read-in conversion is valid only when each K128 group is converted
once.  It must not be repeated by every output-row or split-K task.  If a
producer cannot emit the layout and the activation fans out, use one
independent VDCores quant task that publishes one reusable packed tensor or
lease.  A separate CUDA launch is reserved for offline work or for a measured
persistent-image footprint regression.

Keep an independent conversion/finalization phase when:

- the producer does not own a complete K128 scale group;
- a global statistic or multi-producer reduction must finish first;
- several consumers reuse one packed result;
- consumers require incompatible formats;
- fusing conversion materially increases the common GEMM's code/register
  footprint.

## Operator disposition

Merge into the common GEMM epilogue or writeout:

- router bias/hash/top-6 selection;
- Q-b/KV complete-head RMS and RoPE finalization;
- index-Q RoPE plus Hadamard;
- O-a to O-b quantization and packing;
- paired W1/W3 bounded SwiGLU plus W2 activation packing;
- W2 route scaling;
- attention O-b mHC post when each output tile is independently finalizable;
- vocabulary local argmax;
- FP32-to-BF16 conversion and reduction-target initialization semantics.

Merge non-GEMM phases:

- mHC pre, RMS, and hidden MXFP8 packing;
- attention reduction, inverse RoPE, and O-a MXFP8 packing;
- expert join and FFN mHC post;
- head mHC, final RMS, and vocabulary MXFP8 packing;
- ratio-128 packed history/tail pooling into one local-partial producer.

Retain only true independent/global phases:

- `HC_PRE_RMS_QUANT`;
- one runtime-parameterized `VECTOR_FINALIZE` for global RMS with optional
  RoPE, Hadamard, or native packing;
- one selected attention implementation, plus a reducer only if the selected
  split implementation requires it;
- `POOL_PARTIAL` for the 512-wide global-RMS history path;
- index score and global top-k;
- `EXPERT_JOIN_HC_POST`;
- `HC_HEAD_RMS_QUANT`;
- global argmax reduction.

RoPE-table population is one-time fixed-address memory initialization, not a
steady-state compute task.  Control instructions such as loop and terminate
remain runtime operations and are not model-domain operators.

## Production-image rules

- Select one native MXFP8 path and one attention path.
- Keep legacy, diagnostics, and mutually exclusive experiments in separate
  manifests.
- Avoid shape-specific handlers when a runtime loop over K128 groups expresses
  the same operation.
- Add epilogue modes to the one common MXFP8 compute task rather than cloning
  epilogue bodies into multiple GEMM specializations.
- Consolidate small global transforms into one vector-finalizer implementation
  rather than separate 128/512/1024/4096 generated handlers.
- Do not add `IssueBarrier`, compute-side thread fences, raw-address operands,
  or a second CUDA launch.

## Verification and acceptance

For each consolidation milestone:

1. Generate and build the full resident operator image, not a focused task
   image.
2. Record selected operator count, `dae2` registers, spills, stack, and SASS
   text size.
3. Validate the full resident output token and relevant intermediate numeric
   tolerances.
4. Measure device-side operator spans inside that same full image.
5. Compare each non-attention operator with the matched vLLM/FlashInfer device
   result; require `vdcores_us <= 0.90 * reference_us`.
6. Measure full-image cold and hot resident latency and reject a local win that
   regresses the full flow.

Attention is correctness- and full-flow-gated first.  Its external 10% target
may be relaxed when the selected implementation materially reduces generated
code or improves the complete resident schedule.

## Common MXFP8 production convergence (2026-08-19)

The production Q-b and shared-W2 call sites now use only
`OP_FP8_GEMV_UMMA_COUPLED_SM100`.  The following legacy handlers were
transitional schedule artifacts and are not selected:

- stream `OUTPUT_GROUPS_2` and `OUTPUT_GROUPS_1` for Q-b's two-M128 work and
  per-SM tail;
- split-K FP32 `OUTPUT_GROUPS_2` and `OUTPUT_GROUPS_1` for shared W2's regular
  work and scheduler tail.

The common task is always M256/K256 with scale pack two.  Repeated resident
families use the allocator's existing linear layer counter to select one
16-byte `[weight_stream, activation_stream]` plan record.  A flag in the
existing coupled command's size field requests this selection; it does not add
a memory opcode, compute opcode, allocation path, or device-side projection
identity.  Immutable weights for every family layer are compacted during
setup, and the token-dependent activation address remains common to the
layer plans.

`benchmarks/deepseek_v4_resident_compact.ops` is the measured production
manifest.  Its 32 entries exactly match the operators queued by the full
43-layer/context-128/full-vocabulary job.  It contains neither the four legacy
FP8 UMMA variants nor the diagnostic `OP_PROFILE_EVENT`.  Relative to the
57-handler diagnostic image, the final image has these build properties:

| Property | Diagnostic image | Production image |
|---|---:|---:|
| selected compute handlers | 57 | 32 |
| `dae2` registers | 221 | 162 |
| stack | 224 B | 160 B |
| spills | 0 B | 0 B |
| static shared memory | 2,752 B | 2,752 B |
| `runtime.o` ELF text | 2,836,877 B | 1,503,013 B |
| SASS dump | 71,057 lines | 37,841 lines |

The exact 32-handler full-image job `20260819T054742Z-3015429` queued 230
compute and 1,148 memory instructions, used one persistent launch, passed the
full-head token-201 reference, and measured 14.617408 ms prime plus 14.427776
ms for its one timed sample.  The previous schedule needed 1,212 memory
instructions because each legacy M128/tail form emitted more commands.

Focused programs containing only the new task were then run from this same
32-handler binary.  They use the resident kernel's built-in device start/end
counters, so no profile compute handler is present:

| Production shape | Cold device envelope | Hot median | Matched DeepGEMM | Result |
|---|---:|---:|---:|---|
| Q-b M32768/K1024, unsplit, 128 SM | 9.248 us | 5.520 us | 6.5608 us | 15.9% faster; passes |
| shared W2 M4096/K2048, split-K2, 32 SM | 8.832 us | 5.056 us | 4.0048 us | 26.2% slower; fails |

An isolated balanced-K W2 screen over 128 SMs reached 3.776 us hot and 6.944
us cold.  That is only 5.7% faster than DeepGEMM and still misses the 10%
gate; it also cannot replace the production placement directly because shared
W2 currently overlaps the six routed W2 partitions.  Keep the common opcode,
but treat shared-W2 placement/reduction scheduling as unresolved performance
work rather than restoring a specialized compute handler.

## Runtime-width vector consolidation (2026-08-19)

The two bounded-SwiGLU handlers for widths 128 and 2048 are now one
`OP_DSV4_SILU_CLAMP_MUL` handler carrying the width in the compute command.
Likewise, the two partial-RoPE handlers now use one `OP_DSV4_ROPE_64` command
carrying the full row width; the operation still rotates only the final 64
dimensions.  The production image fell from 32 to 30 selected handlers.  The
exact full checkpoint job `20260819T061039Z-3036657` queued precisely those
30 handlers—no diagnostic or retired handler—used one persistent launch,
passed token 201, and measured 14.536864 ms cold plus 14.591200 ms for the
following sample.

Every focused number below was measured by running the indicated schedule
from that same 30-handler full-inference binary.  The benchmark schedule is
small, but its generated kernel image is not:

| Task | Full-image cold / hot | Previous cold / hot |
|---|---:|---:|
| bounded SwiGLU K128 | 4.640 / 2.624 us | 6.080 / 2.560 us |
| bounded SwiGLU K2048 | 5.184 / 2.720 us | 6.432 / 2.816 us |
| RoPE D512 forward / inverse, 64 sequential rows | n/a / 96.704 / 97.472 us | n/a / 95.584 / 96.352 us |
| RoPE D128 forward / inverse, 64 sequential rows | n/a / 94.048 / 95.344 us | n/a / 93.856 / 94.080 us |
| coupled Q-b M32768/K1024 | 9.600 / 5.696 us | 9.248 / 5.520 us |
| coupled shared-W2 M4096/K2048 split-K2 | 8.032 / 5.280 us | 8.832 / 5.056 us |

All vector results passed their numeric oracle.  RoPE is on the relaxed
attention path; its 0.2--1.3% local change is accepted for one fewer handler.
The Q-b task remains 13.2% faster than its 6.5608-us DeepGEMM reference in
this final code layout.  Shared W2 remains below its external performance
gate and is still a scheduling issue, not grounds for restoring stale
handlers.

The non-4096 shared-memory RMS variants now use one runtime-width
`OP_RMS_NORM_F16_SMEM`; the register-cached K4096 implementation remains
specialized.  This removes a second selected handler and four globally stale
shape handlers.  The exact 29-handler image uses 168 registers, nine
barriers, a 160-byte stack, 2,752 bytes static shared memory, and no spills.
Its `runtime.o` is 1,445,720 bytes, the CUDA fatbin section is 1,428,640
bytes, and its SASS dump is 36,353 lines.  Relative to the 30-handler image,
the fatbin shrank 34,104 bytes and SASS by 768 lines.

The RMS A/B was also made inside the respective complete inference images:

| RMS width | Old cold / hot | Runtime-width cold / hot |
|---:|---:|---:|
| 128 | 3.712 / 2.528 us | 5.376 / 2.432 us |
| 512 | 4.320 / 2.560 us | 5.472 / 2.560 us |

The higher first-use cost is a cold instruction/layout effect; steady-state
K128 improves and K512 is unchanged.  Full job
`20260819T062415Z-3044065` matched all 29 manifest handlers, passed token 201,
measured 14.849600 ms cold, then 14.661856/14.622464/14.520704 ms (median
14.622464 ms).  A no-inline version was rejected at compile time because the
device-call ABI raised `dae2` to 175 registers and a 224-byte stack.

## Head mHC/RMS epilogue fusion (2026-08-19)

The production head now queues one `OP_DSV4_HC_HEAD_RMS` task.  It forms the
4096-wide mHC result in the final output slot, releases the mHC inputs at their
true last use, and runs the cached K4096 RMS calculation in place before
publishing that slot.  This removes the intermediate HBM store/reload and the
standalone `OP_DSV4_HC_HEAD` and `OP_RMS_NORM_F16_K_4096_SMEM` handlers from
the production image.  FP8 block quantization intentionally remains a
separate 32-SM stage; moving its 32 independent K128 groups into the one-SM
head task would serialize useful parallel work.

Measured from the corresponding complete inference images, the two-stage
head mHC plus RMS envelope was 12.128 us cold / 7.872 us hot.  The fused task
is 10.720 us cold / 6.656 us hot, improving first use by 11.6% and steady
state by 15.4%, with maximum absolute error 0.015625 against the composed
BF16 reference.

The exact 28-handler production build uses 144 registers, nine barriers, a
160-byte stack, 2,752 bytes static shared memory, and zero spills.  Its
`runtime.o` is 1,419,872 bytes, CUDA fatbin section 1,402,792 bytes, and SASS
dump 35,729 lines.  Full job `20260819T064037Z-3126034` matched all 28 queued
handlers with no stale selection, used one persistent launch, passed token
201, and measured 14.961696 ms cold followed by
14.557536/14.459552/14.718368 ms (median 14.557536 ms).  The logical graph
lost one stage and one dependency barrier; maximum per-SM queue depth remains
230 compute and 1,148 memory instructions because another SM stream sets each
maximum.

All post-fusion operator checks were repeated from this exact 28-handler
binary rather than inherited from an earlier selective layout:

| Production task | Cold device envelope | Hot median | Status |
|---|---:|---:|---|
| bounded SwiGLU K128 | 5.984 us | 2.592 us | exact |
| bounded SwiGLU K2048 | 6.464 us | 2.752 us | exact |
| runtime RMS K128 | 6.048 us | 2.512 us | max abs 0.015625 |
| runtime RMS K512 | 6.144 us | 2.608 us | max abs 0.015625 |
| coupled Q-b M32768/K1024 | 10.336 us | 5.824 us | exact; 11.2% below 6.5608-us DeepGEMM |
| coupled shared-W2 M4096/K2048 split-K2 | 8.992 us | 5.152 us | exact; external gate still fails |

Partial-RoPE 100-sample medians in the same image were 96.864/97.408 us for
D512 forward/inverse and 94.720/94.912 us for D128 forward/inverse, each over
64 sequential rows on one SM. These are relaxed attention-path checks. The
standalone bounded-SwiGLU task and small RMS tasks do not beat the older
matched Triton primitive numbers; their acceptance here is code-footprint and
full-flow based, not a claim that the strict external primitive gate passed.

## Normal-runtime and coupled-W2 tuning (2026-08-19)

All measurements in this section use the same exact 28-handler inference
image.  No focused compute-op manifest was built.  The common coupled MXFP8
task now publishes one paired M256 output lease and one contiguous TMA store
instead of two M128 leases/stores.  Its runtime K-pair loop is explicitly not
unrolled, avoiding four copies of the large UMMA/barrier body.  LDU0 and LDU1
split each common weight tile and its scales evenly; activation data and SFB
remain on LDU1.  The load still uses one command, one allocation plan, and the
same two task barriers.

The ordinary `dae2` startup now initializes the 48 independent resident TMEM
barriers with 48 threads.  The four 32-entry queue-barrier arrays likewise use
128 threads, one barrier per thread, and the two LDU control barriers use two
additional threads.  The first arithmetic queue-index form was rejected
because it raised the image from 144 to 154 registers.  Four explicit fixed
thread ranges retain the latency improvement at 144 registers.

The resulting exact image uses 144 registers, nine barriers, a 160-byte stack,
2,752 bytes static shared memory, and zero spills.  `runtime.o` is 1,410,496
bytes, its CUDA fatbin section is 1,393,416 bytes, and the SASS dump contains
35,649 lines.  Final focused schedules launched this full binary and measured:

| Production task or placement | Cold device envelope | Hot median | Matched reference | Gate |
|---|---:|---:|---:|---|
| bounded SwiGLU K128 | 4.224 us | 2.560 us | older Triton primitive | fails strict primitive gate |
| bounded SwiGLU K2048 | 4.608 us | 2.752 us | 1.517 us Triton | fails |
| runtime RMS K128 | 5.088 us | 2.336 us | older Triton primitive | fails strict primitive gate |
| runtime RMS K512 | 5.536 us | 2.528 us | 1.338 us Triton | fails |
| coupled Q-b M32768/K1024, 128 SM | 10.048 us | 5.440 us | 6.5608 us DeepGEMM | 17.1% faster; passes |
| coupled shared W2, production 32-task split-K2 | 9.024 us | 4.832 us | 4.0048 us DeepGEMM | 20.7% slower; fails |
| coupled shared W2, balanced 128-task split-K8 | 7.264 us | 3.520 us | 4.0048 us DeepGEMM | 12.1% faster; local pass |

The balanced W2 number is not reported as a production pass.  Job
`20260819T074524Z-3596219` tried that placement in the full DAG and emitted
token zero: routed and shared W2 are still concurrently live even though their
stage objects are appended in order.  Reusing the routed SM range without a
new dependency therefore races the existing accumulator/join schedule.  The
experiment was reverted; the accepted full flow retains the overlapping
32-task shared-W2 placement.

Full job `20260819T074106Z-3562492` selected and queued exactly the same 28
handlers, with no stale handler, used one persistent launch, and passed token
201.  Paired stores reduced the maximum queue image to 230 compute and 1,144
memory commands.  It measured 14.736704 ms prime and
14.733632/14.545056/14.404256 ms hot (median 14.545056 ms), so the runtime and
load changes do not regress the previous 14.557536-ms full-flow milestone.

## Producer-owned SwiGLU and the exact 27-handler image (2026-08-19)

Both FFN gate/up paths now consume the gate result at the up projection's true
producer epilogue.  Routed W1 retains each M128 BF16 gate tile in an LDU-local
register lease; routed W3 reuses the activation, consumes that lease, applies
bounded SwiGLU, and stores the middle activation.  The shared FP8 W1/W3 path
interleaves row shards on the same SM: W1 retains each at-most-15-row gate
tile, and the immediately following W3 consumes it and applies bounded SwiGLU
before its only HBM store.  In both cases the lease is released at final use.
There is no gate/up HBM round trip and no new compute opcode.

Consequently the full command stream no longer queues
`OP_DSV4_SILU_CLAMP_MUL`, and the compact inference manifest removes it.  This
was not measured using a focused selector.  The release extension was built
from all 27 operators used by the complete 43-layer/context-128/full-vocabulary
decode and no others.  The program reported exactly the same 27 unique
operators, proving that the selected image had neither a missing nor a stale
handler.

The exact image uses 144 registers, nine barriers, a 160-byte stack, 2,752
bytes static shared memory, and zero spills.  `runtime.o` is 1,398,776 bytes,
its CUDA fatbin section is 1,381,696 bytes, and the SASS dump contains 35,473
lines.  Full job `20260819T081621Z-3859119` used one persistent launch, passed
token 201, and reduced the resident graph to 2,960 logical stages, 277 queue
stages, 226 maximum compute commands, and 1,136 maximum memory commands.  It
measured 14.501312 ms prime and 14.485600/14.257312/14.198976 ms hot (median
14.257312 ms).  Relative to the immediately preceding 28-handler routed-only
fusion result (14.573184 ms prime and 14.493792 ms one hot sample), prime is
0.5% lower and the new hot median is 1.6% lower.

All refreshed task envelopes below launched the exact same 27-handler binary:

| Production task or placement | Cold device envelope | Hot median | Result |
|---|---:|---:|---|
| fused shared gate+up+SwiGLU M2048/K4096, 56 SM | 33.248 us | 29.024 us | exact quantized BF16 oracle |
| coupled Q-b M32768/K1024, 128 SM | 9.312 us | 5.312 us | exact; 19.0% faster than 6.5608-us DeepGEMM |
| coupled shared W2, production split-K2/32 tasks | 8.864 us | 4.864 us | exact; 21.5% slower than 4.0048-us DeepGEMM |
| runtime RMS K128 | 5.600 us | 2.464 us | max abs 0.015625; external gate still fails |
| runtime RMS K512 | 5.696 us | 2.624 us | max abs 0.015625; external gate still fails |

The removed standalone SwiGLU is intentionally absent from this table: adding
its handler merely to time it would make the image stale and change the code
layout being evaluated.  Its replacement is measured as the actual fused
producer stage and by the complete inference result.

## Pool finalizers and the exact 25-handler image (2026-08-19)

The final standalone runtime RMS uses were also ownership artifacts.  The
ratio-128 attention compressor still performs packed four-SM pooling, but its
BF16 result now goes directly through the already-selected weighted
RMS/RoPE finalizer instead of separate RMS and RoPE stages.  The ratio-4
attention compressor applies pooling, weighted RMS, and RoPE in its existing
one-SM fused finalizer.  The ratio-4 index compressor uses the same finalizer
with its Hadamard epilogue enabled.  This removes intermediate normalized and
rotated HBM rows on those paths.

The resulting command stream no longer uses `OP_RMS_NORM_F16_SMEM`,
`OP_DSV4_GATED_POOL`, or `OP_DSV4_HADAMARD`; it adds the already implemented
`OP_DSV4_GATED_POOL_RMS_ROPE`.  An initial 26-handler validation exposed
Hadamard as the last stale selection, so its timings were discarded.  The
accepted release was rebuilt with exactly the 25 unique operators reported by
the full inference job.

The exact 25-handler image uses 140 registers, nine barriers, a 160-byte
stack, 2,752 bytes static shared memory, and zero spills.  `runtime.o` is
1,373,632 bytes, its CUDA fatbin section is 1,356,552 bytes, and the SASS dump
contains 34,945 lines.  Compared with the exact 27-handler image, this removes
25,144 binary/fatbin bytes, 528 SASS lines, and four registers.

Full job `20260819T083103Z-3994318` selected and queued exactly the same 25
handlers, used one persistent launch, and passed token 201.  The graph fell to
2,835 logical stages, 266 queue stages, and 310 dependency barriers; maximum
per-SM commands remained 226 compute and 1,136 memory.  It measured 13.790272
ms prime and 13.856224/13.749088/13.737344 ms hot (median 13.749088 ms).  This
is 4.9% lower prime and 3.6% lower hot median than the preceding exact
27-handler image.

Only still-selected tasks were refreshed from this exact binary:

| Production task or placement | Cold device envelope | Hot median | Result |
|---|---:|---:|---|
| fused shared gate+up+SwiGLU M2048/K4096, 56 SM | 32.832 us | 29.088 us | exact quantized BF16 oracle |
| coupled Q-b M32768/K1024, 128 SM | 9.408 us | 5.152 us | exact; 21.5% faster than 6.5608-us DeepGEMM |
| coupled shared W2, production split-K2/32 tasks | 7.808 us | 4.672 us | exact; 16.7% slower than 4.0048-us DeepGEMM |

There is no standalone RMS or Hadamard number for this image because neither
handler is selected.  Reselecting dead handlers for a microbenchmark would
violate the full-image measurement boundary and perturb the dispatch layout.

An overlap-safe balanced-K shared-W2 candidate reached 4.032 us in an
isolated invocation of this binary, but the complete 25-handler command stream
stalled after preparation.  That candidate was aborted and production
split-K2 restored.  Its isolated number is rejected: a schedule that does not
complete the full inference image has no reportable performance.

## Full-image acceptance boundary after shared-W2 tuning (2026-08-19)

Subsequent balanced-K work fixed the looped ring-phase correctness problem,
but it did not pass the full-image performance boundary.  The locally fastest
64-SM shared-W2 schedule measured 3.456 us in a focused invocation (13.7%
faster than the 4.0048-us DeepGEMM reference), yet exact 25-handler full job
`20260819T093637Z-382901` measured 14.000352 ms hot median.  That is 1.83%
slower than the accepted 13.749088-ms full-image baseline, so both the balanced
placement and its persistent allocator phase cursor were rejected.  Wider,
fully serialized, and direct-retirement variants were also slower in the full
flow even when their isolated shared-W2 envelope improved.

Production therefore retains the ordinary 32-task split-K2 shared-W2
placement, static per-task ring phase, and UMMA-completion relay.  The restored
artifact was rebuilt from `deepseek_v4_resident_compact.ops`, which selects all
and only the 25 handlers observed in the inference graph.  It uses 140
registers, nine barriers, a 160-byte stack, 2,752 bytes static shared memory,
and zero spills.  The current `runtime.o`/fatbin/SASS sizes are
1,382,616/1,364,152 bytes/35,169 lines.

Confirmation job `20260819T094658Z-476515` queued the same 25 handlers in one
persistent launch, passed token 201, and retained the 2,835 logical stages,
266 queue stages, 310 dependency barriers, 226 maximum compute commands, and
1,136 maximum memory commands.  It measured 13.928480 ms prime and
14.133184/13.784832/13.770144 ms hot (median 13.784832 ms).  The 0.26% delta
from the earlier 13.749088-ms median is treated as run noise, not a regression.
Focused numbers remain useful for diagnosis, but no optimization is accepted
or reported as a production win unless this complete no-stale image also
passes correctness and end-to-end timing.

### Rejected common-W2 load variants and final exact-image refresh

The remaining shared-W2 experiments kept the common compute opcode and were
screened from the exact 25-handler binary. A 64-SM balanced schedule retaining
the normal UMMA-completion relay reached 3.808 us locally, but full job
`20260819T095455Z-544781` regressed to a 14.118624-ms hot median. Separate
scale-arrival barriers, whether scale-first or interleaved with the weight
chunks, measured 4.896 and 4.960 us locally. BF16 cross-split reduction was
4.896 us. Replacing each LDU's two 16-KiB bulk weight requests with one
32-KiB unswizzled TMA tile regressed shared W2 to 5.696 us. All four variants
were rejected and removed; none is part of the production image.

Temporary device counters showed why load-only changes did not help the
ordinary 32-task schedule: allocator/dependency and operand waits were zero,
scale copy was 0.256 us, UMMA issue was 1.568 us, and the task's remaining
roughly 2.4 us was completion/drain plus lease retirement. The diagnostic
counters were removed after this measurement.

The restored production source was rebuilt from
`deepseek_v4_resident_compact.ops`. The compiler selected exactly 25 of 107
available handlers; `dae2` uses 140 registers, nine barriers, a 160-byte
stack, 2,752 bytes static shared memory, and zero spills. Full checkpoint job
`20260819T102833Z-844950` independently reported the same 25 queued handlers,
one persistent launch, 2,835 logical stages, 266 queue stages, 310 dependency
barriers, 226 compute commands, and 1,136 memory commands. It passed token
201 and measured 14.034912 ms prime plus
14.014048/13.796928/13.780544 ms hot (median 13.796928 ms). That is 0.09%
above the restored 13.784832-ms confirmation median and is treated as noise.

The following focused schedules were launched without recompilation from that
same full-inference binary; they are device-envelope measurements, not
focused-handler images:

| Selected production task | Cold device envelope | Hot median | Result |
|---|---:|---:|---|
| head mHC + RMS K4096 | 10.240 us | 6.304 us | max abs 0.015625 |
| fused shared gate+up+SwiGLU M2048/K4096 | 33.472 us | 29.184 us | exact |
| coupled Q-b M32768/K1024, 128 SM | 8.928 us | 5.376 us | exact; 18.1% faster than 6.5608-us DeepGEMM |
| coupled shared W2, split-K2/32 tasks | 7.872 us | 4.848 us | exact; 21.1% slower than 4.0048-us DeepGEMM |

The 117 affected schedule/runtime tests passed before the exact-image build.
Shared W2 remains the only open strict GEMM performance gate; the full-image
methodology does not permit accepting its faster balanced micro-schedule while
the complete inference stream slows down.

### Rejected paired routed-W2 repartition

A subsequent screen paired adjacent routed-W2 M128 outputs in one pipelined
M256 compute command.  This allowed routed W2 to use 14 SMs per expert and
left 68 SMs for a 64-worker shared-W2 split-K4 schedule.  Its focused shared-W2
screen was 6.720 us cold and 3.984 us hot, but that number was diagnostic only.
The candidate was compiled as the complete 25-handler inference image, not as
a focused selector; it used 142 registers, nine barriers, a 160-byte stack,
2,752 bytes of static shared memory, and no spills.

The authoritative full-image job was `20260819T105452Z-1084895`: all 43
layers, context 128, all 129,280 vocabulary rows, one Python enqueue, and one
persistent kernel launch.  The program independently reported exactly the 25
handlers in the candidate manifest, so it had neither missing inference
operators nor stale alternatives.  It passed the full-head token-201 check,
but measured 14.909216 ms prime and
15.124000/14.508032/14.628928 ms hot (median 14.628928 ms).  Relative to the
accepted 13.796928-ms exact-image median, this is a 0.832000-ms or 6.03%
regression.  The graph retained 2,835 logical stages, 266 queue stages, 310
barriers, and 226 compute commands, while maximum memory commands increased
from 1,136 to 1,216.  The paired task therefore failed the full-flow gate even
though it exposed more SMs to shared W2.

The paired routed schedule, 14-SM placement, split-K4 override, and persistent
implicit ring-phase cursors were removed.  The production manifest again
selects `OP_NVFP4_GEMV_UMMA_FP32_SM100`, command-encoded ring phases, routed
W2 at 16 SMs per expert, and shared W2 split-K2 over 32 workers.  Rebuilding
that exact 25-handler image restored 140 registers, nine barriers, a 160-byte
stack, 2,752 bytes of static shared memory, and zero spills.  The known-good
checkpoint timing was not rerun after this source-identical restoration.

### Rejected static-phase 14-SM repartition

A second 14-SM routed-W2 experiment kept the ordinary M128 routed compute
handler and changed only Python scheduling: routed W2 distributed 32 outputs
over 14 SMs per expert, while shared W2 used split-K4 on the remaining 64
workers.  Because split-K4 advances the persistent two-stage MXFP8 ring by two
phases on those workers, setup linked each common task's incoming phase per
physical SM.  Layers 0 and 1 were emitted as separate static blocks, and the
repeated HCA+CSA body was required to have zero net phase advance.  No device
cursor, runtime branch, opcode, or additional compute handler was introduced.

Only the complete image result is authoritative.  Full job
`20260819T113402Z-1434623` ran all 43 layers at context 128 and vocabulary
129,280, queued exactly the same 25 essential handlers, used one persistent
VDcores launch, and passed the full-head token-201 check.  It measured
14.567488 ms prime and 14.542848/14.354528/14.406400 ms hot (median
14.406400 ms), 0.609472 ms or 4.42% slower than the accepted 13.796928-ms
median.  Static expansion of layers 0 and 1 increased queue stages from 266
to 327, maximum compute commands from 226 to 259, and maximum memory commands
from 1,136 to 1,366; logical work and the 25-handler operator image were
otherwise unchanged.

The repartition, split-K4 selection, and setup phase linker were removed, and
the compact phase-neutral 16-SM routed/split-K2 shared schedule was restored.
The general routed-activation last-use correction discovered during the
screen remains covered by a scheduler unit test; it does not alter the
production 16-SM command stream.  The known-good full timing was not rerun
after restoring source-identical production scheduling.

## Final production boundary

The 24-entry compact manifest is the exact inference image, not a benchmark
subset. Its remaining non-GEMM tasks are retained for concrete ownership or
global-dependency reasons:

- `HC_PRE_RMS` and `HC_HEAD_RMS` are the accepted mHC fusions. Hidden/head
  quantization stays separate because K128 groups are distributed over many
  SMs after the one-SM global normalization.
- ordinary FP8, native UMMA-B FP8, and native NVFP4 quantizers produce three
  genuinely different consumer layouts. Each exists only where the producer
  cannot already emit that layout once for all consumers.
- `ROUTE_TOP6`, argmax partial/reduce, and attention/index selection cross
  outputs owned by multiple GEMM shards. They require a join/reduction; simply
  appending their whole operation to one producer epilogue would be incorrect.
- scalar ratio-4 pooling and packed ratio-128 pooling retain their distinct
  input layouts and SM decompositions. Scalar pooling owns RMS/RoPE and the
  optional index Hadamard epilogue; packed pooling hands its BF16 row directly
  to the common RMS/RoPE finalizer.
- partial RoPE remains a standalone operation where its input has multiple
  producers. Bounded SwiGLU is owned by the routed and shared gate/up producer
  epilogues. Runtime-width RMS and generic Hadamard have no standalone
  production handlers.
- `HC_POST` follows the shared-plus-six-routed expert join, so no individual
  W2 producer owns enough data to absorb it.
- one attention implementation is selected. RoPE-table preload is one-time
  resident initialization; loop and termination commands are VM control, not
  alternate model operators.

The previous vocabulary-GEMV local-argmax fusion remains deliberately absent:
its controlled A/B was token-correct but bimodal and slower. Reintroducing it
would violate the performance gate merely to remove one handler.

### Exact-image measurement refresh

Performance remains defined by the complete inference image, not by compiling
only the operator under study.  The production artifact was rebuilt from
`benchmarks/deepseek_v4_resident_compact.ops`; it selected exactly 25 of 107
available handlers and retained the established 140-register, nine-barrier,
160-byte-stack, 2,752-byte-static-shared-memory, zero-spill footprint.  The
full graph independently queued the same 25 handlers, so the measured binary
contained every required operator and no stale alternative.

Full checkpoint job `20260819T121317Z-1791252` ran all 43 layers at context
128 and vocabulary 129,280 through one Python enqueue and one persistent
VDcores kernel launch.  It restored the accepted graph of 2,835 logical
stages, 266 queue stages, 310 dependency barriers, 226 maximum compute
commands, and 1,136 maximum memory commands.  The full-head oracle passed with
token 201.  The prime launch measured 14.065888 ms; three hot samples were
14.088352, 13.818560, and 13.901088 ms, for a 13.901088-ms median.  This is
0.75% above the previous 13.796928-ms exact-image median and is within the
observed run-to-run spread.

A temporary one-group M128 version of the common MXFP8 task was rejected
before this run because it failed the smallest numeric check.  During
restoration, pairing the production M256 compute epilogue with the obsolete
two-store scheduler exposed a useful invariant: the common task consumes one
combined 256-row output lease.  The mismatched form produced four extra
per-SM memory commands and stalled the full image.  Restoring the paired store
returned the graph to 1,136 memory commands and gave bit-exact output for both
M128 accumulator halves.  Neither failed form contributes a performance
number or handler to the production result above.

## O-a conversion fanout consolidation (2026-08-19)

Legacy attention O-a was the sole production user of
`OP_FP8_BLOCK128_GEMV_BF16_SM100`.  That handler converted the same 4,096-wide
BF16 activation independently in each output-row shard.  Production now
quantizes each of the eight O-a groups once into its existing FP8 payload and
one-scale-per-K128 buffers, publishes the group with a direct dependency, and
then uses the already-selected `OP_FP8_BLOCK128_GEMV_SM100`.  No new opcode,
buffer, layout, or CUDA launch was introduced.  The now-unused resident-builder
helper and the stale BF16-input GEMV selector entry were removed; the generic
BF16-input task remains available only in diagnostic source/manifests.

Performance was not measured from a reduced O-a selector.  The rebuilt
`benchmarks/deepseek_v4_resident_compact.ops` artifact contains all and only
the 24 handlers required by inference.  The complete graph independently
reported the identical 24-name set.  The build retains 140 registers, nine
barriers, a 160-byte stack, 2,752 bytes of static shared memory, and zero
spills.  Relative to the preceding exact 25-handler artifact, `runtime.o`
falls from 1,382,616 to 1,333,416 bytes, its CUDA fatbin from 1,364,152 to
1,314,960 bytes, and its SASS dump from 35,169 to 34,065 lines.

Full checkpoint job `20260819T122847Z-1935561` ran all 43 layers at context
128 and vocabulary 129,280 through one Python enqueue and one persistent
VDcores kernel launch.  It passed the full-head oracle with token 201.  The
extra once-per-group quantization stages expand the represented loop graph to
3,179 logical stages, 298 queue stages, 366 dependency barriers, 230 maximum
compute commands, and 1,164 maximum memory commands, but do not materially
change the full flow.  Prime measured 14.109056 ms and hot samples were
14.199616, 13.835808, and 13.927808 ms (median 13.927808 ms).  Against the
preceding exact-image 13.901088-ms median this is +0.19%, within observed run
noise, while removing one generated handler and 49,192 fatbin bytes.  O-a is
part of the relaxed attention boundary; correctness, code size, and complete
inference latency therefore accept this consolidation.

## Full-image native vocabulary head (2026-08-19)

The full-vocabulary projection now uses the already-selected common native
MXFP8 quantizer and coupled UMMA GEMV instead of the scalar block-FP8 GEMV.
The checkpoint head is packed once during setup; the resident graph quantizes
the final normalized activation directly into the native activation layout,
then writes BF16 logits for the unchanged argmax partial/reduce stages.  This
adds no compute handler and does not remove the scalar block-FP8 path, which
the shared gate/up projection still requires.

All timing below came from the complete production image, not a focused
selector.  `build/generated/dae/selected_compute_ops.inc` and
`benchmarks/deepseek_v4_resident_compact.ops` decode to the same ordered
24-entry operator list.  The installed artifact used `runtime.o`
SHA-256 `e275de9b9749bfc923de1daf6d8c884c88edee49a1e672b3cdf17de483295d07`;
there was no CUDA rebuild between the full inference run and the focused
device-envelope run.  Thus the focused result includes the instruction and
register footprint of every essential inference operator and no stale
alternative.

Full checkpoint job `20260819T125055Z-2133510` independently queued those
same 24 handlers, ran all 43 layers at context 128 and vocabulary 129,280 in
one Python enqueue and one persistent VDcores launch, and passed the full-head
oracle with token 201.  Prime was 13.789056 ms; hot samples were
13.794784/13.882240/13.685376 ms (median 13.794784 ms).  Relative to the
immediate scalar-head image's 13.927808-ms median this is -0.133024 ms
(-0.96%).  The represented graph remains 3,179 logical stages, 298 queue
stages, and 366 dependency barriers; replacing the scalar head reduces the
maximum per-SM compute/memory command counts from 230/1,164 to 177/886.

Job `20260819T125238Z-2148932` then measured only the head envelope while
loading that unchanged full image.  For M129280/K4096 on 152 SMs, native
activation quantization measured 3.488 us median, the coupled head measured
97.888 us cold and 86.512 us hot median (85.536 us minimum, 87.744 us
maximum), and numerical max-absolute error was 0.001953.  The matched
DeepGEMM head baseline is 75.421 us, so the common head remains 14.7% slower
and misses the strict 10%-faster target of 67.879 us.  The production change
is retained for its full-flow and queue-footprint improvement, but this
per-operator performance gate remains explicitly open.

A broader conversion of Q-a, KV, O-a, and O-b to the same native coupled path
was rejected before this boundary.  Although it reduced the graph to 206
maximum compute and 996 maximum memory commands, exact-full-image job
`20260819T124151Z-2050479` regressed the hot median to 19.961472 ms.  Same-image
one-layer isolation placed the first regressions in Q-a (+24.512 us), KV
(+6.912 us after Q-a), and O-b (+15.392 us); O-a also introduced a roughly
505-us tail.  Fewer commands therefore did not compensate for producer/join
placement costs.  None of those rejected projection modes or buffers remains
in production source or in the 24-handler image.

## Isolated non-GEMM matrix from the final full image (2026-08-19)

The final measurement loaded the exact generated 24-handler inference image,
not a per-task build.  Its `runtime.o` SHA-256 is
`abb618e3969a75465d43426009cd6e21894bc9995014a8f52862ef0eb705051e`;
the extension SHA-256 is
`cabeb498c8274d9c15695575a52425717dec154196652270bef3efb3cb087b37`.
The image uses 140 registers, nine barriers, a 160-byte stack, 2,752 bytes of
static shared memory, and has zero spills.  Excluding the six projection GEMM
handlers and the two VM control handlers leaves 16 selected non-GEMM handlers.
All 27 production-shape/layout variants below passed their independent
correctness or exact-layout oracle.

Times are microseconds from the resident kernel's device envelope: one fresh
process supplied the cold sample, followed by 20 warmups and 100 hot samples.
The fixed-table operations really preload all four RoPE tables in the same
persistent launch.  Their daggered cold value is therefore the combined
preload-plus-task envelope; their displayed hot value subtracts the measured
3.200-us standalone preload median.  This subtraction is not applied to cold
samples because a single cold difference is too noisy.

| Selected handler / production variant | Cold (us) | Hot median (us) |
|---|---:|---:|
| `PRELOAD_ROPE_TABLES` (four tables) | 6.464 | 3.200 |
| `HC_PRE_RMS` | 14.240 | 10.528 |
| `HC_PRE_RMS`, zero residual-square sum | 12.768 | 10.976 |
| `FP8_QUANT_128`, K=4096 | 4.352 | 2.400 |
| `RMS_FP8_QUANT_UMMA_B`, K=1024 | 3.904 | 2.848 |
| `RMS_ROPE_512_64`, Q64 | 7.264† | 1.152 |
| `RMS_ROPE_512_64`, KV1 weighted | 6.400† | 1.568 |
| `CONTIGUOUS_ATTENTION_512_BLOCK4`, rows=128 | 63.712 | 60.512 |
| `CONTIGUOUS_ATTENTION_512_BLOCK4`, rows=129 | 64.480 | 61.600 |
| `CONTIGUOUS_ATTENTION_512_BLOCK4`, rows=160 | 77.952 | 74.912 |
| `ROPE_64`, D=512 forward | 7.136† | 1.152 |
| `ROPE_64`, D=512 inverse | 7.264† | 1.152 |
| `ROPE_64`, D=128 forward | 6.976† | 0.960 |
| `HC_POST`, BF16 input | 9.248 | 6.768 |
| `HC_POST`, FP32 input | 9.152 | 6.752 |
| `NVFP4_QUANT_UMMA_B`, K=2048 | 6.720 | 3.136 |
| `NVFP4_QUANT_UMMA_B`, K=4096 | 6.368 | 3.136 |
| `GATED_POOL_PACKED8_SHARD128`, ratio=128 | 13.024 | 10.976 |
| `GATED_POOL_RMS_ROPE`, D=512 | 14.720† | 9.824 |
| `GATED_POOL_RMS_ROPE`, Hadamard D=128 | 14.976† | 10.112 |
| `ARGMAX_SMEM_PARTIAL_BF16` | 5.248 | 2.944 |
| `ROUTE_TOP6`, score mode | 9.856 | 6.976 |
| `ROUTE_TOP6`, hash mode | 6.272 | 3.648 |
| `HC_HEAD_RMS` | 10.816 | 6.720 |
| `ARGMAX_SMEM_REDUCE_BF16` | 4.224 | 2.496 |
| `FP8_QUANT_UMMA_B`, K=2048 | 4.096 | 2.112 |
| `FP8_QUANT_UMMA_B`, K=4096 | 4.032 | 2.144 |

The comparison below uses actual installed framework primitives, preallocated
outputs where their API permits, and CUDA-graph hot medians (20 operations per
replay, 100 samples).  Framework single-call "cold" numbers include Python,
dispatcher, and cubin-control work between CUDA events and are not comparable
to the persistent-kernel device envelope, so they are deliberately omitted.
Negative delta means VDcores is faster.

| Matched task boundary | VDcores hot (us) | vLLM / FlashInfer / DeepGEMM (us) | Delta |
|---|---:|---:|---:|
| mHC pre-RMS | 10.528 | vLLM TileLang 4.1432 | +154.1% |
| scalar FP8 K4096 | 2.400 | DeepGEMM packed FP8 2.2216 | +8.0% |
| fused RMS + FP8 K1024 | 2.848 | vLLM RMS + DeepGEMM pack 4.1064 | -30.6% |
| attention, rows=128 | 60.512 | FlashInfer TRTLLM decode 8.7984 | +587.8% |
| attention, rows=129 | 61.600 | FlashInfer TRTLLM decode 10.1432 | +507.3% |
| attention, rows=160 | 74.912 | FlashInfer TRTLLM decode 10.2536 | +630.6% |
| mHC post, BF16 | 6.768 | vLLM TileLang 2.6448 | +155.9% |
| NVFP4 K2048 | 3.136 | FlashInfer NVFP4 quant 2.8136 | +11.5% |
| NVFP4 K4096 | 3.136 | FlashInfer NVFP4 quant 2.9848 | +5.1% |
| top-6 router, score | 6.976 | vLLM 3.8488 | +81.2% |
| top-6 router, hash | 3.648 | vLLM 4.1216 | -11.5% |
| native FP8 K2048 | 2.112 | DeepGEMM packed FP8 2.2288 | -5.2% |
| native FP8 K4096 | 2.144 | DeepGEMM packed FP8 2.2216 | -3.5% |
| argmax partial + reduce | 5.440 | vLLM/Torch argmax 25.0816 | -78.3% |

The vLLM mHC-pre baseline receives the projection result and residual square
sum pre-supplied, matching the VDcores boundary with GEMM excluded.  The fused
RMS/FP8 comparison necessarily uses two framework kernels.  The argmax VDcores
number is the conservative sum of two separately measured resident envelopes,
each including its own VM startup.

There is no shape-and-math-matched standalone primitive in the installed
vLLM, FlashInfer, or DeepGEMM versions for the one-time table preload,
standalone partial RoPE, fused RMS/RoPE, either packed pooling form, FP32 mHC
post, or mHC-head/RMS after the projection is supplied.  These are left
unmatched rather than compared with framework composites that perform extra
cache conversion/insertion or recompute an excluded projection.

The reproducible harnesses are
`benchmarks/deepseek_v4_nongemm_full_image.py` and
`benchmarks/deepseek_v4_nongemm_framework_baselines.py`; raw logs are
`.agentlog/2026-08-19-nongemm-full-image-results.txt` and
`.agentlog/2026-08-19-nongemm-framework-baselines.txt`.
