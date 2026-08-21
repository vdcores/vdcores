# DeepSeek-V4 Routing-To-FFN-Ready Optimization

This note is the implementation plan and acceptance contract for preparing all
FFN inputs from the normalized BF16 hidden vector inside one resident VDCores
launch.  The production path must use the normal allocator/LDU/compute/STU
runtime.  Raw addresses are allowed only for small scalar or metadata records
whose direct access is demonstrably faster and whose producer barrier already
establishes global visibility.

## Boundary And Dependency Graph

Input:

- one normalized contiguous BF16 hidden vector `[4096]`
- contiguous BF16 router weights `[256,4096]`

Outputs required before routed Linear-1 can start:

- FP32 router logits `[256]`
- top-6 FP32 weights and expert ids
- exact native MXFP8 activation data and UE8M0 scale-factor records expected by
  the retained MXFP4 x MXFP8 Linear-1 task

The resident schedule is a fork/join rather than two serialized stages:

```text
ffn_input_ready
  |-- BF16 router GEMV -> router_logits_ready -> top-6 -> experts_ready
  `-- BF16 -> native MXFP8 pack -------------------------> mx8_ready

shared Linear-1 waits on mx8_ready
routed Linear-1 waits on mx8_ready and experts_ready
```

Only dependent activation and metadata loads carry `.bar(...)` dependencies.
Static router and FFN weight loads remain unbarriered.  Do not add an alloc-warp
issue barrier, PDL dependency, compute-side fence, or whole-stage barrier.

## Reference Schedules To Adapt

- vLLM 0.27.1 uses its CTA-local dot-product backend for decode-sized BF16
  routing: one output row per CTA, vectorized BF16 loads, FP32 accumulation,
  warp reduction, and a small shared cross-warp reduction.  VDCores should
  preserve the arithmetic contract but tune rows per resident task because a
  persistent CTA cannot rely on the conventional grid scheduler to run a
  second wave cheaply.
- vLLM's DSV4 score router uses one warp per token, holds 256 scores in
  registers, and performs six unrolled top-k reductions.  The VDCores score
  task should remove its repeated four-warp shared-memory round trips.
- FlashInfer's MXFP8 quantizer uses independent K32 scale groups, two or four
  threads per group, BF16-pair max/convert operations, and packed global
  accesses.  VDCores must emit the final Linear-1 data/SFB layout directly,
  without the existing K128 repeated-scale intermediate.

Reference source versions:

- `vllm/v0.27.1`: `ll_bf16.py`, `_ll_bf16_dotprod.py`, and `dsv4_topk.py`
- `flashinfer/v0.6.16`: `quantization/kernels/mxfp8_quantize.py`

## Implementation Order

1. Add a correctness-first native MXFP8 FFN-input task.  Start with eight
   independent K512 tasks.  Each task loads 1 KiB BF16 through LDU, computes
   sixteen K32 scales, creates the required N8 replication/SFB swizzle in one
   allocator-owned record, and publishes it through STU.
2. Put the eight conversion tasks on resident CTAs concurrently with the BF16
   router tasks.  Initially reserve eight SMs for conversion and keep router
   work on the remaining SMs; then co-tune placement with the winning router
   task shape.
3. Vectorize the existing BF16 router inner loop, emit FP32 logits, and
   constexpr-sweep one, two, and four output rows per resident task.  Prefer a
   single selected template instance in the full image.  Test the existing
   one-row shape rather than assuming vLLM's ordinary-kernel optimum transfers
   to VDCores; two rows gives 128 one-wave tasks and is the first candidate.
4. Replace score top-6 with a one-warp register implementation.  Keep hash
   routing as a separate constexpr fast path that touches only its six selected
   experts.  Pack the tiny ids/weights output into one STU record when that
   reduces commands; raw address is an opt-in measurement variant only.
5. If routing still dominates, compare grouped dot-product and the existing
   BF16 UMMA/split-K primitive.  Do not introduce split-K reduction merely to
   match another framework's shape.

## Selected Milestone And Rejected Score Phasing

The selected projection shape is two experts per task over 128 resident SMs.
One LDU command jointly stages the 8-KiB hidden vector and the adjacent 16-KiB
two-row weight tile, eliminating the second operand wait without changing the
246-register, zero-spill full-image envelope.  The normalized MXFP8 branch runs
in parallel on eight disjoint SMs, while one final SM owns route selection.

A normal allocator-owned phased LDU prototype replaced the whole-projection
score barrier with consecutive producer counters while retaining one 2-KiB
destination lease and one compute operand.  It was exact but did not improve
the full image: eight 256-byte chunks measured 7.088 us, four 512-byte chunks
measured 6.192 us, and two 1024-byte chunks measured 5.664 us, equal to the
selected single-TMA path.  TMA issue/transaction granularity consumes all
available overlap.  The opcode, handler, Python instruction, barrier roles,
and runtime option were removed; retain the single whole-score barrier/load.

FlashInfer's sort-once/shift-winner `Sort<8>` selection was also tested as a
constexpr replacement for the prepared-score path.  It reduced the full image
from 246 to 240 registers with zero spills, but isolated top-6 rose to 2.768 us
and the 100-sample full boundary rose to 5.696 us.  The selected repeated local
scan remains 5.664 us in the same full image.  Keep the SM100 two-redux
argmax; lower register count alone is not the routing frontier.

Source review clarified that FlashInfer 0.6.16 itself sorts at most four local
candidates per lane and hierarchically merges per-warp lists.  The rejected
`Sort<8>` form was our expansion to the one-warp E256 mapping, not a literal
copy.  vLLM 0.27.1 instead uses one warp per token and six repeated reductions,
which remains the better shape match.  FlashInfer's public cluster/radix top-k
is intended for long vocabulary rows and is not appropriate for E256/K6.

The compact image now selects a dedicated prepared-score opcode.  It calls
only `task_dsv4_route_top6<true>` and therefore removes the legacy transform
branch/body from the selected dispatch image while retaining the legacy opcode
for diagnostic manifests.  This preserves the allocator-owned 2-KiB prepared
score load and the 64-byte RawAddress metadata output.  The build changed
`dae2` from 246 to 244 registers with the same 224-byte stack and zero spills;
`runtime.o` fell from 2,377,056 to 2,242,872 bytes and its CUDA fatbin from
2,359,776 to 2,225,592 bytes, both 134,184 bytes smaller.  Isolated prepared
top-6 remained 2.464 us.  The complete boundary measured 5.664 us in the short
profile run and 5.728 us over 100 samples, with maximum absolute error
2.4e-7.  Device timestamps place native MXFP8 readiness at 2.816 us, the
router-projection tail at 3.328 us, and top-6 completion at 5.632 us.  Opcode
specialization is an image-size win, not a latency win; the remaining critical
path is projection completion followed by about 2.3 us of score handoff and
selection.

A follow-up attempt reduced the prepared-score record from 2 KiB to 1 KiB by
storing only the biased selection score and reloading the original score after
selection.  The dependent global reads made both boundaries worse: isolated
top-6 rose to 2.944 us and the full router-to-FFN-ready boundary rose to
6.032 us.  Keep the paired ``float2 {original, selection}`` record and do not
repeat that variant.  Temporary per-subphase top-6 counters used to prepare the
next experiment were removed when this exploration was stopped.

## Correctness Gates

- Compare every active MXFP8 data byte and every active K32 UE8M0 scale against
  a host/FlashInfer-format reference, including zero, subnormal, saturation,
  NaN-policy, and random inputs.
- Verify the batch-1 N8 replication contract separately from the existing
  eight-useful-row FFN benchmark.
- Compare FP32 router logits, top-6 ids, tie breaking, normalized weights, and
  final FFN output against the framework/reference path.
- Run repeated launches in the generated full image to catch stale barriers,
  leaked allocator slots, and visibility errors.

## Performance Measurement And Acceptance

Use device-side timestamps in the full generated inference image for:

- normalized BF16 input visible
- first/last router task and router logits visible
- top-6 start/end and routing metadata visible
- first/last MXFP8 task and native activation visible
- first shared/routed Linear-1 dependent load issue

The primary interval is:

```text
BF16 input visible -> max(experts_ready, mx8_ready)
```

Report cold and hot device time, per-SM finish tails, register/spill state, and
the generated image text size.  Compare against matched vLLM BF16 gate plus
production DSV4 top-6 and FlashInfer exact-layout MXFP8 quantization.  Success
requires the VDCores interval in the full essential-operator image to be at
least 10% faster than the matched framework boundary while preserving the
normal VDCores contract and repeated-launch correctness.
