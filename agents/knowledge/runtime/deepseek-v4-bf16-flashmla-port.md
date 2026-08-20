# DeepSeek-V4 BF16 FlashMLA-Style Attention Port

Date: 2026-08-19

This is the implementation contract for replacing the current DeepSeek-V4
K32 split-attention producer with a FlashMLA-style SM100 dataflow while
retaining the normal VDCores allocator/LDU/STU runtime.

## Fixed contracts

- Q is BF16 `[64, 512]`.
- The KV cache remains BF16 `[rows, 512]`; there is no cache dequantization,
  cache-scale stream, raw-FP8 stage, or packed 584-byte cache record.
- The CTA remains the normal 256-thread VDCores block: one 128-thread compute
  warpgroup plus allocator, STU, LDU0, and LDU1.  BF16 operands require no
  dequantization or layout-transform warpgroup, so a second 128-thread compute
  group must not be added.  No specialized runtime branch is allowed.
- The producer emits a locally normalized BF16 partial plus FP32 `(max,mass)`
  metadata.  The reducer applies the attention sink, inverse RoPE, and native
  O-a FP8 quantization.
- Reduction remains CTA-parallel with one SM per head.  Spare SMs are favored
  over assigning multiple heads to one CTA.
- Small scalar/vector inputs use dependency-ordered raw addresses.  They must
  not consume allocator slots or M2C messages.
- At most one new memory operator may be added.  It must be a general
  allocator-leased internal-slot/barrier stream, analogous to the common MXFP
  coupled stream rather than an attention-only sequence of load opcodes.
- Python queues all producer and reducer commands before one persistent-kernel
  launch.  Producer STU completion directly releases reducer data barriers;
  there is no issue barrier or PDL launch dependency.

## Producer dataflow

1. Partition selected KV rows in B64 blocks.
2. Load Q once per partition into the QK-compatible shared layout.  Return the
   Q slots immediately after the final QK UMMA read.
3. Lease one 16-slot internal stage for the B64 block.  It contains two
   64-KiB shared images of the same BF16 source rows because QK K-major and PV
   MN-major operands have different physical SW128 layouts.
4. Under the same command and allocator lease, LDU0 issues the K-major TMA and
   LDU1 issues four adjacent D128 MN-major TMA copies for V.  There is no
   compute-side dequantization, repack, or additional warpgroup.
5. Arm each transaction barrier before issuing every TMA associated with that
   phase.  This is required for the four-copy V stream; arming after issue can
   let an early copy complete against an uninitialized transaction count.
6. Use one BF16 probability tile and one PV product.  Keep all 512 FP32 output
   dimensions in TMEM across the partition, with online max/mass correction.
7. Return a stage-empty token immediately after its final PV read.  Release
   the ring lease after the last stage is consumed.
8. Store one BF16 `[64,512]` partial and its FP32 metadata through STU.

The current TMEM map reuses columns `0..63` for the score/probability tile and
columns beginning at `128` for one D128 FP32 PV result at a time.  Q remains in
shared memory and therefore consumes no TMEM columns.

## General internal-ring memory operator

The single added memory opcode must accept a device plan record rather than
hard-code attention tensors.  Its contract is:

- allocator obtains one contiguous lease and publishes it once;
- the plan supplies descriptor ids, source coordinates or index vectors,
  byte/column ranges, stage count, iteration count, and internal full/empty
  barrier ids;
- both LDU ports may consume the same immutable command and split work by
  `port_id`;
- descriptor prefetch and cache policy are selected once at stream entry;
- stage allocation and phase state remain local for the complete command;
- compute observes fixed stage offsets and full barriers, then returns empty
  barriers; and
- the final consumer returns the whole lease to the allocator.

Dense MXFP and BF16 KV streams should be representable by the same execution
mechanism even when their plan encodings and TMA ranks differ.

## Reduction and handoff

Use 64 reducer CTAs, one per head.  Each reducer reads its own split metadata,
sink, inverse-RoPE row, and BF16 partial rows through raw pointers after its
producer dependency reaches zero.  It merges in FP32, applies inverse RoPE to
the final 64 dimensions, and emits four native FP8 O-a records.  Unused native
record bytes remain uninitialized.

## Resource budget

- one dual-layout KV stage: 16 slots;
- Q staging peak: 8 slots, exactly filling the normal 24-slot arena;
- after Q release: the 16-slot ring and eight-slot output reuse those same 24
  slots; probability storage is in TMEM, not an allocator slot;
- the ring is retired after the final PV read, before metadata allocation; and
- the kernel needs only the normal 256-thread runtime block and 512 TMEM
  columns.

## Acceptance gates

Correctness must cover SWA, C4A, and C128A at the production context-128
shape, repeated launches, tail masking, sink handling, inverse RoPE, and native
FP8 output layout.

Device-envelope performance is compared against the measured vLLM 0.27.1
FlashMLA attention plus equivalent inverse-RoPE/native-FP8 work:

| Shape | vLLM hot | Required VDCores hot (10% faster) |
|---|---:|---:|
| SWA | 13.795 us | <= 12.416 us |
| C4A / C128A | 14.205 us | <= 12.785 us |

The current VDCores reference is 14.816 us producer, 6.336 us reducer,
21.600 us combined hot, and 25.056 us cold.  Optimization proceeds from
isolated producer and reducer counters to the complete persistent schedule.

## Current implementation and measured result (2026-08-20)

The selected implementation uses two B64 producer CTAs, two reducer CTAs per
head, and the normal allocator/LDU/STU runtime.  The producer owns one 16-slot
dual-layout KV ring and a separate eight-slot Q load.  Reducers consume one
barriered raw pointer record, merge in FP32, apply inverse RoPE, and write the
native FP8 O-a records directly.  Producer stores are ordered 2,3,0,1 and
release the two D256 output groups independently.

At rows 128, repeated correctness passes with max-absolute error `0.00390625`,
mean-absolute error about `0.0000275`, and cosine about `0.99986`.  Tail cases
1, 63, 64, 65, 96, 127, 128, and 129 also pass repeated value-level checks;
UE8M0 boundary bytes may differ by one exponent while representing the same
accepted numerical result.

The compact attention image measures:

| Component | Hot device time |
|---|---:|
| producer | 9.664 us |
| reducer | 3.104 us |
| arithmetic sum | 12.768 us |
| fused schedule | 12.704 us |

The matched vLLM 0.27.1 C4A path is 12.360 us for complete BF16 FlashMLA,
1.7104 us for inverse-RoPE/native-FP8 packing, and 14.2032 us combined.  The
12.704-us compact VDCores result is therefore 10.6% faster.  The individual
producer and vLLM-attention values are not comparable because vLLM's first
stage already includes split reduction.

The 26-op inference image measures 9.952 us producer and 3.296 us reducer, an
arithmetic 13.248 us.  Relative to the compact image, the penalties are 0.288
and 0.192 us.  A hot-CFG dispatch hint does not add a specialized runtime path
or kernel text and recovers about 0.13 us; the residual is large-image code
layout/I-cache exposure.

Only one fence was removable: a redundant compute-group join after disjoint
raw metadata stores.  It saved 0.064 us in both the producer and fused
envelopes.  The tcgen05 proxy fences around QK, probability publication, PV,
and reusable TMEM addresses remain required by PTX ordering and repeated
correctness.  Device counters put all four reusable-TMEM joins at only 0.544
us, so fences are not the dominant cost.

Resident composition also needs phase continuity for the existing coupled
FP8 ring.  `SequentialProgram` now tracks its two-stage parity per physical SM
and rebases matching compute and LDU commands together.  This fixes the Q-a to
Q-b deadlock, and the resident prefix through all eight attention reducer/O-a
shards completes.  A later full-flow stall remains at shared W2: its identical
M4096/K2048 split-2 FP32 task passes alone at 5.472 us, so that remaining issue
belongs to full FFN allocator/ring-state composition rather than the attention
dependency chain.
