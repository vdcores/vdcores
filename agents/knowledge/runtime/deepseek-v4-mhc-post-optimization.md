# DeepSeek-V4 mHC-post optimization

## Goal

Optimize both the standalone mHC-post operation and its composed handoff into
the following mHC projection/pre-RMS boundary.  Each selected path must be at
least 10% faster than its matched vLLM/FlashInfer device-time baseline,
preserve the ordinary VDCores compute-task plus memory-task contract, and
remain correct on repeated launches.  Comparisons with the old VDCores path
are diagnostic only and cannot satisfy this gate.

The router boundary is deliberately BF16.  The normalized hidden activation
and the router weight stay BF16 and continue through `OP_DSV4_BF16_GEMV`.
Native MXFP8 output may be produced for other projections, but it must not
replace or silently quantize the BF16 router input.

## Current cost and dataflow

The standalone schedule shards width 4096 over 32 SMs.  Every shard currently
uses one branch load, four residual-row loads, separate four-float post and
sixteen-float combination-matrix loads, and four output stores.  This is 11
memory commands and 11 normal shared-memory leases per SM even though each
shard carries only about 2.3 KiB of useful data.  In particular, the 80-byte
coefficient record consumes two 8-KiB slots and two allocator/LDU round trips.

The mathematical boundary preserves the producer-native branch dtype:

```text
branch BF16 (attention) or FP32 (FFN accumulator) + residual BF16[4,4096]
    -> mHC post -> residual BF16[4,4096]
    -> FP32xBF16 metadata projection -> mHC pre/RMS
    -> normalized hidden BF16[4096] -> BF16 router
```

The FFN route reduction accumulates and writes FP32, so mHC post consumes that
FP32 buffer directly.  The attention branch remains BF16.  These are mutually
exclusive input modes of one common task, not two independent operators, and
neither boundary inserts a conversion task.

## Selected implementation

1. Pack the four post coefficients and 4x4 combination matrix into one
   contiguous, 64-byte-aligned 20-float record.  Encode its raw address, branch
   dtype, and power-of-two shard width directly in the compute instruction.
   This consumes no shared slot or memory command and is a narrow-metadata
   exception, not a bulk-data transport path.
2. Load the four strided residual rows with one row-major 2-D TMA operation and
   store the four output rows with one row-major 2-D TMA operation.  Branch
   data remains one ordinary LDU load.  The compute task returns every normal
   input lease and publishes the output lease through STU.
3. Keep one mHC-post opcode and one mathematical implementation.  A task flag
   selects the producer-native BF16 or FP32 branch load; coefficient handling,
   residual math, and output transport are shared.  Do not generate separate
   fused/non-fused math operators.
4. For the fused/composed path, first reuse that same implementation and
   schedule state.  Fuse post into a following projection only if profiling
   proves that removing the intermediate BF16 write/read beats the code-size
   and register cost.  Do not clone post math into each GEMM producer.
5. Preserve direct data dependencies.  The producing STU/memory-command
   barrier is the visibility edge for raw metadata; add no issue barrier,
   compute-side fence, or independent CUDA launch.

## Fusion decision order

Evaluate these in order and stop at the smallest implementation meeting the
10% target:

1. compact standalone post followed by the existing projection/pre-RMS tasks
   in one persistent command image;
2. allocator-free lease handoff from post output to the next common projection
   task;
3. one common post-plus-projection mode that emits the existing 24 projection
   partials and square sum, followed by the existing pre/RMS finalizer.

Do not fuse post into attention and FFN GEMM epilogues independently unless a
measured gain justifies duplicated large-handler code.  The preferred fused
boundary mirrors the model's common post-to-next-pre transition and therefore
serves both branches.

## Measurement and acceptance

Use the selected full inference operator image for controls and candidates.
Measure built-in device start/end counters; CUDA event time is secondary.

- Standalone BF16 post is checked against vLLM's matched TileLang mHC-post
  device time.  The current 2.6448-us vLLM reference sets a <=2.3803-us gate.
  FP32 is retained for the FFN producer-native handoff and reported separately
  when no matched framework primitive exists.
- Validate that the FFN FP32 route accumulator arrives at post directly, with
  no intermediate BF16 conversion.
- The fused path is the complete post -> 24-row projection -> pre/RMS boundary
  in one persistent launch.  The measured vLLM 0.27.1 TileLang reference has a
  6.5120-us hot median, which sets a <=5.8608-us VDCores gate.
- Validate against a float reference for each producer-native branch dtype,
  all four output rows, repeated launches, and the packed coefficient layout.
- Also retain same-image old/new controls to attribute wins, but do not use
  them as acceptance evidence.
- Verify the full checkpoint schedule and BF16 router output handoff.  Reject
  local wins that add an unrequested conversion or regress the full image.
- Record selected handler count, `dae2` registers, stack/spills, instruction
  count, cold device envelope, and hot min/median/max.
