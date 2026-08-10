# DeepSeek-V4-Flash runtime contract

## Scope

The target is single-GPU decode of `DeepSeek-V4-Flash-NVFP4`: routed expert
weights use ModelOpt NVFP4, while the remaining quantized linear weights use
E4M3 plus UE8M0 block-128 scales.  Functional model coverage comes before
kernel tuning; performance work should be driven by an end-to-end profile.

## Model shape

- 43 transformer layers, hidden size 4096, 4 hyper-connections.
- 256 routed experts plus one shared expert; top-6 routed experts are active.
- Routed expert intermediate size 2048.
- 64 query heads, one KV head, head dimension 512.
- Alternating compressed-attention ratios 4 and 128, with a 128-token sliding
  window, plus three hash layers.

## Quantized tensor contracts

- NVFP4 weight: packed `uint8 [M,K/2]`, E4M3 scale `[M,K/16]`, and scalar
  FP32 `weight_scale_2`.  Activations use the same packed/per-16 form and a
  scalar input scale.  The GEMV output multiplier is the product of the two
  scalar scales.
- FP8 weight/activation: E4M3 values.  Weight scales are UE8M0 per 128x128
  tile; activation scales are UE8M0 per contiguous K128 block.
- Decode outputs are BF16 with FP32 accumulation in the current tasks.

## Implemented task foundation

- A raw-checkpoint CUDA-core NVFP4 GEMV used as the current fast functional
  path and correctness oracle.
- A native SM100 block-scaled UMMA NVFP4 path with verified operand swizzles,
  E4M3 scale placement, UTCCP-to-TMEM transfer, and all eight output columns.
- A raw-checkpoint E4M3/UE8M0 block-128 FP8 GEMV.
- Python checkpoint-contract quantize/dequantize helpers and standalone
  correctness/latency benchmarks.

The native UMMA path is intentionally not the production performance claim
yet: it synchronously reformats every K256 slice and only has one natural SM
per M128 tile.  Defer its prepacking/TMA/split-K redesign until the broad
model path can be profiled.

## Broad functional task coverage

The correctness-first single-token path now covers every DeepSeek-specific
operation needed to assemble decode without substituting another framework's
model math:

- BF16-to-E4M3/UE8M0 block-128 activation quantization and BF16-to-ModelOpt
  packed NVFP4/per-16-E4M3 activation quantization.
- Partial interleaved RoPE over the final 64 of 512 dimensions, including the
  inverse output rotation.
- Sparse 64-head, 512-dimensional attention over supplied window/compressed
  indices with the learned denominator-only attention sink.
- Ratio-4/ratio-128 gated compressed-KV pooling, normalized Hadamard rotation,
  learned 64x128 index scoring, top-512 selection, and decode index helpers.
- Sqrt-softplus top-6 routing, hash routing, bounded 2048-wide SwiGLU, routed
  plus shared expert reduction, and the existing quantized projection tasks.
- 512/1024 RMSNorm, FP32-weight/BF16-input small projection, mHC pre/Sinkhorn,
  mHC post, and final/MTP mHC-head reduction.

The FP8 and FP32 schedules address each output shard through pointer offsets,
so their instruction fields no longer cap matrices at 65,535 rows.  The FP8
path was exercised at the actual 129,280-by-4,096 LM-head shape.

## Verified GB200 baselines (2026-08-10)

- NVFP4 CUDA, M2048 K4096, 128 SMs: bit-exact BF16 reference; 8.256 us median
  task time in a five-iteration smoke.
- NVFP4 native UMMA, M2048 K4096, 16 SMs: all eight columns bit-exact; 79.712
  us median task time.  This is a correctness baseline, not a tuned result.
- FP8 block-128, M4096 K4096, 152 SMs: max absolute error 1.5e-5 versus the
  quantized reference; 16.320 us median kernel span.
- Broad functional sweep, one GB200: all 21 checks passed, including exact
  activation-quantized bit patterns, 64x512 sparse attention with top-k 512,
  both compression ratios, learned indexing, MoE routing/reduction, bounded
  SwiGLU, and all mHC stages.  The selective image uses 56 registers, nine
  barriers, a 4,160-byte stack frame, no spills, and 14,720 bytes static shared
  memory.
- FP8 LM-head shape M129280 K4096, 152 SMs: max absolute error 0.003906 and
  656.800 us for the one-iteration functional check.  M4096 K4096 remained at
  15.840 us median after pointer-offset sharding.

These functional timings are exploration data, not parity claims.  In
particular, the serial correctness top-512 selector measured 46.703 ms and the
NVFP4 activation quantizer measured 364.256 us at K4096.  Keep them visible in
the first end-to-end profile, but do not tune them in isolation before the
model flow is assembled.

All GPU checks must run through the cluster MPI launcher with one rank and the
target checkout on `PYTHONPATH`.
