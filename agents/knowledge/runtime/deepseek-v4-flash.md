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
  packed NVFP4/per-16-E4M3 activation quantization.  Both schedules shard only
  at complete scale-block boundaries and can use all available SMs.
- Partial interleaved RoPE over the final 64 of 512 dimensions, including the
  inverse output rotation.
- Sparse 64-head, 512-dimensional attention over supplied window/compressed
  indices with the learned denominator-only attention sink.
- Ratio-4/ratio-128 gated compressed-KV pooling, normalized Hadamard rotation,
  learned 64x128 index scoring, exact streaming top-512 selection, and decode
  index helpers.  The selector retains 512 candidates while merging 512-row
  chunks, so it remains exact beyond the initial 1024 rows without a large
  thread-local array.
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
  SwiGLU, and all mHC stages.  After parallel quantization/index selection, the
  selective image uses 56 registers, nine barriers, a 112-byte stack frame, no
  spills, and 14,720 bytes static shared memory.
- FP8 LM-head shape M129280 K4096, 152 SMs: max absolute error 0.003906 and
  656.800 us for the one-iteration functional check.  M4096 K4096 remained at
  15.840 us median after pointer-offset sharding.

## Matched task comparison

`benchmarks/deepseek_v4_triton_tasks.py` supplies shape- and math-matched
Triton references for the task classes where Triton is the applicable
baseline.  It uses CUDA-graph replay to remove Python launch overhead and
performs bit-exact checks for both activation quantizers.  The VDCores sweep
uses repeated internal profile spans.  On the same single GB200, representative
medians in microseconds were:

| Task | Shape | VDCores | Triton |
| --- | --- | ---: | ---: |
| FP8 activation quantization | K4096 | 2.656 | 2.032 |
| NVFP4 activation quantization | K4096 | 3.680 | 1.984 |
| Sparse attention | H64, D512, K512 | 197.856 | 101.779 |
| Index score | rows640, H64, D128 | 19.008 | 2.646 |
| Top-512 | rows640 | 39.648 | 20.912 |
| RMSNorm | K512 | 2.144 | 1.338 |
| RMSNorm | K1024 | 2.112 | 1.626 |
| Bounded SwiGLU | K2048 | 2.176 | 1.517 |

At 4096 index rows, exact VDCores index score/top-512 measured 63.840/280.896
us versus Triton/PyTorch 7.152/33.187 us.  These remaining attention and
indexing gaps are explicit end-to-end profile targets, not parity claims.
The structural pass reduced K4096 NVFP4 quantization from 361.216 to 3.680 us
and rows640 top-512 from 47.500 ms to 39.648 us.

FA4 is not a semantic reference for the supplied-index, shared-D512 sparse
attention task: its standard dense/paged decode interface does not implement
the model's selected-index plus denominator-only-sink contract.  FlashInfer is
the applicable external reference for native NVFP4 GEMV and paged attention;
run those comparisons only from a worker that has its matching CUDA-13
environment and a free GPU.  Do not silently substitute an unavailable
package or disturb an occupied device.

These timings remain exploration data.  Assemble and profile the whole model
before doing finer task tuning.

All GPU checks must run through the cluster MPI launcher with one rank and the
target checkout on `PYTHONPATH`.
