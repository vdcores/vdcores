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
- A BF16-weight/input/output GEMV with FP32 accumulation for checkpoint
  linears that are intentionally unquantized: routing, compression, index
  weighting, embeddings, and the vocabulary head.
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
- Partial interleaved RoPE over the final 64 dimensions of both the 512-wide
  attention heads and 128-wide indexer heads, including the inverse attention
  output rotation.
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
path was stress-tested at 129,280-by-4,096; the actual checkpoint head is
unquantized BF16 rather than FP8.

## Verified GB200 baselines (2026-08-10)

- NVFP4 CUDA, M2048 K4096, 128 SMs: bit-exact BF16 reference; 8.256 us median
  task time in a five-iteration smoke.
- NVFP4 native UMMA, M2048 K4096, 16 SMs: all eight columns bit-exact; 79.712
  us median task time.  This is a correctness baseline, not a tuned result.
- FP8 block-128, M4096 K4096, 152 SMs: max absolute error 1.5e-5 versus the
  quantized reference; 16.320 us median kernel span.
- BF16 checkpoint GEMV, M256 K4096, 152 SMs: bit-exact BF16 reference and
  9.920 us in a one-iteration router-shape smoke.
- Broad functional sweep, one GB200: all 21 checks passed, including exact
  activation-quantized bit patterns, 64x512 sparse attention with top-k 512,
  both compression ratios, learned indexing, MoE routing/reduction, bounded
  SwiGLU, and all mHC stages.  After parallel quantization/index selection, the
  selective image uses 56 registers, nine barriers, a 112-byte stack frame, no
  spills, and 14,720 bytes static shared memory.
- FP8 LM-head shape M129280 K4096, 152 SMs: max absolute error 0.003906 and
  656.800 us for the one-iteration projection-shape stress check.  M4096 K4096
  remained at 15.840 us median after pointer-offset sharding.

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

## Synthetic single-GPU decode flow

`python/dae/deepseek_v4_flow.py` describes the official per-token layer order
and cache/index cardinalities.  `benchmarks/deepseek_v4_synthetic_decode.py`
connects that plan to VDCores schedules for all attention families, both
compressors, CSA indexing, mHC residual paths, NVFP4 routed experts, the FP8
shared expert and other quantized projections, and the final head.  It keeps
checkpoint-sized tensor dimensions while reusing one deterministic tensor per
weight shape; this makes it a topology/dataflow test rather than a checkpoint,
quality, memory-footprint, or TBT claim.

On one GB200, the following breadth checks completed with finite residuals and
logits:

- 43 layers at positions 0 and 3: window-only startup followed by the first
  ratio-4 compressor/index-cache boundary.
- 43 layers at position 127: 2 SWA, 21 CSA, 20 HCA, and the full 129,280-token
  vocabulary head.
- 43 layers at position 4095: the same layer mix with 1,024 compressed CSA
  rows, exact top-512 selection, and 640 attention candidates.

The synthetic graph deliberately remains untuned and currently executes many
individual task launches.  The next functional milestone is loading real
checkpoint tensors into this flow; only then should its profile drive task
fusion, launch reduction, and TBT work.

## Real-checkpoint preflight

`python/dae/deepseek_v4_checkpoint.py` generates and validates the exact raw
non-MTP checkpoint contract, parses safetensors headers without materializing
payloads, and lazily loads only explicitly requested tensors while preserving
their FP8/NVFP4 layouts.  `benchmarks/deepseek_v4_checkpoint_audit.py` can audit
either a local download or HTTPS range-read headers from the official NVIDIA
repository.

At NVIDIA revision `7fc18be2b215ae48260383d4a228ec8a033046f7`, all 46 remote
shard headers passed: 135,235 total tensors, including 133,660 base-inference
tensors and 1,575 MTP tensors.  Total tensor payload is 168,266,793,544 bytes;
the base model without MTP is 164,673,005,788 bytes (153.36 GiB).  A 189,471
MiB GB200 therefore has about 31.67 GiB left before caches, activations, task
images, and allocator overhead.  This is a viable but tight single-GPU fit.

The pinned checkpoint is downloaded on worker `10.0.16.24` at
`/mnt/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4`; its Hugging Face cache is
also worker-local at `/mnt/checkpoints/huggingface-cache`.  The model directory
occupies 157 GiB on local EXT4 storage.  Its local header audit passed the same
135,235-tensor, 46-shard contract as the remote audit.  Checkpoint-backed jobs
must be pinned to this host because `/mnt` is worker-local; checkpoints must
never be copied into the NFS source tree or a worker home directory.

`DeepSeekV4Checkpoint.load_fp8_linear()` and `load_nvfp4_linear()` bind a
named checkpoint prefix to the raw schedule-ready tensors without
dequantizing or rewriting it.  The real-checkpoint task smoke on one GB200
passed for `layers.2.attn.wq_a` (E4M3/UE8M0, exact against the quantized
reference) and `layers.2.ffn.experts.0.w1` (packed NVFP4, 0.048096 maximum
absolute BF16 error).  This verifies checkpoint-to-VDCores dtype, layout, and
scalar-scale routing for both quantization families; it is not yet a complete
real-weight transformer layer.

The CUDA-13/Blackwell vLLM 0.23.0 environment on that worker completed a real
TP=1, one-GPU, two-token inference at context 128.  vLLM selected FP4 experts
through FlashInfer TRT-LLM, FP8 DeepGEMM linears, FP8 MLA KV cache, and FP8
Lightning Indexer cache.  It loaded 153.97 GiB of model state, retained 25.57
GiB for KV cache (63,071 tokens), and exited cleanly.  The environment's PyPI
CUDA compiler had drifted to 13.3 while its runtime headers remained 13.0;
pinning both `CUDA_HOME` and `CUDA_PATH` to `/usr/local/cuda` selected the
coherent system CUDA 13.0 toolchain and fixed TileLang mHC compilation without
changing installed packages.

The first cold start spent 1,177 seconds profiling, compiling cached SM100
operators, warming DeepGEMM, and running FlashInfer's built-in 21-profile MoE
selection.  The one-sample eager harness reported a 0.736501 ms first-to-second
token interval.  Treat this only as an E2E functional smoke result: it had no
warmup or statistical sample set and is not a framework TBT baseline or parity
claim.

All GPU checks must run through the cluster MPI launcher with one rank and the
target checkout on `PYTHONPATH`.  Runs using this checkpoint must also select
worker `10.0.16.24` and pass the explicit `/mnt` checkpoint path.
