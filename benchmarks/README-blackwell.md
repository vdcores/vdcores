# Blackwell task benchmarks

These results were collected on one 152-SM NVIDIA GB200 (SM100, CUDA 13.0)
with BF16 Llama-3.1-8B decode shapes. Times are GPU-side kernel durations with
warm data. Correctness is checked against a PyTorch FP32-softmax reference and
the retained kernels stay below 1% mean-relative error.

## Decode attention

The native SM100 path keeps the online-softmax accumulator in TMEM, drains only
the four live GQA rows through one warp, converts probabilities back to TMEM,
and feeds them directly to the tensor-memory/smem UMMA PV stage. Split-KV
publishes normalized partial output plus log2 LSE; its reducer overlaps LSE
preprocessing with the TMA load of partial output.

The selector in `python/dae/attention_config.py` chooses from the measured
KV64/KV128 and split-count variants. FlashInfer 0.6.15 is the best result among
its available `auto` tensor-core and non-tensor-core decode variants.

| Batch | Sequence | KV tile | Splits | SMs | VDCores (us) | FlashInfer (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 64 | 1 | 8 | 4.064 | 4.397 |
| 1 | 128 | 128 | 1 | 8 | 5.376 | 4.706 |
| 1 | 512 | 64 | 8 | 64 | 6.912 | 6.856 |
| 1 | 2048 | 128 | 16 | 128 | 8.672 | 8.187 |
| 2 | 128 | 128 | 1 | 16 | 5.280 | 5.013 |
| 2 | 512 | 64 | 8 | 128 | 7.136 | 7.470 |
| 4 | 128 | 128 | 1 | 32 | 5.280 | 5.014 |
| 4 | 512 | 128 | 4 | 128 | 7.904 | 8.085 |

Run `app/python/attention_simple_decoding.py` for the unsplit cases and
`app/python/attention_split_kv.py` for split-KV. The comparison harness is
`benchmarks/blackwell_flashinfer_decode.py`.

## GEMV

SM100 GEMV uses native BF16 UMMA with F32 accumulation in TMEM and a
TMEM-to-register-to-smem output path. M128 is the measured winner for regular
Llama projections and is retained alongside M64 for layouts that require the
smaller tile.

| Shape (M x N x K) | SMs | M64 (us) | M128 (us) | M128 gain |
| --- | ---: | ---: | ---: | ---: |
| 1024 x 8 x 4096 | 64 | 4.288 | 4.000 | 6.7% |
| 4096 x 8 x 4096 | 128 | 7.200 | 6.560 | 8.9% |
| 8192 x 8 x 4096 | 128 | 12.096 | 11.200 | 7.4% |
| 4096 x 8 x 14336 | 128 | 22.464 | 20.960 | 6.7% |

Use `benchmarks/blackwell_gemv.py` to reproduce a shape and
`tests/blackwell_gemv_smoke.py` for strict single-tile correctness.

## Implementation references

The retained implementation follows the SM100 programming model and UMMA/TMEM
layouts described by the [CUDA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/13.0.1/blackwell-tuning-guide/index.html),
the [CUDA PTX ISA](https://docs.nvidia.com/cuda/archive/13.0.2/parallel-thread-execution/contents.html),
and the [CUTLASS Blackwell CuTe tutorial](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/blackwell/01_mma_sm100.cu).
Kernel organization and comparison regimes were cross-checked against
[FlashInfer](https://github.com/flashinfer-ai/flashinfer),
[vLLM's Blackwell support tracking](https://github.com/vllm-project/vllm/issues/18153),
[SGLang attention backends](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md),
and [FlashAttention-4](https://github.com/Dao-AILab/flash-attention/issues/2456).
