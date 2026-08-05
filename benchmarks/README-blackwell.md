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

## Llama-3.1-8B single-token schedule

The retained decode path processes batch 8 with one new token per request and
one VDCores megakernel launch per decode step. Persistent multi-token fusion is
deliberately left for a later milestone. A bulk C++ launch API submits a
sequence of independent one-token kernels without repeating Python-side
validation, packing, or cache-policy setup.

The 152-SM schedule uses four measured choices:

- all projection weights are packed as contiguous M64K256 UMMA/TMA tiles;
- QKV uses four active GQA rows per KV head and KV128 decode tiles;
- the gate/up prefix is balanced over three waves across all 152 SMs, while
  the 8,192-row tail keeps gate, up, and SwiGLU values in registers;
- output projection creates exactly 152 tasks with mixed K-folding, and the
  24 auxiliary SMs overlap one low-K down-projection task apiece with the
  register-forwarded MLP tail. Only the 12 affected M tiles cross an explicit
  reduction barrier before their high-K contribution.

The selection runs below used one GB200 on `10.0.16.24`, 128 decode steps,
three warmup sequences, and 7-9 measured sequences. They show why the final
auxiliary-SM overlap is intentionally limited to one half-tile per SM.

| Schedule variant | Median sequence (ms) | TBT (ms) | Kept |
| --- | ---: | ---: | :---: |
| Earlier 128-main/24-aux placement | 470.86 | 3.679 | no |
| Three-wave MLP prefix, uniform output/down | 431.49 | 3.371 | no |
| 152-task output, uniform down | 429.78 | 3.358 | no |
| 152-task output + one auxiliary down half-tile | 402.82 | 3.147 | yes |
| One full down tile per auxiliary SM | 428.03 | 3.344 | no |
| Two down half-tiles on eight auxiliary SMs | 436.11 | 3.407 | no |

The final same-node comparison used `10.0.16.25:0`, the local
Llama-3.1-8B-Instruct BF16 checkpoint, batch 8, input length 1, output length
128, and each framework's default CUDA-graph path. VDCores reports the median
wall time around all 128 one-token launches. vLLM's CLI reports total request
latency, so both its raw total/output value and a stricter cross-run decode
estimate are shown. The latter subtracts the separately measured one-output
p50 (6.496 ms) and divides the remainder by 127; it is an estimate, not a
direct phase timer.

| System | Version / measure | Median TBT (ms) | VDCores reduction |
| --- | --- | ---: | ---: |
| VDCores | 401.380 ms / 128 steps | **3.136** | - |
| vLLM | 0.23.0, 429.979 ms / 128 outputs | 3.359 | 6.7% |
| vLLM | cross-run decode estimate | 3.335 | 6.0% |
| SGLang | 0.5.12.post1, reported decode median | 3.820 | 17.9% |

Reproduce the retained path with:

```bash
python app/python/llama3/sched.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  -N 128 --control-flow --bench 9
```

Tensor-level validation passes for a non-control-flow step, exact greedy tokens
match Hugging Face for 16 consecutive one-token launches, and an exact-token
test starting with 127 prefetched prompt tokens passes across the KV128 block
boundary. `tests/blackwell_runtime_smoke.py` also covers synchronous,
asynchronous, and bulk sequence launches on all 152 SMs.

## Implementation references

The retained implementation follows the SM100 programming model and UMMA/TMEM
layouts described by the [CUDA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/13.0.1/blackwell-tuning-guide/index.html),
the [CUDA PTX ISA tcgen05/TMEM synchronization model](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html),
and [CUTLASS Blackwell GEMM guidance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html),
including its TMEM-to-register epilogues and Blackwell-specific pipelines.
Kernel organization and comparison regimes were cross-checked against the
[Triton Blackwell persistent-matmul example](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py),
[FlashInfer decode APIs and SM100 CuTe path](https://github.com/flashinfer-ai/flashinfer),
[vLLM's CUDA-graph model runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py),
[SGLang's decode runner and attention backends](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py),
and the [FlashAttention-4 CuTeDSL implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py).
