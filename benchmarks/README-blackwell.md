# Blackwell task benchmarks

These results were collected on one 152-SM NVIDIA GB200 (SM100, CUDA 13.0)
with BF16 Llama-3.1-8B decode shapes. Times are GPU-side kernel durations with
warm data. Correctness is checked against a PyTorch FP32-softmax reference and
the retained kernels stay below 1% mean-relative error.

## Decode attention

The native SM100 path keeps QK and PV accumulators in TMEM. One warp drains the
four live GQA score rows into a compact FP32 shared stage, all four compute
warps perform row-parallel online softmax, and the same special-slot storage is
overwritten with swizzled BF16 probabilities for shared/shared UMMA PV. The
path supports one or multiple KV tiles, rescales the prior TMEM output online,
and finishes with aligned BF16x4 TMEM-to-global stores. Split-KV publishes
normalized partial output plus log2 LSE; its reducer overlaps warp-distributed
LSE preprocessing with the TMA load of partial output.

The selector in `python/dae/attention_config.py` chooses from the measured
KV64/KV128 and split-count variants. FlashInfer 0.6.15 is the best result among
its available generic-wrapper variants. The vLLM and SGLang columns instead
exercise the exact decode calls selected by those framework versions, which is
why they can differ materially from the generic FlashInfer wrapper result.

| Batch | Sequence | KV tile | Splits | SMs | VDCores (us) | FlashInfer 0.6.15 (us) | vLLM 0.23.0 (us) | SGLang 0.5.12 (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 64 | 1 | 8 | **3.424** | 4.397 | 4.510 | 5.681 |
| 1 | 128 | 128 | 1 | 8 | **3.904** | 4.706 | 4.431 | 5.728 |
| 1 | 512 | 64 | 8 | 64 | 6.112 | 6.856 | **5.360** | 5.773 |
| 1 | 2048 | 128 | 16 | 128 | 7.040 | 8.187 | **5.782** | 6.833 |
| 2 | 128 | 128 | 1 | 16 | **3.936** | 5.013 | 4.398 | 5.770 |
| 2 | 512 | 64 | 8 | 128 | 6.208 | 7.470 | **5.360** | 5.978 |
| 4 | 128 | 128 | 1 | 32 | **3.936** | 5.014 | 4.500 | 5.158 |
| 4 | 512 | 128 | 4 | 128 | 6.688 | 8.085 | 5.365 | **5.155** |
| 8 | 128 | 128 | 1 | 64 | **4.032** | - | 4.679 | 5.556 |
| 8 | 512 | 128 | 2 | 128 | 8.960 | - | **5.579** | 5.656 |

Run `app/python/attention_simple_decoding.py` for the unsplit cases and
`app/python/attention_split_kv.py` for split-KV. The comparison harness is
`benchmarks/blackwell_flashinfer_decode.py`.

The retained code is the minimum winner from a broader search. A CUDA-core,
non-UMMA SDPA prototype was correct but measured 4.832 us at B1/S32 and 7.488
us at B1/S64, so it was removed. A fused cross-CTA atomic rendezvous measured
6.208 us at B1/S512 versus 6.080-6.112 us for the retained TMA reducer, and a
direct-global reducer epilogue did not improve the median; both were also
removed. Long-context split reduction remains the attention bottleneck.

## GEMV

SM100 GEMV uses native BF16 UMMA with F32 accumulation in TMEM and a
TMEM-to-register-to-smem output path. The LM-head specialization reuses each
eight-token B tile across four M128 output tiles, keeps four accumulators in
separate TMEM column ranges, and drains BF16 directly to global memory.

| Shape (M x N x K) | SMs | M64 (us) | M128 (us) | M128 gain |
| --- | ---: | ---: | ---: | ---: |
| 1024 x 8 x 4096 | 64 | 4.288 | 4.000 | 6.7% |
| 4096 x 8 x 4096 | 128 | 7.200 | 6.560 | 8.9% |
| 8192 x 8 x 4096 | 128 | 12.096 | 11.200 | 7.4% |
| 4096 x 8 x 14336 | 128 | 22.464 | 20.960 | 6.7% |

The exact two-epoch padded LM head (131072 x 8 x 4096 total) measures 147.840
us with the grouped direct path and exact BF16 agreement with the isolated
reference. The same framework comparison measures 149.703 us in vLLM and
149.781 us in SGLang.

Use `benchmarks/blackwell_gemv.py` to reproduce a shape and
`tests/blackwell_gemv_smoke.py` for strict single-tile correctness.

## Per-operator comparison with vLLM and SGLang

Llama-3.1-8B is dense, so "per expert" here means per task/operator rather
than per MoE expert. These BF16 measurements target batch 8 with one new token
per request. VDCores reports the span from the first participating SM start to
the final participating SM completion after five warmups; the framework
harness reports the median per operation inside a CUDA graph (7 samples, 20
replays, 10 operations per replay). All values use warm data on a 152-SM GB200.

The installed framework source selects the following exact paths:

| Task | VDCores | vLLM 0.23.0 | SGLang 0.5.12.post1 |
| --- | --- | --- | --- |
| Token embedding | CC0 row selection feeds the first RMS task directly | Unquantized `F.embedding` | Unquantized `F.embedding` |
| BF16 projections | Native M64/M128 UMMA, tiled weights, F32 TMEM accumulation | Unquantized `F.linear`; fused QKV and fused gate/up | Unquantized `F.linear`; fused QKV and fused gate/up |
| Decode attention | KV64/KV128 QK in TMEM, four-warp shared-P softmax, SS-UMMA PV | FlashInfer 0.6.12 TRTLLM batch decode, page 16, actual maximum sequence | FlashInfer 0.6.11.post1 TRTLLM batch decode, page 64, model maximum sequence 131072 |
| RMSNorm | `SchedRMSShared` | `vllm._custom_ops.rms_norm` | `sgl_kernel.rmsnorm` |
| SwiGLU | Shared-memory prefix plus register-forwarded tail | `torch.ops._C.silu_and_mul` | `sgl_kernel.silu_and_mul` |
| RoPE | Q/K consume projection registers; K writes its cache row | `vllm._custom_ops.rotary_embedding` | SGLang JIT in-place RoPE |
| KV append | V projection and K RoPE stores | `reshape_and_cache_flash` | SGLang JIT `store_cache` |
| Greedy select | 128-SM two-slice reduction | `torch.argmax` | `torch.argmax` |

Shape-matched projection probes are useful even where the full frameworks fuse
components. A dash means that the VDCores production stage is deliberately
split, fused, or overlapped and therefore has no equivalent standalone launch.

| Projection/task scope, BF16 B8 | VDCores (us) | vLLM (us) | SGLang (us) | Result |
| --- | ---: | ---: | ---: | --- |
| KV component, 1024 x 8 x 4096 | **4.352** | 5.333 | 5.343 | VDCores is 18% faster |
| Q/O component, 4096 x 8 x 4096 | 7.216 | **5.863** | 5.922 | M64 VDCores is 23% behind vLLM |
| Fused QKV, 6144 x 8 x 4096 | - | **6.358** | 6.364 | Framework-selected fused scope |
| Gate or up component, 14336 x 8 x 4096 | - | **18.876** | 19.106 | VDCores uses 6144/8192 pipeline partitions |
| Fused gate/up, 28672 x 8 x 4096 | - | 39.284 | **37.123** | Framework-selected fused scope |
| Down, 4096 x 8 x 14336 | 22.768 | 19.264 | **18.893** | VDCores is 20% behind SGLang |
| Padded LM head, 131072 x 8 x 4096 | **147.840** | 149.703 | 149.781 | VDCores leads by 1.2-1.3% |

The production LM head is two 65,536-row epochs. Each of 128 SMs owns four
disjoint M128 tiles and reuses every B tile across them, avoiding fold
reduction, logits clearing, shared-memory output staging, and output TMA. The
two raw output descriptors use dedicated special slots 30 and 31 so memory-VM
lookahead cannot overwrite either pointer with the following argmax task. The
framework linear paths are the same PyTorch operation, so their small
differences are independent-process measurement variation rather than
different kernels.

| Non-projection task, BF16 B8 | VDCores (us) | vLLM (us) | SGLang (us) | Scope note |
| --- | ---: | ---: | ---: | --- |
| RMSNorm, 8 x 4096 | 2.496 | 2.490 | **2.100** | VDCores matches vLLM within 0.3%; SGLang leads by 19% |
| Fused add + RMSNorm, 8 x 4096 | - | 2.697 | **2.308** | VDCores folds residual add into the preceding projection reduction |
| Materialized SwiGLU prefix, 8 x 6144 | **2.560** | 2.682 | 2.919 | Three 2048-wide shards; VDCores leads by 5%/12% |
| Q+K RoPE | 2.304 (Q only) | 2.899 | **1.473** | VDCores Q-only probe is not scope-equivalent to joint Q+K |
| K+V cache append | fused | 2.485 | **1.079** | No standalone VDCores launch |
| Greedy argmax, 8 x 131072 | **7.360** | 11.521 | 11.749 | VDCores is 36-37% faster |

These isolated times must not be summed to predict TBT. VDCores is one
persistent megakernel: Q/K/V are separately placed, K/V stores are fused,
residual adds occur in TMA reductions, the 8,192-row MLP tail forwards through
registers, and 24 auxiliary SMs overlap low-K down projection with that tail.
That cross-task pipeline is why end-to-end VDCores can lead while several
standalone probes trail. B8/S128 attention now leads vLLM by 14% and SGLang by
27%, and the grouped LM head leads both frameworks. The largest remaining task
gaps are long-context B8/S512 attention (61% behind vLLM), and the Q/O and down
projections.

Reproduce the exact framework probes with
`benchmarks/blackwell_framework_tasks.py`, the isolated VDCores non-GEMV tasks
with `benchmarks/blackwell_vdcores_tasks.py`, and projection/LM-head epochs with
`benchmarks/blackwell_gemv.py`. Every retained correctness result is below 1%
mean-relative error.

## Llama-3.1-8B single-token schedule

The retained decode path processes batch 8 with one new token per request and
one VDCores megakernel launch per decode step. Persistent multi-token fusion is
deliberately left for a later milestone. A bulk C++ launch API submits a
sequence of independent one-token kernels without repeating Python-side
validation, packing, or cache-policy setup.

The 152-SM schedule uses four measured choices:

- all projection weights are packed as contiguous M64K256 UMMA/TMA tiles;
- QKV uses four active GQA rows per KV head and KV128 decode tiles;
- decode attention uses the four-warp shared-P/SS-UMMA path and writes its
  BF16x4 epilogue directly, avoiding the prior output TMA staging pass;
- the gate/up prefix is balanced over three waves across all 152 SMs, while
  the 8,192-row tail keeps gate, up, and SwiGLU values in registers;
- the materialized 6,144-wide SwiGLU prefix uses three 2,048-element shards
  per token across all 24 auxiliary SMs, with aligned 128-bit shared-memory
  loads and stores;
- the LM head assigns four M128 tiles to each of 128 SMs, reuses each input
  tile four times, and drains four TMEM accumulators directly to logits;
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
| 152-task output + one auxiliary down half-tile | 402.82 | 3.147 | no |
| One full down tile per auxiliary SM | 428.03 | 3.344 | no |
| Two down half-tiles on eight auxiliary SMs | 436.11 | 3.407 | no |
| 128-bit, three-way materialized SwiGLU | 393.86 | 3.077 | no |
| Shared-P attention + unified multi-tile path | 383.26 | 2.994 | no |
| Four-output grouped LM head | **382.13** | **2.985** | yes |

The framework comparison and current VDCores result used the matching 152-SM
GB200 on `10.0.16.25:0`. Both used the local
Llama-3.1-8B-Instruct BF16 checkpoint, batch 8, input length 1, output length
128, and each framework's default CUDA-graph path. VDCores reports the median
wall time around all 128 one-token launches. vLLM's CLI reports total request
latency, so both its raw total/output value and a stricter cross-run decode
estimate are shown. The latter subtracts the separately measured one-output
p50 (6.496 ms) and divides the remainder by 127; it is an estimate, not a
direct phase timer.

| System | Version / measure | Median TBT (ms) | VDCores reduction |
| --- | --- | ---: | ---: |
| VDCores | 382.133 ms / 128 steps | **2.985** | - |
| vLLM | 0.23.0, 429.979 ms / 128 outputs | 3.359 | 11.1% |
| vLLM | cross-run decode estimate | 3.335 | 10.5% |
| SGLang | 0.5.12.post1, reported decode median | 3.820 | 21.9% |

Reproduce the retained path with:

```bash
python app/python/llama3/sched.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  -N 128 --control-flow --bench 9
```

Tensor-level validation passes for a non-control-flow step, exact greedy tokens
match Hugging Face, and a 130-step launch crosses from one KV128 block to two
with the unified online-softmax path. The exact 12-operator Llama image uses
202 registers, 9 barriers, a 96-byte stack frame, and zero spills.
`tests/blackwell_runtime_smoke.py` also covers synchronous, asynchronous, and
bulk sequence launches on all 152 SMs.

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
