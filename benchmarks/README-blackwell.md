# Blackwell task benchmarks

These results were collected on one 152-SM NVIDIA GB200 (SM100, CUDA 13.0)
with BF16 Llama-3.1-8B decode shapes. Times are GPU-side kernel durations with
warm data. Correctness is checked against a PyTorch FP32-softmax reference and
the retained kernels stay below 1% mean-relative error.

## Decode attention

The native SM100 winner swaps the two UMMA operands following CUTLASS' low-
latency TGV GQA formulation: QK is `K[128,128] * Q[8,128]` and PV is
`V[128,128] * P[8,128]`. Q therefore occupies only an eight-row, 2 KiB TMA
tile instead of a padded 64-row tile. A 32-DP TMEM load assigns one sequence or
output row to each compute thread, so all four compute warps perform the four
live GQA softmax rows and online output correction in parallel. Split-KV emits
unnormalized partial output plus local `(max, sum)` metadata. The reducer reads
the final-output pointer from the same barriered metadata record, applies the
global softmax correction, and stores directly to the output without a second
raw-address instruction or output TMA stage.

The selector in `python/dae/attention_config.py` chooses from the measured
KV64/KV128 and split-count variants. FlashInfer 0.6.15 is the best result among
its available generic-wrapper variants. The vLLM and SGLang columns instead
exercise the exact decode calls selected by those framework versions, which is
why they can differ materially from the generic FlashInfer wrapper result.

| Batch | Sequence | KV tile | Splits | SMs | VDCores (us) | FlashInfer 0.6.15 (us) | vLLM 0.23.0 (us) | SGLang 0.5.12 (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | 128 | 1 | 8 | **3.072** | 4.397 | 4.510 | 5.681 |
| 1 | 128 | 128 | 1 | 8 | **3.264** | 4.706 | 4.431 | 5.728 |
| 1 | 512 | 128 | 4 | 32 | **4.384** | 6.856 | 5.360 | 5.773 |
| 1 | 2048 | 128 | 16 | 128 | **4.768** | 8.187 | 5.782 | 6.833 |
| 2 | 128 | 128 | 1 | 16 | **3.264** | 5.013 | 4.398 | 5.770 |
| 2 | 512 | 128 | 4 | 64 | **4.512** | 7.470 | 5.360 | 5.978 |
| 4 | 128 | 128 | 1 | 32 | **3.264** | 5.014 | 4.500 | 5.158 |
| 4 | 512 | 128 | 4 | 128 | **4.704** | 8.085 | 5.365 | 5.155 |
| 8 | 128 | 128 | 1 | 64 | **3.328** | - | 4.679 | 5.556 |
| 8 | 512 | 128 | 2 | 128 | 6.400 | - | **5.579** | 5.656 |

Run `app/python/attention_simple_decoding.py` for the unsplit cases and
`app/python/attention_split_kv.py` for split-KV. The comparison harness is
`benchmarks/blackwell_flashinfer_decode.py`.

The retained code is the minimum winner from a broader search. A CUDA-core,
non-UMMA SDPA prototype was correct but measured 4.832 us at B1/S32 and 7.488
us at B1/S64, so it was removed. A FA4-style second barrier that launched the
next QK before the current softmax raised the task image from 96 to 128
registers and regressed B8/S256 from 4.640 to 4.928 us. Dedicated reducer SMs,
an atomic rendezvous, and normalized partial-output variants were also slower.
The retained two-split reducer assigns one warp to each live query row and
combines both BF16 partials with aligned 64-bit loads/stores. In a 500-epoch
A/B run at B8/S512 it reduced the median from 6.560 to 6.464 us while masked
S500 and S2048 cases retained less than 1% mean-relative error. Following the
CUTLASS TGV BMM2 correction path, the output accumulator now bypasses CUTE
partitioning and issues raw 32-DP `tcgen05.ld/st` for only the four live GQA
columns. This halves each thread's output registers and further lowers B8/S512
to 6.400 us.
The remaining long-context limitation is high-batch work per CTA: at S2048,
B1/B2/B4/B8 measure 4.768/6.560/9.216/13.776 us as each task owns
1/2/4/8 KV128 blocks.

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

For 4096-row projections, four disjoint M128 accumulators share every B tile.
Their epilogues occupy one 8 KiB shared slot and one strided rank-4 TMA
reduction. This measures 5.792 us for K4096 and 18.704 us for K14336, both
below the exact framework component probes.

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
| Decode attention | Swapped-A/B KV128 QK/PV in TMEM, Q8 TMA, four-warp softmax, raw split reduction | FlashInfer 0.6.12 TRTLLM batch decode, page 16, actual maximum sequence | FlashInfer 0.6.11.post1 TRTLLM batch decode, page 64, model maximum sequence 131072 |
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
| Q/O component, 4096 x 8 x 4096 | **5.792** | 5.863 | 5.922 | VDCores leads by 1.2-2.2% |
| Fused QKV, 6144 x 8 x 4096 | - | **6.358** | 6.364 | Framework-selected fused scope |
| Gate or up component, 14336 x 8 x 4096 | - | **18.876** | 19.106 | VDCores uses 6144/8192 pipeline partitions |
| Fused gate/up, 28672 x 8 x 4096 | - | 39.284 | **37.123** | Framework-selected fused scope |
| Down, 4096 x 8 x 14336 | **18.704** | 19.264 | 18.893 | VDCores leads by 1.0-2.9% |
| Padded LM head, 131072 x 8 x 4096 | **147.840** | 149.703 | 149.781 | VDCores leads by 1.2-1.3% |

The grouped down row is an isolated task result. Production keeps the existing
split/overlapped M64 schedule: integrating the fold-16 grouped result caused
32-layer BF16 reduction drift, while a lower-drift fold-8 phased variant was
slower at 21.888 us and changed a control-flow argmax. The retained schedule
still passes four-token greedy correctness and measures 2.986 ms TBT over 128
steps (job `20260805T094633Z-354598`).

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
| RMSNorm, 8 x 4096 | 2.272 | 2.681 | **2.069** | Two 64-thread rows per SM; VDCores leads vLLM by 15%, SGLang leads by 9.8% |
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
standalone probes trail. B8/S128 attention now leads vLLM by 28% and SGLang by
40%, and grouped Q/O, down, and LM-head projections lead both frameworks. The
remaining attention gap is B8/S512, where VDCores is 15% behind vLLM after
reducing the prior 61% deficit; SGLang's standalone RMSNorm also remains faster
than the VDCores stage.

Reproduce the exact framework probes with
`benchmarks/blackwell_framework_tasks.py`, the isolated VDCores non-GEMV tasks
with `benchmarks/blackwell_vdcores_tasks.py`, and projection/LM-head epochs with
`benchmarks/blackwell_gemv.py`. Every retained correctness result is below 1%
mean-relative error.

The RMS selector uses one 64-thread row at B1/B2/B4 and two concurrent rows at
B8. Its selected B1/B2/B4/B8 medians are 2.144/2.144/2.208/2.272 us, versus
vLLM's 2.463/2.689/2.680/2.681 us and SGLang's
1.858/2.071/2.073/2.069 us. Each row keeps aligned BF16 input packs in
registers, preloads its weight packs while the cross-warp reduction completes,
and uses a row-local named barrier distinct from the runtime queue barrier.
Rejected variants include 32 and 128 threads per row, input reload from shared
memory, direct-global output, port-1 input TMA, and a prefetched global-weight
path. Use `--rms-rows-per-sm 1` or `2` to reproduce either topology; the default
`0` selects the measured B1-B8 optimum.

## Llama-3.1-8B single-token schedule

The retained decode path processes batch 8 with one new token per request and
one VDCores megakernel launch per decode step. Persistent multi-token fusion is
deliberately left for a later milestone. A bulk C++ launch API submits a
sequence of independent one-token kernels without repeating Python-side
validation, packing, or cache-policy setup.

The 152-SM schedule uses four measured choices:

- all projection weights are packed as contiguous M64K256 UMMA/TMA tiles;
- QKV uses four active GQA rows per KV head and KV128 decode tiles;
- decode attention uses the swapped-A/B Q8 path and writes its BF16 epilogue
  directly, avoiding both the padded Q64 load and output TMA staging pass;
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
| Four-output grouped LM head | 382.13 | 2.985 | no |
| Swapped Q8 attention + 128-register image | **377.53** | **2.949** | yes |

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
| VDCores | 377.529 ms / 128 steps | **2.949** | - |
| vLLM | 0.23.0, 429.979 ms / 128 outputs | 3.359 | 12.2% |
| vLLM | cross-run decode estimate | 3.335 | 11.6% |
| SGLang | 0.5.12.post1, reported decode median | 3.820 | 22.8% |

Reproduce the retained path with:

```bash
python app/python/llama3/sched.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  -N 128 --control-flow --bench 9
```

Tensor-level validation passes for a non-control-flow step, exact greedy tokens
match Hugging Face, and a 130-step launch crosses from one KV128 block to two
with the unified online-softmax path. The exact 12-operator Llama image uses
128 registers, 10 barriers, a 96-byte stack frame, and zero spills. Removing
the prior padded attention opcode from the selective image lowered the
megakernel-wide register allocation from roughly 202 registers.
`tests/blackwell_runtime_smoke.py` also covers synchronous, asynchronous, and
bulk sequence launches on all 152 SMs.

## Implementation references

The retained implementation follows the SM100 programming model and UMMA/TMEM
layouts described by the [CUDA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/13.0.1/blackwell-tuning-guide/index.html),
the [CUDA PTX ISA tcgen05/TMEM synchronization model](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html),
and [CUTLASS Blackwell GEMM guidance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html),
including its TMEM-to-register epilogues and Blackwell-specific pipelines.
Kernel organization and comparison regimes were cross-checked against the
[CUTLASS SM100 low-latency GQA example](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa),
[Triton Blackwell persistent-matmul example](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py),
[FlashInfer decode APIs and SM100 CuTe path](https://github.com/flashinfer-ai/flashinfer),
[vLLM's CUDA-graph model runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py),
[SGLang's decode runner and attention backends](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py),
and the [FlashAttention-4 CuTeDSL implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd.py).
