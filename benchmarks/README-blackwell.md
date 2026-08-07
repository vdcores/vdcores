# Blackwell task benchmarks

These results were collected on one 152-SM NVIDIA GB200 (SM100, CUDA 13.0)
with BF16 Llama-3.1-8B decode shapes. Times are GPU-side kernel durations with
warm data. Correctness is checked against a PyTorch FP32-softmax reference and
the retained kernels stay below 1% mean-relative error.

## Decode attention

The native SM100 winner swaps the two UMMA operands following CUTLASS' low-
latency TGV GQA formulation: QK is `K[128,128] * Q[8,128]` and PV is
`V[128,128] * P[8,128]`. Q therefore occupies only an eight-row, 2 KiB TMA
tile instead of a padded 64-row tile. A raw 32-DP TMEM load assigns one sequence
row to each compute thread; scores are transposed through shared memory so each
compute warp owns one live GQA query and each lane reduces four tokens. Split-KV
emits
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
| 1 | 64 | 128 | 1 | 8 | **3.008** | 4.397 | 4.510 | 5.681 |
| 1 | 128 | 128 | 1 | 8 | **3.040** | 4.706 | 4.431 | 5.728 |
| 1 | 512 | 128 | 4 | 32 | **4.320** | 6.856 | 5.360 | 5.773 |
| 1 | 2048 | 128 | 16 | 128 | **4.736** | 8.187 | 5.782 | 6.833 |
| 2 | 128 | 128 | 1 | 16 | **3.232** | 5.013 | 4.398 | 5.770 |
| 2 | 512 | 128 | 4 | 64 | **4.400** | 7.470 | 5.360 | 5.978 |
| 4 | 128 | 128 | 1 | 32 | **3.232** | 5.014 | 4.500 | 5.158 |
| 4 | 512 | 128 | 4 | 128 | **4.560** | 8.085 | 5.365 | 5.155 |
| 8 | 128 | 128 | 1 | 64 | **3.264** | - | 4.679 | 5.556 |
| 8 | 512 | 128 | 2 | 128 | 6.144 | - | **5.579** | 5.656 |

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
to 6.400 us. The retained score path applies the same four-column raw load,
stages the scores in unused scratch space, and assigns one warp to each query.
Each lane reduces four tokens locally before a single warp reduction, removing
the two CTA-wide max/sum exchanges and lowering the task image from 96 to 64
registers. B8/S512 reaches 6.176 us. Splitting each four-row reducer over two
SMs was correct but slower at 6.208 us because it duplicated task and TMA cost.
Returning each K slot as soon as its QK scores reach shared memory lets the
memory warps prefetch during softmax/PV, lowers the image to 63 registers, and
brings B8/S512 to 6.144 us. Issuing QK(i+1) before PV(i) was slower at 6.336 us.
The remaining long-context limitation is high-batch work per CTA: at S2048,
B1/B2/B4/B8 measure 4.736/6.304/8.448/12.672 us as each task owns
1/2/4/8 KV128 blocks.

## GEMV

SM100 GEMV uses native BF16 UMMA with F32 accumulation in TMEM and a
TMEM-to-register-to-smem output path. The LM-head specialization reuses each
eight-token B tile across four M128 output tiles, keeps four accumulators in
separate TMEM column ranges, and can reduce the BF16 epilogue directly to
compact argmax records without materializing the padded logits tensor.

| Shape (M x N x K) | SMs | M64 (us) | M128 (us) | M128 gain |
| --- | ---: | ---: | ---: | ---: |
| 1024 x 8 x 4096 | 64 | 4.288 | 4.000 | 6.7% |
| 4096 x 8 x 4096 | 128 | 7.200 | 6.560 | 8.9% |
| 8192 x 8 x 4096 | 128 | 12.096 | 11.200 | 7.4% |
| 4096 x 8 x 14336 | 128 | 22.464 | 20.960 | 6.7% |

The exact two-epoch padded LM head (131072 x 8 x 4096 total) measures 147.840
us with the diagnostic grouped direct-output path and exact BF16 agreement
with the isolated reference. The same materialized projection comparison
measures 149.703 us in vLLM and 149.781 us in SGLang. Production retains the
same four-output TMEM accumulation but folds argmax into its epilogue.

For 4096-row projections, four disjoint M128 accumulators share every B tile.
Their epilogues occupy one 8 KiB shared slot and one strided rank-4 TMA
reduction. This measures 5.792 us for K4096 and 18.704 us for K14336, both
below the exact framework component probes.

Use `benchmarks/blackwell_gemv.py` to reproduce a shape and
`tests/blackwell_gemv_smoke.py` for strict single-tile correctness.

### Projection-to-RoPE shared-memory handoff

Two M64 projection/RoPE paths remain available.  The two-operator diagnostic
path ends GEMV with `RegStore`, keeps that slot in shared memory, and passes it
to the existing RoPE task with `RegLoad`; only RoPE performs the final TMA
reduction/store.  The selected fused path rotates the TMEM-to-shared GEMV
epilogue in the projection task before the same final store.  Neither path
materializes the projection intermediate in global memory or reloads it with
TMA.

The table below uses 5,001 iterations at the production batch-eight Q and K
shapes.  Q uses 128 SMs/fold two and K uses 64 SMs/fold four.  Both paths stay
below 0.35% mean-relative error.  The framework total is a component sum of
the separate Q projection, separate K projection, and joint Q+K RoPE probes;
it is intentionally not the frameworks' fused-QKV scope.

| Projection + RoPE scope | Two operators (us) | Fused epilogue (us) | vLLM component sum (us) | SGLang component sum (us) |
| --- | ---: | ---: | ---: | ---: |
| Q, 4096 x 8 x 4096, fold 2 | 7.616 | **7.328** | - | - |
| K, 1024 x 8 x 4096, fold 4 | 4.736 | **4.544** | - | - |
| Q + K + RoPE component sum | 12.352 | **11.872** | 14.095 | 12.738 |

The fused task loads one head-local cosine/sine pair per compute thread after
submitting the final UMMA group and keeps it live across the completion wait.
Replaying the same task with those loads after the wait measured 7.392/4.672
us for Q/K, so final-group prefetch contributes 0.192 us of the 0.480 us
per-layer fused advantage.  A 1,001-sample S128 same-image
two-op/fused/two-op sandwich measured 2.684960/2.643360/2.686016 ms; fusion
saves 42.128 us (1.57%) against the control mean after 32 layers.  The larger
end-to-end gain shows that deleting the task boundary also shortens the layer
critical path.

Llama therefore selects fusion by default while retaining the two-operator
path for diagnostics with `VDCORES_FUSED_QK_ROPE=0`.  The minimal production
operator manifest omits the unused standalone RoPE opcode; the stage-profile
manifest contains both paths for same-image comparisons.  Set
`GEMV_TWO_OP_ROPE=1` or `GEMV_FUSED_ROPE=1` in
`benchmarks/blackwell_gemv.py` to reproduce the isolated alternatives.
The exact 11-op production image uses 96 registers, nine barriers, a 96-byte
stack, and zero spills; its final 1,001-sample S128 median is 2.642304 ms.

### Per-head QKV-to-attention readiness

The production schedule no longer holds all eight KV heads behind one Q
barrier and one combined K/V barrier. Each head has independent Q and K/V
readiness, and attention is placed head-major on the same low eight-SM group
that produces that head's V slice. The high group produces the matching Q
fold and K slice concurrently. This changes neither projection task count nor
tensor coordinates; it lets attention for a ready head overlap the tail of
the other heads.

A 1,001-sample coarse/per-head/coarse S128 sandwich measured
2.635520/2.625248/2.636096 ms with the resident internal timer, a 10.560 us
(0.40%) gain against the control mean. Two-head readiness groups measured
2.639360 ms and were not retained. The exact 11-op image remains at 96
registers, nine hardware barriers, a 96-byte stack, and zero spills; its final
production median is 2.626368 ms. Full S128 tensor validation and exact
four-token resident reuse pass.

### Phased attention-to-output schedule

Output projection now observes separate shared frontiers for KV head 0, head
1, and heads 2--7. Its K512 activation repeat already matches one KV head, so
weights can issue independently while only the not-yet-ready activation
segment waits. Early-K contributors run on SMs 64--139 outside the attention
placement; late-K contributors use the complementary SMs. The schedule still
has exactly 152 output-reduction tasks and uses the unchanged M64 compute op.

The profiling-free coarse/phased/coarse S128 medians are
2.627072/2.620544/2.627168 ms, a 6.576 us (0.25%) gain. Full S128 tensor and
exact-token correctness pass, as does two-token resident barrier reuse. The
final exact-image median is 2.620128 ms; the image remains spill-free at 96
registers, nine hardware barriers, and a 96-byte stack.

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
| RMSNorm, 8 x 4096 | 2.080 | 2.681 | **2.069** | One 128-thread row per SM; VDCores leads vLLM by 22% and is within 0.6% of SGLang |
| Fused add + RMSNorm, 8 x 4096 | - | 2.697 | **2.308** | VDCores folds residual add into the preceding projection reduction |
| Materialized SwiGLU prefix, 8 x 6144 | **2.560** | 2.682 | 2.919 | Three 2048-wide shards; VDCores leads by 5%/12% |
| Q+K RoPE | 2.304 (Q only) | 2.899 | **1.473** | VDCores Q-only probe is not scope-equivalent to joint Q+K |
| K+V cache append | fused | 2.485 | **1.079** | No standalone VDCores launch |
| Greedy argmax, 8 x 131072 | **7.360** | 11.521 | 11.749 | VDCores is 36-37% faster |

These isolated times must not be summed to predict TBT. VDCores is one
persistent megakernel: Q/K/V are separately placed, K/V stores are fused,
residual adds occur in TMA reductions, the 8,192-row MLP tail forwards through
an on-SM slot into an overlapped up/SwiGLU epilogue, and 24 auxiliary SMs
overlap low-K down projection with that tail.
That cross-task pipeline is why end-to-end VDCores can lead while several
standalone probes trail. B8/S128 attention now leads vLLM by 28% and SGLang by
40%, and grouped Q/O, down, and LM-head projections lead both frameworks. The
remaining attention gap is B8/S512, where VDCores is 10% behind vLLM after
reducing the prior 61% deficit. Standalone RMSNorm is within 0.6% of SGLang at
B8, ahead at B2/B4, and still trails SGLang at B1.

Reproduce the exact framework probes with
`benchmarks/blackwell_framework_tasks.py`, the isolated VDCores non-GEMV tasks
with `benchmarks/blackwell_vdcores_tasks.py`, and projection/LM-head epochs with
`benchmarks/blackwell_gemv.py`. Every retained correctness result is below 1%
mean-relative error.

The RMS selector uses one 128-thread row per SM at B1/B2/B4/B8. Its repeated
selected medians are 1.920/1.952/1.984/2.080 us, versus vLLM's
2.463/2.689/2.680/2.681 us and SGLang's 1.858/2.071/2.073/2.069 us. VDCores
therefore leads vLLM by 22-27%, leads SGLang by 4-6% at B2/B4, is within 0.6%
at B8, and trails by 3.3% at B1. All four compute warps cache aligned 128-bit
BF16 input and weight packs. Four warp partials cross shared memory once; each
warp leader computes the final reduction and inverse RMS, then broadcasts it
within the warp. Two independent BF16 pair accumulators shorten the square-sum
dependency chain without increasing the 68-register minimal image. The paired
64-thread-row path remains available for two contiguous rows, but its roughly
2.18 us B8 result loses to placing one row on each SM.

These standalone RMS medians use the selective RMS+terminate image, matching
the per-kernel scope of the framework probes. The 128-register production
megakernel is qualified separately by the end-to-end result below.

The embedding stage also keeps its separate residual-copy operator. RMSNorm
on eight SMs and the 8 KiB copy on eight other SMs overlap at batch eight. A
dual-output RMS prototype reused the cached input but measured 2.688 us versus
2.464 us for the existing pair in a 500-iteration same-process A/B. Since both
outputs were correct and fusion regressed the stage by 9.1%, the prototype was
removed.

Rejected variants include 32-thread rows, separate row-local barriers,
per-thread final scalar work, input reload from shared memory, direct-global
output, port-1 input TMA, early global-weight prefetch, four square-sum
accumulators, and 32-byte shared-memory packs. Use `--rms-rows-per-sm 1` or `2`
to reproduce either topology; the default `0` selects one 128-thread row per
SM. The selected task still uses TMA/shared-memory load and store memory ops.

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
  the 8,192-row tail retains each gate tile on-SM and computes its sigmoid
  under the final up-projection UMMA group before direct SwiGLU writeback;
- the materialized 6,144-wide SwiGLU prefix uses three 2,048-element shards
  per token across all 24 auxiliary SMs, with aligned 128-bit shared-memory
  loads and stores;
- the LM head assigns four M128 tiles to each of 128 SMs, reuses each input
  tile four times, and reduces each TMEM epilogue to one compact maximum per
  task/token instead of writing and rereading 2 MiB of padded logits;
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
| Swapped Q8 attention + 128-register image | 377.53 | 2.949 | no |
| One-row-per-SM 128-thread RMSNorm | **377.31** | **2.948** | yes |

The system comparison intentionally follows each deployment model instead of
forcing the same launch stopwatch on both. VDCores reports the internal
cross-SM `globaltimer` span, from the first participating SM entering the
resident megakernel schedule to the last participating SM reaching
`TerminateC`. A new streamed batch does not pay another host launch on this
critical path. vLLM and SGLang retain their launch-inclusive CUDA-graph decode
times because they dispatch work for each decode step.

The fixed-context comparison uses the same local Llama-3.1-8B-Instruct BF16
checkpoint, batch 8, and `10.0.16.24:2`. For a requested context `C`, each
framework receives `C - 1` prompt tokens and produces exactly two tokens. The
interval from the first to the second output is therefore one decode step
whose attention sees exactly `C` KV tokens; prefill is excluded. Each row uses
a separate framework process and an engine configured only for that context
(`max_model_len=C+1` in vLLM and `context_length=C+16` in SGLang). This avoids
silently giving the S64/S128 rows the scheduler and graph capacity of the S512
engine. vLLM's batch-token budget is `8 * (C - 1)`; SGLang disables chunked
prefill and uses the same full-batch prefill budget. These settings are
required at long context: with either framework's smaller default prefill
budget, the first-to-second-token interval can include unfinished prefill from
other requests and is not a decode-only result. The S64-S512 framework rows
are medians of 30 samples after five warmups. The S1K-S32K framework rows are
medians of three samples after one warmup because every isolated sample first
constructs all eight full contexts. Each sample takes the maximum request
interval in the strict eight-request batch. There is no HTTP transport.

| Fixed context | VDCores internal (ms) | vLLM 0.23.0 launch-inclusive (ms) | SGLang 0.5.12.post1 launch-inclusive (ms) | Fastest | VDCores vs vLLM | VDCores vs SGLang |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 64 | **2.812** | 2.842 | 3.381 | VDCores | 1.1% faster | 16.8% faster |
| 128 | **2.811** | 2.816 | 3.312 | VDCores | 0.2% faster | 15.1% faster |
| 256 | **2.802** | 3.448 | 3.410 | VDCores | 18.7% faster | 17.8% faster |
| 512 | **2.851** | 3.499 | 3.683 | VDCores | 18.5% faster | 22.6% faster |
| 1,024 | **3.006** | 3.584 | 3.691 | VDCores | 16.1% faster | 18.6% faster |
| 2,048 | **3.326** | 3.848 | 3.918 | VDCores | 13.6% faster | 15.1% faster |
| 4,096 | 4.358[^s4k] | **4.164** | 4.281 | vLLM | 4.7% slower | 1.8% slower |
| 8,192 | --[^long-vdc] | 4.877 | **4.862** | SGLang | -- | -- |
| 16,384 | --[^long-vdc] | 6.285 | **6.125** | SGLang | -- | -- |
| 32,768 | --[^long-vdc] | 8.999 | **8.540** | SGLang | -- | -- |

[^s4k]: The S4K VDCores number is a 30-sample diagnostic median after five
  warmups. A subsequent 501-iteration stress run hit the long-context fault
  described below, so this row is not marked as a qualified VDCores result.
[^long-vdc]: No VDCores timing is reported. The temporary synthetic-context
  probe encountered an illegal memory access during the 32-layer resident
  schedule's K/V TMA layer transition from S6K onward and was reverted. The
  standalone attention task still passes at S8K, localizing this boundary to
  repeated layer scheduling rather than the single attention task.

vLLM uses its engine-core first/last-token timestamps and automatically selects
the FlashInfer HND backend. SGLang uses its streaming engine metric with the
FlashInfer backend and page size 64. Its experimental piecewise-prefill graph
is disabled after its own 16K-token compiler warmup failed on this stack; the
full decode CUDA graph remains enabled. Thus both framework columns retain
decode scheduling and launch overhead, while the VDCores column remains the
resident-megakernel internal span.

Final balanced-down 501-sample medians are 2.811616/2.810784/2.851328 ms at
S64/S128/S512 in job `20260806T011823Z-2068499`. The temporary long-context
VDCores probe measured 3.005920/3.325632 ms at S1K/S2K (501 samples) in job
`20260806T153156Z-745260` and 4.358496 ms at S4K (30 samples) in job
`20260806T153352Z-753912`; its S6K reproducer is
`20260806T153049Z-738092`. Strict short-context vLLM jobs are
`20260805T205321Z-66365` (S64), `20260805T205112Z-50840` (S128), and
`20260805T205544Z-83058` (S512). Strict SGLang jobs are
`20260805T213318Z-356270` (S64), `20260805T213450Z-368285` (S128), and
`20260805T213620Z-380084` (S512). The corrected S1K-S32K sweeps are
`20260806T150649Z-613023` for vLLM and `20260806T152302Z-698299` for SGLang.
The S256 jobs are `20260806T154512Z-812206` for VDCores,
`20260806T154029Z-788615` for vLLM, and `20260806T154257Z-801610` for
SGLang.

### Why vLLM steps up between S128 and S256

The 0.632 ms increase from S128 to S256 is not an attention-kernel threshold.
The exact vLLM FlashInfer call measures 4.414/4.507/5.440 us at
S128/S256/S512. Thus S128 to S256 adds only 0.093 us per layer, or about 3 us
over all 32 layers; S256 to S512 adds about 30 us. The complete S128 and S256
Nsight traces both contain 396 decode-plus-sampling kernels. Under identical
node tracing, their GPU spans are 3,332.576 and 3,288.799 us and their CPU
decode execution ranges are 1,168.608 and 1,161.024 us, respectively. Nsight
perturbs absolute latency, but neither topology nor device critical path has a
corresponding 0.632 ms jump.

The step comes from vLLM's asynchronous scheduling pipeline. A diagnostic
sweep inside one S256-capacity engine measures 2.856/2.847/3.122/3.452/3.456
ms at S128/S160/S192/S224/S256: S192 is the crossover and is visibly between
the two latency regimes. With asynchronous scheduling disabled, the same
contexts measure 4.257/4.241/4.257/4.254/4.267 ms, eliminating the step. The
short-context engine overlaps more of sampling, bookkeeping, and next-step
dispatch; around S192-S224 that overlap loses a pipeline phase and exposes a
roughly 0.6 ms bubble. Once in the slower phase, extending attention from S256
to S512 adds only its small kernel cost, explaining the nearly flat vLLM
S256-S512 result. The main table intentionally retains vLLM's production
launch-inclusive asynchronous behavior.

The exact-attention audit is job `20260806T160029Z-889385`; the paired async
and non-async sweeps are `20260806T160652Z-920346` and
`20260806T161311Z-957601`. S128/S256 node traces are
`20260806T161002Z-940301` and `20260806T155738Z-878196`.

### S128 kernel versus schedule audit

The S128 short-context deficit was split into identical-shape projection
probes, VDCores stage-frontier timestamps, and a vLLM CUDA-graph trace. The
framework task probes use CUDA events around repeated graph replays; VDCores
uses the resident kernel's cross-SM `globaltimer`. These are device-side task
scopes and deliberately exclude the launch-inclusive system comparison above.

| BF16 B8 projection, shape M x N x K | Production M64 VDCores (us) | M128 VDCores (us) | Group-4 M128 diagnostic (us) | vLLM (us) | SGLang (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Output, 4096 x 8 x 4096 | 7.328 | 6.560 | 6.112 | **5.949** | **5.949** |
| Down, 4096 x 8 x 14336 | 21.888 | 20.480 | 19.296 | 19.367 | **18.405** |

The group-4 path is diagnostic only: its fold-16 BF16 reduction ordering does
not satisfy the qualified 32-layer schedule. Its wider x4 TMEM drain now edges
the vLLM down probe by 0.4%, but remains 4.8% behind SGLang. The production
M64 path intentionally retains the qualified arithmetic order; its gain comes
from balancing and overlapping the complete layer schedule rather than using
the numerically rejected M128 reduction.

Temporary compute-stream markers then split the production S128 critical path:

| VDCores stage | Median (us) | Matching vLLM task sum (us) | Matching SGLang task sum (us) |
| --- | ---: | ---: | ---: |
| QKV + RoPE + KV append | 11.500 | 11.646 | **8.837** |
| Decode attention | **4.000** | 4.527 | 5.533 |
| Output projection + post-attention RMS | 9.500 | 8.384 | **8.204** |
| Gate/up + SwiGLU | **36.750** | 41.937 | 42.382 |
| Down projection + reduction completion + next RMS | 23.000 | 21.803 | **20.660** |
| Q-buffer clear | 0.250 | - | - |

Within the final row, the VDCores split is 21.000 us for down compute and
2.000 us for reduction completion plus the next RMS. The reduction is already
hidden behind the persistent memory/compute pipeline, and Q clear is already
placed on auxiliary SMs. The two projection/RMS stages trail the matching
vLLM sums by about 2.31 us/layer, or 74 us across 32 layers. That is the main
actionable kernel-level deficit; moving clear or adding another RMS fusion is
not.

The vLLM Nsight trace independently verifies a schedule-level contribution.
Its strict-B8 decode consists of 385 CUDA-graph nodes followed by 11 sampler
nodes. Under tracing, their summed kernel duration is 3437.791 us while the
critical-path span is 3332.255 us, exposing 105.536 us (3.07%) of graph-node
concurrency. Nsight instrumentation inflates absolute durations, so that value
is topology evidence rather than an untraced latency estimate. It nevertheless
shows why standalone task sums cannot predict the 91 us launch-inclusive S128
lead. The practical conclusion is: projection kernels are the VDCores-side
bottleneck, while vLLM also benefits from graph-level overlap.

The tuning is backed by three profiling scopes. Standalone VDCores task probes
use the resident runtime's cross-SM `globaltimer`; matching vLLM/SGLang task
probes use CUDA events around warmed CUDA-graph replays. Temporary megakernel
frontier markers split an S128 layer into QKV/RoPE/cache (11.500 us), attention
(4.000 us), output/RMS (9.500 us), MLP (36.750 us), down/reduction/RMS
(23.000 us), and clear (0.250 us). Per-SM event timelines expose which work is
actually concurrent, while Nsight is used only to verify framework graph
topology because its instrumentation changes absolute latency.

### Current S128 Blackwell projection follow-up

The retained M64 epilogue now follows CUTLASS's SM100 FP32-to-16-bit policy:
`16dp256b1x` drains TMEM to registers and `stmatrix` writes the converted
fragment to the existing shared TMA-reduction slot. It preserves the M64 UMMA
and reduction order. A wider x4 drain also reduces the existing diagnostic
M128 group-4 down probe from 19.680 to 19.296 us; x8 regressed to 19.776 us.

The larger end-to-end gain is scheduling. The old down projection mapped 104
SMs to a 28-K256-tile serial tail while the other 48 SMs carried only 12--16
tiles. The retained split uses 192 low-K fold-3 tasks and 256 high-K fold-4
tasks, all with K2048. Its placement gives 144 SMs three tasks and eight SMs
two tasks, reducing the critical load to 24 tiles. High-K waits directly on
the runtime barrier released by the register-forwarded SwiGLU tail; the next
RMS barrier counts every one of the 448 down contributors. No thread fence or
implicit memory-warp join is used.

| S128 measurement | Previous | Retained | Change |
| --- | ---: | ---: | ---: |
| Down schedule, resident `globaltimer` (us) | 22.816 | **21.888** | -4.1% |
| Full single-token decode, resident `globaltimer` (ms) | 2.900 | **2.811** | -3.1% |
| Fixed-context vLLM decode (ms) | 2.816 | 2.816 | VDCores now 0.2% faster |
| Fixed-context SGLang decode (ms) | 3.312 | 3.312 | VDCores now 15.1% faster |

The down schedule comparison is a paired 501-iteration standalone probe from
job `20260806T005532Z-1899332`; mean-relative BF16 error is 0.245% for the old
layout and 0.319% for the balanced layout. The final S128 confirmation is job
`20260806T011823Z-2068499`; an independent 501-iteration run reached 2.802048
ms in job `20260806T005129Z-1869027`.

Reproduce the down-stage comparison with the exact Llama compute image:

```bash
GEMV_M=4096 GEMV_K=14336 GEMV_SMS=152 GEMV_TILE_M=64 \
GEMV_TILE_PACKED=1 GEMV_DOWN_SCHEDULE=legacy GEMV_ITERS=501 \
  python benchmarks/blackwell_gemv.py
GEMV_M=4096 GEMV_K=14336 GEMV_SMS=152 GEMV_TILE_M=64 \
GEMV_TILE_PACKED=1 GEMV_DOWN_SCHEDULE=balanced GEMV_ITERS=501 \
  python benchmarks/blackwell_gemv.py
```

### Spare-SM and fusion follow-up

The 24 SMs outside the 128-SM rectangular projection grid were explicitly
tested rather than assumed idle. An all-152-SM LM-head partition measured
about 2.951 ms at S128 and lost to the 128-SM schedule because the extra,
uneven tasks increased queue and HBM traffic. A balanced all-SM MLP prefix
measured 3.230 ms because it disturbed the register-forwarded tail. Moving Q
clear earlier measured 2.917 ms versus 2.911 ms paired; the retained late clear
is already hidden on auxiliary SMs.

The "Group2 M64" probe means one CTA computes two independent M64N8 output
tiles for the same K slice and reuses the staged B tile. Its clean output
projection grid has 32 row pairs x 4 K partitions = 128 tasks. That is a task
factorization, not a 128-SM hardware limit; the other 24 SMs can run auxiliary
work. It required a separate compute opcode because the existing M64 opcode
owns one accumulator/output tile, while the M128 group-4 opcode has a different
TMEM layout and BF16 reduction order. Group2 improved the isolated K4096 probe
from 7.552 to 6.752 us, but was neutral in a projection prefix and either
failed the qualified 32-layer logits threshold or, for the passing 1536-row
subset, regressed S128 to about 2.951 ms. The experiment was therefore removed.

The retained fusion is instead LM-head epilogue argmax. Each of the 256
projection tasks emits one 16-byte `{value, absolute_index}` record per token;
eight reducer SMs consume those records. It avoids the padded logits
write/read and separate materialized argmax. In a fused/materialized/fused
500-step S128 sandwich, the two fused medians average 2.899672 ms versus
2.911456 ms materialized, a repeatable 11.784 us reduction. Splitting a
materialized early-argmax stage over 64 tasks regressed by about 32 us, and an
auxiliary-only 16-task version was neutral, so neither pipeline was retained.

### Observer-owned M2C readiness

The resident runtime now treats loaded-operand readiness as a producer-owned
phase. The load VCore is the only mbarrier participant, and all 128 compute
threads observe completion with an acquire parity wait. This removes 128
compute arrivals from each of roughly 4,530 handoffs per token while retaining
the existing memory-op path and TMA visibility rules.

At B8/S128, a same-source 501-sample internal-timer comparison improved from
2.734976 to 2.683328 ms (-51.648 us, -1.89%). The default observer build
passed full tensor validation and exact four-token resident control flow. The
selective image uses 126 registers, nine barriers, a 96-byte stack, and no
spills. `make m2c_legacy=1` restores the all-compute-thread arrival path for
A/B diagnostics.

Reproduce the retained path with:

```bash
# Rebuild the exact spill-free 11-operator image used below.
DAE_COMPUTE_OPS_FILE=benchmarks/blackwell_llama8b_fused_argmax.ops \
  make -B pyext

# Internal resident-megakernel span. Supply a prompt with the requested token
# count; the Llama tokenizer adds its BOS token.
python app/python/llama3/sched.py \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --prompt '<fixed-length prompt>' \
  -N 1 --no-control-flow --bench 100

# Run each context in a separate process from each framework's qualified
# environment. The benchmark rejects comma-separated multi-context runs so
# engine capacity cannot leak across rows.
python benchmarks/blackwell_fixed_context_decode.py \
  --framework vllm \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --contexts 128 --batch 8 --warmups 5 --samples 30
python benchmarks/blackwell_fixed_context_decode.py \
  --framework sglang \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --contexts 128 --batch 8 --warmups 5 --samples 30
```

Tensor-level validation passes for a non-control-flow step, exact greedy tokens
match Hugging Face, and a 130-step launch crosses from one KV128 block to two
with the unified online-softmax path. The exact 11-operator Llama image uses
128 registers, 9 barriers, a 96-byte stack frame, and zero spills. Removing
the prior padded attention opcode from the selective image lowered the
megakernel-wide register allocation from roughly 202 registers.
Clean balanced-path checks pass at S1 and four-token control flow in
`20260806T005328Z-1885899`; the latter exactly matches
`[75987, 57918, 706, 264]`. The S128 tensor check and exact final token pass in
`20260806T005041Z-1862547`.
`tests/blackwell_runtime_smoke.py` also covers synchronous, asynchronous, and
bulk sequence launches on all 152 SMs.

## Implementation references

The retained implementation follows the SM100 programming model and UMMA/TMEM
layouts described by the [CUDA Blackwell tuning guide](https://docs.nvidia.com/cuda/archive/13.0.1/blackwell-tuning-guide/index.html),
the [CUDA PTX ISA tcgen05/TMEM synchronization model](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html),
and [CUTLASS Blackwell GEMM guidance](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html),
including its TMEM-to-register epilogues and Blackwell-specific pipelines.
The exact wide-load and shared-store selection follows the official
[CUTLASS SM100 epilogue builder](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/epilogue/collective/builders/sm100_builder.inl).
Kernel organization and comparison regimes were cross-checked against the
[CUTLASS SM100 low-latency GQA example](https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa),
[Triton Blackwell persistent-matmul example](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py),
[FlashInfer decode APIs and SM100 CuTe path](https://github.com/flashinfer-ai/flashinfer),
[vLLM's CUDA-graph model runner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu/model_runner.py),
[SGLang's decode runner and attention backends](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/model_executor/model_runner.py),
and the [FlashAttention-4 SM100 CuTeDSL implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/flash_fwd_sm100.py).
