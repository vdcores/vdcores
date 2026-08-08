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

### Issuer-owned UMMA/TMEM pipeline

Ordinary M64N8 projection tasks now keep the per-tile M2C dequeue, UMMA issue,
UMMA completion wait, and slot release on the issuing compute warp. The other
three compute warps advance their private queue cursors without repeating the
acquire waits, then join once for the four-warp TMEM epilogue. The handoff uses
`tcgen05.fence::before_thread_sync`, the existing 128-thread compute barrier,
and `tcgen05.fence::after_thread_sync`; it does not add a memory-warp join or a
thread fence.

A temporary internal task probe on the M4096/K4096 fold-two shape attributed
2.624 us (39.2%) to operand arrival, 2.784 us (41.6%) to UMMA issue, and only
0.320 us (4.8%) to UMMA completion. The probe was removed after directing the
optimization. Paired 501-iteration standalone medians on the same selective
image were:

| Projection schedule | Compute tiles/task | Previous (us) | Issuer-owned (us) | Gain |
| --- | ---: | ---: | ---: | ---: |
| M2048 x N8 x K4096, 128 SMs/fold 4 | 4 | 4.160 | **4.128** | 0.8% |
| M4096 x N8 x K4096, 128 SMs/fold 2 | 8 | 6.816 | **6.688** | 1.9% |
| M6144 x N8 x K4096, 96 SMs/fold 1 | 16 | 11.296 | **11.200** | 0.8% |
| Balanced M4096 x N8 x K14336, 152 SMs | 12/16 | 21.664 | **21.504** | 0.7% |

An S128 control/issuer/control sandwich measured
2.623840/2.608672/2.624512 ms with the resident internal timer, a 15.504 us
(0.59%) gain against the control mean. The final minimal 11-operator image
measures 2.610624 ms over 501 iterations, 7.29% faster than the strict vLLM
2.816003 ms baseline. It uses 96 registers, nine barriers, a 96-byte stack,
and no spills. Full S128 tensor validation, exact token 24748, and exact
four-token resident reuse all pass (jobs `20260807T195543Z-3887464`,
`20260807T195611Z-3888026`, and `20260807T195656Z-3888515`).

Two follow-up pipeline-depth ideas were rejected and removed. Sending B to the
otherwise-idle second load warp on an unbarriered task regressed M4096/K4096
from 6.688 to 7.040 us; the second port is useful for bypassing a blocked
activation load, not for increasing steady-state TMA throughput. Halving the K
tile to 128 while retaining a four-tile B cadence reduced a TMA stage from 18
slots to nine, but doubled UMMA issue/completion points and regressed the same
shape to 10.432 us (job `20260807T200451Z-3899356`).

Issuer ownership was also tested in the fused tail up-projection/SwiGLU task
and removed.  Merely letting the three non-issuer warps advance to the retained
gate `RegLoad` early was neutral: the control/variant/control medians were
2.608224/2.613344/2.618912 ms, or 0.224 us faster than the control mean.  Warp
0 still evaluated its own sigmoid fragment and remained the join's critical
warp.  A stronger sidecar version used warps 1--3 to transform all 512 gate
elements in place while warp 0 owned UMMA, then reloaded each native fragment
after the legal TMEM handoff.  Exact-token and tensor validation passed (job
`20260807T203352Z-3930180`), but its control/variant/control medians were
2.614720/2.613888/2.615136 ms, only a 1.040 us (0.040%) gain.  Stage profiling
showed why: the local `gate_tail` to `silu_tail` maximum increased from 11.360
to 12.256 us because the 96-thread shared rewrite and reload cost more than the
hidden SFU work.  The experimental opcode, selector, and task were removed.

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

An issuer-owned fused-RoPE variant was also evaluated and removed.  Warp 0
owned the per-tile M2C waits, UMMA issue/completion, and releases while the
other compute warps advanced their private cursors and prefetched their final
cosine/sine pairs before the shared TMEM epilogue.  In 501-iteration probes it
improved Q from 7.296 to 7.104 us and K from 4.512 to 4.416 us, and exact S128
validation passed.  The full-model control/variant/control medians were
2.609888/2.608224/2.607232 ms, however: only 0.336 us (0.013%) against the
control mean.  The extra opcode and selector were therefore not retained
(jobs `20260807T201641Z-3913839`, `20260807T201719Z-3914435`, and
`20260807T201802Z-3915123`).

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

### Packed SwiGLU shard ownership

The three materialized 2,048-element SwiGLU shards retain their independent
readiness barriers and all 24 token-shard tasks, but no longer round-robin
unrelated shards over the 24 auxiliary SMs.  Shards 0 and 1 are both placed
on SMs 128--135 after those CTAs finish the shard-1 up-projection tail.  Shard
2 stays on SMs 144--151, which are also its late producer/consumer group.
Early down-projection owners therefore cannot be parked behind an unrelated
shard-2 input frontier.

A profiling-disabled control/packed/control sandwich on the same 13-op image
measured 2.617888/2.594016/2.618592 ms, a 24.224 us gain against the control
mean (jobs `20260807T232617Z-4075840`, `20260807T232655Z-4076228`, and
`20260807T232733Z-4076690`).  Moving shard 2 to bases 136 or 140 measured
2.598976/2.599520 ms in 201-sample screens; the retained base 144 measured
2.592256 ms (jobs `20260807T232855Z-4077926`,
`20260807T232935Z-4078389`, and `20260807T233014Z-4078979`).

The exact 11-op production image reaches a fresh 2.586304 ms S128 median over
501 internal-timer samples in `20260807T233431Z-4082641`.  It remains at 96
registers, nine barriers, a 96-byte stack, and zero spills.  Full S128 tensor
validation and exact token 24748 pass in `20260807T232532Z-4075060`; four
resident reuses across KV128 exactly match `[24748, 24748, 24748, 24748]` in
`20260807T233102Z-4079837`.  `VDCORES_PACKED_SILU_SHARDS=0` restores the old
round-robin placement for diagnostics.

### Rejected RMS physical-ownership move

Moving the unchanged eight-token RMS tasks away from SMs 0--7 did not expose
useful operand pre-staging.  With packed SwiGLU ownership enabled, moving the
post-attention RMS to bases 64 and 128 measured 2.587264 and 2.594400 ms over
201 internal-timer samples (jobs `20260807T233726Z-4084994` and
`20260807T233807Z-4085619`).  Moving the next-layer pre-attention RMS to bases
64 and 128 measured 2.584672 and 2.589696 ms (jobs
`20260807T233850Z-4086100` and `20260807T234038Z-4087542`) versus the fresh
2.586304 ms production median.  The only apparent improvement was 1.632 us,
below run-phase variance, while auxiliary ownership could be worse by 8.096
us.  Fixed SM0--7 placement and the minimal schedule were restored.

Separating packed SwiGLU shard 0 from shard 1 was also screened after the new
frontier trace showed a 3.424 us maximum local SiLU interval (job
`20260807T234426Z-4090696`).  Moving shard 0
onto main-path SMs 120--127 regressed from 2.590240 to 2.670624 ms because it
extended the gate/up-tail owners (jobs `20260807T234544Z-4091879` and
`20260807T234627Z-4092575`).  Moving it to the remaining auxiliary subgroup,
SMs 136--143, initially appeared 3.504 us faster, but a 501-sample pair was
2.588480 versus 2.589312 ms (jobs `20260807T234848Z-4094941` and
`20260807T234928Z-4095458`).  The 0.832 us delta is below run variance: the
existing paired shard-0/1 work is already hidden behind the main gate/up tail,
so no selector or extra placement remains.

### Rejected phased register-fused MLP tail

Two structural forms tried to expose the independent 8,192-element
register-forwarded gate/up tail earlier.  A coarse reorder ran the complete
tail and 48 high-K down tasks before the materialized prefix.  It remained
correct (job `20260807T235247Z-4098203`) but measured 2.881152 ms in
`20260807T235328Z-4098657`: advancing the tail also delayed every prefix gate
producer, so low-K readiness lost almost a full tail per layer.

The finer form preserved the production order and split tail readiness into
two K4096 phases.  Existing fold-4 down tasks used
`SchedGemvPhasedActivation`, so each task selected its half's barrier inside
one grouped memory program instead of recreating the older four-schedule
handoff penalty.  The VM's 10-bit barrier field required sharing head 7's Q
and KV counter; a separate screen showed that merge alone was neutral at
2.596864 versus 2.596640 ms.  Full tensor correctness and token 24748 passed
in `20260808T000102Z-4105406`, and four resident tokens exactly matched in
`20260808T000655Z-4110115`.

The diagnostic 13-op image initially appeared promising: a 501-sample
control/phase/control sandwich measured 2.592608/2.586624/2.594656 ms, a
7.008 us gain (jobs `20260808T000306Z-4107092`,
`20260808T000345Z-4107660`, and `20260808T000428Z-4108302`).  Its marker trace
moved the final-layer next-RMS frontier from 77.024 to 75.040 us.  The exact
11-op production image did not confirm that magnitude: it measured
2.588832/2.588128/2.590848 ms, only 1.712 us faster than the control mean
(jobs `20260808T000930Z-4111870`, `20260808T001011Z-4112465`, and
`20260808T001056Z-4113476`).  That is below the retention threshold for an
extra per-layer frontier and a Q/KV dependency merge, so all phased-tail,
barrier-sharing, and selector code was removed.

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

Two completion-preserving LM-head ownership variants were subsequently tested
and removed.  Unlike the earlier unsafe completion grouping, both retained one
UMMA completion and immediate operand release per output group.  The first
parked three compute warps during the mainloop and rejoined once for the
four-accumulator epilogue.  It passed exact S128 correctness (job
`20260807T204635Z-3944856`) but measured
2.610976/2.617600/2.616448 ms in a control/variant/control sandwich, a 3.888 us
regression.  Stage profiling localized 3.488 us of that cost to the first LM
epoch; the warm second epoch was unchanged.

The stronger version used four one-way named barriers.  After each final
output-group completion, warp 0 published that disjoint TMEM range and
continued issuing, while warps 1--3 drained and compared their native argmax
fragments.  It was exact and repeat-safe for 501 resident runs (jobs
`20260807T205651Z-3953651` and `20260807T205810Z-3954445`), but increased the
image from 96 registers/nine barriers to 106 registers/13 barriers.  Its
control/variant/control medians were 2.615520/2.617312/2.615296 ms, a 1.904 us
regression.  The warm second LM epoch improved 70.176 to 69.664 us, but the
first epoch worsened 80.576 to 83.136 us; handler/barrier footprint exceeded
the 0.512 us steady overlap.  The extra opcode, barriers, and selector were
therefore removed rather than retaining a complex sub-microsecond path.

The next experiment split fused LM-head argmax into a hierarchical pipeline.
After the first 128-task logits epoch, eight auxiliary SMs each reduced one
token's 128 compact records while the main 128 SMs ran the second epoch.  The
summary replaced the last epoch-0 record, so the final reducer consumed one
contiguous 129-record range instead of 256 records.  The path used explicit
128-thread compute barriers before its C2M publications; it did not rely on a
thread fence to join a memory warp that never entered the compute task.

The pipeline passed full S128 tensor checks and produced the exact token 24748
(job `20260807T211437Z-3968085`).  Its unprofiled 501-run control/variant/control
medians were 2.614976/2.614464/2.619616 ms (jobs
`20260807T211526Z-3968535`, `20260807T211603Z-3969081`, and
`20260807T211643Z-3969665`).  The apparent 2.832 us advantage over the control
mean was smaller than run-to-run drift.  Stage traces showed why the design did
not provide a robust tail reduction: the auxiliary reduction itself remained
about 15--16 us after epoch 0 became ready, and the measured LM-head/argmax
frontier increased from about 231.2 to 233.4 us (jobs
`20260807T211858Z-3971545` and `20260807T211937Z-3971892`).  The extra opcodes,
barriers, and schedule were removed.

The following down-projection experiment tested whether wider M tiles could
use the otherwise idle Blackwell SMs more effectively.  The first three
K2048 shards retained 192 M64 tasks, while the K6144--14336 tail changed from
256 M64 tasks to 128 M128 tasks.  The placement kept the critical issue load
balanced at 24 UMMA instructions on both the 128 primary and 24 auxiliary
SMs, preserved all seven BF16 reduction contributors, and preserved the
existing SiLU readiness boundaries.  A generic M128 implementation passed
full S128 correctness but regressed the 501-sample median by 47.040 us.

To separate tile width from runtime synchronization, a temporary M128
issuer-owned opcode then gave warp 0 the same M2C/UMMA/release protocol as the
retained M64 path.  It compiled at the unchanged 96 registers, nine barriers,
96-byte stack, and zero spills, and passed every S128 tensor check plus exact
token 24748 in job `20260807T214340Z-3996557`.  Its unprofiled
control/variant/control medians were 2.608928/2.656864/2.615712 ms (jobs
`20260807T214438Z-3997646`, `20260807T214518Z-3998427`, and
`20260807T214555Z-3998985`), a 44.544 us regression versus the bracketing
control mean.  The wider tasks reduce command count but make twice as much
work indivisible and delay useful producer/consumer interleaving in every
layer.  The M128 opcode, duplicate packing, selector, and schedule were
removed; the retained down projection remains uniformly M64.

A deferred Q-cleanup proof then moved the unchanged 2 MiB zero stream out of
the 32 layer bodies.  All layer Q rows were temporarily made contiguous, and
24 auxiliary SMs cleared the complete range while the 128 primary SMs ran the
LM head.  The final layer's three attention-output barriers protected Q
lifetime, and the token-completion barrier joined cleanup before resident
reuse.  The first implementation incorrectly waited on that token barrier
inside the auxiliary stream before SM128 could execute the barrier-restore
task; job `20260807T215928Z-4010850` exposed the cycle and was stopped after
its exact worker PID was verified.  Publishing completion without the early
wait fixed the protocol, passed every S128 tensor check, and produced exact
token 24748 in job `20260807T220350Z-4014844`.

The safe batched form still lost.  Releasing the token barrier from every
1 KiB store measured 2.621248 ms against 2.612832/2.609920 ms controls, a
9.872 us regression (jobs `20260807T220431Z-4015238`,
`20260807T220515Z-4016024`, and `20260807T220558Z-4016515`).  A second form
used ordered per-CTA streams and published only each CTA's final store,
reducing 2,048 barrier updates to 24.  It measured 2.625568 ms against the
bracketing 2.609920/2.613856 ms controls, a 13.680 us regression (jobs
`20260807T220759Z-4018311` and `20260807T220838Z-4018991`).  The original
per-layer clears were already mostly hidden; concentrating the same traffic
in the bandwidth-bound LM window costs more than removing their small visible
tail.  Contiguous Q storage, batch scheduling, selector, and completion
changes were removed.

A producer-aggregated C2M experiment kept the store VCore as the final
participant in the existing 129-count phase, so the 32-entry queue retained
its backpressure.  Ordinary issuer-owned GEMV epilogues first joined their
four compute warps, then lane 0 contributed an update count of 128 instead of
executing 128 separate nonblocking arrivals.  The 11-op image remained at 96
registers, nine barriers, a 96-byte stack, and zero spills.  Full S128 tensor
validation and exact token 24748 passed in job
`20260807T221619Z-4025434`.

The full compute join measured 2.614208 ms versus bracketing production
controls of 2.610880 and 2.608576 ms, a 4.480 us regression (jobs
`20260807T221658Z-4025874` and `20260807T222512Z-4032749`).  A split-phase
named-barrier form let warps 1--3 arrive and continue while issuer warp 0
waited before C2M publication.  It measured 2.616544 ms against
2.610688/2.608576 ms controls, a 6.912 us regression (jobs
`20260807T221904Z-4027728`, `20260807T222236Z-4030879`, and
`20260807T222512Z-4032749`).  The original per-thread C2M arrivals are
nonblocking and cheaper than adding a recurring compute rendezvous.  The
aggregate queue method, split barrier, build selector, and GEMV branch were
removed.

A warp-coalesced M2C-wait experiment tested whether issuer warp lane 0 could
absorb the variable-duration observer spin before the other 31 lanes consumed
the loaded operand.  The conservative implementation made lane 0 complete the
wait, executed `__syncwarp()`, and then made every other consuming lane execute
one already-ready acquire wait.  This follows the PTX mbarrier visibility rule,
which guarantees producer writes to the thread that executes a successful
`mbarrier.test_wait`/`mbarrier.try_wait`; `__syncwarp()` orders memory among its
participants but does not document transitive propagation of that async-proxy
acquire.  See the official [PTX mbarrier
semantics](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait)
and [CUDA warp synchronization
semantics](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#synchronization-functions).

The extra warp synchronization failed the standalone gate.  K4096 M64 rose
from 6.912 to 7.680 us, and the balanced K14336/down shape rose from 21.600 to
28.064 us (controls `20260807T222804Z-4035435` and
`20260807T222831Z-4035802`; variants `20260807T223059Z-4037035` and
`20260807T223128Z-4037257`).  A leader-only acquire was not retained because
the documented visibility guarantee applies to the executing thread.  No
end-to-end run was warranted after both task shapes regressed; the helper,
selector, and GEMV branches were removed.

### Rejected K512 projection stage

A wider ordinary-projection stage tested whether halving the number of
memory-to-compute handoffs and UMMA groups could expose more useful overlap.
The proof changed M64N8K256/B4 into M64N8K512/B2 while preserving each
activation group's K1024 cadence.  It also added a fixed 64 KiB rank-5 TMA
load because the memory instruction's 16-bit byte-count field cannot encode
65536.  The experimental image was spill-free and the all-scope Llama run
passed every S128 tensor check with exact token 24748.

The isolated, repeatedly reused matrices favored the wider stage:

| Shape | K256 | K512 | Change |
| --- | ---: | ---: | ---: |
| M1024 K4096 / 64 SMs | 4.096 us | 3.744 us | -8.6% |
| M4096 K4096 / 128 SMs | 6.944 us | 6.240 us | -10.1% |
| M6144 K4096 / 96 SMs | 11.488 us | 9.760 us | -15.0% |
| M8192 K4096 / 128 SMs | 11.648 us | 9.952 us | -14.6% |
| M4096 K14336 / 152 SMs | 21.792 us | 21.856 us | +0.3% |

The result did not transfer to the resident inference schedule.  Against a
2.618624 ms same-image control, V-only measured 2.637216 ms (+18.592 us),
output-only measured 2.687488 ms (+68.864 us), MLP-only measured 2.821312 ms
(+202.688 us), and all eligible projections measured 2.852544 ms
(+233.920 us).  Those runs are `20260807T230222Z-4055721`,
`20260807T230301Z-4056062`, `20260807T230342Z-4056490`, and
`20260807T230031Z-4054735`; the control is
`20260807T225950Z-4054291`.

An eight-epoch M6144/K4096 probe using eight distinct matrices reduced the
apparent gain to 84.128 versus 83.168 us (-1.1%; jobs
`20260807T230426Z-4056780` and `20260807T230459Z-4057166`).  The hot-matrix
microbenchmark therefore overstated the benefit, while the full resident
schedule exposed the dominant cost: each K512 operand consumes eight
contiguous 8 KiB slots and retires at twice the granularity, reducing memory
VCore allocator/load runahead and delaying cross-task interleaving.  The
K512 opcodes, fixed-size memory operation, scheduler selectors, and benchmark
variant were removed; production retains the finer K256 stage.
After removal, the exact 11-op production image rebuilt at 96 registers, nine
barriers, a 96-byte stack, and zero spills.  A fresh 501-sample S128 run
measured a 2.614752 ms internal median and 2.597792 ms minimum in job
`20260807T231129Z-4062821`, consistent with the retained 2.608576--2.609728 ms
production medians.

### Rejected compact register-to-TMEM projection path

A native-M64 tensor/shared proof removed the padded M128 UTCCP source by
loading the compact M64 weight tile into registers and storing it into the
four native TMEM datapath bands.  It compared serialized staging, ping-pong
staging, staging plus disjoint epilogue drain, reuse of the drain group, and
an eight-compute-warp form.  The compact mapping itself is valid: a separate
M64K16 test passed 10,001 bit-exact repetitions with zero mismatches in job
`20260807T231714Z-4068128`.

The 132-CTA timing suite nevertheless rejected the mechanism:

| K | Production SS | Best compact R2T | Eight-warp SS control | Result |
| ---: | ---: | ---: | ---: | --- |
| 256 | **0.897130 us** | 1.921630 us | 0.960740 us | R2T +114.2%; eight warps +7.1% |
| 1024 | **7.173560 us** | 8.453400 us | 7.894200 us | R2T +17.8%; eight warps +10.0% |

Ping-pong plus disjoint drain did overlap real work: it reduced the compact
path from 2.305440 to 1.921630 us at K256 and from 9.482040 to 8.453400 us at
K1024.  Register-mediated shared-to-TMEM movement itself still costs more than
the overlap removes, and the production K256 stage is the worst case.  The
eight-warp R2T form measured 2.049740/8.751720 us and did not reverse the
result.  Every timed form passed 101 exact checks with zero errors in job
`20260807T231739Z-4068687`; no runtime integration was warranted.  The proof
binaries had no recoverable source in the canonical tree or worker-local
paths, so they remain diagnostic artifacts rather than retained code.

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
# Rebuild the current exact spill-free 12-operator image.
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
with the unified online-softmax path. At that milestone, the exact
11-operator Llama image used
128 registers, 9 barriers, a 96-byte stack frame, and zero spills. Removing
the prior padded attention opcode from the selective image lowered the
megakernel-wide register allocation from roughly 202 registers.
Clean balanced-path checks pass at S1 and four-token control flow in
`20260806T005328Z-1885899`; the latter exactly matches
`[75987, 57918, 706, 264]`. The S128 tensor check and exact final token pass in
`20260806T005041Z-1862547`.
`tests/blackwell_runtime_smoke.py` also covers synchronous, asynchronous, and
bulk sequence launches on all 152 SMs.

### Retained cross-layer Q/K ownership overlap

The last down-projection frontier leaves SMs 104--127 later than the other Q
owners, while all 24 auxiliary SMs have crossed the layer boundary.  The
retained schedule therefore moves Q fold 1 for heads 5--7 from SMs 104--127
to SMs 128--151.  K for those heads stays on SMs 104--127, so the two
independent projections run concurrently after next-layer RMS instead of
serializing on one CTA.  Output ranges, K folds, per-head barriers, compute
opcodes, and reduction arithmetic are unchanged.

This placement exposed an implicit dependency: K had inherited the RMS
acquire from its colocated Q command.  The final form adds that load barrier
only to the three vacated K head schedules.  The unchecked form corrupted K
immediately; the explicit form passed every S128 tensor threshold and exact
token 24748 in `20260808T010836Z-4161441`.  The final default configuration
repeated that full validation in `20260808T011748Z-4166632`.

On the exact 11-op image, a 1,001-sample control/overlap/control sandwich
measured 2.589344/2.576576/2.588384 ms in jobs
`20260808T011324Z-4164348`, `20260808T011405Z-4164605`, and
`20260808T011446Z-4165097`.  The retained placement saves 12.288 us / 0.47%
against the control mean and reaches 2.576576 ms at B8/S128, 8.50% ahead of
the 2.816003 ms strict vLLM baseline under the agreed accounting.  Four
resident steps across KV128 exactly matched
`[24748, 24748, 24748, 24748]` in `20260808T011532Z-4165458`.
`VDCORES_Q_FOLD1_AUX_TAIL=0` restores the colocated schedule for A/B runs.
The image remains at 96 registers, nine barriers, a 96-byte stack, and zero
spills.

`VDCORES_STAGE_PROFILE_DETAIL=name[,name...]` prints the complete per-SM
frontier for selected existing profile markers.  This is diagnostic only and
does not add a compute opcode or execute when stage profiling is disabled.

### Retained Q/V versus K/V work rebalance

After the auxiliary-Q split, SMs 104--127 own only one short K task while the
original Q owners still serialize Q and V.  Per-head profiles selected V
heads 3, 6, and 7 as the slow grouped-output contributors.  Their V tasks now
run after K on SMs 104--127, leaving the original owners with Q alone.  This
turns three long-Q-plus-short-V chains into long Q in parallel with short
K-plus-short-V chains; all tensor ranges and head barrier counts remain
unchanged.

As with the Q move, the memory program needs an explicit semantic edge.  A
first timing-only form appeared about 15 us faster but loaded stale next-layer
RMS data and failed at layer 1.  Adding the RMS load barrier directly to each
moved V schedule passed every S128 tensor threshold and exact token 24748 in
`20260808T012706Z-4174330`; the final default repeated that validation in
`20260808T013403Z-4180262`.

On the exact 11-op image, valid control/rebalanced/control 1,001-sample
medians were 2.581376/2.574240/2.582592 ms in jobs
`20260808T013026Z-4177165`, `20260808T013107Z-4177759`, and
`20260808T013150Z-4178381`.  The 7.744 us / 0.30% gain comes entirely from
physical task ownership.  Four resident steps exactly matched the reference
in `20260808T013237Z-4179012`.  `VDCORES_V_K_TAIL=0` restores the prior V
placement for A/B runs; the production image remains at 96 registers, nine
barriers, a 96-byte stack, and zero spills.

### Rejected mixed LM-head epoch order

A schedule-only proof split each 65,536-row LM-head epoch into two 64-CTA
halves.  SM0--63 traversed epoch 0 then epoch 1, while SM64--127 traversed
epoch 1 then epoch 0.  Thus both weight halves streamed in each physical wave
without changing the 256 direct4 tasks, partial-record indices, fused argmax,
or final barrier count.  Full S128 correctness and exact token 24748 passed in
`20260808T013705Z-4182901`.

The reordered path measured 2.574368 ms over 501 internal samples in
`20260808T013743Z-4183357`, versus 2.573696 ms for the ordinary epoch order in
`20260808T013821Z-4183936`.  The 0.672 us regression shows that merely mixing
the two HBM streams does not remove the first-epoch cost.  The coordinate
adapters, four half schedules, and selector were removed.
The restored retained schedule measured a fresh 2.568928 ms in
`20260808T013929Z-4185130`, 8.77% ahead of the strict vLLM baseline.

### Absolute stage frontiers and rejected V remapping

Per-SM marker times have two useful origins.  `frontier_us` remains relative
to that SM's own layer-start marker, while `absolute_us` is relative to the
earliest layer start across the resident grid.  The latter is required for
placement decisions because different CTAs enter a layer at different times.
`VDCORES_STAGE_PROFILE_DETAIL=name[,name...]` now prints both without adding
an opcode or executing in the profiling-disabled production image.

The absolute profile in `20260808T014257Z-4187811` showed Q completion
balanced at about 11.4--12.6 us.  K heads 5--7 completed around 9.4 us, versus
roughly 15 us for heads 0--4, so V heads 5/6/7 were tested on the three K-only
groups in place of the retained 3/6/7 set.  The alternate mapping passed every
S128 tensor threshold and exact token 24748 in
`20260808T014527Z-4190044`.

It did not improve the converged frontier.  On the identical 13-op marker
image, a 301-sample control/alternate/control sequence measured
2.578112/2.578880/2.579136 ms in `20260808T014357Z-4188846`,
`20260808T014605Z-4190577`, and `20260808T014648Z-4191076`.  The alternate
lies inside control drift, so production retains heads 3/6/7.  Component
readiness is not sufficient when the head's attention/output chain is limited
by a different owner.

### Rejected absolute-prefix rebalance and dual-port LM loads

The absolute profile also exposed why the auxiliary MLP prefix cannot simply
be spread over otherwise earlier main CTAs.  A schedule-only proof capped the
192 gate/up prefix tasks at two per CTA and moved the late up tasks onto
SM0--39.  Full S128 correctness and token 24748 passed in
`20260808T015048Z-4194191`, but the 301-sample internal median was
2.835872 ms in `20260808T015129Z-833`.  The profile in
`20260808T015207Z-1581` showed prefix completion improving from about 62 to
57 us while the layer frontier worsened from about 84 to 97 us: the moved
work ran in front of those CTAs' independent register-forwarded gate/up tail.
The placement selector was removed.

Two LM-head load-VCore variants were also rejected.  Moving the small shared
activation load to port 1 was neutral at 2.573504 versus 2.573536 ms in
`20260808T015734Z-5956` and `20260808T015814Z-6642`.  Splitting two of the
four weight-group TMA streams across port 1 passed every S128 tensor threshold
and exact token 24748 in `20260808T015932Z-7875`, but a longer
control/variant/control sequence measured 2.573920/2.575264/2.571872 ms in
`20260808T020011Z-8115`, `20260808T020041Z-8504`, and
`20260808T020111Z-8907`.  The 2.368 us regression against the control mean
shows that two TMA issuers still feed one ordered M2C/UMMA/slot-retirement
pipeline.  All port selectors and generic schedule hooks were removed.

### Rejected parked light-compute warps

A selectable runtime prototype added one or two compute-only helper warps
after the existing four compute and four memory warps.  The helpers slept on
a named barrier between commands rather than polling the instruction stream,
and every data publication still used the compute barriers and the normal
memory-op path.  This tested extra CUDA throughput without changing the
well-qualified 128-thread implementations or making the memory VCore join an
invalid thread fence.

Three placements were screened in one spill-free selectable image.  A paired
SwiGLU command assigned gate and up to separate warp groups.  A wide command
joined both groups only for the late 2,048-element shard.  A balanced RMS
command split the 512 BF16 vector packs 384/128 between the original four
warps and two helpers, then used one six-warp reduction and one ordinary C2M
publication.  Full S128 validation produced exact token 24748 for each
mechanism; the two-helper image used 96 registers, 11 barriers, a 96-byte
stack, and zero spills.

Same-image 501-sample internal-timer sandwiches were all neutral or worse:

| Extra-thread scope | Disabled controls (ms) | Enabled (ms) | Delta |
| --- | ---: | ---: | ---: |
| One-warp paired SwiGLU | 2.594816 / 2.595520 | 2.596640 | +1.472 us |
| Two-warp paired SwiGLU | 2.594048 / 2.597920 | 2.594528 | -1.456 us |
| Two-warp wide late shard | 2.599552 / 2.603232 | 2.602016 | +0.624 us |
| Pair plus wide late shard | 2.599552 / 2.603232 | 2.602528 | +1.136 us |
| Six-warp RMS | 2.592128 / 2.593984 | 2.593536 | +0.480 us |

The apparent 1.456 us paired result is below the run-phase spread, and marker
profiles showed the same 3.168 us maximum MLP-prefix-to-SwiGLU frontier with
and without it: the earlier pair remained hidden behind late shard 2.  The
wide and RMS forms exposed the recurring six-warp rendezvous and did not
repay it with their small amount of elementwise work.  Relevant jobs are
`20260808T002914Z-4128202`--`20260808T003053Z-4130061` for one helper,
`20260808T003410Z-4133114`--`20260808T003527Z-4134186` for two helpers,
`20260808T004316Z-4141337`--`20260808T004517Z-4142812` for wide SwiGLU, and
`20260808T005045Z-4147638`, `20260808T005317Z-4149560`, and
`20260808T005358Z-4149974` for RMS.  The helper opcodes, mailbox, barriers,
selectors, and wider launch were removed; production remains the minimal
eight-warp runtime and the exact 11-op image.  The restored image passed every
S128 tensor threshold and exact token 24748 in `20260808T005920Z-4154831`;
its fresh 501-sample internal median was 2.588416 ms in
`20260808T005959Z-4155420`.

### Parked full-warpgroup base-cost qualification

An experimental build used a 384-thread CTA with the normal
four-warp task interpreter, four parked auxiliary compute warps, and the
unchanged four memory warps. Existing tasks and M2C/C2M barriers still have
only 128 compute participants. A 501-sample default/aux/default S128
sandwich measured 2.569824/2.574720/2.569120 ms, so the parked group costs
5.248 us (0.20%) before it performed useful work.

Paired projection chains did not repay that budget. Two independent UMMA
issuers were neutral at K2048 (12.480 versus 12.512 us) and 0.224 us slower at
K4096. Giving the sidecar only the completed TMEM epilogue was neutral for two
tasks, but four tasks regressed from 26.464 to 26.880 us. A narrower
32-producer-plus-128-consumer handoff still regressed two/four tasks by
0.256/0.192 us. The gate/up form passed full S128 correctness and exact token
24748, but its 2.579200 ms median was only 0.416 us inside same-image control
drift and roughly 9.7 us slower than the neighboring production image. All
paired-op, mailbox, barrier, schedule, and wider-runtime code was removed.
The restored 11-op image returned to 96 registers, nine barriers, a 96-byte
stack, and zero spills; four-token control-flow correctness passed and its
fresh 501-sample S128 internal median was 2.569056 ms.

### Rejected distributed atomic LM-head reduction

An absolute stage profile showed about 15.8 us between the slowest distributed
LM-head completion marker and the final token marker. Compute-side atomic
maxima, 16-way sharded atomic maxima, and a store-VCore-owned atomic handoff
were tested as alternatives to the 256-record reducer. All forms used packed
64-bit value/index keys and passed exact token correctness; the final staged
store form also passed four consecutive resident steps in
`20260808T033523Z-86948`.

Same-image 501-sample internal-timer sandwiches were:

| Reduction organization | Controls (ms) | Variant (ms) | Delta |
| --- | ---: | ---: | ---: |
| One compute-side key | 2.576192 / 2.575520 | 2.574464 | -1.392 us |
| 16 sharded compute-side keys | 2.565248 / 2.572288 | 2.570688 | +1.920 us |
| Store-VCore atomics | 2.570080 / 2.570816 | 2.570240 | -0.208 us |
| Store-VCore epoch retention | 2.569920 / 2.570528 | 2.572448 | +2.224 us |

The two apparent gains are below control drift. The staged form halved partial
publications but still lost, and its selectable image required 128 rather than
96 registers. This confirms that the existing reducer substantially overlaps
the LM-head tail; replacing its records with contended global atomics does not
remove a serial 15.8 us. All experimental runtime and task code was removed.
The restored 11-op image passed all 20 runtime tests and four resident greedy
steps in `20260808T034403Z-94356`; its fresh 501-sample S128 internal median
was 2.570208 ms in `20260808T034444Z-95073`.

### Rejected same-owner auxiliary up pairing

A narrow paired-M64 task reused each normalized-activation B tile only for two
up outputs already serialized on the same CTA and feeding the same MLP shard.
Both outputs retained independent accumulators, their original BF16 order, two
TMA stores, and two barrier releases. The selectable image stayed at 96
registers/nine barriers/zero spills, and four resident steps were exact in
`20260808T035827Z-106651`.

| Scope | Controls (ms) | Paired (ms) | Delta |
| --- | ---: | ---: | ---: |
| Auxiliary shards 1 and 2 | 2.568832 / 2.572832 | 2.573824 | +2.992 us |
| Critical shard 2 only | 2.572832 / 2.569152 | 2.572832 | +1.840 us |

The marker run `20260808T040245Z-110233` showed a genuine local improvement:
the slow auxiliary prefix became 0.5--2.0 us earlier and its shard-SiLU
frontier improved about 1.2 us. The complete layer frontier was unchanged
(84.416 versus 84.288 us control), because the denser paired weight issue and
longer B-slot lifetime delayed concurrent main/down traffic. This is operand
interference, not missing CUDA/TMEM compute throughput, so adding a helper
warp to the paired task would not address the measured limit. All proof code
was removed. The restored 11-op image passed all 20 runtime tests and four
exact resident steps in `20260808T041134Z-116929`; its fresh 501-sample S128
internal median was 2.569664 ms in `20260808T041219Z-117575`.

### Rejected two-phase next-layer RMS handoff

A dependency-changing RMS proof split the 4,096-element hidden row into two
ordinary K2048 memory-op loads. Down-projection stores published independent
low/high M-range barriers; one 128-thread RMS task cached the low half in
registers, used a compute-group barrier before releasing that shared slot,
then consumed the high half and preserved the production BF16 accumulation
order. It did not create partial/finalizer tasks or a global round trip. The
12-op image remained at 96 registers, nine hardware barriers, a 96-byte stack,
and zero spills. Single-step tensor checks and token 24748 passed in
`20260808T042924Z-132101`; four resident steps were exact in
`20260808T043048Z-133145`.

The layer group already uses 30 counters across 32 layers, so a 31st counter
overflowed the memory instruction's 10-bit barrier field in
`20260808T042354Z-127489`. Sharing head 7's Q and KV readiness counter
reclaimed one ID. That sharing alone was neutral: 2.574784 ms versus the
2.575456 ms production control in `20260808T042751Z-130619` and
`20260808T042712Z-130059`.

With RMS left on SMs 0--7, its 501-sample median was 2.574368 ms in
`20260808T042838Z-131504`, only 0.416 us faster than the shared-counter
control. Those owners still had late high-half down tasks ahead of RMS in
their compute streams. Moving RMS to SMs 136--143 let it enter after only the
early down work and wait for the high half inside the task, but measured
2.577216 ms in `20260808T043127Z-133802`, a 2.432 us regression versus that
control.

The marker pair `20260808T043338Z-135482` /
`20260808T043419Z-136106` confirms that the mechanism worked locally: the
next-RMS absolute completion moved from about 83.8 us to 82.1 us. It did not
move the converged boundary. Auxiliary Q-clear completion remained about
83.7--84.4 us in the control and 83.8--84.3 us in the staged form. A ready
memory operand therefore was not sufficient: compute-stream placement first
hid roughly 1.7 us, but a different compute/clear track still owned layer
progress. All split barriers, staged task/opcode, head-counter sharing, and
selectors were removed. Future overlap work must correlate compute issue,
both load VCore streams, slot lifetime, writeback, and synchronization on one
timeline before specializing another consumer.
The restored 11-op image passed all 20 runtime tests and four exact resident
steps in `20260808T044341Z-144054`; its fresh 501-sample S128 internal median
was 2.568672 ms in `20260808T044420Z-144879`.

The retained diagnostic support now supplies that shared view.  Build with
`make track_profile=1` and set `VDCORES_TRACK_PROFILE=1`; it reports per-SM
allocator slot stalls, both LDU queue/dependency tracks, compute M2C waits,
and store queue/service from reserved profile events.  A simultaneous stage
trace in `20260808T045710Z-155849` showed the final Q-clear cohort on
SM136--143 ending at 85.4--85.9 us after next RMS reached about 84.9 us, while
the other clear cohorts ended around 80--81 us.  This is a Q-lifetime and
placement candidate, not evidence that the store engine is saturated: median
store service was only 106.080 us over the full token in
`20260808T045415Z-153452`.  The profiling-free image remains unchanged at 96
registers, nine barriers, a 96-byte stack, and zero spills; its fresh S128
median was 2.575904 ms in `20260808T050051Z-159185`.

### Retained one-tile late Q cleanup

The clear-tail profile led to four mechanism screens. Removing clear made a
fresh token correct but regressed the median by about 5.9 us: the zero stores
were useful pacing for descriptor-slot reuse. Allowing one pending TMA-store
source-read group also regressed by 1.856 us, and moving clear behind phased
attention-output barriers regressed by roughly 125 us because it displaced
output/MLP work. A barrier-correct overwrite/reduce Q fold removed clear but
serialized Q writeback and measured 2.685408 ms versus a 2.580416 ms control.
All four proof paths were removed.

The retained change keeps the same 64 stores and barriers but places one
store on each CTA in SM88--151 instead of two or three on 24 auxiliary CTAs.
A same-image late24/late64/late24 comparison measured
2.580096/2.579488/2.582400 ms over 1,001 internal samples, a 1.760 us gain
against the control mean. The final 11-op image remains at 96 registers,
nine hardware barriers, a 96-byte stack, and zero spills. Full S128 tensor
validation passed in `20260808T060649Z-214439`, four resident tokens matched
`[24748, 24748, 24748, 24748]` in `20260808T060732Z-215233`, and the fresh
1,001-sample internal median was 2.572704 ms in
`20260808T060813Z-215760`.

### Rejected post-cleanup staged-RMS revisit

After the distributed clear stopped owning the layer boundary, the two-phase
down-to-next-RMS proof was rebuilt and extended to four phases. Both forms
kept the ordinary memory-op path, cached each published row range in
registers, and used compute-group barriers before returning shared slots. A
full S128 tensor check passed for two phases in `20260808T064051Z-243028` and
for four phases in `20260808T065037Z-250626`.

The first two-phase screen appeared 2.928 us faster, but a fresh exact-image
1,001-sample control/staged/control qualification measured
2.577472/2.577088/2.574880 ms in `20260808T070043Z-259638`,
`20260808T070127Z-260163`, and `20260808T070202Z-260745`: the staged form is
0.912 us slower than the control mean. Four phases measured 2.579136 ms
against 2.573440/2.573984 ms controls (`20260808T065252Z-252754`,
`20260808T065124Z-251323`, `20260808T065336Z-253360`). Moving the RMS owners
to SM128 or later also lost up to 9.216 us. Extra producer frontiers and
compute barriers therefore do not buy a stable boundary gain; all opcodes,
schedule hooks, barrier sharing, and selectors were removed.

### Rejected deeper Q-clean pipelines

The existing schedule already allocates a private Q buffer for every layer,
so two proofs used that lifetime directly. First, all 2,048 layer/tile stores
were batched onto SM128--151 beside the main-SM LM head. A new system barrier
counted completed stores and gated buffer reuse. Full S128 correctness passed
in `20260808T071429Z-270421`, but control/batch/control measured
2.572544/2.586240/2.572512 ms (`20260808T071513Z-270969`,
`20260808T071550Z-271454`, `20260808T071627Z-272083`): +13.712 us. Removing
the per-layer cleanup pulses lost useful allocator/load pacing, and the batch
left a token-end tail despite LM-head overlap.

Second, each layer cleared the preceding layer's rotated Q descriptor after
current Q/K/V projection. It waited on current pre-attention RMS readiness,
then ran one tile per CTA on SM88--151 concurrently with attention, preserving
the original store count and cadence without another barrier. Full S128
correctness passed in `20260808T071853Z-274287`; four resident tokens matched
`[24748] * 4` in `20260808T072416Z-278446`. Placement screens favored the
64-CTA SM88--151 form over 24 auxiliary CTAs, 48 tail CTAs, or SM64--127, but
its apparent 2.2 us gain did not survive the final minimal-code sandwich:
control/delayed/control was 2.571104/2.571904/2.572416 ms in
`20260808T072636Z-280421`, `20260808T072714Z-281051`, and
`20260808T072755Z-281318`, or +0.144 us versus the control mean. Both deep
pipelines and all selectors were removed.

### Rejected down-projection load-order changes

Two schedule-only proofs separated slot pressure from useful prefetch. An
allocator-level gate blocked each down task until its SiLU input barrier was
ready, preventing LDU0 weights from running ahead of the blocked LDU1
activation. Against a 2.573536 ms control, gating low-K work measured
2.587328 ms and gating high-K work measured 2.615104 ms in
`20260808T073211Z-285531`, `20260808T073247Z-286280`, and
`20260808T073324Z-286831`. The early weight stream is therefore productive
overlap, not removable queue waste.

The complementary proof kept that prefetch but sent the latter two of each
four weight tiles to LDU1 after the activation command. Low-only and high-only
forms were neutral at 2.573952/2.574400 ms; enabling both regressed to
2.578240 ms (`20260808T073514Z-288238`, `20260808T073551Z-289022`,
`20260808T073627Z-289737`). Both LDUs still feed one ordered M2C/UMMA stream,
and extra weight traffic on LDU1 interferes with the dependency-bearing
activation path. All gates, port hooks, and selectors were removed.

### Retained LM-head epoch-1 tail offload

The first 128-task LM-head epoch consistently leaves SM96--103 behind the
rest of the grid.  The second epoch formerly put another task on those same
CTAs while SM128--135 were idle, making the eight fused-argmax partial
releases part of the reducer's global-barrier tail.  The retained placement
moves only logical epoch-1 tasks 96--103 onto physical SM128--135.  It keeps
all 256 logical tasks, weight coordinates, partial-record indices, memory
operations, and barrier release counts unchanged; no opcode or synchronization
primitive is added.

A 1,001-sample control/offload/control/offload qualification measured
2.518560/2.516256/2.518976/2.516544 ms in
`20260808T141741Z-629432`.  The two controls average 2.518768 ms and the two
offload runs average 2.516400 ms, a stable 2.368 us schedule-only gain.  The
same-job marker proof in `20260808T142517Z-635403` moved the LM completion
frontier from 225.568 to 223.552 us and the eight-reducer completion frontier
from 240.192 to 226.784 us.  Marker perturbation exaggerates the end-to-end
effect, but confirms that earlier partial-barrier release—not a timestamp
artifact—is the mechanism.

The final selector-free schedule passed every S128 tensor threshold and exact
token 24748, then matched four resident steps
`[24748, 24748, 24748, 24748]` across KV128 in
`20260808T143044Z-640105`.  Its exact 11-op production image remains at 96
registers, nine hardware barriers, a 96-byte stack, 7,024 bytes of static
shared memory, and zero spills.  A fresh 1,001-sample internal-timer run in
`20260808T142953Z-639148` measured 2.518112 ms median at B8/S128.  This is
10.58% faster than the strict 2.816003 ms vLLM baseline and 16.291 us below
the 2.534403 ms threshold for a 10% win.

### Rejected native-152 single-wave LM head

A mechanism proof replaced the 256 four-group LM tasks on 128 logical owners
with one seven-group task on every physical SM.  It kept seven accumulators in
TMEM, emitted 152 partial records, and reduced those records directly.  The
uniform layout padded the 131,072-row projection to 136,192 rows, spending
3.9% more weight traffic to give every CTA the same critical-path work.  This
was intentionally the latency-balanced upper-bound test before building an
exact-work mixed six/seven-group layout.

The proof passed all S128 tensor thresholds and exact token 24748 in
`20260808T143746Z-646319`.  Its 11-op image still used 96 registers, nine
hardware barriers, a 96-byte stack, 7,024 bytes of static shared memory, and
zero spills, so occupancy or spilling did not cause the result.  Nevertheless,
its 1,001-sample internal median was 2.551200 ms in
`20260808T143827Z-647190`, 33.088 us slower than the retained 2.518112 ms.

The marker image in `20260808T144044Z-649223` localized the regression.  The
single seven-group LM wave took 149.536 us at p50 and 163.872 us at its tail;
the eight-reducer frontier reached 257.504 us.  A larger per-task operand set
reduces the number of activation loads, but also consumes more shared slots
per K step and loses the useful two-command prefetch/retirement cadence of the
retained four-group tasks.  An exact mixed layout would remove only the 3.9%
padding while retaining 112 seven-group critical tasks and adding imbalance,
so it cannot plausibly recover the measured 33 us loss.  Both proof opcodes,
the 152-record reducer, schedule selector, and manifests were removed.

### Rejected full auxiliary LM-tail expansion

The retained epoch-1 remap uses only SM128--135 because they are the measured
early auxiliary cohort.  A schedule-only upper bound also moved logical tasks
104--119 onto SM136--151, using all 24 auxiliary SMs without changing task
coordinates, partial records, or barrier counts.  In one same-GPU
8/24/8/24 sequence, the 301-sample fixed-eight medians were
2.516192/2.516224 ms and the full-auxiliary medians were
2.518912/2.520864 ms (`20260808T144745Z-655181`).  Full auxiliary ownership
therefore regressed 3.680 us against the fixed-eight mean.

This is also the mechanism bound for dynamic epoch-1 claiming.  The extra
CTAs can be early on the compute timeline while retaining late asynchronous
LDU/allocator history, so a compute-side atomic queue would select owners from
the wrong readiness signal.  A memory-VCore claimant would additionally need
to communicate dynamic weight, vocabulary, and partial-record coordinates to
the compute VCore.  That new handoff is not justified when simply exposing all
24 candidates already loses.  The count selector was removed and the minimal
fixed-eight remap retained.

### Rejected higher-ILP shared SwiGLU shard

The higher-resource task screen specialized the single-token K2048 shared
SwiGLU shard. Each compute thread loaded both of its 128-bit gate/up packs
before arithmetic and kept all 16 BF16 lanes live so the compiler could issue
their fast exponentials independently. It added no threads, memory traffic,
queue handoff, or barrier. The exact image remained at 96 registers, nine
hardware barriers, a 96-byte stack, 7,024 bytes of static shared memory, and
zero spills because another selected path already set the kernel-wide register
ceiling.

The mechanism worked in isolation. On one locked GB200, an in-image
baseline/ILP/baseline/ILP comparison over 2,001 internal samples measured
2.560/2.464/2.560/2.464 us for the complete 24-SM, three-shard B8 task in
`20260808T150135Z-665075`, a repeatable 96-ns or 3.75% improvement with zero
error. It did not improve the inference critical path. Two 1,001-sample ILP
runs measured 2.520832 and 2.520992 ms around a freshly rebuilt 2.519840-ms
control (`20260808T150234Z-665880`, `20260808T150459Z-667290`, and
`20260808T150713Z-668575`), a 1.072-us regression against the control.

The standalone launch makes the 24 identical SFU/shared-store programs the
whole frontier; in the resident schedule their more concentrated completion
burst competes with concurrent down-projection traffic. Shortening a task is
therefore insufficient when its issue shape worsens cross-track overlap. The
specialized path, build flag, and benchmark selector were removed; retain the
ordinary two-pack loop.

### Rejected batched nonissuer GEMV cursor repair

The issuer-owned M64 GEMV lets compute warp 0 dequeue operands, issue UMMA,
wait for completion, and release slots. The other three compute warps normally
walk the same K-tile loop only to advance their private M2C cursor and TMEM
phase before the four-warp epilogue fence. A mechanism proof replaced those
repeated branches and advances with one parity-preserving cursor update. It did
not change an operand, memory instruction, UMMA group, slot lifetime, fence, or
task result, and the exact image stayed at 96 registers, nine barriers, a
96-byte stack, 7,024 bytes of static shared memory, and zero spills.

Matched 2,001-sample task measurements showed that the skipped work was not
critical. M4096/K4096 measured 6.880 us for the control and 6.848 us for the
batched form (`20260808T152536Z-679039` and
`20260808T152238Z-677492`), only 32 ns apart. The complete balanced
M4096/K14336 down stage was exactly 20.832 us in both builds
(`20260808T152604Z-679427` and `20260808T152305Z-677761`). At S128, the
batched form measured 2.520064 ms versus 2.517824 ms for the freshly rebuilt
control (`20260808T152336Z-677878` and `20260808T152633Z-679742`), a
2.240-us regression.

Nonissuer cursor maintenance executes while the issuing warp is waiting on
the real operand/UMMA path, so removing it cannot shorten the task frontier.
Making the three warps reach the recurring compute rendezvous earlier instead
changes scheduler/barrier phase. The batch API, task branch, and build flag
were removed; retain the per-tile cursor walk.

### Rejected higher-register resident kernel

The exact single-token image reserves 219 KiB of dynamic shared memory, so it
is structurally limited to one resident CTA per SM. A kernel-wide resource
proof added `__launch_bounds__(256, 1)`, allowing ptxas to spend registers on
more aggressive scheduling without changing tasks or runtime instructions.
The resulting image used 150 registers instead of 96, with the same nine
barriers, 96-byte stack, 7,024 bytes of static shared memory, and zero spills.

Its 1,001-sample S128 internal median was 2.524576 ms in
`20260808T153011Z-681475`, 6.752 us slower than the freshly rebuilt
2.517824-ms control in `20260808T152633Z-679742`. An intermediate compiler
budget used `__launch_bounds__(256, 2)`: dynamic shared memory still admitted
only one physical CTA, while the launch-bound register heuristic produced a
126-register image. It measured 2.524960 ms in
`20260808T153419Z-683995`, 7.136 us slower than control.

NVIDIA documents that `--maxrregcount` is ignored for kernels carrying launch
bounds, so the two-block compiler hint was the supported way to probe the
intermediate allocation. Both higher-resource images regress by essentially
the same amount. The single resident CTA is not register-starved; compiler
freedom changes its global instruction/latency balance without shortening the
cross-track frontier. The launch-bound selector was removed and the 96-register
production image retained.

### Rejected next-RMS-owner down-tail offload

The final-layer marker trace put next-RMS owners SM0--7 near 81.8 us absolute
while several CTAs appeared compute-idle earlier. Two schedule-only proofs
moved their eight high-K down contributors without changing arithmetic,
memory instructions, or the 104 `bar_layer` releases. Sending them to
SM144--151 added a second high-K wave on those CTAs and measured 2.556800 ms
in `20260808T154326Z-689370`, 39.488 us slower than the nearby
2.517312-ms control in `20260808T154614Z-690599`.

The narrower proof used the physical SM96--103 hole left by the retained
down-tail offload: logical tasks 8--95 kept their owners, logical tasks 0--7
moved to SM96--103, and SM0--7 entered RMS immediately. Its 1,001-sample
internal median was 2.522784 ms in `20260808T154533Z-690336`, still 5.472 us
slower than control. A marker pass showed why. The relocated tasks made the
global down frontier arrive on SM96--103 around 80.6--80.9 us, and next RMS
completed near 83.2 us rather than the control's 81.44 us. Those CTAs were
compute-idle but retained unfavorable allocator/LDU history; placing another
task there delayed the dependency-bearing memory track. Both remaps and the
selector were removed.

### Rejected sparse additional LM-tail offload

After the retained eight-task epoch-1 offload, a detailed marker image put
logical tasks 53, 60, and 58 at the remaining LM completion tail. A sparse
schedule proof moved either task 53 alone or all three tasks to otherwise-idle
SM144 onward. This was the low-bandwidth complement to the rejected 16-task
auxiliary expansion; all logical coordinates, partial records, and the
256-release argmax barrier remained unchanged.

The 1,001-sample control/three-task/one-task/control sequence measured
2.517312/2.517952/2.517792/2.519872 ms in
`20260808T154614Z-690599`, `20260808T155247Z-694081`,
`20260808T155326Z-694441`, and `20260808T155416Z-694652`. Against the
2.518592-ms control mean, the apparent gains are only 0.640 and 0.800 us,
below neighboring run movement and much smaller than the qualified fixed-eight
gain. The profile-selected sparse mapping was removed; retain the simple
eight-task topology rather than encoding sub-microsecond tail noise.

### Rejected split-direction shared-slot allocation

The 27-slot exact image occupies 231,280 of the GB200's 232,448-byte opt-in
shared-memory limit, so another physical 8-KiB slot does not fit. A capacity
proof instead targeted fragmentation: one-slot activations and stores were
allocated from the high end while multi-slot GEMV weights retained low-end
first-fit. The arena, instruction stream, queues, tasks, and barriers were
unchanged, and the image stayed at 96 registers, nine barriers, a 96-byte
stack, 7,024 bytes of static shared memory, and zero spills.

The mechanism reduced median allocator stall from about 607 to 458 us in the
track image, but median compute M2C wait rose from about 720 to 946 us. Its
1,001-sample profiling-free median was 2.525664 ms in
`20260808T160016Z-697621`. Reversing the regions so one-slot operands kept
their low-address cadence and weights grew from the high end measured
2.527296 ms in `20260808T160512Z-700522`. Relative to the neighboring
2.518592-ms two-control mean, the two forms regress 7.072 and 8.704 us.

Contiguous availability alone is not the objective: the production first-fit
pattern also paces M2C publication and consumption. Both split-direction
selectors were removed. Future allocator work should preserve physical slot
order and change admission/order only when the consumer can use the operand.

### Rejected fused Up/SwiGLU gate-slot reuse

The fused tail's gate RegStore slot becomes payload-dead after the T2R load,
so three proofs reused it for the up/SwiGLU output instead of allocating a
second store slot. Copying the special-slot TMA descriptor into the gate slot
was correct but measured 2.523040 ms with a compute-group rendezvous and
2.522592 ms with the sufficient warp-local rendezvous, versus a fresh
2.519104-ms control. Directly passing separate descriptor/source slots to the
store queue was invalid because the unallocated descriptor could be
overwritten before delayed store service; job `20260808T163532Z-716775`
failed final-hidden/final-RMS thresholds.

A lifetime-correct memory-control-warp operator recorded the gate RegStore
mask and issued the output TMA descriptor against that occupied slot. It
passed full correctness and kept the exact 96-register, nine-barrier,
96-byte-stack, zero-spill image, but allocation-loop and control-op forms
measured 2.541248 and 2.542016 ms. Matched track runs showed why: median
allocator stall improved 553.312 to 494.080 us, but compute M2C wait worsened
926.624 to 944.992 us and LDU0/LDU1 queue-idle time increased by
25.696/35.808 us. The output allocation is useful admission pacing. All
proof code was removed; do not reuse a retained slot unless consumer
placement changes preserve the original publication cadence.

### Rejected cross-task gate/up activation retention

Gate and up tail projections share the same RMS activation.  A cross-task
proof retained gate's four K1024 activation groups—eight physical shared
slots—and reused them in the following up/SwiGLU task on the same CTA.  It
preserved the two operators and their compute order and passed full S128
tensor/token correctness (`20260808T170003Z-730493` and
`20260808T171454Z-738493`).

The matched track profile removed 128 LDU0 commands and M2C publications per
CTA and reduced median compute M2C wait from 924.352 to 900.736 us, but median
allocator stall increased from 544.512 to 630.400 us.  Partial retention was
worse because one to three retained groups required separate reload and reuse
RepeatM phases: the zero/one/two/three/four-group 301-sample medians were
2.525088/2.529952/2.539520/2.550656/2.524864 ms.

On an exact 11-op, 96-register image, full retention measured 2.525760 ms in
`20260808T171531Z-739081`, versus 2.522976 ms for same-image normal reloads in
`20260808T171616Z-739425` and 2.517504 ms for fresh production in
`20260808T165134Z-725676`.  The duplicate loads are hidden well enough that
their slot release and publication cadence are more valuable than the saved
traffic.  All retention code and selectors were removed.

### Retained 152-SM M128 output projection

The output projection now uses native M128K128-packed weights instead of the
shared M64K256 projection family.  It keeps exactly 152 independent tasks:
the first six M128 rows use eight K512 folds (48 tasks), while the remaining
26 rows use four K1024 folds (104 tasks).  The early K<2048 folds occupy
physical SM64--139 outside the attention owners; late folds occupy the
complementary SM0--63 and SM140--151 set.  Total arithmetic and the 152
partial reductions are unchanged, but each task consumes half as many B
activation bytes and gives TMEM/register epilogue work twice the M extent.

A task-count sweep validates that the gain is not merely fewer tasks.  The
301-sample internal medians for 128, 136, 144, and 152 tasks were 2.519712,
2.506848, 2.508512, and 2.506432 ms.  In the exact 12-op image, 1,001-sample
medians were 2.511072 ms for 136 tasks and 2.508832 ms for 152 tasks, versus
2.521664 ms for the M64 control (`20260808T174050Z-753404`,
`20260808T174133Z-753675`, and `20260808T174215Z-754048`).  Retaining all 152
owners therefore wins 12.832 us while preserving one logical output task per
physical SM.

The matched multi-track profile explains the improvement.  Relative to M64,
median compute M2C wait fell from 715.136 to 696.352 us, store service fell
from 115.296 to 113.504 us, and the median count of contended compute calls
fell from 40 to 30.  Allocator stall was effectively flat at
607.392/608.352 us; LDU1 dependency wait rose from 650.304 to 656.000 us.
Thus the useful mechanism is a shorter activation-to-UMMA/epilogue path, not
extra shared-slot capacity (`20260808T174449Z-755477` and
`20260808T174532Z-755968`).

The minimal default passed full S128 tensor validation and exact token 24748
in `20260808T175059Z-758790`.  Four resident decode steps crossing the KV128
boundary exactly matched `[24748, 24748, 24748, 24748]` in
`20260808T175136Z-759221`, and all 34 schedule/runtime host tests pass.  Its
fresh 1,001-sample profiling-free internal median is 2.507136 ms in
`20260808T175211Z-759358`, 10.968% faster than strict vLLM's 2.816003-ms S128
baseline and 27.267 us beyond the 10%-lead target.  The current selective
image has 12 compute opcodes and remains at 96 registers, nine barriers, a
96-byte stack, 7,024 bytes of static shared memory, and zero spills.

### Rejected M128 materialized-MLP prefix

An analogous M128 proof retiled the separately materialized 6,144-row gate
and up prefixes without fusing the operators.  Each projection retained its
96 tasks as 48 M128 tiles with two K2048 folds, preserving physical owners
and per-task weight work while halving B-activation bytes.  Because both
folds reduce concurrently, 16 auxiliary CTAs cleared one 12-KiB token row of
each output before attention and joined that completion with post-attention
RMS.  The looped clear stores must carry the resource-group flag; the first
ungrouped proof arrived only on layer 0's barrier and stalled at layer 1.

The corrected path passed every S128 tensor threshold and exact token 24748
in `20260808T181150Z-771165`.  Clear-placement medians for bases 104, 120,
128, and 136 were 2.514752, 2.512672, 2.510784, and 2.508192 ms over 301
samples.  The best strict 1,001-sample candidate pair averaged 2.513216 ms
(`20260808T181453Z-773045` and `20260808T181614Z-773686`), versus 2.510848 ms
for the intervening M64 control (`20260808T181535Z-773247`): a 2.368-us
regression.  A no-clear fold-1 form reduced each projection to 48 tasks but
lost system bandwidth parallelism and measured 3.306528 ms in
`20260808T181810Z-774738`.

Track profiles isolate the loss.  Relative to M64, fold-2 M128 reduced median
LDU0/LDU1 queue wait by 22.848/27.712 us, but compute M2C wait rose from
941.408 to 989.408 us, allocator stall from 550.656 to 561.216 us, store
service from 118.464 to 125.664 us, and store-barrier service from 104.864 to
117.632 us (`20260808T182032Z-776155` and
`20260808T182111Z-776347`).  Suppressing all clears left compute M2C wait at
987.776 us and the profiling kernel span unchanged in
`20260808T182210Z-776910`; per-layer independent buffers therefore cannot
repair this split-publication cadence.  All proof buffers, descriptors,
selectors, and schedules were removed; retain the M64 materialized prefix.

### Rejected M128 issuer-owned output specialization

The retained output schedule already uses M128 tiles, so a mechanism proof
compiled a dedicated issuer-owned M128N8 opcode analogous to the retained
M64N8 path.  It changed only which compute warp dequeued operands, issued
UMMA, and retired shared slots; tile coordinates, 152 physical owners,
arithmetic, stores, and reductions remained identical.  Full S128 tensor
validation and exact token 24748 passed in `20260808T184143Z-786922`.

The isolated output partitions did not get faster.  Over 2,001 internal
samples, the six-owner M768/K512 partition was exactly 3.520 us for both
generic and issuer-owned forms (`20260808T183933Z-785794` and
`20260808T184002Z-785978`).  The 26-owner M3328/K1024 partition was exactly
6.240 us for both (`20260808T184030Z-786420` and
`20260808T184056Z-786557`); issuer ownership also raised their average service
times by 20 and 15 ns, respectively.

There was a small resident-schedule effect, but it was below the threshold for
another selected opcode.  A strict 1,001-sample generic/issuer/generic bracket
measured 2.507392/2.506400/2.509088 ms
(`20260808T184220Z-787569`, `20260808T184302Z-788062`, and
`20260808T184359Z-788352`).  The issuer form is 1.840 us or 0.073% faster than
the two-control mean, with no isolated task gain.  That marginal queue-phase
shift does not justify expanding the minimal 12-op image to 13 opcodes.  The
specialization, selector, benchmark path, and manifest entries were removed.

### Rejected rebalanced 152-SM LM waves

The retained LM head runs two 128-task waves, with four M128 output groups per
task.  A schedule-only proof redistributed the same 256 tasks, padded rows,
partial records, and argmax reduction into 136+120, 144+112, and 152+104
waves.  It introduced no new opcode or larger task: the first wave simply
admitted more physical CTAs and the second wave used correspondingly fewer.

More concurrent weight streams made the first wave slower.  Against a
2.505568-ms 301-sample control (`20260808T185046Z-792430`), second-wave-base-0
medians were 2.509536 ms for 136+120, 2.506848 ms for 144+112, and 2.511808 ms
for 152+104 (`20260808T185329Z-793770`, `20260808T185407Z-794169`, and
`20260808T184957Z-791895`).  Moving the 104-task second wave to bases 24 and
48 worsened the full-152 form to 2.513664 and 2.514208 ms
(`20260808T185126Z-792645` and `20260808T185212Z-793142`).

Matched stage profiles explain the reversal.  The retained 128-task first
wave measured 84.032 us at p50 and 86.112 us at max, while the 144-task wave
rose to 88.192 and 92.288 us.  Its smaller second wave recovered only
2.464 us at p50 (63.168 versus 65.632 us), and the final LM frontier max moved
from 227.616 to 230.944 us (`20260808T185631Z-795604` and
`20260808T185713Z-795918`).  The 24 spare CTAs therefore do not provide free
LM throughput: extra simultaneous weight streams lengthen the shared
load/allocator path enough to dominate the reduced second wave.  All proof
weight slicing, placement selectors, and schedule changes were removed.

### Rejected high-resource fused-eight LM task

The next proof kept the retained 128 concurrent LM streams but fused each
CTA's two four-output tasks into one eight-output task.  This doubled live
TMEM accumulator groups, halved partial records, and removed one task
boundary.  A direct K512 load body needs one B plus 32 A memory steps, but the
resident memory-warp `RepeatM` window is physically encoded by 32 lanes and a
five-bit skip count.  The first executable form therefore used K256 B batches
and the existing 17-step repeat cadence.

That K256 form passed full S128 tensor validation and exact token 24748 in
`20260808T190807Z-802168`, and its 14-op image remained at 96 registers with
no spills.  It nevertheless measured 2.551392 ms versus 2.507072 ms for the
same-image two-task control over 301 samples (`20260808T190923Z-802975` and
`20260808T190846Z-802612`), a 44.320-us regression.  Wider per-K UMMA issue
therefore loses substantially more than the fused boundary saves.

A second form explicitly spent resources to preserve the proven four-output
cadence and K512 B traffic.  Phase 0 retained all eight K512 B slots while it
computed output groups 0--3; phase 1 revisited those slots for groups 4--7
before releasing them.  The fully unrolled path raised the kernel from 96 to
206 registers, with the same nine barriers, 96-byte stack, 7,024-byte static
shared allocation, and zero spills.  Since the 219-KiB dynamic allocation
already limits the kernel to one CTA/SM, this was a pure higher-resource task
test rather than an occupancy tradeoff.  It also passed full S128 validation
and exact token 24748 in `20260808T191328Z-805075`.

Long B-slot lifetime overwhelmed the saved traffic.  In the same 206-register
image, the retained-B form measured 2.680032 ms versus 2.513024 ms for the
ordinary control (`20260808T191443Z-805473` and
`20260808T191406Z-805225`), a 167.008-us regression.  Matched track profiles
showed median compute M2C wait improve by 24.672 us and total allocator/load
commands fall by nine, but allocator stall rose 148.704 us, stall events rose
from 358 to 562, and retries from 3,533 to 4,508.  Median LDU0/LDU1 queue wait
rose by 118.144/157.920 us and store-queue wait by 262.176 us
(`20260808T191709Z-806782` and `20260808T191751Z-807107`).  Eight retained
8-KiB slots leave too little of the 27-slot arena for four-slot A tiles, so
the local reuse win empties the global load/compute pipeline.  Both fused
opcodes, the 128-record reducer, schedule paths, and selectors were removed.

### Retained head-pair attention-to-output readiness

The M128 output schedule consumes attention results in K-aligned head groups.
Its prior readiness split was heads `(0)`, `(1)`, and `(2--7)`, so the K1024
partition for heads 2--3 unnecessarily waited for heads 4--7.  The default now
publishes four groups, `(0)`, `(1)`, `(2--3)`, and `(4--7)`.  This is only a
finer cross-stage dependency: output arithmetic, task placement, stores, and
the 12-op compute image are unchanged.  To fit the extra group in the 10-bit
logical-barrier encoding, head 7's Q and KV producers share one cumulative
readiness counter; full single-step and looped correctness qualify the alias.

Matched stage images show the intended overlap.  The absolute completion max
of the heads-2/3 K1024 output partition moved from 25.856 to 24.992 us, and the
full output frontier moved from 27.648 to 27.008 us
(`20260808T193256Z-815721` and `20260808T193337Z-816288`).  A profiling-free
1,001-sample control/candidate/control bracket measured
2.510112/2.507168/2.510240 ms (`20260808T192703Z-812296`,
`20260808T192755Z-812831`, and `20260808T193020Z-814290`), a 3.008-us gain
against the control mean without changing a task opcode.

The selector-free default passed every S128 tensor threshold and exact token
24748 in `20260808T193616Z-817287`; four resident decode steps crossing the
KV128 boundary exactly matched `[24748, 24748, 24748, 24748]` in
`20260808T193658Z-817742`.  All 34 schedule/runtime host tests pass.  Its fresh
1,001-sample internal median is 2.507712 ms in `20260808T193733Z-818093`,
10.948% faster than strict vLLM's 2.816003-ms S128 baseline and 26.691 us past
the 10%-lead target.  The exact production image remains at 12 compute
opcodes, 96 registers, nine barriers, a 96-byte stack, 7,024 bytes of static
shared memory, and zero spills.

### Retained final head-pair output group

The remaining `(4--7)` group still coupled the K1024 partition for heads 6--7
to heads 4--5.  Stage traces show that this is a real readiness gap: in the
four-group control, heads 6--7 completed attention at about 17.92--18.21 us
absolute while heads 4--5 extended to about 19.14 us
(`20260808T194445Z-822199`).  The default now uses five groups, `(0)`, `(1)`,
`(2--3)`, `(4--5)`, and `(6--7)`.  Heads 6 and 7 reuse their Q counters for
KV readiness, keeping the schedule within the same logical-barrier field and
leaving all output tasks and physical owners unchanged.

An isolation run distinguishes finer readiness from the extra counter alias.
Sharing head 6's counter while retaining four output groups measured
2.508256 ms, whereas the five-group form measured 2.499072 ms over 301
samples (`20260808T194818Z-824213` and `20260808T194856Z-824615`).  The
strict 1,001-sample qualification measured 2.499040 ms between 2.507712- and
2.507648-ms four-group controls (`20260808T194153Z-820595` and
`20260808T194231Z-820845`, with the first control from
`20260808T193733Z-818093`), an 8.640-us gain against their mean.  The useful
mechanism is therefore the cross-stage dependency split, not reducing the
logical-barrier count.

The selector-free five-group default passed full S128 tensor/token validation
in `20260808T194946Z-825046` and four resident tokens exactly matched
`[24748, 24748, 24748, 24748]` across KV128 in
`20260808T195020Z-825344`.  All 34 schedule/runtime host tests pass.  Its
fresh 1,001-sample internal median is 2.499584 ms in
`20260808T195056Z-825783`, 11.236% faster than strict vLLM's 2.816003-ms S128
baseline and 34.819 us past the 10%-lead target.  The exact image remains at
12 compute opcodes, 96 registers, nine barriers, a 96-byte stack, 7,024 bytes
of static shared memory, and zero spills.

### Rejected final-down-wave spare-SM remaps

A combined stage/track trace re-audited the final 104-contributor high-K down
wave after the five output-readiness groups.  Several physical CTAs outside
that wave finish their compute programs earlier, but the full per-SM tracks
in `20260808T195613Z-828350` distinguish three very different histories:
SM96--103 have high M2C and allocator pressure, SM136--143 have low store
queue time but high LDU1 dependency time, and SM104--119 are intermediate.
The diagnostic now accepts `VDCORES_TRACK_PROFILE_DETAIL=name[,name...]` to
print those complete per-SM tracks.  The `down_high1` stage marker was also
corrected to cover its actual SM0--95 and SM128--135 owners rather than the
contiguous-but-wrong SM0--103 approximation.

Two schedule proofs preserved all arithmetic, 104 releases, and logical tile
coordinates.  Moving logical tasks 16--23 to SM136--143 measured 2.634016 ms,
and moving tasks 16--31 to SM104--119 measured 2.580448 ms, versus a
2.495040-ms control over 301 internal samples
(`20260808T200013Z-830512`, `20260808T200056Z-831122`, and
`20260808T195933Z-830348`).  An explicit physical-owner map then expressed
each remap inside one unchanged logical task wave, ruling out extra schedule
boundaries: the two forms still measured 2.641024 and 2.583584 ms
(`20260808T200244Z-831988` and `20260808T200326Z-832254`).

The matched track-only control/mid104 pair explains the 95.392-us profiled
span increase (`20260808T200534Z-833460` and
`20260808T200613Z-833764`).  Median compute M2C wait rose 96.736 us, allocator
stall 99.872 us, LDU0 queue wait 80.288 us, LDU1 dependency wait 99.104 us,
and store queue wait 99.264 us.  The relocated CTAs are free only on their
compute stream; their pending load history delays the layer barrier and that
delay propagates to every track in the next layer.  Both remaps, their
selectors, and the unused explicit-map API were removed.  The restored exact
production image remains 12-op/96-register/spill-free, all 34 host tests pass,
and its fresh 301-sample median is 2.498944 ms in
`20260808T200908Z-835733`.

### Retained M128 fused-Q/RoPE task

The Q projection now spends the same 128 physical owners on wider Blackwell
tasks.  Each query head changes from two K2048 folds with eight M64 tasks per
fold to four K1024 folds with four M128 tasks per fold.  Arithmetic and the
16 readiness releases per head are unchanged.  The physical owner set also
stays SM0--103 plus SM128--151, so the proof changes task shape and activation
traffic without conflating the already-qualified auxiliary placement.  Each
task keeps the same M*K weight work but loads half as much RMS activation and
uses the native 32-datapath M128 TMEM-to-register epilogue.

An isolated full-Q wave at 128 SMs measured 7.424/6.880/7.392 us for the
M64/M128/M64 2,001-sample bracket (`20260808T201738Z-840787`,
`20260808T201803Z-841020`, and `20260808T201830Z-841482`).  The M128 task is
7.1--7.3% faster while remaining within the existing 128 compute threads and
shared-slot protocol.  In the full 13-op image, a profiling-free S128
control/candidate/control bracket measured 2.498880/2.480064/2.498816 ms
(`20260808T202338Z-844289`, `20260808T202415Z-844615`, and
`20260808T202453Z-845019`), a retained 18.784-us/token improvement.

Matched stage profiles show why the task result survives composition.  Q
completion p50/max moved from 9.920/12.640 to 8.256/11.808 us, and the
per-head attention frontier max moved from 17.632 to 15.168 us
(`20260808T202719Z-846179` and `20260808T202756Z-846500`).  Smaller Q
activation requests therefore advance K/V/attention through the shared load
queues instead of merely shortening an isolated compute interval.

Full S128 tensor validation and exact token 24748 passed in the final
production image in `20260808T203304Z-849354`; four resident decode steps
across KV128 exactly
matched `[24748, 24748, 24748, 24748]` in `20260808T202839Z-847040`.  All 34
host tests pass.  The production kernel remains at 96 registers, nine
barriers, a 96-byte stack, 7,024 bytes of static shared memory, and zero
spills.  The M64 Q selector and temporary probe manifest were removed; M64
fused RoPE remains only because K projection still uses that task.  A fresh
selector-free 1,001-sample internal median is 2.474624 ms in
`20260808T203214Z-849046`, 12.123% faster than strict vLLM's 2.816003-ms S128
baseline and 59.779 us beyond the 10%-lead target.

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
