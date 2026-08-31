# Blackwell BF16 multimodel optimization milestone

This milestone closes the 2026-08-30 through 2026-08-31 exploration on one
NVIDIA GB200 GPU on worker `10.0.16.34`.  It integrates the best fully qualified
BF16 implementations for the four requested model lanes.  The integration
adds no FP8 model path, new opcode, allocator publication exception, or
unqualified experiment.

The requested small-model aliases map to the checkpoints available on the
cluster:

- `llama3-1b` is `unsloth/Llama-3.2-1B-Instruct`.
- `qwen3-1b` is `Qwen/Qwen3-1.7B`.

## Retained implementations

| Model | Retained source milestone | Material retained change |
|:--|:--|:--|
| Llama-3.1-8B | `27f8c41`, batch follow-on through `f7777c1` | Lean namespaced Blackwell runtime, logical B1--B8, seq-major KV cache |
| Llama-3.2-1B | `89df3ee` | Issuer-only up/SwiGLU and LM-head handoffs, shard-local down readiness |
| Qwen3-8B | `720c9d7` through `584c3bf` | Q-buffer correctness, M128 LM head, direct attention output, phased down, balanced down owners |
| Qwen3-1.7B | `8971e9e`, documented by `cbbc912` | Register-forwarded full gate/up SwiGLU with split down readiness |

The Llama-3.2-1B branch's later generic MXFP8 commits are intentionally not
part of this milestone.  The dirty Qwen3-8B Group4-prefix experiment is also
excluded because it had not completed full correctness and matched timing at
the freeze point.

## Median decode latency

Lower is better.  This table reuses existing measurements; no closure run was
performed.  The framework columns are BF16 decode at exactly C128.  The
VDCores column reports its actual measured context and internal device timing:
the logical-batch sweeps are C1 for the Llama and small-Qwen lanes and C2 for
Qwen3-8B.  These timing boundaries and contexts are not directly comparable,
so the table deliberately makes no VDCores/framework speedup claim.

| Model | Batch | VDCores context | VDCores existing (ms) | vLLM BF16 C128 (ms) | SGLang BF16 C128 (ms) |
|:--|--:|:--:|--:|--:|--:|
| Llama-3.1-8B | 1 | C1 | 2.501280 | 2.801810 | 3.187549 |
| Llama-3.1-8B | 2 | C1 | 2.504928 | 2.736208 | 3.366307 |
| Llama-3.1-8B | 4 | C1 | 2.507152 | 2.830291 | 3.389763 |
| Llama-3.1-8B | 8 | C1 | 2.492736 | 2.843123 | 3.373474 |
| Llama-3.2-1B | 1 | C1 | 0.678144 | 0.318281 | 1.730291 |
| Llama-3.2-1B | 2 | C1 | 0.679616 | 0.332170 | 1.770964 |
| Llama-3.2-1B | 4 | C1 | 0.681808 | 0.351210 | 1.683249 |
| Llama-3.2-1B | 8 | C1 | 0.684832 | 0.369546 | 1.754931 |
| Qwen3-8B | 1 | C2 | 2.842656 | 3.108603 | 3.473574 |
| Qwen3-8B | 2 | C2, prior port | 3.001520 [^qwen-prior] | 3.125051 | 3.656299 |
| Qwen3-8B | 4 | C2, prior port | 2.958224 [^qwen-prior] | 3.174845 | 3.656171 |
| Qwen3-8B | 8 | C2 | 2.830880 | 3.093018 | 3.666604 |
| Qwen3-1.7B | 1 | C1 | 1.153392 | 0.957885 | 1.486380 |
| Qwen3-1.7B | 2 | C1 | 1.153040 | 1.030142 | 1.609935 |
| Qwen3-1.7B | 4 | C1 | 1.154176 | 1.107104 | 1.690033 |
| Qwen3-1.7B | 8 | C1 | 1.165600 | 0.998845 | 1.762260 |

[^qwen-prior]: The final Qwen3-8B head `584c3bf` has accepted B1 and B8
    timings only.  B2 and B4 are the existing one-prefill measurements from
    the earlier accepted port at `e2494e7`; they are retained for table
    completeness and are not presented as final-head measurements.

The original lean Llama-3.1-8B refinement also recorded a 2.484224-ms S128
internal median before logical batch controls were introduced.  Because that
run does not carry a logical B1/B2/B4/B8 label, it is not substituted into the
batch table.

## Improvement against each VDCores lane's retained control

| Model | Compared points | Retained change |
|:--|:--|--:|
| Llama-3.1-8B | S128 unchanged copy 3.109568 -> 2.484224 ms | -20.11% |
| Llama-3.2-1B | B1 0.696768 -> 0.678144 ms; B8 0.701264 -> 0.684832 ms | -2.67%; -2.34% |
| Qwen3-8B | B1 2.961696 -> 2.842656 ms; B8 2.956224 -> 2.830880 ms | -4.02%; -4.24% |
| Qwen3-1.7B | B1 1.343376 -> 1.153392 ms; B8 1.362992 -> 1.165600 ms | -14.14%; -14.48% |

## Correctness gates

Mean-relative tensor error below 5% is the acceptance criterion.  The retained
jobs passed that gate:

| Model | Accepted evidence |
|:--|:--|
| Llama-3.1-8B | B8 all-row checks below 3.304%; position-140 cache traversal below 2.494% |
| Llama-3.2-1B | B1/B8 worst 2.346%/2.913%; position-128 three-block cache check below 2.892% |
| Qwen3-8B | Final-head B1/B8 worst 2.397%/2.806%, token 422 on every live row |
| Qwen3-1.7B | B1/B8/position-65 suite worst 4.137% |

Exact commands, job identifiers, distributions, and build profiles are in:

- [Llama-3.1-8B results](blackwell_llama3_8b_results.md)
- [Llama-3.2-1B results](blackwell_llama32_1b_results.md)
- [Qwen3-8B results](blackwell_qwen3_8b_results.md)
- [Qwen3-1.7B results](blackwell_qwen3_1p7b_results.md)
- [vLLM BF16 C128 baselines](blackwell_vllm_multimodel_batch_results.md)
- [SGLang BF16 C128 baselines](blackwell_sglang_multimodel_batch_results.md)
