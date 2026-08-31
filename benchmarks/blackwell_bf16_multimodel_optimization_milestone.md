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

Lower is better.  Every row is now context-matched at C128: 127 cached prefix
tokens plus one current decode token.  VDCores uses three unmeasured warmups
and 20 measured resident launches and reports the median internal device
interval.  The vLLM and SGLang columns reuse the previously accepted BF16 C128
framework results unchanged; neither framework was rerun for this correction.
Framework timing includes its serving/IPC observation boundary, so the table
is context- and dtype-matched but not a pure kernel-to-kernel comparison.

| Model | Batch | VDCores BF16 C128 (ms) | vLLM BF16 C128 (ms) | SGLang BF16 C128 (ms) |
|:--|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 2.488144 | 2.801810 | 3.187549 |
| Llama-3.1-8B | 2 | 2.494496 | 2.736208 | 3.366307 |
| Llama-3.1-8B | 4 | 2.498176 | 2.830291 | 3.389763 |
| Llama-3.1-8B | 8 | 2.495088 | 2.843123 | 3.373474 |
| Llama-3.2-1B | 1 | 0.694256 | 0.318281 | 1.730291 |
| Llama-3.2-1B | 2 | 0.699200 | 0.332170 | 1.770964 |
| Llama-3.2-1B | 4 | 0.702048 | 0.351210 | 1.683249 |
| Llama-3.2-1B | 8 | 0.706112 | 0.369546 | 1.754931 |
| Qwen3-8B | 1 | 2.835808 | 3.108603 | 3.473574 |
| Qwen3-8B | 2 | 2.827728 | 3.125051 | 3.656299 |
| Qwen3-8B | 4 | 2.846864 | 3.174845 | 3.656171 |
| Qwen3-8B | 8 | 2.839152 | 3.093018 | 3.666604 |
| Qwen3-1.7B | 1 | 1.364928 | 0.957885 | 1.486380 |
| Qwen3-1.7B | 2 | 1.365984 | 1.030142 | 1.609935 |
| Qwen3-1.7B | 4 | 1.361584 | 1.107104 | 1.690033 |
| Qwen3-1.7B | 8 | 1.387152 [^qwen17-c128] | 0.998845 | 1.762260 |

[^qwen17-c128]: The C128 B8 run returned the exact reference token on every
    checked launch, but its repeated tensor checks were numerically marginal:
    the worst observations were 5.228% for one logits shard and 5.239% for
    fused SwiGLU.  The B8 value is retained as performance evidence, not as a
    strict below-5% correctness qualification.

## Improvement against each VDCores lane's retained control

| Model | Compared points | Retained change |
|:--|:--|--:|
| Llama-3.1-8B | S128 unchanged copy 3.109568 -> 2.484224 ms | -20.11% |
| Llama-3.2-1B | B1 0.696768 -> 0.678144 ms; B8 0.701264 -> 0.684832 ms | -2.67%; -2.34% |
| Qwen3-8B | B1 2.961696 -> 2.842656 ms; B8 2.956224 -> 2.830880 ms | -4.02%; -4.24% |
| Qwen3-1.7B | B1 1.343376 -> 1.153392 ms; B8 1.362992 -> 1.165600 ms | -14.14%; -14.48% |

## Correctness gates

Mean-relative tensor error below 5% is the acceptance criterion.  The retained
jobs produced the following C128 evidence:

| Model | Accepted evidence |
|:--|:--|
| Llama-3.1-8B | C128 B8 all-row checks passed; worst reported tensor error 3.573% |
| Llama-3.2-1B | C128 B8 passed; worst reported tensor error 3.104%, exact token 315 on every row |
| Qwen3-8B | C128 B8 passed; worst reported tensor error 4.183%, exact token 198 on every row |
| Qwen3-1.7B | C128 B1 passed with worst 4.188% and exact token 198; C128 B8 exact-token checks passed but repeated tensor maxima reached 5.239% |

Exact commands, job identifiers, distributions, and build profiles are in:

- [Llama-3.1-8B results](blackwell_llama3_8b_results.md)
- [Llama-3.2-1B results](blackwell_llama32_1b_results.md)
- [Qwen3-8B results](blackwell_qwen3_8b_results.md)
- [Qwen3-1.7B results](blackwell_qwen3_1p7b_results.md)
- [vLLM BF16 C128 baselines](blackwell_vllm_multimodel_batch_results.md)
- [SGLang BF16 C128 baselines](blackwell_sglang_multimodel_batch_results.md)
