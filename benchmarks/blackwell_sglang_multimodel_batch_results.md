# SGLang Blackwell multimodel batch results

Measured on worker `10.0.16.34` on 2026-08-30 with SGLang
`0.5.12.post1`. The four local BF16 checkpoints were:

- `/mnt/checkpoints/unsloth/Meta-Llama-3.1-8B-Instruct`
- `/mnt/checkpoints/unsloth/Llama-3.2-1B-Instruct`
- `/mnt/checkpoints/Qwen/Qwen3-8B`
- `/mnt/checkpoints/Qwen/Qwen3-1.7B`

Qwen3-1.7B is the locally available tier corresponding to the requested
Qwen3-1B evaluation.

## Method

Each reported batch row (B1, B2, B4, and B8) ran
`benchmarks/blackwell_fixed_context_decode.py` directly in a fresh engine
process. The prompt contained 127 copies of token 791 and the engine generated
two tokens, so the measured first-to-second-token interval is one decode step
over exactly 128 KV tokens. Each row used three unmeasured warmups and 21
measured samples. For a batch, the sample latency is the slowest request's
decode interval; aggregate throughput is `batch * 1000 / median_ms`.

The engine used BF16, FlashInfer attention, `moe_runner_backend=auto`, page
size 64, disabled radix cache, unchunked strict prefill, CUDA graphs through
the tested batch size, and `mem_fraction_static=0.8`. No DeepSeek-specific
flag or cache dtype was used. All accepted rows ran serially with no other
accepted framework measurement active on the worker.

## Results

| Model | Batch | Median (ms) | P90 (ms) | Aggregate req/s |
|:--|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 3.187549 | 3.203390 | 313.72 |
| Llama-3.1-8B | 2 | 3.366307 | 3.406244 | 594.12 |
| Llama-3.1-8B | 4 | 3.389763 | 3.414980 | 1,180.02 |
| Llama-3.1-8B | 8 | 3.373474 | 3.502758 | 2,371.44 |
| Llama-3.2-1B | 1 | 1.730291 | 1.802901 | 577.94 |
| Llama-3.2-1B | 2 | 1.770964 | 1.810485 | 1,129.33 |
| Llama-3.2-1B | 4 | 1.683249 | 1.770772 | 2,376.36 |
| Llama-3.2-1B | 8 | 1.754931 | 1.877591 | 4,558.58 |
| Qwen3-8B | 1 | 3.473574 | 3.503175 | 287.89 |
| Qwen3-8B | 2 | 3.656299 | 3.672491 | 547.00 |
| Qwen3-8B | 4 | 3.656171 | 3.694540 | 1,094.04 |
| Qwen3-8B | 8 | 3.666604 | 3.778735 | 2,181.86 |
| Qwen3-1.7B | 1 | 1.486380 | 1.508012 | 672.78 |
| Qwen3-1.7B | 2 | 1.609935 | 1.671153 | 1,242.29 |
| Qwen3-1.7B | 4 | 1.690033 | 1.727634 | 2,366.82 |
| Qwen3-1.7B | 8 | 1.762260 | 1.859894 | 4,539.63 |

## Jobs and environment

| Checkpoint | Accepted cluster job |
|:--|:--|
| Meta-Llama-3.1-8B-Instruct | `20260830T201648Z-1459836` |
| Llama-3.2-1B-Instruct | `20260830T195527Z-1333591` |
| Qwen3-8B | `20260830T202715Z-1529648` |
| Qwen3-1.7B | `20260830T200542Z-1394493` |

Every job used one cooperative MPI GPU allocation on `10.0.16.34`; its shell
ran batches sequentially and launched a new Python process for each row. All
16 reported B1/B2/B4/B8 rows emitted `FIXED_CONTEXT_RESULT` and exited zero.

The validated environment contained Python 3.12.3, PyTorch `2.11.0+cu130`
with CUDA 13.0, `sglang-kernel=0.4.2.post2`,
`flashinfer=0.6.11.post1`, and `flash-mla=1.0.0+15f13e5`. Exact launcher,
configuration, min/median/P90/max result lines, discarded smoke records, and
the FlashMLA transfer checksum are preserved under
`.agentlog/sglang-multimodel-20260830/accepted-results.log`.
