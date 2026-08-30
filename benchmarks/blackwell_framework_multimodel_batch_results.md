# Blackwell VDCores, vLLM, and SGLang multimodel baselines

Measured on one NVIDIA GB200 on worker `10.0.16.34` on 2026-08-30. Per the
accepted reporting scope, this comparison retains only batch sizes 1, 2, 4,
and 8.

## Measurement contract

- Frameworks: vLLM `0.27.1` and SGLang `0.5.12.post1`.
- Each request has 127 copies of token 791 and generates two tokens. The
  reported first-to-second-token interval is one decode step attending to
  exactly 128 KV tokens.
- Each row uses a fresh engine process, three unmeasured warmups, and 21
  measured requests. Model load, prefill, JIT, graph capture, and kernel
  tuning are outside the retained interval.
- A batch sample is the slowest request's interval. All retained rows ran
  serially, with no other accepted framework measurement active on the
  worker.
- Models and KV state use BF16. vLLM uses `kv_cache_dtype=auto`; SGLang uses
  its native dense-model BF16 path. No DeepSeek-specific option or shim is
  enabled.
- At the user's direction, the VDCores column below reuses the previously
  accepted resident decode-step data rather than rerunning it. Those values
  used three warmups and 20 samples, and report the internal device interval.
  The Llama and Qwen3-1.7B rows are position-0 (C1); Qwen3-8B has one prefill
  token (C2). They are included for visibility but are not context-matched to
  the C128 framework columns.

vLLM records engine-core token timestamps, while SGLang derives its interval
from tokenizer-manager completion timing. Both isolate the same single decode
iteration, but the framework/IPC boundary is not identical. The values below
should therefore be read as framework-observed latency, not pure kernel time.

## Median decode latency

`VDCores existing` is the earlier unmatched-context measurement described
above. `SGLang/vLLM` is the ratio of the two C128 framework medians; values
above one favor vLLM for this framework-level measurement.

| Model | Batch | VDCores existing (ms) | vLLM C128 (ms) | SGLang C128 (ms) | SGLang/vLLM |
|:--|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 2.501280 | 2.801810 | 3.187549 | 1.138x |
| Llama-3.1-8B | 2 | 2.504928 | 2.736208 | 3.366307 | 1.230x |
| Llama-3.1-8B | 4 | 2.507152 | 2.830291 | 3.389763 | 1.198x |
| Llama-3.1-8B | 8 | 2.492736 | 2.843123 | 3.373474 | 1.187x |
| Llama-3.2-1B | 1 | 0.696768 | 0.318281 | 1.730291 | 5.436x |
| Llama-3.2-1B | 2 | 0.701024 | 0.332170 | 1.770964 | 5.331x |
| Llama-3.2-1B | 4 | 0.704592 | 0.351210 | 1.683249 | 4.793x |
| Llama-3.2-1B | 8 | 0.701264 | 0.369546 | 1.754931 | 4.749x |
| Qwen3-8B | 1 | 2.961696 | 3.108603 | 3.473574 | 1.117x |
| Qwen3-8B | 2 | 3.001520 | 3.125051 | 3.656299 | 1.170x |
| Qwen3-8B | 4 | 2.958224 | 3.174845 | 3.656171 | 1.152x |
| Qwen3-8B | 8 | 2.956224 | 3.093018 | 3.666604 | 1.185x |
| Qwen3-1.7B | 1 | 1.397280 | 0.957885 | 1.486380 | 1.552x |
| Qwen3-1.7B | 2 | 1.414608 | 1.030142 | 1.609935 | 1.563x |
| Qwen3-1.7B | 4 | 1.392064 | 1.107104 | 1.690033 | 1.527x |
| Qwen3-1.7B | 8 | 1.428736 | 0.998845 | 1.762260 | 1.764x |

## B8 aggregate throughput

Aggregate throughput is `batch * 1000 / median_ms`.

| Model | VDCores existing (req/s) | vLLM C128 (req/s) | SGLang C128 (req/s) |
|:--|--:|--:|--:|
| Llama-3.1-8B | 3,209.3 | 2,813.81 | 2,371.44 |
| Llama-3.2-1B | 11,408.0 | 21,648.18 | 4,558.58 |
| Qwen3-8B | 2,706.15 | 2,586.47 | 2,181.86 |
| Qwen3-1.7B | 5,599.35 | 8,009.25 | 4,539.63 |

The requested small-model aliases correspond to the available
`Llama-3.2-1B-Instruct` and `Qwen3-1.7B` checkpoints. Detailed distributions,
job identifiers, environment versions, and checkpoint paths are recorded in:

- [vLLM results](blackwell_vllm_multimodel_batch_results.md)
- [SGLang results](blackwell_sglang_multimodel_batch_results.md)
- [existing VDCores results](blackwell_multimodel_batch_results.md)

Because the reused VDCores rows have a different context and timing boundary,
this table does not make a direct VDCores/framework speedup claim.
