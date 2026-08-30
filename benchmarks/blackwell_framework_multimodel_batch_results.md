# Blackwell vLLM and SGLang multimodel baselines

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

vLLM records engine-core token timestamps, while SGLang derives its interval
from tokenizer-manager completion timing. Both isolate the same single decode
iteration, but the framework/IPC boundary is not identical. The values below
should therefore be read as framework-observed latency, not pure kernel time.

## Median decode latency

`SGLang/vLLM` is the ratio of the two median latencies; values above one favor
vLLM for this framework-level measurement.

| Model | Batch | vLLM (ms) | SGLang (ms) | SGLang/vLLM |
|:--|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 2.801810 | 3.187549 | 1.138x |
| Llama-3.1-8B | 2 | 2.736208 | 3.366307 | 1.230x |
| Llama-3.1-8B | 4 | 2.830291 | 3.389763 | 1.198x |
| Llama-3.1-8B | 8 | 2.843123 | 3.373474 | 1.187x |
| Llama-3.2-1B | 1 | 0.318281 | 1.730291 | 5.436x |
| Llama-3.2-1B | 2 | 0.332170 | 1.770964 | 5.331x |
| Llama-3.2-1B | 4 | 0.351210 | 1.683249 | 4.793x |
| Llama-3.2-1B | 8 | 0.369546 | 1.754931 | 4.749x |
| Qwen3-8B | 1 | 3.108603 | 3.473574 | 1.117x |
| Qwen3-8B | 2 | 3.125051 | 3.656299 | 1.170x |
| Qwen3-8B | 4 | 3.174845 | 3.656171 | 1.152x |
| Qwen3-8B | 8 | 3.093018 | 3.666604 | 1.185x |
| Qwen3-1.7B | 1 | 0.957885 | 1.486380 | 1.552x |
| Qwen3-1.7B | 2 | 1.030142 | 1.609935 | 1.563x |
| Qwen3-1.7B | 4 | 1.107104 | 1.690033 | 1.527x |
| Qwen3-1.7B | 8 | 0.998845 | 1.762260 | 1.764x |

## B8 aggregate throughput

Aggregate throughput is `batch * 1000 / median_ms`.

| Model | vLLM (req/s) | SGLang (req/s) |
|:--|--:|--:|
| Llama-3.1-8B | 2,813.81 | 2,371.44 |
| Llama-3.2-1B | 21,648.18 | 4,558.58 |
| Qwen3-8B | 2,586.47 | 2,181.86 |
| Qwen3-1.7B | 8,009.25 | 4,539.63 |

The requested small-model aliases correspond to the available
`Llama-3.2-1B-Instruct` and `Qwen3-1.7B` checkpoints. Detailed distributions,
job identifiers, environment versions, and checkpoint paths are recorded in:

- [vLLM results](blackwell_vllm_multimodel_batch_results.md)
- [SGLang results](blackwell_sglang_multimodel_batch_results.md)

The earlier VDCores table measures a different position-0 workload. It is not
mixed into this C128 comparison; a direct VDCores/framework speedup claim
requires a matching C128 VDCores timing sweep.
