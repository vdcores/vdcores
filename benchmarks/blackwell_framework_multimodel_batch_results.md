# Blackwell VDCores, vLLM, and SGLang multimodel baselines

> The optimized BF16 milestone and its updated VDCores rows are recorded in
> [Blackwell BF16 multimodel optimization milestone](blackwell_bf16_multimodel_optimization_milestone.md).
> The vLLM and SGLang values below preserve the original 2026-08-30
> framework-baseline aggregation.  Only the VDCores column was updated with
> the context-matched C128 sweep on 2026-08-31.

The framework rows were measured on one NVIDIA GB200 on worker `10.0.16.34`
on 2026-08-30; the VDCores C128 correction used the same worker on 2026-08-31.
Per the accepted reporting scope, this comparison retains only batch sizes 1,
2, 4, and 8.

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
- VDCores uses the same C128 shape (127 cached prefix rows plus the current
  decode row), three unmeasured warmups, and 20 measured resident launches.
  It reports the median internal device interval. No framework was rerun when
  these VDCores rows were added.

vLLM records engine-core token timestamps, while SGLang derives its interval
from tokenizer-manager completion timing. Both isolate the same single decode
iteration, but the framework/IPC boundary is not identical. The values below
should therefore be read as framework-observed latency, not pure kernel time.

## Median decode latency

`SGLang/vLLM` is the ratio of the two C128 framework medians; values above one
favor vLLM for this framework-level measurement.

| Model | Batch | VDCores C128 (ms) | vLLM C128 (ms) | SGLang C128 (ms) | SGLang/vLLM |
|:--|--:|--:|--:|--:|--:|
| Llama-3.1-8B | 1 | 2.488144 | 2.801810 | 3.187549 | 1.138x |
| Llama-3.1-8B | 2 | 2.494496 | 2.736208 | 3.366307 | 1.230x |
| Llama-3.1-8B | 4 | 2.498176 | 2.830291 | 3.389763 | 1.198x |
| Llama-3.1-8B | 8 | 2.495088 | 2.843123 | 3.373474 | 1.187x |
| Llama-3.2-1B | 1 | 0.694256 | 0.318281 | 1.730291 | 5.436x |
| Llama-3.2-1B | 2 | 0.699200 | 0.332170 | 1.770964 | 5.331x |
| Llama-3.2-1B | 4 | 0.702048 | 0.351210 | 1.683249 | 4.793x |
| Llama-3.2-1B | 8 | 0.706112 | 0.369546 | 1.754931 | 4.749x |
| Qwen3-8B | 1 | 2.835808 | 3.108603 | 3.473574 | 1.117x |
| Qwen3-8B | 2 | 2.827728 | 3.125051 | 3.656299 | 1.170x |
| Qwen3-8B | 4 | 2.846864 | 3.174845 | 3.656171 | 1.152x |
| Qwen3-8B | 8 | 2.839152 | 3.093018 | 3.666604 | 1.185x |
| Qwen3-1.7B | 1 | 1.364928 | 0.957885 | 1.486380 | 1.552x |
| Qwen3-1.7B | 2 | 1.365984 | 1.030142 | 1.609935 | 1.563x |
| Qwen3-1.7B | 4 | 1.361584 | 1.107104 | 1.690033 | 1.527x |
| Qwen3-1.7B | 8 | 1.387152 [^qwen17-b8] | 0.998845 | 1.762260 | 1.764x |

[^qwen17-b8]: Exact-token checks passed at C128 B8, but repeated tensor checks
    were marginal, with a worst observed mean-relative error of 5.239%. See
    the optimized milestone for the qualification boundary.

## B8 aggregate throughput

Aggregate throughput is `batch * 1000 / median_ms`.

| Model | VDCores C128 (req/s) | vLLM C128 (req/s) | SGLang C128 (req/s) |
|:--|--:|--:|--:|
| Llama-3.1-8B | 3,206.30 | 2,813.81 | 2,371.44 |
| Llama-3.2-1B | 11,329.65 | 21,648.18 | 4,558.58 |
| Qwen3-8B | 2,817.74 | 2,586.47 | 2,181.86 |
| Qwen3-1.7B | 5,767.21 | 8,009.25 | 4,539.63 |

The requested small-model aliases correspond to the available
`Llama-3.2-1B-Instruct` and `Qwen3-1.7B` checkpoints. Detailed distributions,
job identifiers, environment versions, and checkpoint paths are recorded in:

- [vLLM results](blackwell_vllm_multimodel_batch_results.md)
- [SGLang results](blackwell_sglang_multimodel_batch_results.md)
- [historical VDCores B1--B8 results](blackwell_multimodel_batch_results.md)
- [context-matched optimized milestone](blackwell_bf16_multimodel_optimization_milestone.md)

The contexts and BF16 dtype are matched. The VDCores internal device interval
and framework-observed serving intervals still have different timing
boundaries, so small percentage differences should not be read as pure-kernel
speedups.
