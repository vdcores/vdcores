# Blackwell vLLM fixed-context decode baselines

These results were collected on worker `10.0.16.34` (NVIDIA GB200, compute
capability 10.0) with vLLM 0.27.1. The worker-local environment was
`/mnt/checkpoints/envs/vllm-0.27.1`: Python 3.12.13, PyTorch 2.13.0+cu130,
CUDA 13.0, and FlashInfer 0.6.16.post3. MPI environment validation passed in
job `20260830T195033Z-1304490`.

## Method

Each row invokes `benchmarks/blackwell_fixed_context_decode.py` in a fresh
vLLM engine process, with no concurrent accepted framework measurement on the
worker. Every request has 127 fixed prompt tokens and produces two tokens. The
reported latency is the framework timestamp interval from the first to the
second generated token, so the measured decode step sees a context length of
128. Runs use BF16 weights, `kv_cache_dtype=auto`, strict full-batch prefill,
`gpu_memory_utilization=0.8`, three warmups, and 21 timed samples. Throughput is
`batch * 1000 / median_ms`. No DeepSeek-specific flags or compatibility shim
are used. Only requested batch sizes 1, 2, 4, and 8 are reported.

## Results

| Model | Batch | Median latency (ms) | Aggregate requests/s |
|:--|--:|--:|--:|
| Llama-3.2-1B-Instruct | 1 | 0.318281 | 3,141.88 |
| Llama-3.2-1B-Instruct | 2 | 0.332170 | 6,021.01 |
| Llama-3.2-1B-Instruct | 4 | 0.351210 | 11,389.20 |
| Llama-3.2-1B-Instruct | 8 | 0.369546 | 21,648.18 |
| Qwen3-1.7B | 1 | 0.957885 | 1,043.97 |
| Qwen3-1.7B | 2 | 1.030142 | 1,941.48 |
| Qwen3-1.7B | 4 | 1.107104 | 3,613.03 |
| Qwen3-1.7B | 8 | 0.998845 | 8,009.25 |
| Llama-3.1-8B-Instruct | 1 | 2.801810 | 356.91 |
| Llama-3.1-8B-Instruct | 2 | 2.736208 | 730.94 |
| Llama-3.1-8B-Instruct | 4 | 2.830291 | 1,413.28 |
| Llama-3.1-8B-Instruct | 8 | 2.843123 | 2,813.81 |
| Qwen3-8B | 1 | 3.108603 | 321.69 |
| Qwen3-8B | 2 | 3.125051 | 639.99 |
| Qwen3-8B | 4 | 3.174845 | 1,259.90 |
| Qwen3-8B | 8 | 3.093018 | 2,586.47 |

The smaller Qwen checkpoint available on the worker is Qwen3-1.7B, at
`/mnt/checkpoints/Qwen/Qwen3-1.7B`; it is the checkpoint used for the requested
Qwen3 small-model lane. The other checkpoints are
`/mnt/checkpoints/unsloth/Llama-3.2-1B-Instruct`,
`/mnt/checkpoints/unsloth/Meta-Llama-3.1-8B-Instruct`, and
`/mnt/checkpoints/Qwen/Qwen3-8B`.

## MPI jobs

| Model / batches | Job ID |
|:--|:--|
| Llama-3.2-1B-Instruct, B1/B2/B4/B8 | `20260830T203759Z-1592163` |
| Qwen3-1.7B, B1/B2/B4/B8 | `20260830T204523Z-1638486` |
| Llama-3.1-8B-Instruct, B1/B2 | `20260830T210018Z-1725917` |
| Llama-3.1-8B-Instruct, B4 | `20260830T210255Z-1740052` |
| Llama-3.1-8B-Instruct, B8 | `20260830T210431Z-1749213` |
| Qwen3-8B, B1 | `20260830T210558Z-1758281` |
| Qwen3-8B, B2 | `20260830T210737Z-1766986` |
| Qwen3-8B, B4 | `20260830T210921Z-1776855` |
| Qwen3-8B, B8 | `20260830T211104Z-1787432` |

Raw min/median/p90/max result lines and environment evidence are retained in
`.agentlog/2026-08-30-vllm-multimodel-batch.md`.
