# Blackwell LLM batch 1-8 evaluation

Measured on 2026-08-30 on one GPU and one MPI rank on worker
`10.0.16.34`. The worker exposes NVIDIA GB200 Blackwell GPUs and
the checkpoint-backed conda environment from `setup.sh`.

The requested 1B aliases use the checkpoints available on the cluster:

- `llama3-1b` is `unsloth/Llama-3.2-1B-Instruct`.
- `qwen3-1b` is the official-size `Qwen/Qwen3-1.7B` checkpoint.

All schedules retain the qualified physical projection width `N=8` while the
reported batch is the logical live request count. Each latency is the median
device interval for one fixed decode step after three unmeasured warmups and
20 measured launches. Model loading, checkpoint transfer, schedule
construction, and one-time weight preparation are excluded. Aggregate
throughput is `logical_batch / median_latency`; it is resident decode-step
throughput rather than end-to-end serving throughput.

## Median decode-step latency

| Logical batch | Llama 3.1 8B (ms) | Llama 3.2 1B (ms) | Qwen3 8B (ms) | Qwen3 1.7B (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.501280 | 0.696768 | 2.961696 | 1.397280 |
| 2 | 2.504928 | 0.701024 | 3.001520 | 1.414608 |
| 3 | 2.508320 | 0.704016 | 3.005968 | 1.391072 |
| 4 | 2.507152 | 0.704592 | 2.958224 | 1.392064 |
| 5 | 2.492096 | 0.700704 | 2.964928 | 1.387152 |
| 6 | 2.505312 | 0.701216 | 3.000768 | 1.346304 |
| 7 | 2.495072 | 0.702784 | 2.958176 | 1.364016 |
| 8 | 2.492736 | 0.701264 | 2.956224 | 1.428736 |

## Aggregate decode throughput

| Logical batch | Llama 3.1 8B (req/s) | Llama 3.2 1B (req/s) | Qwen3 8B (req/s) | Qwen3 1.7B (req/s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 399.8 | 1,435.2 | 337.64 | 715.68 |
| 2 | 798.4 | 2,853.0 | 666.33 | 1,413.82 |
| 3 | 1,196.0 | 4,261.3 | 998.01 | 2,156.61 |
| 4 | 1,595.5 | 5,677.0 | 1,352.16 | 2,873.43 |
| 5 | 2,006.4 | 7,135.7 | 1,686.38 | 3,604.51 |
| 6 | 2,395.0 | 8,556.6 | 1,999.49 | 4,456.65 |
| 7 | 2,805.6 | 9,960.4 | 2,366.32 | 5,131.90 |
| 8 | 3,209.3 | 11,408.0 | 2,706.15 | 5,599.35 |

## Correctness gate

Mean relative tensor error below 5% is the authoritative acceptance gate.
Exact next-token agreement is recorded as an informational diagnostic and
does not override an accepted tensor result.

| Model | B=8 checkpoint evidence | Multi-block KV evidence | Status |
| --- | --- | --- | --- |
| Llama 3.1 8B | Every check below 5%; final hidden error 3.304% | Position 140, two KV128 blocks, at most 2.494% | PASS |
| Llama 3.2 1B | Worst reported tensor error 4.857% | Position 128, three KV64 blocks, attention oracle 0.190% | PASS |
| Qwen3 8B | Worst reported tensor error 2.888% | Position 65, two KV64 blocks, all Q/K/V/O live | PASS |
| Qwen3 1.7B | Integrated-tree worst tensor error 4.118% | Position 65, two KV64 blocks, 4.059% overall and 0.410% attention | PASS |

The position-0 Llama 3.1 8B fused argmax differed from the dense reference in
one accepted run (`76944` versus `75987`), while its position-140 run matched
token 264. The other documented checkpoint runs matched their reference
tokens. These token results remain non-gating.

## Optimization outcomes

- Llama 3.1 8B now supports logical batch 1-8 with a seq-major batched KV
  cache and validates request rows beyond row zero.
- Llama 3.2 1B uses the existing native HDIM64 Blackwell attention path. Its
  build uses 96 registers and reports zero spills.
- Qwen3 8B reuses the existing issuer-only M64N8 UMMA task and tile-major
  weights. Against the matched pre-optimization image, median latency improved
  4.47% at B1 and 1.15% at B8. Its build uses 233 registers and zero spills.
- Qwen3 1.7B's material optimization is corrected full-fold SM placement,
  reducing B8 from 120.310 ms to 1.429 ms. A matched A/B found generic
  row-major M64N8 faster than issuer-only/tile-major by 7.36% at B1 and 0.54%
  at B8, so the final image retains the generic existing task. Its build uses
  233 registers and zero spills.

No model port adds a new opcode or changes allocator/writeback publication
behavior. Detailed commands, job identifiers, build profiles, and tensor
checks are in the per-model reports:

- [Llama 3.1 8B](blackwell_llama3_8b_results.md)
- [Llama 3.2 1B](blackwell_llama32_1b_results.md)
- [Qwen3 8B](blackwell_qwen3_8b_results.md)
- [Qwen3 1.7B](blackwell_qwen3_1p7b_results.md)
