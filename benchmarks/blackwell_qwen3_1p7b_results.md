# Qwen3-1.7B Blackwell decode results

Measured on worker `10.0.16.34` on 2026-08-30 with checkpoint
`/mnt/checkpoints/Qwen/Qwen3-1.7B`. The tested implementation is commit
`8971e9e` on `port/qwen3-1b-blackwell`.

The schedule retains the qualified physical GEMV width `N=8` and exposes
`--batch-size=1..8` as the logical request count. Request-dependent embedding,
RMS, copy, attention, argmax, and output rows use the logical count; dense
projections remain padded to eight rows. The KV cache is physically seq-major
so every request/head pair remains addressable by the existing TMA builders.

## Decode-step latency

Each row measures one position-0 decode step after three unmeasured warmups and
20 measured launches (`DAE_BENCH_WARMUP=3`, `-b 20`). Model loading and
schedule construction are outside the timed region. Latency is the median
device execution interval reported by the DAE profiler. Aggregate throughput
is logical batch divided by median latency.

| Logical batch | Median latency (ns) | Median latency (ms) | Aggregate req/s |
| ---: | ---: | ---: | ---: |
| 1 | 1,153,392 | 1.153392 | 867.01 |
| 2 | 1,153,040 | 1.153040 | 1,734.55 |
| 4 | 1,154,176 | 1.154176 | 3,465.68 |
| 8 | 1,165,600 | 1.165600 | 6,863.42 |

Cluster job: `20260830T233517Z-2649692`.

The sweep used the worktree-local extension through absolute `PYTHONPATH` and
ran this model command for each batch value:

```bash
python app/python/qwen3_1p7b/sched.py \
  --model-name /mnt/checkpoints/Qwen/Qwen3-1.7B \
  --max-seq-len 128 --batch-size BATCH -b 20
```

## Optimization A/B

The accepted schedule forwards all 6,144 gate and up rows through the existing
per-SM `RegStore`/`RegLoad` path. It removes the 4,096-row materialized prefix,
its two global-store descriptors, and its input barrier. Two register SwiGLU
groups preserve independent 4,096/2,048-row readiness so low-K down projection
can overlap the high tail without changing the existing M64N8 operator or
runtime publication behavior.

The matched control was captured before the edit with the same 10-op SM100
image and 3-warmup/20-sample method (`20260830T225259Z-2388240`).

| Batch | Control (ms) | Register-fused (ms) | Latency change | Requested ceiling (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.343376 | 1.153392 | -14.14% | 0.766308 |
| 2 | 1.347968 | 1.153040 | -14.46% | 0.824114 |
| 4 | 1.382512 | 1.154176 | -16.52% | 0.885683 |
| 8 | 1.362992 | 1.165600 | -14.48% | 0.799076 |

This is a material improvement at every accepted batch, but it does not meet
the requested framework-relative latency ceilings. A grouped M128 direct-output
experiment was rejected after its launch failed to make progress; none of its
operator selections or schedule changes are present in the accepted image.

The original under-filled projection placement measured 112.214384 ms at B1
and 120.310352 ms at B8 (jobs `20260830T184934Z-932685` and
`20260830T185209Z-950363`). Correct full-fold placement accounts for the
material speedup: with otherwise generic row-major M64N8 GEMV it measured
1.285696/1.410880 ms at B1/B8 (`20260830T190105Z-1004199`).

A matched issuer-only/tile-major image with the same corrected placement
measured 1.387888/1.418608 ms (`20260830T185829Z-993182`), so the generic
row-major task was 7.36% faster at B1 and 0.54% faster at B8 in this A/B. The
evidence-backed final image therefore keeps generic M64N8 and the corrected
Q/K/V/out/down fold counts; it does not retain packed-weight streaming.

## Correctness and cache traversal

Job `20260830T233441Z-2646448` ran all three full 28-layer checks against the
dense checkpoint reference. Every reported tensor passed the 5% mean-relative
error gate.

| Case | Worst reported tensor error | Attention row error | Exact token (informational) |
| --- | ---: | ---: | ---: |
| B1, position 0 | 3.469% | 1.907% | 25 |
| B8, position 0 | 2.538% | 1.988% | 25 |
| B8, position 65 (65-token prefill) | 4.137% | 0.398% | 52 |

The position-65 case traverses two 64-token KV blocks. It checks current K/V
and attention output for request 0 and request 7, proving the seq-major cache
store/load coordinates and repeat strides beyond the first block. Exact token
agreement is recorded as a diagnostic; the tensor-error gate is authoritative.
Gate and up projections are intentionally not materialized by the optimized
schedule; their first public tensor is the fused SwiGLU output, which is checked
along with final hidden, RMS, logits, and both edge-request attention rows.

## Build

- Target: SM100a Blackwell selective image, 232 instruction slots.
- Compute image: 10 selected operations plus one dynamic family, from 112
  available operations.
- `ptxas`: 233 registers, nine barriers, 336-byte stack, 8,048 bytes static
  shared memory, and zero spills.
- Runtime dynamic shared memory: 219 KiB. The image launched successfully with
  both static and dynamic allocations active.
- No new opcode, allocator behavior, or publication mechanism was added.

## Context-matched C128 milestone sweep

The 2026-08-31 correction seeds 127 checkpoint-backed prefix KV rows and
decodes at position 127. Three warmups and 20 measured resident launches were
used; framework baselines were not rerun.

| Logical batch | Median latency (ns) | Median latency (ms) |
| ---: | ---: | ---: |
| 1 | 1,364,928 | 1.364928 |
| 2 | 1,365,984 | 1.365984 |
| 4 | 1,361,584 | 1.361584 |
| 8 | 1,387,152 | 1.387152 |

Performance job: `20260831T053638Z-1226097`. C128 B1 correctness job
`20260831T053614Z-1222284` passed every tensor gate with a worst reported
mean-relative error of 4.188%, and its output exactly matched reference token
198. Repeated C128 B8 checks (`20260831T053410Z-1203310` and
`20260831T053526Z-1214397`) also matched token 198, but were numerically
marginal: their worst observations were 5.228% on one logits shard and 5.239%
on fused SwiGLU. The B8 latency is therefore retained as performance evidence,
not a strict below-5% qualification.

The command shape was:

```bash
DAE_BENCH_WARMUP=3 python app/python/qwen3_1p7b/sched.py \
  --model-name /mnt/checkpoints/Qwen/Qwen3-1.7B \
  --batch-size BATCH --prefill-length 127 --max-seq-len 512 \
  -N 1 --bench 20
```
