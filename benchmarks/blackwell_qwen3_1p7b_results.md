# Qwen3-1.7B Blackwell decode results

Measured on worker `10.0.16.34` on 2026-08-30 with checkpoint
`/mnt/checkpoints/Qwen/Qwen3-1.7B`. The tested implementation is commit
`f7b7fb8` on `port/qwen3-1b-blackwell`.

The schedule retains the qualified physical GEMV width `N=8` and exposes
`--batch-size=1..8` as the logical request count. Request-dependent embedding,
RMS, copy, attention, argmax, and output rows use the logical count; dense
projections remain padded to eight rows. The KV cache is physically seq-major
so every request/head pair remains addressable by the existing TMA builders.

## Decode-step latency

Each row measures one position-0 decode step after three unmeasured warmups and
over 20 measured launches (`DAE_BENCH_WARMUP=3`, `-b 20`). Model loading and
schedule construction are outside the timed region. Latency is the median
device execution interval reported by the DAE profiler. Aggregate throughput
is logical batch divided by median latency.

| Logical batch | Median latency (ns) | Median latency (ms) | Aggregate req/s |
| ---: | ---: | ---: | ---: |
| 1 | 1,397,280 | 1.397280 | 715.68 |
| 2 | 1,414,608 | 1.414608 | 1,413.82 |
| 3 | 1,391,072 | 1.391072 | 2,156.61 |
| 4 | 1,392,064 | 1.392064 | 2,873.43 |
| 5 | 1,387,152 | 1.387152 | 3,604.51 |
| 6 | 1,346,304 | 1.346304 | 4,456.65 |
| 7 | 1,364,016 | 1.364016 | 5,131.90 |
| 8 | 1,428,736 | 1.428736 | 5,599.35 |

Cluster job: `20260830T190526Z-1030814`.

The sweep used the worktree-local extension through absolute `PYTHONPATH` and
ran this model command for each batch value:

```bash
python app/python/qwen3_1p7b/sched.py \
  --model-name /mnt/checkpoints/Qwen/Qwen3-1.7B \
  --max-seq-len 128 --batch-size BATCH -b 20
```

## Optimization A/B

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

Job `20260830T190446Z-1026378` ran all three full 28-layer checks against the
dense checkpoint reference. Every reported tensor passed the 5% mean-relative
error gate.

| Case | Worst reported tensor error | Attention row error | Exact token (informational) |
| --- | ---: | ---: | ---: |
| B1, position 0 | 3.000% | 1.847% | 25 |
| B8, position 0 | 2.428% | 1.895% | 25 |
| B8, position 65 (65-token prefill) | 4.059% | 0.410% | 52 |

The position-65 case traverses two 64-token KV blocks. It checks current K/V
and attention output for request 0 and request 7, proving the seq-major cache
store/load coordinates and repeat strides beyond the first block. Exact token
agreement is recorded as a diagnostic; the tensor-error gate is authoritative.

## Build

- Target: SM100a Blackwell selective image, 232 instruction slots.
- Compute image: 10 selected operations plus one dynamic family, from 112
  available operations.
- `ptxas`: 233 registers, nine barriers, 336-byte stack, 8,048 bytes static
  shared memory, and zero spills.
- Runtime dynamic shared memory: 219 KiB. The image launched successfully with
  both static and dynamic allocations active.
- No new opcode, allocator behavior, or publication mechanism was added.
