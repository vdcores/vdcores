# Qwen3-8B Blackwell decode results

Measured on 2026-08-30 through 2026-08-31 with checkpoint
`/mnt/checkpoints/Qwen/Qwen3-8B` on worker `10.0.16.34` (one NVIDIA GB200
GPU, one MPI rank). The initial batch sweep used commit `e2494e7` on
`port/qwen3-8b-blackwell`; the retained BF16 optimization chain ends at
`584c3bf` on `opt/qwen3-8b-blackwell-bf16`.

## Method

- One prefill token and one measured decode token, `--max-seq-len 64`.
- Logical request batch 1 through 8 over the qualified physical UMMA N=8
  tile. Only live request rows participate in attention, RMS, and argmax.
- Three unmeasured warmups followed by 20 measured launches:
  `DAE_BENCH_WARMUP=3`, `-N 1 --bench 20`.
- Latency is the median device execution interval reported by the DAE
  profiler. Aggregate throughput is `batch * 1e9 / median_ns` and excludes
  model loading, one-time tile-major weight packing, and host setup.
- Every run used the worktree-local Python extension through an absolute
  `PYTHONPATH`; checkpoints and Hugging Face metadata were offline.

## Batch sweep

| Logical batch | Job | Median latency (ns) | Median latency (ms) | Aggregate requests/s |
| ---: | --- | ---: | ---: | ---: |
| 1 | `20260830T184754Z-923648` | 2,961,696 | 2.961696 | 337.64 |
| 2 | `20260830T184818Z-925846` | 3,001,520 | 3.001520 | 666.33 |
| 3 | `20260830T184837Z-927691` | 3,005,968 | 3.005968 | 998.01 |
| 4 | `20260830T184856Z-929303` | 2,958,224 | 2.958224 | 1,352.16 |
| 5 | `20260830T184912Z-930848` | 2,964,928 | 2.964928 | 1,686.38 |
| 6 | `20260830T184929Z-932384` | 3,000,768 | 3.000768 | 1,999.49 |
| 7 | `20260830T184947Z-933827` | 2,958,176 | 2.958176 | 2,366.32 |
| 8 | `20260830T185004Z-935235` | 2,956,224 | 2.956224 | 2,706.15 |

The existing Llama Blackwell issuer-only M64N8 UMMA operator and tile-major
weight layout improved the matched batch-1 median from 3,100,128 ns to
2,961,696 ns (4.47%) and the matched batch-8 median from 2,990,672 ns to
2,956,224 ns (1.15%). Baseline jobs were `20260830T184107Z-880759` and
`20260830T184128Z-883093`, respectively. No new opcode, allocator behavior,
or publication mechanism was introduced.

### Retained phased down projection

The down projection now keeps each owner's prefix and tail products in one
ordinary BF16 UMMA accumulator and performs one final reduction store. The two
folds retain exactly 128 owners and six K1024 repeats per owner: each fold
observes two prefix repeats after `bar_silu_out1` and four tail repeats after
`bar_silu_out2`. This removes one compute dispatch and one intermediate
reduction store per owner per layer without changing the M2C/C2M or allocator
publication protocol. `--no-fused-down-phases` selects the prior two-task
schedule for matched controls.

On the same SM100 image with `--max-seq-len 128`, three warmups, and 301
measured iterations, the full-token control/candidate/control medians were:

| Batch | Control 1 | Phased down | Control 2 | Control mean | Saving |
| ---: | --- | --- | --- | ---: | ---: |
| 1 | 2.961600 ms (`20260831T022821Z-3727764`) | 2.951136 ms (`20260831T022854Z-3730697`) | 3.001984 ms (`20260831T022925Z-3733341`) | 2.981792 ms | 30.656 us (1.03%) |
| 8 | 2.959776 ms (`20260831T022957Z-3736089`) | 2.956000 ms (`20260831T023029Z-3738846`) | 2.964896 ms (`20260831T023102Z-3742529`) | 2.962336 ms | 6.336 us (0.21%) |

A one-layer final-RMS 301-iteration bracket measured 78.240/77.664/78.016 us
(`20260831T022614Z-3713222`, `20260831T022641Z-3715507`, and
`20260831T022708Z-3719011`), a 0.464-us improvement over the control-endpoint
mean. The composed full-token gain is therefore consistent with a small
per-layer queue/dispatch reduction rather than a new compute primitive.

### Retained critical-path down-owner offload

On the composed BF16 image at `7d48fe9`, the phased down projection has 128
logical owners on the 132-SM GB300.  Physical SMs 0--3 also carry the earliest
live attention requests, while SMs 128--131 only carry the short prefix-SiLU
task before the down frontier.  The retained placement maps logical down
owners 0--3 onto physical SMs 128--131.  It preserves all 128 contributors,
the same K segments, reduction stores, barriers, and compute operators; only
physical ownership changes.  `--no-down-tail-offload` selects the matched
placement control.

With `--max-seq-len 128`, three warmups, and 301 measured iterations:

| Batch | Control | Down-owner offload | Saving |
| ---: | ---: | ---: | ---: |
| 1 | 2.936256 ms (`20260831T025341Z-3883376`) | 2.842656 ms (`20260831T025418Z-3888616`) | 93.600 us (3.19%) |
| 8 | 2.936160 ms (`20260831T025525Z-3896378`) | 2.830880 ms (`20260831T025454Z-3892296`) | 105.280 us (3.59%) |

Full 36-layer BF16 reference checks passed at batch 1 and batch 8 in jobs
`20260831T025559Z-3899631` and `20260831T025635Z-3902811`.  The worst reported
mean-relative errors were 2.397% and 2.806%, respectively, and both runs
returned token 422 on every live request row.

## Correctness

Full 36-layer reference checks use a 5% mean-relative-error gate for every
reported projection, fused Q/K RMS+RoPE result, SiLU output, final hidden/RMS
state, logits slice, and cross-request consistency check.

| Batch | Job | Worst reported tensor error | Final tokens |
| ---: | --- | ---: | --- |
| 1 | `20260830T184526Z-907712` | 2.659% | `[422]` |
| 8 | `20260830T184556Z-910073` | 2.888% | `[422, 422, 422, 422, 422, 422, 422, 422]` |
| 8 repeat | `20260830T184704Z-918688` | 2.668% | `[422, 422, 422, 422, 422, 422, 422, 422]` |
| 1 phased down | `20260831T023141Z-3745358` | 2.513% | `[422]` |
| 8 phased down | `20260831T023212Z-3749900` | 2.222% | `[422, 422, 422, 422, 422, 422, 422, 422]` |

The repeated batch-8 run confirms the same accepted token on every live row
and keeps every tensor metric below 5%. Token agreement is recorded as an
additional signal; tensor error is the acceptance criterion.

The optimized image also passed a two-KV-block attention smoke at token
position 65 with a 128-token cache (`20260830T185046Z-940688`): all Q, K, V,
and attention-output tensors were finite and nonzero. The existing K4096
interleaved SiLU handler smoke passed in `20260830T182540Z-782716`.

## Build

- Target: SM100a, native UMMA, selective Qwen3-8B image.
- Selected compute operators: 11 plus one dynamic family, from 112 available.
- `ptxas`: 233 registers, 9 barriers, 336-byte stack, 8,624-byte static shared
  memory, and zero spills.
- Runtime profile: 256 instruction slots and 218 KiB dynamic shared memory.
- Fused attention keeps the existing Q/K RMS normalization and RoPE path;
  dense projections use the existing Llama Blackwell issuer-only M64N8
  operator.

## Context-matched C128 milestone sweep

The 2026-08-31 correction seeds 127 checkpoint-backed prefix KV rows and
decodes at position 127. Three warmups and 20 measured resident launches were
used; framework baselines were not rerun.

| Logical batch | Median latency (ns) | Median latency (ms) |
| ---: | ---: | ---: |
| 1 | 2,835,808 | 2.835808 |
| 2 | 2,827,728 | 2.827728 |
| 4 | 2,846,864 | 2.846864 |
| 8 | 2,839,152 | 2.839152 |

Performance job: `20260831T053131Z-1172356`. C128 B8 correctness job
`20260831T053106Z-1168105` passed every tensor gate, with a worst reported
mean-relative error of 4.183%; all eight outputs exactly matched reference
token 198.

The command shape was:

```bash
DAE_BENCH_WARMUP=3 python app/python/qwen3/sched.py \
  --model-name /mnt/checkpoints/Qwen/Qwen3-8B \
  --batch-size BATCH --prefill-length 127 --max-seq-len 512 \
  -N 1 --bench 20
```
