# Qwen3-8B Blackwell decode results

Measured on 2026-08-30 with checkpoint
`/mnt/checkpoints/Qwen/Qwen3-8B` on worker `10.0.16.34` (one NVIDIA GB200
GPU, one MPI rank). The tested implementation is commit `e2494e7` on
`port/qwen3-8b-blackwell`.

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
