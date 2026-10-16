# Minimal Hopper runtime series

2026-10-16, `hopper-runtime-minimal`, based on `main` `fd75b8d`.
Retains all six previously accepted runtime improvements in separate commits:
queue acquire polling, constant-index load ports, bulk barrier resets,
store/allocator warp placement, a private free-slot cache, and tensor-map prefetch.
Task kernels, model schedules and queue participant counts are unchanged.

The cleanup makes participant polling the queue's wait implementation directly
and uses constant indices instead of changing the load-queue API. Kernel stack
falls from 80 to 32 bytes; static local-load/store sites fall from 14/13 to 6/4.
The release cache and host reset implementation match the accepted backport.

## Component comparisons

Device milliseconds, H100 80GB HBM3 / 132 SMs, batch eight, one decode step.
All GPU work used sequential foreground `na run` jobs. Fixed 14-op manifest,
CUDA 13.0.88, Torch 2.14.1+cu130. The first three rows average two interleaved
1,001-sample run medians; subsequent rows are 501-sample component screens.
Each excludes five warmups and uses a fresh process with writable buffers reset
before every launch. Each runtime commit records its comparison for both models.

| Cumulative runtime | Llama 3.1 8B | Llama 3.2 1B |
|---|---:|---:|
| Main (initial controls) | 5.431152 | 1.040192 |
| + Queue acquire polling | 5.388880 | 1.028944 |
| + Constant-index load ports | 5.364384 | 1.035152 |
| + Bulk barrier resets | 5.359584 | 1.033248 |
| + Store/allocator warp placement | 5.351040 | 1.027552 |
| + Cached free-slot bitmap | 5.339392 | 1.024096 |
| + Tensor-map prefetch | 5.319392 | 0.998656 |

The port change costs 0.006208 ms on 1B relative to polling alone. All accepted
components are retained as requested; acceptance applies to the complete series.
1B measurements are timing-only, with correctness deliberately unchecked.

## Final comparison against main

One `na` allocation, frozen clean-build binaries, 1,001 measured launches per
fresh process, both models at each ABBA step. Device time is latest compute end
minus earliest compute start; reset/model loading are excluded.

| Order | 8B device ms | 1B device ms |
|---|---:|---:|
| Main A1 | 5.432032 | 1.035072 |
| Branch B1 | 5.318880 | 0.994272 |
| Branch B2 | 5.318944 | 0.999584 |
| Main A2 | 5.430816 | 1.038016 |

Mean of run medians: 8B **5.431424 -> 5.318912 ms**,
saving **0.112512 ms / 2.07%**. Resampling contiguous
50-launch blocks, 5,000 draws, gives a 95% interval of
**[0.110944, 0.113832] ms**, above the 0.05 ms gate.
This interval describes these runs, not variation across machines or workloads.
1B: **1.036544 -> 0.996928 ms**, saving **0.039616 ms / 3.82%**.

Synchronous launch, including device execution: 8B
**12.769343 -> 7.085242 ms (44.51% lower)**;
1B **5.057170 -> 2.339005 ms (53.75% lower)**.

## Validation and reproduction

Every step passed a clean fixed-manifest build, exact queue/GEMV tests and a
fresh 8B control-flow reference. Repeated 8B launches still show the existing
intermittent token issue: final ABBA had 3 deviations/2,012 launches on main and
4/2,012 on the branch, including warmups. No repair of that issue is claimed.
The restored sources and binary exactly match the frozen final candidate.

```bash
make clean
make pyext DAE_COMPUTE_OPS_FILE=tests/llama_runtime.ops
na run python tests/runtime_queue.py
na run python tests/runtime_gemv.py
na run python tests/script/benchmark_llama_runtime.py 8b --checkpoint /path/to/8b --samples 1001 --output 8b.json
na run python tests/script/benchmark_llama_runtime.py 1b --checkpoint /path/to/1b --samples 1001 --output 1b.json
na run python tests/script/benchmark_llama_runtime.py 8b --checkpoint /path/to/8b --correctness
```

8B uses `unsloth/Meta-Llama-3.1-8B-Instruct` revision
`a2856192dd7c25b842431f39c179a6c2c2f627d1`; input token/position are 791/0
for 8B and 791/10 for 1B. Local artifacts, raw samples, frozen sources/binaries,
per-step logs and confirmation analysis are under
`.agentlog/2026-10-16-hopper-runtime-minimal/`.
