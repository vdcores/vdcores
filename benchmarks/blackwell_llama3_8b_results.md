# Llama 3.1 8B Blackwell decode results

Measured on worker `10.0.16.34` on 2026-08-30 with checkpoint
`/mnt/checkpoints/unsloth/Meta-Llama-3.1-8B-Instruct`.

The tuned schedule retains the qualified physical UMMA/GEMV width `N=8`.
`--batch-size` selects one through eight logical request rows for the
request-dependent work and outputs. The KV cache is physically seq-major so
request and head remain jointly addressable by the existing TMA descriptors.

## Decode-step latency

Each row measures one fixed position-0 decode step (`-N 1
--no-control-flow`), after three warmups, over 20 timed samples. Model loading
and schedule construction are outside the timed region. Aggregate throughput
is logical batch divided by median step latency.

| Logical batch | Median (ms) | Aggregate req/s |
| ---: | ---: | ---: |
| 1 | 2.501280 | 399.8 |
| 2 | 2.504928 | 798.4 |
| 3 | 2.508320 | 1,196.0 |
| 4 | 2.507152 | 1,595.5 |
| 5 | 2.492096 | 2,006.4 |
| 6 | 2.505312 | 2,395.0 |
| 7 | 2.495072 | 2,805.6 |
| 8 | 2.492736 | 3,209.3 |

Cluster job: `20260830T183631Z-853651`.

The sweep used `DAE_BENCH_WARMUP=3` and the following model command for each
batch value:

```bash
python app/python/llama3/sched.py \
  --model /mnt/checkpoints/unsloth/Meta-Llama-3.1-8B-Instruct \
  --batch-size BATCH --no-control-flow -N 1 --bench 20
```

## Correctness and cache traversal

- B=1 and B=8 full-checkpoint jobs passed every tensor check under the 5%
  acceptance criterion. At B=8, request 7 K/V errors in the first two layers
  remained below 0.95%, final attention-row divergence was 0.954%, final
  hidden/RMS errors were 3.304%/3.282%, and all eight output rows agreed. The
  position-0 fused argmax token differed from the dense reference (`76944`
  versus `75987`), which is recorded as a non-gating exact-token diagnostic.
- Job `20260830T183557Z-850519` used a 140-token prefill and decoded at
  position 140, traversing two KV128 blocks. Every B=8 request row passed;
  final hidden/RMS errors were at most 2.487%/2.494%, and all eight tokens
  exactly matched reference token 264.

