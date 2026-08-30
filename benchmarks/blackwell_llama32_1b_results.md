# Llama 3.2 1B Blackwell decode results

Measured on worker `10.0.16.34` on 2026-08-30 with checkpoint
`/mnt/checkpoints/unsloth/Llama-3.2-1B-Instruct`.

The schedule keeps the proven physical UMMA/GEMV width `N=8`. `--batch-size`
selects the logical active request count from 1 through 8 for embedding, RMS,
copy, attention, argmax, and token rows. Projection buffers remain padded to
eight rows.

## Decode-step latency

Each row measures one resident position-0 decode step (`-N 1`), after three
warmups, over 20 timed samples. Model loading and schedule construction are
outside the timed region. Aggregate throughput is logical batch divided by
mean step latency.

| Logical batch | Min (us) | Median (us) | Mean (us) | Max (us) | Aggregate req/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 692.896 | 696.768 | 697.130 | 701.824 | 1,434.5 |
| 2 | 696.448 | 701.024 | 700.688 | 704.768 | 2,854.3 |
| 3 | 701.184 | 704.016 | 704.736 | 715.232 | 4,256.9 |
| 4 | 702.080 | 704.592 | 704.822 | 712.288 | 5,675.2 |
| 5 | 698.144 | 700.704 | 700.797 | 704.992 | 7,134.7 |
| 6 | 697.888 | 701.216 | 701.048 | 702.912 | 8,558.6 |
| 7 | 700.928 | 702.784 | 703.080 | 707.232 | 9,956.2 |
| 8 | 698.784 | 701.264 | 701.845 | 708.192 | 11,398.5 |

Cluster job: `20260830T184001Z-873568`.

### Optimized schedule

The accepted schedule fuses the existing M64 issuer-only up-projection and
SiLU operation, hands each 2,048-row SiLU shard directly to its matching
down-projection shard, and uses the issuer-only M64 operation for the
materialized LM-head epochs. The output projection remains M128: the faster
M64/B2 fold-4 alternative failed the request-row cache validation, while its
correct fold-2 placement was slower.

The following is the final default candidate, using the same resident
position-0 decode step, three warmups, and 20 timed samples as the baseline.

| Logical batch | Min (us) | Median (us) | Mean (us) | Max (us) | Baseline median (us) | Median change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 673.152 | 678.144 | 678.038 | 684.064 | 696.768 | -2.67% |
| 2 | 677.568 | 679.616 | 680.160 | 684.672 | 701.024 | -3.05% |
| 4 | 680.416 | 681.808 | 682.334 | 687.872 | 704.592 | -3.23% |
| 8 | 682.752 | 684.832 | 685.133 | 688.544 | 701.264 | -2.34% |

Final sweep jobs: B1 `20260830T232401Z-2580007`, B2
`20260830T232418Z-2581386`, B4 `20260830T232436Z-2582529`, and B8
`20260830T232453Z-2583925`.

These medians do not meet the requested 0.254625/0.265736/0.280968/0.295637
ms B1/B2/B4/B8 thresholds. The current BF16 projection path remains dominated
by checkpoint-weight traffic. Existing FP8 paths are not drop-in substitutes:
the block-128 schedule is single-vector, while the native N8 stream/split path
requires activation quantization, prepacked weights, and dispatch handlers that
are not present in the lean Llama Blackwell runtime.

The sweep command was:

```bash
/home/azhpcuser/.codex/skills/gpu-cluster/scripts/mpi-run \
  -n 1 --host 10.0.16.34 --wait 60 \
  --env PYTHONPATH=/home/azhpcuser/jiaxin-shared/vdcores-project/vdcores-agent-llama32-1b/python \
  --env PYTHONPYCACHEPREFIX=/tmp/vdcores-pycache-llama32-1b-final-sweep \
  -- env PATH=/mnt/vdcores/env/miniconda3/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -c 'for batch in 1 2 3 4 5 6 7 8; do DAE_BENCH_WARMUP=3 python app/python/llama32_1b/sched.py --batch-size "$batch" -N 1 --model-name /mnt/checkpoints/unsloth/Llama-3.2-1B-Instruct --hf-cache-dir /mnt/checkpoints/huggingface -b 20; done'
```

## Correctness and cache traversal

- Final optimized B=1 post-build job `20260830T232804Z-2601175` passed every tensor gate;
  its worst logit error was 2.346%, and both generated and materialized tokens
  exactly matched reference token 315.
- Final optimized B=8 post-build job `20260830T232828Z-2603746` passed every request row;
  its worst tensor error was 2.913%, and all generated
  and materialized tokens exactly matched reference token 315.
- Final optimized post-build M128 cache job `20260830T232851Z-2606556` decoded position
  128 through three KV blocks. Current K/V request divergence was at most
  1.073%/2.298%, final attention-output divergence was 2.892%, the explicit
  attention oracle error was at most 0.192%, and all eight tokens were 13.
- The rejected M64/B2 fold-4 cache job `20260830T231853Z-2548449` kept every
  request live and passed the attention oracle (0.190%), but K/V/output request
  divergence reached 6.069%/12.089%/12.042%; it is not the default.

- B=8 post-build full-checkpoint job `20260830T184202Z-887496` validated every request
  row. Layer 0 Q/K/V errors were 0.196%/0.230%/0.457% on every row. Layer 1
  Q/K/V worst errors were 0.447%/0.437%/0.783%. Final attention-output row
  divergence was at most 2.285%, final hidden error at most 3.737%, and the
  worst logit error was 4.857%. All eight DAE and materialized-logit tokens
  exactly matched reference token 315.
- B=1 full-checkpoint job `20260830T183742Z-860289` passed all tensor checks;
  final hidden error was 1.688% and the exact reference/generated token was
  315.
- The seq-major cache smoke job `20260830T183715Z-857244` seeded identical
  nonzero history for all eight requests and decoded at position 128. It
  traversed three 64-token KV blocks; all current K/V and attention-output rows
  were live, and the explicit three-block PyTorch attention oracle differed by
  at most 0.190%. All eight generated tokens were 13.

The optimized lean Blackwell build selects 13 compute operations (two dynamic), uses 96
registers and nine barriers, has a 208-byte stack frame and 7,088 bytes of
static shared memory, and reports zero spills.

The existing Llama-8B fused GEMV/argmax head was evaluated at the 1B model's
`K=2048`, but repeated runs produced nondeterministic corrupt partial maxima.
It is therefore not used here; the stable materialized-logits head is retained.
