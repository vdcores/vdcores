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

The lean Blackwell build selects 12 compute operations (two dynamic), uses 96
registers and nine barriers, has a 208-byte stack frame and 7,088 bytes of
static shared memory, and reports zero spills.

The existing Llama-8B fused GEMV/argmax head was evaluated at the 1B model's
`K=2048`, but repeated runs produced nondeterministic corrupt partial maxima.
It is therefore not used here; the stable materialized-logits head is retained.
