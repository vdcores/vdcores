# CPU/GPU Placement Experiment

Use this after the fixed attention-to-CPU-to-GPU handoff benchmark has passed
on a coherent GH200 or GB200 system.

## Build

The experiment uses both the full VDCores extension and the small handoff
extension:

```bash
make pyext
make handoff
```

## Measure

Run GPU benchmarks sequentially. The default sweep compares the same float64
dot product on CPU and GPU at six work sizes. Both paths start after split-KV
attention and finish through the same prequeued GPU successor.

```bash
python tests/script/check_attention_placement.py \
  --iterations 100 \
  --warmup 5 \
  --output results/cpu_gpu_handoff/YYYY-MM-DD-placement.json
```

Use `--work-items 64,256,1024` for a shorter sweep. The script alternates CPU
and GPU order at every size, validates both dot products, and rechecks the
split-attention output against the PyTorch reference.

## Consume The Measurements

`python/dae/placement.py` loads the result with
`MeasuredCrossoverPolicy.from_benchmark(...)`. The policy interpolates between
measured sizes, adds CPU and GPU queue-delay estimates, and can prefer the only
device predicted to meet a deadline. It clamps outside the measured range;
that is a visible limitation, not evidence about unmeasured sizes.

This policy is a first placement layer. It does not add a CPU worker pool,
encode CPU operations in the VDCores instruction stream, or infer contention
automatically.
