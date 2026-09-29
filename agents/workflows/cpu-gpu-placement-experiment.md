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

## Plan Or Run One Adaptive Chain

After a real calibration file exists, inspect the decision without CUDA:

```bash
python tests/script/run_adaptive_attention_handoff.py \
  --policy results/cpu_gpu_handoff/YYYY-MM-DD-placement.json \
  --work-items 4096 \
  --plan-only
```

Remove `--plan-only` on the coherent machine to execute the selected path. Use
`--cpu-queue-us`, `--gpu-queue-us`, or `--deadline-us` to supply scheduling
context. Work sizes outside the measured range fail by default; explicitly pass
`--allow-outside-range` to clamp to the nearest measurement.

The adaptive runner builds a three-node dependency plan with
`python/dae/heterogeneous.py`: GPU attention, a flexible CPU/GPU dot product,
and a fixed GPU successor. Plan-only mode never imports the CUDA extension.
