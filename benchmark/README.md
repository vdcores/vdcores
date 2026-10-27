# Benchmarks

Run from the repository root using Conda base and a CUDA 13.0 toolkit matching
Torch.

```bash
# Compile the runtime, including the two operators used by the NVLink suite.
make pyext DAE_COMPUTE_OPS=OP_COPY,OP_TERMINATEC
pip install ninja  # needed for the small raw-CUDA baseline extension
python -m benchmark --list
na run -n=2 python -m benchmark nvlink --num-gpus 2 -b 20 \
    --json output/nvlink.json
```

`python benchmark/run.py ...` is an equivalent thin entry point. Select multiple
suites by name, or `all`; suites execute sequentially. `-b [ITERATIONS]` / `--bench`
sets measurement repetitions (default 20). `--help` and `--list` do not initialize
CUDA. Leave `CUDA_VISIBLE_DEVICES` as set by `na`; when already inside an
allocation, invoke Python directly instead of nesting `na run`.

## NVLink suite

Requires exactly two explicitly selected visible Hopper GPUs, peer access in
both directions, and an NVLink connection verified with `nvidia-smi topo -m`.
CUDA UUIDs map visible ordinals to physical GPUs. The suite fails with a clear
error on an allocation connected only by PCIe.

- Compares VDCores 1D TMA load/copy/store with a vectorized raw CUDA SM copy and
  CUDA's copy engine (`cudaMemcpyAsync` with enabled peer UVA pointers). Each SM
  copy pulls from a peer source into a local destination. Duplex uses independent
  buffers and launches one transfer on each GPU simultaneously.
- Sweeps 16, 64, and 256 MiB **per direction**, and 8/16/32 KiB VDCores tiles.
  Override payloads with `--sizes-mib 64 256 512`, SM count with `--num-sms 64`,
  or warmups with `--warmup 5`. All VDCores SMs are used by default.
- Compilation, allocation, schedule upload, warmup, and exact byte-for-byte
  validation happen outside timing. Baseline compilation uses Torch's extension
  cache. Each timed graph contains eight copies to remove Python dispatch gaps.
  Both start and end events use the same GPU clock; a control stream waits for
  completion on both GPUs in duplex mode. VDCores timings include the complete
  kernel, including store completion.
- Reports median and maximum measured payload GB/s in decimal units, counting
  each byte crossing NVLink once. Duplex sums both directions. The final peak
  for each implementation/direction is the fastest measured size/tile/sample;
  it is an observed throughput, not a hardware specification. Small payloads may
  benefit from caches; use payloads larger than L2 for sustained bandwidth.
- JSON includes device UUIDs, topology, parameters, all timing samples, results,
  and measured peaks. Run benchmarks sequentially for credible measurements.

The baseline uses `cudaMemcpyAsync` because `cudaMemcpyPeerAsync` rejects stream
capture on the tested CUDA toolkit. Runtime device configuration also stays
outside capture; warmup establishes those settings before graph recording.

If a local `.vdcore.build` file selects other operators, the suite reports a
rebuild instruction. To preserve model operators, include `OP_COPY` and
`OP_TERMINATEC` in the full selection rather than replacing it with the minimal
example above.

## Adding a suite

Add one small module under `suites/`, exporting `run(args)` and returning a
JSON-serializable dictionary. It is discovered automatically. Keep suite-specific
GPU imports inside `run`, provide a raw-system baseline, validate results outside
timing, and state the work/byte counting convention. Put any baseline CUDA code
under `cuda/`. Shared selection, repetitions, and JSON output belong in
`runner.py`; the entry points stay thin.
