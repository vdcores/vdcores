# Benchmark suites

- `benchmark/` contains the thin `__main__.py` / `run.py`
  entry points, shared CLI in `runner.py`, and automatically discovered suite
  modules in `suites/`. Each suite exports `run(args)` and returns JSON data.
- Use `python -m benchmark --list` or `--help` without initializing CUDA. Select
  one or more suites, or `all`; never run performance suites in parallel.
- Build the runtime first with `make pyext`. NVLink needs `OP_DUMMY` and
  `OP_TERMINATEC` in the selected compute ops. An ignored local `.vdcore.build`
  file can omit these; supply an explicit operator superset when needed.
- The raw NVLink baseline compiles through Torch's extension cache and needs
  Ninja: `pip install ninja`. No external benchmark repository is needed.
- Allocate with `na run -n=2 python -m benchmark nvlink --num-gpus 2 -b 20
  --json output/nvlink.json`. Inside an existing `NA_JOB`, run Python directly.
- The NVLink suite checks physical topology through CUDA-visible UUIDs and
  `nvidia-smi`; CUDA peer capability alone is not sufficient to label bandwidth
  as NVLink. Default measurements sweep 16/64/256 MiB buffers with fixed 8 KiB
  VDCores tiles. Compare raw CUDA and VDCores load/store roundtrips, with one
  executor on either GPU or both executors running concurrently.
- Graph batches exclude host dispatch and setup. Cross-device event waits gate
  both GPUs; elapsed time always uses two events from one GPU. Count only payload
  crossing the link: read plus write (`2 * buffer_bytes` per executor), in
  decimal GB/s. Verify peer buffers remain unchanged outside timing. JSON
  retains samples and measurement parameters.
- VDCores relays `TmaLoad1D(peer).writeback()` through
  `Dummy(count, writeback=True)` to the store warp using the same shared slot.
  Upload and timing stay within the benchmark suite. Keep buffers and their
  peer tensors alive until timing completes.
- Short GPU benchmark check:
  `na run -n=2 python -m benchmark nvlink --num-gpus 2 --sizes-mib 64 -b 3 --warmup 1`.
