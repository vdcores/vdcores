# Benchmark suites

- `benchmark/` contains the thin `__main__.py` / `run.py`
  entry points, shared CLI in `runner.py`, and automatically discovered suite
  modules in `suites/`. Each suite exports `run(args)` and returns JSON data.
- Use `python -m benchmark --list` or `--help` without initializing CUDA. Select
  one or more suites, or `all`; never run performance suites in parallel.
- Build the runtime first with `make pyext`. NVLink needs `OP_COPY` and
  `OP_TERMINATEC` in the selected compute ops. An ignored local `.vdcore.build`
  file can omit these; supply an explicit operator superset when needed.
- The raw NVLink baseline compiles through Torch's extension cache and needs
  Ninja: `pip install ninja`. No external benchmark repository is needed.
- Allocate with `na run -n=2 python -m benchmark nvlink --num-gpus 2 -b 20
  --json output/nvlink.json`. Inside an existing `NA_JOB`, run Python directly.
- The NVLink suite checks physical topology through CUDA-visible UUIDs and
  `nvidia-smi`; CUDA peer capability alone is not sufficient to label bandwidth
  as NVLink. Default measurements sweep 16/64/256 MiB per direction and
  8/16/32 KiB VDCores tiles, with two unidirectional cases and simultaneous duplex.
- Graph batches exclude host dispatch and setup. Cross-device event waits gate
  both GPUs; elapsed time always uses two events from one GPU. Count only payload
  crossing the link, once per direction, in decimal GB/s. Keep exact data checks
  outside timing. JSON retains samples and measurement parameters.
- `cudaMemcpyPeerAsync` rejected capture on the tested CUDA 13 toolkit; use
  `cudaMemcpyAsync` with enabled peer UVA pointers for the copy-engine baseline.
  Upload and graph timing stay within the benchmark suite. Keep buffers and their
  peer tensors alive until timing completes.
- GPU correctness smoke check:
  `na run -n=2 python -m unittest discover -s tests -p test_nvlink.py`.
