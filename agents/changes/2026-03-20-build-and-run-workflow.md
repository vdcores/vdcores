# 2026-03-20 Build And Run Workflow Check

## Purpose

Exercise the requested VDCores develop/test loop by rebuilding the extension with `make pyext` and running `python app/python/llama3/sched.py -b`, then record the standard workflow.

## Repository State

- `git status --short` was clean before the run.

## Commands Run

```bash
make pyext
python app/python/llama3/sched.py -b
```

## Results

- `make pyext` failed during the Torch extension build.
- Failure cause: CUDA version mismatch detected by `torch.utils.cpp_extension`.
- Detected local CUDA toolkit: `12.5`
- PyTorch CUDA version: `13.0`
- `python app/python/llama3/sched.py -b` succeeded.

## Benchmark Outcome

- 132 SMs
- 1 iteration
- Average duration: `80486408.48 ns`
- Average execution time: `80501632.00 ns`

## Files Added

- `agents/workflows/development-and-test.md`

## Notes

- This confirms the standard execution command for the Llama3 demo benchmark path.
- Rebuilds should be treated as required only when CUDA runtime or kernel code changes.
- Python-only changes can use the benchmark command directly without a rebuild.
