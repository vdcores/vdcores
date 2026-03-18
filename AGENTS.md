# Project Guide

This repository is a CUDA-heavy `dae` project with Python scheduling scripts and a compiled Torch extension.

## Layout

- `app/python/`: runnable model-specific scripts and experiments.
- `app/python/llama3/`: Llama3 schedule, reference hooks, and attention verification helpers.
- `app/python/qwen3/`: Qwen3 schedule variants.
- `python/dae/`: Python package code for launcher, scheduler, and compiler support.
- `src/`, `include/`: CUDA/C++ extension sources and headers.
- `tests/`: test assets and scripts.

## Environment

- Preferred Python environment: Conda `base`.
- Typical activation:

- The project builds a CUDA extension via `make pyext`, linking `src/torch_runtime.cu` and `runtime.o`.

## Working Rules

- Check the repo state before editing so unrelated user changes are not described or overwritten.
- Keep edits focused and fast when the task is a single requested implementation.
- Prefer existing local reference helpers, especially under `app/python/llama3/`, over re-deriving model math.
- Use `agents/changes/` to log what changed.
- Use `agents/workflows/` to log durable project-specific workflow knowledge.

## Verification

- Use the lightest meaningful check first, such as `python -m py_compile ...` for Python edits.
- For schedule scripts like `app/python/llama3/sched.py`, full verification may require a machine with CUDA access.
- If a runtime check fails for environment reasons, record that clearly in the change log rather than guessing.
