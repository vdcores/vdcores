# Project Map

This note summarizes the stable structure confirmed during repository initialization.

## Main Entry Points

- `README.md`: high-level project overview, setup path, and demo commands.
- `Makefile`: builds `runtime.o` from `src/runtime.cu` and installs the Python extension with `make pyext`.
- `setup.py`: packages the `dae` Python module and compiles `src/torch_runtime.cu` linked against `runtime.o`.
- `setup.sh`: reference environment bootstrap for Conda, CUDA Toolkit 13.0.2, and Python dependencies.

## Code Areas

- `app/python/`: runnable experiments and schedule prototypes.
- `app/python/llama3/sched.py`: primary end-to-end decoding demo for `meta-llama/Llama-3.1-8B-Instruct`.
- `app/python/llama3/reference.py` and `app/python/llama3/llama_attention_reference.py`: local reference helpers worth checking before re-deriving model math.
- `app/python/gemv_mma_out.py`: dedicated correctness harness for the isolated `N=8` MMA GEMV operator path.
- `app/python/qwen3/`: Qwen 3 client, layer, utilities, and schedule variants.
- `python/dae/launcher.py`: Python launch plumbing between schedules and the runtime.
- `python/dae/schedule.py`: scheduling interface and composition layer.
- `python/dae/model.py`: model-side Python support code.
- `include/dae/`: runtime abstractions such as allocator, launcher, queues, runtime, and virtual cores.
- `include/task/`: CUDA task building blocks including attention, GEMV, RMSNorm, RoPE, SiLU, WGMMA, and argmax.
- `src/runtime.cu`: runtime implementation compiled to `runtime.o`.
- `src/torch_runtime.cu`: Torch extension binding source.

## Operational Notes

- The repository currently includes built extension artifacts under `python/dae/`.
- Full runtime verification may require Hopper-class CUDA hardware.
- For Python-only edits, start with light checks such as `python -m py_compile`.
- `app/python/llama3/sched.py` now includes a `--correctness` mode for a single-token, single-decoding-step validation against `app/python/llama3/reference.py`.
- `python/dae/schedule.py` now treats SM-count placement as a post-construction concern across the main scheduler classes, including `SchedArgmax`.
- `python/dae/launcher.py` and `app/python/llama3/sched.py` now support late-bound barrier counts: barrier ids are still built early, and the llama path now binds selected placement-dependent layer/system barriers from a generic scan of the placed schedule bundle's barrier-releasing memory instructions rather than from a handwritten per-bar table.
- The llama/qwen shared-memory SiLU stages are no longer only inline callables in the app scripts; they now have dedicated schedule classes in `python/dae/schedule.py` for the interleaved phase and the fused register-backed phase.
- `app/python/llama3/sched.py` now follows the newer schedule-construction style: build dependency-only schedules first, attach mostly-static bars immediately after construction, then apply `place(...)` in a grouped step before submission to `dae.i(...)`.
