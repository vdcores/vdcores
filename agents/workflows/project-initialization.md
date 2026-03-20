# Project Initialization Workflow

Use this when an agent is starting work in `vdcores` without prior local context.

1. Check repository state with `git status --short`.
2. Read `README.md` and `AGENTS.md`.
3. Inspect the top-level structure and confirm the main code areas:
   - `app/python/` for runnable experiments and model schedules
   - `app/python/llama3/` for the Llama 3 demo and reference helpers
   - `app/python/qwen3/` for Qwen 3 schedule variants
   - `python/dae/` for the Python package and scheduling interface
   - `include/dae/` and `src/` for the CUDA runtime and Torch extension bindings
   - `include/task/` for task primitives such as attention, GEMV, RMSNorm, RoPE, SiLU, WGMMA, and argmax
   - `tests/` for utility scripts and test assets
4. If build or environment details matter, inspect:
   - `setup.sh` for reference environment setup
   - `Makefile` for `runtime.o` and `make pyext`
   - `setup.py` for the `dae.runtime` extension packaging
5. Create or update agent notes:
   - add a dated file in `agents/changes/`
   - capture durable findings in `agents/knowledge/`
   - refine this workflow when repeated steps become clear
6. Verify with the least expensive meaningful command first.
7. If CUDA or hardware requirements block verification, document that explicitly in the change log.

## Current Environment Facts

- Preferred environment is Conda `base`.
- The reference setup installs CUDA Toolkit 13.0.2.
- The build is currently configured for Hopper `sm_90a`.
- The Python extension package name is `dae`.
