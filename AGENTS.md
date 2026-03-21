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
- Reference bootstrap is in `setup.sh`.
- Typical dependencies: CUDA Toolkit 13.0.x, PyTorch CUDA 13.0 wheel, `numpy`, `transformers`, and `accelerate`.
- Current build targets Hopper (`sm_90a`) in both `Makefile` and `setup.py`.
- The project builds a CUDA extension via `make pyext`, linking `src/torch_runtime.cu` and `runtime.o`.

## Agent Workspace

- Store concise in-tree agent summaries under `agents/`.
- Store task-by-task working logs under the gitignored `.agentlog/` directory.
- Use `agents/workflows/` for durable, reusable procedures that future agents should follow.
- Use `agents/knowledge/` for concise project knowledge captured from repo inspection, such as structure maps, entry points, and distilled lessons worth keeping in-tree.
- Keep agent docs short, factual, and tied to concrete files or commands.

## Working Rules

- Check the repo state before editing so unrelated user changes are not described or overwritten.
- Keep edits focused and fast when the task is a single requested implementation.
- Prefer existing local reference helpers, especially under `app/python/llama3/`, over re-deriving model math.
- Use `.agentlog/` for detailed task logs, verification notes, and environment blockers that do not belong in version control.
- Use `agents/workflows/` to log durable project-specific workflow knowledge.
- Update `agents/knowledge/` when you learn something structural or high-signal that will help later tasks.
- Persist important changes and newly learned durable knowledge back into the tracked `agents/` docs; do not leave that context only in `.agentlog/`.

## Initialization Workflow

When starting fresh in this repository:

1. Check repo state with `git status --short` before editing.
2. Read `README.md` and `AGENTS.md` to understand the runtime model, supported environment, and agent expectations.
3. Inspect the relevant layout before making assumptions:
   `app/python/`, `python/dae/`, `include/dae/`, `include/task/`, `src/`, and `tests/`.
4. If the task touches setup or build behavior, also inspect `Makefile`, `setup.py`, and `setup.sh`.
5. Create or update agent notes:
   - add a dated entry under `.agentlog/` for task-specific actions, verification, and blockers;
   - add or refine reusable guidance in `agents/workflows/`;
   - add structural or distilled high-signal notes in `agents/knowledge/` when new project understanding is gained;
   - summarize important changes and durable takeaways into tracked `agents/` files during the same task.
6. Verify with the lightest meaningful check first and record any environment limitations in `.agentlog/`.

## Development And Test

- Standard workflow is documented in `agents/workflows/development-and-test.md`.
- If CUDA runtime or kernel code changes, run `make pyext` before the Python app.
- If only Python code changes, you can skip `make pyext` and run the target application directly.
- For the Llama3 benchmark path, use `python app/python/llama3/sched.py -b`.
- Rebuilds require the local CUDA toolkit to match the CUDA version used by the installed PyTorch build.

## Logging Expectations

- Name `.agentlog/` entries with an ISO date prefix when possible, for example `2026-03-20-project-initialization.md`.
- Each `.agentlog/` entry should capture: purpose, files touched, commands run for verification, and blockers or assumptions.
- Workflow docs should describe repeatable steps, not one-off task history.
- Knowledge docs should summarize stable facts and distilled lessons that are worth keeping in-tree, and point to concrete entry files.
- If a task produces an important repo change or durable lesson, record a concise persistent summary in `agents/workflows/` or `agents/knowledge/` instead of leaving it only in local logs.

## Verification

- Use the lightest meaningful check first, such as `python -m py_compile ...` for Python edits.
- For schedule scripts like `app/python/llama3/sched.py`, full verification may require a machine with CUDA access.
- If a runtime check fails for environment reasons, record that clearly in `.agentlog/` rather than guessing.
