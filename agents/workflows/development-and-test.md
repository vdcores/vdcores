# Development And Test Workflow

Use this as the standard loop when developing VDCores applications.

## When CUDA Runtime Or Kernel Code Changes

If you change CUDA runtime or kernel code, rebuild the extension first.

```bash
make pyext
python app/python/llama3/sched.py -b
```

Typical runtime-touching areas include:

- `src/`
- `include/dae/`
- `include/task/`
- `setup.py`
- `Makefile`

## When Only Python Code Changes

If you only change Python-side code, you can skip the rebuild and run the application directly.

```bash
python app/python/llama3/sched.py -b
```

## Llama Correctness Check

Use this when changing the Python scheduling flow for the Llama 3 demo and you want a correctness-oriented check instead of only launch or benchmark coverage.

```bash
python app/python/llama3/sched.py --correctness
```

Current behavior of `--correctness`:

- runs the single input token / single decoding step path
- launches the DAE schedule
- captures a PyTorch reference pass through `app/python/llama3/reference.py`
- checks exact final-token agreement
- checks selected intermediate tensors against a `5%` mean relative diff budget
- checks final logits against a `10%` mean relative diff budget

Prefer this mode after refactors in:

- `python/dae/schedule.py`
- `python/dae/model.py`
- `app/python/llama3/sched.py`
- `app/python/llama3/reference.py`

Typical Python-only areas include:

- `app/python/`
- `python/dae/`

## Notes

- `-b` is the short form of `--bench` and defaults to one benchmark iteration.
- `app/python/llama3/sched.py --help` still performs model loading before printing usage, so prefer using it sparingly on large checkpoints.
- When a task reveals a stable repo fact or a reusable procedure, update `agents/knowledge/` or `agents/workflows/` in the same task instead of leaving it only in `.agentlog/`.
- When a task changes the standard workflow or exposes a durable verification lesson, persist the concise takeaway in tracked `agents/` docs before closing the task.
- `make pyext` requires the local CUDA toolkit version to match the CUDA version used by the installed PyTorch build.
- Put machine-specific verification outcomes and dated run history in `.agentlog/`, then distill only durable conclusions back into `agents/`.
