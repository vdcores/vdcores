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

## Launch Hang Detection

When testing a new schedule, a common failure mode is a barrier deadlock: the app prints `[launch]` and then never makes forward progress. Use the timeout wrapper in `tests/script/run_with_launch_timeout.py` to catch that class of bug quickly.

```bash
python tests/script/run_with_launch_timeout.py \
  --post-launch-timeout 60 \
  --post-launch-idle-timeout 20 \
  -- python app/python/llama32_1b/sched.py --correctness
```

Notes:

- The wrapper starts its timer only after it sees the launch marker, which avoids treating checkpoint download or model loading as hangs.
- It forwards child output live and prints the recent output tail on timeout.
- A timeout after `[launch]` is a strong hint that a barrier count or dependency release is missing.

## Stage Schedule Debugging

When a new schedule hangs, do not debug all layers at once. Narrow it in this order:

```bash
python tests/script/run_with_launch_timeout.py \
  --post-launch-timeout 120 \
  --post-launch-idle-timeout 20 \
  -- python app/python/llama32_1b/sched.py \
    --debug-num-layers 1 \
    --debug-stop-after q_rope
```

Then expand gradually:

- first one layer, operator by operator with `--debug-stop-after`
- then one full layer
- then two layers
- only after those pass, restore the full layer count

This catches two common bugs quickly:

- a missing release or wait that deadlocks immediately after one operator is added
- an implicit dependency that only breaks after repartitioning work onto disjoint SM ranges

Typical Python-only areas include:

- `app/python/`
- `python/dae/`

For fused task work, reread [agents/knowledge/task-queue-semantics.md](/home1/11362/depctg/vdcores/agents/knowledge/task-queue-semantics.md) before changing queue order or adding side inputs. The `m2c.pop()` order must match the schedule's emitted memory order.

## Notes

- `-b` is the short form of `--bench` and defaults to one benchmark iteration.
- `app/python/llama3/sched.py --help` still performs model loading before printing usage, so prefer using it sparingly on large checkpoints.
- When a task reveals a stable repo fact or a reusable procedure, update `agents/knowledge/` or `agents/workflows/` in the same task instead of leaving it only in `.agentlog/`.
- When a task changes the standard workflow or exposes a durable verification lesson, persist the concise takeaway in tracked `agents/` docs before closing the task.
- `make pyext` requires the local CUDA toolkit version to match the CUDA version used by the installed PyTorch build.
- In this environment, `make pyext` failed because the detected CUDA version was `12.5` while PyTorch was built with CUDA `13.0`.
- The benchmark command succeeded against the existing environment and extension artifacts.
- For the isolated `N=8` MMA GEMV path, use `app/python/gemv_mma_out.py` as the dedicated harness instead of modifying `app/python/gemv_out.py`.
- If `make pyext` fails immediately with an unsupported GCC version from the active Conda compiler toolchain, retry from a reset shell state with:
  This repo already verified this exact recovery path on 2026-03-21; prefer it over ad hoc `nvcc` flag changes.

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate
conda activate
make pyext
```

- The MMA GEMV harness defaults to `M=4096`, `K=4096`, and `N=8`, and supports quick smaller checks through `GEMV_M`, `GEMV_K`, and `GEMV_SMS`.
- Before performance benchmarking `app/python/llama32_1b/sched.py`, clear leftover Python workers with `killall python || true`; stale decode jobs can distort both timing and apparent correctness.
- Never run GPU performance benchmarks in parallel. For this repo, credible timing comes from sequential runs only; overlapping jobs contend for the device and can corrupt both throughput numbers and debugging conclusions.
- For risky multi-token or partial-stage experiments on the Llama 3.2 1B path, prefer `tests/script/run_with_launch_timeout.py` first to separate deadlocks from slow-but-completing schedules.
- For longer multi-token timing on the current Llama 3.2 1B branch, prefer fresh-process `-b 1` measurements over larger `-b` counts until launch-state reset behavior is audited; repeated launches in one process showed inconsistent timings.

## Last Verified Result

On 2026-03-21:

- `python -m py_compile app/python/gemv_mma_out.py python/dae/launcher.py`: succeeded
- `python -m py_compile app/python/llama32_1b/sched.py python/dae/launcher.py`: succeeded
- `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate && make pyext`: succeeded
- `GEMV_M=64 GEMV_K=256 GEMV_SMS=1 python app/python/gemv_mma_out.py -l`: succeeded with `0.0%` average diff
- `python app/python/gemv_mma_out.py -l`: succeeded with `0.0%` average diff
- `python app/python/gemv_out.py -l`: succeeded
- `python tests/script/run_with_launch_timeout.py --post-launch-timeout 20 --post-launch-idle-timeout 10 -- python app/python/llama32_1b/sched.py -N 2 -l`: succeeded
- `python tests/script/run_with_launch_timeout.py --post-launch-timeout 20 --post-launch-idle-timeout 10 -- python app/python/llama32_1b/sched.py -N 2 --debug-num-layers 8 -l`: succeeded

On 2026-03-20:

- `make pyext`: failed due to CUDA version mismatch (`12.5` local toolkit vs `13.0` PyTorch)
- `python app/python/llama3/sched.py -b`: succeeded
- `python app/python/llama3/sched.py --correctness`: succeeded
- Put machine-specific verification outcomes and dated run history in `.agentlog/`, then distill only durable conclusions back into `agents/`.
