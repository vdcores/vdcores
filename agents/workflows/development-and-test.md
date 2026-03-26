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

## Notes

- `-b` is the short form of `--bench` and defaults to one benchmark iteration.
- `app/python/llama3/sched.py --help` still performs model loading before printing usage, so prefer using it sparingly on large checkpoints.
- When a task reveals a stable repo fact or a reusable procedure, update `agents/knowledge/` or `agents/workflows/` in the same task instead of leaving it only in `.agentlog/`.
- When a task changes the standard workflow or exposes a durable verification lesson, persist the concise takeaway in tracked `agents/` docs before closing the task.
- `make pyext` requires the local CUDA toolkit version to match the CUDA version used by the installed PyTorch build.
- To build only a subset of compute-warp operators, pass `DAE_COMPUTE_OPS=OP_A,OP_B,...` to `make pyext`; leaving it unset keeps the full supported compute-op set.
- As a file-based alternative, put one operator symbol per line in a repo-root `dae_compute_ops.vdcore.build` file, or point `DAE_COMPUTE_OPS_FILE` at another file; the build prints which source it used.
- To generate that file from a built launcher without hand-copying names, use the app-level `--write-compute-ops [path]` option exposed through `python/dae/util.py`'s `dae_app(...)`.
- A clean `make pyext` generates `build/generated/dae/selected_compute_ops.inc`, `build/generated/dae/compute_opcode_order.inc`, and `build/generated/dae/dynamic_compute_handlers.inc` in the Makefile before compiling `runtime.o`; `setup.py` consumes those generated includes but does not create them.
- For family-backed compute instructions such as the new canonical GEMV strings, `-w` writes the family string directly; after changing that file you must rebuild `dae.runtime` before those instructions can serialize successfully, because opcode resolution happens lazily through the rebuilt `runtime.opcode` export.
- For GEMV-family support specifically, `include/dae/opcode.cuh.inc` now declares only family rules, not checked-in GEMV instances. The concrete canonical GEMV strings still come from Python or `-w`, and `make pyext` materializes only those requested instances into the generated opcode/handler includes before the unchanged Python GEMV wrappers can run.
- When adjusting a dynamic compute family, keep the declarative `DAE_DEFINE_COMP_FAMILY(...)` entry in `include/dae/opcode.cuh.inc` as the source of truth for both field order and numeric constraints. Runtime Python should read those definitions from `dae.runtime.compute_family_specs`, while build-time tools should reuse the shared parser helpers rather than adding a second mangling/parsing grammar.
- `make pyext` should now print the `[compute-ops] ...` summary once per build; if it starts repeating, check whether multiple generated include targets are invoking the generator separately instead of through the shared stamp target.
- In this environment, `make pyext` failed because the detected CUDA version was `12.5` while PyTorch was built with CUDA `13.0`.
- The benchmark command succeeded against the existing environment and extension artifacts.
- For the isolated `N=8` MMA GEMV path, use `app/python/gemv_mma_out.py` as the dedicated harness instead of modifying `app/python/gemv_out.py`.
- For isolated decode-attention kernel changes, use `app/python/attention_simple_decoding.py` as the primary correctness and quick-timing harness; it exercises the shared attention opcode path without requiring a full model schedule.
- Set `ATTENTION_IMPL=mma` when you want that harness to exercise the explicit non-Hopper MMA attention opcodes; leave it unset (or use `ATTENTION_IMPL=hopper`) to stay on the default Hopper GMMA path.
- `app/python/attention.py` currently calls the attention instruction with a stale constructor signature and is not a reliable smoke test until that script is updated.
- If `make pyext` fails immediately with an unsupported GCC version from the active Conda compiler toolchain, retry from a reset shell state with:

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
