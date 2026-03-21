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
- When a task reveals a stable repo fact or a reusable procedure, update `agents/knowledge/` or `agents/workflows/` in the same task instead of leaving it only in the change log.
- `make pyext` requires the local CUDA toolkit version to match the CUDA version used by the installed PyTorch build.
- In this environment, `make pyext` failed because the detected CUDA version was `12.5` while PyTorch was built with CUDA `13.0`.
- The benchmark command succeeded against the existing environment and extension artifacts.
- For the isolated `N=8` MMA GEMV path, use `app/python/gemv_mma_out.py` as the dedicated harness instead of modifying `app/python/gemv_out.py`.
- If `make pyext` fails immediately with an unsupported GCC version from the active Conda compiler toolchain, retry from a reset shell state with:

```bash
source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate
make pyext
```

- The MMA GEMV harness defaults to `M=4096`, `K=4096`, and `N=8`, and supports quick smaller checks through `GEMV_M`, `GEMV_K`, and `GEMV_SMS`.

## Last Verified Result

On 2026-03-21:

- `python -m py_compile app/python/gemv_mma_out.py python/dae/launcher.py`: succeeded
- `source /home1/11362/depctg/miniconda3/etc/profile.d/conda.sh && conda deactivate && conda activate && make pyext`: succeeded
- `GEMV_M=64 GEMV_K=256 GEMV_SMS=1 python app/python/gemv_mma_out.py -l`: succeeded with `0.0%` average diff
- `python app/python/gemv_mma_out.py -l`: succeeded with `0.0%` average diff
- `python app/python/gemv_out.py -l`: succeeded

On 2026-03-20:

- `make pyext`: failed due to CUDA version mismatch (`12.5` local toolkit vs `13.0` PyTorch)
- `python app/python/llama3/sched.py -b`: succeeded
- `python app/python/llama3/sched.py --correctness`: succeeded
