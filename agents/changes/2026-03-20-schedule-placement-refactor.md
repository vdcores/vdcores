## Purpose

Refactor `python/dae/schedule.py` so schedule objects are created dependency-first, then placed later with `place(num_sms, base_sm=0)`. Add `ListSchedule` for split-based composition and update the llama path to use the new placement flow.

## Files Touched

- `python/dae/schedule.py`
- `python/dae/model.py`
- `app/python/llama3/reference.py`
- `app/python/llama3/sched.py`
- `app/python/qwen3/sched.py`
- `app/python/rmsnorm.py`
- `app/python/gemv_mlp_nosync.py`
- `app/python/gemv_mlp_mixed.py`
- `app/python/gemv_kvq_proj.py`
- `app/python/gemv_mlp_silu_no_cric_sync.py`
- `app/python/gemv_rope.py`
- `agents/workflows/development-and-test.md`
- `agents/workflows/project-initialization.md`
- `agents/knowledge/project-map.md`

## What Changed

- Moved shared SM placement and generic barrier state into the base `Schedule` class.
- Added `Schedule.place(...)` and role-based `Schedule.bar(...)`.
- Added `ListSchedule` to wrap split/composed schedules while forwarding boundary barriers and placement.
- Added optional `ListSchedule.warn_on_boundary_bars(...)` support so list-level boundary-only barrier forwarding can emit a warning when desired.
- Refactored the main schedule subclasses used by the llama/qwen demos to build dependencies first and place later.
- Updated llama and qwen demo schedule construction to use `.place(...)`.
- Updated a small set of direct-constructor example scripts to use `.place(...)` so they still match the refactored scheduler API.
- Removed the temporary compatibility barrier aliases from `Schedule` and updated remaining call sites to use `bar("role", ...)`.
- Added a llama-only `--correctness` mode that forces the single-token / single-decoding-step flow, captures a PyTorch reference pass, and checks selected intermediate/final tensors against threshold budgets plus exact final-token agreement.
- Logged the new correctness-test workflow in `agents/workflows/development-and-test.md` and recorded the llama correctness mode in `agents/knowledge/project-map.md`.
- Added an explicit reminder in the reusable workflow docs to record new stable project knowledge and reusable procedures as they are discovered.
- Removed constructor-owned `num_sms` from `SchedArgmax` and moved it to `place(...)` like the other main schedule classes.
- Replaced the llama/qwen inline `silu1` and `silu_fused` scheduling callables with dedicated schedule classes in `python/dae/schedule.py`.

## Verification

- `python -m py_compile python/dae/schedule.py python/dae/model.py app/python/llama3/sched.py app/python/qwen3/sched.py`
- `python -m py_compile python/dae/schedule.py python/dae/model.py app/python/llama3/sched.py app/python/qwen3/sched.py app/python/rmsnorm.py app/python/gemv_mlp_nosync.py app/python/gemv_mlp_mixed.py app/python/gemv_kvq_proj.py app/python/gemv_mlp_silu_no_cric_sync.py app/python/gemv_rope.py`
- `python app/python/llama3/sched.py --launch`
- `python -m py_compile app/python/llama3/reference.py app/python/llama3/sched.py`
- `python app/python/llama3/sched.py --correctness`
- `python -m py_compile python/dae/schedule.py app/python/llama3/sched.py app/python/qwen3/sched.py app/python/argmax.py`
- `python app/python/llama3/sched.py --correctness`

## Notes

- The llama demo completed schedule construction and launch successfully in this environment.
- The correctness mode passed with exact final-token match and the configured intermediate/logit thresholds.
- I left unrelated existing workspace changes untouched.
