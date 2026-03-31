# LoRA Scheduling Demo Notes

- Demo entry point: `app/python/lora_fixed_rank_demo.py`
- Current scope:
  - one fixed LoRA pipeline
  - fixed rank `64`
  - group sizes `[64, 64, 8, 8]`
  - no padding
- The demo currently builds:
  - `64`-token groups: `SchedGemm` shrink + `SchedGemm` expand
  - `8`-token groups: `SchedGemv` shrink + `SchedGemv` expand
  - one barrier between each shrink/expand pair

## Important Constraint

This first version is intentionally not schedule-optimized yet.

- it uses one fixed SM allocation (`64`)
- it reuses the same SM range for every large-group shrink and expand stage
- it uses a smaller shrink allocation for `8`-token GEMV groups and a larger expand allocation afterward
- it is meant only to validate that a simple grouped LoRA GEMM pipeline launches and matches reference math before adding smarter scheduling

## Timing Hook

- `include/dae/context.cuh` has a compile-time flag `dae2EnableComputeProfile`.
- When enabled, the runtime appends per-SM compute-op durations into `dae.profile[sm, 2:]` in execution order.
- `python/dae/util.py` exposes `read_compute_durations(dae)` to read those per-SM duration lists for post-processing.
