# LoRA Scheduling Demo Notes

- Demo entry point: `app/python/lora_fixed_rank_demo.py`
- Current scope:
  - LoRA update path only
  - fixed rank `64`
  - fixed `feat_out = 4096`
  - adapter groups are already contiguous in the batch
- The demo compares:
  - `baseline`: equal SM split across adapter groups with a fixed `split_M(...)` policy derived from that equal split
  - `adaptive`: group-size-aware SM allocation plus a matching `split_M(...)` choice per group

## Important Constraint

The current GEMV instruction set does not expose a direct fixed-rank LoRA second stage for `rank = 64`.

- `Gemv_M64N8` uses `K = 256`
- `Gemv_M128N8` uses `K = 128`
- there is no current GEMV atom with `K = 64`

Because of that, the demo precomposes each adapter's LoRA delta weight `AB` on the Python side and schedules the LoRA-only update as grouped GEMVs over the adapter-local delta weights.

This keeps the tutorial focused on scheduling:

- different adapter groups still map to different discontiguous weight tensors
- same-adapter requests still stay adjacent
- the schedule difference comes from SM partitioning, `split_M(...)`, and concurrent GPU space sharing across groups
