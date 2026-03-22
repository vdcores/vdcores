# Attention Queue Semantics

Use this note before debugging `include/task/attention.cuh` or changing the attention schedule wiring.

## Queue Ordering

- The compute side consumes memory results strictly in `m2c.pop()` order.
- That order is the same order the schedule emitted allocating memory instructions for that SM.
- If a schedule inserts a new TMA load before `loadQ`, the compute kernel must pop that side-input slot before popping `Q`.

## What Does And Does Not Use The Queue

- Regular TMA/shared-memory allocations flow through the allocator, `m2c`, and `c2m`.
- `RawAddress(...)` does not flow through `m2c`; it writes a special-slot `MInst` into `st_insts` and kernels read it through `slot_2_glob_ptr(...)`.
- This means a fused kernel can mix both styles:
  - queued shared inputs via `m2c.pop()`
  - raw global pointers via `st_insts`

## Decode Attention Ordering

For the single-token decode path in `python/dae/schedule.py`, the intended memory order is:

1. optional side-input TMA, such as the current RoPE row
2. `Q`
3. repeated streamed `K` blocks
4. repeated streamed `V` blocks
5. output store slot

The compute kernel should return slots only after the last use:

- side-input slot: after all Q/K transforms are finished
- `Q` slot: after all KV blocks are finished
- previous `K` / `V` slots: after the next block has become current
- output slot: after the final writeback copy into shared memory

## K/V Cache Expectations

- The K/V cache store path should update only the current token position, not an entire logical KV tile.
- For Qwen single-token bring-up, validate this directly against `attnKs[..., token_pos, :]` and confirm later positions remain zero/untouched.
- The fused attention kernel should operate on the loaded shared-memory K tile for the current SM's head; it must not mutate the global K cache as part of attention.
