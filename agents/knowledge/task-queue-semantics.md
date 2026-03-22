# Task Queue Semantics

Use this note before changing any task kernel in `include/task/` or the memory instruction order emitted from `python/dae/schedule.py`.

## Core Contract

- The task ABI is queue-ordered: compute-side consumers see shared-memory allocations strictly in `m2c.pop()` order.
- That order is defined by the schedule's emitted allocating memory instructions for the same SM.
- Every task kernel in `include/task/` hard-codes its expected input order by the sequence of `m2c.pop()` calls, so schedule edits and kernel edits must stay in lockstep.

## Runtime Model

- `include/dae/queue.cuh` implements `m2c` and `c2m` as barrier queues. `m2c.pop()` waits for the producing memory warp, then advances the consumer pointer.
- `c2m.push(...)` has two distinct meanings in `SizeBoundedBarrierAllocQueue`:
  - default `push(...)`: frees one or more shared-memory slots back to the allocator
  - `push<..., true>(...)`: publishes a writeback-complete slot to the memory side
- Slot values are bitmasks for regular shared-memory slots. Helpers such as `extract(...)` convert a one-hot slot mask back into a slot index when a task needs the shared-memory address.

## Two Input Families

- Regular TMA/shared-memory allocations flow through the allocator and therefore through `m2c` and `c2m`.
- `RawAddress(...)` uses special slots in `st_insts` instead of allocator-managed shared-memory slots.
- A task that calls `slot_2_glob_ptr(st_insts, slot)` is consuming a predeclared raw global pointer, not a shared-memory queue slot.
- A fused task may mix both styles, but only allocator-managed shared-memory objects participate in the `m2c` and `c2m` slot lifecycle.

## Patterns Confirmed Across Tasks

- `include/task/gemv.cuh`, `include/task/wgmma.cuh`, `include/task/rope.cuh`, `include/task/rms_norm.cuh`, and `include/task/silu.cuh` all follow the same rule: pop inputs in schedule order, then free or write back only after the last use.
- Output buffers are usually popped last and returned with `c2m.push<..., true>(...)` after the final store into shared memory or global memory.
- Read-only shared-memory inputs are typically returned with plain `c2m.push(...)` after the next stage has consumed them.
- Some tasks return multiple read-only inputs as a combined bitmask such as `slot_a | slot_b`; that is still one allocator release event covering both slots.
- Raw-address-only tasks such as the global RMSNorm, SiLU, and argmax paths still use `m2c.pop()` to receive special-slot ids, but they do not free those ids back through allocator-style `c2m.push(...)`.

## Attention Is One Instance

- `include/task/attention.cuh` follows the same contract rather than defining a special one.
- In `SchedAttentionDecoding.schedule(...)`, the current queue order is:
  1. optional side-input TMA when `use_tma_side_input=True`
  2. `Q`
  3. streamed `K` blocks
  4. streamed `V` blocks
  5. output store slot
- The attention kernel returns side input, `Q`, and previous `K` and `V` slots only after their last logical use, then publishes the output slot with writeback semantics.

## Editing Rule Of Thumb

- When a schedule inserts, removes, or reorders a memory instruction that allocates shared memory, update the corresponding task's `m2c.pop()` sequence and final `c2m.push(...)` timing in the same change.
- When debugging bad dataflow, inspect both sides:
  - schedule emission order in `python/dae/schedule.py`
  - task consumption and release order in the relevant `include/task/*.cuh`
