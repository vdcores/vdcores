# VDCores Queue Notes

These notes summarize the queue/runtime behavior that mattered while debugging fused Qwen3 decode attention.

## Entry Points

- Runtime interpreter: [include/dae/dae2.cuh](/home1/11362/depctg/vdcores/include/dae/dae2.cuh)
- Queue definitions: [include/dae/queue.cuh](/home1/11362/depctg/vdcores/include/dae/queue.cuh)
- Alloc warp: [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh)
- Store warp: [include/dae/pipeline/stwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/stwarp.cuh)
- Attention task: [include/task/attention.cuh](/home1/11362/depctg/vdcores/include/task/attention.cuh)

## Core Queues

- `m2c`:
  - memory-to-compute queue
  - compute pops slot ids from here when data is ready in shared memory
- `c2m`:
  - compute-to-memory queue
  - compute uses this to either free slots or request writeback
- `m2ld`:
  - memory-to-load pipeline queue for the load warps

## The Important Rule

- Popping a slot from `m2c` is not enough.
- Every popped slot must eventually be returned through `c2m`, either:
  - as a plain release for temporary data, or
  - as a writeback request for data that must be stored to global memory

If a path pops extra slots and forgets to return them, the allocator eventually stalls and the launch often looks like a barrier deadlock.

## How To Return Slots

- Temporary input slots:
  - `c2m.push(thread_id, slot_id)`
- Writeback slots:
  - `c2m.template push<0, true>(thread_id, slot_id)`

The writeback form is required for slots that correspond to `OP_ALLOC_WB_*` memory instructions. Without it, the slot may be freed without the store warp executing the actual global write.

## Why This Matters In Attention

- Fused Qwen3 decode attention introduced extra aux inputs:
  - Q-norm weights
  - K-norm weights
  - RoPE row
  - later, a packed side-input slot replaced the three separate aux loads
- Each aux slot popped by attention had to be returned through `c2m`.
- The transformed-K cache write slot had to use the writeback path, not the plain free path.

## Store Warp Behavior

- `stwarp_execute_singlethread(...)` in [stwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/stwarp.cuh) reads writeback requests from `c2m`.
- It executes the actual:
  - `cp_async_bulk` for 1D stores
  - `cp.async.bulk.tensor.*` for descriptor-backed TMA stores
- After the store completes, it frees the slot.

## Deadlock Debugging Heuristic

- If the app prints `[launch]` and then goes idle:
  - first suspect a queue protocol bug or barrier release mismatch
  - especially check whether every new `m2c.pop()` has a matching `c2m` return
- This was the right diagnosis for the earlier fused Qwen3 attention deadlock.

## TMA/Descriptor Lessons

- A descriptor-backed writeback path is usually cleaner than a raw-address side channel when data should land in an existing scheduled tensor.
- For BF16 `tensor1d` descriptors in the current path, the effective `cord(...)` offset that worked for head placement was in BF16-element units, not raw bytes.
- A wrong unit here causes regular placement errors such as writes landing on alternating heads.
