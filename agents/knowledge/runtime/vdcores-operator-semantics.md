# VDCores Operator Semantics

This note records the field meanings and state effects of the operators defined in the checked-in runtime sources.

Primary references:

- [include/dae/context.cuh](/home1/11362/depctg/vdcores/include/dae/context.cuh)
- [include/dae/opcode.cuh.inc](/home1/11362/depctg/vdcores/include/dae/opcode.cuh.inc)
- [include/dae/compute_dispatch.cuh](/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh)
- [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh)
- [include/dae/pipeline/ldwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/ldwarp.cuh)
- [include/dae/pipeline/stwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/stwarp.cuh)
- [python/dae/instructions.py](/home1/11362/depctg/vdcores/python/dae/instructions.py)

## Instruction Formats

### `CInst`

- layout:
  - `opcode`
  - `args[0..2]`
- all args are `uint16`

### `MInst`

- layout:
  - `opcode`
  - `size`
  - `num_slots`
  - `arg`
  - `address` or `coords[4]`
- helpers:
  - `nslot()`
    - low 6 bits of `num_slots`
  - `bar()`
    - upper bits of `num_slots`

## Memory-Opcode Flags

The low 6 opcode bits are flags:

- `ALLOCATE`
  - allocator-backed op
- `WRITEBACK`
  - store-side op
- `GROUP`
  - apply `LoopM` resource-group shift to `num_slots/arg`
- `JUMP`
  - after this allocating op, use the active `RepeatM` loop-back path
- `BARRIER`
  - use the global `bars[]` dependency table
- `PORT`
  - route load work to load port `1` instead of `0`

## Memory Operators

The memory-op registry declared in [include/dae/opcode.cuh.inc](/home1/11362/depctg/vdcores/include/dae/opcode.cuh.inc) is:

- `OP_TERMINATE`
- `OP_REPEAT`
- `OP_LOOP`
- `OP_SHIFT_RESOURCE`
- `OP_ISSUE_BARRIER`
- `OP_CC0`
- `OP_CC0_ROW_BYTES`
- `OP_ALLOC_REG_LOAD`
- `OP_ALLOC_TMA_LOAD_1D`
- `OP_ALLOC_TMA_LOAD_TENSOR_1D`
- `OP_ALLOC_TMA_LOAD_2D`
- `OP_ALLOC_TMA_LOAD_3D`
- `OP_ALLOC_TMA_LOAD_4D`
- `OP_ALLOC_TMA_LOAD_5D_FIX0`
- `OP_ALLOC_WB_REG_STORE`
- `OP_ALLOC_WB_TMA_STORE_1D`
- `OP_ALLOC_WB_TMA_STORE_2D`
- `OP_ALLOC_WB_TMA_STORE_3D`
- `OP_ALLOC_WB_TMA_STORE_4D`
- `OP_ALLOC_WB_TMA_STORE_5D_FIX0`
- `OP_ALLOC_WB_TMA_REDUCE_ADD_2D`
- `OP_ALLOC_WB_TMA_REDUCE_ADD_3D`
- `OP_ALLOC_WB_RAW_ADDRESS`

### Control-flow and non-alloc ops

- `OP_TERMINATE`
  - fields:
    - none
  - effect:
    - stop the alloc warp
    - push `SLOT_END` to both load ports

- `OP_REPEAT`
  - fields:
    - `size` = repeat count
    - `address` or `coords` = per-iteration delta
    - low byte of `num_slots` = first lane/register to seed
    - high byte of `num_slots` = end lane/register
  - effect:
    - seed `gpr[0]` and clear `gpr[1]`
    - later allocating `JUMP` instructions consume this repeat state
    - set `loop_counter = size`
    - set `loop_start_pc = pc + 1`
  - registers changed:
    - `loop_counter`
    - `loop_start_pc`
    - selected lanes of `gpr[0]`
    - selected lanes of `gpr[1]`

- `OP_LOOP`
  - fields:
    - `size` = trip count
    - `num_slots` = control lane that owns the loop counter
    - `coords[0]` = jump target pc
    - packed `coords[2:3]` = group-shift increment
  - effect:
    - update `next_pc`
    - update the resource-group `shift`
    - does not allocate or touch shared memory directly
  - registers changed:
    - control lane `jmp_cnt`
    - `next_pc`
    - `shift`

- `OP_ISSUE_BARRIER`
  - fields:
    - encoded barrier id in `num_slots`
  - effect:
    - alloc warp spins until `bars[bar] == 0`

- `OP_CC0`
  - fields:
    - `address` = pointer to token id
    - `arg` = log2(row_bytes)
  - effect:
    - read token id
    - set `loop_counter = 1`
    - set `loop_start_pc = pc + 1`
    - seed lane `0` of `gpr[1] = token << arg`
    - next allocating op sees an address offset for embedding-row selection
  - registers changed:
    - `loop_counter`
    - `loop_start_pc`
    - lane `0` of `gpr[1]`

- `OP_CC0_ROW_BYTES`
  - fields:
    - `address` = pointer to token id
    - `size` = row width in bytes
  - effect:
    - same role as `OP_CC0`, but uses `token * size` for non-power-of-two row widths
    - also sets `loop_counter = 1` and `loop_start_pc = pc + 1`
  - registers changed:
    - `loop_counter`
    - `loop_start_pc`
    - lane `0` of `gpr[1]`

- `OP_SHIFT_RESOURCE`
  - fields:
    - not documented by a checked-in Python wrapper
  - effect:
    - declared in the opcode registry, but there is no checked-in `allocwarp` case for it in this snapshot
  - status:
    - declared but unhandled in the checked-in runtime path

### Load-side allocating ops

- `OP_ALLOC_TMA_LOAD_1D`
  - fields:
    - `size` = byte count
    - `address` = global source pointer
  - effect:
    - allocate normal shared-memory slots
    - async copy global bytes into the slot span
    - publish a ready token to `m2c`

- `OP_ALLOC_TMA_LOAD_TENSOR_1D`
  - fields:
    - `arg` = TMA descriptor index
    - `address` = 1D tensor coordinate
    - `size` = byte count
  - effect:
    - issue 1D descriptor-backed TMA load into shared memory

- `OP_ALLOC_TMA_LOAD_2D`
  - fields:
    - `arg` = descriptor index
    - `coords[0:1]` = tensor coords
    - `size` = byte count
  - effect:
    - issue 2D TMA load

- `OP_ALLOC_TMA_LOAD_3D`
  - fields:
    - `arg` = descriptor index
    - `coords[0:2]`
    - `size`
  - effect:
    - issue 3D TMA load

- `OP_ALLOC_TMA_LOAD_4D`
  - fields:
    - `arg` = descriptor index
    - `coords[0:3]`
    - `size`
  - effect:
    - issue 4D TMA load

- `OP_ALLOC_TMA_LOAD_5D_FIX0`
  - fields:
    - `arg` = descriptor index
    - `coords[0:3]`
    - `size`
  - effect:
    - issue 5D TMA load with hardcoded leading coordinate `0`

Barrier behavior for load ops:

- if `BARRIER` is set and `WRITEBACK` is not set, the load warp waits for `bars[bar] == 0` before issuing the load

### Store-side / writeback ops

- `OP_ALLOC_WB_TMA_STORE_1D`
  - fields:
    - `size`
    - `address` = global destination pointer
  - effect:
    - store warp copies shared-memory slot contents to global memory

- `OP_ALLOC_WB_TMA_STORE_2D`
- `OP_ALLOC_WB_TMA_STORE_3D`
- `OP_ALLOC_WB_TMA_STORE_4D`
- `OP_ALLOC_WB_TMA_STORE_5D_FIX0`
  - fields:
    - `arg` = descriptor index
    - `coords[...]` = tensor destination coordinates
    - `size`
  - effect:
    - store warp issues descriptor-backed TMA store from shared memory

- `OP_ALLOC_WB_TMA_REDUCE_ADD_2D`
- `OP_ALLOC_WB_TMA_REDUCE_ADD_3D`
  - fields:
    - same shape as the matching TMA store ops
  - effect:
    - store warp issues `cp.reduce.async ... add` instead of plain store

Barrier behavior for writeback ops:

- if `BARRIER` is set, the store warp waits for the async store to complete and then decrements `bars[bar]`

### Pseudo-register and raw-address ops

- `OP_ALLOC_WB_REG_STORE`
  - fields:
    - `size` = register id
    - `nslot()` = size of the remembered slot span
  - observed effect:
    - loader computes the allocated slot mask and records it in `regFile[reg_id]`
    - loader also publishes that mask to `m2c` with the sign bit set
  - practical meaning:
    - this op captures a shared-memory slot span so later `RegLoad(reg_id)` can re-emit it without a fresh TMA load

- `OP_ALLOC_REG_LOAD`
  - fields:
    - `size` = register id
    - `nslot()` is encoded as a special slot id in Python
  - observed effect:
    - loader publishes `regFile[reg_id]` to `m2c`
  - practical meaning:
    - reload a previously remembered slot mask

- `OP_ALLOC_WB_RAW_ADDRESS`
  - fields:
    - `nslot()` is a special slot id
    - `arg` is also set to that slot id in Python
    - `address` = raw global pointer
  - observed effect:
    - allocator bypasses normal slot allocation
    - the pointer becomes available through `st_insts[special_slot].address`
  - practical meaning:
    - carry raw global pointers through the VM for compute kernels that write or read global memory directly

## Compute Operators

The compute-op registry declared in [include/dae/opcode.cuh.inc](/home1/11362/depctg/vdcores/include/dae/opcode.cuh.inc) is larger than the checked-in handler set. The checked-in control-flow opcodes are:

- `OP_TERMINATEC`
- `OP_LOOPC`
- `OP_DUMMY`
- `OP_COPY`

## Control ops

- `OP_TERMINATEC`
  - args:
    - none
  - effect:
    - set `finish = true`
    - queue a `0` token to `c2m`
    - write profiling end timestamp

- `OP_LOOPC`
  - args:
    - `args[0]` = repeat count
    - `args[1]` = target pc
  - effect:
    - if `++count < args[0]`, jump to `pc = args[1]`
    - else clear `count`
  - registers changed:
    - `count`
    - maybe `pc`

- `OP_DUMMY`
  - args:
    - `args[0]` = number of input tokens to consume
    - `args[1]` = optional nanosleep delay
  - effect:
    - pop one `m2c` token per iteration
    - optionally sleep
    - return the same token through `c2m`

- `OP_COPY`
  - args:
    - `args[0]` = number of copies
    - `args[1]` = number of `uint32` words to move
  - effect:
    - per iteration:
      - pop source token
      - pop destination token
      - copy shared memory source to shared memory destination
      - queue destination as writeback
      - free source

## GEMV / GEMM families

The checked-in runtime has two declarative compute families in [include/dae/opcode.cuh.inc](/home1/11362/depctg/vdcores/include/dae/opcode.cuh.inc):

- `GEMV_WGMMA`
  - fields:
    - `M`
    - `N`
    - `K`
    - `BLOAD`
    - `RESIDUAL`
  - Python args:
    - `args[0]` = number of K tiles
    - `args[1]` = prefetch distance
  - task behavior from [include/task/gemv.cuh](/home1/11362/depctg/vdcores/include/task/gemv.cuh):
    - optional residual input first if `RESIDUAL=1`
    - then a stream of `A` tiles and periodically reused `B` tiles
    - final output tile is written to a shared-memory output slot and queued as writeback

- `GEMV_MMA`
  - fields:
    - `M`
    - `N`
    - `K`
  - Python args:
    - `args[0]` = number of K tiles
  - task behavior:
    - shared-memory `A/B` tiles in, one shared-memory output tile out

Static GEMM handlers:

- `OP_GEMM_M64N64`
- `OP_GEMM_M64N64K64`
- `OP_GEMM_M64N128K64`
  - args:
    - `args[0]` = number of K tiles
  - effect:
    - consume streams of shared-memory `A/B` tiles
    - accumulate in registers
    - write one shared-memory output tile
    - queue output as writeback

## Attention ops

Shared packing rules from [include/dae/compute_dispatch.cuh](/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh):

- non-split variants:
  - `args[0]` = `num_kv_block`
  - `args[1]`:
    - low 8 bits = `num_active_q`
    - high 8 bits = `last_kv_active_token_len`
  - `args[2]`:
    - bit 0 = `need_norm`
    - bit 1 = `need_rope`

- split variants:
  - `args[0]`:
    - low 12 bits = `num_kv_block`
    - high 4 bits = `split_idx`
  - `args[1]`:
    - low 8 bits = `num_active_q`
    - high 8 bits = `last_kv_active_token_len`
  - `args[2]` = `kv_start_idx`

Operators:

- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim`
- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64`
- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split`
- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA`
- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA`
- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA`
  - effect:
    - optionally pop packed side-input weights/rope row and a K-store output slot
    - pop Q tile
    - per KV block:
      - pop K tile
      - pop V tile
      - run grouped flash-attention update
      - free old K/V tiles as soon as safe
    - free Q and optional side-input
    - queue K-store slot as writeback if fused norm/rope path is active
    - pop O output slot and queue it as writeback
    - split-KV variants also pop a raw-address LSE output slot and write LSE values directly to global memory

- `OP_ATTN_SPLIT_POST_REDUCE`
  - args:
    - `args[0]` = `num_split`
  - effect:
    - pop raw-address LSE buffer
    - pop shared-memory split-output tensor
    - reduce split outputs with LSE-derived weights
    - pop final output slot and queue it as writeback

## Elementwise / reduction ops

- `OP_SILU_MUL_SHARED_BF16_K_4096_INTER`
  - args:
    - `args[0]` = number of active tokens
  - effect:
    - pop output slot, gate slot, up slot
    - compute SiLU(gate) * up in shared memory
    - queue output as writeback
    - free gate and up

- `OP_SILU_MUL_SHARED_BF16_K_64_SW128`
  - args:
    - `args[0]` = number of active tokens
  - effect:
    - same slot protocol, different shared-memory layout

- `OP_RMS_NORM_F16_K_4096_SMEM`
- `OP_RMS_NORM_F16_K_2048_SMEM`
- `OP_RMS_NORM_F16_K_5120_SMEM`
- `OP_RMS_NORM_F16_K_128_SMEM`
  - args:
    - `args[0]` = number of tokens
    - `args[1]` = BF16-encoded epsilon
  - effect:
    - pop weights, input, output shared-memory slots
    - run RMSNorm row by row
    - queue output as writeback
    - free weights and input

- `OP_ARGMAX_PARTIAL_bf16_1152_50688_132`
- `OP_ARGMAX_PARTIAL_bf16_1024_65536_128`
  - args:
    - `args[0]` = number of active tokens
  - effect:
    - pop raw-address input pointer
    - pop raw-address output-value pointer
    - pop raw-address output-index pointer
    - compute per-chunk partial argmax directly in global memory

- `OP_ARGMAX_REDUCE_bf16_1152_132`
- `OP_ARGMAX_REDUCE_bf16_1024_128`
  - args:
    - `args[0]` = number of active tokens
  - effect:
    - pop raw-address partial-values pointer
    - pop raw-address partial-indices pointer
    - pop raw-address final-output pointer
    - compute final argmax directly in global memory

- `OP_ROPE_INTERLEAVE_512`
  - args:
    - none
  - effect:
    - pop RoPE table slot
    - pop input slot
    - pop output slot
    - apply interleaved complex multiply
    - queue output as writeback
    - free input and table

## Coverage Mismatches Worth Knowing

The source tree currently has three layers of operator availability.

### Declared in the opcode registry

Many opcodes are declared in [include/dae/opcode.cuh.inc](/home1/11362/depctg/vdcores/include/dae/opcode.cuh.inc).

### Handled in the checked-in compute dispatch

[include/dae/compute_dispatch.cuh](/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh) only contains a subset of concrete handlers in-tree.

### Actually selected in the current build artifact

[build/generated/dae/selected_compute_ops.inc](/home1/11362/depctg/vdcores/build/generated/dae/selected_compute_ops.inc) currently selects only:

- `OP_COPY`
- `OP_TERMINATEC`

Also, some Python wrappers reference opcode names that are declared but do not have a checked-in handler in the current source snapshot, including:

- `OP_RMS_NORM_F16_K_4096`
- `OP_DEBUG_WGMMA`
- `OP_GEMV_M64_PREFETCH`
- `OP_GEMV_M192`
- `OP_GEMV_M64N8_ROPE_128`
- `OP_WGMMA_M64N256K16_F16`
- `OP_WGMMA_M64N256K16_BF16`

Treat those as source/interface drift until their lowering path is confirmed through generated handlers or updated dispatch code.

There is also a checked-in argument-packing mismatch worth keeping in mind:

- the shared non-split attention handler expects `args[1]` to pack:
  - low 8 bits = `num_active_q`
  - high 8 bits = `last_kv_active_token_len`
- but the Python wrappers:
  - `ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA`
  - `ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA`
  currently pass `last_kv_active_token_len` directly as `args[1]`

So those two MMA wrapper classes should not be treated as field-compatible with the checked-in handler until that packing is reconciled.
