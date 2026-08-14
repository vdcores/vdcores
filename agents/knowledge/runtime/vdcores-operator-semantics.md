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
- `OP_ALLOC_ROUTED_TMA_LOAD_1D`
- `OP_ALLOC_LDU_LOAD_1D`
- `OP_ALLOC_WB_STU_STORE_1D`
- `OP_ALLOC_INDEXED_TMA_LOAD_1D`
- `OP_ALLOC_INDIRECT_TMA_LOAD_1D`
- `OP_ALLOC_INDIRECT_LDU_LOAD_1D`
- `OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_1D`
- `OP_ALLOC_INDIRECT_INDEXED_TMA_LOAD_1D`

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
    - `arg == 0` = legacy repeat mode
    - `arg & 0x8000` = counter-derived offset mode
    - `arg & 0x4000` = add the selected counter value to the repeat count
    - `arg & 0x2000` = accumulate into `gpr[1]` instead of replacing it
    - `arg & 0x00ff` = source memory-loop counter lane in counter mode
  - effect:
    - seed `gpr[0]` and clear `gpr[1]`
    - in counter mode, seed `gpr[1] = delta * shfl(jmp_cnt, arg & 0x00ff)`
    - when not in accumulate mode, set `loop_counter = size`, or `size + shfl(jmp_cnt, arg & 0x00ff)` in count-counter mode
    - in accumulate mode, add the selected delta contribution into `gpr[1]` without resetting `loop_counter` or `loop_start_pc`; this supports combined token/block address offsets before one allocating instruction
    - later allocating `JUMP` instructions consume this repeat state
    - `RepeatM.offsetByCounters(...)` keeps the original zero seed active and writes accumulated contributions into the lane selected by the final consumer's `pc - loop_start_pc`
    - set `loop_start_pc = pc + 1` when not in accumulate mode
  - registers changed:
    - `loop_counter`
    - `loop_start_pc`
    - selected lanes of `gpr[0]`
    - selected lanes of `gpr[1]`

- `OP_LOOP`
  - fields:
    - `size` = trip count
    - `num_slots` = control lane whose per-thread `jmp_cnt` owns this loop
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
    - read token id with an L2-cached global load
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
    - reads the token id with the same L2-cached global load as `OP_CC0`
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
- routed/indexed address resolution happens in LDU after this dependency wait; it does
  not require an alloc-warp `OP_ISSUE_BARRIER`

- `OP_ALLOC_LDU_LOAD_1D`
  - synchronously copies small or non-16-byte-aligned metadata from global
    memory into normal shared slots on LDU
- `OP_ALLOC_ROUTED_TMA_LOAD_1D`
  - resolves a top-6 expert field from the HBM routing table and TMA-loads the
    selected payload into normal shared slots
- `OP_ALLOC_INDEXED_TMA_LOAD_1D`
  - reads one runtime row index from a compact HBM record and TMA-loads that row
  - `RepeatM` may advance through 24-byte records for long gather streams

- `OP_ALLOC_INDIRECT_TMA_LOAD_1D`
  - `address` points to one HBM `uint64` source-pointer entry
  - LDU resolves that pointer and bulk-loads `size` bytes into normal slots
- `OP_ALLOC_INDIRECT_LDU_LOAD_1D`
  - same pointer resolution, with the arbitrary-size synchronous LDU copy path
- `OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_1D`
  - `address` points to a two-word HBM descriptor: fixed route-id address and
    current layer `RoutedAddressTable` state address
  - LDU reads the selected expert from the fixed route result, then resolves the
    field from the current layer table
- `OP_ALLOC_INDIRECT_INDEXED_TMA_LOAD_1D`
  - `address` points to one HBM pointer to the ordinary 24-byte indexed record
  - LDU resolves the record before applying its runtime row index

The four `OP_ALLOC_LAYER_*_LOAD_1D` forms have the same LDU behavior as their
indirect counterparts, but the allocator first advances the pointer-column
entry by its current linear layer index (eight bytes, or sixteen bytes for the
routed descriptor). `OP_RESET_INDIRECT_LAYER` resets that allocator-only
index once before a repeated family; the memory `LOOP` advances it once per
logical body. This avoids emitting address arithmetic before every load and
does not expose the scheduler's nested-loop shape in the load ISA.

`OP_LDU_PROFILE_LAYER` is diagnostic control flow rather than a data load. It
is queued on both LDU ports, waits on an attached layer-tail dependency, and
lets port 0 record `globaltimer[event_base + ldu_local_counter]` before
incrementing that counter. The ordinary LDU barrier-reload command records its
own post-reload counter range in tracking builds. Neither operation involves
compute threads, an issue barrier, or a thread fence.

There is deliberately no indirect store opcode: fixed scratch outputs use
fixed addresses and persistent layer caches use regular strided STU addresses.

### Store-side / writeback ops

- `OP_ALLOC_WB_TMA_STORE_1D`
  - fields:
    - `size`
    - `address` = global destination pointer
  - effect:
    - store warp copies shared-memory slot contents to global memory

- `OP_ALLOC_WB_STU_STORE_1D`
  - synchronously stores small or non-16-byte-aligned shared metadata through
    STU

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

### Pseudo-register and legacy raw-address ops

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
    - legacy escape hatch for compute kernels that write or read global memory
      directly; new tasks normally must not use it
    - the opt-in SM100 FP8 compact-scale experiment is a narrow exception:
      compute captures one immutable scale-row pointer, reads one UE8M0 byte
      per K tile, expands it in existing activation-slot padding, and issues
      the TMEM copy itself; the production resident schedule keeps the packed
      LDU path because the compact task did not win latency

- `OP_ALLOC_ROUTED_TMA_LOAD_1D`
  - fields:
    - `nslot()` = payload slot span
    - `address` = HBM routing-state base
    - low three `arg` bits = route rank in `[0,6)`
    - remaining `arg` bits = pointer-table field id
    - `size` = tensor byte count
    - optional input `BARRIER` = route-completion dependency
  - HBM state layout:
    - eight `int32` route-id words (first six valid)
    - `int32` pointer-field stride and expert count
    - two padding words
    - row-major `uint64 [expert, field]` pointer table
  - observed effect:
    - LDU reads the selected expert id and resolved pointer through L2-cached
      global loads
    - LDU copies the selected field into allocator-owned shared slots before
      publishing the mask
    - compute consumes shared payload and returns the mask through `c2m`
  - scheduling rule:
    - the first routed address after a router carries the route dependency;
      later lookups on the same LDU port are ordered behind it

Task handlers must not call `__threadfence()`. Compute-only synchronization
uses compute-group barriers; cross-core visibility is owned by LDU/STU queue
completion and memory barriers.

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
    - `args[2]` = compute loop-counter register id
  - effect:
    - if `++count[args[2]] < args[0]`, jump to `pc = args[1]`
    - else clear `count[args[2]]`
  - registers changed:
    - `count[args[2]]`
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

The native FP8 UMMA path uses the same generated-family mechanism rather than
runtime argument dispatch:

- `FP8_GEMV_UMMA_STREAM_SM100`
  - family fields:
    - `SCALE_PACK` = 1, 2, or 4 adjacent K128 scales per native scale record
    - `OUTPUT_GROUPS` = one or two M128 accumulators sharing one activation
      stream and delayed epilogues
  - Python args:
    - `args[0]` = number of K128 tiles
- `FP8_GEMV_UMMA_SPLITK_SM100`
  - family fields:
    - `SCALE_PACK`
    - `OUTPUT_GROUPS`
    - `REDUCTION_BYTES` = 2 for BF16 or 4 for FP32
  - Python args:
    - `args[0]` = number of K128 tiles in this shard
- `DSV4_FP8_QUANT_UMMA_B_SM100`
  - family fields:
    - `SCALE_PACK`
  - Python args:
    - `args[0]` = number of K128 tiles

Only canonical family instances named in the selected `.ops` manifest receive
opcodes and generated handlers. Each generated handler calls one fixed C++
template directly; scale packing, grouped-row count, and reduction type are
not decoded from `CInst.args` in the persistent kernel.

The shaped routed NVFP4 handler is runtime-packed rather than a generated
family:

- `OP_NVFP4_GEMV_UMMA_K512_FP32_SM100`
  - Python args:
    - `args[0]` = number of K512 stages in `[1,8]`
    - `args[1]` = task-local shared scale-ring stages
    - low eight bits of `args[2]` = adjacent weight stages per M2C load
    - bit eight of `args[2]` = retain the activation slot span after the task
  - input tokens:
    - one 128-byte metadata record containing FP32 alpha and compact SFA/SFB
      base addresses
    - one contiguous activation-data span covering the complete K shard
    - streamed 32-KiB K512 weight-data records
  - effect:
    - copies compact scales through task-local shared/TMEM staging
    - accumulates every K512 stage into one resident FP32 TMEM accumulator
    - drains FP32 once after the full shard and queues the output token
    - normally releases the activation with the final weight record
    - when the retain bit is set, leaves that normal-slot mask allocated so a
      same-LDU `RegLoad` can republish it to the adjacent task; the eventual
      non-retaining consumer must release it

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
    - bit 2 = add a compute loop counter to `last_kv_active_token_len`
    - bit 3 = add a compute loop counter to `num_kv_block`
    - bits 4..7 = counter register id for bit 3
    - bits 8..15 = counter register id for bit 2

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

[build/generated/dae/selected_compute_ops.inc](/home1/11362/depctg/vdcores/build/generated/dae/selected_compute_ops.inc) is generated build state, so re-read it before claiming an op is runnable in the local extension. At integration time on 2026-04-20, the current checkout selected:

- `OP_RMS_NORM_F16_K_4096_SMEM`
- `OP_GEMV_WGMMA__M_64__N_8__K_256__BLOAD_4__RESIDUAL_0`
- `OP_ROPE_INTERLEAVE_512`
- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim`
- `OP_SILU_MUL_SHARED_BF16_K_64_SW128`
- `OP_LOOPC`
- `OP_ARGMAX_PARTIAL_bf16_1024_65536_128`
- `OP_ARGMAX_REDUCE_bf16_1024_128`
- `OP_TERMINATEC`
- `OP_COPY`
- `OP_SILU_MUL_SHARED_BF16_K_4096_INTER`

Also, some Python wrappers reference opcode names that are declared but do not have a checked-in handler in the current source snapshot, including:

- `OP_RMS_NORM_F16_K_4096`
- `OP_DEBUG_WGMMA`
- `OP_GEMV_M64_PREFETCH`
- `OP_GEMV_M192`
- `OP_GEMV_M64N8_ROPE_128`
- `OP_WGMMA_M64N256K16_F16`
- `OP_WGMMA_M64N256K16_BF16`

Treat those as source/interface drift until their lowering path is confirmed through generated handlers or updated dispatch code.

There is also one handler declaration that is not declared in the checked-in opcode registry:

- `OP_GEMV_M64N8_MMA`

Treat the dynamic `GEMV_MMA` family as the current declared MMA GEMV path unless `OP_GEMV_M64N8_MMA` is reintroduced into [include/dae/opcode.cuh.inc](/home1/11362/depctg/vdcores/include/dae/opcode.cuh.inc).
