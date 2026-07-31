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

### `CommInst`

- independent 16-byte communication format:
  - `opcode`
  - `size`
  - `arg0`
  - `arg1`
  - `address`
- the first four fields are `uint16`; `address` is `uint64`
- communication opcodes have no memory flags and are consumed only by the
  optional communication warp

### `PoolInst`

- independent 16-byte pool format with the same field widths as `CommInst`;
- stored in a separate instruction array and never decoded by the ordinary
  communication interpreter;
- each wire opcode selects one compile-time execute-warp type during host
  kernel assembly.

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
    - when a schedule adds `WRITEBACK | BARRIER`, compute returns the special
      slot as an ordinary one-hot `c2m` mask; the store warp performs no copy
      and decrements the encoded normal barrier
    - writeback is supported for special slots `24..30`; slots `31..32` are
      input-only because their one-hot values are not positive signed `c2m`
      payloads
  - practical meaning:
    - carry raw global pointers through the VM for compute kernels that write or read global memory directly

## Communication Operators

The registry is `include/dae/communication_opcode.cuh.inc`. Python classes are
`CommunicationInstruction` subclasses. The ordinary runtime exposes the ABI
but rejects nonempty communication streams because it has no consumer warp.

- `COMM_TERMINATE`: stop the communication PC loop.
- `COMM_WAIT_BARRIER`: `size` is a local `bars[]` id; wait for zero.
- `COMM_RECORD_EVENT`: `size` is a per-block profile index; lane 0 records
  `globaltimer`.
- `COMM_NVSHMEM_PUT`:
  - `address` is the same symmetric source/destination address;
  - `size | arg0 << 16` is bytes;
  - `arg1[7:0]` is target PE and `arg1[15:8]` is signal id;
  - warp put-with-signal followed by quiet.
- `COMM_NVSHMEM_WAIT`: `size` is signal id and `address` is the expected value.
- `COMM_MEMORY_POOL_SUBMIT`:
  - `address` is a 128-byte request, `size` its submit signal, `arg0` pool PE;
  - warp put-with-signal and quiet publishes the mailbox.
- `COMM_MEMORY_POOL_WAIT`: `address` is the request whose completion sequence
  is awaited; `arg0` identifies the pool PE, selecting a GPU-scope atomic
  acquire for a local completion or an NVSHMEM wait for a remote completion.
- `COMM_MEMORY_POOL_RUN`:
  - `address` is `MemoryPoolConfig`;
  - `size | arg0 << 16` is the expected completion count;
  - lanes poll distinct mailboxes, ballot ready requests, execute one selected
    request cooperatively, then advance its dependency/completion state.
## Pool Operators

The registry is `include/dae/pool_opcode.cuh.inc`. Pool operators are
`PoolInstruction` subclasses and execute only in a kernel assembly containing
their registered execute-warp type.

- `POOL_SLICE_EXCHANGE` and `POOL_SLICE_WEIGHTED_EXCHANGE` are eight-warp
  `PoolInst` variants:
  - `address` is a `PoolSliceConfig` pointer;
  - `size` is the first source-writer chunk barrier;
  - `arg0` is the first contiguous reader-dispatch barrier;
  - `arg1` is the first contiguous reader-compute barrier;
  - host dispatch instantiates the matching generic or weighted execute-warp
    type and every thread enters it before ordinary VM state is allocated;
  - dispatch queue `DATA` instructions execute as `DynamicRead<Copy>`;
  - for weighted combine, dispatch-derived route metadata is converted locally
    by `RESERVE_ROUTES` into immutable `DynamicRead<ReduceAdd>` plans while
    activation DATA is still in flight; every plan names its ordinary
    compute-barrier subset and static source-row shard, so combine sends no
    second metadata packet;
  - Copy and ReduceAdd enter one compile-time specialized dynamic-read
    executor; there is no transform switch in the PoolInst hot loop;
  - exact per-reader DATA counters release ordinary dispatch barriers before
    unrelated queue `END` instructions retire;
  - it directly PUTs source rows, executes the typed dynamic reads, publishes
    payload-coupled return groups, and source-scatters/reduces the result;
  - a fixed pool assembly contains no compute/memory/communication
    interpreters; a mixed assembly may run them only on other blocks.
- `POOL_SLICE_GIN_WEIGHTED_EXCHANGE` has the same instruction fields,
  metadata ABI, `DynamicRead<Copy>`, and `DynamicRead<ReduceAdd>` semantics as
  the weighted operator. Its execute-warp type is present only in the
  compile-time NCCL GIN assembly; all remotely accessed objects must be views
  of its registered HBM window. The raw build changes dispatch WQE formation,
  not the PoolInst dependency semantics.

Generic dependency semantics are in `memory-pool-protocol.md`. The batched gathered
read ABI, warp roles, and ordering rules are in `pool-slice-dynamic-read.md`
and `vdcores-communication-core.md`.

## Compute Operators

The compute-op registry declared in [include/dae/opcode.cuh.inc](/home1/11362/depctg/vdcores/include/dae/opcode.cuh.inc) is larger than the checked-in handler set. The checked-in control-flow opcodes are:

Task device functions under `include/task/` must never use `noinline`, including
configuration-dependent aliases of it. Keep task entry points force-inlined so
the selected operator specializes into its VDCores interpreter; isolate large
runtime roles through compile-time core assembly rather than device calls.
`tests/test_task_inlining.py` enforces this source contract.

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
