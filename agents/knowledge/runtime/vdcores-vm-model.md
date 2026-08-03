# VDCores VM Model

This note distills the checked-in runtime into a VM-style model. It is based on:

- [include/dae/context.cuh](/home1/11362/depctg/vdcores/include/dae/context.cuh)
- [include/dae/dae2.cuh](/home1/11362/depctg/vdcores/include/dae/dae2.cuh)
- [include/dae/virtualcore.cuh](/home1/11362/depctg/vdcores/include/dae/virtualcore.cuh)
- [include/dae/queue.cuh](/home1/11362/depctg/vdcores/include/dae/queue.cuh)
- [include/dae/allocator.cuh](/home1/11362/depctg/vdcores/include/dae/allocator.cuh)
- [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh)
- [include/dae/pipeline/ldwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/ldwarp.cuh)
- [include/dae/pipeline/stwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/stwarp.cuh)
- [include/dae/compute_dispatch.cuh](/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh)
- [python/dae/instructions.py](/home1/11362/depctg/vdcores/python/dae/instructions.py)
- [python/dae/launcher.py](/home1/11362/depctg/vdcores/python/dae/launcher.py)

## Runtime Shape

- One `dae2` block runs per SM.
- Each block has `256` threads:
  - `4` compute warps, threads `0..127`
  - `4` memory-side warps, threads `128..255`
- Current fixed configuration from [include/dae/context.cuh](/home1/11362/depctg/vdcores/include/dae/context.cuh):
  - `24` normal shared-memory slots
  - `9` special slots
  - slot size `8 KiB`
  - `512` instructions per SM when `dae2LoadInstructions=true`
  - `1024` TMA descriptors max
  - `1024` global barrier ids max

## Per-SM State

Each SM/block owns:

- Shared copies of the compute and memory instruction streams
- `st_insts[numSlots + numSpecialSlots]`
  - the allocated-memory instruction table
  - entry `st_insts[slot]` is the metadata for that live slot
- `slot_avail`
  - a shared `uint32` bitmap
  - bit `1` means the corresponding normal slot is free
- Dynamic shared memory, aligned to `1 KiB`
  - normal slot `s` starts at `smem_base + s * 8 KiB`
- `scratch_space[32]`
  - shared scratch used by some compute ops such as argmax

The kernel also accepts an optional process-wide `uint64_t*` signal array and
forwards it unchanged to each alloc warp. No checked-in memory opcode consumes
that pointer yet.

## Virtual Cores

The runtime is easiest to view as one compute VM plus one memory VM per SM.

### Compute VM

The compute side is a simple PC loop over `CInst`:

- registers:
  - `pc`
  - `count[]`
  - `finish`
- dispatch:
  - `dispatch_compute_instruction(...)` in [include/dae/compute_dispatch.cuh](/home1/11362/depctg/vdcores/include/dae/compute_dispatch.cuh)
- active width:
  - most kernels expect the full 128-thread compute group

Control-flow detail:

- fetch is effectively:
  - `inst = cinsts[pc]`
  - `pc = pc + 1`
- `OP_LOOPC` then optionally overwrites that post-incremented `pc`
  - `inst.args[2]` selects the counter register
  - if `++count[inst.args[2]] < inst.args[0]`, set `pc = inst.args[1]`
  - else set `count[inst.args[2]] = 0`

Practical convention:

- counter `0` is used for the per-layer compute loop
- counter `1` is used by Llama control-flow decode for the top-level token loop
- counter `2` is used by Llama full-KV-block decode for the outer block loop
- non-split decode attention may add selected counter values to `last_kv_active_token_len` and `num_kv_blocks`

### Memory VM

The memory side is split into four warp roles:

- alloc warp:
  - interprets `MInst`
  - allocates slots
  - writes `st_insts`
  - routes work to load ports
- store warp:
  - consumes writeback requests from compute
  - performs TMA/global stores
  - frees slots after completion
- two load warps:
  - one per async load port
  - execute TMA/global-load side effects
  - publish ready tokens to compute

The alloc warp carries the explicit memory-VM register state in [include/dae/virtualcore.cuh](/home1/11362/depctg/vdcores/include/dae/virtualcore.cuh):

- `gpr[0]`
  - per-lane delta register
- `gpr[1]`
  - per-lane accumulator register
- `slot_alloc`
  - most recent allocated lead slot
- `loop_counter`
  - repeat counter for zero-overhead repeating alloc ops
- `jmp_cnt`
  - per-thread counter for `OP_LOOP`
- `loop_start_pc`
  - repeat-loop body start
- `port`
  - selected load port, `0` or `1`
- predicates:
  - `pred_continue`
  - `pred_jump`
  - `pred_allocate`

The alloc-warp interpreter also keeps three local control variables in [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh):

- `pc`
  - current memory instruction index
- `next_pc`
  - pc for the next fetch
- `shift`
  - the packed resource-group increment later consumed by `GROUP`

## Queue Model

The live protocol centers on three queues.

### `m2c`

- Type: `SizeBoundedBarrierQueue<int, 32>`
- Direction: memory to compute
- Payload:
  - usually a slot mask for normal slots
  - sometimes a special-slot id or a register-restored mask
- Meaning:
  - the data behind this token is ready for compute consumption

### `m2ld[2]`

- Type: `SizeBoundedBarrierQueue<int, 32>`
- Direction: alloc warp to load warp
- Payload:
  - packed `LdCmd { slot, bar, opcode }`
- Meaning:
  - issue the load-side behavior for `st_insts[slot]` on port `0` or `1`

### `c2m`

- Type: `SizeBoundedBarrierAllocQueue<32>`
- Direction: compute to store/free path
- Two modes:
  - plain `push(...)`:
    frees slots immediately by OR-ing the returned mask into `slot_avail`
  - `push<..., true>(...)`:
    queues a writeback/completion token for the store warp

## Slot Encoding

There are two token shapes in flight.

### Normal slot token

- A normal allocation returns a contiguous slot mask.
- Example:
  - a 2-slot allocation at lead slot `5` becomes bits `5` and `6`
- Consumers that need a base slot call `extract(mask)`.

### Special slot token

- Requests with `nslot >= numSlots` bypass the shared-memory allocator.
- The returned value is used directly as a special-slot index into `st_insts`.
- This is how raw global pointers and some register/address carriers travel through the VM.

Compiled-mode note:

- `st_insts[]` is semantic slot metadata, not a mandatory write on every compiled alloc step.
- In interpreted alloc, normal shared-memory producers still materialize the full `MInst` into `st_insts[lead_slot]` because later memory-side stages consume that metadata generically.
- In compiled mode, a producer only needs to write the `st_insts[slot]` fields that some later compiled path still reads.
- In the current compiled support set, only `OP_ALLOC_WB_RAW_ADDRESS` still requires `st_insts[slot].address`, because the store path recovers the raw global pointer through `slot_2_glob_ptr(st_insts, slot)`.
- Ordinary compiled shared-memory producers and current reg-carrier paths do not need a full `st_insts` materialization once their consumers are lowered from the frozen compiled spec.

## Allocator Model

The active allocator is [SharedMemoryAllocator](/home1/11362/depctg/vdcores/include/dae/allocator.cuh):

- storage:
  - `slot_avail` in shared memory
- policy:
  - contiguous allocation only
  - first lead lane whose `req`-bit mask fits wins
- effect:
  - winning lead lane clears those bits with `atomicAnd`
  - the alloc warp stores the full `MInst` into `st_insts[lead_slot]`

Normal-slot lifetime:

1. alloc warp allocates a slot span
2. alloc warp writes `st_insts[lead_slot]`
3. load warp performs the load-side effect
4. load warp publishes a ready token to `m2c`
5. compute pops the token
6. compute must return it through `c2m`
7. either:
   - it is freed immediately, or
   - the store warp writes it back and then frees it

Important rule:

- Every `m2c.pop()` for a normal shared-memory token must eventually be matched by a `c2m` return.
- Missing returns stall `slot_avail` and eventually deadlock the schedule.

## Group And Accumulate Registers

The runtime has two distinct address/resource stepping mechanisms.

### `OP_REPEAT` plus `JUMP`

This is the address-accumulation path.

- `OP_REPEAT` seeds:
  - `loop_counter = inst.size`
  - `loop_start_pc = pc + 1`
  - for lane range `[reg_start, reg_end)`:
    - `gpr[0] = inst.address`
    - `gpr[1] = 0`
- before each instruction executes, the alloc warp also computes:
  - `addr_accum = shuffle(gpr[1], lane = pc - loop_start_pc)`
- when `loop_counter != 0`, the current allocating instruction gets:
  - `inst.address += addr_accum`
- While `loop_counter > 0`, each later allocating instruction sees:
  - `inst.address += gpr[1]`
- If that allocating instruction also has the `JUMP` flag:
  - `loop_counter--`
  - jump back to `loop_start_pc` if more iterations remain
  - `gpr[1] += gpr[0]`

Practical meaning:

- `gpr[0]` is the per-iteration address delta
- `gpr[1]` is the accumulated offset applied to later alloc ops
- `loop_counter` is only decremented on an allocating instruction that carries `JUMP`
- `RepeatM.on(...)` in Python builds this pattern by:
  - inserting one or more `OP_REPEAT` seed instructions
  - marking the final allocating step with `JUMP`

Register transition summary:

- on `OP_REPEAT`:
  - changed:
    - `loop_counter`
    - `loop_start_pc`
    - selected lanes of `gpr[0]`
    - selected lanes of `gpr[1]`
  - unchanged:
    - `jmp_cnt`
    - `shift`
- on a later allocating `JUMP` instruction:
  - changed:
    - `loop_counter`
    - maybe `next_pc = loop_start_pc`
    - `gpr[1] += gpr[0]`
  - unchanged:
    - `jmp_cnt`
    - `shift`

### `OP_LOOP` plus `GROUP`

This is the resource-group stepping path.

- `OP_LOOP` does not touch `gpr[0]/gpr[1]`
- Instead it updates:
  - `next_pc`
  - a 32-bit `shift` value
- it uses `jmp_cnt` as its own loop counter
- `num_slots` selects the alloc-warp lane whose per-thread `jmp_cnt` owns that loop
- `shift` is packed so that:
  - low 16 bits adjust `num_slots`
    - in practice this is used as `bar_shift << 6`, so slot count stays fixed while the encoded barrier id changes
  - high 16 bits adjust `arg`
    - in practice this shifts TMA descriptor ids
- Any allocating memory instruction with the `GROUP` flag set executes:
  - `inst.shifter += shift`

Practical meaning:

- `GROUP` marks instructions whose resource ids should advance across loop iterations
- `LoopM(..., resource_group=...)` or `LoopM(..., tma_shift=..., bar_shift=...)` defines that step size
- [python/dae/launcher.py](/home1/11362/depctg/vdcores/python/dae/launcher.py) exposes `ResourceGroup.get_shift()` as:
  - TMA increment = number of TMA descriptors in the group
  - barrier increment = number of barrier ids in the group

Register transition summary:

- on `OP_LOOP`:
  - changed:
    - control lane increments its own `jmp_cnt`
    - maybe `next_pc = coords[0]`
    - maybe `shift += packed(coords[2:3])`
    - or, on the terminal iteration:
      - `jmp_cnt = 0`
      - `shift = 0`
  - unchanged:
    - `gpr[0]`
    - `gpr[1]`
    - `loop_counter`
- on a later allocating `GROUP` instruction:
  - changed:
    - the in-flight `inst.num_slots` / `inst.arg` view via `inst.shifter += shift`
  - unchanged:
    - the VM registers themselves

### `CC0` and `CC0_ROW_BYTES`

These are small control-flow helpers for embedding-row address selection.

- `CC0` reads a token id from `inst.address`
- then it sets:
  - `loop_counter = 1`
  - `loop_start_pc = pc + 1`
  - lane `0`: `gpr[1] = token << inst.arg`
- `CC0_ROW_BYTES` is the same pattern except:
  - lane `0`: `gpr[1] = token * inst.size`

Practical meaning:

- both ops seed the address accumulator for the following allocating instruction
- the runtime comment in [include/dae/pipeline/allocwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/allocwarp.cuh) says a single `tmaload1D` should come right after `CC0`
- that schedule convention matters, because `CC0` itself does not clear the accumulator afterward

Register transition summary:

- on `CC0` / `CC0_ROW_BYTES`:
  - changed:
    - `loop_counter = 1`
    - `loop_start_pc = pc + 1`
    - lane `0` of `gpr[1]`
  - unchanged:
    - `gpr[0]`
    - `jmp_cnt`
    - `shift`

### Counter-Derived `OP_REPEAT`

`OP_REPEAT` also has a control-flow offset mode used by multi-token schedules.

- `inst.arg & 0x8000` enables counter mode
- `inst.arg & 0x4000` adds the selected counter value to `loop_counter`
- `inst.arg & 0x2000` accumulates into `gpr[1]` instead of replacing it
- `inst.arg & 0x00ff` selects an alloc-warp lane
- the selected lane's per-thread `jmp_cnt` is broadcast with `shfl`
- selected accumulator lanes are seeded with `inst.address * selected_jmp_cnt`

Practical meaning:

- a top-level `LoopM(..., reg=1)` can repeat one memory body over tokens
- `RepeatM.offsetByCounter(1, inst, delta)` applies `base + delta * token_iteration` without unrolling the body
- `RepeatM.offsetByCounters([(1, token_delta), (2, block_delta)], inst)` combines multiple loop counters for one address update; the Llama3 full-block decode path uses this for token plus KV-block position offsets
- `RepeatM.on(count, ..., count_counter_reg=2)` emits a repeat whose trip count is `count + jmp_cnt[2]`; the Llama3 full-block attention path uses this to load all previous full KV blocks as the outer block loop advances
- this does not create multiple memory counters per thread; it reuses the 32 existing per-lane `jmp_cnt` values

## Barrier Model

There are two barrier classes in play.

### Queue barriers

Used internally by `m2c`, `c2m`, and `m2ld`.

- `m2c` and `c2m` barriers include the full 128-thread compute group plus one memory-side producer/consumer thread
- each `m2ld` barrier includes:
  - alloc warp lane `0`
  - the corresponding load warp lane `0`

### Global schedule barriers

The `bars` array is the externally visible dependency table.

- read side:
  - `OP_ISSUE_BARRIER` spins until `bars[bar] == 0`
  - load ops with `MEM_OP_FLAGS_BARRIER` also wait for `bars[bar] == 0` before issuing
- write side:
  - writeback ops with `MEM_OP_FLAGS_BARRIER` decrement `bars[bar]` after the async store finishes

## Register-Like Facilities

Two operator pairs act like VM-side pseudo-registers.

### `RegStore` / `RegLoad`

Observed behavior from [include/dae/pipeline/ldwarp.cuh](/home1/11362/depctg/vdcores/include/dae/pipeline/ldwarp.cuh):

- `RegStore(reg_id)` records the allocated slot mask into `regFile[reg_id]` in the load warp
- `RegLoad(reg_id)` later republishes that mask back onto `m2c`

This behaves like a tiny loader-local register file of remembered slot masks. The checked-in implementation sizes it as `int regFile[4]`.

### `RawAddress`

Observed behavior from [python/dae/instructions.py](/home1/11362/depctg/vdcores/python/dae/instructions.py):

- `RawAddress(tensor, slot_id)` reserves a special slot id
- the pointer is stored in `st_insts[slot_id].address`
- compute kernels recover it with `slot_2_glob_ptr(...)`

This is the path used for outputs that compute writes directly to global memory without a shared-memory writeback tile.

## State Transition Sketch

For a normal allocating load/store instruction:

1. alloc warp fetches `MInst`
2. optional address/resource rewriting happens:
   - accumulator on `address`
   - group shift on `num_slots/arg`
3. allocator finds a slot span
4. alloc warp stores the rewritten `MInst` into `st_insts[lead_slot]`
5. alloc warp sends:
   - a ready-token slot mask placeholder to `m2c`
   - an `LdCmd` to the chosen `m2ld` port
6. load warp performs the load-side action, then arrives on the corresponding `m2c` barrier
7. compute pops from `m2c`, uses the slot, then returns it through `c2m`
8. plain-return tokens free immediately; writeback tokens go through the store warp first

## Current Build Caveat

The checked-in source defines many compute handlers, but the generated build selection in [build/generated/dae/selected_compute_ops.inc](/home1/11362/depctg/vdcores/build/generated/dae/selected_compute_ops.inc) controls what the local extension dispatches. Re-read that generated file before claiming an op is runnable. At integration time on 2026-04-20, it included:

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

So the VM model above is the source-level model, not a substitute for checking the exact generated operator subset compiled into the local extension artifact.
