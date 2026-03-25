#pragma once

#include "virtualcore.cuh"

static __device__ __forceinline__ void prefetch_inst_window(
    const int lane_id, const MInst* insts, uint32_t target_pc) {
  if constexpr (!dae2LoadInstructions) {
    if (lane_id == 0)
      prefetch_l1(insts + (target_pc % numInsts));
  }
}

template<typename M2C_Type, typename M2LD_Type>
__device__ __forceinline__ void allocwarp_execute(
    const int lane_id,
    M2C_Type &m2c, M2LD_Type m2ld[2], const MInst* smem_minsts, int *flags,
    MInst *st_insts, const void *smem_base, const CUtensorMap *tma_descs, int *bars
) {
  static_assert(numSlots < 32, "Too many slots for single warp");

  // register flags
  MInst inst;
  uint32_t pc = 0, next_pc = 0;
  // parameter shift
  uint32_t shift = 0;

  MemoryVirtualCore di;
  di.init();
  SharedMemoryAllocator<numSlots> alloc;

  __syncwarp();

  while (di.pred_continue) {
    inst = smem_minsts[next_pc % numInsts];
    // async zone after all shared memory read
    // IF/ID
    // 1. try to fetch a instruction
    // TODO(zhiyuang): inst to use is quite close. optimize? e.g, vector load?
    pc = next_pc;
    prefetch_inst_window(lane_id, smem_minsts, pc + 2);
    uint64_t addr_accum = __shfl_sync(
        0xFFFFFFFF, di.gpr[MVC_GPR_ACC], pc - di.gpr_32[MVC_GPR32_LOOP_START_PC]);

    __mprint("[exec][pc=%d]: opcode=%04x m2c.ptr=%d m2ld[0].ptr=%d m2ld[1].ptr=%d",
            pc, inst.opcode, m2c.ptr, m2ld[0].ptr, m2ld[1].ptr);
    // __smprint(0, lane_id, "[exec][pc=%d]: opcode=%04x m2c.ptr=%d m2ld[0].ptr=%d m2ld[1].ptr=%d",
    //         pc, inst.opcode, m2c.ptr, m2ld[0].ptr, m2ld[1].ptr);
    // end of async zone

    di.inst_decode(inst);
    auto &curld = m2ld[di.port];

    // ID.A: modification to the instruction
    // A1. shift the address field
    // load the address anyway regardless of allocate or not
    // TODO(zhiyuang): sometimes shuffle (esp, on 64bit) is slow?
    if (lane_id == 0 && di.id_repeat()) {
      inst.address += addr_accum;
      __mprint("[Loop][loop_counter=%d] Updated address addr + 0x? -> 0x%lx",
                di.gpr_32[MVC_GPR32_LOOP_COUNTER], inst.address);
    }

    // A2. shift the arg field for group instructions (usually with tmas and bars)
    if (inst.opcode & MEM_OP_FLAGS_GROUP) {
      __mprint("[Group] Before update: shift %x: bar=%d arg=%d nslot=%d",
        shift, inst.bar(), inst.arg, inst.nslot());
      inst.shifter += shift;
      __mprint("[Group] Updated: shift %x: bar=%d arg=%d nslot=%d",
        shift, inst.bar(), inst.arg, inst.nslot());
      // __smprint(0, lane_id, "[Group] Updated: shift %x: bar=%d arg=%d nslot=%d bar=%d",
      //   shift, inst.bar(), inst.arg, inst.nslot(), inst.opcode & MEM_OP_FLAGS_BARRIER);
    }

    // TODO(zhiyuang): let the allocator decide whether to stall
    // ID.C: resource allocation
    // we also commit in the alloc
    int alloc_mask = 0;
    if (di.pred_allocate) {
      while (true) {
        di.slot_alloc = alloc.allocate(lane_id, flags, inst.nslot(), alloc_mask);
        // TODO(zhiyuang): reorder this store

        __mprint("[id] after allocation: allocate=%d slot=%d",
          di.pred_allocate, di.slot_alloc);

        if (di.slot_alloc >= 0)
          break;

        __nanosleep(allocRetrySleepCycles);
      }
    }

    // if not stall we continue to execute memory or compute insts
    next_pc = pc + 1;

    // store the instruction into the slot
    if (di.pred_allocate) {
      // parallel_copy<sizeof(MInst)>(lane_id, &inst, &st_insts[di.slot_alloc]);
      // TODO(zhiyuang): do we need this syncwarp here?
      // __syncwarp();
      if (lane_id == 0) {
        st_insts[di.slot_alloc] = inst;
        m2c.put(alloc_mask);

        LdCmd ld;
        ld.init(di.slot_alloc, m2c.ptr, inst.opcode);

        curld.put(ld.raw);
        // TODO(zhiyuang): change the return value of allocate

        // TODO(zhiyuang): double push could be optimize? maybe put the barrier to the last
        m2c.advance();
        curld.commit();
        curld.advance();
      }

      // have to keep this branch
      if (di.pred_jump) {
        --di.gpr_32[MVC_GPR32_LOOP_COUNTER];
        if (di.gpr_32[MVC_GPR32_LOOP_COUNTER] > 0) {
          next_pc = di.gpr_32[MVC_GPR32_LOOP_START_PC];
          // prefetch_inst_window(lane_id, smem_minsts, next_pc + 2);
        }
        di.gpr[MVC_GPR_ACC] += di.gpr[MVC_GPR_DELTA];
      }
    } else { // Executing the non-allocation instructions (control flow instructions)
      switch (op(inst.opcode)) {
        // memory barrier ops
        case op(OP_TERMINATE): {
          di.pred_continue = false;
          if (lane_id == 0) {
            m2ld[0].push(SLOT_END);
            m2ld[1].push(SLOT_END);
          }
        }
        break;
        // repeat instruction will repeat the following instructions with NO overhead
        case op(OP_REPEAT): {
          di.gpr_32[MVC_GPR32_LOOP_COUNTER] = inst.size; // minus the current one
          di.gpr_32[MVC_GPR32_LOOP_START_PC] = pc + 1 - inst.arg; // the instruction arg as base arg
          auto reg_start = inst.num_slots & 0xFF;
          auto reg_end = inst.num_slots >> 8;
          if (lane_id >= reg_start && lane_id < reg_end) {
            di.gpr[MVC_GPR_DELTA] = inst.address; // loop offset
            di.gpr[MVC_GPR_ACC] = 0;
          }
        }
        break;
        case op(OP_LOOP): {
          prefetch_inst_window(lane_id, smem_minsts, inst.coords[0] + 1);
          // F0: jump to a different pc after certain iterations
          if (__memory_tid() == inst.num_slots) {
            if (++di.gpr_32[MVC_GPR32_JMP_CNT] < inst.size) {
              next_pc = (unsigned)inst.coords[0];
              // F2: update the shift for group instructions
              shift += *(const uint32_t *)&inst.coords[2];
            } else {
              di.gpr_32[MVC_GPR32_JMP_CNT] = 0;
              shift = 0;
            }
          }
          next_pc = __shfl_sync(0xFFFFFFFF, next_pc, inst.num_slots);
          shift = __shfl_sync(0xFFFFFFFF, shift, inst.num_slots);
          __mprint("Loop: pc=%d reg=%d count=%d reg0=%d target_pc=%d arg_offset=%u",
            pc, inst.num_slots, inst.size, di.gpr_32[MVC_GPR32_JMP_CNT], next_pc, shift);
        }
        break;
        case op(OP_ISSUE_BARRIER): {
          if (lane_id == 0) {
            volatile int *bar = bars + inst.bar();
            while (*bar != 0) {
              __nanosleep(barrierPollSleepCycles);
            }
            __mprint("Issue barrier %d passed", inst.bar());
          }
          break;
        }
        // CV here for custom variation
        case op(OP_CC0): {
          // CC0: embedding operator. A single tmaload1D instruction should come right after this one
          int token = *(int *)(inst.address);
          di.gpr_32[MVC_GPR32_LOOP_COUNTER] = 1;
          di.gpr_32[MVC_GPR32_LOOP_START_PC] = pc + 1;
          if (lane_id == 0) {
            di.gpr[MVC_GPR_ACC] = token << inst.arg;
          }
          break;
        }
        case op(OP_CC0_ROW_BYTES): {
          // Generalized CC0 path for non-power-of-two embedding row widths.
          int token = *(int *)(inst.address);
          di.gpr_32[MVC_GPR32_LOOP_COUNTER] = 1;
          di.gpr_32[MVC_GPR32_LOOP_START_PC] = pc + 1;
          if (lane_id == 0) {
            di.gpr[MVC_GPR_ACC] = token * inst.size;
          }
          break;
        }
        default:
          // opcode we do not want to handle
          __mprint("Unknown mem opcode: %04x op=%d\n", inst.opcode, op(inst.opcode));
          // assert(false && "Unknown mem opcode");
        break;
      }
    }

    __mprint("branch cur_pc=%d next_pc=%d loop=%d loop_counter=%d",
      pc,
      next_pc,
      di.gpr_32[MVC_GPR32_LOOP_COUNTER] > 0,
      di.gpr_32[MVC_GPR32_LOOP_COUNTER]);

  }

  // __print(lane_id, "End of Alloc warp execution");
  __mprint("End of allocwarp");
}
