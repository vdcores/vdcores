#pragma once

#include "virtualcore.cuh"

static __device__ __forceinline__ void prefetch_inst_window(
    const int lane_id, const MInst* insts, uint32_t target_pc) {
  if constexpr (!dae2LoadInstructions) {
    if (lane_id == 0)
      prefetch_l1(insts + (target_pc % numInsts));
  }
}

static constexpr uint16_t repeatCounterModeFlag = 0x8000U;
static constexpr uint16_t repeatCountCounterModeFlag = 0x4000U;
static constexpr uint16_t repeatAccumulateModeFlag = 0x2000U;
static constexpr uint16_t repeatSkipCountMask = 0x1F00U;
static constexpr int repeatSkipCountShift = 8;
static constexpr uint16_t repeatCounterRegMask = 0x00FFU;

template<typename M2C_Type, typename M2LD_Type>
__device__ __forceinline__ void allocwarp_execute(
    const int lane_id,
    M2C_Type &m2c, M2LD_Type m2ld[2], const MInst* smem_minsts, int *flags,
    MInst *st_insts, const void *smem_base, const CUtensorMap *tma_descs, int *bars,
    cuda::barrier<cuda::thread_scope_block> *ldu_control_publish_barrier,
    const LoopCounters &initial_loop_counts
#if defined(DAE_TRACK_PROFILE)
    , const int sm_id, uint64_t *g_events
#endif
) {
  static_assert(numSlots < 32, "Too many slots for single warp");

  // register flags
  MInst inst;
  uint32_t pc = 0, next_pc = 0;
  // parameter shift
  uint32_t shift = 0;

  MemoryVirtualCore di;
  di.init();
  uint32_t indirect_layer_index = 0;
  if (lane_id < numComputeLoopCounters) {
    di.jmp_cnt = initial_loop_counts.values[lane_id];
  }
  SharedMemoryAllocator<numSlots> alloc;

#if defined(DAE_TRACK_PROFILE)
  uint64_t slot_stall_ns = 0;
  uint64_t slot_stall_events = 0;
  uint64_t slot_retries = 0;
  uint64_t issue_barrier_ns = 0;
  uint64_t issue_barrier_contended = 0;
  uint64_t allocation_instructions = 0;
#endif

  __syncwarp();

  while (di.pred_continue) {
    inst = smem_minsts[next_pc % numInsts];
    // async zone after all shared memory read
    // IF/ID
    // 1. try to fetch a instruction
    // TODO(zhiyuang): inst to use is quite close. optimize? e.g, vector load?
    pc = next_pc;
    prefetch_inst_window(lane_id, smem_minsts, pc + 2);
    uint64_t addr_accum = __shfl_sync(0xFFFFFFFF, di.gpr[1], pc - di.loop_start_pc);

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
                di.loop_counter, inst.address);
    }

    // All layer-indexed dynamic loads in one repeated body share this linear
    // allocator index. The loop control advances it once per logical body,
    // replacing per-load address-arithmetic instruction sequences.
    const int decoded_op = op(inst.opcode);
    if (decoded_op >= op(OP_ALLOC_LAYER_TMA_LOAD_1D) &&
        decoded_op <= op(OP_ALLOC_LAYER_INDEXED_TMA_LOAD_1D)) {
      if (lane_id == 0) {
        inst.address += uint64_t(indirect_layer_index) * 8;
      }
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
#if defined(DAE_TRACK_PROFILE)
      bool slot_stalled = false;
      uint64_t slot_stall_start = 0;
#endif
      while (true) {
        di.slot_alloc = alloc.allocate(lane_id, flags, inst.nslot(), alloc_mask);
        // TODO(zhiyuang): reorder this store

        __mprint("[id] after allocation: allocate=%d slot=%d",
          di.pred_allocate, di.slot_alloc);

        if (di.slot_alloc >= 0) {
#if defined(DAE_TRACK_PROFILE)
          if (lane_id == 0) {
            ++allocation_instructions;
            if (slot_stalled) {
              slot_stall_ns +=
                  cuda::ptx::get_sreg_globaltimer() - slot_stall_start;
              ++slot_stall_events;
            }
          }
#endif
          break;
        }

#if defined(DAE_TRACK_PROFILE)
        if (lane_id == 0) {
          if (!slot_stalled) {
            slot_stalled = true;
            slot_stall_start = cuda::ptx::get_sreg_globaltimer();
          }
          ++slot_retries;
        }
#endif

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
        --di.loop_counter;
        if (di.loop_counter > 0) {
          next_pc = di.loop_start_pc;
          // prefetch_inst_window(lane_id, smem_minsts, next_pc + 2);
        }
        di.gpr[1] += di.gpr[0];
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
          const bool counter_mode = inst.arg & repeatCounterModeFlag;
          const bool count_counter_mode = inst.arg & repeatCountCounterModeFlag;
          const bool accumulate_mode = inst.arg & repeatAccumulateModeFlag;
          const int counter_reg = inst.arg & repeatCounterRegMask;
          const int counter_value = __shfl_sync(ALL_THREADS, di.jmp_cnt, counter_reg);
          const int repeat_count = inst.size + (count_counter_mode ? counter_value : 0);
          const int skip_count = (inst.arg & repeatSkipCountMask) >> repeatSkipCountShift;
          // Accumulator repeats extend the active seed; they do not start a new
          // repeat window, so the final consumer keeps the original pc distance.
          if (!accumulate_mode) {
            di.loop_counter = repeat_count; // minus the current one
            di.loop_start_pc = pc + 1;
            if (repeat_count == 0 && skip_count > 0) {
              next_pc = pc + 1 + skip_count;
            }
          }
          auto reg_start = inst.num_slots & 0xFF;
          auto reg_end = inst.num_slots >> 8;
          // TODO(zhiyuang): will this slowdown the critical path? if so we can also put the counter value in gpr and shuffle together with reg0
          if (lane_id >= reg_start && lane_id < reg_end) {
            if (accumulate_mode) {
              di.gpr[1] += counter_mode ? inst.address * counter_value : inst.address;
            } else {
              di.gpr[0] = inst.address; // loop offset
              di.gpr[1] = counter_mode ? inst.address * counter_value : 0;
            }
          }
        }
        break;
        case op(OP_LOOP): {
          prefetch_inst_window(lane_id, smem_minsts, inst.coords[0] + 1);
          // F0: jump to a different pc after certain iterations
          if (__memory_tid() == inst.num_slots) {
            if (++di.jmp_cnt < inst.size) {
              next_pc = (unsigned)inst.coords[0];
              // F2: update the shift for group instructions
              shift += *(const uint32_t *)&inst.coords[2];
            } else {
              di.jmp_cnt = 0;
              shift = 0;
            }
          }
          next_pc = __shfl_sync(0xFFFFFFFF, next_pc, inst.num_slots);
          shift = __shfl_sync(0xFFFFFFFF, shift, inst.num_slots);
          if (inst.coords[1] & 0x1)
            ++indirect_layer_index;
          __mprint("Loop: pc=%d reg=%d count=%d reg0=%d target_pc=%d arg_offset=%u",
            pc, inst.num_slots, inst.size, __shfl_sync(ALL_THREADS, di.jmp_cnt, inst.num_slots), next_pc, shift);
        }
        break;
        case op(OP_RESET_INDIRECT_LAYER): {
          indirect_layer_index = 0;
        }
        break;
        case op(OP_LDU_RELOAD_BARRIERS):
        case op(OP_LDU_PROFILE_LAYER):
        case op(OP_LDU_SET_AFFINE_EXPERT_BASE): {
          if (lane_id == 0) {
            const int special_slot = inst.nslot();
            if (special_slot < numSlots ||
                special_slot + 1 >= numSlots + numSpecialSlots) {
              asm volatile("trap;");
            }
            for (int port = 0; port < 2; ++port) {
              const int slot = special_slot + port;
              st_insts[slot] = inst;
              LdCmd ld;
              ld.init(slot, 0, inst.opcode);
              m2ld[port].put(ld.raw);
              m2ld[port].commit();
              m2ld[port].advance();
            }
          }
          __syncwarp();
          // Do not allow a later control command to overwrite the shared
          // metadata slots until both LDU handlers have copied this command.
          ldu_control_publish_barrier->arrive_and_wait();
        }
        break;
        case op(OP_ISSUE_BARRIER): {
          if (lane_id == 0) {
            volatile int *bar = bars + inst.bar();
#if defined(DAE_TRACK_PROFILE)
            const uint64_t barrier_start = cuda::ptx::get_sreg_globaltimer();
            if (*bar != 0)
              ++issue_barrier_contended;
#endif
            while (*bar != 0) {
              __nanosleep(barrierPollSleepCycles);
            }
#if defined(DAE_TRACK_PROFILE)
            issue_barrier_ns +=
                cuda::ptx::get_sreg_globaltimer() - barrier_start;
#endif
            __mprint("Issue barrier %d passed", inst.bar());
          }
          break;
        }
        // CV here for custom variation
        case op(OP_CC0): {
          // CC0: embedding operator. A single tmaload1D instruction should come right after this one
          int token = load_l2((const int *)(inst.address));
          di.loop_counter = 1;
          di.loop_start_pc = pc + 1;
          if (lane_id == 0) {
            di.gpr[1] = token << inst.arg;
          }
          break;
        }
        case op(OP_CC0_ROW_BYTES): {
          // Generalized CC0 path for non-power-of-two embedding row widths.
          int token = load_l2((const int *)(inst.address));
          di.loop_counter = 1;
          di.loop_start_pc = pc + 1;
          if (lane_id == 0) {
            di.gpr[1] = token * inst.size;
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
      pc, next_pc, di.loop_counter > 0, di.loop_counter);

  }

  // __print(lane_id, "End of Alloc warp execution");
#if defined(DAE_TRACK_PROFILE)
  if (lane_id == 0) {
    const int event_base = sm_id * numProfileEvents;
    g_events[event_base + DAE_TRACK_ALLOC_SLOT_STALL_NS] = slot_stall_ns;
    g_events[event_base + DAE_TRACK_ALLOC_SLOT_STALL_EVENTS] = slot_stall_events;
    g_events[event_base + DAE_TRACK_ALLOC_SLOT_RETRIES] = slot_retries;
    g_events[event_base + DAE_TRACK_ALLOC_ISSUE_BARRIER_NS] = issue_barrier_ns;
    g_events[event_base + DAE_TRACK_ALLOC_ISSUE_BARRIER_CONTENDED] =
        issue_barrier_contended;
    g_events[event_base + DAE_TRACK_ALLOC_INSTRUCTIONS] =
        allocation_instructions;
  }
#endif
  __mprint("End of allocwarp");
}
