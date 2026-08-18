#pragma once

#include <cuda/atomic>

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

template<typename M2LD_Type>
__device__ __forceinline__ void
allocwarp_execute_mxfp_resident_ffn_fast(
    const int lane_id, M2LD_Type m2ld[2], const MInst *minsts,
    MInst *st_insts, int *bars, uint64_t *tmem_mma_barriers
#if defined(DAE_TRACK_PROFILE)
    , const int sm_id, uint64_t *g_events
#endif
) {
  static_assert(
      !mxfpResidentFfnFastMemoryDispatchEnabled || numInsts == 2,
      "resident FFN fast memory dispatch requires command + terminate");

  if (lane_id == 0) {
    // This remains a normal queued VDCores memory task: the allocator warp
    // publishes the immutable mailbox and LDU0 acquires/dequeues the command.
    // Only the generic decoder for the fixed two-command image is omitted.
    const MInst inst = minsts[0];
    const uint8_t special_slot = uint8_t(inst.nslot());
    st_insts[special_slot] = inst;

    LdCmd ld;
    ld.init(special_slot, 0, inst.opcode);
    m2ld[0].put(ld.raw);
    m2ld[0].commit();
    m2ld[0].advance();
    if constexpr (mxfpResidentDownLdu1ZeroEnabled) {
      m2ld[1].put(ld.raw);
      m2ld[1].commit();
      m2ld[1].advance();
    }
    m2ld[0].push(SLOT_END);
    m2ld[1].push(SLOT_END);

    if constexpr (mxfpResidentDownSplitLduEnabled) {
      using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
      // This warp has finished its only queued dispatch. Pause it until LDU0
      // has issued the complete Linear-1 stream, then make it the dedicated
      // observer for the two global reduction dependencies. LDU0 and LDU1
      // remain free to produce Down weight/SFA and activation/SFB.
      auto *poll_start = reinterpret_cast<TxBarrier *>(
          tmem_mma_barriers + mxfpDownResidentLdu1PollStartBarrier);
      auto *reduction_ready = reinterpret_cast<TxBarrier *>(
          tmem_mma_barriers +
          mxfpDownResidentReductionReadyBarrierBase);
      poll_start->wait(0);
      const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
      #pragma unroll
      for (int task = 0; task < 2; ++task) {
        const auto *metadata = reinterpret_cast<const uint8_t *>(
            load_l2_u64(plan + 1 + task));
        const uint32_t task_bar = uint32_t(load_l2_u64(
            reinterpret_cast<const uint64_t *>(metadata + 32)) >> 32);
        cuda::atomic_ref<int, cuda::thread_scope_device> ready(
            bars[task_bar]);
        while (ready.load(cuda::memory_order_acquire) != 0) {
          __nanosleep(128);
        }
        reduction_ready[task].arrive();
      }
    }

#if defined(DAE_TRACK_PROFILE)
    const int event_base = sm_id * numProfileEvents;
    g_events[event_base + DAE_TRACK_ALLOC_SLOT_STALL_NS] = 0;
    g_events[event_base + DAE_TRACK_ALLOC_SLOT_STALL_EVENTS] = 0;
    g_events[event_base + DAE_TRACK_ALLOC_SLOT_RETRIES] = 0;
    g_events[event_base + DAE_TRACK_ALLOC_ISSUE_BARRIER_NS] = 0;
    g_events[event_base + DAE_TRACK_ALLOC_ISSUE_BARRIER_CONTENDED] = 0;
    g_events[event_base + DAE_TRACK_ALLOC_INSTRUCTIONS] = 0;
#endif
  }
}

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
#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
  bool mx_scale_bases_inflight = false;
#endif
#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
  uint32_t mx_down_weight_ring_mask = 0;
#endif
#if DAE_MXFP_GATE_UP_LDU_WEIGHT_RING && DAE_MXFP_DOWN_LDU_WEIGHT_RING
  uint32_t mx_weight_ring_handoff_mask = 0;
#endif

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
    if ((decoded_op >= op(OP_ALLOC_LAYER_TMA_LOAD_1D) &&
         decoded_op <= op(OP_ALLOC_LAYER_INDEXED_TMA_LOAD_1D)) ||
        decoded_op == op(OP_ALLOC_LAYER_ROUTED_TMA_LOAD_BASE_1D)) {
      const int entry_bytes =
          (decoded_op == op(OP_ALLOC_LAYER_ROUTED_TMA_LOAD_1D) ||
           decoded_op == op(OP_ALLOC_LAYER_ROUTED_TMA_LOAD_BASE_1D)) ? 16 : 8;
      if (lane_id == 0) {
        inst.address += uint64_t(indirect_layer_index) * entry_bytes;
      }
    }
    if (decoded_op == op(OP_ALLOC_LAYER_TMA_LOAD_4D) && lane_id == 0) {
      inst.coords[3] += indirect_layer_index;
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

#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
    if (decoded_op == op(OP_ALLOC_TMA_LOAD_MX_SCALE_BASE_1D)) {
      const int operand = inst.arg;
      const int special_slot = inst.nslot();
      if (operand >= 2 || di.port != operand ||
          special_slot != numSlots + 6 + operand ||
          special_slot >= numSlots + numSpecialSlots) {
        asm volatile("trap;");
      }
      // Both LDUs acknowledge after copying their base mailbox. Delay all 32
      // allocator arrivals until a later task is about to overwrite operand
      // zero; a single-tile schedule pays no reuse rendezvous.
      if (operand == 0 && mx_scale_bases_inflight) {
        ldu_control_publish_barrier->arrive_and_wait();
        mx_scale_bases_inflight = false;
      }
      if (operand == 1) {
        mx_scale_bases_inflight = true;
      }
    }
#endif

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

#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
    if (di.pred_allocate &&
        decoded_op == op(OP_ALLOC_TMA_LOAD_MX_DOWN_WEIGHT_RING_5D)) {
      // Continuations do not allocate. Keep the original lease mask in the
      // allocator warp so each compact command can publish that same storage
      // to its matching compute task while LDU0 retains ownership.
      mx_down_weight_ring_mask = uint32_t(alloc_mask);
    }
#endif
#if DAE_MXFP_GATE_UP_LDU_WEIGHT_RING && DAE_MXFP_DOWN_LDU_WEIGHT_RING
    if (di.pred_allocate &&
        decoded_op == op(OP_ALLOC_TMA_LOAD_MX_WEIGHT_RING_HANDOFF_5D)) {
      // The allocator keeps the source lease live. The following target is a
      // non-allocating publication of this same mask to Linear-2 compute.
      mx_weight_ring_handoff_mask = uint32_t(alloc_mask);
    }
#endif

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
        case op(OP_LDU_PROFILE_LAYER): {
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
        case op(OP_TMA_LOAD_MX_GATE_UP_RESIDENT):
        case op(OP_TMA_LOAD_MX_DOWN_RESIDENT):
        case op(OP_TMA_LOAD_MX_RESIDENT_FFN): {
          // A dedicated resident plan owns fixed shared-memory addresses, so
          // this command bypasses both slot allocation and M2C publication.
          // The isolated Linear-1 schedule submits exactly one such command
          // per worker; its special mailbox remains immutable until LDU0 has
          // consumed the complete plan.
          if (lane_id == 0) {
            const int special_slot = inst.nslot();
            st_insts[special_slot] = inst;
            LdCmd ld;
            ld.init(uint8_t(special_slot), 0, inst.opcode);
            curld.put(ld.raw);
            curld.commit();
            curld.advance();
          }
        }
        break;
        case op(OP_TMA_LOAD_MX_WEIGHT_RING_CONTINUE_5D): {
          // This compact command carries no allocator lease and no M2C
          // operand. It stays on LDU0's FIFO so the active retained-ring
          // handler can consume it as the gate-to-up continuation without
          // returning through the allocator loop.
          if (lane_id == 0) {
            LdCmd ld;
            ld.init(inst.nslot(), 0, inst.opcode);
            curld.put(ld.raw);
            curld.commit();
            curld.advance();
          }
        }
        break;
#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
        case op(OP_TMA_LOAD_MX_DOWN_WEIGHT_RING_CONTINUE_5D): {
          // Reserve one ordinary M2C publication for the next compute task,
          // but keep the eight physical slots leased to the active LDU0
          // handler. The next output-task coordinate travels directly in the
          // compact LdCmd slot byte, so no special mailbox is rewritten.
          if (lane_id == 0) {
            m2c.put(int(mx_down_weight_ring_mask));
            LdCmd ld;
            ld.init(uint8_t(inst.num_slots), m2c.ptr, inst.opcode);
            curld.put(ld.raw);
            m2c.advance();
            curld.commit();
            curld.advance();
          }
        }
        break;
#endif
#if DAE_MXFP_GATE_UP_LDU_WEIGHT_RING && DAE_MXFP_DOWN_LDU_WEIGHT_RING
        case op(OP_TMA_LOAD_MX_DOWN_WEIGHT_RING_HANDOFF_5D): {
          // Preserve the complete target descriptor/task command in the
          // special mailbox consumed by the still-running LDU0 source. The
          // down compute consumes the original lease mask. Its persistent
          // down barrier bank remains separate from the still-retiring gate
          // stage, so no barrier objects are recreated at the transition.
          if (lane_id == 0) {
            const int special_slot = inst.nslot();
            st_insts[special_slot] = inst;
            mx_down_weight_ring_mask = mx_weight_ring_handoff_mask;
            m2c.put(int(mx_down_weight_ring_mask));
            LdCmd ld;
            ld.init(uint8_t(special_slot), m2c.ptr, inst.opcode);
            curld.put(ld.raw);
            m2c.advance();
            curld.commit();
            curld.advance();
          }
        }
        break;
#endif
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
