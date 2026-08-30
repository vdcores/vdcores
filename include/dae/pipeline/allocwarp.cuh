#pragma once

#include <cuda/atomic>

#include "mxfp_resident_ffn.cuh"
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
static constexpr uint16_t repeatCounterRegMask = 0x0003U;
static constexpr uint16_t repeatCounterShiftMask = 0x001CU;
static constexpr int repeatCounterShiftShift = 2;
static constexpr uint16_t repeatCounterMaskBitsMask = 0x00E0U;
static constexpr int repeatCounterMaskBitsShift = 5;

static __device__ __forceinline__ void allocwarp_wait_ldu_publication(
    const int lane_id,
    cuda::barrier<cuda::thread_scope_block> *barrier) {
  if (lane_id == 0)
    barrier->arrive_and_wait();
  // Lane zero exclusively owns the mailbox. The next allocator iteration's
  // full-mask shuffle reconverges the warp before any lane-derived state is
  // consumed, so a second explicit warp barrier is unnecessary here.
}

static __device__ __forceinline__ void
allocwarp_observe_mxfp_resident_down_ready(
    const MInst &inst, int *bars, uint64_t *tmem_mma_barriers,
    const uint32_t resident_phase) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  // The allocator has published both resident LDU commands and has no more
  // useful issue work. Keep the existing overlap by observing the two
  // device-scope reduction dependencies while LDU0 and LDU1 produce Down.
  auto *poll_start = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentLdu1PollStartBarrier);
  auto *reduction_ready = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentReductionReadyBarrierBase);
  const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
  const int down_task_count = load_l2(
      reinterpret_cast<const int *>(plan + 3));
  if (!(inst.arg & dae_mxfp_resident_ffn::kCoupledDownOnly)) {
    poll_start->wait(resident_phase);
  }
  #pragma unroll 1
  for (int task = 0; task < down_task_count; ++task) {
    const auto *metadata = reinterpret_cast<const uint8_t *>(
        load_l2_u64(plan + 1 + task));
    const uint32_t task_bar = uint32_t(load_l2_u64(
        reinterpret_cast<const uint64_t *>(metadata + 32)) >> 32);
    cuda::atomic_ref<int, cuda::thread_scope_device> ready(bars[task_bar]);
    while (ready.load(cuda::memory_order_acquire) != 0) {
      __nanosleep(128);
    }
    reduction_ready[task].arrive();
  }
}

template<typename M2C_Type, typename M2LD_Type>
__device__ __forceinline__ void allocwarp_execute(
    const int lane_id,
    M2C_Type &m2c, M2LD_Type m2ld[2], const MInst* smem_minsts, int *flags,
    MInst *st_insts, const void *smem_base, const CUtensorMap *tma_descs, int *bars,
    cuda::barrier<cuda::thread_scope_block> *ldu_control_publish_barrier,
    const LoopCounters &initial_loop_counts,
    uint64_t *tmem_mma_barriers
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
  uint32_t mxfp_resident_down_phase = 0;
  if (lane_id < numComputeLoopCounters) {
    di.jmp_cnt = initial_loop_counts.values[lane_id];
  }
  SharedMemoryAllocator<numSlots> alloc;
#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
  bool mx_scale_bases_inflight = false;
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
    const int decoded_op = op(inst.opcode);
    const bool layer_repeat_offset =
        decoded_op == op(OP_ALLOC_LAYER_TMA_LOAD_1D) ||
        decoded_op == op(OP_ALLOC_LAYER_LDU_LOAD_1D);
    // A1. shift the address field
    // load the address anyway regardless of allocate or not
    // TODO(zhiyuang): sometimes shuffle (esp, on 64bit) is slow?
    if (lane_id == 0 && di.id_repeat()) {
      if (layer_repeat_offset) {
        // Keep the common short-context path in LDU. Its existing 16-bit arg
        // can carry a 16-byte-granular offset without another instruction.
        if ((addr_accum >> 4) <= 0xFFFFU) {
          inst.arg = static_cast<uint16_t>(addr_accum >> 4);
        }
      } else {
        inst.address += addr_accum;
      }
      __mprint("[Loop][loop_counter=%d] Updated address addr + 0x? -> 0x%lx",
                di.loop_counter, inst.address);
    }

    // All layer-indexed dynamic loads in one repeated body share this linear
    // allocator index. The loop control advances it once per logical body,
    // replacing per-load address-arithmetic instruction sequences.
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
    constexpr uint16_t kProfileStoreEventFlag = 1U << 15;
    constexpr uint16_t kProfileStoreAllocationFlag = 1U << 14;
    constexpr uint16_t kProfileStoreEventMask = (1U << 14) - 1;
    const bool profile_store_allocation =
        decoded_op == op(OP_ALLOC_WB_TMA_STORE_1D) &&
        (inst.arg & (kProfileStoreEventFlag |
                     kProfileStoreAllocationFlag)) ==
            (kProfileStoreEventFlag | kProfileStoreAllocationFlag);
    const int profile_store_event = inst.arg & kProfileStoreEventMask;
    if (profile_store_allocation && lane_id == 0) {
      g_events[sm_id * numProfileEvents + profile_store_event] =
          cuda::ptx::get_sreg_globaltimer();
    }
#endif
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
    if (lane_id == 0 && layer_repeat_offset && di.id_repeat() &&
        (addr_accum >> 4) > 0xFFFFU) {
      // Long-context byte offsets do not fit the compact LDU encoding.
      // Resolve the already layer-selected pointer here and turn this into
      // the equivalent direct load. This moves the one pointer read from LDU
      // to the allocator; it adds neither a command nor a global transaction.
      inst.address = load_l2_u64(
          reinterpret_cast<const uint64_t *>(inst.address)) + addr_accum;
      constexpr uint16_t kMemoryFlags = (1U << flagBits) - 1U;
      const uint16_t direct_opcode =
          decoded_op == op(OP_ALLOC_LAYER_TMA_LOAD_1D)
          ? OP_ALLOC_TMA_LOAD_1D
          : OP_ALLOC_LDU_LOAD_1D;
      inst.opcode = (direct_opcode & ~kMemoryFlags) |
                    (inst.opcode & kMemoryFlags);
    }
    if (decoded_op == op(OP_ALLOC_LAYER_TMA_LOAD_4D) && lane_id == 0) {
      inst.coords[3] += indirect_layer_index;
    }
    if ((decoded_op == op(OP_ALLOC_WB_LAYER_TMA_REDUCE_ADD_3D) ||
         decoded_op == op(OP_ALLOC_WB_LAYER_TMA_STORE_3D)) && lane_id == 0) {
      inst.coords[2] += indirect_layer_index;
    }
    if (decoded_op == op(OP_TMA_LOAD_MX_COUPLED_STREAM) &&
        (inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask) ==
            dae_mxfp_resident_ffn::kCoupledFp8Gemv &&
        (inst.size & dae_mxfp_resident_ffn::kCoupledLayerIndexedSize)) {
      if (lane_id == 0) {
        inst.address +=
            uint64_t(indirect_layer_index) *
            2 * sizeof(uint64_t);
        inst.size &= dae_mxfp_resident_ffn::kCoupledStreamLengthMask;
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
    if (decoded_op == op(OP_LDU_ASYNC_RELOAD_BARRIERS) &&
        (inst.opcode & MEM_OP_FLAGS_GROUP)) {
      constexpr uint16_t kShiftTarget = 1U << 13;
      const int bank_shift = shift >> slotBits;
      const int count = inst.size & ((1U << slotBits) - 1);
      const int input_bar = (inst.size >> slotBits) + bank_shift;
      if (inst.arg & kShiftTarget) {
        inst.arg = (inst.arg & ~((1U << 10) - 1)) |
            ((inst.arg + bank_shift) & ((1U << 10) - 1));
      }
      inst.size = count | (input_bar << slotBits);
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
        allocwarp_wait_ldu_publication(
            lane_id, ldu_control_publish_barrier);
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
        const bool fixed_internal_ring =
            decoded_op == op(OP_TMA_LOAD_MX_COUPLED_STREAM) &&
            (inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask) ==
                dae_mxfp_resident_ffn::kCoupledTmaRing;
        di.slot_alloc = fixed_internal_ring
            ? alloc.allocate_at_zero(
                  lane_id, flags, inst.nslot(), alloc_mask)
            : alloc.allocate(
                  lane_id, flags, inst.nslot(), alloc_mask);
        // TODO(zhiyuang): reorder this store

        __mprint("[id] after allocation: allocate=%d slot=%d",
          di.pred_allocate, di.slot_alloc);

        if (di.slot_alloc >= 0) {
#if defined(DAE_TRACK_PROFILE)
          if (lane_id == 0 &&
              decoded_op == op(OP_ALLOC_WB_RAW_ADDRESS) &&
              inst.size != 0) {
            g_events[sm_id * numProfileEvents + int(inst.size) - 1] =
                cuda::ptx::get_sreg_globaltimer();
          }
#endif
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
          if (profile_store_allocation && lane_id == 0) {
            g_events[sm_id * numProfileEvents + profile_store_event + 1] =
                cuda::ptx::get_sreg_globaltimer();
          }
#endif
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
        const uint16_t coupled_kind =
            inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask;
        const bool allocator_owned_coupled =
            decoded_op == op(OP_TMA_LOAD_MX_COUPLED_STREAM) &&
            (coupled_kind == dae_mxfp_resident_ffn::kCoupledFp8Gemv ||
             coupled_kind == dae_mxfp_resident_ffn::kCoupledTmaRing);
        st_insts[di.slot_alloc] = inst;
        m2c.put(alloc_mask);

        LdCmd ld;
        ld.init(di.slot_alloc, m2c.ptr, inst.opcode);
        const bool direct_writeback_publication =
            (inst.opcode & MEM_OP_FLAGS_WRITEBACK) != 0 &&
            decoded_op != op(OP_ALLOC_RW_TMA_2D);
        if (allocator_owned_coupled || direct_writeback_publication) {
          // The lease itself is the compute operand. Publish it immediately;
          // compute observes the per-stage transaction barriers before
          // touching payload bytes. A writeback lease likewise has no load
          // producer: compute fills it and STU remains its only consumer.
          m2c.commit();
          if (allocator_owned_coupled) {
            #pragma unroll
            for (int port = 0; port < 2; ++port) {
              m2ld[port].put(ld.raw);
              m2ld[port].commit();
              m2ld[port].advance();
            }
          }
          m2c.advance();
        } else {
          curld.put(ld.raw);
          // TODO(zhiyuang): change the return value of allocate

          // TODO(zhiyuang): double push could be optimize? maybe put the barrier to the last
          m2c.advance();
          curld.commit();
          curld.advance();
        }
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
          const int raw_counter_value =
              __shfl_sync(ALL_THREADS, di.jmp_cnt, counter_reg);
          const int counter_shift =
              (inst.arg & repeatCounterShiftMask) >> repeatCounterShiftShift;
          const int counter_mask_bits =
              (inst.arg & repeatCounterMaskBitsMask) >>
              repeatCounterMaskBitsShift;
          const int counter_mask = counter_mask_bits == 0
              ? -1
              : (1 << counter_mask_bits) - 1;
          const int counter_value =
              (raw_counter_value >> counter_shift) & counter_mask;
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
        case op(OP_LDU_RELOAD_BARRIERS): {
          constexpr uint16_t kSkipFinalLoop = 1U << 14;
          if (inst.arg & kSkipFinalLoop) {
            const int loop_reg = (inst.arg >> 10) & 0x3;
            // Terminal elision is valid only for a one-bank repeated block,
            // so its reload is immediately followed by the authoritative
            // outer LOOPM descriptor. Keep the full reload-range width in
            // inst.size; full-model images can span more than 255 barriers.
            const int loop_count = smem_minsts[pc + 1].size;
            const int completed_backedges = __shfl_sync(
                ALL_THREADS, di.jmp_cnt, loop_reg);
            // The following OP_LOOP increments the counter. On the final
            // body, leave the completion barrier at zero and let the next
            // activation load consume it directly instead of draining both
            // LDU FIFOs through a terminal device-wide reload.
            if (completed_backedges + 1 == loop_count) {
              break;
            }
          }
          if (lane_id == 0) {
            const int special_slot = inst.nslot();
            if (special_slot < numSlots ||
                special_slot >= numSlots + numSpecialSlots) {
              asm volatile("trap;");
            }
            st_insts[special_slot] = inst;
            for (int port = 0; port < 2; ++port) {
              LdCmd ld;
              ld.init(special_slot, 0, inst.opcode);
              m2ld[port].put(ld.raw);
              m2ld[port].commit();
              m2ld[port].advance();
            }
          }
          __syncwarp();
          // Do not allow a later control command to overwrite the metadata
          // until both LDU handlers have consumed the command.
          allocwarp_wait_ldu_publication(
              lane_id, ldu_control_publish_barrier);
        }
        break;
        case op(OP_LDU_ASYNC_RELOAD_BARRIERS): {
          if constexpr (dae2AsyncBarrierReload) {
            constexpr uint16_t kSkipInitialLoop = 1U << 12;
            const bool skip_initial_loop =
                (inst.arg & kSkipInitialLoop) &&
                __shfl_sync(ALL_THREADS, di.jmp_cnt, 0) == 0 &&
                __shfl_sync(ALL_THREADS, di.jmp_cnt, 1) == 0;
            if (skip_initial_loop) {
              if (lane_id == 0) {
                atomicSub(&bars[inst.bar()], 1);
              }
              break;
            }
            if (lane_id == 0) {
              const int special_slot = inst.nslot();
              if (special_slot < numSlots ||
                  special_slot >= numSlots + numSpecialSlots) {
                asm volatile("trap;");
              }
              st_insts[special_slot] = inst;
              for (int port = 0; port < 2; ++port) {
                LdCmd ld;
                ld.init(special_slot, 0, inst.opcode);
                m2ld[port].put(ld.raw);
                m2ld[port].commit();
                m2ld[port].advance();
              }
            }
            __syncwarp();
            allocwarp_wait_ldu_publication(
                lane_id, ldu_control_publish_barrier);
          }
        }
        break;
        case op(OP_LDU_WAIT_BARRIER): {
          if constexpr (dae2AsyncBarrierReload) {
            const int generation = __shfl_sync(
                ALL_THREADS, di.jmp_cnt, inst.arg);
            if (lane_id == 0) {
              inst.arg = generation;
              const int special_slot = inst.nslot();
              st_insts[special_slot] = inst;
              for (int port = 0; port < 2; ++port) {
                LdCmd ld;
                ld.init(special_slot, 0, inst.opcode);
                m2ld[port].put(ld.raw);
                m2ld[port].commit();
                m2ld[port].advance();
              }
            }
            __syncwarp();
            allocwarp_wait_ldu_publication(
                lane_id, ldu_control_publish_barrier);
          }
        }
        break;
        case op(OP_TMA_LOAD_MX_COUPLED_STREAM): {
          // A dedicated resident plan owns fixed shared-memory addresses, so
          // this command bypasses both slot allocation and M2C publication.
          // Each in-flight stream has an immutable special mailbox. Chained
          // streams still publish distinct mailboxes before the first LDU
          // handler consumes the next queue entry locally.
          if (lane_id == 0) {
            const int special_slot = inst.nslot();
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            const uint16_t stream_kind =
                inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask;
            int detail_event = -1;
            if (stream_kind == dae_mxfp_resident_ffn::kCoupledLinear1) {
              detail_event = mxfpFfnDetailAllocatorLinear1;
            } else if (
                stream_kind == dae_mxfp_resident_ffn::kCoupledDownWeight) {
              detail_event = mxfpFfnDetailAllocatorDownWeight;
            } else if (
                stream_kind == dae_mxfp_resident_ffn::kCoupledDownActivation) {
              detail_event = mxfpFfnDetailAllocatorDownActivation;
            }
            if (detail_event >= 0) {
              g_events[sm_id * numProfileEvents + detail_event] =
                  cuda::ptx::get_sreg_globaltimer();
            }
#endif
            st_insts[special_slot] = inst;
            LdCmd ld;
            ld.init(uint8_t(special_slot), 0, inst.opcode);
            curld.put(ld.raw);
            curld.commit();
            curld.advance();
            if ((inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask) ==
                    dae_mxfp_resident_ffn::kCoupledDownActivation) {
              allocwarp_observe_mxfp_resident_down_ready(
                  inst, bars, tmem_mma_barriers,
                  mxfp_resident_down_phase);
              mxfp_resident_down_phase ^= 1U;
            }
          }
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

      // Fixed-address resident commands bypass allocation but can still be
      // the final consumer in an allocator repeat window.  Retire that window
      // here just as the allocating path above does, otherwise later commands
      // decode their offsets from lanes beyond the 32-lane repeat window.
      if (di.pred_jump) {
        --di.loop_counter;
        if (di.loop_counter > 0) {
          next_pc = di.loop_start_pc;
        }
        di.gpr[1] += di.gpr[0];
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
