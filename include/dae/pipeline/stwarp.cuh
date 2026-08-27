#pragma once

#include "virtualcore.cuh"

// TODO(zhiyuang): attach bars to the writeback
template<typename C2M_Type>
__device__ __forceinline__ void stwarp_execute_singlethread(
    C2M_Type &c2m, const MInst* slot_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars
#if defined(DAE_TRACK_PROFILE)
    , const int sm_id, uint64_t *g_events
#endif
    ) {

  __stprint("[ST Warp] Start ST warp execution");

#if defined(DAE_TRACK_PROFILE)
  uint64_t service_ns = 0;
  uint64_t barrier_service_ns = 0;
  uint64_t commands = 0;
  uint64_t barrier_commands = 0;
#endif
#if defined(DAE_STU_HISTORY_PROFILE)
  uint64_t recent_service_begin[stuHistoryCommands] = {};
  uint64_t recent_service_end[stuHistoryCommands] = {};
  uint16_t recent_opcode[stuHistoryCommands] = {};
  uint32_t recent_count = 0;
  uint64_t pop_begin = cuda::ptx::get_sreg_globaltimer();
#endif
    
  int slot_mask = c2m.pop();
  while (slot_mask) {
#if defined(DAE_TRACK_PROFILE)
    const uint64_t service_start = cuda::ptx::get_sreg_globaltimer();
    ++commands;
#endif
  
    auto slot = extract(slot_mask);
    bool do_free = true;

    __stprint("Receive ST slot: slot=%d", slot);

    auto &inst = slot_insts[slot];
    uint16_t opcode = inst.opcode;
#if defined(DAE_TRACK_PROFILE)
    const int raw_profile_event =
        op(opcode) == op(OP_ALLOC_WB_RAW_ADDRESS) && inst.size != 0
        ? int(inst.size) - 1
        : -1;
    if (raw_profile_event >= 0) {
      g_events[sm_id * numProfileEvents + raw_profile_event + 1] =
          service_start;
#if defined(DAE_STU_HISTORY_PROFILE)
      g_events[sm_id * numProfileEvents + stuRawPopBeginEvent] = pop_begin;
      const uint32_t service_queue_slot =
          (c2m.ptr + 32 - 1) % 32;
      g_events[sm_id * numProfileEvents + stuRawServiceIdentityEvent] =
          (uint64_t(slot) << 32) | service_queue_slot;
#endif
#if defined(DAE_STU_HISTORY_PROFILE)
      const uint32_t history_count =
          min(recent_count, uint32_t(stuHistoryCommands));
      const uint32_t history_start = recent_count < stuHistoryCommands
          ? 0
          : recent_count % stuHistoryCommands;
#pragma unroll
      for (uint32_t history = 0; history < stuHistoryCommands; ++history) {
        if (history < history_count) {
          const uint32_t source =
              (history_start + history) % stuHistoryCommands;
          const int event = stuHistoryEventBase + 3 * history;
          g_events[sm_id * numProfileEvents + event + 0] =
              recent_opcode[source];
          g_events[sm_id * numProfileEvents + event + 1] =
              recent_service_begin[source];
          g_events[sm_id * numProfileEvents + event + 2] =
              recent_service_end[source];
        }
      }
      g_events[sm_id * numProfileEvents + stuHistoryEventBase +
               3 * stuHistoryCommands] = history_count;
#endif
    }
#endif
    // all ops are writeback ops

    switch(op(opcode)) {
      case op(OP_ALLOC_WB_TMA_STORE_1D):
      {
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global,
          cuda::ptx::space_shared,
          (void*)(inst.address),
          (const void *)(get_slot_address(smem_base, slot)),
          inst.size
        );
        cuda::ptx::cp_async_bulk_commit_group();
      } 
        break;
      case op(OP_ALLOC_WB_STU_STORE_1D):
      {
        auto *dst = reinterpret_cast<unsigned char *>(inst.address);
        const auto *src = static_cast<const unsigned char *>(
            get_slot_address(smem_base, slot));
        int offset = 0;
        if ((reinterpret_cast<uintptr_t>(dst) & 0xF) == 0) {
          for (; offset + 16 <= inst.size; offset += 16) {
            *reinterpret_cast<uint4 *>(dst + offset) =
                *reinterpret_cast<const uint4 *>(src + offset);
          }
        }
        if ((reinterpret_cast<uintptr_t>(dst + offset) & 0x3) == 0) {
          for (; offset + 4 <= inst.size; offset += 4) {
            *reinterpret_cast<uint32_t *>(dst + offset) =
                *reinterpret_cast<const uint32_t *>(src + offset);
          }
        }
        if ((reinterpret_cast<uintptr_t>(dst + offset) & 0x1) == 0) {
          for (; offset + 2 <= inst.size; offset += 2) {
            *reinterpret_cast<uint16_t *>(dst + offset) =
                *reinterpret_cast<const uint16_t *>(src + offset);
          }
        }
        for (; offset < inst.size; ++offset) {
          dst[offset] = src[offset];
        }
      }
        break;
      case op(OP_ALLOC_RW_TMA_2D):
      {
        asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
          "[%0, {%1, %2}], [%3];\n"
          :
          : "l"((void *)(tma_descs + inst.coords[3])),
            "r"((int)inst.coords[0]),
            "r"((int)inst.coords[1]),
            "r"((uint32_t)__cvta_generic_to_shared(
                get_slot_address(smem_base, slot)))
          : "memory");
        cuda::ptx::cp_async_bulk_commit_group();
      }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_PAIR_2D):
      {
        const uint32_t transfer_size = uint32_t(inst.size);
        const uint32_t tile_size = transfer_size / 2;
        const uint16_t *cord = inst.coords;
        const uint32_t smem = uint32_t(__cvta_generic_to_shared(
            get_slot_address(smem_base, slot)));
        asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
          "[%0, {%1, %2}], [%3];\n"
          "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
          "[%0, {%4, %2}], [%5];\n"
          :
          : "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"(smem),
            "r"((int)cord[0] + (int)cord[2]),
            "r"(smem + tile_size)
          : "memory");
        cuda::ptx::cp_async_bulk_commit_group();
      }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_2D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 2D Store: desc_idx=%d size=%d cord=(%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1]);
          asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
            "[%0, {%1, %2}], [%3];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_4D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 4D Store: desc_idx=%d size=%d cord=(%d,%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
          asm volatile(
            "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
            "[%0, {%1, %2, %3, %4}], [%5];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((int)cord[3]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_3D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 3D Store: desc_idx=%d size=%d cord=(%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2]);
          asm volatile(
            "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
            "[%0, {%1, %2, %3}], [%4];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_5D_FIX0):
        {
          const uint16_t *cord = inst.coords;
          // harcode first coord to be 0
          __stprint("TMA 5D Store: desc_idx=%d size=%d cord=(0,%d,%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
          asm volatile(
            "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
            "[%0, {0, %1, %2, %3, %4}], [%5];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((int)cord[3]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_REDUCE_ADD_2D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 2D Reduce-Add: desc_idx=%d size=%d cord=(%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1]);
          asm volatile(
            "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "
            "[%0, {%1, %2}], [%3];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_REDUCE_ADD_3D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 3D Reduce-Add: desc_idx=%d size=%d cord=(%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2]);
          asm volatile(
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group "
            "[%0, {%1, %2, %3}], [%4];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_REDUCE_ADD_4D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 4D Reduce-Add: desc_idx=%d size=%d cord=(%d,%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
          asm volatile(
            "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.bulk_group "
            "[%0, {%1, %2, %3, %4}], [%5];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((int)cord[3]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      default:
        // unknown opcode
        __stprint("Unknown mem wb opcode: slot_mask=%x slot=%d op=%d opcode=%04x\n", slot_mask, slot, op(inst.opcode), inst.opcode);
        do_free = false;
        break;
    }

    // do bar for all instructions all at once
    if (opcode & MEM_OP_FLAGS_BARRIER) {
#if defined(DAE_TRACK_PROFILE)
      ++barrier_commands;
#endif
      // cuda::std::atomic_ref<int> bar {bars[inst.bar()]};
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE) || \
    defined(DAE_ATTENTION_DETAIL_PROFILE)
      constexpr uint16_t kProfileStoreEventFlag = 1U << 15;
      constexpr uint16_t kProfileStoreAllocationFlag = 1U << 14;
      constexpr uint16_t kProfileStoreEventMask = (1U << 14) - 1;
      if (op(opcode) == op(OP_ALLOC_WB_TMA_STORE_1D) &&
          (inst.arg & kProfileStoreEventFlag)) {
        const int event_id =
            (inst.arg & kProfileStoreEventMask) +
            ((inst.arg & kProfileStoreAllocationFlag) ? 2 : 0);
        g_events[sm_id * numProfileEvents +
                 event_id] =
            cuda::ptx::get_sreg_globaltimer();
      }
#endif
      atomicSub(&bars[inst.bar()], 1);
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
      if (op(opcode) == op(OP_ALLOC_WB_TMA_STORE_1D) &&
          (inst.arg & kProfileStoreEventFlag)) {
        const int event_id = inst.arg & kProfileStoreEventMask;
        g_events[sm_id * numProfileEvents + event_id + 1] =
            cuda::ptx::get_sreg_globaltimer();
      }
#endif
#if defined(DAE_TRACK_PROFILE)
      if (raw_profile_event >= 0) {
        g_events[sm_id * numProfileEvents + raw_profile_event + 2] =
            cuda::ptx::get_sreg_globaltimer();
      }
#endif
      // int current_cnt = bar.fetch_sub(1, cuda::std::memory_order_release);
      // __stprint("Arrive for barrier %d, remaining count=%d", inst.bar(), current_cnt - 1);
      // if (inst.bar() == 0)
      //   printf("[sm=%d] Arrive for barrier %d, remaining count=%d\n", blockIdx.x, inst.bar(), current_cnt - 1);
    } else {
      cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
    }

    // write back to free the slot
    __stprint("finish slot=%d op=%d flags=%02x",
      slot, op(inst.opcode), opcode & ((1 << flagBits) - 1));

    if (do_free)
      c2m.reset(slot_mask);
#if defined(DAE_TRACK_PROFILE)
    const uint64_t service_end = cuda::ptx::get_sreg_globaltimer();
    const uint64_t current_service_ns = service_end - service_start;
    service_ns += current_service_ns;
    if (opcode & MEM_OP_FLAGS_BARRIER)
      barrier_service_ns += current_service_ns;
#endif
#if defined(DAE_STU_HISTORY_PROFILE)
    const uint32_t history_slot = recent_count % stuHistoryCommands;
    recent_service_begin[history_slot] = service_start;
    recent_service_end[history_slot] = service_end;
    recent_opcode[history_slot] = op(opcode);
    ++recent_count;
    pop_begin = cuda::ptx::get_sreg_globaltimer();
#endif
    slot_mask = c2m.pop();
  }

  __stprint("End of ST warp execution");
#if defined(DAE_TRACK_PROFILE)
  const int event_base = sm_id * numProfileEvents;
  g_events[event_base + DAE_TRACK_STORE_QUEUE_WAIT_NS] = c2m.track_wait_ns;
  g_events[event_base + DAE_TRACK_STORE_QUEUE_WAIT_CALLS] =
      c2m.track_wait_calls;
  g_events[event_base + DAE_TRACK_STORE_SERVICE_NS] = service_ns;
  g_events[event_base + DAE_TRACK_STORE_BARRIER_SERVICE_NS] =
      barrier_service_ns;
  g_events[event_base + DAE_TRACK_STORE_COMMANDS] = commands;
  g_events[event_base + DAE_TRACK_STORE_BARRIER_COMMANDS] = barrier_commands;
#endif
}
