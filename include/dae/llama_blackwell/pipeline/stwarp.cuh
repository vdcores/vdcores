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
#if defined(DAE_TRACK_PROFILE) && defined(DAE_TRACK_LLAMA8B_STORE_SERVICE)
    int llama_store_event = -1;
    if (opcode & MEM_OP_FLAGS_BARRIER) {
      const int layer_relative_bar = static_cast<int>(inst.bar()) - 4;
      const int layer_bar_role = layer_relative_bar >= 0
          ? layer_relative_bar % 30
          : -1;
      llama_store_event = layer_bar_role == 1
          ? DAE_TRACK_LLAMA_OUT_STORE_START
          : (layer_bar_role == 0 ? DAE_TRACK_LLAMA_LAYER_STORE_START : -1);
      if (llama_store_event >= 0) {
        g_events[sm_id * numProfileEvents + llama_store_event] = service_start;
      }
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
#if defined(DAE_TRACK_PROFILE) && defined(DAE_TRACK_LLAMA8B_STORE_SERVICE)
      if (llama_store_event >= 0) {
        const int event_base = sm_id * numProfileEvents + llama_store_event;
        g_events[event_base + 1] = cuda::ptx::get_sreg_globaltimer();
        g_events[event_base + 2] = commands;
      }
#endif
#if defined(DAE_TRACK_PROFILE) && \
    (defined(DAE_TRACK_LLAMA8B_FRONTIER) || \
     defined(DAE_TRACK_LLAMA8B_LAYER_TIMELINE))
      const int barrier_before_release = atomicSub(&bars[inst.bar()], 1);
      if (barrier_before_release == 1) {
        const int layer_relative_bar = static_cast<int>(inst.bar()) - 4;
        const int layer_bar_role = layer_relative_bar >= 0
            ? layer_relative_bar % 30
            : -1;
#if defined(DAE_TRACK_LLAMA8B_FRONTIER)
        const int frontier_event = layer_bar_role == 1
            ? DAE_TRACK_LLAMA_OUT_RELEASE
            : (layer_bar_role == 0 ? DAE_TRACK_LLAMA_LAYER_RELEASE : -1);
        if (frontier_event >= 0) {
          g_events[sm_id * numProfileEvents + frontier_event] =
              cuda::ptx::get_sreg_globaltimer();
        }
#endif
#if defined(DAE_TRACK_LLAMA8B_LAYER_TIMELINE)
        int release_base = -1;
        int release_counter = -1;
        int release_limit = 0;
        if (layer_bar_role == 0) {
          release_base = daeTrackLlamaLayerReleaseBase;
          release_counter = daeTrackLlamaLayerReleaseCounter;
          release_limit = daeTrackLlamaLayers;
        } else if (layer_bar_role == 8) {
          // Includes the embedding RMS release before layer 0 and the final
          // next-RMS release consumed by LM head: 32 layers plus one.
          release_base = daeTrackLlamaPreRmsReleaseBase;
          release_counter = daeTrackLlamaPreRmsReleaseCounter;
          release_limit = daeTrackLlamaLayers + 1;
        }
        if (release_base >= 0 && gridDim.x > daeTrackLlamaGlobalRow) {
          auto *counter = reinterpret_cast<unsigned long long *>(
              g_events + daeTrackLlamaGlobalRow * numProfileEvents +
              release_counter);
          const unsigned long long occurrence = atomicAdd(counter, 1ULL);
          if (occurrence < static_cast<unsigned long long>(release_limit)) {
            const uint64_t timestamp = cuda::ptx::get_sreg_globaltimer();
            const uint32_t timestamp_delta = static_cast<uint32_t>(
                timestamp - g_events[sm_id * numProfileEvents + 0]);
            g_events[daeTrackLlamaGlobalRow * numProfileEvents + release_base +
                     occurrence] =
                static_cast<uint64_t>(timestamp_delta) |
                (static_cast<uint64_t>(sm_id) << 32) |
                (static_cast<uint64_t>(inst.coords[0] & 0x0fff) << 40) |
                (static_cast<uint64_t>(inst.coords[1] & 0x0fff) << 52);
          }
        }
#endif
      }
#else
      atomicSub(&bars[inst.bar()], 1);
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
    const uint64_t current_service_ns =
        cuda::ptx::get_sreg_globaltimer() - service_start;
    service_ns += current_service_ns;
    if (opcode & MEM_OP_FLAGS_BARRIER)
      barrier_service_ns += current_service_ns;
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
