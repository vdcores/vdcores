#pragma once

#include "virtualcore.cuh"

template<typename C2M_Type>
__device__ __forceinline__ void
stwarp_execute_mxfp_resident_down_reduction(
    C2M_Type &c2m, const MInst *minsts, const void *smem_base,
    const CUtensorMap *tma_descs, int *bars,
    uint64_t *tmem_mma_barriers
#if defined(DAE_TRACK_PROFILE)
    , const int sm_id, uint64_t *g_events
#endif
) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  auto *store_ready = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentStoreReadyBarrierBase);
  auto *store_done = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentStoreDoneBarrierBase);
  const auto *plan = reinterpret_cast<const uint64_t *>(minsts[0].address);
  const int task_count = load_l2(reinterpret_cast<const int *>(plan + 3));
  const uint32_t source = uint32_t(__cvta_generic_to_shared(
      static_cast<const uint8_t *>(smem_base) +
      dae_mxfp_resident_ffn::kDownOutputOffset));

#if defined(DAE_TRACK_PROFILE)
  const uint64_t service_start = cuda::ptx::get_sreg_globaltimer();
#endif
  #pragma unroll
  for (int task = 0; task < 2; ++task) {
    if (task >= task_count) {
      break;
    }
    store_ready[task].wait(0);
    const auto *metadata = reinterpret_cast<const uint8_t *>(
        load_l2_u64(plan + 1 + task));
    const uint64_t tma_info = load_l2_u64(
        reinterpret_cast<const uint64_t *>(metadata + 24));
    const uint16_t output_tma_index = uint16_t(tma_info >> 16);
    const uint32_t output_task = uint32_t(tma_info >> 32);
    const uint64_t barrier_info = load_l2_u64(
        reinterpret_cast<const uint64_t *>(metadata + 32));
    const uint32_t reduce_bar = uint32_t(barrier_info >> 32);
    const uint32_t resident_flags = *reinterpret_cast<const uint32_t *>(
        metadata + 68);
    const bool reduce_from_zero = (resident_flags & 1U) != 0;
    constexpr int kTileM = 128;
    constexpr int kDownTilesPerExpert = 4096 / kTileM;
    const int expert = int(output_task) / kDownTilesPerExpert;
    const int output_m_tile = int(output_task) % kDownTilesPerExpert;
    const int row = output_m_tile * kTileM;

    if (expert == 0 && !reduce_from_zero) {
      asm volatile(
          "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
          "[%0, {%1, %2}], [%3];\n"
          :
          : "l"((void *)(tma_descs + output_tma_index)),
            "r"(0), "r"(row), "r"(source)
          : "memory");
    } else {
      asm volatile(
          "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "
          "[%0, {%1, %2}], [%3];\n"
          :
          : "l"((void *)(tma_descs + output_tma_index)),
            "r"(0), "r"(row), "r"(source)
          : "memory");
    }
    cuda::ptx::cp_async_bulk_commit_group();
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    cuda::ptx::fence_proxy_async();
    if (expert == 0 && !reduce_from_zero) {
      cuda::atomic_ref<int, cuda::thread_scope_device> shared_ready(
          bars[reduce_bar]);
      shared_ready.fetch_sub(1, cuda::memory_order_release);
    }
    store_done[task].arrive();
  }

  // Compute publishes the ordinary zero terminator only after both STU
  // acknowledgements, preserving the normal C2M queue lifetime.
  (void)c2m.template pop<>();
#if defined(DAE_TRACK_PROFILE)
  const int event_base = sm_id * numProfileEvents;
  g_events[event_base + DAE_TRACK_STORE_QUEUE_WAIT_NS] = c2m.track_wait_ns;
  g_events[event_base + DAE_TRACK_STORE_QUEUE_WAIT_CALLS] =
      c2m.track_wait_calls;
  g_events[event_base + DAE_TRACK_STORE_SERVICE_NS] =
      cuda::ptx::get_sreg_globaltimer() - service_start;
  g_events[event_base + DAE_TRACK_STORE_BARRIER_SERVICE_NS] = 0;
  g_events[event_base + DAE_TRACK_STORE_COMMANDS] = task_count;
  g_events[event_base + DAE_TRACK_STORE_BARRIER_COMMANDS] = 0;
#endif
}

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
      atomicSub(&bars[inst.bar()], 1);
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
