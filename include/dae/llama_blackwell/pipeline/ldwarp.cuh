#pragma once

#include "virtualcore.cuh"

template<typename M2LD_Type, typename M2C_Type>
__device__ __forceinline__ void ldwarp_execute_singlethread(
    M2LD_Type &m2ld, M2C_Type &m2c,
    const MInst *st_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars
#if defined(DAE_TRACK_PROFILE)
    , const int sm_id, const int port_id, uint64_t *g_events
#endif
    ) {

  __ldprint("[LD Warp] Start LD warp execution");

  int regFile[4];
#if defined(DAE_TRACK_PROFILE)
  uint64_t dependency_wait_ns = 0;
  uint64_t dependency_contended = 0;
  uint64_t commands = 0;
  uint64_t max_dependency_wait_ns = 0;
  uint64_t max_dependency_bar = 0;
  uint64_t max_dependency_command = 0;
#if defined(DAE_TRACK_LLAMA8B_DEPENDENCY_HISTOGRAM)
  uint64_t dependency_role_ns[3] = {};
#endif
#if defined(DAE_TRACK_LLAMA8B_LAYER_TIMELINE)
  uint32_t llama_layer_wait_index = 0;
  uint32_t llama_pre_rms_wait_index = 0;
#endif
#endif
  m2ld.wait();
  LdCmd cmd { .raw = m2ld.data[m2ld.ptr] };

  while (cmd.slot != SLOT_END) {
#if defined(DAE_TRACK_PROFILE)
    ++commands;
#endif
    auto &slot = cmd.slot;
    auto inst = st_insts[slot];

    m2ld.advance();

    auto &opcode = cmd.opcode;
    auto &bar = cmd.bar;

    __ldprint("Receive LD cmd: slot=%d bar=%d opcode=%d", slot, bar, op(opcode));

    // If its a readbar, we do the readbar
    // TODO(zhiyuang): wait bar here if bar is set
    if ((opcode & MEM_OP_FLAGS_BARRIER) && !(opcode & MEM_OP_FLAGS_WRITEBACK)) {
      volatile int *bar = bars + inst.bar();
#if defined(DAE_TRACK_PROFILE)
      const uint64_t dependency_start = cuda::ptx::get_sreg_globaltimer();
      const bool dependency_was_contended = *bar != 0;
      if (dependency_was_contended)
        ++dependency_contended;
#endif
      // bool first_wait = true;
      // if (blockIdx.x == 0 && first_wait) {
      //   printf("[LD][sm=%d] check bar=%d bars[bar]=%d\n", blockIdx.x, inst.bar(), *bar);
      // }
      while (*bar != 0) {
        // busy wait
        __nanosleep(barrierPollSleepCycles);
        // if (blockIdx.x == 0 && first_wait) {
        //   printf("[LD][sm=%d] waiting bar=%d bars[bar]=%d\n", blockIdx.x, inst.bar(), *bar);
        //   first_wait = false;
        // }
      }
#if defined(DAE_TRACK_PROFILE)
      const uint64_t dependency_elapsed =
          cuda::ptx::get_sreg_globaltimer() - dependency_start;
      dependency_wait_ns += dependency_elapsed;
#if defined(DAE_TRACK_LLAMA8B_LAYER_TIMELINE)
      // The qualified Llama schedule has exactly one bar_layer read on LDU0
      // for each RMS owner and exactly one bar_pre_attn_rms read on LDU1 for
      // each Q owner per layer.  Keep their resident-loop occurrences instead
      // of overwriting the final one.  Local CTA-relative uint32 nanoseconds
      // cover far more than a single-token diagnostic and leave one uint64
      // profile cell per layer.
      const int timeline_relative_bar = static_cast<int>(inst.bar()) - 4;
      const int timeline_bar_role = timeline_relative_bar >= 0
          ? timeline_relative_bar % 30
          : -1;
      uint32_t *timeline_index = nullptr;
      int timeline_base = -1;
      if (port_id == 0 && timeline_bar_role == 0) {
        timeline_index = &llama_layer_wait_index;
        timeline_base = daeTrackLlamaLayerWaitBase;
      } else if (port_id == 1 && timeline_bar_role == 8) {
        timeline_index = &llama_pre_rms_wait_index;
        timeline_base = daeTrackLlamaPreRmsWaitBase;
      }
      if (timeline_index != nullptr) {
        const uint32_t occurrence = (*timeline_index)++;
        if (occurrence < daeTrackLlamaLayers) {
          const uint64_t cta_start =
              g_events[sm_id * numProfileEvents + 0];
          const uint32_t start_delta =
              static_cast<uint32_t>(dependency_start - cta_start);
          const uint32_t end_delta = static_cast<uint32_t>(
              dependency_start + dependency_elapsed - cta_start);
          g_events[sm_id * numProfileEvents + timeline_base + occurrence] =
              (static_cast<uint64_t>(end_delta) << 32) | start_delta;
        }
      }
#endif
#if defined(DAE_TRACK_LLAMA8B_FRONTIER)
      // Record the final-layer consumer edge for the two whole-stage Llama
      // barriers.  Repeated layers overwrite the same per-SM cells; the last
      // timestamp is therefore the resident loop's final occurrence.
      const int frontier_relative_bar = static_cast<int>(inst.bar()) - 4;
      const int frontier_bar_role = frontier_relative_bar >= 0
          ? frontier_relative_bar % 30
          : -1;
      if (port_id == 0 &&
          (frontier_bar_role == 1 || frontier_bar_role == 0)) {
        const int frontier_base = frontier_bar_role == 1
            ? DAE_TRACK_LLAMA_OUT_WAIT_START
            : DAE_TRACK_LLAMA_LAYER_WAIT_START;
        const int event_base = sm_id * numProfileEvents;
        g_events[event_base + frontier_base] = dependency_start;
        g_events[event_base + frontier_base + 1] =
            dependency_start + dependency_elapsed;
      }
#endif
      if (dependency_elapsed > max_dependency_wait_ns) {
        max_dependency_wait_ns = dependency_elapsed;
        max_dependency_bar = inst.bar();
        max_dependency_command = commands;
      }
#if defined(DAE_TRACK_LLAMA8B_DEPENDENCY_HISTOGRAM)
      // The qualified Llama-8B image allocates 30 barriers per layer starting
      // at ID 4: layer=0, out_mlp=1, pre_rms=8, and post_rms=9.  Keep this
      // app-specific classification behind an explicit diagnostic build flag.
      const int layer_relative_bar = static_cast<int>(inst.bar()) - 4;
      const int layer_bar_role = layer_relative_bar >= 0
          ? layer_relative_bar % 30
          : -1;
      int dependency_role = 2;
      if (port_id == 0) {
        if (layer_bar_role == 1)
          dependency_role = 0;
        else if (layer_bar_role == 0)
          dependency_role = 1;
      } else {
        if (layer_bar_role == 8)
          dependency_role = 0;
        else if (layer_bar_role == 9)
          dependency_role = 1;
      }
      dependency_role_ns[dependency_role] += dependency_elapsed;
#endif
#if defined(DAE_TRACK_DEPENDENCY_TRACE_SM)
      if (dependency_was_contended &&
          sm_id == DAE_TRACK_DEPENDENCY_TRACE_SM) {
        printf(
            "[track-dependency] sm=%d port=%d command=%llu bar=%u wait_ns=%llu\n",
            sm_id,
            port_id,
            static_cast<unsigned long long>(commands),
            static_cast<unsigned>(inst.bar()),
            static_cast<unsigned long long>(dependency_elapsed));
      }
#endif
#endif
      __ldprint("wait for global barrier before load: bar=%d", inst.bar());
    };

    // TODO(zhiyuang): change location?
    switch(op(opcode)) {
      case op(OP_ALLOC_TMA_LOAD_1D): {
        __ldprint("TMA 1D Load: size=%d", inst.size);
        // We need to get a slot ID first, as we will use its barrier
        cuda::device::memcpy_async_tx(
            (char *)(get_slot_address(smem_base, slot)),
            (char *)(inst.address),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]
        );
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_TENSOR_1D): {
        __ldprint("TMA Tensor 1D Load: size=%d", inst.size);
        asm volatile(
          "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2}], [%3];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((uint32_t)inst.address),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_2D): {
        const uint16_t *cord = inst.coords;
        __ldprint("TMA 2D Load: desc_idx=%d size=%d cord=(%d,%d)", inst.arg, inst.size, cord[0], cord[1]);
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3}], [%4];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_3D): {
        const uint16_t *cord = inst.coords;
        __ldprint("TMA 3D Load: desc_idx=%d size=%d cord=(%d,%d,%d)", inst.arg, inst.size, cord[0], cord[1], cord[2]);
        asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3, %4}], [%5];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((int)cord[2]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_4D): {
        const uint16_t *cord = inst.coords;
        __ldprint("TMA 4D Load: desc_idx=%d size=%d cord=(%d,%d,%d,%d)",
          inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
        asm volatile(
          "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((int)cord[2]),
            "r"((int)cord[3]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_5D_FIX0): {
        const uint16_t *cord = inst.coords;
        // hardcode first coord to be 0
        __ldprint("TMA 5D Load: desc_idx=%d size=%d cord=(0,%d,%d,%d,%d)",
          inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
        asm volatile(
          "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {0, %2, %3, %4, %5}], [%6];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((int)cord[2]),
            "r"((int)cord[3]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_WB_REG_STORE): {
        // TODO(zhiyuang): recalculate the mask or read from smem?
        int slotMask = mkSlotMask(slot, inst.nslot());
        m2c.data[bar] = slotMask | 0x80000000U; // set high bit to invalidate the writeback
        regFile[inst.size] = slotMask;
        __ldprint("[REG] store: reg_id=%d slot=%d nslot=%d bar=%d slotMask=0x%X",
          inst.size, slot, inst.nslot(), bar, slotMask);
        break;
      }
      case op(OP_ALLOC_REG_LOAD): {
        m2c.data[bar] = regFile[inst.size];
        __ldprint("[REG] load: reg_id=%d bar=%d slotMask=0x%X", inst.size, bar, regFile[inst.size]);
        break;
      }
    }

    // m2c data should be prepared in the CFU
    (void)m2c.barriers[bar].arrive();

    m2ld.wait();
    cmd.raw = m2ld.data[m2ld.ptr];
  } // End of LD warp loop

  __ldprint("End of LD warp execution");
#if defined(DAE_TRACK_PROFILE)
  const int event_base = sm_id * numProfileEvents;
  const int port_base = port_id == 0
      ? DAE_TRACK_LDU0_QUEUE_WAIT_NS
      : DAE_TRACK_LDU1_QUEUE_WAIT_NS;
  g_events[event_base + port_base + 0] = m2ld.track_wait_ns;
  g_events[event_base + port_base + 1] = m2ld.track_wait_calls;
  g_events[event_base + port_base + 2] = dependency_wait_ns;
  g_events[event_base + port_base + 3] = dependency_contended;
  g_events[event_base + port_base + 4] = commands;
#if !defined(DAE_TRACK_LLAMA8B_FRONTIER) && \
    !defined(DAE_TRACK_LLAMA8B_STORE_SERVICE)
  const int max_dependency_base = port_id == 0
      ? DAE_TRACK_LDU0_MAX_DEPENDENCY_NS
      : DAE_TRACK_LDU1_MAX_DEPENDENCY_NS;
  g_events[event_base + max_dependency_base + 0] = max_dependency_wait_ns;
  g_events[event_base + max_dependency_base + 1] = max_dependency_bar;
  g_events[event_base + max_dependency_base + 2] = max_dependency_command;
#if defined(DAE_TRACK_LLAMA8B_DEPENDENCY_HISTOGRAM)
  g_events[event_base + max_dependency_base + 0] = dependency_role_ns[0];
  g_events[event_base + max_dependency_base + 1] = dependency_role_ns[1];
  g_events[event_base + max_dependency_base + 2] = dependency_role_ns[2];
#endif
#endif
#endif
  // __print(0, "End of LD warp execution");
}
