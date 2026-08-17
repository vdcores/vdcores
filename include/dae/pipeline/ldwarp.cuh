#pragma once

#include <cuda/atomic>

#include <cutlass/arch/barrier.h>

#include "virtualcore.cuh"

constexpr int kLduRouteCount = 6;

struct LduRouteExperts {
  int rank0;
  int rank1;
  int rank2;
  int rank3;
  int rank4;
  int rank5;
};

__device__ __forceinline__ int ldu_route_expert(
    const LduRouteExperts &route_experts, int rank) {
  switch (rank) {
    case 0: return route_experts.rank0;
    case 1: return route_experts.rank1;
    case 2: return route_experts.rank2;
    case 3: return route_experts.rank3;
    case 4: return route_experts.rank4;
    case 5: return route_experts.rank5;
    default: return -1;
  }
}

__device__ __forceinline__ void ldu_cache_route_experts(
    uint64_t route_address, uint64_t &cached_route_address,
    LduRouteExperts &route_experts) {
  if (route_address == cached_route_address) {
    return;
  }
  const auto *route_ids = reinterpret_cast<const int *>(route_address);
  route_experts.rank0 = load_l2(route_ids + 0);
  route_experts.rank1 = load_l2(route_ids + 1);
  route_experts.rank2 = load_l2(route_ids + 2);
  route_experts.rank3 = load_l2(route_ids + 3);
  route_experts.rank4 = load_l2(route_ids + 4);
  route_experts.rank5 = load_l2(route_ids + 5);
  cached_route_address = route_address;
}

__device__ __forceinline__ void ldu_ensure_route_experts(
    uint64_t route_address, uint64_t &cached_route_address,
    LduRouteExperts &route_experts) {
  ldu_cache_route_experts(
      route_address, cached_route_address, route_experts);
}

__device__ __forceinline__ uint64_t ldu_resolve_routed_address(
    uint64_t state_address, uint16_t encoded_field_rank,
    const LduRouteExperts &route_experts) {
  constexpr int kHeaderInts = 12;
  const auto *header = reinterpret_cast<const int *>(state_address);
  const int route_rank = encoded_field_rank & 0x7;
  const int pointer_field = encoded_field_rank >> 3;
  if (route_rank < 0 || route_rank >= kLduRouteCount) {
    return 0;
  }
  const int expert = ldu_route_expert(route_experts, route_rank);
  const int field_stride = load_l2(header + 8);
  const int expert_count = load_l2(header + 9);
  if (expert < 0 || expert >= expert_count ||
      pointer_field < 0 || pointer_field >= field_stride) {
    return 0;
  }
  const auto *pointer_table =
      reinterpret_cast<const uint64_t *>(header + kHeaderInts);
  return load_l2_u64(
      pointer_table + expert * field_stride + pointer_field);
}

#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
template<typename M2LD_Type, typename M2C_Type>
__device__ __forceinline__ void ldu_execute_mxfp_down_weight_ring(
    M2LD_Type &m2ld, M2C_Type &m2c,
    const MInst inst, const uint8_t slot, const uint8_t bar,
    const void *smem_base, const CUtensorMap *tma_descs,
    int *slot_avail, uint64_t *tmem_mma_barriers
#if defined(DAE_TRACK_PROFILE)
    , uint64_t &commands
#endif
    ) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = mxfpDownLduWeightRingStages;
  constexpr int kTilesPerTask = 8;
  constexpr int kWeightPackedBytes = 16 * 1024;
  constexpr int kWeightStageBytes = 32 * 1024;

  // Publish allocation visibility independently of tile readiness. Each
  // compute task waits on the resident weight-full phases before UMMA.
  static_cast<void>(m2c.barriers[bar].arrive());
  auto *weight_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownLduWeightRingFullBarrierBase);
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownLduWeightRingEmptyBarrierBase);
  auto *weight_ring = static_cast<uint8_t *>(
      get_slot_address(smem_base, slot));
  uint32_t empty_phase[kStages] = {};
  uint8_t output_task = uint8_t(inst.coords[3]);

  for (int task = 0; task < int(inst.size); ++task) {
    if (task != 0) {
      // The allocator warp reserved this task's M2C entry and embedded its
      // output tile directly in the command. Descriptor, ring, and phase state
      // remain local to this long-running LDU invocation.
      m2ld.wait();
      const LdCmd continuation {
          .raw = m2ld.data[m2ld.ptr]
      };
      m2ld.advance();
      output_task = continuation.slot;
      static_cast<void>(m2c.barriers[continuation.bar].arrive());
#if defined(DAE_TRACK_PROFILE)
      ++commands;
#endif
    }

    #pragma unroll
    for (int tile = 0; tile < kTilesPerTask; ++tile) {
      const int stage = tile % kStages;
      stage_empty[stage].wait(empty_phase[stage]);
      empty_phase[stage] ^= 1U;
      const uint32_t destination = static_cast<uint32_t>(
          __cvta_generic_to_shared(
              weight_ring + stage * kWeightStageBytes));
      const uint32_t barrier = static_cast<uint32_t>(
          __cvta_generic_to_shared(weight_full + stage));
      asm volatile(
          "cp.async.bulk.tensor.5d.shared::cluster.global."
          "mbarrier::complete_tx::bytes "
          "[%0], [%1, {0, %2, %3, %4, %5}], [%6];"
          :: "r"(destination), "l"(tma_descs + inst.arg),
             "r"(0), "r"(0), "r"(tile), "r"(int(output_task)),
             "r"(barrier)
          : "memory");
      weight_full[stage].arrive_and_expect_tx(kWeightPackedBytes);
    }
  }

  // This is the lease's true last-use point: both final stage-empty phases
  // prove that no UMMA can still read either 32-KiB half.
  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    stage_empty[stage].wait(empty_phase[stage]);
  }
  atomicOr(slot_avail, int(mkSlotMask(slot, 8)));
}
#endif

template<typename M2LD_Type, typename M2C_Type>
__device__ __forceinline__ void ldwarp_execute_singlethread(
    M2LD_Type &m2ld, M2C_Type &m2c,
    MInst *st_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars,
    int *slot_avail,
    cuda::barrier<cuda::thread_scope_block> *ldu_control_barrier,
    cuda::barrier<cuda::thread_scope_block> *ldu_control_publish_barrier,
    uint64_t *tmem_mma_barriers,
    const int port_id
#if defined(DAE_TRACK_PROFILE)
    , const int sm_id, uint64_t *g_events
#endif
    ) {

  __ldprint("[LD Warp] Start LD warp execution");

  int regFile[4];
  uint64_t routedBaseAddress = 0;
  uint64_t cachedRouteAddress = 0;
  LduRouteExperts cachedRouteExperts;
#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
  // SFA and SFB may be assigned independently to either LDU. Track the
  // observed empty phase per (operand, shared stage), not merely per port.
  uint32_t mxScalePhaseMask = 0;
  uint64_t mxScaleBase[2] = {};
  uint32_t mxScaleTile[2] = {};
#endif
#if defined(DAE_TRACK_PROFILE)
  uint64_t dependency_wait_ns = 0;
  uint64_t dependency_contended = 0;
  uint64_t commands = 0;
  uint32_t profile_layer_counter = 0;
  uint32_t profile_reload_counter = 0;
#endif
#if defined(DAE_TRACK_MXFP_TIMELINE)
  uint32_t profile_mx_weight_tma_counter = 0;
  uint32_t profile_mx_activation_tma_counter = 0;
#endif
  m2ld.wait();
  LdCmd cmd { .raw = m2ld.data[m2ld.ptr] };

  while (cmd.slot != SLOT_END) {
#if defined(DAE_TRACK_PROFILE)
    ++commands;
#endif
    auto &slot = cmd.slot;
#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
    auto &opcode = cmd.opcode;
    MInst inst{};
    // Compact direct-scale commands carry their operand/stage in LdCmd::slot
    // and derive the source from LDU-local state. Their repeatedly overwritten
    // special MInst mailboxes are intentionally never read.
    if (op(opcode) != op(OP_ALLOC_TMA_LOAD_MX_SCALE_1D))
      inst = st_insts[slot];
#else
    auto inst = st_insts[slot];
#endif

    m2ld.advance();

#if !DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
    auto &opcode = cmd.opcode;
#endif
    auto &bar = cmd.bar;
    bool produces_compute_operand = true;

    if (op(opcode) == op(OP_LDU_RELOAD_BARRIERS) ||
        op(opcode) == op(OP_LDU_PROFILE_LAYER))
      ldu_control_publish_barrier->arrive_and_wait();

    __ldprint("Receive LD cmd: slot=%d bar=%d opcode=%d", slot, bar, op(opcode));

    // If its a readbar, we do the readbar
    // TODO(zhiyuang): wait bar here if bar is set
    if ((opcode & MEM_OP_FLAGS_BARRIER) && !(opcode & MEM_OP_FLAGS_WRITEBACK)) {
      volatile int *bar = bars + inst.bar();
#if defined(DAE_TRACK_PROFILE)
      const uint64_t dependency_start = cuda::ptx::get_sreg_globaltimer();
      if (*bar != 0)
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
      dependency_wait_ns +=
          cuda::ptx::get_sreg_globaltimer() - dependency_start;
#endif
      __ldprint("wait for global barrier before load: bar=%d", inst.bar());
    };

    // TODO(zhiyuang): change location?
    switch(op(opcode)) {
      case op(OP_ALLOC_TMA_LOAD_REG_1D):
      case op(OP_ALLOC_TMA_LOAD_1D): {
        // MInst::size is uint16_t.  Reserve the otherwise-invalid
        // size=0/arg=0xffff combination for one 64 KiB allocator load so a
        // shaped task can move two adjacent 32 KiB K512 records with one M2C
        // command.  Keep the retained-register form untouched because its
        // arg field names the destination register.
        constexpr uint16_t kTmaLoad64KMarker = 0xffff;
        const uint32_t transfer_size =
            op(opcode) == op(OP_ALLOC_TMA_LOAD_1D) &&
                    inst.size == 0 && inst.arg == kTmaLoad64KMarker &&
                    inst.nslot() == 8
                ? 64U * 1024U
                : uint32_t(inst.size);
        if (transfer_size == 0) {
          asm volatile("trap;");
        }
#if defined(DAE_TRACK_MXFP_TIMELINE)
        if (op(opcode) == op(OP_ALLOC_TMA_LOAD_1D) &&
            profile_mx_activation_tma_counter < 8) {
          g_events[sm_id * numProfileEvents +
                   mxfpProfileActivationTmaIssueBase +
                   profile_mx_activation_tma_counter] =
              cuda::ptx::get_sreg_globaltimer();
          ++profile_mx_activation_tma_counter;
        }
#endif
        __ldprint("TMA 1D Load: size=%u", transfer_size);
        // We need to get a slot ID first, as we will use its barrier
        cuda::device::memcpy_async_tx(
            (char *)(get_slot_address(smem_base, slot)),
            (char *)(inst.address),
            cuda::aligned_size_t<16>(transfer_size),
            m2c.barriers[bar]
        );
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(transfer_size)
        );
        if (op(opcode) == op(OP_ALLOC_TMA_LOAD_REG_1D)) {
          if (inst.arg >= 4) {
            asm volatile("trap;");
          }
          regFile[inst.arg] = mkSlotMask(slot, inst.nslot());
          __ldprint(
              "[REG] retain TMA: reg_id=%d slot=%d nslot=%d mask=0x%X",
              inst.arg, slot, inst.nslot(), regFile[inst.arg]);
        }
        break; }
#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
      case op(OP_ALLOC_TMA_LOAD_MX_SCALE_1D):
      case op(OP_ALLOC_TMA_LOAD_MX_SCALE_BASE_1D): {
        constexpr uint32_t kScaleHalfBytes = 2048;
        constexpr uint32_t kScaleStageBytes = 2 * kScaleHalfBytes;
        int operand = 0;
        int stage = 0;
        if (op(opcode) == op(OP_ALLOC_TMA_LOAD_MX_SCALE_BASE_1D)) {
          operand = inst.arg;
          if (operand >= 2 || operand != port_id ||
              slot != numSlots + 6 + operand || inst.address == 0) {
            asm volatile("trap;");
          }
          mxScaleBase[operand] = inst.address;
          mxScaleTile[operand] = 0;
          static_cast<void>(ldu_control_publish_barrier->arrive());
        } else {
          const int encoded = int(slot) - numSlots;
          if (encoded < 0 || encoded >= 2 * mxfp4Mxfp8TmaScaleStages) {
            asm volatile("trap;");
          }
          operand = encoded / mxfp4Mxfp8TmaScaleStages;
          stage = encoded % mxfp4Mxfp8TmaScaleStages;
        }
        const int phase_index = operand * mxfp4Mxfp8TmaScaleStages + stage;
        const uint32_t scale_tile = mxScaleTile[operand];
        if (mxScaleBase[operand] == 0 ||
            scale_tile >= 8 ||
            stage != int(scale_tile % mxfp4Mxfp8TmaScaleStages)) {
          asm volatile("trap;");
        }

#if defined(DAE_TRACK_MXFP_TIMELINE)
        const int producer_start_base = operand == 0
            ? mxfpProfileSfaProducerStartBase
            : mxfpProfileSfbProducerStartBase;
        g_events[sm_id * numProfileEvents + producer_start_base + scale_tile] =
            cuda::ptx::get_sreg_globaltimer();
#endif
        const uint32_t phase = (mxScalePhaseMask >> phase_index) & 1U;
        bool empty = cuda::ptx::mbarrier_try_wait_parity(
            cuda::ptx::sem_acquire,
            cuda::ptx::scope_cta,
            tmem_mma_barriers + mxfp4Mxfp8TmaScaleBarrierBase + stage,
            phase);
        while (!empty) {
          __nanosleep(barrierPollSleepCycles);
          empty = cuda::ptx::mbarrier_try_wait_parity(
              cuda::ptx::sem_acquire,
              cuda::ptx::scope_cta,
              tmem_mma_barriers + mxfp4Mxfp8TmaScaleBarrierBase + stage,
              phase);
        }
        mxScalePhaseMask ^= 1U << phase_index;
#if defined(DAE_TRACK_MXFP_TIMELINE)
        const int producer_ready_base = operand == 0
            ? mxfpProfileSfaProducerReadyBase
            : mxfpProfileSfbProducerReadyBase;
        g_events[sm_id * numProfileEvents + producer_ready_base + scale_tile] =
            cuda::ptx::get_sreg_globaltimer();
#endif

        auto *destination = static_cast<char *>(
            get_slot_address(smem_base, numSlots));
        destination += stage * kScaleStageBytes + operand * kScaleHalfBytes;
        cuda::device::memcpy_async_tx(
            destination,
            reinterpret_cast<const char *>(
                mxScaleBase[operand] +
                uint64_t(scale_tile) * kScaleHalfBytes),
            cuda::aligned_size_t<16>(kScaleHalfBytes),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar],
            cuda::aligned_size_t<16>(kScaleHalfBytes));
        ++mxScaleTile[operand];
        break; }
#endif
#if DAE_MXFP_GATE_UP_LDU_WEIGHT_RING && \
    !DAE_MXFP_GATE_UP_DIRECT_ACTIVATION
      case op(OP_ALLOC_TMA_LOAD_MX_WEIGHT_RING_5D): {
        using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
        constexpr int kStages = mxfpLduWeightRingStages;
        constexpr int kTilesPerProjection = 8;
        constexpr int kWeightPackedBytes = 32 * 1024;
        constexpr int kWeightStageBytes = 64 * 1024;

        // Allocation publication is independent of weight readiness. Compute
        // learns the ring base now, then each UMMA stage waits on weight_full.
        // The handler owns the lease until every final consumer has produced
        // the matching resident empty phase.
        static_cast<void>(m2c.barriers[bar].arrive());
        produces_compute_operand = false;
        auto *weight_full = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpLduWeightRingFullBarrierBase);
        auto *stage_empty = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpLduWeightRingEmptyBarrierBase);
        auto *weight_ring = static_cast<uint8_t *>(
            get_slot_address(smem_base, slot));
        uint32_t empty_phase[kStages] = {};

        #pragma unroll
        for (int projection = 0; projection < 2; ++projection) {
          if (projection == 1) {
            // Keep the lease, barrier phases, and descriptor state in this
            // LDU invocation. Only the compact queue marker is consumed.
            m2ld.wait();
            const LdCmd continuation {
                .raw = m2ld.data[m2ld.ptr]
            };
            static_cast<void>(continuation);
            m2ld.advance();
#if defined(DAE_TRACK_PROFILE)
            ++commands;
#endif
          }
          const uint16_t descriptor_index =
              projection == 0 ? inst.size : inst.arg;
          #pragma unroll
          for (int tile = 0; tile < kTilesPerProjection; ++tile) {
            const int stage = tile % kStages;
            stage_empty[stage].wait(empty_phase[stage]);
            empty_phase[stage] ^= 1U;
            const uint32_t destination = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    weight_ring + stage * kWeightStageBytes));
            const uint32_t barrier = static_cast<uint32_t>(
                __cvta_generic_to_shared(weight_full + stage));
            asm volatile(
                "cp.async.bulk.tensor.5d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {0, %2, %3, %4, %5}], [%6];"
                :: "r"(destination), "l"(tma_descs + descriptor_index),
                   "r"(0), "r"(0), "r"(tile), "r"(int(inst.coords[3])),
                   "r"(barrier)
                : "memory");
            weight_full[stage].arrive_and_expect_tx(kWeightPackedBytes);
          }
        }

        #pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
          stage_empty[stage].wait(empty_phase[stage]);
        }
        atomicOr(slot_avail, int(mkSlotMask(slot, 16)));
#if defined(DAE_TRACK_MXFP_TIMELINE)
        g_events[sm_id * numProfileEvents + mxfpProfileWeightRingRelease] =
            cuda::ptx::get_sreg_globaltimer();
#endif
        break; }
#endif
#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
      case op(OP_ALLOC_TMA_LOAD_MX_DOWN_WEIGHT_RING_5D): {
        produces_compute_operand = false;
        ldu_execute_mxfp_down_weight_ring(
            m2ld, m2c, inst, slot, bar,
            smem_base, tma_descs, slot_avail, tmem_mma_barriers
#if defined(DAE_TRACK_PROFILE)
            , commands
#endif
            );
        break; }
#endif
      case op(OP_ALLOC_INDIRECT_TMA_LOAD_1D):
      case op(OP_ALLOC_LAYER_TMA_LOAD_1D): {
        const uint64_t resolved = load_l2_u64(
            reinterpret_cast<const uint64_t *>(inst.address));
        if (resolved == 0) {
          asm volatile("trap;");
        }
        __ldprint(
            "Indirect TMA 1D load: size=%d resolved=0x%lx",
            inst.size, resolved);
        cuda::device::memcpy_async_tx(
            static_cast<char *>(get_slot_address(smem_base, slot)),
            reinterpret_cast<const char *>(resolved),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar], cuda::aligned_size_t<16>(inst.size));
        break; }
      case op(OP_ALLOC_LDU_LOAD_1D): {
        __ldprint("LDU 1D load: size=%d", inst.size);
        auto *dst = static_cast<unsigned char *>(
            get_slot_address(smem_base, slot));
        const auto *src = reinterpret_cast<const unsigned char *>(inst.address);
        int offset = 0;
        if ((reinterpret_cast<uintptr_t>(src) & 0xF) == 0) {
          for (; offset + 16 <= inst.size; offset += 16) {
            *reinterpret_cast<uint4 *>(dst + offset) =
                *reinterpret_cast<const uint4 *>(src + offset);
          }
        }
        if ((reinterpret_cast<uintptr_t>(src + offset) & 0x3) == 0) {
          for (; offset + 4 <= inst.size; offset += 4) {
            *reinterpret_cast<uint32_t *>(dst + offset) =
                *reinterpret_cast<const uint32_t *>(src + offset);
          }
        }
        if ((reinterpret_cast<uintptr_t>(src + offset) & 0x1) == 0) {
          for (; offset + 2 <= inst.size; offset += 2) {
            *reinterpret_cast<uint16_t *>(dst + offset) =
                *reinterpret_cast<const uint16_t *>(src + offset);
          }
        }
        for (; offset < inst.size; ++offset) {
          dst[offset] = src[offset];
        }
        break; }
      case op(OP_ALLOC_INDIRECT_LDU_LOAD_1D):
      case op(OP_ALLOC_LAYER_LDU_LOAD_1D): {
        const uint64_t resolved = load_l2_u64(
            reinterpret_cast<const uint64_t *>(inst.address));
        if (resolved == 0) {
          asm volatile("trap;");
        }
        __ldprint(
            "Indirect LDU 1D load: size=%d resolved=0x%lx",
            inst.size, resolved);
        auto *dst = static_cast<unsigned char *>(
            get_slot_address(smem_base, slot));
        const auto *src = reinterpret_cast<const unsigned char *>(resolved);
        int offset = 0;
        if ((resolved & 0xF) == 0) {
          for (; offset + 16 <= inst.size; offset += 16) {
            *reinterpret_cast<uint4 *>(dst + offset) =
                *reinterpret_cast<const uint4 *>(src + offset);
          }
        }
        if ((reinterpret_cast<uintptr_t>(src + offset) & 0x3) == 0) {
          for (; offset + 4 <= inst.size; offset += 4) {
            *reinterpret_cast<uint32_t *>(dst + offset) =
                *reinterpret_cast<const uint32_t *>(src + offset);
          }
        }
        if ((reinterpret_cast<uintptr_t>(src + offset) & 0x1) == 0) {
          for (; offset + 2 <= inst.size; offset += 2) {
            *reinterpret_cast<uint16_t *>(dst + offset) =
                *reinterpret_cast<const uint16_t *>(src + offset);
          }
        }
        for (; offset < inst.size; ++offset) {
          dst[offset] = src[offset];
        }
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
      case op(OP_ALLOC_LAYER_TMA_LOAD_4D):
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
#if defined(DAE_TRACK_MXFP_TIMELINE)
        if (profile_mx_weight_tma_counter < 8) {
          g_events[sm_id * numProfileEvents + mxfpProfileWeightTmaIssueBase +
                   profile_mx_weight_tma_counter] =
              cuda::ptx::get_sreg_globaltimer();
          ++profile_mx_weight_tma_counter;
        }
#endif
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
      case op(OP_ALLOC_ROUTED_TMA_LOAD_1D): {
        // HBM layout: eight int32 route-id slots, uint32 field stride,
        // uint32 expert count, two padding words, then row-major uint64
        // pointer entries. The low three arg bits select the route rank; the
        // remaining bits select the pointer field.
        ldu_ensure_route_experts(
            inst.address, cachedRouteAddress, cachedRouteExperts);
        const uint64_t resolved = ldu_resolve_routed_address(
            inst.address, inst.arg, cachedRouteExperts);
        if (resolved == 0) {
          asm volatile("trap;");
        }
        __ldprint(
            "Routed TMA 1D load: rank=%d field=%d size=%d resolved=0x%lx",
            inst.arg & 0x7, inst.arg >> 3, inst.size, resolved);
        cuda::device::memcpy_async_tx(
            static_cast<char *>(get_slot_address(smem_base, slot)),
            reinterpret_cast<const char *>(resolved),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar], cuda::aligned_size_t<16>(inst.size));
        break;
      }
      case op(OP_ALLOC_ROUTED_TMA_LOAD_BASE_1D): {
        ldu_ensure_route_experts(
            inst.address, cachedRouteAddress, cachedRouteExperts);
        const uint64_t resolved = ldu_resolve_routed_address(
            inst.address, inst.arg, cachedRouteExperts);
        if (resolved == 0) {
          asm volatile("trap;");
        }
        routedBaseAddress = resolved;
        __ldprint(
            "Routed TMA base: rank=%d field=%d size=%d resolved=0x%lx",
            inst.arg & 0x7, inst.arg >> 3, inst.size, resolved);
        cuda::device::memcpy_async_tx(
            static_cast<char *>(get_slot_address(smem_base, slot)),
            reinterpret_cast<const char *>(resolved),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar], cuda::aligned_size_t<16>(inst.size));
        break;
      }
      case op(OP_ALLOC_TMA_LOAD_ADDRESS_REG_1D): {
        if (inst.arg != 0 || routedBaseAddress == 0) {
          asm volatile("trap;");
        }
        const uint64_t resolved = routedBaseAddress + inst.address;
        __ldprint(
            "Address-register TMA: reg=%d offset=%lu size=%d resolved=0x%lx",
            inst.arg, inst.address, inst.size, resolved);
        cuda::device::memcpy_async_tx(
            static_cast<char *>(get_slot_address(smem_base, slot)),
            reinterpret_cast<const char *>(resolved),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar], cuda::aligned_size_t<16>(inst.size));
        break;
      }
      case op(OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_1D):
      case op(OP_ALLOC_LAYER_ROUTED_TMA_LOAD_1D):
      case op(OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_BASE_1D):
      case op(OP_ALLOC_LAYER_ROUTED_TMA_LOAD_BASE_1D): {
        constexpr int kHeaderInts = 12;
        // HBM descriptor: a fixed route-result pointer followed by the
        // current layer's ordinary RoutedAddressTable state pointer.
        const auto *descriptor =
            reinterpret_cast<const uint64_t *>(inst.address);
        const uint64_t route_address = load_l2_u64(descriptor + 0);
        const uint64_t state_address = load_l2_u64(descriptor + 1);
        if (route_address == 0 || state_address == 0) {
          asm volatile("trap;");
        }
        const auto *header = reinterpret_cast<const int *>(state_address);
        ldu_ensure_route_experts(
            route_address, cachedRouteAddress, cachedRouteExperts);
        const int route_rank = inst.arg & 0x7;
        const int pointer_field = inst.arg >> 3;
        uint64_t resolved = 0;
        if (route_rank >= 0 && route_rank < kLduRouteCount) {
          const int expert = ldu_route_expert(
              cachedRouteExperts, route_rank);
          const int field_stride = load_l2(header + 8);
          const int expert_count = load_l2(header + 9);
          if (expert >= 0 && expert < expert_count &&
              pointer_field >= 0 && pointer_field < field_stride) {
            const auto *pointer_table =
                reinterpret_cast<const uint64_t *>(header + kHeaderInts);
            resolved = load_l2_u64(
                pointer_table + expert * field_stride + pointer_field);
          }
        }
        if (resolved == 0) {
          asm volatile("trap;");
        }
        if (op(inst.opcode) == op(OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_BASE_1D) ||
            op(inst.opcode) == op(OP_ALLOC_LAYER_ROUTED_TMA_LOAD_BASE_1D)) {
          routedBaseAddress = resolved;
        }
        __ldprint(
            "Indirect routed TMA 1D load: rank=%d field=%d size=%d "
            "state=0x%lx resolved=0x%lx",
            route_rank, pointer_field, inst.size, state_address, resolved);
        cuda::device::memcpy_async_tx(
            static_cast<char *>(get_slot_address(smem_base, slot)),
            reinterpret_cast<const char *>(resolved),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar], cuda::aligned_size_t<16>(inst.size));
        break;
      }
      case op(OP_ALLOC_INDEXED_TMA_LOAD_1D): {
        // HBM record: base pointer, pointer to one int32 index, then packed
        // uint32 (row count, row stride). RepeatM advances across records.
        const auto *state = reinterpret_cast<const uint64_t *>(inst.address);
        const uint64_t base = load_l2_u64(state + 0);
        const uint64_t index_address = load_l2_u64(state + 1);
        const uint64_t shape = load_l2_u64(state + 2);
        const int row_count = static_cast<int>(shape & 0xFFFFFFFFULL);
        const int row_stride = static_cast<int>(shape >> 32);
        const int row = load_l2(
            reinterpret_cast<const int *>(index_address));
        if (base == 0 || index_address == 0 || row >= row_count ||
            row_stride < inst.size) {
          asm volatile("trap;");
        }
        if (row < 0) {
          auto *dst = static_cast<unsigned char *>(
              get_slot_address(smem_base, slot));
          for (int offset = 0; offset < inst.size; ++offset) {
            dst[offset] = 0;
          }
          break;
        }
        const uint64_t resolved = base + uint64_t(row) * row_stride;
        __ldprint(
            "Indexed TMA 1D load: row=%d size=%d resolved=0x%lx",
            row, inst.size, resolved);
        cuda::device::memcpy_async_tx(
            static_cast<char *>(get_slot_address(smem_base, slot)),
            reinterpret_cast<const char *>(resolved),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar], cuda::aligned_size_t<16>(inst.size));
        break;
      }
      case op(OP_ALLOC_INDIRECT_INDEXED_TMA_LOAD_1D):
      case op(OP_ALLOC_LAYER_INDEXED_TMA_LOAD_1D): {
        const uint64_t record_address = load_l2_u64(
            reinterpret_cast<const uint64_t *>(inst.address));
        if (record_address == 0) {
          asm volatile("trap;");
        }
        const auto *state = reinterpret_cast<const uint64_t *>(record_address);
        const uint64_t base = load_l2_u64(state + 0);
        const uint64_t index_address = load_l2_u64(state + 1);
        const uint64_t shape = load_l2_u64(state + 2);
        const int row_count = static_cast<int>(shape & 0xFFFFFFFFULL);
        const int row_stride = static_cast<int>(shape >> 32);
        if (base == 0 || index_address == 0) {
          asm volatile("trap;");
        }
        const int row = load_l2(reinterpret_cast<const int *>(index_address));
        if (row >= row_count || row_stride < inst.size) {
          asm volatile("trap;");
        }
        if (row < 0) {
          auto *dst = static_cast<unsigned char *>(
              get_slot_address(smem_base, slot));
          for (int offset = 0; offset < inst.size; ++offset) {
            dst[offset] = 0;
          }
          break;
        }
        const uint64_t resolved = base + uint64_t(row) * row_stride;
        cuda::device::memcpy_async_tx(
            static_cast<char *>(get_slot_address(smem_base, slot)),
            reinterpret_cast<const char *>(resolved),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]);
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar], cuda::aligned_size_t<16>(inst.size));
        break;
      }
      case op(OP_LDU_RELOAD_BARRIERS): {
        produces_compute_operand = false;
        // Route results are constant throughout one loop iteration.  Drop the
        // LDU-local all-rank cache only after both ports have drained so the
        // next layer/step cannot observe stale expert IDs.
        cachedRouteAddress = 0;
        // Both LDU lanes reach this point only after all earlier commands on
        // their own ports have drained and the loop-tail STU dependency has
        // reached zero. The first rendezvous therefore makes it safe for each
        // block's port 0 to restore a disjoint slice of the active bank.
        ldu_control_barrier->arrive_and_wait();
        if (port_id == 0) {
          int *arrivals = bars + lduBarrierReloadArrival;
          const int count = inst.size;
          // The attached completion barrier is the last barrier in the
          // active shifted bank. Derive that bank's first barrier instead of
          // restoring every bank on every loop iteration.
          const int first_bar = inst.bar() + 1 - count;
          const int *source = reinterpret_cast<const int *>(inst.address);
          if (first_bar < inst.arg || count <= 0 ||
              (first_bar - inst.arg) % count != 0 ||
              first_bar + count > lduBarrierReloadArrival) {
            asm volatile("trap;");
          }
          for (int offset = blockIdx.x; offset < count;
               offset += gridDim.x) {
            cuda::atomic_ref<int, cuda::thread_scope_device> destination(
                bars[first_bar + offset]);
            destination.store(
                load_l2(source + first_bar + offset),
                cuda::memory_order_relaxed);
          }

          // The release/acquire chain on the single arrivals word orders every
          // block's disjoint counter stores. All blocks observe the completed
          // phase before either LDU port can advance into the next iteration.
          cuda::atomic_ref<int, cuda::thread_scope_device> arrivals_ref(
              *arrivals);
          const int ticket = arrivals_ref.fetch_add(
              1, cuda::memory_order_acq_rel);
          const int phase_end = (ticket / gridDim.x + 1) * gridDim.x;
          while (arrivals_ref.load(cuda::memory_order_acquire) < phase_end) {
            __nanosleep(barrierPollSleepCycles);
          }
        }
        // Port 1 cannot consume a following loop iteration until port 0 has
        // observed the device-wide completion phase after restoring counters.
        ldu_control_barrier->arrive_and_wait();
#if defined(DAE_TRACK_PROFILE)
        if (port_id == 0 && inst.nslot() == numSlots + 2) {
          const int event_id = reloadProfileEventBase + profile_reload_counter;
          if (event_id >= trackProfileEventBase) {
            asm volatile("trap;");
          }
          g_events[sm_id * numProfileEvents + event_id] =
              cuda::ptx::get_sreg_globaltimer();
          ++profile_reload_counter;
        }
#endif
        break;
      }
      case op(OP_LDU_PROFILE_LAYER): {
        produces_compute_operand = false;
#if defined(DAE_TRACK_PROFILE)
        if (port_id == 0) {
          const int event_id = inst.arg + profile_layer_counter;
          if (profile_layer_counter >= inst.size ||
              event_id < layerProfileEventBase ||
              event_id >= reloadProfileEventBase) {
            asm volatile("trap;");
          }
          g_events[sm_id * numProfileEvents + event_id] =
              cuda::ptx::get_sreg_globaltimer();
          ++profile_layer_counter;
        }
#endif
        break;
      }
    }

    // m2c data should be prepared in the CFU
    if (produces_compute_operand)
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
#endif
  // __print(0, "End of LD warp execution");
}
