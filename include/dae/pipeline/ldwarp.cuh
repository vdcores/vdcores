#pragma once

#include <cuda/atomic>

#include <cutlass/arch/barrier.h>

#include "mxfp_resident_ffn.cuh"
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

__device__ __forceinline__ uint64_t ldu_mxfp_streaming_cache_policy() {
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
      : "=l"(policy));
  return policy;
}

template<bool Streaming>
__device__ __forceinline__ void ldu_issue_mxfp_weight_tma(
    const uint32_t destination, const CUtensorMap *descriptor,
    const int tile, const int output_task, const uint32_t barrier,
    const uint64_t cache_policy) {
  if constexpr (Streaming) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global."
        "mbarrier::complete_tx::bytes.L2::cache_hint "
        "[%0], [%1, {0, %2, %3, %4, %5}], [%6], %7;"
        :: "r"(destination), "l"(descriptor),
           "r"(0), "r"(0), "r"(tile), "r"(output_task),
           "r"(barrier), "l"(cache_policy)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global."
        "mbarrier::complete_tx::bytes "
        "[%0], [%1, {0, %2, %3, %4, %5}], [%6];"
        :: "r"(destination), "l"(descriptor),
           "r"(0), "r"(0), "r"(tile), "r"(output_task),
           "r"(barrier)
        : "memory");
  }
}

__device__ __forceinline__ void ldu_prefetch_mxfp_weight_tma(
    const CUtensorMap *descriptor, const int tile, const int output_task,
    const uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint "
      "[%0, {0, %1, %2, %3, %4}], %5;"
      :: "l"(descriptor), "r"(0), "r"(0), "r"(tile), "r"(output_task),
         "l"(cache_policy)
      : "memory");
}

#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
template<typename M2LD_Type, typename M2C_Type>
__device__ __forceinline__ void ldu_execute_mxfp_down_weight_ring(
    M2LD_Type &m2ld, M2C_Type &m2c,
    const MInst inst, const uint8_t slot, const uint8_t bar,
    const void *smem_base, const CUtensorMap *tma_descs,
    int *slot_avail,
    cutlass::arch::ClusterTransactionBarrier *weight_full,
    cutlass::arch::ClusterTransactionBarrier *weight_scale_full,
    cutlass::arch::ClusterTransactionBarrier *stage_empty,
    uint32_t *empty_phase,
    const int release_slots
#if defined(DAE_TRACK_PROFILE)
    , uint64_t &commands
#endif
    ) {
  constexpr int kStages = mxfpDownLduWeightRingStages;
  constexpr int kTilesPerTask = 8;
  constexpr int kWeightPackedBytes = 16 * 1024;
  constexpr int kWeightStageBytes = 32 * 1024;
  constexpr int kScalePackedBytes = 1024;
  constexpr int kScaleStageBytes = 2 * kScalePackedBytes;

  uint64_t cache_policy = 0;
  if constexpr (mxfpWeightPrefetchEnabled) {
    cache_policy = ldu_mxfp_streaming_cache_policy();
  }

  // Publish allocation visibility independently of tile readiness. Each
  // compute task waits on the resident weight-full phases before UMMA.
  static_cast<void>(m2c.barriers[bar].arrive());
  auto *weight_ring = static_cast<uint8_t *>(
      get_slot_address(smem_base, slot));
  auto *scale_ring = weight_ring + kStages * kWeightStageBytes;
  uint8_t output_task = uint8_t(inst.coords[3]);
  uint64_t scale_base = 0;
  if constexpr (mxfpWeightScaleTmaEnabled) {
    scale_base = *reinterpret_cast<const uint64_t *>(
        tma_descs + inst.coords[0]);
  }

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
      ldu_issue_mxfp_weight_tma<mxfpWeightPrefetchEnabled>(
          destination, tma_descs + inst.arg, tile, int(output_task), barrier,
          cache_policy);
      if constexpr (mxfpDownWeightScaleSeparateBarrierEnabled) {
        // Submit the latency-critical transformed weight first. The small SFA
        // transaction still has an independent completion phase and can reach
        // TMEM before the weight becomes visible, without occupying the first
        // request position on this LDU.
        weight_full[stage].arrive_and_expect_tx(kWeightPackedBytes);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            scale_ring + stage * kScaleStageBytes,
            reinterpret_cast<const uint8_t *>(scale_base) +
                (int(output_task) * kTilesPerTask + tile) *
                    kScalePackedBytes,
            uint32_t(kScalePackedBytes),
            reinterpret_cast<uint64_t *>(weight_scale_full + stage));
        weight_scale_full[stage].arrive_and_expect_tx(kScalePackedBytes);
      } else if constexpr (mxfpWeightScaleTmaEnabled) {
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            scale_ring + stage * kScaleStageBytes,
            reinterpret_cast<const uint8_t *>(scale_base) +
                (int(output_task) * kTilesPerTask + tile) *
                    kScalePackedBytes,
            uint32_t(kScalePackedBytes),
            reinterpret_cast<uint64_t *>(weight_full + stage));
        weight_full[stage].arrive_and_expect_tx(
            kWeightPackedBytes + kScalePackedBytes);
      } else {
        weight_full[stage].arrive_and_expect_tx(kWeightPackedBytes);
      }
      if constexpr (mxfpWeightPrefetchEnabled) {
        if (tile + 1 < kTilesPerTask) {
          ldu_prefetch_mxfp_weight_tma(
              tma_descs + inst.arg, tile + 1, int(output_task), cache_policy);
        }
      }
    }
  }

  // This is the lease's true last-use point: both final stage-empty phases
  // prove that no UMMA can still read either 32-KiB half.
  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    stage_empty[stage].wait(empty_phase[stage]);
  }
  if (release_slots != 0) {
    atomicOr(slot_avail, int(mkSlotMask(slot, release_slots)));
  }
}
#endif

#if DAE_MXFP_GATE_UP_LDU_WEIGHT_RING && \
    !DAE_MXFP_GATE_UP_DIRECT_ACTIVATION
__device__ __noinline__ void ldu_execute_mxfp_resident_linear1(
    const MInst inst, const void *smem_base, const CUtensorMap *tma_descs,
    uint64_t *tmem_mma_barriers
#if defined(DAE_TRACK_MXFP_TIMELINE)
    , const int sm_id, uint64_t *g_events
#endif
    ) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = dae_mxfp_resident_ffn::kLinear1Stages;
  constexpr int kTilesPerProjection = 8;
  constexpr int kWeightPackedBytes = 32 * 1024;
  constexpr int kWeightStageBytes =
      dae_mxfp_resident_ffn::kLinear1WeightStageBytes;
  constexpr int kScalePackedBytes = 2048;
  constexpr int kScaleStageBytes =
      dae_mxfp_resident_ffn::kLinear1ScaleStageBytes;
  constexpr int kActivationBytes =
      dae_mxfp_resident_ffn::kLinear1ActivationBytes;
  constexpr int kActivationChunkBytes = 16 * 1024;

  const auto *metadata = reinterpret_cast<const uint8_t *>(inst.address);
  const auto *activation_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 0)));
  const auto *gate_scale_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 16)));
  const auto *activation_scale_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 24)));
  const auto *up_scale_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 32)));
  const uint64_t tma_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 40));
  const uint16_t gate_descriptor_index = uint16_t(tma_info);
  const uint16_t up_descriptor_index = uint16_t(tma_info >> 16);
  const int output_tile = int(uint32_t(tma_info >> 32));

  auto *resident_base = reinterpret_cast<uint8_t *>(
      const_cast<void *>(smem_base));
  auto *weight_ring = resident_base +
      dae_mxfp_resident_ffn::kLinear1WeightRingOffset;
  auto *scale_ring = resident_base +
      dae_mxfp_resident_ffn::kLinear1ScaleRingOffset;
  auto *activation = resident_base +
      dae_mxfp_resident_ffn::kLinear1ActivationOffset;
  auto *weight_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpLduWeightRingFullBarrierBase);
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpLduWeightRingEmptyBarrierBase);
  uint32_t empty_phase[kStages] = {};
  uint64_t cache_policy = 0;
  if constexpr (mxfpWeightPrefetchEnabled) {
    cache_policy = ldu_mxfp_streaming_cache_policy();
  }

  #pragma unroll
  for (int projection = 0; projection < 2; ++projection) {
    const auto *weight_scale_global = projection == 0
        ? gate_scale_global
        : up_scale_global;
    const uint16_t descriptor_index = projection == 0
        ? gate_descriptor_index
        : up_descriptor_index;
    #pragma unroll
    for (int tile = 0; tile < kTilesPerProjection; ++tile) {
      const int operation = projection * kTilesPerProjection + tile;
      const int stage = operation % kStages;
      stage_empty[stage].wait(empty_phase[stage]);
      empty_phase[stage] ^= 1U;

      const uint32_t destination = static_cast<uint32_t>(
          __cvta_generic_to_shared(
              weight_ring + stage * kWeightStageBytes));
      const uint32_t barrier = static_cast<uint32_t>(
          __cvta_generic_to_shared(weight_full + stage));
      ldu_issue_mxfp_weight_tma<mxfpWeightPrefetchEnabled>(
          destination, tma_descs + descriptor_index, tile, output_tile,
          barrier, cache_policy);
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          scale_ring + stage * kScaleStageBytes,
          weight_scale_global + tile * kScalePackedBytes,
          uint32_t(kScalePackedBytes),
          reinterpret_cast<uint64_t *>(weight_full + stage));
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          scale_ring + stage * kScaleStageBytes + kScalePackedBytes,
          activation_scale_global + tile * kScalePackedBytes,
          uint32_t(kScalePackedBytes),
          reinterpret_cast<uint64_t *>(weight_full + stage));

      int transaction_bytes = kWeightPackedBytes + 2 * kScalePackedBytes;
      if (projection == 0 && tile == 0) {
        #pragma unroll
        for (int chunk = 0;
             chunk < kActivationBytes / kActivationChunkBytes; ++chunk) {
          cuda::ptx::cp_async_bulk(
              cuda::ptx::space_shared,
              cuda::ptx::space_global,
              activation + chunk * kActivationChunkBytes,
              activation_global + chunk * kActivationChunkBytes,
              uint32_t(kActivationChunkBytes),
              reinterpret_cast<uint64_t *>(weight_full + stage));
        }
        transaction_bytes += kActivationBytes;
      }
      weight_full[stage].arrive_and_expect_tx(transaction_bytes);

      if constexpr (mxfpWeightPrefetchEnabled) {
        if (tile + 1 < kTilesPerProjection) {
          ldu_prefetch_mxfp_weight_tma(
              tma_descs + descriptor_index, tile + 1, output_tile,
              cache_policy);
        }
      }
    }
  }

  if constexpr (!mxfpResidentFfnOverlapDownPrefetchEnabled) {
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      stage_empty[stage].wait(empty_phase[stage]);
    }
  }
#if defined(DAE_TRACK_MXFP_TIMELINE)
  // With full-FFN overlap enabled this marks final Linear-1 TMA issue rather
  // than final ring consumption. The disjoint Down ring may begin issuing at
  // once; the blockwise activation path still waits on producer readiness.
  g_events[sm_id * numProfileEvents + mxfpProfileWeightRingRelease] =
      cuda::ptx::get_sreg_globaltimer();
#endif
}
#endif

#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
template<bool ProduceActivation = true>
__device__ __noinline__ void ldu_execute_mxfp_resident_down(
    const MInst inst, const void *smem_base, const CUtensorMap *tma_descs,
    int *bars, uint64_t *tmem_mma_barriers) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = dae_mxfp_resident_ffn::kDownStages;
  constexpr int kTiles = 8;
  constexpr int kK128PerTile = 2;
  constexpr int kWeightPackedBytes = 16 * 1024;
  constexpr int kWeightScaleBytes = 1024;
  constexpr int kActivationRecordBytes = 1536;
  constexpr int kActivationDataBytes = 1024;
  constexpr int kActivationScaleBytes = 512;

  const auto *metadata = reinterpret_cast<const uint8_t *>(inst.address);
  const auto *weight_scale_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 0)));
  const auto *activation_records_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 8)));
  const uint64_t tma_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 24));
  const uint16_t weight_tma_index = uint16_t(tma_info);
  const int output_task = int(uint32_t(tma_info >> 32));
  const uint64_t barrier_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 32));
  const uint32_t ready_bar = uint32_t(barrier_info);
  const uint32_t k_start_tile = uint32_t(load_l2(
      reinterpret_cast<const int *>(metadata + 64)));
  const uint32_t resident_flags = uint32_t(load_l2(
      reinterpret_cast<const int *>(metadata + 68)));
  const int ready_bar_stride = (resident_flags & 2U) != 0 ? 8 : 1;
  const bool blockwise_ready = (resident_flags & 4U) != 0;

  auto *resident_base = reinterpret_cast<uint8_t *>(
      const_cast<void *>(smem_base));
  auto *weight_ring = resident_base +
      dae_mxfp_resident_ffn::kDownWeightRingOffset;
  auto *scale_ring = resident_base +
      dae_mxfp_resident_ffn::kDownScaleRingOffset;
  auto *activation_ring = resident_base +
      dae_mxfp_resident_ffn::kDownActivationRingOffset;
  auto *weight_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownLduWeightRingFullBarrierBase);
  auto *operand_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentOperandFullBarrierBase);
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownLduWeightRingEmptyBarrierBase);
  uint32_t empty_phase[kStages] = {};
  uint64_t cache_policy = 0;
  if constexpr (mxfpWeightPrefetchEnabled) {
    cache_policy = ldu_mxfp_streaming_cache_policy();
  }

  if constexpr (ProduceActivation) {
    if (!blockwise_ready && ready_bar != 0xFFFFFFFFU) {
      volatile int *ready = bars + ready_bar;
      bool pending = true;
      while (pending) {
        pending = false;
        #pragma unroll
        for (int record = 0; record < 16; ++record) {
          pending |= ready[record * ready_bar_stride] != 0;
        }
        if (pending) {
          __nanosleep(256);
        }
      }
      asm volatile("fence.acquire.gpu;" ::: "memory");
    }
  }

  #pragma unroll
  for (int tile = 0; tile < kTiles; ++tile) {
    const int stage = tile % kStages;
    stage_empty[stage].wait(empty_phase[stage]);
    empty_phase[stage] ^= 1U;

    const uint32_t weight_destination = static_cast<uint32_t>(
        __cvta_generic_to_shared(
            weight_ring + stage *
                dae_mxfp_resident_ffn::kDownWeightStageBytes));
    const uint32_t weight_barrier = static_cast<uint32_t>(
        __cvta_generic_to_shared(weight_full + stage));
    ldu_issue_mxfp_weight_tma<mxfpWeightPrefetchEnabled>(
        weight_destination, tma_descs + weight_tma_index,
        int(k_start_tile) + tile, output_task, weight_barrier, cache_policy);
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        scale_ring + stage *
            dae_mxfp_resident_ffn::kDownScaleStageBytes,
        weight_scale_global + tile * kWeightScaleBytes,
        uint32_t(kWeightScaleBytes),
        reinterpret_cast<uint64_t *>(weight_full + stage));
    weight_full[stage].arrive_and_expect_tx(
        kWeightPackedBytes + kWeightScaleBytes);
    if constexpr (mxfpWeightPrefetchEnabled) {
      if (tile + 1 < kTiles) {
        ldu_prefetch_mxfp_weight_tma(
            tma_descs + weight_tma_index,
            int(k_start_tile) + tile + 1, output_task, cache_policy);
      }
    }

    if constexpr (ProduceActivation) {
      if (blockwise_ready && ready_bar != 0xFFFFFFFFU) {
        volatile int *ready = bars + ready_bar;
        bool pending = true;
        while (pending) {
          pending = false;
          #pragma unroll
          for (int subtile = 0; subtile < kK128PerTile; ++subtile) {
            const int record =
                (int(k_start_tile) + tile) * kK128PerTile + subtile;
            pending |= ready[record * ready_bar_stride] != 0;
          }
          if (pending) {
            __nanosleep(256);
          }
        }
        asm volatile("fence.acquire.gpu;" ::: "memory");
      }

      #pragma unroll
      for (int subtile = 0; subtile < kK128PerTile; ++subtile) {
        const int record_index =
            (int(k_start_tile) + tile) * kK128PerTile + subtile;
        const auto *record = activation_records_global +
            record_index * kActivationRecordBytes;
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            activation_ring + stage *
                dae_mxfp_resident_ffn::kDownActivationStageBytes +
                subtile * kActivationDataBytes,
            record,
            uint32_t(kActivationDataBytes),
            reinterpret_cast<uint64_t *>(operand_full + stage));
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            scale_ring + stage *
                dae_mxfp_resident_ffn::kDownScaleStageBytes +
                kWeightScaleBytes + subtile * kActivationScaleBytes,
            record + kActivationDataBytes,
            uint32_t(kActivationScaleBytes),
            reinterpret_cast<uint64_t *>(operand_full + stage));
      }
      operand_full[stage].arrive_and_expect_tx(
          kK128PerTile * (kActivationDataBytes + kActivationScaleBytes));
    }
  }

  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    stage_empty[stage].wait(empty_phase[stage]);
  }
}

__device__ __noinline__ void ldu_execute_mxfp_resident_down_activation(
    const MInst inst, const void *smem_base, int *bars,
    uint64_t *tmem_mma_barriers) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = dae_mxfp_resident_ffn::kDownStages;
  constexpr int kTiles = 8;
  constexpr int kK128PerTile = 2;
  constexpr int kWeightScaleBytes = 1024;
  constexpr int kActivationRecordBytes = 1536;
  constexpr int kActivationDataBytes = 1024;
  constexpr int kActivationScaleBytes = 512;

  const auto *metadata = reinterpret_cast<const uint8_t *>(inst.address);
  const auto *activation_records_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 8)));
  const uint64_t barrier_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 32));
  const uint32_t ready_bar = uint32_t(barrier_info);
  const uint32_t k_start_tile = uint32_t(load_l2(
      reinterpret_cast<const int *>(metadata + 64)));
  const uint32_t resident_flags = uint32_t(load_l2(
      reinterpret_cast<const int *>(metadata + 68)));
  const int ready_bar_stride = (resident_flags & 2U) != 0 ? 8 : 1;
  const bool blockwise_ready = (resident_flags & 4U) != 0;

  auto *resident_base = reinterpret_cast<uint8_t *>(
      const_cast<void *>(smem_base));
  auto *scale_ring = resident_base +
      dae_mxfp_resident_ffn::kDownScaleRingOffset;
  auto *activation_ring = resident_base +
      dae_mxfp_resident_ffn::kDownActivationRingOffset;
  auto *operand_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentOperandFullBarrierBase);
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownLduWeightRingEmptyBarrierBase);
  uint32_t empty_phase[kStages] = {};

  if (!blockwise_ready && ready_bar != 0xFFFFFFFFU) {
    volatile int *ready = bars + ready_bar;
    bool pending = true;
    while (pending) {
      pending = false;
      #pragma unroll
      for (int record = 0; record < 16; ++record) {
        pending |= ready[record * ready_bar_stride] != 0;
      }
      if (pending) {
        __nanosleep(256);
      }
    }
    asm volatile("fence.acquire.gpu;" ::: "memory");
  }

  #pragma unroll
  for (int tile = 0; tile < kTiles; ++tile) {
    const int stage = tile % kStages;
    stage_empty[stage].wait(empty_phase[stage]);
    empty_phase[stage] ^= 1U;

    if (blockwise_ready && ready_bar != 0xFFFFFFFFU) {
      volatile int *ready = bars + ready_bar;
      bool pending = true;
      while (pending) {
        pending = false;
        #pragma unroll
        for (int subtile = 0; subtile < kK128PerTile; ++subtile) {
          const int record =
              (int(k_start_tile) + tile) * kK128PerTile + subtile;
          pending |= ready[record * ready_bar_stride] != 0;
        }
        if (pending) {
          __nanosleep(256);
        }
      }
      asm volatile("fence.acquire.gpu;" ::: "memory");
    }

    #pragma unroll
    for (int subtile = 0; subtile < kK128PerTile; ++subtile) {
      const int record_index =
          (int(k_start_tile) + tile) * kK128PerTile + subtile;
      const auto *record = activation_records_global +
          record_index * kActivationRecordBytes;
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          activation_ring + stage *
              dae_mxfp_resident_ffn::kDownActivationStageBytes +
              subtile * kActivationDataBytes,
          record,
          uint32_t(kActivationDataBytes),
          reinterpret_cast<uint64_t *>(operand_full + stage));
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          scale_ring + stage *
              dae_mxfp_resident_ffn::kDownScaleStageBytes +
              kWeightScaleBytes + subtile * kActivationScaleBytes,
          record + kActivationDataBytes,
          uint32_t(kActivationScaleBytes),
          reinterpret_cast<uint64_t *>(operand_full + stage));
    }
    operand_full[stage].arrive_and_expect_tx(
        kK128PerTile * (kActivationDataBytes + kActivationScaleBytes));
  }

  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    stage_empty[stage].wait(empty_phase[stage]);
  }
}

#endif

#if DAE_MXFP_GATE_UP_LDU_WEIGHT_RING && \
    !DAE_MXFP_GATE_UP_DIRECT_ACTIVATION && \
    DAE_MXFP_DOWN_LDU_WEIGHT_RING
__device__ __forceinline__ void ldu_execute_mxfp_resident_ffn_aux(
    const MInst inst, const void *smem_base, int *bars,
    uint64_t *tmem_mma_barriers) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
  const auto *metadata = reinterpret_cast<const uint8_t *>(
      load_l2_u64(plan + 1));
  const auto *second_metadata = reinterpret_cast<const uint8_t *>(
      load_l2_u64(plan + 2));
  const uint64_t tma_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 24));
  const uint64_t barrier_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 32));
  const uint32_t output_task = uint32_t(tma_info >> 32);
  const uint32_t reduce_bar = uint32_t(barrier_info >> 32);
  const uint32_t second_reduce_bar = uint32_t(load_l2_u64(
      reinterpret_cast<const uint64_t *>(second_metadata + 32)) >> 32);

  if (output_task < 32) {
    // Python has initialized both destinations. Publish their device-scope
    // reduction dependencies without doing any output work in the LDU.
    asm volatile("fence.release.gpu;" ::: "memory");
    *reinterpret_cast<volatile int *>(bars + reduce_bar) = 0;
    *reinterpret_cast<volatile int *>(bars + second_reduce_bar) = 0;
  }

  auto *reduction_ready = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentReductionReadyBarrierBase);
  auto *poll_start = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentLdu1PollStartBarrier);
  poll_start->wait(0);
  if constexpr (mxfpResidentDownSplitLduEnabled) {
    // LDU1 owns only activation/SFB. The normal allocator command that
    // dispatched this operator observes reduction readiness independently.
    auto *linear1_empty = reinterpret_cast<TxBarrier *>(
        tmem_mma_barriers + mxfpLduWeightRingEmptyBarrierBase);
    linear1_empty[1].wait(0);
    MInst task_inst {};
    #pragma unroll
    for (int task = 0; task < 2; ++task) {
      task_inst.address = load_l2_u64(plan + 1 + task);
      ldu_execute_mxfp_resident_down_activation(
          task_inst, smem_base, bars, tmem_mma_barriers);
    }
  } else {
    const uint32_t task_bars[2] = {reduce_bar, second_reduce_bar};
    #pragma unroll
    for (int task = 0; task < 2; ++task) {
      cuda::atomic_ref<int, cuda::thread_scope_device> ready(
          bars[task_bars[task]]);
      while (ready.load(cuda::memory_order_acquire) != 0) {
        __nanosleep(128);
      }
      reduction_ready[task].arrive();
    }
  }
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
    !DAE_MXFP_GATE_UP_DIRECT_ACTIVATION && \
    DAE_MXFP_DOWN_LDU_WEIGHT_RING
      case op(OP_TMA_LOAD_MX_RESIDENT_FFN): {
        produces_compute_operand = false;
        const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
        MInst task_inst = inst;
        task_inst.address = load_l2_u64(plan + 0);
        ldu_execute_mxfp_resident_linear1(
            task_inst, smem_base, tma_descs, tmem_mma_barriers
#if defined(DAE_TRACK_MXFP_TIMELINE)
            , sm_id, g_events
#endif
            );
        if constexpr (mxfpResidentDownLdu1ZeroEnabled) {
          auto *poll_start = reinterpret_cast<
              cutlass::arch::ClusterTransactionBarrier *>(
              tmem_mma_barriers + mxfpDownResidentLdu1PollStartBarrier);
          poll_start->arrive();
        }
        // The resident plan contains at most two Down tasks. Fetch their
        // immutable metadata while Linear-1's last stage is still retiring,
        // then consume the cached pointers after ownership transfers. This
        // is software interleaving only: command order and barriers are
        // unchanged.
        const int down_task_count = load_l2(
            reinterpret_cast<const int *>(plan + 3));
        const uint64_t down_task_address0 = down_task_count > 0
            ? load_l2_u64(plan + 1)
            : 0;
        const uint64_t down_task_address1 = down_task_count > 1
            ? load_l2_u64(plan + 2)
            : 0;
        if constexpr (mxfpResidentDownSplitLduEnabled) {
          auto *linear1_empty = reinterpret_cast<
              cutlass::arch::ClusterTransactionBarrier *>(
              tmem_mma_barriers + mxfpLduWeightRingEmptyBarrierBase);
          linear1_empty[0].wait(0);
        }
        for (int task = 0; task < down_task_count; ++task) {
          task_inst.address = task == 0
              ? down_task_address0
              : down_task_address1;
          ldu_execute_mxfp_resident_down<!mxfpResidentDownSplitLduEnabled>(
              task_inst, smem_base, tma_descs, bars, tmem_mma_barriers);
        }
        break; }
      case op(OP_TMA_LOAD_MX_RESIDENT_FFN_AUX): {
        produces_compute_operand = false;
        ldu_execute_mxfp_resident_ffn_aux(
            inst, smem_base, bars, tmem_mma_barriers);
        break; }
#endif
#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
      case op(OP_TMA_LOAD_MX_DOWN_RESIDENT): {
        produces_compute_operand = false;
        const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
        const int task_count = load_l2(
            reinterpret_cast<const int *>(plan + 2));
        for (int task = 0; task < task_count; ++task) {
          MInst task_inst = inst;
          task_inst.address = load_l2_u64(plan + task);
          ldu_execute_mxfp_resident_down(
              task_inst, smem_base, tma_descs, bars, tmem_mma_barriers);
        }
        break; }
#endif
#if DAE_MXFP_GATE_UP_LDU_WEIGHT_RING && \
    !DAE_MXFP_GATE_UP_DIRECT_ACTIVATION
      case op(OP_TMA_LOAD_MX_GATE_UP_RESIDENT): {
        using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
        constexpr int kStages =
            dae_mxfp_resident_ffn::kLinear1Stages;
        constexpr int kTilesPerProjection = 8;
        constexpr int kWeightPackedBytes = 32 * 1024;
        constexpr int kWeightStageBytes =
            dae_mxfp_resident_ffn::kLinear1WeightStageBytes;
        constexpr int kScalePackedBytes = 2048;
        constexpr int kScaleStageBytes =
            dae_mxfp_resident_ffn::kLinear1ScaleStageBytes;
        constexpr int kActivationBytes =
            dae_mxfp_resident_ffn::kLinear1ActivationBytes;
        constexpr int kActivationChunkBytes = 16 * 1024;

        produces_compute_operand = false;
        const auto *metadata = reinterpret_cast<const uint8_t *>(
            inst.address);
        const auto *activation_global = reinterpret_cast<const uint8_t *>(
            load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 0)));
        const auto *gate_scale_global = reinterpret_cast<const uint8_t *>(
            load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 16)));
        const auto *activation_scale_global =
            reinterpret_cast<const uint8_t *>(load_l2_u64(
                reinterpret_cast<const uint64_t *>(metadata + 24)));
        const auto *up_scale_global = reinterpret_cast<const uint8_t *>(
            load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 32)));
        const uint64_t tma_info = load_l2_u64(
            reinterpret_cast<const uint64_t *>(metadata + 40));
        const uint16_t descriptor_indices[2] = {
            uint16_t(tma_info), uint16_t(tma_info >> 16)};
        const int output_tile = int(uint32_t(tma_info >> 32));

        auto *resident_base = reinterpret_cast<uint8_t *>(
            const_cast<void *>(smem_base));
        auto *weight_ring = resident_base +
            dae_mxfp_resident_ffn::kLinear1WeightRingOffset;
        auto *scale_ring = resident_base +
            dae_mxfp_resident_ffn::kLinear1ScaleRingOffset;
        auto *activation = resident_base +
            dae_mxfp_resident_ffn::kLinear1ActivationOffset;
        auto *weight_full = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpLduWeightRingFullBarrierBase);
        auto *stage_empty = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpLduWeightRingEmptyBarrierBase);
        uint32_t empty_phase[kStages] = {};
        uint64_t cache_policy = 0;
        if constexpr (mxfpWeightPrefetchEnabled) {
          cache_policy = ldu_mxfp_streaming_cache_policy();
        }

        #pragma unroll
        for (int projection = 0; projection < 2; ++projection) {
          const auto *weight_scale_global = projection == 0
              ? gate_scale_global
              : up_scale_global;
          #pragma unroll
          for (int tile = 0; tile < kTilesPerProjection; ++tile) {
            const int operation = projection * kTilesPerProjection + tile;
            const int stage = operation % kStages;
            stage_empty[stage].wait(empty_phase[stage]);
            empty_phase[stage] ^= 1U;

            const uint32_t destination = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    weight_ring + stage * kWeightStageBytes));
            const uint32_t barrier = static_cast<uint32_t>(
                __cvta_generic_to_shared(weight_full + stage));
            ldu_issue_mxfp_weight_tma<mxfpWeightPrefetchEnabled>(
                destination, tma_descs + descriptor_indices[projection],
                tile, output_tile, barrier, cache_policy);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                scale_ring + stage * kScaleStageBytes,
                weight_scale_global + tile * kScalePackedBytes,
                uint32_t(kScalePackedBytes),
                reinterpret_cast<uint64_t *>(weight_full + stage));
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                scale_ring + stage * kScaleStageBytes + kScalePackedBytes,
                activation_scale_global + tile * kScalePackedBytes,
                uint32_t(kScalePackedBytes),
                reinterpret_cast<uint64_t *>(weight_full + stage));

            int transaction_bytes =
                kWeightPackedBytes + 2 * kScalePackedBytes;
            if (projection == 0 && tile == 0) {
              #pragma unroll
              for (int chunk = 0;
                   chunk < kActivationBytes / kActivationChunkBytes;
                   ++chunk) {
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_shared,
                    cuda::ptx::space_global,
                    activation + chunk * kActivationChunkBytes,
                    activation_global + chunk * kActivationChunkBytes,
                    uint32_t(kActivationChunkBytes),
                    reinterpret_cast<uint64_t *>(weight_full + stage));
              }
              transaction_bytes += kActivationBytes;
            }
            weight_full[stage].arrive_and_expect_tx(transaction_bytes);

            if constexpr (mxfpWeightPrefetchEnabled) {
              if (tile + 1 < kTilesPerProjection) {
                ldu_prefetch_mxfp_weight_tma(
                    tma_descs + descriptor_indices[projection], tile + 1,
                    output_tile, cache_policy);
              }
            }
          }
        }

        #pragma unroll
        for (int stage = 0; stage < kStages; ++stage) {
          stage_empty[stage].wait(empty_phase[stage]);
        }
#if defined(DAE_TRACK_MXFP_TIMELINE)
        g_events[sm_id * numProfileEvents + mxfpProfileWeightRingRelease] =
            cuda::ptx::get_sreg_globaltimer();
#endif
        break; }

      case op(OP_ALLOC_TMA_LOAD_MX_WEIGHT_RING_5D):
#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
      case op(OP_ALLOC_TMA_LOAD_MX_WEIGHT_RING_HANDOFF_5D):
#endif
      {
        using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
        constexpr int kStages = mxfpLduWeightRingStages;
        constexpr int kTilesPerProjection = 8;
        constexpr int kWeightPackedBytes = 32 * 1024;
        constexpr int kWeightStageBytes = 64 * 1024;
        constexpr int kScalePackedBytes = 2048;
        constexpr int kScaleStageBytes = 2 * kScalePackedBytes;
        constexpr int kLeaseSlots =
            16 + (mxfpWeightScaleTmaEnabled ? 1 : 0);

        // Allocation publication is independent of weight readiness. Compute
        // learns the ring base now, then each UMMA stage waits on weight_full.
        // The handler owns the lease until every final consumer has produced
        // the matching resident empty phase.
        static_cast<void>(m2c.barriers[bar].arrive());
        produces_compute_operand = false;
        auto *weight_full = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpLduWeightRingFullBarrierBase);
        auto *weight_scale_full = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpLduWeightScaleFullBarrierBase);
        auto *stage_empty = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpLduWeightRingEmptyBarrierBase);
        auto *weight_ring = static_cast<uint8_t *>(
            get_slot_address(smem_base, slot));
        auto *scale_ring = weight_ring + kStages * kWeightStageBytes;
        uint32_t empty_phase[kStages] = {};
        uint64_t cache_policy = 0;
        if constexpr (mxfpWeightPrefetchEnabled) {
          cache_policy = ldu_mxfp_streaming_cache_policy();
        }

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
          uint64_t scale_base = 0;
          if constexpr (mxfpWeightScaleTmaEnabled) {
            scale_base = *reinterpret_cast<const uint64_t *>(
                tma_descs + inst.coords[projection]);
          }
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
            ldu_issue_mxfp_weight_tma<mxfpWeightPrefetchEnabled>(
                destination, tma_descs + descriptor_index, tile,
                int(inst.coords[3]), barrier, cache_policy);
            if constexpr (mxfpGateUpWeightScaleSeparateBarrierEnabled) {
              weight_full[stage].arrive_and_expect_tx(kWeightPackedBytes);
              cuda::ptx::cp_async_bulk(
                  cuda::ptx::space_shared,
                  cuda::ptx::space_global,
                  scale_ring + stage * kScaleStageBytes,
                  reinterpret_cast<const uint8_t *>(scale_base) +
                      (int(inst.coords[3]) * kTilesPerProjection + tile) *
                          kScalePackedBytes,
                  uint32_t(kScalePackedBytes),
                  reinterpret_cast<uint64_t *>(weight_scale_full + stage));
              weight_scale_full[stage].arrive_and_expect_tx(
                  kScalePackedBytes);
            } else if constexpr (mxfpWeightScaleTmaEnabled) {
              cuda::ptx::cp_async_bulk(
                  cuda::ptx::space_shared,
                  cuda::ptx::space_global,
                  scale_ring + stage * kScaleStageBytes,
                  reinterpret_cast<const uint8_t *>(scale_base) +
                      (int(inst.coords[3]) * kTilesPerProjection + tile) *
                          kScalePackedBytes,
                  uint32_t(kScalePackedBytes),
                  reinterpret_cast<uint64_t *>(weight_full + stage));
              weight_full[stage].arrive_and_expect_tx(
                  kWeightPackedBytes + kScalePackedBytes);
            } else {
              weight_full[stage].arrive_and_expect_tx(kWeightPackedBytes);
            }
            if constexpr (mxfpWeightPrefetchEnabled) {
              if (tile + 1 < kTilesPerProjection) {
                ldu_prefetch_mxfp_weight_tma(
                    tma_descs + descriptor_index, tile + 1,
                    int(inst.coords[3]), cache_policy);
              }
            }
          }
        }

        const bool transfer_to_down =
            op(opcode) == op(OP_ALLOC_TMA_LOAD_MX_WEIGHT_RING_HANDOFF_5D);
#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
        if (transfer_to_down) {
          // The following logical ring command is already queued on this
          // same LDU. Transfer the lease locally and preserve the target's
          // descriptor/task chain without another allocator transaction.
          m2ld.wait();
          const LdCmd handoff {
              .raw = m2ld.data[m2ld.ptr]
          };
          m2ld.advance();
          const MInst down_inst = st_insts[handoff.slot];
#if defined(DAE_TRACK_PROFILE)
          ++commands;
#endif
          // A K256 down ring needs only the first 64 KiB of this lease. Wait
          // for both Linear-1 physical stages before starting its traffic:
          // issuing after stage 0 alone contends with the gate/up tail and is
          // measurably slower even though the address ranges are disjoint.
          #pragma unroll
          for (int stage = 0; stage < kStages; ++stage) {
            stage_empty[stage].wait(empty_phase[stage]);
          }
          auto *down_weight_full = reinterpret_cast<TxBarrier *>(
              tmem_mma_barriers + mxfpDownLduWeightRingFullBarrierBase);
          auto *down_weight_scale_full = reinterpret_cast<TxBarrier *>(
              tmem_mma_barriers + mxfpDownLduWeightScaleFullBarrierBase);
          auto *down_stage_empty = reinterpret_cast<TxBarrier *>(
              tmem_mma_barriers + mxfpDownLduWeightRingEmptyBarrierBase);
          uint32_t down_empty_phase[mxfpDownLduWeightRingStages] = {};
          ldu_execute_mxfp_down_weight_ring(
              m2ld, m2c, down_inst, slot, handoff.bar,
              smem_base, tma_descs, slot_avail,
              down_weight_full, down_weight_scale_full,
              down_stage_empty, down_empty_phase,
              0
#if defined(DAE_TRACK_PROFILE)
              , commands
#endif
              );
          // Down may consume half or all of the transferred storage. Allocator
          // ownership remains the original 16/17-slot lease until its last
          // resident phase retires.
          atomicOr(slot_avail, int(mkSlotMask(slot, kLeaseSlots)));
        } else
#endif
        {
          #pragma unroll
          for (int stage = 0; stage < kStages; ++stage) {
            stage_empty[stage].wait(empty_phase[stage]);
          }
          atomicOr(slot_avail, int(mkSlotMask(slot, kLeaseSlots)));
        }
#if defined(DAE_TRACK_MXFP_TIMELINE)
        g_events[sm_id * numProfileEvents + mxfpProfileWeightRingRelease] =
            cuda::ptx::get_sreg_globaltimer();
#endif
        break; }
#endif
#if DAE_MXFP_DOWN_LDU_WEIGHT_RING
      case op(OP_ALLOC_TMA_LOAD_MX_DOWN_WEIGHT_RING_5D): {
        using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
        constexpr int kDownLeaseSlots =
            4 * mxfpDownLduWeightRingStages +
            (mxfpWeightScaleTmaEnabled ? 1 : 0);
        produces_compute_operand = false;
        auto *weight_full = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpDownLduWeightRingFullBarrierBase);
        auto *weight_scale_full = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpDownLduWeightScaleFullBarrierBase);
        auto *stage_empty = reinterpret_cast<TxBarrier *>(
            tmem_mma_barriers + mxfpDownLduWeightRingEmptyBarrierBase);
        uint32_t empty_phase[mxfpDownLduWeightRingStages] = {};
        ldu_execute_mxfp_down_weight_ring(
            m2ld, m2c, inst, slot, bar,
            smem_base, tma_descs, slot_avail,
            weight_full, weight_scale_full, stage_empty, empty_phase,
            kDownLeaseSlots
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
