#pragma once

#include <cuda/atomic>

#include <cutlass/arch/barrier.h>

#include "internal_ring_stream.cuh"
#include "mxfp_resident_ffn.cuh"
#include "virtualcore.cuh"

#ifndef DAE_FP8_COUPLED_COMPLETION_PORT_MASK
#define DAE_FP8_COUPLED_COMPLETION_PORT_MASK 3
#endif

static_assert(
    DAE_FP8_COUPLED_COMPLETION_PORT_MASK >= 0 &&
    DAE_FP8_COUPLED_COMPLETION_PORT_MASK <= 3);

constexpr int kLduRouteCount = 6;

__device__ __forceinline__ int ldu_reload_fetch_add(
    int *address, int value, cuda::memory_order order) {
  if constexpr (reloadBarrierUseAtomicAdd) {
    // CUDA built-in atomics are device-scope relaxed operations. The reload
    // done phase supplies its release ordering with __threadfence() before
    // entering this helper; the ready phase has no preceding stores to
    // publish.
    return atomicAdd(address, value);
  } else {
    cuda::atomic_ref<int, cuda::thread_scope_device> ref(*address);
    return ref.fetch_add(value, order);
  }
}

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

__device__ __forceinline__ void ldu_issue_mxfp_streaming_bulk(
    void *destination, const void *source, const uint32_t bytes,
    void *barrier, const uint64_t cache_policy) {
  const uint32_t destination_smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(destination));
  const uint32_t barrier_smem = static_cast<uint32_t>(
      __cvta_generic_to_shared(barrier));
  asm volatile(
      "cp.async.bulk.shared::cluster.global."
      "mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;"
      :: "r"(destination_smem), "l"(source), "r"(bytes),
         "r"(barrier_smem), "l"(cache_policy)
      : "memory");
}

__device__ __forceinline__ void ldu_issue_mxfp_weight_tma(
    const uint32_t destination, const CUtensorMap *descriptor,
    const int tile, const int output_task, const uint32_t barrier,
    const uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global."
      "mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {0, %2, %3, %4, %5}], [%6], %7;"
      :: "r"(destination), "l"(descriptor),
         "r"(0), "r"(0), "r"(tile), "r"(output_task),
         "r"(barrier), "l"(cache_policy)
      : "memory");
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

__device__ __forceinline__ uint32_t ldu_tensor_transfer_bytes(
    const MInst &inst) {
  // MInst::size is uint16.  Descriptor-backed 64-KiB tensor tiles reserve
  // size=0 with an eight-slot lease; rank and descriptor id still come from
  // the ordinary tensor opcode/arg fields.
  return inst.size == 0 && inst.nslot() == 8
      ? 64U * 1024U
      : uint32_t(inst.size);
}

__device__ __forceinline__ void ldu_issue_internal_ring_tma(
    const int rank, const uint32_t destination,
    const CUtensorMap *descriptor, const int32_t *coordinates,
    const uint32_t barrier, const uint64_t cache_policy) {
  switch (rank) {
    case 1:
      asm volatile(
          "cp.async.bulk.tensor.1d.shared::cluster.global."
          "mbarrier::complete_tx::bytes.L2::cache_hint "
          "[%0], [%1, {%2}], [%3], %4;"
          :: "r"(destination), "l"(descriptor),
             "r"(coordinates[0]), "r"(barrier), "l"(cache_policy)
          : "memory");
      break;
    case 2:
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global."
          "mbarrier::complete_tx::bytes.L2::cache_hint "
          "[%0], [%1, {%2, %3}], [%4], %5;"
          :: "r"(destination), "l"(descriptor),
             "r"(coordinates[0]), "r"(coordinates[1]),
             "r"(barrier), "l"(cache_policy)
          : "memory");
      break;
    case 3:
      asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global."
          "mbarrier::complete_tx::bytes.L2::cache_hint "
          "[%0], [%1, {%2, %3, %4}], [%5], %6;"
          :: "r"(destination), "l"(descriptor),
             "r"(coordinates[0]), "r"(coordinates[1]),
             "r"(coordinates[2]), "r"(barrier), "l"(cache_policy)
          : "memory");
      break;
    case 4:
      asm volatile(
          "cp.async.bulk.tensor.4d.shared::cluster.global."
          "mbarrier::complete_tx::bytes.L2::cache_hint "
          "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
          :: "r"(destination), "l"(descriptor),
             "r"(coordinates[0]), "r"(coordinates[1]),
             "r"(coordinates[2]), "r"(coordinates[3]),
             "r"(barrier), "l"(cache_policy)
          : "memory");
      break;
  }
}

// General allocator-owned descriptor stream.  The command reaches both LDUs,
// but the plan's port mask may leave either lane idle.  Idle lanes still
// advance their logical empty-barrier phases so a later command can activate
// that LDU without rebuilding or resetting the persistent barrier bank.
__device__ __noinline__ void ldu_execute_internal_ring_stream(
    const MInst inst, const int slot, const int port_id,
    const void *smem_base, const CUtensorMap *tma_descs,
    uint64_t *tmem_mma_barriers, uint32_t &empty_phase_mask) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  using namespace dae_internal_ring;

  const auto *plan = reinterpret_cast<const TmaPlan *>(inst.address);
  const TmaLanePlan &lane = plan->lanes[port_id];
  const int stages =
      (inst.arg & dae_mxfp_resident_ffn::kCoupledStagesMask) >>
      dae_mxfp_resident_ffn::kCoupledStagesShift;
  const int port_mask =
      (inst.arg & dae_mxfp_resident_ffn::kCoupledPortMask) >>
      dae_mxfp_resident_ffn::kCoupledPortMaskShift;
  const bool active = (port_mask & (1 << port_id)) != 0;
  const uint32_t stage_bytes = load_l2(
      reinterpret_cast<const int *>(&plan->stage_bytes));
  const uint32_t flags = load_l2(
      reinterpret_cast<const int *>(&plan->flags));
  auto *ring = static_cast<uint8_t *>(get_slot_address(smem_base, slot));
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + internalRingEmptyBarrierBase);
  auto *stage_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + internalRingFullBarrierBase +
      port_id * internalRingStages);
  const uint64_t cache_policy = ldu_mxfp_streaming_cache_policy();
  __ldprint(
      "internal ring port=%d active=%d stages=%d iterations=%d rank=%d "
      "issues=%d tx=%u dst=%u stride=%u",
      port_id, int(active), stages, int(inst.size), int(lane.rank),
      int(lane.issue_count), lane.transaction_bytes,
      lane.destination_offset, lane.destination_issue_stride);

  for (int iteration = 0; iteration < int(inst.size); ++iteration) {
    const int stage = iteration % stages;
    const uint32_t phase = (empty_phase_mask >> stage) & 1U;
    if (active) {
      stage_empty[stage].wait(phase);
      __ldprint(
          "internal ring port=%d stage=%d empty phase=%u ready",
          port_id, stage, phase);
      const uint32_t barrier = static_cast<uint32_t>(
          __cvta_generic_to_shared(stage_full + stage));
      const uint32_t stage_destination = static_cast<uint32_t>(
          __cvta_generic_to_shared(ring + stage * stage_bytes));
      const uint16_t descriptor_index = lane.descriptor_index;
      const int rank = lane.rank;
      const int issue_count = lane.issue_count;
      const uint32_t destination_offset = lane.destination_offset;
      const uint32_t destination_issue_stride =
          lane.destination_issue_stride;
      const uint32_t expected_bytes =
          lane.transaction_bytes * issue_count;
      int32_t coordinates[kMaxRank];
      #pragma unroll
      for (int coordinate = 0; coordinate < kMaxRank; ++coordinate) {
        coordinates[coordinate] = lane.coordinates[coordinate] +
            iteration * lane.iteration_delta[coordinate];
      }

      // Arm the phase before any TMA can complete against it.  This is
      // required when one logical port contributes multiple copies to the
      // same stage: an early copy must not decrement an unarmed phase.
      stage_full[stage].arrive_and_expect_tx(expected_bytes);
      for (int issue = 0; issue < issue_count; ++issue) {
        int32_t issue_coordinates[kMaxRank];
        #pragma unroll
        for (int coordinate = 0; coordinate < kMaxRank; ++coordinate) {
          issue_coordinates[coordinate] = coordinates[coordinate] +
              issue * lane.issue_delta[coordinate];
        }
        ldu_issue_internal_ring_tma(
            rank,
            stage_destination + destination_offset +
                issue * destination_issue_stride,
            tma_descs + descriptor_index, issue_coordinates,
            barrier, cache_policy);
        __ldprint(
            "internal ring port=%d issue=%d coords=(%d,%d,%d,%d)",
            port_id, issue, issue_coordinates[0], issue_coordinates[1],
            issue_coordinates[2], issue_coordinates[3]);
      }
      __ldprint(
          "internal ring port=%d stage=%d expected=%u",
          port_id, stage, expected_bytes);
    }
    empty_phase_mask ^= 1U << stage;
  }
  static_cast<void>(flags);
}

// Common allocator-owned MXFP8 x MXFP8 stream. Each stage carries both M128
// output groups for one K256 pair, while LDU1 supplies their activation.
__device__ __forceinline__ void ldu_execute_mxfp8_coupled_stream(
    const MInst inst, const int slot, const int port_id,
    const void *smem_base, uint64_t *tmem_mma_barriers,
    uint32_t &pair_base_state
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
    , const int sm_id, uint64_t *g_events, const bool detail_capture
#endif
    ) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages =
      dae_mxfp_resident_ffn::kFp8CoupledStages;
  constexpr int kWeightBytes =
      dae_mxfp_resident_ffn::kFp8CoupledWeightDataBytes +
      dae_mxfp_resident_ffn::kFp8CoupledWeightScaleBytes;
  constexpr int kActivationBytes =
      dae_mxfp_resident_ffn::kFp8CoupledActivationDataBytes +
      dae_mxfp_resident_ffn::kFp8CoupledActivationScaleBytes;
  constexpr int kActivationTileBytes = 2 * 1024;
  constexpr int kActivationDataBytes =
      dae_mxfp_resident_ffn::kFp8CoupledActivationDataBytes / 2;
  constexpr int kBulkBytes = 16 * 1024;
  static_assert(
      dae_mxfp_resident_ffn::kFp8CoupledWeightDataBytes % kBulkBytes == 0);
  const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
  const uint64_t detail_source_begin = detail_capture
      ? cuda::ptx::get_sreg_globaltimer()
      : 0;
#endif
  const auto *source = reinterpret_cast<const uint8_t *>(
      load_l2_u64(plan + port_id));
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
  if (detail_capture) {
    const uint64_t detail_source_end = cuda::ptx::get_sreg_globaltimer();
    const uint64_t detail_source_duration = min(
        detail_source_end - detail_source_begin, 0xfffffULL);
    g_events[sm_id * numProfileEvents +
             fp8CoupledDetailSourceEventBase + port_id] =
        uint64_t(uint32_t(detail_source_begin)) |
        (detail_source_duration << 32);
  }
#endif
  auto *ring = static_cast<uint8_t *>(
      get_slot_address(smem_base, slot));
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfp8CoupledEmptyBarrierBase);
  auto *stage_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers +
      (port_id == 0 ? mxfp8CoupledWeightFullBarrierBase
                    : mxfp8CoupledActivationFullBarrierBase));
  const int phase_base = int(pair_base_state);
  pair_base_state = uint32_t(
      (phase_base + inst.size) % (2 * kStages));
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
  if (detail_capture) {
    g_events[sm_id * numProfileEvents +
             fp8CoupledDetailWaitEventBase + port_id * 3] =
        uint64_t(uint32_t(phase_base)) |
        (uint64_t(uint32_t(inst.size)) << 32);
  }
  uint32_t detail_used_stage_mask = 0;
  uint32_t detail_last_phase[kStages] = {};
#endif
  for (int pair = 0; pair < inst.size; ++pair) {
    const int global_pair = phase_base + pair;
    const int stage = global_pair % kStages;
    const int phase = (global_pair / kStages) & 1;
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
    detail_used_stage_mask |= 1U << stage;
    detail_last_phase[stage] = uint32_t(phase);
    uint64_t detail_empty_wait_begin = 0;
    bool detail_expected_ready = false;
    bool detail_opposite_ready = false;
    if (detail_capture && pair < 2) {
      detail_empty_wait_begin = cuda::ptx::get_sreg_globaltimer();
      detail_expected_ready = cuda::ptx::mbarrier_try_wait_parity(
          cuda::ptx::sem_acquire,
          cuda::ptx::scope_cta,
          reinterpret_cast<uint64_t *>(stage_empty + stage),
          phase);
      detail_opposite_ready = cuda::ptx::mbarrier_try_wait_parity(
          cuda::ptx::sem_acquire,
          cuda::ptx::scope_cta,
          reinterpret_cast<uint64_t *>(stage_empty + stage),
          phase ^ 1);
    }
#endif
    stage_empty[stage].wait(phase);
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
    if (detail_capture && pair < 2) {
      const uint64_t detail_empty_wait_end =
          cuda::ptx::get_sreg_globaltimer();
      const uint64_t duration = min(
          detail_empty_wait_end - detail_empty_wait_begin, 0xfffffULL);
      g_events[sm_id * numProfileEvents + fp8CoupledDetailWaitEventBase +
               port_id * 3 + 1 + pair] =
          uint64_t(uint32_t(detail_empty_wait_begin)) |
          (duration << 32) |
          (uint64_t(detail_expected_ready) << 52) |
          (uint64_t(detail_opposite_ready) << 53) |
          (uint64_t(stage) << 54) |
          (uint64_t(phase) << 55);
    }
#endif
    auto *destination = ring +
        stage * dae_mxfp_resident_ffn::kFp8CoupledStageBytes;

    if (port_id == 0) {
      const auto *record = source + uint64_t(pair) * kWeightBytes;
      const uint64_t cache_policy = ldu_mxfp_streaming_cache_policy();
      #pragma unroll
      for (int chunk = 0;
           chunk < dae_mxfp_resident_ffn::kFp8CoupledWeightDataBytes /
                       kBulkBytes;
           ++chunk) {
        const int offset = chunk * kBulkBytes;
        ldu_issue_mxfp_streaming_bulk(
            destination + offset,
            record + offset,
            uint32_t(kBulkBytes),
          stage_full + stage,
          cache_policy);
      }
      ldu_issue_mxfp_streaming_bulk(
          destination +
              dae_mxfp_resident_ffn::kFp8CoupledWeightScaleOffset,
          record + dae_mxfp_resident_ffn::kFp8CoupledWeightDataBytes,
          uint32_t(dae_mxfp_resident_ffn::kFp8CoupledWeightScaleBytes),
          stage_full + stage,
          cache_policy);
      stage_full[stage].arrive_and_expect_tx(kWeightBytes);
    } else {
      const auto *record =
          source + uint64_t(pair) * 2 * kActivationTileBytes;
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          destination +
              dae_mxfp_resident_ffn::kFp8CoupledActivationDataOffset,
          record,
          uint32_t(kActivationDataBytes),
          reinterpret_cast<uint64_t *>(stage_full + stage));
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          destination +
              dae_mxfp_resident_ffn::kFp8CoupledActivationDataOffset +
              kActivationDataBytes,
          record + kActivationTileBytes,
          uint32_t(kActivationDataBytes),
          reinterpret_cast<uint64_t *>(stage_full + stage));
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared,
          cuda::ptx::space_global,
          destination +
              dae_mxfp_resident_ffn::kFp8CoupledActivationScaleOffset,
          record + kActivationDataBytes,
          uint32_t(dae_mxfp_resident_ffn::kFp8CoupledActivationScaleBytes),
          reinterpret_cast<uint64_t *>(stage_full + stage));
      stage_full[stage].arrive_and_expect_tx(kActivationBytes);
    }
  }
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
  // Diagnostic observer only: all earlier generations were necessarily
  // consumed before their stage could be reused, so waiting on the final
  // generation of each used stage gives the command's true async-copy
  // completion frontier.  This adds no arrival and changes no phase state.
  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    if ((DAE_FP8_COUPLED_COMPLETION_PORT_MASK & (1 << port_id)) &&
        (detail_used_stage_mask & (1U << stage))) {
      stage_full[stage].wait(detail_last_phase[stage]);
    }
  }
#endif
}

template <bool DynamicExpert>
__device__ __noinline__ void ldu_execute_mxfp_coupled_linear1(
    const MInst inst, const void *smem_base, const CUtensorMap *tma_descs,
    uint64_t *tmem_mma_barriers
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    , const int sm_id, uint64_t *g_events
#endif
    ) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = dae_mxfp_resident_ffn::kLinear1Stages;
  constexpr int kOperations = 16;
  constexpr int kWeightPackedBytes = 32 * 1024;
  constexpr int kWeightStageBytes =
      dae_mxfp_resident_ffn::kLinear1WeightStageBytes;
  constexpr int kWeightScaleBytes = 2 * 1024;
  constexpr int kActivationScaleBytes = 2 * 1024;
  constexpr int kScaleStageBytes =
      dae_mxfp_resident_ffn::kLinear1ScaleStageBytes;
  constexpr int kActivationBytes =
      dae_mxfp_resident_ffn::kLinear1ActivationBytes;
  constexpr int kActivationChunkBytes = 16 * 1024;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  const uint64_t detail_prologue_begin =
      cuda::ptx::get_sreg_globaltimer();
  uint64_t detail_stage_empty_wait_ns = 0;
#endif

  const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
  const auto *activation_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(plan + 6));
  const auto *scale_stream_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(plan + 4));
  const auto *activation_scale_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(plan + 7));
  const uint64_t tma_info = load_l2_u64(plan + 5);
  const uint16_t descriptor_index = uint16_t(tma_info);
  int output_tile = int(uint32_t(tma_info >> 32));
  if constexpr (DynamicExpert) {
    const auto *route_record = reinterpret_cast<const uint8_t *>(
        load_l2_u64(plan + 8));
    const uint64_t selector = load_l2_u64(plan + 9);
    const int route_rank = int(uint32_t(selector));
    const int local_slice = int(uint32_t(selector >> 32));
    const uint32_t task_base = uint32_t(load_l2(
        reinterpret_cast<const int *>(route_record + 64) + route_rank));
    output_tile = int(task_base) + local_slice;
    scale_stream_global +=
        uint64_t(output_tile) * kOperations * kWeightScaleBytes;
  }

  auto *resident_base = reinterpret_cast<uint8_t *>(
      const_cast<void *>(smem_base));
  auto *weight_ring = resident_base +
      dae_mxfp_resident_ffn::kLinear1WeightRingOffset;
  auto *scale_ring = resident_base +
      dae_mxfp_resident_ffn::kLinear1ScaleRingOffset;
  auto *activation = resident_base +
      dae_mxfp_resident_ffn::kLinear1ActivationOffset;
  auto *weight_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpResidentLinear1FullBarrierBase);
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpResidentLinear1EmptyBarrierBase);
  uint32_t empty_phase[kStages] = {};
  const uint64_t cache_policy = ldu_mxfp_streaming_cache_policy();
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  g_events[sm_id * numProfileEvents +
           mxfpFfnDetailLdu0Linear1PrologueNs] =
      cuda::ptx::get_sreg_globaltimer() - detail_prologue_begin;
#endif

  #pragma unroll
  for (int operation = 0; operation < kOperations; ++operation) {
    const int stage = operation % kStages;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    const uint64_t detail_empty_wait_begin =
        cuda::ptx::get_sreg_globaltimer();
#endif
    stage_empty[stage].wait(empty_phase[stage]);
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    detail_stage_empty_wait_ns +=
        cuda::ptx::get_sreg_globaltimer() - detail_empty_wait_begin;
#endif
    empty_phase[stage] ^= 1U;

    const uint32_t destination = static_cast<uint32_t>(
        __cvta_generic_to_shared(
            weight_ring + stage * kWeightStageBytes));
    const uint32_t barrier = static_cast<uint32_t>(
        __cvta_generic_to_shared(weight_full + stage));
    ldu_issue_mxfp_weight_tma(
        destination, tma_descs + descriptor_index, operation, output_tile,
        barrier, cache_policy);
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        scale_ring + stage * kScaleStageBytes,
        scale_stream_global + operation * kWeightScaleBytes,
        uint32_t(kWeightScaleBytes),
        reinterpret_cast<uint64_t *>(weight_full + stage));
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        scale_ring + stage * kScaleStageBytes + kWeightScaleBytes,
        activation_scale_global + (operation & 7) * kActivationScaleBytes,
        uint32_t(kActivationScaleBytes),
        reinterpret_cast<uint64_t *>(weight_full + stage));

    int transaction_bytes =
        kWeightPackedBytes + kWeightScaleBytes + kActivationScaleBytes;
    if (operation == 0) {
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

    if (operation + 1 < kOperations) {
      ldu_prefetch_mxfp_weight_tma(
          tma_descs + descriptor_index, operation + 1, output_tile,
          cache_policy);
    }
  }
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  g_events[sm_id * numProfileEvents +
           mxfpFfnDetailLdu0Linear1EmptyWaitNs] =
      detail_stage_empty_wait_ns;
#endif
}

template <bool DynamicExpert>
__device__ __noinline__ void ldu_execute_mxfp_down_weight_stream(
    const MInst inst, const void *smem_base, const CUtensorMap *tma_descs,
    uint64_t *tmem_mma_barriers, const uint64_t *plan
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    , uint64_t &detail_stage_empty_wait_ns
#endif
    ) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = dae_mxfp_resident_ffn::kDownStages;
  constexpr int kFullTiles = 8;
  constexpr int kWeightPackedBytes = 16 * 1024;
  constexpr int kWeightScaleBytes = 1024;

  const auto *metadata = reinterpret_cast<const uint8_t *>(inst.address);
  const auto *weight_scale_global = reinterpret_cast<const uint8_t *>(
      load_l2_u64(reinterpret_cast<const uint64_t *>(metadata + 0)));
  const uint64_t tma_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 24));
  uint16_t weight_tma_index = uint16_t(tma_info);
  int output_task = int(uint32_t(tma_info >> 32));
  const uint32_t k_start_tile = uint32_t(load_l2(
      reinterpret_cast<const int *>(metadata + 64)));
  const uint32_t resident_flags = uint32_t(load_l2(
      reinterpret_cast<const int *>(metadata + 68)));
  const int tiles =
      (resident_flags & dae_mxfp_resident_ffn::kDownSplitK2)
          ? kFullTiles / 2
          : kFullTiles;
  if constexpr (DynamicExpert) {
    const auto *route_record = reinterpret_cast<const uint8_t *>(
        load_l2_u64(plan + 8));
    const int plan_route_rank = load_l2(
        reinterpret_cast<const int *>(plan + 9));
    const int task_route_rank = load_l2(
        reinterpret_cast<const int *>(metadata + 72));
    const int route_rank =
        (resident_flags & dae_mxfp_resident_ffn::kDownPerTaskRoute)
            ? task_route_rank
            : plan_route_rank;
    const int local_output_tile = output_task % 32;
    if (route_rank >= 0) {
      const uint32_t task_base = uint32_t(load_l2(
          reinterpret_cast<const int *>(route_record + 96) + route_rank));
      output_task = int(task_base) + local_output_tile;
      weight_scale_global = reinterpret_cast<const uint8_t *>(
          load_l2_u64(plan + 10)) +
          (uint64_t(output_task) * kFullTiles + k_start_tile) *
              kWeightScaleBytes;
      weight_tma_index = uint16_t(load_l2_u64(plan + 11));
    }
  }

  auto *resident_base = reinterpret_cast<uint8_t *>(
      const_cast<void *>(smem_base));
  auto *weight_ring = resident_base +
      dae_mxfp_resident_ffn::kDownWeightRingOffset;
  auto *scale_ring = resident_base +
      dae_mxfp_resident_ffn::kDownScaleRingOffset;
  auto *weight_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpResidentDownWeightFullBarrierBase);
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpResidentDownEmptyBarrierBase);
  uint32_t empty_phase[kStages] = {};
  const uint64_t cache_policy = ldu_mxfp_streaming_cache_policy();

  #pragma unroll
  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % kStages;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    const uint64_t detail_empty_wait_begin =
        cuda::ptx::get_sreg_globaltimer();
#endif
    stage_empty[stage].wait(empty_phase[stage]);
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    detail_stage_empty_wait_ns +=
        cuda::ptx::get_sreg_globaltimer() - detail_empty_wait_begin;
#endif
    empty_phase[stage] ^= 1U;

    const uint32_t weight_destination = static_cast<uint32_t>(
        __cvta_generic_to_shared(
            weight_ring + stage *
                dae_mxfp_resident_ffn::kDownWeightStageBytes));
    const uint32_t weight_barrier = static_cast<uint32_t>(
        __cvta_generic_to_shared(weight_full + stage));
    ldu_issue_mxfp_weight_tma(
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
    if (tile + 1 < tiles) {
      ldu_prefetch_mxfp_weight_tma(
          tma_descs + weight_tma_index,
          int(k_start_tile) + tile + 1, output_task, cache_policy);
    }
  }

  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    stage_empty[stage].wait(empty_phase[stage]);
  }
}

__device__ __noinline__ void ldu_execute_mxfp_resident_down_activation(
    const MInst inst, const void *smem_base, int *bars,
    uint64_t *tmem_mma_barriers
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    , uint64_t &detail_stage_empty_wait_ns
#endif
    ) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  constexpr int kStages = dae_mxfp_resident_ffn::kDownStages;
  constexpr int kFullTiles = 8;
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
  const int tiles =
      (resident_flags & dae_mxfp_resident_ffn::kDownSplitK2)
          ? kFullTiles / 2
          : kFullTiles;
  const int ready_bar_stride =
      (resident_flags & dae_mxfp_resident_ffn::kDownReadyStride8) != 0
          ? 8
          : 1;
  const bool blockwise_ready =
      (resident_flags & dae_mxfp_resident_ffn::kDownBlockwiseReady) != 0;

  auto *resident_base = reinterpret_cast<uint8_t *>(
      const_cast<void *>(smem_base));
  auto *scale_ring = resident_base +
      dae_mxfp_resident_ffn::kDownScaleRingOffset;
  auto *activation_ring = resident_base +
      dae_mxfp_resident_ffn::kDownActivationRingOffset;
  auto *operand_full = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpDownResidentOperandFullBarrierBase);
  auto *stage_empty = reinterpret_cast<TxBarrier *>(
      tmem_mma_barriers + mxfpResidentDownEmptyBarrierBase);
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
  for (int tile = 0; tile < tiles; ++tile) {
    const int stage = tile % kStages;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    const uint64_t detail_empty_wait_begin =
        cuda::ptx::get_sreg_globaltimer();
#endif
    stage_empty[stage].wait(empty_phase[stage]);
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    detail_stage_empty_wait_ns +=
        cuda::ptx::get_sreg_globaltimer() - detail_empty_wait_begin;
#endif
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

__device__ __forceinline__ void ldu_execute_mxfp_down_activation_stream(
    const MInst inst, const void *smem_base, int *bars,
    uint64_t *tmem_mma_barriers, const uint32_t resident_phase
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    , const int sm_id, uint64_t *g_events
#endif
    ) {
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;
  const auto *plan = reinterpret_cast<const uint64_t *>(inst.address);
  const int down_task_count = load_l2(
      reinterpret_cast<const int *>(plan + 3));
  const auto *metadata = reinterpret_cast<const uint8_t *>(
      load_l2_u64(plan + 1));
  const uint64_t tma_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 24));
  const uint64_t barrier_info = load_l2_u64(
      reinterpret_cast<const uint64_t *>(metadata + 32));
  const uint32_t output_task = uint32_t(tma_info >> 32);
  const uint32_t reduce_bar = uint32_t(barrier_info >> 32);
  const uint32_t resident_flags = uint32_t(load_l2(
      reinterpret_cast<const int *>(metadata + 68)));

  if (output_task < 32) {
    // Externally initialized destinations can publish immediately.  In the
    // shared-first FP32 path, expert zero instead releases each counter after
    // its initial TMA copy has completed.
    if ((resident_flags & 1U) != 0) {
      asm volatile("fence.release.gpu;" ::: "memory");
      *reinterpret_cast<volatile int *>(bars + reduce_bar) = 0;
    }
  }
  if (down_task_count > 1) {
    const auto *second_metadata = reinterpret_cast<const uint8_t *>(
        load_l2_u64(plan + 2));
    const uint64_t second_tma_info = load_l2_u64(
        reinterpret_cast<const uint64_t *>(second_metadata + 24));
    const uint32_t second_output_task = uint32_t(second_tma_info >> 32);
    const uint32_t second_resident_flags = uint32_t(load_l2(
        reinterpret_cast<const int *>(second_metadata + 68)));
    if (second_output_task < 32 && (second_resident_flags & 1U) != 0) {
      const uint32_t second_reduce_bar = uint32_t(load_l2_u64(
          reinterpret_cast<const uint64_t *>(second_metadata + 32)) >> 32);
      asm volatile("fence.release.gpu;" ::: "memory");
      *reinterpret_cast<volatile int *>(bars + second_reduce_bar) = 0;
    }
  }

  if (!(inst.arg & dae_mxfp_resident_ffn::kCoupledDownOnly)) {
    auto *poll_start = reinterpret_cast<TxBarrier *>(
        tmem_mma_barriers + mxfpDownResidentLdu1PollStartBarrier);
    poll_start->wait(resident_phase);
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    g_events[sm_id * numProfileEvents + mxfpFfnDetailLdu1PollReady] =
        cuda::ptx::get_sreg_globaltimer();
#endif
    // LDU1 owns activation/SFB only. The allocator observes the independent
    // reduction dependencies while this stream waits for Linear-1 data.
    auto *linear1_empty = reinterpret_cast<TxBarrier *>(
        tmem_mma_barriers + mxfpResidentLinear1EmptyBarrierBase);
    linear1_empty[1].wait(0);
  }
  MInst task_inst {};
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  uint64_t detail_stage_empty_wait_ns = 0;
#endif
  #pragma unroll 1
  for (int task = 0; task < down_task_count; ++task) {
    task_inst.address = load_l2_u64(plan + 1 + task);
    ldu_execute_mxfp_resident_down_activation(
        task_inst, smem_base, bars, tmem_mma_barriers
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
        , detail_stage_empty_wait_ns
#endif
        );
  }
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  g_events[sm_id * numProfileEvents +
           mxfpFfnDetailLdu1DownEmptyWaitNs] =
      detail_stage_empty_wait_ns;
#endif
}

template<typename M2LD_Type, typename M2C_Type>
__device__ __forceinline__ void ldwarp_execute_singlethread(
    M2LD_Type &m2ld, M2C_Type &m2c,
    MInst *st_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars,
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
  uint32_t mxfp8_coupled_pair_base = 0;
  uint32_t internal_ring_empty_phase_mask = 0;
  uint32_t mxfp_resident_down_phase = 0;
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
#if !defined(DAE_MXFP_FFN_DETAIL_PROFILE) && \
    !defined(DAE_STU_HISTORY_PROFILE)
  uint32_t profile_reload_counter = 0;
#endif
#endif
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
  uint32_t fp8_coupled_detail_index = 0;
  uint32_t fp8_coupled_detail_loop_tail_reloads = 0;
  bool fp8_coupled_detail_complete = false;
#endif
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  uint64_t detail_previous_begin = 0;
  uint64_t detail_previous_end = 0;
  uint16_t detail_previous_opcode = 0;
#endif
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
  bool attention_detail_ring_seen = false;
  bool attention_detail_q_complete = false;
#endif
  m2ld.wait();
  LdCmd cmd { .raw = m2ld.data[m2ld.ptr] };

  while (cmd.slot != SLOT_END) {
#if defined(DAE_TRACK_PROFILE)
    ++commands;
#endif
    auto &slot = cmd.slot;
    auto &opcode = cmd.opcode;
#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
    MInst inst{};
    // Compact direct-scale commands carry their operand/stage in LdCmd::slot
    // and derive the source from LDU-local state. Their repeatedly overwritten
    // special MInst mailboxes are intentionally never read.
    if (op(opcode) != op(OP_ALLOC_TMA_LOAD_MX_SCALE_1D))
      inst = st_insts[slot];
#else
    auto inst = st_insts[slot];
#endif
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
    const bool attention_detail_ring_command =
        port_id == 1 &&
        op(opcode) == op(OP_TMA_LOAD_MX_COUPLED_STREAM) &&
        (inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask) ==
            dae_mxfp_resident_ffn::kCoupledTmaRing;
    const bool attention_detail_q_command =
        port_id == 1 && attention_detail_ring_seen &&
        !attention_detail_q_complete &&
        op(opcode) == op(OP_ALLOC_TMA_LOAD_3D) &&
        inst.size == 0 && inst.nslot() == 8;
    if (attention_detail_ring_command) {
      g_events[sm_id * numProfileEvents + detailProfileEventBase + 40] =
          cuda::ptx::get_sreg_globaltimer();
    }
    if (attention_detail_q_command) {
      g_events[sm_id * numProfileEvents + detailProfileEventBase + 43] =
          cuda::ptx::get_sreg_globaltimer();
    }
#endif

    m2ld.advance();
    const bool async_control_command =
        op(opcode) == op(OP_LDU_ASYNC_RELOAD_BARRIERS) ||
        op(opcode) == op(OP_LDU_WAIT_BARRIER);
    if (async_control_command) {
      // Both LDUs have copied the immutable descriptor into registers, so the
      // allocator may advance while the handlers wait or clear independently.
      ldu_control_publish_barrier->arrive_and_wait();
    }
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE) || \
    defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
    const uint64_t detail_command_begin =
        cuda::ptx::get_sreg_globaltimer();
#endif

    auto &bar = cmd.bar;
    bool produces_compute_operand = true;
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
    bool detail_executed_fp8_coupled = false;
#endif
    __ldprint("Receive LD cmd: slot=%d bar=%d opcode=%d", slot, bar, op(opcode));

    // If its a readbar, we do the readbar
    // TODO(zhiyuang): wait bar here if bar is set
    // The common FP8 stream carries immutable weights on LDU0 and the
    // producer-dependent activation on LDU1.  Let LDU0 fill the first ring
    // stages while the activation dependency is still outstanding; only the
    // activation port needs to observe the stage input barrier.
    const bool prefetch_coupled_fp8_weight =
        port_id == 0 &&
        op(opcode) == op(OP_TMA_LOAD_MX_COUPLED_STREAM) &&
        (inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask) ==
            dae_mxfp_resident_ffn::kCoupledFp8Gemv;
    if ((opcode & MEM_OP_FLAGS_BARRIER) &&
        !(opcode & MEM_OP_FLAGS_WRITEBACK) &&
        op(opcode) != op(OP_LDU_WAIT_BARRIER) &&
        !prefetch_coupled_fp8_weight) {
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
      // A barriered raw-address command is itself the producer dependency:
      // compute will dereference HBM directly instead of issuing a TMA after
      // this wait.  Carry the device-scope release/acquire edge through the
      // LDU-to-compute mailbox before publishing that pointer.
      if (op(opcode) == op(OP_ALLOC_WB_RAW_ADDRESS)) {
        asm volatile("fence.acquire.gpu;" ::: "memory");
      }
#if defined(DAE_TRACK_PROFILE)
      dependency_wait_ns +=
          cuda::ptx::get_sreg_globaltimer() - dependency_start;
#endif
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
      if (port_id == 0 &&
          op(opcode) == op(OP_TMA_LOAD_MX_COUPLED_STREAM) &&
          (inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask) ==
              dae_mxfp_resident_ffn::kCoupledLinear1) {
        g_events[sm_id * numProfileEvents +
                 mxfpFfnDetailLdu0Linear1AfterDependency] =
            cuda::ptx::get_sreg_globaltimer();
      }
#endif
      __ldprint("wait for global barrier before load: bar=%d", dependency_bar);
    };
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
    if (attention_detail_ring_command) {
      g_events[sm_id * numProfileEvents + detailProfileEventBase + 41] =
          cuda::ptx::get_sreg_globaltimer();
    }
    if (attention_detail_q_command) {
      g_events[sm_id * numProfileEvents + detailProfileEventBase + 44] =
          cuda::ptx::get_sreg_globaltimer();
    }
#endif

    if (op(opcode) == op(OP_ALLOC_RW_TMA_2D) &&
        inst.coords[2] != 0xFFFFU) {
      volatile int *input_bar = bars + inst.coords[2];
      while (*input_bar != 0) {
        __nanosleep(barrierPollSleepCycles);
      }
    }

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
      case op(OP_ALLOC_TMA_LOAD_PAIR_1D): {
        const uint32_t transfer_size = uint32_t(inst.size);
        const uint32_t first_size = uint32_t(inst.arg);
        const auto *address_plan =
            reinterpret_cast<const uint64_t *>(inst.address);
        const uint64_t first_address = load_l2_u64(address_plan);
        const uint64_t second_address = load_l2_u64(address_plan + 1);
        char *destination = static_cast<char *>(
            get_slot_address(smem_base, slot));
        const uint32_t shared_destination = uint32_t(
            __cvta_generic_to_shared(destination));
        const uint32_t transaction_barrier = uint32_t(
            __cvta_generic_to_shared(m2c.native_bar(bar)));
        // Register the complete transaction before either independent copy can
        // retire, so both completion decrements belong to this barrier phase.
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar],
            cuda::aligned_size_t<16>(transfer_size));
        // NVCC 13.0 drops the second consecutive libcudacxx
        // memcpy_async_tx call in this runtime-dispatched basic block.  Keep
        // both copies in one volatile PTX statement, matching the established
        // paired-2D implementation below.
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            "[%0], [%1], %2, [%3];\n"
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            "[%4], [%5], %6, [%3];\n"
            :
            : "r"(shared_destination),
              "l"(first_address),
              "r"(first_size),
              "r"(transaction_barrier),
              "r"(shared_destination + first_size),
              "l"(second_address),
              "r"(transfer_size - first_size)
            : "memory");
        break; }
      case op(OP_ALLOC_RW_TMA_2D): {
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3}], [%4];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(
                get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)inst.coords[0]),
            "r"((int)inst.coords[1]),
            "r"((uint32_t)__cvta_generic_to_shared(m2c.native_bar(bar)))
          : "memory");
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar],
            cuda::aligned_size_t<16>(uint32_t(inst.size)));
        break; }
      case op(OP_ALLOC_TMA_LOAD_PAIR_2D): {
        const uint32_t transfer_size = uint32_t(inst.size);
        const uint32_t tile_size = transfer_size / 2;
        const uint16_t *cord = inst.coords;
        const uint32_t smem = uint32_t(__cvta_generic_to_shared(
            get_slot_address(smem_base, slot)));
        const uint32_t tx_bar = uint32_t(__cvta_generic_to_shared(
            m2c.native_bar(bar)));
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3}], [%4];\n"
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%5], [%1, {%6, %3}], [%4];\n"
          :
          : "r"(smem),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"(tx_bar),
            "r"(smem + tile_size),
            "r"((int)cord[0] + (int)cord[2])
          : "memory");
        cuda::device::barrier_expect_tx(
            m2c.barriers[bar],
            cuda::aligned_size_t<16>(transfer_size));
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
      case op(OP_TMA_LOAD_MX_COUPLED_STREAM): {
        produces_compute_operand = false;
        MInst stream_inst = inst;
        while (true) {
          const uint16_t stream_kind =
              stream_inst.arg & dae_mxfp_resident_ffn::kCoupledKindMask;
          if (stream_kind == dae_mxfp_resident_ffn::kCoupledFp8Gemv) {
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
            // The C128 Q-b projection is the unique pre-attention coupled
            // stream with four K256 pairs.  Keep the rolling prefix armed
            // across Q-a/KV and stop only at that command.
            detail_executed_fp8_coupled = stream_inst.size == 4;
#endif
            ldu_execute_mxfp8_coupled_stream(
                stream_inst, slot, port_id, smem_base, tmem_mma_barriers,
                mxfp8_coupled_pair_base
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
                , sm_id, g_events,
                !fp8_coupled_detail_complete &&
                    detail_executed_fp8_coupled
#endif
                );
          } else if (
              stream_kind == dae_mxfp_resident_ffn::kCoupledTmaRing) {
            ldu_execute_internal_ring_stream(
                stream_inst, slot, port_id, smem_base, tma_descs,
                tmem_mma_barriers, internal_ring_empty_phase_mask);
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
            if (attention_detail_ring_command) {
              g_events[sm_id * numProfileEvents +
                       detailProfileEventBase + 42] =
                  cuda::ptx::get_sreg_globaltimer();
            }
            attention_detail_ring_seen = true;
#endif
          } else if (
              stream_kind == dae_mxfp_resident_ffn::kCoupledLinear1) {
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            if (port_id == 0) {
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0PreviousBegin] =
                  detail_previous_begin;
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0PreviousEnd] =
                  detail_previous_end;
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0PreviousOpcode] =
                  detail_previous_opcode;
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0Linear1Begin] =
                  detail_command_begin;
            }
#endif
            if (stream_inst.arg &
                dae_mxfp_resident_ffn::kCoupledDynamicExpert) {
              ldu_execute_mxfp_coupled_linear1<true>(
                  stream_inst, smem_base, tma_descs, tmem_mma_barriers
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
                  , sm_id, g_events
#endif
                  );
            } else {
              ldu_execute_mxfp_coupled_linear1<false>(
                  stream_inst, smem_base, tma_descs, tmem_mma_barriers
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
                  , sm_id, g_events
#endif
                  );
            }
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            if (port_id == 0) {
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0Linear1End] =
                  cuda::ptx::get_sreg_globaltimer();
            }
#endif
            auto *poll_start = reinterpret_cast<
                cutlass::arch::ClusterTransactionBarrier *>(
                tmem_mma_barriers + mxfpDownResidentLdu1PollStartBarrier);
            poll_start->arrive();
          } else if (
              stream_kind == dae_mxfp_resident_ffn::kCoupledDownWeight) {
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            if (port_id == 0) {
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0DownBegin] =
                  cuda::ptx::get_sreg_globaltimer();
            }
#endif
            const auto *plan = reinterpret_cast<const uint64_t *>(
                stream_inst.address);
            const int down_task_count = load_l2(
                reinterpret_cast<const int *>(plan + 3));
            const uint64_t down_task_address0 = down_task_count > 0
                ? load_l2_u64(plan + 1)
                : 0;
            const uint64_t down_task_address1 = down_task_count > 1
                ? load_l2_u64(plan + 2)
                : 0;
            auto *linear1_empty = reinterpret_cast<
                cutlass::arch::ClusterTransactionBarrier *>(
                tmem_mma_barriers + mxfpResidentLinear1EmptyBarrierBase);
            if (!(stream_inst.arg &
                  dae_mxfp_resident_ffn::kCoupledDownOnly)) {
              linear1_empty[0].wait(0);
            }
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            if (port_id == 0) {
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0DownReady] =
                  cuda::ptx::get_sreg_globaltimer();
            }
#endif
            MInst task_inst = stream_inst;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            uint64_t detail_stage_empty_wait_ns = 0;
#endif
            for (int task = 0; task < down_task_count; ++task) {
              task_inst.address = task == 0
                  ? down_task_address0
                  : down_task_address1;
              if (stream_inst.arg &
                  dae_mxfp_resident_ffn::kCoupledDynamicExpert) {
                ldu_execute_mxfp_down_weight_stream<true>(
                    task_inst, smem_base, tma_descs,
                    tmem_mma_barriers, plan
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
                    , detail_stage_empty_wait_ns
#endif
                    );
              } else {
                ldu_execute_mxfp_down_weight_stream<false>(
                    task_inst, smem_base, tma_descs,
                    tmem_mma_barriers, nullptr
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
                    , detail_stage_empty_wait_ns
#endif
                    );
              }
            }
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            if (port_id == 0) {
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0DownEmptyWaitNs] =
                  detail_stage_empty_wait_ns;
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu0DownEnd] =
                  cuda::ptx::get_sreg_globaltimer();
            }
#endif
          } else {
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            if (port_id == 1) {
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu1ActivationBegin] =
                  cuda::ptx::get_sreg_globaltimer();
            }
#endif
            ldu_execute_mxfp_down_activation_stream(
                stream_inst, smem_base, bars, tmem_mma_barriers,
                mxfp_resident_down_phase
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
                , sm_id, g_events
#endif
                );
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
            if (port_id == 1) {
              g_events[sm_id * numProfileEvents +
                       mxfpFfnDetailLdu1ActivationEnd] =
                  cuda::ptx::get_sreg_globaltimer();
            }
#endif
            mxfp_resident_down_phase ^= 1U;
          }

          if (!(stream_inst.arg &
                dae_mxfp_resident_ffn::kCoupledLocalChain)) {
            break;
          }
          // Python proved adjacency, same LDU, and same shared area. Consume
          // the next immutable command locally; persistent barrier phase and
          // ring ownership never return to the allocator or outer dispatcher.
          m2ld.wait();
          const LdCmd next_stream {
              .raw = m2ld.data[m2ld.ptr]
          };
          m2ld.advance();
          stream_inst = st_insts[next_stream.slot];
#if defined(DAE_TRACK_PROFILE)
          ++commands;
#endif
        }
        break; }
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
        const uint32_t transfer_size = ldu_tensor_transfer_bytes(inst);
        __ldprint("TMA Tensor 1D Load: size=%u", transfer_size);
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
          cuda::aligned_size_t<16>(transfer_size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_2D): {
        const uint32_t transfer_size = ldu_tensor_transfer_bytes(inst);
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
          cuda::aligned_size_t<16>(transfer_size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_3D): {
        const uint32_t transfer_size = ldu_tensor_transfer_bytes(inst);
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
          cuda::aligned_size_t<16>(transfer_size)
        );
        break; }
      case op(OP_ALLOC_LAYER_TMA_LOAD_4D):
      case op(OP_ALLOC_TMA_LOAD_4D): {
        const uint32_t transfer_size = ldu_tensor_transfer_bytes(inst);
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
          cuda::aligned_size_t<16>(transfer_size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_5D_FIX0): {
        const uint32_t transfer_size = ldu_tensor_transfer_bytes(inst);
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
          cuda::aligned_size_t<16>(transfer_size)
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
#if defined(DAE_TRACK_PROFILE) && \
    !defined(DAE_MXFP_FFN_DETAIL_PROFILE) && \
    !defined(DAE_STU_HISTORY_PROFILE)
        const bool profile_reload =
            port_id == 0 && inst.nslot() == numSlots + 2;
        const int reload_event_id =
            reloadProfileEventBase + profile_reload_counter;
        if (profile_reload) {
          if (reload_event_id >= trackProfileEventBase) {
            asm volatile("trap;");
          }
          // Keep the start time in the diagnostic buffer. Retaining it in a
          // register across two warp and two device-wide rendezvous makes the
          // track-only megakernel carry a long-lived 64-bit value and proved
          // unreliable after many loop iterations.
          g_events[sm_id * numProfileEvents + reload_event_id] =
              cuda::ptx::get_sreg_globaltimer();
        }
#endif
        constexpr uint16_t kFirstBarMask = (1U << 10) - 1;
        const int source_first_bar = inst.arg & kFirstBarMask;
        // Route results are constant throughout one loop iteration.  Drop the
        // LDU-local all-rank cache only after both ports have drained so the
        // next layer/step cannot observe stale expert IDs.
        cachedRouteAddress = 0;
        // Both LDU lanes reach this point only after all earlier commands on
        // their own ports have drained and the loop-tail STU dependency has
        // reached zero. The first rendezvous hands that fact to port 0.
        ldu_control_barrier->arrive_and_wait();
        if (port_id == 0) {
          // A zero-valued completion counter is one generation of a reusable
          // global barrier. No block may restore that counter until every
          // block has consumed the zero generation. Otherwise a fast block
          // can publish the next nonzero generation while a late LDU is still
          // waiting for the old one, creating a cycle. This device-wide ready
          // phase is the direct loop-carried dependency; it is not an
          // allocator IssueBarrier.
          cuda::atomic_ref<int, cuda::thread_scope_device> ready_ref(
              bars[lduBarrierReloadArrival]);
          const int ready_ticket = ldu_reload_fetch_add(
              &bars[lduBarrierReloadArrival], 1,
              cuda::memory_order_acq_rel);
          const int ready_phase_end =
              (ready_ticket / gridDim.x + 1) * gridDim.x;
          while (ready_ref.load(cuda::memory_order_acquire) <
                 ready_phase_end) {
            __nanosleep(reloadBarrierPollSleepCycles);
          }

          // Resident mbarriers are persistent objects. Their cyclic data
          // rings complete an even number of generations per layer, while
          // allocator, LDU1, and compute carry explicit phase state for the
          // three one-shot Down handoffs. Never reinitialize live barrier
          // storage at this cross-layer rendezvous.
          const int count = inst.size;
          // The attached completion barrier is the last barrier in the
          // active shifted bank. Derive that bank's first barrier instead of
          // restoring every bank on every loop iteration.
          const int first_bar = inst.bar() + 1 - count;
          const int *source = reinterpret_cast<const int *>(inst.address);
          if (first_bar < source_first_bar || count <= 0 ||
              (first_bar - source_first_bar) % count != 0 ||
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

          // The second phase orders every block's disjoint counter stores.
          // All blocks observe the completed reload before either LDU port
          // can advance into the next iteration.
          cuda::atomic_ref<int, cuda::thread_scope_device> done_ref(
              bars[lduBarrierReloadDone]);
          if constexpr (reloadBarrierUseAtomicAdd) {
            __threadfence();
          }
          const int done_ticket = ldu_reload_fetch_add(
              &bars[lduBarrierReloadDone], 1,
              cuda::memory_order_acq_rel);
          const int done_phase_end =
              (done_ticket / gridDim.x + 1) * gridDim.x;
          while (done_ref.load(cuda::memory_order_acquire) < done_phase_end) {
            __nanosleep(reloadBarrierPollSleepCycles);
          }

        }
        // Port 1 cannot consume a following loop iteration until port 0 has
        // observed the device-wide completion phase after restoring counters.
        ldu_control_barrier->arrive_and_wait();
#if defined(DAE_TRACK_PROFILE) && \
    !defined(DAE_MXFP_FFN_DETAIL_PROFILE) && \
    !defined(DAE_STU_HISTORY_PROFILE)
        if (profile_reload) {
          uint64_t &profile_value =
              g_events[sm_id * numProfileEvents + reload_event_id];
          profile_value =
              cuda::ptx::get_sreg_globaltimer() - profile_value;
          ++profile_reload_counter;
        }
#endif
        break;
      }
      case op(OP_LDU_ASYNC_RELOAD_BARRIERS): {
        produces_compute_operand = false;
        if constexpr (dae2AsyncBarrierReload) {
          if (port_id == 0) {
            const int first_bar = inst.arg & ((1U << 10) - 1);
            const int count = inst.size & ((1U << 6) - 1);
            const int input_bar = inst.size >> 6;
            const int *source = reinterpret_cast<const int *>(inst.address);
            constexpr uint16_t kBankReadyCompletion = 1U << 14;
            constexpr uint16_t kBankReadyLeader = 1U << 15;
            const bool bank_ready_completion =
                inst.arg & kBankReadyCompletion;
            cuda::atomic_ref<int, cuda::thread_scope_device> worker_join(
                bars[inst.bar()]);
            if (bank_ready_completion) {
              // Arm the clear-worker rendezvous while the current compute
              // tail is still live. The adjacent lower counter is a monotonic
              // bank-ready generation observed by every LDU entry command.
              if (inst.arg & kBankReadyLeader) {
                atomicExch(&bars[inst.bar()],
                           3 * asyncBarrierReloadWorkers);
              } else {
                while (worker_join.load(cuda::memory_order_acquire) == 0) {
                  __nanosleep(reloadBarrierPollSleepCycles);
                }
              }
              atomicSub(&bars[inst.bar()], 1);
              while (worker_join.load(cuda::memory_order_acquire) >
                     2 * asyncBarrierReloadWorkers) {
                __nanosleep(reloadBarrierPollSleepCycles);
              }
            }
            cuda::atomic_ref<int, cuda::thread_scope_device> input_dependency(
                bars[input_bar]);
            while (input_dependency.load(cuda::memory_order_acquire) != 0) {
              __nanosleep(reloadBarrierPollSleepCycles);
            }
            if (bank_ready_completion) {
              // The completion counter itself belongs to one worker's clear
              // slice. Make every worker consume its zero generation before
              // that slice can publish the next nonzero generation.
              atomicSub(&bars[inst.bar()], 1);
              while (worker_join.load(cuda::memory_order_acquire) >
                     asyncBarrierReloadWorkers) {
                __nanosleep(reloadBarrierPollSleepCycles);
              }
            }
            for (int offset = 0; offset < count; ++offset) {
              cuda::atomic_ref<int, cuda::thread_scope_device> destination(
                  bars[first_bar + offset]);
              if (!bank_ready_completion) {
                // The task-local internal range has no inactive branches, so
                // each old counter is also an exact last-use dependency.
                while (destination.load(cuda::memory_order_acquire) != 0) {
                  __nanosleep(reloadBarrierPollSleepCycles);
                }
              }
              destination.store(
                  load_l2(source + offset), cuda::memory_order_relaxed);
            }
            // Publish the slice stores to the ordinary sequential join. The
            // built-in atomic is device-scope relaxed, so supply the release
            // edge explicitly.
            __threadfence();
            if (bank_ready_completion) {
              const int previous = worker_join.fetch_sub(
                  1, cuda::memory_order_acq_rel);
              if (previous == 1) {
                __threadfence();
                atomicAdd(&bars[inst.bar() - 1], 1);
              }
            } else {
              atomicSub(&bars[inst.bar()], 1);
            }
          }
          // Port 1 only acknowledges the immutable mailbox. Keeping this
          // rendezvous local lets LDU0 perform the clear without involving
          // any CTA outside the selected worker subset.
          ldu_control_barrier->arrive_and_wait();
        }
        break;
      }
      case op(OP_LDU_WAIT_BARRIER): {
        produces_compute_operand = false;
        if constexpr (dae2AsyncBarrierReload) {
          // The allocator encodes the current outer-loop generation into the
          // immutable mailbox. Every LDU port observes that exact generation
          // before it can dequeue any command from the reused bank, and the
          // boundary also invalidates route IDs cached in handler registers.
          cachedRouteAddress = 0;
          cuda::atomic_ref<int, cuda::thread_scope_device> ready(
              bars[inst.bar()]);
          while (ready.load(cuda::memory_order_acquire) < inst.arg) {
            __nanosleep(reloadBarrierPollSleepCycles);
          }
        }
        break;
      }
    }

    if (op(opcode) == op(OP_LDU_RELOAD_BARRIERS)) {
      ldu_control_publish_barrier->arrive_and_wait();
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
      if (port_id == 0 && inst.nslot() == numSlots + 2) {
        if ((fp8_coupled_detail_loop_tail_reloads & 1U) == 0) {
          g_events[sm_id * numProfileEvents + fp8CoupledReloadEndEvent] =
              cuda::ptx::get_sreg_globaltimer();
        }
      }
#endif
    }

#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
    // Retain a rolling tail of the LDU prefix after the loop-tail reload and
    // stop on the first coupled FP8 projection.  This shows which queue-local
    // command holds a port immediately before Q-a without growing the profile
    // image.  The reload itself defines the trace origin and is omitted.
    if (op(opcode) == op(OP_LDU_RELOAD_BARRIERS) &&
        inst.nslot() == numSlots + 2) {
      // The two-layer reproducer loops one compact layer body twice.  Drop
      // the first layer's trace, then retain the second layer's Q-a prefix
      // through its trailing reload so the host can inspect it.  The next
      // token's first reload re-arms the same diagnostic window.
      ++fp8_coupled_detail_loop_tail_reloads;
      if ((fp8_coupled_detail_loop_tail_reloads & 1U) != 0) {
        fp8_coupled_detail_index = 0;
        fp8_coupled_detail_complete = false;
      }
    } else if (op(opcode) != op(OP_LDU_RELOAD_BARRIERS) &&
               !fp8_coupled_detail_complete) {
      const uint64_t detail_command_end =
          cuda::ptx::get_sreg_globaltimer();
      const uint64_t duration = min(
          detail_command_end - detail_command_begin, 0xfffffULL);
      const uint64_t normalized_opcode = detail_executed_fp8_coupled
          ? 0x838ULL
          : (uint64_t(op(opcode)) & 0xfffULL);
      const int event_id = fp8CoupledDetailLduEventBase +
          port_id * fp8CoupledDetailCommands +
          int(fp8_coupled_detail_index % fp8CoupledDetailCommands);
      g_events[sm_id * numProfileEvents + event_id] =
          uint64_t(uint32_t(detail_command_begin)) |
          (duration << 32) | (normalized_opcode << 52);
      ++fp8_coupled_detail_index;
      fp8_coupled_detail_complete = detail_executed_fp8_coupled;
    }
#endif

#if defined(DAE_ATTENTION_DETAIL_PROFILE)
    if (attention_detail_q_command) {
      g_events[sm_id * numProfileEvents + detailProfileEventBase + 45] =
          cuda::ptx::get_sreg_globaltimer();
      attention_detail_q_complete = true;
    }
#endif

#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    detail_previous_begin = detail_command_begin;
    detail_previous_end = cuda::ptx::get_sreg_globaltimer();
    detail_previous_opcode = opcode;
#endif

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
