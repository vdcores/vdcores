#pragma once

#include "context.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>
#include <cutlass/numeric_types.h>

// Build the SMEM descriptor tensor on device. Persistent DAE tasks receive
// allocator-owned raw pointers instead of constructing these descriptors in
// host-side CUTLASS argument plumbing.
template <class UtccpOp, class TEngine, class TLayout>
__device__ __forceinline__ auto dae_mxfp_get_utccp_smem_desc_tensor(
    cute::Tensor<TEngine, TLayout> const &smem_tensor) {
  using namespace cute;
  using VecLayout = decltype(layout<0>(TLayout{}));
  static_assert(VecLayout::rank == 2 && shape<1>(VecLayout{}) == 1);
  static_assert(is_smem<TEngine>::value);
  static_assert(is_static<VecLayout>::value);

  using Value = typename TEngine::value_type;
  using Traits = Copy_Traits<UtccpOp>;
  auto core_shape = take<0, 2>(
      upcast<sizeof_bits_v<Value>>(typename Traits::ValID{}).shape());
  Layout vec_layout = flatten(layout<0>(VecLayout{}));
  Layout core_layout = vec_layout.with_shape(core_shape);
  Tensor core_tensor = group_modes<0, 2>(
      make_tensor(smem_tensor.data(), core_layout));
  Tensor descriptor =
      make_tensor<UMMA::smem_desc<UMMA::Major::K>>(core_tensor);
  return make_tensor(
      descriptor.data(), recast_layout<Value, uint128_t>(smem_tensor.layout()));
}

template <int Bytes>
__device__ __forceinline__ void dae_mxfp_cp_async_scale_stage(
    const uint8_t *global_source,
    uint8_t *shared_destination,
    int lane) {
  static_assert(Bytes % int(sizeof(uint4)) == 0);
  constexpr int kVectors = Bytes / int(sizeof(uint4));
  #pragma unroll
  for (int vector = lane; vector < kVectors;
       vector += numThreadsPerWarp) {
    const auto *global_vector = global_source +
        vector * int(sizeof(uint4));
    auto *shared_vector = shared_destination +
        vector * int(sizeof(uint4));
    const uint32_t shared_address = static_cast<uint32_t>(
        __cvta_generic_to_shared(shared_vector));
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;"
        :: "r"(shared_address), "l"(global_vector)
        : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}

// Named barriers 12..15 join issuer warp 0 with the two independent scale
// producer warps. Producers arrive without waiting after their global-to-SMEM
// copies; the issuer waits only when it reaches that K512 stage. Warp 1 is
// dedicated to completion and allocator retirement.
static constexpr int mxfp4Mxfp8ScaleReadyBarrierBase = 12;
static constexpr int mxfp4Mxfp8ScaleReadyThreads = 3 * numThreadsPerWarp;

// Fixed-shape W4A8 projection. One task owns M128, one logical activation row
// replicated to N8, and eight K512 tiles (K4096). HBM already contains the
// exact native layouts. Packed FP4 weights are expanded/swizzled by TMA into
// four concatenated 16 KiB K128 SMEM records inside each K512 allocation.
// K is the independently streamed weight/compute tile and BLoad is the number
// of consecutive activation tiles in one allocation, matching GEMV_WGMMA's
// K/BLOAD contract. BLoad=8 is one full-activation load; 1/2/4 are tiled
// streaming points. Weight allocations remain one K512 tile in every case.
//
// ScaleFromMetadata=false token order:
//   per activation chunk: activation data, then per K512 stage:
//     direct SFA scale, direct SFB scale, packed-to-native weight data
// The direct scale commands bypass the normal allocator and TMA into a
// dedicated task-tail ring of 4-KiB stages.
// ScaleFromMetadata=true token order:
//   per activation chunk: activation data, then per K512 stage: weight data
// Metadata byte offsets 16 and 24 hold the weight- and activation-scale bases.
template <bool ScaleFromMetadata, int K, int BLoad,
          typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_mxfp4_mxfp8_gemv_umma_k512_fp32_sm100(
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    const uint8_t *metadata,
    uint32_t &tmem_mma_phase,
    uint32_t &pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m
#if defined(DAE_TRACK_MXFP_TIMELINE)
    , int sm_id, uint64_t *g_events
#endif
    ) {
  using namespace cute;
  // The mixed F8/F6/F4 family consumes FP4 after TMA's unpack-to-SMEM
  // transform.  The unpacksmem tag is not merely an allocation hint: it also
  // selects E2M1 (format 5) in the mixed-family instruction descriptor.
  // float_e2m1_t selects the standalone MXF4 encoding (format 1), which this
  // instruction would interpret as E5M2.
  using Weight = cutlass::detail::float_e2m1_unpacksmem_t;
  using Activation = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Accum = float;

  static_assert(K == 512, "native MXFP4/MXFP8 currently supports K512 tiles");
  static_assert(BLoad == 1 || BLoad == 2 || BLoad == 4 || BLoad == 8);
  static_assert(
      ScaleFromMetadata || mxfp4Mxfp8DirectTmaEnabled,
      "direct MXFP4/MXFP8 TMA task requires mxfp_direct_tma=1");
  constexpr int kNumWeightTiles = 4096 / K;
  static_assert(kNumWeightTiles % BLoad == 0);
  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kTileK = 128;
  constexpr int kK128PerWeightTile = K / kTileK;
  constexpr int kScaleVector = 32;
  constexpr int kStages = fp8UmmaPipelineStages;
  constexpr int kFullBarrierBase = fp8UmmaPipelineBarrierBase;
  constexpr int kEmptyBarrierBase = kFullBarrierBase + kStages;

  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = SM100_MMA_MXF8F6F4_SS<
      Weight, Activation, Accum, Scale, kTileM, kTileN,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;

  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);
  auto mma_shape_a = partition_shape_A(
      tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto mma_shape_b = partition_shape_B(
      tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  // MXF8/F6/F4 uses an 8-bit SMEM allocation unit for every input format.
  // FP4 packing is selected by the UMMA instruction descriptor; using an
  // FP4-width swizzle atom here changes descriptor byte addressing and is not
  // equivalent. This mirrors Element{A,B}Mma_SmemAllocType in CUTLASS's
  // SM100 block-scaled builder.
  auto layout_sA = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<uint8_t>{}, mma_shape_a);
  auto layout_sB = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<uint8_t>{}, mma_shape_b);
  using LayoutSFA = decltype(
      ScaleConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape{}));
  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));

  constexpr int kDescriptorAlignment = 128;
  constexpr int kWeightK128Bytes =
      (cosize_v<decltype(layout_sA)> +
       kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kActivationK128Bytes =
      (cosize_v<decltype(layout_sB)> + kDescriptorAlignment - 1) &
      -kDescriptorAlignment;
  constexpr int kSfaK128Bytes = cosize_v<LayoutSFA>;
  constexpr int kSfbK128Bytes = cosize_v<LayoutSFB>;
  constexpr int kWeightTileBytes =
      kK128PerWeightTile * kWeightK128Bytes;
  constexpr int kActivationTileBytes =
      kK128PerWeightTile * kActivationK128Bytes;
  constexpr int kSfaTileBytes = kK128PerWeightTile * kSfaK128Bytes;
  constexpr int kSfbTileBytes = kK128PerWeightTile * kSfbK128Bytes;
  static_assert(kWeightK128Bytes == 16384);
  static_assert(kActivationK128Bytes == 1024);
  static_assert(kSfaK128Bytes == 512);
  static_assert(kSfbK128Bytes == 512);
  static_assert(kWeightTileBytes == 65536);
  static_assert(kActivationTileBytes == 4096);
  static_assert(kSfaTileBytes == 2048);
  static_assert(kSfbTileBytes == 2048);

  auto logical_c = make_tensor(
      make_smem_ptr(static_cast<Accum *>(nullptr)),
      make_layout(
          make_shape(Int<kTileM>{}, Int<kTileN>{}),
          make_stride(Int<kTileN>{}, Int<1>{})));
  auto cta_c = cta_mma.partition_C(logical_c);
  auto tmem_acc = cta_mma.make_fragment_C(cta_c);
  tmem_acc.data() = tmem_base_ptr;
  auto tmem_sfa_probe = make_tensor<typename TiledMma::FrgTypeSFA>(
      shape(LayoutSFA{}));
  auto tmem_sfb_probe = make_tensor<typename TiledMma::FrgTypeSFB>(
      shape(LayoutSFB{}));
  const int accumulator_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_acc));
  const int sfa_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_sfa_probe));
  const int sfb_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_sfb_probe));
  // One SFA/SFB pair is sufficient per pipeline stage. UTCCP for K128(i+1)
  // follows the dependent UMMA for K128(i) in the same TCGEN issue stream,
  // so the columns can be reused without a completion wait.
  const int scale_stage_columns = sfa_columns + sfb_columns;
  const uint32_t scale_pipeline_base =
      tmem_base_ptr + accumulator_columns;
  if (accumulator_columns + kStages * scale_stage_columns >
      cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns) {
    asm volatile("trap;");
  }

  using Utccp = SM100_UTCCP_4x32dp128bit_1cta;
  const int tid = __compute_tid();
  const int warp = tid / numThreadsPerWarp;
  const int lane = tid & (numThreadsPerWarp - 1);
#if defined(DAE_TRACK_MXFP_TIMELINE)
  auto *profile_events = g_events + sm_id * numProfileEvents;
  if (tid == 0) {
    profile_events[mxfpProfileTaskEntry] =
        cuda::ptx::get_sreg_globaltimer();
  }
#endif

  const uint8_t *weight_scale_global = nullptr;
  const uint8_t *activation_scale_global = nullptr;
  if constexpr (ScaleFromMetadata) {
    if (warp == 2) {
      weight_scale_global = reinterpret_cast<const uint8_t *>(
          *reinterpret_cast<const uint64_t *>(metadata + 16));
    } else if (warp == 3) {
      activation_scale_global = reinterpret_cast<const uint8_t *>(
          *reinterpret_cast<const uint64_t *>(metadata + 24));
    }
  }

  constexpr int kScaleScratchStageBytes =
      kSfaTileBytes + kSfbTileBytes;
  constexpr int kScaleScratchBytes = kStages * kScaleScratchStageBytes;
  constexpr int kDirectScaleScratchBytes =
      mxfp4Mxfp8TmaScaleStages * kScaleScratchStageBytes;
  constexpr int kTaskScratchBytes =
      dynamicSmemBytes - numSlots * slotSizeKb * 1024;
  constexpr int kSmemBaseAlignmentSlack = 1023;
  static_assert(
      !ScaleFromMetadata ||
      kTaskScratchBytes >= kScaleScratchBytes +
          kSmemBaseAlignmentSlack,
      "MXFP4/MXFP8 scale ring must fit after the allocator arena");
  static_assert(
      ScaleFromMetadata ||
      kTaskScratchBytes >= kDirectScaleScratchBytes +
          kSmemBaseAlignmentSlack,
      "direct MXFP4/MXFP8 TMA scale stages must fit after the allocator arena");
  auto *scale_scratch = static_cast<uint8_t *>(
      get_slot_address(smem_base, numSlots));

  #pragma unroll
  for (int chunk_start = 0; chunk_start < kNumWeightTiles;
       chunk_start += BLoad) {
    int activation_data_slots = 0;
    uint8_t *activation_data_base = nullptr;

    if (warp < 2) {
      activation_data_slots = m2c.template pop<0>();
      if (warp == 0) {
        activation_data_base = static_cast<uint8_t *>(
            get_slot_address(smem_base, extract(activation_data_slots)));
      }
    } else {
      m2c.advance();
    }
#if defined(DAE_TRACK_MXFP_TIMELINE)
    if (tid == 0) {
      const uint64_t activation_ready = cuda::ptx::get_sreg_globaltimer();
      #pragma unroll
      for (int profile_tile = chunk_start;
           profile_tile < chunk_start + BLoad; ++profile_tile) {
        profile_events[mxfpProfileActivationReadyBase + profile_tile] =
            activation_ready;
      }
    }
#endif
    #pragma unroll
    for (int local_stage = 0;
         local_stage < BLoad;
         ++local_stage) {
      const int tile = chunk_start + local_stage;
      const int stage = tile % kStages;
      const int generation = tile / kStages;
      int weight_data_slots = 0;
      uint8_t *weight_data_base = nullptr;

#if defined(DAE_TRACK_MXFP_TIMELINE)
      if constexpr (ScaleFromMetadata) {
        if (tid == 2 * numThreadsPerWarp) {
          profile_events[mxfpProfileSfaProducerStartBase + tile] =
              cuda::ptx::get_sreg_globaltimer();
        } else if (tid == 3 * numThreadsPerWarp) {
          profile_events[mxfpProfileSfbProducerStartBase + tile] =
              cuda::ptx::get_sreg_globaltimer();
        }
      }
#endif

      if constexpr (!ScaleFromMetadata) {
        // Only the issuer needs the TMA acquire. Other compute warps retain
        // identical queue positions without joining the scale dependency.
        if (warp == 0) {
          (void)m2c.template pop<0>();
          (void)m2c.template pop<0>();
#if defined(DAE_TRACK_MXFP_TIMELINE)
          if (tid == 0) {
            profile_events[mxfpProfileScaleReadyBase + tile] =
                cuda::ptx::get_sreg_globaltimer();
          }
#endif
        } else {
          m2c.advance_by(2);
        }
      }

      if (warp < 2) {
        weight_data_slots = m2c.template pop<0>();
        if (warp == 0) {
          weight_data_base = static_cast<uint8_t *>(
              get_slot_address(smem_base, extract(weight_data_slots)));
        }
      } else {
        m2c.advance();
      }
#if defined(DAE_TRACK_MXFP_TIMELINE)
      if (tid == 0) {
        profile_events[mxfpProfileWeightReadyBase + tile] =
            cuda::ptx::get_sreg_globaltimer();
      }
#endif
      // The issuer protects every TMEM scale stage before reuse. Metadata
      // producers observe the same completion so they can safely refill the
      // corresponding shared scratch stage in parallel.
      if (generation > 0 &&
          (warp == 0 || (ScaleFromMetadata && warp >= 2))) {
        const uint32_t stage_phase =
            (pipeline_phase_mask >> stage) & 1U;
        cute::wait_barrier(
            tmem_mma_barrier[kEmptyBarrierBase + stage],
            stage_phase ^ uint32_t((generation - 1) & 1));
      }

      if constexpr (ScaleFromMetadata) {
        if (warp >= 2) {
          auto *stage_scratch =
              scale_scratch + stage * kScaleScratchStageBytes;
          const uint8_t *global_source = warp == 2
              ? weight_scale_global + tile * kSfaTileBytes
              : activation_scale_global + tile * kSfbTileBytes;
          auto *shared_destination = warp == 2
              ? stage_scratch
              : stage_scratch + kSfaTileBytes;
          dae_mxfp_cp_async_scale_stage<kSfaTileBytes>(
              global_source, shared_destination, lane);
          asm volatile("cp.async.wait_group 0;" ::: "memory");
          __syncwarp();
          cutlass::arch::fence_view_async_shared();
#if defined(DAE_TRACK_MXFP_TIMELINE)
          if (lane == 0) {
            const int event_base = warp == 2
                ? mxfpProfileSfaProducerReadyBase
                : mxfpProfileSfbProducerReadyBase;
            profile_events[event_base + tile] =
                cuda::ptx::get_sreg_globaltimer();
          }
#endif
          __arrive_barrier_unaligned<mxfp4Mxfp8ScaleReadyThreads>(
              mxfp4Mxfp8ScaleReadyBarrierBase + stage);
        } else if (warp == 0) {
          __sync_barrier_unaligned<mxfp4Mxfp8ScaleReadyThreads>(
              mxfp4Mxfp8ScaleReadyBarrierBase + stage);
#if defined(DAE_TRACK_MXFP_TIMELINE)
          if (tid == 0) {
            profile_events[mxfpProfileScaleReadyBase + tile] =
                cuda::ptx::get_sreg_globaltimer();
          }
#endif
        }
      }

      if (warp == 0) {
        const uint8_t *sfa_source = nullptr;
        const uint8_t *sfb_source = nullptr;
        if constexpr (ScaleFromMetadata) {
          auto *stage_scratch =
              scale_scratch + stage * kScaleScratchStageBytes;
          sfa_source = stage_scratch;
          sfb_source = stage_scratch + kSfaTileBytes;
        } else {
          auto *stage_scratch =
              scale_scratch +
              (tile % mxfp4Mxfp8TmaScaleStages) * kScaleScratchStageBytes;
          sfa_source = stage_scratch;
          sfb_source = stage_scratch + kSfaTileBytes;
        }

        // UMMA issue and commit are warp-collective even though their PTX
        // encoding elects one lane internally. Keep every issuer lane on the
        // same control path. Interleave each K128 scale copy with its dependent
        // UMMA bundle, exactly matching CUTLASS's TCGEN ordering contract.
        #pragma unroll
        for (int subtile = 0; subtile < kK128PerWeightTile; ++subtile) {
          auto sA = make_tensor(
              make_smem_ptr(reinterpret_cast<uint8_t *>(
                  weight_data_base + subtile * kWeightK128Bytes)),
              layout_sA);
          auto sB = make_tensor(
              make_smem_ptr(reinterpret_cast<Activation *>(
                  activation_data_base +
                  local_stage * kActivationTileBytes +
                  subtile * kActivationK128Bytes)),
              layout_sB);
          auto tCrA = cta_mma.make_fragment_A(sA);
          auto tCrB = cta_mma.make_fragment_B(sB);
          const uint32_t scale_substage_base =
              scale_pipeline_base + stage * scale_stage_columns;
          auto stage_sfa = make_tensor<typename TiledMma::FrgTypeSFA>(
              shape(LayoutSFA{}));
          auto stage_sfb = make_tensor<typename TiledMma::FrgTypeSFB>(
              shape(LayoutSFB{}));
          stage_sfa.data() = scale_substage_base;
          stage_sfb.data() = scale_substage_base + sfa_columns;
          if (elect_one_sync()) {
            auto compact_sfa = make_tensor(
                stage_sfa.data(), filter_zeros(stage_sfa.layout()));
            auto compact_sfb = make_tensor(
                stage_sfb.data(), filter_zeros(stage_sfb.layout()));
            auto copy_sfa = make_utccp_copy(Utccp{}, compact_sfa);
            auto copy_sfb = make_utccp_copy(Utccp{}, compact_sfb);
            auto copy_sfa_slice = copy_sfa.get_slice(0);
            auto copy_sfb_slice = copy_sfb.get_slice(0);
            auto smem_sfa = make_tensor(
                make_smem_ptr(reinterpret_cast<Scale *>(
                    const_cast<uint8_t *>(sfa_source) +
                    subtile * kSfaK128Bytes)),
                LayoutSFA{});
            auto smem_sfb = make_tensor(
                make_smem_ptr(reinterpret_cast<Scale *>(
                    const_cast<uint8_t *>(sfb_source) +
                    subtile * kSfbK128Bytes)),
                LayoutSFB{});
            auto smem_sfa_compact = make_tensor(
                smem_sfa.data(), filter_zeros(smem_sfa.layout()));
            auto smem_sfb_compact = make_tensor(
                smem_sfb.data(), filter_zeros(smem_sfb.layout()));
            auto copy_sfa_src = dae_mxfp_get_utccp_smem_desc_tensor<Utccp>(
                copy_sfa_slice.partition_S(smem_sfa_compact));
            auto copy_sfb_src = dae_mxfp_get_utccp_smem_desc_tensor<Utccp>(
                copy_sfb_slice.partition_S(smem_sfb_compact));
            auto copy_sfa_dst = copy_sfa_slice.partition_D(compact_sfa);
            auto copy_sfb_dst = copy_sfb_slice.partition_D(compact_sfb);
            copy(copy_sfa, copy_sfa_src, copy_sfa_dst);
            copy(copy_sfb, copy_sfb_src, copy_sfb_dst);
          }
          #pragma unroll
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            const auto accumulate = tile == 0 && subtile == 0 && k_block == 0
                ? UMMA::ScaleOut::Zero
                : UMMA::ScaleOut::One;
            gemm(
                tiled_mma.with(
                    accumulate,
                    stage_sfa(_, _, k_block),
                    stage_sfb(_, _, k_block)),
                tCrA(_, _, k_block),
                tCrB(_, _, k_block),
                tmem_acc);
          }
        }
        cutlass::arch::umma_arrive(
            tmem_mma_barrier + kFullBarrierBase + stage);
#if defined(DAE_TRACK_MXFP_TIMELINE)
        if (tid == 0) {
          profile_events[mxfpProfileUmmaIssueBase + tile] =
              cuda::ptx::get_sreg_globaltimer();
        }
#endif
      } else if (warp == 1) {
        cute::wait_barrier(
            tmem_mma_barrier[kFullBarrierBase + stage],
            ((pipeline_phase_mask >> stage) & 1U) ^
                uint32_t(generation & 1));
#if defined(DAE_TRACK_MXFP_TIMELINE)
        if (tid == numThreadsPerWarp) {
          profile_events[mxfpProfileUmmaCompleteBase + tile] =
              cuda::ptx::get_sreg_globaltimer();
        }
#endif
        int release_slots = weight_data_slots;
        if (local_stage + 1 == BLoad) {
          release_slots |= activation_data_slots;
        }
        c2m.template push<numThreadsPerWarp>(tid, release_slots);
        if (tid == numThreadsPerWarp) {
          cuda::ptx::mbarrier_arrive(
              cuda::ptx::sem_release,
              cuda::ptx::scope_cta,
              cuda::ptx::space_shared,
              tmem_mma_barrier + kEmptyBarrierBase + stage);
          if constexpr (!ScaleFromMetadata) {
            cuda::ptx::mbarrier_arrive(
                cuda::ptx::sem_release,
                cuda::ptx::scope_cta,
                cuda::ptx::space_shared,
                tmem_mma_barrier + mxfp4Mxfp8TmaScaleBarrierBase +
                    (tile % mxfp4Mxfp8TmaScaleStages));
          }
        }
      }
    }
  }

  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_compute_group(128);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  #pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    const int stage_uses =
        (kNumWeightTiles + kStages - 1 - stage) / kStages;
    if (stage_uses & 1) {
      pipeline_phase_mask ^= 1U << stage;
    }
  }

  const int output_slots = m2c.template pop<0>();
#if defined(DAE_TRACK_MXFP_TIMELINE)
  if (tid == 0) {
    profile_events[mxfpProfileOutputReady] =
        cuda::ptx::get_sreg_globaltimer();
  }
#endif
  auto *output = static_cast<float *>(
      get_slot_address(smem_base, extract(output_slots)));
  auto coord_c = make_identity_tensor(
      make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto cta_coord_c = cta_mma.partition_C(coord_c);
  using TmemLoad = SM100_TMEM_LOAD_32dp32b1x;
  auto tAcc = tmem_acc(make_coord(_, _), _0{}, _0{});
  auto cAcc = cta_coord_c(make_coord(_, _), _0{}, _0{});
  auto tiled_t2r = make_tmem_copy(TmemLoad{}, tAcc);
  const int thread_idx = tid % size(tiled_t2r);
  auto thread_t2r = tiled_t2r.get_slice(thread_idx);
  auto thread_tmem = thread_t2r.partition_S(tAcc);
  auto thread_coord = thread_t2r.partition_D(cAcc);
  auto r_acc = make_tensor<Accum>(shape(thread_coord));
  copy(tiled_t2r, thread_tmem, r_acc);
  #pragma unroll
  for (int index = 0; index < size(r_acc); ++index) {
    const int row = int(get<0>(thread_coord(index)));
    const int col = int(get<1>(thread_coord(index)));
    if (row < kTileM && col == 0) {
      output[row] = r_acc(index);
    }
  }

  __sync_compute_group(128);
  c2m.template push<0, true>(tid, output_slots);
#if defined(DAE_TRACK_MXFP_TIMELINE)
  if (tid == 0) {
    profile_events[mxfpProfileTaskEnd] =
        cuda::ptx::get_sreg_globaltimer();
  }
#endif
}
