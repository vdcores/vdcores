#pragma once

#include "mxfp4_mxfp8_gate_up_silu.cuh"

// Full-FFN Linear-2 task.  One CTA owns one expert/M128 output tile and
// consumes the sixteen native MXFP8 records produced by fused Linear-1.
// Those records are copied directly into the UMMA-B K128 images; no format
// conversion, row replication, or intermediate repack occurs.
template <int KBundles, int RingStages, int BundleK,
          int SyncBarrierId, int TmemColumns,
          int ScratchOffsetBytes, int ScratchCapacityBytes, int ThreadOffset,
          bool UseLduWeightRing, bool ResidentAllTma,
          typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_mxfp4_mxfp8_down_fixed_ring_sm100(
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *resident_mma_barriers,
    const CUtensorMap *tma_descs,
    const uint8_t *metadata,
    int *global_bars,
    M2CQueue &m2c,
    C2MQueue &c2m,
    int resident_task_index
    ) {
  using namespace cute;
  using Weight = cutlass::detail::float_e2m1_unpacksmem_t;
  using Activation = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Accum = float;
  using Output = cutlass::bfloat16_t;
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kNativeOutputRows = 8;
  constexpr int kTileK = 128;
  constexpr int kBundleK = BundleK;
  constexpr int kK128PerBundle = kBundleK / kTileK;
  static_assert(KBundles == 2 || KBundles == 4 || KBundles == 8);
  static_assert(
      RingStages == 1 || RingStages == 2 || RingStages == 3 ||
          RingStages == 4);
  static_assert(BundleK == 256 || BundleK == 512);
  static_assert(
      !UseLduWeightRing ||
          ((RingStages == 2 || RingStages == 3 || RingStages == 4) &&
           BundleK == 256));
  static_assert(
      UseLduWeightRing == ResidentAllTma,
      "LDU-owned Down storage is reserved for the resident FFN");
  constexpr bool kLduWeightScaleTma = ResidentAllTma;
  constexpr bool kSeparateWeightScaleBarrier = false;
  constexpr int kTotalK = KBundles * kBundleK;
  constexpr int kNumKTiles = kTotalK / kBundleK;
  constexpr int kRingStages = RingStages;
  constexpr int kScaleVector = 32;
  constexpr int kWeightPackedBytes = kTileM * kBundleK / 2;
  constexpr int kActivationRecordBytes = 1536;
  constexpr int kActivationRecordDataBytes = kNativeOutputRows * kTileK;
  constexpr int kActivationRecordScaleBytes = 512;
  constexpr int kDownTilesPerExpert = 4096 / kTileM;
  constexpr int kOutputElements = kTileM * kNativeOutputRows;
  constexpr int kOutputBytes = kOutputElements * int(sizeof(Output));

  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = cute::SM100_MMA_MXF8F6F4_SS<
      Weight, Activation, Accum, Scale, kTileM, kTileN,
      cute::UMMA::Major::K, cute::UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;

  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);
  auto mma_shape_a = partition_shape_A(
      tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto mma_shape_b = partition_shape_B(
      tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto layout_sA = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<uint8_t>{}, mma_shape_a);
  auto layout_sB = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<uint8_t>{}, mma_shape_b);
  using LayoutSFA = decltype(
      ScaleConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape{}));
  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));

  constexpr int kWeightK128Bytes = cosize_v<decltype(layout_sA)>;
  constexpr int kActivationK128Bytes = cosize_v<decltype(layout_sB)>;
  constexpr int kSfaK128Bytes = cosize_v<LayoutSFA>;
  constexpr int kSfbK128Bytes = cosize_v<LayoutSFB>;
  constexpr int kWeightStageBytes =
      kK128PerBundle * kWeightK128Bytes;
  constexpr int kActivationStageBytes =
      kK128PerBundle * kActivationK128Bytes;
  constexpr int kSfaStageBytes = kK128PerBundle * kSfaK128Bytes;
  constexpr int kSfbStageBytes = kK128PerBundle * kSfbK128Bytes;
  constexpr int kScaleStageBytes = kSfaStageBytes + kSfbStageBytes;
  static_assert(kWeightStageBytes == kBundleK * kTileM);
  static_assert(kActivationStageBytes == kBundleK * kTileN);
  static_assert(
      kSfaStageBytes == kK128PerBundle * 512 &&
      kSfbStageBytes == kK128PerBundle * 512);

  auto accumulator_shape = partition_shape_C(
      tiled_mma, make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto accumulator = TiledMma::make_fragment_C(accumulator_shape);
  auto tmem_sfa_probe = make_tensor<typename TiledMma::FrgTypeSFA>(
      shape(LayoutSFA{}));
  auto tmem_sfb_probe = make_tensor<typename TiledMma::FrgTypeSFB>(
      shape(LayoutSFB{}));
  const int accumulator_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(accumulator));
  const int sfa_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_sfa_probe));
  const int sfb_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_sfb_probe));
  accumulator.data() = tmem_base_ptr;
  constexpr int kUtccpColumns = 4;
  const int sfb_column_offset = mxfpGateUpPaddedTmemScale
      ? kUtccpColumns
      : sfa_columns;
  const int scale_tmem_subtile_columns = mxfpGateUpPaddedTmemScale
      ? 2 * kUtccpColumns
      : sfa_columns + sfb_columns;
  const int scale_tmem_stage_columns =
      scale_tmem_subtile_columns * kK128PerBundle;
  const uint32_t scale_tmem_base =
      tmem_base_ptr + accumulator_columns;
  if (accumulator_columns + kRingStages * scale_tmem_stage_columns >
      TmemColumns) {
    asm volatile("trap;");
  }

  const int tid = __compute_tid() - ThreadOffset;
  const int warp = tid / numThreadsPerWarp;
  const int lane = tid & (numThreadsPerWarp - 1);
  (void)c2m;
  (void)resident_task_index;

  const auto *weight_scale_global = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 0));
  const auto *activation_records_global = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 8));
  const uint64_t tma_info =
      *reinterpret_cast<const uint64_t *>(metadata + 24);
  const uint16_t weight_tma_index = uint16_t(tma_info);
  const uint16_t output_tma_index = uint16_t(tma_info >> 16);
  const uint32_t output_task = uint32_t(tma_info >> 32);
  const uint64_t barrier_info =
      *reinterpret_cast<const uint64_t *>(metadata + 32);
  const uint32_t ready_bar = uint32_t(barrier_info);
  const uint32_t reduce_bar = uint32_t(barrier_info >> 32);
  const float route_scale =
      *reinterpret_cast<const float *>(metadata + 40);
  const uint32_t resident_flags =
      *reinterpret_cast<const uint32_t *>(metadata + 68);
  const bool reduce_from_zero = (resident_flags & 1U) != 0;
  const int ready_bar_stride = (resident_flags & 2U) != 0 ? 8 : 1;
  const bool blockwise_ready = (resident_flags & 4U) != 0;
  auto *final_output_global = reinterpret_cast<Output *>(
      *reinterpret_cast<const uint64_t *>(metadata + 48));
  const uint64_t layout_info =
      *reinterpret_cast<const uint64_t *>(metadata + 56);
  const uint32_t weight_scale_tile_stride = uint32_t(layout_info) != 0
      ? uint32_t(layout_info)
      : uint32_t(kSfaStageBytes);
  const bool weight_k_tile_major = uint32_t(layout_info >> 32) != 0;
  const uint32_t k_start_tile =
      *reinterpret_cast<const uint32_t *>(metadata + 64);
  const int expert = int(output_task) / kDownTilesPerExpert;
  const int output_m_tile = int(output_task) % kDownTilesPerExpert;

  // The shared-expert CTA initializes its FP32 destination and publishes the
  // zero-ready edge used by all seven TMA reduce-add producers.
  if constexpr (!ResidentAllTma) {
    if (reduce_from_zero && expert == 0) {
      for (int index = tid; index < kOutputElements; index += 128) {
        final_output_global[index] = Output(0.0f);
      }
      __sync_barrier<SyncBarrierId, 128>();
      if (tid == 0) {
        asm volatile("fence.release.gpu;" ::: "memory");
        *reinterpret_cast<volatile int *>(global_bars + reduce_bar) = 0;
      }
      __sync_barrier<SyncBarrierId, 128>();
    }
  }

  if (!ResidentAllTma && !blockwise_ready && tid == 0 &&
      ready_bar != 0xFFFFFFFFU) {
    volatile int *ready = global_bars + ready_bar;
    bool pending = true;
    while (pending) {
      pending = false;
      #pragma unroll
      for (int slice = 0; slice < 16; ++slice) {
        pending |= ready[slice * ready_bar_stride] != 0;
      }
      if (pending) {
        __nanosleep(256);
      }
    }
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
  }
  __sync_barrier<SyncBarrierId, 128>();

  constexpr int kWeightRingBytes = kRingStages * kWeightStageBytes;
  constexpr int kActivationRingBytes =
      kRingStages * kActivationStageBytes;
  constexpr int kScaleRingBytes = kRingStages * kScaleStageBytes;
  constexpr int kLocalScaleRingBytes = kLduWeightScaleTma
      ? 0
      : kScaleRingBytes;
  constexpr int kLocalWeightRingBytes =
      UseLduWeightRing ? 0 : kWeightRingBytes;
  constexpr int kLocalBarrierArrays = UseLduWeightRing ? 2 : 4;
  constexpr int kBarrierBytes = kLocalBarrierArrays * kRingStages *
      int(sizeof(TxBarrier));
  constexpr int kOutputOffsetUnaligned =
      kLocalScaleRingBytes + kActivationRingBytes + kLocalWeightRingBytes +
      kBarrierBytes;
  constexpr int kOutputOffset =
      (kOutputOffsetUnaligned + 127) & ~127;
  constexpr int kFixedScratchBytes = kOutputOffset + kOutputBytes;
  constexpr int kTaskScratchBytes =
      ScratchCapacityBytes - ScratchOffsetBytes;
  static_assert(
      kTaskScratchBytes >= kFixedScratchBytes + 1023,
      "fixed MXFP4/MXFP8 down ring does not fit task-local shared memory");
  auto *fixed_base =
      static_cast<uint8_t *>(smem_base) + ScratchOffsetBytes;
  auto *local_scale_ring = fixed_base;
  auto *activation_ring = local_scale_ring + kLocalScaleRingBytes;
  auto *local_weight_ring = activation_ring + kActivationRingBytes;
  auto *local_barriers = reinterpret_cast<TxBarrier *>(
      local_weight_ring + kLocalWeightRingBytes);
  uint8_t *weight_ring;
  uint8_t *scale_ring;
  TxBarrier *weight_full;
  TxBarrier *weight_scale_full;
  TxBarrier *operand_full;
  TxBarrier *umma_full;
  TxBarrier *stage_empty;
  if constexpr (ResidentAllTma) {
    auto *resident_base = static_cast<uint8_t *>(smem_base);
    weight_ring = resident_base +
        dae_mxfp_resident_ffn::kDownWeightRingOffset;
    scale_ring = resident_base +
        dae_mxfp_resident_ffn::kDownScaleRingOffset;
    activation_ring = resident_base +
        dae_mxfp_resident_ffn::kDownActivationRingOffset;
    weight_full = reinterpret_cast<TxBarrier *>(
        resident_mma_barriers + mxfpResidentDownWeightFullBarrierBase);
    weight_scale_full = weight_full;
    operand_full = reinterpret_cast<TxBarrier *>(
        resident_mma_barriers + mxfpDownResidentOperandFullBarrierBase);
    umma_full = local_barriers;
    stage_empty = reinterpret_cast<TxBarrier *>(
        resident_mma_barriers + mxfpResidentDownEmptyBarrierBase);
  } else {
    weight_ring = local_weight_ring;
    scale_ring = local_scale_ring;
    weight_full = local_barriers;
    weight_scale_full = weight_full;
    operand_full = weight_full + kRingStages;
    umma_full = operand_full + kRingStages;
    stage_empty = umma_full + kRingStages;
  }
  auto *output_smem = reinterpret_cast<Output *>(
      ResidentAllTma
          ? static_cast<uint8_t *>(smem_base) +
              dae_mxfp_resident_ffn::kDownOutputOffset
          : fixed_base + kOutputOffset);

  if (warp == 1 && lane == 0) {
    #pragma unroll
    for (int stage = 0; stage < kRingStages; ++stage) {
      if constexpr (!UseLduWeightRing) {
        weight_full[stage].init(1);
        stage_empty[stage].init(1);
      }
      if constexpr (!ResidentAllTma) {
        operand_full[stage].init(kLduWeightScaleTma ? 1 : 2);
      }
      umma_full[stage].init(1);
    }
    cutlass::arch::fence_barrier_init();
  }
  __sync_barrier<SyncBarrierId, 128>();

  if (warp == 1) {
    if constexpr (!UseLduWeightRing) {
      if (elect_one_sync()) {
        #pragma unroll
        for (int tile = 0; tile < kNumKTiles; ++tile) {
          const int stage = tile % kRingStages;
          const int phase = (tile / kRingStages) & 1;
          if (tile >= kRingStages) {
            stage_empty[stage].wait(phase ^ 1);
          }
          const uint32_t destination = static_cast<uint32_t>(
              __cvta_generic_to_shared(
                  weight_ring + stage * kWeightStageBytes));
          const uint32_t barrier = static_cast<uint32_t>(
              __cvta_generic_to_shared(weight_full + stage));
          const int weight_coord3 =
              weight_k_tile_major ? int(output_task) : k_start_tile + tile;
          const int weight_coord4 =
              weight_k_tile_major ? k_start_tile + tile : int(output_task);
          asm volatile(
              "cp.async.bulk.tensor.5d.shared::cluster.global."
              "mbarrier::complete_tx::bytes "
              "[%0], [%1, {0, %2, %3, %4, %5}], [%6];"
              :: "r"(destination), "l"(tma_descs + weight_tma_index),
                 "r"(0), "r"(0), "r"(weight_coord3),
                 "r"(weight_coord4),
                 "r"(barrier)
              : "memory");
          weight_full[stage].arrive_and_expect_tx(kWeightPackedBytes);
        }
      }
    }
  } else if (warp == 2) {
    #pragma unroll
    for (int tile = 0; tile < kNumKTiles; ++tile) {
      const int stage = tile % kRingStages;
      const int phase = (tile / kRingStages) & 1;
      if constexpr (kRingStages == 1) {
        if (tile >= 1) {
          const int retire_phase = (tile - 1) & 1;
          umma_full[0].wait(retire_phase);
          if (lane == 0) {
            stage_empty[0].arrive();
          }
        }
      }
      if (tile >= kRingStages) {
        stage_empty[stage].wait(
            phase ^ (UseLduWeightRing ? 0 : 1));
      }
      if constexpr (!kLduWeightScaleTma) {
        dae_mxfp_cp_async_scale_stage<kSfaStageBytes>(
            weight_scale_global + tile * weight_scale_tile_stride,
            scale_ring + stage * kScaleStageBytes, lane);
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
        cutlass::arch::fence_view_async_shared();
        if (lane == 0) {
          operand_full[stage].arrive();
        }
      }

      // Keep two K bundles in flight. Retiring tile n-1 after publishing
      // tile n permits the issuer to submit the next bundle without waiting
      // for the preceding UMMA at the issue point.
      if constexpr (kRingStages > 1) {
        if (tile >= 1) {
          const int retire_tile = tile - 1;
          const int retire_stage = retire_tile % kRingStages;
          const int retire_phase = (retire_tile / kRingStages) & 1;
          umma_full[retire_stage].wait(retire_phase);
          if (lane == 0) {
            stage_empty[retire_stage].arrive();
          }
        }
      }
    }
    constexpr int kLastTile = kNumKTiles - 1;
    constexpr int kLastStage = kLastTile % kRingStages;
    constexpr int kLastPhase = (kLastTile / kRingStages) & 1;
    umma_full[kLastStage].wait(kLastPhase);
    if (lane == 0) {
      stage_empty[kLastStage].arrive();
    }
  } else if (warp == 3) {
    if constexpr (!ResidentAllTma) {
      #pragma unroll
      for (int tile = 0; tile < kNumKTiles; ++tile) {
        const int stage = tile % kRingStages;
        const int phase = (tile / kRingStages) & 1;
        if (tile >= kRingStages) {
          stage_empty[stage].wait(
              phase ^ (UseLduWeightRing ? 0 : 1));
        }
        if (blockwise_ready && ready_bar != 0xFFFFFFFFU) {
          if (lane == 0) {
            volatile int *ready = global_bars + ready_bar;
            bool pending = true;
            while (pending) {
              pending = false;
              #pragma unroll
              for (int subtile = 0; subtile < kK128PerBundle; ++subtile) {
                const int record =
                    (k_start_tile + tile) * kK128PerBundle + subtile;
                pending |= ready[record * ready_bar_stride] != 0;
              }
              if (pending) {
                __nanosleep(256);
              }
            }
            asm volatile("fence.acquire.gpu;" ::: "memory");
          }
          __syncwarp();
        }
        #pragma unroll
        for (int subtile = 0; subtile < kK128PerBundle; ++subtile) {
          const auto *record = activation_records_global +
              ((k_start_tile + tile) * kK128PerBundle + subtile) *
                  kActivationRecordBytes;
          dae_mxfp_cp_async_scale_stage<kActivationRecordDataBytes>(
              record,
              activation_ring + stage * kActivationStageBytes +
                  subtile * kActivationK128Bytes,
              lane);
          dae_mxfp_cp_async_scale_stage<kActivationRecordScaleBytes>(
              record + kActivationRecordDataBytes,
              scale_ring + stage * kScaleStageBytes + kSfaStageBytes +
                  subtile * kSfbK128Bytes,
              lane);
        }
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncwarp();
        cutlass::arch::fence_view_async_shared();
        if (lane == 0) {
          operand_full[stage].arrive();
        }
      }
    }
  } else if (warp == 0) {
    using Utccp = SM100_UTCCP_4x32dp128bit_1cta;
    #pragma unroll
    for (int tile = 0; tile < kNumKTiles; ++tile) {
      const int stage = tile % kRingStages;
      const int phase = (tile / kRingStages) & 1;
      if constexpr (kSeparateWeightScaleBarrier) {
        weight_scale_full[stage].wait(phase);
        operand_full[stage].wait(phase);
        #pragma unroll
        for (int subtile = 0; subtile < kK128PerBundle; ++subtile) {
          const uint32_t scale_subtile_base =
              scale_tmem_base + stage * scale_tmem_stage_columns +
              subtile * scale_tmem_subtile_columns;
          dae_mxfp_copy_scale_subtile_to_tmem<
              kSfaStageBytes, kSfaK128Bytes, kSfbK128Bytes,
              Utccp, Scale, typename TiledMma::FrgTypeSFA,
              typename TiledMma::FrgTypeSFB, LayoutSFA, LayoutSFB>(
                  scale_ring + stage * kScaleStageBytes,
                  subtile, scale_subtile_base, sfb_column_offset);
        }
        weight_full[stage].wait(phase);
      } else {
        weight_full[stage].wait(phase);
        operand_full[stage].wait(phase);
      }
      #pragma unroll
      for (int subtile = 0; subtile < kK128PerBundle; ++subtile) {
        const uint32_t scale_subtile_base =
            scale_tmem_base + stage * scale_tmem_stage_columns +
            subtile * scale_tmem_subtile_columns;
        auto stage_tmem_sfa =
            make_tensor<typename TiledMma::FrgTypeSFA>(shape(LayoutSFA{}));
        auto stage_tmem_sfb =
            make_tensor<typename TiledMma::FrgTypeSFB>(shape(LayoutSFB{}));
        stage_tmem_sfa.data() = scale_subtile_base;
        stage_tmem_sfb.data() = scale_subtile_base + sfb_column_offset;
        auto sA = make_tensor(
            make_smem_ptr(reinterpret_cast<uint8_t *>(
                weight_ring + stage * kWeightStageBytes +
                subtile * kWeightK128Bytes)),
            layout_sA);
        auto sB = make_tensor(
            make_smem_ptr(reinterpret_cast<Activation *>(
                activation_ring + stage * kActivationStageBytes +
                subtile * kActivationK128Bytes)),
            layout_sB);
        auto frag_a = TiledMma::make_fragment_A(sA);
        auto frag_b = TiledMma::make_fragment_B(sB);

        if constexpr (!kSeparateWeightScaleBarrier) {
          dae_mxfp_copy_scale_subtile_to_tmem<
              kSfaStageBytes, kSfaK128Bytes, kSfbK128Bytes,
              Utccp, Scale, typename TiledMma::FrgTypeSFA,
              typename TiledMma::FrgTypeSFB, LayoutSFA, LayoutSFB>(
                  scale_ring + stage * kScaleStageBytes,
                  subtile, scale_subtile_base, sfb_column_offset);
        }

        #pragma unroll
        for (int k_block = 0; k_block < size<2>(frag_a); ++k_block) {
          const auto accumulate =
              tile == 0 && subtile == 0 && k_block == 0
              ? UMMA::ScaleOut::Zero
              : UMMA::ScaleOut::One;
          dae_mxfp_gate_up_issue_raw_umma<
              Weight, Activation, Accum, Scale, kTileM, kTileN>(
                  frag_a(_, _, k_block), frag_b(_, _, k_block), accumulator,
                  accumulate, stage_tmem_sfa(_, _, k_block),
                  stage_tmem_sfb(_, _, k_block));
        }
      }
      cutlass::arch::umma_arrive(
          reinterpret_cast<uint64_t *>(umma_full + stage));
    }
  }

  constexpr int kLastTile = kNumKTiles - 1;
  constexpr int kLastStage = kLastTile % kRingStages;
  constexpr int kLastPhase = (kLastTile / kRingStages) & 1;
  if constexpr (!UseLduWeightRing) {
    if (warp == 1) {
      stage_empty[kLastStage].wait(kLastPhase);
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_barrier<SyncBarrierId, 128>();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  auto coord_c = make_identity_tensor(
      make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto cta_coord_c = cta_mma.partition_C(coord_c);
  using TmemLoad = SM100_TMEM_LOAD_32dp32b1x;
  auto accumulator_tmem = accumulator(make_coord(_, _), _0{}, _0{});
  auto c_acc = cta_coord_c(make_coord(_, _), _0{}, _0{});
  auto tiled_t2r = make_tmem_copy(TmemLoad{}, accumulator_tmem);
  if (tid < size(tiled_t2r)) {
    auto thread_t2r = tiled_t2r.get_slice(tid);
    auto thread_tmem = thread_t2r.partition_S(accumulator_tmem);
    auto thread_coord = thread_t2r.partition_D(c_acc);
    auto registers = make_tensor<Accum>(shape(thread_coord));
    copy(tiled_t2r, thread_tmem, registers);
    cutlass::arch::fence_view_async_tmem_load();
    #pragma unroll
    for (int index = 0; index < size(registers); ++index) {
      const int row = int(get<0>(thread_coord(index)));
      const int column = int(get<1>(thread_coord(index)));
      if (row < kTileM && column < kNativeOutputRows) {
        output_smem[row * kNativeOutputRows + column] =
            Output(registers(index) * route_scale);
      }
    }
  }
  __sync_barrier<SyncBarrierId, 128>();

  if (tid == 0 && (reduce_from_zero || expert != 0)) {
    if constexpr (ResidentAllTma) {
      auto *reduction_ready = reinterpret_cast<TxBarrier *>(
          resident_mma_barriers +
          mxfpDownResidentReductionReadyBarrierBase);
      reduction_ready[output_m_tile >= kDownTilesPerExpert / 2].wait(0);
    } else {
      cuda::atomic_ref<int, cuda::thread_scope_device> shared_ready(
          global_bars[reduce_bar]);
      while (shared_ready.load(cuda::memory_order_acquire) != 0) {
        __nanosleep(128);
      }
    }
  }
  if constexpr (!ResidentAllTma) {
    __sync_barrier<SyncBarrierId, 128>();
  }

  // The shared expert establishes the BF16 destination with a bulk copy; the
  // routed experts reduce-add into the Python-initialized destination.
  if (tid == 0) {
      const uint32_t source = uint32_t(__cvta_generic_to_shared(output_smem));
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
            global_bars[reduce_bar]);
        shared_ready.fetch_sub(1, cuda::memory_order_release);
      }
  }
  __sync_barrier<SyncBarrierId, 128>();
}
