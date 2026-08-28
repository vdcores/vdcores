#pragma once

#include "mxfp_resident_ffn.cuh"
#include "mxfp4_mxfp8_umma.cuh"

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>
#include <cutlass/numeric_types.h>
#include <cuda/atomic>
#include <cuda_bf16.h>

#include <type_traits>

// Finalized task-owned and resident FFN layout. Superseded tuning variants
// remain available in commit 79022cc for archaeology only.
static constexpr bool mxfpGateUpRawUmma = true;
static constexpr bool mxfpGateUpPaddedTmemScale = false;
static constexpr bool mxfpGateUpFixedBulkScale = false;
static constexpr bool mxfpGateUpSubtileScaleSlots = true;
static constexpr bool mxfpGateUpDirectOutput = true;
static constexpr bool mxfpGateUpDirectActivationEnabled = true;
static constexpr int mxfpGateUpDirectActivationTiles = 8;
static constexpr int mxfpGateUpFixedOutputRows = 8;
static constexpr bool mxfpGateUpFixedBf16Epilogue = false;

template <
    int SfaStageBytes, int SfaK128Bytes, int SfbK128Bytes,
    class Utccp, class Scale, class FrgSFA, class FrgSFB,
    class LayoutSFA, class LayoutSFB>
__device__ __forceinline__ void dae_mxfp_copy_scale_subtile_to_tmem(
    uint8_t *stage_scale, int subtile, uint32_t scale_subtile_base,
    int sfb_column_offset) {
  using namespace cute;
  auto stage_tmem_sfa = make_tensor<FrgSFA>(shape(LayoutSFA{}));
  auto stage_tmem_sfb = make_tensor<FrgSFB>(shape(LayoutSFB{}));
  stage_tmem_sfa.data() = scale_subtile_base;
  stage_tmem_sfb.data() = scale_subtile_base + sfb_column_offset;
  if (elect_one_sync()) {
    auto compact_sfa = make_tensor(
        stage_tmem_sfa.data(), filter_zeros(stage_tmem_sfa.layout()));
    auto compact_sfb = make_tensor(
        stage_tmem_sfb.data(), filter_zeros(stage_tmem_sfb.layout()));
    auto copy_sfa = make_utccp_copy(Utccp{}, compact_sfa);
    auto copy_sfb = make_utccp_copy(Utccp{}, compact_sfb);
    auto copy_sfa_slice = copy_sfa.get_slice(0);
    auto copy_sfb_slice = copy_sfb.get_slice(0);
    auto smem_sfa = make_tensor(
        make_smem_ptr(reinterpret_cast<Scale *>(
            stage_scale + subtile * SfaK128Bytes)),
        LayoutSFA{});
    auto smem_sfb = make_tensor(
        make_smem_ptr(reinterpret_cast<Scale *>(
            stage_scale + SfaStageBytes + subtile * SfbK128Bytes)),
        LayoutSFB{});
    auto smem_sfa_compact = make_tensor(
        smem_sfa.data(), filter_zeros(smem_sfa.layout()));
    auto smem_sfb_compact = make_tensor(
        smem_sfb.data(), filter_zeros(smem_sfb.layout()));
    auto copy_sfa_src = dae_mxfp_get_utccp_smem_desc_tensor<Utccp>(
        copy_sfa_slice.partition_S(smem_sfa_compact));
    auto copy_sfb_src = dae_mxfp_get_utccp_smem_desc_tensor<Utccp>(
        copy_sfb_slice.partition_S(smem_sfb_compact));
    copy(copy_sfa, copy_sfa_src, copy_sfa_slice.partition_D(compact_sfa));
    copy(copy_sfb, copy_sfb_src, copy_sfb_slice.partition_D(compact_sfb));
  }
}

template <
    int OutputRows, class GateSilu, class TiledT2R,
    class GateTmem, class CoordTensor>
__device__ __forceinline__ void dae_mxfp_gate_silu_registers(
    int tid,
    const TiledT2R &tiled_t2r,
    const GateTmem &gate_tmem,
    const CoordTensor &c_acc,
    GateSilu (&gate_silu)[OutputRows]) {
  using namespace cute;
  auto thread_t2r = tiled_t2r.get_slice(tid);
  auto thread_tmem = thread_t2r.partition_S(gate_tmem);
  auto thread_coord = thread_t2r.partition_D(c_acc);
  auto registers = make_tensor<float>(shape(thread_coord));
  copy(tiled_t2r, thread_tmem, registers);
  cutlass::arch::fence_view_async_tmem_load();
  #pragma unroll
  for (int index = 0; index < size(registers); ++index) {
    const int row = int(get<0>(thread_coord(index)));
    const int column = int(get<1>(thread_coord(index)));
    if (row < 128 && column < OutputRows) {
      // DeepSeek-V4's checkpoint contract uses bounded SwiGLU.  Clamp while
      // the accumulator is still FP32 so the native MXFP8 record matches the
      // offline/PyTorch path without another task or shared-memory pass.
      const float gate = fminf(registers(index), 10.0f);
      if constexpr (std::is_same_v<GateSilu, __nv_bfloat16>) {
        const float rounded_gate =
            __bfloat162float(__float2bfloat16_rn(gate));
        gate_silu[column] = __float2bfloat16_rn(
            rounded_gate / (1.0f + __expf(-rounded_gate)));
      } else {
        gate_silu[column] = gate / (1.0f + __expf(-gate));
      }
    }
  }
}

template <
    class Weight, class Activation, class Accum, class Scale,
    int TileM, int TileN,
    class FragA, class FragB, class AccTensor,
    class SfaTensor, class SfbTensor>
__device__ __forceinline__ void dae_mxfp_gate_up_issue_raw_umma(
    const FragA &frag_a,
    const FragB &frag_b,
    AccTensor &accumulator,
    cute::UMMA::ScaleOut accumulate,
    const SfaTensor &sfa,
    const SfbTensor &sfb) {
  const uint64_t desc_a = frag_a[0];
  const uint64_t desc_b = frag_b[0];
  const uint32_t tmem_c = cute::raw_pointer_cast(accumulator.data());
  const uint32_t tmem_sfa = cute::raw_pointer_cast(sfa.data());
  const uint32_t tmem_sfb = cute::raw_pointer_cast(sfb.data());
  const uint64_t instruction =
      cute::UMMA::make_runtime_instr_desc_block_scaled<
          Weight, Activation, Accum, Scale,
          TileM, TileN, cute::UMMA::Major::K, cute::UMMA::Major::K>(
              tmem_sfa, tmem_sfb);
  cute::SM100_MMA_MXF8F6F4_SS<
      Weight, Activation, Accum, Scale, TileM, TileN,
      cute::UMMA::Major::K, cute::UMMA::Major::K>::fma(
          desc_a, desc_b, tmem_c, uint32_t(accumulate), instruction,
          tmem_sfa, tmem_sfb);
}

// Task-specialized Linear-1 mainloop. Activation crosses M2C/C2M while the
// default packed-weight ring is allocated once and retained by LDU0 across
// gate and up. The task-owned producer remains as an A/B control. Warp 0 is
// the raw UMMA issuer and warps 2/3 produce native scales.
// Gate is completed first and remains in TMEM while most of the up stream is
// prepared. Each warp converts its gate fragment to lane-local SiLU registers
// late enough to overlap the remaining up work without a shared-memory image.
template <
    int K, int RingStages, int TileN, bool PublishReady,
    bool UseLduWeightRing, bool ResidentAllTma,
    typename M2CQueue, typename C2MQueue>
__device__ __noinline__ void
task_mxfp4_mxfp8_gate_up_silu_fixed_ring_sm100(
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barriers,
    const CUtensorMap *tma_descs,
    const uint8_t *metadata,
    int *global_bars,
    M2CQueue &m2c,
    C2MQueue &c2m
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
    , const int detail_sm_id, uint64_t *detail_events
#endif
    ) {
  using namespace cute;
  using Weight = cutlass::detail::float_e2m1_unpacksmem_t;
  using Activation = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Accum = float;
  using TxBarrier = cutlass::arch::ClusterTransactionBarrier;

  static_assert(K == 128 || K == 512, "fixed-ring K must be 128 or 512");
  static_assert(
      (K == 128 && (RingStages == 10 || RingStages == 11)) ||
          (K == 512 && (RingStages == 2 || RingStages == 3)),
      "fixed-ring stage count does not match streamed K");
  static_assert(
      !UseLduWeightRing || (K == 512 && RingStages == 2),
      "retained LDU weight ring requires the K512 two-stage task");
  static_assert(
      UseLduWeightRing == ResidentAllTma,
      "LDU-owned Linear-1 storage is reserved for the resident FFN");
  constexpr bool kLduWeightScaleTma = ResidentAllTma;
  constexpr bool kSeparateWeightScaleBarrier = false;
  constexpr bool kStreamDirectActivation =
      !ResidentAllTma && mxfpGateUpDirectActivationEnabled &&
      mxfpGateUpDirectActivationTiles == 1;
  static_assert(
      !kStreamDirectActivation || (K == 512 && RingStages == 3),
      "streamed direct activation requires the K512 three-stage image");
  constexpr int kTileM = 128;
  constexpr int kTileN = TileN;
  constexpr int kNativeActivationRows = 8;
  static_assert(kTileN == 8 || kTileN == 16);
  static_assert(
      !kStreamDirectActivation || kTileN == kNativeActivationRows,
      "streamed activation currently requires the native N8 UMMA tile");
  constexpr int kTileK = 128;
  constexpr int kK128PerTile = K / kTileK;
  constexpr int kNumKTiles = 4096 / K;
  constexpr int kScaleVector = 32;
  constexpr int kWeightPackedBytes = kTileM * K / 2;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  uint64_t detail_weight_full_wait_ns = 0;
  uint64_t detail_umma_full_wait_ns = 0;
#endif

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
  constexpr int kWeightStageBytes = kK128PerTile * kWeightK128Bytes;
  constexpr int kActivationStageBytes =
      kK128PerTile * kActivationK128Bytes;
  constexpr int kActivationGlobalStageBytes =
      K * kNativeActivationRows;
  constexpr int kSfaStageBytes = kK128PerTile * kSfaK128Bytes;
  constexpr int kSfbStageBytes = kK128PerTile * kSfbK128Bytes;
  constexpr int kScaleStageBytes = kSfaStageBytes + kSfbStageBytes;
  constexpr int kActivationBytes = kNumKTiles * kActivationStageBytes;
  constexpr int kActivationGlobalBytes =
      kNumKTiles * kActivationGlobalStageBytes;
  static_assert(kActivationBytes == kTileN * 4096);
  static_assert(kActivationGlobalBytes == 32 * 1024);
  static_assert(kWeightStageBytes == K * kTileM);
  static_assert(kActivationStageBytes == K * kTileN);
  static_assert(kSfaStageBytes == K * 4 && kSfbStageBytes == K * 4);

  auto logical_c = make_tensor(
      make_smem_ptr(static_cast<Accum *>(nullptr)),
      make_layout(
          make_shape(Int<kTileM>{}, Int<kTileN>{}),
          make_stride(Int<kTileN>{}, Int<1>{})));
  auto cta_c = cta_mma.partition_C(logical_c);
  auto gate_acc = cta_mma.make_fragment_C(cta_c);
  auto up_acc = cta_mma.make_fragment_C(cta_c);
  auto tmem_sfa_probe = make_tensor<typename TiledMma::FrgTypeSFA>(
      shape(LayoutSFA{}));
  auto tmem_sfb_probe = make_tensor<typename TiledMma::FrgTypeSFB>(
      shape(LayoutSFB{}));
  const int accumulator_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(gate_acc));
  const int sfa_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_sfa_probe));
  const int sfb_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_sfb_probe));
  gate_acc.data() = tmem_base_ptr;
  up_acc.data() = tmem_base_ptr + accumulator_columns;
  // Each in-flight SMEM stage owns a matching pair of TMEM scale slots. Reusing
  // one pair immediately after issue races the asynchronous UMMA scale read;
  // stage_empty protects both the SMEM image and these TMEM columns until the
  // corresponding UMMA completion. tcgen05.cp ... warpx4 destinations are
  // four-column aligned even though CUTE reports one logical scale column.
  constexpr int kUtccpColumns = 4;
  const int sfb_column_offset = mxfpGateUpPaddedTmemScale
      ? kUtccpColumns
      : sfa_columns;
  const int scale_tmem_subtile_columns = mxfpGateUpPaddedTmemScale
      ? 2 * kUtccpColumns
      : sfa_columns + sfb_columns;
  const int scale_tmem_stage_columns = scale_tmem_subtile_columns *
      (mxfpGateUpSubtileScaleSlots ? kK128PerTile : 1);
  const uint32_t scale_tmem_base =
      tmem_base_ptr + 2 * accumulator_columns;
  if (2 * accumulator_columns + RingStages * scale_tmem_stage_columns >
      cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns) {
    asm volatile("trap;");
  }

  const int tid = __compute_tid();
  const int warp = tid / numThreadsPerWarp;
  const int lane = tid & (numThreadsPerWarp - 1);
  int activation_slots = 0;
  uint8_t *activation_base = static_cast<uint8_t *>(smem_base);
  if constexpr (ResidentAllTma) {
    activation_base = static_cast<uint8_t *>(smem_base) +
        dae_mxfp_resident_ffn::kLinear1ActivationOffset;
  } else if constexpr (!mxfpGateUpDirectActivationEnabled) {
    activation_base = nullptr;
    if (warp < 2) {
      activation_slots = m2c.template pop<0>();
      if (warp == 0) {
        activation_base = static_cast<uint8_t *>(
            get_slot_address(smem_base, extract(activation_slots)));
      }
    } else {
      m2c.advance();
    }
  }

  uint8_t *ldu_weight_ring = nullptr;
  if constexpr (ResidentAllTma) {
    ldu_weight_ring = static_cast<uint8_t *>(smem_base) +
        dae_mxfp_resident_ffn::kLinear1WeightRingOffset;
  } else if constexpr (UseLduWeightRing) {
    if constexpr (kLduWeightScaleTma) {
      const int weight_slots = m2c.template pop<0>();
      ldu_weight_ring = static_cast<uint8_t *>(
          get_slot_address(smem_base, extract(weight_slots)));
    } else {
      if (warp == 0) {
        const int weight_slots = m2c.template pop<0>();
        ldu_weight_ring = static_cast<uint8_t *>(
            get_slot_address(smem_base, extract(weight_slots)));
      } else {
        m2c.advance();
      }
    }
  }

  const auto *activation_data_global = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 0));
  const uint64_t tma_info =
      *reinterpret_cast<const uint64_t *>(metadata + 40);
  const uint16_t gate_tma_index = uint16_t(tma_info);
  const uint16_t up_tma_index = uint16_t(tma_info >> 16);
  const uint32_t output_tile = uint32_t(tma_info >> 32);
  const auto *gate_scale_global = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 16));
  const auto *activation_scale_global = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 24));
  const auto *up_scale_global = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 32));
  auto *direct_output_global = reinterpret_cast<uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 48));
  const uint64_t layout_info =
      *reinterpret_cast<const uint64_t *>(metadata + 56);
  const uint32_t weight_scale_tile_stride = uint32_t(layout_info) != 0
      ? uint32_t(layout_info)
      : uint32_t(kSfaStageBytes);
  const bool weight_k_tile_major = uint32_t(layout_info >> 32) != 0;
  const uint32_t ready_bar =
      *reinterpret_cast<const uint32_t *>(metadata + 64);

  constexpr int kTaskWeightRingBytes = UseLduWeightRing
      ? 0
      : RingStages * kWeightStageBytes;
  constexpr int kScaleRingBytes = RingStages * kScaleStageBytes;
  constexpr int kTaskScaleRingBytes = kLduWeightScaleTma
      ? 0
      : kScaleRingBytes;
  constexpr int kActivationRingBytes = kStreamDirectActivation
      ? RingStages * kActivationStageBytes
      : 0;
  constexpr int kBarrierBytes =
      ((UseLduWeightRing ? 2 : 4) * RingStages +
       (kStreamDirectActivation ? 0 : 1) +
       2) *
      int(sizeof(TxBarrier));
  constexpr int kOutputRows = mxfpGateUpFixedOutputRows;
  using GateSilu = std::conditional_t<
      mxfpGateUpFixedBf16Epilogue, __nv_bfloat16, float>;
  constexpr int kFixedScratchBytes =
      kTaskWeightRingBytes + kTaskScaleRingBytes + kActivationRingBytes +
      kBarrierBytes;
  constexpr int kAllocatorArenaBytes = numSlots * slotSizeKb * 1024;
  constexpr int kFixedOffset = kStreamDirectActivation
      ? 0
      : (mxfpGateUpDirectActivationEnabled &&
                 kActivationBytes > kAllocatorArenaBytes
             ? kActivationBytes
             : kAllocatorArenaBytes);
  constexpr int kTaskScratchBytes =
      dynamicSmemBytes - kFixedOffset;
  static_assert(
      kTaskScratchBytes >= kFixedScratchBytes + 1023,
      "fixed weight ring does not fit behind the configured allocator arena");
  auto *fixed_base = kStreamDirectActivation
      ? static_cast<uint8_t *>(smem_base)
      : static_cast<uint8_t *>(smem_base) + kFixedOffset;
  // Without LDU scale TMA, keep the compact task-owned scale ring first. In
  // combined mode it instead occupies the final slot of the retained lease.
  auto *task_scale_ring = fixed_base;
  auto *activation_ring = task_scale_ring + kTaskScaleRingBytes;
  auto *task_weight_ring = activation_ring + kActivationRingBytes;
  auto *local_barrier_base = task_weight_ring + kTaskWeightRingBytes;
  auto *weight_ring = UseLduWeightRing
      ? ldu_weight_ring
      : task_weight_ring;
  auto *scale_ring = ResidentAllTma
      ? static_cast<uint8_t *>(smem_base) +
          dae_mxfp_resident_ffn::kLinear1ScaleRingOffset
      : (kLduWeightScaleTma
          ? ldu_weight_ring + RingStages * kWeightStageBytes
          : task_scale_ring);
  TxBarrier *weight_full;
  TxBarrier *weight_scale_full;
  TxBarrier *scale_full;
  TxBarrier *umma_full;
  TxBarrier *stage_empty;
  TxBarrier *activation_full;
  TxBarrier *projection_ready;
  if constexpr (UseLduWeightRing) {
    weight_full = reinterpret_cast<TxBarrier *>(
        tmem_mma_barriers + mxfpResidentLinear1FullBarrierBase);
    weight_scale_full = weight_full;
    stage_empty = reinterpret_cast<TxBarrier *>(
        tmem_mma_barriers + mxfpResidentLinear1EmptyBarrierBase);
    scale_full = reinterpret_cast<TxBarrier *>(local_barrier_base);
    umma_full = scale_full + RingStages;
    activation_full = umma_full + RingStages;
  } else {
    weight_full = reinterpret_cast<TxBarrier *>(local_barrier_base);
    weight_scale_full = weight_full;
    scale_full = weight_full + RingStages;
    umma_full = scale_full + RingStages;
    stage_empty = umma_full + RingStages;
    activation_full = stage_empty + RingStages;
  }
  // Persistent full/empty phases can wrap an even number of times before a
  // late gate-register reader joins. Task-local one-shot tokens provide an
  // unambiguous completion frontier for each projection.
  projection_ready = activation_full +
      (kStreamDirectActivation ? 0 : 1);
  GateSilu register_gate_silu[kOutputRows];

  if (warp == 1 && lane == 0) {
    #pragma unroll
    for (int stage = 0; stage < RingStages; ++stage) {
      if constexpr (!UseLduWeightRing) {
        weight_full[stage].init(1);
        stage_empty[stage].init(1);
      }
      scale_full[stage].init(
          kLduWeightScaleTma ? 1 :
          (mxfpGateUpFixedBulkScale ? 1 : 2));
      umma_full[stage].init(1);
    }
    if constexpr (!kStreamDirectActivation) {
      activation_full->init(1);
    }
    projection_ready[0].init(1);
    projection_ready[1].init(1);
    cutlass::arch::fence_barrier_init();
  }
  __sync_compute_group(128);

  if constexpr (
      !ResidentAllTma && mxfpGateUpDirectActivationEnabled &&
      !kStreamDirectActivation) {
    if (warp == 1 && lane == 0) {
      activation_full->arrive_and_expect_tx(kActivationGlobalBytes);
      if constexpr (kTileN == kNativeActivationRows) {
        #pragma unroll
        for (int chunk = 0; chunk < kActivationBytes / (16 * 1024); ++chunk) {
          cuda::ptx::cp_async_bulk(
              cuda::ptx::space_shared,
              cuda::ptx::space_global,
              activation_base + chunk * 16 * 1024,
              activation_data_global + chunk * 16 * 1024,
              uint32_t(16 * 1024),
              reinterpret_cast<uint64_t *>(activation_full));
        }
      } else {
        #pragma unroll
        for (int tile = 0; tile < kNumKTiles; ++tile) {
          #pragma unroll
          for (int subtile = 0; subtile < kK128PerTile; ++subtile) {
            constexpr int kNativeActivationK128Bytes =
                kNativeActivationRows * kTileK;
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                activation_base + tile * kActivationStageBytes +
                    subtile * kActivationK128Bytes,
                activation_data_global + tile * kActivationGlobalStageBytes +
                    subtile * kNativeActivationK128Bytes,
                uint32_t(kNativeActivationK128Bytes),
                reinterpret_cast<uint64_t *>(activation_full));
          }
        }
      }
    }
    if (warp == 0) {
      activation_full->wait(0);
    }
  }

  using Utccp = SM100_UTCCP_4x32dp128bit_1cta;
  auto coord_c = make_identity_tensor(
      make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto cta_coord_c = cta_mma.partition_C(coord_c);
  using TmemLoad = SM100_TMEM_LOAD_32dp32b1x;
  auto gate_tmem = gate_acc(make_coord(_, _), _0{}, _0{});
  auto up_tmem = up_acc(make_coord(_, _), _0{}, _0{});
  auto c_acc = cta_coord_c(make_coord(_, _), _0{}, _0{});
  auto tiled_t2r = make_tmem_copy(TmemLoad{}, gate_tmem);

  // Keep the expanded gate/up schedule behind a device-call boundary so it
  // does not inflate the persistent dispatch handler's instruction footprint.
  #pragma unroll 2
  for (int projection = 0; projection < 2; ++projection) {
    const uint16_t descriptor_index =
        projection == 0 ? gate_tma_index : up_tma_index;
    const uint8_t *weight_scale_global =
        projection == 0 ? gate_scale_global : up_scale_global;

    if (warp == 1) {
      if constexpr (!UseLduWeightRing) {
        if (elect_one_sync()) {
          #pragma unroll 4
          for (int tile = 0; tile < kNumKTiles; ++tile) {
            const int operation = projection * kNumKTiles + tile;
            const int stage = operation % RingStages;
            const int phase = (operation / RingStages) & 1;
            if (operation >= RingStages) {
              stage_empty[stage].wait(phase ^ 1);
            }
            const uint32_t destination = static_cast<uint32_t>(
                __cvta_generic_to_shared(
                    weight_ring + stage * kWeightStageBytes));
            const uint32_t barrier = static_cast<uint32_t>(
                __cvta_generic_to_shared(weight_full + stage));
            const int weight_coord3 =
                weight_k_tile_major ? int(output_tile) : tile;
            const int weight_coord4 =
                weight_k_tile_major ? tile : int(output_tile);
            asm volatile(
                "cp.async.bulk.tensor.5d.shared::cluster.global."
                "mbarrier::complete_tx::bytes "
                "[%0], [%1, {0, %2, %3, %4, %5}], [%6];"
                :: "r"(destination), "l"(tma_descs + descriptor_index),
                   "r"(0), "r"(0), "r"(weight_coord3), "r"(weight_coord4),
                   "r"(barrier)
                : "memory");
            if constexpr (kStreamDirectActivation) {
              cuda::ptx::cp_async_bulk(
                  cuda::ptx::space_shared,
                  cuda::ptx::space_global,
                  activation_ring + stage * kActivationStageBytes,
                  activation_data_global + tile * kActivationGlobalStageBytes,
                  uint32_t(kActivationGlobalStageBytes),
                  reinterpret_cast<uint64_t *>(weight_full + stage));
            }
            // TMA accounting follows packed HBM bytes even though the 16U4
            // transform expands the shared-memory image.
            weight_full[stage].arrive_and_expect_tx(
                kWeightPackedBytes +
                (kStreamDirectActivation ? kActivationGlobalStageBytes : 0));
          }
        }
      }
      if (projection == 1) {
        projection_ready[0].wait(0);
        // By the time the up-weight ring is full, reusing its two physical
        // stages has retired the complete gate stream. Drain warp 1's TMEM
        // datapath while warp 0 consumes the prefetched up weights.
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        dae_mxfp_gate_silu_registers<kOutputRows>(
            tid, tiled_t2r, gate_tmem, c_acc, register_gate_silu);
      }
    } else if (warp >= 2) {
      #pragma unroll 2
      for (int tile = 0; tile < kNumKTiles; ++tile) {
        const int operation = projection * kNumKTiles + tile;
        const int stage = operation % RingStages;
        const int phase = (operation / RingStages) & 1;
        if (operation >= RingStages) {
          stage_empty[stage].wait(
              phase ^ (UseLduWeightRing ? 0 : 1));
        }
        auto *stage_scale = scale_ring + stage * kScaleStageBytes;
        if constexpr (ResidentAllTma) {
          // LDU0 attached SFA and SFB to the same resident transaction as
          // the transformed weight. These warps only retire completed UMMA
          // stages; they submit no memory operations in this path.
        } else if constexpr (kLduWeightScaleTma) {
          // LDU0 attached SFA to the resident weight-full transaction. Keep
          // only producer-dependent activation scale on the compute side.
          if (warp == 3) {
            dae_mxfp_cp_async_scale_stage<kSfbStageBytes>(
                activation_scale_global + tile * kSfbStageBytes,
                stage_scale + kSfaStageBytes, lane);
            asm volatile("cp.async.wait_group 0;" ::: "memory");
            __syncwarp();
            cutlass::arch::fence_view_async_shared();
            if (lane == 0) {
              scale_full[stage].arrive();
            }
          }
        } else if constexpr (mxfpGateUpFixedBulkScale) {
          // One producer submits both scale records to the bulk engine. The
          // transaction barrier flips only after all 4 KiB are visible, so
          // warp 0 can UTCCP directly without a producer-side wait. Warp 3 is
          // left free to reach the gate-SiLU frontier early.
          if (warp == 2 && lane == 0) {
            scale_full[stage].arrive_and_expect_tx(kScaleStageBytes);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                stage_scale,
                weight_scale_global + tile * weight_scale_tile_stride,
                uint32_t(kSfaStageBytes),
                reinterpret_cast<uint64_t *>(scale_full + stage));
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                stage_scale + kSfaStageBytes,
                activation_scale_global + tile * kSfbStageBytes,
                uint32_t(kSfbStageBytes),
                reinterpret_cast<uint64_t *>(scale_full + stage));
          }
        } else {
          const uint8_t *source = warp == 2
              ? weight_scale_global + tile * weight_scale_tile_stride
              : activation_scale_global + tile * kSfbStageBytes;
          auto *destination = warp == 2
              ? stage_scale
              : stage_scale + kSfaStageBytes;
          dae_mxfp_cp_async_scale_stage<kSfaStageBytes>(
              source, destination, lane);
          asm volatile("cp.async.wait_group 0;" ::: "memory");
          __syncwarp();
          cutlass::arch::fence_view_async_shared();
          if (lane == 0) {
            scale_full[stage].arrive();
          }
        }
        // Warp 2 retires a short UMMA window. This is a conventional
        // full/empty pipeline: tcgen completion flips umma_full, then the
        // retire warp releases the SMEM/TMEM stage through stage_empty.
        // Keeping four tiles live preserves asynchronous issue while avoiding
        // a dedicated fifth compute warp.
        constexpr int kRetireLag = RingStages > 4 ? 4 : RingStages - 1;
        if (warp == 2 && tile >= kRetireLag) {
          const int retire_tile = tile - kRetireLag;
          const int retire_operation =
              projection * kNumKTiles + retire_tile;
          const int retire_stage = retire_operation % RingStages;
          const int retire_phase = (retire_operation / RingStages) & 1;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
          const uint64_t detail_umma_wait_begin =
              cuda::ptx::get_sreg_globaltimer();
#endif
          umma_full[retire_stage].wait(retire_phase);
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
          if constexpr (ResidentAllTma) {
            detail_umma_full_wait_ns +=
                cuda::ptx::get_sreg_globaltimer() -
                detail_umma_wait_begin;
          }
#endif
          if (lane == 0) {
            stage_empty[retire_stage].arrive();
          }
        }
        // Preserve the gate accumulator in TMEM while six of the eight up
        // scale tiles are prepared. The final two scale tiles and the UMMA
        // issuer then overlap this warp-local TMEM drain and SiLU work.
        constexpr int kLateGateDrainTile = kNumKTiles - 3;
        if (projection == 1 && tile == kLateGateDrainTile) {
          projection_ready[0].wait(0);
          asm volatile(
              "tcgen05.fence::after_thread_sync;" ::: "memory");
          dae_mxfp_gate_silu_registers<kOutputRows>(
              tid, tiled_t2r, gate_tmem, c_acc, register_gate_silu);
        }
      }
      if (warp == 2) {
        constexpr int kRetireLag = RingStages > 4 ? 4 : RingStages - 1;
        #pragma unroll
        for (int retire_tile = kNumKTiles - kRetireLag;
             retire_tile < kNumKTiles; ++retire_tile) {
          const int retire_operation =
              projection * kNumKTiles + retire_tile;
          const int retire_stage = retire_operation % RingStages;
          const int retire_phase = (retire_operation / RingStages) & 1;
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
          const uint64_t detail_umma_wait_begin =
              cuda::ptx::get_sreg_globaltimer();
#endif
          umma_full[retire_stage].wait(retire_phase);
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
          if constexpr (ResidentAllTma) {
            detail_umma_full_wait_ns +=
                cuda::ptx::get_sreg_globaltimer() -
                detail_umma_wait_begin;
          }
#endif
          if (lane == 0) {
            stage_empty[retire_stage].arrive();
          }
        }
        if (lane == 0) {
          projection_ready[projection].arrive();
        }
      }
    } else if (warp == 0) {
      #pragma unroll 2
      for (int tile = 0; tile < kNumKTiles; ++tile) {
        const int operation = projection * kNumKTiles + tile;
        const int stage = operation % RingStages;
        const int phase = (operation / RingStages) & 1;
        if constexpr (kSeparateWeightScaleBarrier) {
          weight_scale_full[stage].wait(phase);
          scale_full[stage].wait(phase);
          #pragma unroll
          for (int subtile = 0; subtile < kK128PerTile; ++subtile) {
            const int scale_subtile =
                mxfpGateUpSubtileScaleSlots ? subtile : 0;
            const uint32_t scale_subtile_base =
                scale_tmem_base + stage * scale_tmem_stage_columns +
                scale_subtile * scale_tmem_subtile_columns;
            dae_mxfp_copy_scale_subtile_to_tmem<
                kSfaStageBytes, kSfaK128Bytes, kSfbK128Bytes,
                Utccp, Scale, typename TiledMma::FrgTypeSFA,
                typename TiledMma::FrgTypeSFB, LayoutSFA, LayoutSFB>(
                    scale_ring + stage * kScaleStageBytes,
                    subtile, scale_subtile_base, sfb_column_offset);
          }
          // Scale UTCCP is now in flight/complete in TMEM while the much
          // larger transformed-weight transaction reaches visibility.
          weight_full[stage].wait(phase);
        } else if constexpr (ResidentAllTma) {
          // The single resident token covers transformed weight, SFA, SFB,
          // and (for gate tile zero) the immutable full activation image.
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
          const uint64_t detail_weight_wait_begin =
              cuda::ptx::get_sreg_globaltimer();
#endif
          weight_full[stage].wait(phase);
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
          detail_weight_full_wait_ns +=
              cuda::ptx::get_sreg_globaltimer() -
              detail_weight_wait_begin;
#endif
        } else {
          weight_full[stage].wait(phase);
          scale_full[stage].wait(phase);
        }

#if defined(DAE_DEBUG_PRINT)
        if (projection == 0 && tile == 0 && lane == 0) {
          auto *sample = weight_ring + stage * kWeightStageBytes;
          printf(
              "fixed-weight quarters=%08x,%08x,%08x,%08x\n",
              *reinterpret_cast<const uint32_t *>(sample),
              *reinterpret_cast<const uint32_t *>(sample + 4096),
              *reinterpret_cast<const uint32_t *>(sample + 8192),
              *reinterpret_cast<const uint32_t *>(sample + 12288));
        }
#endif

        #pragma unroll
        for (int subtile = 0; subtile < kK128PerTile; ++subtile) {
          // Give each K128 member of a K512 bundle its own task-local TMEM
          // scale slot. The following UTCCP can then execute while the prior
          // UMMA still reads its immutable scale columns. stage_empty retires
          // all members of this stage together after the UMMA bundle arrives.
          const int scale_subtile =
              mxfpGateUpSubtileScaleSlots ? subtile : 0;
          const uint32_t scale_subtile_base =
              scale_tmem_base + stage * scale_tmem_stage_columns +
              scale_subtile * scale_tmem_subtile_columns;
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
                  (kStreamDirectActivation
                       ? activation_ring + stage * kActivationStageBytes
                       : activation_base + tile * kActivationStageBytes) +
                  subtile * kActivationK128Bytes)),
              layout_sB);
          auto frag_a = cta_mma.make_fragment_A(sA);
          auto frag_b = cta_mma.make_fragment_B(sB);
#if defined(DAE_DEBUG_PRINT)
          if (projection == 0 && tile == 0 && subtile == 0 && lane == 0) {
            printf(
                "fixed-fragment K=%d a_size=%d a_k_blocks=%d b_size=%d "
                "b_k_blocks=%d acc_cols=%d sfa_cols=%d sfb_cols=%d\n",
                K, int(size(frag_a)), int(size<2>(frag_a)),
                int(size(frag_b)), int(size<2>(frag_b)),
                accumulator_columns, sfa_columns, sfb_columns);
          }
#endif
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
            auto frag_a_k = frag_a(_, _, k_block);
            auto frag_b_k = frag_b(_, _, k_block);
            auto sfa_k = stage_tmem_sfa(_, _, k_block);
            auto sfb_k = stage_tmem_sfb(_, _, k_block);
            if constexpr (mxfpGateUpRawUmma) {
              if (projection == 0) {
                dae_mxfp_gate_up_issue_raw_umma<
                    Weight, Activation, Accum, Scale, kTileM, kTileN>(
                        frag_a_k, frag_b_k, gate_acc, accumulate,
                        sfa_k, sfb_k);
              } else {
                dae_mxfp_gate_up_issue_raw_umma<
                    Weight, Activation, Accum, Scale, kTileM, kTileN>(
                        frag_a_k, frag_b_k, up_acc, accumulate,
                        sfa_k, sfb_k);
              }
            } else {
              if (projection == 0) {
                gemm(
                    tiled_mma.with(accumulate, sfa_k, sfb_k),
                    frag_a_k, frag_b_k, gate_acc);
              } else {
                gemm(
                    tiled_mma.with(accumulate, sfa_k, sfb_k),
                    frag_a_k, frag_b_k, up_acc);
              }
            }
          }
        }
        cutlass::arch::umma_arrive(
            reinterpret_cast<uint64_t *>(umma_full + stage));
#if defined(DAE_DEBUG_PRINT)
        umma_full[stage].wait(phase);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        auto debug_t2r = tiled_t2r.get_slice(tid);
        auto debug_tmem = debug_t2r.partition_S(
            projection == 0 ? gate_tmem : up_tmem);
        auto debug_coord = debug_t2r.partition_D(c_acc);
        auto debug_acc = make_tensor<Accum>(shape(debug_coord));
        copy(tiled_t2r, debug_tmem, debug_acc);
        cutlass::arch::fence_view_async_tmem_load();
        if (lane == 0) {
          printf(
              "fixed-running-acc projection=%d tile=%d value=%g\n",
              projection, tile, double(debug_acc(0)));
        }
#endif
      }
      if (projection == 1) {
        // The final up commit is asynchronous and targets a disjoint TMEM
        // allocation, so warp 0 can evaluate its gate rows in the shadow of
        // the last up bundle without perturbing the up issue stream.
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        dae_mxfp_gate_silu_registers<kOutputRows>(
            tid, tiled_t2r, gate_tmem, c_acc, register_gate_silu);
      }
    }
  }

  if (warp == 1) {
    __syncwarp();
    projection_ready[1].wait(0);
    if constexpr (
        !ResidentAllTma && !mxfpGateUpDirectActivationEnabled) {
      c2m.template push<numThreadsPerWarp>(tid, activation_slots);
    }
  }

  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_compute_group(128);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

#if defined(DAE_DEBUG_PRINT)
  {
    auto debug_t2r = tiled_t2r.get_slice(tid);
    auto debug_coord = debug_t2r.partition_D(c_acc);
    auto debug_gate = make_tensor<Accum>(shape(debug_coord));
    auto debug_up = make_tensor<Accum>(shape(debug_coord));
    copy(tiled_t2r, debug_t2r.partition_S(gate_tmem), debug_gate);
    copy(tiled_t2r, debug_t2r.partition_S(up_tmem), debug_up);
    cutlass::arch::fence_view_async_tmem_load();
    if (tid == 0) {
      printf(
          "fixed-final-acc gate0=%g up0=%g registers=%d\n",
          double(debug_gate(0)), double(debug_up(0)), int(size(debug_gate)));
    }
  }
#endif

  auto *quant_scales = reinterpret_cast<float *>(scale_ring);
  constexpr int kOutputScratchOffset =
      (kOutputRows * (kTileM / kScaleVector) * int(sizeof(float)) + 1023) &
      -1024;
  static_assert(
      kScaleRingBytes >=
          kOutputScratchOffset + kTileM * kOutputRows + kSfbK128Bytes,
      "fixed gate/up scale ring cannot hold the full N8 epilogue");
  int output_slots = 0;
  uint8_t *data_output = nullptr;
  if constexpr (mxfpGateUpDirectOutput) {
    // All scale images are dead after the final UMMA. Keep the temporary
    // N8 SwiGLU values and scales at the head, then pack the contiguous native
    // output in the aligned remainder without another allocator transaction.
    data_output = scale_ring + kOutputScratchOffset;
  } else {
    output_slots = m2c.template pop<0>();
    data_output = static_cast<uint8_t *>(
        get_slot_address(smem_base, extract(output_slots)));
  }
  // The fixed-ring schedule allocates one contiguous native output record.
  // Keeping data and scales in the same allocator slot removes a second
  // allocation/store rendezvous from the post-UMMA critical path.
  auto *scale_output = data_output + kTileM * kOutputRows;

  float swiglu_values[kOutputRows];
  #pragma unroll
  for (int output_row = 0; output_row < kOutputRows; ++output_row) {
    swiglu_values[output_row] = 0.0f;
  }
  if (tid < size(tiled_t2r)) {
    auto thread_t2r = tiled_t2r.get_slice(tid);
    auto thread_up_tmem = thread_t2r.partition_S(up_tmem);
    auto thread_coord = thread_t2r.partition_D(c_acc);
    auto up_registers = make_tensor<Accum>(shape(thread_coord));
    copy(tiled_t2r, thread_up_tmem, up_registers);
    cutlass::arch::fence_view_async_tmem_load();
    #pragma unroll
    for (int index = 0; index < size(up_registers); ++index) {
      const int row = int(get<0>(thread_coord(index)));
      const int column = int(get<1>(thread_coord(index)));
      if (row < kTileM && column < kOutputRows) {
        const float bounded_up = fminf(
            fmaxf(up_registers(index), -10.0f), 10.0f);
        if constexpr (mxfpGateUpFixedBf16Epilogue) {
          swiglu_values[column] = __bfloat162float(__hmul(
              register_gate_silu[column],
              __float2bfloat16_rn(bounded_up)));
        } else {
          swiglu_values[column] =
              float(register_gate_silu[column]) * bounded_up;
        }
      }
    }
  }

  #pragma unroll
  for (int output_row = 0; output_row < kOutputRows; ++output_row) {
    float maximum = fabsf(swiglu_values[output_row]);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(
          maximum,
          __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
    }
    if (lane == 0) {
      const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
      const float exponent = ceilf(log2f(requested));
      quant_scales[output_row * (kTileM / kScaleVector) + warp] =
          exp2f(fminf(fmaxf(exponent, -127.0f), 127.0f));
    }
  }
  __sync_compute_group(128);

  const int source_chunk = tid / 16;
  const int byte_in_chunk = tid % 16;
  #pragma unroll
  for (int output_row = 0; output_row < kOutputRows; ++output_row) {
    const float value = swiglu_values[output_row];
    const float quant_scale =
        quant_scales[output_row * (kTileM / kScaleVector) + warp];
    const Activation quantized = value == 0.0f
        ? Activation(0.0f)
        : Activation(fminf(fmaxf(value / quant_scale, -448.0f), 448.0f));
    const int destination_chunk = source_chunk ^ output_row;
    reinterpret_cast<Activation *>(data_output)[
        output_row * kTileK + destination_chunk * 16 + byte_in_chunk] =
            quantized;
  }

  using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
  const auto logical_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});
  if (tid < kOutputRows * (kTileK / kScaleVector)) {
    const int output_row = tid / (kTileK / kScaleVector);
    const int scale_fragment = tid % (kTileK / kScaleVector);
    const int destination = int(
        logical_sfb(output_row, scale_fragment * kScaleVector));
    reinterpret_cast<Scale *>(scale_output)[destination] =
        Scale(quant_scales[
            output_row * (kTileM / kScaleVector) + scale_fragment]);
  }
  __sync_compute_group(128);
  if constexpr (mxfpGateUpDirectOutput) {
    if (tid == 0) {
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global,
          cuda::ptx::space_shared,
          direct_output_global,
          data_output,
          uint32_t(kTileM * kOutputRows + kSfbK128Bytes));
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      cuda::ptx::fence_proxy_async();
      if constexpr (PublishReady) {
        if (ready_bar != 0xFFFFFFFFU) {
          asm volatile("fence.release.gpu;" ::: "memory");
          *reinterpret_cast<volatile int *>(global_bars + ready_bar) = 0;
        }
      }
    }
    if constexpr (PublishReady) {
      __sync_compute_group(128);
    }
  } else {
    c2m.template push<31, true, false>(tid, output_slots);
  }
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
  if (detail_events != nullptr && detail_sm_id >= 0) {
    if (tid == 0) {
      detail_events[
          detail_sm_id * numProfileEvents +
          mxfpFfnDetailComputeLinear1WeightWaitNs] =
          detail_weight_full_wait_ns;
    } else if (tid == 2 * numThreadsPerWarp) {
      detail_events[
          detail_sm_id * numProfileEvents +
          mxfpFfnDetailComputeLinear1UmmaWaitNs] =
          detail_umma_full_wait_ns;
    }
  }
#endif
}
