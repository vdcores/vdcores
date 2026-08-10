#pragma once

#include "context.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>
#include <cutlass/numeric_types.h>

// CUTLASS 4.6.1's get_utccp_smem_desc_tensor is intentionally a host-side
// constexpr helper because normal CUTLASS kernels build this object in their
// host argument plumbing.  DAE decodes raw pointers inside a persistent device
// kernel, so build the identical descriptor tensor on the device instead.
template <class UtccpOp, class TEngine, class TLayout>
__device__ __forceinline__ auto dae_get_utccp_smem_desc_tensor(
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

// Native block-scaled tensor-core correctness path for checkpoint-shaped
// DeepSeek-V4 expert GEMV.  One SM owns a full 128-row output tile.  The
// checkpoint remains row-major; compute threads stage each K256 slice into the
// UMMA shared-memory layouts and UTCCP moves its E4M3 scale factors into TMEM.
// This path is also the arithmetic/layout oracle for the later pipelined path.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_sm100(
    int rows,
    int k,
    int output_columns,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using CheckpointScale = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue4m3_t;
  using Accum = float;
  using Output = cutlass::bfloat16_t;

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kTileK = 256;
  constexpr int kScaleVector = 16;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = SM100_MMA_MXF4_SS<
      Fp4, Fp4, Accum, Scale,
      kTileM, kTileN, kScaleVector,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;

  static_assert(kTileK % TiledMma::K == 0);

  const int weight_slot = m2c.template pop<0>();
  const auto *weight = static_cast<const uint8_t *>(
      slot_2_glob_ptr(st_insts, weight_slot));
  const int weight_scale_slot = m2c.template pop<0>();
  const auto *weight_scale = static_cast<const CheckpointScale *>(
      slot_2_glob_ptr(st_insts, weight_scale_slot));
  const int input_slot = m2c.template pop<0>();
  const auto *input = static_cast<const uint8_t *>(
      slot_2_glob_ptr(st_insts, input_slot));
  const int input_scale_slot = m2c.template pop<0>();
  const auto *input_scale = static_cast<const CheckpointScale *>(
      slot_2_glob_ptr(st_insts, input_scale_slot));
  const int alpha_slot = m2c.template pop<0>();
  const auto *alpha_ptr = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, alpha_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<Output *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  const int packed_row_stride = k / 2;
  const int scale_row_stride = k / kScaleVector;
  const int num_k_tiles = k / kTileK;
  const float alpha = *alpha_ptr;

  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);
  auto mma_shape_a = partition_shape_A(
      tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto mma_shape_b = partition_shape_B(
      tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto layout_sA = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Fp4>{}, mma_shape_a);
  auto layout_sB = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Fp4>{}, mma_shape_b);
  using LayoutSFA = decltype(
      ScaleConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape{}));
  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));

  constexpr int kABytes = (cosize_v<decltype(layout_sA)> + 1) / 2;
  constexpr int kBBytes = (cosize_v<decltype(layout_sB)> + 1) / 2;
  constexpr int kSFABytes = cosize_v<LayoutSFA>;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  // UMMA shared descriptors encode addresses in 128-byte units.  CUTE's
  // cosize is the logical extent, not a placement/alignment guarantee.
  constexpr int kDescriptorAlignment = 128;
  constexpr int kAStorageBytes =
      (kABytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kBStorageBytes =
      (kBBytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kSFAStorageBytes =
      (kSFABytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kScratchBytes =
      kAStorageBytes + kBStorageBytes + kSFAStorageBytes + kSFBBytes;
  static_assert(kScratchBytes <= 3 * slotSizeKb * 1024,
                "NVFP4 UMMA scratch must fit in three physical slots");

  auto *scratch = static_cast<uint8_t *>(smem_base);
  auto *sA_bytes = scratch;
  auto *sB_bytes = sA_bytes + kAStorageBytes;
  auto *sSFA_bytes = sB_bytes + kBStorageBytes;
  auto *sSFB_bytes = sSFA_bytes + kSFAStorageBytes;

  auto sA = make_tensor(
      make_smem_ptr(reinterpret_cast<Fp4 *>(sA_bytes)), layout_sA);
  auto sB = make_tensor(
      make_smem_ptr(reinterpret_cast<Fp4 *>(sB_bytes)), layout_sB);
  auto tCrA = cta_mma.make_fragment_A(sA);
  auto tCrB = cta_mma.make_fragment_B(sB);

  auto logical_c = make_tensor(
      make_smem_ptr(static_cast<Accum *>(nullptr)),
      make_layout(
          make_shape(Int<kTileM>{}, Int<kTileN>{}),
          make_stride(Int<kTileN>{}, Int<1>{})));
  auto cta_c = cta_mma.partition_C(logical_c);
  auto tmem_acc = cta_mma.make_fragment_C(cta_c);
  tmem_acc.data() = tmem_base_ptr;

  auto tCtSFA = make_tensor<typename TiledMma::FrgTypeSFA>(
      shape(LayoutSFA{}));
  auto tCtSFB = make_tensor<typename TiledMma::FrgTypeSFB>(
      shape(LayoutSFB{}));
  tCtSFA.data() = tmem_base_ptr +
      cutlass::detail::find_tmem_tensor_col_offset(tmem_acc);
  tCtSFB.data() = tCtSFA.data().get() +
      cutlass::detail::find_tmem_tensor_col_offset(tCtSFA);

  auto tCsSFA = make_tensor(
      make_smem_ptr(reinterpret_cast<Scale *>(sSFA_bytes)), LayoutSFA{});
  auto tCsSFB = make_tensor(
      make_smem_ptr(reinterpret_cast<Scale *>(sSFB_bytes)), LayoutSFB{});
  auto tCsSFA_compact = make_tensor(
      tCsSFA.data(), filter_zeros(tCsSFA.layout()));
  auto tCtSFA_compact = make_tensor(
      tCtSFA.data(), filter_zeros(tCtSFA.layout()));
  auto tCsSFB_compact = make_tensor(
      tCsSFB.data(), filter_zeros(tCsSFB.layout()));
  auto tCtSFB_compact = make_tensor(
      tCtSFB.data(), filter_zeros(tCtSFB.layout()));
  using Utccp = SM100_UTCCP_4x32dp128bit_1cta;
  auto copy_sfa = make_utccp_copy(Utccp{}, tCtSFA_compact);
  auto copy_sfb = make_utccp_copy(Utccp{}, tCtSFB_compact);
  auto copy_sfa_slice = copy_sfa.get_slice(0);
  auto copy_sfb_slice = copy_sfb.get_slice(0);
  auto copy_sfa_src_raw = copy_sfa_slice.partition_S(tCsSFA_compact);
  auto copy_sfb_src_raw = copy_sfb_slice.partition_S(tCsSFB_compact);
  auto copy_sfa_src =
      dae_get_utccp_smem_desc_tensor<Utccp>(copy_sfa_src_raw);
  auto copy_sfb_src =
      dae_get_utccp_smem_desc_tensor<Utccp>(copy_sfb_src_raw);
  auto copy_sfa_dst = copy_sfa_slice.partition_D(tCtSFA_compact);
  auto copy_sfb_dst = copy_sfb_slice.partition_D(tCtSFB_compact);

  using ScaleProblemShape =
      Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
  const auto logical_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(ScaleProblemShape{});
  const auto logical_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});
  constexpr int kPackedSegmentsA = kTileM * kTileK / 32;
  constexpr int kPackedSegmentsB = kTileN * kTileK / 32;
  constexpr int kScalesA = kTileM * kTileK / kScaleVector;
  constexpr int kScalesB = kTileN * kTileK / kScaleVector;

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int tile = 0; tile < num_k_tiles; ++tile) {
    for (int segment = tid; segment < kPackedSegmentsA; segment += 128) {
      const int row = segment / (kTileK / 32);
      const int segment_k = segment % (kTileK / 32);
      const int logical_k = segment_k * 32;
      const int mma_k = logical_k / TiledMma::K;
      const int k_in_mma = logical_k % TiledMma::K;
      const auto *source = weight + row * packed_row_stride +
          tile * (kTileK / 2);
      constexpr int kRowPhaseXor = 9 * kScaleVector;
      const int source_xor = ((row >> 2) & 1) ? kRowPhaseXor : 0;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int destination_k = logical_k + half * 16;
        const int dst_byte = int(layout_sA(make_coord(
            make_coord(row, k_in_mma + half * 16), 0, mma_k))) / 2;
        uint2 values{};
        if (row < rows) {
          const int source_k = destination_k ^ source_xor;
          values = *reinterpret_cast<const uint2 *>(source + source_k / 2);
        }
        *reinterpret_cast<uint2 *>(sA_bytes + dst_byte) = values;
      }
    }
    for (int segment = tid; segment < kPackedSegmentsB; segment += 128) {
      const int segment_k = segment % (kTileK / 32);
      const int logical_k = segment_k * 32;
      const int row = segment / (kTileK / 32);
      const int mma_k = logical_k / TiledMma::K;
      const int k_in_mma = logical_k % TiledMma::K;
      const auto *source = input + tile * (kTileK / 2);
      constexpr int kRowPhaseXor = 9 * kScaleVector;
      const int source_xor = ((row >> 2) & 1) ? kRowPhaseXor : 0;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int destination_k = logical_k + half * 16;
        const int dst_byte = int(layout_sB(make_coord(
            make_coord(row, k_in_mma + half * 16), 0, mma_k))) / 2;
        const int source_k = destination_k ^ source_xor;
        const uint2 values =
            *reinterpret_cast<const uint2 *>(source + source_k / 2);
        *reinterpret_cast<uint2 *>(sB_bytes + dst_byte) = values;
      }
    }
    for (int index = tid; index < kScalesA; index += 128) {
      const int row = index / (kTileK / kScaleVector);
      const int sf = index % (kTileK / kScaleVector);
      const int dst = int(logical_sfa(row, sf * kScaleVector));
      const int source_sf = sf ^ ((sf & 8) ? 1 : 0);
      sSFA_bytes[dst] = row < rows
          ? reinterpret_cast<const uint8_t *>(weight_scale)[
                row * scale_row_stride +
                tile * (kTileK / kScaleVector) + source_sf]
          : 0;
    }
    for (int index = tid; index < kScalesB; index += 128) {
      const int sf = index % (kTileK / kScaleVector);
      const int row = index / (kTileK / kScaleVector);
      const int dst = int(logical_sfb(
          row, sf * kScaleVector));
      const int source_sf = sf ^ ((sf & 8) ? 1 : 0);
      sSFB_bytes[dst] = reinterpret_cast<const uint8_t *>(input_scale)[
          tile * (kTileK / kScaleVector) + source_sf];
    }
    __sync_compute_group(128);

    if (tid < 32 && elect_one_sync()) {
      copy(copy_sfa, copy_sfa_src, copy_sfa_dst);
      copy(copy_sfb, copy_sfb_src, copy_sfb_dst);
    }
    if (tid < 32) {
#pragma unroll
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        gemm(
            tiled_mma.with(
                tiled_mma.accumulate_,
                tCtSFA(_, _, k_block),
                tCtSFB(_, _, k_block)),
            tCrA(_, _, k_block),
            tCrB(_, _, k_block),
            tmem_acc);
        tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
      cutlass::arch::umma_arrive(tmem_mma_barrier);
    }
    cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
    tmem_mma_phase ^= 1;
  }

  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_compute_group(128);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  auto coord_c = make_identity_tensor(
      make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto cta_coord_c = cta_mma.partition_C(coord_c);
  using TmemLoad = SM100_TMEM_LOAD_32dp32b1x;
  // Match CUTLASS's SM100 no-SMEM epilogue: strip the MMA tiling modes before
  // constructing the TMEM copy and map each warp's lane onto the 32 hardware
  // datapaths.  Passing the unsliced accumulator or a 0..127 thread id makes
  // CUTE interpret MMA/N modes as extra datapaths.
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
    if (row < rows && col < output_columns) {
      output[row * output_columns + col] = Output(r_acc(index) * alpha);
    }
  }

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}
