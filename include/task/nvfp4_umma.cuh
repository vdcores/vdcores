#pragma once

#include "context.cuh"
#include "deepseek_v4.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// The resident kernel owns one full-SM TMEM allocation. Individual tasks plan
// non-overlapping column ranges inside that allocation without touching the
// runtime allocator or carrying shape-specific offsets across instructions.
class DaeTaskTmemAllocator {
 public:
  __device__ __forceinline__ explicit DaeTaskTmemAllocator(uint32_t base)
      : base_(base), cursor_(base) {}

  __device__ __forceinline__ uint32_t allocate(int columns) {
    const uint32_t result = cursor_;
    cursor_ += static_cast<uint32_t>(columns);
    return result;
  }

  __device__ __forceinline__ uint32_t used_columns() const {
    return cursor_ - base_;
  }

 private:
  uint32_t base_;
  uint32_t cursor_;
};

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

  const int weight_slots = m2c.template pop<0>();
  const int weight_slot = extract(weight_slots);
  const auto *weight = static_cast<const uint8_t *>(
      get_slot_address(smem_base, weight_slot));
  const int weight_scale_slots = m2c.template pop<0>();
  const int weight_scale_slot = extract(weight_scale_slots);
  const auto *weight_scale = static_cast<const CheckpointScale *>(
      get_slot_address(smem_base, weight_scale_slot));
  const int input_slots = m2c.template pop<0>();
  const int input_slot = extract(input_slots);
  const auto *input = static_cast<const uint8_t *>(
      get_slot_address(smem_base, input_slot));
  const int input_scale_slots = m2c.template pop<0>();
  const int input_scale_slot = extract(input_scale_slots);
  const auto *input_scale = static_cast<const CheckpointScale *>(
      get_slot_address(smem_base, input_scale_slot));
  const int alpha_slots = m2c.template pop<0>();
  const int alpha_slot = extract(alpha_slots);
  const auto *alpha_ptr = static_cast<const float *>(
      get_slot_address(smem_base, alpha_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<Output *>(
      get_slot_address(smem_base, output_slot));

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
  c2m.push(
      tid,
      weight_slots | weight_scale_slots | input_slots |
          input_scale_slots | alpha_slots);
  c2m.template push<0, true>(tid, output_slots);
}

// Setup-only layout conversion. The data input has already been placed in its
// native swizzle by TMA. This task copies those bytes beside the matching
// native scale layout so the token-time path can fetch data and scales with
// one ordinary TMA transaction. Kind 0 packs M128 weight tiles; kind 1 packs
// the replicated N8 activation tile.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_umma_prepack_sm100(
    int kind,
    int num_k_tiles,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using CheckpointScale = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue4m3_t;
  using Accum = float;

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

  TiledMma tiled_mma;
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
  constexpr int kAlignment = 128;
  constexpr int kABytes = (cosize_v<decltype(layout_sA)> + 1) / 2;
  constexpr int kBBytes = (cosize_v<decltype(layout_sB)> + 1) / 2;
  constexpr int kAStorageBytes =
      (kABytes + kAlignment - 1) & -kAlignment;
  constexpr int kBStorageBytes =
      (kBBytes + kAlignment - 1) & -kAlignment;
  constexpr int kSFABytes = cosize_v<LayoutSFA>;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  static_assert(kAStorageBytes + kSFABytes == 18432);
  static_assert(kBStorageBytes + kSFBBytes == 3072);

  using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
  const auto logical_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(ScaleProblemShape{});
  const auto logical_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});
  constexpr int kScalesA = kTileM * kTileK / kScaleVector;
  constexpr int kScalesB = kTileN * kTileK / kScaleVector;
  cutlass::NumericConverter<Scale, CheckpointScale> convert_scale;
  const int tid = __compute_tid();

  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int data_slots = m2c.template pop<0>();
    const auto *data = static_cast<const uint8_t *>(
        get_slot_address(smem_base, extract(data_slots)));
    const int scale_slots = m2c.template pop<0>();
    const auto *scale = static_cast<const CheckpointScale *>(
        get_slot_address(smem_base, extract(scale_slots)));
    const int output_slots = m2c.template pop<0>();
    auto *output = static_cast<uint8_t *>(
        get_slot_address(smem_base, extract(output_slots)));

    const int data_bytes = kind == 0 ? kABytes : kBBytes;
    const int scale_offset = kind == 0 ? kAStorageBytes : kBStorageBytes;
    const int scale_bytes = kind == 0 ? kSFABytes : kSFBBytes;
    for (int offset = tid * 16; offset < data_bytes; offset += 128 * 16) {
      *reinterpret_cast<uint4 *>(output + offset) =
          *reinterpret_cast<const uint4 *>(data + offset);
    }
    for (int offset = tid; offset < scale_bytes; offset += 128) {
      output[scale_offset + offset] = 0;
    }
    __sync_compute_group(128);

    if (kind == 0) {
      auto *packed_scale = reinterpret_cast<Scale *>(
          output + kAStorageBytes);
      for (int index = tid; index < kScalesA; index += 128) {
        const int row = index / (kTileK / kScaleVector);
        const int sf = index % (kTileK / kScaleVector);
        const int dst = int(logical_sfa(row, sf * kScaleVector));
        packed_scale[dst] = convert_scale(
            scale[row * (kTileK / kScaleVector) + sf]);
      }
    } else {
      auto *packed_scale = reinterpret_cast<Scale *>(
          output + kBStorageBytes);
      for (int index = tid; index < kScalesB; index += 128) {
        const int sf = index % (kTileK / kScaleVector);
        const int row = index / (kTileK / kScaleVector);
        const int dst = int(logical_sfb(row, sf * kScaleVector));
        packed_scale[dst] = convert_scale(scale[sf]);
      }
    }
    __sync_compute_group(128);
    c2m.push(tid, data_slots | scale_slots);
    c2m.template push<0, true>(tid, output_slots);
  }
}

// Quantize token-dependent BF16 activations directly into the combined native
// N8/K256 UMMA B layout. This avoids materializing raw packed data/scales and
// then copying them through a separate preprocessing stage.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_nvfp4_quant_umma_b_sm100(
    int num_k_tiles,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Accum = float;

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

  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));
  using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
  const auto logical_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});

  constexpr int kAlignment = 128;
  constexpr int kBBytes = kTileN * kTileK / 2;
  constexpr int kBStorageBytes =
      (kBBytes + kAlignment - 1) & -kAlignment;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  static_assert(kBStorageBytes + kSFBBytes == 3072);

  const int input_slots = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, extract(input_slots)));
  const int global_scale_slots = m2c.template pop<0>();
  const auto *global_scale = static_cast<const float *>(
      get_slot_address(smem_base, extract(global_scale_slots)));
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<uint8_t *>(
      get_slot_address(smem_base, extract(output_slots)));

  constexpr int kThreadsPerBlock = 8;
  const int tid = __compute_tid();
  const int block_lane = tid & (kThreadsPerBlock - 1);
  const int block_group = tid / kThreadsPerBlock;
  const int warp_lane = tid & 31;
  const unsigned block_mask =
      0xFFU << (warp_lane & ~(kThreadsPerBlock - 1));
  const float model_scale = global_scale[0];
  auto *quant_denominators = static_cast<float *>(task_scratch);
  const auto *input_pairs =
      reinterpret_cast<const __nv_bfloat162 *>(input);

  for (int tile = 0; tile < num_k_tiles; ++tile) {
    auto *tile_output = output + tile * (kBStorageBytes + kSFBBytes);
    auto *packed_scale = reinterpret_cast<Scale *>(
        tile_output + kBStorageBytes);
    for (int offset = tid; offset < kSFBBytes; offset += 128) {
      tile_output[kBStorageBytes + offset] = 0;
    }
    __sync_compute_group(128);

    const int block = block_group;
    const __nv_bfloat162 pair = input_pairs[
        tile * (kTileK / 2) + block * (kScaleVector / 2) + block_lane];
    const float2 values = __bfloat1622float2(pair);
    float maximum = fmaxf(fabsf(values.x), fabsf(values.y));
    for (int offset = kThreadsPerBlock / 2; offset > 0; offset >>= 1) {
      maximum = fmaxf(
          maximum,
          __shfl_down_sync(
              block_mask, maximum, offset, kThreadsPerBlock));
    }
    if (block_lane == 0) {
      const float block_scale =
          dsv4_ceil_e4m3(dsv4_div_rn(maximum, 6.0f * model_scale));
      quant_denominators[block_group] = block_scale * model_scale;
      for (int row = 0; row < kTileN; ++row) {
        const int dst = int(logical_sfb(row, block * kScaleVector));
        packed_scale[dst] = Scale(block_scale);
      }
    }
    __syncwarp(block_mask);

    const float quant_denominator = quant_denominators[block_group];
    const uint8_t low = dsv4_nearest_fp4(
        dsv4_div_rn(values.x, quant_denominator));
    const uint8_t high = dsv4_nearest_fp4(
        dsv4_div_rn(values.y, quant_denominator));
    const uint8_t packed = low | (high << 4);
    // TMA's 128-byte swizzle treats each row as eight 16-byte chunks.
    // Physical destination chunk d receives logical source chunk d xor row.
    // Invert that relation here while each 8-thread group owns one 8-byte
    // half-chunk of freshly quantized packed values.
    const int source_chunk = block / 2;
    const int half = block & 1;
    for (int row = 0; row < kTileN; ++row) {
      const int destination_chunk = source_chunk ^ row;
      const int destination =
          row * (kTileK / 2) + destination_chunk * 16 +
          half * kThreadsPerBlock + block_lane;
      tile_output[destination] = packed;
    }
    __sync_compute_group(128);
  }

  c2m.push(tid, input_slots | global_scale_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Finalize linear split-K sums at the only required dependency boundary:
// apply bounded SwiGLU in FP32 and directly emit the native N8/K256 NVFP4 B
// layout consumed by W2.  No BF16 middle tensor or repack task is materialized.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_dsv4_fp32_swiglu_nvfp4_quant_umma_b_sm100(
    int num_k_tiles,
    float limit,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Accum = float;

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
  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));
  using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
  const auto logical_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});

  constexpr int kAlignment = 128;
  constexpr int kBBytes = kTileN * kTileK / 2;
  constexpr int kBStorageBytes =
      (kBBytes + kAlignment - 1) & -kAlignment;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  static_assert(kBStorageBytes + kSFBBytes == 3072);

  const int gate_slots = m2c.template pop<0>();
  const auto *gate = static_cast<const float *>(
      get_slot_address(smem_base, extract(gate_slots)));
  const int up_slots = m2c.template pop<0>();
  const auto *up = static_cast<const float *>(
      get_slot_address(smem_base, extract(up_slots)));
  const int global_scale_slots = m2c.template pop<0>();
  const auto *global_scale = static_cast<const float *>(
      get_slot_address(smem_base, extract(global_scale_slots)));
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<uint8_t *>(
      get_slot_address(smem_base, extract(output_slots)));

  constexpr int kThreadsPerBlock = 8;
  const int tid = __compute_tid();
  const int block_lane = tid & (kThreadsPerBlock - 1);
  const int block_group = tid / kThreadsPerBlock;
  const int warp_lane = tid & 31;
  const unsigned block_mask =
      0xFFU << (warp_lane & ~(kThreadsPerBlock - 1));
  const float model_scale = global_scale[0];
  auto *quant_denominators = static_cast<float *>(task_scratch);

  for (int tile = 0; tile < num_k_tiles; ++tile) {
    auto *tile_output = output + tile * (kBStorageBytes + kSFBBytes);
    auto *packed_scale = reinterpret_cast<Scale *>(
        tile_output + kBStorageBytes);
    for (int offset = tid; offset < kSFBBytes; offset += 128) {
      tile_output[kBStorageBytes + offset] = 0;
    }
    __sync_compute_group(128);

    const int block = block_group;
    const int pair_index =
        tile * (kTileK / 2) + block * (kScaleVector / 2) + block_lane;
    const float2 gate_values =
        reinterpret_cast<const float2 *>(gate)[pair_index];
    const float2 up_values =
        reinterpret_cast<const float2 *>(up)[pair_index];
    float2 values;
    const float gate0 = fminf(gate_values.x, limit);
    const float gate1 = fminf(gate_values.y, limit);
    const float up0 = fminf(fmaxf(up_values.x, -limit), limit);
    const float up1 = fminf(fmaxf(up_values.y, -limit), limit);
    values.x = gate0 / (1.0f + expf(-gate0)) * up0;
    values.y = gate1 / (1.0f + expf(-gate1)) * up1;

    float maximum = fmaxf(fabsf(values.x), fabsf(values.y));
    for (int offset = kThreadsPerBlock / 2; offset > 0; offset >>= 1) {
      maximum = fmaxf(
          maximum,
          __shfl_down_sync(
              block_mask, maximum, offset, kThreadsPerBlock));
    }
    if (block_lane == 0) {
      const float block_scale =
          dsv4_ceil_e4m3(dsv4_div_rn(maximum, 6.0f * model_scale));
      quant_denominators[block_group] = block_scale * model_scale;
      for (int row = 0; row < kTileN; ++row) {
        const int dst = int(logical_sfb(row, block * kScaleVector));
        packed_scale[dst] = Scale(block_scale);
      }
    }
    __syncwarp(block_mask);

    const float quant_denominator = quant_denominators[block_group];
    const uint8_t low = dsv4_nearest_fp4(
        dsv4_div_rn(values.x, quant_denominator));
    const uint8_t high = dsv4_nearest_fp4(
        dsv4_div_rn(values.y, quant_denominator));
    const uint8_t packed = low | (high << 4);
    const int source_chunk = block / 2;
    const int half = block & 1;
    for (int row = 0; row < kTileN; ++row) {
      const int destination_chunk = source_chunk ^ row;
      const int destination =
          row * (kTileK / 2) + destination_chunk * 16 +
          half * kThreadsPerBlock + block_lane;
      tile_output[destination] = packed;
    }
    __sync_compute_group(128);
  }

  c2m.push(tid, gate_slots | up_slots | global_scale_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Performance-oriented native path. LDU places prepacked data and scale bytes
// in one shared allocation for each operand and streams K256 tiles through
// native block-scaled UMMA.
// Each load contains one M128/K256 weight tile or one broadcast N8/K256
// activation tile directly in its UMMA data and scale layouts. Compute moves
// the native scales to TMEM, issues block-scaled UMMA, and retires each tile
// before consuming the next.
// No global address is visible here; the same operand stream can later be fed
// by routed LDU commands without changing this task.
template <typename Output, bool LoadOutputScale,
          typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_stream_impl_sm100(
    int num_k_tiles,
    int retain_activation,
    int bulk_activation,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Accum = float;

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

  const int alpha_slots = m2c.template pop<0>();
  const int alpha_slot = extract(alpha_slots);
  const auto *alpha_ptr = static_cast<const float *>(
      get_slot_address(smem_base, alpha_slot));
  const float alpha = *alpha_ptr;
  int output_scale_slots = 0;
  float output_scale = 1.0f;
  if constexpr (LoadOutputScale) {
    output_scale_slots = m2c.template pop<0>();
    const auto *output_scale_ptr = static_cast<const float *>(
        get_slot_address(smem_base, extract(output_scale_slots)));
    output_scale = *output_scale_ptr;
  }
  const int tid = __compute_tid();
  const int warp = tid / numThreadsPerWarp;

  int bulk_input_slots = 0;
  uint8_t *bulk_input_base = nullptr;
  if (bulk_activation) {
    if (warp == 0) {
      bulk_input_slots = m2c.template pop<0>();
      bulk_input_base = static_cast<uint8_t *>(
          get_slot_address(smem_base, extract(bulk_input_slots)));
    } else {
      m2c.advance();
    }
  }

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

  constexpr int kDescriptorAlignment = 128;
  constexpr int kABytes = (cosize_v<decltype(layout_sA)> + 1) / 2;
  constexpr int kBBytes = (cosize_v<decltype(layout_sB)> + 1) / 2;
  constexpr int kAStorageBytes =
      (kABytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kBStorageBytes =
      (kBBytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kSFABytes = cosize_v<LayoutSFA>;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  static_assert(kAStorageBytes + kSFABytes == 18432);
  static_assert(kBStorageBytes + kSFBBytes == 3072);

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

  auto tCtSFA_compact = make_tensor(
      tCtSFA.data(), filter_zeros(tCtSFA.layout()));
  auto tCtSFB_compact = make_tensor(
      tCtSFB.data(), filter_zeros(tCtSFB.layout()));
  using Utccp = SM100_UTCCP_4x32dp128bit_1cta;
  auto copy_sfa = make_utccp_copy(Utccp{}, tCtSFA_compact);
  auto copy_sfb = make_utccp_copy(Utccp{}, tCtSFB_compact);
  auto copy_sfa_slice = copy_sfa.get_slice(0);
  auto copy_sfb_slice = copy_sfb.get_slice(0);
  auto copy_sfa_dst = copy_sfa_slice.partition_D(tCtSFA_compact);
  auto copy_sfb_dst = copy_sfb_slice.partition_D(tCtSFB_compact);

  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
  for (int tile = 0; tile < num_k_tiles; ++tile) {
    int weight_slots = 0;
    int input_slots = bulk_input_slots;
    if (warp == 0) {
      weight_slots = m2c.template pop<0>();
      if (!bulk_activation) {
        input_slots = m2c.template pop<0>();
      }
    } else {
      m2c.advance();
      if (!bulk_activation) {
        m2c.advance();
      }
    }

    if (warp == 0) {
      auto *weight_base = static_cast<uint8_t *>(
          get_slot_address(smem_base, extract(weight_slots)));
      auto sA = make_tensor(
          make_smem_ptr(reinterpret_cast<Fp4 *>(weight_base)),
          layout_sA);
      auto tCrA = cta_mma.make_fragment_A(sA);
      auto tCsSFA = make_tensor(
          make_smem_ptr(reinterpret_cast<Scale *>(
              weight_base + kAStorageBytes)),
          LayoutSFA{});
      auto tCsSFA_compact = make_tensor(
          tCsSFA.data(), filter_zeros(tCsSFA.layout()));
      auto copy_sfa_src_raw = copy_sfa_slice.partition_S(tCsSFA_compact);
      auto copy_sfa_src =
          dae_get_utccp_smem_desc_tensor<Utccp>(copy_sfa_src_raw);

      auto *input_base = bulk_activation
          ? bulk_input_base + tile * (kBStorageBytes + kSFBBytes)
          : static_cast<uint8_t *>(
                get_slot_address(smem_base, extract(input_slots)));
      auto sB = make_tensor(
          make_smem_ptr(reinterpret_cast<Fp4 *>(input_base)),
          layout_sB);
      auto tCrB = cta_mma.make_fragment_B(sB);
      auto tCsSFB = make_tensor(
          make_smem_ptr(reinterpret_cast<Scale *>(
              input_base + kBStorageBytes)),
          LayoutSFB{});
      auto tCsSFB_compact = make_tensor(
          tCsSFB.data(), filter_zeros(tCsSFB.layout()));
      auto copy_sfb_src_raw = copy_sfb_slice.partition_S(tCsSFB_compact);
      auto copy_sfb_src =
          dae_get_utccp_smem_desc_tensor<Utccp>(copy_sfb_src_raw);

      if (elect_one_sync()) {
        copy(copy_sfa, copy_sfa_src, copy_sfa_dst);
        copy(copy_sfb, copy_sfb_src, copy_sfb_dst);
      }
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
      cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
      tmem_mma_phase ^= 1;
      c2m.template push<0>(tid, weight_slots);
      if (!bulk_activation) {
        c2m.template push<0>(tid, input_slots);
      }
    }
  }

  if (warp == 0 && bulk_activation && !retain_activation) {
    c2m.template push<0>(tid, bulk_input_slots);
  }

  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_compute_group(128);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<Output *>(
      get_slot_address(smem_base, output_slot));
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
      output[row] = Output(r_acc(index) * alpha * output_scale);
    }
  }

  __sync_compute_group(128);
  c2m.push(tid, alpha_slots);
  if constexpr (LoadOutputScale) {
    c2m.push(tid, output_scale_slots);
  }
  c2m.template push<0, true>(tid, output_slots);
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_stream_sm100(
    int num_k_tiles,
    int retain_activation,
    int bulk_activation,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  task_nvfp4_gemv_umma_stream_impl_sm100<
      cutlass::bfloat16_t, false>(
      num_k_tiles, retain_activation, bulk_activation, smem_base,
      tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase, m2c, c2m);
}

// The compute contract is independent of how K was scheduled: accumulate the
// supplied K stream and emit exactly one FP32 M128 result. The following STU
// instruction selects an ordinary write or reduce-add. The extra scalar is
// one for an unscaled K shard and the dynamic route weight for expert
// aggregation.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_fp32_sm100(
    int num_k_tiles,
    int retain_activation,
    int bulk_activation,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  task_nvfp4_gemv_umma_stream_impl_sm100<float, true>(
      num_k_tiles, retain_activation, bulk_activation, smem_base, tmem_base_ptr,
      tmem_mma_barrier, tmem_mma_phase, m2c, c2m);
}

// A task-local K256 stage ring keeps the LDU stream ahead of UMMA. Its depth is
// selected in the resident context so source-guided stage-count trials do not
// change this task's queue or accumulation contract. Activation loads use the
// same tunable chunking protocol as the WGMMA and native FP8 paths: one shared
// allocation contains activation_tiles_per_load consecutive tiles, while
// weight tiles always stream individually. Warp 0 owns scale staging and MMA
// issue; warp 1 independently observes completion and retires each weight plus
// its activation chunk after the final use. Each stage receives private
// SFA/SFB TMEM columns from the task-local allocator, so a later K tile cannot
// overwrite scales or shared operands belonging to an in-flight UMMA.
template <typename Output, bool LoadOutputScale, int OutputGroups,
          typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_pipeline_impl_sm100(
    int num_k_tiles,
    int retain_activation,
    int activation_tiles_per_load,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Accum = float;

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kTileK = 256;
  constexpr int kScaleVector = 16;
  constexpr int kStages = nvfp4UmmaPipelineStages;
  constexpr int kFullBarrierBase = nvfp4UmmaPipelineBarrierBase;
  constexpr int kEmptyBarrierBase = kFullBarrierBase + kStages;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = SM100_MMA_MXF4_SS<
      Fp4, Fp4, Accum, Scale,
      kTileM, kTileN, kScaleVector,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;
  static_assert(OutputGroups == 1 || OutputGroups == 2);

  const int alpha0_slots = m2c.template pop<0>();
  const auto *alpha0_ptr = static_cast<const float *>(
      get_slot_address(smem_base, extract(alpha0_slots)));
  const float alpha0 = *alpha0_ptr;
  int alpha1_slots = 0;
  float alpha1 = alpha0;
  if constexpr (OutputGroups == 2) {
    alpha1_slots = m2c.template pop<0>();
    const auto *alpha1_ptr = static_cast<const float *>(
        get_slot_address(smem_base, extract(alpha1_slots)));
    alpha1 = *alpha1_ptr;
  }
  int output_scale_slots = 0;
  float output_scale = 1.0f;
  if constexpr (LoadOutputScale) {
    output_scale_slots = m2c.template pop<0>();
    const auto *output_scale_ptr = static_cast<const float *>(
        get_slot_address(smem_base, extract(output_scale_slots)));
    output_scale = *output_scale_ptr;
  }
  const int tid = __compute_tid();
  const int warp = tid / numThreadsPerWarp;
  if (num_k_tiles <= 0 || activation_tiles_per_load <= 0 ||
      activation_tiles_per_load > num_k_tiles ||
      (retain_activation && activation_tiles_per_load != num_k_tiles)) {
    asm volatile("trap;");
  }

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

  constexpr int kDescriptorAlignment = 128;
  constexpr int kABytes = (cosize_v<decltype(layout_sA)> + 1) / 2;
  constexpr int kBBytes = (cosize_v<decltype(layout_sB)> + 1) / 2;
  constexpr int kAStorageBytes =
      (kABytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kBStorageBytes =
      (kBBytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kSFABytes = cosize_v<LayoutSFA>;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  constexpr int kActivationTileBytes = kBStorageBytes + kSFBBytes;
  static_assert(kAStorageBytes + kSFABytes == 18432);
  static_assert(kActivationTileBytes == 3072);

  auto logical_c = make_tensor(
      make_smem_ptr(static_cast<Accum *>(nullptr)),
      make_layout(
          make_shape(Int<kTileM>{}, Int<kTileN>{}),
          make_stride(Int<kTileN>{}, Int<1>{})));
  auto cta_c = cta_mma.partition_C(logical_c);
  auto tmem_acc = cta_mma.make_fragment_C(cta_c);

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
  const int scale_stage_columns =
      OutputGroups * sfa_columns + sfb_columns;

  DaeTaskTmemAllocator tmem_allocator(tmem_base_ptr);
  const uint32_t accumulator_pipeline_base =
      tmem_allocator.allocate(OutputGroups * accumulator_columns);
  tmem_acc.data() = accumulator_pipeline_base;
  const uint32_t scale_pipeline_base =
      tmem_allocator.allocate(kStages * scale_stage_columns);
  if (tmem_allocator.used_columns() >
      cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns) {
    asm volatile("trap;");
  }

  using Utccp = SM100_UTCCP_4x32dp128bit_1cta;
  for (int chunk_start = 0; chunk_start < num_k_tiles;
       chunk_start += activation_tiles_per_load) {
    int activation_slots = 0;
    if (warp < 2) {
      activation_slots = m2c.template pop<0>();
    } else {
      m2c.advance();
    }
    auto *activation_chunk_base = warp == 0
        ? static_cast<uint8_t *>(
              get_slot_address(smem_base, extract(activation_slots)))
        : nullptr;
    const int remaining = num_k_tiles - chunk_start;
    const int chunk_tiles = remaining < activation_tiles_per_load
        ? remaining
        : activation_tiles_per_load;
    for (int tile_in_chunk = 0; tile_in_chunk < chunk_tiles;
         ++tile_in_chunk) {
      const int tile = chunk_start + tile_in_chunk;
      const int stage = tile % kStages;
      const int generation = tile / kStages;
      if (warp == 0) {
        if (generation > 0) {
          const uint32_t stage_phase =
              (pipeline_phase_mask >> stage) & 1U;
          cute::wait_barrier(
              tmem_mma_barrier[kEmptyBarrierBase + stage],
              stage_phase ^ uint32_t((generation - 1) & 1));
        }

        auto *tile_activation = activation_chunk_base +
            tile_in_chunk * kActivationTileBytes;
        auto sB = make_tensor(
            make_smem_ptr(reinterpret_cast<Fp4 *>(tile_activation)),
            layout_sB);
        auto tCrB = cta_mma.make_fragment_B(sB);
        auto tCsSFB = make_tensor(
            make_smem_ptr(reinterpret_cast<Scale *>(
                tile_activation + kBStorageBytes)),
            LayoutSFB{});
        auto tCsSFB_compact = make_tensor(
            tCsSFB.data(), filter_zeros(tCsSFB.layout()));

        auto stage_sfb = make_tensor<typename TiledMma::FrgTypeSFB>(
            shape(LayoutSFB{}));
        const uint32_t stage_base =
            scale_pipeline_base + stage * scale_stage_columns;
        stage_sfb.data() =
            stage_base + OutputGroups * sfa_columns;
        auto stage_sfb_compact = make_tensor(
            stage_sfb.data(), filter_zeros(stage_sfb.layout()));
        auto copy_sfb = make_utccp_copy(Utccp{}, stage_sfb_compact);
        auto copy_sfb_slice = copy_sfb.get_slice(0);
        auto copy_sfb_src_raw = copy_sfb_slice.partition_S(tCsSFB_compact);
        auto copy_sfb_src =
            dae_get_utccp_smem_desc_tensor<Utccp>(copy_sfb_src_raw);
        auto copy_sfb_dst = copy_sfb_slice.partition_D(stage_sfb_compact);

        if (elect_one_sync()) {
          copy(copy_sfb, copy_sfb_src, copy_sfb_dst);
        }

#pragma unroll
        for (int output_group = 0; output_group < OutputGroups;
             ++output_group) {
          const int weight_slots = m2c.template pop<0>();
          auto *weight_base = static_cast<uint8_t *>(
              get_slot_address(smem_base, extract(weight_slots)));
          auto sA = make_tensor(
              make_smem_ptr(reinterpret_cast<Fp4 *>(weight_base)), layout_sA);
          auto tCrA = cta_mma.make_fragment_A(sA);
          auto tCsSFA = make_tensor(
              make_smem_ptr(reinterpret_cast<Scale *>(
                  weight_base + kAStorageBytes)),
              LayoutSFA{});
          auto tCsSFA_compact = make_tensor(
              tCsSFA.data(), filter_zeros(tCsSFA.layout()));
          auto stage_sfa = make_tensor<typename TiledMma::FrgTypeSFA>(
              shape(LayoutSFA{}));
          stage_sfa.data() = stage_base + output_group * sfa_columns;
          auto stage_sfa_compact = make_tensor(
              stage_sfa.data(), filter_zeros(stage_sfa.layout()));
          auto copy_sfa = make_utccp_copy(Utccp{}, stage_sfa_compact);
          auto copy_sfa_slice = copy_sfa.get_slice(0);
          auto copy_sfa_src_raw =
              copy_sfa_slice.partition_S(tCsSFA_compact);
          auto copy_sfa_src =
              dae_get_utccp_smem_desc_tensor<Utccp>(copy_sfa_src_raw);
          auto copy_sfa_dst =
              copy_sfa_slice.partition_D(stage_sfa_compact);
          if (elect_one_sync()) {
            copy(copy_sfa, copy_sfa_src, copy_sfa_dst);
          }

          auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
          group_tmem_acc.data() =
              accumulator_pipeline_base + output_group * accumulator_columns;
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            const auto accumulate = tile == 0 && k_block == 0
                ? UMMA::ScaleOut::Zero
                : UMMA::ScaleOut::One;
            gemm(
                tiled_mma.with(
                    accumulate,
                    stage_sfa(_, _, k_block),
                    stage_sfb(_, _, k_block)),
                tCrA(_, _, k_block),
                tCrB(_, _, k_block),
                group_tmem_acc);
          }
        }
        cutlass::arch::umma_arrive(
            tmem_mma_barrier + kFullBarrierBase + stage);
      } else if (warp == 1) {
        int release_slots = 0;
#pragma unroll
        for (int output_group = 0; output_group < OutputGroups;
             ++output_group) {
          release_slots |= m2c.template pop<0>();
        }
        cute::wait_barrier(
            tmem_mma_barrier[kFullBarrierBase + stage],
            ((pipeline_phase_mask >> stage) & 1U)
                ^ uint32_t(generation & 1));
        if (tile_in_chunk + 1 == chunk_tiles && !retain_activation) {
          release_slots |= activation_slots;
        }
        c2m.template push<numThreadsPerWarp>(tid, release_slots);
        if (tid == numThreadsPerWarp) {
          cuda::ptx::mbarrier_arrive(
              cuda::ptx::sem_release,
              cuda::ptx::scope_cta,
              cuda::ptx::space_shared,
              tmem_mma_barrier + kEmptyBarrierBase + stage);
        }
      } else {
#pragma unroll
        for (int output_group = 0; output_group < OutputGroups;
             ++output_group) {
          m2c.advance();
        }
      }
    }
  }

  const int final_tile = num_k_tiles - 1;
  cute::wait_barrier(
      tmem_mma_barrier[kFullBarrierBase + final_tile % kStages],
      ((pipeline_phase_mask >> (final_tile % kStages)) & 1U)
          ^ uint32_t((final_tile / kStages) & 1));
#pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    const int stage_uses = (num_k_tiles + kStages - 1 - stage) / kStages;
    if (stage_uses & 1) {
      pipeline_phase_mask ^= 1U << stage;
    }
  }

  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_compute_group(128);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  auto coord_c = make_identity_tensor(
      make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto cta_coord_c = cta_mma.partition_C(coord_c);
  using TmemLoad = SM100_TMEM_LOAD_32dp32b1x;
  auto cAcc = cta_coord_c(make_coord(_, _), _0{}, _0{});
#pragma unroll
  for (int output_group = 0; output_group < OutputGroups; ++output_group) {
    const int output_slots = m2c.template pop<0>();
    auto *output = static_cast<Output *>(
        get_slot_address(smem_base, extract(output_slots)));
    auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
    group_tmem_acc.data() =
        accumulator_pipeline_base + output_group * accumulator_columns;
    auto tAcc = group_tmem_acc(make_coord(_, _), _0{}, _0{});
    auto tiled_t2r = make_tmem_copy(TmemLoad{}, tAcc);
    const int thread_idx = tid % size(tiled_t2r);
    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    auto thread_tmem = thread_t2r.partition_S(tAcc);
    auto thread_coord = thread_t2r.partition_D(cAcc);
    auto r_acc = make_tensor<Accum>(shape(thread_coord));
    copy(tiled_t2r, thread_tmem, r_acc);
    const float group_alpha = output_group == 0 ? alpha0 : alpha1;
#pragma unroll
    for (int index = 0; index < size(r_acc); ++index) {
      const int row = int(get<0>(thread_coord(index)));
      const int col = int(get<1>(thread_coord(index)));
      if (row < kTileM && col == 0) {
        output[row] = Output(r_acc(index) * group_alpha * output_scale);
      }
    }
    __sync_compute_group(128);
    c2m.template push<0, true>(tid, output_slots);
  }

  c2m.push(tid, alpha0_slots | alpha1_slots);
  if constexpr (LoadOutputScale) {
    c2m.push(tid, output_scale_slots);
  }
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_pipeline_sm100(
    int num_k_tiles,
    int retain_activation,
    int activation_tiles_per_load,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  task_nvfp4_gemv_umma_pipeline_impl_sm100<cutlass::bfloat16_t, false, 1>(
      num_k_tiles, retain_activation, activation_tiles_per_load, smem_base,
      tmem_base_ptr, tmem_mma_barrier, pipeline_phase_mask, m2c, c2m);
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_pipeline_fp32_sm100(
    int num_k_tiles,
    int retain_activation,
    int activation_tiles_per_load,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  task_nvfp4_gemv_umma_pipeline_impl_sm100<float, true, 1>(
      num_k_tiles, retain_activation, activation_tiles_per_load, smem_base,
      tmem_base_ptr, tmem_mma_barrier, pipeline_phase_mask, m2c, c2m);
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_nvfp4_gemv_umma_pipeline_fp32_group2_sm100(
    int num_k_tiles,
    int retain_activation,
    int activation_tiles_per_load,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  task_nvfp4_gemv_umma_pipeline_impl_sm100<float, true, 2>(
      num_k_tiles, retain_activation, activation_tiles_per_load, smem_base,
      tmem_base_ptr, tmem_mma_barrier, pipeline_phase_mask, m2c, c2m);
}

// Issue one K256 operand's four K64 block-scaled UMMAs under a single elected
// lane. CUTE's scalar atom performs a warp election for every K64 block; the
// shaped path has already materialized all descriptors, so one PTX bundle can
// submit the four independent commands back-to-back.
__device__ __forceinline__ void dae_nvfp4_umma_issue_k256(
    uint32_t tmem_c,
    uint32_t first_accumulate,
    uint64_t descriptor_a,
    uint64_t descriptor_b,
    uint32_t instruction_descriptor,
    uint32_t tmem_sfa,
    uint32_t tmem_sfb) {
#if defined(CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED)
  asm volatile(
        "{\n\t"
        ".reg .pred p_first, p_one;\n\t"
        "setp.ne.b32 p_first, %5, 0;\n\t"
        "setp.ne.b32 p_one, 1, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%4], %0, %1, %6, [%2], [%3], p_first;\n\t"
        "add.u64 %0, %0, 2;\n\t"
        "add.u64 %1, %1, 2;\n\t"
        "add.u32 %2, %2, 4;\n\t"
        "add.u32 %3, %3, 4;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%4], %0, %1, %6, [%2], [%3], p_one;\n\t"
        "add.u64 %0, %0, 2;\n\t"
        "add.u64 %1, %1, 2;\n\t"
        "add.u32 %2, %2, 4;\n\t"
        "add.u32 %3, %3, 4;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%4], %0, %1, %6, [%2], [%3], p_one;\n\t"
        "add.u64 %0, %0, 2;\n\t"
        "add.u64 %1, %1, 2;\n\t"
        "add.u32 %2, %2, 4;\n\t"
        "add.u32 %3, %3, 4;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
        "[%4], %0, %1, %6, [%2], [%3], p_one;\n\t"
#else
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%4], %0, %1, %6, [%2], [%3], p_first;\n\t"
        "add.u64 %0, %0, 2;\n\t"
        "add.u64 %1, %1, 2;\n\t"
        "add.u32 %2, %2, 4;\n\t"
        "add.u32 %3, %3, 4;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%4], %0, %1, %6, [%2], [%3], p_one;\n\t"
        "add.u64 %0, %0, 2;\n\t"
        "add.u64 %1, %1, 2;\n\t"
        "add.u32 %2, %2, 4;\n\t"
        "add.u32 %3, %3, 4;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%4], %0, %1, %6, [%2], [%3], p_one;\n\t"
        "add.u64 %0, %0, 2;\n\t"
        "add.u64 %1, %1, 2;\n\t"
        "add.u32 %2, %2, 4;\n\t"
        "add.u32 %3, %3, 4;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%4], %0, %1, %6, [%2], [%3], p_one;\n\t"
#endif
        "}\n"
        : "+l"(descriptor_a), "+l"(descriptor_b),
          "+r"(tmem_sfa), "+r"(tmem_sfb)
        : "r"(tmem_c), "r"(first_accumulate),
          "r"(instruction_descriptor));
#else
  CUTE_INVALID_CONTROL_PATH(
      "Attempting K256 NVFP4 UMMA issue without SM100 MXF4 support");
#endif
}

// Shape-specialized routed Linear-1 experiment. Weight and activation data are
// packed as K512 records and still arrive through M2C-owned allocator slots.
// Their scale records live in separate contiguous HBM arrays. For the initial
// verification experiment both compact scale-array addresses arrive in a
// preloaded metadata record; raw-address operands can replace that record
// later. The compute issuer copies each K512 scale pair through a fixed
// two-entry scratch ring while the preceding UMMA remains in flight. This keeps
// the M2C/C2M ownership contract without spending allocator slots on scale
// payloads.
template <
    int StaticNumKTiles = 0,
    int StaticScaleStages = 0,
    int StaticWeightTilesPerLoad = 0,
    typename M2CQueue,
    typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_umma_k512_fp32_sm100(
    int num_k_tiles,
    int scale_stages,
    int weight_tiles_per_load,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_ue4m3_t;
  using Accum = float;

  if constexpr (StaticNumKTiles > 0) {
    num_k_tiles = StaticNumKTiles;
  }
  if constexpr (StaticScaleStages > 0) {
    scale_stages = StaticScaleStages;
  }
  if constexpr (StaticWeightTilesPerLoad > 0) {
    weight_tiles_per_load = StaticWeightTilesPerLoad;
  }

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  // One scheduling stage owns two independently described K256 operands. This
  // is the K512/u2 organization used by the shaped path: it halves M2C load
  // commands without assuming that a single K512 CUTE swizzle is bytewise
  // compatible with two checkpoint K256 records.
  constexpr int kTileK = 256;
  constexpr int kK256PerStage = 2;
  constexpr int kScaleVector = 16;
  constexpr int kStages = nvfp4UmmaPipelineStages;
  constexpr int kFullBarrierBase = nvfp4UmmaPipelineBarrierBase;
  constexpr int kEmptyBarrierBase = kFullBarrierBase + kStages;
  constexpr int kScaleBarrierBase = nvfp4ScaleCopyBarrierBase;
  constexpr int kScaleEmptyBarrierBase = 11;
  constexpr int kScratchStageBytes = 8 * 1024;
  constexpr int kScratchStages = nvfp4ScaleCopyBarrierCount;
  constexpr int kScaleScratchBytes = kScratchStages * kScratchStageBytes;
  constexpr int kTaskScratchBytes =
      dynamicSmemBytes - numSlots * slotSizeKb * 1024;
  constexpr int kSmemBaseAlignmentSlack = 1023;
  static_assert(
      kTaskScratchBytes >= kScaleScratchBytes + kSmemBaseAlignmentSlack,
      "K512 NVFP4 scale ring must fit after the allocator arena");

  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = SM100_MMA_MXF4_SS<
      Fp4, Fp4, Accum, Scale,
      kTileM, kTileN, kScaleVector,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;

  const int tid = __compute_tid();
  const int warp = tid / numThreadsPerWarp;
  const int metadata_slots = m2c.template pop<0>();
  const auto *metadata = static_cast<const uint8_t *>(
      get_slot_address(smem_base, extract(metadata_slots)));
  const float alpha = *reinterpret_cast<const float *>(metadata);
  const auto *weight_scale_base = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 16));
  const auto *activation_scale_base = reinterpret_cast<const uint8_t *>(
      *reinterpret_cast<const uint64_t *>(metadata + 24));

  if (num_k_tiles <= 0 || num_k_tiles > 8 ||
      scale_stages <= 0 || scale_stages > kScratchStages ||
      weight_tiles_per_load <= 0 || weight_tiles_per_load > 2 ||
      num_k_tiles % weight_tiles_per_load) {
    asm volatile("trap;");
  }

  int activation_slots = 0;
  if (warp < 2) {
    activation_slots = m2c.template pop<0>();
  } else {
    m2c.advance();
  }
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

  constexpr int kDescriptorAlignment = 128;
  constexpr int kABytes = (cosize_v<decltype(layout_sA)> + 1) / 2;
  constexpr int kBBytes = (cosize_v<decltype(layout_sB)> + 1) / 2;
  constexpr int kAStorageBytes =
      (kABytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kBStorageBytes =
      (kBBytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kSFABytes = cosize_v<LayoutSFA>;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  static_assert(kAStorageBytes == 16384);
  static_assert(kBStorageBytes == 1024);
  static_assert(kSFABytes == 2048);
  static_assert(kSFBBytes == 2048);
  static_assert(
      kK256PerStage * (kSFABytes + kSFBBytes) == kScratchStageBytes);
  constexpr int kWeightScaleRecordBytes = kK256PerStage * kSFABytes;
  auto *scale_scratch = static_cast<uint8_t *>(
      get_slot_address(smem_base, numSlots));

  // Launch the first scale-ring fill as soon as the preloaded address becomes
  // available. These transfers run concurrently with TMEM layout setup; the
  // producer only waits when its corresponding K tile reaches the issue loop.
  if (warp == 2 && (tid & (numThreadsPerWarp - 1)) == 0) {
#pragma unroll
    for (int scratch_stage = 0; scratch_stage < kScratchStages;
         ++scratch_stage) {
      if (scratch_stage < scale_stages && scratch_stage < num_k_tiles) {
        auto &scale_barrier =
            tmem_mma_barrier[kScaleBarrierBase + scratch_stage];
        auto *stage_scratch =
            scale_scratch + scratch_stage * kScratchStageBytes;
        cute::set_barrier_transaction_bytes(
            scale_barrier, kWeightScaleRecordBytes);
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            stage_scratch,
            weight_scale_base + scratch_stage * kWeightScaleRecordBytes,
            uint32_t(kWeightScaleRecordBytes),
            &scale_barrier);
      }
    }
  }

  auto logical_c = make_tensor(
      make_smem_ptr(static_cast<Accum *>(nullptr)),
      make_layout(
          make_shape(Int<kTileM>{}, Int<kTileN>{}),
          make_stride(Int<kTileN>{}, Int<1>{})));
  auto cta_c = cta_mma.partition_C(logical_c);
  auto tmem_acc = cta_mma.make_fragment_C(cta_c);
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
  const int scale_stage_columns =
      kK256PerStage * (sfa_columns + sfb_columns);

  DaeTaskTmemAllocator tmem_allocator(tmem_base_ptr);
  const uint32_t accumulator_pipeline_base =
      tmem_allocator.allocate(accumulator_columns);
  tmem_acc.data() = accumulator_pipeline_base;
  const uint32_t scale_pipeline_base =
      tmem_allocator.allocate(kStages * scale_stage_columns);
  if (tmem_allocator.used_columns() >
      cute::TMEM::Allocator1Sm::Sm100TmemCapacityColumns) {
    asm volatile("trap;");
  }

  // Only 32 activation scale bytes per K512 record live in HBM. Keep the
  // padded N8 SFB regions zero and update their 256 live bytes in-place for
  // each stage instead of fetching 4 KiB of replicated scale data.
  if (warp == 3) {
    const int lane = tid & (numThreadsPerWarp - 1);
#pragma unroll
    for (int scratch_stage = 0; scratch_stage < kScratchStages;
         ++scratch_stage) {
      auto *activation_scale_scratch = scale_scratch +
          scratch_stage * kScratchStageBytes + kK256PerStage * kSFABytes;
      auto *activation_scale_vectors =
          reinterpret_cast<uint4 *>(activation_scale_scratch);
      constexpr int kActivationScaleVectors =
          kK256PerStage * kSFBBytes / int(sizeof(uint4));
#pragma unroll
      for (int vector = lane; vector < kActivationScaleVectors;
           vector += numThreadsPerWarp) {
        activation_scale_vectors[vector] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  uint8_t *activation_base = nullptr;
  if (warp == 0) {
    activation_base = static_cast<uint8_t *>(
        get_slot_address(smem_base, extract(activation_slots)));
  }
  using Utccp = SM100_UTCCP_4x32dp128bit_1cta;
  constexpr int kWeightDataRecordBytes =
      kK256PerStage * kAStorageBytes;
  int live_weight_slots = 0;
  uint8_t *live_weight_base = nullptr;
#pragma unroll
  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int stage = tile % kStages;
    const int generation = tile / kStages;
    auto *stage_scratch =
        scale_scratch + (tile % scale_stages) * kScratchStageBytes;

    // Warp 0 waits only for its allocator/TMEM stage. Warps 2 and 3 prepare the
    // next weight and activation scales while warp 0 is issuing the preceding
    // stage. The compute-group rendezvous closes that producer generation
    // before any warp can start the next one; this is required by the resident
    // CTA, where a producer otherwise can lap a named barrier before warp 0
    // consumes it.
    if (warp == 0 && generation > 0) {
      const uint32_t stage_phase =
          (pipeline_phase_mask >> stage) & 1U;
      cute::wait_barrier(
          tmem_mma_barrier[kEmptyBarrierBase + stage],
          stage_phase ^ uint32_t((generation - 1) & 1));
    }
    if (warp >= 2) {
      constexpr int kPackedScaleGroups =
          kK256PerStage * kTileK / kScaleVector / 4;
      const int lane = tid & (numThreadsPerWarp - 1);
      uint32_t local_scale_group = 0;
      if (warp == 3 && lane < kPackedScaleGroups) {
        const auto *packed_activation_scale =
            reinterpret_cast<const uint32_t *>(
                activation_scale_base + tile * kPackedScaleGroups * 4);
        // This compact 32-byte read does not touch the scratch stage. Start it
        // before the empty-stage wait so global latency overlaps warp 0's
        // preceding UTCCP/UMMA issue; the value remains in warp-3 registers
        // until that stage is safe to overwrite.
        local_scale_group = packed_activation_scale[lane];
      }
      if (tile >= scale_stages) {
        if (tile % scale_stages == 0) {
          __sync_barrier_unaligned<
              kScaleEmptyBarrierBase, 3 * numThreadsPerWarp>();
        } else {
          __sync_barrier_unaligned<
              kScaleEmptyBarrierBase + 1, 3 * numThreadsPerWarp>();
        }
      }
      if (warp == 2) {
        const int scratch_stage = tile % scale_stages;
        const int scratch_generation = tile / scale_stages;
        auto &scale_barrier =
            tmem_mma_barrier[kScaleBarrierBase + scratch_stage];
        if ((tid & (numThreadsPerWarp - 1)) == 0) {
          if (tile >= scale_stages) {
            cute::set_barrier_transaction_bytes(
                scale_barrier, kWeightScaleRecordBytes);
            cuda::ptx::cp_async_bulk(
                cuda::ptx::space_shared,
                cuda::ptx::space_global,
                stage_scratch,
                weight_scale_base + tile * kWeightScaleRecordBytes,
                uint32_t(kWeightScaleRecordBytes),
                &scale_barrier);
          }
          cute::wait_barrier(
              scale_barrier,
              ((pipeline_phase_mask >> (kStages + scratch_stage)) & 1U) ^
                  uint32_t(scratch_generation & 1));
        }
        __syncwarp();
      } else {
        auto *expanded_activation_scale =
            stage_scratch + kWeightScaleRecordBytes;
        // The N8 SFB layout places each group of four consecutive scale bytes
        // at group*512 + row*16 within a K256 subtile. Broadcast one packed
        // group at a time and let lanes 0..7 write distinct row banks.
#pragma unroll
        for (int scale_group = 0;
             scale_group < kPackedScaleGroups;
             ++scale_group) {
          const uint32_t packed_scale = __shfl_sync(
              0xFFFFFFFFU, local_scale_group, scale_group);
          if (lane < kTileN) {
            const int scale_subtile = scale_group / 4;
            const int local_group = scale_group % 4;
            *reinterpret_cast<uint32_t *>(
                expanded_activation_scale + scale_subtile * kSFBBytes +
                local_group * 512 + lane * 16) = packed_scale;
          }
        }
        __syncwarp();
        cutlass::arch::fence_view_async_shared();
      }
    }
    __sync_compute_group(128);

    if (warp == 0) {
      if (tile % weight_tiles_per_load == 0) {
        live_weight_slots = m2c.template pop<0>();
        live_weight_base = static_cast<uint8_t *>(
            get_slot_address(smem_base, extract(live_weight_slots)));
      }
      auto *weight_base = live_weight_base +
          (tile % weight_tiles_per_load) * kWeightDataRecordBytes;
      // Save the mask separately: the retire warp observes the same M2C item
      // and returns it only after the corresponding completion barrier.
      const uint32_t stage_base =
          scale_pipeline_base + stage * scale_stage_columns;
      if (elect_one_sync()) {
#pragma unroll
      for (int subtile = 0; subtile < kK256PerStage; ++subtile) {
        auto sA = make_tensor(
            make_smem_ptr(reinterpret_cast<Fp4 *>(
                weight_base + subtile * kAStorageBytes)),
            layout_sA);
        auto tCrA = cta_mma.make_fragment_A(sA);
        auto *tile_activation = activation_base +
            (tile * kK256PerStage + subtile) * kBStorageBytes;
        auto sB = make_tensor(
            make_smem_ptr(reinterpret_cast<Fp4 *>(tile_activation)),
            layout_sB);
        auto tCrB = cta_mma.make_fragment_B(sB);

        const uint32_t scale_substage_base = stage_base +
            subtile * (sfa_columns + sfb_columns);
        auto stage_sfa = make_tensor<typename TiledMma::FrgTypeSFA>(
            shape(LayoutSFA{}));
        stage_sfa.data() = scale_substage_base;
        auto stage_sfb = make_tensor<typename TiledMma::FrgTypeSFB>(
            shape(LayoutSFB{}));
        stage_sfb.data() = scale_substage_base + sfa_columns;

        auto smem_sfa = make_tensor(
            make_smem_ptr(reinterpret_cast<Scale *>(
                stage_scratch + subtile * kSFABytes)),
            LayoutSFA{});
        auto smem_sfb = make_tensor(
            make_smem_ptr(reinterpret_cast<Scale *>(
                stage_scratch + kK256PerStage * kSFABytes +
                subtile * kSFBBytes)),
            LayoutSFB{});
        auto compact_sfa = make_tensor(
            stage_sfa.data(), filter_zeros(stage_sfa.layout()));
        auto compact_sfb = make_tensor(
            stage_sfb.data(), filter_zeros(stage_sfb.layout()));
        auto copy_sfa = make_utccp_copy(Utccp{}, compact_sfa);
        auto copy_sfb = make_utccp_copy(Utccp{}, compact_sfb);
        auto copy_sfa_slice = copy_sfa.get_slice(0);
        auto copy_sfb_slice = copy_sfb.get_slice(0);
        auto smem_sfa_compact = make_tensor(
            smem_sfa.data(), filter_zeros(smem_sfa.layout()));
        auto smem_sfb_compact = make_tensor(
            smem_sfb.data(), filter_zeros(smem_sfb.layout()));
        auto copy_sfa_src = dae_get_utccp_smem_desc_tensor<Utccp>(
            copy_sfa_slice.partition_S(smem_sfa_compact));
        auto copy_sfb_src = dae_get_utccp_smem_desc_tensor<Utccp>(
            copy_sfb_slice.partition_S(smem_sfb_compact));
        auto copy_sfa_dst = copy_sfa_slice.partition_D(compact_sfa);
        auto copy_sfb_dst = copy_sfb_slice.partition_D(compact_sfb);
        copy(copy_sfa, copy_sfa_src, copy_sfa_dst);
        copy(copy_sfb, copy_sfb_src, copy_sfb_dst);

        CUTE_STATIC_ASSERT_V(size<2>(tCrA) == Int<4>{});
        const uint64_t descriptor_a = tCrA(_, _, 0)[0];
        const uint64_t descriptor_b = tCrB(_, _, 0)[0];
        const uint32_t tmem_sfa_address =
            raw_pointer_cast(stage_sfa(_, _, 0).data());
        const uint32_t tmem_sfb_address =
            raw_pointer_cast(stage_sfb(_, _, 0).data());
        // All task-local scale allocations are column aligned. The four K64
        // fragments therefore advance by four TMEM columns without changing
        // the subword IDs encoded in the instruction descriptor.
        const uint32_t instruction_descriptor = uint32_t(tiled_mma.idesc_);
        const auto first_accumulate = tile == 0 && subtile == 0
            ? UMMA::ScaleOut::Zero
            : UMMA::ScaleOut::One;
        dae_nvfp4_umma_issue_k256(
            raw_pointer_cast(tmem_acc.data()),
            uint32_t(first_accumulate),
            descriptor_a,
            descriptor_b,
            instruction_descriptor,
            tmem_sfa_address,
            tmem_sfb_address);
      }
      const uint32_t completion_barrier = cute::cast_smem_ptr_to_uint(
          tmem_mma_barrier + kFullBarrierBase + stage);
      asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
          "shared::cluster.b64 [%0];"
          :: "r"(completion_barrier) : "memory");
      }
      // The SMEM scale record is no longer needed once both UTCCPs and their
      // dependent UMMAs have been issued. Reconverge the issuer warp before
      // announcing consumption: named-barrier counts are warp-granular even
      // though only the elected lane submits the UMMA bundle above.
      if (tile % scale_stages == 0) {
        __arrive_barrier_unaligned<
            kScaleEmptyBarrierBase, 3 * numThreadsPerWarp>();
      } else {
        __arrive_barrier_unaligned<
            kScaleEmptyBarrierBase + 1, 3 * numThreadsPerWarp>();
      }
    } else if (warp == 1) {
      if (tile % weight_tiles_per_load == 0) {
        live_weight_slots = m2c.template pop<0>();
      }
      cute::wait_barrier(
          tmem_mma_barrier[kFullBarrierBase + stage],
          ((pipeline_phase_mask >> stage) & 1U) ^
              uint32_t(generation & 1));
      if ((tile + 1) % weight_tiles_per_load == 0 ||
          tile + 1 == num_k_tiles) {
        int release_slots = live_weight_slots;
        if (tile + 1 == num_k_tiles) {
          release_slots |= activation_slots;
        }
        c2m.template push<numThreadsPerWarp>(tid, release_slots);
      }
      if (tid == numThreadsPerWarp) {
        cuda::ptx::mbarrier_arrive(
            cuda::ptx::sem_release,
            cuda::ptx::scope_cta,
            cuda::ptx::space_shared,
            tmem_mma_barrier + kEmptyBarrierBase + stage);
      }
    } else {
      if (tile % weight_tiles_per_load == 0) {
        m2c.advance();
      }
    }
  }

  // Complete the final scratch-consumed generations so the named barriers are
  // reset before the persistent CTA executes another task.
  if (warp >= 2) {
    if (scale_stages > 0) {
      __sync_barrier_unaligned<
          kScaleEmptyBarrierBase, 3 * numThreadsPerWarp>();
    }
    if (scale_stages > 1 && num_k_tiles > 1) {
      __sync_barrier_unaligned<
          kScaleEmptyBarrierBase + 1, 3 * numThreadsPerWarp>();
    }
  }

  const int final_tile = num_k_tiles - 1;
  cute::wait_barrier(
      tmem_mma_barrier[kFullBarrierBase + final_tile % kStages],
      ((pipeline_phase_mask >> (final_tile % kStages)) & 1U) ^
          uint32_t((final_tile / kStages) & 1));
#pragma unroll
  for (int stage = 0; stage < kStages; ++stage) {
    const int stage_uses = (num_k_tiles + kStages - 1 - stage) / kStages;
    if (stage_uses & 1) {
      pipeline_phase_mask ^= 1U << stage;
    }
  }
#pragma unroll
  for (int scratch_stage = 0; scratch_stage < kScratchStages;
       ++scratch_stage) {
    const int stage_uses = scratch_stage < scale_stages
        ? (num_k_tiles + scale_stages - 1 - scratch_stage) / scale_stages
        : 0;
    if (stage_uses & 1) {
      pipeline_phase_mask ^= 1U << (kStages + scratch_stage);
    }
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_compute_group(128);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int output_slots = m2c.template pop<0>();
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
      output[row] = r_acc(index) * alpha;
    }
  }

  __sync_compute_group(128);
  c2m.push(tid, metadata_slots);
  c2m.template push<0, true>(tid, output_slots);
}
