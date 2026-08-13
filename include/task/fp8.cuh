#pragma once

#include "context.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_tmem_helper.hpp>

template <class UtccpOp, class TEngine, class TLayout>
__device__ __forceinline__ auto dae_fp8_get_utccp_smem_desc_tensor(
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

// Decode-time FP8 matrix-vector multiply for DeepSeek's native block-128
// checkpoint tensors.  Weights and activations are E4M3; both scale tensors
// are UE8M0.  The weight scale is shared by each logical 128x128 weight tile.
__device__ __forceinline__ void fp8_block128_gemv_compute_sm100(
    int rows,
    int k,
    int row_in_scale_block,
    const cutlass::float_e4m3_t *weight,
    const cutlass::float_ue8m0_t *weight_scale,
    const cutlass::float_e4m3_t *input,
    const cutlass::float_ue8m0_t *input_scale,
    cutlass::bfloat16_t *output) {
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using InputFragment = cutlass::Array<Fp8, 32>;
  using FloatFragment = cutlass::Array<float, 32>;

  static_assert(sizeof(InputFragment) == 32,
                "32 FP8 values must occupy one 256-bit load");
  const int tid = __compute_tid();
  const int lane_in_group = tid & 15;
  const int row_group = tid >> 4;
  const unsigned group_mask = (tid & 16) ? 0xffff0000U : 0x0000ffffU;
  constexpr int kRowsPerWave = 8;
  constexpr int kValuesPerFragment = 32;

  const int num_fragments = k / kValuesPerFragment;
  const int scale_k_stride = k / 128;
  cutlass::NumericArrayConverter<float, Fp8, kValuesPerFragment>
      convert_fp8;
  cutlass::NumericConverter<float, Scale> convert_scale;

  for (int local_row = row_group; local_row < rows;
       local_row += kRowsPerWave) {
    const int scale_row = row_in_scale_block + local_row;
    float partial = 0.0f;
    for (int fragment_idx = lane_in_group;
         fragment_idx < num_fragments;
         fragment_idx += 16) {
      const auto weight_fragment =
          *reinterpret_cast<const InputFragment *>(
              weight + local_row * k + fragment_idx * kValuesPerFragment);
      const auto input_fragment =
          *reinterpret_cast<const InputFragment *>(
              input + fragment_idx * kValuesPerFragment);
      const FloatFragment weight_values = convert_fp8(weight_fragment);
      const FloatFragment input_values = convert_fp8(input_fragment);
      const int scale_k = fragment_idx / 4;
      const float scale =
          convert_scale(
              weight_scale[(scale_row / 128) * scale_k_stride + scale_k]) *
          convert_scale(input_scale[scale_k]);

#pragma unroll
      for (int element = 0; element < kValuesPerFragment; ++element) {
        partial = fmaf(weight_values[element], input_values[element] * scale,
                       partial);
      }
    }

#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(group_mask, partial, offset, 16);
    }
    if (lane_in_group == 0) {
      output[local_row] = cutlass::bfloat16_t(partial);
    }
  }
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_fp8_block128_gemv_sm100(
    int rows,
    int k,
    int row_in_scale_block,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  const int weight_slots = m2c.template pop<0>();
  const int weight_slot = extract(weight_slots);
  const auto *weight = static_cast<const Fp8 *>(
      get_slot_address(smem_base, weight_slot));
  const int weight_scale_slots = m2c.template pop<0>();
  const int weight_scale_slot = extract(weight_scale_slots);
  const auto *weight_scale = static_cast<const Scale *>(
      get_slot_address(smem_base, weight_scale_slot));
  const int input_slots = m2c.template pop<0>();
  const int input_slot = extract(input_slots);
  const auto *input = static_cast<const Fp8 *>(
      get_slot_address(smem_base, input_slot));
  const int input_scale_slots = m2c.template pop<0>();
  const int input_scale_slot = extract(input_scale_slots);
  const auto *input_scale = static_cast<const Scale *>(
      get_slot_address(smem_base, input_scale_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<cutlass::bfloat16_t *>(
      get_slot_address(smem_base, output_slot));

  const int tid = __compute_tid();
  fp8_block128_gemv_compute_sm100(
      rows, k, row_in_scale_block, weight, weight_scale,
      input, input_scale, output);

  __sync_compute_group(128);
  c2m.push(
      tid,
      weight_slots | weight_scale_slots | input_slots | input_scale_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// VDCores adaptive fusion for row-sharded FP8 projections. Every projection
// SM needs the complete activation vector, so quantize the BF16 source once in
// that SM's special shared scratch and consume it immediately. Four warps
// cover independent block-128 ranges and rendezvous once before GEMV.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_fp8_block128_gemv_bf16_sm100(
    int rows,
    int k,
    int row_in_scale_block,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  constexpr int kScratchBytes =
      dynamicSmemBytes - numSlots * slotSizeKb * 1024;
  if (k <= 0 || k % 128 || k + k / 128 > kScratchBytes) {
    asm volatile("trap;");
  }

  const int weight_slots = m2c.template pop<0>();
  const auto *weight = static_cast<const Fp8 *>(
      get_slot_address(smem_base, extract(weight_slots)));
  const int weight_scale_slots = m2c.template pop<0>();
  const auto *weight_scale = static_cast<const Scale *>(
      get_slot_address(smem_base, extract(weight_scale_slots)));
  const int input_slots = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, extract(input_slots)));
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<cutlass::bfloat16_t *>(
      get_slot_address(smem_base, extract(output_slots)));

  auto *quantized = static_cast<Fp8 *>(task_scratch);
  auto *input_scale = reinterpret_cast<Scale *>(quantized + k);
  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int kValuesPerLane = 4;
  const int blocks = k / 128;

  for (int block = warp; block < blocks; block += 4) {
    float values[kValuesPerLane];
    float maximum = 0.0f;
#pragma unroll 1
    for (int item = 0; item < kValuesPerLane; ++item) {
      values[item] = __bfloat162float(
          input[block * 128 + lane + item * 32]);
      maximum = fmaxf(maximum, fabsf(values[item]));
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(
          maximum,
          __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
    }
    float scale = 0.0f;
    if (lane == 0) {
      const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
      const float exponent = ceilf(log2f(requested));
      scale = exp2f(fminf(fmaxf(exponent, -127.0f), 127.0f));
      input_scale[block] = Scale(scale);
    }
    scale = __shfl_sync(0xFFFFFFFFU, scale, 0);
#pragma unroll 1
    for (int item = 0; item < kValuesPerLane; ++item) {
      quantized[block * 128 + lane + item * 32] = Fp8(
          fminf(fmaxf(values[item] / scale, -448.0f), 448.0f));
    }
  }
  __sync_compute_group(128);

  fp8_block128_gemv_compute_sm100(
      rows, k, row_in_scale_block, weight, weight_scale,
      quantized, input_scale, output);

  __sync_compute_group(128);
  c2m.push(tid, weight_slots | weight_scale_slots | input_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Setup-only conversion from one TMA-swizzled FP8 data tile plus one raw
// block-128 UE8M0 scale into the combined native MXF8 data/scale layout.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_fp8_umma_prepack_sm100(
    int kind,
    int num_k_tiles,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Accum = float;

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kTileK = 128;
  constexpr int kScaleVector = 32;
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = SM100_MMA_MXF8F6F4_SS<
      Fp8, Fp8, Accum, Scale, kTileM, kTileN,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;

  TiledMma tiled_mma;
  auto mma_shape_a = partition_shape_A(
      tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto mma_shape_b = partition_shape_B(
      tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto layout_sA = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Fp8>{}, mma_shape_a);
  auto layout_sB = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Fp8>{}, mma_shape_b);
  using LayoutSFA = decltype(
      ScaleConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape{}));
  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));
  constexpr int kAlignment = 128;
  constexpr int kABytes = cosize_v<decltype(layout_sA)>;
  constexpr int kBBytes = cosize_v<decltype(layout_sB)>;
  constexpr int kAStorageBytes =
      (kABytes + kAlignment - 1) & -kAlignment;
  constexpr int kBStorageBytes =
      (kBBytes + kAlignment - 1) & -kAlignment;
  constexpr int kSFABytes = cosize_v<LayoutSFA>;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  constexpr int kBTileBytes = 2048;
  static_assert(kAStorageBytes + kSFABytes == 16896);
  static_assert(kBStorageBytes + kSFBBytes <= kBTileBytes);

  using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
  const auto logical_sfa =
      ScaleConfig::tile_atom_to_shape_SFA(ScaleProblemShape{});
  const auto logical_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});
  constexpr int kScalesA = kTileM * kTileK / kScaleVector;
  constexpr int kScalesB = kTileN * kTileK / kScaleVector;
  const int tid = __compute_tid();

  for (int tile = 0; tile < num_k_tiles; ++tile) {
    const int data_slots = m2c.template pop<0>();
    const auto *data = static_cast<const uint8_t *>(
        get_slot_address(smem_base, extract(data_slots)));
    const int scale_slots = m2c.template pop<0>();
    const auto *scale = static_cast<const Scale *>(
        get_slot_address(smem_base, extract(scale_slots)));
    const int output_slots = m2c.template pop<0>();
    auto *output = static_cast<uint8_t *>(
        get_slot_address(smem_base, extract(output_slots)));

    const int data_bytes = kind == 0 ? kABytes : kBBytes;
    const int scale_offset = kind == 0 ? kAStorageBytes : kBStorageBytes;
    const int tile_bytes = kind == 0
        ? kAStorageBytes + kSFABytes
        : kBTileBytes;
    for (int offset = tid * 16; offset < data_bytes; offset += 128 * 16) {
      *reinterpret_cast<uint4 *>(output + offset) =
          *reinterpret_cast<const uint4 *>(data + offset);
    }
    for (int offset = tid; offset < tile_bytes - scale_offset; offset += 128) {
      output[scale_offset + offset] = 0;
    }
    __sync_compute_group(128);

    const Scale block_scale = scale[0];
    auto *packed_scale = reinterpret_cast<Scale *>(output + scale_offset);
    if (kind == 0) {
      for (int index = tid; index < kScalesA; index += 128) {
        const int row = index / (kTileK / kScaleVector);
        const int sf = index % (kTileK / kScaleVector);
        const int dst = int(logical_sfa(row, sf * kScaleVector));
        packed_scale[dst] = block_scale;
      }
    } else {
      for (int index = tid; index < kScalesB; index += 128) {
        const int row = index / (kTileK / kScaleVector);
        const int sf = index % (kTileK / kScaleVector);
        const int dst = int(logical_sfb(row, sf * kScaleVector));
        packed_scale[dst] = block_scale;
      }
    }
    __sync_compute_group(128);
    c2m.push(tid, data_slots | scale_slots);
    c2m.template push<0, true>(tid, output_slots);
  }
}

// Quantize BF16 directly into the combined N8/K128 native MXF8 B layout.
template <int ScalePack, typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_fp8_quant_umma_b_sm100(
    int num_k_tiles,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Accum = float;

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kTileK = 128;
  constexpr int kScaleVector = 32;
  static_assert(ScalePack == 1 || ScalePack == 2 || ScalePack == 4);
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = SM100_MMA_MXF8F6F4_SS<
      Fp8, Fp8, Accum, Scale, kTileM, kTileN,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));
  using ScaleConfig = cutlass::detail::Sm1xxBlockScaledConfig<kScaleVector>;
  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));
  using ScaleProblemShape = Shape<Int<kTileM>, Int<128>, Int<kTileK>>;
  const auto logical_sfb =
      ScaleConfig::tile_atom_to_shape_SFB(ScaleProblemShape{});
  constexpr int kBBytes = kTileN * kTileK;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  constexpr int kBTileBytes = 2048;
  static_assert(kBBytes + kSFBBytes <= kBTileBytes);

  const int input_slots = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, extract(input_slots)));
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<uint8_t *>(
      get_slot_address(smem_base, extract(output_slots)));
  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(task_scratch);

  if constexpr (ScalePack == 1) {
    for (int tile = 0; tile < num_k_tiles; ++tile) {
      auto *tile_output = output + tile * kBTileBytes;
      const float value = __bfloat162float(input[tile * kTileK + tid]);
      float maximum = fabsf(value);
      for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
      }
      if (lane == 0) {
        shared[warp] = maximum;
      }
      for (int offset = tid; offset < kBTileBytes - kBBytes; offset += 128) {
        tile_output[kBBytes + offset] = 0;
      }
      __sync_compute_group(128);
      if (tid == 0) {
        maximum = fmaxf(
            fmaxf(shared[0], shared[1]), fmaxf(shared[2], shared[3]));
        const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
        const float exponent = ceilf(log2f(requested));
        shared[4] = exp2f(fminf(fmaxf(exponent, -127.0f), 127.0f));
      }
      __sync_compute_group(128);

      const Fp8 quantized = value == 0.0f
          ? Fp8(0.0f)
          : Fp8(fminf(fmaxf(value / shared[4], -448.0f), 448.0f));
      const int source_chunk = tid / 16;
      const int byte_in_chunk = tid % 16;
      for (int row = 0; row < kTileN; ++row) {
        const int destination_chunk = source_chunk ^ row;
        reinterpret_cast<Fp8 *>(tile_output)[
            row * kTileK + destination_chunk * 16 + byte_in_chunk] = quantized;
      }
      auto *packed_scale =
          reinterpret_cast<Scale *>(tile_output + kBBytes);
      const Scale block_scale = Scale(shared[4]);
      for (int index = tid; index < kTileN * (kTileK / kScaleVector);
           index += 128) {
        const int row = index / (kTileK / kScaleVector);
        const int sf = index % (kTileK / kScaleVector);
        const int dst = int(logical_sfb(row, sf * kScaleVector));
        packed_scale[dst] = block_scale;
      }
      __sync_compute_group(128);
    }
  } else {
    constexpr int kThreadsPerTile = 128 / ScalePack;
    constexpr int kValuesPerThread = ScalePack;
    constexpr int kWarpsPerTile = kThreadsPerTile / 32;
    const int tile_in_group = tid / kThreadsPerTile;
    const int tile_tid = tid % kThreadsPerTile;

    for (int group_start = 0; group_start < num_k_tiles;
         group_start += ScalePack) {
      float values[kValuesPerThread];
      float maximum = 0.0f;
      #pragma unroll
      for (int value_id = 0; value_id < kValuesPerThread; ++value_id) {
        const int element = tile_tid + value_id * kThreadsPerTile;
        const float value = __bfloat162float(
            input[(group_start + tile_in_group) * kTileK + element]);
        values[value_id] = value;
        maximum = fmaxf(maximum, fabsf(value));
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
      }
      if (lane == 0) {
        shared[warp] = maximum;
      }
      #pragma unroll
      for (int pack_tile = 0; pack_tile < ScalePack; ++pack_tile) {
        auto *tile_output = output +
            (group_start + pack_tile) * kBTileBytes;
        for (int offset = tid; offset < kBTileBytes - kBBytes;
             offset += 128) {
          tile_output[kBBytes + offset] = 0;
        }
      }
      __sync_compute_group(128);
      if (tile_tid == 0) {
        maximum = 0.0f;
        #pragma unroll
        for (int tile_warp = 0; tile_warp < kWarpsPerTile; ++tile_warp) {
          maximum = fmaxf(
              maximum,
              shared[tile_in_group * kWarpsPerTile + tile_warp]);
        }
        const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
        const float exponent = ceilf(log2f(requested));
        shared[16 + tile_in_group] =
            exp2f(fminf(fmaxf(exponent, -127.0f), 127.0f));
      }
      __sync_compute_group(128);

      auto *tile_output = output +
          (group_start + tile_in_group) * kBTileBytes;
      const float block_scale = shared[16 + tile_in_group];
      #pragma unroll
      for (int value_id = 0; value_id < kValuesPerThread; ++value_id) {
        const int element = tile_tid + value_id * kThreadsPerTile;
        const float value = values[value_id];
        const Fp8 quantized = value == 0.0f
            ? Fp8(0.0f)
            : Fp8(fminf(fmaxf(value / block_scale, -448.0f), 448.0f));
        const int source_chunk = element / 16;
        const int byte_in_chunk = element % 16;
        #pragma unroll
        for (int row = 0; row < kTileN; ++row) {
          const int destination_chunk = source_chunk ^ row;
          reinterpret_cast<Fp8 *>(tile_output)[
              row * kTileK + destination_chunk * 16 + byte_in_chunk] =
              quantized;
        }
      }
      if (tid < kTileN * ScalePack) {
        const int row = tid / ScalePack;
        const int sf = tid % ScalePack;
        auto *packed_scale = reinterpret_cast<Scale *>(
            output + group_start * kBTileBytes + kBBytes);
        const int dst = int(logical_sfb(row, sf * kScaleVector));
        packed_scale[dst] = Scale(shared[16 + sf]);
      }
      __sync_compute_group(128);
    }
  }

  c2m.push(tid, input_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Decode-time native MXF8 path. LDU streams combined activation and weight
// records through separate load ports. Each allocation already contains both
// swizzled FP8 data and its native UE8M0 scale layout, so compute sees only
// shared addresses and never resolves an HBM pointer.
template <int ScalePack, int OutputGroups, bool SplitK, typename SplitOutput,
          typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_fp8_gemv_umma_stream_impl_sm100(
    int num_k_tiles,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    uint32_t &fp8_umma_pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Accum = float;
  using Output = cutlass::bfloat16_t;

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kTileK = 128;
  constexpr int kScaleVector = 32;
  constexpr int kActivationTilesPerChunk = ScalePack == 1 ? 4 : 8;
  static_assert(ScalePack == 1 || ScalePack == 2 || ScalePack == 4);
  static_assert(OutputGroups == 1 || OutputGroups == 2);
  static_assert(ScalePack != 1 || OutputGroups == 1);
  static_assert(kActivationTilesPerChunk % ScalePack == 0);
  using TileShape = Shape<Int<kTileM>, Int<kTileN>, Int<kTileK>>;
  using Atom = SM100_MMA_MXF8F6F4_SS<
      Fp8, Fp8, Accum, Scale, kTileM, kTileN,
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
      UMMA::Layout_K_SW128_Atom<Fp8>{}, mma_shape_a);
  auto layout_sB = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Fp8>{}, mma_shape_b);
  using LayoutSFA = decltype(
      ScaleConfig::deduce_smem_layoutSFA(TiledMma{}, TileShape{}));
  using LayoutSFB = decltype(
      ScaleConfig::deduce_smem_layoutSFB(TiledMma{}, TileShape{}));

  constexpr int kDescriptorAlignment = 128;
  constexpr int kABytes = cosize_v<decltype(layout_sA)>;
  constexpr int kBBytes = cosize_v<decltype(layout_sB)>;
  constexpr int kAStorageBytes =
      (kABytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kBStorageBytes =
      (kBBytes + kDescriptorAlignment - 1) & -kDescriptorAlignment;
  constexpr int kSFABytes = cosize_v<LayoutSFA>;
  constexpr int kSFBBytes = cosize_v<LayoutSFB>;
  constexpr int kWeightTileBytes = kAStorageBytes + kSFABytes;
  constexpr int kActivationTileBytes = 2048;
  static_assert(kWeightTileBytes == 16896);
  static_assert(kBStorageBytes + kSFBBytes <= kActivationTileBytes);

  auto logical_c = make_tensor(
      make_smem_ptr(static_cast<Accum *>(nullptr)),
      make_layout(
          make_shape(Int<kTileM>{}, Int<kTileN>{}),
          make_stride(Int<kTileN>{}, Int<1>{})));
  auto cta_c = cta_mma.partition_C(logical_c);
  auto tmem_acc = cta_mma.make_fragment_C(cta_c);
  tmem_acc.data() = tmem_base_ptr;
  const int accumulator_columns = int(
      cutlass::detail::find_tmem_tensor_col_offset(tmem_acc));

  auto tCtSFA = make_tensor<typename TiledMma::FrgTypeSFA>(
      shape(LayoutSFA{}));
  auto tCtSFB = make_tensor<typename TiledMma::FrgTypeSFB>(
      shape(LayoutSFB{}));
  tCtSFA.data() =
      tmem_base_ptr + accumulator_columns * OutputGroups;
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

  const int tid = __compute_tid();
  const int warp = tid / numThreadsPerWarp;
  if constexpr (ScalePack == 1) {
    // Preserve the known-good task exactly for compatibility and regression
    // isolation. Packed variants below use the same operands and epilogue but
    // split issue from retirement so four K128 groups can remain in flight.
    for (int chunk_start = 0; chunk_start < num_k_tiles;
         chunk_start += kActivationTilesPerChunk) {
      const int activation_slots = m2c.template pop<0>();
      auto *activation_chunk_base = static_cast<uint8_t *>(
          get_slot_address(smem_base, extract(activation_slots)));
      const int remaining = num_k_tiles - chunk_start;
      const int chunk_tiles = remaining < kActivationTilesPerChunk
          ? remaining
          : kActivationTilesPerChunk;
      for (int tile_in_chunk = 0; tile_in_chunk < chunk_tiles;
           ++tile_in_chunk) {
        const int tile = chunk_start + tile_in_chunk;
        const int weight_slots = m2c.template pop<0>();
        auto *weight_base = static_cast<uint8_t *>(
            get_slot_address(smem_base, extract(weight_slots)));
        auto sA = make_tensor(
            make_smem_ptr(reinterpret_cast<Fp8 *>(weight_base)), layout_sA);
        auto tCrA = cta_mma.make_fragment_A(sA);
        auto tCsSFA = make_tensor(
            make_smem_ptr(reinterpret_cast<Scale *>(
                weight_base + kAStorageBytes)),
            LayoutSFA{});
        auto tCsSFA_compact = make_tensor(
            tCsSFA.data(), filter_zeros(tCsSFA.layout()));
        auto copy_sfa_src_raw = copy_sfa_slice.partition_S(tCsSFA_compact);
        auto copy_sfa_src =
            dae_fp8_get_utccp_smem_desc_tensor<Utccp>(copy_sfa_src_raw);

        auto *activation_base = activation_chunk_base +
            tile_in_chunk * kActivationTileBytes;
        auto sB = make_tensor(
            make_smem_ptr(reinterpret_cast<Fp8 *>(activation_base)), layout_sB);
        auto tCrB = cta_mma.make_fragment_B(sB);
        auto tCsSFB = make_tensor(
            make_smem_ptr(reinterpret_cast<Scale *>(
                activation_base + kBStorageBytes)),
            LayoutSFB{});
        auto tCsSFB_compact = make_tensor(
            tCsSFB.data(), filter_zeros(tCsSFB.layout()));
        auto copy_sfb_src_raw = copy_sfb_slice.partition_S(tCsSFB_compact);
        auto copy_sfb_src =
            dae_fp8_get_utccp_smem_desc_tensor<Utccp>(copy_sfb_src_raw);

        if (tid < 32 && elect_one_sync()) {
          copy(copy_sfa, copy_sfa_src, copy_sfa_dst);
          copy(copy_sfb, copy_sfb_src, copy_sfb_dst);
        }
        if (tid < 32) {
          for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            const auto accumulate = tile == 0 && k_block == 0
                ? UMMA::ScaleOut::Zero
                : UMMA::ScaleOut::One;
            gemm(
                tiled_mma.with(
                    accumulate,
                    tCtSFA(_, _, k_block),
                    tCtSFB(_, _, k_block)),
                tCrA(_, _, k_block),
                tCrB(_, _, k_block),
                tmem_acc);
          }
          cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
        c2m.push(tid, weight_slots);
      }
      c2m.push(tid, activation_slots);
    }
  } else {
    constexpr int kStages = fp8UmmaPipelineStages;
    constexpr int kFullBarrierBase = fp8UmmaPipelineBarrierBase;
    constexpr int kEmptyBarrierBase = kFullBarrierBase + kStages;

    int live_activation_slots = 0;
    int live_group_weight_slots = 0;
    int pipeline_group = 0;
    for (int chunk_start = 0; chunk_start < num_k_tiles;
         chunk_start += kActivationTilesPerChunk) {
      if (warp < 2) {
        live_activation_slots = m2c.template pop<0>();
      } else {
        m2c.advance();
      }
      auto *activation_chunk_base = warp == 0
          ? static_cast<uint8_t *>(
                get_slot_address(smem_base, extract(live_activation_slots)))
          : nullptr;
      const int remaining = num_k_tiles - chunk_start;
      const int chunk_tiles = remaining < kActivationTilesPerChunk
          ? remaining
          : kActivationTilesPerChunk;

      for (int scale_start = 0; scale_start < chunk_tiles;
           scale_start += ScalePack) {
        #pragma unroll
        for (int output_group = 0; output_group < OutputGroups;
             ++output_group) {
          const int stage = pipeline_group % kStages;
          const int generation = pipeline_group / kStages;
          for (int scale_id = 0; scale_id < ScalePack; ++scale_id) {
            const int tile_in_chunk = scale_start + scale_id;
            const int tile = chunk_start + tile_in_chunk;

            if (warp == 0) {
              if (scale_id == 0 && generation > 0 && output_group == 0) {
                const uint32_t stage_phase =
                    (fp8_umma_pipeline_phase_mask >> stage) & 1U;
                cute::wait_barrier(
                    tmem_mma_barrier[kEmptyBarrierBase + stage],
                    stage_phase ^ uint32_t((generation - 1) & 1));
              }
              const int weight_slots = m2c.template pop<0>();
              auto *weight_base = static_cast<uint8_t *>(
                  get_slot_address(smem_base, extract(weight_slots)));
              auto sA = make_tensor(
                  make_smem_ptr(reinterpret_cast<Fp8 *>(weight_base)),
                  layout_sA);
              auto tCrA = cta_mma.make_fragment_A(sA);

              auto *activation_base = activation_chunk_base +
                  tile_in_chunk * kActivationTileBytes;
              auto sB = make_tensor(
                  make_smem_ptr(reinterpret_cast<Fp8 *>(activation_base)),
                  layout_sB);
              auto tCrB = cta_mma.make_fragment_B(sB);

              if (scale_id == 0 && elect_one_sync()) {
                auto tCsSFA = make_tensor(
                    make_smem_ptr(reinterpret_cast<Scale *>(
                        weight_base + kAStorageBytes)),
                    LayoutSFA{});
                auto tCsSFA_compact = make_tensor(
                    tCsSFA.data(), filter_zeros(tCsSFA.layout()));
                auto copy_sfa_src_raw =
                    copy_sfa_slice.partition_S(tCsSFA_compact);
                auto copy_sfa_src =
                    dae_fp8_get_utccp_smem_desc_tensor<Utccp>(
                        copy_sfa_src_raw);
                copy(copy_sfa, copy_sfa_src, copy_sfa_dst);

                if (output_group == 0) {
                  auto tCsSFB = make_tensor(
                      make_smem_ptr(reinterpret_cast<Scale *>(
                          activation_base + kBStorageBytes)),
                      LayoutSFB{});
                  auto tCsSFB_compact = make_tensor(
                      tCsSFB.data(), filter_zeros(tCsSFB.layout()));
                  auto copy_sfb_src_raw =
                      copy_sfb_slice.partition_S(tCsSFB_compact);
                  auto copy_sfb_src =
                      dae_fp8_get_utccp_smem_desc_tensor<Utccp>(
                          copy_sfb_src_raw);
                  copy(copy_sfb, copy_sfb_src, copy_sfb_dst);
                }
              }

              auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
              group_tmem_acc.data() =
                  tmem_base_ptr + output_group * accumulator_columns;
              for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
                const auto accumulate = tile == 0 && k_block == 0
                    ? UMMA::ScaleOut::Zero
                    : UMMA::ScaleOut::One;
                gemm(
                    tiled_mma.with(
                        accumulate,
                        tCtSFA(_, _, scale_id),
                        tCtSFB(_, _, scale_id)),
                    tCrA(_, _, k_block),
                    tCrB(_, _, k_block),
                    group_tmem_acc);
              }
              if (scale_id + 1 == ScalePack &&
                  output_group + 1 == OutputGroups) {
                cutlass::arch::umma_arrive(
                    tmem_mma_barrier + kFullBarrierBase + stage);
              }
            } else if (warp == 1) {
              const int weight_slots = m2c.template pop<0>();
              live_group_weight_slots |= weight_slots;
              if (scale_id + 1 == ScalePack &&
                  output_group + 1 == OutputGroups) {
                cute::wait_barrier(
                    tmem_mma_barrier[kFullBarrierBase + stage],
                    ((fp8_umma_pipeline_phase_mask >> stage) & 1U)
                        ^ uint32_t(generation & 1));
                int release_slots = live_group_weight_slots;
                if (output_group + 1 == OutputGroups &&
                    tile_in_chunk + 1 == chunk_tiles) {
                  release_slots |= live_activation_slots;
                }
                c2m.template push<numThreadsPerWarp>(tid, release_slots);
                live_group_weight_slots = 0;
                if (tid == numThreadsPerWarp) {
                  cuda::ptx::mbarrier_arrive(
                      cuda::ptx::sem_release,
                      cuda::ptx::scope_cta,
                      cuda::ptx::space_shared,
                      tmem_mma_barrier + kEmptyBarrierBase + stage);
                }
              }
            } else {
              m2c.advance();
            }
          }
        }
        ++pipeline_group;
      }
    }

    const int num_groups = num_k_tiles / ScalePack;
    const int final_group = num_groups - 1;
    cute::wait_barrier(
        tmem_mma_barrier[kFullBarrierBase + final_group % kStages],
        ((fp8_umma_pipeline_phase_mask >> (final_group % kStages)) & 1U)
            ^ uint32_t((final_group / kStages) & 1));
    #pragma unroll
    for (int stage = 0; stage < kStages; ++stage) {
      const int stage_uses = (num_groups + kStages - 1 - stage) / kStages;
      if (stage_uses & 1) {
        fp8_umma_pipeline_phase_mask ^= 1U << stage;
      }
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
    auto *output_base = get_slot_address(smem_base, extract(output_slots));
    auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
    group_tmem_acc.data() =
        tmem_base_ptr + output_group * accumulator_columns;
    auto tAcc = group_tmem_acc(make_coord(_, _), _0{}, _0{});
    auto tiled_t2r = make_tmem_copy(TmemLoad{}, tAcc);
    const int thread_idx = tid % size(tiled_t2r);
    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    auto thread_tmem = thread_t2r.partition_S(tAcc);
    auto thread_coord = thread_t2r.partition_D(cAcc);
    auto r_acc = make_tensor<Accum>(shape(thread_coord));
    copy(tiled_t2r, thread_tmem, r_acc);
    for (int index = 0; index < size(r_acc); ++index) {
      const int row = int(get<0>(thread_coord(index)));
      const int col = int(get<1>(thread_coord(index)));
      if (row < kTileM && col == 0) {
        if constexpr (SplitK) {
          static_cast<SplitOutput *>(output_base)[row] =
              SplitOutput(r_acc(index));
        } else {
          auto *output = static_cast<Output *>(output_base);
          output[row] = Output(r_acc(index));
        }
      }
    }
    __sync_compute_group(128);
    c2m.template push<0, true>(tid, output_slots);
  }
}

template <int ScalePack, int OutputGroups, typename M2CQueue,
          typename C2MQueue>
__device__ __forceinline__ void task_fp8_gemv_umma_stream_sm100(
    int num_k_tiles,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    uint32_t &fp8_umma_pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  task_fp8_gemv_umma_stream_impl_sm100<
      ScalePack, OutputGroups, false, float>(
      num_k_tiles, smem_base, tmem_base_ptr, tmem_mma_barrier,
      tmem_mma_phase, fp8_umma_pipeline_phase_mask, m2c, c2m);
}

template <int ScalePack, int OutputGroups, typename SplitOutput,
          typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_fp8_gemv_umma_splitk_sm100(
    int num_k_tiles,
    void *smem_base,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    uint32_t &fp8_umma_pipeline_phase_mask,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  task_fp8_gemv_umma_stream_impl_sm100<
      ScalePack, OutputGroups, true, SplitOutput>(
      num_k_tiles, smem_base, tmem_base_ptr, tmem_mma_barrier,
      tmem_mma_phase, fp8_umma_pipeline_phase_mask, m2c, c2m);
}
