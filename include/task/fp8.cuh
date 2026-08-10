#pragma once

#include "context.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// Decode-time FP8 matrix-vector multiply for DeepSeek's native block-128
// checkpoint tensors.  Weights and activations are E4M3; both scale tensors
// are UE8M0.  The weight scale is shared by each logical 128x128 weight tile.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_fp8_block128_gemv_sm100(
    int rows,
    int k,
    int row_in_scale_block,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using InputFragment = cutlass::Array<Fp8, 32>;
  using FloatFragment = cutlass::Array<float, 32>;

  static_assert(sizeof(InputFragment) == 32,
                "32 FP8 values must occupy one 256-bit load");

  const int weight_slot = m2c.template pop<0>();
  const auto *weight = static_cast<const Fp8 *>(
      slot_2_glob_ptr(st_insts, weight_slot));
  const int weight_scale_slot = m2c.template pop<0>();
  const auto *weight_scale = static_cast<const Scale *>(
      slot_2_glob_ptr(st_insts, weight_scale_slot));
  const int input_slot = m2c.template pop<0>();
  const auto *input = static_cast<const Fp8 *>(
      slot_2_glob_ptr(st_insts, input_slot));
  const int input_scale_slot = m2c.template pop<0>();
  const auto *input_scale = static_cast<const Scale *>(
      slot_2_glob_ptr(st_insts, input_scale_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<cutlass::bfloat16_t *>(
      slot_2_glob_ptr(st_insts, output_slot));

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

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}
