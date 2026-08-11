#pragma once

#include "context.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cutlass/array.h>
#include <cutlass/bfloat16.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cuda_fp16.h>

// Decode-time NVFP4 matrix-vector multiply for ModelOpt checkpoints.
//
// The checkpoint representation is deliberately consumed without a global
// layout conversion:
//   weight       [M, K / 2] uint8, two E2M1 values per byte
//   weight_scale [M, K / 16] E4M3
//   activation   [K / 2] uint8, two E2M1 values per byte
//   input_scale  [K / 16] E4M3
//   alpha        scalar float32 (weight_scale_2 * input_scale)
//
// The memory virtual core loads the selected expert shard and the activation
// operands into allocator-owned shared slots.  This task never dereferences a
// global pointer; routing and address resolution remain entirely in LDU.
//
// This helper is a register-compact adaptation of the packed conversion path
// in CUTLASS GemvBlockScaled.  One 32-bit word contains eight E2M1 values; the
// native SM100 conversion produces four F16x2 pairs which are multiplied and
// reduced before the next packed word is expanded.
__device__ __forceinline__ uint32_t nvfp4_dot8x2_sm100(
    uint32_t lhs,
    uint32_t rhs) {
  uint32_t result;
  asm volatile(
      "{\n"
      ".reg .b8 a0, a1, a2, a3;\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      ".reg .f16x2 ah0, ah1, ah2, ah3;\n"
      ".reg .f16x2 bh0, bh1, bh2, bh3;\n"
      ".reg .f16x2 p0, p1, p2, p3;\n"
      "mov.b32 {a0, a1, a2, a3}, %1;\n"
      "mov.b32 {b0, b1, b2, b3}, %2;\n"
      "cvt.rn.f16x2.e2m1x2 ah0, a0;\n"
      "cvt.rn.f16x2.e2m1x2 ah1, a1;\n"
      "cvt.rn.f16x2.e2m1x2 ah2, a2;\n"
      "cvt.rn.f16x2.e2m1x2 ah3, a3;\n"
      "cvt.rn.f16x2.e2m1x2 bh0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 bh1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 bh2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 bh3, b3;\n"
      "mov.b32 p0, 0;\n"
      "mov.b32 p1, 0;\n"
      "mov.b32 p2, 0;\n"
      "mov.b32 p3, 0;\n"
      "fma.rn.f16x2 p0, ah0, bh0, p0;\n"
      "fma.rn.f16x2 p1, ah1, bh1, p1;\n"
      "fma.rn.f16x2 p2, ah2, bh2, p2;\n"
      "fma.rn.f16x2 p3, ah3, bh3, p3;\n"
      "add.rn.f16x2 p0, p0, p1;\n"
      "add.rn.f16x2 p2, p2, p3;\n"
      "add.rn.f16x2 p0, p0, p2;\n"
      "mov.b32 %0, p0;\n"
      "}\n"
      : "=r"(result)
      : "r"(lhs), "r"(rhs));
  return result;
}

__device__ __forceinline__ void nvfp4_scale_pair_sm100(
    uint16_t lhs,
    uint16_t rhs,
    float &scale0,
    float &scale1) {
  asm volatile(
      "{\n"
      ".reg .f16x2 ah, bh;\n"
      ".reg .f16 a0, a1, b0, b1;\n"
      ".reg .f32 af0, af1, bf0, bf1;\n"
      "cvt.rn.f16x2.e4m3x2 ah, %2;\n"
      "cvt.rn.f16x2.e4m3x2 bh, %3;\n"
      "mov.b32 {a0, a1}, ah;\n"
      "mov.b32 {b0, b1}, bh;\n"
      "cvt.f32.f16 af0, a0;\n"
      "cvt.f32.f16 af1, a1;\n"
      "cvt.f32.f16 bf0, b0;\n"
      "cvt.f32.f16 bf1, b1;\n"
      "mul.rn.f32 %0, af0, bf0;\n"
      "mul.rn.f32 %1, af1, bf1;\n"
      "}\n"
      : "=f"(scale0), "=f"(scale1)
      : "h"(lhs), "h"(rhs));
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_nvfp4_gemv_sm100(
    int rows,
    int k,
    int retain_activation,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Fp4 = cutlass::float_e2m1_t;
  using Scale = cutlass::float_e4m3_t;
  using PackedFragment = cutlass::Array<Fp4, 32>;

  static_assert(sizeof(PackedFragment) == 16,
                "32 packed FP4 values must occupy one 128-bit load");

  const int weight_slots = m2c.template pop<0>();
  const int weight_slot = extract(weight_slots);
  const auto *weight = static_cast<const uint8_t *>(
      get_slot_address(smem_base, weight_slot));
  const int weight_scale_slots = m2c.template pop<0>();
  const int weight_scale_slot = extract(weight_scale_slots);
  const auto *weight_scale = static_cast<const Scale *>(
      get_slot_address(smem_base, weight_scale_slot));
  const int input_slots = m2c.template pop<0>();
  const int input_slot = extract(input_slots);
  const auto *input = static_cast<const uint8_t *>(
      get_slot_address(smem_base, input_slot));
  const int input_scale_slots = m2c.template pop<0>();
  const int input_scale_slot = extract(input_scale_slots);
  const auto *input_scale = static_cast<const Scale *>(
      get_slot_address(smem_base, input_scale_slot));
  const int alpha_slots = m2c.template pop<0>();
  const int alpha_slot = extract(alpha_slots);
  const auto *alpha_ptr = static_cast<const float *>(
      get_slot_address(smem_base, alpha_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<cutlass::bfloat16_t *>(
      get_slot_address(smem_base, output_slot));

  const int tid = __compute_tid();
  constexpr int kThreadsPerRow = 8;
  constexpr int kRowsPerWave = 128 / kThreadsPerRow;
  const int lane_in_group = tid & (kThreadsPerRow - 1);
  const int row_group = tid / kThreadsPerRow;
  constexpr int kValuesPerFragment = 32;
  const unsigned group_mask = 0xffU << (tid & 24);

  const int packed_row_stride = k >> 1;
  const int scale_row_stride = k >> 4;
  const int num_fragments = k / kValuesPerFragment;
  const float alpha = *alpha_ptr;

  for (int row = row_group; row < rows; row += kRowsPerWave) {
    float partial = 0.0f;
    for (int fragment_idx = lane_in_group;
         fragment_idx < num_fragments;
         fragment_idx += kThreadsPerRow) {
      const int packed_offset = fragment_idx * sizeof(PackedFragment);
      const auto weight_fragment =
          *reinterpret_cast<const PackedFragment *>(
              weight + row * packed_row_stride + packed_offset);
      const auto input_fragment =
          *reinterpret_cast<const PackedFragment *>(input + packed_offset);
      const auto *weight_words =
          reinterpret_cast<const uint32_t *>(&weight_fragment);
      const auto *input_words =
          reinterpret_cast<const uint32_t *>(&input_fragment);

      const int sf = fragment_idx * 2;
      const uint16_t weight_scale_pair =
          *reinterpret_cast<const uint16_t *>(
              weight_scale + row * scale_row_stride + sf);
      const uint16_t input_scale_pair =
          *reinterpret_cast<const uint16_t *>(input_scale + sf);
      float scale0;
      float scale1;
      nvfp4_scale_pair_sm100(
          weight_scale_pair, input_scale_pair, scale0, scale1);

      const uint32_t dot0_bits0 =
          nvfp4_dot8x2_sm100(weight_words[0], input_words[0]);
      const uint32_t dot0_bits1 =
          nvfp4_dot8x2_sm100(weight_words[1], input_words[1]);
      const uint32_t dot1_bits0 =
          nvfp4_dot8x2_sm100(weight_words[2], input_words[2]);
      const uint32_t dot1_bits1 =
          nvfp4_dot8x2_sm100(weight_words[3], input_words[3]);
      const __half2 dot0_pairs = __hadd2(
          *reinterpret_cast<const __half2 *>(&dot0_bits0),
          *reinterpret_cast<const __half2 *>(&dot0_bits1));
      const __half2 dot1_pairs = __hadd2(
          *reinterpret_cast<const __half2 *>(&dot1_bits0),
          *reinterpret_cast<const __half2 *>(&dot1_bits1));
      const float dot0 =
          __half2float(__low2half(dot0_pairs)) +
          __half2float(__high2half(dot0_pairs));
      const float dot1 =
          __half2float(__low2half(dot1_pairs)) +
          __half2float(__high2half(dot1_pairs));
      partial = fmaf(dot0, scale0, partial);
      partial = fmaf(dot1, scale1, partial);
    }

#pragma unroll
    for (int offset = kThreadsPerRow / 2; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(
          group_mask, partial, offset, kThreadsPerRow);
    }
    if (lane_in_group == 0) {
      output[row] = cutlass::bfloat16_t(partial * alpha);
    }
  }

  __sync_compute_group(128);
  int release_slots = weight_slots | weight_scale_slots | alpha_slots;
  if (!retain_activation) {
    release_slots |= input_slots | input_scale_slots;
  }
  c2m.push(tid, release_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}
