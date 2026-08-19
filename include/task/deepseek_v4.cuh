#pragma once

#include "context.cuh"
#include "rms_norm.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cuda_bf16.h>
#include <cutlass/numeric_types.h>

__device__ __forceinline__ float dsv4_sigmoid(float value) {
  return 1.0f / (1.0f + __expf(-value));
}

__device__ __forceinline__ float dsv4_softplus(float value) {
  return fmaxf(value, 0.0f) + log1pf(__expf(-fabsf(value)));
}

struct Dsv4Bf16x8 {
  uint4 raw;
};

enum class Dsv4HcPreRmsMetadataMode : int {
  SeparateShared = 0,
  PackedShared = 1,
  PackedRaw = 2,
};

// The production image has one mHC-pre/RMS transport contract.  Keep the
// choice compile-time so alternate experiments do not leave a mode branch in
// the resident compute handler.
inline constexpr Dsv4HcPreRmsMetadataMode dsv4HcPreRmsMetadataMode =
    Dsv4HcPreRmsMetadataMode::PackedRaw;
inline constexpr int dsv4HcPreRmsSinkhornIters = 20;
inline constexpr float dsv4HcPreRmsEpsilon = 1.0e-6f;
inline constexpr float dsv4HcPreRmsNormEpsilon = 1.0e-6f;
inline constexpr int dsv4HcPreRmsScaleOffset = 28;
inline constexpr int dsv4HcPreRmsBaseOffset = 31;

__device__ __forceinline__ Dsv4Bf16x8 dsv4_load_bf16x8(
    const __nv_bfloat16 *pointer) {
  Dsv4Bf16x8 value;
  value.raw = *reinterpret_cast<const uint4 *>(pointer);
  return value;
}

__device__ __forceinline__ __nv_bfloat162 dsv4_bf16x8_pair(
    const Dsv4Bf16x8 &value,
    int pair) {
  const uint32_t bits = pair == 0 ? value.raw.x :
                        pair == 1 ? value.raw.y :
                        pair == 2 ? value.raw.z : value.raw.w;
  __nv_bfloat162 result;
  *reinterpret_cast<uint32_t *>(&result) = bits;
  return result;
}

__device__ __forceinline__ void dsv4_store_bf16x8(
    __nv_bfloat16 *pointer,
    const Dsv4Bf16x8 &value) {
  *reinterpret_cast<uint4 *>(pointer) = value.raw;
}

__device__ __forceinline__ float dsv4_hc_mix_hidden_vector(
    const __nv_bfloat16 *residual,
    const float *pre,
    __nv_bfloat16 *hidden,
    int dim) {
  constexpr int kHc = 4;
  constexpr int kHidden = 4096;
  constexpr int kVectorWidth = 8;
  float values[kVectorWidth] = {};
#pragma unroll
  for (int branch = 0; branch < kHc; ++branch) {
    const Dsv4Bf16x8 packed = dsv4_load_bf16x8(
        residual + branch * kHidden + dim);
#pragma unroll
    for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
      const float2 branch_values = __bfloat1622float2(
          dsv4_bf16x8_pair(packed, pair));
      values[pair * 2] = fmaf(
          pre[branch], branch_values.x, values[pair * 2]);
      values[pair * 2 + 1] = fmaf(
          pre[branch], branch_values.y, values[pair * 2 + 1]);
    }
  }
  Dsv4Bf16x8 rounded;
  auto *rounded_pairs = reinterpret_cast<__nv_bfloat162 *>(&rounded.raw);
  float sum_squares = 0.0f;
#pragma unroll
  for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
    rounded_pairs[pair] = __float22bfloat162_rn(
        make_float2(values[pair * 2], values[pair * 2 + 1]));
    const float2 rounded_values = __bfloat1622float2(rounded_pairs[pair]);
    sum_squares = fmaf(
        rounded_values.x, rounded_values.x, sum_squares);
    sum_squares = fmaf(
        rounded_values.y, rounded_values.y, sum_squares);
  }
  dsv4_store_bf16x8(hidden + dim, rounded);
  return sum_squares;
}

__device__ __forceinline__ float dsv4_div_rn(
    float numerator, float denominator) {
  float result;
  asm volatile("div.rn.f32 %0, %1, %2;"
               : "=f"(result) : "f"(numerator), "f"(denominator));
  return result;
}

__device__ __forceinline__ float dsv4_ceil_e4m3(float value) {
  value = fminf(fmaxf(value, 0x1p-9f), 448.0f);
  if (value < 0x1p-6f) {
    return ceilf(value * 512.0f) / 512.0f;
  }
  float exponent = floorf(log2f(value));
  float mantissa = ceilf((value / exp2f(exponent) - 1.0f) * 8.0f);
  if (mantissa >= 8.0f) {
    exponent += 1.0f;
    mantissa = 0.0f;
  }
  return exp2f(exponent) * (1.0f + mantissa / 8.0f);
}

__device__ __forceinline__ uint8_t dsv4_nearest_fp4(float value) {
  const float magnitude = fabsf(value);
  int code = int(magnitude > 0.25f) + int(magnitude > 0.75f) +
             int(magnitude > 1.25f) + int(magnitude > 1.75f) +
             int(magnitude > 2.5f) + int(magnitude > 3.5f) +
             int(magnitude > 5.0f);
  if (value < 0.0f && code != 0) {
    code += 8;
  }
  return static_cast<uint8_t>(code);
}

// Quantize one BF16 activation vector to the E4M3/UE8M0 block-128 contract
// consumed by the non-expert FP8 GEMV task.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_fp8_quant128(
    int k,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;

  const int input_slots = m2c.template pop<0>();
  const int input_slot = extract(input_slots);
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, input_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<Fp8 *>(
      get_slot_address(smem_base, output_slot));
  const int scale_slots = m2c.template pop<0>();
  const int scale_slot = extract(scale_slots);
  auto *scales = static_cast<Scale *>(
      get_slot_address(smem_base, scale_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(task_scratch);
  for (int block = 0; block < k / 128; ++block) {
    const float value = __bfloat162float(input[block * 128 + tid]);
    float maximum = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      maximum = fmaxf(
          maximum,
          __shfl_down_sync(0xFFFFFFFFU, maximum, offset));
    }
    if (lane == 0) {
      shared[warp] = maximum;
    }
    __sync_compute_group(128);
    if (tid == 0) {
      maximum = fmaxf(
          fmaxf(shared[0], shared[1]), fmaxf(shared[2], shared[3]));
      const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
      const float exponent = ceilf(log2f(requested));
      shared[0] = exp2f(fminf(fmaxf(exponent, -127.0f), 127.0f));
      scales[block] = Scale(shared[0]);
    }
    __sync_compute_group(128);
    output[block * 128 + tid] = Fp8(
        fminf(fmaxf(value / shared[0], -448.0f), 448.0f));
    __sync_compute_group(128);
  }

  c2m.push(tid, input_slots);
  c2m.template push<31, true, false>(tid, output_slots);
  c2m.template push<31, true, false>(tid, scale_slots);
}

// Quantize one BF16 activation vector to ModelOpt's packed E2M1/per-16 E4M3
// representation using the checkpoint-provided scalar dequantization scale.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_nvfp4_quant16(
    int k,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Scale = cutlass::float_e4m3_t;

  const int input_slots = m2c.template pop<0>();
  const int input_slot = extract(input_slots);
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, input_slot));
  const int global_scale_slots = m2c.template pop<0>();
  const int global_scale_slot = extract(global_scale_slots);
  const auto *global_scale = static_cast<const float *>(
      get_slot_address(smem_base, global_scale_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<uint8_t *>(
      get_slot_address(smem_base, output_slot));
  const int scale_slots = m2c.template pop<0>();
  const int scale_slot = extract(scale_slots);
  auto *scales = static_cast<Scale *>(
      get_slot_address(smem_base, scale_slot));

  constexpr int kThreadsPerBlock = 8;
  constexpr int kBlocksPerComputeGroup = 128 / kThreadsPerBlock;
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

  for (int block = block_group; block < k / 16;
       block += kBlocksPerComputeGroup) {
    const __nv_bfloat162 pair = input_pairs[block * 8 + block_lane];
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
      scales[block] = Scale(block_scale);
      quant_denominators[block_group] = block_scale * model_scale;
    }
    __syncwarp(block_mask);
    const float quant_denominator = quant_denominators[block_group];
    const uint8_t low = dsv4_nearest_fp4(
        dsv4_div_rn(values.x, quant_denominator));
    const uint8_t high = dsv4_nearest_fp4(
        dsv4_div_rn(values.y, quant_denominator));
    output[block * 8 + block_lane] = low | (high << 4);
  }
  __sync_compute_group(128);

  c2m.push(tid, input_slots | global_scale_slots);
  c2m.template push<31, true, false>(tid, output_slots);
  c2m.template push<31, true, false>(tid, scale_slots);
}

// A resident block keeps immutable RoPE metadata in the high end of its fixed
// task scratch.  The production image's largest low-end scratch user is the
// 8-KiB top-k workspace, leaving a wide gap before these four 256-byte tables.
static constexpr int kDsv4RopeTableElements = 32 * 2;
static constexpr int kDsv4RopeTableBytes =
    kDsv4RopeTableElements * sizeof(float);
static constexpr int kDsv4MaxResidentRopeTables = 4;
static constexpr int kDsv4ResidentRopeMetadataBytes =
    kDsv4MaxResidentRopeTables * kDsv4RopeTableBytes;
static constexpr int kDsv4TaskScratchBytes =
    dynamicSmemBytes - numSlots * slotSizeKb * 1024;
#ifndef DAE_DSV4_ROPE_METADATA_OFFSET_KB
#define DAE_DSV4_ROPE_METADATA_OFFSET_KB 16
#endif
static constexpr int kDsv4ResidentRopeMetadataOffset =
    DAE_DSV4_ROPE_METADATA_OFFSET_KB * 1024;
static constexpr int kDsv4SmemBaseAlignmentSlack = 1023;
static_assert(
    kDsv4TaskScratchBytes >=
        kDsv4ResidentRopeMetadataOffset +
        kDsv4ResidentRopeMetadataBytes + kDsv4SmemBaseAlignmentSlack,
    "DeepSeek resident scratch must fit fixed RoPE metadata after alignment");

__device__ __forceinline__ float *dsv4_resident_rope_table(
    void *smem_base,
    int table_id) {
  auto *task_scratch = static_cast<unsigned char *>(
      get_slot_address(smem_base, numSlots));
  return reinterpret_cast<float *>(
      task_scratch + kDsv4ResidentRopeMetadataOffset +
      table_id * kDsv4RopeTableBytes);
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_preload_rope_tables(
    int num_tables,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  if (num_tables <= 0 || num_tables > kDsv4MaxResidentRopeTables) {
    asm volatile("trap;");
  }

  const int tid = __compute_tid();
  int input_slots = 0;
  for (int table_id = 0; table_id < num_tables; ++table_id) {
    const int table_slots = m2c.template pop<0>();
    input_slots |= table_slots;
    const auto *source = static_cast<const float *>(
        get_slot_address(smem_base, extract(table_slots)));
    auto *target = dsv4_resident_rope_table(smem_base, table_id);
    for (int item = tid; item < kDsv4RopeTableElements; item += 128) {
      target[item] = source[item];
    }
  }

  // All four compute warps publish the fixed tables before their source slots
  // are returned. Memory threads are independent and never join this barrier.
  __sync_compute_group(128);
  c2m.push(tid, input_slots);
}

// Apply the DeepSeek partial rotary embedding to the final 64 dimensions of
// each attention (512-wide) or indexer (128-wide) row.  The table is float32
// [32, 2] in (cos, sin) order.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_rope_64(
    int rows,
    int head_dim,
    bool inverse,
    int fixed_table_selector,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kRopeDim = 64;
  const int rope_start = head_dim - kRopeDim;

  const int input_slots = m2c.template pop<0>();
  const int input_slot = extract(input_slots);
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, input_slot));
  int table_slots = 0;
  const float *table = nullptr;
  if (fixed_table_selector == 0) {
    table_slots = m2c.template pop<0>();
    table = static_cast<const float *>(
        get_slot_address(smem_base, extract(table_slots)));
  } else {
    if (fixed_table_selector > kDsv4MaxResidentRopeTables) {
      asm volatile("trap;");
    }
    table = dsv4_resident_rope_table(smem_base, fixed_table_selector - 1);
  }
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));

  const int tid = __compute_tid();
  constexpr int kPairsPerRow = kRopeDim / 2;
  for (int item = tid; item < rows * kPairsPerRow; item += 128) {
    const int row = item / kPairsPerRow;
    const int pair = item % kPairsPerRow;
    const int offset = row * head_dim + rope_start + pair * 2;
    const float even = __bfloat162float(input[offset]);
    const float odd = __bfloat162float(input[offset + 1]);
    const float cosine = table[pair * 2];
    float sine = table[pair * 2 + 1];
    if (inverse) {
      sine = -sine;
    }
    output[offset] = __float2bfloat16(even * cosine - odd * sine);
    output[offset + 1] = __float2bfloat16(even * sine + odd * cosine);
  }

  // Preserve the non-rotary dimensions when input and output do not alias.
  for (int item = tid; item < rows * rope_start; item += 128) {
    const int row = item / rope_start;
    const int dim = item % rope_start;
    output[row * head_dim + dim] = input[row * head_dim + dim];
  }

  __sync_compute_group(128);
  c2m.push(tid, input_slots | table_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Decode-specialized attention epilogue.  One resident compute group owns one
// 512-wide row, keeps normalization in FP32 scratch, and applies the partial
// rotary transform before the only BF16 write.  Q rows omit the weight load;
// KV/cache rows consume the learned 512-wide RMS weight.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_rms_rope_512_64(
    bool weighted,
    int fixed_table_selector,
    __nv_bfloat16 epsilon,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHeadDim = 512;
  constexpr int kRopeDim = 64;
  constexpr int kRopeStart = kHeadDim - kRopeDim;
  constexpr int kValuesPerThread = kHeadDim / 128;

  const int input_slots = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, extract(input_slots)));
  int weight_slots = 0;
  const __nv_bfloat16 *weight = nullptr;
  if (weighted) {
    weight_slots = m2c.template pop<0>();
    weight = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, extract(weight_slots)));
  }
  int table_slots = 0;
  const float *table = nullptr;
  if (fixed_table_selector == 0) {
    table_slots = m2c.template pop<0>();
    table = static_cast<const float *>(
        get_slot_address(smem_base, extract(table_slots)));
  } else {
    if (fixed_table_selector > kDsv4MaxResidentRopeTables) {
      asm volatile("trap;");
    }
    table = dsv4_resident_rope_table(
        smem_base, fixed_table_selector - 1);
  }
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, extract(output_slots)));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *normalized = static_cast<float *>(task_scratch);
  auto *reduction = normalized + kHeadDim;

  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    const float value = __bfloat162float(input[dim]);
    normalized[dim] = value;
    sum = fmaf(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0) {
    reduction[warp] = sum;
  }
  __sync_compute_group(128);
  if (tid == 0) {
    const float total = reduction[0] + reduction[1] +
                        reduction[2] + reduction[3];
    reduction[4] = rsqrtf(
        total / float(kHeadDim) + __bfloat162float(epsilon));
  }
  __sync_compute_group(128);

  const float rms_rcp = reduction[4];
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    float value = normalized[dim] * rms_rcp;
    if (weighted) {
      value *= __bfloat162float(weight[dim]);
    }
    normalized[dim] = value;
  }
  __sync_compute_group(128);

  if (tid < kRopeDim / 2) {
    const int offset = kRopeStart + tid * 2;
    const float even = normalized[offset];
    const float odd = normalized[offset + 1];
    const float cosine = table[tid * 2];
    const float sine = table[tid * 2 + 1];
    normalized[offset] = even * cosine - odd * sine;
    normalized[offset + 1] = even * sine + odd * cosine;
  }
  __sync_compute_group(128);

#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    output[dim] = __float2bfloat16(normalized[dim]);
  }
  __sync_compute_group(128);

  c2m.push(tid, input_slots | weight_slots | table_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Split-K projection epilogue for Q/KV.  The UMMA task reduces directly into
// FP32 HBM; this task consumes that accumulator once, performs the complete
// per-head RMS statistic and rotary transform in FP32, and emits only the
// attention-ready BF16 row.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_fp32_rms_rope_512_64(
    bool weighted,
    int fixed_table_selector,
    __nv_bfloat16 epsilon,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHeadDim = 512;
  constexpr int kRopeDim = 64;
  constexpr int kRopeStart = kHeadDim - kRopeDim;
  constexpr int kValuesPerThread = kHeadDim / 128;

  const int input_slots = m2c.template pop<0>();
  const auto *input = static_cast<const float *>(
      get_slot_address(smem_base, extract(input_slots)));
  int weight_slots = 0;
  const __nv_bfloat16 *weight = nullptr;
  if (weighted) {
    weight_slots = m2c.template pop<0>();
    weight = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, extract(weight_slots)));
  }
  int table_slots = 0;
  const float *table = nullptr;
  if (fixed_table_selector == 0) {
    table_slots = m2c.template pop<0>();
    table = static_cast<const float *>(
        get_slot_address(smem_base, extract(table_slots)));
  } else {
    if (fixed_table_selector > kDsv4MaxResidentRopeTables) {
      asm volatile("trap;");
    }
    table = dsv4_resident_rope_table(
        smem_base, fixed_table_selector - 1);
  }
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, extract(output_slots)));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *normalized = static_cast<float *>(task_scratch);
  auto *reduction = normalized + kHeadDim;

  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    const float value = input[dim];
    normalized[dim] = value;
    sum = fmaf(value, value, sum);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0) {
    reduction[warp] = sum;
  }
  __sync_compute_group(128);
  if (tid == 0) {
    const float total = reduction[0] + reduction[1] +
                        reduction[2] + reduction[3];
    reduction[4] = rsqrtf(
        total / float(kHeadDim) + __bfloat162float(epsilon));
  }
  __sync_compute_group(128);

  const float rms_rcp = reduction[4];
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    float value = normalized[dim] * rms_rcp;
    if (weighted) {
      value *= __bfloat162float(weight[dim]);
    }
    normalized[dim] = value;
  }
  __sync_compute_group(128);

  if (tid < kRopeDim / 2) {
    const int offset = kRopeStart + tid * 2;
    const float even = normalized[offset];
    const float odd = normalized[offset + 1];
    const float cosine = table[tid * 2];
    const float sine = table[tid * 2 + 1];
    normalized[offset] = even * cosine - odd * sine;
    normalized[offset + 1] = even * sine + odd * cosine;
  }
  __sync_compute_group(128);

#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    output[dim] = __float2bfloat16(normalized[dim]);
  }
  __sync_compute_group(128);

  c2m.push(tid, input_slots | weight_slots | table_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Index-Q split-K epilogue.  RoPE and the normalized Walsh-Hadamard transform
// stay in one FP32 scratch tile, so neither the raw nor the rotary-only
// intermediate is materialized in HBM.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_fp32_rope_hadamard_128(
    int fixed_table_selector,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kWidth = 128;
  constexpr int kRopeDim = 64;
  constexpr int kRopeStart = kWidth - kRopeDim;

  const int input_slots = m2c.template pop<0>();
  const auto *input = static_cast<const float *>(
      get_slot_address(smem_base, extract(input_slots)));
  int table_slots = 0;
  const float *table = nullptr;
  if (fixed_table_selector == 0) {
    table_slots = m2c.template pop<0>();
    table = static_cast<const float *>(
        get_slot_address(smem_base, extract(table_slots)));
  } else {
    if (fixed_table_selector > kDsv4MaxResidentRopeTables) {
      asm volatile("trap;");
    }
    table = dsv4_resident_rope_table(
        smem_base, fixed_table_selector - 1);
  }
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, extract(output_slots)));

  const int tid = __compute_tid();
  auto *values = static_cast<float *>(task_scratch);
  values[tid] = input[tid];
  __sync_compute_group(128);

  if (tid < kRopeDim / 2) {
    const int offset = kRopeStart + tid * 2;
    const float even = values[offset];
    const float odd = values[offset + 1];
    const float cosine = table[tid * 2];
    const float sine = table[tid * 2 + 1];
    values[offset] = even * cosine - odd * sine;
    values[offset + 1] = even * sine + odd * cosine;
  }
  __sync_compute_group(128);

  for (int stride = 1; stride < kWidth; stride <<= 1) {
    if (tid < kWidth / 2) {
      const int group = tid / stride;
      const int offset = tid - group * stride;
      const int lhs = group * (stride * 2) + offset;
      const int rhs = lhs + stride;
      const float lhs_value = values[lhs];
      const float rhs_value = values[rhs];
      values[lhs] = lhs_value + rhs_value;
      values[rhs] = lhs_value - rhs_value;
    }
    __sync_compute_group(128);
  }

  output[tid] = __float2bfloat16(values[tid] * rsqrtf(float(kWidth)));
  __sync_compute_group(128);
  c2m.push(tid, input_slots | table_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Correctness-first sparse decode attention for the DeepSeek 64x512 query and
// shared 512-wide KV cache.  One resident SM owns one query head and walks the
// selected KV indices with online softmax.  The attention sink is represented
// as an extra denominator-only logit, matching the official inference path.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_sparse_attention_512(
    int topk,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHeadDim = 512;
  constexpr int kValuesPerThread = kHeadDim / 128;
  constexpr float kSoftmaxScale = 0.04419417382415922f;  // 1 / sqrt(512)

  const int q_slots = m2c.template pop<0>();
  const int q_slot = extract(q_slots);
  const auto *q = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, q_slot));
  const int indices_slots = m2c.template pop<0>();
  const int indices_slot = extract(indices_slots);
  const auto *indices = static_cast<const int *>(
      get_slot_address(smem_base, indices_slot));
  const int sink_slots = m2c.template pop<0>();
  const int sink_slot = extract(sink_slots);
  const auto *sink = static_cast<const float *>(
      get_slot_address(smem_base, sink_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *warp_reduce = static_cast<float *>(task_scratch);

  float q_values[kValuesPerThread];
  float accum[kValuesPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    q_values[item] = __bfloat162float(q[tid + item * 128]);
  }

  float running_max = sink[0];
  float running_sum = 1.0f;
  for (int selected = 0; selected < topk; ++selected) {
    const int kv_row = indices[selected];
    const int kv_slots = m2c.template pop<0>();
    const int kv_slot = extract(kv_slots);
    const auto *kv_ptr = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, kv_slot));
    if (kv_row < 0) {
      __sync_compute_group(128);
      c2m.push(tid, kv_slots);
      continue;
    }
    float partial = 0.0f;
    float kv_values[kValuesPerThread];
#pragma unroll
    for (int item = 0; item < kValuesPerThread; ++item) {
      kv_values[item] =
          __bfloat162float(kv_ptr[tid + item * 128]);
      partial = fmaf(q_values[item], kv_values[item], partial);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset);
    }
    if (lane == 0) {
      warp_reduce[warp] = partial;
    }
    __sync_compute_group(128);
    if (tid == 0) {
      warp_reduce[0] = warp_reduce[0] + warp_reduce[1] +
                       warp_reduce[2] + warp_reduce[3];
    }
    __sync_compute_group(128);

    const float score = warp_reduce[0] * kSoftmaxScale;
    const float new_max = fmaxf(running_max, score);
    const float old_scale = __expf(running_max - new_max);
    const float probability = __expf(score - new_max);
    running_sum = running_sum * old_scale + probability;
#pragma unroll
    for (int item = 0; item < kValuesPerThread; ++item) {
      accum[item] = accum[item] * old_scale + probability * kv_values[item];
    }
    running_max = new_max;
    __sync_compute_group(128);
    c2m.push(tid, kv_slots);
  }

  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));
  const float inverse_sum = 1.0f / running_sum;
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    output[tid + item * 128] = __float2bfloat16(accum[item] * inverse_sum);
  }

  __sync_compute_group(128);
  c2m.push(tid, q_slots | indices_slots | sink_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Four contiguous rows share one TMA slot and one online-softmax step.  Each
// compute warp owns one score row, so the 512-wide dot needs only a warp
// reduction.  Thread zero evaluates the scalar exponentials once and
// publishes four probabilities through the compute scratchpad; all threads
// then update disjoint output dimensions.  Shape policy selects this task only
// when the chosen set is the entire contiguous cache.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_contiguous_attention_512_block4(
    int rows,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHeadDim = 512;
  constexpr int kRowsPerBatch = 4;
  constexpr int kQValuesPerLane = kHeadDim / 32;
  constexpr int kOutputValuesPerThread = kHeadDim / 128;
  constexpr float kSoftmaxScale = 0.04419417382415922f;

  const int q_slots = m2c.template pop<0>();
  const int q_slot = extract(q_slots);
  const auto *q = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, q_slot));
  const int sink_slots = m2c.template pop<0>();
  const int sink_slot = extract(sink_slots);
  const auto *sink = static_cast<const float *>(
      get_slot_address(smem_base, sink_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(task_scratch);

  float q_values[kQValuesPerLane];
#pragma unroll 1
  for (int item = 0; item < kQValuesPerLane; ++item) {
    q_values[item] = __bfloat162float(q[lane + item * 32]);
  }
  float accum[kOutputValuesPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
  float running_max = 0.0f;
  float running_sum = 0.0f;
  if (tid == 0) {
    running_max = sink[0];
    running_sum = 1.0f;
  }

  for (int batch_start = 0; batch_start < rows;
       batch_start += kRowsPerBatch) {
    const int remaining = rows - batch_start;
    const int batch_rows =
        remaining < kRowsPerBatch ? remaining : kRowsPerBatch;
    const int kv_slots = m2c.template pop<0>();
    const int kv_slot = extract(kv_slots);
    const auto *kv_batch = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, kv_slot));

    if (warp < batch_rows) {
      const auto *kv = kv_batch + warp * kHeadDim;
      float partial = 0.0f;
#pragma unroll 1
      for (int item = 0; item < kQValuesPerLane; ++item) {
        partial = fmaf(
            q_values[item],
            __bfloat162float(kv[lane + item * 32]),
            partial);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset);
      }
      if (lane == 0) {
        shared[warp] = partial * kSoftmaxScale;
      }
    }
    __sync_compute_group(128);

    if (tid == 0) {
      float next_max = running_max;
#pragma unroll 1
      for (int row = 0; row < batch_rows; ++row) {
        next_max = fmaxf(next_max, shared[row]);
      }
      const float old_scale = __expf(running_max - next_max);
      float next_sum = running_sum * old_scale;
      shared[4] = old_scale;
#pragma unroll 1
      for (int row = 0; row < batch_rows; ++row) {
        const float probability = __expf(shared[row] - next_max);
        shared[5 + row] = probability;
        next_sum += probability;
      }
      running_max = next_max;
      running_sum = next_sum;
    }
    __sync_compute_group(128);

    const float old_scale = shared[4];
#pragma unroll 1
    for (int item = 0; item < kOutputValuesPerThread; ++item) {
      const int dim = tid + item * 128;
      float update = 0.0f;
#pragma unroll 1
      for (int row = 0; row < batch_rows; ++row) {
        update = fmaf(
            shared[5 + row],
            __bfloat162float(kv_batch[row * kHeadDim + dim]),
            update);
      }
      accum[item] = accum[item] * old_scale + update;
    }
    __sync_compute_group(128);

    c2m.push(tid, kv_slots);
  }

  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));
  if (tid == 0) {
    shared[9] = 1.0f / running_sum;
  }
  __sync_compute_group(128);
  const float inverse_sum = shared[9];
#pragma unroll 1
  for (int item = 0; item < kOutputValuesPerThread; ++item) {
    output[tid + item * 128] =
        __float2bfloat16(accum[item] * inverse_sum);
  }

  __sync_compute_group(128);
  c2m.push(tid, q_slots | sink_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Select DeepSeek's top-6 routed experts from 256 gate logits.  Hash layers
// provide the six expert ids directly but still use the transformed scores as
// routing weights.
__device__ __forceinline__ bool dsv4_route_score_better(
    float candidate_score,
    int candidate_expert,
    float current_score,
    int current_expert) {
  return candidate_score > current_score ||
      (candidate_score == current_score && candidate_expert < current_expert);
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_route_top6(
    bool hash_routing,
    float route_scale,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kExperts = 256;
  constexpr int kTopK = 6;

  const int logits_slots = m2c.template pop<0>();
  const int logits_slot = extract(logits_slots);
  const auto *logits = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, logits_slot));
  const int bias_slots = m2c.template pop<0>();
  const int bias_slot = extract(bias_slots);
  const auto *bias = static_cast<const float *>(
      get_slot_address(smem_base, bias_slot));
  const int hash_slots = m2c.template pop<0>();
  const int hash_slot = extract(hash_slots);
  const auto *hash_indices = static_cast<const int *>(
      get_slot_address(smem_base, hash_slot));
  const int indices_slots = m2c.template pop<0>();
  const int indices_slot = extract(indices_slots);
  auto *output_indices = static_cast<int *>(
      get_slot_address(smem_base, indices_slot));
  const int weights_slots = m2c.template pop<0>();
  const int weights_slot = extract(weights_slots);
  auto *output_weights = static_cast<float *>(
      get_slot_address(smem_base, weights_slot));

  const int tid = __compute_tid();
  auto *original_scores = static_cast<float *>(task_scratch);
  auto *selection_scores = original_scores + kExperts;
  auto *warp_scores = selection_scores + kExperts;
  auto *warp_indices = reinterpret_cast<int *>(warp_scores + 4);
  auto *selected_indices = warp_indices + 4;
  for (int expert = tid; expert < kExperts; expert += 128) {
    const float transformed =
        sqrtf(dsv4_softplus(__bfloat162float(logits[expert])));
    original_scores[expert] = transformed;
    selection_scores[expert] = transformed + bias[expert];
  }
  __sync_compute_group(128);

  if (hash_routing) {
    if (tid == 0) {
      for (int rank = 0; rank < kTopK; ++rank) {
        selected_indices[rank] = hash_indices[rank];
      }
    }
  } else {
    const int lane = tid & 31;
    const int warp = tid >> 5;
    int first_expert = tid;
    int second_expert = tid + 128;
    float first_score = selection_scores[first_expert];
    float second_score = selection_scores[second_expert];
    for (int rank = 0; rank < kTopK; ++rank) {
      float best_score = first_score;
      int best_expert = first_expert;
      if (dsv4_route_score_better(
              second_score, second_expert, best_score, best_expert)) {
        best_score = second_score;
        best_expert = second_expert;
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        const float candidate_score = __shfl_down_sync(
            0xFFFFFFFFU, best_score, offset);
        const int candidate_expert = __shfl_down_sync(
            0xFFFFFFFFU, best_expert, offset);
        if (lane + offset < 32 && dsv4_route_score_better(
                candidate_score,
                candidate_expert,
                best_score,
                best_expert)) {
          best_score = candidate_score;
          best_expert = candidate_expert;
        }
      }
      if (lane == 0) {
        warp_scores[warp] = best_score;
        warp_indices[warp] = best_expert;
      }
      __sync_compute_group(128);

      if (warp == 0) {
        best_score = lane < 4
            ? warp_scores[lane]
            : -__int_as_float(0x7f800000);
        best_expert = lane < 4 ? warp_indices[lane] : 0x7fffffff;
        for (int offset = 16; offset > 0; offset >>= 1) {
          const float candidate_score = __shfl_down_sync(
              0xFFFFFFFFU, best_score, offset);
          const int candidate_expert = __shfl_down_sync(
              0xFFFFFFFFU, best_expert, offset);
          if (lane + offset < 32 && dsv4_route_score_better(
                  candidate_score,
                  candidate_expert,
                  best_score,
                  best_expert)) {
            best_score = candidate_score;
            best_expert = candidate_expert;
          }
        }
        if (lane == 0) {
          selected_indices[rank] = best_expert;
        }
      }
      __sync_compute_group(128);
      const int selected = selected_indices[rank];
      if (first_expert == selected) {
        first_score = -__int_as_float(0x7f800000);
      }
      if (second_expert == selected) {
        second_score = -__int_as_float(0x7f800000);
      }
    }
  }

  if (tid == 0) {
    float weight_sum = 0.0f;
    for (int rank = 0; rank < kTopK; ++rank) {
      weight_sum += original_scores[selected_indices[rank]];
    }
    const float normalization = route_scale / weight_sum;
    for (int rank = 0; rank < kTopK; ++rank) {
      output_indices[rank] = selected_indices[rank];
      output_weights[rank] =
          original_scores[selected_indices[rank]] * normalization;
    }
  }

  __sync_compute_group(128);
  c2m.push(tid, logits_slots | bias_slots | hash_slots);
  c2m.template push<0, true>(tid, indices_slots);
  c2m.template push<0, true>(tid, weights_slots);
}

// Sum the six routed expert down projections and one shared expert.  Applying
// the route weight after w2 is equivalent to the official pre-w2 weighting
// because the down projection is linear.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_expert_reduce(
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHidden = 4096;
  constexpr int kTopK = 6;

  const int routed_slots = m2c.template pop<0>();
  const int routed_slot = extract(routed_slots);
  const auto *routed = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, routed_slot));
  const int weights_slots = m2c.template pop<0>();
  const int weights_slot = extract(weights_slots);
  const auto *weights = static_cast<const float *>(
      get_slot_address(smem_base, weights_slot));
  const int shared_slots = m2c.template pop<0>();
  const int shared_slot = extract(shared_slots);
  const auto *shared = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, shared_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));

  const int tid = __compute_tid();
  for (int dim = tid; dim < kHidden; dim += 128) {
    float value = __bfloat162float(shared[dim]);
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
      value = fmaf(
          weights[rank],
          __bfloat162float(routed[rank * kHidden + dim]),
          value);
    }
    output[dim] = __float2bfloat16(value);
  }

  __sync_compute_group(128);
  c2m.push(tid, routed_slots | weights_slots | shared_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// FP32-weight/BF16-input GEMV for the small mHC mixing projections.  It is
// deliberately scalar and correctness-oriented; one SM owns one or more rows.
template <bool EmitPreRmsMetadata, typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_fp32_bf16_gemv(
    int k,
    int tile_k,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *warp_reduce = static_cast<float *>(task_scratch);
  float partial = 0.0f;
  float square_partial;
  if constexpr (EmitPreRmsMetadata) {
    square_partial = 0.0f;
  }
  for (int column_start = 0; column_start < k; column_start += tile_k) {
    const int columns = min(tile_k, k - column_start);
    const int weight_slots = m2c.template pop<0>();
    const int weight_slot = extract(weight_slots);
    const auto *weight = static_cast<const float *>(
        get_slot_address(smem_base, weight_slot));
    const int input_slots = m2c.template pop<0>();
    const int input_slot = extract(input_slots);
    const auto *input = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, input_slot));
    for (int column = tid; column < columns; column += 128) {
      const float input_value = __bfloat162float(input[column]);
      partial = fmaf(
          weight[column],
          input_value,
          partial);
      if constexpr (EmitPreRmsMetadata) {
        square_partial = fmaf(input_value, input_value, square_partial);
      }
    }
    __sync_compute_group(128);
    c2m.push(tid, weight_slots | input_slots);
  }
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<float *>(
      get_slot_address(smem_base, output_slot));
  int square_output_slots;
  float *square_output;
  int scale_slots;
  const float *scale;
  int base_slots;
  const float *base;
  int metadata_tail_slots;
  float *metadata_tail;
  if constexpr (EmitPreRmsMetadata) {
    square_output_slots = m2c.template pop<0>();
    square_output = static_cast<float *>(
        get_slot_address(smem_base, extract(square_output_slots)));
    scale_slots = m2c.template pop<0>();
    scale = static_cast<const float *>(
        get_slot_address(smem_base, extract(scale_slots)));
    base_slots = m2c.template pop<0>();
    base = static_cast<const float *>(
        get_slot_address(smem_base, extract(base_slots)));
    metadata_tail_slots = m2c.template pop<0>();
    metadata_tail = static_cast<float *>(
        get_slot_address(smem_base, extract(metadata_tail_slots)));
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset);
    if constexpr (EmitPreRmsMetadata) {
      square_partial += __shfl_down_sync(
          0xFFFFFFFFU, square_partial, offset);
    }
  }
  if (lane == 0) {
    warp_reduce[warp] = partial;
    if constexpr (EmitPreRmsMetadata) {
      warp_reduce[4 + warp] = square_partial;
    }
  }
  __sync_compute_group(128);
  if (tid == 0) {
    output[0] = warp_reduce[0] + warp_reduce[1] +
                warp_reduce[2] + warp_reduce[3];
    if constexpr (EmitPreRmsMetadata) {
      square_output[0] = warp_reduce[4] + warp_reduce[5] +
                         warp_reduce[6] + warp_reduce[7];
    }
  }
  if constexpr (EmitPreRmsMetadata) {
    if (tid < 3) {
      metadata_tail[tid] = scale[tid];
    }
    if (tid < 24) {
      metadata_tail[3 + tid] = base[tid];
    }
    // metadata_tail[27] is alignment padding and intentionally uninitialized.
  }
  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, output_slots);
  if constexpr (EmitPreRmsMetadata) {
    c2m.template push<31, true, false>(tid, square_output_slots);
    c2m.push(tid, scale_slots | base_slots);
    c2m.template push<31, true, false>(tid, metadata_tail_slots);
  }
}

// BF16-weight/input/output GEMV for checkpoint linears that are intentionally
// not quantized (router, compressors, indexer weights, embedding head).  FP32
// accumulation retains the same correctness-first row sharding as the mHC
// projection above without expanding large BF16 checkpoint matrices to FP32.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_bf16_gemv(
    int k,
    int tile_k,
    bool output_fp32,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *warp_reduce = static_cast<float *>(task_scratch);
  float partial = 0.0f;
  for (int column_start = 0; column_start < k; column_start += tile_k) {
    const int columns = min(tile_k, k - column_start);
    const int weight_slots = m2c.template pop<0>();
    const int weight_slot = extract(weight_slots);
    const auto *weight = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, weight_slot));
    const int input_slots = m2c.template pop<0>();
    const int input_slot = extract(input_slots);
    const auto *input = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, input_slot));
    for (int column = tid; column < columns; column += 128) {
      partial = fmaf(
          __bfloat162float(weight[column]),
          __bfloat162float(input[column]),
          partial);
    }
    __sync_compute_group(128);
    c2m.push(tid, weight_slots | input_slots);
  }
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  void *output = get_slot_address(smem_base, output_slot);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset);
  }
  if (lane == 0) {
    warp_reduce[warp] = partial;
  }
  __sync_compute_group(128);
  if (tid == 0) {
    const float value = warp_reduce[0] + warp_reduce[1] +
                        warp_reduce[2] + warp_reduce[3];
    if (output_fp32) {
      static_cast<float *>(output)[0] = value;
    } else {
      static_cast<__nv_bfloat16 *>(output)[0] = __float2bfloat16(value);
    }
  }
  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Normalized Walsh-Hadamard transform used by the ratio-4 indexer before its
// simulated FP4 dot products.  One SM owns one 128- or 512-wide row.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hadamard(
    int width,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  const int input_slots = m2c.template pop<0>();
  const int input_slot = extract(input_slots);
  const auto *input = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, input_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));

  const int tid = __compute_tid();
  auto *values = static_cast<float *>(task_scratch);
  for (int dim = tid; dim < width; dim += 128) {
    values[dim] = __bfloat162float(input[dim]);
  }
  __sync_compute_group(128);

  for (int stride = 1; stride < width; stride <<= 1) {
    for (int pair = tid; pair < width / 2; pair += 128) {
      const int group = pair / stride;
      const int offset = pair - group * stride;
      const int lhs = group * (stride * 2) + offset;
      const int rhs = lhs + stride;
      const float lhs_value = values[lhs];
      const float rhs_value = values[rhs];
      values[lhs] = lhs_value + rhs_value;
      values[rhs] = lhs_value - rhs_value;
    }
    __sync_compute_group(128);
  }

  const float scale = rsqrtf(float(width));
  for (int dim = tid; dim < width; dim += 128) {
    output[dim] = __float2bfloat16(values[dim] * scale);
  }
  __sync_compute_group(128);
  c2m.push(tid, input_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Dimension-wise gated pooling for compressed KV state.  The caller supplies
// the contiguous rows selected by the ratio-4 overlap rule or ratio-128 rule,
// with positional APE already added to the FP32 scores.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_gated_pool(
    int pool_rows,
    int width,
    bool tail_bias,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  const int tid = __compute_tid();
  constexpr int kMaxValuesPerThread = 4;
  float maximum[kMaxValuesPerThread];
  float denominator[kMaxValuesPerThread];
  float numerator[kMaxValuesPerThread];
#pragma unroll
  for (int item = 0; item < kMaxValuesPerThread; ++item) {
    maximum[item] = -__int_as_float(0x7f800000);
    denominator[item] = 0.0f;
    numerator[item] = 0.0f;
  }

  for (int row = 0; row < pool_rows; ++row) {
    const int values_slots = m2c.template pop<0>();
    const int values_slot = extract(values_slots);
    const auto *values = static_cast<const float *>(
        get_slot_address(smem_base, values_slot));
    const int scores_slots = m2c.template pop<0>();
    const int scores_slot = extract(scores_slots);
    const auto *scores = static_cast<const float *>(
        get_slot_address(smem_base, scores_slot));
    int bias_slots = 0;
    const float *bias = nullptr;
    if (tail_bias && row + 1 == pool_rows) {
      bias_slots = m2c.template pop<0>();
      const int bias_slot = extract(bias_slots);
      bias = static_cast<const float *>(
          get_slot_address(smem_base, bias_slot));
    }
#pragma unroll
    for (int item = 0; item < kMaxValuesPerThread; ++item) {
      const int dim = tid + item * 128;
      if (dim < width) {
        const float score = scores[dim] + (bias == nullptr ? 0.0f : bias[dim]);
        const float next_max = fmaxf(maximum[item], score);
        const float old_scale = __expf(maximum[item] - next_max);
        const float probability = __expf(score - next_max);
        denominator[item] = denominator[item] * old_scale + probability;
        numerator[item] = numerator[item] * old_scale + probability * values[dim];
        maximum[item] = next_max;
      }
    }
    __sync_compute_group(128);
    c2m.push(tid, values_slots | scores_slots | bias_slots);
  }

  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));
#pragma unroll
  for (int item = 0; item < kMaxValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    if (dim < width) {
      output[dim] = __float2bfloat16(numerator[item] / denominator[item]);
    }
  }
  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Ratio-4 compressor epilogue.  Pooling remains FP32 through weighted RMSNorm,
// RoPE, and the optional index-cache Hadamard transform; only the final cache
// row is written to HBM.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_gated_pool_rms_rope(
    int pool_rows,
    int width,
    bool tail_bias,
    bool hadamard,
    int fixed_table_selector,
    __nv_bfloat16 epsilon,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  if ((width != 128 && width != 512) || (hadamard && width != 128)) {
    asm volatile("trap;");
  }
  const int tid = __compute_tid();
  constexpr int kMaxValuesPerThread = 4;
  constexpr int kRopeDim = 64;
  float maximum[kMaxValuesPerThread];
  float denominator[kMaxValuesPerThread];
  float numerator[kMaxValuesPerThread];
#pragma unroll
  for (int item = 0; item < kMaxValuesPerThread; ++item) {
    maximum[item] = -__int_as_float(0x7f800000);
    denominator[item] = 0.0f;
    numerator[item] = 0.0f;
  }

  for (int row = 0; row < pool_rows; ++row) {
    const int values_slots = m2c.template pop<0>();
    const auto *values = static_cast<const float *>(
        get_slot_address(smem_base, extract(values_slots)));
    const int scores_slots = m2c.template pop<0>();
    const auto *scores = static_cast<const float *>(
        get_slot_address(smem_base, extract(scores_slots)));
    int bias_slots = 0;
    const float *bias = nullptr;
    if (tail_bias && row + 1 == pool_rows) {
      bias_slots = m2c.template pop<0>();
      bias = static_cast<const float *>(
          get_slot_address(smem_base, extract(bias_slots)));
    }
#pragma unroll
    for (int item = 0; item < kMaxValuesPerThread; ++item) {
      const int dim = tid + item * 128;
      if (dim < width) {
        const float score = scores[dim] +
            (bias == nullptr ? 0.0f : bias[dim]);
        const float next_max = fmaxf(maximum[item], score);
        const float old_scale = __expf(maximum[item] - next_max);
        const float probability = __expf(score - next_max);
        denominator[item] =
            denominator[item] * old_scale + probability;
        numerator[item] =
            numerator[item] * old_scale + probability * values[dim];
        maximum[item] = next_max;
      }
    }
    __sync_compute_group(128);
    c2m.push(tid, values_slots | scores_slots | bias_slots);
  }

  const int weight_slots = m2c.template pop<0>();
  const auto *weight = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, extract(weight_slots)));
  int table_slots = 0;
  const float *table = nullptr;
  if (fixed_table_selector == 0) {
    table_slots = m2c.template pop<0>();
    table = static_cast<const float *>(
        get_slot_address(smem_base, extract(table_slots)));
  } else {
    if (fixed_table_selector > kDsv4MaxResidentRopeTables) {
      asm volatile("trap;");
    }
    table = dsv4_resident_rope_table(
        smem_base, fixed_table_selector - 1);
  }
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, extract(output_slots)));

  auto *values = static_cast<float *>(task_scratch);
  auto *reduction = values + width;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float sum = 0.0f;
#pragma unroll
  for (int item = 0; item < kMaxValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    if (dim < width) {
      const float value = numerator[item] / denominator[item];
      values[dim] = value;
      sum = fmaf(value, value, sum);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane == 0) {
    reduction[warp] = sum;
  }
  __sync_compute_group(128);
  if (tid == 0) {
    const float total = reduction[0] + reduction[1] +
                        reduction[2] + reduction[3];
    reduction[4] = rsqrtf(
        total / float(width) + __bfloat162float(epsilon));
  }
  __sync_compute_group(128);

  const float rms_rcp = reduction[4];
#pragma unroll
  for (int item = 0; item < kMaxValuesPerThread; ++item) {
    const int dim = tid + item * 128;
    if (dim < width) {
      values[dim] =
          values[dim] * rms_rcp * __bfloat162float(weight[dim]);
    }
  }
  __sync_compute_group(128);

  if (tid < kRopeDim / 2) {
    const int rope_start = width - kRopeDim;
    const int offset = rope_start + tid * 2;
    const float even = values[offset];
    const float odd = values[offset + 1];
    const float cosine = table[tid * 2];
    const float sine = table[tid * 2 + 1];
    values[offset] = even * cosine - odd * sine;
    values[offset + 1] = even * sine + odd * cosine;
  }
  __sync_compute_group(128);

  if (hadamard) {
    for (int stride = 1; stride < width; stride <<= 1) {
      if (tid < width / 2) {
        const int group = tid / stride;
        const int offset = tid - group * stride;
        const int lhs = group * (stride * 2) + offset;
        const int rhs = lhs + stride;
        const float lhs_value = values[lhs];
        const float rhs_value = values[rhs];
        values[lhs] = lhs_value + rhs_value;
        values[rhs] = lhs_value - rhs_value;
      }
      __sync_compute_group(128);
    }
  }

  const float output_scale = hadamard ? rsqrtf(float(width)) : 1.0f;
  for (int dim = tid; dim < width; dim += 128) {
    output[dim] = __float2bfloat16(values[dim] * output_scale);
  }
  __sync_compute_group(128);
  c2m.push(tid, weight_slots | table_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Dimension-sharded gated pooling over immutable, prepacked history.  One
// 8-KiB TMA carries eight rows of both values and scores for 128 dimensions.
// The dynamic tail remains in its producer layout and is loaded separately;
// no inter-stage layout copy is introduced.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_gated_pool_packed8_shard128(
    int history_rows,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kWidth = 128;
  constexpr int kRowsPerBlock = 8;
  const int tid = __compute_tid();
  float maximum = -__int_as_float(0x7f800000);
  float denominator = 0.0f;
  float numerator = 0.0f;

  for (int row_start = 0; row_start < history_rows;
       row_start += kRowsPerBlock) {
    const int block_slots = m2c.template pop<0>();
    const int block_slot = extract(block_slots);
    const auto *block = static_cast<const float *>(
        get_slot_address(smem_base, block_slot));
    const int remaining = history_rows - row_start;
    const int block_rows =
        remaining < kRowsPerBlock ? remaining : kRowsPerBlock;
#pragma unroll 1
    for (int row = 0; row < block_rows; ++row) {
      const float value = block[(row * 2) * kWidth + tid];
      const float score = block[(row * 2 + 1) * kWidth + tid];
      const float next_max = fmaxf(maximum, score);
      const float old_scale = __expf(maximum - next_max);
      const float probability = __expf(score - next_max);
      denominator = denominator * old_scale + probability;
      numerator = numerator * old_scale + probability * value;
      maximum = next_max;
    }
    __sync_compute_group(128);
    c2m.push(tid, block_slots);
  }

  const int tail_values_slots = m2c.template pop<0>();
  const int tail_values_slot = extract(tail_values_slots);
  const auto *tail_values = static_cast<const float *>(
      get_slot_address(smem_base, tail_values_slot));
  const int tail_scores_slots = m2c.template pop<0>();
  const int tail_scores_slot = extract(tail_scores_slots);
  const auto *tail_scores = static_cast<const float *>(
      get_slot_address(smem_base, tail_scores_slot));
  const int tail_bias_slots = m2c.template pop<0>();
  const int tail_bias_slot = extract(tail_bias_slots);
  const auto *tail_bias = static_cast<const float *>(
      get_slot_address(smem_base, tail_bias_slot));
  const float tail_score = tail_scores[tid] + tail_bias[tid];
  const float next_max = fmaxf(maximum, tail_score);
  const float old_scale = __expf(maximum - next_max);
  const float probability = __expf(tail_score - next_max);
  denominator = denominator * old_scale + probability;
  numerator = numerator * old_scale + probability * tail_values[tid];

  __sync_compute_group(128);
  c2m.push(
      tid, tail_values_slots | tail_scores_slots | tail_bias_slots);

  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));
  output[tid] = __float2bfloat16(numerator / denominator);
  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Four-way HCA pooling phase.  Each SM owns 128 dimensions, retains pooled
// values in FP32, and publishes one exact local sum-of-squares.  The tiny
// four-scalar join lets the following phase normalize all shards in parallel.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_dsv4_gated_pool_packed8_rms_partial(
    int history_rows,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kWidth = 128;
  constexpr int kRowsPerBlock = 8;
  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float maximum = -__int_as_float(0x7f800000);
  float denominator = 0.0f;
  float numerator = 0.0f;

  for (int row_start = 0; row_start < history_rows;
       row_start += kRowsPerBlock) {
    const int block_slots = m2c.template pop<0>();
    const auto *block = static_cast<const float *>(
        get_slot_address(smem_base, extract(block_slots)));
    const int remaining = history_rows - row_start;
    const int block_rows =
        remaining < kRowsPerBlock ? remaining : kRowsPerBlock;
#pragma unroll 1
    for (int row = 0; row < block_rows; ++row) {
      const float value = block[(row * 2) * kWidth + tid];
      const float score = block[(row * 2 + 1) * kWidth + tid];
      const float next_max = fmaxf(maximum, score);
      const float old_scale = __expf(maximum - next_max);
      const float probability = __expf(score - next_max);
      denominator = denominator * old_scale + probability;
      numerator = numerator * old_scale + probability * value;
      maximum = next_max;
    }
    __sync_compute_group(128);
    c2m.push(tid, block_slots);
  }

  const int tail_values_slots = m2c.template pop<0>();
  const auto *tail_values = static_cast<const float *>(
      get_slot_address(smem_base, extract(tail_values_slots)));
  const int tail_scores_slots = m2c.template pop<0>();
  const auto *tail_scores = static_cast<const float *>(
      get_slot_address(smem_base, extract(tail_scores_slots)));
  const int tail_bias_slots = m2c.template pop<0>();
  const auto *tail_bias = static_cast<const float *>(
      get_slot_address(smem_base, extract(tail_bias_slots)));
  const float tail_score = tail_scores[tid] + tail_bias[tid];
  const float next_max = fmaxf(maximum, tail_score);
  const float old_scale = __expf(maximum - next_max);
  const float probability = __expf(tail_score - next_max);
  denominator = denominator * old_scale + probability;
  numerator = numerator * old_scale + probability * tail_values[tid];
  const float pooled = numerator / denominator;

  __sync_compute_group(128);
  c2m.push(
      tid, tail_values_slots | tail_scores_slots | tail_bias_slots);

  const int pooled_slots = m2c.template pop<0>();
  auto *pooled_output = static_cast<float *>(
      get_slot_address(smem_base, extract(pooled_slots)));
  const int partial_slots = m2c.template pop<0>();
  auto *partial_output = static_cast<float *>(
      get_slot_address(smem_base, extract(partial_slots)));
  pooled_output[tid] = pooled;

  float sum = pooled * pooled;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  auto *warp_sums = static_cast<float *>(task_scratch);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __sync_compute_group(128);
  if (tid == 0) {
    partial_output[0] =
        warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
  }
  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, pooled_slots);
  c2m.template push<31, true, false>(tid, partial_slots);
}

// HCA history is immutable for the current decode step and does not depend on
// the current-token compressor projection.  Preserve its numerically stable
// online-softmax state so it can run on a disjoint SM band while wkv/wgate is
// still producing the tail.  Each update needs one exponential: a new maximum
// makes the new row's probability exactly one, while an old maximum makes the
// existing accumulator scale exactly one.
__device__ __forceinline__ void dsv4_gated_pool_online_update(
    float value,
    float score,
    float &maximum,
    float &denominator,
    float &numerator) {
  if (score > maximum) {
    const float old_scale = __expf(maximum - score);
    denominator = denominator * old_scale + 1.0f;
    numerator = numerator * old_scale + value;
    maximum = score;
  } else {
    const float probability = __expf(score - maximum);
    denominator += probability;
    numerator = fmaf(probability, value, numerator);
  }
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void
task_dsv4_gated_pool_packed8_history_state(
    int history_rows,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kWidth = 128;
  constexpr int kRowsPerBlock = 8;
  const int tid = __compute_tid();
  float maximum = -__int_as_float(0x7f800000);
  float denominator = 0.0f;
  float numerator = 0.0f;

  for (int row_start = 0; row_start < history_rows;
       row_start += kRowsPerBlock) {
    const int block_slots = m2c.template pop<0>();
    const auto *block = static_cast<const float *>(
        get_slot_address(smem_base, extract(block_slots)));
    const int remaining = history_rows - row_start;
    const int block_rows =
        remaining < kRowsPerBlock ? remaining : kRowsPerBlock;
#pragma unroll 1
    for (int row = 0; row < block_rows; ++row) {
      dsv4_gated_pool_online_update(
          block[(row * 2) * kWidth + tid],
          block[(row * 2 + 1) * kWidth + tid],
          maximum,
          denominator,
          numerator);
    }
    __sync_compute_group(128);
    c2m.push(tid, block_slots);
  }

  const int state_slots = m2c.template pop<0>();
  auto *state = static_cast<float *>(
      get_slot_address(smem_base, extract(state_slots)));
  state[tid] = maximum;
  state[kWidth + tid] = denominator;
  state[2 * kWidth + tid] = numerator;
  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, state_slots);
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_gated_pool_tail_rms_partial(
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kWidth = 128;
  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int state_slots = m2c.template pop<0>();
  const auto *state = static_cast<const float *>(
      get_slot_address(smem_base, extract(state_slots)));
  float maximum = state[tid];
  float denominator = state[kWidth + tid];
  float numerator = state[2 * kWidth + tid];

  const int tail_values_slots = m2c.template pop<0>();
  const auto *tail_values = static_cast<const float *>(
      get_slot_address(smem_base, extract(tail_values_slots)));
  const int tail_scores_slots = m2c.template pop<0>();
  const auto *tail_scores = static_cast<const float *>(
      get_slot_address(smem_base, extract(tail_scores_slots)));
  const int tail_bias_slots = m2c.template pop<0>();
  const auto *tail_bias = static_cast<const float *>(
      get_slot_address(smem_base, extract(tail_bias_slots)));
  dsv4_gated_pool_online_update(
      tail_values[tid],
      tail_scores[tid] + tail_bias[tid],
      maximum,
      denominator,
      numerator);
  const float pooled = numerator / denominator;

  __sync_compute_group(128);
  c2m.push(
      tid,
      state_slots | tail_values_slots | tail_scores_slots | tail_bias_slots);

  const int pooled_slots = m2c.template pop<0>();
  auto *pooled_output = static_cast<float *>(
      get_slot_address(smem_base, extract(pooled_slots)));
  const int partial_slots = m2c.template pop<0>();
  auto *partial_output = static_cast<float *>(
      get_slot_address(smem_base, extract(partial_slots)));
  pooled_output[tid] = pooled;

  float sum = pooled * pooled;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFU, sum, offset);
  }
  auto *warp_sums = static_cast<float *>(task_scratch);
  if (lane == 0) {
    warp_sums[warp] = sum;
  }
  __sync_compute_group(128);
  if (tid == 0) {
    partial_output[0] =
        warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3];
  }
  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, pooled_slots);
  c2m.template push<31, true, false>(tid, partial_slots);
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_fp32_rms_rope_shard128(
    int shard,
    int fixed_table_selector,
    __nv_bfloat16 epsilon,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kShardWidth = 128;
  constexpr int kFullWidth = 512;
  constexpr int kRopeDim = 64;
  if (shard < 0 || shard >= kFullWidth / kShardWidth) {
    asm volatile("trap;");
  }

  const int input_slots = m2c.template pop<0>();
  const auto *input = static_cast<const float *>(
      get_slot_address(smem_base, extract(input_slots)));
  const int partial_slots = m2c.template pop<0>();
  const auto *partials = static_cast<const float *>(
      get_slot_address(smem_base, extract(partial_slots)));
  const int weight_slots = m2c.template pop<0>();
  const auto *weight = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, extract(weight_slots)));
  int table_slots = 0;
  const float *table = nullptr;
  if (fixed_table_selector == 0) {
    table_slots = m2c.template pop<0>();
    table = static_cast<const float *>(
        get_slot_address(smem_base, extract(table_slots)));
  } else {
    if (fixed_table_selector > kDsv4MaxResidentRopeTables) {
      asm volatile("trap;");
    }
    table = dsv4_resident_rope_table(
        smem_base, fixed_table_selector - 1);
  }
  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, extract(output_slots)));

  const int tid = __compute_tid();
  const float total =
      partials[0] + partials[1] + partials[2] + partials[3];
  const float rms_rcp = rsqrtf(
      total / float(kFullWidth) + __bfloat162float(epsilon));
  auto *values = static_cast<float *>(task_scratch);
  values[tid] =
      input[tid] * rms_rcp * __bfloat162float(weight[tid]);
  __sync_compute_group(128);

  if (shard == 3 && tid < kRopeDim / 2) {
    const int offset = kShardWidth - kRopeDim + tid * 2;
    const float even = values[offset];
    const float odd = values[offset + 1];
    const float cosine = table[tid * 2];
    const float sine = table[tid * 2 + 1];
    values[offset] = even * cosine - odd * sine;
    values[offset + 1] = even * sine + odd * cosine;
  }
  __sync_compute_group(128);
  output[tid] = __float2bfloat16(values[tid]);
  __sync_compute_group(128);

  c2m.push(tid, input_slots | partial_slots | weight_slots | table_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Learned ratio-4 index score.  Each SM handles a contiguous KV-row shard;
// within a row the four warps independently reduce one head at a time.  This
// keeps all warps useful and needs only one warpgroup reduction per row.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_index_score(
    int rows,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHeads = 64;
  constexpr int kHeadDim = 128;

  const int q_slots = m2c.template pop<0>();
  const int q_slot = extract(q_slots);
  const auto *q = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, q_slot));
  const int kv_slots = m2c.template pop<0>();
  const int kv_slot = extract(kv_slots);
  const auto *kv = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, kv_slot));
  const int weights_slots = m2c.template pop<0>();
  const int weights_slot = extract(weights_slots);
  const auto *head_weights = static_cast<const float *>(
      get_slot_address(smem_base, weights_slot));
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<float *>(
      get_slot_address(smem_base, output_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *warp_reduce = static_cast<float *>(task_scratch);
  for (int row = 0; row < rows; ++row) {
    float warp_score = 0.0f;
    for (int head = warp; head < kHeads; head += 4) {
      float partial = 0.0f;
#pragma unroll
      for (int item = 0; item < 4; ++item) {
        const int dim = lane + item * 32;
        partial = fmaf(
            __bfloat162float(q[head * kHeadDim + dim]),
            __bfloat162float(kv[row * kHeadDim + dim]),
            partial);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFFU, partial, offset);
      }
      if (lane == 0) {
        warp_score = fmaf(
            fmaxf(partial, 0.0f), head_weights[head], warp_score);
      }
    }
    if (lane == 0) {
      warp_reduce[warp] = warp_score;
    }
    __sync_compute_group(128);
    if (tid == 0) {
      output[row] = warp_reduce[0] + warp_reduce[1] +
                    warp_reduce[2] + warp_reduce[3];
    }
    __sync_compute_group(128);
  }

  c2m.push(tid, q_slots | kv_slots | weights_slots);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Exact streaming top-k selection for indexer scores.  A 1024-element shared
// bitonic network produces the first candidates, then merges each subsequent
// 512-row chunk with the retained top 512.  Decode contexts up to the model
// maximum fit in the uint16 row count after ratio-4 compression.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_topk512(
    int rows,
    int topk,
    int index_offset,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kSortSize = 1024;
  constexpr int kRetained = 512;
  const int tid = __compute_tid();
  const int lane = tid & 31;

  // Decode starts with at most 32 compressed CSA rows.  Sorting those rows in
  // the generic 1024-item shared-memory network spends 55 synchronization
  // phases on padding.  Keep this exact small-row path entirely in one warp;
  // all four compute warps still participate in the queue hand-off.
  if (rows <= 32) {
    const int scores_slots = m2c.template pop<0>();
    const int scores_slot = extract(scores_slots);
    const float *scores = static_cast<const float *>(
        get_slot_address(smem_base, scores_slot));
    const int output_slots = m2c.template pop<0>();
    const int output_slot = extract(output_slots);
    auto *output = static_cast<int *>(
        get_slot_address(smem_base, output_slot));

    if (tid < 32) {
      constexpr unsigned kWarpMask = 0xFFFFFFFFU;
      float score = lane < rows
          ? scores[lane]
          : -__int_as_float(0x7f800000);
      int index = lane < rows ? lane : -1;
#pragma unroll
      for (int width = 2; width <= 32; width <<= 1) {
#pragma unroll
        for (int stride = width >> 1; stride > 0; stride >>= 1) {
          const float peer_score = __shfl_xor_sync(
              kWarpMask, score, stride);
          const int peer_index = __shfl_xor_sync(
              kWarpMask, index, stride);
          const bool peer_less =
              peer_score < score ||
              (peer_score == score && peer_index < index);
          const bool self_less =
              score < peer_score ||
              (score == peer_score && index < peer_index);
          const bool ascending = (lane & width) == 0;
          const bool lower_lane = (lane & stride) == 0;
          const bool choose_min = ascending == lower_lane;
          if ((choose_min && peer_less) || (!choose_min && self_less)) {
            score = peer_score;
            index = peer_index;
          }
        }
      }
      const int selected = __shfl_sync(kWarpMask, index, 31 - lane);
      if (lane < topk) {
        output[lane] = selected + index_offset;
      }
    }
    __sync_compute_group(128);
    c2m.push(tid, scores_slots);
    c2m.template push<31, true, false>(tid, output_slots);
    return;
  }

  auto *sort_scores = static_cast<float *>(task_scratch);
  auto *sort_indices = reinterpret_cast<int *>(sort_scores + kSortSize);

  int processed = min(rows, kSortSize);
  int scores_slots = m2c.template pop<0>();
  int scores_slot = extract(scores_slots);
  const float *scores = static_cast<const float *>(
      get_slot_address(smem_base, scores_slot));
  for (int item = tid; item < kSortSize; item += 128) {
    sort_scores[item] = item < processed
        ? scores[item]
        : -__int_as_float(0x7f800000);
    sort_indices[item] = item < processed ? item : -1;
  }
  __sync_compute_group(128);
  c2m.push(tid, scores_slots);

  while (true) {
    for (int width = 2; width <= kSortSize; width <<= 1) {
      for (int stride = width >> 1; stride > 0; stride >>= 1) {
        for (int item = tid; item < kSortSize; item += 128) {
          const int peer = item ^ stride;
          if (peer > item) {
            const float left = sort_scores[item];
            const float right = sort_scores[peer];
            const int left_index = sort_indices[item];
            const int right_index = sort_indices[peer];
            const bool ascending = (item & width) == 0;
            const bool swap = ascending ? left > right : left < right;
            if (swap) {
              sort_scores[item] = right;
              sort_scores[peer] = left;
              sort_indices[item] = right_index;
              sort_indices[peer] = left_index;
            }
          }
        }
        __sync_compute_group(128);
      }
    }

    if (processed >= rows) {
      break;
    }
    const int chunk = min(kRetained, rows - processed);
    scores_slots = m2c.template pop<0>();
    scores_slot = extract(scores_slots);
    scores = static_cast<const float *>(
        get_slot_address(smem_base, scores_slot));
    for (int item = tid; item < kRetained; item += 128) {
      sort_scores[item] = item < chunk
          ? scores[item]
          : -__int_as_float(0x7f800000);
      sort_indices[item] = item < chunk ? processed + item : -1;
    }
    processed += chunk;
    __sync_compute_group(128);
    c2m.push(tid, scores_slots);
  }

  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<int *>(
      get_slot_address(smem_base, output_slot));
  for (int rank = tid; rank < topk; rank += 128) {
    output[rank] = sort_indices[kSortSize - 1 - rank] + index_offset;
  }

  __sync_compute_group(128);
  c2m.template push<31, true, false>(tid, output_slots);
}

// Final/MTP hyper-connection head: normalize the four raw projection values,
// form sigmoid pre coefficients, and reduce [4,4096] to one hidden vector.
template <bool FuseRms, typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hc_head(
    float epsilon,
    __nv_bfloat16 rms_epsilon,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHc = 4;
  constexpr int kHidden = 4096;

  const int residual_slots = m2c.template pop<0>();
  const int residual_slot = extract(residual_slots);
  const auto *residual = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, residual_slot));
  const int mixes_slots = m2c.template pop<0>();
  const int mixes_slot = extract(mixes_slots);
  const auto *mixes = static_cast<const float *>(
      get_slot_address(smem_base, mixes_slot));
  const int scale_slots = m2c.template pop<0>();
  const int scale_slot = extract(scale_slots);
  const auto *scale = static_cast<const float *>(
      get_slot_address(smem_base, scale_slot));
  const int base_slots = m2c.template pop<0>();
  const int base_slot = extract(base_slots);
  const auto *base = static_cast<const float *>(
      get_slot_address(smem_base, base_slot));
  int rms_weight_slots = 0;
  const __nv_bfloat16 *rms_weight = nullptr;
  if constexpr (FuseRms) {
    rms_weight_slots = m2c.template pop<0>();
    rms_weight = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, extract(rms_weight_slots)));
  }
  const int output_slots = m2c.template pop<0>();
  const int output_slot = extract(output_slots);
  auto *output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, output_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(task_scratch);
  auto *warp_reduce = shared;
  auto *pre = shared + 4;

  float sum_squares = 0.0f;
  for (int item = tid; item < kHc * kHidden; item += 128) {
    const float value = __bfloat162float(residual[item]);
    sum_squares = fmaf(value, value, sum_squares);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_squares += __shfl_down_sync(0xFFFFFFFFU, sum_squares, offset);
  }
  if (lane == 0) {
    warp_reduce[warp] = sum_squares;
  }
  __sync_compute_group(128);
  if (tid == 0) {
    const float total = warp_reduce[0] + warp_reduce[1] +
                        warp_reduce[2] + warp_reduce[3];
    const float normalization =
        rsqrtf(total / float(kHc * kHidden) + 1.0e-6f);
#pragma unroll
    for (int branch = 0; branch < kHc; ++branch) {
      pre[branch] = dsv4_sigmoid(
          mixes[branch] * normalization * scale[0] + base[branch]) + epsilon;
    }
  }
  __sync_compute_group(128);

  for (int dim = tid; dim < kHidden; dim += 128) {
    float value = 0.0f;
#pragma unroll
    for (int branch = 0; branch < kHc; ++branch) {
      value = fmaf(
          pre[branch],
          __bfloat162float(residual[branch * kHidden + dim]),
          value);
    }
    output[dim] = __float2bfloat16(value);
  }

  __sync_compute_group(128);
  c2m.push(tid, residual_slots | mixes_slots | scale_slots | base_slots);
  if constexpr (FuseRms) {
    _rms_helper_one_row<kHidden, 128>(
        rms_weight, output, output, shared, rms_epsilon);
    __sync_compute_group(128);
    c2m.push(tid, rms_weight_slots);
  }
  c2m.template push<31, true, false>(tid, output_slots);
}

// Convert 24 mHC projection values into pre/post/comb coefficients and form
// the pre-branch 4096-wide hidden vector.  The residual is [4,4096].
template <bool FuseRms, typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hc_pre(
    int sinkhorn_iters,
    float epsilon,
    __nv_bfloat16 rms_epsilon,
    bool zero_fp32_output,
    void *smem_base,
    void *task_scratch,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHc = 4;
  constexpr int kHidden = 4096;

  const int residual_slots = m2c.template pop<0>();
  const int residual_slot = extract(residual_slots);
  const auto *residual = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, residual_slot));
  int mixes_slots = 0;
  const float *mixes = nullptr;
  int scale_slots = 0;
  const float *scale = nullptr;
  int base_slots = 0;
  const float *base = nullptr;
  int norm_weight_slots = 0;
  const __nv_bfloat16 *norm_weight = nullptr;
  int output_slots = 0;
  __nv_bfloat16 *output = nullptr;
  int post_slots = 0;
  float *post_output = nullptr;
  int comb_slots = 0;
  float *comb_output = nullptr;
  int zero_output_slots = 0;
  float *zero_output = nullptr;

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(task_scratch);
  auto *warp_reduce = shared;
  auto *pre = shared + 4;
  auto *post_values = shared + 9;
  auto *comb_values = shared + 13;

  float sum_squares = 0.0f;
  constexpr int kVectorWidth = 8;
  for (int item = tid * kVectorWidth; item < kHc * kHidden;
       item += 128 * kVectorWidth) {
    const Dsv4Bf16x8 packed = dsv4_load_bf16x8(residual + item);
#pragma unroll
    for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
      const float2 values = __bfloat1622float2(
          dsv4_bf16x8_pair(packed, pair));
      sum_squares = fmaf(values.x, values.x, sum_squares);
      sum_squares = fmaf(values.y, values.y, sum_squares);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum_squares += __shfl_down_sync(0xFFFFFFFFU, sum_squares, offset);
  }
  if (lane == 0) {
    warp_reduce[warp] = sum_squares;
  }
  __sync_compute_group(128);

  mixes_slots = m2c.template pop<0>();
  mixes = static_cast<const float *>(
      get_slot_address(smem_base, extract(mixes_slots)));
  scale_slots = m2c.template pop<0>();
  scale = static_cast<const float *>(
      get_slot_address(smem_base, extract(scale_slots)));
  base_slots = m2c.template pop<0>();
  base = static_cast<const float *>(
      get_slot_address(smem_base, extract(base_slots)));

  if (tid == 0) {
    const float total = warp_reduce[0] + warp_reduce[1] +
                        warp_reduce[2] + warp_reduce[3];
    const float rsqrt = rsqrtf(total / float(kHc * kHidden) + 1.0e-6f);
    shared[8] = rsqrt;
#pragma unroll
    for (int index = 0; index < kHc; ++index) {
      pre[index] = dsv4_sigmoid(
          mixes[index] * rsqrt * scale[0] + base[index]) + epsilon;
    }
  }
  __sync_compute_group(128);

  if (warp == 0 && lane < kHc) {
    const float rsqrt = shared[8];
    post_values[lane] = 2.0f * dsv4_sigmoid(
        mixes[kHc + lane] * rsqrt * scale[1] + base[kHc + lane]);
    float comb[kHc];
    float row_max = -__int_as_float(0x7f800000);
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      const int index = lane * kHc + column;
      comb[column] = mixes[2 * kHc + index] * rsqrt * scale[2] +
                     base[2 * kHc + index];
      row_max = fmaxf(row_max, comb[column]);
    }
    float row_sum = 0.0f;
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      comb[column] = __expf(comb[column] - row_max);
      row_sum += comb[column];
    }
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      comb[column] = comb[column] / row_sum + epsilon;
    }
    constexpr unsigned kHcMask = (1U << kHc) - 1U;
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      float column_sum = comb[column];
      column_sum += __shfl_xor_sync(kHcMask, column_sum, 1);
      column_sum += __shfl_xor_sync(kHcMask, column_sum, 2);
      comb[column] /= column_sum + epsilon;
    }
    for (int iteration = 1; iteration < sinkhorn_iters; ++iteration) {
      row_sum = comb[0] + comb[1] + comb[2] + comb[3] + epsilon;
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        comb[column] /= row_sum;
      }
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        float column_sum = comb[column];
        column_sum += __shfl_xor_sync(kHcMask, column_sum, 1);
        column_sum += __shfl_xor_sync(kHcMask, column_sum, 2);
        comb[column] /= column_sum + epsilon;
      }
    }
#pragma unroll
    for (int column = 0; column < kHc; ++column) {
      comb_values[lane * kHc + column] = comb[column];
    }
  }

  auto *hidden = reinterpret_cast<__nv_bfloat16 *>(shared + 64);
  if constexpr (FuseRms) {
    float hidden_sum = 0.0f;
    if (warp != 0) {
      constexpr int kWorkers = 128 - 32;
      const int worker = tid - 32;
      for (int dim = worker * kVectorWidth; dim < kHidden;
           dim += kWorkers * kVectorWidth) {
        float values[kVectorWidth] = {};
#pragma unroll
        for (int branch = 0; branch < kHc; ++branch) {
          const Dsv4Bf16x8 packed = dsv4_load_bf16x8(
              residual + branch * kHidden + dim);
#pragma unroll
          for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
            const float2 branch_values = __bfloat1622float2(
                dsv4_bf16x8_pair(packed, pair));
            values[pair * 2] = fmaf(
                pre[branch], branch_values.x, values[pair * 2]);
            values[pair * 2 + 1] = fmaf(
                pre[branch], branch_values.y, values[pair * 2 + 1]);
          }
        }
        Dsv4Bf16x8 rounded;
        auto *rounded_pairs = reinterpret_cast<__nv_bfloat162 *>(&rounded.raw);
#pragma unroll
        for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
          rounded_pairs[pair] = __float22bfloat162_rn(
              make_float2(values[pair * 2], values[pair * 2 + 1]));
          const float2 rounded_values = __bfloat1622float2(rounded_pairs[pair]);
          hidden_sum = fmaf(
              rounded_values.x, rounded_values.x, hidden_sum);
          hidden_sum = fmaf(
              rounded_values.y, rounded_values.y, hidden_sum);
        }
        dsv4_store_bf16x8(hidden + dim, rounded);
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        hidden_sum += __shfl_down_sync(
            0xFFFFFFFFU, hidden_sum, offset);
      }
      if (lane == 0) {
        warp_reduce[warp - 1] = hidden_sum;
      }
    }
    __sync_compute_group(128);
  } else {
    __sync_compute_group(128);
  }

  if constexpr (FuseRms) {
    norm_weight_slots = m2c.template pop<0>();
    norm_weight = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, extract(norm_weight_slots)));
  }
  output_slots = m2c.template pop<0>();
  output = static_cast<__nv_bfloat16 *>(
      get_slot_address(smem_base, extract(output_slots)));
  post_slots = m2c.template pop<0>();
  post_output = static_cast<float *>(
      get_slot_address(smem_base, extract(post_slots)));
  comb_slots = m2c.template pop<0>();
  comb_output = static_cast<float *>(
      get_slot_address(smem_base, extract(comb_slots)));
  if (zero_fp32_output) {
    zero_output_slots = m2c.template pop<0>();
    zero_output = static_cast<float *>(
        get_slot_address(smem_base, extract(zero_output_slots)));
  }
  if (tid < kHc) {
    post_output[tid] = post_values[tid];
  }
  if (tid < kHc * kHc) {
    comb_output[tid] = comb_values[tid];
  }

  if constexpr (FuseRms) {
    if (tid == 0) {
      const float total = warp_reduce[0] + warp_reduce[1] + warp_reduce[2];
      warp_reduce[3] = rsqrtf(
          total / float(kHidden) + __bfloat162float(rms_epsilon));
    }
    __sync_compute_group(128);
    const float rms_rcp = warp_reduce[3];
    for (int dim = tid * kVectorWidth; dim < kHidden;
         dim += 128 * kVectorWidth) {
      const Dsv4Bf16x8 hidden_values = dsv4_load_bf16x8(hidden + dim);
      const Dsv4Bf16x8 weight_values = dsv4_load_bf16x8(norm_weight + dim);
      Dsv4Bf16x8 normalized;
      auto *normalized_pairs =
          reinterpret_cast<__nv_bfloat162 *>(&normalized.raw);
#pragma unroll
      for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
        const float2 hidden_pair = __bfloat1622float2(
            dsv4_bf16x8_pair(hidden_values, pair));
        const float2 weight_pair = __bfloat1622float2(
            dsv4_bf16x8_pair(weight_values, pair));
        normalized_pairs[pair] = __float22bfloat162_rn(make_float2(
            hidden_pair.x * rms_rcp * weight_pair.x,
            hidden_pair.y * rms_rcp * weight_pair.y));
      }
      dsv4_store_bf16x8(output + dim, normalized);
      if (zero_fp32_output) {
        auto *zero_vectors = reinterpret_cast<uint4 *>(zero_output + dim);
        zero_vectors[0] = make_uint4(0, 0, 0, 0);
        zero_vectors[1] = make_uint4(0, 0, 0, 0);
      }
    }
  } else {
    for (int dim = tid; dim < kHidden; dim += 128) {
      float value = 0.0f;
#pragma unroll
      for (int branch = 0; branch < kHc; ++branch) {
        value = fmaf(
            pre[branch],
            __bfloat162float(residual[branch * kHidden + dim]),
            value);
      }
      output[dim] = __float2bfloat16(value);
      if (zero_fp32_output) {
        zero_output[dim] = 0.0f;
      }
    }
  }

  __sync_compute_group(128);
  c2m.push(
      tid, residual_slots | mixes_slots | scale_slots | base_slots |
      norm_weight_slots);
  c2m.template push<31, true, false>(tid, output_slots);
  c2m.template push<31, true, false>(tid, post_slots);
  c2m.template push<31, true, false>(tid, comb_slots);
  if (zero_fp32_output) {
    c2m.template push<31, true, false>(tid, zero_output_slots);
  }
}

template <Dsv4HcPreRmsMetadataMode MetadataMode, bool ZeroFp32Output,
          bool OutputFp8,
          typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hc_pre_rms(
    void *smem_base,
    void *task_scratch,
    const MInst *st_insts,
    int sm_id,
    uint64_t *g_events,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHc = 4;
  constexpr int kHidden = 4096;
  constexpr int kVectorWidth = 8;
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
#if !defined(DAE_TRACK_PROFILE)
  (void)sm_id;
  (void)g_events;
#endif

  int square_sum_slots;
  int mixes_slots;
  int scale_slots;
  int base_slots;
  int packed_metadata_slot;
  int norm_weight_slots;
  const __nv_bfloat16 *norm_weight;
  const float *square_sum;
  const float *mixes;
  const float *scale;
  const float *base;
  if constexpr (MetadataMode == Dsv4HcPreRmsMetadataMode::SeparateShared) {
    square_sum_slots = m2c.template pop<0>();
    square_sum = static_cast<const float *>(
        get_slot_address(smem_base, extract(square_sum_slots)));
    mixes_slots = m2c.template pop<0>();
    mixes = static_cast<const float *>(
        get_slot_address(smem_base, extract(mixes_slots)));
    scale_slots = m2c.template pop<0>();
    scale = static_cast<const float *>(
        get_slot_address(smem_base, extract(scale_slots)));
    base_slots = m2c.template pop<0>();
    base = static_cast<const float *>(
        get_slot_address(smem_base, extract(base_slots)));
  } else {
    packed_metadata_slot = m2c.template pop<0>();
    const float *metadata = nullptr;
    if constexpr (MetadataMode == Dsv4HcPreRmsMetadataMode::PackedShared) {
      metadata = static_cast<const float *>(get_slot_address(
          smem_base, extract(packed_metadata_slot)));
    } else {
      static_assert(MetadataMode == Dsv4HcPreRmsMetadataMode::PackedRaw);
      metadata = static_cast<const float *>(slot_2_glob_ptr(
          st_insts, packed_metadata_slot));
    }
    square_sum = metadata;
    mixes = metadata + 1;
    scale = metadata + dsv4HcPreRmsScaleOffset;
    base = metadata + dsv4HcPreRmsBaseOffset;
  }

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(task_scratch);
  auto *warp_reduce = shared;
  auto *pre = shared + 4;
  auto *post_values = shared + 9;
  auto *comb_values = shared + 13;

  if (warp == 0) {
    float coefficient_rstd = 0.0f;
    if (lane == 0) {
      coefficient_rstd = rsqrtf(
          square_sum[0] / float(kHc * kHidden) + 1.0e-6f);
    }
    coefficient_rstd = __shfl_sync(
        0xFFFFFFFFU, coefficient_rstd, 0);
    if (lane < kHc) {
      pre[lane] = dsv4_sigmoid(
          mixes[lane] * coefficient_rstd * scale[0] + base[lane]) +
          dsv4HcPreRmsEpsilon;
    }
    if (lane == 0) {
      shared[8] = coefficient_rstd;
      *reinterpret_cast<int *>(shared + 29) = 0;
      *reinterpret_cast<int *>(shared + 30) = 0;
    }
  }
  __sync_compute_group(128);

  const int residual_slots = m2c.template pop<0>();
  const auto *residual = static_cast<const __nv_bfloat16 *>(
      get_slot_address(smem_base, extract(residual_slots)));
#if defined(DAE_TRACK_PROFILE)
  if (tid == 0) {
    g_events[sm_id * numProfileEvents + 125] =
        cuda::ptx::get_sreg_globaltimer();
  }
#endif
  auto *hidden = reinterpret_cast<__nv_bfloat16 *>(shared + 64);

  // Warp zero owns the coefficient transform while the other three warps
  // consume the already-published pre coefficients.  One element per lane
  // turns each 4x4 row/column normalization into two shuffles instead of four
  // serial scalar normalizations in each of four lanes.
  if (warp == 0) {
    const float coefficient_rstd = shared[8];
    if (lane < kHc) {
      post_values[lane] = 2.0f * dsv4_sigmoid(
          mixes[kHc + lane] * coefficient_rstd * scale[1] +
          base[kHc + lane]);
    }
    if (lane < kHc * kHc) {
      float comb = mixes[2 * kHc + lane] * coefficient_rstd * scale[2] +
                   base[2 * kHc + lane];
      constexpr unsigned kCombMask = (1U << (kHc * kHc)) - 1U;
      float row_max = comb;
      row_max = fmaxf(
          row_max, __shfl_xor_sync(kCombMask, row_max, 1));
      row_max = fmaxf(
          row_max, __shfl_xor_sync(kCombMask, row_max, 2));
      comb = __expf(comb - row_max);
      float row_sum = comb;
      row_sum += __shfl_xor_sync(kCombMask, row_sum, 1);
      row_sum += __shfl_xor_sync(kCombMask, row_sum, 2);
      comb = comb / row_sum + dsv4HcPreRmsEpsilon;
      float column_sum = comb;
      column_sum += __shfl_xor_sync(kCombMask, column_sum, 4);
      column_sum += __shfl_xor_sync(kCombMask, column_sum, 8);
      comb /= column_sum + dsv4HcPreRmsEpsilon;
#pragma unroll
      for (int iteration = 1;
           iteration < dsv4HcPreRmsSinkhornIters;
           ++iteration) {
        row_sum = comb;
        row_sum += __shfl_xor_sync(kCombMask, row_sum, 1);
        row_sum += __shfl_xor_sync(kCombMask, row_sum, 2);
        comb /= row_sum + dsv4HcPreRmsEpsilon;
        column_sum = comb;
        column_sum += __shfl_xor_sync(kCombMask, column_sum, 4);
        column_sum += __shfl_xor_sync(kCombMask, column_sum, 8);
        comb /= column_sum + dsv4HcPreRmsEpsilon;
      }
      comb_values[lane] = comb;
    }
#if defined(DAE_TRACK_PROFILE)
    if (tid == 0) {
      g_events[sm_id * numProfileEvents + 124] =
          cuda::ptx::get_sreg_globaltimer();
    }
#endif
  }

  auto *worker_count = reinterpret_cast<int *>(shared + 29);
  auto *rms_ready = reinterpret_cast<int *>(shared + 30);
  auto *worker_rms_rcp = shared + 31;
  if (warp != 0) {
    constexpr int kWorkerThreads = 96;
    const int worker = tid - 32;
    float hidden_sum = 0.0f;
    for (int dim = worker * kVectorWidth; dim < kHidden;
         dim += kWorkerThreads * kVectorWidth) {
      hidden_sum += dsv4_hc_mix_hidden_vector(
          residual, pre, hidden, dim);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      hidden_sum += __shfl_down_sync(0xFFFFFFFFU, hidden_sum, offset);
    }
    if (lane == 0) {
      warp_reduce[warp] = hidden_sum;
      __threadfence_block();
      atomicAdd(worker_count, 1);
      while (atomicAdd(worker_count, 0) < 3) {
      }
    }
    __syncwarp();
    if (tid == 32) {
      const float total =
          warp_reduce[1] + warp_reduce[2] + warp_reduce[3];
      *worker_rms_rcp = rsqrtf(
          total / float(kHidden) + dsv4HcPreRmsNormEpsilon);
      __threadfence_block();
      atomicExch(rms_ready, 1);
    }
    if (lane == 0) {
      while (atomicAdd(rms_ready, 0) == 0) {
      }
    }
    __syncwarp();
  }

  norm_weight_slots = m2c.template pop<0>();
  norm_weight = static_cast<const __nv_bfloat16 *>(get_slot_address(
      smem_base, extract(norm_weight_slots)));
  const int output_slots = m2c.template pop<0>();
  __nv_bfloat16 *output;
  Fp8 *fp8_output;
  if constexpr (OutputFp8) {
    fp8_output = static_cast<Fp8 *>(
        get_slot_address(smem_base, extract(output_slots)));
  } else {
    output = static_cast<__nv_bfloat16 *>(
        get_slot_address(smem_base, extract(output_slots)));
  }
  auto *global_output = static_cast<uint8_t *>(slot_2_glob_ptr(
      st_insts, extract(output_slots)));
  constexpr int kPrimaryOutputBytes = OutputFp8 ? kHidden : kHidden * 2;
  auto *post_output = reinterpret_cast<float *>(
      global_output + kPrimaryOutputBytes);
  auto *comb_output = post_output + kHc;
  int fp8_scale_slots;
  Scale *fp8_scale;
  if constexpr (OutputFp8) {
    fp8_scale_slots = m2c.template pop<0>();
    fp8_scale = static_cast<Scale *>(get_slot_address(
        smem_base, extract(fp8_scale_slots)));
  }
  int zero_output_slots;
  float *zero_output;
  if constexpr (ZeroFp32Output) {
    zero_output_slots = m2c.template pop<0>();
    zero_output = static_cast<float *>(
        get_slot_address(smem_base, extract(zero_output_slots)));
  }

  if (warp == 0) {
    if (lane < kHc) {
      post_output[lane] = post_values[lane];
    }
    if (lane < kHc * kHc) {
      comb_output[lane] = comb_values[lane];
    }
  } else if constexpr (!OutputFp8) {
    constexpr int kWorkerThreads = 96;
    const int worker = tid - 32;
    const float rms_rcp = *worker_rms_rcp;
    for (int dim = worker * kVectorWidth; dim < kHidden;
         dim += kWorkerThreads * kVectorWidth) {
      const Dsv4Bf16x8 hidden_values = dsv4_load_bf16x8(hidden + dim);
      const Dsv4Bf16x8 weight_values = dsv4_load_bf16x8(norm_weight + dim);
      Dsv4Bf16x8 normalized;
      auto *normalized_pairs =
          reinterpret_cast<__nv_bfloat162 *>(&normalized.raw);
#pragma unroll
      for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
        const float2 hidden_pair = __bfloat1622float2(
            dsv4_bf16x8_pair(hidden_values, pair));
        const float2 weight_pair = __bfloat1622float2(
            dsv4_bf16x8_pair(weight_values, pair));
        normalized_pairs[pair] = __float22bfloat162_rn(make_float2(
            hidden_pair.x * rms_rcp * weight_pair.x,
            hidden_pair.y * rms_rcp * weight_pair.y));
      }
      dsv4_store_bf16x8(output + dim, normalized);
      if constexpr (ZeroFp32Output) {
        auto *zero_vectors = reinterpret_cast<uint4 *>(zero_output + dim);
        zero_vectors[0] = make_uint4(0, 0, 0, 0);
        zero_vectors[1] = make_uint4(0, 0, 0, 0);
      }
    }
#if defined(DAE_TRACK_PROFILE)
    if (tid == 32) {
      g_events[sm_id * numProfileEvents + 126] =
          cuda::ptx::get_sreg_globaltimer();
    }
#endif
  }

  if constexpr (OutputFp8) {
    // Eight 16-lane groups quantize eight independent K128 blocks at once.
    // Each lane owns one aligned BF16x8 vector. The BF16 rounding remains part
    // of the model math, but only the selected E4M3 representation is written.
    constexpr int kBlock = 128;
    constexpr int kGroups = 128 / (kBlock / kVectorWidth);
    constexpr int kLanesPerGroup = kBlock / kVectorWidth;
    constexpr int kBlocks = kHidden / kBlock;
    const int group = tid / kLanesPerGroup;
    const int group_lane = tid % kLanesPerGroup;
    const unsigned group_mask =
        (tid & 16) == 0 ? 0x0000FFFFU : 0xFFFF0000U;
    const float rms_rcp = *worker_rms_rcp;
    for (int block_base = 0; block_base < kBlocks; block_base += kGroups) {
      const int block = block_base + group;
      const int dim = block * kBlock + group_lane * kVectorWidth;
      const Dsv4Bf16x8 hidden_values = dsv4_load_bf16x8(hidden + dim);
      const Dsv4Bf16x8 weight_values = dsv4_load_bf16x8(norm_weight + dim);
      Dsv4Bf16x8 normalized;
      auto *normalized_pairs =
          reinterpret_cast<__nv_bfloat162 *>(&normalized.raw);
      float normalized_values[kVectorWidth];
      float maximum = 0.0f;
#pragma unroll
      for (int pair = 0; pair < kVectorWidth / 2; ++pair) {
        const float2 hidden_pair = __bfloat1622float2(
            dsv4_bf16x8_pair(hidden_values, pair));
        const float2 weight_pair = __bfloat1622float2(
            dsv4_bf16x8_pair(weight_values, pair));
        normalized_pairs[pair] = __float22bfloat162_rn(make_float2(
            hidden_pair.x * rms_rcp * weight_pair.x,
            hidden_pair.y * rms_rcp * weight_pair.y));
        const float2 rounded = __bfloat1622float2(normalized_pairs[pair]);
        normalized_values[pair * 2] = rounded.x;
        normalized_values[pair * 2 + 1] = rounded.y;
        maximum = fmaxf(maximum, fabsf(rounded.x));
        maximum = fmaxf(maximum, fabsf(rounded.y));
      }
#pragma unroll
      for (int offset = kLanesPerGroup / 2; offset > 0; offset >>= 1) {
        maximum = fmaxf(
            maximum,
            __shfl_down_sync(group_mask, maximum, offset, kLanesPerGroup));
      }
      float block_scale = 0.0f;
      if (group_lane == 0) {
        const float requested = fmaxf(maximum / 448.0f, 0x1p-127f);
        const float exponent = ceilf(log2f(requested));
        block_scale = exp2f(fminf(fmaxf(exponent, -127.0f), 127.0f));
        fp8_scale[block] = Scale(block_scale);
      }
      block_scale = __shfl_sync(
          group_mask, block_scale, 0, kLanesPerGroup);
#pragma unroll
      for (int element = 0; element < kVectorWidth; ++element) {
        fp8_output[dim + element] = Fp8(fminf(
            fmaxf(normalized_values[element] / block_scale, -448.0f),
            448.0f));
      }
      if constexpr (ZeroFp32Output) {
        auto *zero_vectors = reinterpret_cast<uint4 *>(zero_output + dim);
        zero_vectors[0] = make_uint4(0, 0, 0, 0);
        zero_vectors[1] = make_uint4(0, 0, 0, 0);
      }
    }
#if defined(DAE_TRACK_PROFILE)
    if (tid == 0) {
      g_events[sm_id * numProfileEvents + 126] =
          cuda::ptx::get_sreg_globaltimer();
    }
#endif
  }

  __sync_compute_group(128);
  int input_slots = residual_slots | norm_weight_slots;
  if constexpr (MetadataMode == Dsv4HcPreRmsMetadataMode::SeparateShared) {
    input_slots |= square_sum_slots | mixes_slots | scale_slots | base_slots;
  } else if constexpr (
      MetadataMode == Dsv4HcPreRmsMetadataMode::PackedShared) {
    input_slots |= packed_metadata_slot;
  }
  c2m.push(tid, input_slots);
  c2m.template push<31, true, false>(tid, output_slots);
  if constexpr (OutputFp8) {
    c2m.template push<31, true, false>(tid, fp8_scale_slots);
  }
  if constexpr (ZeroFp32Output) {
    c2m.template push<31, true, false>(tid, zero_output_slots);
  }
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hc_post(
    int width,
    bool branch_fp32,
    void *smem_base,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHc = 4;

  const int branch_slots = m2c.template pop<0>();
  const auto *branch = get_slot_address(smem_base, extract(branch_slots));
  int residual_slots[kHc];
  const __nv_bfloat16 *residual[kHc];
  for (int branch_index = 0; branch_index < kHc; ++branch_index) {
    residual_slots[branch_index] = m2c.template pop<0>();
    residual[branch_index] = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, extract(residual_slots[branch_index])));
  }
  const int post_slots = m2c.template pop<0>();
  const auto *post = static_cast<const float *>(
      get_slot_address(smem_base, extract(post_slots)));
  const int comb_slots = m2c.template pop<0>();
  const auto *comb = static_cast<const float *>(
      get_slot_address(smem_base, extract(comb_slots)));
  int output_slots[kHc];
  __nv_bfloat16 *output[kHc];
  for (int branch_index = 0; branch_index < kHc; ++branch_index) {
    output_slots[branch_index] = m2c.template pop<0>();
    output[branch_index] = static_cast<__nv_bfloat16 *>(
        get_slot_address(smem_base, extract(output_slots[branch_index])));
  }

  const int tid = __compute_tid();
  for (int output_branch = 0; output_branch < kHc; ++output_branch) {
    for (int dim = tid; dim < width; dim += 128) {
      const float branch_value = branch_fp32
          ? static_cast<const float *>(branch)[dim]
          : __bfloat162float(
                static_cast<const __nv_bfloat16 *>(branch)[dim]);
      float value = post[output_branch] * branch_value;
      for (int input_branch = 0; input_branch < kHc; ++input_branch) {
        // The model updates streams with comb^T @ residual.
        value = fmaf(
            comb[input_branch * kHc + output_branch],
            __bfloat162float(residual[input_branch][dim]),
            value);
      }
      output[output_branch][dim] = __float2bfloat16(value);
    }
  }

  __sync_compute_group(128);
  int input_slots = branch_slots | post_slots | comb_slots;
  for (int branch_index = 0; branch_index < kHc; ++branch_index) {
    input_slots |= residual_slots[branch_index];
  }
  c2m.push(tid, input_slots);
  for (int branch_index = 0; branch_index < kHc; ++branch_index) {
    c2m.template push<31, true, false>(tid, output_slots[branch_index]);
  }
}
