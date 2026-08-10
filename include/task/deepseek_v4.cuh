#pragma once

#include "context.cuh"
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
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Fp8 = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;

  const int input_slot = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, input_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<Fp8 *>(
      slot_2_glob_ptr(st_insts, output_slot));
  const int scale_slot = m2c.template pop<0>();
  auto *scales = static_cast<Scale *>(
      slot_2_glob_ptr(st_insts, scale_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(smem_base);
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

  __threadfence();
  c2m.template push<31, true, false>(
      tid, (1U << output_slot) | (1U << scale_slot));
}

// Quantize one BF16 activation vector to ModelOpt's packed E2M1/per-16 E4M3
// representation using the checkpoint-provided scalar dequantization scale.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_nvfp4_quant16(
    int k,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using Scale = cutlass::float_e4m3_t;

  const int input_slot = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, input_slot));
  const int global_scale_slot = m2c.template pop<0>();
  const auto *global_scale = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, global_scale_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<uint8_t *>(
      slot_2_glob_ptr(st_insts, output_slot));
  const int scale_slot = m2c.template pop<0>();
  auto *scales = static_cast<Scale *>(
      slot_2_glob_ptr(st_insts, scale_slot));

  const int tid = __compute_tid();
  auto *shared = static_cast<float *>(smem_base);
  for (int block = 0; block < k / 16; ++block) {
    if (tid == 0) {
      float maximum = 0.0f;
#pragma unroll
      for (int element = 0; element < 16; ++element) {
        maximum = fmaxf(
            maximum,
            fabsf(__bfloat162float(input[block * 16 + element])));
      }
      shared[0] = dsv4_ceil_e4m3(
          maximum / (6.0f * global_scale[0]));
      scales[block] = Scale(shared[0]);
    }
    __sync_compute_group(128);
    if (tid < 8) {
      const int first = block * 16 + tid * 2;
      const float inverse_scale = 1.0f / (shared[0] * global_scale[0]);
      const uint8_t low = dsv4_nearest_fp4(
          __bfloat162float(input[first]) * inverse_scale);
      const uint8_t high = dsv4_nearest_fp4(
          __bfloat162float(input[first + 1]) * inverse_scale);
      output[block * 8 + tid] = low | (high << 4);
    }
    __sync_compute_group(128);
  }

  __threadfence();
  c2m.template push<31, true, false>(
      tid, (1U << output_slot) | (1U << scale_slot));
}

// Apply the DeepSeek partial rotary embedding to the final 64 dimensions of
// each attention (512-wide) or indexer (128-wide) row.  The table is float32
// [32, 2] in (cos, sin) order.
template <int kHeadDim, typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_rope_64(
    int rows,
    bool inverse,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kRopeDim = 64;
  constexpr int kRopeStart = kHeadDim - kRopeDim;

  const int input_slot = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, input_slot));
  const int table_slot = m2c.template pop<0>();
  const auto *table = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, table_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  constexpr int kPairsPerRow = kRopeDim / 2;
  for (int item = tid; item < rows * kPairsPerRow; item += 128) {
    const int row = item / kPairsPerRow;
    const int pair = item % kPairsPerRow;
    const int offset = row * kHeadDim + kRopeStart + pair * 2;
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
  for (int item = tid; item < rows * kRopeStart; item += 128) {
    const int row = item / kRopeStart;
    const int dim = item % kRopeStart;
    output[row * kHeadDim + dim] = input[row * kHeadDim + dim];
  }

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// Correctness-first sparse decode attention for the DeepSeek 64x512 query and
// shared 512-wide KV cache.  One resident SM owns one query head and walks the
// selected KV indices with online softmax.  The attention sink is represented
// as an extra denominator-only logit, matching the official inference path.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_sparse_attention_512(
    int head,
    int topk,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHeadDim = 512;
  constexpr int kValuesPerThread = kHeadDim / 128;
  constexpr float kSoftmaxScale = 0.04419417382415922f;  // 1 / sqrt(512)

  const int q_slot = m2c.template pop<0>();
  const auto *q = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, q_slot)) + head * kHeadDim;
  const int kv_slot = m2c.template pop<0>();
  const auto *kv = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, kv_slot));
  const int indices_slot = m2c.template pop<0>();
  const auto *indices = static_cast<const int *>(
      slot_2_glob_ptr(st_insts, indices_slot));
  const int sink_slot = m2c.template pop<0>();
  const auto *sink = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, sink_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot)) + head * kHeadDim;

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *warp_reduce = static_cast<float *>(smem_base);

  float q_values[kValuesPerThread];
  float accum[kValuesPerThread] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    q_values[item] = __bfloat162float(q[tid + item * 128]);
  }

  float running_max = sink[head];
  float running_sum = 1.0f;
  for (int selected = 0; selected < topk; ++selected) {
    const int kv_row = indices[selected];
    if (kv_row < 0) {
      continue;
    }
    const auto *kv_ptr = kv + kv_row * kHeadDim;
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
  }

  const float inverse_sum = 1.0f / running_sum;
#pragma unroll
  for (int item = 0; item < kValuesPerThread; ++item) {
    output[tid + item * 128] = __float2bfloat16(accum[item] * inverse_sum);
  }

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// Select DeepSeek's top-6 routed experts from 256 gate logits.  Hash layers
// provide the six expert ids directly but still use the transformed scores as
// routing weights.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_route_top6(
    bool hash_routing,
    float route_scale,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kExperts = 256;
  constexpr int kTopK = 6;

  const int logits_slot = m2c.template pop<0>();
  const auto *logits = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, logits_slot));
  const int bias_slot = m2c.template pop<0>();
  const auto *bias = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, bias_slot));
  const int hash_slot = m2c.template pop<0>();
  const auto *hash_indices = static_cast<const int *>(
      slot_2_glob_ptr(st_insts, hash_slot));
  const int indices_slot = m2c.template pop<0>();
  auto *output_indices = static_cast<int *>(
      slot_2_glob_ptr(st_insts, indices_slot));
  const int weights_slot = m2c.template pop<0>();
  auto *output_weights = static_cast<float *>(
      slot_2_glob_ptr(st_insts, weights_slot));

  const int tid = __compute_tid();
  auto *original_scores = static_cast<float *>(smem_base);
  auto *selection_scores = original_scores + kExperts;
  for (int expert = tid; expert < kExperts; expert += 128) {
    const float transformed =
        sqrtf(dsv4_softplus(__bfloat162float(logits[expert])));
    original_scores[expert] = transformed;
    selection_scores[expert] = transformed + bias[expert];
  }
  __sync_compute_group(128);

  if (tid == 0) {
    int selected_indices[kTopK];
    if (hash_routing) {
#pragma unroll
      for (int rank = 0; rank < kTopK; ++rank) {
        selected_indices[rank] = hash_indices[rank];
      }
    } else {
      float selected_scores[kTopK];
#pragma unroll
      for (int rank = 0; rank < kTopK; ++rank) {
        selected_scores[rank] = -__int_as_float(0x7f800000);
        selected_indices[rank] = -1;
      }
      for (int expert = 0; expert < kExperts; ++expert) {
        const float candidate = selection_scores[expert];
        int insert = kTopK;
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
          if (candidate > selected_scores[rank]) {
            insert = rank;
            break;
          }
        }
        if (insert < kTopK) {
          for (int rank = kTopK - 1; rank > insert; --rank) {
            selected_scores[rank] = selected_scores[rank - 1];
            selected_indices[rank] = selected_indices[rank - 1];
          }
          selected_scores[insert] = candidate;
          selected_indices[insert] = expert;
        }
      }
    }

    float weight_sum = 0.0f;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
      weight_sum += original_scores[selected_indices[rank]];
    }
    const float normalization = route_scale / weight_sum;
#pragma unroll
    for (int rank = 0; rank < kTopK; ++rank) {
      output_indices[rank] = selected_indices[rank];
      output_weights[rank] =
          original_scores[selected_indices[rank]] * normalization;
    }
  }

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(
      tid, (1U << indices_slot) | (1U << weights_slot));
}

// Sum the six routed expert down projections and one shared expert.  Applying
// the route weight after w2 is equivalent to the official pre-w2 weighting
// because the down projection is linear.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_expert_reduce(
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHidden = 4096;
  constexpr int kTopK = 6;

  const int routed_slot = m2c.template pop<0>();
  const auto *routed = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, routed_slot));
  const int weights_slot = m2c.template pop<0>();
  const auto *weights = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, weights_slot));
  const int shared_slot = m2c.template pop<0>();
  const auto *shared = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, shared_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot));

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
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// FP32-weight/BF16-input GEMV for the small mHC mixing projections.  It is
// deliberately scalar and correctness-oriented; one SM owns one or more rows.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_fp32_bf16_gemv(
    int rows,
    int k,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  const int weight_slot = m2c.template pop<0>();
  const auto *weight = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, weight_slot));
  const int input_slot = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, input_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<float *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *warp_reduce = static_cast<float *>(smem_base);
  for (int local_row = 0; local_row < rows; ++local_row) {
    float partial = 0.0f;
    for (int column = tid; column < k; column += 128) {
      partial = fmaf(
          weight[local_row * k + column],
          __bfloat162float(input[column]),
          partial);
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
      output[local_row] = warp_reduce[0] + warp_reduce[1] +
                          warp_reduce[2] + warp_reduce[3];
    }
    __sync_compute_group(128);
  }

  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// Normalized Walsh-Hadamard transform used by the ratio-4 indexer before its
// simulated FP4 dot products.  One SM owns one 128- or 512-wide row.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hadamard(
    int row,
    int width,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  const int input_slot = m2c.template pop<0>();
  const auto *input = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, input_slot)) + row * width;
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot)) + row * width;

  const int tid = __compute_tid();
  auto *values = static_cast<float *>(smem_base);
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
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// Dimension-wise gated pooling for compressed KV state.  The caller supplies
// the contiguous rows selected by the ratio-4 overlap rule or ratio-128 rule,
// with positional APE already added to the FP32 scores.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_gated_pool(
    int pool_rows,
    int width,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  const int values_slot = m2c.template pop<0>();
  const auto *values = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, values_slot));
  const int scores_slot = m2c.template pop<0>();
  const auto *scores = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, scores_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  for (int dim = tid; dim < width; dim += 128) {
    float maximum = -__int_as_float(0x7f800000);
    for (int row = 0; row < pool_rows; ++row) {
      maximum = fmaxf(maximum, scores[row * width + dim]);
    }
    float denominator = 0.0f;
    float numerator = 0.0f;
    for (int row = 0; row < pool_rows; ++row) {
      const float probability = __expf(scores[row * width + dim] - maximum);
      denominator += probability;
      numerator = fmaf(probability, values[row * width + dim], numerator);
    }
    output[dim] = __float2bfloat16(numerator / denominator);
  }

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// Learned ratio-4 index score.  Each SM handles a contiguous KV-row shard;
// within a row the four warps independently reduce one head at a time.  This
// keeps all warps useful and needs only one warpgroup reduction per row.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_index_score(
    int rows,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHeads = 64;
  constexpr int kHeadDim = 128;

  const int q_slot = m2c.template pop<0>();
  const auto *q = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, q_slot));
  const int kv_slot = m2c.template pop<0>();
  const auto *kv = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, kv_slot));
  const int weights_slot = m2c.template pop<0>();
  const auto *head_weights = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, weights_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<float *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *warp_reduce = static_cast<float *>(smem_base);
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

  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
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
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kSortSize = 1024;
  constexpr int kRetained = 512;
  const int scores_slot = m2c.template pop<0>();
  const auto *scores = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, scores_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<int *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  auto *sort_scores = static_cast<float *>(smem_base);
  auto *sort_indices = reinterpret_cast<int *>(sort_scores + kSortSize);

  int processed = min(rows, kSortSize);
  for (int item = tid; item < kSortSize; item += 128) {
    sort_scores[item] = item < processed
        ? scores[item]
        : -__int_as_float(0x7f800000);
    sort_indices[item] = item < processed ? item : -1;
  }
  __sync_compute_group(128);

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
    for (int item = tid; item < kRetained; item += 128) {
      sort_scores[item] = item < chunk
          ? scores[processed + item]
          : -__int_as_float(0x7f800000);
      sort_indices[item] = item < chunk ? processed + item : -1;
    }
    processed += chunk;
    __sync_compute_group(128);
  }

  for (int rank = tid; rank < topk; rank += 128) {
    output[rank] = sort_indices[kSortSize - 1 - rank] + index_offset;
  }

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// Final/MTP hyper-connection head: normalize the four raw projection values,
// form sigmoid pre coefficients, and reduce [4,4096] to one hidden vector.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hc_head(
    float epsilon,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHc = 4;
  constexpr int kHidden = 4096;

  const int residual_slot = m2c.template pop<0>();
  const auto *residual = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, residual_slot));
  const int mixes_slot = m2c.template pop<0>();
  const auto *mixes = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, mixes_slot));
  const int scale_slot = m2c.template pop<0>();
  const auto *scale = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, scale_slot));
  const int base_slot = m2c.template pop<0>();
  const auto *base = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, base_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(smem_base);
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
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}

// Convert 24 mHC projection values into pre/post/comb coefficients and form
// the pre-branch 4096-wide hidden vector.  The residual is [4,4096].
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hc_pre(
    int sinkhorn_iters,
    float epsilon,
    void *smem_base,
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHc = 4;
  constexpr int kHidden = 4096;

  const int residual_slot = m2c.template pop<0>();
  const auto *residual = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, residual_slot));
  const int mixes_slot = m2c.template pop<0>();
  const auto *mixes = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, mixes_slot));
  const int scale_slot = m2c.template pop<0>();
  const auto *scale = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, scale_slot));
  const int base_slot = m2c.template pop<0>();
  const auto *base = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, base_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot));
  const int post_slot = m2c.template pop<0>();
  auto *post_output = static_cast<float *>(
      slot_2_glob_ptr(st_insts, post_slot));
  const int comb_slot = m2c.template pop<0>();
  auto *comb_output = static_cast<float *>(
      slot_2_glob_ptr(st_insts, comb_slot));

  const int tid = __compute_tid();
  const int lane = tid & 31;
  const int warp = tid >> 5;
  auto *shared = static_cast<float *>(smem_base);
  auto *warp_reduce = shared;
  auto *pre = shared + 4;
  auto *post = pre + kHc;
  auto *comb = post + kHc;

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
    const float rsqrt = rsqrtf(total / float(kHc * kHidden) + 1.0e-6f);
#pragma unroll
    for (int index = 0; index < kHc; ++index) {
      pre[index] = dsv4_sigmoid(
          mixes[index] * rsqrt * scale[0] + base[index]) + epsilon;
      post[index] = 2.0f * dsv4_sigmoid(
          mixes[kHc + index] * rsqrt * scale[1] + base[kHc + index]);
      post_output[index] = post[index];
    }

#pragma unroll
    for (int row = 0; row < kHc; ++row) {
      float row_max = -__int_as_float(0x7f800000);
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        const int index = row * kHc + column;
        comb[index] = mixes[2 * kHc + index] * rsqrt * scale[2] +
                      base[2 * kHc + index];
        row_max = fmaxf(row_max, comb[index]);
      }
      float row_sum = 0.0f;
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        const int index = row * kHc + column;
        comb[index] = __expf(comb[index] - row_max);
        row_sum += comb[index];
      }
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        comb[row * kHc + column] =
            comb[row * kHc + column] / row_sum + epsilon;
      }
    }

    for (int iteration = 0; iteration < sinkhorn_iters; ++iteration) {
#pragma unroll
      for (int column = 0; column < kHc; ++column) {
        float column_sum = 0.0f;
#pragma unroll
        for (int row = 0; row < kHc; ++row) {
          column_sum += comb[row * kHc + column];
        }
#pragma unroll
        for (int row = 0; row < kHc; ++row) {
          comb[row * kHc + column] /= column_sum + epsilon;
        }
      }
      if (iteration + 1 == sinkhorn_iters) {
        break;
      }
#pragma unroll
      for (int row = 0; row < kHc; ++row) {
        float row_sum = 0.0f;
#pragma unroll
        for (int column = 0; column < kHc; ++column) {
          row_sum += comb[row * kHc + column];
        }
#pragma unroll
        for (int column = 0; column < kHc; ++column) {
          comb[row * kHc + column] /= row_sum + epsilon;
        }
      }
    }
#pragma unroll
    for (int index = 0; index < kHc * kHc; ++index) {
      comb_output[index] = comb[index];
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
  __threadfence();
  c2m.template push<31, true, false>(
      tid, (1U << output_slot) | (1U << post_slot) | (1U << comb_slot));
}

template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_hc_post(
    const MInst *st_insts,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  constexpr int kHc = 4;
  constexpr int kHidden = 4096;

  const int branch_slot = m2c.template pop<0>();
  const auto *branch = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, branch_slot));
  const int residual_slot = m2c.template pop<0>();
  const auto *residual = static_cast<const __nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, residual_slot));
  const int post_slot = m2c.template pop<0>();
  const auto *post = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, post_slot));
  const int comb_slot = m2c.template pop<0>();
  const auto *comb = static_cast<const float *>(
      slot_2_glob_ptr(st_insts, comb_slot));
  const int output_slot = m2c.template pop<0>();
  auto *output = static_cast<__nv_bfloat16 *>(
      slot_2_glob_ptr(st_insts, output_slot));

  const int tid = __compute_tid();
  for (int item = tid; item < kHc * kHidden; item += 128) {
    const int output_branch = item / kHidden;
    const int dim = item % kHidden;
    float value = post[output_branch] * __bfloat162float(branch[dim]);
#pragma unroll
    for (int input_branch = 0; input_branch < kHc; ++input_branch) {
      value = fmaf(
          comb[output_branch * kHc + input_branch],
          __bfloat162float(residual[input_branch * kHidden + dim]),
          value);
    }
    output[item] = __float2bfloat16(value);
  }

  __sync_compute_group(128);
  __threadfence();
  c2m.template push<31, true, false>(tid, 1U << output_slot);
}
