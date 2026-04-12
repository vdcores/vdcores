#pragma once

#include "opt_attention_barriers.cuh"
#include "opt_attention_types.cuh"

#include <cuda_runtime.h>

#include <cfloat>

namespace opt_attention {

template <typename scalar_t>
struct SharedStorage {
  PipelineBarriers barriers;
  alignas(16) scalar_t k[2][kKvTile * kHeadDim];
  alignas(16) scalar_t v[2][kKvTile * kHeadDim];
  alignas(16) float scores[kKvTile];
};

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* query_ptr(const OptAttentionParams& params, int batch, int head) {
  const auto* base = static_cast<const scalar_t*>(params.query);
  return base + batch * params.q_stride_b + head * params.q_stride_h;
}

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* key_ptr(
    const OptAttentionParams& params,
    int batch,
    int head,
    int token) {
  const auto* base = static_cast<const scalar_t*>(params.key);
  return base + batch * params.k_stride_b + head * params.k_stride_h + token * params.k_stride_s;
}

template <typename scalar_t>
__device__ __forceinline__ const scalar_t* value_ptr(
    const OptAttentionParams& params,
    int batch,
    int head,
    int token) {
  const auto* base = static_cast<const scalar_t*>(params.value);
  return base + batch * params.v_stride_b + head * params.v_stride_h + token * params.v_stride_s;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t* output_ptr(const OptAttentionParams& params, int batch, int head) {
  auto* base = static_cast<scalar_t*>(params.output);
  return base + batch * params.o_stride_b + head * params.o_stride_h;
}

template <typename scalar_t>
__device__ __forceinline__ void producer_kv_copy_tile(
    const OptAttentionParams& params,
    SharedStorage<scalar_t>& smem,
    const scalar_t* k0,
    const scalar_t* v0,
    int stage,
    int lane) {
  constexpr int elems_per_vec = sizeof(uint4) / sizeof(scalar_t);
  constexpr int vec_count = kKvTile * kHeadDim / elems_per_vec;
  constexpr int copies_per_lane = vec_count / kProducerThreads;
  static_assert(vec_count % kProducerThreads == 0);

  auto* sk_vec = reinterpret_cast<uint4*>(&smem.k[stage][0]);
  auto* sv_vec = reinterpret_cast<uint4*>(&smem.v[stage][0]);
  #pragma unroll 1
  for (int iter = 0; iter < copies_per_lane; ++iter) {
    const int vec = iter * kProducerThreads + lane;
    const int elem = vec * elems_per_vec;
    const int row = elem / kHeadDim;
    const int col = elem - row * kHeadDim;
    const scalar_t* k = k0 + row * params.k_stride_s + col;
    const scalar_t* v = v0 + row * params.v_stride_s + col;
    cp_async_16B(&sk_vec[vec], k);
    cp_async_16B(&sv_vec[vec], v);
  }
}

template <typename scalar_t>
__device__ void producer_kv(
    const OptAttentionParams& params,
    SharedStorage<scalar_t>& smem,
    int kv_start,
    int kv_end,
    int batch,
    int head) {
  const int lane = threadIdx.x - kComputeThreads;
  const int chunk_len = max(0, kv_end - kv_start);
  const int num_blocks = (chunk_len + kKvTile - 1) / kKvTile;

  for (int block = 0; block < num_blocks; ++block) {
    const int stage = block & 1;
    if (block >= 2) {
      smem.barriers.kv_drain[stage].arrive_and_wait();
    }

    const int block_start = kv_start + block * kKvTile;
    const scalar_t* k0 = key_ptr<scalar_t>(params, batch, head, block_start);
    const scalar_t* v0 = value_ptr<scalar_t>(params, batch, head, block_start);

    producer_kv_copy_tile(params, smem, k0, v0, stage, lane);
    producer_async_arrive(smem.barriers.kv_fill[stage]);
    producer_arrive(smem.barriers.kv_fill[stage]);
  }
}

__device__ __forceinline__ float mask_value(const OptAttentionParams& params, int batch, int token) {
  if (params.mask == nullptr) {
    return 0.0f;
  }
  return params.mask[batch * params.m_stride_b + token * params.m_stride_s];
}

template <typename scalar_t>
__device__ void compute_score_group(
    const OptAttentionParams& params,
    SharedStorage<scalar_t>& smem,
    int batch,
    int head,
    int block,
    int stage,
    int kv_start,
    int row_base,
    int compute_tid) {
  const int warp_id = compute_tid / 32;
  const int lane_id = compute_tid & 31;
  const int row = row_base + warp_id;
  const int token = kv_start + block * kKvTile + row;
  const scalar_t* q = query_ptr<scalar_t>(params, batch, head);

  float partial = 0.0f;
  #pragma unroll
  for (int col = lane_id; col < kHeadDim; col += 32) {
    const float q_value = ScalarIO<scalar_t>::load(&q[col]);
    const float k = ScalarIO<scalar_t>::load(&smem.k[stage][row * kHeadDim + col]);
    partial += q_value * k;
  }

  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xffffffffU, partial, offset);
  }

  if (lane_id == 0) {
    smem.scores[row] = partial * params.scaling + mask_value(params, batch, token);
  }
  compute_group_sync();
}

template <typename scalar_t>
__device__ void compute_attention(
    const OptAttentionParams& params,
    SharedStorage<scalar_t>& smem,
    int kv_start,
    int kv_end,
    int batch,
    int head,
    int split_idx) {
  const int compute_tid = threadIdx.x;
  const int chunk_len = max(0, kv_end - kv_start);
  const int num_blocks = (chunk_len + kKvTile - 1) / kKvTile;
  float row_max = -FLT_MAX;
  float row_sum = 0.0f;
  float acc = 0.0f;

  for (int block = 0; block < num_blocks; ++block) {
    const int stage = block & 1;
    smem.barriers.kv_fill[stage].arrive_and_wait();

    #pragma unroll
    for (int row_base = 0; row_base < kKvTile; row_base += 4) {
      compute_score_group(params, smem, batch, head, block, stage, kv_start, row_base, compute_tid);

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = row_base + i;
        const float score = smem.scores[row];
        const float value = ScalarIO<scalar_t>::load(&smem.v[stage][row * kHeadDim + compute_tid]);
        const float new_max = fmaxf(row_max, score);
        const float old_scale = (row_max == -FLT_MAX) ? 0.0f : __expf(row_max - new_max);
        const float score_scale = __expf(score - new_max);
        acc = acc * old_scale + value * score_scale;
        row_sum = row_sum * old_scale + score_scale;
        row_max = new_max;
      }
      compute_group_sync();
    }

    compute_arrive(smem.barriers.kv_drain[stage]);
  }

  if (params.num_splits <= 1) {
    scalar_t* out = output_ptr<scalar_t>(params, batch, head);
    const float normalized = row_sum == 0.0f ? 0.0f : acc / row_sum;
    out[compute_tid * params.o_stride_d] = ScalarIO<scalar_t>::store(normalized);
  } else {
    const int partial_base = ((batch * params.num_heads + head) * params.num_splits + split_idx);
    params.partial_out[partial_base * kHeadDim + compute_tid] = acc;
    if (compute_tid == 0) {
      params.partial_m[partial_base] = row_max;
      params.partial_l[partial_base] = row_sum;
    }
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kThreads, 1) void opt_attention_decode_kernel(OptAttentionParams params) {
  __shared__ SharedStorage<scalar_t> smem;
  init_pipeline_barriers(smem.barriers);

  const int split_idx = blockIdx.x % params.num_splits;
  const int head = (blockIdx.x / params.num_splits) % params.num_heads;
  const int batch = blockIdx.x / (params.num_heads * params.num_splits);
  const int kv_start = split_idx * params.split_size;
  const int kv_end = min(params.key_seq_len, kv_start + params.split_size);

  if (threadIdx.x < kComputeThreads) {
    compute_attention(params, smem, kv_start, kv_end, batch, head, split_idx);
  } else {
    producer_kv(params, smem, kv_start, kv_end, batch, head);
  }
}

template <typename scalar_t>
__global__ __launch_bounds__(kComputeThreads, 1) void opt_attention_reduce_splits_kernel(OptAttentionParams params) {
  const int compute_tid = threadIdx.x;
  const int head = blockIdx.x % params.num_heads;
  const int batch = blockIdx.x / params.num_heads;
  const int partial_base = (batch * params.num_heads + head) * params.num_splits;

  float global_m = -FLT_MAX;
  for (int split = 0; split < params.num_splits; ++split) {
    global_m = fmaxf(global_m, params.partial_m[partial_base + split]);
  }

  float denom = 0.0f;
  float acc = 0.0f;
  for (int split = 0; split < params.num_splits; ++split) {
    const float m = params.partial_m[partial_base + split];
    const float l = params.partial_l[partial_base + split];
    const float scale = (m == -FLT_MAX || global_m == -FLT_MAX) ? 0.0f : expf(m - global_m);
    denom += l * scale;
    acc += params.partial_out[(partial_base + split) * kHeadDim + compute_tid] * scale;
  }

  scalar_t* out = output_ptr<scalar_t>(params, batch, head);
  out[compute_tid * params.o_stride_d] = ScalarIO<scalar_t>::store(denom == 0.0f ? 0.0f : acc / denom);
}

}  // namespace opt_attention
