#pragma once

#include "virtualcore.cuh"
#include "type.cuh"

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    T other = (T)__shfl_down_sync(0xffffffff, (float)val, offset);
    val = other > val ? other : val;
  }
  return val;
}

template <typename data_t, int NUM_EXPERTS = 128, int TOP_K = 8,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_router_topk_softmax(
    int num_active_tokens,
    void *base,
    const MInst *st_insts,
    M2C_Type &m2c,
    C2M_Type &c2m) {
  __activate_compute_group(128);
  int tid = __compute_tid();

  int logits_slot = m2c.template pop<0>();
  auto *logits = (data_t *)get_slot_address(base, extract(logits_slot));
  int weight_slot = m2c.template pop<0>();
  auto *weights = (data_t *)get_slot_address(base, extract(weight_slot));
  int idx_slot = m2c.template pop<0>();
  auto *indices = idx_slot < numSlots
      ? (int *)get_slot_address(base, extract(idx_slot))
      : (int *)slot_2_glob_ptr(st_insts, idx_slot);

  __shared__ float probs[NUM_EXPERTS];
  __shared__ float warp_tmp[NUM_EXPERTS / 32];

  if (tid == 0) {
    for (int i = 0; i < TOP_K * 32; ++i) {
      weights[i] = data_t(0);
    }
  }
  __sync_compute_group(128);

  for (int token = 0; token < num_active_tokens; ++token) {
    float val = tid < NUM_EXPERTS ? (float)logits[token * NUM_EXPERTS + tid] : -FLT_MAX;
    float wmax = warp_reduce_max(val);
    if ((tid & 31) == 0) {
      warp_tmp[tid >> 5] = wmax;
    }
    __sync_compute_group(128);

    if (tid < 32) {
      float block_max = tid < (NUM_EXPERTS / 32) ? warp_tmp[tid] : -FLT_MAX;
      block_max = warp_reduce_max(block_max);
      if (tid == 0) {
        warp_tmp[0] = block_max;
      }
    }
    __sync_compute_group(128);
    float max_val = warp_tmp[0];

    float prob = tid < NUM_EXPERTS ? expf(val - max_val) : 0.0f;
    if (tid < NUM_EXPERTS) {
      probs[tid] = prob;
    }
    float wsum = warp_reduce_sum(prob);
    if ((tid & 31) == 0) {
      warp_tmp[tid >> 5] = wsum;
    }
    __sync_compute_group(128);

    if (tid < 32) {
      float block_sum = tid < (NUM_EXPERTS / 32) ? warp_tmp[tid] : 0.0f;
      block_sum = warp_reduce_sum(block_sum);
      if (tid == 0) {
        warp_tmp[0] = block_sum;
      }
    }
    __sync_compute_group(128);
    float sum_val = warp_tmp[0];

    if (tid == 0) {
      float top_probs[TOP_K];
      for (int expert = 0; expert < NUM_EXPERTS; ++expert) {
        probs[expert] = probs[expert] / sum_val;
      }

      #pragma unroll
      for (int i = 0; i < TOP_K; ++i) {
        float best_prob = -1.0f;
        int best_idx = 0;
        for (int expert = 0; expert < NUM_EXPERTS; ++expert) {
          float normalized = probs[expert];
          if (normalized > best_prob) {
            best_prob = normalized;
            best_idx = expert;
          }
        }
        top_probs[i] = best_prob;
        if (token == 0) {
          indices[i] = best_idx;
        }
        probs[best_idx] = -1.0f;
      }

      float top_sum = 0.0f;
      #pragma unroll
      for (int i = 0; i < TOP_K; ++i) {
        top_sum += top_probs[i];
      }
      float inv_top_sum = top_sum > 0.0f ? 1.0f / top_sum : 0.0f;
      #pragma unroll
      for (int i = 0; i < TOP_K; ++i) {
        weights[i * 32 + token] = (data_t)(top_probs[i] * inv_top_sum);
      }
    }
    __sync_compute_group(128);
  }

  c2m.template push<0, true>(tid, weight_slot);
  if (idx_slot < numSlots) {
    c2m.template push<0, true>(tid, idx_slot);
  }
  c2m.push(tid, logits_slot);
}

template <typename data_t, int K,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_scale_rows(
    int num_active_tokens,
    void *base,
    M2C_Type &m2c,
    C2M_Type &c2m) {
  __activate_compute_group(128);
  int tid = __compute_tid();

  int out_slot = m2c.template pop<0>();
  auto *out_ptr = (data_t *)get_slot_address(base, extract(out_slot));
  int in_slot = m2c.template pop<0>();
  auto *in_ptr = (data_t *)get_slot_address(base, extract(in_slot));
  int weight_slot = m2c.template pop<0>();
  auto *weight_ptr = (data_t *)get_slot_address(base, extract(weight_slot));

  for (int token = 0; token < num_active_tokens; ++token) {
    float scale = (float)weight_ptr[token];
    for (int i = tid; i < K; i += 128) {
      out_ptr[token * K + i] = (data_t)((float)in_ptr[token * K + i] * scale);
    }
  }

  c2m.template push<0, true>(tid, out_slot);
  c2m.push(tid, in_slot | weight_slot);
}
