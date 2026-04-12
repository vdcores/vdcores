#pragma once

#include <cuda.h>

#include <cstdint>

namespace opt_attention {

#ifndef OPT_ATTENTION_USE_TMA
#define OPT_ATTENTION_USE_TMA 1
#endif

constexpr int kHeadDim = 128;
constexpr int kKvTile = 64;
constexpr int kComputeThreads = 128;
constexpr int kProducerThreads = 32;
constexpr int kThreads = kComputeThreads + kProducerThreads;
constexpr bool kUseTensorTma = OPT_ATTENTION_USE_TMA != 0;

struct OptAttentionParams {
  const void* query;
  const void* key;
  const void* value;
  const float* mask;
  void* output;
  float* partial_out;
  float* partial_m;
  float* partial_l;
  const CUtensorMap* tma_descs;

  int batch_size;
  int num_heads;
  int key_seq_len;
  int num_splits;
  int split_size;
  float scaling;

  int64_t q_stride_b;
  int64_t q_stride_h;
  int64_t q_stride_q;
  int64_t q_stride_d;

  int64_t k_stride_b;
  int64_t k_stride_h;
  int64_t k_stride_s;
  int64_t k_stride_d;

  int64_t v_stride_b;
  int64_t v_stride_h;
  int64_t v_stride_s;
  int64_t v_stride_d;

  int64_t o_stride_b;
  int64_t o_stride_q;
  int64_t o_stride_h;
  int64_t o_stride_d;

  int64_t m_stride_b;
  int64_t m_stride_s;
};

}  // namespace opt_attention
