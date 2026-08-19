#pragma once

#include "context.cuh"
#include "type.cuh"

#include <type_traits>

template<int HIDDIM_SIZE, int N_COMPUTE_THREAD, typename T>
__device__ __forceinline__ void _rms_helper_one_row(
    const T* weights,
    const T* input,
    T* output,
    float* smem_reduce,
    const T epsilon
) {
  int thread_id = __compute_tid();
  int lane_id = thread_id % 32;

  using Tr = F16Traits<T>;
  using vec2_t = typename Tr::vec2_t;
  const vec2_t* input2  = reinterpret_cast<const vec2_t*>(input);
  const vec2_t* weights2 = reinterpret_cast<const vec2_t*>(weights);
  vec2_t* output2 = reinterpret_cast<vec2_t*>(output);

  vec2_t sum2_0 = make_bfloat162(0, 0);
  vec2_t sum2_1 = make_bfloat162(0, 0);
  constexpr bool kCacheInput =
      HIDDIM_SIZE == 4096 &&
      (N_COMPUTE_THREAD == 64 || N_COMPUTE_THREAD == 128) &&
      std::is_same_v<T, __nv_bfloat16>;
  constexpr bool kCacheWeight =
      HIDDIM_SIZE == 4096 &&
      (N_COMPUTE_THREAD == 64 || N_COMPUTE_THREAD == 128) &&
      std::is_same_v<T, __nv_bfloat16>;
  struct alignas(16) Pack128 {
    vec2_t values[4];
  };
  constexpr int kPacksPerThread =
      kCacheInput ? HIDDIM_SIZE / 8 / N_COMPUTE_THREAD : 1;
  Pack128 input_cache[kPacksPerThread];
  Pack128 weight_cache[kCacheWeight ? kPacksPerThread : 1];
  if constexpr (kCacheInput) {
    const Pack128* input_pack = reinterpret_cast<const Pack128*>(input);
    #pragma unroll
    for (int j = 0; j < kPacksPerThread; ++j) {
      const Pack128 pack = input_pack[thread_id + j * N_COMPUTE_THREAD];
      input_cache[j] = pack;
      sum2_0 = __hfma2(pack.values[0], pack.values[0], sum2_0);
      sum2_1 = __hfma2(pack.values[1], pack.values[1], sum2_1);
      sum2_0 = __hfma2(pack.values[2], pack.values[2], sum2_0);
      sum2_1 = __hfma2(pack.values[3], pack.values[3], sum2_1);
    }
  } else {
    #pragma unroll
    for (int i = thread_id; i < HIDDIM_SIZE / 2; i += N_COMPUTE_THREAD) {
      vec2_t val = input2[i];
      sum2_0 = __hfma2(val, val, sum2_0);
    }
  }

  float sum = __bfloat162float(sum2_0.x) + __bfloat162float(sum2_0.y);
  if constexpr (kCacheInput) {
    sum += __bfloat162float(sum2_1.x) + __bfloat162float(sum2_1.y);
  }

  // reduce within warp
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
  }

  if (lane_id == 0) 
    smem_reduce[thread_id / 32] = sum;
  if constexpr (N_COMPUTE_THREAD == 64) {
    __sync_barrier<9, 64>();
  } else {
    __sync_compute_group(N_COMPUTE_THREAD);
  }

  float rms_rcp;
  if constexpr (kCacheWeight) {
    if (lane_id == 0) {
      sum = 0.0f;
      #pragma unroll
      for (int i = 0; i < N_COMPUTE_THREAD / 32; ++i) {
        sum += smem_reduce[i];
      }
    }
    const Pack128* weights_pack = reinterpret_cast<const Pack128*>(weights);
    #pragma unroll
    for (int j = 0; j < kPacksPerThread; ++j) {
      weight_cache[j] = weights_pack[thread_id + j * N_COMPUTE_THREAD];
    }
    if (lane_id == 0) {
      rms_rcp = rsqrtf(
          sum / float(HIDDIM_SIZE) + Tr::to_float(epsilon));
    }
    rms_rcp = __shfl_sync(0xFFFFFFFFU, rms_rcp, 0);
  } else {
    // Generic widths publish one final shared result before normalization.
    if (thread_id == 0) {
      #pragma unroll
      for (int i = 1; i < N_COMPUTE_THREAD / 32; i++)
        sum += smem_reduce[i];
      smem_reduce[0] = sum;
    }
    __sync_compute_group(N_COMPUTE_THREAD);
    sum = smem_reduce[0];
    rms_rcp = rsqrtf(
        sum / float(HIDDIM_SIZE) + Tr::to_float(epsilon));
  }

  // final scale
  vec2_t scale2 = make_bfloat162(rms_rcp, rms_rcp);
  if constexpr (kCacheInput) {
    const Pack128* weights_pack = reinterpret_cast<const Pack128*>(weights);
    Pack128* output_pack = reinterpret_cast<Pack128*>(output);
    #pragma unroll
    for (int j = 0; j < kPacksPerThread; ++j) {
      const int i = thread_id + j * N_COMPUTE_THREAD;
      const Pack128 weight =
          kCacheWeight ? weight_cache[j] : weights_pack[i];
      Pack128 out;
      #pragma unroll
      for (int k = 0; k < 4; ++k) {
        const vec2_t o = __hmul2(input_cache[j].values[k], scale2);
        out.values[k] = __hmul2(o, weight.values[k]);
      }
      output_pack[i] = out;
    }
  } else {
    #pragma unroll
    for (int i = thread_id; i < HIDDIM_SIZE / 2; i += N_COMPUTE_THREAD) {
      vec2_t o = __hmul2(input2[i], scale2);
      output2[i] = __hmul2(o, weights2[i]);
    }
  }
}

// The non-4096 resident RMS shapes have identical work and synchronization;
// only the number of BF16 pairs differs. Keep 4096's register-cached fast
// path templated and use this one compact runtime loop for the other widths.
__device__ __forceinline__ void _rms_helper_one_row_runtime_bf16(
    const __nv_bfloat16* weights,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    float* smem_reduce,
    int hidden_size,
    __nv_bfloat16 epsilon
) {
  constexpr int kThreads = 128;
  const int thread_id = __compute_tid();
  const int lane_id = thread_id % 32;
  const int pair_count = hidden_size / 2;
  const auto* input2 = reinterpret_cast<const __nv_bfloat162*>(input);
  const auto* weights2 = reinterpret_cast<const __nv_bfloat162*>(weights);
  auto* output2 = reinterpret_cast<__nv_bfloat162*>(output);

  __nv_bfloat162 sum2 = make_bfloat162(0, 0);
  for (int i = thread_id; i < pair_count; i += kThreads) {
    const __nv_bfloat162 value = input2[i];
    sum2 = __hfma2(value, value, sum2);
  }
  float sum = __bfloat162float(sum2.x) + __bfloat162float(sum2.y);
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane_id == 0) {
    smem_reduce[thread_id / 32] = sum;
  }
  __sync_compute_group(kThreads);

  if (thread_id == 0) {
    #pragma unroll
    for (int warp = 1; warp < kThreads / 32; ++warp) {
      sum += smem_reduce[warp];
    }
    smem_reduce[0] = sum;
  }
  __sync_compute_group(kThreads);

  const float rms_rcp = rsqrtf(
      smem_reduce[0] / float(hidden_size) + __bfloat162float(epsilon));
  const __nv_bfloat162 scale2 = make_bfloat162(rms_rcp, rms_rcp);
  for (int i = thread_id; i < pair_count; i += kThreads) {
    output2[i] = __hmul2(__hmul2(input2[i], scale2), weights2[i]);
  }
}

__device__ __forceinline__ void _rms_helper_two_rows_4096_bf16(
    const __nv_bfloat16* weights,
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    float* smem_reduce,
    const __nv_bfloat16 epsilon
) {
  constexpr int kThreadsPerRow = 64;
  constexpr int kElementsPerPack = 8;
  constexpr int kPacksPerRow = 4096 / kElementsPerPack;
  constexpr int kPacksPerThread = kPacksPerRow / kThreadsPerRow;
  using vec2_t = __nv_bfloat162;
  struct alignas(16) Pack128 {
    vec2_t values[4];
  };

  const int thread_id = __compute_tid();
  const int row = thread_id / kThreadsPerRow;
  const int row_thread = thread_id % kThreadsPerRow;
  const int lane_id = thread_id % 32;
  const Pack128* input_pack = reinterpret_cast<const Pack128*>(input);
  const Pack128* weights_pack = reinterpret_cast<const Pack128*>(weights);
  Pack128* output_pack = reinterpret_cast<Pack128*>(output);
  Pack128 input_cache[kPacksPerThread];
  Pack128 weight_cache[kPacksPerThread];
  vec2_t sum2 = make_bfloat162(0, 0);

  #pragma unroll
  for (int j = 0; j < kPacksPerThread; ++j) {
    const int pack_id = row_thread + j * kThreadsPerRow;
    const Pack128 pack = input_pack[row * kPacksPerRow + pack_id];
    input_cache[j] = pack;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      sum2 = __hfma2(pack.values[k], pack.values[k], sum2);
    }
  }

  float sum = __bfloat162float(sum2.x) + __bfloat162float(sum2.y);
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
  }
  if (lane_id == 0) {
    smem_reduce[thread_id / 32] = sum;
  }
  __sync_compute_group(128);

  if (lane_id == 0) {
    sum = smem_reduce[row * 2] + smem_reduce[row * 2 + 1];
  }
  #pragma unroll
  for (int j = 0; j < kPacksPerThread; ++j) {
    weight_cache[j] = weights_pack[row_thread + j * kThreadsPerRow];
  }
  float rms_rcp;
  if (lane_id == 0) {
    rms_rcp = rsqrtf(sum / 4096.0f + __bfloat162float(epsilon));
  }
  rms_rcp = __shfl_sync(0xFFFFFFFFU, rms_rcp, 0);
  const vec2_t scale2 = make_bfloat162(rms_rcp, rms_rcp);
  #pragma unroll
  for (int j = 0; j < kPacksPerThread; ++j) {
    const int pack_id = row_thread + j * kThreadsPerRow;
    Pack128 out;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      const vec2_t scaled = __hmul2(input_cache[j].values[k], scale2);
      out.values[k] = __hmul2(scaled, weight_cache[j].values[k]);
    }
    output_pack[row * kPacksPerRow + pack_id] = out;
  }
}

template<int HIDDIM_SIZE, typename data_t,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_rms_norm_f16_from_glob(
    void *base,
    const MInst *st_insts,
    const int num_token,
    const data_t epsilon,
    float *smem_reduce,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
  // TODO(zijian): this assume K major input
  static_assert(HIDDIM_SIZE % 2 == 0, "HIDDIM_SIZE must be even for half2 load");
  constexpr int nThreads = 128;
  __activate_compute_group(nThreads);

  // base address should be the start of the first token
  const int weights_addr_slot = m2c.template pop<0>();
  data_t* base_weights_addr = (data_t*)slot_2_glob_ptr(st_insts, weights_addr_slot);
  const int raw_addr_slot = m2c.template pop<0>();
  data_t* base_input_ptr = (data_t*)slot_2_glob_ptr(st_insts, raw_addr_slot);
  const int out_addr_slot = m2c.template pop<0>();
  data_t* base_out_ptr = (data_t*)slot_2_glob_ptr(st_insts, out_addr_slot);

  #pragma unroll
  for (int token_id = 0; token_id < num_token; token_id++) {
    // offset input address to current token
    data_t* input_ptr = base_input_ptr + token_id * HIDDIM_SIZE;
    data_t* output_ptr = base_out_ptr + token_id * HIDDIM_SIZE;
    _rms_helper_one_row<HIDDIM_SIZE, nThreads>(base_weights_addr, input_ptr, output_ptr, smem_reduce, epsilon);
  }
  c2m.template push<31, true, false>(__compute_tid(), out_addr_slot);
}

template<int HIDDIM_SIZE, typename data_t,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_rms_norm_f16_from_smem(
    void *base,
    const int num_token,
    const data_t epsilon,
    float *smem_reduce,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
  // TODO(zijian): this assume K major input
  static_assert(HIDDIM_SIZE % 2 == 0, "HIDDIM_SIZE must be even for half2 load");

  constexpr int nThreads = 128;
  int thread_id = __compute_tid();

  // base address should be the start of the first token
  const int weights_slot = m2c.template pop<0>();
  data_t* weights_ptr = (data_t*)get_slot_address(base, extract(weights_slot));
  const int in_addr_slot = m2c.template pop<0>();
  data_t* base_input_ptr = (data_t*)get_slot_address(base, extract(in_addr_slot));
  const int out_addr_slot = m2c.template pop<0>();
  data_t* base_out_ptr = (data_t*)get_slot_address(base, extract(out_addr_slot));

  if constexpr (HIDDIM_SIZE == 4096 &&
                std::is_same_v<data_t, __nv_bfloat16>) {
    if (num_token == 2) {
      _rms_helper_two_rows_4096_bf16(
          weights_ptr, base_input_ptr, base_out_ptr, smem_reduce, epsilon);
    } else {
      #pragma unroll
      for (int token_id = 0; token_id < num_token; token_id++) {
        data_t* input_ptr = base_input_ptr + token_id * HIDDIM_SIZE;
        data_t* output_ptr = base_out_ptr + token_id * HIDDIM_SIZE;
        if (thread_id < nThreads) {
          _rms_helper_one_row<HIDDIM_SIZE, nThreads>(
              weights_ptr, input_ptr, output_ptr, smem_reduce, epsilon);
        }
      }
    }
  } else {
    #pragma unroll
    for (int token_id = 0; token_id < num_token; token_id++) {
      data_t* input_ptr = base_input_ptr + token_id * HIDDIM_SIZE;
      data_t* output_ptr = base_out_ptr + token_id * HIDDIM_SIZE;
      _rms_helper_one_row<HIDDIM_SIZE, nThreads>(
          weights_ptr, input_ptr, output_ptr, smem_reduce, epsilon);
    }
  }
  

  // TODO(zhiyuang): do we need sync here?
  // __sync_compute_group(nThreads);

  c2m.template push<0, true>(thread_id, out_addr_slot);
  c2m.push(thread_id, in_addr_slot | weights_slot);
}

template<typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_rms_norm_f16_from_smem_runtime(
    void *base,
    int num_token,
    int hidden_size,
    __nv_bfloat16 epsilon,
    float *smem_reduce,
    M2C_Type& m2c,
    C2M_Type& c2m
) {
  const int thread_id = __compute_tid();
  const int weights_slot = m2c.template pop<0>();
  const auto* weights_ptr = reinterpret_cast<const __nv_bfloat16*>(
      get_slot_address(base, extract(weights_slot)));
  const int input_slot = m2c.template pop<0>();
  const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(
      get_slot_address(base, extract(input_slot)));
  const int output_slot = m2c.template pop<0>();
  auto* output_ptr = reinterpret_cast<__nv_bfloat16*>(
      get_slot_address(base, extract(output_slot)));

  for (int token_id = 0; token_id < num_token; ++token_id) {
    _rms_helper_one_row_runtime_bf16(
        weights_ptr,
        input_ptr + token_id * hidden_size,
        output_ptr + token_id * hidden_size,
        smem_reduce,
        hidden_size,
        epsilon);
  }

  c2m.template push<0, true>(thread_id, output_slot);
  c2m.push(thread_id, input_slot | weights_slot);
}
