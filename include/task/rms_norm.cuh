#pragma once

#include "context.cuh"
#include "type.cuh"

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

  vec2_t sum2 = make_bfloat162(0, 0);
  #pragma unroll
  for (int i = thread_id; i < HIDDIM_SIZE / 2; i += N_COMPUTE_THREAD) {
    vec2_t val = input2[i];
    sum2 = __hfma2(val, val, sum2);
  }

  float sum = __bfloat162float(sum2.x) + __bfloat162float(sum2.y);

  // reduce within warp
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    sum += __shfl_xor_sync(0xFFFFFFFFU, sum, offset);
  }

  if (lane_id == 0) 
    smem_reduce[thread_id / 32] = sum;
  __sync_compute_group(N_COMPUTE_THREAD);

  // final reduce by first warp
  if (thread_id == 0) {
    #pragma unroll
    for (int i = 1; i < N_COMPUTE_THREAD / 32; i++)
      sum += smem_reduce[i];
    smem_reduce[0] = sum;
  }
  __sync_compute_group(N_COMPUTE_THREAD);

  float rms_rcp = rsqrtf(smem_reduce[0] / float(HIDDIM_SIZE) + Tr::to_float(epsilon));

  // final scale
  vec2_t scale2 = make_bfloat162(rms_rcp, rms_rcp);
  #pragma unroll
  for (int i = thread_id; i < HIDDIM_SIZE / 2; i += N_COMPUTE_THREAD) {
    vec2_t o = __hmul2(input2[i], scale2);
    output2[i] = __hmul2(o, weights2[i]);
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

  #pragma unroll
  for (int token_id = 0; token_id < num_token; token_id++) {
    // offset input address to current token
    data_t* input_ptr = base_input_ptr + token_id * HIDDIM_SIZE;
    data_t* output_ptr = base_out_ptr + token_id * HIDDIM_SIZE;
    _rms_helper_one_row<HIDDIM_SIZE, nThreads>(weights_ptr, input_ptr, output_ptr, smem_reduce, epsilon);
  }
  

  // TODO(zhiyuang): do we need sync here?
  // __sync_compute_group(nThreads);

  c2m.template push<0, true>(thread_id, out_addr_slot);
  c2m.push(thread_id, in_addr_slot | weights_slot);
}

