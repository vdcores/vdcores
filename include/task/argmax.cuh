#pragma once

#include <cfloat>

#include "virtualcore.cuh"
#include "type.cuh"

template <typename T>
__device__ __forceinline__ void warp_reduce_max_idx(T &val, long long &idx) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    float tmp = __shfl_down_sync(0xffffffff, (float)val, offset);
    T other_val = (T)tmp;
    long long other_idx = __shfl_down_sync(0xffffffff, idx, offset);
    if (other_val > val) {
      val = other_val;
      idx = other_idx;
    }
  }
}

template <typename T>
__device__ __forceinline__ void block_reduce_max_idx(void *smem, T &val, long long &idx) {
  // Align the shared memory to 128 bytes
  constexpr int nThreads = 128;
  constexpr int nWarps = nThreads / numThreadsPerWarp;

  long long *smem_idxs = (long long *)smem;
  T* smem_vals = (T *)(smem_idxs + nWarps);

  warp_reduce_max_idx(val, idx);

  int my_lane_id = threadIdx.x % numThreadsPerWarp;
  int my_warp_id = threadIdx.x / numThreadsPerWarp;

  if (my_lane_id == 0) {
    smem_vals[my_warp_id] = val;
    smem_idxs[my_warp_id] = idx;
  }

  __sync_compute_group(nThreads);

  // Only thread 0 holds the final result
  if (threadIdx.x == 0) {
    T block_max_val = T(-FLT_MAX);
    long long block_max_idx = -1;

    int num_warps = 4;
    for (int i = 0; i < num_warps; i++) {
      T warp_val = smem_vals[i];
      long long warp_idx = smem_idxs[i];
      if (warp_val > block_max_val) {
        block_max_val = warp_val;
        block_max_idx = warp_idx;
      }
    }

    val = block_max_val;
    idx = block_max_idx;
  }
}

// assume half, vectorize it
template <int CHUNK_SIZE, int I_STRIDE, int O_STRIDE, typename data_t>
__device__ __forceinline__ void _argmax_partial_helper(
  int num_active_tokens,
  void *stratchpad,
  const typename F16Traits<data_t>::vec2_t *__restrict__ input,
  data_t *__restrict__ output_val,
  long long *__restrict__ output_idx
) {
  constexpr int nThreads = 128;
  int tid = threadIdx.x;

  #pragma unroll
  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    data_t local_max = (data_t)(-FLT_MAX);
    long long local_idx = -1;
    #pragma unroll
    for (int i = tid; i < CHUNK_SIZE / 2; i += nThreads) {
      auto val = input[i + batch_idx * I_STRIDE / 2];
      if (val.x > local_max) {
        local_max = val.x;
        local_idx = i * 2;
      }
      if (val.y > local_max) {
        local_max = val.y;
        local_idx = i * 2 + 1;
      }
    }
    block_reduce_max_idx(stratchpad, local_max, local_idx);

    if (tid == 0) {
      output_val[batch_idx * O_STRIDE] = local_max;
      output_idx[batch_idx * O_STRIDE] = local_idx;
    }
  }
}


// TODO(zijian): reuse helper if go smem
template <int CHUNK_SIZE, int I_STRIDE, int O_TASKS, typename data_t,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_argmax_partial(int num_active_tokens,
    void *smem_base, const MInst *st_insts, void *stratchpad, M2C_Type &m2c, C2M_Type &c2m) {
  using Tr = F16Traits<data_t>;
  using vec2_t = typename Tr::vec2_t;

  int tid = threadIdx.x;
  auto input_slot = m2c.template pop<0>();
  vec2_t const *input = (vec2_t const *)slot_2_glob_ptr(st_insts, input_slot);
  auto output_val_slot = m2c.template pop<0>();
  data_t *output_val = (data_t *)slot_2_glob_ptr(st_insts, output_val_slot);
  auto output_idx_slot = m2c.template pop<0>();
  long long *output_idx = (long long *)slot_2_glob_ptr(st_insts, output_idx_slot);

  _argmax_partial_helper<CHUNK_SIZE, I_STRIDE, O_TASKS>(
    num_active_tokens, stratchpad, input, output_val, output_idx
  );

  c2m.template push<31, true, false>(tid, 1 << output_val_slot);
  // TODO(zhiyuang): check if need to write
  c2m.template push<31, true, false>(tid, 1 << output_idx_slot);
}

template <int CHUNK_SIZE, int NUM_PARTIAL_TASKS, typename data_t,
          typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_argmax_reduce_kernel(int num_active_tokens, 
    void *smem_base, const MInst *st_insts, void *stratchpad, M2C_Type &m2c, C2M_Type &c2m) {
  using Tr = F16Traits<data_t>;
  using vec2_t = typename Tr::vec2_t;

  // TODO(zijian): is vectorize really helpful here ?
  auto output_val_slot = m2c.template pop<0>();
  data_t const *__restrict__ output_val = (data_t const *)slot_2_glob_ptr(st_insts, output_val_slot);
  auto output_idx_slot = m2c.template pop<0>();
  long long const *__restrict__ output_idx = (long long const *)slot_2_glob_ptr(st_insts, output_idx_slot);

  auto output_slot = m2c.template pop<0>();
  long long *__restrict__ output_final = (long long *)slot_2_glob_ptr(st_insts, output_slot);
                        
  int tid = threadIdx.x;
  constexpr int nThreads = 128;

  #pragma unroll
  for (int batch_idx = 0; batch_idx < num_active_tokens; batch_idx++) {
    data_t local_max = data_t(-FLT_MAX);
    // Pack (chunk_index, relative_index) into a single 64-bit integer
    long long local_packed_idx = -1;

    #pragma unroll
    for (int i = tid; i < NUM_PARTIAL_TASKS; i += nThreads) {
      data_t current_val = output_val[i + batch_idx * NUM_PARTIAL_TASKS];
      if (current_val > local_max) {
        local_max = current_val;
        // Higher 32 bits for chunk_index (i), lower 32 for relative_index
        local_packed_idx = ((long long)i << 32) |
                           output_idx[i + batch_idx * NUM_PARTIAL_TASKS];
      }
    }

    block_reduce_max_idx(stratchpad, local_max, local_packed_idx);

    if (tid == 0) {
      if (local_packed_idx != -1) {
        long long winning_chunk_idx = local_packed_idx >> 32;
        long long winning_relative_idx = local_packed_idx & 0xFFFFFFFF;
        output_final[batch_idx] =
            winning_chunk_idx * CHUNK_SIZE + winning_relative_idx;
      } else {
        output_final[batch_idx] = -1;
      }
    }
  }

  c2m.template push<31, true, false>(tid, 1 << output_slot);
}
