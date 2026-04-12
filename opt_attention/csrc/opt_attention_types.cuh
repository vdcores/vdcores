#pragma once

#include "opt_attention_params.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace opt_attention {

template <typename T>
struct ScalarIO;

template <>
struct ScalarIO<half> {
  __device__ __forceinline__ static float load(const half* ptr) {
    return __half2float(*ptr);
  }

  __device__ __forceinline__ static half store(float value) {
    return __float2half_rn(value);
  }

  __device__ __forceinline__ static half zero() {
    return __float2half_rn(0.0f);
  }
};

template <>
struct ScalarIO<__nv_bfloat16> {
  __device__ __forceinline__ static float load(const __nv_bfloat16* ptr) {
    return __bfloat162float(*ptr);
  }

  __device__ __forceinline__ static __nv_bfloat16 store(float value) {
    return __float2bfloat16(value);
  }

  __device__ __forceinline__ static __nv_bfloat16 zero() {
    return __float2bfloat16(0.0f);
  }
};

__device__ __forceinline__ void compute_group_sync() {
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

__device__ __forceinline__ void cp_async_16B(void* shared_dst, const void* global_src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned shared_addr = static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(shared_addr), "l"(global_src) : "memory");
#else
  *reinterpret_cast<uint4*>(shared_dst) = *reinterpret_cast<const uint4*>(global_src);
#endif
}

}  // namespace opt_attention
