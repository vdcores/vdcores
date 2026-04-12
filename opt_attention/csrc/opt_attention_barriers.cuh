#pragma once

#include "opt_attention_types.cuh"

#include <cuda/barrier>

namespace opt_attention {

struct PipelineBarriers {
  cuda::barrier<cuda::thread_scope_block> kv_fill[2];
  cuda::barrier<cuda::thread_scope_block> kv_drain[2];
};

__device__ __forceinline__ void init_pipeline_barriers(PipelineBarriers& barriers) {
  if (threadIdx.x == 0) {
    init(&barriers.kv_fill[0], kComputeThreads + kProducerThreads);
    init(&barriers.kv_fill[1], kComputeThreads + kProducerThreads);
    init(&barriers.kv_drain[0], kComputeThreads + kProducerThreads);
    init(&barriers.kv_drain[1], kComputeThreads + kProducerThreads);
  }
  __syncthreads();
}

__device__ __forceinline__ void producer_arrive(cuda::barrier<cuda::thread_scope_block>& barrier) {
  __threadfence_block();
  static_cast<void>(barrier.arrive());
}

__device__ __forceinline__ void producer_async_arrive(cuda::barrier<cuda::thread_scope_block>& barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned barrier_addr = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
  asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];" ::"r"(barrier_addr) : "memory");
#endif
}

__device__ __forceinline__ void compute_arrive(cuda::barrier<cuda::thread_scope_block>& barrier) {
  __threadfence_block();
  static_cast<void>(barrier.arrive());
}

}  // namespace opt_attention
