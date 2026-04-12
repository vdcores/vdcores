#include "opt_attention_kernel.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace opt_attention {

template <typename scalar_t>
static void launch_typed(const OptAttentionParams& params) {
  const dim3 grid(params.batch_size * params.num_heads * params.num_splits);
  const dim3 block(kThreads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  opt_attention_decode_kernel<scalar_t><<<grid, block, 0, stream>>>(params);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  if (params.num_splits > 1) {
    const dim3 reduce_grid(params.batch_size * params.num_heads);
    const dim3 reduce_block(kComputeThreads);
    opt_attention_reduce_splits_kernel<scalar_t><<<reduce_grid, reduce_block, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void launch_decode(const OptAttentionParams& params, at::ScalarType dtype) {
  if (dtype == at::kHalf) {
    launch_typed<half>(params);
    return;
  }
  if (dtype == at::kBFloat16) {
    launch_typed<__nv_bfloat16>(params);
    return;
  }
  TORCH_CHECK(false, "opt_attention only supports float16 and bfloat16");
}

}  // namespace opt_attention
