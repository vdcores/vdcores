#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>


__global__ void manual_reduction_kernel(const __nv_bfloat16* partial0, const __nv_bfloat16* partial1, __nv_bfloat16* output, int numel) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        float a = __bfloat162float(partial0[idx]);
        float b = __bfloat162float(partial1[idx]);
        output[idx] = __float2bfloat16(a + b);
    }
}


void manual_reduction(
    torch::Tensor partial0,
    torch::Tensor partial1,
    torch::Tensor output
) {
    TORCH_CHECK(partial0.is_cuda(), "partial0 must be a CUDA tensor");
    TORCH_CHECK(partial1.is_cuda(), "partial1 must be a CUDA tensor");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");

    TORCH_CHECK(partial0.dtype() == torch::kBFloat16, "partial0 must be BF16");
    TORCH_CHECK(partial1.dtype() == torch::kBFloat16, "partial1 must be BF16");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be BF16");

    TORCH_CHECK(partial0.numel() == partial1.numel(), "partials must have same numel");
    TORCH_CHECK(partial0.numel() == output.numel(), "output must have same numel");

    TORCH_CHECK(partial0.is_contiguous(), "partial0 must be contiguous");
    TORCH_CHECK(partial1.is_contiguous(), "partial1 must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    int numel = partial0.numel();

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    manual_reduction_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(partial0.data_ptr<at::BFloat16>()), reinterpret_cast<const __nv_bfloat16*>(partial1.data_ptr<at::BFloat16>()), reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()), numel);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("manual_reduction", &manual_reduction, "Manual BF16 partial-output reduction");
}