#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

static void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

// Both pointers name the same peer buffer. Separate kernel arguments ensure
// the compiler emits the coalesced loads and stores even for an in-place call.
__global__ void peer_roundtrip(uint4* dst, const uint4* src, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

static void roundtrip_kernel(uintptr_t dst, uintptr_t src, size_t bytes,
                             int blocks, uintptr_t stream) {
  if (bytes % sizeof(uint4) != 0 || blocks < 1)
    throw std::runtime_error("roundtrip requires 16-byte size alignment and positive blocks");
  peer_roundtrip<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<uint4*>(dst), reinterpret_cast<const uint4*>(src),
      bytes / sizeof(uint4));
  check(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("roundtrip_kernel", &roundtrip_kernel);
}
