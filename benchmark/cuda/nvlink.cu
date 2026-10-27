#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>

static void check(cudaError_t error) {
  if (error != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(error));
}

// A conventional coalesced SM copy, with a peer source and local destination.
__global__ void peer_copy(uint4* dst, const uint4* src, size_t count) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += static_cast<size_t>(gridDim.x) * blockDim.x)
    dst[i] = src[i];
}

static void copy_kernel(uintptr_t dst, uintptr_t src, size_t bytes,
                        int blocks, uintptr_t stream) {
  if (bytes % sizeof(uint4) != 0 || blocks < 1)
    throw std::runtime_error("copy requires 16-byte size alignment and positive blocks");
  peer_copy<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      reinterpret_cast<uint4*>(dst), reinterpret_cast<const uint4*>(src),
      bytes / sizeof(uint4));
  check(cudaGetLastError());
}

static void copy_engine(uintptr_t dst, uintptr_t src, size_t bytes, uintptr_t stream) {
  // Peer access is already enabled. Unlike cudaMemcpyPeerAsync, this UVA
  // device-to-device form supports stream capture on the tested CUDA toolkit.
  check(cudaMemcpyAsync(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src),
                       bytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("copy_kernel", &copy_kernel);
  module.def("copy_engine", &copy_engine);
}
