#ifndef DAE_ENABLE_NVSHMEM
#error "torch_nvshmem_runtime.cu requires DAE_ENABLE_NVSHMEM"
#endif

#include <torch/extension.h>

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <cuda_runtime_api.h>

#define NVSHMEMI_HOST_ONLY
#include <nvshmem_host.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

namespace py = pybind11;

namespace {

struct SymmetricAllocation {
  void* ptr = nullptr;
  size_t bytes = 0;
};

std::mutex allocation_mutex;
std::vector<SymmetricAllocation> allocations;
int allocation_device = -1;

void require_initialized() {
  const int status = nvshmemx_init_status();
  TORCH_CHECK(
      status >= NVSHMEM_STATUS_IS_INITIALIZED &&
          status <= NVSHMEM_STATUS_FULL_MPG,
      "NVSHMEM is not initialized; call dae.nvshmem.init() first");
}

int current_device() {
  int device = -1;
  const cudaError_t status = cudaGetDevice(&device);
  TORCH_CHECK(
      status == cudaSuccess,
      "cudaGetDevice failed: ",
      cudaGetErrorString(status));
  return device;
}

size_t tensor_bytes(
    const std::vector<int64_t>& shape,
    at::ScalarType dtype) {
  TORCH_CHECK(dtype != at::ScalarType::Undefined, "dtype must be defined");

  size_t elements = 1;
  for (const int64_t dimension : shape) {
    TORCH_CHECK(dimension >= 0, "tensor dimensions must be non-negative");
    if (dimension == 0) {
      elements = 0;
      continue;
    }
    TORCH_CHECK(
        elements <= std::numeric_limits<size_t>::max() /
            static_cast<size_t>(dimension),
        "tensor element count overflows size_t");
    elements *= static_cast<size_t>(dimension);
  }

  const size_t element_size = c10::elementSize(dtype);
  TORCH_CHECK(
      elements <= std::numeric_limits<size_t>::max() / element_size,
      "tensor byte size overflows size_t");
  return elements * element_size;
}

torch::Tensor allocate_tensor_unlocked(
    const std::vector<int64_t>& shape,
    at::ScalarType dtype,
    bool zeroed) {
  require_initialized();

  const int device = current_device();
  if (allocation_device < 0) {
    allocation_device = device;
  }
  TORCH_CHECK(
      device == allocation_device,
      "All DAE NVSHMEM allocations must use CUDA device ",
      allocation_device,
      ", but device ",
      device,
      " is current");

  const size_t logical_bytes = tensor_bytes(shape, dtype);
  const size_t allocation_bytes = std::max<size_t>(logical_bytes, 1);
  void* pointer = zeroed
      ? nvshmem_calloc(1, allocation_bytes)
      : nvshmem_malloc(allocation_bytes);
  TORCH_CHECK(
      pointer != nullptr,
      "NVSHMEM symmetric allocation failed for ",
      allocation_bytes,
      " bytes; increase NVSHMEM_SYMMETRIC_SIZE and keep allocation order "
      "identical across PEs");

  allocations.push_back({pointer, allocation_bytes});
  const auto options = torch::TensorOptions()
                           .dtype(dtype)
                           .device(torch::Device(torch::kCUDA, device));
  return at::from_blob(pointer, shape, [](void*) {}, options);
}

torch::Tensor allocate_tensor(
    const std::vector<int64_t>& shape,
    at::ScalarType dtype,
    bool zeroed) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  return allocate_tensor_unlocked(shape, dtype, zeroed);
}

bool is_symmetric_tensor(const torch::Tensor& tensor) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  if (!tensor.defined() || !tensor.is_cuda()) {
    return false;
  }

  const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  for (const SymmetricAllocation& allocation : allocations) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(allocation.ptr);
    const uintptr_t end = begin + allocation.bytes;
    if (address >= begin && address < end) {
      return true;
    }
  }
  return false;
}

void release_allocations() {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  if (allocations.empty()) {
    return;
  }
  require_initialized();

  for (auto allocation = allocations.rbegin();
       allocation != allocations.rend();
       ++allocation) {
    nvshmem_free(allocation->ptr);
  }
  allocations.clear();
  allocation_device = -1;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "DAE symmetric tensor allocation for NVSHMEM4Py";
  m.attr("NVSHMEM_ENABLED") = true;
  m.def(
      "allocate_tensor",
      &allocate_tensor,
      py::arg("shape"),
      py::arg("dtype"),
      py::arg("zeroed") = false);
  m.def("is_symmetric_tensor", &is_symmetric_tensor, py::arg("tensor"));
  m.def("release_allocations", &release_allocations);
}
