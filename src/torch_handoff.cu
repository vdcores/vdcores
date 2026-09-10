#include <torch/extension.h>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <thread>

namespace py = pybind11;

static int32_t* check_cpu_counter(torch::Tensor counter) {
  TORCH_CHECK(counter.defined(), "counter must be defined");
  TORCH_CHECK(counter.device().is_cpu(), "counter must be a CPU tensor");
  TORCH_CHECK(counter.scalar_type() == torch::kInt32, "counter must have dtype torch.int32");
  TORCH_CHECK(counter.numel() == 1, "counter must contain exactly one element");
  TORCH_CHECK(counter.is_contiguous(), "counter must be contiguous");

  auto* ptr = counter.data_ptr<int32_t>();
  const auto address = reinterpret_cast<uintptr_t>(ptr);
  TORCH_CHECK(
      address % std::atomic_ref<int32_t>::required_alignment == 0,
      "counter is not aligned for std::atomic_ref<int32_t>");
  return ptr;
}

static int current_device() {
  int device = 0;
  const auto err = cudaGetDevice(&device);
  TORCH_CHECK(err == cudaSuccess, "cudaGetDevice failed: ", cudaGetErrorString(err));
  return device;
}

static int device_attribute(cudaDeviceAttr attribute) {
  int value = 0;
  const auto err = cudaDeviceGetAttribute(&value, attribute, current_device());
  TORCH_CHECK(err == cudaSuccess, "cudaDeviceGetAttribute failed: ", cudaGetErrorString(err));
  return value;
}

static py::dict handoff_capabilities() {
  cudaDeviceProp prop{};
  const auto err = cudaGetDeviceProperties(&prop, current_device());
  TORCH_CHECK(err == cudaSuccess, "cudaGetDeviceProperties failed: ", cudaGetErrorString(err));

  py::dict result;
  result["device_name"] = std::string(prop.name);
  result["pageable_memory_access"] =
      device_attribute(cudaDevAttrPageableMemoryAccess) != 0;
  result["uses_host_page_tables"] =
      device_attribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables) != 0;
  result["host_native_atomics"] =
      device_attribute(cudaDevAttrHostNativeAtomicSupported) != 0;
  result["concurrent_managed_access"] =
      device_attribute(cudaDevAttrConcurrentManagedAccess) != 0;
  return result;
}

static int32_t* gpu_accessible_counter_ptr(torch::Tensor counter) {
  auto* host_ptr = check_cpu_counter(counter);

  cudaPointerAttributes attributes{};
  auto err = cudaPointerGetAttributes(&attributes, host_ptr);
  if (err == cudaErrorInvalidValue) {
    cudaGetLastError();
    attributes.type = cudaMemoryTypeUnregistered;
  } else {
    TORCH_CHECK(err == cudaSuccess,
                "cudaPointerGetAttributes failed: ", cudaGetErrorString(err));
  }

  if (attributes.type == cudaMemoryTypeUnregistered) {
    TORCH_CHECK(
        device_attribute(cudaDevAttrPageableMemoryAccess) != 0,
        "this GPU cannot directly access an ordinary CPU tensor; use a GH200/GB200 "
        "or a Linux HMM configuration with pageable memory access");
    return host_ptr;
  }

  if (attributes.type == cudaMemoryTypeHost) {
    TORCH_CHECK(
        device_attribute(cudaDevAttrHostNativeAtomicSupported) != 0,
        "mapped host memory does not support native CPU/GPU read-modify-write atomics "
        "on this system");
    if (attributes.devicePointer != nullptr) {
      return static_cast<int32_t*>(attributes.devicePointer);
    }

    void* device_ptr = nullptr;
    err = cudaHostGetDevicePointer(&device_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
    return static_cast<int32_t*>(device_ptr);
  }

  if (attributes.type == cudaMemoryTypeManaged) {
    TORCH_CHECK(
        device_attribute(cudaDevAttrConcurrentManagedAccess) != 0,
        "managed counter does not support concurrent CPU/GPU access on this system");
    return host_ptr;
  }

  TORCH_CHECK(false, "counter must use system, mapped host, or managed memory");
  return nullptr;
}

__global__ void atomic_add_kernel(int32_t* counter, int32_t delta) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    cuda::atomic_ref<int32_t, cuda::thread_scope_system> atomic_counter(*counter);
    atomic_counter.fetch_add(delta, cuda::memory_order_acq_rel);
  }
}

__global__ void atomic_wait_add_kernel(
    int32_t* counter,
    int32_t target,
    int32_t delta,
    uint64_t timeout_cycles) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  cuda::atomic_ref<int32_t, cuda::thread_scope_system> atomic_counter(*counter);
  const uint64_t start = clock64();
  while (atomic_counter.load(cuda::memory_order_acquire) < target) {
    if (clock64() - start >= timeout_cycles) {
      return;
    }
    __nanosleep(64);
  }
  atomic_counter.fetch_add(delta, cuda::memory_order_acq_rel);
}

static int64_t cpu_atomic_load(torch::Tensor counter) {
  std::atomic_ref<int32_t> atomic_counter(*check_cpu_counter(counter));
  TORCH_CHECK(atomic_counter.is_lock_free(), "counter atomic is not lock-free on this CPU");
  return atomic_counter.load(std::memory_order_acquire);
}

static void cpu_atomic_store(torch::Tensor counter, int64_t value) {
  TORCH_CHECK(value >= std::numeric_limits<int32_t>::min() &&
                  value <= std::numeric_limits<int32_t>::max(),
              "value is outside the int32 range");
  std::atomic_ref<int32_t> atomic_counter(*check_cpu_counter(counter));
  TORCH_CHECK(atomic_counter.is_lock_free(), "counter atomic is not lock-free on this CPU");
  atomic_counter.store(static_cast<int32_t>(value), std::memory_order_release);
}

static int64_t cpu_atomic_add(torch::Tensor counter, int64_t delta) {
  TORCH_CHECK(delta >= std::numeric_limits<int32_t>::min() &&
                  delta <= std::numeric_limits<int32_t>::max(),
              "delta is outside the int32 range");
  std::atomic_ref<int32_t> atomic_counter(*check_cpu_counter(counter));
  TORCH_CHECK(atomic_counter.is_lock_free(), "counter atomic is not lock-free on this CPU");
  return atomic_counter.fetch_add(static_cast<int32_t>(delta), std::memory_order_acq_rel);
}

static int64_t cpu_atomic_wait(torch::Tensor counter, int64_t target, int64_t timeout_ms) {
  TORCH_CHECK(target >= std::numeric_limits<int32_t>::min() &&
                  target <= std::numeric_limits<int32_t>::max(),
              "target is outside the int32 range");
  TORCH_CHECK(timeout_ms > 0, "timeout_ms must be positive");

  std::atomic_ref<int32_t> atomic_counter(*check_cpu_counter(counter));
  TORCH_CHECK(atomic_counter.is_lock_free(), "counter atomic is not lock-free on this CPU");
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  const auto expected = static_cast<int32_t>(target);

  py::gil_scoped_release release;
  uint32_t spins = 0;
  while (std::chrono::steady_clock::now() < deadline) {
    const int32_t observed = atomic_counter.load(std::memory_order_acquire);
    if (observed >= expected) {
      return observed;
    }
    if ((++spins & 0x3ffU) == 0) {
      std::this_thread::yield();
    }
  }

  TORCH_CHECK(false, "timed out waiting for counter to reach ", target);
  return 0;
}

static void gpu_atomic_add(torch::Tensor counter, int64_t delta, int64_t stream_id) {
  TORCH_CHECK(delta >= std::numeric_limits<int32_t>::min() &&
                  delta <= std::numeric_limits<int32_t>::max(),
              "delta is outside the int32 range");
  auto* device_ptr = gpu_accessible_counter_ptr(counter);
  auto stream = reinterpret_cast<cudaStream_t>(stream_id);
  atomic_add_kernel<<<1, 1, 0, stream>>>(device_ptr, static_cast<int32_t>(delta));
  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "handoff atomic kernel launch failed: ", cudaGetErrorString(err));
}

static void gpu_atomic_wait_add(
    torch::Tensor counter,
    int64_t target,
    int64_t delta,
    int64_t timeout_ms,
    int64_t stream_id) {
  TORCH_CHECK(target >= std::numeric_limits<int32_t>::min() &&
                  target <= std::numeric_limits<int32_t>::max(),
              "target is outside the int32 range");
  TORCH_CHECK(delta >= std::numeric_limits<int32_t>::min() &&
                  delta <= std::numeric_limits<int32_t>::max(),
              "delta is outside the int32 range");
  TORCH_CHECK(timeout_ms > 0, "timeout_ms must be positive");

  auto* device_ptr = gpu_accessible_counter_ptr(counter);
  const uint64_t timeout_cycles =
      static_cast<uint64_t>(device_attribute(cudaDevAttrClockRate)) *
      static_cast<uint64_t>(timeout_ms);
  auto stream = reinterpret_cast<cudaStream_t>(stream_id);
  atomic_wait_add_kernel<<<1, 1, 0, stream>>>(
      device_ptr,
      static_cast<int32_t>(target),
      static_cast<int32_t>(delta),
      timeout_cycles);
  const auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess,
              "handoff wait kernel launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("handoff_capabilities", &handoff_capabilities,
        "Report CUDA capabilities relevant to CPU/GPU atomic handoff");
  m.def("cpu_atomic_load", &cpu_atomic_load, py::arg("counter"),
        "Load a one-element CPU int32 tensor with acquire ordering");
  m.def("cpu_atomic_store", &cpu_atomic_store,
        py::arg("counter"), py::arg("value"),
        "Store a one-element CPU int32 tensor with release ordering");
  m.def("cpu_atomic_add", &cpu_atomic_add,
        py::arg("counter"), py::arg("delta") = 1,
        "Atomically add to a CPU int32 tensor and return its prior value");
  m.def("cpu_atomic_wait", &cpu_atomic_wait,
        py::arg("counter"), py::arg("target"), py::arg("timeout_ms") = 5000,
        "Wait until a CPU int32 tensor reaches a target value");
  m.def("gpu_atomic_add", &gpu_atomic_add,
        py::arg("counter"), py::arg("delta") = 1, py::arg("stream") = 0,
        "Asynchronously add to a CPU counter from the GPU at system scope");
  m.def("gpu_atomic_wait_add", &gpu_atomic_wait_add,
        py::arg("counter"), py::arg("target"), py::arg("delta") = 1,
        py::arg("timeout_ms") = 5000, py::arg("stream") = 0,
        "Wait on a CPU counter from the GPU, then atomically add to it");
}
