#include <torch/extension.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
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
}
