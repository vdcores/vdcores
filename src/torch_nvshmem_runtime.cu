#include <torch/extension.h>

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>

#include <cuda_runtime_api.h>
#include <mpi.h>

#define NVSHMEMI_HOST_ONLY
#include <nvshmem_host.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct SymmetricAllocation {
  void* ptr = nullptr;
  size_t bytes = 0;
};

struct RuntimeState {
  bool initialized = false;
  bool owns_mpi = false;
  bool owns_nvshmem = false;
  int mpi_thread_level = MPI_THREAD_SINGLE;
  int rank = -1;
  int world_size = 0;
  int local_rank = -1;
  int local_size = 0;
  int device = -1;
  int pe = -1;
  int num_pes = 0;
  MPI_Comm local_comm = MPI_COMM_NULL;
  std::vector<SymmetricAllocation> allocations;
  torch::Tensor signal_space;
  int64_t signal_count = 0;
};

RuntimeState state;
std::mutex state_mutex;

std::string mpi_error_string(int error) {
  char message[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(error, message, &length);
  return std::string(message, static_cast<size_t>(std::max(length, 0)));
}

void check_mpi(int status, const char* operation) {
  TORCH_CHECK(
      status == MPI_SUCCESS,
      operation,
      " failed: ",
      mpi_error_string(status));
}

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorString(status));
}

void set_environment(const char* name, const std::string& value, bool overwrite) {
  TORCH_CHECK(
      setenv(name, value.c_str(), overwrite ? 1 : 0) == 0,
      "Could not set ",
      name);
}

void configure_nvshmem_environment(const std::string& symmetric_size) {
  set_environment("NVSHMEM_BOOTSTRAP", "MPI", false);
  set_environment("NVSHMEM_REMOTE_TRANSPORT", "ibrc", false);
  set_environment("NVSHMEM_IB_ENABLE_IBGDA", "1", false);
  set_environment("NVSHMEM_IBGDA_NIC_HANDLER", "gpu", false);
  if (symmetric_size.empty()) {
    set_environment("NVSHMEM_SYMMETRIC_SIZE", "512M", false);
  } else {
    set_environment("NVSHMEM_SYMMETRIC_SIZE", symmetric_size, true);
  }
}

void require_initialized() {
  TORCH_CHECK(
      state.initialized,
      "NVSHMEM runtime is not initialized; call dae.nvshmem.init() first");
}

py::dict runtime_info_unlocked() {
  require_initialized();

  int nvshmem_major = 0;
  int nvshmem_minor = 0;
  int nvshmem_patch = 0;
  char nvshmem_name[NVSHMEM_MAX_NAME_LEN] = {};
  nvshmem_info_get_version(&nvshmem_major, &nvshmem_minor);
  nvshmemx_vendor_get_version_info(
      &nvshmem_major, &nvshmem_minor, &nvshmem_patch);
  nvshmem_info_get_name(nvshmem_name);

  py::dict result;
  result["rank"] = state.rank;
  result["world_size"] = state.world_size;
  result["local_rank"] = state.local_rank;
  result["local_size"] = state.local_size;
  result["device"] = state.device;
  result["pe"] = state.pe;
  result["num_pes"] = state.num_pes;
  result["mpi_thread_level"] = state.mpi_thread_level;
  result["owns_mpi"] = state.owns_mpi;
  result["owns_nvshmem"] = state.owns_nvshmem;
  result["nvshmem_name"] = std::string(nvshmem_name);
  result["nvshmem_version"] = py::make_tuple(
      nvshmem_major, nvshmem_minor, nvshmem_patch);
  const char* heap_size = std::getenv("NVSHMEM_SYMMETRIC_SIZE");
  result["symmetric_size"] = heap_size == nullptr ? "" : heap_size;
  result["allocation_count"] = state.allocations.size();
  return result;
}

py::dict initialize(const std::string& symmetric_size, int requested_device) {
  std::lock_guard<std::mutex> lock(state_mutex);
  if (state.initialized) {
    TORCH_CHECK(
        requested_device < 0 || requested_device == state.device,
        "NVSHMEM is already initialized on CUDA device ",
        state.device,
        ", not requested device ",
        requested_device);
    return runtime_info_unlocked();
  }

  configure_nvshmem_environment(symmetric_size);
  RuntimeState pending;

  try {
    int mpi_finalized = 0;
    check_mpi(MPI_Finalized(&mpi_finalized), "MPI_Finalized");
    TORCH_CHECK(!mpi_finalized, "MPI was already finalized and cannot be reinitialized");

    int mpi_initialized = 0;
    check_mpi(MPI_Initialized(&mpi_initialized), "MPI_Initialized");
    if (!mpi_initialized) {
      int argc = 0;
      char** argv = nullptr;
      check_mpi(
          MPI_Init_thread(
              &argc,
              &argv,
              MPI_THREAD_SERIALIZED,
              &pending.mpi_thread_level),
          "MPI_Init_thread");
      pending.owns_mpi = true;
    } else {
      check_mpi(MPI_Query_thread(&pending.mpi_thread_level), "MPI_Query_thread");
    }

    check_mpi(MPI_Comm_rank(MPI_COMM_WORLD, &pending.rank), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(MPI_COMM_WORLD, &pending.world_size), "MPI_Comm_size");
    check_mpi(
        MPI_Comm_split_type(
            MPI_COMM_WORLD,
            MPI_COMM_TYPE_SHARED,
            pending.rank,
            MPI_INFO_NULL,
            &pending.local_comm),
        "MPI_Comm_split_type");
    check_mpi(
        MPI_Comm_rank(pending.local_comm, &pending.local_rank),
        "local MPI_Comm_rank");
    check_mpi(
        MPI_Comm_size(pending.local_comm, &pending.local_size),
        "local MPI_Comm_size");

    int device_count = 0;
    check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    TORCH_CHECK(device_count > 0, "No CUDA devices are visible to this MPI rank");
    TORCH_CHECK(
        pending.local_size <= device_count,
        "Local MPI ranks (",
        pending.local_size,
        ") exceed visible CUDA devices (",
        device_count,
        "); launch one MPI rank per GPU");

    pending.device = requested_device < 0 ? pending.local_rank : requested_device;
    TORCH_CHECK(
        pending.device >= 0 && pending.device < device_count,
        "CUDA device ",
        pending.device,
        " is outside [0, ",
        device_count,
        ")");
    check_cuda(cudaSetDevice(pending.device), "cudaSetDevice");

    const int init_status = nvshmemx_init_status();
    if (init_status < NVSHMEM_STATUS_IS_INITIALIZED) {
      MPI_Comm mpi_comm = MPI_COMM_WORLD;
      nvshmemx_init_attr_t attributes = NVSHMEMX_INIT_ATTR_INITIALIZER;
      attributes.mpi_comm = &mpi_comm;
      const int status = nvshmemx_init_attr(
          NVSHMEMX_INIT_WITH_MPI_COMM, &attributes);
      TORCH_CHECK(status == 0, "nvshmemx_init_attr failed with status ", status);
      pending.owns_nvshmem = true;
    }

    pending.pe = nvshmem_my_pe();
    pending.num_pes = nvshmem_n_pes();
    TORCH_CHECK(
        pending.num_pes == pending.world_size,
        "NVSHMEM PE count ",
        pending.num_pes,
        " does not match MPI world size ",
        pending.world_size);
  } catch (...) {
    if (pending.owns_nvshmem) {
      nvshmem_finalize();
    }
    if (pending.local_comm != MPI_COMM_NULL) {
      MPI_Comm_free(&pending.local_comm);
    }
    if (pending.owns_mpi) {
      int mpi_finalized = 0;
      if (MPI_Finalized(&mpi_finalized) == MPI_SUCCESS && !mpi_finalized) {
        MPI_Finalize();
      }
    }
    throw;
  }

  pending.initialized = true;
  state = std::move(pending);
  return runtime_info_unlocked();
}

size_t tensor_bytes(const std::vector<int64_t>& shape, at::ScalarType dtype) {
  TORCH_CHECK(dtype != at::ScalarType::Undefined, "dtype must be defined");
  size_t elements = 1;
  for (int64_t dimension : shape) {
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
  const size_t logical_bytes = tensor_bytes(shape, dtype);
  const size_t allocation_bytes = std::max<size_t>(logical_bytes, 1);
  void* pointer = zeroed
      ? nvshmem_calloc(1, allocation_bytes)
      : nvshmem_malloc(allocation_bytes);
  TORCH_CHECK(
      pointer != nullptr,
      "NVSHMEM symmetric allocation failed for ",
      allocation_bytes,
      " bytes; increase NVSHMEM_SYMMETRIC_SIZE and keep allocation order identical across PEs");

  state.allocations.push_back({pointer, allocation_bytes});
  const auto options = torch::TensorOptions()
                           .dtype(dtype)
                           .device(torch::Device(torch::kCUDA, state.device));
  return at::from_blob(pointer, shape, [](void*) {}, options);
}

torch::Tensor allocate_tensor(
    const std::vector<int64_t>& shape,
    at::ScalarType dtype,
    bool zeroed) {
  std::lock_guard<std::mutex> lock(state_mutex);
  return allocate_tensor_unlocked(shape, dtype, zeroed);
}

torch::Tensor initialize_signal_space(int64_t signal_count) {
  std::lock_guard<std::mutex> lock(state_mutex);
  require_initialized();
  TORCH_CHECK(signal_count > 0, "signal_count must be positive");
  if (state.signal_space.defined()) {
    TORCH_CHECK(
        signal_count == state.signal_count,
        "Signal space is already initialized with ",
        state.signal_count,
        " entries, not ",
        signal_count);
    return state.signal_space;
  }

  state.signal_space = allocate_tensor_unlocked(
      {signal_count}, at::ScalarType::UInt64, true);
  state.signal_count = signal_count;
  nvshmem_barrier_all();
  return state.signal_space;
}

torch::Tensor get_signal_space() {
  std::lock_guard<std::mutex> lock(state_mutex);
  require_initialized();
  TORCH_CHECK(
      state.signal_space.defined(),
      "Signal space is not initialized; call init_signal_space() first");
  return state.signal_space;
}

bool is_symmetric_tensor(const torch::Tensor& tensor) {
  std::lock_guard<std::mutex> lock(state_mutex);
  if (!state.initialized || !tensor.defined() || !tensor.is_cuda() ||
      tensor.get_device() != state.device) {
    return false;
  }

  const uintptr_t address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
  for (const SymmetricAllocation& allocation : state.allocations) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(allocation.ptr);
    const uintptr_t end = begin + allocation.bytes;
    if (address >= begin && address < end) {
      return true;
    }
  }
  return false;
}

uint64_t* signal_address_unlocked(int64_t index) {
  require_initialized();
  TORCH_CHECK(
      state.signal_space.defined(),
      "Signal space is not initialized; call init_signal_space() first");
  TORCH_CHECK(
      index >= 0 && index < state.signal_count,
      "signal index ",
      index,
      " is outside [0, ",
      state.signal_count,
      ")");
  return state.signal_space.data_ptr<uint64_t>() + index;
}

void signal_on_stream(
    int64_t index,
    uint64_t value,
    int operation,
    int pe,
    int64_t stream) {
  std::lock_guard<std::mutex> lock(state_mutex);
  TORCH_CHECK(
      operation == NVSHMEM_SIGNAL_SET || operation == NVSHMEM_SIGNAL_ADD,
      "unsupported NVSHMEM signal operation ",
      operation);
  TORCH_CHECK(pe >= 0 && pe < state.num_pes, "target PE is out of range");
  nvshmemx_signal_op_on_stream(
      signal_address_unlocked(index),
      value,
      operation,
      pe,
      reinterpret_cast<cudaStream_t>(stream));
}

void wait_signal_on_stream(
    int64_t index,
    int comparison,
    uint64_t value,
    int64_t stream) {
  std::lock_guard<std::mutex> lock(state_mutex);
  TORCH_CHECK(
      comparison >= NVSHMEM_CMP_EQ && comparison <= NVSHMEM_CMP_GE,
      "unsupported NVSHMEM comparison ",
      comparison);
  nvshmemx_signal_wait_until_on_stream(
      signal_address_unlocked(index),
      comparison,
      value,
      reinterpret_cast<cudaStream_t>(stream));
}

void quiet_on_stream(int64_t stream) {
  std::lock_guard<std::mutex> lock(state_mutex);
  require_initialized();
  nvshmemx_quiet_on_stream(reinterpret_cast<cudaStream_t>(stream));
}

void barrier_all() {
  std::lock_guard<std::mutex> lock(state_mutex);
  require_initialized();
  nvshmem_barrier_all();
}

py::dict runtime_info() {
  std::lock_guard<std::mutex> lock(state_mutex);
  return runtime_info_unlocked();
}

bool is_initialized() {
  std::lock_guard<std::mutex> lock(state_mutex);
  return state.initialized;
}

void finalize() {
  std::lock_guard<std::mutex> lock(state_mutex);
  if (!state.initialized) {
    return;
  }

  check_cuda(cudaSetDevice(state.device), "cudaSetDevice");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  nvshmem_barrier_all();

  state.signal_space = torch::Tensor();
  state.signal_count = 0;
  for (auto allocation = state.allocations.rbegin();
       allocation != state.allocations.rend();
       ++allocation) {
    nvshmem_free(allocation->ptr);
  }
  state.allocations.clear();

  if (state.owns_nvshmem) {
    nvshmem_finalize();
  }
  if (state.local_comm != MPI_COMM_NULL) {
    check_mpi(MPI_Comm_free(&state.local_comm), "MPI_Comm_free");
  }
  if (state.owns_mpi) {
    int mpi_finalized = 0;
    check_mpi(MPI_Finalized(&mpi_finalized), "MPI_Finalized");
    if (!mpi_finalized) {
      check_mpi(MPI_Finalize(), "MPI_Finalize");
    }
  }

  state = RuntimeState{};
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.doc() = "Optional MPI-bootstrapped NVSHMEM runtime for DAE";
  module.def(
      "initialize",
      &initialize,
      py::arg("symmetric_size") = "",
      py::arg("device") = -1,
      "Initialize MPI and NVSHMEM, selecting the node-local CUDA device");
  module.def("is_initialized", &is_initialized);
  module.def("info", &runtime_info);
  module.def(
      "allocate_tensor",
      &allocate_tensor,
      py::arg("shape"),
      py::arg("dtype"),
      py::arg("zeroed") = false,
      "Collectively allocate a contiguous CUDA tensor in the NVSHMEM symmetric heap");
  module.def(
      "init_signal_space",
      &initialize_signal_space,
      py::arg("signal_count"),
      "Collectively allocate and zero the process-global uint64 signal space");
  module.def("get_signal_space", &get_signal_space);
  module.def("is_symmetric_tensor", &is_symmetric_tensor, py::arg("tensor"));
  module.def(
      "signal_on_stream",
      &signal_on_stream,
      py::arg("index"),
      py::arg("value"),
      py::arg("operation"),
      py::arg("pe"),
      py::arg("stream"));
  module.def(
      "wait_signal_on_stream",
      &wait_signal_on_stream,
      py::arg("index"),
      py::arg("comparison"),
      py::arg("value"),
      py::arg("stream"));
  module.def("quiet_on_stream", &quiet_on_stream, py::arg("stream"));
  module.def("barrier_all", &barrier_all);
  module.def("finalize", &finalize);

  module.attr("SIGNAL_SET") = static_cast<int>(NVSHMEM_SIGNAL_SET);
  module.attr("SIGNAL_ADD") = static_cast<int>(NVSHMEM_SIGNAL_ADD);
  module.attr("CMP_EQ") = static_cast<int>(NVSHMEM_CMP_EQ);
  module.attr("CMP_NE") = static_cast<int>(NVSHMEM_CMP_NE);
  module.attr("CMP_GT") = static_cast<int>(NVSHMEM_CMP_GT);
  module.attr("CMP_LE") = static_cast<int>(NVSHMEM_CMP_LE);
  module.attr("CMP_LT") = static_cast<int>(NVSHMEM_CMP_LT);
  module.attr("CMP_GE") = static_cast<int>(NVSHMEM_CMP_GE);
}
