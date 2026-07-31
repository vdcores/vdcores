#include <torch/extension.h>

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

namespace py = pybind11;

namespace {

struct MulticastAllocation {
  std::vector<int> devices;
  std::vector<CUmemGenericAllocationHandle> memory_handles;
  std::vector<CUdeviceptr> unicast_addresses;
  std::vector<CUdeviceptr> multicast_addresses;
  CUmemGenericAllocationHandle multicast_handle = 0;
  size_t bytes = 0;
};

std::mutex allocation_mutex;
std::vector<MulticastAllocation> allocations;

void check_cuda(cudaError_t status, const char* operation) {
  TORCH_CHECK(
      status == cudaSuccess,
      operation,
      " failed: ",
      cudaGetErrorString(status));
}

void check_driver(CUresult status, const char* operation) {
  if (status == CUDA_SUCCESS)
    return;
  const char* name = nullptr;
  const char* message = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &message);
  TORCH_CHECK(false, operation, " failed: ", name, " (", message, ")");
}

void select_device(int device) {
  check_cuda(cudaSetDevice(device), "cudaSetDevice");
  check_cuda(cudaFree(nullptr), "CUDA primary-context initialization");
}

void validate_devices(const std::vector<int>& devices) {
  TORCH_CHECK(!devices.empty(), "at least one CUDA device is required");
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  for (size_t index = 0; index < devices.size(); ++index) {
    const int device = devices[index];
    TORCH_CHECK(device >= 0 && device < count, "invalid CUDA device ", device);
    TORCH_CHECK(
        std::find(devices.begin(), devices.begin() + index, device) ==
            devices.begin() + index,
        "duplicate CUDA device ",
        device);
    int multicast = 0;
    check_driver(
        cuDeviceGetAttribute(
            &multicast, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device),
        "cuDeviceGetAttribute(MULTICAST_SUPPORTED)");
    TORCH_CHECK(multicast != 0, "CUDA device ", device, " lacks multicast");
  }
}

void enable_peer_access(const std::vector<int>& devices) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  validate_devices(devices);
  for (const int source : devices) {
    select_device(source);
    for (const int target : devices) {
      if (source == target)
        continue;
      int accessible = 0;
      check_cuda(
          cudaDeviceCanAccessPeer(&accessible, source, target),
          "cudaDeviceCanAccessPeer");
      TORCH_CHECK(
          accessible != 0,
          "CUDA device ",
          source,
          " cannot directly access local peer ",
          target);
      const cudaError_t status = cudaDeviceEnablePeerAccess(target, 0);
      if (status != cudaSuccess && status != cudaErrorPeerAccessAlreadyEnabled)
        check_cuda(status, "cudaDeviceEnablePeerAccess");
      if (status == cudaErrorPeerAccessAlreadyEnabled)
        cudaGetLastError();
    }
  }
}

py::tuple allocate_multicast(
    const std::vector<int>& devices,
    size_t requested_bytes) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  validate_devices(devices);
  TORCH_CHECK(requested_bytes > 0, "multicast allocation must be nonempty");

  CUmulticastObjectProp multicast_prop{};
  multicast_prop.numDevices = devices.size();
  multicast_prop.handleTypes = CU_MEM_HANDLE_TYPE_NONE;
  multicast_prop.flags = 0;
  size_t multicast_granularity = 0;
  check_driver(
      cuMulticastGetGranularity(
          &multicast_granularity,
          &multicast_prop,
          CU_MULTICAST_GRANULARITY_RECOMMENDED),
      "cuMulticastGetGranularity");
  const size_t bytes =
      (requested_bytes + multicast_granularity - 1) /
      multicast_granularity * multicast_granularity;
  multicast_prop.size = bytes;

  MulticastAllocation allocation;
  allocation.devices = devices;
  allocation.bytes = bytes;
  allocation.memory_handles.resize(devices.size());
  allocation.unicast_addresses.resize(devices.size());
  allocation.multicast_addresses.resize(devices.size());
  check_driver(
      cuMulticastCreate(&allocation.multicast_handle, &multicast_prop),
      "cuMulticastCreate");
  for (const int device : devices)
    check_driver(
        cuMulticastAddDevice(allocation.multicast_handle, device),
        "cuMulticastAddDevice");

  for (size_t index = 0; index < devices.size(); ++index) {
    const int device = devices[index];
    select_device(device);
    CUmemAllocationProp memory_prop{};
    memory_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memory_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memory_prop.location.id = device;
    memory_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    size_t memory_granularity = 0;
    check_driver(
        cuMemGetAllocationGranularity(
            &memory_granularity,
            &memory_prop,
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
        "cuMemGetAllocationGranularity");
    TORCH_CHECK(
        bytes % memory_granularity == 0,
        "multicast and physical allocation granularities are incompatible");
    check_driver(
        cuMemCreate(
            &allocation.memory_handles[index], bytes, &memory_prop, 0),
        "cuMemCreate");
    check_driver(
        cuMulticastBindMem(
            allocation.multicast_handle,
            0,
            allocation.memory_handles[index],
            0,
            bytes,
            0),
        "cuMulticastBindMem");

    check_driver(
        cuMemAddressReserve(
            &allocation.unicast_addresses[index], bytes, 0, 0, 0),
        "cuMemAddressReserve(unicast)");
    check_driver(
        cuMemMap(
            allocation.unicast_addresses[index],
            bytes,
            0,
            allocation.memory_handles[index],
            0),
        "cuMemMap(unicast)");
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check_driver(
        cuMemSetAccess(
            allocation.unicast_addresses[index], bytes, &access, 1),
        "cuMemSetAccess(unicast)");

    check_driver(
        cuMemAddressReserve(
            &allocation.multicast_addresses[index], bytes, 0, 0, 0),
        "cuMemAddressReserve(multicast)");
    check_driver(
        cuMemMap(
            allocation.multicast_addresses[index],
            bytes,
            0,
            allocation.multicast_handle,
            0),
        "cuMemMap(multicast)");
    check_driver(
        cuMemSetAccess(
            allocation.multicast_addresses[index], bytes, &access, 1),
        "cuMemSetAccess(multicast)");
    check_cuda(
        cudaMemset(
            reinterpret_cast<void*>(allocation.unicast_addresses[index]),
            0,
            bytes),
        "cudaMemset(multicast backing)");
  }

  py::list tensors;
  py::list multicast_addresses;
  for (size_t index = 0; index < devices.size(); ++index) {
    const auto options = torch::TensorOptions()
        .dtype(torch::kUInt8)
        .device(torch::Device(torch::kCUDA, devices[index]));
    tensors.append(at::from_blob(
        reinterpret_cast<void*>(allocation.unicast_addresses[index]),
        {static_cast<int64_t>(requested_bytes)},
        [](void*) {},
        options));
    multicast_addresses.append(
        py::int_(allocation.multicast_addresses[index]));
  }
  allocations.push_back(std::move(allocation));
  return py::make_tuple(tensors, multicast_addresses, bytes);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "CUDA peer and multicast allocation for local GB300 pool slices";
  m.def("enable_peer_access", &enable_peer_access, py::arg("devices"));
  m.def(
      "allocate_multicast",
      &allocate_multicast,
      py::arg("devices"),
      py::arg("bytes"));
}
