#include <torch/extension.h>

#include <ATen/ops/from_blob.h>
#include <c10/core/ScalarType.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
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

struct FabricMemoryMapping {
  int device = -1;
  CUmemGenericAllocationHandle memory_handle = 0;
  CUdeviceptr address = 0;
  size_t bytes = 0;
  bool owner = false;
};

struct FabricMulticastAllocation {
  int device = -1;
  CUmemGenericAllocationHandle multicast_handle = 0;
  CUmemGenericAllocationHandle memory_handle = 0;
  CUdeviceptr unicast_address = 0;
  CUdeviceptr multicast_address = 0;
  size_t bytes = 0;
  size_t address_alignment = 0;
  bool device_added = false;
  bool bound = false;
};

std::mutex allocation_mutex;
std::vector<MulticastAllocation> allocations;
std::vector<std::unique_ptr<FabricMemoryMapping>> fabric_mappings;
std::vector<std::unique_ptr<FabricMulticastAllocation>>
    fabric_multicast_allocations;

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

void validate_fabric_device(int device, bool require_multicast) {
  int count = 0;
  check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
  TORCH_CHECK(device >= 0 && device < count, "invalid CUDA device ", device);
  int fabric = 0;
  check_driver(
      cuDeviceGetAttribute(
          &fabric,
          CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
          device),
      "cuDeviceGetAttribute(HANDLE_TYPE_FABRIC_SUPPORTED)");
  TORCH_CHECK(
      fabric != 0,
      "CUDA device ",
      device,
      " does not support Fabric allocation handles");
  if (require_multicast) {
    int multicast = 0;
    check_driver(
        cuDeviceGetAttribute(
            &multicast,
            CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
            device),
        "cuDeviceGetAttribute(MULTICAST_SUPPORTED)");
    TORCH_CHECK(
        multicast != 0, "CUDA device ", device, " lacks multicast");
  }
}

size_t round_up(size_t value, size_t alignment) {
  TORCH_CHECK(alignment != 0, "CUDA allocation alignment must be nonzero");
  TORCH_CHECK(
      value <= SIZE_MAX - (alignment - 1),
      "CUDA allocation size overflows size_t");
  return (value + alignment - 1) / alignment * alignment;
}

CUmemAllocationProp fabric_memory_prop(int device) {
  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  return prop;
}

CUmemAccessDesc read_write_access(int device) {
  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return access;
}

py::bytes encode_fabric_handle(const CUmemFabricHandle& handle) {
  return py::bytes(
      reinterpret_cast<const char*>(&handle), sizeof(CUmemFabricHandle));
}

CUmemFabricHandle decode_fabric_handle(const py::bytes& encoded) {
  const std::string bytes = encoded;
  TORCH_CHECK(
      bytes.size() == sizeof(CUmemFabricHandle),
      "invalid CUDA Fabric handle size: expected ",
      sizeof(CUmemFabricHandle),
      ", got ",
      bytes.size());
  CUmemFabricHandle handle{};
  std::memcpy(&handle, bytes.data(), sizeof(handle));
  return handle;
}

torch::Tensor byte_tensor(
    CUdeviceptr address, size_t requested_bytes, int device) {
  TORCH_CHECK(
      requested_bytes <= static_cast<size_t>(INT64_MAX),
      "CUDA tensor view exceeds int64 capacity");
  const auto options = torch::TensorOptions()
      .dtype(torch::kUInt8)
      .device(torch::Device(torch::kCUDA, device));
  return at::from_blob(
      reinterpret_cast<void*>(address),
      {static_cast<int64_t>(requested_bytes)},
      [](void*) {},
      options);
}

py::tuple allocate_fabric_arena(int device, size_t requested_bytes) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  TORCH_CHECK(requested_bytes > 0, "Fabric arena must be nonempty");
  select_device(device);
  validate_fabric_device(device, false);

  const CUmemAllocationProp prop = fabric_memory_prop(device);
  size_t granularity = 0;
  check_driver(
      cuMemGetAllocationGranularity(
          &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity(Fabric arena)");
  const size_t bytes = round_up(requested_bytes, granularity);

  auto mapping = std::make_unique<FabricMemoryMapping>();
  mapping->device = device;
  mapping->bytes = bytes;
  mapping->owner = true;
  check_driver(
      cuMemCreate(&mapping->memory_handle, bytes, &prop, 0),
      "cuMemCreate(Fabric arena)");
  CUmemFabricHandle fabric_handle{};
  check_driver(
      cuMemExportToShareableHandle(
          &fabric_handle,
          mapping->memory_handle,
          CU_MEM_HANDLE_TYPE_FABRIC,
          0),
      "cuMemExportToShareableHandle(Fabric arena)");
  check_driver(
      cuMemAddressReserve(&mapping->address, bytes, granularity, 0, 0),
      "cuMemAddressReserve(Fabric arena)");
  check_driver(
      cuMemMap(mapping->address, bytes, 0, mapping->memory_handle, 0),
      "cuMemMap(Fabric arena)");
  const CUmemAccessDesc access = read_write_access(device);
  check_driver(
      cuMemSetAccess(mapping->address, bytes, &access, 1),
      "cuMemSetAccess(Fabric arena)");
  check_cuda(
      cudaMemset(reinterpret_cast<void*>(mapping->address), 0, bytes),
      "cudaMemset(Fabric arena)");

  torch::Tensor tensor = byte_tensor(mapping->address, requested_bytes, device);
  fabric_mappings.push_back(std::move(mapping));
  return py::make_tuple(tensor, encode_fabric_handle(fabric_handle), bytes);
}

uint64_t import_fabric_arena(
    int device, const py::bytes& encoded_handle, size_t bytes) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  TORCH_CHECK(bytes > 0, "imported Fabric arena must be nonempty");
  select_device(device);
  validate_fabric_device(device, false);
  const CUmemFabricHandle fabric_handle =
      decode_fabric_handle(encoded_handle);

  auto mapping = std::make_unique<FabricMemoryMapping>();
  mapping->device = device;
  mapping->bytes = bytes;
  check_driver(
      cuMemImportFromShareableHandle(
          &mapping->memory_handle,
          const_cast<CUmemFabricHandle*>(&fabric_handle),
          CU_MEM_HANDLE_TYPE_FABRIC),
      "cuMemImportFromShareableHandle(Fabric arena)");
  check_driver(
      cuMemAddressReserve(&mapping->address, bytes, 0, 0, 0),
      "cuMemAddressReserve(imported Fabric arena)");
  check_driver(
      cuMemMap(mapping->address, bytes, 0, mapping->memory_handle, 0),
      "cuMemMap(imported Fabric arena)");
  const CUmemAccessDesc access = read_write_access(device);
  check_driver(
      cuMemSetAccess(mapping->address, bytes, &access, 1),
      "cuMemSetAccess(imported Fabric arena)");

  const uint64_t address = mapping->address;
  fabric_mappings.push_back(std::move(mapping));
  return address;
}

py::tuple create_fabric_multicast(
    int device, size_t num_devices, size_t requested_bytes) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  TORCH_CHECK(num_devices > 0, "multicast team must be nonempty");
  TORCH_CHECK(
      num_devices <= 32, "multicast team exceeds PoolSlice PE capacity");
  TORCH_CHECK(requested_bytes > 0, "multicast allocation must be nonempty");
  select_device(device);
  validate_fabric_device(device, true);

  CUmulticastObjectProp multicast_prop{};
  multicast_prop.numDevices = num_devices;
  multicast_prop.handleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  multicast_prop.flags = 0;
  size_t multicast_granularity = 0;
  check_driver(
      cuMulticastGetGranularity(
          &multicast_granularity,
          &multicast_prop,
          CU_MULTICAST_GRANULARITY_RECOMMENDED),
      "cuMulticastGetGranularity(Fabric multicast)");
  const CUmemAllocationProp memory_prop = fabric_memory_prop(device);
  size_t memory_granularity = 0;
  check_driver(
      cuMemGetAllocationGranularity(
          &memory_granularity,
          &memory_prop,
          CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity(Fabric multicast backing)");
  const size_t alignment =
      std::lcm(multicast_granularity, memory_granularity);
  const size_t bytes = round_up(requested_bytes, alignment);
  multicast_prop.size = bytes;

  auto allocation = std::make_unique<FabricMulticastAllocation>();
  allocation->device = device;
  allocation->bytes = bytes;
  allocation->address_alignment = alignment;
  check_driver(
      cuMulticastCreate(&allocation->multicast_handle, &multicast_prop),
      "cuMulticastCreate(Fabric multicast)");
  CUmemFabricHandle fabric_handle{};
  check_driver(
      cuMemExportToShareableHandle(
          &fabric_handle,
          allocation->multicast_handle,
          CU_MEM_HANDLE_TYPE_FABRIC,
          0),
      "cuMemExportToShareableHandle(Fabric multicast)");

  const size_t allocation_id = fabric_multicast_allocations.size();
  fabric_multicast_allocations.push_back(std::move(allocation));
  return py::make_tuple(
      allocation_id,
      encode_fabric_handle(fabric_handle),
      bytes,
      alignment);
}

size_t import_fabric_multicast(
    int device,
    const py::bytes& encoded_handle,
    size_t bytes,
    size_t address_alignment) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  TORCH_CHECK(bytes > 0, "imported multicast allocation must be nonempty");
  TORCH_CHECK(
      address_alignment > 0,
      "imported multicast address alignment must be nonzero");
  select_device(device);
  validate_fabric_device(device, true);
  const CUmemFabricHandle fabric_handle =
      decode_fabric_handle(encoded_handle);

  auto allocation = std::make_unique<FabricMulticastAllocation>();
  allocation->device = device;
  allocation->bytes = bytes;
  allocation->address_alignment = address_alignment;
  check_driver(
      cuMemImportFromShareableHandle(
          &allocation->multicast_handle,
          const_cast<CUmemFabricHandle*>(&fabric_handle),
          CU_MEM_HANDLE_TYPE_FABRIC),
      "cuMemImportFromShareableHandle(Fabric multicast)");
  const size_t allocation_id = fabric_multicast_allocations.size();
  fabric_multicast_allocations.push_back(std::move(allocation));
  return allocation_id;
}

FabricMulticastAllocation& fabric_multicast_allocation(size_t allocation_id) {
  TORCH_CHECK(
      allocation_id < fabric_multicast_allocations.size(),
      "invalid Fabric multicast allocation id ",
      allocation_id);
  return *fabric_multicast_allocations[allocation_id];
}

void add_fabric_multicast_device(size_t allocation_id, int device) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  FabricMulticastAllocation& allocation =
      fabric_multicast_allocation(allocation_id);
  TORCH_CHECK(
      device == allocation.device,
      "multicast allocation belongs to CUDA device ",
      allocation.device,
      ", not ",
      device);
  TORCH_CHECK(!allocation.device_added, "multicast device already added");
  select_device(device);
  check_driver(
      cuMulticastAddDevice(allocation.multicast_handle, device),
      "cuMulticastAddDevice(Fabric multicast)");
  allocation.device_added = true;
}

py::tuple bind_fabric_multicast(
    size_t allocation_id, int device, size_t requested_bytes) {
  std::lock_guard<std::mutex> lock(allocation_mutex);
  FabricMulticastAllocation& allocation =
      fabric_multicast_allocation(allocation_id);
  TORCH_CHECK(
      device == allocation.device,
      "multicast allocation belongs to CUDA device ",
      allocation.device,
      ", not ",
      device);
  TORCH_CHECK(allocation.device_added, "add the multicast device before bind");
  TORCH_CHECK(!allocation.bound, "multicast backing already bound");
  TORCH_CHECK(
      requested_bytes > 0 && requested_bytes <= allocation.bytes,
      "invalid multicast tensor view size");
  select_device(device);

  const CUmemAllocationProp memory_prop = fabric_memory_prop(device);
  size_t memory_granularity = 0;
  check_driver(
      cuMemGetAllocationGranularity(
          &memory_granularity,
          &memory_prop,
          CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
      "cuMemGetAllocationGranularity(Fabric multicast backing)");
  TORCH_CHECK(
      allocation.bytes % memory_granularity == 0,
      "multicast and physical allocation granularities are incompatible");
  check_driver(
      cuMemCreate(
          &allocation.memory_handle,
          allocation.bytes,
          &memory_prop,
          0),
      "cuMemCreate(Fabric multicast backing)");
  check_driver(
      cuMulticastBindMem(
          allocation.multicast_handle,
          0,
          allocation.memory_handle,
          0,
          allocation.bytes,
          0),
      "cuMulticastBindMem(Fabric multicast)");

  check_driver(
      cuMemAddressReserve(
          &allocation.unicast_address,
          allocation.bytes,
          memory_granularity,
          0,
          0),
      "cuMemAddressReserve(Fabric multicast unicast)");
  check_driver(
      cuMemMap(
          allocation.unicast_address,
          allocation.bytes,
          0,
          allocation.memory_handle,
          0),
      "cuMemMap(Fabric multicast unicast)");
  const CUmemAccessDesc access = read_write_access(device);
  check_driver(
      cuMemSetAccess(
          allocation.unicast_address, allocation.bytes, &access, 1),
      "cuMemSetAccess(Fabric multicast unicast)");

  check_driver(
      cuMemAddressReserve(
          &allocation.multicast_address,
          allocation.bytes,
          allocation.address_alignment,
          0,
          0),
      "cuMemAddressReserve(Fabric multicast alias)");
  check_driver(
      cuMemMap(
          allocation.multicast_address,
          allocation.bytes,
          0,
          allocation.multicast_handle,
          0),
      "cuMemMap(Fabric multicast alias)");
  check_driver(
      cuMemSetAccess(
          allocation.multicast_address, allocation.bytes, &access, 1),
      "cuMemSetAccess(Fabric multicast alias)");
  check_cuda(
      cudaMemset(
          reinterpret_cast<void*>(allocation.unicast_address),
          0,
          allocation.bytes),
      "cudaMemset(Fabric multicast backing)");
  allocation.bound = true;

  return py::make_tuple(
      byte_tensor(allocation.unicast_address, requested_bytes, device),
      py::int_(allocation.multicast_address));
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
  m.doc() =
      "CUDA peer, Fabric, and multicast allocation for GB300 pool slices";
  m.def("enable_peer_access", &enable_peer_access, py::arg("devices"));
  m.def(
      "allocate_multicast",
      &allocate_multicast,
      py::arg("devices"),
      py::arg("bytes"));
  m.def(
      "allocate_fabric_arena",
      &allocate_fabric_arena,
      py::arg("device"),
      py::arg("bytes"));
  m.def(
      "import_fabric_arena",
      &import_fabric_arena,
      py::arg("device"),
      py::arg("handle"),
      py::arg("bytes"));
  m.def(
      "create_fabric_multicast",
      &create_fabric_multicast,
      py::arg("device"),
      py::arg("num_devices"),
      py::arg("bytes"));
  m.def(
      "import_fabric_multicast",
      &import_fabric_multicast,
      py::arg("device"),
      py::arg("handle"),
      py::arg("bytes"),
      py::arg("address_alignment"));
  m.def(
      "add_fabric_multicast_device",
      &add_fabric_multicast_device,
      py::arg("allocation_id"),
      py::arg("device"));
  m.def(
      "bind_fabric_multicast",
      &bind_fabric_multicast,
      py::arg("allocation_id"),
      py::arg("device"),
      py::arg("requested_bytes"));
}
