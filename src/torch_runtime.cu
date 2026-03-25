#include "dae/runtime.cuh"
#include "dae/context.cuh"

#include <torch/extension.h>
#include <pybind11/stl.h>

#include <cuda.h>            // Driver API
#include <cuda_runtime.h>

#include <vector>
#include <cstdint>

namespace py = pybind11;

// function 1: set smem size
size_t py_set_smem_size(int64_t device_id, size_t requested_size) {
  return set_smem_size(static_cast<int>(device_id), requested_size);
}

template <typename T>
static inline T* check_tensor_ptr(torch::Tensor t, const char* name) {
  TORCH_CHECK(t.defined(), name, " must be defined");
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == torch::kUInt8, name, " must be uint8");
  TORCH_CHECK(t.dim() == 2, name, " must be rank-2");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");

  const int64_t rows = t.size(0);
  const int64_t cols = t.size(1);

  TORCH_CHECK(cols == (int64_t)sizeof(T),
              name, " second dimension must equal sizeof(T) = ",
              sizeof(T), " but got ", cols);

  // Now memory layout is guaranteed to be:
  // rows contiguous records of sizeof(T) bytes each.
  auto* p = reinterpret_cast<T*>(t.data_ptr<uint8_t>());

  // Alignment safety (important for 16-byte aligned structs)
  uintptr_t addr = reinterpret_cast<uintptr_t>(p);
  TORCH_CHECK(addr % alignof(T) == 0,
              name, " misaligned pointer: address mod alignof(T) = ",
              (addr % alignof(T)));

  return p;
}

static void set_persistent_cache() {
  // This function can be used to set up any global state or configuration
  // needed for persistent caching. For now, it's a placeholder.

  cudaDeviceProp prop{};
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  // printf("L2 size: %d bytes\n", prop.l2CacheSize);
  // printf("persistingL2CacheMaxSize: %zu bytes\n", prop.persistingL2CacheMaxSize);
  // printf("accessPolicyMaxWindowSize: %zu bytes\n", prop.accessPolicyMaxWindowSize);

  size_t recommended_size = prop.persistingL2CacheMaxSize * 2 / 8; // Example heuristic

  size_t set_aside = std::min<size_t>(recommended_size, prop.persistingL2CacheMaxSize);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside);
  // printf("persistentCacheSize: %zu bytes\n", set_aside);
}

static int64_t current_or_specified_device(int64_t device_id) {
  if (device_id >= 0) {
    return device_id;
  }
  int current_device = 0;
  cudaGetDevice(&current_device);
  return current_device;
}

// function 2: launch_dae
int py_launch_dae(
    int64_t device_id,
    int64_t num_sms,
    size_t smem_size,
    torch::Tensor compute_insts_bytes,   // uint8 buffer
    torch::Tensor memory_insts_bytes,    // uint8 buffer
    torch::Tensor tma_descs_bytes,       // uint8 buffer
    torch::Tensor bars_int32,            // int32
    torch::Tensor profile_u64,           // uint64
    int64_t stream
) {
  const int launch_device_id = static_cast<int>(current_or_specified_device(device_id));
  cudaSetDevice(launch_device_id);
  set_persistent_cache();

  // fixed for H100 for now
  TORCH_CHECK(num_sms >= 0 && num_sms <= 132, "num_sms out of range");

  // Make sure we run on the right device/stream
  auto cinst = check_tensor_ptr<CInst>(compute_insts_bytes, "compute_insts_bytes");
  auto minst = check_tensor_ptr<MInst>(memory_insts_bytes, "memory_insts_bytes");
  auto tma = check_tensor_ptr<CUtensorMap>(tma_descs_bytes, "tma_descs_bytes");
  auto bars = check_tensor_ptr<int>(bars_int32, "bars_int32");
  auto prof = check_tensor_ptr<uint64_t>(profile_u64, "profile_u64");

  cudaError_t st = launch_dae(
      launch_device_id,
      static_cast<int>(num_sms), smem_size,
      cinst, minst, tma,
      bars, prof, stream
  );

  TORCH_CHECK(st == cudaSuccess, "launch_dae failed: ", cudaGetErrorString(st));

  // Return something meaningful; often you return profile or nothing.
  return 0;
}

// function 3: build TMA descriptors
static inline CUtensorMapInterleave to_interleave(int64_t interleave) {
  switch (interleave) {
    case 0: return CU_TENSOR_MAP_INTERLEAVE_NONE;
    case 16: return CU_TENSOR_MAP_INTERLEAVE_16B;
    case 32: return CU_TENSOR_MAP_INTERLEAVE_32B;
    default: TORCH_CHECK(false, "Unsupported interleave=", interleave, " (expected 0/16/32)");
  }
}

static inline CUtensorMapSwizzle to_swizzle(int64_t swizzle_bytes) {
  switch (swizzle_bytes) {
    case 0:   return CU_TENSOR_MAP_SWIZZLE_NONE;
    case 32:  return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64:  return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    default: TORCH_CHECK(false, "Unsupported swizzle_bytes=", swizzle_bytes, " (expected 0/32/64/128)");
  }
}

static inline CUtensorMapDataType to_dtype(torch::ScalarType st) {
  // Extend as you need
  switch (st) {
    case torch::kFloat16:  return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case torch::kBFloat16: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case torch::kFloat32:  return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case torch::kUInt8:    return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case torch::kInt32:    return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case torch::kUInt32:   return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype for TMA: ", c10::toString(st));
  }
}

// Build a CUtensorMap descriptor for a tensor.
// Arguments that must be consistent with your kernel's expected layout.
//
// shape:          sizes in elements, rank R
// strides_bytes:  strides in BYTES, rank R  (yes, bytes; not elements)
// box_dim:        tile dimensions in elements, rank R
// elem_strides:   element strides inside the tile, rank R (often all-ones)
// swizzle_bytes:  0/32/64/128
// interleave:     0 for NONE, 1 for 16B, 2 for 32B (optional; use NONE if unsure)
// l2_promo:       0 NONE, 1 64B, 2 128B, 3 256B (varies; use 256B commonly)
// oob_fill:       0 NONE, 1 NAN (float) etc (usually NONE)
torch::Tensor py_build_tma_desc(
    torch::Tensor base,                    // CUDA tensor providing base_ptr + device
    std::vector<int64_t> shape,            // length R
    std::vector<int64_t> strides_bytes,    // length R
    std::vector<int64_t> box_dim,          // length R
    std::vector<int64_t> elem_strides,     // length R
    int64_t swizzle_bytes,
    int64_t interleave_bytes
) {
  TORCH_CHECK(base.defined(), "base must be defined");
  TORCH_CHECK(base.is_cuda(), "base must be a CUDA tensor");
  TORCH_CHECK(base.numel() > 0, "base must have storage");
  TORCH_CHECK(shape.size() == strides_bytes.size() + 1, "shape and strides_bytes must have same length");
  TORCH_CHECK(shape.size() == box_dim.size(), "shape and box_dim must have same length");
  TORCH_CHECK(shape.size() == elem_strides.size(), "shape and elem_strides must have same length");

  const int R = (int)shape.size();
  TORCH_CHECK(R >= 1 && R <= 5, "tensorRank=", R, " not supported here (adjust if needed)");

  // Allocate descriptor storage on device as opaque bytes
  auto desc = torch::empty({(int64_t)sizeof(CUtensorMap)},
                           torch::TensorOptions().dtype(torch::kUInt8));

  cudaSetDevice(base.get_device());

  // Prepare arrays
  std::vector<cuuint64_t> gdim(5, 0);
  std::vector<cuuint64_t> gstride(5, 0);
  std::vector<cuuint32_t> bdim(5, 0);
  std::vector<cuuint32_t> estride(5, 0);

  for (int i = 0; i < R; i++) {
    TORCH_CHECK(shape[i] > 0, "shape[", i, "] must be > 0");
    TORCH_CHECK(box_dim[i] > 0, "box_dim[", i, "] must be > 0");
    TORCH_CHECK(elem_strides[i] > 0, "elem_strides[", i, "] must be > 0");
    gdim[i]    = (cuuint64_t)shape[i];
    bdim[i]    = (cuuint32_t)box_dim[i];
    estride[i] = (cuuint32_t)elem_strides[i];

    if (i < R - 1) {
      // TORCH_CHECK(strides_bytes[i] > 0, "strides_bytes[", i, "] must be > 0");
      gstride[i] = (cuuint64_t)strides_bytes[i];
    } else
      gstride[i] = (cuuint64_t)0; // last stride is not used by hardware, can be 0
  }

  CUtensorMapDataType dtype = to_dtype(base.scalar_type());
  CUtensorMapSwizzle swz = to_swizzle(swizzle_bytes);
  CUtensorMapInterleave interleave = to_interleave(interleave_bytes);

  // CUtensorMapL2promotion l2p = CU_TENSOR_MAP_L2_PROMOTION_NONE;
  CUtensorMapL2promotion l2p = CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  CUtensorMapFloatOOBfill oob = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

  // Fill descriptor in device memory
  CUtensorMap* tma = reinterpret_cast<CUtensorMap*>(desc.data_ptr<uint8_t>());

  CUresult r = cuTensorMapEncodeTiled(
      tma,
      dtype,
      (cuuint32_t)R,
      (void*)base.data_ptr(),
      gdim.data(),
      gstride.data(),
      bdim.data(),
      estride.data(),
      interleave,
      swz,
      l2p,
      oob
  );

  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed with error code ", r);

  return desc;
}

enum CachePolicy : int {
  DAE_CACHE_NORMAL = cudaAccessPropertyNormal,
  DAE_CACHE_STREAMING = cudaAccessPropertyStreaming,
  DAE_CACHE_PERSISTING = cudaAccessPropertyPersisting
};

// Set cache policy for a CUDA tensor on the specified stream.
void py_tensor_set_cache_policy(torch::Tensor t, int64_t stream_id, float hit_ratio, int hit_policy, int miss_policy) {
  TORCH_CHECK(t.defined(), "Tensor must be defined");
  TORCH_CHECK(t.is_cuda(), "Tensor must be a CUDA tensor");
  TORCH_CHECK(t.numel() > 0, "Tensor must have storage");

  cudaSetDevice(t.get_device());

  // Get the current CUDA stream
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);

  cudaAccessPolicyWindow apw{};
  apw.base_ptr  = (void*)t.data_ptr();          // some device pointer
  cudaDeviceProp prop{};
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);

  size_t requested_bytes = (size_t)t.numel() * (size_t)t.element_size();
  if (prop.accessPolicyMaxWindowSize > 0) {
    requested_bytes = std::min(requested_bytes, static_cast<size_t>(prop.accessPolicyMaxWindowSize));
  }
  apw.num_bytes = requested_bytes;
  apw.hitRatio  = hit_ratio;                    // 0..1

  apw.hitProp = static_cast<cudaAccessProperty>(hit_policy);
  apw.missProp = static_cast<cudaAccessProperty>(miss_policy);

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = apw;
  auto err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  TORCH_CHECK(err == cudaSuccess, "cudaStreamSetAttribute failed: ", cudaGetErrorString(err));
}

void py_enable_peer_access(std::vector<int64_t> gpu_ids) {
  std::vector<int> devices;
  devices.reserve(gpu_ids.size());
  for (int64_t gpu_id : gpu_ids) {
    devices.push_back(static_cast<int>(gpu_id));
  }
  enable_peer_access(devices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto op = m.def_submodule("opcode", "DAE2 OpCodes");
  #define DAE_OP(name, value) op.attr(#name) = (int)name;
  #include "dae/opcode.cuh.inc"
  #undef DAE_OP

  py::list supported_compute_ops;
  #define DAE_COMPUTE_OP(name) supported_compute_ops.append(py::str(#name));
  #include "dae/selected_compute_ops.inc"
  #undef DAE_COMPUTE_OP
  m.attr("supported_compute_ops") = supported_compute_ops;

  auto config = m.def_submodule("config", "DAE2 Configuration Constants");
  config.attr("slot_size") = slotSizeKb * 1024;
  config.attr("num_slots") = numSlots;
  config.attr("max_insts") = numInsts;
  config.attr("num_profile_events") = numProfileEvents;
  config.attr("max_tmas") = numTmas;
  config.attr("max_bars") = numBars;
  config.attr("num_special_slots") = numSpecialSlots;

  // auto flag = m.def_submodule("flag", "DAE2 Instruction Flags");
  // flag.attr("jump") = MEM_OP_FLAGS_JUMP;
  // flag.attr("writeback") = MEM_OP_FLAGS_WRITEBACK;
  // flag.attr("group") = MEM_OP_FLAGS_GROUP;
  // flag.attr("barrier") = MEM_OP_FLAGS_BARRIER;
  // flag.attr("port") = MEM_OP_FLAGS_PORT;

  // auto cache = m.def_submodule("cache_policy", "DAE2 Cache Policy Constants");
  // cache.attr("normal") = DAE_CACHE_NORMAL;
  // cache.attr("streaming") = DAE_CACHE_STREAMING;
  // cache.attr("persisting") = DAE_CACHE_PERSISTING;

  m.def("set_smem_size", &py_set_smem_size,
            "Set dynamic shared memory size for DAE2 kernel");
  m.def("launch_dae", &py_launch_dae,
            "Launch DAE2 kernel with given parameters");
  m.def("build_tma_desc", &py_build_tma_desc,
            "Build CUtensorMap descriptor for given tensor and layout");
  m.def("set_cache_policy", &py_tensor_set_cache_policy,
            "Set cache policy for a CUDA tensor on the specified stream");
  m.def("enable_peer_access", &py_enable_peer_access,
            "Enable CUDA peer access for all provided GPU ids");
}
