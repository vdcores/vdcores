#include "dae/runtime.cuh"
#include "dae/context.cuh"

#include <torch/extension.h>

#include <cuda.h>            // Driver API
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <cstdint>
#include <string>
#include <vector>

namespace py = pybind11;

// function 1: set smem size
size_t py_set_smem_size(size_t requested_size) {
  return set_smem_size(requested_size);
}

#ifdef DAE_ENABLE_NCCL_GIN
uint32_t py_configure_pool_gin_transport(
    uint64_t host_dev_comm,
    uint64_t window_handle,
    uint64_t arena_base,
    uint64_t arena_bytes) {
  uint32_t context_count = 0;
  const cudaError_t status = configure_pool_gin_transport(
      reinterpret_cast<const void*>(host_dev_comm),
      window_handle,
      arena_base,
      arena_bytes,
      &context_count);
  TORCH_CHECK(
      status == cudaSuccess,
      "configure_pool_gin_transport failed: ",
      cudaGetErrorString(status));
  return context_count;
}
#endif

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

static inline uint64_t* check_signal_array_ptr(
    const torch::Tensor& signal_array,
    const char* name) {
  TORCH_CHECK(signal_array.defined(), name, " must be defined");
  TORCH_CHECK(signal_array.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(
      signal_array.scalar_type() == torch::kUInt64,
      name,
      " must have dtype uint64");
  TORCH_CHECK(signal_array.dim() == 1, name, " must be rank-1");
  TORCH_CHECK(signal_array.is_contiguous(), name, " must be contiguous");
  return signal_array.data_ptr<uint64_t>();
}

static cudaDeviceProp current_device_prop() {
  cudaDeviceProp prop{};
  int dev = 0;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop, dev);
  return prop;
}

static std::optional<size_t> env_size_t(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return std::nullopt;
  }
  char* end = nullptr;
  unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (end == raw || (end != nullptr && *end != '\0')) {
    return std::nullopt;
  }
  return static_cast<size_t>(parsed);
}

static std::optional<double> env_double(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return std::nullopt;
  }
  char* end = nullptr;
  double parsed = std::strtod(raw, &end);
  if (end == raw || (end != nullptr && *end != '\0')) {
    return std::nullopt;
  }
  return parsed;
}

static size_t select_persisting_l2_size(const cudaDeviceProp& prop) {
  const size_t max_size = static_cast<size_t>(prop.persistingL2CacheMaxSize);
  if (max_size == 0) {
    return 0;
  }
  if (auto requested_bytes = env_size_t("DAE_PERSISTING_L2_BYTES")) {
    return std::min(*requested_bytes, max_size);
  }

  const double requested_fraction = env_double("DAE_PERSISTING_L2_FRACTION").value_or(0.0625);
  const double clamped_fraction = std::clamp(requested_fraction, 0.0, 1.0);
  return std::min(static_cast<size_t>(clamped_fraction * max_size), max_size);
}

static CUtensorMapL2promotion select_tma_l2_promotion() {
  const char* raw = std::getenv("DAE_TMA_L2_PROMOTION");
  if (raw == nullptr || raw[0] == '\0') {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }

  const std::string value(raw);
  if (value == "0" || value == "none") {
    return CU_TENSOR_MAP_L2_PROMOTION_NONE;
  }
  if (value == "64" || value == "64b" || value == "l2_64b") {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
  }
  if (value == "128" || value == "128b" || value == "l2_128b") {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  }
  if (value == "256" || value == "256b" || value == "l2_256b") {
    return CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }
  TORCH_CHECK(false, "Unsupported DAE_TMA_L2_PROMOTION=", value, " (expected none/64/128/256)");
}

static void set_persistent_cache() {
  const cudaDeviceProp prop = current_device_prop();

  // printf("L2 size: %d bytes\n", prop.l2CacheSize);
  // printf("persistingL2CacheMaxSize: %zu bytes\n", prop.persistingL2CacheMaxSize);
  // printf("accessPolicyMaxWindowSize: %zu bytes\n", prop.accessPolicyMaxWindowSize);

  const size_t set_aside = select_persisting_l2_size(prop);
  cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, set_aside);
  // printf("persistentCacheSize: %zu bytes\n", set_aside);
}

// function 2: launch_dae
int py_launch_dae(
    int64_t num_sms,
    size_t smem_size,
    torch::Tensor compute_insts_bytes,   // uint8 buffer
    torch::Tensor memory_insts_bytes,    // uint8 buffer
    torch::Tensor communication_insts_bytes,
    torch::Tensor pool_insts_bytes,
    torch::Tensor tma_descs_bytes,       // uint8 buffer
    torch::Tensor bars_int32,            // int32
    torch::Tensor profile_u64,           // uint64
    int64_t stream,
    std::optional<torch::Tensor> signal_array_u64,
    std::optional<torch::Tensor> core_configs_bytes,
    int64_t kernel_variant,
    int64_t pool_inst_opcode
) {
  set_persistent_cache();

  // fixed for H100 for now
  TORCH_CHECK(num_sms >= 0 && num_sms <= 132, "num_sms out of range");

  // Make sure we run on the right device/stream
  auto cinst = check_tensor_ptr<CInst>(compute_insts_bytes, "compute_insts_bytes");
  auto minst = check_tensor_ptr<MInst>(memory_insts_bytes, "memory_insts_bytes");
  auto comminst = check_tensor_ptr<CommInst>(
      communication_insts_bytes, "communication_insts_bytes");
  auto poolinst = check_tensor_ptr<PoolInst>(
      pool_insts_bytes, "pool_insts_bytes");
  auto tma = check_tensor_ptr<CUtensorMap>(tma_descs_bytes, "tma_descs_bytes");
  auto bars = check_tensor_ptr<int>(bars_int32, "bars_int32");
  auto prof = check_tensor_ptr<uint64_t>(profile_u64, "profile_u64");
  uint64_t* signal_array = nullptr;
  if (signal_array_u64) {
    signal_array = check_signal_array_ptr(*signal_array_u64, "signal_array_u64");
  }
  const DaeCoreConfig* core_configs = nullptr;
  if (core_configs_bytes) {
    core_configs = check_tensor_ptr<DaeCoreConfig>(
        *core_configs_bytes, "core_configs_bytes");
    TORCH_CHECK(
        core_configs_bytes->size(0) == num_sms,
        "core_configs_bytes must contain one record per launched block");
  }
  TORCH_CHECK(
      kernel_variant >= DAE_KERNEL_AUTO &&
          kernel_variant <= DAE_KERNEL_RUNTIME_COMMUNICATION,
      "kernel_variant is outside the DaeKernelVariant range");
  TORCH_CHECK(
      pool_inst_opcode >= 0 && pool_inst_opcode <= UINT16_MAX,
      "pool_inst_opcode must fit in uint16");

  cudaError_t st = launch_dae(
      static_cast<int>(num_sms), smem_size,
      cinst, minst, comminst, poolinst, tma,
      bars, signal_array, prof, stream,
      core_configs,
      static_cast<DaeKernelVariant>(kernel_variant),
      static_cast<uint16_t>(pool_inst_opcode)
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

  CUtensorMapL2promotion l2p = select_tma_l2_promotion();
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
void py_reset_cache_policy(int64_t stream_id) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);
  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow.base_ptr = nullptr;
  attr.accessPolicyWindow.num_bytes = 0;
  attr.accessPolicyWindow.hitRatio = 0.0f;
  attr.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  attr.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  auto err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  TORCH_CHECK(err == cudaSuccess, "cudaStreamSetAttribute reset failed: ", cudaGetErrorString(err));
}

void py_tensor_set_cache_policy(
    torch::Tensor t,
    int64_t stream_id,
    float hit_ratio,
    int hit_policy,
    int miss_policy,
    int64_t num_bytes) {
  TORCH_CHECK(t.defined(), "Tensor must be defined");
  TORCH_CHECK(t.is_cuda(), "Tensor must be a CUDA tensor");
  TORCH_CHECK(t.numel() > 0, "Tensor must have storage");

  // Get the current CUDA stream
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_id);

  cudaAccessPolicyWindow apw{};
  apw.base_ptr  = (void*)t.data_ptr();          // some device pointer
  const cudaDeviceProp prop = current_device_prop();

  const size_t tensor_bytes = (size_t)t.numel() * (size_t)t.element_size();
  size_t requested_bytes = tensor_bytes;
  if (num_bytes > 0) {
    requested_bytes = std::min(requested_bytes, static_cast<size_t>(num_bytes));
  }
  if (prop.accessPolicyMaxWindowSize > 0) {
    requested_bytes = std::min(requested_bytes, static_cast<size_t>(prop.accessPolicyMaxWindowSize));
  }
  TORCH_CHECK(requested_bytes > 0, "cache window must be non-zero");
  apw.num_bytes = requested_bytes;
  apw.hitRatio  = hit_ratio;                    // 0..1

  apw.hitProp = static_cast<cudaAccessProperty>(hit_policy);
  apw.missProp = static_cast<cudaAccessProperty>(miss_policy);

  cudaStreamAttrValue attr{};
  attr.accessPolicyWindow = apw;
  auto err = cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr);
  TORCH_CHECK(err == cudaSuccess, "cudaStreamSetAttribute failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  auto op = m.def_submodule("opcode", "DAE2 OpCodes");
  #define DAE_OP(name, value) op.attr(#name) = (int)name;
  #include "dae/opcode.cuh.inc"
  #undef DAE_OP

  py::list compute_family_specs;
  #define DAE_OP(name, value)
  #define DAE_DEFINE_COMP_FAMILY(name, ...) { \
    py::dict spec; \
    spec["family"] = py::str(#name); \
    spec["definition"] = py::str(#__VA_ARGS__); \
    compute_family_specs.append(spec); \
  }
  #include "dae/opcode.cuh.inc"
  #undef DAE_OP
  #undef DAE_DEFINE_COMP_FAMILY
  m.attr("compute_family_specs") = compute_family_specs;

  auto comm_opcode = m.def_submodule(
      "comm_opcode", "VDCores communication instruction opcodes");
  #define DAE_COMM_OP(name, value) comm_opcode.attr(#name) = py::int_(value);
  #include "dae/communication_opcode.cuh.inc"
  #undef DAE_COMM_OP

  auto pool_opcode = m.def_submodule(
      "pool_opcode", "VDCores compile-time pool instruction opcodes");
  py::dict pool_execute_warp_types;
  #define DAE_POOL_OP(name, value, execute_warp_type) \
    pool_opcode.attr(#name) = py::int_(value); \
    pool_execute_warp_types[py::int_(value)] = py::str(#execute_warp_type);
  #include "dae/pool_opcode.cuh.inc"
  #undef DAE_POOL_OP
  m.attr("pool_execute_warp_types") = pool_execute_warp_types;

  py::list supported_compute_ops;
  #define DAE_COMPUTE_OP(name) supported_compute_ops.append(py::str(#name));
  #include "dae/selected_compute_ops.inc"
  #undef DAE_COMPUTE_OP
  m.attr("supported_compute_ops") = supported_compute_ops;

  auto config = m.def_submodule("config", "DAE2 Configuration Constants");
  config.attr("slot_size") = slotSizeKb * 1024;
  config.attr("num_slots") = numSlots;
  config.attr("max_insts") = numInsts;
  config.attr("max_comm_insts") = numCommInsts;
  config.attr("max_pool_insts") = numPoolInsts;
  config.attr("num_profile_events") = numProfileEvents;
  config.attr("max_tmas") = numTmas;
  config.attr("max_bars") = numBars;
  config.attr("num_special_slots") = numSpecialSlots;
  config.attr("core_config_bytes") = sizeof(DaeCoreConfig);
  config.attr("compute_warps") = daeComputeWarps;
  config.attr("default_load_warps") = daeDefaultLoadWarps;
  config.attr("default_core_warps") = daeDefaultCoreWarps;
  config.attr("runtime_core_warps") = daeRuntimeCoreWarps;
  config.attr("runtime_communication_core_warps") =
      daeRuntimeCommunicationCoreWarps;
  config.attr("pool_slice_warps") = daePoolSliceWarps;
  config.attr("pool_slice_warp_qp_completion") =
      daePoolSliceWarpQpCompletion;
  config.attr("pool_slice_completion_slots") =
      daePoolSliceCompletionSlots;
  config.attr("pool_slice_raw_sgl") = daePoolSliceRawSgl;
  config.attr("pool_slice_raw_sgl_width") = daePoolSliceRawSglWidth;
  config.attr("kernel_auto") = static_cast<int>(DAE_KERNEL_AUTO);
  config.attr("kernel_compute_memory") =
      static_cast<int>(DAE_KERNEL_COMPUTE_MEMORY);
  config.attr("kernel_compute_memory_one_load") =
      static_cast<int>(DAE_KERNEL_COMPUTE_MEMORY_ONE_LOAD);
  config.attr("kernel_runtime") = static_cast<int>(DAE_KERNEL_RUNTIME);
  config.attr("kernel_pool") = static_cast<int>(DAE_KERNEL_POOL);
  config.attr("kernel_runtime_communication") =
      static_cast<int>(DAE_KERNEL_RUNTIME_COMMUNICATION);
#ifdef DAE_ENABLE_NVSHMEM
  config.attr("nvshmem_enabled") = true;
  m.def(
      "_nvshmem_module_init",
      &nvshmem_module_init,
      "Initialize NVSHMEM device state for the DAE CUDA module");
  m.def(
      "_nvshmem_module_finalize",
      &nvshmem_module_finalize,
      "Finalize NVSHMEM device state for the DAE CUDA module");
#else
  config.attr("nvshmem_enabled") = false;
#endif
#ifdef DAE_ENABLE_NCCL_GIN
  config.attr("nccl_gin_enabled") = true;
  m.def(
      "_configure_pool_gin_transport",
      &py_configure_pool_gin_transport,
      py::arg("host_dev_comm"),
      py::arg("window_handle"),
      py::arg("arena_base"),
      py::arg("arena_bytes"),
      "Install the process-local NCCL GIN device communicator and pool window");
#else
  config.attr("nccl_gin_enabled") = false;
#endif

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
  m.def(
      "launch_dae",
      &py_launch_dae,
      py::arg("num_sms"),
      py::arg("smem_size"),
      py::arg("compute_insts_bytes"),
      py::arg("memory_insts_bytes"),
      py::arg("communication_insts_bytes"),
      py::arg("pool_insts_bytes"),
      py::arg("tma_descs_bytes"),
      py::arg("bars_int32"),
      py::arg("profile_u64"),
      py::arg("stream"),
      py::arg("signal_array_u64") = std::nullopt,
      py::arg("core_configs_bytes") = std::nullopt,
      py::arg("kernel_variant") = static_cast<int>(DAE_KERNEL_AUTO),
      py::arg("pool_inst_opcode") = 0,
      "Launch a fixed or runtime-configurable DAE2 core assembly");
  m.def("build_tma_desc", &py_build_tma_desc,
            "Build CUtensorMap descriptor for given tensor and layout");
  m.def("reset_cache_policy", &py_reset_cache_policy,
            py::arg("stream"),
            "Clear the access-policy window on the specified CUDA stream");
  m.def("set_cache_policy", &py_tensor_set_cache_policy,
            py::arg("tensor"),
            py::arg("stream"),
            py::arg("hit_ratio"),
            py::arg("hit_policy"),
            py::arg("miss_policy"),
            py::arg("num_bytes") = -1,
            "Set cache policy for a CUDA tensor on the specified stream");
}
