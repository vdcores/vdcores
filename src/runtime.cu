#include "dae2.cuh"
#include "runtime.cuh"

#include <cuda.h>

#if __has_include("dae/generated_compiled_runtime.cuh")
#include "dae/generated_compiled_runtime.cuh"
#define DAE_HAS_GENERATED_COMPILED_RUNTIME 1
#else
#define DAE_HAS_GENERATED_COMPILED_RUNTIME 0
#endif

size_t set_smem_size(size_t smem_size) {
    cudaError_t err = cudaFuncSetAttribute(
        dae2,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );
    if (err != cudaSuccess) {
        std::cerr << "Kernel set parameter failed: " << cudaGetErrorString(err) << std::endl;
    }
#if DAE_HAS_GENERATED_COMPILED_RUNTIME
    err = dae::compiled::set_generated_compiled_runtime_smem_size(smem_size);
    if (err != cudaSuccess) {
        std::cerr << "Compiled runtime set parameter failed: " << cudaGetErrorString(err) << std::endl;
    }
#endif
    return smem_size;
}

cudaError_t launch_dae(
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  int64_t stream
) {
  // wait for all pre-launch meta-data copying
  cudaDeviceSynchronize();
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  dae2<<<numSMs, numThreads, smem_size, cuda_stream>>>(
    compute_instructions,
    memory_instructions,
    tma_descs,
    bars,
    profile
  );
  // TODO(zhiyuang): check launch error here?

  cudaDeviceSynchronize();

  return cudaGetLastError();
}

cudaError_t launch_compiled_dae(
  int numSMs,
  size_t smem_size,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  int64_t stream
) {
#if DAE_HAS_GENERATED_COMPILED_RUNTIME
  cudaDeviceSynchronize();
  return dae::compiled::launch_generated_compiled_runtime(
    numSMs,
    smem_size,
    tma_descs,
    bars,
    profile,
    stream
  );
#else
  (void)numSMs;
  (void)smem_size;
  (void)tma_descs;
  (void)bars;
  (void)profile;
  (void)stream;
  return cudaErrorNotSupported;
#endif
}

const char *compiled_runtime_tag() {
#if DAE_HAS_GENERATED_COMPILED_RUNTIME
  return dae::compiled::kCompiledProgramTag;
#else
  return "";
#endif
}

CUtensorMap create_tma_descriptor(
  CUtensorMapDataType data_type,
  int dims,
  void * base,
  std::array<uint64_t, 5> global_dims,
  std::array<uint32_t, 5> box_dims,
  CUtensorMapSwizzle swizzle,
  std::array<uint64_t, 5> global_strides_opt
) {
  assert(dims <= 5 && "Maximum 5 dimensions supported");

  CUtensorMap desc;

  int element_size = -1; // default to BF16

  if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT8) {
    element_size = 1;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT16 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_BFLOAT16) {
    element_size = 2;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT32 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_INT32) {
    element_size = 4;
  } else if (data_type == CU_TENSOR_MAP_DATA_TYPE_UINT64 ||
             data_type == CU_TENSOR_MAP_DATA_TYPE_INT64) {
    element_size = 8;
  }
  assert(element_size > 0 && "Unsupported data type");

  uint64_t global_strides[5];
  uint32_t box_strides[5];

  // Calculate global strides using cumulative products
  global_strides[0] = global_dims[0] * element_size;
  for (int i = 1; i < dims - 1; i++) {
    global_strides[i] = global_strides[i-1] * global_dims[i];
  }

  // Box strides are always 1 (contiguous within each tile)
  for (int i = 0; i < dims; i++) {
    box_strides[i] = 1;
  }

  auto result = cuTensorMapEncodeTiled(
    &desc,
    data_type,
    dims,
    base,
    global_dims.data(),
    // we go with a compact layout if no strides are provided
    global_strides_opt[0] == 0 ? global_strides : global_strides_opt.data(),
    box_dims.data(),
    box_strides,

    CU_TENSOR_MAP_INTERLEAVE_NONE,    // No interleaving
    swizzle,       // Swizzle mode
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B,  // No L2 promotion
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE // No special OOB handling
  );
  assert(result == CUDA_SUCCESS && "Failed to create tensor map");
  
  return desc;
}
