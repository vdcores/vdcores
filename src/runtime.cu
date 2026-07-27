#include "dae2.cuh"
#include "runtime.cuh"
#ifdef DAE_ENABLE_NVSHMEM
#include "runtime_comm.cuh"
#endif

#include <cuda.h>

#ifdef DAE_ENABLE_NVSHMEM
#include <nvshmemx.h>

namespace {
CUmodule nvshmem_module = nullptr;
}

int nvshmem_module_init() {
    if (nvshmem_module != nullptr) {
        return 0;
    }

    cudaFunction_t function = nullptr;
    cudaError_t runtime_status = cudaGetFuncBySymbol(
        &function,
        reinterpret_cast<const void *>(
            dae2<NoPoolInstExecuteWarp, 2, 0, false, true>)
    );
    if (runtime_status != cudaSuccess) {
        return static_cast<int>(runtime_status);
    }

    CUmodule module = nullptr;
    CUresult driver_status = cuFuncGetModule(
        &module,
        reinterpret_cast<CUfunction>(function)
    );
    if (driver_status != CUDA_SUCCESS) {
        return static_cast<int>(driver_status);
    }

    int status = nvshmemx_cumodule_init(module);
    if (status == 0) {
        nvshmem_module = module;
    }
    return status;
}

int nvshmem_module_finalize() {
    if (nvshmem_module == nullptr) {
        return 0;
    }

    int status = nvshmemx_cumodule_finalize(nvshmem_module);
    if (status == 0) {
        nvshmem_module = nullptr;
    }
    return status;
}
#endif

size_t set_smem_size(size_t smem_size) {
    cudaError_t err = cudaFuncSetAttribute(
        dae2<NoPoolInstExecuteWarp, 2, 0, false, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(
            dae2<NoPoolInstExecuteWarp, 1, 0, false, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }
    if (err == cudaSuccess) {
        err = cudaFuncSetAttribute(
            dae2<NoPoolInstExecuteWarp, 2, 0, true, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
    }
#ifdef DAE_ENABLE_NVSHMEM
    // Every PoolInst registry entry owns two concrete assemblies: a mixed
    // runtime envelope and a pool-only envelope.  Adding an instruction does
    // not modify dae2 or the default compute/memory assembly.
    #define DAE_POOL_OP(name, value, execute_warp_type) do { \
      if (err == cudaSuccess) { \
        err = cudaFuncSetAttribute( \
            dae2<execute_warp_type, 2, 0, true, true>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            smem_size); \
      } \
      if (err == cudaSuccess) { \
        err = cudaFuncSetAttribute( \
            dae2<execute_warp_type, 0, 0, false, false>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            smem_size); \
      } \
    } while (0);
    #include "dae/pool_opcode.cuh.inc"
    #undef DAE_POOL_OP
    if (err == cudaSuccess)
        err = set_runtime_communication_smem_size(smem_size);
#endif
    if (err != cudaSuccess) {
        std::cerr << "Kernel set parameter failed: " << cudaGetErrorString(err) << std::endl;
    }
    return smem_size;
}

#ifdef DAE_ENABLE_NVSHMEM
template <typename PoolInstExecuteWarp>
static cudaError_t launch_runtime_pool_inst(
    int num_sms,
    size_t smem_size,
    CInst* compute_instructions,
    MInst* memory_instructions,
    CommInst* communication_instructions,
    PoolInst* pool_instructions,
    CUtensorMap* tma_descs,
    int* bars,
    uint64_t* signal_array,
    uint64_t* profile,
    const DaeCoreConfig* core_configs,
    cudaStream_t stream) {
  dae2<PoolInstExecuteWarp, 2, 0, true, true>
      <<<num_sms,
         (daeDefaultCoreWarps > PoolInstExecuteWarp::num_warps
              ? daeDefaultCoreWarps
              : PoolInstExecuteWarp::num_warps) * numThreadsPerWarp,
         smem_size,
         stream>>>(
          compute_instructions,
          memory_instructions,
          communication_instructions,
          pool_instructions,
          tma_descs,
          bars,
          signal_array,
          profile,
          core_configs);
  return cudaGetLastError();
}

template <typename PoolInstExecuteWarp>
static cudaError_t launch_fixed_pool_inst(
    int num_sms,
    CInst* compute_instructions,
    MInst* memory_instructions,
    CommInst* communication_instructions,
    PoolInst* pool_instructions,
    CUtensorMap* tma_descs,
    int* bars,
    uint64_t* signal_array,
    uint64_t* profile,
    const DaeCoreConfig* core_configs,
    cudaStream_t stream) {
  dae2<PoolInstExecuteWarp, 0, 0, false, false>
      <<<num_sms,
         PoolInstExecuteWarp::num_warps * numThreadsPerWarp,
         0,
         stream>>>(
          compute_instructions,
          memory_instructions,
          communication_instructions,
          pool_instructions,
          tma_descs,
          bars,
          signal_array,
          profile,
          core_configs);
  return cudaGetLastError();
}

static cudaError_t launch_selected_pool_inst(
    uint16_t pool_inst_opcode,
    bool pool_only,
    int num_sms,
    size_t smem_size,
    CInst* compute_instructions,
    MInst* memory_instructions,
    CommInst* communication_instructions,
    PoolInst* pool_instructions,
    CUtensorMap* tma_descs,
    int* bars,
    uint64_t* signal_array,
    uint64_t* profile,
    const DaeCoreConfig* core_configs,
    cudaStream_t stream) {
  switch (pool_inst_opcode) {
    #define DAE_POOL_OP(name, value, execute_warp_type) \
      case name: \
        return pool_only \
            ? launch_fixed_pool_inst<execute_warp_type>( \
                  num_sms, compute_instructions, memory_instructions, \
                  communication_instructions, pool_instructions, tma_descs, \
                  bars, signal_array, profile, core_configs, stream) \
            : launch_runtime_pool_inst<execute_warp_type>( \
                  num_sms, smem_size, compute_instructions, \
                  memory_instructions, communication_instructions, \
                  pool_instructions, tma_descs, bars, signal_array, profile, \
                  core_configs, stream);
    #include "dae/pool_opcode.cuh.inc"
    #undef DAE_POOL_OP
    default:
      return cudaErrorNotSupported;
  }
}
#endif

cudaError_t launch_dae(
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CommInst* communication_instructions,
  PoolInst* pool_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * signal_array,
  uint64_t * profile,
  int64_t stream,
  const DaeCoreConfig* core_configs,
  DaeKernelVariant kernel_variant,
  uint16_t pool_inst_opcode
) {
  // wait for all pre-launch meta-data copying
  cudaDeviceSynchronize();
  cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (kernel_variant == DAE_KERNEL_AUTO) {
    kernel_variant = core_configs == nullptr
        ? DAE_KERNEL_COMPUTE_MEMORY
        : DAE_KERNEL_RUNTIME;
  }

  switch (kernel_variant) {
    case DAE_KERNEL_COMPUTE_MEMORY:
      if (pool_inst_opcode != 0)
        return cudaErrorInvalidValue;
      dae2<NoPoolInstExecuteWarp, 2, 0, false, true>
          <<<numSMs,
             daeDefaultCoreWarps * numThreadsPerWarp,
             smem_size,
             cuda_stream>>>(
              compute_instructions,
              memory_instructions,
              communication_instructions,
              pool_instructions,
              tma_descs,
              bars,
              signal_array,
              profile,
              core_configs);
      break;

    case DAE_KERNEL_COMPUTE_MEMORY_ONE_LOAD:
      if (pool_inst_opcode != 0)
        return cudaErrorInvalidValue;
      dae2<NoPoolInstExecuteWarp, 1, 0, false, true>
          <<<numSMs,
             (daeComputeWarps + daeMemoryControlWarps + 1) * numThreadsPerWarp,
             smem_size,
             cuda_stream>>>(
              compute_instructions,
              memory_instructions,
              communication_instructions,
              pool_instructions,
              tma_descs,
              bars,
              signal_array,
              profile,
              core_configs);
      break;

    case DAE_KERNEL_RUNTIME:
      if (pool_inst_opcode == 0) {
        dae2<NoPoolInstExecuteWarp, 2, 0, true, true>
            <<<numSMs,
               daeDefaultCoreWarps * numThreadsPerWarp,
               smem_size,
               cuda_stream>>>(
                compute_instructions,
                memory_instructions,
                communication_instructions,
                pool_instructions,
                tma_descs,
                bars,
                signal_array,
                profile,
                core_configs);
      } else {
#ifdef DAE_ENABLE_NVSHMEM
        const cudaError_t launch_status = launch_selected_pool_inst(
            pool_inst_opcode,
            false,
            numSMs,
            smem_size,
            compute_instructions,
            memory_instructions,
            communication_instructions,
            pool_instructions,
            tma_descs,
            bars,
            signal_array,
            profile,
            core_configs,
            cuda_stream);
        if (launch_status != cudaSuccess)
          return launch_status;
#else
        return cudaErrorNotSupported;
#endif
      }
      break;

    case DAE_KERNEL_POOL:
#ifdef DAE_ENABLE_NVSHMEM
      {
      const cudaError_t launch_status = launch_selected_pool_inst(
          pool_inst_opcode,
          true,
          numSMs,
          smem_size,
          compute_instructions,
          memory_instructions,
          communication_instructions,
          pool_instructions,
          tma_descs,
          bars,
          signal_array,
          profile,
          core_configs,
          cuda_stream);
      if (launch_status != cudaSuccess)
        return launch_status;
      break;
      }
#else
      return cudaErrorNotSupported;
#endif

    case DAE_KERNEL_RUNTIME_COMMUNICATION:
#ifdef DAE_ENABLE_NVSHMEM
      {
      if (pool_inst_opcode != 0)
        return cudaErrorInvalidValue;
      const cudaError_t launch_status = launch_dae_runtime_communication(
          numSMs,
          smem_size,
          compute_instructions,
          memory_instructions,
          communication_instructions,
          pool_instructions,
          tma_descs,
          bars,
          signal_array,
          profile,
          core_configs,
          cuda_stream);
      if (launch_status != cudaSuccess)
        return launch_status;
      break;
      }
#else
      return cudaErrorNotSupported;
#endif

    default:
      return cudaErrorInvalidValue;
  }
  // TODO(zhiyuang): check launch error here?

  cudaDeviceSynchronize();

  return cudaGetLastError();
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
