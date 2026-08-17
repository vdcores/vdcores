#pragma once

#include "context.cuh"
#include <cuda.h>

// runtime interface for DAE kernels
size_t set_smem_size(size_t smem_size = (1024 * 212));

#ifdef DAE_ENABLE_NVSHMEM
int nvshmem_module_init();
int nvshmem_module_finalize();
#endif

#ifdef DAE_ENABLE_NCCL_GIN
cudaError_t configure_pool_gin_transport(
    const void* host_dev_comm,
    uint64_t window_handle,
    uint64_t arena_base,
    uint64_t arena_bytes,
    uint32_t* context_count);
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
  const DaeCoreConfig* core_configs = nullptr,
  DaeKernelVariant kernel_variant = DAE_KERNEL_AUTO,
  uint16_t pool_inst_opcode = 0
);

CUtensorMap create_tma_descriptor(
  CUtensorMapDataType data_type,
  int dims,
  void * base,
  std::array<uint64_t, 5> global_dims,
  std::array<uint32_t, 5> box_dims,
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
  std::array<uint64_t, 5> global_strides_opt = {0, 0, 0, 0, 0}
);
