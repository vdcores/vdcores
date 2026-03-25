#pragma once

#include "context.cuh"
#include <cuda.h>
#include <vector>

// runtime interface for DAE kernels
size_t set_smem_size(int device_id, size_t smem_size = (1024 * 212));

cudaError_t launch_dae(
  int device_id,
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  int64_t stream
);

void enable_peer_access(const std::vector<int>& device_ids);

CUtensorMap create_tma_descriptor(
  CUtensorMapDataType data_type,
  int dims,
  void * base,
  std::array<uint64_t, 5> global_dims,
  std::array<uint32_t, 5> box_dims,
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE,
  std::array<uint64_t, 5> global_strides_opt = {0, 0, 0, 0, 0}
);
