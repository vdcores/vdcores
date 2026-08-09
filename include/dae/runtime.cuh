#pragma once

#include "context.cuh"
#include <cuda.h>
#include <vector>

// runtime interface for DAE kernels
size_t set_smem_size(size_t smem_size = dynamicSmemBytes);

cudaError_t launch_dae(
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  LoopCounters initial_loop_counts = {},
  int64_t stream = 0,
  bool synchronize = true
);

cudaError_t launch_dae_sequence(
  int numSMs,
  size_t smem_size,
  CInst* compute_instructions,
  MInst* memory_instructions,
  CUtensorMap* tma_descs,
  int * bars,
  uint64_t * profile,
  const std::vector<LoopCounters>& initial_loop_counts,
  int64_t stream = 0,
  bool synchronize = true
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
