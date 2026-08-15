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

#if defined(DAE_FFN_SPECIALIZED_KERNELS)
cudaError_t launch_dae_ffn_linear1_direct(
  int num_blocks, size_t smem_size, const uint8_t *metadata,
  CUtensorMap *tma_descs, int *bars, int reduction_bar_base,
  int reduction_tiles, uint64_t *profile, int64_t stream = 0);

cudaError_t launch_dae_ffn_down_direct(
  int num_blocks, size_t smem_size, const uint8_t *metadata,
  CUtensorMap *tma_descs, int *bars, uint64_t *profile, int64_t stream = 0);

#endif

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
