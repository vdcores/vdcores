#pragma once

#include "context.cuh"

#include <cuda_runtime.h>

#ifdef DAE_ENABLE_NVSHMEM
cudaError_t set_runtime_communication_smem_size(size_t smem_size);

cudaError_t launch_dae_runtime_communication(
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
    cudaStream_t stream);
#endif

