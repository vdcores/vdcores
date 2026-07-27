#include "dae2.cuh"
#include "runtime_comm.cuh"

#ifndef DAE_ENABLE_NVSHMEM
#error "runtime_comm.cu is an NVSHMEM-only specialized assembly"
#endif

cudaError_t set_runtime_communication_smem_size(size_t smem_size) {
  return cudaFuncSetAttribute(
      dae2<NoPoolInstExecuteWarp, 2, 1, true, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);
}

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
    cudaStream_t stream) {
  dae2<NoPoolInstExecuteWarp, 2, 1, true, true>
      <<<num_sms,
         daeRuntimeCommunicationCoreWarps * numThreadsPerWarp,
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
