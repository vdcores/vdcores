#pragma once

#if !defined(DAE_ENABLE_NVSHMEM) && !defined(DAE_ENABLE_NCCL_GIN) && \
    !defined(DAE_ENABLE_LOCAL_POOL)
#error "poolinst.cuh requires a pool transport"
#endif
#if (defined(DAE_ENABLE_NVSHMEM) + defined(DAE_ENABLE_NCCL_GIN) + \
     defined(DAE_ENABLE_LOCAL_POOL)) > 1
#error "PoolInst transport backends are compile-time exclusive"
#endif

#include "context.cuh"
#include "pool_slice.cuh"

// PoolInst execution is assembled by type, not decoded by the ordinary
// communication interpreter.  Despite the name "execute warp", an executor
// may own the entire multi-warp CTA.  A new PoolInst adds one executor with
// this interface and one entry in pool_opcode.cuh.inc.
struct NoPoolInstExecuteWarp {
  static constexpr uint32_t num_warps = 0;
  static constexpr int max_registers = daeWideRegisterLimit;
};

#if defined(DAE_ENABLE_NVSHMEM) || defined(DAE_ENABLE_LOCAL_POOL)
struct PoolSliceExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst* instructions,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    // Host assembly selection guarantees the PoolInst type.  There is no
    // opcode check, switch, or defensive fallback in the specialized CTA.
    pool_slice_exchange<false, num_warps>(
        instructions,
        bars,
        signal_array,
        g_events,
        thread_id);
  }
};

struct PoolSliceWeightedExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_WEIGHTED_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst* instructions,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_exchange<true, num_warps>(
        instructions,
        bars,
        signal_array,
        g_events,
        thread_id);
  }
};

struct PoolSliceHostWeightedExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_HOST_WEIGHTED_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst* instructions,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_host_weighted_exchange<num_warps>(
        instructions,
        bars,
        signal_array,
        g_events,
        thread_id);
  }
};

#ifdef DAE_ENABLE_LOCAL_POOL
// The scheduler, metadata protocol, and DynamicRead command stream are shared
// with weighted forwarding. Only ReduceAdd's worker implementation and finish
// boundary are specialized, so the hot worker contains no backend branch.
struct PoolSliceMultimemExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_MULTIMEM_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst* instructions,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_exchange<true, num_warps, true>(
        instructions,
        bars,
        signal_array,
        g_events,
        thread_id);
  }
};

// Arbitrary top-k routing may place a token's expert outputs on several GPUs.
// Destinations therefore publish readiness only; the source GPU follows its
// token-major route table through CUDA Fabric and performs the one final sum.
struct PoolSliceSourceGatherExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_SOURCE_GATHER_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst* instructions,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_exchange<true, num_warps, false, false, true, 4>(
        instructions,
        bars,
        signal_array,
        g_events,
        thread_id);
  }
};

// Hybrid assembly: this sole PoolInst CTA performs initialization, metadata
// acceptance, readiness publication, and ring scheduling only. Payload,
// route, and return work executes through the ordinary cooperative CInst.
struct PoolSliceSourceGatherSchedulerExecuteWarp {
  static constexpr uint16_t opcode =
      POOL_SLICE_SOURCE_GATHER_SCHEDULER;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst* instructions,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_exchange<
        true,
        num_warps,
        false,
        false,
        true,
        4,
        false,
        true>(
        instructions,
        bars,
        signal_array,
        g_events,
        thread_id);
  }
};
#endif
#endif

#ifdef DAE_ENABLE_NCCL_GIN
struct PoolSliceGinWeightedExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_GIN_WEIGHTED_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst* instructions,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_exchange<true, num_warps>(
        instructions,
        bars,
        signal_array,
        g_events,
        thread_id);
  }
};
#endif
