#pragma once

#ifndef DAE_ENABLE_NVSHMEM
#error "poolinst.cuh requires DAE_ENABLE_NVSHMEM"
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

struct PoolSliceExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst& inst,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    // Host assembly selection guarantees the PoolInst type.  There is no
    // opcode check, switch, or defensive fallback in the specialized CTA.
    pool_slice_exchange<false, num_warps>(
        reinterpret_cast<const PoolSliceConfig*>(inst.address),
        bars,
        signal_array,
        g_events,
        inst.size,
        inst.arg0,
        inst.arg1,
        thread_id);
  }
};

struct PoolSliceWeightedExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_WEIGHTED_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst& inst,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_exchange<true, num_warps>(
        reinterpret_cast<const PoolSliceConfig*>(inst.address),
        bars,
        signal_array,
        g_events,
        inst.size,
        inst.arg0,
        inst.arg1,
        thread_id);
  }
};

struct PoolSliceHostWeightedExchangeExecuteWarp {
  static constexpr uint16_t opcode = POOL_SLICE_HOST_WEIGHTED_EXCHANGE;
  static constexpr uint32_t num_warps = daePoolSliceWarps;
  static constexpr int max_registers = daeWideRegisterLimit;

  static __device__ __forceinline__ void execute(
      const PoolInst& inst,
      int* bars,
      uint64_t* signal_array,
      uint64_t* g_events,
      uint32_t physical_warps,
      uint32_t thread_id) {
    (void)physical_warps;
    pool_slice_host_weighted_exchange<num_warps>(
        reinterpret_cast<const PoolSliceHostConfig*>(inst.address),
        bars,
        signal_array,
        g_events,
        inst.size,
        inst.arg0,
        inst.arg1,
        thread_id);
  }
};
