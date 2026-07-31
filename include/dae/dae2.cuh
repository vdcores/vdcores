#pragma once

#include "virtualcore.cuh"

#include "allocator.cuh"
#include "queue.cuh"
#include "compute_dispatch.cuh"

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <bit>

// pipeline stages
#include "pipeline/allocwarp.cuh"
#include "pipeline/ldwarp.cuh"
#include "pipeline/stwarp.cuh"
#ifdef DAE_ENABLE_NVSHMEM
#include "pipeline/commwarp.cuh"
#endif
#if defined(DAE_ENABLE_NVSHMEM) || defined(DAE_ENABLE_NCCL_GIN) || \
    defined(DAE_ENABLE_LOCAL_POOL)
#include "pipeline/poolinst.cuh"
#else
struct NoPoolInstExecuteWarp {
  static constexpr uint32_t num_warps = 0;
  static constexpr int max_registers = daeWideRegisterLimit;
};
#endif

static __device__ __forceinline__ void* align_to(void* ptr, size_t align) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned = (addr + align - 1) & ~(align - 1);
  return reinterpret_cast<void*>(aligned);
}

template <typename PoolInstExecuteWarp,
          int FixedLoadWarps,
          int FixedCommunicationWarps,
          bool RuntimeSelectable,
          bool VmEnabled>
static __global__
__maxnreg__(FixedCommunicationWarps == 0
                ? PoolInstExecuteWarp::max_registers
                : daeNineWarpRegisterLimit)
void dae2(
    const CInst* __restrict__ compute_instructions,
    const MInst* __restrict__ memory_instructions,
    const CommInst* __restrict__ communication_instructions,
    const PoolInst* __restrict__ pool_instructions,
    const CUtensorMap* __restrict__ tma_descs,
    int* __restrict__ bars,
    uint64_t* __restrict__ signal_array,
    uint64_t* __restrict__ g_events,
    const DaeCoreConfig* __restrict__ core_configs) {
  static_assert(FixedLoadWarps >= 0 && FixedLoadWarps <= daeMaxLoadWarps);
  static_assert(FixedCommunicationWarps >= 0 &&
                FixedCommunicationWarps <= 1);
  static_assert(VmEnabled || RuntimeSelectable == false);
  static_assert(PoolInstExecuteWarp::num_warps <= 32);
  static_assert(PoolInstExecuteWarp::max_registers > 0 &&
                PoolInstExecuteWarp::max_registers <= 255);

  constexpr int fixed_vm_warps =
      daeComputeWarps + daeMemoryControlWarps + FixedLoadWarps +
      FixedCommunicationWarps;
  constexpr int runtime_vm_warps =
      daeDefaultCoreWarps + FixedCommunicationWarps;
  constexpr int compiled_warps = !VmEnabled
      ? PoolInstExecuteWarp::num_warps
      : (RuntimeSelectable
             ? (runtime_vm_warps > PoolInstExecuteWarp::num_warps
                    ? runtime_vm_warps
                    : PoolInstExecuteWarp::num_warps)
             : fixed_vm_warps);
  static_assert(compiled_warps > 0);

  const int sm_id = blockIdx.x;
  const int thread_id = threadIdx.x;
  const int lane_id = thread_id % numThreadsPerWarp;

  DaeCoreConfig core;
  if constexpr (!VmEnabled) {
    core = dae_pool_core(compiled_warps);
  } else if constexpr (RuntimeSelectable) {
    core = core_configs == nullptr
        ? dae_compute_memory_core()
        : core_configs[sm_id];
  } else {
    core = dae_compute_memory_core(
        FixedLoadWarps, FixedCommunicationWarps);
  }

  __kprint(
      "[DAE2 SM %d] Kernel launched with %d threads (%d warps), core kind=%u "
      "compute=%u load=%u communication=%u\n",
      sm_id,
      blockDim.x,
      blockDim.x / numThreadsPerWarp,
      static_cast<unsigned>(core.kind),
      static_cast<unsigned>(core.compute_warps),
      static_cast<unsigned>(core.load_warps),
      static_cast<unsigned>(core.communication_warps));

  if constexpr (RuntimeSelectable) {
    const bool valid_compute_memory =
        core.kind == DAE_CORE_COMPUTE_MEMORY &&
        core.compute_warps == daeComputeWarps &&
        (core.load_warps == 1 || core.load_warps == 2) &&
        core.communication_warps <= FixedCommunicationWarps &&
        core.pool_warps == 0;
    const bool valid_pool =
        PoolInstExecuteWarp::num_warps != 0 && core.kind == DAE_CORE_POOL &&
        core.compute_warps == 0 && core.load_warps == 0 &&
        core.communication_warps == 0 &&
        (core.pool_warps == 0 || core.pool_warps == compiled_warps);
    const bool valid_inactive =
        core.kind == DAE_CORE_INACTIVE && core.compute_warps == 0 &&
        core.load_warps == 0 && core.communication_warps == 0 &&
        core.pool_warps == 0;
    if (!valid_compute_memory && !valid_pool &&
        !valid_inactive) {
      if (thread_id == 0)
        asm volatile("trap;");
      return;
    }
  }

  if (core.kind == DAE_CORE_INACTIVE) {
    if (thread_id == 0 && g_events != nullptr) {
      const uint64_t now = cuda::ptx::get_sreg_globaltimer();
      const int event_base = sm_id * numProfileEvents;
      g_events[event_base] = now;
      g_events[event_base + 1] = now;
    }
    return;
  }

#if defined(DAE_ENABLE_NVSHMEM) || defined(DAE_ENABLE_NCCL_GIN) || \
    defined(DAE_ENABLE_LOCAL_POOL)
  // PoolInst has its own compile-time execute-warp type and instruction array;
  // it is never decoded by the ordinary communication warp.
  if constexpr (PoolInstExecuteWarp::num_warps != 0) {
    if (core.kind == DAE_CORE_POOL) {
      PoolInstExecuteWarp::execute(
          pool_instructions + sm_id * numPoolInsts,
          bars,
          signal_array,
          g_events,
          compiled_warps,
          thread_id);
      return;
    }
  }
#endif

  if constexpr (!VmEnabled) {
    // A pool-only build does not instantiate the ordinary VM.
    return;
  } else {
    const CInst* __restrict__ cinsts;
    const MInst* __restrict__ minsts;
#ifdef DAE_ENABLE_NVSHMEM
    const CommInst* __restrict__ comminsts;
#endif

    if constexpr (dae2LoadInstructions) {
      __shared__ CInst smem_cinsts[numInsts];
      __shared__ MInst smem_minsts[numInsts];
#ifdef DAE_ENABLE_NVSHMEM
      __shared__ CommInst smem_comminsts[numCommInsts];
#endif

      for (int i = thread_id; i < numInsts; i += blockDim.x) {
        smem_cinsts[i] = compute_instructions[sm_id * numInsts + i];
        smem_minsts[i] = memory_instructions[sm_id * numInsts + i];
      }
#ifdef DAE_ENABLE_NVSHMEM
      if constexpr (FixedCommunicationWarps != 0) {
        for (int i = thread_id; i < numCommInsts; i += blockDim.x) {
          smem_comminsts[i] =
              communication_instructions[sm_id * numCommInsts + i];
        }
      }
#endif

      cinsts = smem_cinsts;
      minsts = smem_minsts;
#ifdef DAE_ENABLE_NVSHMEM
      comminsts = smem_comminsts;
#endif
    } else {
      cinsts = compute_instructions + sm_id * numInsts;
      minsts = memory_instructions + sm_id * numInsts;
#ifdef DAE_ENABLE_NVSHMEM
      comminsts = communication_instructions + sm_id * numCommInsts;
#endif
    }

    constexpr int numQueueElements = 32;
    __shared__ MInst st_insts[numSlots + numSpecialSlots];

    __shared__ int slot_avail;
    if (thread_id == 0)
      slot_avail = (1U << numSlots) - 1;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ cuda::barrier<cuda::thread_scope_block>
        barriers[4][numQueueElements];
    assert(numQueueElements <= blockDim.x && "Too many slots for barriers");
    if (thread_id < numQueueElements) {
      init(&barriers[0][thread_id], numThreadsM2CBarrier);
      init(&barriers[1][thread_id], numThreadsC2MBarrier);
      init(&barriers[2][thread_id], numThreadsLDBarrier);
      init(&barriers[3][thread_id], numThreadsLDBarrier);
    }

    __shared__ int m2c_data[numQueueElements];
    __shared__ int c2m_data[numQueueElements];
    __shared__ int m2ld_data[daeMaxLoadWarps][numQueueElements];

    SizeBoundedBarrierQueue<int, numQueueElements> m2c{
        .barriers = barriers[0], .data = m2c_data, .ptr = 0};
    SizeBoundedBarrierAllocQueue<numQueueElements> c2m{
        barriers[1], c2m_data, 0, &slot_avail};
    SizeBoundedBarrierQueue<int, numQueueElements> m2ld[daeMaxLoadWarps] = {
        {.barriers = barriers[2], .data = m2ld_data[0], .ptr = 0},
        {.barriers = barriers[3], .data = m2ld_data[1], .ptr = 0}};

    extern __shared__ uint8_t shared_mem[];
    void* smem_base = align_to(reinterpret_cast<void*>(shared_mem), 1024);
    __shared__ uint64_t scratch_space[32];

    if (thread_id == 0 && g_events != nullptr) {
      const int event_base = sm_id * numProfileEvents;
      g_events[event_base] = cuda::ptx::get_sreg_globaltimer();
    }

    __syncthreads();

    const int memory_warps = daeMemoryControlWarps + core.load_warps;
    const int memory_begin = core.compute_warps * numThreadsPerWarp;
    const int communication_begin =
        (core.compute_warps + memory_warps) * numThreadsPerWarp;

    if (thread_id < memory_begin) {
      CInst inst;
      uint32_t pc = 0;
      uint32_t count[numComputeLoopCounters] = {};
      bool finish = false;

      while (!finish) {
        inst = cinsts[(pc++) % numInsts];
        __cprint(
            "Executing instruction at PC %d: opcode=%04x", pc - 1, inst.opcode);
        dispatch_compute_instruction(
            sm_id,
            thread_id,
            pc,
            count,
            finish,
            inst,
            smem_base,
            scratch_space,
            st_insts,
            m2c,
            c2m,
            g_events);
      }
      __cprint("Finished execution pc=%d", pc - 1);
    } else if (thread_id < communication_begin) {
      const int memory_warp_id =
          (thread_id - memory_begin) / numThreadsPerWarp;
      if (memory_warp_id == 0) {
        if constexpr (RuntimeSelectable) {
          if (core.load_warps == 1) {
            allocwarp_execute<1>(
                lane_id,
                m2c,
                m2ld,
                minsts,
                &slot_avail,
                st_insts,
                smem_base,
                tma_descs,
                bars,
                signal_array);
          } else {
            allocwarp_execute<2>(
                lane_id,
                m2c,
                m2ld,
                minsts,
                &slot_avail,
                st_insts,
                smem_base,
                tma_descs,
                bars,
                signal_array);
          }
        } else {
          allocwarp_execute<FixedLoadWarps>(
              lane_id,
              m2c,
              m2ld,
              minsts,
              &slot_avail,
              st_insts,
              smem_base,
              tma_descs,
              bars,
              signal_array);
        }
      } else if (memory_warp_id == 1) {
        if (lane_id == 0) {
          stwarp_execute_singlethread(
              c2m, st_insts, smem_base, tma_descs, bars);
        }
      } else if (lane_id == 0) {
        const int port_id = memory_warp_id - daeMemoryControlWarps;
        ldwarp_execute_singlethread(
            m2ld[port_id], m2c, st_insts, smem_base, tma_descs, bars);
      }
    }
#ifdef DAE_ENABLE_NVSHMEM
    else if constexpr (FixedCommunicationWarps != 0) {
      if (core.communication_warps != 0 &&
          thread_id < communication_begin + numThreadsPerWarp) {
        communicationwarp_execute(
            lane_id, comminsts, bars, signal_array, g_events);
      }
    }
#endif
  }
}
