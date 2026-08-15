#pragma once

#include "virtualcore.cuh"

#include "allocator.cuh"
#include "queue.cuh"
#include "compute_dispatch.cuh"

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <bit>

#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

// pipeline stages
#include "pipeline/allocwarp.cuh"
#include "pipeline/ldwarp.cuh"
#include "pipeline/stwarp.cuh"

static __device__ __forceinline__ void * align_to(void *ptr, size_t align) {
  uintptr_t addr = (uintptr_t)ptr;
  uintptr_t aligned = (addr + align - 1) & ~(align - 1);
  return (void*)aligned;
}

// TODO(zhiyuang): decide this maxnreg size.
// Also with setnreg for computation and memory separately?
static __global__
void dae2(
  const CInst* __restrict__ compute_instructions,
  const MInst* __restrict__ memory_instructions,
  const CUtensorMap* __restrict__ tma_descs,
  int * __restrict__ bars,
  uint64_t *  __restrict__ g_events,
  LoopCounters initial_loop_counts
) {

  int sm_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int warp_id = (thread_id % 128) / 32;
  int lane_id = thread_id % 32;


  __kprint("[DAE2 SM %d] Kernel launched with %d threads (%d warps)\n", sm_id, blockDim.x, blockDim.x / 32);


  const CInst* __restrict__ cinsts;
  const MInst* __restrict__ minsts;

  // local datastructures
  if constexpr (dae2LoadInstructions) {
    __shared__ CInst smem_cinsts[numInsts];
    __shared__ MInst smem_minsts[numInsts];

    for (int i = thread_id; i < numInsts; i += blockDim.x) {
      smem_cinsts[i] = compute_instructions[sm_id * numInsts + i];
      smem_minsts[i] = memory_instructions[sm_id * numInsts + i];
    }

    cinsts = smem_cinsts;
    minsts = smem_minsts;
  } else {
    cinsts = compute_instructions + sm_id * numInsts;
    minsts = memory_instructions + sm_id * numInsts;
  }

  // intermidate insts
  constexpr int numQueueElements = 32;
  __shared__ MInst st_insts[numSlots + numSpecialSlots]; // we can have some special slots that don't go through the allocator, for special purposes like reduction output, argmax output, etc. these are indexed from numSlots and above.

  // allocator
  // TODO(zhiyuang): align this to lane 31 to avoid bank conflict?
  __shared__ int slot_avail;
  if (thread_id == 0)
    slot_avail = (1U << numSlots) - 1; // one bit per physical allocator slot

  // Init the queues
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> barriers[4][numQueueElements];
  __shared__ cuda::barrier<cuda::thread_scope_block> ldu_control_barrier;
  __shared__ cuda::barrier<cuda::thread_scope_block> ldu_control_publish_barrier;
  assert(numQueueElements <= blockDim.x && "Too many slots for barriers");
  if (threadIdx.x < numQueueElements) {
    init(&barriers[0][threadIdx.x], numThreadsM2CBarrier);
    init(&barriers[1][threadIdx.x], numThreadsC2MBarrier);
    init(&barriers[2][threadIdx.x], numThreadsLDBarrier);
    init(&barriers[3][threadIdx.x], numThreadsLDBarrier);
  }
  if (threadIdx.x == 0) {
    init(&ldu_control_barrier, 2);
    init(&ldu_control_publish_barrier, numThreadsPerWarp + 2);
  }

  __shared__ int m2c_data[numQueueElements];
  __shared__ int c2m_data[numQueueElements];
  __shared__ int m2ld_data[2][numQueueElements];

  SizeBoundedBarrierQueue<int, numQueueElements, dae2M2CObserverWait> m2c {
    .barriers = barriers[0], .data = m2c_data, .ptr = 0
  };
  SizeBoundedBarrierAllocQueue<numQueueElements> c2m {
    barriers[1], c2m_data, 0, &slot_avail
  };
  SizeBoundedBarrierQueue<int, numQueueElements> m2ld[2] = {
    { .barriers = barriers[2], .data = m2ld_data[0], .ptr = 0 },
    { .barriers = barriers[3], .data = m2ld_data[1], .ptr = 0 }
  };

  // init the slots
  extern __shared__ uint8_t shared_mem[];
  void * smem_base = align_to((void*)shared_mem, 1024); // align to 1KB

  // alloc a small scratch space for temporary data
  // argmax uses this
  __shared__ uint64_t scratch_space[32]; // 8-bytes aligned

  // Blackwell UMMA accumulates in tensor memory.  VDCores keeps one resident
  // block per SM, so acquire TMEM once for the lifetime of the megakernel and
  // reuse it across sequential compute tasks.
  __shared__ alignas(16) uint32_t tmem_base_ptr;
  // Barrier ownership is declared in context.cuh. The FP8/NVFP4 banks are
  // task-local planning over this resident array; no TMEM/runtime allocation
  // protocol changes between projection shapes.
  __shared__ alignas(16) uint64_t tmem_mma_barriers[tmemMmaBarrierCount];
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (thread_id / numThreadsPerWarp == 0) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &tmem_base_ptr);
  }
  if (thread_id == 0) {
    #pragma unroll
    for (int i = 0; i < tmemMmaBarrierCount; ++i) {
      cute::initialize_barrier(tmem_mma_barriers[i], 1);
    }
    // Direct scale stages begin empty. Both LDU streams observe this phase;
    // the completion warp produces each subsequent empty phase.
    #pragma unroll
    for (int i = 0; i < mxfp4Mxfp8TmaScaleBarrierCount; ++i) {
      cuda::ptx::mbarrier_arrive(
          cuda::ptx::sem_release,
          cuda::ptx::scope_cta,
          cuda::ptx::space_shared,
          tmem_mma_barriers + mxfp4Mxfp8TmaScaleBarrierBase + i);
    }
  }
#else
  if (thread_id == 0) {
    tmem_base_ptr = 0;
    #pragma unroll
    for (int i = 0; i < tmemMmaBarrierCount; ++i) {
      tmem_mma_barriers[i] = 0;
    }
  }
#endif

  if (threadIdx.x == 0) {
    int event_base = sm_id * numProfileEvents;
    g_events[event_base + 0] = cuda::ptx::get_sreg_globaltimer();
#if defined(DAE_TRACK_PROFILE)
    uint32_t physical_sm_id;
    uint64_t sm_clock_start;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(physical_sm_id));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(sm_clock_start));
    g_events[event_base + DAE_TRACK_PHYSICAL_SM_ID] = physical_sm_id;
    g_events[event_base + DAE_TRACK_SM_CLOCK_START] = sm_clock_start;
#endif
  }

  __syncthreads();

  // start memory and computation execution
  if (threadIdx.x < numComputeWarps * 32) {
    CInst inst;
    uint32_t pc = 0;
    uint32_t count[numComputeLoopCounters];
    #pragma unroll
    for (int i = 0; i < numComputeLoopCounters; ++i) {
      count[i] = initial_loop_counts.values[i];
    }
    uint32_t tmem_mma_phase = 0;
    uint32_t fp8_umma_pipeline_phase_mask = 0;
    uint32_t nvfp4_umma_pipeline_phase_mask = 0;
    bool finish = false;

    while (!finish) {
      inst = cinsts[(pc++) % numInsts];
    
      __cprint("Executing instruction at PC %d: opcode=%04x", pc - 1, inst.opcode);
      dispatch_compute_instruction(
        sm_id,
        thread_id,
        pc,
        count,
        finish,
        inst,
        smem_base,
        tmem_base_ptr,
        tmem_mma_barriers,
        tmem_mma_phase,
        fp8_umma_pipeline_phase_mask,
        nvfp4_umma_pipeline_phase_mask,
        scratch_space,
        st_insts,
        tma_descs,
        m2c,
        c2m,
        g_events
      );
      // if (blockIdx.x == 0 && threadIdx.x == 0) {
      //   printf("[COMP] after execution: pc=%d, opcode=%04x\n", pc-1, inst.opcode);
      // }
    }
    __cprint("Finished execution pc=%d", pc-1);
  } else { // memory warp group
    // TODO(zhiyuang): reduce the register usage in memory warps
    // cuda::ptx::set_max_nreg();

    // TODO(zhiyuang): change this to threadIdx.x predicates. will be faster than lane_id based?
    if (warp_id == 0) {
      allocwarp_execute(
        lane_id,
        m2c, m2ld, minsts, &slot_avail,
        st_insts, smem_base, tma_descs, bars,
        &ldu_control_publish_barrier,
        initial_loop_counts
#if defined(DAE_TRACK_PROFILE)
        , sm_id, g_events
#endif
      );
    } else if (warp_id == 1) {
      if (lane_id == 0) {
        stwarp_execute_singlethread(
          c2m, st_insts,
          smem_base, tma_descs, bars
#if defined(DAE_TRACK_PROFILE)
          , sm_id, g_events
#endif
        );
      }
    } else if (warp_id >= 2) { // LD Warps 0-1
      if (lane_id == 0) {
        int port_id = warp_id - 2;
        ldwarp_execute_singlethread(
          m2ld[port_id], m2c,
          st_insts,
          smem_base, tma_descs, bars, &ldu_control_barrier,
          &ldu_control_publish_barrier,
#if DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
          tmem_mma_barriers,
#endif
          port_id
#if defined(DAE_TRACK_PROFILE)
          , sm_id, g_events
#endif
        );
      }
    } // End of warps
  } // End of memory warp group

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  __syncthreads();
  if (thread_id / numThreadsPerWarp == 0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
#endif

  // end of megakernel
}
