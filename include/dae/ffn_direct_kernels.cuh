#pragma once

#include "context.cuh"
#include "task/mxfp4_mxfp8_gate_up_silu.cuh"
#include "task/mxfp4_mxfp8_ffn.cuh"

#include <cute/arch/tmem_allocator_sm100.hpp>

// The selected FFN family owns every movement operation inside its task body:
// direct bulk activation/weight/scale loads and direct native output stores.
// These empty queue types make that compile-time contract explicit while the
// arithmetic and TMEM schedule remain the same VCore task implementations.
struct DaeFfnDirectQueue {};

static_assert(
    mxfpGateUpDirectActivationEnabled && mxfpGateUpDirectOutputEnabled,
    "focused FFN entrypoints require direct activation and native output");

static __device__ __forceinline__ void *dae_ffn_align_to(
    void *pointer, size_t alignment) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  return reinterpret_cast<void *>(
      (address + alignment - 1) & ~(alignment - 1));
}

static __global__ __launch_bounds__(128, 1)
void dae_ffn_linear1_direct_kernel(
    const uint8_t *__restrict__ metadata,
    const CUtensorMap *__restrict__ tma_descs,
    int *__restrict__ bars,
    int reduction_bar_base,
    int reduction_tiles,
    uint64_t *__restrict__ profile) {
  const int worker = int(blockIdx.x);
  if (threadIdx.x == 0 && worker < reduction_tiles) {
    // The two focused kernels are stream ordered. Reset the down-projection
    // sense flags here so the timed graph needs no standalone barrier memcpy.
    bars[reduction_bar_base + worker] = 1;
  }
  if (threadIdx.x == 0) {
    profile[worker * numProfileEvents] = cuda::ptx::get_sreg_globaltimer();
  }

  extern __shared__ uint8_t dynamic_shared[];
  void *smem_base = dae_ffn_align_to(dynamic_shared, 1024);
  __shared__ alignas(16) uint32_t tmem_base_ptr;
  constexpr int kLinear1TmemColumns = 512;
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (threadIdx.x / numThreadsPerWarp == 0) {
    tmem_allocator.allocate(kLinear1TmemColumns, &tmem_base_ptr);
  }
  __syncthreads();

  DaeFfnDirectQueue m2c;
  DaeFfnDirectQueue c2m;
  constexpr int kLinear1ScratchBytes = 176 * 1024;
  constexpr int kAllocatorArenaBytes = numSlots * slotSizeKb * 1024;
  static_assert(
      dynamicSmemBytes >= kLinear1ScratchBytes + kAllocatorArenaBytes,
      "fused Linear-1 scratch and allocator slots must be disjoint");
  // The accepted fused task retains its hot activation/ring addresses. Slots
  // live in a separate tail arena and remain untouched by direct movement.
  // Physical range [kLinear1ScratchBytes, dynamicSmemBytes) is reserved for
  // the allocator arena; the direct task never addresses it.
  task_mxfp4_mxfp8_gate_up_silu_fixed_ring_sm100<
      512, mxfpGateUpDirectActivationTiles == 1 ? 3 : 2,
      8, false, false, false>(
      smem_base, tmem_base_ptr, nullptr, tma_descs,
      metadata + worker * 128, bars, m2c, c2m
#if defined(DAE_TRACK_MXFP_TIMELINE)
      , worker, profile
#endif
      );

  __syncthreads();
  if (threadIdx.x / numThreadsPerWarp == 0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, kLinear1TmemColumns);
  }
  if (threadIdx.x == 0) {
    profile[worker * numProfileEvents + 1] =
        cuda::ptx::get_sreg_globaltimer();
  }
}

static __global__ __launch_bounds__(256, 1)
void dae_ffn_down_direct_kernel(
    const uint8_t *__restrict__ metadata,
    const CUtensorMap *__restrict__ tma_descs,
    int *__restrict__ bars,
    uint64_t *__restrict__ profile) {
  const int worker = int(blockIdx.x);
  if (threadIdx.x == 0) {
    profile[worker * numProfileEvents] = cuda::ptx::get_sreg_globaltimer();
  }

  extern __shared__ uint8_t dynamic_shared[];
  void *smem_base = dae_ffn_align_to(dynamic_shared, 1024);
  __shared__ alignas(16) uint32_t tmem_base_ptr;
  // One resident CTA carries two independent four-warp down tasks.  This uses
  // the same eight-warp worker shape as the allocator kernel without turning
  // the second half into an LDU/STU path.  Both tasks retain separate TMEM,
  // scratch, and named-barrier ownership.
  constexpr int kDownTmemColumnsPerTask = 64;
  constexpr int kDownTmemColumns = 2 * kDownTmemColumnsPerTask;
  cute::TMEM::Allocator1Sm tmem_allocator{};
  if (threadIdx.x / numThreadsPerWarp == 0) {
    tmem_allocator.allocate(kDownTmemColumns, &tmem_base_ptr);
  }
  __syncthreads();

  DaeFfnDirectQueue m2c;
  DaeFfnDirectQueue c2m;
  constexpr int kDownScratchBytes = 80 * 1024;
  constexpr int kPairedScratchBytes = 2 * kDownScratchBytes;
  static_assert(
      dynamicSmemBytes >= kPairedScratchBytes,
      "paired down scratch exceeds the compiled shared-memory capacity");
  // The focused launch gives these two scratchpads a standalone 160-KiB
  // allocation. The generic allocator handler instead starts one identical
  // 80-KiB task scratchpad after its ordinary slot arena.
  const auto *first_metadata = metadata + worker * 128;
  const auto *second_metadata =
      metadata + (worker + int(gridDim.x)) * 128;
  if (threadIdx.x < 128) {
      task_mxfp4_mxfp8_down_fixed_ring_sm100<
          8, 2, 256, 1, kDownTmemColumnsPerTask,
          0, kDownScratchBytes, 0, false, false>(
          smem_base, tmem_base_ptr, nullptr, tma_descs,
        first_metadata, bars, m2c, c2m
#if defined(DAE_TRACK_MXFP_TIMELINE)
        , worker, profile
#endif
    );
  } else {
    if (*reinterpret_cast<const uint64_t *>(second_metadata) != 0) {
        task_mxfp4_mxfp8_down_fixed_ring_sm100<
            8, 2, 256, 2, kDownTmemColumnsPerTask,
            kDownScratchBytes, 2 * kDownScratchBytes, 128, false, false>(
            smem_base, tmem_base_ptr + kDownTmemColumnsPerTask,
            nullptr, tma_descs,
          second_metadata, bars, m2c, c2m
#if defined(DAE_TRACK_MXFP_TIMELINE)
          , worker, profile
#endif
      );
    }
  }

  __syncthreads();
  if (threadIdx.x / numThreadsPerWarp == 0) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(tmem_base_ptr, kDownTmemColumns);
  }
  if (threadIdx.x == 0) {
    profile[worker * numProfileEvents + 1] =
        cuda::ptx::get_sreg_globaltimer();
  }
}
