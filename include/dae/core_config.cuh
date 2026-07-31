#pragma once

#include <cstdint>

// A DAE kernel has a compile-time physical envelope (the launched CTA size)
// and, for the runtime-selectable envelope, one logical configuration per
// block. CUDA requires every block in a grid to have the same blockDim, so a
// runtime configuration can disable/reassign warps but cannot reclaim their
// launch resources. The fixed variants below provide that resource reduction.
static constexpr uint8_t daeComputeWarps = 4;
static constexpr uint8_t daeDefaultLoadWarps = 2;
static constexpr uint8_t daeMaxLoadWarps = 2;
static constexpr uint8_t daeMemoryControlWarps = 2;  // allocator + store
static constexpr uint8_t daeDefaultMemoryWarps =
    daeMemoryControlWarps + daeDefaultLoadWarps;
static constexpr uint8_t daeDefaultCoreWarps =
    daeComputeWarps + daeDefaultMemoryWarps;
static constexpr int daeWideRegisterLimit = 255;
static constexpr int daeNineWarpRegisterLimit = 168;

#if defined(DAE_ENABLE_NVSHMEM) || defined(DAE_ENABLE_NCCL_GIN)
#ifdef DAE_ENABLE_NVSHMEM
static constexpr uint8_t daeRuntimeCommunicationWarps = 1;
#else
static constexpr uint8_t daeRuntimeCommunicationWarps = 0;
#endif
#ifndef DAE_POOL_SLICE_WARPS
#define DAE_POOL_SLICE_WARPS 8
#endif
#ifndef DAE_POOL_SLICE_WARP_QP_COMPLETION
#define DAE_POOL_SLICE_WARP_QP_COMPLETION 0
#endif
#ifndef DAE_POOL_SLICE_RAW_SGL
#define DAE_POOL_SLICE_RAW_SGL 0
#endif
#ifndef DAE_POOL_SLICE_RAW_SGL_WIDTH
#define DAE_POOL_SLICE_RAW_SGL_WIDTH 8
#endif
static constexpr uint8_t daePoolSliceWarps = DAE_POOL_SLICE_WARPS;
static_assert(daePoolSliceWarps >= 3 && daePoolSliceWarps <= 32);
static constexpr bool daePoolSliceWarpQpCompletion =
    DAE_POOL_SLICE_WARP_QP_COMPLETION != 0;
static_assert(
    DAE_POOL_SLICE_WARP_QP_COMPLETION == 0 ||
    DAE_POOL_SLICE_WARP_QP_COMPLETION == 1);
static constexpr uint8_t daePoolSliceCompletionSlots =
    daePoolSliceWarpQpCompletion ? daePoolSliceWarps : 1;
static constexpr bool daePoolSliceRawSgl = DAE_POOL_SLICE_RAW_SGL != 0;
static constexpr uint8_t daePoolSliceRawSglWidth =
    DAE_POOL_SLICE_RAW_SGL_WIDTH;
static_assert(DAE_POOL_SLICE_RAW_SGL == 0 || DAE_POOL_SLICE_RAW_SGL == 1);
static_assert(
    !daePoolSliceRawSgl ||
    (daePoolSliceRawSglWidth >= 1 && daePoolSliceRawSglWidth <= 30));
// The common heterogeneous pool envelope deliberately excludes the ordinary
// communication interpreter. At eight warps it can alternate per block between
// compute+memory and the currently registered PoolInst without the nine-warp
// register ceiling.
static constexpr uint8_t daeRuntimeCoreWarps =
    daeDefaultCoreWarps > daePoolSliceWarps
        ? daeDefaultCoreWarps
        : daePoolSliceWarps;
static constexpr uint8_t daeRuntimeCommunicationCoreWarps =
    daeDefaultCoreWarps + daeRuntimeCommunicationWarps;
#else
static constexpr uint8_t daeRuntimeCommunicationWarps = 0;
static constexpr uint8_t daePoolSliceWarps = 0;
static constexpr bool daePoolSliceWarpQpCompletion = false;
static constexpr uint8_t daePoolSliceCompletionSlots = 0;
static constexpr bool daePoolSliceRawSgl = false;
static constexpr uint8_t daePoolSliceRawSglWidth = 0;
static constexpr uint8_t daeRuntimeCoreWarps = daeDefaultCoreWarps;
static constexpr uint8_t daeRuntimeCommunicationCoreWarps =
    daeRuntimeCoreWarps;
#endif

enum DaeCoreKind : uint8_t {
  // Four compute warps plus allocator/store and one or two load warps. An
  // optional communication warp consumes its own CommInst stream.
  DAE_CORE_COMPUTE_MEMORY = 0,
  // Every physical warp cooperates on one statically compiled PoolInst.
  // This is deliberately separate from the ordinary CommInst interpreter.
  DAE_CORE_POOL = 1,
  // The block intentionally executes no virtual core.
  DAE_CORE_INACTIVE = 2,
};

// Stable byte ABI shared with python/dae/core.py.
struct alignas(8) DaeCoreConfig {
  uint8_t kind;
  uint8_t compute_warps;
  uint8_t load_warps;
  uint8_t communication_warps;
  uint8_t pool_warps;
  uint8_t flags;
  uint8_t reserved[2];
};
static_assert(sizeof(DaeCoreConfig) == 8, "DaeCoreConfig ABI changed");

enum DaeKernelVariant : uint32_t {
  // Null config -> default; non-null config -> runtime-selectable envelope.
  DAE_KERNEL_AUTO = 0,
  // Fixed 4 compute + allocator + store + 2 load warps (8 warps).
  DAE_KERNEL_COMPUTE_MEMORY = 1,
  // Fixed 4 compute + allocator + store + 1 load warp (7 warps).
  DAE_KERNEL_COMPUTE_MEMORY_ONE_LOAD = 2,
  // Maximum envelope with a per-block DaeCoreConfig (9 warps with NVSHMEM).
  DAE_KERNEL_RUNTIME = 3,
  // Fixed selected-PoolInst envelope; no compute/memory/CommInst VM exists.
  DAE_KERNEL_POOL = 4,
  // Maximum envelope for blocks that combine compute/memory with a CommInst
  // warp. Kept separate so the common eight-warp kernels are not register-capped.
  DAE_KERNEL_RUNTIME_COMMUNICATION = 5,
};

static __device__ __host__ __forceinline__ constexpr DaeCoreConfig
dae_compute_memory_core(uint8_t load_warps = daeDefaultLoadWarps,
                        uint8_t communication_warps = 0) {
  return DaeCoreConfig{
      DAE_CORE_COMPUTE_MEMORY,
      daeComputeWarps,
      load_warps,
      communication_warps,
      0,
      0,
      {0, 0}};
}

static __device__ __host__ __forceinline__ constexpr DaeCoreConfig
dae_pool_core(uint8_t pool_warps = 0) {
  return DaeCoreConfig{
      DAE_CORE_POOL,
      0,
      0,
      0,
      pool_warps,
      0,
      {0, 0}};
}

static __device__ __host__ __forceinline__ constexpr DaeCoreConfig
dae_inactive_core() {
  return DaeCoreConfig{DAE_CORE_INACTIVE, 0, 0, 0, 0, 0, {0, 0}};
}
