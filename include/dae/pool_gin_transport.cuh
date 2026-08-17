#pragma once

#ifndef DAE_ENABLE_NCCL_GIN
#error "pool_gin_transport.cuh requires DAE_ENABLE_NCCL_GIN"
#endif

#include <nccl_device.h>

#include <cuda/atomic>
#include <cuda/std/cstdint>

// One NCCL device communicator and one registered HBM window back the complete
// process-local pool slice.  Host setup installs this state once; PoolInst
// instructions retain the transport-independent 192-byte PoolSliceConfig ABI.
// All addresses passed by the protocol are local VAs inside `arena_base` and
// are translated to window offsets at the transport boundary.
struct alignas(16) PoolGinTransportState {
  ncclDevComm dev_comm;
  ncclWindow_t window;
  uint64_t arena_base;
  uint64_t arena_bytes;
};

static __device__ __constant__ PoolGinTransportState
    dae_pool_gin_transport_state;

static __device__ __forceinline__ size_t pool_gin_offset(
    const void* address) {
  return reinterpret_cast<uint64_t>(address) -
      dae_pool_gin_transport_state.arena_base;
}

static __device__ __forceinline__ uint32_t pool_gin_context(
    uint32_t salt = 0) {
  const uint32_t count =
      dae_pool_gin_transport_state.dev_comm.ginContextCount;
  if (count == 1)
    return 0;
  // Keep data and metadata on disjoint QPs.  Shared QPs let long activation
  // WQEs delay the route envelope which makes those activations consumable.
  const uint32_t metadata_count = count >= 4 ? count / 4 : 1;
  const uint32_t data_count = count - metadata_count;
  if (salt == 1) {
    return data_count +
        static_cast<uint32_t>(blockIdx.x) % metadata_count;
  }
  return static_cast<uint32_t>(blockIdx.x) % data_count;
}

// Contexts can be shared by independently scheduled PoolInst CTAs.  GPU mode
// gives GDAKI the exact reservation scope it needs; context-count tuning still
// spreads requests over multiple QPs without relying on a block/QP bijection.
// Use the public ncclGin facade. Host setup requires GDAKI, so its runtime
// backend discriminator is stable; using the facade also keeps VA-signal
// descriptor construction on NCCL's supported API surface.
using PoolGin = ncclGin;

static __device__ __forceinline__ PoolGin pool_gin(
    uint32_t salt = 0) {
  return PoolGin{
      dae_pool_gin_transport_state.dev_comm,
      static_cast<int>(pool_gin_context(salt)),
      NCCL_GIN_RESOURCE_SHARING_GPU};
}

static __device__ __forceinline__ void pool_gin_put_warp(
    void* destination,
    const void* source,
    size_t bytes,
    int target_pe,
    bool aggregate,
    uint32_t salt = 0) {
  if (bytes == 0)
    return;
  const uint32_t flags = aggregate
      ? static_cast<uint32_t>(ncclGinOptFlagsAggregateRequests)
      : static_cast<uint32_t>(ncclGinOptFlagsDefault);
  pool_gin(salt).put(
      ncclTeamWorld(dae_pool_gin_transport_state.dev_comm),
      target_pe,
      dae_pool_gin_transport_state.window,
      pool_gin_offset(destination),
      dae_pool_gin_transport_state.window,
      pool_gin_offset(source),
      bytes,
      ncclGin_None{},
      ncclGin_None{},
      ncclCoopWarp{},
      ncclGin_None{},
      cuda::thread_scope_device,
      cuda::thread_scope_device,
      flags);
}

// Exact-generation writes are used for one-producer readiness slots.  When
// earlier puts use AggregateRequests on this same context, this final request
// publishes their WQEs and rings the QP once. RC ordering makes the generation
// visible only after the preceding payload writes to that peer.
static __device__ __forceinline__ void pool_gin_set_thread(
    uint64_t* destination,
    uint64_t value,
    int target_pe,
    uint32_t salt = 0) {
  pool_gin(salt).putValue<uint64_t>(
      ncclTeamWorld(dae_pool_gin_transport_state.dev_comm),
      target_pe,
      dae_pool_gin_transport_state.window,
      pool_gin_offset(destination),
      value,
      ncclGin_None{},
      ncclCoopThread{},
      ncclGin_None{},
      cuda::thread_scope_device,
      cuda::thread_scope_device,
      ncclGinOptFlagsDefault);
}

// Metadata has a many-message monotonic counter rather than a one-producer
// generation. A strong VA add couples the packet and its precise dependency
// word in one GIN operation; it does not order unrelated contexts or messages.
static __device__ __forceinline__ void pool_gin_put_add_signal_warp(
    void* destination,
    const void* source,
    size_t bytes,
    uint64_t* signal,
    uint64_t delta,
    int target_pe,
    uint32_t salt = 1) {
  pool_gin(salt).put(
      ncclTeamWorld(dae_pool_gin_transport_state.dev_comm),
      target_pe,
      dae_pool_gin_transport_state.window,
      pool_gin_offset(destination),
      dae_pool_gin_transport_state.window,
      pool_gin_offset(source),
      bytes,
      ncclGin_StrongVASignalAdd{
          dae_pool_gin_transport_state.window,
          pool_gin_offset(signal),
          delta},
      ncclGin_None{},
      ncclCoopWarp{},
      ncclGin_None{},
      cuda::thread_scope_device,
      cuda::thread_scope_device,
      ncclGinOptFlagsDefault);
}

static __device__ __forceinline__ void pool_gin_flush_block() {
  pool_gin().flush(ncclCoopCta{}, cuda::memory_order_acquire);
  if (dae_pool_gin_transport_state.dev_comm.ginContextCount > 1)
    pool_gin(1).flush(ncclCoopCta{}, cuda::memory_order_acquire);
}
