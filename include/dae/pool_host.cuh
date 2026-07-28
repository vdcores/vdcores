#pragma once

#include "pool_host_abi.h"

static __device__ __forceinline__ uint64_t pool_host_load_acquire_system(
    const uint64_t* address) {
  uint64_t value;
  asm volatile(
      "ld.acquire.sys.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(address)
      : "memory");
  return value;
}

static __device__ __forceinline__ void pool_host_store_release_system(
    uint64_t* address, uint64_t value) {
  asm volatile(
      "st.release.sys.global.u64 [%0], %1;"
      :
      : "l"(address), "l"(value)
      : "memory");
}

static __device__ __forceinline__ uint64_t pool_host_reserve_generation_warp(
    uint64_t* producer_generation, uint32_t lane) {
  uint64_t generation = 0;
  if (lane == 0) {
    generation = atomicAdd(
        reinterpret_cast<unsigned long long*>(producer_generation), 1ULL) + 1;
  }
  return __shfl_sync(0xffffffffU, generation, 0);
}

// The request owns exactly one ring slot. Indexed requests gather arbitrary
// activation rows; contiguous requests synthesize row indices in-place for a
// reduced return interval. The slot release is the only GPU/Grace handoff.
template <bool Indexed>
static __device__ __forceinline__ void pool_host_publish_request_warp(
    HostSglRingMemory* memory,
    uint64_t generation,
    const uint32_t* source_row_indices,
    uint32_t source_row_begin,
    uint32_t row_count,
    uint32_t local_lkey,
    uint32_t remote_rkey,
    uint64_t source_base,
    uint64_t source_stride,
    uint32_t row_bytes,
    uint64_t remote_data,
    uint64_t remote_signal,
    uint64_t sequence,
    uint32_t lane) {
  auto* slots = reinterpret_cast<HostSglRingSlot*>(memory->slots_address);
  HostSglRingSlot& slot =
      slots[(generation - 1) % hostSglRingCapacity];

  uint32_t reusable = generation <= hostSglRingCapacity;
  while (reusable == 0) {
    if (lane == 0) {
      reusable = pool_host_load_acquire_system(&slot.consumed_generation) >=
          generation - hostSglRingCapacity;
    }
    reusable = __shfl_sync(0xffffffffU, reusable, 0);
  }

  uint32_t* destination_indices =
      reinterpret_cast<uint32_t*>(memory->indices_address) +
      static_cast<uint64_t>((generation - 1) % hostSglRingCapacity) *
          hostSglRingMaxRows;
  for (uint32_t row = lane; row < row_count; row += 32) {
    if constexpr (Indexed) {
      destination_indices[row] = source_row_indices[source_row_begin + row];
    } else {
      destination_indices[row] = source_row_begin + row;
    }
  }
  if (lane == 0) {
    HostSglRequest& request = slot.request;
    request.local_lkey = local_lkey;
    request.remote_rkey = remote_rkey;
    request.source_base = source_base;
    request.source_stride = source_stride;
    request.row_bytes = row_bytes;
    request.row_count = row_count;
    request.remote_data = remote_data;
    request.remote_signal = remote_signal;
    request.sequence = sequence;
  }
  __syncwarp();
  if (lane == 0)
    pool_host_store_release_system(&slot.ready_generation, generation);
}

static __device__ __forceinline__ void pool_host_publish_epoch_end_thread(
    HostSglRingMemory* memory,
    uint64_t* producer_generation,
    uint64_t sequence) {
  const uint64_t generation = atomicAdd(
      reinterpret_cast<unsigned long long*>(producer_generation), 1ULL) + 1;
  auto* slots = reinterpret_cast<HostSglRingSlot*>(memory->slots_address);
  HostSglRingSlot& slot =
      slots[(generation - 1) % hostSglRingCapacity];
  if (generation > hostSglRingCapacity) {
    while (pool_host_load_acquire_system(&slot.consumed_generation) <
           generation - hostSglRingCapacity) {
    }
  }
  HostSglRequest& request = slot.request;
  request.local_lkey = 0;
  request.remote_rkey = 0;
  request.source_base = 0;
  request.source_stride = 0;
  request.row_bytes = 0;
  request.row_count = 0;
  request.remote_data = 0;
  request.remote_signal = 0;
  request.sequence = sequence;
  pool_host_store_release_system(&slot.ready_generation, generation);
}
