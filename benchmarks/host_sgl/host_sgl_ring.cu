#include "host_sgl_ring_abi.h"

#include <cuda_runtime_api.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

__device__ __forceinline__ uint64_t load_acquire_system(
    const uint64_t* address) {
  uint64_t value;
  asm volatile(
      "ld.acquire.sys.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(address)
      : "memory");
  return value;
}

__device__ __forceinline__ void store_release_system(
    uint64_t* address, uint64_t value) {
  asm volatile(
      "st.release.sys.global.u64 [%0], %1;"
      :
      : "l"(address), "l"(value)
      : "memory");
}

__device__ __forceinline__ void publish_ring_request(
    HostSglRingSlot* slots,
    uint32_t* indices,
    uint64_t generation,
    uint32_t message,
    const uint32_t* source_row_indices,
    HostSglRouteGroup group,
    uint32_t local_lkey,
    uint32_t remote_rkey,
    uint64_t source_base,
    uint64_t source_stride,
    uint32_t row_bytes,
    uint64_t remote_data_base,
    uint64_t remote_signal_base,
    uint64_t remote_signal_stride) {
  const uint32_t lane = threadIdx.x & 31U;
  HostSglRingSlot& slot =
      slots[(generation - 1) % hostSglRingCapacity];

  uint32_t reusable = generation <= hostSglRingCapacity;
  while (reusable == 0) {
    if (lane == 0) {
      reusable = load_acquire_system(&slot.consumed_generation) >=
          generation - hostSglRingCapacity;
    }
    reusable = __shfl_sync(0xffffffffU, reusable, 0);
  }

  uint32_t* destination_indices = indices +
      static_cast<uint64_t>((generation - 1) % hostSglRingCapacity) *
          hostSglRingMaxRows;
  for (uint32_t row = lane; row < group.row_count; row += 32) {
    destination_indices[row] = source_row_indices[
        static_cast<uint64_t>(group.source_row_begin) + row];
  }
  if (lane == 0) {
    HostSglRequest& request = slot.request;
    request.local_lkey = local_lkey;
    request.remote_rkey = remote_rkey;
    request.source_base = source_base;
    request.source_stride = source_stride;
    request.row_bytes = row_bytes;
    request.row_count = group.row_count;
    request.remote_data =
        remote_data_base +
        static_cast<uint64_t>(group.remote_row_begin) * row_bytes;
    request.remote_signal = remote_signal_base +
        static_cast<uint64_t>(message) * remote_signal_stride;
    request.sequence = generation;
  }
  __syncwarp();
  if (lane == 0)
    store_release_system(&slot.ready_generation, generation);
}

__global__ void publish_ring_requests(
    HostSglRingSlot* slots,
    uint32_t* indices,
    uint64_t first_generation,
    uint32_t message_count,
    const uint32_t* source_row_indices,
    const HostSglRouteGroup* route_groups,
    uint32_t local_lkey,
    uint32_t remote_rkey,
    uint64_t source_base,
    uint64_t source_stride,
    uint32_t row_bytes,
    uint64_t remote_data_base,
    uint64_t remote_signal_base,
    uint64_t remote_signal_stride) {
  const uint32_t message = blockIdx.x;
  if (message >= message_count)
    return;
  const uint64_t generation = first_generation + message;
  publish_ring_request(
      slots,
      indices,
      generation,
      message,
      source_row_indices,
      route_groups[message],
      local_lkey,
      remote_rkey,
      source_base,
      source_stride,
      row_bytes,
      remote_data_base,
      remote_signal_base,
      remote_signal_stride);
}

// One communication-specialized CTA publishes every peer ring. Route groups
// already name their peer and compact destination offset, so device work is a
// flat warp-strided loop with no per-QP kernel or stream.
__global__ void publish_ring_group_resident(
    const HostSglPeerRoute* peers,
    const uint32_t* source_row_indices,
    const HostSglRouteGroup* route_groups,
    uint32_t group_count,
    uint64_t first_generation,
    uint32_t round_count,
    uint32_t local_lkey,
    uint64_t source_base,
    uint64_t source_stride,
    uint32_t row_bytes,
    uint64_t remote_signal_stride) {
  const uint32_t warp = threadIdx.x >> 5;
  constexpr uint32_t resident_warps = 8;
  for (uint32_t round = 0; round < round_count; ++round) {
    for (uint32_t task = warp; task < group_count; task += resident_warps) {
      const HostSglRouteGroup group = route_groups[task];
      const HostSglPeerRoute peer = peers[group.peer_index];
      const uint32_t message = task - static_cast<uint32_t>(peer.group_begin);
      auto* memory = reinterpret_cast<HostSglRingMemory*>(peer.ring_memory);
      const uint64_t generation = first_generation +
          static_cast<uint64_t>(round) * peer.group_count + message;
      publish_ring_request(
          reinterpret_cast<HostSglRingSlot*>(memory->slots_address),
          reinterpret_cast<uint32_t*>(memory->indices_address),
          generation,
          message,
          source_row_indices,
          group,
          local_lkey,
          static_cast<uint32_t>(peer.remote_rkey),
          source_base,
          source_stride,
          row_bytes,
          peer.remote_data_base,
          peer.remote_signal_base,
          remote_signal_stride);
    }
  }
}

void set_cuda_error(char* error, size_t error_bytes, const char* format, ...) {
  if (error == nullptr || error_bytes == 0)
    return;
  va_list arguments;
  va_start(arguments, format);
  std::vsnprintf(error, error_bytes, format, arguments);
  va_end(arguments);
  error[error_bytes - 1] = '\0';
}

}  // namespace

extern "C" int host_sgl_publish_ring_cuda(
    void* memory_pointer,
    uint64_t first_generation,
    uint32_t message_count,
    const uint32_t* source_row_indices,
    const HostSglRouteGroup* route_groups,
    uint32_t local_lkey,
    uint32_t remote_rkey,
    uint64_t source_base,
    uint64_t source_stride,
    uint32_t row_bytes,
    uint64_t remote_data_base,
    uint64_t remote_signal_base,
    uint64_t remote_signal_stride,
    void* stream_pointer,
    char* error,
    size_t error_bytes) {
  auto* memory = static_cast<HostSglRingMemory*>(memory_pointer);
  if (memory == nullptr || first_generation == 0 || message_count == 0 ||
      message_count > hostSglRingMaxMessages ||
      source_row_indices == nullptr || route_groups == nullptr ||
      source_stride < row_bytes ||
      row_bytes == 0 || source_base == 0 || remote_data_base == 0 ||
      remote_signal_base == 0) {
    set_cuda_error(error, error_bytes, "invalid coherent-ring publication");
    return -1;
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  publish_ring_requests<<<message_count, 32, 0, stream>>>(
      reinterpret_cast<HostSglRingSlot*>(memory->slots_address),
      reinterpret_cast<uint32_t*>(memory->indices_address),
      first_generation,
      message_count,
      source_row_indices,
      route_groups,
      local_lkey,
      remote_rkey,
      source_base,
      source_stride,
      row_bytes,
      remote_data_base,
      remote_signal_base,
      remote_signal_stride);
  const cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    set_cuda_error(error, error_bytes, "publish_ring_requests: %s",
                   cudaGetErrorString(result));
    return static_cast<int>(result);
  }
  return 0;
}

extern "C" int host_sgl_publish_ring_group_resident_cuda(
    const HostSglPeerRoute* peers,
    uint32_t peer_count,
    const uint32_t* source_row_indices,
    const HostSglRouteGroup* route_groups,
    uint32_t group_count,
    uint64_t first_generation,
    uint32_t round_count,
    uint32_t local_lkey,
    uint64_t source_base,
    uint64_t source_stride,
    uint32_t row_bytes,
    uint64_t remote_signal_stride,
    void* stream_pointer,
    char* error,
    size_t error_bytes) {
  if (peers == nullptr || peer_count == 0 || peer_count > 32 ||
      source_row_indices == nullptr || route_groups == nullptr ||
      group_count == 0 || first_generation == 0 || round_count == 0 ||
      source_stride < row_bytes || row_bytes == 0 || source_base == 0 ||
      remote_signal_stride == 0) {
    set_cuda_error(
        error, error_bytes, "invalid resident coherent-ring group publication");
    return -1;
  }
  auto stream = reinterpret_cast<cudaStream_t>(stream_pointer);
  publish_ring_group_resident<<<1, 256, 0, stream>>>(
      peers,
      source_row_indices,
      route_groups,
      group_count,
      first_generation,
      round_count,
      local_lkey,
      source_base,
      source_stride,
      row_bytes,
      remote_signal_stride);
  const cudaError_t result = cudaPeekAtLastError();
  if (result != cudaSuccess) {
    set_cuda_error(error, error_bytes, "publish_ring_group_resident: %s",
                   cudaGetErrorString(result));
    return static_cast<int>(result);
  }
  return 0;
}
