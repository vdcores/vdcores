#pragma once

#include "context.cuh"
#include "scoped_atomic.cuh"

#ifdef DAE_ENABLE_NVSHMEM

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>

static constexpr uint32_t memoryPoolNoDependency = UINT32_MAX;

enum MemoryPoolRequestOpcode : uint32_t {
  MEMORY_POOL_WRITE = 1,
  MEMORY_POOL_READ = 2,
  MEMORY_POOL_SCATTER = 3,
  MEMORY_POOL_GATHER = 4,
};

enum MemoryPoolRequestFlags : uint32_t {
  MEMORY_POOL_FLAGS_NONE = 0,
  MEMORY_POOL_REDUCE_SUM_F32 = 1U << 0,
};

enum MemoryPoolStatus : uint64_t {
  MEMORY_POOL_STATUS_OK = 0,
  MEMORY_POOL_STATUS_BAD_CONFIG = 1,
  MEMORY_POOL_STATUS_BAD_OPCODE = 2,
  MEMORY_POOL_STATUS_POOL_RANGE = 3,
  MEMORY_POOL_STATUS_DEPENDENCY_RANGE = 4,
  MEMORY_POOL_STATUS_SIGNAL_RANGE = 5,
  MEMORY_POOL_STATUS_ROUTE_RANGE = 6,
  MEMORY_POOL_STATUS_SCRATCH_RANGE = 7,
  MEMORY_POOL_STATUS_REDUCE_FORMAT = 8,
  MEMORY_POOL_STATUS_SEQUENCE = 9,
};

// Python packs the same stable 128-byte ABI in python/dae/memory_pool.py.
struct alignas(16) MemoryPoolRequest {
  uint64_t sequence;
  uint64_t source_address;
  uint64_t destination_address;
  uint64_t route_address;
  uint64_t pool_offset;
  uint64_t bytes;
  uint64_t wait_value;
  uint64_t signal_delta;

  uint32_t opcode;
  uint32_t flags;
  uint32_t source_pe;
  uint32_t target_pe;
  uint32_t completion_pe;
  uint32_t completion_signal;
  uint32_t wait_slot;
  uint32_t signal_slot;
  uint32_t row_bytes;
  uint32_t row_count;
  uint32_t source_stride;
  uint32_t destination_stride;

  uint64_t user_tag;
  uint64_t reserved;
};
static_assert(sizeof(MemoryPoolRequest) == 128, "MemoryPoolRequest ABI changed");

// All pointer fields name HBM buffers. Remotely accessed buffers must be in the
// NVSHMEM symmetric heap; config itself only needs to be visible to this GPU.
struct alignas(16) MemoryPoolConfig {
  uint64_t mailboxes_address;
  uint64_t pool_data_address;
  uint64_t data_scratch_address;
  uint64_t route_scratch_address;
  uint64_t dependencies_address;
  uint64_t consumed_sequences_address;
  uint64_t control_address;
  uint64_t pool_bytes;
  uint64_t data_scratch_bytes;

  uint32_t mailbox_count;
  uint32_t dependency_count;
  uint32_t submit_signal_base;
  uint32_t signal_count;
  uint32_t route_capacity;
  uint32_t flags;
  uint32_t reserved_u32[8];
};
static_assert(sizeof(MemoryPoolConfig) == 128, "MemoryPoolConfig ABI changed");

static __device__ __forceinline__ bool memory_pool_span_fits(
    uint64_t offset, uint64_t bytes, uint64_t capacity) {
  return offset <= capacity && bytes <= capacity - offset;
}

static __device__ __forceinline__ void memory_pool_copy_local(
    void* destination, const void* source, uint64_t bytes) {
  auto* dst = reinterpret_cast<uint8_t*>(destination);
  const auto* src = reinterpret_cast<const uint8_t*>(source);
  for (uint64_t i = 0; i < bytes; ++i)
    dst[i] = src[i];
}

// Dependency slots are pool-local HBM synchronization objects. A release
// increment publishes exactly the data operation named by one request; a
// matching acquire load gates only requests that name that slot/value.
static __device__ __forceinline__ bool memory_pool_dependency_ready(
    const uint64_t* address, uint64_t expected) {
  return dae_atomic_load_acquire_gpu(address) >= expected;
}

static __device__ __forceinline__ void memory_pool_dependency_release(
    uint64_t* address, uint64_t delta) {
  dae_atomic_add_release_gpu(address, delta);
}

// Consumed sequences prevent mailbox replay; they do not publish request
// payload. Keep this bookkeeping atomic but deliberately unordered.
static __device__ __forceinline__ uint64_t memory_pool_sequence_load(
    const uint64_t* address) {
  return dae_atomic_load_relaxed_gpu(const_cast<uint64_t*>(address));
}

static __device__ __forceinline__ void memory_pool_sequence_store(
    uint64_t* address, uint64_t value) {
  dae_atomic_store_relaxed_gpu(address, value);
}

// A local completion is a same-device message and uses the completion word as
// its release/acquire object. A remote completion remains an NVSHMEM signal;
// the caller has already quieted any RMA whose delivery it acknowledges.
static __device__ __forceinline__ void memory_pool_publish_completion(
    uint64_t* signal_address,
    uint64_t sequence,
    uint32_t completion_pe) {
  if (completion_pe == static_cast<uint32_t>(nvshmem_my_pe())) {
    dae_atomic_store_release_gpu(signal_address, sequence);
    return;
  }
  nvshmemx_signal_op(
      signal_address, sequence, NVSHMEM_SIGNAL_SET, completion_pe);
  nvshmem_quiet();
}

static __device__ __forceinline__ void memory_pool_wait_completion_local(
    uint64_t* signal_address, uint64_t expected) {
  while (dae_atomic_load_acquire_gpu(signal_address) < expected)
    __nanosleep(barrierPollSleepCycles);
}

static __device__ __forceinline__ void memory_pool_get(
    void* destination, const void* source, uint64_t bytes, int source_pe) {
  if (source_pe == nvshmem_my_pe()) {
    memory_pool_copy_local(destination, source, bytes);
    return;
  }
  nvshmem_getmem_nbi(destination, source, static_cast<size_t>(bytes), source_pe);
  nvshmem_quiet();
}

static __device__ __forceinline__ void memory_pool_put(
    void* destination, const void* source, uint64_t bytes, int target_pe) {
  if (target_pe == nvshmem_my_pe()) {
    memory_pool_copy_local(destination, source, bytes);
    return;
  }
  nvshmem_putmem_nbi(destination, source, static_cast<size_t>(bytes), target_pe);
  nvshmem_quiet();
}

static __device__ __forceinline__ uint64_t memory_pool_row_offset(
    uint64_t base, uint32_t row, uint32_t stride, bool* valid) {
  if (stride != 0 && static_cast<uint64_t>(row) > (UINT64_MAX - base) / stride) {
    *valid = false;
    return 0;
  }
  *valid = true;
  return base + static_cast<uint64_t>(row) * stride;
}

static __device__ __forceinline__ MemoryPoolStatus memory_pool_execute_request(
    const MemoryPoolRequest& request,
    const MemoryPoolConfig& config) {
  auto* pool_data = reinterpret_cast<uint8_t*>(config.pool_data_address);
  auto* data_scratch = reinterpret_cast<uint8_t*>(config.data_scratch_address);
  auto* route_scratch = reinterpret_cast<uint32_t*>(config.route_scratch_address);

  if (request.opcode == MEMORY_POOL_WRITE) {
    if (!memory_pool_span_fits(request.pool_offset, request.bytes, config.pool_bytes))
      return MEMORY_POOL_STATUS_POOL_RANGE;

    void* pool_destination = pool_data + request.pool_offset;
    const void* remote_source = reinterpret_cast<const void*>(request.source_address);
    if (request.flags & MEMORY_POOL_REDUCE_SUM_F32) {
      if (request.bytes > config.data_scratch_bytes)
        return MEMORY_POOL_STATUS_SCRATCH_RANGE;
      if ((request.bytes & (sizeof(float) - 1)) != 0)
        return MEMORY_POOL_STATUS_REDUCE_FORMAT;
      memory_pool_get(data_scratch, remote_source, request.bytes, request.source_pe);
      auto* destination_f32 = reinterpret_cast<float*>(pool_destination);
      const auto* source_f32 = reinterpret_cast<const float*>(data_scratch);
      for (uint64_t i = 0; i < request.bytes / sizeof(float); ++i)
        destination_f32[i] += source_f32[i];
      return MEMORY_POOL_STATUS_OK;
    }

    memory_pool_get(pool_destination, remote_source, request.bytes, request.source_pe);
    return MEMORY_POOL_STATUS_OK;
  }

  if (request.opcode == MEMORY_POOL_READ) {
    if (!memory_pool_span_fits(request.pool_offset, request.bytes, config.pool_bytes))
      return MEMORY_POOL_STATUS_POOL_RANGE;
    memory_pool_put(
        reinterpret_cast<void*>(request.destination_address),
        pool_data + request.pool_offset,
        request.bytes,
        request.target_pe);
    return MEMORY_POOL_STATUS_OK;
  }

  if (request.opcode != MEMORY_POOL_SCATTER && request.opcode != MEMORY_POOL_GATHER)
    return MEMORY_POOL_STATUS_BAD_OPCODE;
  if (request.row_count > config.route_capacity)
    return MEMORY_POOL_STATUS_ROUTE_RANGE;
  if (request.row_count != 0 && (request.row_bytes == 0 || request.route_address == 0))
    return MEMORY_POOL_STATUS_ROUTE_RANGE;

  const int route_pe = request.opcode == MEMORY_POOL_SCATTER
      ? static_cast<int>(request.source_pe)
      : static_cast<int>(request.target_pe);
  memory_pool_get(
      route_scratch,
      reinterpret_cast<const void*>(request.route_address),
      static_cast<uint64_t>(request.row_count) * sizeof(uint32_t),
      route_pe);

  const uint32_t source_stride = request.source_stride == 0
      ? request.row_bytes
      : request.source_stride;
  const uint32_t destination_stride = request.destination_stride == 0
      ? request.row_bytes
      : request.destination_stride;

  for (uint32_t row = 0; row < request.row_count; ++row) {
    bool valid = false;
    if (request.opcode == MEMORY_POOL_SCATTER) {
      const uint64_t pool_row = memory_pool_row_offset(
          request.pool_offset, route_scratch[row], destination_stride, &valid);
      if (!valid || !memory_pool_span_fits(pool_row, request.row_bytes, config.pool_bytes))
        return MEMORY_POOL_STATUS_POOL_RANGE;
      memory_pool_get(
          pool_data + pool_row,
          reinterpret_cast<const uint8_t*>(request.source_address) +
              static_cast<uint64_t>(row) * source_stride,
          request.row_bytes,
          request.source_pe);
    } else {
      const uint64_t pool_row = memory_pool_row_offset(
          request.pool_offset, route_scratch[row], source_stride, &valid);
      if (!valid || !memory_pool_span_fits(pool_row, request.row_bytes, config.pool_bytes))
        return MEMORY_POOL_STATUS_POOL_RANGE;
      memory_pool_put(
          reinterpret_cast<uint8_t*>(request.destination_address) +
              static_cast<uint64_t>(row) * destination_stride,
          pool_data + pool_row,
          request.row_bytes,
          request.target_pe);
    }
  }
  return MEMORY_POOL_STATUS_OK;
}

static __device__ __forceinline__ void memory_pool_copy_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t lane) {
  auto* dst = reinterpret_cast<uint8_t*>(destination);
  const auto* src = reinterpret_cast<const uint8_t*>(source);
  for (uint64_t offset = lane; offset < bytes; offset += 32)
    dst[offset] = src[offset];
}

static __device__ __forceinline__ void memory_pool_get_nbi_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    int source_pe,
    uint32_t lane) {
  if (bytes == 0)
    return;
  if (source_pe == nvshmem_my_pe()) {
    memory_pool_copy_warp(destination, source, bytes, lane);
    return;
  }
  nvshmemx_getmem_nbi_warp(
      destination, source, static_cast<size_t>(bytes), source_pe);
}

static __device__ __forceinline__ void memory_pool_put_nbi_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    int target_pe,
    uint32_t lane) {
  if (bytes == 0)
    return;
  if (target_pe == nvshmem_my_pe()) {
    memory_pool_copy_warp(destination, source, bytes, lane);
    return;
  }
  nvshmemx_putmem_nbi_warp(
      destination, source, static_cast<size_t>(bytes), target_pe);
}

static __device__ __forceinline__ void memory_pool_complete_warp(
    uint32_t lane, bool issued_remote) {
  __syncwarp();
  if (lane == 0 && issued_remote)
    nvshmem_quiet();
  __syncwarp();
}

static __device__ __forceinline__ MemoryPoolStatus
memory_pool_execute_request_warp(
    const MemoryPoolRequest& request,
    const MemoryPoolConfig& config,
    uint32_t lane) {
  auto* pool_data = reinterpret_cast<uint8_t*>(config.pool_data_address);
  auto* data_scratch = reinterpret_cast<uint8_t*>(config.data_scratch_address);
  auto* route_scratch = reinterpret_cast<uint32_t*>(config.route_scratch_address);

  if (request.opcode == MEMORY_POOL_WRITE) {
    if (!memory_pool_span_fits(request.pool_offset, request.bytes, config.pool_bytes))
      return MEMORY_POOL_STATUS_POOL_RANGE;

    void* pool_destination = pool_data + request.pool_offset;
    const void* remote_source = reinterpret_cast<const void*>(request.source_address);
    if (request.flags & MEMORY_POOL_REDUCE_SUM_F32) {
      if (request.bytes > config.data_scratch_bytes)
        return MEMORY_POOL_STATUS_SCRATCH_RANGE;
      if ((request.bytes & (sizeof(float) - 1)) != 0)
        return MEMORY_POOL_STATUS_REDUCE_FORMAT;
      memory_pool_get_nbi_warp(
          data_scratch, remote_source, request.bytes, request.source_pe, lane);
      memory_pool_complete_warp(
          lane, request.source_pe != static_cast<uint32_t>(nvshmem_my_pe()));
      auto* destination_f32 = reinterpret_cast<float*>(pool_destination);
      const auto* source_f32 = reinterpret_cast<const float*>(data_scratch);
      for (uint64_t index = lane;
           index < request.bytes / sizeof(float);
           index += 32)
        destination_f32[index] += source_f32[index];
      memory_pool_complete_warp(lane, false);
      return MEMORY_POOL_STATUS_OK;
    }

    memory_pool_get_nbi_warp(
        pool_destination, remote_source, request.bytes, request.source_pe, lane);
    memory_pool_complete_warp(
        lane, request.source_pe != static_cast<uint32_t>(nvshmem_my_pe()));
    return MEMORY_POOL_STATUS_OK;
  }

  if (request.opcode == MEMORY_POOL_READ) {
    if (!memory_pool_span_fits(request.pool_offset, request.bytes, config.pool_bytes))
      return MEMORY_POOL_STATUS_POOL_RANGE;
    memory_pool_put_nbi_warp(
        reinterpret_cast<void*>(request.destination_address),
        pool_data + request.pool_offset,
        request.bytes,
        request.target_pe,
        lane);
    memory_pool_complete_warp(
        lane, request.target_pe != static_cast<uint32_t>(nvshmem_my_pe()));
    return MEMORY_POOL_STATUS_OK;
  }

  if (request.opcode != MEMORY_POOL_SCATTER && request.opcode != MEMORY_POOL_GATHER)
    return MEMORY_POOL_STATUS_BAD_OPCODE;
  if (request.row_count > config.route_capacity)
    return MEMORY_POOL_STATUS_ROUTE_RANGE;
  if (request.row_count != 0 && (request.row_bytes == 0 || request.route_address == 0))
    return MEMORY_POOL_STATUS_ROUTE_RANGE;

  const int route_pe = request.opcode == MEMORY_POOL_SCATTER
      ? static_cast<int>(request.source_pe)
      : static_cast<int>(request.target_pe);
  memory_pool_get_nbi_warp(
      route_scratch,
      reinterpret_cast<const void*>(request.route_address),
      static_cast<uint64_t>(request.row_count) * sizeof(uint32_t),
      route_pe,
      lane);
  memory_pool_complete_warp(lane, route_pe != nvshmem_my_pe());

  const uint32_t source_stride = request.source_stride == 0
      ? request.row_bytes
      : request.source_stride;
  const uint32_t destination_stride = request.destination_stride == 0
      ? request.row_bytes
      : request.destination_stride;

  for (uint32_t row = 0; row < request.row_count; ++row) {
    bool valid = false;
    if (request.opcode == MEMORY_POOL_SCATTER) {
      const uint64_t pool_row = memory_pool_row_offset(
          request.pool_offset, route_scratch[row], destination_stride, &valid);
      if (!valid || !memory_pool_span_fits(pool_row, request.row_bytes, config.pool_bytes))
        return MEMORY_POOL_STATUS_POOL_RANGE;
      memory_pool_get_nbi_warp(
          pool_data + pool_row,
          reinterpret_cast<const uint8_t*>(request.source_address) +
              static_cast<uint64_t>(row) * source_stride,
          request.row_bytes,
          request.source_pe,
          lane);
    } else {
      const uint64_t pool_row = memory_pool_row_offset(
          request.pool_offset, route_scratch[row], source_stride, &valid);
      if (!valid || !memory_pool_span_fits(pool_row, request.row_bytes, config.pool_bytes))
        return MEMORY_POOL_STATUS_POOL_RANGE;
      memory_pool_put_nbi_warp(
          reinterpret_cast<uint8_t*>(request.destination_address) +
              static_cast<uint64_t>(row) * destination_stride,
          pool_data + pool_row,
          request.row_bytes,
          request.target_pe,
          lane);
    }
  }
  const uint32_t data_pe = request.opcode == MEMORY_POOL_SCATTER
      ? request.source_pe
      : request.target_pe;
  memory_pool_complete_warp(
      lane, data_pe != static_cast<uint32_t>(nvshmem_my_pe()));
  return MEMORY_POOL_STATUS_OK;
}

static __device__ __forceinline__ void memory_pool_record_control(
    const MemoryPoolConfig& config,
    MemoryPoolStatus status,
    uint64_t completed,
    uint64_t mailbox,
    uint64_t user_tag) {
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  control[0] = status;
  control[1] = completed;
  control[2] = mailbox;
  control[3] = user_tag;
  // Control words are telemetry, not a protocol signal. Host readers observe
  // them only after kernel/stream completion.
}

static __device__ __forceinline__ void memory_pool_run_singlethread(
    const MemoryPoolConfig* config_pointer,
    uint64_t* signal_array,
    uint32_t expected_requests) {
  if (config_pointer == nullptr || signal_array == nullptr)
    return;

  const MemoryPoolConfig config = *config_pointer;
  if (config.mailboxes_address == 0 || config.pool_data_address == 0 ||
      config.dependencies_address == 0 || config.consumed_sequences_address == 0 ||
      config.control_address == 0 || config.mailbox_count == 0 ||
      config.submit_signal_base > config.signal_count ||
      config.mailbox_count > config.signal_count - config.submit_signal_base) {
    if (config.control_address != 0)
      memory_pool_record_control(config, MEMORY_POOL_STATUS_BAD_CONFIG, 0, 0, 0);
    return;
  }

  const auto* mailboxes = reinterpret_cast<const MemoryPoolRequest*>(
      config.mailboxes_address);
  auto* dependencies = reinterpret_cast<uint64_t*>(config.dependencies_address);
  auto* consumed = reinterpret_cast<uint64_t*>(config.consumed_sequences_address);

  uint64_t completed = 0;
  memory_pool_record_control(config, MEMORY_POOL_STATUS_OK, completed, 0, 0);
  while (completed < expected_requests) {
    bool made_progress = false;
    for (uint32_t mailbox = 0;
         mailbox < config.mailbox_count && completed < expected_requests;
         ++mailbox) {
      const uint64_t published = nvshmem_signal_fetch(
          signal_array + config.submit_signal_base + mailbox);
      if (published <= memory_pool_sequence_load(consumed + mailbox))
        continue;

      // put-with-signal publishes the request before the signal. Keep the
      // compiler from moving the mailbox load above the volatile signal load.
      asm volatile("" ::: "memory");
      const MemoryPoolRequest request = mailboxes[mailbox];
      if (request.sequence != published) {
        memory_pool_record_control(
            config, MEMORY_POOL_STATUS_SEQUENCE, completed, mailbox, request.user_tag);
        return;
      }
      if (request.completion_signal >= config.signal_count) {
        memory_pool_record_control(
            config, MEMORY_POOL_STATUS_SIGNAL_RANGE, completed, mailbox, request.user_tag);
        return;
      }
      if (request.wait_slot != memoryPoolNoDependency) {
        if (request.wait_slot >= config.dependency_count) {
          memory_pool_record_control(
              config, MEMORY_POOL_STATUS_DEPENDENCY_RANGE, completed, mailbox,
              request.user_tag);
          return;
        }
        if (!memory_pool_dependency_ready(
                dependencies + request.wait_slot, request.wait_value))
          continue;
      }
      if (request.signal_slot != memoryPoolNoDependency &&
          request.signal_slot >= config.dependency_count) {
        memory_pool_record_control(
            config, MEMORY_POOL_STATUS_DEPENDENCY_RANGE, completed, mailbox,
            request.user_tag);
        return;
      }

      const MemoryPoolStatus status = memory_pool_execute_request(request, config);
      if (status != MEMORY_POOL_STATUS_OK) {
        memory_pool_record_control(config, status, completed, mailbox, request.user_tag);
        return;
      }

      if (request.signal_slot != memoryPoolNoDependency)
        memory_pool_dependency_release(
            dependencies + request.signal_slot, request.signal_delta);
      memory_pool_sequence_store(consumed + mailbox, request.sequence);

      memory_pool_publish_completion(
          signal_array + request.completion_signal,
          request.sequence,
          request.completion_pe);

      ++completed;
      made_progress = true;
      memory_pool_record_control(
          config, MEMORY_POOL_STATUS_OK, completed, mailbox, request.user_tag);
    }
    if (!made_progress)
      __nanosleep(barrierPollSleepCycles);
  }
}

static __device__ __noinline__ void memory_pool_run_warp(
    const MemoryPoolConfig* config_pointer,
    uint64_t* signal_array,
    uint32_t expected_requests,
    uint32_t lane) {
  __shared__ MemoryPoolConfig shared_config;
  __shared__ MemoryPoolRequest selected_request;

  if (config_pointer == nullptr || signal_array == nullptr)
    return;
  if (lane == 0)
    shared_config = *config_pointer;
  __syncwarp();
  const MemoryPoolConfig config = shared_config;
  if (config.mailboxes_address == 0 || config.pool_data_address == 0 ||
      config.dependencies_address == 0 || config.consumed_sequences_address == 0 ||
      config.control_address == 0 || config.mailbox_count == 0 ||
      config.submit_signal_base > config.signal_count ||
      config.mailbox_count > config.signal_count - config.submit_signal_base) {
    if (lane == 0 && config.control_address != 0)
      memory_pool_record_control(config, MEMORY_POOL_STATUS_BAD_CONFIG, 0, 0, 0);
    __syncwarp();
    return;
  }

  const auto* mailboxes = reinterpret_cast<const MemoryPoolRequest*>(
      config.mailboxes_address);
  auto* dependencies = reinterpret_cast<uint64_t*>(config.dependencies_address);
  auto* consumed = reinterpret_cast<uint64_t*>(config.consumed_sequences_address);

  uint64_t completed = 0;
  if (lane == 0)
    memory_pool_record_control(config, MEMORY_POOL_STATUS_OK, completed, 0, 0);
  __syncwarp();

  while (completed < expected_requests) {
    bool made_progress = false;
    for (uint32_t base = 0;
         base < config.mailbox_count && completed < expected_requests;
         base += 32) {
      const uint32_t mailbox = base + lane;
      uint64_t published = 0;
      bool candidate = false;
      if (mailbox < config.mailbox_count) {
        published = nvshmem_signal_fetch(
            signal_array + config.submit_signal_base + mailbox);
        candidate = published > memory_pool_sequence_load(consumed + mailbox);
        if (candidate) {
          asm volatile("" ::: "memory");
          const MemoryPoolRequest* request = mailboxes + mailbox;
          if (request->sequence == published &&
              request->wait_slot != memoryPoolNoDependency &&
              request->wait_slot < config.dependency_count &&
              !memory_pool_dependency_ready(
                  dependencies + request->wait_slot, request->wait_value))
            candidate = false;
        }
      }

      uint32_t ready_mask = __ballot_sync(0xffffffffU, candidate);
      while (ready_mask != 0 && completed < expected_requests) {
        const uint32_t selected_lane = __ffs(ready_mask) - 1;
        const uint32_t selected_mailbox = __shfl_sync(
            0xffffffffU, mailbox, selected_lane);
        const uint64_t selected_sequence = __shfl_sync(
            0xffffffffU, published, selected_lane);
        if (lane == 0)
          selected_request = mailboxes[selected_mailbox];
        __syncwarp();
        const MemoryPoolRequest request = selected_request;

        MemoryPoolStatus validation = MEMORY_POOL_STATUS_OK;
        bool dependency_ready = true;
        if (lane == 0) {
          if (request.sequence != selected_sequence) {
            validation = MEMORY_POOL_STATUS_SEQUENCE;
          } else if (request.completion_signal >= config.signal_count) {
            validation = MEMORY_POOL_STATUS_SIGNAL_RANGE;
          } else if (request.wait_slot != memoryPoolNoDependency &&
                     request.wait_slot >= config.dependency_count) {
            validation = MEMORY_POOL_STATUS_DEPENDENCY_RANGE;
          } else if (request.signal_slot != memoryPoolNoDependency &&
                     request.signal_slot >= config.dependency_count) {
            validation = MEMORY_POOL_STATUS_DEPENDENCY_RANGE;
          } else if (request.wait_slot != memoryPoolNoDependency &&
                     !memory_pool_dependency_ready(
                         dependencies + request.wait_slot,
                         request.wait_value)) {
            dependency_ready = false;
          }
        }
        validation = static_cast<MemoryPoolStatus>(__shfl_sync(
            0xffffffffU, static_cast<uint32_t>(validation), 0));
        dependency_ready = __shfl_sync(
            0xffffffffU, static_cast<uint32_t>(dependency_ready), 0);

        if (validation != MEMORY_POOL_STATUS_OK) {
          if (lane == 0)
            memory_pool_record_control(
                config,
                validation,
                completed,
                selected_mailbox,
                request.user_tag);
          __syncwarp();
          return;
        }
        if (!dependency_ready) {
          ready_mask &= ~(1U << selected_lane);
          continue;
        }

        const MemoryPoolStatus status = memory_pool_execute_request_warp(
            request, config, lane);
        if (status != MEMORY_POOL_STATUS_OK) {
          if (lane == 0)
            memory_pool_record_control(
                config, status, completed, selected_mailbox, request.user_tag);
          __syncwarp();
          return;
        }

        if (lane == 0) {
          if (request.signal_slot != memoryPoolNoDependency)
            memory_pool_dependency_release(
                dependencies + request.signal_slot, request.signal_delta);
          memory_pool_sequence_store(
              consumed + selected_mailbox, request.sequence);
          memory_pool_publish_completion(
              signal_array + request.completion_signal,
              request.sequence,
              request.completion_pe);
          ++completed;
          memory_pool_record_control(
              config,
              MEMORY_POOL_STATUS_OK,
              completed,
              selected_mailbox,
              request.user_tag);
        }
        completed = __shfl_sync(0xffffffffU, completed, 0);
        made_progress = true;
        ready_mask &= ~(1U << selected_lane);
        __syncwarp();
      }
    }
    if (!made_progress)
      __nanosleep(barrierPollSleepCycles);
  }
}

#endif  // DAE_ENABLE_NVSHMEM
