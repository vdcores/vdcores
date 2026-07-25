#pragma once

#include "context.cuh"

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

static __device__ __forceinline__ void memory_pool_get(
    void* destination, const void* source, uint64_t bytes, int source_pe) {
  if (source_pe == nvshmem_my_pe()) {
    memory_pool_copy_local(destination, source, bytes);
    __threadfence_system();
    return;
  }
  nvshmem_getmem_nbi(destination, source, static_cast<size_t>(bytes), source_pe);
  nvshmem_quiet();
}

static __device__ __forceinline__ void memory_pool_put(
    void* destination, const void* source, uint64_t bytes, int target_pe) {
  if (target_pe == nvshmem_my_pe()) {
    memory_pool_copy_local(destination, source, bytes);
    __threadfence_system();
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
      __threadfence_system();
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
  __threadfence_system();
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
      if (published <= consumed[mailbox])
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
        if (dependencies[request.wait_slot] < request.wait_value)
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

      if (request.signal_slot != memoryPoolNoDependency) {
        dependencies[request.signal_slot] += request.signal_delta;
        __threadfence_system();
      }
      consumed[mailbox] = request.sequence;
      __threadfence_system();

      nvshmemx_signal_op(
          signal_array + request.completion_signal,
          request.sequence,
          NVSHMEM_SIGNAL_SET,
          request.completion_pe);
      nvshmem_quiet();

      ++completed;
      made_progress = true;
      memory_pool_record_control(
          config, MEMORY_POOL_STATUS_OK, completed, mailbox, request.user_tag);
    }
    if (!made_progress)
      __nanosleep(barrierPollSleepCycles);
  }
}

#endif  // DAE_ENABLE_NVSHMEM
