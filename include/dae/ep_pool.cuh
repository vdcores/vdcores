#pragma once

#include "context.cuh"
#include "ep_pool_abi.cuh"

#ifndef DAE_ENABLE_NVSHMEM
#error "ep_pool.cuh requires DAE_ENABLE_NVSHMEM"
#endif

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>

static __device__ __forceinline__ void ep_pool_set_status(
    const EpPoolConfig& config, EpPoolStatus status) {
  if (config.control_address == 0)
    return;
  auto* control = reinterpret_cast<unsigned long long*>(config.control_address);
  atomicCAS(
      control,
      static_cast<unsigned long long>(EP_POOL_STATUS_OK),
      static_cast<unsigned long long>(status));
}

static __device__ __forceinline__ bool ep_pool_u64_product_fits(
    uint64_t left, uint64_t right, uint64_t* product) {
  if (left != 0 && right > UINT64_MAX / left)
    return false;
  *product = left * right;
  return true;
}

static __device__ __forceinline__ bool ep_pool_signal_range_fits(
    uint32_t base, uint64_t count, uint32_t capacity) {
  return base <= capacity && count <= static_cast<uint64_t>(capacity - base);
}

static __device__ __forceinline__ bool ep_pool_valid_config(
    const EpPoolConfig& config) {
  uint64_t required_expert_bytes = 0;
  const uint64_t dispatch_signal_count =
      static_cast<uint64_t>(config.num_experts) * config.num_pes;
  return config.source_address != 0 &&
      config.packed_source_address != 0 &&
      config.expert_input_address != 0 &&
      config.expert_output_address != 0 &&
      config.return_inbox_address != 0 &&
      config.returned_address != 0 &&
      config.send_offsets_address != 0 &&
      config.send_rows_address != 0 &&
      config.send_origin_rows_address != 0 &&
      config.send_batches_address != 0 &&
      config.receive_batches_address != 0 &&
      config.expert_tails_address != 0 &&
      config.sequence_address != 0 &&
      config.control_address != 0 &&
      config.row_bytes >= epPoolMinimumRowBytes &&
      config.row_bytes % epPoolAlignmentBytes == 0 &&
      config.source_capacity_rows != 0 &&
      config.return_capacity_rows != 0 &&
      config.route_capacity != 0 &&
      config.active_rows <= config.route_capacity &&
      config.source_stride >= config.row_bytes &&
      config.source_stride % epPoolAlignmentBytes == 0 &&
      config.expert_row_stride == config.row_bytes &&
      config.return_stride >= config.row_bytes &&
      config.return_stride % epPoolAlignmentBytes == 0 &&
      config.experts_per_pe != 0 &&
      config.num_pes != 0 &&
      config.num_pes <= epPoolMaxPes &&
      config.my_pe < config.num_pes &&
      config.num_experts != 0 &&
      config.num_experts <= epPoolMaxExperts &&
      config.num_experts == config.experts_per_pe * config.num_pes &&
      ep_pool_u64_product_fits(
          config.expert_capacity_rows,
          config.expert_row_stride,
          &required_expert_bytes) &&
      config.expert_stride >= required_expert_bytes &&
      config.expert_stride % epPoolAlignmentBytes == 0 &&
      ep_pool_signal_range_fits(
          config.dispatch_signal_base,
          dispatch_signal_count,
          config.signal_count) &&
      ep_pool_signal_range_fits(
          config.return_signal_base,
          config.num_experts,
          config.signal_count) &&
      ep_pool_signal_range_fits(
          config.reset_signal_base,
          config.num_pes,
          config.signal_count);
}

// The integrated communication path deliberately supports only aligned,
// contiguous LLM activation rows.  This keeps the hot copy loop to one form.
static __device__ __forceinline__ void ep_pool_copy_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t lane) {
  auto* dst = reinterpret_cast<uint4*>(destination);
  const auto* src = reinterpret_cast<const uint4*>(source);
  const uint64_t vectors = bytes / sizeof(uint4);
  constexpr uint64_t copyIlp = 4;
  uint64_t index = lane;
  for (; index + (copyIlp - 1) * 32 < vectors; index += copyIlp * 32) {
    const uint4 value0 = src[index];
    const uint4 value1 = src[index + 32];
    const uint4 value2 = src[index + 64];
    const uint4 value3 = src[index + 96];
    dst[index] = value0;
    dst[index + 32] = value1;
    dst[index + 64] = value2;
    dst[index + 96] = value3;
  }
  for (; index < vectors; index += 32)
    dst[index] = src[index];
}

static __device__ __forceinline__ void ep_pool_put_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t target_pe,
    uint32_t my_pe,
    uint32_t lane) {
  if (bytes == 0)
    return;
  if (target_pe == my_pe) {
    ep_pool_copy_warp(destination, source, bytes, lane);
    return;
  }
  nvshmemx_putmem_warp(
      destination, source, static_cast<size_t>(bytes), target_pe);
}

static __device__ __forceinline__ void ep_pool_put_signal_nbi_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint64_t* signal_array,
    uint32_t signal_id,
    uint64_t sequence,
    uint32_t target_pe,
    uint32_t my_pe,
    uint32_t lane) {
  if (target_pe == my_pe) {
    if (bytes != 0)
      ep_pool_copy_warp(destination, source, bytes, lane);
    __syncwarp();
    if (lane == 0) {
      __threadfence();
      atomicExch(
          reinterpret_cast<unsigned long long*>(signal_array + signal_id),
          static_cast<unsigned long long>(sequence));
    }
    __syncwarp();
    return;
  }
  if (bytes != 0) {
    nvshmemx_putmem_signal_nbi_warp(
        destination,
        source,
        static_cast<size_t>(bytes),
        signal_array + signal_id,
        sequence,
        NVSHMEM_SIGNAL_SET,
        target_pe);
  } else {
    if (lane == 0) {
      nvshmemx_signal_op(
          signal_array + signal_id,
          sequence,
          NVSHMEM_SIGNAL_SET,
          target_pe);
    }
    __syncwarp();
  }
}

static __device__ __forceinline__ void ep_pool_publish_peer(
    uint64_t* signal_array,
    uint32_t signal_id,
    uint64_t sequence,
    uint32_t target_pe,
    uint32_t my_pe,
    uint32_t lane) {
  __syncwarp();
  if (lane == 0) {
    if (target_pe == my_pe) {
      __threadfence();
      atomicExch(
          reinterpret_cast<unsigned long long*>(signal_array + signal_id),
          static_cast<unsigned long long>(sequence));
    } else {
      nvshmemx_signal_op(
          signal_array + signal_id,
          sequence,
          NVSHMEM_SIGNAL_SET,
          target_pe);
    }
  }
  __syncwarp();
}

static __device__ __forceinline__ uint64_t ep_pool_sequence(
    const EpPoolConfig& config) {
  const auto* sequence = reinterpret_cast<const unsigned long long*>(
      config.sequence_address);
  return atomicAdd(const_cast<unsigned long long*>(sequence), 0ULL);
}

static __device__ __forceinline__ bool ep_pool_routes_valid(
    const EpPoolConfig& config,
    uint32_t begin,
    uint32_t end,
    uint32_t lane) {
  const auto* send_rows = reinterpret_cast<const uint32_t*>(
      config.send_rows_address);
  const auto* send_origin_rows = reinterpret_cast<const uint32_t*>(
      config.send_origin_rows_address);
  bool valid = begin <= end && end <= config.active_rows;
  if (valid) {
    for (uint32_t index = begin + lane; index < end; index += 32) {
      valid = valid &&
          send_rows[index] < config.source_capacity_rows &&
          send_origin_rows[index] < config.return_capacity_rows;
    }
  }
  return __all_sync(0xffffffffU, valid);
}

static __device__ __forceinline__ bool ep_pool_batch_valid(
    const EpPoolConfig& config,
    const EpPoolBatch& batch,
    uint64_t sequence,
    uint32_t source_pe,
    uint32_t local_expert) {
  return batch.sequence == sequence &&
      batch.source_pe == source_pe &&
      batch.local_expert == local_expert &&
      batch.flags == EP_POOL_BATCH_FLAGS_NONE &&
      batch.source_base <= config.route_capacity &&
      batch.row_count <= config.route_capacity - batch.source_base &&
      batch.base_row <= config.expert_capacity_rows &&
      batch.row_count <= config.expert_capacity_rows - batch.base_row;
}

static __device__ __forceinline__ void ep_pool_wait_signal_warp(
    uint64_t* signal_array,
    uint32_t signal_base,
    uint64_t sequence,
    uint32_t count,
    uint32_t lane) {
  bool ready = lane >= count;
  while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
    if (!ready)
      ready = nvshmem_signal_fetch(signal_array + signal_base + lane) >= sequence;
    if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
      __nanosleep(barrierPollSleepCycles);
  }
}

static __device__ __noinline__ void ep_pool_reset(
    const EpPoolConfig& config,
    int* bars,
    uint64_t* signal_array,
    uint32_t release_barrier,
    uint32_t lane) {
  if (!ep_pool_valid_config(config)) {
    if (lane == 0)
      ep_pool_set_status(config, EP_POOL_STATUS_BAD_CONFIG);
    __syncwarp();
    return;
  }
  auto* tails = reinterpret_cast<unsigned long long*>(
      config.expert_tails_address);
  auto* control = reinterpret_cast<unsigned long long*>(config.control_address);
  for (uint32_t index = lane; index < config.experts_per_pe; index += 32)
    tails[index] = 0;
  for (uint32_t index = lane; index < 8; index += 32)
    control[index] = 0;
  __threadfence_system();
  __syncwarp();

  const uint64_t sequence = ep_pool_sequence(config);
  for (uint32_t target_pe = 0; target_pe < config.num_pes; ++target_pe) {
    ep_pool_publish_peer(
        signal_array,
        config.reset_signal_base + config.my_pe,
        sequence,
        target_pe,
        config.my_pe,
        lane);
  }
  ep_pool_wait_signal_warp(
      signal_array,
      config.reset_signal_base,
      sequence,
      config.num_pes,
      lane);
  if (lane == 0) {
    control[4] = sequence;
    __threadfence();
    atomicSub(bars + release_barrier, 1);
  }
  __syncwarp();
}

static __device__ __noinline__ void ep_pool_dispatch_expert(
    const EpPoolConfig& config,
    int* bars,
    uint64_t* signal_array,
    uint32_t global_expert,
    uint32_t release_barrier,
    uint32_t lane) {
  if (!ep_pool_valid_config(config) || global_expert >= config.num_experts) {
    if (lane == 0)
      ep_pool_set_status(config, EP_POOL_STATUS_BAD_CONFIG);
    __syncwarp();
    return;
  }

  const uint64_t sequence = ep_pool_sequence(config);
  const uint32_t target_pe = global_expert / config.experts_per_pe;
  const uint32_t local_expert = global_expert % config.experts_per_pe;
  const auto* offsets = reinterpret_cast<const uint32_t*>(
      config.send_offsets_address);
  const auto* rows = reinterpret_cast<const uint32_t*>(config.send_rows_address);
  const auto* source = reinterpret_cast<const uint8_t*>(config.source_address);
  auto* packed = reinterpret_cast<uint8_t*>(config.packed_source_address);
  auto* send_batches = reinterpret_cast<EpPoolBatch*>(
      config.send_batches_address);
  auto* receive_batches = reinterpret_cast<EpPoolBatch*>(
      config.receive_batches_address);
  auto* tails = reinterpret_cast<uint64_t*>(config.expert_tails_address);
  auto* expert_input = reinterpret_cast<uint8_t*>(config.expert_input_address);
  auto* control = reinterpret_cast<unsigned long long*>(config.control_address);

  const uint32_t begin = offsets[global_expert];
  const uint32_t end = offsets[global_expert + 1];
  const bool routes_valid = ep_pool_routes_valid(config, begin, end, lane);
  uint32_t count = routes_valid ? end - begin : 0;
  if (!routes_valid)
    ep_pool_set_status(config, EP_POOL_STATUS_ROUTE_RANGE);

  uint64_t base_row = 0;
  if (lane == 0 && count != 0) {
    if (target_pe == config.my_pe) {
      base_row = atomicAdd(
          reinterpret_cast<unsigned long long*>(tails + local_expert),
          static_cast<unsigned long long>(count));
    } else {
      base_row = nvshmem_uint64_atomic_fetch_add(
          tails + local_expert,
          static_cast<uint64_t>(count),
          target_pe);
    }
  }
  base_row = __shfl_sync(0xffffffffU, base_row, 0);
  const bool capacity_valid =
      base_row <= config.expert_capacity_rows &&
      count <= config.expert_capacity_rows - base_row;
  if (!capacity_valid) {
    ep_pool_set_status(config, EP_POOL_STATUS_CAPACITY);
    count = 0;
  }

  EpPoolBatch* batch = send_batches + global_expert;
  if (lane == 0) {
    batch->sequence = sequence;
    batch->base_row = base_row;
    batch->source_base = begin;
    batch->row_count = count;
    batch->source_pe = config.my_pe;
    batch->local_expert = local_expert;
    batch->flags = routes_valid && capacity_valid
        ? EP_POOL_BATCH_FLAGS_NONE
        : EP_POOL_BATCH_FLAGS_ERROR;
    batch->reserved_u32[0] = 0;
    batch->reserved_u32[1] = 0;
    batch->reserved_u32[2] = 0;
  }
  __syncwarp();

  // This tiny blocking put returns as soon as the descriptor source can be
  // reused.  Its remote delivery proceeds while the warp gathers rows.
  ep_pool_put_warp(
      receive_batches +
          static_cast<uint64_t>(local_expert) * config.num_pes + config.my_pe,
      batch,
      sizeof(EpPoolBatch),
      target_pe,
      config.my_pe,
      lane);

  // Metadata is already in flight while the warp gathers source rows.  A
  // local expert is gathered directly into its final pool rows; a remote
  // expert uses one contiguous packed message for its RMA.
  if (count != 0) {
    for (uint32_t index = begin; index < end; ++index) {
      uint8_t* destination = packed +
          static_cast<uint64_t>(index) * config.row_bytes;
      if (target_pe == config.my_pe) {
        destination = expert_input +
            static_cast<uint64_t>(local_expert) * config.expert_stride +
            (base_row + index - begin) * config.expert_row_stride;
      }
      ep_pool_copy_warp(
          destination,
          source + static_cast<uint64_t>(rows[index]) * config.source_stride,
          config.row_bytes,
          lane);
    }
  }
  __syncwarp();
  if (target_pe != config.my_pe) {
    // A fence orders the preceding blocking descriptor put before the
    // put-with-signal.  The signal is the payload's remote completion point.
    if (lane == 0)
      nvshmem_fence();
    __syncwarp();
  }
  ep_pool_put_signal_nbi_warp(
      expert_input +
          static_cast<uint64_t>(local_expert) * config.expert_stride +
          base_row * config.expert_row_stride,
      packed + static_cast<uint64_t>(begin) * config.row_bytes,
      target_pe == config.my_pe
          ? 0
          : static_cast<uint64_t>(count) * config.row_bytes,
      signal_array,
      config.dispatch_signal_base + global_expert * config.num_pes + config.my_pe,
      sequence,
      target_pe,
      config.my_pe,
      lane);

  if (config.my_pe != target_pe)
    return;

  const uint32_t signal_base =
      config.dispatch_signal_base + global_expert * config.num_pes;
  ep_pool_wait_signal_warp(
      signal_array, signal_base, sequence, config.num_pes, lane);
  if (lane < config.num_pes) {
    const EpPoolBatch received = receive_batches[
        static_cast<uint64_t>(local_expert) * config.num_pes + lane];
    if (!ep_pool_batch_valid(config, received, sequence, lane, local_expert)) {
      ep_pool_set_status(config, EP_POOL_STATUS_BATCH);
    } else {
      atomicAdd(control + 1, 1ULL);
      atomicAdd(
          control + 2,
          static_cast<unsigned long long>(received.row_count));
    }
  }
  __syncwarp();
  if (lane == 0) {
    __threadfence();
    atomicSub(bars + release_barrier, 1);
  }
  __syncwarp();
}

static __device__ __noinline__ void ep_pool_return_expert(
    const EpPoolConfig& config,
    int* bars,
    uint64_t* signal_array,
    uint32_t global_expert,
    uint32_t wait_barrier,
    uint32_t lane) {
  if (!ep_pool_valid_config(config) || global_expert >= config.num_experts) {
    if (lane == 0)
      ep_pool_set_status(config, EP_POOL_STATUS_BAD_CONFIG);
    __syncwarp();
    return;
  }

  const uint64_t sequence = ep_pool_sequence(config);
  const uint32_t target_pe = global_expert / config.experts_per_pe;
  const uint32_t local_expert = global_expert % config.experts_per_pe;
  const auto* receive_batches = reinterpret_cast<const EpPoolBatch*>(
      config.receive_batches_address);
  const auto* expert_output = reinterpret_cast<const uint8_t*>(
      config.expert_output_address);
  auto* return_inbox = reinterpret_cast<uint8_t*>(config.return_inbox_address);

  if (config.my_pe == target_pe) {
    if (lane == 0) {
      volatile int* bar = bars + wait_barrier;
      while (*bar != 0)
        __nanosleep(barrierPollSleepCycles);
    }
    __syncwarp();

    for (uint32_t source_pe = 0; source_pe < config.num_pes; ++source_pe) {
      const EpPoolBatch batch = receive_batches[
          static_cast<uint64_t>(local_expert) * config.num_pes + source_pe];
      const bool batch_valid = ep_pool_batch_valid(
          config, batch, sequence, source_pe, local_expert);
      if (!batch_valid)
        ep_pool_set_status(config, EP_POOL_STATUS_BATCH);
      const uint32_t source_base = batch_valid ? batch.source_base : 0;
      const uint64_t base_row = batch_valid ? batch.base_row : 0;
      const uint32_t row_count = batch_valid ? batch.row_count : 0;
      ep_pool_put_signal_nbi_warp(
          return_inbox +
              static_cast<uint64_t>(source_base) * config.row_bytes,
          expert_output +
              static_cast<uint64_t>(local_expert) * config.expert_stride +
              base_row * config.expert_row_stride,
          source_pe == config.my_pe
              ? 0
              : static_cast<uint64_t>(row_count) * config.row_bytes,
          signal_array,
          config.return_signal_base + global_expert,
          sequence,
          source_pe,
          config.my_pe,
          lane);
    }
    // One quiet after all peer batches makes expert_output reusable when this
    // VDCores program exits without serializing the individual sends.
    __syncwarp();
    if (lane == 0)
      nvshmem_quiet();
    __syncwarp();
  }

  ep_pool_wait_signal_warp(
      signal_array,
      config.return_signal_base + global_expert,
      sequence,
      1,
      lane);

  const auto* offsets = reinterpret_cast<const uint32_t*>(
      config.send_offsets_address);
  const auto* origins = reinterpret_cast<const uint32_t*>(
      config.send_origin_rows_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);
  const uint32_t begin = offsets[global_expert];
  const uint32_t end = offsets[global_expert + 1];
  const EpPoolBatch local_batch = receive_batches[
      static_cast<uint64_t>(local_expert) * config.num_pes + config.my_pe];
  const bool local_batch_valid = target_pe != config.my_pe ||
      (ep_pool_batch_valid(
           config, local_batch, sequence, config.my_pe, local_expert) &&
       local_batch.source_base == begin &&
       local_batch.row_count == end - begin);
  if (!local_batch_valid)
    ep_pool_set_status(config, EP_POOL_STATUS_BATCH);
  for (uint32_t index = begin; index < end; ++index) {
    const uint32_t origin = origins[index];
    if (origin >= config.return_capacity_rows) {
      ep_pool_set_status(config, EP_POOL_STATUS_ROUTE_RANGE);
      continue;
    }
    const uint8_t* source = return_inbox +
        static_cast<uint64_t>(index) * config.row_bytes;
    if (target_pe == config.my_pe && local_batch_valid) {
      source = expert_output +
          static_cast<uint64_t>(local_expert) * config.expert_stride +
          (local_batch.base_row + index - begin) * config.expert_row_stride;
    }
    if (target_pe != config.my_pe || local_batch_valid) {
      ep_pool_copy_warp(
          returned + static_cast<uint64_t>(origin) * config.return_stride,
          source,
          config.row_bytes,
          lane);
    }
  }
  __syncwarp();
  if (lane == 0) {
    auto* control = reinterpret_cast<unsigned long long*>(config.control_address);
    atomicAdd(control + 3, 1ULL);
  }
  __syncwarp();
}
