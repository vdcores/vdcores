#pragma once

#include "context.cuh"
#include "pool_slice_abi.cuh"

#ifndef DAE_ENABLE_NVSHMEM
#error "pool_slice.cuh requires DAE_ENABLE_NVSHMEM"
#endif

#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstddef>
#include <cstdint>

static __device__ __forceinline__ void pool_slice_set_status(
    const PoolSliceConfig& config, PoolSliceStatus status) {
  if (config.control_address == 0)
    return;
  auto* control = reinterpret_cast<unsigned long long*>(config.control_address);
  atomicCAS(
      control,
      static_cast<unsigned long long>(POOL_SLICE_STATUS_OK),
      static_cast<unsigned long long>(status));
}

static __device__ __forceinline__ bool pool_slice_u64_product_fits(
    uint64_t left, uint64_t right, uint64_t* product) {
  if (left != 0 && right > UINT64_MAX / left)
    return false;
  *product = left * right;
  return true;
}

static __device__ __forceinline__ bool pool_slice_signal_range_fits(
    uint32_t base, uint32_t count, uint32_t capacity) {
  return base <= capacity && count <= capacity - base;
}

static __device__ __forceinline__ bool pool_slice_signal_ranges_disjoint(
    uint32_t left,
    uint32_t right,
    uint32_t count) {
  return static_cast<uint64_t>(left) + count <= right ||
      static_cast<uint64_t>(right) + count <= left;
}

static __device__ __forceinline__ bool pool_slice_valid_config(
    const PoolSliceConfig& config) {
  uint64_t required_expert_bytes = 0;
  return config.source_address != 0 &&
      config.token_pool_address != 0 &&
      config.expert_input_address != 0 &&
      config.expert_output_address != 0 &&
      config.return_inbox_address != 0 &&
      config.returned_address != 0 &&
      config.send_offsets_address != 0 &&
      config.send_rows_address != 0 &&
      config.send_origin_rows_address != 0 &&
      config.send_batches_address != 0 &&
      config.receive_batches_address != 0 &&
      config.offsets_inbox_address != 0 &&
      config.rows_inbox_address != 0 &&
      config.receive_routes_address != 0 &&
      config.reader_tails_address != 0 &&
      config.sequence_address != 0 &&
      config.group_ready_address != 0 &&
      config.control_address != 0 &&
      config.row_bytes >= poolSliceMinimumRowBytes &&
      config.row_bytes % poolSliceAlignmentBytes == 0 &&
      config.source_stride >= config.row_bytes &&
      config.source_stride % poolSliceAlignmentBytes == 0 &&
      config.pool_stride >= config.row_bytes &&
      config.pool_stride % poolSliceAlignmentBytes == 0 &&
      config.expert_row_stride == config.row_bytes &&
      config.return_stride >= config.row_bytes &&
      config.return_stride % poolSliceAlignmentBytes == 0 &&
      config.active_rows <= config.route_capacity &&
      config.token_capacity != 0 &&
      config.route_capacity != 0 &&
      config.expert_capacity_rows != 0 &&
      config.local_readers != 0 &&
      config.local_readers < 132 &&
      config.num_pes != 0 &&
      config.num_pes <= poolSliceMaxPes &&
      config.my_pe < config.num_pes &&
      config.return_capacity_rows != 0 &&
      (config.flags & ~POOL_SLICE_FLAGS_STREAMING_GATHER) == 0 &&
      (config.data_stages == 1 || config.data_stages == 2) &&
      ((config.data_stages == 1 && config.early_ready_rows == 0) ||
       (config.data_stages == 2 &&
        (config.flags & POOL_SLICE_FLAGS_STREAMING_GATHER) != 0 &&
        config.early_ready_rows != 0 &&
        config.early_ready_rows < config.token_capacity)) &&
      pool_slice_u64_product_fits(
          config.expert_capacity_rows,
          config.expert_row_stride,
          &required_expert_bytes) &&
      config.expert_stride >= required_expert_bytes &&
      config.expert_stride % poolSliceAlignmentBytes == 0 &&
      pool_slice_signal_range_fits(
          config.queue_signal_base, config.num_pes, config.signal_count) &&
      pool_slice_signal_range_fits(
          config.data_signal_base, config.num_pes, config.signal_count) &&
      pool_slice_signal_range_fits(
          config.return_signal_base, config.num_pes, config.signal_count) &&
      pool_slice_signal_ranges_disjoint(
          config.queue_signal_base, config.data_signal_base, config.num_pes) &&
      pool_slice_signal_ranges_disjoint(
          config.queue_signal_base, config.return_signal_base, config.num_pes) &&
      pool_slice_signal_ranges_disjoint(
          config.data_signal_base, config.return_signal_base, config.num_pes);
}

static __device__ __forceinline__ uint64_t pool_slice_sequence(
    const PoolSliceConfig& config) {
  auto* sequence = reinterpret_cast<unsigned long long*>(
      config.sequence_address);
  return atomicAdd(sequence, 0ULL);
}

static __device__ __forceinline__ uint64_t pool_slice_data_signal_value(
    uint64_t sequence,
    uint32_t completed_stages,
    uint32_t data_stages) {
  return sequence * data_stages - (data_stages - completed_stages);
}

// The hot data path deliberately supports only aligned, fixed-width LLM rows.
static __device__ __forceinline__ void pool_slice_copy_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t lane) {
  const uintptr_t alignment = reinterpret_cast<uintptr_t>(destination) |
      reinterpret_cast<uintptr_t>(source) | bytes;
  if ((alignment & (poolSliceAlignmentBytes - 1)) != 0) {
    auto* dst_bytes = reinterpret_cast<uint8_t*>(destination);
    const auto* src_bytes = reinterpret_cast<const uint8_t*>(source);
    for (uint64_t index = lane; index < bytes; index += 32)
      dst_bytes[index] = src_bytes[index];
    return;
  }
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

static __device__ __forceinline__ void pool_slice_get_nbi_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t source_pe,
    uint32_t my_pe,
    uint32_t lane) {
  if (bytes == 0)
    return;
  if (source_pe == my_pe) {
    pool_slice_copy_warp(destination, source, bytes, lane);
    return;
  }
  nvshmemx_getmem_nbi_warp(
      destination, source, static_cast<size_t>(bytes), source_pe);
}

static __device__ __forceinline__ void pool_slice_put_nbi_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t target_pe,
    uint32_t my_pe,
    uint32_t lane) {
  if (bytes == 0)
    return;
  if (target_pe == my_pe) {
    pool_slice_copy_warp(destination, source, bytes, lane);
    return;
  }
  nvshmemx_putmem_nbi_warp(
      destination, source, static_cast<size_t>(bytes), target_pe);
}

static __device__ __forceinline__ void pool_slice_complete_warp(
    uint32_t lane) {
  __syncwarp();
  if (lane == 0) {
    __threadfence_system();
    nvshmem_quiet();
  }
  __syncwarp();
}

static __device__ __forceinline__ void pool_slice_publish_signal(
    uint64_t* signal_array,
    uint32_t signal_id,
    uint64_t sequence,
    uint32_t target_pe,
    uint32_t my_pe,
    uint32_t lane) {
  __syncwarp();
  if (lane == 0) {
    if (target_pe == my_pe) {
      __threadfence_system();
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

static __device__ __forceinline__ void pool_slice_wait_signals_warp(
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

static __device__ __forceinline__ void pool_slice_wait_barriers_warp(
    int* bars,
    uint32_t barrier_base,
    uint32_t count,
    uint32_t lane) {
  for (uint32_t base = 0; base < count; base += 32) {
    const uint32_t index = base + lane;
    bool ready = index >= count;
    while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
      if (!ready)
        ready = *reinterpret_cast<volatile int*>(bars + barrier_base + index) == 0;
      if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
  }
}

static __device__ __forceinline__ uint32_t pool_slice_pe_mask(
    uint32_t num_pes) {
  return num_pes == 32 ? 0xffffffffU : (1U << num_pes) - 1U;
}

// Rotate the PE order so remote operations are issued before the synchronous
// local HBM copy. This leaves the local copy as useful overlap for remote NBI
// traffic and avoids making PE 0 the systematic rank-max straggler.
static __device__ __forceinline__ uint32_t pool_slice_remote_first_pe(
    uint32_t index,
    uint32_t my_pe,
    uint32_t num_pes) {
  const uint32_t pe = index + my_pe + 1;
  return pe >= num_pes ? pe - num_pes : pe;
}

static __device__ __forceinline__ void pool_slice_record_profile(
    uint64_t* g_events,
    uint32_t event,
    uint32_t lane) {
  if (g_events != nullptr && lane == 0) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents + event] =
        cuda::ptx::get_sreg_globaltimer();
  }
  __syncwarp();
}

// Transport-only producer action. Route semantics are not executed here: the
// source only publishes one descriptor into each target pool slice's own
// per-source mailbox.
static __device__ __noinline__ void pool_slice_publish(
    const PoolSliceConfig* config_pointer,
    uint64_t* signal_array,
    uint32_t lane) {
  __shared__ PoolSliceConfig shared_config;
  if (config_pointer == nullptr || signal_array == nullptr)
    return;
  if (lane == 0)
    shared_config = *config_pointer;
  __syncwarp();
  const PoolSliceConfig& config = shared_config;
  if (!pool_slice_valid_config(config)) {
    if (lane == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
    __syncwarp();
    return;
  }

  const uint64_t sequence = pool_slice_sequence(config);
  auto* send_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.send_batches_address);
  auto* receive_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.receive_batches_address);
  const auto* send_offsets = reinterpret_cast<const uint32_t*>(
      config.send_offsets_address);

  for (uint32_t target_pe = lane;
       target_pe < config.num_pes;
       target_pe += 32) {
    PoolSlicePublishBatch& batch = send_batches[target_pe];
    batch.sequence = sequence;
    batch.source_pe = config.my_pe;
    batch.target_pe = target_pe;
    batch.active_rows = config.active_rows;
    batch.flags = POOL_SLICE_BATCH_FLAGS_NONE;
    const uint32_t reader_begin = target_pe * config.local_readers;
    batch.route_begin = send_offsets[reader_begin];
    batch.route_end = send_offsets[reader_begin + config.local_readers];
  }
  __syncwarp();
  if (lane == 0)
    __threadfence_system();
  __syncwarp();

  for (uint32_t target_pe = 0; target_pe < config.num_pes; ++target_pe) {
    PoolSlicePublishBatch* destination = receive_batches + config.my_pe;
    const PoolSlicePublishBatch* source = send_batches + target_pe;
    if (target_pe == config.my_pe) {
      pool_slice_copy_warp(
          destination, source, sizeof(PoolSlicePublishBatch), lane);
      __syncwarp();
      if (lane == 0) {
        __threadfence_system();
        atomicExch(
            reinterpret_cast<unsigned long long*>(
                signal_array + config.queue_signal_base + config.my_pe),
            static_cast<unsigned long long>(sequence));
      }
      __syncwarp();
    } else {
      nvshmemx_putmem_signal_nbi_warp(
          destination,
          source,
          sizeof(PoolSlicePublishBatch),
          signal_array + config.queue_signal_base + config.my_pe,
          sequence,
          NVSHMEM_SIGNAL_SET,
          target_pe);
    }
  }
  // The queue doorbell, rather than local source completion, gates every
  // consumer.  The first metadata quiet in gather also retires these NBI
  // publications before send_batches can be reused, so a separate quiet here
  // would add a serialized network round trip to the fixed protocol floor.
  __syncwarp();
}

static __device__ __noinline__ void pool_slice_gather(
    const PoolSliceConfig* config_pointer,
    int* bars,
    uint64_t* signal_array,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t lane) {
  __shared__ PoolSliceConfig shared_config;
  __shared__ uint32_t shared_status;
  if (config_pointer == nullptr || bars == nullptr || signal_array == nullptr)
    return;
  if (lane == 0) {
    shared_config = *config_pointer;
    shared_status = POOL_SLICE_STATUS_OK;
  }
  __syncwarp();
  const PoolSliceConfig& config = shared_config;
  if (!pool_slice_valid_config(config)) {
    if (lane == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
    __syncwarp();
    return;
  }

  const uint64_t sequence = pool_slice_sequence(config);
  auto* control = reinterpret_cast<unsigned long long*>(config.control_address);
  auto* group_ready = reinterpret_cast<unsigned long long*>(
      config.group_ready_address);
  auto* reader_tails = reinterpret_cast<unsigned long long*>(
      config.reader_tails_address);
  auto* receive_batches = reinterpret_cast<const PoolSlicePublishBatch*>(
      config.receive_batches_address);
  auto* offsets_inbox = reinterpret_cast<uint32_t*>(
      config.offsets_inbox_address);
  auto* rows_inbox = reinterpret_cast<uint32_t*>(config.rows_inbox_address);
  auto* receive_routes = reinterpret_cast<PoolSliceReceiveBatch*>(
      config.receive_routes_address);
  const auto* send_offsets = reinterpret_cast<const uint32_t*>(
      config.send_offsets_address);
  const auto* send_rows = reinterpret_cast<const uint32_t*>(
      config.send_rows_address);

  for (uint32_t index = lane;
       index < poolSliceControlWords;
       index += 32)
    control[index] = 0;
  for (uint32_t index = lane; index < config.local_readers; index += 32)
    reader_tails[index] = 0;
  if (lane == 0)
    *group_ready = 0;
  __syncwarp();
  if (lane == 0)
    __threadfence_system();
  __syncwarp();

  // Queue signals are transport doorbells. Consuming all source descriptors
  // closes the shared sender group for every dynamic read on this slice.
  pool_slice_wait_signals_warp(
      signal_array,
      config.queue_signal_base,
      sequence,
      config.num_pes,
      lane);

  const uint32_t global_reader_base = config.my_pe * config.local_readers;
  for (uint32_t source_pe = 0;
       source_pe < config.num_pes;
       ++source_pe) {
    const PoolSlicePublishBatch batch = receive_batches[source_pe];
    if (lane == 0 &&
        (batch.sequence != sequence ||
         batch.source_pe != source_pe ||
         batch.target_pe != config.my_pe ||
         batch.active_rows > config.route_capacity ||
         batch.route_begin > batch.route_end ||
         batch.route_end > batch.active_rows ||
         batch.flags != POOL_SLICE_BATCH_FLAGS_NONE)) {
      shared_status = batch.sequence != sequence
          ? POOL_SLICE_STATUS_SEQUENCE
          : POOL_SLICE_STATUS_BATCH;
    }
    __syncwarp();
    uint32_t* destination = offsets_inbox +
        static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
    if (config.local_readers == 1) {
      if (lane == 0) {
        destination[0] = batch.route_begin;
        destination[1] = batch.route_end;
      }
      __syncwarp();
    } else {
      pool_slice_get_nbi_warp(
          destination,
          send_offsets + global_reader_base,
          static_cast<uint64_t>(config.local_readers + 1) * sizeof(uint32_t),
          source_pe,
          config.my_pe,
          lane);
    }
  }
  if (config.local_readers == 1) {
    __syncwarp();
  } else {
    pool_slice_complete_warp(lane);
  }
  if (shared_status != POOL_SLICE_STATUS_OK) {
    if (lane == 0)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(shared_status));
    __syncwarp();
    return;
  }

  if (lane == 0) {
    for (uint32_t source_pe = 0;
         source_pe < config.num_pes &&
             shared_status == POOL_SLICE_STATUS_OK;
         ++source_pe) {
      const uint32_t* offsets = offsets_inbox +
          static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
      const PoolSlicePublishBatch batch = receive_batches[source_pe];
      const uint32_t source_begin = offsets[0];
      const uint32_t source_end = offsets[config.local_readers];
      if (source_begin != batch.route_begin ||
          source_end != batch.route_end ||
          source_end - source_begin > config.route_capacity) {
        shared_status = POOL_SLICE_STATUS_ROUTE_RANGE;
        break;
      }
      for (uint32_t local_reader = 0;
           local_reader < config.local_readers;
           ++local_reader) {
        const uint32_t begin = offsets[local_reader];
        const uint32_t end = offsets[local_reader + 1];
        if (begin > end || begin < source_begin || end > source_end) {
          shared_status = POOL_SLICE_STATUS_ROUTE_RANGE;
          break;
        }
        const uint32_t count = end - begin;
        const uint64_t base_row = reader_tails[local_reader];
        if (base_row > config.expert_capacity_rows ||
            count > config.expert_capacity_rows - base_row) {
          shared_status = POOL_SLICE_STATUS_CAPACITY;
          break;
        }
        reader_tails[local_reader] = base_row + count;
        PoolSliceReceiveBatch& route = receive_routes[
            static_cast<uint64_t>(local_reader) * config.num_pes + source_pe];
        route.sequence = sequence;
        route.base_row = base_row;
        route.source_begin = begin;
        route.row_count = count;
        route.source_pe = source_pe;
        route.local_reader = local_reader;
        route.flags = POOL_SLICE_BATCH_FLAGS_NONE;
        route.reserved_u32[0] = 0;
        route.reserved_u32[1] = 0;
        route.reserved_u32[2] = 0;
      }
    }
  }
  __syncwarp();
  if (shared_status != POOL_SLICE_STATUS_OK) {
    if (lane == 0)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(shared_status));
    __syncwarp();
    return;
  }

  // Pull each source's metadata slice into pool-owned HBM. Metadata movement
  // is batched per source and completed with one quiet for the whole slice.
  for (uint32_t source_pe = 0;
       source_pe < config.num_pes;
       ++source_pe) {
    const uint32_t* offsets = offsets_inbox +
        static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
    const uint32_t source_begin = offsets[0];
    const uint32_t source_count =
        offsets[config.local_readers] - source_begin;
    pool_slice_get_nbi_warp(
        rows_inbox + static_cast<uint64_t>(source_pe) * config.route_capacity,
        send_rows + source_begin,
        static_cast<uint64_t>(source_count) * sizeof(uint32_t),
        source_pe,
        config.my_pe,
        lane);
  }
  pool_slice_complete_warp(lane);

  bool routes_valid = true;
  for (uint32_t source_pe = 0;
       source_pe < config.num_pes;
       ++source_pe) {
    const uint32_t* offsets = offsets_inbox +
        static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
    const uint32_t source_count =
        offsets[config.local_readers] - offsets[0];
    const uint32_t* rows = rows_inbox +
        static_cast<uint64_t>(source_pe) * config.route_capacity;
    for (uint32_t index = lane; index < source_count; index += 32) {
      routes_valid = routes_valid && rows[index] < config.token_capacity;
    }
  }
  if (!__all_sync(0xffffffffU, routes_valid)) {
    if (lane == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_ROUTE_RANGE);
    __syncwarp();
    return;
  }

  // The pool memory/compute VMs perform the local token write. Only the pool
  // communication warp publishes its readiness to every destination PE.
  pool_slice_wait_barriers_warp(bars, write_barrier, 1, lane);
  for (uint32_t target_pe = 0;
       target_pe < config.num_pes;
       ++target_pe) {
    pool_slice_publish_signal(
        signal_array,
        config.data_signal_base + config.my_pe,
        sequence,
        target_pe,
        config.my_pe,
        lane);
  }
  // All peers wait on these source-indexed signals.  Waiting on the reciprocal
  // set below provides global progress without an extra local quiet phase.
  pool_slice_wait_signals_warp(
      signal_array,
      config.data_signal_base,
      sequence,
      config.num_pes,
      lane);

  const auto* token_pool = reinterpret_cast<const uint8_t*>(
      config.token_pool_address);
  auto* expert_input = reinterpret_cast<uint8_t*>(
      config.expert_input_address);
  for (uint32_t source_pe = 0;
       source_pe < config.num_pes;
       ++source_pe) {
    const uint32_t* offsets = offsets_inbox +
        static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
    const uint32_t source_begin = offsets[0];
    const uint32_t* rows = rows_inbox +
        static_cast<uint64_t>(source_pe) * config.route_capacity;
    for (uint32_t local_reader = 0;
         local_reader < config.local_readers;
         ++local_reader) {
      const PoolSliceReceiveBatch route = receive_routes[
          static_cast<uint64_t>(local_reader) * config.num_pes + source_pe];
      const uint32_t metadata_begin =
          offsets[local_reader] - source_begin;
      for (uint32_t index = 0; index < route.row_count; ++index) {
        const uint32_t source_row = rows[metadata_begin + index];
        pool_slice_get_nbi_warp(
            expert_input +
                static_cast<uint64_t>(local_reader) * config.expert_stride +
                (route.base_row + index) * config.expert_row_stride,
            token_pool + static_cast<uint64_t>(source_row) * config.pool_stride,
            config.row_bytes,
            source_pe,
            config.my_pe,
            lane);
      }
    }
  }
  pool_slice_complete_warp(lane);

  if (lane == 0) {
    uint64_t received_rows = 0;
    for (uint32_t local_reader = 0;
         local_reader < config.local_readers;
         ++local_reader)
      received_rows += reader_tails[local_reader];
    *group_ready = sequence;
    control[1] = config.num_pes;
    control[2] = received_rows;
    control[4] = sequence;
    __threadfence_system();
    for (uint32_t local_reader = 0;
         local_reader < config.local_readers;
         ++local_reader)
      atomicSub(bars + dispatch_barrier_base + local_reader, 1);
  }
  __syncwarp();
}

// Issue one source-row interval for every ready source. Two-stage activation
// readiness calls this once for the early prefix and once for the remainder;
// each row is therefore fetched exactly once while both batches can remain
// NBI and in flight until the final gather quiet.
static __device__ __forceinline__ uint32_t pool_slice_issue_payload_stage(
    const PoolSliceConfig& config,
    uint32_t source_mask,
    uint32_t source_row_begin,
    uint32_t source_row_end,
    uint64_t* g_events,
    uint32_t lane) {
  const auto* offsets_inbox = reinterpret_cast<const uint32_t*>(
      config.offsets_inbox_address);
  const auto* rows_inbox = reinterpret_cast<const uint32_t*>(
      config.rows_inbox_address);
  const auto* receive_routes =
      reinterpret_cast<const PoolSliceReceiveBatch*>(
          config.receive_routes_address);
  const auto* token_pool = reinterpret_cast<const uint8_t*>(
      config.token_pool_address);
  auto* expert_input = reinterpret_cast<uint8_t*>(
      config.expert_input_address);
  uint32_t issued_mask = 0;
  uint32_t payload_recorded = g_events == nullptr;
  if (g_events != nullptr && lane == 0) {
    payload_recorded = g_events[
        static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
        poolSliceProfileFirstPayload] != 0;
  }
  payload_recorded = __shfl_sync(0xffffffffU, payload_recorded, 0);

  for (uint32_t source_index = 0;
       source_index < config.num_pes;
       ++source_index) {
    const uint32_t source_pe = pool_slice_remote_first_pe(
        source_index, config.my_pe, config.num_pes);
    const uint32_t source_bit = 1U << source_pe;
    if ((source_mask & source_bit) == 0)
      continue;
    const uint32_t* offsets = offsets_inbox +
        static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
    const uint32_t source_begin = offsets[0];
    const uint32_t* rows = rows_inbox +
        static_cast<uint64_t>(source_pe) * config.route_capacity;
    bool source_issued = false;
    for (uint32_t local_reader = 0;
         local_reader < config.local_readers;
         ++local_reader) {
      const PoolSliceReceiveBatch route = receive_routes[
          static_cast<uint64_t>(local_reader) * config.num_pes + source_pe];
      const uint32_t metadata_begin = offsets[local_reader] - source_begin;
      for (uint32_t index = 0; index < route.row_count; ++index) {
        const uint32_t source_row = rows[metadata_begin + index];
        if (source_row < source_row_begin || source_row >= source_row_end)
          continue;
        if (!payload_recorded) {
          pool_slice_record_profile(
              g_events, poolSliceProfileFirstPayload, lane);
          payload_recorded = true;
        }
        source_issued = true;
        pool_slice_get_nbi_warp(
            expert_input +
                static_cast<uint64_t>(local_reader) * config.expert_stride +
                (route.base_row + index) * config.expert_row_stride,
            token_pool + static_cast<uint64_t>(source_row) * config.pool_stride,
            config.row_bytes,
            source_pe,
            config.my_pe,
            lane);
      }
    }
    if (source_issued)
      issued_mask |= source_bit;
  }
  return issued_mask;
}

// Event-driven gather. Descriptor ingestion, the local source write, remote
// source readiness, and payload gets advance independently. A metadata wave
// uses one quiet to make newly fetched route rows consumable; payload gets
// remain NBI and in flight while the pool continues scanning later senders.
static __device__ __noinline__ void pool_slice_gather_streaming(
    const PoolSliceConfig* config_pointer,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t lane) {
  __shared__ PoolSliceConfig shared_config;
  __shared__ uint32_t shared_status;
  if (config_pointer == nullptr || bars == nullptr || signal_array == nullptr)
    return;
  if (lane == 0) {
    shared_config = *config_pointer;
    shared_status = POOL_SLICE_STATUS_OK;
    if (g_events != nullptr) {
      uint64_t* block_events =
          g_events + static_cast<uint64_t>(blockIdx.x) * numProfileEvents;
      block_events[poolSliceProfileDataPublished] = 0;
      block_events[poolSliceProfileFirstPayload] = 0;
      block_events[poolSliceProfileMetadataClosed] = 0;
      block_events[poolSliceProfilePayloadDone] = 0;
      block_events[poolSliceProfileFirstDataPublished] = 0;
    }
  }
  __syncwarp();
  const PoolSliceConfig& config = shared_config;
  if (!pool_slice_valid_config(config)) {
    if (lane == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
    __syncwarp();
    return;
  }

  const uint64_t sequence = pool_slice_sequence(config);
  if (sequence > UINT64_MAX / config.data_stages) {
    if (lane == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_SEQUENCE);
    __syncwarp();
    return;
  }
  auto* control = reinterpret_cast<unsigned long long*>(config.control_address);
  auto* group_ready = reinterpret_cast<unsigned long long*>(
      config.group_ready_address);
  auto* reader_tails = reinterpret_cast<unsigned long long*>(
      config.reader_tails_address);
  const auto* receive_batches =
      reinterpret_cast<const PoolSlicePublishBatch*>(
          config.receive_batches_address);
  auto* offsets_inbox = reinterpret_cast<uint32_t*>(
      config.offsets_inbox_address);
  auto* rows_inbox = reinterpret_cast<uint32_t*>(config.rows_inbox_address);
  auto* receive_routes = reinterpret_cast<PoolSliceReceiveBatch*>(
      config.receive_routes_address);
  const auto* send_offsets = reinterpret_cast<const uint32_t*>(
      config.send_offsets_address);
  const auto* send_rows = reinterpret_cast<const uint32_t*>(
      config.send_rows_address);

  for (uint32_t index = lane;
       index < poolSliceControlWords;
       index += 32)
    control[index] = 0;
  for (uint32_t index = lane; index < config.local_readers; index += 32)
    reader_tails[index] = 0;
  if (lane == 0)
    *group_ready = 0;
  __syncwarp();
  if (lane == 0)
    __threadfence_system();
  __syncwarp();

  const uint32_t expected_mask = pool_slice_pe_mask(config.num_pes);
  const uint32_t global_reader_base = config.my_pe * config.local_readers;
  uint32_t queue_seen_mask = 0;
  uint32_t early_seen_mask =
      config.data_stages == 1 ? expected_mask : 0;
  uint32_t data_seen_mask = 0;
  uint32_t metadata_ready_mask = 0;
  uint32_t early_issued_mask =
      config.data_stages == 1 ? expected_mask : 0;
  uint32_t payload_issued_mask = 0;
  uint32_t needed_data_mask = 0;
  uint32_t metadata_waves = 0;
  uint32_t payload_sources = 0;
  uint32_t inflight_sources = 0;
  uint32_t peak_inflight_sources = 0;
  uint32_t published_stages = 0;
  bool metadata_closed_recorded = false;

  while (published_stages != config.data_stages ||
         metadata_ready_mask != expected_mask ||
         payload_issued_mask != expected_mask) {
    bool made_progress = false;

    if (queue_seen_mask != expected_mask) {
      bool queue_ready = lane >= config.num_pes ||
          (queue_seen_mask & (1U << lane)) != 0;
      if (!queue_ready) {
        queue_ready = nvshmem_signal_fetch(
            signal_array + config.queue_signal_base + lane) >= sequence;
      }
      const uint32_t polled_queue_mask =
          __ballot_sync(0xffffffffU, queue_ready) & expected_mask;
      if ((polled_queue_mask & ~queue_seen_mask) != 0)
        made_progress = true;
      queue_seen_mask |= polled_queue_mask;
    }

    if (payload_issued_mask != expected_mask &&
        data_seen_mask != expected_mask) {
      const bool final_seen = lane < config.num_pes &&
          (data_seen_mask & (1U << lane)) != 0;
      uint64_t observed = final_seen || lane >= config.num_pes
          ? UINT64_MAX
          : nvshmem_signal_fetch(
                signal_array + config.data_signal_base + lane);
      const uint64_t first_value = pool_slice_data_signal_value(
          sequence, 1, config.data_stages);
      const uint64_t final_value = pool_slice_data_signal_value(
          sequence, config.data_stages, config.data_stages);
      const bool early_ready = lane >= config.num_pes ||
          (early_seen_mask & (1U << lane)) != 0 || observed >= first_value;
      const bool data_ready = lane >= config.num_pes ||
          final_seen || observed >= final_value;
      if (config.data_stages == 2) {
        const uint32_t polled_early_mask =
            __ballot_sync(0xffffffffU, early_ready) & expected_mask;
        if ((polled_early_mask & ~early_seen_mask) != 0)
          made_progress = true;
        early_seen_mask |= polled_early_mask;
      }
      const uint32_t polled_data_mask =
          __ballot_sync(0xffffffffU, data_ready) & expected_mask;
      if ((polled_data_mask & ~data_seen_mask) != 0)
        made_progress = true;
      data_seen_mask |= polled_data_mask;
    }

    if (queue_seen_mask == expected_mask && !metadata_closed_recorded) {
      pool_slice_record_profile(
          g_events, poolSliceProfileMetadataClosed, lane);
      metadata_closed_recorded = true;
    }

    if (published_stages != config.data_stages) {
      uint32_t local_write_ready = 0;
      if (lane == 0) {
        local_write_ready =
            *reinterpret_cast<volatile int*>(
                bars + write_barrier + published_stages) == 0;
      }
      local_write_ready = __shfl_sync(
          0xffffffffU, local_write_ready, 0);
      if (local_write_ready != 0) {
        const uint32_t completed_stages = published_stages + 1;
        const uint64_t signal_value = pool_slice_data_signal_value(
            sequence, completed_stages, config.data_stages);
        if (lane == 0)
          __threadfence_system();
        __syncwarp();
        for (uint32_t index = 0;
             index < config.num_pes;
             ++index) {
          const uint32_t target_pe = pool_slice_remote_first_pe(
              index, config.my_pe, config.num_pes);
          pool_slice_publish_signal(
              signal_array,
              config.data_signal_base + config.my_pe,
              signal_value,
              target_pe,
              config.my_pe,
              lane);
        }
        published_stages = completed_stages;
        made_progress = true;
        if (completed_stages == 1) {
          pool_slice_record_profile(
              g_events, poolSliceProfileFirstDataPublished, lane);
        }
        if (completed_stages == config.data_stages) {
          pool_slice_record_profile(
              g_events, poolSliceProfileDataPublished, lane);
        }
      }
    }

    const uint32_t metadata_wave = queue_seen_mask & ~metadata_ready_mask;
    if (metadata_wave != 0) {
      for (uint32_t index = 0;
           index < config.num_pes;
           ++index) {
        const uint32_t source_pe = pool_slice_remote_first_pe(
            index, config.my_pe, config.num_pes);
        const uint32_t source_bit = 1U << source_pe;
        if ((metadata_wave & source_bit) == 0)
          continue;
        const PoolSlicePublishBatch batch = receive_batches[source_pe];
        if (lane == 0 &&
            (batch.sequence != sequence ||
             batch.source_pe != source_pe ||
             batch.target_pe != config.my_pe ||
             batch.active_rows > config.route_capacity ||
             batch.route_begin > batch.route_end ||
             batch.route_end > batch.active_rows ||
             batch.flags != POOL_SLICE_BATCH_FLAGS_NONE)) {
          shared_status = batch.sequence != sequence
              ? POOL_SLICE_STATUS_SEQUENCE
              : POOL_SLICE_STATUS_BATCH;
        }
        __syncwarp();

        uint32_t* offsets = offsets_inbox +
            static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
        if (config.local_readers == 1) {
          if (lane == 0) {
            offsets[0] = batch.route_begin;
            offsets[1] = batch.route_end;
          }
          __syncwarp();
        } else {
          pool_slice_get_nbi_warp(
              offsets,
              send_offsets + global_reader_base,
              static_cast<uint64_t>(config.local_readers + 1) *
                  sizeof(uint32_t),
              source_pe,
              config.my_pe,
              lane);
        }
        pool_slice_get_nbi_warp(
            rows_inbox +
                static_cast<uint64_t>(source_pe) * config.route_capacity,
            send_rows + batch.route_begin,
            static_cast<uint64_t>(batch.route_end - batch.route_begin) *
                sizeof(uint32_t),
            source_pe,
            config.my_pe,
            lane);
      }

      // This makes the new metadata wave consumable. It may also retire
      // payload gets issued by an earlier wave, which is useful bounded
      // backpressure rather than a correctness dependency.
      pool_slice_complete_warp(lane);
      inflight_sources = 0;
      ++metadata_waves;
      if (shared_status != POOL_SLICE_STATUS_OK) {
        if (lane == 0) {
          pool_slice_set_status(
              config, static_cast<PoolSliceStatus>(shared_status));
        }
        __syncwarp();
        return;
      }

      if (lane == 0) {
        for (uint32_t source_pe = 0;
             source_pe < config.num_pes &&
                 shared_status == POOL_SLICE_STATUS_OK;
             ++source_pe) {
          const uint32_t source_bit = 1U << source_pe;
          if ((metadata_wave & source_bit) == 0)
            continue;
          const PoolSlicePublishBatch batch = receive_batches[source_pe];
          const uint32_t* offsets = offsets_inbox +
              static_cast<uint64_t>(source_pe) *
                  (config.local_readers + 1);
          const uint32_t source_begin = offsets[0];
          const uint32_t source_end = offsets[config.local_readers];
          if (source_begin != batch.route_begin ||
              source_end != batch.route_end ||
              source_end - source_begin > config.route_capacity) {
            shared_status = POOL_SLICE_STATUS_ROUTE_RANGE;
            break;
          }
          for (uint32_t local_reader = 0;
               local_reader < config.local_readers;
               ++local_reader) {
            const uint32_t begin = offsets[local_reader];
            const uint32_t end = offsets[local_reader + 1];
            if (begin > end || begin < source_begin || end > source_end) {
              shared_status = POOL_SLICE_STATUS_ROUTE_RANGE;
              break;
            }
            const uint32_t count = end - begin;
            const uint64_t base_row = reader_tails[local_reader];
            if (base_row > config.expert_capacity_rows ||
                count > config.expert_capacity_rows - base_row) {
              shared_status = POOL_SLICE_STATUS_CAPACITY;
              break;
            }
            reader_tails[local_reader] = base_row + count;
            PoolSliceReceiveBatch& route = receive_routes[
                static_cast<uint64_t>(local_reader) * config.num_pes +
                source_pe];
            route.sequence = sequence;
            route.base_row = base_row;
            route.source_begin = begin;
            route.row_count = count;
            route.source_pe = source_pe;
            route.local_reader = local_reader;
            route.flags = POOL_SLICE_BATCH_FLAGS_NONE;
            route.reserved_u32[0] = 0;
            route.reserved_u32[1] = 0;
            route.reserved_u32[2] = 0;
          }
        }
      }
      __syncwarp();
      if (shared_status != POOL_SLICE_STATUS_OK) {
        if (lane == 0) {
          pool_slice_set_status(
              config, static_cast<PoolSliceStatus>(shared_status));
        }
        __syncwarp();
        return;
      }

      bool routes_valid = true;
      for (uint32_t source_pe = 0;
           source_pe < config.num_pes;
           ++source_pe) {
        const uint32_t source_bit = 1U << source_pe;
        if ((metadata_wave & source_bit) == 0)
          continue;
        const uint32_t* offsets = offsets_inbox +
            static_cast<uint64_t>(source_pe) * (config.local_readers + 1);
        const uint32_t source_count =
            offsets[config.local_readers] - offsets[0];
        const uint32_t* rows = rows_inbox +
            static_cast<uint64_t>(source_pe) * config.route_capacity;
        for (uint32_t index = lane; index < source_count; index += 32)
          routes_valid = routes_valid && rows[index] < config.token_capacity;
      }
      if (!__all_sync(0xffffffffU, routes_valid)) {
        if (lane == 0)
          pool_slice_set_status(config, POOL_SLICE_STATUS_ROUTE_RANGE);
        __syncwarp();
        return;
      }

      for (uint32_t source_pe = 0;
           source_pe < config.num_pes;
           ++source_pe) {
        const uint32_t source_bit = 1U << source_pe;
        if ((metadata_wave & source_bit) == 0)
          continue;
        const PoolSlicePublishBatch batch = receive_batches[source_pe];
        if (batch.route_begin == batch.route_end) {
          early_issued_mask |= source_bit;
          payload_issued_mask |= source_bit;
        } else {
          needed_data_mask |= source_bit;
        }
      }
      metadata_ready_mask |= metadata_wave;
      made_progress = true;
    }

    if (config.data_stages == 2) {
      const uint32_t early_wave =
          metadata_ready_mask & early_seen_mask & ~early_issued_mask;
      if (early_wave != 0) {
        const uint32_t issued_mask = pool_slice_issue_payload_stage(
            config,
            early_wave,
            0,
            config.early_ready_rows,
            g_events,
            lane);
        early_issued_mask |= early_wave;
        inflight_sources += __popc(issued_mask);
        if (inflight_sources > peak_inflight_sources)
          peak_inflight_sources = inflight_sources;
        made_progress = true;
      }
    }

    const uint32_t payload_wave =
        metadata_ready_mask & data_seen_mask & ~payload_issued_mask;
    if (payload_wave != 0) {
      const uint32_t issued_mask = pool_slice_issue_payload_stage(
          config,
          payload_wave,
          config.data_stages == 2 ? config.early_ready_rows : 0,
          config.token_capacity,
          g_events,
          lane);
      payload_issued_mask |= payload_wave;
      payload_sources += __popc(payload_wave & needed_data_mask);
      inflight_sources += __popc(issued_mask);
      if (inflight_sources > peak_inflight_sources)
        peak_inflight_sources = inflight_sources;
      made_progress = true;
    }

    if (!made_progress)
      __nanosleep(barrierPollSleepCycles);
  }

  // All source batches have been issued, but readers remain blocked until one
  // quiet makes every early NBI get visible in receiver-owned HBM.
  pool_slice_complete_warp(lane);
  pool_slice_record_profile(g_events, poolSliceProfilePayloadDone, lane);

  if (lane == 0) {
    uint64_t received_rows = 0;
    for (uint32_t local_reader = 0;
         local_reader < config.local_readers;
         ++local_reader)
      received_rows += reader_tails[local_reader];
    *group_ready = sequence;
    control[1] = __popc(metadata_ready_mask);
    control[2] = received_rows;
    control[4] = sequence;
    control[5] = metadata_waves;
    control[6] = payload_sources;
    control[7] = peak_inflight_sources;
    __threadfence_system();
    for (uint32_t local_reader = 0;
         local_reader < config.local_readers;
         ++local_reader)
      atomicSub(bars + dispatch_barrier_base + local_reader, 1);
  }
  __syncwarp();
}

static __device__ __noinline__ void pool_slice_return(
    const PoolSliceConfig* config_pointer,
    int* bars,
    uint64_t* signal_array,
    uint32_t compute_barrier_base,
    uint32_t lane) {
  __shared__ PoolSliceConfig shared_config;
  if (config_pointer == nullptr || bars == nullptr || signal_array == nullptr)
    return;
  if (lane == 0)
    shared_config = *config_pointer;
  __syncwarp();
  const PoolSliceConfig& config = shared_config;
  if (!pool_slice_valid_config(config)) {
    if (lane == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
    __syncwarp();
    return;
  }

  const uint64_t sequence = pool_slice_sequence(config);
  const auto* receive_routes =
      reinterpret_cast<const PoolSliceReceiveBatch*>(
          config.receive_routes_address);
  const auto* expert_output = reinterpret_cast<const uint8_t*>(
      config.expert_output_address);
  auto* return_inbox = reinterpret_cast<uint8_t*>(
      config.return_inbox_address);

  pool_slice_wait_barriers_warp(
      bars, compute_barrier_base, config.local_readers, lane);

  for (uint32_t source_pe = 0;
       source_pe < config.num_pes;
       ++source_pe) {
    for (uint32_t local_reader = 0;
         local_reader < config.local_readers;
         ++local_reader) {
      const PoolSliceReceiveBatch route = receive_routes[
          static_cast<uint64_t>(local_reader) * config.num_pes + source_pe];
      if (route.sequence != sequence || route.source_pe != source_pe ||
          route.local_reader != local_reader ||
          route.flags != POOL_SLICE_BATCH_FLAGS_NONE ||
          route.source_begin > config.route_capacity ||
          route.row_count > config.route_capacity - route.source_begin ||
          route.base_row > config.expert_capacity_rows ||
          route.row_count > config.expert_capacity_rows - route.base_row) {
        if (lane == 0)
          pool_slice_set_status(config, POOL_SLICE_STATUS_BATCH);
        __syncwarp();
        return;
      }
      pool_slice_put_nbi_warp(
          return_inbox +
              static_cast<uint64_t>(route.source_begin) * config.row_bytes,
          expert_output +
              static_cast<uint64_t>(local_reader) * config.expert_stride +
              route.base_row * config.expert_row_stride,
          static_cast<uint64_t>(route.row_count) * config.row_bytes,
          source_pe,
          config.my_pe,
          lane);
    }
  }
  pool_slice_complete_warp(lane);

  // One completion signal per producing pool slice covers all reader batches
  // that slice returned to the source PE.
  for (uint32_t source_pe = 0;
       source_pe < config.num_pes;
       ++source_pe) {
    pool_slice_publish_signal(
        signal_array,
        config.return_signal_base + config.my_pe,
        sequence,
        source_pe,
        config.my_pe,
        lane);
  }
  // Every source waits for the complete reciprocal signal set; the preceding
  // payload quiet is the only data-completion fence required.
  pool_slice_wait_signals_warp(
      signal_array,
      config.return_signal_base,
      sequence,
      config.num_pes,
      lane);

  const auto* origins = reinterpret_cast<const uint32_t*>(
      config.send_origin_rows_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);
  for (uint32_t index = 0; index < config.active_rows; ++index) {
    const uint32_t origin = origins[index];
    if (origin >= config.return_capacity_rows) {
      if (lane == 0)
        pool_slice_set_status(config, POOL_SLICE_STATUS_ROUTE_RANGE);
      continue;
    }
    pool_slice_copy_warp(
        returned + static_cast<uint64_t>(origin) * config.return_stride,
        return_inbox + static_cast<uint64_t>(index) * config.row_bytes,
        config.row_bytes,
        lane);
  }
  __syncwarp();
  if (lane == 0) {
    auto* control = reinterpret_cast<unsigned long long*>(
        config.control_address);
    control[3] = config.num_pes;
    __threadfence_system();
  }
  __syncwarp();
}
