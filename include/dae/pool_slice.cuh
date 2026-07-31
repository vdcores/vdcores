#pragma once

#include "context.cuh"
#include "pool_host.cuh"
#include "pool_slice_abi.cuh"
#include "scoped_atomic.cuh"

#if defined(DAE_ENABLE_NVSHMEM)
#include <nvshmem.h>
#include <nvshmemx.h>
#include <non_abi/device/common/nvshmemi_common_device.cuh>
#elif defined(DAE_ENABLE_NCCL_GIN)
#include "pool_gin_transport.cuh"
#else
#error "pool_slice.cuh requires a PoolInst transport backend"
#endif

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

// NVSHMEM 3.4 does not publish a cooperative quiet wrapper, although its
// pinned device implementation exposes the scope-generic primitive. Keep the
// non-ABI dependency in this isolated helper so the pool macro can distribute
// QP completion work without changing any base VDCores operator.
static __device__ __forceinline__ void pool_slice_quiet_block() {
#ifdef __CUDA_ARCH__
#ifdef DAE_ENABLE_NVSHMEM
  nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>();
#else
  pool_gin_flush_block();
#endif
#endif
}

// PoolInst consumes the same local countdown barriers as ordinary VDCores
// memory operators. A pending producer dependency starts above zero; an
// already satisfied dependency is zero.
static __device__ __forceinline__ bool pool_slice_barrier_ready(
    const int* barrier) {
  return *reinterpret_cast<volatile const int*>(barrier) == 0;
}

#if DAE_POOL_SLICE_RAW_SGL && defined(DAE_ENABLE_NVSHMEM)
#include "pool_ibgda_sgl.cuh"
#elif DAE_POOL_SLICE_RAW_SGL && defined(DAE_ENABLE_NCCL_GIN)
#include "pool_gin_gdaki_sgl.cuh"
#endif

static __device__ __forceinline__ void pool_slice_set_status(
    const PoolSliceConfig& config, PoolSliceStatus status) {
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  atomicCAS(
      reinterpret_cast<unsigned long long*>(control),
      static_cast<unsigned long long>(POOL_SLICE_STATUS_OK),
      static_cast<unsigned long long>(status));
}

static __device__ __forceinline__ uint64_t pool_slice_sequence(
    const PoolSliceConfig& config) {
  const auto* sequence = reinterpret_cast<const unsigned long long*>(
      config.sequence_address);
  return atomicAdd(
      const_cast<unsigned long long*>(sequence),
      static_cast<unsigned long long>(0));
}

static __device__ __forceinline__ void pool_slice_wait_value_warp(
    const uint64_t* address, uint64_t expected, uint32_t lane) {
  uint32_t ready = 0;
  while (ready == 0) {
    if (lane == 0)
      ready = dae_atomic_load_acquire_gpu(address) >= expected;
    ready = __shfl_sync(0xffffffffU, ready, 0);
    if (ready == 0)
      __nanosleep(barrierPollSleepCycles);
  }
}

static __device__ __forceinline__ void pool_slice_wait_generation_warp(
    const uint64_t* generations,
    uint32_t count,
    uint64_t expected,
    uint32_t lane) {
  for (uint32_t base = 0; base < count; base += 32) {
    const uint32_t index = base + lane;
    bool ready = index >= count;
    while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
      if (!ready)
        ready = dae_atomic_load_acquire_gpu(generations + index) >= expected;
      if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
  }
}

// Self-target signals carry same-GPU message ordering. Remote signals stay in
// the NVSHMEM domain so their transport and atomicity contracts remain intact.
static __device__ __forceinline__ void pool_slice_signal_release_local(
    uint64_t* address, uint64_t value) {
  dae_atomic_store_release_gpu(address, value);
}

static __device__ __forceinline__ uint64_t pool_slice_signal_fetch(
    uint64_t* address, bool local) {
#ifdef DAE_ENABLE_NVSHMEM
  return local ? dae_atomic_load_acquire_gpu(address)
               : nvshmem_signal_fetch(address);
#else
  (void)local;
  return dae_atomic_load_acquire_gpu(address);
#endif
}

// The hot path deliberately supports fixed-width, aligned LLM rows.
static __device__ __forceinline__ void pool_slice_copy_warp(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t lane) {
  auto* dst = reinterpret_cast<uint4*>(destination);
  const auto* src = reinterpret_cast<const uint4*>(source);
  const uint64_t vectors = bytes / sizeof(uint4);
  constexpr uint64_t copy_ilp = 4;
  uint64_t index = lane;
  for (; index + (copy_ilp - 1) * 32 < vectors; index += copy_ilp * 32) {
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

// Copy one vector shard of an aligned row. Weighted combine assigns several
// warps to each source token; this keeps the one-contributor fast path from
// redundantly copying the complete row in every warp.
static __device__ __forceinline__ void pool_slice_copy_warp_shard(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t vector_shard,
    uint32_t vector_shards,
    uint32_t lane) {
  auto* dst = reinterpret_cast<uint4*>(destination);
  const auto* src = reinterpret_cast<const uint4*>(source);
  const uint64_t vectors = bytes / sizeof(uint4);
  const uint64_t stride = static_cast<uint64_t>(vector_shards) * 32;
  constexpr uint64_t copy_ilp = 4;
  uint64_t index = static_cast<uint64_t>(vector_shard) * 32 + lane;
  for (; index + (copy_ilp - 1) * stride < vectors;
       index += copy_ilp * stride) {
    const uint4 value0 = src[index];
    const uint4 value1 = src[index + stride];
    const uint4 value2 = src[index + 2 * stride];
    const uint4 value3 = src[index + 3 * stride];
    dst[index] = value0;
    dst[index + stride] = value1;
    dst[index + 2 * stride] = value2;
    dst[index + 3 * stride] = value3;
  }
  for (; index < vectors; index += stride)
    dst[index] = src[index];
}

// Register-only two-source combine for the common spread route on two PEs.
// Keeping this separate from the arbitrary-fan-in fallback prevents its
// accumulator state from creating a per-lane local-memory frame.
static __device__ __forceinline__ void pool_slice_add_bf16_warp_shard(
    void* destination,
    const void* source0,
    const void* source1,
    uint64_t bytes,
    uint32_t vector_shard,
    uint32_t vector_shards,
    uint32_t lane) {
  auto* dst = reinterpret_cast<__nv_bfloat162*>(destination);
  const auto* src0 = reinterpret_cast<const __nv_bfloat162*>(source0);
  const auto* src1 = reinterpret_cast<const __nv_bfloat162*>(source1);
  const uint64_t elements = bytes / sizeof(__nv_bfloat162);
  const uint64_t stride = static_cast<uint64_t>(vector_shards) * 32;
  for (uint64_t element =
           static_cast<uint64_t>(vector_shard) * 32 + lane;
       element < elements;
       element += 4 * stride) {
#pragma unroll
    for (uint32_t item = 0; item < 4; ++item) {
      const uint64_t item_element = element + item * stride;
      if (item_element < elements) {
        dst[item_element] = __hadd2(
            src0[item_element], src1[item_element]);
      }
    }
  }
}

static __device__ __forceinline__ void pool_slice_copy_block(
    void* destination,
    const void* source,
    uint64_t bytes,
    uint32_t thread_id) {
  auto* dst = reinterpret_cast<uint4*>(destination);
  const auto* src = reinterpret_cast<const uint4*>(source);
  const uint64_t vectors = bytes / sizeof(uint4);
  const uint64_t stride = blockDim.x;
  constexpr uint64_t copy_ilp = 4;
  uint64_t index = thread_id;
  for (; index + (copy_ilp - 1) * stride < vectors;
       index += copy_ilp * stride) {
    const uint4 value0 = src[index];
    const uint4 value1 = src[index + stride];
    const uint4 value2 = src[index + 2 * stride];
    const uint4 value3 = src[index + 3 * stride];
    dst[index] = value0;
    dst[index + stride] = value1;
    dst[index + 2 * stride] = value2;
    dst[index + 3 * stride] = value3;
  }
  for (; index < vectors; index += stride)
    dst[index] = src[index];
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
#ifdef DAE_ENABLE_NVSHMEM
  nvshmemx_putmem_nbi_warp(
      destination, source, static_cast<size_t>(bytes), target_pe);
#else
  pool_gin_put_warp(
      destination,
      source,
      static_cast<size_t>(bytes),
      target_pe,
      false);
#endif
}

static __device__ __forceinline__ void pool_slice_signal_set_remote(
    uint64_t* destination,
    uint64_t value,
    uint32_t target_pe) {
#ifdef DAE_ENABLE_NVSHMEM
  nvshmemx_signal_op(
      destination, value, NVSHMEM_SIGNAL_SET, target_pe);
#else
  pool_gin_set_thread(destination, value, target_pe);
#endif
}

static __device__ __forceinline__ void pool_slice_publish_phase_parallel(
    uint64_t* signal_array,
    uint32_t signal_id,
    uint64_t value,
    uint32_t my_pe,
    uint32_t num_pes,
    uint32_t lane,
    const PoolSlicePublishBatch* empty_batches,
    uint64_t sequence) {
  if (num_pes <= 2) {
    for (uint32_t index = 0; index < num_pes; ++index) {
      const uint32_t unwrapped = index + my_pe + 1;
      const uint32_t target_pe =
          unwrapped >= num_pes ? unwrapped - num_pes : unwrapped;
      __syncwarp();
      const bool empty = empty_batches != nullptr &&
          empty_batches[target_pe].sequence == sequence &&
          empty_batches[target_pe].flags == POOL_SLICE_BATCH_FLAGS_NONE &&
          empty_batches[target_pe].active_rows == 0;
      if (lane == 0 && !empty) {
        if (target_pe == my_pe) {
          pool_slice_signal_release_local(signal_array + signal_id, value);
        } else {
          pool_slice_signal_set_remote(
              signal_array + signal_id, value, target_pe);
        }
      }
    }
    __syncwarp();
    return;
  }

  __syncwarp();
  if (lane < num_pes) {
    const uint32_t unwrapped = lane + my_pe + 1;
    const uint32_t target_pe =
        unwrapped >= num_pes ? unwrapped - num_pes : unwrapped;
    const bool empty = empty_batches != nullptr &&
        empty_batches[target_pe].sequence == sequence &&
        empty_batches[target_pe].flags == POOL_SLICE_BATCH_FLAGS_NONE &&
        empty_batches[target_pe].active_rows == 0;
    if (empty) {
      // A valid zero-row descriptor already names completion of this phase.
    } else if (target_pe == my_pe) {
      pool_slice_signal_release_local(signal_array + signal_id, value);
    } else {
      pool_slice_signal_set_remote(
          signal_array + signal_id, value, target_pe);
    }
  }
  __syncwarp();
}

static __device__ __forceinline__ uint32_t pool_slice_pe_mask(
    uint32_t num_pes) {
  return num_pes == 32 ? 0xffffffffU : (1U << num_pes) - 1U;
}

static __device__ __forceinline__ uint32_t pool_slice_remote_first_pe(
    uint32_t index,
    uint32_t my_pe,
    uint32_t num_pes) {
  const uint32_t pe = index + my_pe + 1;
  return pe >= num_pes ? pe - num_pes : pe;
}

static __device__ __forceinline__ void pool_slice_record_profile(
    uint64_t* g_events, uint32_t event, uint32_t lane) {
  if (g_events != nullptr && lane == 0) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents + event] =
        cuda::ptx::get_sreg_globaltimer();
  }
  __syncwarp();
}

// Streaming dispatch derives G from the current target's unique activation
// output. Reader-sharded gathers expose one CTA task per local expert, so one
// group already creates substantial destination parallelism. Aim for groups
// near 256 KiB through four PEs and 512 KiB at eight or more PEs, with no
// more than 32 rows.  The larger 8-PE group avoids doubling ordered queue-head
// work after every source shard has already shrunk to about 32 rows.
static __device__ __forceinline__ uint32_t pool_slice_stream_group_count(
    uint32_t active_rows,
    uint32_t row_bytes,
    uint32_t max_groups,
    uint32_t num_pes) {
  if (active_rows == 0)
    return 0;
  constexpr uint64_t target_group_bytes_low_pe = 256ULL * 1024;
  constexpr uint64_t target_group_bytes_high_pe = 512ULL * 1024;
  const uint64_t target_group_bytes = num_pes >= 8
      ? target_group_bytes_high_pe
      : target_group_bytes_low_pe;
  constexpr uint32_t target_group_rows = 32;
  const uint64_t payload_bytes =
      static_cast<uint64_t>(active_rows) * row_bytes;
  const uint64_t byte_groups =
      1 + (payload_bytes - 1) / target_group_bytes;
  const uint64_t row_groups =
      (static_cast<uint64_t>(active_rows) + target_group_rows - 1) /
      target_group_rows;
  uint64_t groups = byte_groups > row_groups ? byte_groups : row_groups;
  groups = groups < max_groups ? groups : max_groups;
  groups = groups < active_rows ? groups : active_rows;
  return static_cast<uint32_t>(groups);
}

static __device__ __forceinline__ uint32_t
pool_slice_stream_dispatch_worker_count(
    const PoolSliceConfig& config,
    uint64_t* control) {
  if (config.pool_count == 1)
    return 1;
  const uint64_t send_total = dae_atomic_load_relaxed_gpu(
      control + poolSliceControlStreamSendTotal);
  const uint64_t gather_work = control[4] *
      static_cast<uint64_t>(config.num_pes) * config.local_readers;
  uint64_t useful_workers = gather_work + send_total;
  useful_workers = useful_workers == 0 ? 1 : useful_workers;
  useful_workers = useful_workers < config.pool_count
      ? useful_workers
      : config.pool_count - 1;
  return static_cast<uint32_t>(useful_workers);
}

static __device__ __forceinline__ void pool_slice_stream_group_range(
    uint32_t active_rows,
    uint32_t row_bytes,
    uint32_t group_count,
    uint32_t group,
    uint32_t* row_begin,
    uint32_t* row_end) {
  if (group_count == 1) {
    *row_begin = 0;
    *row_end = active_rows;
    return;
  }
  *row_begin = static_cast<uint32_t>(
      static_cast<uint64_t>(active_rows) * group / group_count);
  *row_end = static_cast<uint32_t>(
      static_cast<uint64_t>(active_rows) * (group + 1) / group_count);
}

static __device__ __forceinline__ uint64_t* pool_slice_stream_data_ready(
    uint64_t* control,
    uint32_t source_pe,
    uint32_t group,
    uint32_t payload_warp) {
  return control + poolSliceControlStreamDataReady +
      (static_cast<uint64_t>(source_pe) * poolSliceMaxDataGroups + group) *
          poolSliceCompletionSlots +
      payload_warp;
}

static __device__ __forceinline__ uint64_t
pool_slice_stream_data_progress(
    uint64_t sequence, uint32_t completed_segments) {
  return sequence * poolSliceRawSglProgressStride + completed_segments;
}

static __device__ __forceinline__ uint32_t
pool_slice_stream_data_segments(uint32_t row_begin, uint32_t row_end) {
  const uint32_t rows = row_end - row_begin;
  return (rows + poolSliceRawSglWidth - 1) / poolSliceRawSglWidth;
}

static __device__ __forceinline__ void
pool_slice_stream_wait_data_progress_warp(
    uint64_t* control,
    uint32_t source_pe,
    uint32_t ready_slot,
    uint64_t expected,
    uint32_t lane) {
  uint32_t ready = 0;
  while (ready == 0) {
    if (lane == 0) {
      ready = pool_slice_signal_fetch(
                  pool_slice_stream_data_ready(
                      control, source_pe, ready_slot, 0),
                  false) >= expected;
    }
    ready = __shfl_sync(0xffffffffU, ready, 0);
    if (ready == 0)
      __nanosleep(barrierPollSleepCycles);
  }
}

// Queue entries are interleaved by slot. A producer therefore sends only the
// slot rounds that contain RESERVE/COPY/END instructions; the consumer still
// follows explicit in-order messages and never derives the producer's G.
static __device__ __forceinline__ uint32_t
pool_slice_stream_envelope_bytes(
    uint32_t active_rows,
    uint32_t row_bytes,
    uint32_t group_limit,
    uint32_t num_pes) {
  const uint32_t groups = pool_slice_stream_group_count(
      active_rows, row_bytes, group_limit, num_pes);
  const uint32_t slot_rounds = 2 + (groups + 1) / 2;
  return sizeof(PoolSlicePublishBatch) +
      slot_rounds * poolSliceMaxStreamQueues *
          sizeof(PoolSliceQueueEntry);
}

static __device__ __forceinline__ uint32_t pool_slice_stream_queue_index(
    uint32_t source_pe, uint32_t queue) {
  return source_pe * poolSliceMaxStreamQueues + queue;
}

static __device__ __forceinline__ uint64_t
pool_slice_stream_packet_capacity_bytes(uint32_t route_capacity) {
  return (sizeof(PoolSliceMetadataEnvelope) +
          static_cast<uint64_t>(route_capacity) * sizeof(uint32_t) + 15) &
      ~15ULL;
}

static __device__ __forceinline__ uint64_t* pool_slice_stream_queue_head(
    uint64_t* control, uint32_t source_pe, uint32_t queue) {
  return control + poolSliceControlStreamQueueHead +
      pool_slice_stream_queue_index(source_pe, queue);
}

static __device__ __forceinline__ uint64_t* pool_slice_stream_queue_claim(
    uint64_t* control, uint32_t source_pe, uint32_t queue) {
  return control + poolSliceControlStreamQueueClaim +
      pool_slice_stream_queue_index(source_pe, queue);
}

static __device__ __forceinline__ PoolSliceMetadataEnvelope*
pool_slice_stream_envelope(
    PoolSlicePublishBatch* storage,
    uint32_t peer,
    uint32_t route_capacity) {
  const uint64_t packet_bytes =
      pool_slice_stream_packet_capacity_bytes(route_capacity);
  return reinterpret_cast<PoolSliceMetadataEnvelope*>(
      reinterpret_cast<uint8_t*>(storage) + peer * packet_bytes);
}

static __device__ __forceinline__ const PoolSliceMetadataEnvelope*
pool_slice_stream_envelope(
    const PoolSlicePublishBatch* storage,
    uint32_t peer,
    uint32_t route_capacity) {
  const uint64_t packet_bytes =
      pool_slice_stream_packet_capacity_bytes(route_capacity);
  return reinterpret_cast<const PoolSliceMetadataEnvelope*>(
      reinterpret_cast<const uint8_t*>(storage) + peer * packet_bytes);
}

static __device__ __forceinline__ PoolSlicePublishBatch*
pool_slice_stream_batch(
    PoolSlicePublishBatch* storage,
    uint32_t peer,
    uint32_t route_capacity) {
  return &pool_slice_stream_envelope(storage, peer, route_capacity)->batch;
}

static __device__ __forceinline__ const PoolSlicePublishBatch*
pool_slice_stream_batch(
    const PoolSlicePublishBatch* storage,
    uint32_t peer,
    uint32_t route_capacity) {
  return &pool_slice_stream_envelope(storage, peer, route_capacity)->batch;
}

static __device__ __forceinline__ PoolSliceQueueEntry*
pool_slice_stream_queue_entry(
    PoolSlicePublishBatch* storage,
    uint32_t peer,
    uint32_t queue,
    uint32_t slot,
    uint32_t route_capacity) {
  return &pool_slice_stream_envelope(storage, peer, route_capacity)
              ->queues[slot][queue];
}

static __device__ __forceinline__ const PoolSliceQueueEntry*
pool_slice_stream_queue_entry(
    const PoolSlicePublishBatch* storage,
    uint32_t peer,
    uint32_t queue,
    uint32_t slot,
    uint32_t route_capacity) {
  return &pool_slice_stream_envelope(storage, peer, route_capacity)
              ->queues[slot][queue];
}

static __device__ __forceinline__ uint32_t* pool_slice_stream_route_words(
    PoolSlicePublishBatch* storage,
    uint32_t peer,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch& batch) {
  auto* envelope = reinterpret_cast<uint8_t*>(
      pool_slice_stream_envelope(storage, peer, config.route_capacity));
  return reinterpret_cast<uint32_t*>(
      envelope + pool_slice_stream_envelope_bytes(
                     batch.active_rows,
                     config.row_bytes,
                     config.group_limit,
                     config.num_pes));
}

static __device__ __forceinline__ const uint32_t*
pool_slice_stream_route_words(
    const PoolSlicePublishBatch* storage,
    uint32_t peer,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch& batch) {
  const auto* envelope = reinterpret_cast<const uint8_t*>(
      pool_slice_stream_envelope(storage, peer, config.route_capacity));
  return reinterpret_cast<const uint32_t*>(
      envelope + pool_slice_stream_envelope_bytes(
                     batch.active_rows,
                     config.row_bytes,
                     config.group_limit,
                     config.num_pes));
}

static __device__ __forceinline__ uint64_t
pool_slice_stream_queue_retired_mask(uint32_t num_pes) {
  const uint32_t queues = num_pes * poolSliceMaxStreamQueues;
  return queues == 64 ? ~0ULL : (1ULL << queues) - 1;
}

static __device__ __forceinline__ uint32_t pool_slice_stream_route_lower_bound(
    const uint32_t* rows,
    uint32_t begin,
    uint32_t count,
    uint32_t compact_row) {
  uint32_t low = 0;
  uint32_t high = count;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    const uint32_t value = rows[begin + middle] & 0xffffU;
    if (value < compact_row)
      low = middle + 1;
    else
      high = middle;
  }
  return low;
}

// Count the exact nonempty dispatch DATA shards that contribute to one local
// reader. The source metadata is sufficient: activation readiness and queue
// retirement are deliberately absent from this calculation.
static __device__ __forceinline__ uint32_t
pool_slice_stream_reader_data_groups(
    const PoolSlicePublishBatch& batch,
    const uint32_t* source_routes,
    uint32_t local_reader,
    const PoolSliceConfig& config) {
  uint32_t reader_begin = 0;
  for (uint32_t reader = 0; reader < local_reader; ++reader)
    reader_begin += batch.reader_counts[reader];
  const uint32_t reader_count = batch.reader_counts[local_reader];
  const uint32_t groups = pool_slice_stream_group_count(
      batch.active_rows,
      config.row_bytes,
      config.group_limit,
      config.num_pes);
  uint32_t nonempty = 0;
  for (uint32_t group = 0; group < groups; ++group) {
    uint32_t row_begin = 0;
    uint32_t row_end = 0;
    pool_slice_stream_group_range(
        batch.active_rows,
        config.row_bytes,
        groups,
        group,
        &row_begin,
        &row_end);
    const uint32_t begin = pool_slice_stream_route_lower_bound(
        source_routes, reader_begin, reader_count, row_begin);
    const uint32_t end = pool_slice_stream_route_lower_bound(
        source_routes, reader_begin, reader_count, row_end);
    nonempty += begin < end;
  }
  return nonempty;
}

static __device__ __forceinline__ PoolSliceQueueEntry
pool_slice_stream_make_queue_entry(
    uint64_t sequence,
    uint32_t slot,
    uint32_t opcode,
    uint32_t row_begin,
    uint32_t row_end,
    uint32_t ready_slot,
    uint32_t flags) {
  return PoolSliceQueueEntry{
      sequence,
      slot,
      opcode,
      row_begin,
      row_end,
      ready_slot,
      flags};
}

// Materialize a compact instruction stream for each destination.  Grouping is
// a producer concern only: DATA carries its exact compact-row interval,
// and END is the sole consumer-visible termination condition.
static __device__ __noinline__ void pool_slice_stream_build_queues(
    uint32_t target_pe,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch& batch,
    PoolSlicePublishBatch* send_batches) {
  constexpr uint32_t queue_count = poolSliceMaxStreamQueues;
  const uint32_t groups = pool_slice_stream_group_count(
      batch.active_rows,
      config.row_bytes,
      config.group_limit,
      config.num_pes);
  for (uint32_t queue = 0; queue < queue_count; ++queue) {
    uint32_t slot = 0;
    if (queue == 0) {
      *pool_slice_stream_queue_entry(
          send_batches,
          target_pe,
          queue,
          slot,
          config.route_capacity) =
          pool_slice_stream_make_queue_entry(
              batch.sequence,
              slot,
              POOL_SLICE_QUEUE_RESERVE_ROUTES,
              0,
              batch.active_rows,
              UINT32_MAX,
              batch.flags);
      ++slot;
    }
    for (uint32_t group = queue; group < groups; group += queue_count) {
      uint32_t row_begin = 0;
      uint32_t row_end = 0;
      pool_slice_stream_group_range(
          batch.active_rows,
          config.row_bytes,
          groups,
          group,
          &row_begin,
          &row_end);
      *pool_slice_stream_queue_entry(
          send_batches,
          target_pe,
          queue,
          slot,
          config.route_capacity) =
          pool_slice_stream_make_queue_entry(
              batch.sequence,
              slot,
              POOL_SLICE_QUEUE_DATA,
              row_begin,
              row_end,
              group,
              batch.flags);
      ++slot;
    }
    *pool_slice_stream_queue_entry(
        send_batches,
        target_pe,
        queue,
        slot,
        config.route_capacity) =
        pool_slice_stream_make_queue_entry(
            batch.sequence,
            slot,
            POOL_SLICE_QUEUE_END,
            0,
            0,
            UINT32_MAX,
            batch.flags);
  }
}

static __device__ __forceinline__ bool pool_slice_stream_decode_send_task(
    uint32_t task,
    const PoolSliceConfig& config,
    const uint32_t* send_token_counts,
    uint32_t* target_pe,
    uint32_t* group) {
  uint32_t cursor = 0;
  // Send tasks are remote-only and remote-first. A self-source dynamic read
  // acquires its ordinary writer chunk inside gather, so it needs neither a
  // synthetic send task nor a data-ready message. Target-major enumeration
  // preserves the independent metadata/data-plane placement while assigning
  // every runtime-sized group exactly once.
  for (uint32_t index = 0; index + 1 < config.num_pes; ++index) {
    const uint32_t target = pool_slice_remote_first_pe(
        index, config.my_pe, config.num_pes);
    const uint32_t groups = pool_slice_stream_group_count(
        send_token_counts[target],
        config.row_bytes,
        config.group_limit,
        config.num_pes);
    for (uint32_t candidate_group = 0;
         candidate_group < groups;
         ++candidate_group) {
      if (cursor == task) {
        *target_pe = target;
        *group = candidate_group;
        return true;
      }
      ++cursor;
    }
  }
  return false;
}

#if DAE_POOL_SLICE_RAW_SGL
#define DAE_POOL_SLICE_PUBLIC_FALLBACK_QUALIFIER __noinline__
#else
#define DAE_POOL_SLICE_PUBLIC_FALLBACK_QUALIFIER __forceinline__
#endif

// Keep the ordinary NVSHMEM row-PUT path inline in the default build. The raw
// SGL specialization makes it a cold noinline fallback so its loop state does
// not inflate the hot raw sender's live range or spill traffic.
static __device__ DAE_POOL_SLICE_PUBLIC_FALLBACK_QUALIFIER void
pool_slice_stream_put_rows_public(
    uint32_t target_pe,
    uint32_t row_begin,
    uint32_t row_end,
    const PoolSliceConfig& config,
    int* bars,
    uint32_t write_barrier,
    uint32_t* shared_status,
    uint32_t thread_id) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  const uint32_t send_warps = blockDim.x / 32;
  const uint32_t group_rows = row_end - row_begin;
  const uint32_t warp_row_begin = row_begin + static_cast<uint32_t>(
      static_cast<uint64_t>(group_rows) * warp / send_warps);
  const uint32_t warp_row_end = row_begin + static_cast<uint32_t>(
      static_cast<uint64_t>(group_rows) * (warp + 1) / send_warps);
  const auto* token_pool =
      reinterpret_cast<const uint8_t*>(config.token_pool_address);
  auto* delivery_pool =
      reinterpret_cast<uint8_t*>(config.delivery_pool_address);
  const auto* target_rows = reinterpret_cast<const uint32_t*>(
      config.send_token_rows_address) +
      static_cast<uint64_t>(target_pe) * config.token_capacity;

  uint32_t waited_chunk = UINT32_MAX;
  for (uint32_t packed_row = warp_row_begin;
       packed_row < warp_row_end;) {
    uint32_t source_row = 0;
    uint32_t run_rows = 1;
    if (lane == 0) {
      source_row = target_rows[packed_row];
      if (source_row < config.token_capacity) {
        const uint32_t source_chunk =
            source_row / config.write_chunk_rows;
        while (packed_row + run_rows < warp_row_end) {
          const uint32_t candidate = target_rows[packed_row + run_rows];
          if (candidate != source_row + run_rows ||
              candidate / config.write_chunk_rows != source_chunk)
            break;
          ++run_rows;
        }
      }
    }
    source_row = __shfl_sync(0xffffffffU, source_row, 0);
    run_rows = __shfl_sync(0xffffffffU, run_rows, 0);
    if (source_row >= config.token_capacity) {
      if (lane == 0) {
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
      }
      ++packed_row;
      continue;
    }
    const uint32_t chunk = source_row / config.write_chunk_rows;
    if (chunk != waited_chunk) {
      uint32_t write_ready = 0;
      while (write_ready == 0) {
        if (lane == 0)
          write_ready = pool_slice_barrier_ready(
              bars + write_barrier + chunk);
        write_ready = __shfl_sync(0xffffffffU, write_ready, 0);
        if (write_ready == 0)
          __nanosleep(barrierPollSleepCycles);
      }
      waited_chunk = chunk;
    }
    uint8_t* destination = delivery_pool +
        (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
         packed_row) *
            config.row_bytes;
    const uint8_t* source = token_pool +
        static_cast<uint64_t>(source_row) * config.row_bytes;
    const size_t bytes = static_cast<size_t>(run_rows) * config.row_bytes;
#ifdef DAE_ENABLE_NVSHMEM
    nvshmemx_putmem_nbi_warp(destination, source, bytes, target_pe);
#else
    // All row runs in this group share a GIN context. Defer its doorbell until
    // the exact readiness generation below, allowing noncontiguous source
    // regions to travel as one submitted request list.
    pool_gin_put_warp(
        destination, source, bytes, target_pe, true);
#endif
    packed_row += run_rows;
  }
}

#undef DAE_POOL_SLICE_PUBLIC_FALLBACK_QUALIFIER

// One CTA owns one dynamic group. Every warp issues direct puts from the
// authoritative source token slots; no source-side activation staging is
// materialized. Completion scope is compile-time static: a CTA-mapped build
// posts one same-QP generation after all warp WQEs, while a warp-mapped build
// couples each active warp's final run to its own generation. Neither path
// performs a transport-wide completion sweep.
template <bool HostDataPlane>
static __device__ __noinline__ void pool_slice_stream_send_group(
    uint32_t target_pe,
    uint32_t group,
    const PoolSliceConfig& config,
    const PoolSliceHostConfig* host_config,
    int* bars,
    uint32_t write_barrier,
    uint64_t* control,
    uint64_t* g_events,
    uint64_t sequence,
    uint32_t* shared_status,
    uint32_t* shared_first_payload,
  uint32_t thread_id) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  const auto* send_token_rows =
      reinterpret_cast<const uint32_t*>(config.send_token_rows_address);
  const auto* send_token_counts =
      reinterpret_cast<const uint32_t*>(config.send_token_counts_address);
  // Send-task decoding emits remote targets only. Keeping that invariant
  // explicit removes the local/raw transport matrix from the hot helper.
  if (target_pe >= config.num_pes || target_pe == config.my_pe) {
    if (thread_id == 0) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
    }
    return;
  }
  const uint32_t token_count = send_token_counts[target_pe];
  const uint32_t group_count = pool_slice_stream_group_count(
      token_count,
      config.row_bytes,
      config.group_limit,
      config.num_pes);
  if (group_count == 0 || group >= group_count ||
      token_count > config.token_capacity) {
    if (thread_id == 0) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
    }
    return;
  }

  uint32_t row_begin = 0;
  uint32_t row_end = 0;
  pool_slice_stream_group_range(
      token_count,
      config.row_bytes,
      group_count,
      group,
      &row_begin,
      &row_end);
  if constexpr (HostDataPlane) {
    if (thread_id == 0) {
      const auto* generations = reinterpret_cast<const uint64_t*>(
          host_config->producer_generations_address);
      while (dae_atomic_load_acquire_gpu(
                 generations + config.num_pes) < sequence)
        __nanosleep(barrierPollSleepCycles);
    }
    __syncthreads();
  }
  if (thread_id == 0 && atomicCAS(shared_first_payload, 0U, 1U) == 0U &&
      g_events != nullptr) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
             poolSliceProfileFirstPayload] =
        cuda::ptx::get_sreg_globaltimer();
  }
  __syncthreads();

  const uint32_t* target_rows = send_token_rows +
      static_cast<uint64_t>(target_pe) * config.token_capacity;
  if constexpr (HostDataPlane) {
    for (uint32_t packed_row = row_begin + thread_id;
         packed_row < row_end;
         packed_row += blockDim.x) {
      const uint32_t source_row = target_rows[packed_row];
      if (source_row >= config.token_capacity) {
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
        continue;
      }
      const uint32_t chunk = source_row / config.write_chunk_rows;
      while (!pool_slice_barrier_ready(bars + write_barrier + chunk))
        __nanosleep(barrierPollSleepCycles);
    }
    __syncthreads();
    if (warp == 0) {
      const auto* peers = reinterpret_cast<const PoolSliceHostPeer*>(
          host_config->peers_address);
      auto* generations = reinterpret_cast<uint64_t*>(
          host_config->producer_generations_address);
      const PoolSliceHostPeer peer = peers[target_pe];
      const uint64_t generation = pool_host_reserve_generation_warp(
          generations + target_pe, lane);
      uint64_t* local_ready = pool_slice_stream_data_ready(
          control, config.my_pe, group, 0);
      const uint64_t ready_offset =
          reinterpret_cast<uint64_t>(local_ready) - config.control_address;
      pool_host_publish_request_warp<true>(
          reinterpret_cast<HostSglRingMemory*>(peer.ring_memory),
          generation,
          target_rows,
          row_begin,
          row_end - row_begin,
          host_config->local_lkey,
          static_cast<uint32_t>(peer.remote_rkey),
          config.token_pool_address,
          config.row_bytes,
          config.row_bytes,
          peer.remote_delivery_address +
              (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
               row_begin) * config.row_bytes,
          peer.remote_control_address + ready_offset,
          poolSliceRawSgl
              ? pool_slice_stream_data_progress(
                    sequence,
                    pool_slice_stream_data_segments(row_begin, row_end))
              : sequence,
          lane);
    }
    __syncthreads();
    if (thread_id == 0) {
      atomicAdd(
          reinterpret_cast<unsigned long long*>(
              control + poolSliceControlStreamSendDone),
          1ULL);
    }
    __syncthreads();
    return;
  }

#if DAE_POOL_SLICE_RAW_SGL && defined(DAE_ENABLE_NVSHMEM) && \
    defined(__CUDA_ARCH__)
  const auto* token_pool =
      reinterpret_cast<const uint8_t*>(config.token_pool_address);
  auto* delivery_pool =
      reinterpret_cast<uint8_t*>(config.delivery_pool_address);
  if (warp == 0) {
    pool_ibgda_sgl_put_rows_warp(
        delivery_pool +
            (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
             row_begin) * config.row_bytes,
        pool_slice_stream_data_ready(
            control, config.my_pe, group, 0),
        token_pool,
        target_rows,
        row_begin,
        row_end,
        config.row_bytes,
        config.write_chunk_rows,
        target_pe,
        bars,
        write_barrier,
        sequence,
        lane);
  }
#elif DAE_POOL_SLICE_RAW_SGL && defined(DAE_ENABLE_NCCL_GIN) && \
    defined(__CUDA_ARCH__)
  const auto* token_pool =
      reinterpret_cast<const uint8_t*>(config.token_pool_address);
  auto* delivery_pool =
      reinterpret_cast<uint8_t*>(config.delivery_pool_address);
  if (warp == 0) {
    pool_gin_gdaki_sgl_put_rows_warp(
        delivery_pool +
            (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
             row_begin) * config.row_bytes,
        pool_slice_stream_data_ready(
            control, config.my_pe, group, 0),
        token_pool,
        target_rows,
        row_begin,
        row_end,
        config.row_bytes,
        config.write_chunk_rows,
        target_pe,
        bars,
        write_barrier,
        sequence,
        lane);
  }
#else
  const uint32_t send_warps = blockDim.x / 32;
  const uint32_t group_rows = row_end - row_begin;
#ifdef DAE_ENABLE_NVSHMEM
  const uint32_t warp_row_begin = row_begin + static_cast<uint32_t>(
      static_cast<uint64_t>(group_rows) * warp / send_warps);
  const uint32_t warp_row_end = row_begin + static_cast<uint32_t>(
      static_cast<uint64_t>(group_rows) * (warp + 1) / send_warps);
#else
  (void)send_warps;
  (void)group_rows;
#endif
  pool_slice_stream_put_rows_public(
      target_pe,
      row_begin,
      row_end,
      config,
      bars,
      write_barrier,
      shared_status,
      thread_id);

#ifdef DAE_ENABLE_NVSHMEM
  if constexpr (poolSliceWarpQpCompletion) {
    // Each nonempty warp owns one statically mapped QP. Its inline generation
    // write follows every payload WQE on that QP, so a dynamic group can skip
    // arbitrary sequence values without an atomic RMW or source-side state.
    if (lane == 0 && warp_row_begin < warp_row_end) {
      nvshmem_uint64_p(
          pool_slice_stream_data_ready(
              control, config.my_pe, group, warp),
          sequence,
          target_pe);
    }
  }
#endif
#endif

  __syncthreads();
  if (thread_id == 0) {
#if !DAE_POOL_SLICE_RAW_SGL
    if constexpr (!poolSliceWarpQpCompletion) {
      // The static CTA-QP policy maps every producer warp to one ordered RC
      // context. All payload WQEs have been posted before this thread reaches
      // the CTA barrier, so the later same-QP generation names precisely this
      // group without a transport-wide quiet or a system fence.
      const uint64_t ready_value = poolSliceRawSgl
          ? pool_slice_stream_data_progress(
                sequence,
                pool_slice_stream_data_segments(row_begin, row_end))
          : sequence;
#ifdef DAE_ENABLE_NVSHMEM
      nvshmem_uint64_p(
          pool_slice_stream_data_ready(
              control, config.my_pe, group, 0),
          ready_value,
          target_pe);
#else
      pool_gin_set_thread(
          pool_slice_stream_data_ready(
              control, config.my_pe, group, 0),
          ready_value,
          target_pe);
#endif
    }
#endif
    atomicAdd(
        reinterpret_cast<unsigned long long*>(
            control + poolSliceControlStreamSendDone),
        1ULL);
  }
  __syncthreads();
}

static __device__ __forceinline__ uint32_t
pool_slice_reduce_add_source_shards(
    const PoolSliceConfig& config, uint32_t source_pe);
static __device__ __forceinline__ uint32_t
pool_slice_reduce_add_group_count(
    uint32_t rows, uint32_t active_shards, uint32_t row_bytes);
static __device__ __forceinline__ void pool_slice_reduce_add_shard_range(
    uint32_t rows,
    uint32_t shard,
    uint32_t shards,
    uint32_t* row_begin,
    uint32_t* row_end);

template <PoolSliceDynamicReadTransform Transform>
struct PoolSliceDynamicReadExecutor;

// A destination DATA instruction compiled as DynamicRead<Copy> is executed by
// the whole PoolInst CTA:
// one warp owns each
// local reader, and its lanes move that reader's matching activation rows.
// Queue-zero metadata reserves the complete (reader, source) span once.  A
// queue carries the exact compact interval, so the consumer never derives G.
template <bool HostDataPlane, uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_stream_gather_rows(
    uint32_t source_pe,
    uint32_t compact_begin,
    uint32_t compact_end,
    uint32_t ready_slot_and_reader,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    const PoolSliceReceiveBatch* receive_routes,
    const uint32_t* send_token_rows,
    const uint8_t* token_pool,
    const uint8_t* delivery_pool,
    uint8_t* expert_input,
    int* bars,
    uint32_t write_barrier,
    uint64_t sequence,
    uint32_t* shared_status,
    uint32_t thread_id) {
  __shared__ PoolSliceReceiveBatch shared_route;
  __shared__ uint32_t shared_relative_begin;
  __shared__ uint32_t shared_relative_end;
  __shared__ uint32_t shared_route_valid;

  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  const uint32_t ready_slot = ready_slot_and_reader & 0xffffU;
  const uint32_t local_reader = ready_slot_and_reader >> 16;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  static_assert(TotalWarps > 0);
  if (source_pe >= config.num_pes || compact_begin >= compact_end ||
      compact_end > config.token_capacity ||
      ready_slot >= poolSliceMaxDataGroups ||
      local_reader >= config.local_readers) {
    if (thread_id == 0) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
    }
    return;
  }

  const PoolSlicePublishBatch batch = *pool_slice_stream_batch(
      receive_batches, source_pe, config.route_capacity);
  const uint32_t* source_routes = pool_slice_stream_route_words(
      receive_batches, source_pe, config, batch);
  if (thread_id == 0) {
    shared_route = receive_routes[
        static_cast<uint64_t>(local_reader) * config.num_pes + source_pe];
    const PoolSliceReceiveBatch& route = shared_route;
    shared_route_valid = route.sequence == sequence &&
        route.source_pe == source_pe &&
        route.local_reader == local_reader &&
        route.flags == POOL_SLICE_BATCH_FLAGS_NONE &&
        route.source_begin <= config.route_capacity &&
        route.row_count <= config.route_capacity - route.source_begin &&
        route.source_begin >= batch.route_begin &&
        route.source_begin <= batch.route_end &&
        route.row_count <= batch.route_end - route.source_begin &&
        route.base_row <= config.expert_capacity_rows &&
        route.row_count <= config.expert_capacity_rows - route.base_row;
    if (shared_route_valid != 0) {
      shared_relative_begin = pool_slice_stream_route_lower_bound(
          source_routes,
          route.source_begin - batch.route_begin,
          route.row_count,
          compact_begin);
      shared_relative_end = pool_slice_stream_route_lower_bound(
          source_routes,
          route.source_begin - batch.route_begin,
          route.row_count,
          compact_end);
    }
  }
  __syncthreads();
  if (shared_route_valid == 0) {
    if (thread_id == 0) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
    }
    return;
  }

  const PoolSliceReceiveBatch route = shared_route;
  const uint32_t packed_route_begin =
      route.source_begin - batch.route_begin;
  const uint32_t relative_begin = shared_relative_begin;
  const uint32_t relative_end = shared_relative_end;
  bool dense_remote = source_pe != config.my_pe &&
      relative_end - relative_begin == compact_end - compact_begin;
  bool dense_thread_valid = true;
  if (dense_remote) {
    for (uint32_t relative = relative_begin + thread_id;
         relative < relative_end;
         relative += blockDim.x) {
      const uint32_t route_word =
          source_routes[packed_route_begin + relative];
      const uint32_t compact_row = route_word & 0xffffU;
      const uint32_t expected_row =
          compact_begin + relative - relative_begin;
      dense_thread_valid &= compact_row == expected_row;
    }
  }
  dense_remote =
      __syncthreads_and(!dense_remote || dense_thread_valid) && dense_remote;
  if (dense_remote) {
    if constexpr (poolSliceRawSgl && !HostDataPlane) {
      // The queue head becomes claimable after segment zero. Keep the same
      // reader CTA and walk the remainder of the ordered progress word while
      // later SGL writes are still crossing the fabric.
      const uint32_t segment_count = pool_slice_stream_data_segments(
          compact_begin, compact_end);
      for (uint32_t segment = 0; segment < segment_count; ++segment) {
        if (segment != 0 && thread_id == 0) {
          const uint64_t expected = pool_slice_stream_data_progress(
              sequence, segment + 1);
          while (pool_slice_signal_fetch(
                     pool_slice_stream_data_ready(
                         control, source_pe, ready_slot, 0),
                     false) < expected)
            __nanosleep(barrierPollSleepCycles);
        }
        __syncthreads();
        const uint32_t segment_begin =
            compact_begin + segment * poolSliceRawSglWidth;
        const uint32_t segment_end =
            segment_begin + poolSliceRawSglWidth < compact_end
            ? segment_begin + poolSliceRawSglWidth
            : compact_end;
        const uint32_t relative_offset = segment_begin - compact_begin;
        pool_slice_copy_block(
            expert_input +
                (static_cast<uint64_t>(local_reader) *
                     config.expert_capacity_rows +
                 route.base_row + relative_begin + relative_offset) *
                    config.row_bytes,
            delivery_pool +
                (static_cast<uint64_t>(source_pe) * config.token_capacity +
                 segment_begin) *
                    config.row_bytes,
            static_cast<uint64_t>(segment_end - segment_begin) *
                config.row_bytes,
            thread_id);
      }
    } else {
      pool_slice_copy_block(
          expert_input +
              (static_cast<uint64_t>(local_reader) *
                   config.expert_capacity_rows +
               route.base_row + relative_begin) *
                  config.row_bytes,
          delivery_pool +
              (static_cast<uint64_t>(source_pe) * config.token_capacity +
               compact_begin) *
                  config.row_bytes,
          static_cast<uint64_t>(relative_end - relative_begin) *
              config.row_bytes,
          thread_id);
    }
  } else {
    // Sparse and self-source gathers stripe rows across the compiled warps;
    // each warp keeps a full coalesced row copy and its precise writer wait.
    for (uint32_t relative = relative_begin + warp;
         relative < relative_end;
         relative += TotalWarps) {
      uint32_t route_word = 0;
      uint32_t compact_row = 0;
      if (lane == 0) {
        route_word = source_routes[packed_route_begin + relative];
        compact_row = route_word & 0xffffU;
      }
      route_word = __shfl_sync(0xffffffffU, route_word, 0);
      compact_row = __shfl_sync(0xffffffffU, compact_row, 0);
      if (compact_row < compact_begin || compact_row >= compact_end ||
          compact_row >= config.token_capacity) {
        if (lane == 0) {
          atomicCAS(
              shared_status,
              static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
              static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
        }
        continue;
      }
      const uint8_t* source_address = delivery_pool +
          (static_cast<uint64_t>(source_pe) * config.token_capacity +
           compact_row) *
              config.row_bytes;
      if constexpr (poolSliceRawSgl && !HostDataPlane) {
        if (source_pe != config.my_pe) {
          const uint32_t segment =
              (compact_row - compact_begin) / poolSliceRawSglWidth;
          if (segment != 0) {
            pool_slice_stream_wait_data_progress_warp(
                control,
                source_pe,
                ready_slot,
                pool_slice_stream_data_progress(sequence, segment + 1),
                lane);
          }
        }
      }
      if (source_pe == config.my_pe) {
        uint32_t token_row = 0;
        if (lane == 0) {
          token_row = send_token_rows[
              static_cast<uint64_t>(config.my_pe) * config.token_capacity +
              compact_row];
        }
        token_row = __shfl_sync(0xffffffffU, token_row, 0);
        if (token_row >= config.token_capacity) {
          if (lane == 0) {
            atomicCAS(
                shared_status,
                static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
                static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
          }
          continue;
        }
        const uint32_t chunk = token_row / config.write_chunk_rows;
        uint32_t write_ready = 0;
        while (write_ready == 0) {
          if (lane == 0)
            write_ready = pool_slice_barrier_ready(
                bars + write_barrier + chunk);
          write_ready =
              __shfl_sync(0xffffffffU, write_ready, 0);
          if (write_ready == 0)
            __nanosleep(barrierPollSleepCycles);
        }
        source_address = token_pool +
            static_cast<uint64_t>(token_row) * config.row_bytes;
      }
      pool_slice_copy_warp(
          expert_input +
              (static_cast<uint64_t>(local_reader) *
                   config.expert_capacity_rows +
               route.base_row + relative) *
                  config.row_bytes,
          source_address,
          config.row_bytes,
          lane);
    }
  }
  __syncthreads();
  // The final release-add for a reader is its precise expert-input data fence.
  // Metadata computes the expected number of nonempty shards independently,
  // so no queue-wide END or unrelated reader participates in this dependency.
  if (thread_id == 0 && relative_begin < relative_end) {
    dae_atomic_add_release_gpu(
        control + poolSliceControlReaderDataDone + local_reader, 1);
  }
}

static __device__ __forceinline__ float pool_slice_route_weight(
    uint64_t route_word) {
  union {
    uint16_t bits;
    __nv_bfloat16 value;
  } weight;
  weight.bits = static_cast<uint16_t>(route_word >> 32);
  return __bfloat162float(weight.value);
}

// Reduce one compact source token across the experts hosted by this pool
// slice. Gather workers materialize a tiny reverse map while copying the
// expert-major input; one lane per local reader then resolves its row in one
// load and the full warp performs the vector BF16-weighted reduction.
static __device__ __noinline__ void pool_slice_weighted_reduce_token(
    uint32_t source_pe,
    uint32_t packed_row,
    uint32_t vector_shard,
    uint32_t vector_shards,
    const PoolSliceConfig& config,
    const uint64_t* combine_rows,
    const uint8_t* expert_output,
    uint8_t* partial_output,
    uint32_t* shared_status,
    uint32_t lane) {
  uint32_t expert_row = UINT32_MAX;
  uint32_t weight_word = 0;
  if (lane < config.local_readers) {
    const uint32_t reader = lane;
    const uint64_t word = combine_rows[
        (static_cast<uint64_t>(reader) * config.num_pes + source_pe) *
            config.token_capacity +
        packed_row];
    if (word != UINT64_MAX) {
      expert_row = static_cast<uint32_t>(word);
      weight_word = static_cast<uint32_t>(word >> 32);
      if (expert_row >= config.expert_capacity_rows) {
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
        expert_row = UINT32_MAX;
      }
    }
  }

  auto* destination = reinterpret_cast<__nv_bfloat162*>(
      partial_output + static_cast<uint64_t>(packed_row) * config.row_bytes);
  const uint32_t elements = config.row_bytes / sizeof(__nv_bfloat162);
  const uint32_t vector_stride = vector_shards * 32;
  for (uint32_t element = vector_shard * 32 + lane;
       element < elements;
       element += 4 * vector_stride) {
      float2 sums[4][4];
#pragma unroll
      for (uint32_t item = 0; item < 4; ++item) {
#pragma unroll
        for (uint32_t group = 0; group < 4; ++group)
          sums[item][group] = make_float2(0.0f, 0.0f);
      }
#pragma unroll
      for (uint32_t reader = 0;
           reader < poolSliceMaxLocalReaders;
           ++reader) {
        if (reader >= config.local_readers)
          continue;
        const uint32_t reader_row =
            __shfl_sync(0xffffffffU, expert_row, reader);
        if (reader_row == UINT32_MAX)
          continue;
        const auto* row = reinterpret_cast<const __nv_bfloat162*>(
            expert_output +
            (static_cast<uint64_t>(reader) * config.expert_capacity_rows +
             reader_row) *
                config.row_bytes);
        const uint64_t route_weight_word =
            static_cast<uint64_t>(
                __shfl_sync(0xffffffffU, weight_word, reader)) << 32;
        const float weight = pool_slice_route_weight(route_weight_word);
#pragma unroll
        for (uint32_t item = 0; item < 4; ++item) {
          const uint32_t item_element = element + item * vector_stride;
          if (item_element < elements) {
            const float2 value = __bfloat1622float2(row[item_element]);
            sums[item][reader & 3U].x += value.x * weight;
            sums[item][reader & 3U].y += value.y * weight;
          }
        }
      }
#pragma unroll
      for (uint32_t item = 0; item < 4; ++item) {
        const uint32_t item_element = element + item * vector_stride;
        if (item_element < elements) {
          const float sum_x = sums[item][0].x + sums[item][1].x +
              sums[item][2].x + sums[item][3].x;
          const float sum_y = sums[item][0].y + sums[item][1].y +
              sums[item][2].y + sums[item][3].y;
          destination[item_element] =
              __floats2bfloat162_rn(sum_x, sum_y);
        }
      }
  }
  __syncwarp();
}

// Finish the source-side sum of one token across destination-pool partials.
// The producer already owns a sorted compact-token list for every target, so
// lane zero performs at most num_pes short scans and broadcasts the matching
// return-inbox rows to the vector reduction lanes.
static __device__ __noinline__ void pool_slice_weighted_scatter_token(
    uint32_t source_row,
    uint32_t vector_shard,
    uint32_t vector_shards,
    const PoolSliceConfig& config,
    const uint32_t* send_token_rows,
    const uint8_t* return_inbox,
    uint8_t* returned,
    uint32_t* shared_status,
    uint32_t lane) {
  const uint64_t receive_capacity =
      static_cast<uint64_t>(config.num_pes) * config.token_capacity;
  const uint8_t* local_partial_output =
      reinterpret_cast<const uint8_t*>(config.delivery_pool_address) +
      (receive_capacity +
       static_cast<uint64_t>(config.my_pe) * config.token_capacity) *
          config.row_bytes;
  uint32_t packed_row = UINT32_MAX;
  if (lane < config.num_pes) {
    const uint32_t target_pe = lane;
    const uint32_t* inverse_rows =
        send_token_rows +
        static_cast<uint64_t>(config.num_pes + target_pe) *
            config.token_capacity;
    packed_row = inverse_rows[source_row];
    if (packed_row != UINT32_MAX && packed_row >= config.token_capacity) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
      packed_row = UINT32_MAX;
    }
  }

  // A common inference route keeps all experts selected by one token on one
  // pool slice. The reverse-map lanes already identify that case, so bypass
  // BF16->FP32->BF16 reduction and copy the sole weighted partial directly.
  const uint32_t contributor_mask = __ballot_sync(
      0xffffffffU, lane < config.num_pes && packed_row != UINT32_MAX);
  if (__popc(contributor_mask) == 1) {
    const uint32_t target_pe = __ffs(contributor_mask) - 1;
    const uint32_t target_row =
        __shfl_sync(0xffffffffU, packed_row, target_pe);
    const uint8_t* partial = target_pe == config.my_pe
        ? local_partial_output +
            static_cast<uint64_t>(target_row) * config.row_bytes
        : return_inbox +
            (static_cast<uint64_t>(target_pe) * config.token_capacity +
             target_row) *
                config.row_bytes;
    pool_slice_copy_warp_shard(
        returned + static_cast<uint64_t>(source_row) * config.row_bytes,
        partial,
        config.row_bytes,
        vector_shard,
        vector_shards,
        lane);
    __syncwarp();
    return;
  }

  if (__popc(contributor_mask) == 2) {
    const uint32_t target0 = __ffs(contributor_mask) - 1;
    const uint32_t target1 =
        __ffs(contributor_mask & (contributor_mask - 1)) - 1;
    const uint32_t row0 =
        __shfl_sync(0xffffffffU, packed_row, target0);
    const uint32_t row1 =
        __shfl_sync(0xffffffffU, packed_row, target1);
    const uint8_t* partial0 = target0 == config.my_pe
        ? local_partial_output + static_cast<uint64_t>(row0) * config.row_bytes
        : return_inbox +
            (static_cast<uint64_t>(target0) * config.token_capacity + row0) *
                config.row_bytes;
    const uint8_t* partial1 = target1 == config.my_pe
        ? local_partial_output + static_cast<uint64_t>(row1) * config.row_bytes
        : return_inbox +
            (static_cast<uint64_t>(target1) * config.token_capacity + row1) *
                config.row_bytes;
    pool_slice_add_bf16_warp_shard(
        returned + static_cast<uint64_t>(source_row) * config.row_bytes,
        partial0,
        partial1,
        config.row_bytes,
        vector_shard,
        vector_shards,
        lane);
    __syncwarp();
    return;
  }

  auto* destination = reinterpret_cast<__nv_bfloat162*>(
      returned + static_cast<uint64_t>(source_row) * config.row_bytes);
  const uint32_t elements = config.row_bytes / sizeof(__nv_bfloat162);
  const uint32_t vector_stride = vector_shards * 32;
  for (uint32_t element = vector_shard * 32 + lane;
       element < elements;
       element += 4 * vector_stride) {
    float2 sum0 = make_float2(0.0f, 0.0f);
    float2 sum1 = make_float2(0.0f, 0.0f);
    float2 sum2 = make_float2(0.0f, 0.0f);
    float2 sum3 = make_float2(0.0f, 0.0f);
    for (uint32_t target_pe = 0;
         target_pe < config.num_pes;
         ++target_pe) {
      const uint32_t target_row =
          __shfl_sync(0xffffffffU, packed_row, target_pe);
      if (target_row == UINT32_MAX)
        continue;
      const uint8_t* partial_bytes = target_pe == config.my_pe
          ? local_partial_output +
              static_cast<uint64_t>(target_row) * config.row_bytes
          : return_inbox +
              (static_cast<uint64_t>(target_pe) * config.token_capacity +
               target_row) *
                  config.row_bytes;
      const auto* partial =
          reinterpret_cast<const __nv_bfloat162*>(partial_bytes);
      const uint32_t element1 = element + vector_stride;
      const uint32_t element2 = element + 2 * vector_stride;
      const uint32_t element3 = element + 3 * vector_stride;
      const float2 value0 = __bfloat1622float2(partial[element]);
      sum0.x += value0.x;
      sum0.y += value0.y;
      if (element1 < elements) {
        const float2 value1 = __bfloat1622float2(partial[element1]);
        sum1.x += value1.x;
        sum1.y += value1.y;
      }
      if (element2 < elements) {
        const float2 value2 = __bfloat1622float2(partial[element2]);
        sum2.x += value2.x;
        sum2.y += value2.y;
      }
      if (element3 < elements) {
        const float2 value3 = __bfloat1622float2(partial[element3]);
        sum3.x += value3.x;
        sum3.y += value3.y;
      }
    }
    destination[element] = __floats2bfloat162_rn(sum0.x, sum0.y);
    const uint32_t element1 = element + vector_stride;
    const uint32_t element2 = element + 2 * vector_stride;
    const uint32_t element3 = element + 3 * vector_stride;
    if (element1 < elements)
      destination[element1] = __floats2bfloat162_rn(sum1.x, sum1.y);
    if (element2 < elements)
      destination[element2] = __floats2bfloat162_rn(sum2.x, sum2.y);
    if (element3 < elements)
      destination[element3] = __floats2bfloat162_rn(sum3.x, sum3.y);
  }
  __syncwarp();
}

static __device__ __forceinline__ uint32_t
pool_slice_reduce_add_source_shards(
    const PoolSliceConfig& config, uint32_t source_pe) {
  if (source_pe >= config.pool_count)
    return 0;
  return 1 + (config.pool_count - 1 - source_pe) / config.num_pes;
}

static __device__ __forceinline__ uint32_t
pool_slice_reduce_add_group_count(
    uint32_t rows, uint32_t active_shards, uint32_t row_bytes) {
  if (rows == 0 || active_shards == 0)
    return 0;
  constexpr uint64_t target_group_bytes = 256ULL * 1024;
  uint64_t groups =
      (static_cast<uint64_t>(rows) * row_bytes + target_group_bytes - 1) /
      target_group_bytes;
  groups = groups < active_shards ? groups : active_shards;
  groups = groups < poolSliceReturnGroupsPerSource
      ? groups
      : poolSliceReturnGroupsPerSource;
  return static_cast<uint32_t>(groups);
}

static __device__ __forceinline__ void pool_slice_reduce_add_shard_range(
    uint32_t rows,
    uint32_t shard,
    uint32_t shards,
    uint32_t* row_begin,
    uint32_t* row_end) {
  if (shards == 0 || shard >= shards) {
    *row_begin = 0;
    *row_end = 0;
    return;
  }
  *row_begin = static_cast<uint32_t>(
      static_cast<uint64_t>(rows) * shard / shards);
  *row_end = static_cast<uint32_t>(
      static_cast<uint64_t>(rows) * (shard + 1) / shards);
}

static __device__ __forceinline__ PoolSliceDynamicReadPlan*
pool_slice_dynamic_read_plan(uint64_t* control, uint32_t pool_rank) {
  return reinterpret_cast<PoolSliceDynamicReadPlan*>(
      control + poolSliceControlCombinePlan) + pool_rank;
}

// RESERVE_ROUTES already owns the complete source envelope and the final
// destination expert-row allocation. Convert that metadata into the immutable
// DynamicRead<ReduceAdd> stream before activation DATA is required. Plans for
// different sources write disjoint pool ranks, so independently scheduled
// metadata CTAs can build them concurrently without a destination-wide join.
static __device__ __noinline__ void pool_slice_build_reduce_add_plans_source(
    uint32_t source_pe,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch& batch,
    const uint64_t* combine_rows,
    uint64_t* control,
    uint64_t sequence,
    uint32_t* shared_status,
    uint32_t thread_id) {
  const bool batch_valid = batch.active_rows <= config.token_capacity &&
      batch.sequence == sequence && batch.source_pe == source_pe &&
      batch.target_pe == config.my_pe &&
      batch.flags == POOL_SLICE_BATCH_FLAGS_NONE;
  const uint32_t rows = batch_valid ? batch.active_rows : 0;
  const uint32_t available_shards =
      pool_slice_reduce_add_source_shards(config, source_pe);
  for (uint32_t source_shard = thread_id;
       source_shard < available_shards;
       source_shard += blockDim.x) {
    const uint32_t pool_rank =
        source_pe + source_shard * config.num_pes;
    const uint32_t active_shards = rows < available_shards
        ? rows
        : available_shards;
    uint32_t row_begin = 0;
    uint32_t row_end = 0;
    pool_slice_reduce_add_shard_range(
        rows, source_shard, active_shards, &row_begin, &row_end);

    PoolSliceDynamicReadPlan plan{
        sequence,
        source_pe,
        row_begin,
        row_end,
        0,
        UINT32_MAX,
        POOL_SLICE_DYNAMIC_READ_PLAN_EMPTY};
    if (!batch_valid) {
      plan.flags = POOL_SLICE_DYNAMIC_READ_PLAN_ERROR;
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
    } else if (row_begin < row_end) {
      const uint32_t groups = pool_slice_reduce_add_group_count(
          rows, active_shards, config.row_bytes);
      const uint32_t group = static_cast<uint32_t>(
          (static_cast<uint64_t>(source_shard + 1) * groups - 1) /
          active_shards);
      // The reverse map was just expanded from this source's metadata and is
      // still cache-hot. Scan only this fine reduction shard, producing a more
      // precise dependency set than a transport-group mask without adding
      // atomics or bookkeeping to DynamicRead<Copy>'s hot gather loop.
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader) {
        // With locality-preserving top-k placement, a local expert commonly
        // receives every compact token in this source batch. The count proves
        // that waiting on the reader is conservative for every shard, avoiding
        // reverse-map traffic in the production dense case. Sparse readers
        // retain the exact per-shard scan below.
        if (batch.reader_counts[reader] == rows) {
          plan.dependency_mask |= 1U << reader;
          continue;
        }
        if (batch.reader_counts[reader] == 0)
          continue;
        const uint64_t* reader_rows = combine_rows +
            (static_cast<uint64_t>(reader) * config.num_pes + source_pe) *
                config.token_capacity;
        for (uint32_t row = row_begin; row < row_end; ++row) {
          if (reader_rows[row] != UINT64_MAX) {
            plan.dependency_mask |= 1U << reader;
            break;
          }
        }
      }
      plan.ready_slot = group;
      plan.flags = POOL_SLICE_DYNAMIC_READ_PLAN_ACTIVE;
      if (plan.dependency_mask == 0) {
        plan.flags |= POOL_SLICE_DYNAMIC_READ_PLAN_ERROR;
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
      }
    }
    *pool_slice_dynamic_read_plan(control, pool_rank) = plan;
  }
}

static __device__ __forceinline__ uint64_t* pool_slice_reduce_add_return_ready(
    uint64_t* control, uint32_t destination_pe, uint32_t return_rank) {
  return control + poolSliceControlReturnReady +
      static_cast<uint64_t>(destination_pe) * poolSliceMaxReturnReady +
      return_rank;
}

static __device__ __forceinline__ uint32_t*
pool_slice_reduce_add_return_group_count(
    uint64_t* control, uint32_t source_pe, uint32_t group) {
  return reinterpret_cast<uint32_t*>(
      control + poolSliceControlReturnGroupCount +
      static_cast<uint64_t>(source_pe) *
          poolSliceReturnGroupsPerSource +
      group);
}

// DynamicRead<ReduceAdd>: consume one prebuilt combine instruction, reduce
// ready expert rows inside the destination pool slice, and transfer one
// source-owned row shard with a payload-coupled generation. This local half is
// separated from source finalization so the transform stays small. The live
// scheduler still gives Copy exclusive HBM priority through dispatch
// retirement; early ReduceAdd execution was measured and rejected.
template <bool HostDataPlane, uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_dynamic_read_reduce_add_local(
    const PoolSliceConfig& config,
    const PoolSliceHostConfig* host_config,
    int* bars,
    uint64_t* g_events,
    uint32_t compute_barrier_base,
    uint32_t thread_id,
    uint64_t sequence,
    uint32_t* shared_status) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  const auto* receive_batches =
      reinterpret_cast<const PoolSlicePublishBatch*>(
          config.receive_batches_address);
  const auto* combine_rows =
      reinterpret_cast<const uint64_t*>(config.combine_rows_address);
  const auto* expert_output =
      reinterpret_cast<const uint8_t*>(config.expert_output_address);
  auto* delivery_pool =
      reinterpret_cast<uint8_t*>(config.delivery_pool_address);
  auto* return_inbox =
      reinterpret_cast<uint8_t*>(config.return_inbox_address);
  const PoolSliceDynamicReadPlan plan =
      *pool_slice_dynamic_read_plan(control, config.pool_rank);
  const uint32_t expected_source_pe = config.pool_rank % config.num_pes;
  const bool plan_valid = plan.sequence == sequence &&
      plan.source_pe == expected_source_pe &&
      (plan.flags & POOL_SLICE_DYNAMIC_READ_PLAN_ERROR) == 0;
  if (!plan_valid && thread_id == 0) {
    atomicCAS(
        shared_status,
        static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
        static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
  }
  const bool active_shard = plan_valid &&
      (plan.flags & POOL_SLICE_DYNAMIC_READ_PLAN_ACTIVE) != 0 &&
      plan.row_begin < plan.row_end;
  if (warp == 0 && active_shard) {
    const uint32_t dependencies = plan.dependency_mask;
    bool ready = lane >= config.local_readers ||
        (dependencies & (1U << lane)) == 0;
    while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
      if (!ready)
        ready = pool_slice_barrier_ready(
            bars + compute_barrier_base + lane);
      if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
  }
  __syncthreads();

  if (thread_id == 0 && active_shard && g_events != nullptr) {
    const auto previous = atomicCAS(
        reinterpret_cast<unsigned long long*>(
            control + poolSliceControlCombineFirstReady),
        0ULL,
        static_cast<unsigned long long>(sequence));
    if (previous == 0) {
      const uint32_t rank_zero_block = blockIdx.x - config.pool_rank;
      g_events[static_cast<uint64_t>(rank_zero_block) * numProfileEvents +
               poolSliceProfileComputeReady] =
          cuda::ptx::get_sreg_globaltimer();
    }
  }

  if (thread_id == 0 && active_shard && g_events != nullptr) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
             poolSliceProfileReturnReduceStart] =
        cuda::ptx::get_sreg_globaltimer();
  }

  const uint64_t receive_capacity =
      static_cast<uint64_t>(config.num_pes) * config.token_capacity;
  uint8_t* partial_staging =
      delivery_pool + receive_capacity * config.row_bytes;
  const uint32_t source_pe = expected_source_pe;
  const PoolSlicePublishBatch batch = *pool_slice_stream_batch(
      receive_batches, source_pe, config.route_capacity);
  const uint32_t rows = plan_valid ? batch.active_rows : 0;
  const uint32_t available_shards =
      pool_slice_reduce_add_source_shards(config, source_pe);
  const uint32_t active_shards =
      rows < available_shards ? rows : available_shards;
  const uint32_t row_begin = active_shard ? plan.row_begin : 0;
  const uint32_t row_end = active_shard ? plan.row_end : 0;
  uint8_t* partial_output =
      partial_staging +
      static_cast<uint64_t>(source_pe) * config.token_capacity *
          config.row_bytes;
  // The coordinator warp is idle after this plan's named dependencies are
  // ready. Reuse it for the local weighted reduction so every statically
  // assembled PoolInst warp contributes during this bandwidth-bound phase.
  if (active_shard) {
    constexpr uint32_t worker_warps = TotalWarps;
    const uint32_t worker_slot = warp;
    const uint32_t vector_shards = 4;
    const uint32_t reduce_tasks =
        (row_end - row_begin) * vector_shards;
    for (uint32_t task = worker_slot;
         task < reduce_tasks;
         task += worker_warps) {
      const uint32_t packed_row = row_begin + task / vector_shards;
      pool_slice_weighted_reduce_token(
          source_pe,
          packed_row,
          task % vector_shards,
          vector_shards,
          config,
          combine_rows,
          expert_output,
          partial_output,
          shared_status,
          lane);
    }
  }
  __syncthreads();

  if (thread_id == 0 && active_shard && g_events != nullptr) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
             poolSliceProfileReturnReduceDone] =
        cuda::ptx::get_sreg_globaltimer();
  }

  // Keep fine-grained reduction sharding, then coalesce adjacent shards into
  // roughly 256-KiB transport groups. The last release/acquire counter
  // contributor owns the group's one contiguous put plus exact generation.
  const uint32_t active_groups = pool_slice_reduce_add_group_count(
      rows, active_shards, config.row_bytes);
  const uint32_t group = active_shard ? plan.ready_slot : 0;
  const uint32_t transport_warp = group % TotalWarps;
  if (warp == transport_warp && active_shard) {
    uint32_t group_shard_begin = 0;
    uint32_t group_shard_end = 0;
    pool_slice_reduce_add_shard_range(
        active_shards,
        group,
        active_groups,
        &group_shard_begin,
        &group_shard_end);
    uint32_t previous = 0;
    if (lane == 0) {
      previous = dae_atomic_fetch_add_acq_rel_gpu(
          pool_slice_reduce_add_return_group_count(
              control, source_pe, group),
          1);
    }
    previous = __shfl_sync(0xffffffffU, previous, 0);
    if (previous + 1 == group_shard_end - group_shard_begin) {
      const uint32_t group_row_begin = static_cast<uint32_t>(
          static_cast<uint64_t>(rows) * group_shard_begin /
          active_shards);
      const uint32_t group_row_end = static_cast<uint32_t>(
          static_cast<uint64_t>(rows) * group_shard_end /
          active_shards);
      if (lane == 0 && g_events != nullptr) {
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileFirstReturnPut] =
            cuda::ptx::get_sreg_globaltimer();
      }
      __syncwarp();
      const uint8_t* source = partial_output +
          static_cast<uint64_t>(group_row_begin) * config.row_bytes;
      uint8_t* destination = return_inbox +
          (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
           group_row_begin) * config.row_bytes;
      const uint64_t bytes =
          static_cast<uint64_t>(group_row_end - group_row_begin) *
          config.row_bytes;
      const uint32_t group_rank = source_pe + group * config.num_pes;
      uint64_t* ready = pool_slice_reduce_add_return_ready(
          control, config.my_pe, group_rank);
      if (source_pe == config.my_pe) {
        if (lane == 0)
          pool_slice_signal_release_local(ready, sequence);
      } else {
        if constexpr (HostDataPlane) {
          const auto* peers = reinterpret_cast<const PoolSliceHostPeer*>(
              host_config->peers_address);
          auto* generations = reinterpret_cast<uint64_t*>(
              host_config->producer_generations_address);
          const PoolSliceHostPeer peer = peers[source_pe];
          const uint64_t generation = pool_host_reserve_generation_warp(
              generations + source_pe, lane);
          const uint64_t ready_offset =
              reinterpret_cast<uint64_t>(ready) - config.control_address;
          pool_host_publish_request_warp<false>(
              reinterpret_cast<HostSglRingMemory*>(peer.ring_memory),
              generation,
              nullptr,
              group_row_begin,
              group_row_end - group_row_begin,
              host_config->local_lkey,
              static_cast<uint32_t>(peer.remote_rkey),
              reinterpret_cast<uint64_t>(partial_output),
              config.row_bytes,
              config.row_bytes,
              peer.remote_return_inbox_address +
                  (static_cast<uint64_t>(config.my_pe) *
                       config.token_capacity +
                   group_row_begin) * config.row_bytes,
              peer.remote_control_address + ready_offset,
              sequence,
              lane);
        } else {
#if DAE_POOL_SLICE_RAW_SGL && defined(DAE_ENABLE_NVSHMEM) && \
    defined(__CUDA_ARCH__)
          pool_ibgda_put_contiguous_signal_warp(
              destination,
              ready,
              source,
              static_cast<uint32_t>(bytes),
              source_pe,
              sequence,
              lane);
#elif defined(DAE_ENABLE_NVSHMEM)
          nvshmemx_putmem_nbi_warp(
              destination,
              source,
              static_cast<size_t>(bytes),
              source_pe);
          if (lane == 0)
            nvshmem_uint64_p(ready, sequence, source_pe);
#else
          pool_gin_put_warp(
              destination,
              source,
              static_cast<size_t>(bytes),
              source_pe,
              true);
          __syncwarp();
          if (lane == 0)
            pool_gin_set_thread(ready, sequence, source_pe);
#endif
        }
      }
    }
  }
  __syncthreads();
  if (thread_id == 0) {
    if (g_events != nullptr) {
      g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
               poolSliceProfileReturnCtaDone] =
          cuda::ptx::get_sreg_globaltimer();
    }
    if (*shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(*shared_status));
    dae_atomic_store_release_gpu(
        control + poolSliceControlReturnGeneration + config.pool_rank,
        sequence);
  }
  __syncthreads();
}

// Finalize the source-visible return only after every local ReduceAdd
// instruction has completed. Keeping this join outside the local transform
// makes the precise per-CTA completion generation explicit and reduces the
// live state of both helpers without changing their ordered execution.
template <uint32_t TotalWarps>
static __device__ __noinline__ void
pool_slice_dynamic_read_reduce_add_finish(
    const PoolSliceConfig& config,
    uint64_t* g_events,
    uint32_t thread_id,
    uint64_t sequence,
    uint32_t* shared_status) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  const auto* send_token_rows =
      reinterpret_cast<const uint32_t*>(config.send_token_rows_address);
  auto* return_inbox =
      reinterpret_cast<uint8_t*>(config.return_inbox_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);

  if (config.pool_rank == 0 && warp == 0) {
    pool_slice_wait_generation_warp(
        control + poolSliceControlReturnGeneration,
        config.pool_count,
        sequence,
        lane);
    const auto* send_token_counts =
        reinterpret_cast<const uint32_t*>(config.send_token_counts_address);
    const uint32_t source_shards =
        pool_slice_reduce_add_source_shards(config, config.my_pe);
    const uint32_t ready_tasks =
        config.num_pes * poolSliceReturnGroupsPerSource;
    for (uint32_t task = lane; task < ready_tasks; task += 32) {
      const uint32_t destination_pe =
          task / poolSliceReturnGroupsPerSource;
      const uint32_t group = task % poolSliceReturnGroupsPerSource;
      const uint32_t destination_rows = send_token_counts[destination_pe];
      const uint32_t destination_shards =
          destination_rows < source_shards
              ? destination_rows
              : source_shards;
      const uint32_t destination_groups =
          pool_slice_reduce_add_group_count(
              destination_rows, destination_shards, config.row_bytes);
      if (group >= destination_groups)
        continue;
      const uint32_t destination_rank =
          config.my_pe + group * config.num_pes;
      uint64_t* ready = pool_slice_reduce_add_return_ready(
          control, destination_pe, destination_rank);
      while (pool_slice_signal_fetch(
                 ready, destination_pe == config.my_pe) < sequence)
        __nanosleep(barrierPollSleepCycles);
    }
    __syncwarp();
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnPayloadDone, lane);
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnSignalsClosed, lane);
    if (lane == 0)
      dae_atomic_store_release_gpu(
          control + poolSliceControlScatterStart, sequence);
    __syncwarp();
  }

  pool_slice_wait_value_warp(
      control + poolSliceControlScatterStart, sequence, lane);
  const uint32_t global_warp = config.pool_rank * TotalWarps + warp;
  const uint32_t global_warps = config.pool_count * TotalWarps;
  const uint32_t scatter_shards = 2;
  const uint32_t scatter_tasks = config.token_capacity * scatter_shards;
  for (uint32_t task = global_warp;
       task < scatter_tasks;
       task += global_warps) {
    pool_slice_weighted_scatter_token(
        task / scatter_shards,
        task % scatter_shards,
        scatter_shards,
        config,
        send_token_rows,
        return_inbox,
        returned,
        shared_status,
        lane);
  }
  __syncthreads();
  if (thread_id == 0) {
    if (*shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(*shared_status));
    dae_atomic_store_release_gpu(
        control + poolSliceControlScatterGeneration + config.pool_rank,
        sequence);
  }
  __syncthreads();

  if (config.pool_rank == 0 && warp == 0) {
    pool_slice_wait_generation_warp(
        control + poolSliceControlScatterGeneration,
        config.pool_count,
        sequence,
        lane);
    if (lane == 0) {
      control[3] = config.num_pes;
      if (g_events != nullptr) {
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileScatterDone] =
            cuda::ptx::get_sreg_globaltimer();
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileDone] =
            cuda::ptx::get_sreg_globaltimer();
      }
    }
    __syncwarp();
  }
}

// Compile-time dynamic-read registry. Both dispatch and combine enter this
// interface, while each specialization keeps only the state needed by its
// transform. That preserves the queue-driven Copy scheduler and the static
// per-PoolInst ReduceAdd placement without a runtime opcode branch, a union
// context, or additional live registers in either hot path.
template <>
struct PoolSliceDynamicReadExecutor<POOL_SLICE_DYNAMIC_READ_COPY> {
  template <bool HostDataPlane, uint32_t TotalWarps>
  static __device__ __forceinline__ void execute(
      uint32_t source_pe,
      uint32_t compact_begin,
      uint32_t compact_end,
      uint32_t ready_slot_and_reader,
      const PoolSliceConfig& config,
      const PoolSlicePublishBatch* receive_batches,
      const PoolSliceReceiveBatch* receive_routes,
      const uint32_t* send_token_rows,
      const uint8_t* token_pool,
      const uint8_t* delivery_pool,
      uint8_t* expert_input,
      int* bars,
      uint32_t write_barrier,
      uint64_t sequence,
      uint32_t* shared_status,
      uint32_t thread_id) {
    pool_slice_stream_gather_rows<HostDataPlane, TotalWarps>(
        source_pe,
        compact_begin,
        compact_end,
        ready_slot_and_reader,
        config,
        receive_batches,
        receive_routes,
        send_token_rows,
        token_pool,
        delivery_pool,
        expert_input,
        bars,
        write_barrier,
        sequence,
        shared_status,
        thread_id);
  }
};

template <>
struct PoolSliceDynamicReadExecutor<POOL_SLICE_DYNAMIC_READ_REDUCE_ADD> {
  template <bool HostDataPlane, uint32_t TotalWarps>
  static __device__ __forceinline__ void execute(
      const PoolSliceConfig& config,
      const PoolSliceHostConfig* host_config,
      int* bars,
      uint64_t* g_events,
      uint32_t compute_barrier_base,
      uint32_t thread_id,
      uint64_t sequence,
      uint32_t* shared_status) {
    pool_slice_dynamic_read_reduce_add_local<HostDataPlane, TotalWarps>(
        config,
        host_config,
        bars,
        g_events,
        compute_barrier_base,
        thread_id,
        sequence,
        shared_status);
  }

  template <uint32_t TotalWarps>
  static __device__ __forceinline__ void finish(
      const PoolSliceConfig& config,
      uint64_t* g_events,
      uint32_t thread_id,
      uint64_t sequence,
      uint32_t* shared_status) {
    pool_slice_dynamic_read_reduce_add_finish<TotalWarps>(
        config, g_events, thread_id, sequence, shared_status);
  }
};

static __device__ __forceinline__ void
pool_slice_stream_publish_metadata_target(
    const PoolSliceConfig& config,
    uint64_t sequence,
    uint64_t signal_delta,
    uint32_t index,
    uint32_t lane) {
  const uint32_t target_pe = pool_slice_remote_first_pe(
      index, config.my_pe, config.num_pes);
  auto* send_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.send_batches_address);
  auto* receive_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.receive_batches_address);
  const auto* send_rows = reinterpret_cast<const uint32_t*>(
      config.send_rows_address);
  PoolSliceMetadataEnvelope* destination =
      pool_slice_stream_envelope(
          receive_batches, config.my_pe, config.route_capacity);
  PoolSliceMetadataEnvelope* source =
      pool_slice_stream_envelope(
          send_batches, target_pe, config.route_capacity);
  const PoolSlicePublishBatch* source_batch = &source->batch;
  const uint32_t envelope_bytes = pool_slice_stream_envelope_bytes(
      source_batch->active_rows,
      config.row_bytes,
      config.group_limit,
      config.num_pes);
  const uint32_t route_count =
      source_batch->route_end - source_batch->route_begin;
  uint32_t* packed_routes = pool_slice_stream_route_words(
      send_batches, target_pe, config, *source_batch);
  for (uint32_t route = lane; route < route_count; route += 32) {
    packed_routes[route] =
        send_rows[source_batch->route_begin + route];
  }
  __syncwarp();

  // The local fast path copies uint4 vectors, so include packet-local padding
  // after the live route words. The fixed peer stride reserves this padding.
  const uint64_t packet_bytes =
      (envelope_bytes +
       static_cast<uint64_t>(route_count) * sizeof(uint32_t) + 15) &
      ~15ULL;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  uint64_t* ready =
      control + poolSliceControlStreamMetadataTransportReady + config.my_pe;
  if (target_pe == config.my_pe) {
    pool_slice_copy_warp(destination, source, packet_bytes, lane);
    __syncwarp();
    if (lane == 0)
      pool_slice_signal_release_local(ready, sequence);
  } else {
#ifdef DAE_ENABLE_NVSHMEM
    nvshmemx_putmem_signal_nbi_warp(
        destination,
        source,
        static_cast<size_t>(packet_bytes),
        ready,
        signal_delta,
        NVSHMEM_SIGNAL_ADD,
        target_pe);
#else
    pool_gin_put_add_signal_warp(
        destination,
        source,
        static_cast<size_t>(packet_bytes),
        ready,
        signal_delta,
        target_pe);
#endif
  }
  __syncwarp();
}

template <bool HostDataPlane, uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_stream_publish_metadata(
    const PoolSliceConfig& config,
    uint64_t sequence,
    uint64_t signal_delta,
    uint32_t thread_id) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  if (config.pool_count >= config.num_pes) {
    // Every transport publishes one independent fused metadata message per
    // target. Metadata and activation keep independent CTA/QP placement so
    // the two planes can be submitted concurrently.
    if (config.pool_rank < config.num_pes && warp == 0) {
      pool_slice_stream_publish_metadata_target(
          config,
          sequence,
          signal_delta,
          config.pool_rank,
          lane);
    }
    return;
  }
  // A smaller generic assembly falls back to the coordinator CTA's warps.
  if (config.pool_rank != 0)
    return;
  for (uint32_t index = warp;
       index < config.num_pes;
       index += TotalWarps) {
    pool_slice_stream_publish_metadata_target(
        config, sequence, signal_delta, index, lane);
  }
}

// A coordinator lane validates one source envelope as soon as its metadata
// signal arrives.  It does not reconstruct groups or reserve destination
// rows: those are explicit ordered queue instructions executed by PoolInst
// workers.  Data readiness remains independent and may arrive first.
static __device__ __forceinline__ void pool_slice_stream_accept_metadata(
    uint32_t source_pe,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    uint64_t* control,
    uint64_t sequence,
    uint32_t* shared_status) {
  const PoolSlicePublishBatch batch =
      *pool_slice_stream_batch(
          receive_batches, source_pe, config.route_capacity);
  bool valid = batch.sequence == sequence &&
      batch.source_pe == source_pe && batch.target_pe == config.my_pe &&
      batch.active_rows <= config.token_capacity &&
      batch.route_begin <= batch.route_end &&
      batch.route_end <= config.route_capacity &&
      batch.flags == POOL_SLICE_BATCH_FLAGS_NONE;
  uint32_t source_cursor = batch.route_begin;
  for (uint32_t reader = 0; reader < config.local_readers; ++reader) {
    const uint32_t count = batch.reader_counts[reader];
    if (count > config.route_capacity || source_cursor > batch.route_end ||
        count > batch.route_end - source_cursor)
      valid = false;
    source_cursor += count;
  }
  if (source_cursor != batch.route_end)
    valid = false;
  if (!valid) {
    atomicCAS(
        shared_status,
        static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
        static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
    pool_slice_set_status(config, POOL_SLICE_STATUS_BATCH);
  }
  dae_atomic_store_release_gpu(
      control + poolSliceControlStreamMetadataReady + source_pe, sequence);
}

template <bool HostDataPlane, uint32_t TotalWarps>
static __device__ __forceinline__ bool pool_slice_stream_queue_message_ready(
    uint32_t source_pe,
    uint32_t queue,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    uint64_t* control,
    uint64_t sequence,
    bool require_unclaimed) {
  if (dae_atomic_load_acquire_gpu(
          control + poolSliceControlStreamMetadataReady + source_pe) <
      sequence)
    return false;
  constexpr uint32_t queue_count = poolSliceMaxStreamQueues;
  const uint32_t queue_index = source_pe * queue_count + queue;
  const uint64_t retired = dae_atomic_load_acquire_gpu(
      control + poolSliceControlStreamQueueRetiredMask);
  if ((retired & (1ULL << queue_index)) != 0)
    return false;
  uint64_t* claim =
      pool_slice_stream_queue_claim(control, source_pe, queue);
  const uint32_t claim_state = dae_atomic_load_relaxed_gpu(
      reinterpret_cast<const uint32_t*>(claim));
  auto* head_address = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_head(control, source_pe, queue));
  const uint64_t head = atomicAdd(head_address, 0ULL);
  if (head >= poolSliceStreamQueueDepth)
    return true;
  const PoolSliceQueueEntry* message = pool_slice_stream_queue_entry(
      receive_batches,
      source_pe,
      queue,
      static_cast<uint32_t>(head),
      config.route_capacity);
  if (message->sequence != sequence || message->message_index != head)
    return !require_unclaimed || claim_state == 0;
  if (message->opcode != POOL_SLICE_QUEUE_DATA)
    return !require_unclaimed || claim_state == 0;
  // DATA is sharded by local reader. The low word assigns the next
  // reader and the high word counts completed readers; the head stays stable
  // until the final shard advances it.
  if (require_unclaimed &&
      static_cast<uint32_t>(claim_state) >= config.local_readers)
    return false;
  if (dae_atomic_load_acquire_gpu(
          control + poolSliceControlStreamRouteReady + source_pe) < sequence)
    return false;
  if (message->ready_slot >= poolSliceMaxDataGroups)
    return true;
  if (source_pe == config.my_pe)
    return true;
  if constexpr (HostDataPlane || !poolSliceWarpQpCompletion) {
    const uint64_t observed = pool_slice_signal_fetch(
        pool_slice_stream_data_ready(
            control, source_pe, message->ready_slot, 0),
        false);
    if constexpr (poolSliceRawSgl) {
      return observed >= pool_slice_stream_data_progress(
          sequence,
          HostDataPlane
              ? pool_slice_stream_data_segments(
                    message->row_begin, message->row_end)
              : 1);
    }
    return observed >= sequence;
  } else {
    static_assert(TotalWarps == poolSlicePayloadWarps);
    static_assert(
        !poolSliceWarpQpCompletion ||
        poolSliceCompletionSlots == poolSlicePayloadWarps);
    const uint32_t group_rows = message->row_end - message->row_begin;
#pragma unroll
    for (uint32_t payload_warp = 0;
         payload_warp < TotalWarps;
         ++payload_warp) {
      const uint32_t warp_row_begin = message->row_begin +
          static_cast<uint32_t>(
              static_cast<uint64_t>(group_rows) * payload_warp / TotalWarps);
      const uint32_t warp_row_end = message->row_begin +
          static_cast<uint32_t>(
              static_cast<uint64_t>(group_rows) * (payload_warp + 1) /
              TotalWarps);
      if (warp_row_begin < warp_row_end &&
          pool_slice_signal_fetch(
              pool_slice_stream_data_ready(
                  control,
                  source_pe,
                  message->ready_slot,
                  payload_warp),
              false) < sequence)
        return false;
    }
    return true;
  }
}

// The scan is only over queue heads. Warp zero tests them in at most two
// 32-lane waves and elects the first ready head. Revalidation under the claim
// handles another PoolInst CTA advancing the same queue between scan and CAS.
template <bool HostDataPlane, uint32_t TotalWarps>
static __device__ __noinline__ bool pool_slice_stream_claim_queue_head(
    uint32_t queue_index,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    uint64_t* control,
    uint64_t sequence,
    PoolSliceQueueEntry* message,
    uint32_t* local_reader) {
  constexpr uint32_t queue_count = poolSliceMaxStreamQueues;
  const uint32_t source_pe = queue_index / queue_count;
  const uint32_t queue = queue_index % queue_count;
  auto* head_address = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_head(control, source_pe, queue));
  const uint64_t head = atomicAdd(head_address, 0ULL);
  if (head < poolSliceStreamQueueDepth) {
    const PoolSliceQueueEntry candidate = *pool_slice_stream_queue_entry(
        receive_batches,
        source_pe,
        queue,
        static_cast<uint32_t>(head),
        config.route_capacity);
    if (candidate.sequence == sequence &&
        candidate.message_index == head &&
        candidate.opcode == POOL_SLICE_QUEUE_DATA) {
      auto* copy_claim = reinterpret_cast<uint32_t*>(
          pool_slice_stream_queue_claim(control, source_pe, queue));
      uint32_t state = atomicAdd(copy_claim, 0U);
      while (state < config.local_readers) {
        const uint32_t desired = state + 1;
        const uint32_t previous = atomicCAS(copy_claim, state, desired);
        if (previous == state) {
          *message = candidate;
          *local_reader = state;
          return true;
        }
        state = previous;
      }
      return false;
    }
  }

  auto* claim = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_claim(control, source_pe, queue));
  if (dae_atomic_compare_exchange_acquire_gpu(
          reinterpret_cast<uint64_t*>(claim), 0, UINT64_MAX) != 0)
    return false;
  if (!pool_slice_stream_queue_message_ready<HostDataPlane, TotalWarps>(
          source_pe,
          queue,
          config,
          receive_batches,
          control,
          sequence,
          false)) {
    dae_atomic_store_release_gpu(reinterpret_cast<uint64_t*>(claim), 0);
    return false;
  }
  const uint64_t claimed_head = atomicAdd(head_address, 0ULL);
  if (claimed_head >= poolSliceStreamQueueDepth) {
    *message = pool_slice_stream_make_queue_entry(
        sequence,
        static_cast<uint32_t>(claimed_head),
        0,
        0,
        0,
        UINT32_MAX,
        POOL_SLICE_BATCH_FLAGS_ERROR);
  } else {
    *message = *pool_slice_stream_queue_entry(
        receive_batches,
        source_pe,
        queue,
        static_cast<uint32_t>(claimed_head),
        config.route_capacity);
  }
  *local_reader = UINT32_MAX;
  return true;
}

static __device__ __forceinline__ void
pool_slice_dynamic_read_finish_data_head(
    uint64_t* control,
    uint32_t source_pe,
    uint32_t queue,
    uint32_t local_readers) {
  auto* head = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_head(
          control, source_pe, queue));
  uint64_t* claim =
      pool_slice_stream_queue_claim(control, source_pe, queue);
  auto* completed = reinterpret_cast<uint32_t*>(claim) + 1;
  // The acquire-release RMW chain carries every reader CTA's gather stores
  // to the unique last completer before it publishes the new queue head.
  const uint32_t completed_readers =
      dae_atomic_fetch_add_acq_rel_gpu(completed, 1U) + 1;
  if (completed_readers == local_readers) {
    dae_atomic_add_release_gpu(
        reinterpret_cast<uint64_t*>(head), 1ULL);
    dae_atomic_store_release_gpu(claim, 0);
  }
}

static __device__ __forceinline__ void pool_slice_stream_queue_error(
    const PoolSliceConfig& config, uint32_t* shared_status) {
  atomicCAS(
      shared_status,
      static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
      static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
  pool_slice_set_status(config, POOL_SLICE_STATUS_BATCH);
}

static __device__ __forceinline__ bool pool_slice_dynamic_read_data_valid(
    const PoolSliceQueueEntry& message,
    uint32_t source_pe,
    uint32_t queue,
    const PoolSliceConfig& config,
    uint64_t* control,
    uint64_t sequence) {
  const uint64_t head = atomicAdd(
      reinterpret_cast<unsigned long long*>(
          pool_slice_stream_queue_head(control, source_pe, queue)),
      0ULL);
  return message.sequence == sequence && message.message_index == head &&
      message.opcode == POOL_SLICE_QUEUE_DATA &&
      message.flags == POOL_SLICE_BATCH_FLAGS_NONE &&
      message.row_begin < message.row_end &&
      message.row_end <= config.token_capacity &&
      message.ready_slot < poolSliceMaxDataGroups;
}

// Execute RESERVE_ROUTES as a CTA-wide metadata operation. Lane zero performs
// the small destination-row allocation; all threads then expand the packed
// route words and build this source's immutable ReduceAdd plans. No activation
// DATA dependency is consulted, so this overlaps the direct payload path.
template <bool WeightedReturn>
static __device__ __noinline__ void
pool_slice_stream_execute_reserve_routes(
    const PoolSliceQueueEntry& message,
    uint32_t source_pe,
    uint32_t queue,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    PoolSliceReceiveBatch* receive_routes,
    uint64_t* combine_rows,
    uint64_t* control,
    uint64_t sequence,
    uint32_t* shared_status,
    uint32_t thread_id) {
  __shared__ PoolSlicePublishBatch shared_batch;
  __shared__ uint32_t shared_valid;

  if (thread_id == 0) {
    const uint64_t head = atomicAdd(
        reinterpret_cast<unsigned long long*>(
            pool_slice_stream_queue_head(control, source_pe, queue)),
        0ULL);
    shared_batch = *pool_slice_stream_batch(
        receive_batches, source_pe, config.route_capacity);
    const PoolSlicePublishBatch& batch = shared_batch;
    bool valid = message.sequence == sequence &&
        message.message_index == head &&
        message.opcode == POOL_SLICE_QUEUE_RESERVE_ROUTES &&
        message.flags == POOL_SLICE_BATCH_FLAGS_NONE &&
        source_pe < config.num_pes && queue == 0 &&
        message.row_begin == 0 &&
        message.row_end <= config.token_capacity &&
        batch.sequence == sequence && batch.source_pe == source_pe &&
        batch.target_pe == config.my_pe &&
        batch.active_rows == message.row_end &&
        batch.route_begin <= batch.route_end &&
        batch.route_end <= config.route_capacity &&
        batch.flags == POOL_SLICE_BATCH_FLAGS_NONE;
    uint32_t route_cursor = batch.route_begin;
    uint32_t source_rows = 0;
    uint32_t source_batches = 0;
    for (uint32_t reader = 0; reader < config.local_readers; ++reader) {
      const uint32_t count = batch.reader_counts[reader];
      if (route_cursor > batch.route_end ||
          count > batch.route_end - route_cursor)
        valid = false;
      auto* tail = reinterpret_cast<unsigned long long*>(
          control + poolSliceControlReaderRowCount + reader);
      const uint64_t base_row = atomicAdd(
          tail, static_cast<unsigned long long>(count));
      if (base_row > config.expert_capacity_rows ||
          count > config.expert_capacity_rows - base_row)
        valid = false;
      PoolSliceReceiveBatch& route = receive_routes[
          static_cast<uint64_t>(reader) * config.num_pes + source_pe];
      route.sequence = sequence;
      route.base_row = static_cast<uint32_t>(base_row);
      route.source_begin = route_cursor;
      route.row_count = count;
      route.source_pe = source_pe;
      route.local_reader = reader;
      route.flags = valid ? POOL_SLICE_BATCH_FLAGS_NONE
                          : POOL_SLICE_BATCH_FLAGS_ERROR;
      route_cursor += count;
      source_rows += count;
      source_batches += count != 0;
    }
    valid &= route_cursor == batch.route_end &&
        source_rows == batch.route_end - batch.route_begin;
    if (!valid) {
      for (uint32_t reader = 0; reader < config.local_readers; ++reader) {
        receive_routes[
            static_cast<uint64_t>(reader) * config.num_pes + source_pe]
            .flags = POOL_SLICE_BATCH_FLAGS_ERROR;
      }
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
      pool_slice_set_status(config, POOL_SLICE_STATUS_BATCH);
    }
    shared_valid = valid;
    atomicAdd(
        reinterpret_cast<unsigned long long*>(control + 5),
        static_cast<unsigned long long>(source_rows != 0));
    atomicAdd(
        reinterpret_cast<unsigned long long*>(control + 6),
        static_cast<unsigned long long>(source_batches));
  }
  __syncthreads();

  if constexpr (WeightedReturn) {
    const PoolSlicePublishBatch batch = shared_batch;
    const uint32_t* source_routes = pool_slice_stream_route_words(
        receive_batches, source_pe, config, batch);
    if (shared_valid != 0) {
      for (uint32_t reader = 0; reader < config.local_readers; ++reader) {
        const PoolSliceReceiveBatch route = receive_routes[
            static_cast<uint64_t>(reader) * config.num_pes + source_pe];
        const uint32_t packed_begin = route.source_begin - batch.route_begin;
        for (uint32_t relative = thread_id;
             relative < route.row_count;
             relative += blockDim.x) {
          const uint32_t route_word = source_routes[packed_begin + relative];
          const uint32_t compact_row = route_word & 0xffffU;
          if (compact_row >= batch.active_rows) {
            atomicCAS(
                shared_status,
                static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
                static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
            continue;
          }
          combine_rows[
              (static_cast<uint64_t>(reader) * config.num_pes + source_pe) *
                      config.token_capacity +
                  compact_row] =
              (static_cast<uint64_t>(route_word & 0xffff0000U) << 16) |
              static_cast<uint32_t>(route.base_row + relative);
        }
      }
    }
    __syncthreads();
    pool_slice_build_reduce_add_plans_source(
        source_pe,
        config,
        batch,
        combine_rows,
        control,
        sequence,
        shared_status,
        thread_id);
    __syncthreads();
  }

  if (thread_id == 0) {
    // RouteReady covers receive_routes, the weighted reverse map, and every
    // source-owned ReduceAdd plan. DATA needs only the first object; combine
    // acquires the same publication without a later plan-build phase.
    dae_atomic_store_release_gpu(
        control + poolSliceControlStreamRouteReady + source_pe, sequence);
    dae_atomic_add_release_gpu(
        pool_slice_stream_queue_head(control, source_pe, queue), 1);
    dae_atomic_store_release_gpu(
        pool_slice_stream_queue_claim(control, source_pe, queue), 0);
  }
  __syncthreads();
}

// Execute the metadata/control opcodes at a claimed queue head. DATA is
// handled by the full CTA in the caller.  END is an ordered retirement marker;
// no destination-side group count participates in completion.
static __device__ __noinline__ void pool_slice_stream_execute_queue_control(
    const PoolSliceQueueEntry& message,
    uint32_t source_pe,
    uint32_t queue,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    uint64_t* control,
    uint64_t* signal_array,
    uint64_t sequence,
    uint64_t return_value,
    uint32_t* shared_status) {
  bool valid = message.sequence == sequence &&
      source_pe < config.num_pes &&
      queue < poolSliceMaxStreamQueues &&
      message.message_index < poolSliceStreamQueueDepth &&
      message.flags == POOL_SLICE_BATCH_FLAGS_NONE;
  const uint64_t head = atomicAdd(
      reinterpret_cast<unsigned long long*>(
          pool_slice_stream_queue_head(control, source_pe, queue)),
      0ULL);
  valid &= message.message_index == head;
  switch (message.opcode) {
    case POOL_SLICE_QUEUE_END: {
      constexpr uint32_t queue_count = poolSliceMaxStreamQueues;
      const uint32_t queue_index = source_pe * queue_count + queue;
      const uint64_t bit = 1ULL << queue_index;
      auto* retired = reinterpret_cast<unsigned long long*>(
          control + poolSliceControlStreamQueueRetiredMask);
      const uint64_t previous = dae_atomic_fetch_or_acq_rel_gpu(
          reinterpret_cast<uint64_t*>(retired), bit);
      valid &= (previous & bit) == 0;
      const uint64_t source_mask =
          ((1ULL << queue_count) - 1) << (source_pe * queue_count);
      if (((previous | bit) & source_mask) == source_mask) {
        const PoolSlicePublishBatch batch =
            *pool_slice_stream_batch(
                receive_batches, source_pe, config.route_capacity);
        if (batch.route_begin == batch.route_end) {
          if (source_pe == config.my_pe) {
            pool_slice_signal_release_local(
                signal_array + config.signal_base + config.my_pe,
                return_value);
          } else {
            pool_slice_signal_set_remote(
                signal_array + config.signal_base + config.my_pe,
                return_value,
                source_pe);
          }
        }
      }
      break;
    }
    default:
      valid = false;
      break;
  }
  if (!valid)
    pool_slice_stream_queue_error(config, shared_status);
}

// A claimed control head can run through the immediately following control
// instructions without returning to CTA-wide arbitration.  Stop precisely at
// DATA (which may still be waiting on its independent data plane) or
// after END.  The queue remains in order and the claim is released once.
static __device__ __noinline__ void pool_slice_stream_drain_queue_control(
    PoolSliceQueueEntry message,
    uint32_t source_pe,
    uint32_t queue,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    uint64_t* control,
    uint64_t* signal_array,
    uint64_t sequence,
    uint64_t return_value,
    uint32_t* shared_status) {
  auto* head = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_head(control, source_pe, queue));
  auto* claim = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_claim(control, source_pe, queue));
  while (message.opcode != POOL_SLICE_QUEUE_DATA) {
    pool_slice_stream_execute_queue_control(
        message,
        source_pe,
        queue,
        config,
        receive_batches,
        control,
        signal_array,
        sequence,
        return_value,
        shared_status);
    const bool ended = message.opcode == POOL_SLICE_QUEUE_END;
    const uint64_t next = atomicAdd(head, 1ULL) + 1;
    if (ended)
      break;
    if (next >= poolSliceStreamQueueDepth) {
      pool_slice_stream_queue_error(config, shared_status);
      break;
    }
    message = *pool_slice_stream_queue_entry(
        receive_batches,
        source_pe,
        queue,
        static_cast<uint32_t>(next),
        config.route_capacity);
  }
  dae_atomic_store_release_gpu(reinterpret_cast<uint64_t*>(claim), 0);
}

template <uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_return_unweighted(
    const PoolSliceConfig& config,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t compute_barrier_base,
    uint32_t thread_id,
    uint64_t sequence,
    uint64_t return_value,
    uint32_t* shared_status) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  const auto* receive_batches =
      reinterpret_cast<const PoolSlicePublishBatch*>(
          config.receive_batches_address);
  const auto* receive_routes =
      reinterpret_cast<const PoolSliceReceiveBatch*>(
          config.receive_routes_address);
  const auto* expert_output = reinterpret_cast<const uint8_t*>(
      config.expert_output_address);
  auto* return_inbox = reinterpret_cast<uint8_t*>(
      config.return_inbox_address);
  if (warp != 0) {
    constexpr uint32_t worker_warps = TotalWarps - 1;
    const uint32_t worker_slot = warp - 1;
    const uint32_t num_batches = config.local_readers * config.num_pes;
    for (uint32_t task =
             config.pool_rank + worker_slot * config.pool_count;
         task < num_batches;
         task += worker_warps * config.pool_count) {
      const uint32_t reader = task / config.num_pes;
      const uint32_t source_pe = task % config.num_pes;
      uint32_t compute_ready = 0;
      while (compute_ready == 0) {
        if (lane == 0)
          compute_ready = pool_slice_barrier_ready(
              bars + compute_barrier_base + reader);
        compute_ready = __shfl_sync(0xffffffffU, compute_ready, 0);
        if (compute_ready == 0)
          __nanosleep(barrierPollSleepCycles);
      }
      const PoolSliceReceiveBatch route = receive_routes[
          static_cast<uint64_t>(reader) * config.num_pes + source_pe];
      if (route.sequence != sequence || route.source_pe != source_pe ||
          route.local_reader != reader ||
          route.flags != POOL_SLICE_BATCH_FLAGS_NONE ||
          route.source_begin > config.route_capacity ||
          route.row_count > config.route_capacity - route.source_begin ||
          route.base_row > config.expert_capacity_rows ||
          route.row_count > config.expert_capacity_rows - route.base_row) {
        if (lane == 0) {
          atomicCAS(
              shared_status,
              static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
              static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
        }
        continue;
      }
      pool_slice_put_nbi_warp(
          return_inbox +
              static_cast<uint64_t>(route.source_begin) * config.row_bytes,
          expert_output +
              (static_cast<uint64_t>(reader) *
                   config.expert_capacity_rows +
               route.base_row) *
                  config.row_bytes,
          static_cast<uint64_t>(route.row_count) * config.row_bytes,
          source_pe,
          config.my_pe,
          lane);
    }
  }

  if (config.pool_rank == 0 && warp == 0) {
    bool ready = lane >= config.local_readers;
    while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
      if (!ready)
        ready = pool_slice_barrier_ready(
            bars + compute_barrier_base + lane);
      if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
    pool_slice_record_profile(
        g_events, poolSliceProfileComputeReady, lane);
  }
  __syncthreads();
  pool_slice_quiet_block();
  if (thread_id == 0) {
    if (*shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(*shared_status));
    dae_atomic_store_release_gpu(
        control + poolSliceControlReturnGeneration + config.pool_rank,
        sequence);
  }
  __syncthreads();

  if (config.pool_rank == 0 && warp == 0) {
    pool_slice_wait_generation_warp(
        control + poolSliceControlReturnGeneration,
        config.pool_count,
        sequence,
        lane);
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnPayloadDone, lane);
    pool_slice_publish_phase_parallel(
        signal_array,
        config.signal_base + config.my_pe,
        return_value,
        config.my_pe,
        config.num_pes,
        lane,
        receive_batches,
        sequence);

    bool returned_ready = lane >= config.num_pes;
    while (__ballot_sync(0xffffffffU, returned_ready) != 0xffffffffU) {
      if (!returned_ready) {
        returned_ready = pool_slice_signal_fetch(
            signal_array + config.signal_base + lane,
            lane == config.my_pe) >= return_value;
      }
      if (__ballot_sync(0xffffffffU, returned_ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnSignalsClosed, lane);
    if (lane == 0) {
      dae_atomic_store_release_gpu(
          control + poolSliceControlScatterStart, sequence);
    }
    __syncwarp();
  }

  pool_slice_wait_value_warp(
      control + poolSliceControlScatterStart, sequence, lane);
  const auto* origins = reinterpret_cast<const uint32_t*>(
      config.send_origin_rows_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);
  const auto* return_rows = reinterpret_cast<const uint8_t*>(
      config.return_inbox_address);
  const uint32_t global_warp = config.pool_rank * TotalWarps + warp;
  const uint32_t global_warps = config.pool_count * TotalWarps;
  for (uint32_t route = global_warp;
       route < config.active_rows;
       route += global_warps) {
    uint32_t origin = 0;
    if (lane == 0)
      origin = origins[route];
    origin = __shfl_sync(0xffffffffU, origin, 0);
    pool_slice_copy_warp(
        returned + static_cast<uint64_t>(origin) * config.row_bytes,
        return_rows + static_cast<uint64_t>(route) * config.row_bytes,
        config.row_bytes,
        lane);
  }
  __syncthreads();
  if (thread_id == 0) {
    if (*shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(*shared_status));
    dae_atomic_store_release_gpu(
        control + poolSliceControlScatterGeneration + config.pool_rank,
        sequence);
  }
  __syncthreads();

  if (config.pool_rank == 0 && warp == 0) {
    pool_slice_wait_generation_warp(
        control + poolSliceControlScatterGeneration,
        config.pool_count,
        sequence,
        lane);
    if (lane == 0) {
      control[3] = config.num_pes;
      if (g_events != nullptr) {
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileScatterDone] =
            cuda::ptx::get_sreg_globaltimer();
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileDone] =
            cuda::ptx::get_sreg_globaltimer();
      }
    }
    __syncwarp();
  }
}

// Direct-source, dynamically grouped dispatch. Metadata and activation data
// have independent readiness generations. Payload CTAs alternate source-group
// transmission with ready destination queue heads, so a destination copy can
// begin before either the local send plane or the global sender set closes.
// Ordered END bits alone retire the dynamic read and release ordinary VDCores
// readers; the destination never reconstructs producer group counts.
template <bool WeightedReturn, bool HostDataPlane, uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_exchange_streaming(
    const PoolSliceConfig& config,
    const PoolSliceHostConfig* host_config,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t compute_barrier_base,
    uint32_t thread_id,
    uint64_t sequence,
    uint64_t return_value) {
  __shared__ uint32_t shared_status;
  __shared__ uint32_t shared_first_payload;
  __shared__ uint32_t shared_first_gather;
  __shared__ uint32_t shared_send_task;
  __shared__ uint32_t shared_next_send_task;
  __shared__ uint32_t shared_queue_candidate;
  __shared__ uint32_t shared_queue_index;
  __shared__ uint32_t shared_queue_reader;
  __shared__ uint32_t shared_queue_valid;
  __shared__ uint32_t shared_probe;
  __shared__ uint32_t shared_complete;
  __shared__ PoolSliceQueueEntry shared_queue_message;

  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  auto* send_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.send_batches_address);
  auto* receive_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.receive_batches_address);
  auto* combine_rows = reinterpret_cast<uint64_t*>(
      config.combine_rows_address);
  auto* receive_routes = reinterpret_cast<PoolSliceReceiveBatch*>(
      config.receive_routes_address);
  const auto* send_offsets = reinterpret_cast<const uint32_t*>(
      config.send_offsets_address);
  const auto* send_token_rows = reinterpret_cast<const uint32_t*>(
      config.send_token_rows_address);
  const auto* send_token_counts = reinterpret_cast<const uint32_t*>(
      config.send_token_counts_address);
  const auto* token_pool = reinterpret_cast<const uint8_t*>(
      config.token_pool_address);
  auto* delivery_pool = reinterpret_cast<uint8_t*>(
      config.delivery_pool_address);
  auto* expert_input = reinterpret_cast<uint8_t*>(
      config.expert_input_address);

  if (thread_id == 0) {
    shared_status = POOL_SLICE_STATUS_OK;
    shared_first_payload = 0;
    shared_first_gather = 0;
    shared_probe = config.pool_rank %
        (config.num_pes * poolSliceMaxStreamQueues);
    shared_next_send_task = config.pool_count == 1
        ? 0
        : config.pool_rank - 1;
  }
  if (g_events != nullptr) {
    constexpr uint32_t last_profile_event = poolSliceProfileAllReadersReady;
    uint64_t* block_events =
        g_events + static_cast<uint64_t>(blockIdx.x) * numProfileEvents;
    for (uint32_t event = poolSliceProfileStart + thread_id;
         event <= last_profile_event;
         event += blockDim.x)
      block_events[event] = 0;
  }
  __syncthreads();

  if (config.pool_rank == 0) {
    for (uint32_t index = thread_id; index < 5; index += blockDim.x)
      control[index] = 0;
    for (uint32_t reader = thread_id;
         reader < config.local_readers;
         reader += blockDim.x) {
      control[poolSliceControlReaderRowCount + reader] = 0;
      control[poolSliceControlReaderDataDone + reader] = 0;
    }
    // Only these three words are invocation-local scratch. The metadata
    // transport generations start at the next word and remain monotonic:
    // clearing them here could erase an already-arrived remote packet.
    for (uint32_t index = thread_id; index < 3; index += blockDim.x)
      control[poolSliceControlStreamSendTotal + index] = 0;
    for (uint32_t index = thread_id;
         index < poolSliceMaxPes * poolSliceMaxStreamQueues;
         index += blockDim.x) {
      control[poolSliceControlStreamQueueHead + index] = 0;
      control[poolSliceControlStreamQueueClaim + index] = 0;
    }
    if constexpr (WeightedReturn) {
      for (uint32_t index = thread_id;
           index < config.num_pes * poolSliceReturnGroupsPerSource;
           index += blockDim.x)
        control[poolSliceControlReturnGroupCount + index] = 0;
      if (thread_id == 0)
        control[poolSliceControlCombineFirstReady] = 0;
      const uint64_t combine_count =
          static_cast<uint64_t>(config.local_readers) * config.num_pes *
          config.token_capacity;
      for (uint64_t index = thread_id;
           index < combine_count;
           index += blockDim.x)
        combine_rows[index] = UINT64_MAX;
    }
    if (thread_id == 0) {
      if (g_events != nullptr) {
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileStart] =
            cuda::ptx::get_sreg_globaltimer();
      }
    }
    __syncthreads();

    for (uint32_t target_pe = thread_id;
         target_pe < config.num_pes;
         target_pe += blockDim.x) {
      PoolSlicePublishBatch& batch =
          *pool_slice_stream_batch(
              send_batches, target_pe, config.route_capacity);
      const uint32_t reader_begin = target_pe * config.local_readers;
      const uint32_t route_begin = send_offsets[reader_begin];
      const uint32_t route_end =
          send_offsets[reader_begin + config.local_readers];
      bool route_valid = route_begin <= route_end &&
          route_end <= config.active_rows &&
          send_token_counts[target_pe] <= config.token_capacity;
      batch.sequence = sequence;
      batch.source_pe = config.my_pe;
      batch.target_pe = target_pe;
      batch.active_rows = send_token_counts[target_pe];
      batch.flags = POOL_SLICE_BATCH_FLAGS_NONE;
      batch.route_begin = route_begin;
      batch.route_end = route_end;
      uint32_t reader_cursor = route_begin;
      for (uint32_t reader = 0;
           reader < poolSliceMaxLocalReaders;
           ++reader) {
        uint32_t count = 0;
        if (reader < config.local_readers) {
          const uint32_t next = send_offsets[reader_begin + reader + 1];
          if (next < reader_cursor || next > route_end) {
            route_valid = false;
          } else {
            count = next - reader_cursor;
            reader_cursor = next;
          }
        }
        batch.reader_counts[reader] = count;
      }
      if (!route_valid || reader_cursor != route_end) {
        batch.active_rows = 0;
        batch.flags = POOL_SLICE_BATCH_FLAGS_ERROR;
        batch.route_begin = 0;
        batch.route_end = 0;
        for (uint32_t reader = 0;
             reader < poolSliceMaxLocalReaders;
             ++reader)
          batch.reader_counts[reader] = 0;
      }
      pool_slice_stream_build_queues(
          target_pe, config, batch, send_batches);
    }
    __syncthreads();
    if (thread_id == 0) {
      uint64_t send_total = 0;
      uint32_t active_group_count = 0;
      for (uint32_t target_pe = 0;
           target_pe < config.num_pes;
           ++target_pe) {
        const uint32_t target_groups = pool_slice_stream_group_count(
            pool_slice_stream_batch(
                send_batches, target_pe, config.route_capacity)->active_rows,
            config.row_bytes,
            config.group_limit,
            config.num_pes);
        active_group_count = target_groups > active_group_count
            ? target_groups
            : active_group_count;
        if (target_pe != config.my_pe)
          send_total += target_groups;
      }
      control[poolSliceControlStreamSendTotal] = send_total;
      control[4] = active_group_count;
      const uint64_t previous_metadata_sequence =
          dae_atomic_load_relaxed_gpu(
              control + poolSliceControlStreamMetadataSourceSequence);
      control[poolSliceControlStreamMetadataSignalDelta] =
          sequence - previous_metadata_sequence;
      control[poolSliceControlStreamMetadataSourceSequence] = sequence;
      dae_atomic_store_release_gpu(
          control + poolSliceControlStart, sequence);
    }
  } else {
    pool_slice_wait_value_warp(
        control + poolSliceControlStart, sequence, lane);
  }
  __syncthreads();

  // Metadata is one source-owned envelope per destination. It is always an
  // independent fused message, so routing can be accepted while the selected
  // host, public-NVSHMEM, or raw-SGL data plane remains in flight.
  const uint64_t metadata_signal_delta = dae_atomic_load_relaxed_gpu(
      control + poolSliceControlStreamMetadataSignalDelta);
  const uint32_t dispatch_worker_count =
      pool_slice_stream_dispatch_worker_count(config, control);
  pool_slice_stream_publish_metadata<HostDataPlane, TotalWarps>(
      config,
      sequence,
      metadata_signal_delta,
      thread_id);
  __syncthreads();
  if constexpr (HostDataPlane) {
    if (config.pool_rank == 0 && thread_id == 0) {
      auto* generations = reinterpret_cast<uint64_t*>(
          host_config->producer_generations_address);
      dae_atomic_store_release_gpu(generations + config.num_pes, sequence);
    }
  }
  __syncthreads();

  // With multiple CTAs rank zero remains a control plane while all other
  // CTAs alternate direct sends and ready destination gathers. A one-CTA
  // correctness configuration first closes metadata, then runs the same
  // payload loop with the full block.
  if (config.pool_rank == 0 && warp == 0) {
    bool metadata_ready = lane >= config.num_pes;
    bool route_ready = lane >= config.num_pes;
    bool reader_expected_ready = lane >= config.local_readers;
    bool reader_released = lane >= config.local_readers;
    uint32_t reader_expected = 0;
    bool metadata_closed_recorded = false;
    bool dispatch_ready_published = false;
    bool first_reader_ready_recorded = false;
    bool all_readers_ready_recorded = false;
    bool first_data_recorded = false;
    bool data_done_recorded = false;
    bool gather_done_recorded = false;
    while (true) {
      if (!metadata_ready) {
        uint64_t observed = pool_slice_signal_fetch(
            control + poolSliceControlStreamMetadataTransportReady + lane,
            lane == config.my_pe);
        if (observed >= sequence) {
          pool_slice_stream_accept_metadata(
              lane,
              config,
              receive_batches,
              control,
              sequence,
              &shared_status);
          metadata_ready = true;
        }
      }
      const uint32_t metadata_mask =
          __ballot_sync(0xffffffffU, metadata_ready) &
          pool_slice_pe_mask(config.num_pes);
      const bool metadata_complete =
          metadata_mask == pool_slice_pe_mask(config.num_pes);
      if (metadata_complete && !reader_expected_ready) {
        // This scan runs once per local reader and touches only compact route
        // words. Gather completion can precede it: ReaderDataDone is monotonic,
        // so metadata and payload retain independent arrival order.
        for (uint32_t source_pe = 0;
             source_pe < config.num_pes;
             ++source_pe) {
          (void)dae_atomic_load_acquire_gpu(
              control + poolSliceControlStreamMetadataReady + source_pe);
          const PoolSlicePublishBatch batch = *pool_slice_stream_batch(
              receive_batches, source_pe, config.route_capacity);
          const uint32_t* source_routes = pool_slice_stream_route_words(
              receive_batches, source_pe, config, batch);
          reader_expected += pool_slice_stream_reader_data_groups(
              batch, source_routes, lane, config);
        }
        reader_expected_ready = true;
      }
      if (metadata_ready && !route_ready) {
        route_ready = dae_atomic_load_acquire_gpu(
                          control + poolSliceControlStreamRouteReady + lane) >=
            sequence;
      }
      const uint32_t route_mask =
          __ballot_sync(0xffffffffU, route_ready) &
          pool_slice_pe_mask(config.num_pes);
      if (reader_expected_ready && !reader_released) {
        const uint64_t completed = dae_atomic_load_acquire_gpu(
            control + poolSliceControlReaderDataDone + lane);
        if (completed >= reader_expected) {
          atomicSub(bars + dispatch_barrier_base + lane, 1);
          reader_released = true;
        }
      }
      const uint32_t reader_mask =
          __ballot_sync(0xffffffffU, reader_released) &
          pool_slice_pe_mask(config.local_readers);
      uint32_t stop = 0;
      if (lane == 0) {
        const uint64_t send_done = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlStreamSendDone);
        const uint64_t send_total = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlStreamSendTotal);
        if (!first_data_recorded && (send_done != 0 || send_total == 0)) {
          if (g_events != nullptr) {
            const uint64_t now = cuda::ptx::get_sreg_globaltimer();
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfileFirstPayload] = now;
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfileFirstDataPublished] = now;
          }
          first_data_recorded = true;
        }
        if (!data_done_recorded && send_done >= send_total) {
          if (g_events != nullptr) {
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfileDataPublished] =
                cuda::ptx::get_sreg_globaltimer();
          }
          data_done_recorded = true;
        }
        if (!metadata_closed_recorded &&
            metadata_complete) {
          if (g_events != nullptr) {
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfileMetadataClosed] =
                cuda::ptx::get_sreg_globaltimer();
          }
          metadata_closed_recorded = true;
        }
        if (!dispatch_ready_published &&
            route_mask == pool_slice_pe_mask(config.num_pes)) {
          // Every source-specific reverse map and ReduceAdd plan is complete.
          // Publish them while activation DATA and reader computation continue.
          dae_atomic_store_release_gpu(
              control + poolSliceControlDispatchReady, sequence);
          if (g_events != nullptr) {
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfilePlanReady] =
                cuda::ptx::get_sreg_globaltimer();
          }
          dispatch_ready_published = true;
        }
        if (!first_reader_ready_recorded && reader_mask != 0) {
          if (g_events != nullptr) {
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfileFirstReaderReady] =
                cuda::ptx::get_sreg_globaltimer();
          }
          first_reader_ready_recorded = true;
        }
        if (!all_readers_ready_recorded &&
            reader_mask == pool_slice_pe_mask(config.local_readers)) {
          if (g_events != nullptr) {
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfileAllReadersReady] =
                cuda::ptx::get_sreg_globaltimer();
          }
          all_readers_ready_recorded = true;
        }
        if (metadata_complete) {
          if (config.pool_count == 1) {
            stop = 1;
          } else {
            const uint64_t retired = dae_atomic_load_acquire_gpu(
                control + poolSliceControlStreamQueueRetiredMask);
            const uint64_t expected =
                pool_slice_stream_queue_retired_mask(config.num_pes);
            if (!gather_done_recorded && retired == expected) {
              if (g_events != nullptr) {
                g_events[static_cast<uint64_t>(blockIdx.x) *
                             numProfileEvents +
                         poolSliceProfileStreamGatherDone] =
                    cuda::ptx::get_sreg_globaltimer();
              }
              gather_done_recorded = true;
            }
            stop = send_done >= send_total && retired == expected &&
                route_mask == pool_slice_pe_mask(config.num_pes) &&
                reader_mask == pool_slice_pe_mask(config.local_readers);
          }
        }
      }
      stop = __shfl_sync(0xffffffffU, stop, 0);
      if (stop != 0)
        break;
      __nanosleep(barrierPollSleepCycles);
    }
    __syncwarp();
  }
  __syncthreads();

  const bool payload_executor = config.pool_count == 1 ||
      (config.pool_rank != 0 &&
       config.pool_rank <= dispatch_worker_count);
  if (payload_executor) {
    const uint32_t send_stride = config.pool_count == 1
        ? 1
        : dispatch_worker_count;
    while (true) {
      if (thread_id == 0) {
        const uint64_t send_total = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlStreamSendTotal);
        shared_send_task = shared_next_send_task < send_total
            ? shared_next_send_task
            : UINT32_MAX;
        shared_next_send_task += send_stride;
      }
      __syncthreads();
      const uint32_t send_total = static_cast<uint32_t>(
          dae_atomic_load_relaxed_gpu(
              control + poolSliceControlStreamSendTotal));
      bool did_work = false;
      if (shared_send_task < send_total) {
        uint32_t target_pe = 0;
        uint32_t group = 0;
        if (pool_slice_stream_decode_send_task(
              shared_send_task,
              config,
              send_token_counts,
              &target_pe,
              &group)) {
          pool_slice_stream_send_group<HostDataPlane>(
              target_pe,
              group,
              config,
              host_config,
              bars,
              write_barrier,
              control,
              g_events,
              sequence,
              &shared_status,
              &shared_first_payload,
              thread_id);
          did_work = true;
        }
      }

      constexpr uint32_t queue_count = poolSliceMaxStreamQueues;
      const uint32_t total_queue_count = config.num_pes * queue_count;
      if (thread_id == 0) {
        shared_queue_candidate = UINT32_MAX;
        shared_queue_index = UINT32_MAX;
      }
      __syncthreads();
      if (warp == 0) {
        // Test every queue head warp-parallel, starting at this worker's
        // round-robin probe. Claiming remains global, so ready CTAs can steal
        // work from any source without a preferred-head branch or retry.
        for (uint32_t base = 0;
             shared_queue_candidate == UINT32_MAX &&
             base < total_queue_count;
             base += 32) {
          const uint32_t offset = base + lane;
          bool ready = offset < total_queue_count;
          const uint32_t queue_index =
              (shared_probe + offset) % total_queue_count;
          const uint32_t source_pe = queue_index / queue_count;
          const uint32_t queue = queue_index % queue_count;
          ready = ready &&
              pool_slice_stream_queue_message_ready<
                  HostDataPlane, TotalWarps>(
                      source_pe,
                      queue,
                      config,
                      receive_batches,
                      control,
                      sequence,
                      true);
          const uint32_t ready_mask =
              __ballot_sync(0xffffffffU, ready);
          if (lane == 0 && shared_queue_candidate == UINT32_MAX &&
              ready_mask != 0) {
            shared_queue_candidate =
                base + static_cast<uint32_t>(__ffs(ready_mask) - 1);
          }
          __syncwarp();
        }
      }
      __syncthreads();
      if (thread_id == 0 && shared_queue_candidate != UINT32_MAX) {
        const uint32_t queue_index =
            (shared_probe + shared_queue_candidate) % total_queue_count;
        if (pool_slice_stream_claim_queue_head<HostDataPlane, TotalWarps>(
                queue_index,
                config,
                receive_batches,
                control,
                sequence,
                &shared_queue_message,
                &shared_queue_reader)) {
          shared_queue_index = queue_index;
          shared_probe = (queue_index + 1) % total_queue_count;
        }
      }
      __syncthreads();
      if (shared_queue_index != UINT32_MAX) {
        const uint32_t source_pe = shared_queue_index / queue_count;
        const uint32_t queue = shared_queue_index % queue_count;
        if (shared_queue_message.opcode == POOL_SLICE_QUEUE_DATA) {
          if (thread_id == 0) {
            shared_queue_valid = pool_slice_dynamic_read_data_valid(
                shared_queue_message,
                source_pe,
                queue,
                config,
                control,
                sequence);
            if (!shared_queue_valid) {
              pool_slice_stream_queue_error(config, &shared_status);
            } else if (
                atomicCAS(&shared_first_gather, 0U, 1U) == 0U &&
                g_events != nullptr) {
              g_events[
                  static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                  poolSliceProfileFirstGather] =
                  cuda::ptx::get_sreg_globaltimer();
            }
          }
          __syncthreads();
          if (shared_queue_valid) {
            PoolSliceDynamicReadExecutor<
                POOL_SLICE_DYNAMIC_READ_COPY>::
                execute<HostDataPlane, TotalWarps>(
                source_pe,
                shared_queue_message.row_begin,
                shared_queue_message.row_end,
                shared_queue_message.ready_slot |
                    (shared_queue_reader << 16),
                config,
                receive_batches,
                receive_routes,
                send_token_rows,
                token_pool,
                delivery_pool,
                expert_input,
                bars,
                write_barrier,
                sequence,
                &shared_status,
                thread_id);
          }
        } else if (
            shared_queue_message.opcode ==
            POOL_SLICE_QUEUE_RESERVE_ROUTES) {
          pool_slice_stream_execute_reserve_routes<WeightedReturn>(
              shared_queue_message,
              source_pe,
              queue,
              config,
              receive_batches,
              receive_routes,
              combine_rows,
              control,
              sequence,
              &shared_status,
              thread_id);
        } else {
          if (thread_id == 0) {
            pool_slice_stream_drain_queue_control(
                shared_queue_message,
                source_pe,
                queue,
                config,
                receive_batches,
                control,
                signal_array,
                sequence,
                return_value,
                &shared_status);
          }
        }
        __syncthreads();
        if (thread_id == 0 &&
            shared_queue_message.opcode == POOL_SLICE_QUEUE_DATA) {
          pool_slice_dynamic_read_finish_data_head(
              control,
              source_pe,
              queue,
              config.local_readers);
        }
        __syncthreads();
        did_work = true;
      }

      if (thread_id == 0) {
        const uint64_t retired = dae_atomic_load_acquire_gpu(
            control + poolSliceControlStreamQueueRetiredMask);
        const uint64_t expected =
            pool_slice_stream_queue_retired_mask(config.num_pes);
        const uint64_t send_done = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlStreamSendDone);
        const uint64_t expected_send = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlStreamSendTotal);
        shared_complete = retired == expected &&
            send_done >= expected_send;
      }
      __syncthreads();
      if (shared_complete != 0)
        break;
      if (!did_work)
        __nanosleep(barrierPollSleepCycles);
      __syncthreads();
    }
  }
  if (!payload_executor) {
    if (thread_id == 0) {
      const uint64_t expected_retired =
          pool_slice_stream_queue_retired_mask(config.num_pes);
      while (true) {
        const uint64_t retired = dae_atomic_load_acquire_gpu(
            control + poolSliceControlStreamQueueRetiredMask);
        const uint64_t send_done = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlStreamSendDone);
        const uint64_t send_total = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlStreamSendTotal);
        if (retired == expected_retired && send_done >= send_total)
          break;
        __nanosleep(barrierPollSleepCycles);
      }
    }
  }
  __syncthreads();

  if (config.pool_rank == 0 && config.pool_count == 1 && warp == 0) {
    const uint64_t send_done = dae_atomic_load_relaxed_gpu(
        control + poolSliceControlStreamSendDone);
    if (g_events != nullptr) {
      const uint64_t now = cuda::ptx::get_sreg_globaltimer();
      if (send_done != 0) {
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileFirstDataPublished] = now;
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileFirstPayload] = now;
      } else {
        g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                 poolSliceProfileFirstDataPublished] = now;
      }
      g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
               poolSliceProfileDataPublished] = now;
    }
    __syncwarp();
  }

  if (thread_id == 0) {
    if (shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(shared_status));
    dae_atomic_store_release_gpu(
        control + poolSliceControlDispatchGeneration + config.pool_rank,
        sequence);
  }
  __syncthreads();

  if (config.pool_rank == 0 && warp == 0) {
    pool_slice_wait_generation_warp(
        control + poolSliceControlDispatchGeneration,
        config.pool_count,
        sequence,
        lane);
    pool_slice_record_profile(
        g_events, poolSliceProfilePayloadDone, lane);
  }
  __syncthreads();

  if (config.pool_rank == 0 && warp == 0) {
    if (lane == 0) {
      uint64_t received_rows = 0;
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader) {
        const uint64_t rows = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlReaderRowCount + reader);
        received_rows += rows;
      }
      if (shared_status != POOL_SLICE_STATUS_OK)
        pool_slice_set_status(
            config, static_cast<PoolSliceStatus>(shared_status));
      control[1] = config.num_pes;
      control[2] = received_rows;
      if constexpr (WeightedReturn) {
        if (received_rows == 0 && g_events != nullptr) {
          control[poolSliceControlCombineFirstReady] = sequence;
          g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                   poolSliceProfileComputeReady] =
              cuda::ptx::get_sreg_globaltimer();
        }
      }
      // A one-CTA correctness assembly cannot coordinate metadata while it is
      // executing payloads, so close its already-built plans and reader
      // barriers here. Multi-CTA production assemblies published both early.
      if (config.pool_count == 1) {
        dae_atomic_store_release_gpu(
            control + poolSliceControlDispatchReady, sequence);
        for (uint32_t reader = 0;
             reader < config.local_readers;
             ++reader)
          atomicSub(bars + dispatch_barrier_base + reader, 1);
        if (g_events != nullptr) {
          const uint64_t now = cuda::ptx::get_sreg_globaltimer();
          g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                   poolSliceProfilePlanReady] = now;
          g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                   poolSliceProfileFirstReaderReady] = now;
          g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                   poolSliceProfileAllReadersReady] = now;
        }
      }
    }
    __syncwarp();
    pool_slice_record_profile(
        g_events, poolSliceProfileGatherReady, lane);
  }

  pool_slice_wait_value_warp(
      control + poolSliceControlDispatchReady, sequence, lane);
  if constexpr (WeightedReturn) {
    PoolSliceDynamicReadExecutor<
        POOL_SLICE_DYNAMIC_READ_REDUCE_ADD>::
        execute<HostDataPlane, TotalWarps>(
        config,
        host_config,
        bars,
        g_events,
        compute_barrier_base,
        thread_id,
        sequence,
        &shared_status);
    PoolSliceDynamicReadExecutor<
        POOL_SLICE_DYNAMIC_READ_REDUCE_ADD>::
        finish<TotalWarps>(
        config, g_events, thread_id, sequence, &shared_status);
  } else {
    pool_slice_return_unweighted<TotalWarps>(
        config,
        bars,
        signal_array,
        g_events,
        compute_barrier_base,
        thread_id,
        sequence,
        return_value,
        &shared_status);
  }
}


// A PoolInst macro operation invoked by every PoolSliceExchangeExecuteWarp
// thread. PoolInst is separate from CommInst, and this operator has one
// dispatch implementation: direct-source streaming gathered read.
template <bool WeightedReturn, uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_exchange(
    const PoolSliceConfig* config_pointer,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t compute_barrier_base,
    uint32_t thread_id) {
  static_assert(TotalWarps >= 3, "PoolInst requires coordinator plus workers");
  __shared__ PoolSliceConfig shared_config;

  if (thread_id == 0)
    shared_config = *config_pointer;
  __syncthreads();

  const PoolSliceConfig& config = shared_config;
  const uint64_t sequence = pool_slice_sequence(config);
  const uint64_t return_value = sequence;
  pool_slice_exchange_streaming<WeightedReturn, false, TotalWarps>(
      config,
      nullptr,
      bars,
      signal_array,
      g_events,
      write_barrier,
      dispatch_barrier_base,
      compute_barrier_base,
      thread_id,
      sequence,
      return_value);
}

// Host-routed EP is a distinct compile-time PoolInst. It reuses the complete
// base metadata/gather/reduce/scatter state machine and substitutes only the
// payload+ready publication sites above.
template <uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_host_weighted_exchange(
    const PoolSliceHostConfig* config_pointer,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t compute_barrier_base,
    uint32_t thread_id) {
  static_assert(TotalWarps >= 3, "PoolInst requires coordinator plus workers");
  __shared__ PoolSliceHostConfig shared_host_config;

  if (thread_id == 0)
    shared_host_config = *config_pointer;
  __syncthreads();

  const PoolSliceConfig& config = shared_host_config.pool;
  const uint64_t sequence = pool_slice_sequence(config);
  pool_slice_exchange_streaming<true, true, TotalWarps>(
      config,
      &shared_host_config,
      bars,
      signal_array,
      g_events,
      write_barrier,
      dispatch_barrier_base,
      compute_barrier_base,
      thread_id,
      sequence,
      sequence);

  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  if (config.pool_rank == 0 && warp == 0 && lane < config.num_pes &&
      lane != config.my_pe) {
    const auto* peers = reinterpret_cast<const PoolSliceHostPeer*>(
        shared_host_config.peers_address);
    auto* generations = reinterpret_cast<uint64_t*>(
        shared_host_config.producer_generations_address);
    const PoolSliceHostPeer peer = peers[lane];
    pool_host_publish_epoch_end_thread(
        reinterpret_cast<HostSglRingMemory*>(peer.ring_memory),
        generations + lane,
        sequence);
  }
  if (config.pool_rank == 0 && warp == 0) {
    __syncwarp();
    if (lane == 0 && g_events != nullptr) {
      g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
               poolSliceProfileDone] = cuda::ptx::get_sreg_globaltimer();
    }
  }
}
