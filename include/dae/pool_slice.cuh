#pragma once

#include "context.cuh"
#include "pool_host.cuh"
#include "pool_signal.cuh"
#include "pool_slice_abi.cuh"
#include "scoped_atomic.cuh"

#ifndef DAE_ENABLE_NVSHMEM
#error "pool_slice.cuh requires DAE_ENABLE_NVSHMEM"
#endif

#include <nvshmem.h>
#include <nvshmemx.h>
#include <non_abi/device/common/nvshmemi_common_device.cuh>

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

// NVSHMEM 3.4 does not publish a cooperative quiet wrapper, although its
// pinned device implementation exposes the scope-generic primitive. Keep the
// non-ABI dependency in this isolated helper so the pool macro can distribute
// QP completion work without changing any base VDCores operator.
static __device__ __forceinline__ void pool_slice_quiet_block() {
#ifdef __CUDA_ARCH__
  nvshmemi_quiet<NVSHMEMI_THREADGROUP_BLOCK>();
#endif
}

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
  bool ready = lane >= count;
  while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
    if (!ready)
      ready = dae_atomic_load_acquire_gpu(generations + lane) >= expected;
    if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
      __nanosleep(barrierPollSleepCycles);
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
  return local ? dae_atomic_load_acquire_gpu(address)
               : nvshmem_signal_fetch(address);
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

// Route words are the sole 8-byte-aligned local-copy plane. Keeping their
// copy separate lets every payload/descriptor call above remain branch-free.
static __device__ __forceinline__ void pool_slice_copy_route_words_warp(
    uint64_t* destination,
    const uint64_t* source,
    uint64_t words,
    uint32_t lane) {
  for (uint64_t index = lane; index < words; index += 32)
    destination[index] = source[index];
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
          nvshmemx_signal_op(
              signal_array + signal_id,
              value,
              NVSHMEM_SIGNAL_SET,
              target_pe);
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
      nvshmemx_signal_op(
          signal_array + signal_id,
          value,
          NVSHMEM_SIGNAL_SET,
          target_pe);
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
// near 512 KiB and no more than 32 rows: this amortizes IBGDA readiness and
// ordered-head overhead while preserving payload/metadata overlap. Sparse
// targets automatically use fewer groups.
static __device__ __forceinline__ uint32_t pool_slice_stream_group_count(
    uint32_t active_rows,
    uint32_t row_bytes,
    uint32_t max_groups) {
  if (active_rows == 0)
    return 0;
  constexpr uint64_t target_group_bytes = 512ULL * 1024;
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

static __device__ __forceinline__ void pool_slice_stream_group_range(
    uint32_t active_rows,
    uint32_t group_count,
    uint32_t group,
    uint32_t* row_begin,
    uint32_t* row_end) {
  *row_begin = static_cast<uint32_t>(
      static_cast<uint64_t>(active_rows) * group / group_count);
  *row_end = static_cast<uint32_t>(
      static_cast<uint64_t>(active_rows) * (group + 1) / group_count);
}

static __device__ __forceinline__ uint64_t* pool_slice_stream_data_ready(
    uint64_t* control, uint32_t source_pe, uint32_t group) {
  return control + poolSliceControlStreamDataReady +
      static_cast<uint64_t>(source_pe) * poolSliceMaxPoolBlocks + group;
}

// Queue entries are interleaved by slot. A producer therefore sends only the
// slot rounds that contain RESERVE/COPY/END instructions; the consumer still
// follows explicit in-order messages and never derives the producer's G.
static __device__ __forceinline__ uint32_t
pool_slice_stream_envelope_bytes(
    uint32_t active_rows,
    uint32_t row_bytes,
    uint32_t group_limit) {
  const uint32_t groups = pool_slice_stream_group_count(
      active_rows, row_bytes, group_limit);
  const uint32_t slot_rounds = 2 + (groups + 1) / 2;
  return sizeof(PoolSlicePublishBatch) +
      slot_rounds * poolSliceMaxStreamQueues *
          sizeof(PoolSliceQueueEntry);
}

static __device__ __forceinline__ uint32_t pool_slice_stream_queue_index(
    uint32_t source_pe, uint32_t queue) {
  return source_pe * poolSliceMaxStreamQueues + queue;
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
  const uint64_t packet_bytes = sizeof(PoolSliceMetadataEnvelope) +
      static_cast<uint64_t>(route_capacity) * sizeof(uint64_t);
  return reinterpret_cast<PoolSliceMetadataEnvelope*>(
      reinterpret_cast<uint8_t*>(storage) + peer * packet_bytes);
}

static __device__ __forceinline__ const PoolSliceMetadataEnvelope*
pool_slice_stream_envelope(
    const PoolSlicePublishBatch* storage,
    uint32_t peer,
    uint32_t route_capacity) {
  const uint64_t packet_bytes = sizeof(PoolSliceMetadataEnvelope) +
      static_cast<uint64_t>(route_capacity) * sizeof(uint64_t);
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

static __device__ __forceinline__ uint64_t* pool_slice_stream_route_words(
    PoolSlicePublishBatch* storage,
    uint32_t peer,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch& batch) {
  auto* envelope = reinterpret_cast<uint8_t*>(
      pool_slice_stream_envelope(storage, peer, config.route_capacity));
  return reinterpret_cast<uint64_t*>(
      envelope + pool_slice_stream_envelope_bytes(
                     batch.active_rows,
                     config.row_bytes,
                     config.group_limit));
}

static __device__ __forceinline__ const uint64_t*
pool_slice_stream_route_words(
    const PoolSlicePublishBatch* storage,
    uint32_t peer,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch& batch) {
  const auto* envelope = reinterpret_cast<const uint8_t*>(
      pool_slice_stream_envelope(storage, peer, config.route_capacity));
  return reinterpret_cast<const uint64_t*>(
      envelope + pool_slice_stream_envelope_bytes(
                     batch.active_rows,
                     config.row_bytes,
                     config.group_limit));
}

static __device__ __forceinline__ uint64_t
pool_slice_stream_queue_retired_mask(uint32_t num_pes) {
  const uint32_t queues = num_pes * poolSliceMaxStreamQueues;
  return queues == 64 ? ~0ULL : (1ULL << queues) - 1;
}

static __device__ __forceinline__ uint32_t pool_slice_stream_route_lower_bound(
    const uint64_t* rows,
    uint32_t begin,
    uint32_t count,
    uint32_t compact_row) {
  uint32_t low = 0;
  uint32_t high = count;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    const uint32_t value = static_cast<uint32_t>(rows[begin + middle]);
    if (value < compact_row)
      low = middle + 1;
    else
      high = middle;
  }
  return low;
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
// a producer concern only: COPY_ROWS carries its exact compact-row interval,
// and END is the sole consumer-visible termination condition.
static __device__ __noinline__ void pool_slice_stream_build_queues(
    uint32_t target_pe,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch& batch,
    PoolSlicePublishBatch* send_batches) {
  constexpr uint32_t queue_count = poolSliceMaxStreamQueues;
  const uint32_t groups = pool_slice_stream_group_count(
      batch.active_rows, config.row_bytes, config.group_limit);
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
          batch.active_rows, groups, group, &row_begin, &row_end);
      *pool_slice_stream_queue_entry(
          send_batches,
          target_pe,
          queue,
          slot,
          config.route_capacity) =
          pool_slice_stream_make_queue_entry(
              batch.sequence,
              slot,
              POOL_SLICE_QUEUE_COPY_ROWS,
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
  // synthetic send task nor a data-ready message.
  for (uint32_t index = 0; index + 1 < config.num_pes; ++index) {
    const uint32_t target = pool_slice_remote_first_pe(
        index, config.my_pe, config.num_pes);
    const uint32_t groups = pool_slice_stream_group_count(
        send_token_counts[target], config.row_bytes, config.group_limit);
    if (task < cursor + groups) {
      *target_pe = target;
      *group = task - cursor;
      return true;
    }
    cursor += groups;
  }
  return false;
}

// One CTA owns one dynamic group. Every warp issues direct puts from the
// authoritative source token slots; no source-side activation staging is
// materialized. Public-NVSHMEM groups publish readiness after CTA-local quiet;
// each readiness generation names exactly the compact rows it protects.
template <bool HostDataPlane>
static __device__ __noinline__ void pool_slice_stream_send_group(
    uint32_t target_pe,
    uint32_t group,
    const PoolSliceConfig& config,
    const PoolSliceHostConfig* host_config,
    const uint8_t* token_pool,
    uint8_t* delivery_pool,
    const uint32_t* send_token_rows,
    const uint32_t* send_token_counts,
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
      token_count, config.row_bytes, config.group_limit);
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
      token_count, group_count, group, &row_begin, &row_end);
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
      while (!pool_signal_ready(bars + write_barrier + chunk))
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
          control, config.my_pe, group);
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
          sequence,
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

  uint32_t waited_chunk = UINT32_MAX;
  for (uint32_t packed_row = row_begin + warp;
       packed_row < row_end;
       packed_row += blockDim.x / 32) {
    uint32_t source_row = 0;
    if (lane == 0)
      source_row = target_rows[packed_row];
    source_row = __shfl_sync(0xffffffffU, source_row, 0);
    if (source_row >= config.token_capacity) {
      if (lane == 0) {
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
      }
      continue;
    }
    const uint32_t chunk = source_row / config.write_chunk_rows;
    if (chunk != waited_chunk) {
      uint32_t write_ready = 0;
      while (write_ready == 0) {
        if (lane == 0)
          write_ready = pool_signal_ready(bars + write_barrier + chunk);
        write_ready = __shfl_sync(0xffffffffU, write_ready, 0);
        if (write_ready == 0)
          __nanosleep(barrierPollSleepCycles);
      }
      waited_chunk = chunk;
    }
    nvshmemx_putmem_nbi_warp(
        delivery_pool +
            (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
             packed_row) *
                config.row_bytes,
        token_pool +
            static_cast<uint64_t>(source_row) * config.row_bytes,
        static_cast<size_t>(config.row_bytes),
        target_pe);
  }

  __syncthreads();
  pool_slice_quiet_block();
  __syncthreads();
  if (thread_id == 0) {
    uint64_t* ready = pool_slice_stream_data_ready(
        control, config.my_pe, group);
    nvshmemx_signal_op(
        ready, sequence, NVSHMEM_SIGNAL_SET, target_pe);
    atomicAdd(
        reinterpret_cast<unsigned long long*>(
            control + poolSliceControlStreamSendDone),
        1ULL);
  }
  __syncthreads();
}

// A destination COPY_ROWS instruction is executed by the whole PoolInst CTA:
// one warp owns each
// local reader, and its lanes move that reader's matching activation rows.
// Queue-zero metadata reserves the complete (reader, source) span once.  A
// queue carries the exact compact interval, so the consumer never derives G.
template <bool WeightedReturn, uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_stream_gather_rows(
    uint32_t source_pe,
    uint32_t compact_begin,
    uint32_t compact_end,
    uint32_t local_reader,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    const PoolSliceReceiveBatch* receive_routes,
    uint64_t* combine_rows,
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
  static_assert(TotalWarps > 0);
  if (source_pe >= config.num_pes || compact_begin >= compact_end ||
      compact_end > config.token_capacity ||
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
  const uint64_t* source_routes = pool_slice_stream_route_words(
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
      const uint64_t route_word =
          source_routes[packed_route_begin + relative];
      const uint32_t compact_row = static_cast<uint32_t>(route_word);
      const uint32_t expected_row =
          compact_begin + relative - relative_begin;
      dense_thread_valid &= compact_row == expected_row;
    }
  }
  dense_remote =
      __syncthreads_and(!dense_remote || dense_thread_valid) && dense_remote;
  if (dense_remote) {
    if constexpr (WeightedReturn) {
      for (uint32_t relative = relative_begin + thread_id;
           relative < relative_end;
           relative += blockDim.x) {
        const uint64_t route_word =
            source_routes[packed_route_begin + relative];
        const uint32_t compact_row = static_cast<uint32_t>(route_word);
        combine_rows[
            (static_cast<uint64_t>(local_reader) * config.num_pes +
             source_pe) *
                    config.token_capacity +
                compact_row] =
            (route_word & 0xffffffff00000000ULL) |
            static_cast<uint32_t>(route.base_row + relative);
      }
    }
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
  } else {
    // Sparse and self-source gathers stripe rows across the compiled warps;
    // each warp keeps a full coalesced row copy and its precise writer wait.
    for (uint32_t relative = relative_begin + warp;
         relative < relative_end;
         relative += TotalWarps) {
      uint64_t route_word = 0;
      uint32_t compact_row = 0;
      if (lane == 0) {
        route_word = source_routes[packed_route_begin + relative];
        compact_row = static_cast<uint32_t>(route_word);
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
      if constexpr (WeightedReturn) {
        if (lane == 0) {
          combine_rows[
              (static_cast<uint64_t>(local_reader) * config.num_pes +
               source_pe) *
                      config.token_capacity +
                  compact_row] =
              (route_word & 0xffffffff00000000ULL) |
              static_cast<uint32_t>(route.base_row + relative);
        }
      }
      const uint8_t* source_address = delivery_pool +
          (static_cast<uint64_t>(source_pe) * config.token_capacity +
           compact_row) *
              config.row_bytes;
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
            write_ready =
                pool_signal_ready(bars + write_barrier + chunk);
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
pool_slice_weighted_source_shards(
    const PoolSliceConfig& config, uint32_t source_pe) {
  if (source_pe >= config.pool_count)
    return 0;
  return 1 + (config.pool_count - 1 - source_pe) / config.num_pes;
}

static __device__ __forceinline__ void pool_slice_weighted_shard_range(
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

static __device__ __forceinline__ uint64_t* pool_slice_weighted_return_ready(
    uint64_t* control, uint32_t destination_pe, uint32_t pool_rank) {
  return control + poolSliceControlReturnReady +
      static_cast<uint64_t>(destination_pe) * poolSliceMaxPoolBlocks +
      pool_rank;
}

static __device__ __forceinline__ uint32_t*
pool_slice_weighted_return_group_count(
    uint64_t* control, uint32_t source_pe, uint32_t group) {
  return reinterpret_cast<uint32_t*>(
      control + poolSliceControlReturnGroupCount +
      static_cast<uint64_t>(source_pe) *
          poolSliceReturnGroupsPerSource +
      group);
}

// Production EP return: reduce expert rows inside the destination pool slice,
// transfer one source-owned row shard with a payload-coupled generation, then
// reduce the at-most-one partial per destination slice at the source. PoolInst
// rank modulo PE owns the source and rank/PE owns its shard; this static map
// removes the destination-wide quiet and makes expected readiness local math.
template <bool HostDataPlane, uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_return_weighted(
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
  const auto* send_token_rows =
      reinterpret_cast<const uint32_t*>(config.send_token_rows_address);
  const auto* expert_output =
      reinterpret_cast<const uint8_t*>(config.expert_output_address);
  auto* delivery_pool =
      reinterpret_cast<uint8_t*>(config.delivery_pool_address);
  auto* return_inbox =
      reinterpret_cast<uint8_t*>(config.return_inbox_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);
  if (warp == 0) {
    bool ready = lane >= config.local_readers;
    while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
      if (!ready)
        ready = pool_signal_ready(bars + compute_barrier_base + lane);
      if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
    if (config.pool_rank == 0)
      pool_slice_record_profile(
          g_events, poolSliceProfileComputeReady, lane);
  }
  __syncthreads();

  if (thread_id == 0 && g_events != nullptr) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
             poolSliceProfileReturnReduceStart] =
        cuda::ptx::get_sreg_globaltimer();
  }

  const uint64_t receive_capacity =
      static_cast<uint64_t>(config.num_pes) * config.token_capacity;
  uint8_t* partial_staging =
      delivery_pool + receive_capacity * config.row_bytes;
  const uint32_t source_pe = config.pool_rank % config.num_pes;
  const uint32_t source_shard = config.pool_rank / config.num_pes;
  const PoolSlicePublishBatch batch = *pool_slice_stream_batch(
      receive_batches, source_pe, config.route_capacity);
  const bool batch_valid = batch.active_rows <= config.token_capacity &&
      batch.sequence == sequence && batch.source_pe == source_pe &&
      batch.target_pe == config.my_pe &&
      batch.flags == POOL_SLICE_BATCH_FLAGS_NONE;
  if (!batch_valid && thread_id == 0) {
    atomicCAS(
        shared_status,
        static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
        static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
  }
  const uint32_t rows = batch_valid ? batch.active_rows : 0;
  const uint32_t available_shards =
      pool_slice_weighted_source_shards(config, source_pe);
  const uint32_t active_shards =
      rows < available_shards ? rows : available_shards;
  uint32_t row_begin = 0;
  uint32_t row_end = 0;
  pool_slice_weighted_shard_range(
      rows, source_shard, active_shards, &row_begin, &row_end);
  const bool active_shard = row_begin < row_end;
  uint8_t* partial_output =
      partial_staging +
      static_cast<uint64_t>(source_pe) * config.token_capacity *
          config.row_bytes;
  if (warp != 0 && active_shard) {
    constexpr uint32_t worker_warps = TotalWarps - 1;
    const uint32_t worker_slot = warp - 1;
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

  if (thread_id == 0 && g_events != nullptr) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
             poolSliceProfileReturnReduceDone] =
        cuda::ptx::get_sreg_globaltimer();
  }

  // Keep fine-grained reduction sharding, then coalesce adjacent shards into
  // four transport groups. The last release/acquire counter contributor owns
  // the group's one contiguous put-with-signal.
  if (warp == 0 && active_shard) {
    const uint32_t active_groups =
        active_shards < poolSliceReturnGroupsPerSource
        ? active_shards
        : poolSliceReturnGroupsPerSource;
    const uint32_t group = static_cast<uint32_t>(
        (static_cast<uint64_t>(source_shard + 1) * active_groups - 1) /
        active_shards);
    uint32_t group_shard_begin = 0;
    uint32_t group_shard_end = 0;
    pool_slice_weighted_shard_range(
        active_shards,
        group,
        active_groups,
        &group_shard_begin,
        &group_shard_end);
    uint32_t previous = 0;
    if (lane == 0) {
      previous = dae_atomic_fetch_add_acq_rel_gpu(
          pool_slice_weighted_return_group_count(
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
      uint64_t* ready = pool_slice_weighted_return_ready(
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
          nvshmemx_putmem_signal_nbi_warp(
              destination,
              source,
              static_cast<size_t>(bytes),
              ready,
              sequence,
              NVSHMEM_SIGNAL_SET,
              source_pe);
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
  }
  __syncthreads();

  if (config.pool_rank == 0 && warp == 0) {
    const auto* send_token_counts =
        reinterpret_cast<const uint32_t*>(config.send_token_counts_address);
    const uint32_t source_shards =
        pool_slice_weighted_source_shards(config, config.my_pe);
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
          destination_shards < poolSliceReturnGroupsPerSource
          ? destination_shards
          : poolSliceReturnGroupsPerSource;
      if (group >= destination_groups)
        continue;
      const uint32_t destination_rank =
          config.my_pe + group * config.num_pes;
      uint64_t* ready = pool_slice_weighted_return_ready(
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

template <uint32_t TotalWarps>
static __device__ __noinline__ void pool_slice_stream_publish_metadata(
    const PoolSliceConfig& config,
    PoolSlicePublishBatch* send_batches,
    PoolSlicePublishBatch* receive_batches,
    const uint64_t* send_rows,
    uint64_t* control,
    uint64_t sequence,
    uint32_t thread_id) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;

  for (uint32_t index = warp;
       index < config.num_pes;
       index += TotalWarps) {
    const uint32_t target_pe = pool_slice_remote_first_pe(
        index, config.my_pe, config.num_pes);
    PoolSliceMetadataEnvelope* destination =
        pool_slice_stream_envelope(
            receive_batches, config.my_pe, config.route_capacity);
    PoolSliceMetadataEnvelope* source =
        pool_slice_stream_envelope(
            send_batches, target_pe, config.route_capacity);
    const PoolSlicePublishBatch* source_batch = &source->batch;
    const uint32_t envelope_bytes = pool_slice_stream_envelope_bytes(
        source_batch->active_rows, config.row_bytes, config.group_limit);
    const uint32_t route_count =
        source_batch->route_end - source_batch->route_begin;
    uint64_t* packed_routes = pool_slice_stream_route_words(
        send_batches, target_pe, config, *source_batch);
    for (uint32_t route = lane; route < route_count; route += 32) {
      packed_routes[route] =
          send_rows[source_batch->route_begin + route];
    }
    __syncwarp();

    const uint64_t packet_bytes = envelope_bytes +
        static_cast<uint64_t>(route_count) * sizeof(uint64_t);
    uint64_t* transport_ready =
        control + poolSliceControlStreamMetadataTransportReady + config.my_pe;
    if (target_pe == config.my_pe) {
      pool_slice_copy_warp(destination, source, packet_bytes, lane);
      __syncwarp();
      if (lane == 0)
        pool_slice_signal_release_local(transport_ready, sequence);
    } else {
      nvshmemx_putmem_signal_nbi_warp(
          destination,
          source,
          static_cast<size_t>(packet_bytes),
          transport_ready,
          sequence,
          NVSHMEM_SIGNAL_SET,
          target_pe);
    }
    __syncwarp();
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
  if (message->opcode != POOL_SLICE_QUEUE_COPY_ROWS)
    return !require_unclaimed || claim_state == 0;
  // COPY_ROWS is sharded by local reader. The low word assigns the next
  // reader and the high word counts completed readers; the head stays stable
  // until the final shard advances it.
  if (require_unclaimed &&
      static_cast<uint32_t>(claim_state) >= config.local_readers)
    return false;
  if (dae_atomic_load_acquire_gpu(
          control + poolSliceControlStreamRouteReady + source_pe) < sequence)
    return false;
  if (message->ready_slot >= poolSliceMaxPoolBlocks)
    return true;
  return source_pe == config.my_pe ||
      pool_slice_signal_fetch(
          pool_slice_stream_data_ready(
              control, source_pe, message->ready_slot),
          false) >= sequence;
}

// The scan is only over queue heads. There are at most sixteen heads (two per
// PE), so warp zero tests all of them and elects the first ready head with one
// ballot. Revalidation under the claim handles another PoolInst CTA advancing
// the same queue between scan and CAS.
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
        candidate.opcode == POOL_SLICE_QUEUE_COPY_ROWS) {
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
  if (!pool_slice_stream_queue_message_ready(
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
pool_slice_stream_finish_copy_queue_head(
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

static __device__ __forceinline__ bool pool_slice_stream_queue_copy_valid(
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
      message.opcode == POOL_SLICE_QUEUE_COPY_ROWS &&
      message.flags == POOL_SLICE_BATCH_FLAGS_NONE &&
      message.row_begin < message.row_end &&
      message.row_end <= config.token_capacity &&
      message.ready_slot < poolSliceMaxPoolBlocks;
}

// Execute the metadata/control opcodes at a claimed queue head. COPY_ROWS is
// handled by the full CTA in the caller.  END is an ordered retirement marker;
// no destination-side group count participates in completion.
static __device__ __noinline__ void pool_slice_stream_execute_queue_control(
    const PoolSliceQueueEntry& message,
    uint32_t source_pe,
    uint32_t queue,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    PoolSliceReceiveBatch* receive_routes,
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
    case POOL_SLICE_QUEUE_RESERVE_ROUTES: {
      valid &= queue == 0 && message.row_begin == 0 &&
          message.row_end <= config.token_capacity;
      const PoolSlicePublishBatch batch =
          *pool_slice_stream_batch(
              receive_batches, source_pe, config.route_capacity);
      valid &= batch.sequence == sequence && batch.source_pe == source_pe &&
          batch.target_pe == config.my_pe &&
          batch.active_rows == message.row_end &&
          batch.route_begin <= batch.route_end &&
          batch.route_end <= config.route_capacity &&
          batch.flags == POOL_SLICE_BATCH_FLAGS_NONE;
      uint32_t route_cursor = batch.route_begin;
      uint32_t source_rows = 0;
      uint32_t source_batches = 0;
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader) {
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
            static_cast<uint64_t>(reader) * config.num_pes +
            source_pe];
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
        for (uint32_t reader = 0;
             reader < config.local_readers;
             ++reader) {
          receive_routes[
              static_cast<uint64_t>(reader) * config.num_pes + source_pe]
              .flags = POOL_SLICE_BATCH_FLAGS_ERROR;
        }
      }
      atomicAdd(
          reinterpret_cast<unsigned long long*>(control + 5),
          static_cast<unsigned long long>(source_rows != 0));
      atomicAdd(
          reinterpret_cast<unsigned long long*>(control + 6),
          static_cast<unsigned long long>(source_batches));
      dae_atomic_store_release_gpu(
          control + poolSliceControlStreamRouteReady + source_pe, sequence);
      break;
    }
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
            nvshmemx_signal_op(
                signal_array + config.signal_base + config.my_pe,
                return_value,
                NVSHMEM_SIGNAL_SET,
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
// COPY_ROWS (which may still be waiting on its independent data plane) or
// after END.  The queue remains in order and the claim is released once.
static __device__ __noinline__ void pool_slice_stream_drain_queue_control(
    PoolSliceQueueEntry message,
    uint32_t source_pe,
    uint32_t queue,
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches,
    PoolSliceReceiveBatch* receive_routes,
    uint64_t* control,
    uint64_t* signal_array,
    uint64_t sequence,
    uint64_t return_value,
    uint32_t* shared_status) {
  auto* head = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_head(control, source_pe, queue));
  auto* claim = reinterpret_cast<unsigned long long*>(
      pool_slice_stream_queue_claim(control, source_pe, queue));
  while (message.opcode != POOL_SLICE_QUEUE_COPY_ROWS) {
    pool_slice_stream_execute_queue_control(
        message,
        source_pe,
        queue,
        config,
        receive_batches,
        receive_routes,
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
          compute_ready =
              pool_signal_ready(bars + compute_barrier_base + reader);
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
        ready = pool_signal_ready(bars + compute_barrier_base + lane);
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
  const auto* send_rows = reinterpret_cast<const uint64_t*>(
      config.send_rows_address);
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
    constexpr uint32_t last_profile_event = WeightedReturn
        ? poolSliceProfileReturnCtaDone
        : poolSliceProfileStreamGatherDone;
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
            config.group_limit);
        active_group_count = target_groups > active_group_count
            ? target_groups
            : active_group_count;
        if (target_pe != config.my_pe)
          send_total += target_groups;
      }
      control[poolSliceControlStreamSendTotal] = send_total;
      control[4] = active_group_count;
      dae_atomic_store_release_gpu(
          control + poolSliceControlStart, sequence);
    }
  } else {
    pool_slice_wait_value_warp(
        control + poolSliceControlStart, sequence, lane);
  }
  __syncthreads();

  if (config.pool_rank == 0) {
    pool_slice_stream_publish_metadata<TotalWarps>(
        config,
        send_batches,
        receive_batches,
        send_rows,
        control,
        sequence,
        thread_id);
  }
  __syncthreads();

  // With multiple CTAs rank zero remains a control plane while all other
  // CTAs alternate direct sends and ready destination gathers. A one-CTA
  // correctness configuration first closes metadata, then runs the same
  // payload loop with the full block.
  if (config.pool_rank == 0 && warp == 0) {
    bool metadata_ready = lane >= config.num_pes;
    bool metadata_closed_recorded = false;
    bool first_data_recorded = false;
    bool data_done_recorded = false;
    bool gather_done_recorded = false;
    while (true) {
      if (!metadata_ready) {
        const uint64_t observed = pool_slice_signal_fetch(
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
            metadata_mask == pool_slice_pe_mask(config.num_pes)) {
          if (g_events != nullptr) {
            g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                     poolSliceProfileMetadataClosed] =
                cuda::ptx::get_sreg_globaltimer();
          }
          metadata_closed_recorded = true;
        }
        if (metadata_mask == pool_slice_pe_mask(config.num_pes)) {
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
            stop = send_done >= send_total && retired == expected;
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

  uint32_t dispatch_worker_count = 1;
  if (config.pool_count > 1) {
    const uint64_t send_total = dae_atomic_load_relaxed_gpu(
        control + poolSliceControlStreamSendTotal);
    const uint64_t gather_work = control[4] *
        static_cast<uint64_t>(config.num_pes) * config.local_readers;
    uint64_t useful_workers = gather_work + send_total;
    useful_workers = useful_workers == 0 ? 1 : useful_workers;
    useful_workers = useful_workers < config.pool_count
        ? useful_workers
        : config.pool_count - 1;
    dispatch_worker_count = static_cast<uint32_t>(useful_workers);
  }
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
              token_pool,
              delivery_pool,
              send_token_rows,
              send_token_counts,
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
        const uint32_t offset = lane;
        bool ready = offset < total_queue_count;
        const uint32_t queue_index =
            (shared_probe + offset) % total_queue_count;
        const uint32_t source_pe = queue_index / queue_count;
        const uint32_t queue = queue_index % queue_count;
        ready = ready && pool_slice_stream_queue_message_ready(
                source_pe,
                queue,
                config,
                receive_batches,
                control,
                sequence,
                true);
        const uint32_t ready_mask = __ballot_sync(0xffffffffU, ready);
        if (lane == 0 && ready_mask != 0)
          shared_queue_candidate = __ffs(ready_mask) - 1;
      }
      __syncthreads();
      if (thread_id == 0 && shared_queue_candidate != UINT32_MAX) {
        const uint32_t queue_index =
            (shared_probe + shared_queue_candidate) % total_queue_count;
        if (pool_slice_stream_claim_queue_head(
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
        if (shared_queue_message.opcode == POOL_SLICE_QUEUE_COPY_ROWS) {
          if (thread_id == 0) {
            shared_queue_valid = pool_slice_stream_queue_copy_valid(
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
            pool_slice_stream_gather_rows<WeightedReturn, TotalWarps>(
                source_pe,
                shared_queue_message.row_begin,
                shared_queue_message.row_end,
                shared_queue_reader,
                config,
                receive_batches,
                receive_routes,
                combine_rows,
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
        } else {
          if (thread_id == 0) {
            pool_slice_stream_drain_queue_control(
                shared_queue_message,
                source_pe,
                queue,
                config,
                receive_batches,
                receive_routes,
                control,
                signal_array,
                sequence,
                return_value,
                &shared_status);
          }
        }
        __syncthreads();
        if (thread_id == 0 &&
            shared_queue_message.opcode == POOL_SLICE_QUEUE_COPY_ROWS) {
          pool_slice_stream_finish_copy_queue_head(
              control, source_pe, queue, config.local_readers);
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
    if (lane == 0) {
      uint64_t received_rows = 0;
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader) {
        const uint64_t rows = dae_atomic_load_relaxed_gpu(
            control + poolSliceControlReaderRowCount + reader);
        received_rows += rows;
        pool_signal_release(bars + dispatch_barrier_base + reader);
      }
      control[1] = config.num_pes;
      control[2] = received_rows;
      dae_atomic_store_release_gpu(
          control + poolSliceControlDispatchReady, sequence);
    }
    __syncwarp();
    pool_slice_record_profile(
        g_events, poolSliceProfileGatherReady, lane);
  }

  pool_slice_wait_value_warp(
      control + poolSliceControlDispatchReady, sequence, lane);
  if constexpr (WeightedReturn) {
    pool_slice_return_weighted<HostDataPlane, TotalWarps>(
        config,
        host_config,
        bars,
        g_events,
        compute_barrier_base,
        thread_id,
        sequence,
        &shared_status);
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
