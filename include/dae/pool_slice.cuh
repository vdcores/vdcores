#pragma once

#include "context.cuh"
#include "pool_slice_abi.cuh"

#ifndef DAE_ENABLE_NVSHMEM
#error "pool_slice.cuh requires DAE_ENABLE_NVSHMEM"
#endif

#include <nvshmem.h>
#include <nvshmemx.h>
#include <non_abi/device/common/nvshmemi_common_device.cuh>

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

static __device__ __forceinline__ void pool_slice_quiet(
    uint32_t num_pes, uint32_t thread_id) {
  // One remote peer has only one useful RC queue in the recommended profile;
  // a block-wide cooperative quiet costs more than lane 0 polling that queue.
  if (num_pes <= 2) {
    if (thread_id == 0)
      nvshmem_quiet();
    __syncthreads();
    return;
  }
  pool_slice_quiet_block();
}

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

static __device__ __forceinline__ bool pool_slice_valid_config(
    const PoolSliceConfig& config) {
  constexpr uint32_t total_warps =
      numComputeWarps + numMemoryWarps + numCommunicationWarps;
  constexpr uint32_t worker_warps = total_warps - 1;
  uint64_t required_expert_bytes = 0;
  return config.source_address != 0 &&
      config.token_pool_address != 0 &&
      config.delivery_pool_address != 0 &&
      config.expert_input_address != 0 &&
      config.expert_output_address != 0 &&
      config.return_inbox_address != 0 &&
      config.returned_address != 0 &&
      config.send_offsets_address != 0 &&
      config.send_rows_address != 0 &&
      config.send_origin_rows_address != 0 &&
      config.send_batches_address != 0 &&
      config.receive_batches_address != 0 &&
      config.receive_routes_address != 0 &&
      config.sequence_address != 0 &&
      config.group_ready_address != 0 &&
      config.control_address != 0 &&
      config.row_bytes >= poolSliceMinimumRowBytes &&
      config.row_bytes % poolSliceAlignmentBytes == 0 &&
      config.source_stride >= config.row_bytes &&
      config.source_stride % poolSliceAlignmentBytes == 0 &&
      config.pool_stride >= config.row_bytes &&
      config.pool_stride % poolSliceAlignmentBytes == 0 &&
      config.delivery_stride == config.row_bytes &&
      config.expert_row_stride == config.row_bytes &&
      config.return_stride >= config.row_bytes &&
      config.return_stride % poolSliceAlignmentBytes == 0 &&
      config.active_rows <= config.route_capacity &&
      config.token_capacity != 0 &&
      config.route_capacity != 0 &&
      config.expert_capacity_rows != 0 &&
      config.local_readers != 0 &&
      config.local_readers <= poolSliceMaxLocalReaders &&
      config.num_pes != 0 &&
      config.num_pes <= poolSliceMaxPes &&
      config.my_pe < config.num_pes &&
      config.signal_base <= config.signal_count &&
      config.num_pes <= config.signal_count - config.signal_base &&
      config.return_capacity_rows != 0 &&
      config.pack_warps != 0 &&
      config.pack_warps < worker_warps &&
      config.write_chunks != 0 &&
      config.write_chunk_rows != 0 &&
      config.write_chunks ==
          (config.token_capacity + config.write_chunk_rows - 1) /
              config.write_chunk_rows &&
      pool_slice_u64_product_fits(
          config.expert_capacity_rows,
          config.expert_row_stride,
          &required_expert_bytes) &&
      config.expert_stride >= required_expert_bytes &&
      config.expert_stride % poolSliceAlignmentBytes == 0;
}

static __device__ __forceinline__ uint64_t pool_slice_sequence(
    const PoolSliceConfig& config) {
  const auto* sequence = reinterpret_cast<const unsigned long long*>(
      config.sequence_address);
  return atomicAdd(
      const_cast<unsigned long long*>(sequence),
      static_cast<unsigned long long>(0));
}

static __device__ __forceinline__ uint64_t pool_slice_signal_value(
    uint64_t sequence, PoolSliceSignalPhase phase) {
  return (sequence - 1) * poolSliceSignalPhases +
      static_cast<uint32_t>(phase);
}

// The hot path deliberately supports fixed-width, aligned LLM rows.
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

static __device__ __forceinline__ void pool_slice_publish_phase_parallel(
    uint64_t* signal_array,
    uint32_t signal_id,
    uint64_t value,
    uint32_t my_pe,
    uint32_t num_pes,
    uint32_t lane) {
  if (num_pes <= 2) {
    for (uint32_t index = 0; index < num_pes; ++index) {
      const uint32_t unwrapped = index + my_pe + 1;
      const uint32_t target_pe =
          unwrapped >= num_pes ? unwrapped - num_pes : unwrapped;
      __syncwarp();
      if (lane == 0) {
        if (target_pe == my_pe) {
          __threadfence_system();
          atomicExch(
              reinterpret_cast<unsigned long long*>(
                  signal_array + signal_id),
              static_cast<unsigned long long>(value));
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
    if (target_pe == my_pe) {
      __threadfence_system();
      atomicExch(
          reinterpret_cast<unsigned long long*>(signal_array + signal_id),
          static_cast<unsigned long long>(value));
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

static __device__ __forceinline__ void pool_slice_record_first_payload(
    uint64_t* g_events,
    uint32_t* recorded,
    uint32_t lane) {
  if (lane == 0 && atomicCAS(recorded, 0U, 1U) == 0U &&
      g_events != nullptr) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
             poolSliceProfileFirstPayload] =
        cuda::ptx::get_sreg_globaltimer();
  }
  __syncwarp();
}

// A communication-specialized VDCores macro operation. It is invoked by every
// thread in a block selected by COMM_POOL_SLICE_EXCHANGE. The normal block
// shape remains 4 compute + 4 memory + 1 communication warp; this operator
// temporarily gives all nine warps communication roles without changing any
// ordinary virtual-core implementation.
static __device__ __noinline__ void pool_slice_exchange(
    const PoolSliceConfig* config_pointer,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t compute_barrier_base,
    uint32_t thread_id) {
  constexpr uint32_t total_warps =
      numComputeWarps + numMemoryWarps + numCommunicationWarps;

  __shared__ PoolSliceConfig shared_config;
  __shared__ uint32_t shared_status;
  __shared__ uint32_t shared_metadata_mask;
  __shared__ uint32_t shared_data_mask;
  __shared__ uint32_t shared_source_state[poolSliceMaxPes];
  __shared__ uint32_t shared_pack_done;
  __shared__ uint32_t shared_dispatch_issued;
  __shared__ uint32_t shared_return_next;
  __shared__ uint32_t shared_return_issued;
  __shared__ uint32_t shared_first_payload;
  __shared__ uint32_t shared_payload_sources;
  __shared__ uint32_t shared_dispatch_batches;
  __shared__ unsigned long long
      shared_reader_tails[poolSliceMaxLocalReaders];

  if (config_pointer == nullptr || bars == nullptr || signal_array == nullptr)
    return;

  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  if (thread_id == 0)
    shared_config = *config_pointer;
  __syncthreads();
  const PoolSliceConfig& config = shared_config;
  if (!pool_slice_valid_config(config)) {
    if (thread_id == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
    return;
  }

  const uint64_t sequence = pool_slice_sequence(config);
  if (sequence == 0 ||
      sequence - 1 >
          (UINT64_MAX - poolSliceSignalPhases) / poolSliceSignalPhases) {
    if (thread_id == 0)
      pool_slice_set_status(config, POOL_SLICE_STATUS_SEQUENCE);
    return;
  }

  const uint64_t metadata_value = pool_slice_signal_value(
      sequence, POOL_SLICE_SIGNAL_METADATA);
  const uint64_t data_value = pool_slice_signal_value(
      sequence, POOL_SLICE_SIGNAL_DATA);
  const uint64_t return_value = pool_slice_signal_value(
      sequence, POOL_SLICE_SIGNAL_RETURN);
  const uint32_t expected_mask = pool_slice_pe_mask(config.num_pes);
  auto* control = reinterpret_cast<unsigned long long*>(config.control_address);
  auto* group_ready = reinterpret_cast<unsigned long long*>(
      config.group_ready_address);
  auto* send_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.send_batches_address);
  auto* receive_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.receive_batches_address);
  auto* receive_routes = reinterpret_cast<PoolSliceReceiveBatch*>(
      config.receive_routes_address);
  const auto* send_offsets = reinterpret_cast<const uint32_t*>(
      config.send_offsets_address);
  const auto* send_rows = reinterpret_cast<const uint32_t*>(
      config.send_rows_address);
  const auto* token_pool = reinterpret_cast<const uint8_t*>(
      config.token_pool_address);
  auto* delivery_pool = reinterpret_cast<uint8_t*>(
      config.delivery_pool_address);
  auto* expert_input = reinterpret_cast<uint8_t*>(
      config.expert_input_address);

  if (thread_id == 0) {
    shared_status = POOL_SLICE_STATUS_OK;
    shared_metadata_mask = 0;
    shared_data_mask = 0;
    shared_pack_done = 0;
    shared_dispatch_issued = 0;
    shared_return_next = 0;
    shared_return_issued = 0;
    shared_first_payload = 0;
    shared_payload_sources = 0;
    shared_dispatch_batches = 0;
    *group_ready = 0;
  }
  for (uint32_t index = thread_id;
       index < poolSliceMaxPes;
       index += blockDim.x)
    shared_source_state[index] = 0;
  for (uint32_t index = thread_id;
       index < poolSliceMaxLocalReaders;
       index += blockDim.x)
    shared_reader_tails[index] = 0;
  for (uint32_t index = thread_id;
       index < poolSliceControlWords;
       index += blockDim.x)
    control[index] = 0;
  if (g_events != nullptr) {
    uint64_t* block_events =
        g_events + static_cast<uint64_t>(blockIdx.x) * numProfileEvents;
    for (uint32_t event = poolSliceProfileStart + thread_id;
         event <= poolSliceProfileScatterDone;
         event += blockDim.x)
      block_events[event] = 0;
  }
  __syncthreads();
  if (thread_id == 0) {
    __threadfence_system();
    if (g_events != nullptr) {
      g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
               poolSliceProfileStart] = cuda::ptx::get_sreg_globaltimer();
    }
  }
  __syncthreads();

  // Materialize one cache-line descriptor per target. The producer wrote only
  // route metadata; this pool core converts offsets into target-local counts.
  for (uint32_t target_pe = thread_id;
       target_pe < config.num_pes;
       target_pe += blockDim.x) {
    PoolSlicePublishBatch& batch = send_batches[target_pe];
    const uint32_t reader_begin = target_pe * config.local_readers;
    batch.sequence = sequence;
    batch.source_pe = config.my_pe;
    batch.target_pe = target_pe;
    batch.active_rows = config.active_rows;
    batch.flags = POOL_SLICE_BATCH_FLAGS_NONE;
    batch.route_begin = send_offsets[reader_begin];
    batch.route_end = send_offsets[reader_begin + config.local_readers];
    for (uint32_t reader = 0;
         reader < poolSliceMaxLocalReaders;
         ++reader) {
      batch.reader_counts[reader] = reader < config.local_readers
          ? send_offsets[reader_begin + reader + 1] -
                send_offsets[reader_begin + reader]
          : 0;
    }
  }
  __syncthreads();

  // Warp 0 owns publication and all phase signals. Each active lane posts one
  // peer descriptor so small control messages are not serialized by PE.
  if (warp == 0) {
    if (config.num_pes <= 2) {
      for (uint32_t index = 0; index < config.num_pes; ++index) {
        const uint32_t target_pe = pool_slice_remote_first_pe(
            index, config.my_pe, config.num_pes);
        PoolSlicePublishBatch* destination =
            receive_batches + config.my_pe;
        const PoolSlicePublishBatch* source = send_batches + target_pe;
        if (target_pe == config.my_pe) {
          pool_slice_copy_warp(
              destination, source, sizeof(PoolSlicePublishBatch), lane);
          __syncwarp();
          if (lane == 0) {
            __threadfence_system();
            atomicExch(
                reinterpret_cast<unsigned long long*>(
                    signal_array + config.signal_base + config.my_pe),
                static_cast<unsigned long long>(metadata_value));
          }
          __syncwarp();
        } else {
          nvshmemx_putmem_signal_nbi_warp(
              destination,
              source,
              sizeof(PoolSlicePublishBatch),
              signal_array + config.signal_base + config.my_pe,
              metadata_value,
              NVSHMEM_SIGNAL_SET,
              target_pe);
        }
      }
    } else if (lane < config.num_pes) {
      const uint32_t target_pe = pool_slice_remote_first_pe(
          lane, config.my_pe, config.num_pes);
      PoolSlicePublishBatch* destination = receive_batches + config.my_pe;
      const PoolSlicePublishBatch* source = send_batches + target_pe;
      if (target_pe == config.my_pe) {
        *destination = *source;
        __threadfence_system();
        atomicExch(
            reinterpret_cast<unsigned long long*>(
                signal_array + config.signal_base + config.my_pe),
            static_cast<unsigned long long>(metadata_value));
      } else {
        nvshmem_putmem_signal_nbi(
            destination,
            source,
            sizeof(PoolSlicePublishBatch),
            signal_array + config.signal_base + config.my_pe,
            metadata_value,
            NVSHMEM_SIGNAL_SET,
            target_pe);
      }
    }
    __syncwarp();
    if (lane == 0)
      nvshmem_fence();
    __syncwarp();
  }

  // No block barrier here: descriptor publication, route packing, and receive
  // polling are independent. The coordinator's signal scan supplies the
  // actual dependency edge, and the dispatch quiet is their convergence.

  const bool is_pack_warp = warp > 0 && warp <= config.pack_warps;
  const bool is_receive_warp = warp > config.pack_warps;

  // Pack route-major delivery rows as soon as the ordinary VDCores writer has
  // made its token-slot pool ready. Pack warps operate concurrently with the
  // receive workers ingesting remote descriptors.
  if (is_pack_warp) {
    const uint32_t pack_index = warp - 1;
    for (uint32_t route = pack_index;
         route < config.active_rows;
         route += config.pack_warps) {
      uint32_t source_row = 0;
      if (lane == 0)
        source_row = send_rows[route];
      source_row = __shfl_sync(0xffffffffU, source_row, 0);
      if (source_row >= config.token_capacity) {
        if (lane == 0) {
          atomicCAS(
              &shared_status,
              static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
              static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
        }
        continue;
      }

      const uint32_t write_chunk = source_row / config.write_chunk_rows;
      uint32_t write_ready = 0;
      while (write_ready == 0) {
        if (lane == 0) {
          write_ready = *reinterpret_cast<volatile int*>(
              bars + write_barrier + write_chunk) == 0;
        }
        write_ready = __shfl_sync(0xffffffffU, write_ready, 0);
        if (write_ready == 0)
          __nanosleep(barrierPollSleepCycles);
      }
      pool_slice_copy_warp(
          delivery_pool + static_cast<uint64_t>(route) * config.delivery_stride,
          token_pool + static_cast<uint64_t>(source_row) * config.pool_stride,
          config.row_bytes,
          lane);
    }
    __syncwarp();
    // Every lane wrote a disjoint vector of the symmetric source buffer. A
    // system fence before the release counter makes those writes NIC-visible.
    __threadfence_system();
    __syncwarp();
    if (lane == 0)
      atomicAdd(&shared_pack_done, 1U);
  }

  // Receive workers claim metadata-ready sources dynamically. Counts embedded
  // in the descriptor make route resolution local; after the data phase each
  // nonempty reader is one contiguous NBI GET into its assigned range.
  if (is_receive_warp) {
    while (*reinterpret_cast<volatile uint32_t*>(&shared_dispatch_issued) <
           config.num_pes) {
      uint32_t source_pe = UINT32_MAX;
      if (lane == 0) {
        const uint32_t ready_mask =
            *reinterpret_cast<volatile uint32_t*>(&shared_metadata_mask);
        for (uint32_t index = 0; index < config.num_pes; ++index) {
          const uint32_t candidate = pool_slice_remote_first_pe(
              index, config.my_pe, config.num_pes);
          if ((ready_mask & (1U << candidate)) != 0 &&
              atomicCAS(shared_source_state + candidate, 0U, 1U) == 0U) {
            source_pe = candidate;
            break;
          }
        }
      }
      source_pe = __shfl_sync(0xffffffffU, source_pe, 0);
      if (source_pe == UINT32_MAX) {
        __nanosleep(barrierPollSleepCycles);
        continue;
      }

      const PoolSlicePublishBatch batch = receive_batches[source_pe];
      uint32_t total_rows = 0;
      uint32_t batch_valid = 1;
      if (lane == 0) {
        batch_valid = batch.sequence == sequence &&
            batch.source_pe == source_pe &&
            batch.target_pe == config.my_pe &&
            batch.active_rows <= config.route_capacity &&
            batch.route_begin <= batch.route_end &&
            batch.route_end <= batch.active_rows &&
            batch.flags == POOL_SLICE_BATCH_FLAGS_NONE;
        uint32_t source_cursor = batch.route_begin;
        for (uint32_t reader = 0;
             reader < config.local_readers;
             ++reader) {
          const uint32_t count = batch.reader_counts[reader];
          if (count > batch.route_end - source_cursor) {
            batch_valid = 0;
            break;
          }
          const uint64_t base_row = atomicAdd(
              shared_reader_tails + reader,
              static_cast<unsigned long long>(count));
          if (base_row > config.expert_capacity_rows ||
              count > config.expert_capacity_rows - base_row) {
            batch_valid = 0;
          }
          PoolSliceReceiveBatch& route = receive_routes[
              static_cast<uint64_t>(reader) * config.num_pes + source_pe];
          route.sequence = sequence;
          route.base_row = base_row;
          route.source_begin = source_cursor;
          route.row_count = count;
          route.source_pe = source_pe;
          route.local_reader = reader;
          route.flags = POOL_SLICE_BATCH_FLAGS_NONE;
          route.reserved_u32[0] = 0;
          route.reserved_u32[1] = 0;
          route.reserved_u32[2] = 0;
          source_cursor += count;
          total_rows += count;
        }
        if (source_cursor != batch.route_end)
          batch_valid = 0;
        if (!batch_valid) {
          atomicCAS(
              &shared_status,
              static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
              static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
          total_rows = 0;
        }
      }
      total_rows = __shfl_sync(0xffffffffU, total_rows, 0);
      batch_valid = __shfl_sync(0xffffffffU, batch_valid, 0);
      __syncwarp();

      if (batch_valid && total_rows != 0) {
        while ((*reinterpret_cast<volatile uint32_t*>(&shared_data_mask) &
                (1U << source_pe)) == 0)
          __nanosleep(barrierPollSleepCycles);

        pool_slice_record_first_payload(
            g_events, &shared_first_payload, lane);
        if (lane == 0)
          atomicAdd(&shared_payload_sources, 1U);
        for (uint32_t reader = 0;
             reader < config.local_readers;
             ++reader) {
          const PoolSliceReceiveBatch route = receive_routes[
              static_cast<uint64_t>(reader) * config.num_pes + source_pe];
          if (route.row_count == 0)
            continue;
          pool_slice_get_nbi_warp(
              expert_input +
                  static_cast<uint64_t>(reader) * config.expert_stride +
                  route.base_row * config.expert_row_stride,
              delivery_pool +
                  static_cast<uint64_t>(route.source_begin) *
                      config.delivery_stride,
              static_cast<uint64_t>(route.row_count) * config.row_bytes,
              source_pe,
              config.my_pe,
              lane);
          if (lane == 0)
            atomicAdd(&shared_dispatch_batches, 1U);
        }
      }
      __syncwarp();
      if (lane == 0) {
        __threadfence_block();
        atomicExch(shared_source_state + source_pe, 2U);
        atomicAdd(&shared_dispatch_issued, 1U);
      }
    }
  }

  // The coordinator continuously scans all source phase words lane-parallel.
  // It publishes local data as soon as packing completes and waits only until
  // every source batch has been issued, leaving all remote GETs in flight.
  if (warp == 0) {
    bool data_published = false;
    bool metadata_closed_recorded = false;
    while (!data_published ||
           *reinterpret_cast<volatile uint32_t*>(&shared_dispatch_issued) <
               config.num_pes) {
      const uint64_t observed = lane < config.num_pes
          ? nvshmem_signal_fetch(signal_array + config.signal_base + lane)
          : UINT64_MAX;
      const uint32_t metadata_mask =
          __ballot_sync(0xffffffffU, observed >= metadata_value) & expected_mask;
      const uint32_t data_mask =
          __ballot_sync(0xffffffffU, observed >= data_value) & expected_mask;
      if (lane == 0) {
        shared_metadata_mask |= metadata_mask;
        shared_data_mask |= data_mask;
        __threadfence_block();
      }
      if (metadata_mask == expected_mask && !metadata_closed_recorded) {
        pool_slice_record_profile(
            g_events, poolSliceProfileMetadataClosed, lane);
        metadata_closed_recorded = true;
      }

      if (!data_published &&
          *reinterpret_cast<volatile uint32_t*>(&shared_pack_done) ==
              config.pack_warps) {
        if (lane == 0)
          __threadfence_system();
        __syncwarp();
        pool_slice_publish_phase_parallel(
            signal_array,
            config.signal_base + config.my_pe,
            data_value,
            config.my_pe,
            config.num_pes,
            lane);
        data_published = true;
        pool_slice_record_profile(
            g_events, poolSliceProfileFirstDataPublished, lane);
        pool_slice_record_profile(
            g_events, poolSliceProfileDataPublished, lane);
      }
      if (!data_published ||
          *reinterpret_cast<volatile uint32_t*>(&shared_dispatch_issued) <
              config.num_pes)
        __nanosleep(barrierPollSleepCycles);
    }

  }

  // NVSHMEM 3.4's public device quiet is thread-scoped. The pinned runtime's
  // block-scope implementation distributes RC/DCI completion polling across
  // the block, which avoids making coordinator lane 0 walk every QP after the
  // workers have issued their batches. All roles are converged here.
  __syncthreads();
  pool_slice_quiet(config.num_pes, thread_id);

  if (warp == 0) {
    pool_slice_record_profile(
        g_events, poolSliceProfilePayloadDone, lane);

    if (lane == 0) {
      uint64_t received_rows = 0;
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader)
        received_rows += shared_reader_tails[reader];
      *group_ready = sequence;
      control[1] = config.num_pes;
      control[2] = received_rows;
      control[4] = sequence;
      control[5] = shared_payload_sources;
      control[6] = shared_dispatch_batches;
      control[7] = config.pack_warps;
      if (shared_status != POOL_SLICE_STATUS_OK)
        pool_slice_set_status(
            config, static_cast<PoolSliceStatus>(shared_status));
      __threadfence_system();
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader)
        atomicSub(bars + dispatch_barrier_base + reader, 1);
    }
    __syncwarp();
    pool_slice_record_profile(
        g_events, poolSliceProfileGatherReady, lane);
  }

  __syncthreads();

  // Reader blocks are ordinary VDCores programs. Only the coordinator polls
  // their completion barriers; all communication workers sleep at the block
  // barrier until expert output is ready.
  if (warp == 0) {
    bool ready = lane >= config.local_readers;
    while (__ballot_sync(0xffffffffU, ready) != 0xffffffffU) {
      if (!ready) {
        ready = *reinterpret_cast<volatile int*>(
            bars + compute_barrier_base + lane) == 0;
      }
      if (__ballot_sync(0xffffffffU, ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
    pool_slice_record_profile(
        g_events, poolSliceProfileComputeReady, lane);
  }
  __syncthreads();

  if (thread_id == 0) {
    shared_return_next = 0;
    shared_return_issued = 0;
  }
  __syncthreads();

  const auto* expert_output = reinterpret_cast<const uint8_t*>(
      config.expert_output_address);
  auto* return_inbox = reinterpret_cast<uint8_t*>(
      config.return_inbox_address);

  if (warp != 0) {
    while (true) {
      uint32_t source_pe = UINT32_MAX;
      if (lane == 0)
        source_pe = atomicAdd(&shared_return_next, 1U);
      source_pe = __shfl_sync(0xffffffffU, source_pe, 0);
      if (source_pe >= config.num_pes)
        break;
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader) {
        const PoolSliceReceiveBatch route = receive_routes[
            static_cast<uint64_t>(reader) * config.num_pes + source_pe];
        if (route.sequence != sequence ||
            route.source_pe != source_pe ||
            route.local_reader != reader ||
            route.flags != POOL_SLICE_BATCH_FLAGS_NONE ||
            route.source_begin > config.route_capacity ||
            route.row_count > config.route_capacity - route.source_begin ||
            route.base_row > config.expert_capacity_rows ||
            route.row_count > config.expert_capacity_rows - route.base_row) {
          if (lane == 0) {
            atomicCAS(
                &shared_status,
                static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
                static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
          }
          continue;
        }
        pool_slice_put_nbi_warp(
            return_inbox +
                static_cast<uint64_t>(route.source_begin) * config.row_bytes,
            expert_output +
                static_cast<uint64_t>(reader) * config.expert_stride +
                route.base_row * config.expert_row_stride,
            static_cast<uint64_t>(route.row_count) * config.row_bytes,
            source_pe,
            config.my_pe,
            lane);
      }
      __syncwarp();
      if (lane == 0)
        atomicAdd(&shared_return_issued, 1U);
    }
  }

  if (warp == 0)
    while (*reinterpret_cast<volatile uint32_t*>(&shared_return_issued) <
           config.num_pes)
      __nanosleep(barrierPollSleepCycles);

  __syncthreads();
  pool_slice_quiet(config.num_pes, thread_id);
  if (warp == 0)
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnPayloadDone, lane);
  __syncthreads();

  if (warp == 0) {
    pool_slice_publish_phase_parallel(
        signal_array,
        config.signal_base + config.my_pe,
        return_value,
        config.my_pe,
        config.num_pes,
        lane);

    bool returned = lane >= config.num_pes;
    while (__ballot_sync(0xffffffffU, returned) != 0xffffffffU) {
      if (!returned) {
        returned = nvshmem_signal_fetch(
            signal_array + config.signal_base + lane) >= return_value;
      }
      if (__ballot_sync(0xffffffffU, returned) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnSignalsClosed, lane);
  }
  __syncthreads();

  const auto* origins = reinterpret_cast<const uint32_t*>(
      config.send_origin_rows_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);
  for (uint32_t route = warp;
       route < config.active_rows;
       route += total_warps) {
    uint32_t origin = 0;
    if (lane == 0)
      origin = origins[route];
    origin = __shfl_sync(0xffffffffU, origin, 0);
    if (origin >= config.return_capacity_rows) {
      if (lane == 0) {
        atomicCAS(
            &shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
      }
      continue;
    }
    pool_slice_copy_warp(
        returned + static_cast<uint64_t>(origin) * config.return_stride,
        return_inbox + static_cast<uint64_t>(route) * config.row_bytes,
        config.row_bytes,
        lane);
  }
  __syncthreads();

  if (thread_id == 0) {
    control[3] = config.num_pes;
    if (shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(shared_status));
    __threadfence_system();
    if (g_events != nullptr) {
      g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
               poolSliceProfileScatterDone] = cuda::ptx::get_sreg_globaltimer();
      g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
               poolSliceProfileDone] = cuda::ptx::get_sreg_globaltimer();
    }
  }
  __syncthreads();
}
