#pragma once

#include "context.cuh"
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
  if (config.control_address == 0)
    return;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  atomicCAS(
      reinterpret_cast<unsigned long long*>(control),
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
    const PoolSliceConfig& config, uint32_t total_warps) {
  const uint32_t worker_warps = total_warps - 1;
  uint64_t required_expert_bytes = 0;
  return config.combine_rows_address != 0 &&
      config.token_pool_address != 0 &&
      config.delivery_pool_address != 0 &&
      config.expert_input_address != 0 &&
      config.expert_output_address != 0 &&
      config.return_inbox_address != 0 &&
      config.returned_address != 0 &&
      config.send_offsets_address != 0 &&
      config.send_rows_address != 0 &&
      config.send_origin_rows_address != 0 &&
      config.send_token_rows_address != 0 &&
      config.send_token_counts_address != 0 &&
      config.send_batches_address != 0 &&
      config.receive_batches_address != 0 &&
      config.receive_rows_address != 0 &&
      config.receive_routes_address != 0 &&
      config.sequence_address != 0 &&
      config.group_ready_address != 0 &&
      config.control_address != 0 &&
      config.row_bytes >= poolSliceMinimumRowBytes &&
      config.row_bytes % poolSliceAlignmentBytes == 0 &&
      config.reducer_count != 0 &&
      config.reducer_count <= poolSliceMaxExternalReducers &&
      config.pool_stride >= config.row_bytes &&
      config.pool_stride % poolSliceAlignmentBytes == 0 &&
      config.pool_stride == config.row_bytes &&
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
      total_warps >= 3 &&
      config.pool_count != 0 &&
      config.pool_count <= poolSliceMaxPoolBlocks &&
      config.pool_rank < config.pool_count &&
      (config.flags &
       ~(POOL_SLICE_FLAGS_DEDICATED_COORDINATOR |
         POOL_SLICE_FLAGS_PUT_PHASE_WORDS |
         POOL_SLICE_FLAGS_PIPELINED_RETURN |
         POOL_SLICE_FLAGS_READER_PIPELINE |
         POOL_SLICE_FLAGS_WEIGHTED_RETURN |
         POOL_SLICE_FLAGS_EXTERNAL_WEIGHTED_REDUCER |
         POOL_SLICE_FLAGS_EXTERNAL_TOKEN_REDUCER)) == 0 &&
      ((config.flags & POOL_SLICE_FLAGS_WEIGHTED_RETURN) == 0 ||
       ((config.flags & POOL_SLICE_FLAGS_PIPELINED_RETURN) == 0 &&
        config.return_capacity_rows >= config.token_capacity)) &&
      ((config.flags & POOL_SLICE_FLAGS_EXTERNAL_WEIGHTED_REDUCER) == 0 ||
       (config.flags & POOL_SLICE_FLAGS_WEIGHTED_RETURN) != 0) &&
      ((config.flags & POOL_SLICE_FLAGS_EXTERNAL_TOKEN_REDUCER) == 0 ||
       (config.flags & POOL_SLICE_FLAGS_EXTERNAL_WEIGHTED_REDUCER) != 0) &&
      (((config.flags & POOL_SLICE_FLAGS_EXTERNAL_TOKEN_REDUCER) != 0) ||
       config.reducer_count == config.local_readers) &&
      ((config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0 ||
       config.pool_count > 1) &&
      config.dispatch_mode == POOL_SLICE_DISPATCH_POOL_GATHER &&
      config.pack_warps != 0 &&
      config.pack_warps <= worker_warps &&
      config.expert_capacity_rows >=
          static_cast<uint64_t>(config.num_pes) * config.token_capacity &&
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
    uint32_t lane,
    bool put_phase_words,
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
          if (put_phase_words)
            nvshmem_uint64_p(signal_array + signal_id, value, target_pe);
          else
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
      if (put_phase_words)
        nvshmem_uint64_p(signal_array + signal_id, value, target_pe);
      else
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

// Pool dispatch writes each activation once per destination pool slice. The
// producer metadata supplies a sorted unique-token list per target and maps
// every expert route to an index in that list. PoolInst packs one contiguous
// target shard, transfers it once, and leaves expert fanout to local gathered
// reads.
static __device__ __noinline__ void pool_slice_replicate_target_shard(
    uint32_t target_pe,
    uint32_t token_shard,
    uint32_t token_shards,
    const PoolSliceConfig& config,
    const uint8_t* token_pool,
    uint8_t* delivery_pool,
    const uint32_t* send_token_rows,
    const uint32_t* send_token_counts,
    int* bars,
    uint32_t write_barrier,
    uint32_t* shared_status,
    uint64_t* g_events,
    uint32_t* first_payload,
    uint32_t lane) {
  const uint32_t token_count = send_token_counts[target_pe];
  if (token_count > config.token_capacity) {
    if (lane == 0) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
    }
    return;
  }
  // The source-owned pool is already the authoritative storage for a local
  // dynamic read. Packing it into the self receive segment would add a full
  // HBM read/write pass before the gathered read; local gather resolves the
  // compact index directly through send_token_rows instead.
  if (target_pe == config.my_pe)
    return;
  const uint32_t row_begin = static_cast<uint32_t>(
      (static_cast<uint64_t>(token_count) * token_shard) /
      token_shards);
  const uint32_t row_end = static_cast<uint32_t>(
      (static_cast<uint64_t>(token_count) * (token_shard + 1)) /
      token_shards);
  if (row_begin == row_end)
    return;

  const uint64_t receive_rows =
      static_cast<uint64_t>(config.num_pes) * config.token_capacity;
  uint8_t* packed = delivery_pool +
      (receive_rows +
       static_cast<uint64_t>(target_pe) * config.token_capacity) *
          config.delivery_stride;
  const uint32_t* target_rows =
      send_token_rows +
      static_cast<uint64_t>(target_pe) * config.token_capacity;
  uint32_t waited_chunk = UINT32_MAX;
  for (uint32_t packed_row = row_begin;
       packed_row < row_end;
       ++packed_row) {
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
      return;
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
    pool_slice_copy_warp(
        packed + static_cast<uint64_t>(packed_row) * config.delivery_stride,
        token_pool + static_cast<uint64_t>(source_row) * config.pool_stride,
        config.row_bytes,
        lane);
  }

  __syncwarp();
  pool_slice_record_first_payload(g_events, first_payload, lane);
  pool_slice_put_nbi_warp(
      delivery_pool +
          (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
           row_begin) *
              config.delivery_stride,
      packed + static_cast<uint64_t>(row_begin) * config.delivery_stride,
      static_cast<uint64_t>(row_end - row_begin) * config.row_bytes,
      target_pe,
      config.my_pe,
      lane);
}

static __device__ __noinline__ void pool_slice_gather_reader_group(
    uint32_t local_reader,
    uint32_t source_pe,
    uint32_t route_shard,
    uint32_t route_shards,
    const PoolSliceConfig& config,
    const PoolSliceReceiveBatch* receive_routes,
    const uint64_t* receive_rows,
    uint64_t* combine_rows,
    const uint32_t* send_token_rows,
    const uint8_t* token_pool,
    const uint8_t* delivery_pool,
    int* bars,
    uint32_t write_barrier,
    uint8_t* expert_input,
    uint32_t* shared_status,
    uint32_t lane) {
  const PoolSliceReceiveBatch route = receive_routes[
      static_cast<uint64_t>(local_reader) * config.num_pes + source_pe];
  if (route.source_begin > config.route_capacity ||
      route.row_count > config.route_capacity - route.source_begin ||
      route.base_row > config.expert_capacity_rows ||
      route.row_count > config.expert_capacity_rows - route.base_row) {
    if (lane == 0) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
    }
    return;
  }

  const uint64_t* source_rows =
      receive_rows + static_cast<uint64_t>(source_pe) * config.route_capacity;
  uint32_t waited_chunk = UINT32_MAX;
  for (uint32_t relative = route_shard;
       relative < route.row_count;
       relative += route_shards) {
    uint32_t source_row = 0;
    uint64_t route_word = 0;
    if (lane == 0) {
      route_word = source_rows[route.source_begin + relative];
      source_row = static_cast<uint32_t>(route_word);
    }
    source_row = __shfl_sync(0xffffffffU, source_row, 0);
    if (source_row >= config.token_capacity) {
      if (lane == 0) {
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
      }
      return;
    }
    if (lane == 0 &&
        (config.flags & POOL_SLICE_FLAGS_WEIGHTED_RETURN) != 0) {
      combine_rows[
          (static_cast<uint64_t>(local_reader) * config.num_pes + source_pe) *
              config.token_capacity +
          source_row] =
          (route_word & 0xffffffff00000000ULL) |
          static_cast<uint32_t>(route.base_row + relative);
    }
    const uint8_t* source_address = delivery_pool +
        (static_cast<uint64_t>(source_pe) * config.token_capacity +
         source_row) *
            config.delivery_stride;
    if (source_pe == config.my_pe) {
      uint32_t token_row = 0;
      if (lane == 0) {
        token_row = send_token_rows[
            static_cast<uint64_t>(config.my_pe) * config.token_capacity +
            source_row];
      }
      token_row = __shfl_sync(0xffffffffU, token_row, 0);
      if (token_row >= config.token_capacity) {
        if (lane == 0) {
          atomicCAS(
              shared_status,
              static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
              static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
        }
        return;
      }
      const uint32_t chunk = token_row / config.write_chunk_rows;
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
      source_address = token_pool +
          static_cast<uint64_t>(token_row) * config.pool_stride;
    }
    pool_slice_copy_warp(
        expert_input +
            static_cast<uint64_t>(local_reader) * config.expert_stride +
            (route.base_row + relative) * config.expert_row_stride,
        source_address,
        config.row_bytes,
        lane);
  }
}

static __device__ __forceinline__ void pool_slice_complete_reader_shard(
    const PoolSliceConfig& config,
    uint64_t* control,
    int* bars,
    uint32_t dispatch_barrier_base,
    uint32_t local_reader,
    uint32_t expected_shards,
    uint64_t sequence,
    uint32_t* shared_status,
    uint32_t lane) {
  __syncwarp();
  if (lane == 0) {
    const uint64_t previous = dae_atomic_fetch_add_acq_rel_gpu(
        control + poolSliceControlReaderGatherCount + local_reader, 1);
    if (previous + 1 == expected_shards) {
      dae_atomic_store_release_gpu(
          control + poolSliceControlReaderReady + local_reader, sequence);
      pool_signal_release(bars + dispatch_barrier_base + local_reader);
    } else if (previous >= expected_shards) {
      atomicCAS(
          shared_status,
          static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
          static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
    }
  }
  __syncwarp();
}

static __device__ __forceinline__ uint64_t pool_slice_return_batch_fetch(
    uint64_t* address, bool local) {
  return local ? dae_atomic_load_acquire_gpu(address)
               : nvshmem_signal_fetch(address);
}

static __device__ __forceinline__ void pool_slice_wait_return_batch_warp(
    uint64_t* address,
    uint64_t sequence,
    bool local,
    uint32_t lane) {
  uint32_t ready = 0;
  while (ready == 0) {
    if (lane == 0)
      ready = pool_slice_return_batch_fetch(address, local) >= sequence;
    ready = __shfl_sync(0xffffffffU, ready, 0);
    if (ready == 0)
      __nanosleep(barrierPollSleepCycles);
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
            static_cast<uint64_t>(reader) * config.expert_stride +
            static_cast<uint64_t>(reader_row) * config.expert_row_stride);
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

  auto* destination = reinterpret_cast<__nv_bfloat162*>(
      returned + static_cast<uint64_t>(source_row) * config.return_stride);
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
    for (uint32_t target_pe = 0;
         target_pe < config.num_pes;
         ++target_pe) {
      const uint32_t target_row =
          __shfl_sync(0xffffffffU, packed_row, target_pe);
      if (target_row == UINT32_MAX)
        continue;
      const auto* partial = reinterpret_cast<const __nv_bfloat162*>(
          return_inbox +
          (static_cast<uint64_t>(target_pe) * config.token_capacity +
           target_row) *
              config.row_bytes);
#pragma unroll
      for (uint32_t item = 0; item < 4; ++item) {
        const uint32_t item_element = element + item * vector_stride;
        if (item_element < elements) {
          const float2 value = __bfloat1622float2(partial[item_element]);
          sums[item][target_pe & 3U].x += value.x;
          sums[item][target_pe & 3U].y += value.y;
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

static __device__ __forceinline__ uint64_t pool_slice_weighted_total_rows(
    const PoolSliceConfig& config,
    const PoolSlicePublishBatch* receive_batches) {
  uint64_t total = 0;
  for (uint32_t source_pe = 0; source_pe < config.num_pes; ++source_pe) {
    const uint32_t rows = receive_batches[source_pe].active_rows;
    if (rows <= config.token_capacity)
      total += rows;
  }
  return total;
}

static __device__ __forceinline__ void pool_slice_weighted_source_range(
    uint32_t rows,
    uint64_t source_base,
    uint64_t flat_begin,
    uint64_t flat_end,
    uint32_t* row_begin,
    uint32_t* row_end) {
  const uint64_t source_end = source_base + rows;
  const uint64_t intersection_begin =
      flat_begin > source_base ? flat_begin : source_base;
  const uint64_t intersection_end =
      flat_end < source_end ? flat_end : source_end;
  if (intersection_begin >= intersection_end) {
    *row_begin = 0;
    *row_end = 0;
    return;
  }
  *row_begin = static_cast<uint32_t>(intersection_begin - source_base);
  *row_end = static_cast<uint32_t>(intersection_end - source_base);
}

// Production EP return: reduce expert rows inside the destination pool slice,
// transfer one contiguous partial-token batch to every source, then reduce the
// at-most-one partial per destination slice at the source. Token ranges are
// statically disjoint across PoolInst CTAs, so staging and network PUTs need no
// cross-CTA fence. CTA quiet followed by a GPU-scope generation gives the
// coordinator the precise completion fact needed before phase publication.
static __device__ __noinline__ void pool_slice_return_weighted(
    const PoolSliceConfig& config,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t compute_barrier_base,
    uint32_t total_warps,
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
  const bool external_reducer =
      (config.flags & POOL_SLICE_FLAGS_EXTERNAL_WEIGHTED_REDUCER) != 0;
  const bool payload_executor =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0 ||
      config.pool_rank != 0;
  const uint32_t payload_pool_count =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0
      ? config.pool_count
      : config.pool_count - 1;
  const uint32_t payload_pool_rank =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0
      ? config.pool_rank
      : (config.pool_rank == 0 ? 0 : config.pool_rank - 1);

  if (warp == 0) {
    bool ready = lane >= config.reducer_count;
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
  const uint64_t total_return_rows =
      pool_slice_weighted_total_rows(config, receive_batches);
  const uint64_t flat_row_begin =
      total_return_rows * payload_pool_rank / payload_pool_count;
  const uint64_t flat_row_end =
      total_return_rows * (payload_pool_rank + 1) / payload_pool_count;
  uint8_t* partial_staging = external_reducer
      ? return_inbox + receive_capacity * config.row_bytes
      : delivery_pool + receive_capacity * config.delivery_stride;
  if (!external_reducer && warp != 0 && payload_executor) {
    const uint32_t worker_warps = total_warps - 1;
    const uint32_t worker_slot = warp - 1;
    const uint32_t vector_shards = 4;
    uint64_t source_base = 0;
    for (uint32_t source_pe = 0;
         source_pe < config.num_pes;
         ++source_pe) {
      const PoolSlicePublishBatch batch = receive_batches[source_pe];
      const uint32_t batch_rows = batch.active_rows <= config.token_capacity
          ? batch.active_rows
          : 0;
      if (batch.active_rows > config.token_capacity ||
          batch.sequence != sequence ||
          batch.source_pe != source_pe ||
          batch.target_pe != config.my_pe ||
          batch.flags != POOL_SLICE_BATCH_FLAGS_NONE) {
        if (lane == 0)
          atomicCAS(
              shared_status,
              static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
              static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
        source_base += batch_rows;
        continue;
      }
      uint8_t* partial_output =
          partial_staging +
          static_cast<uint64_t>(source_pe) * config.token_capacity *
              config.row_bytes;
      uint32_t row_begin = 0;
      uint32_t row_end = 0;
      pool_slice_weighted_source_range(
          batch.active_rows,
          source_base,
          flat_row_begin,
          flat_row_end,
          &row_begin,
          &row_end);
      source_base += batch.active_rows;
      const uint32_t row_count = row_end - row_begin;
      const uint32_t reduce_tasks = row_count * vector_shards;
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
  }
  __syncthreads();

  if (thread_id == 0 && g_events != nullptr) {
    g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
             poolSliceProfileReturnReduceDone] =
        cuda::ptx::get_sreg_globaltimer();
  }

  // One warp posts this CTA's contiguous token shard for every source. The
  // local batch follows the same layout but uses a GPU-local copy. Sharding
  // across PoolInst CTAs exposes enough independent warps to hide expert-row
  // load latency while preserving batched network writes.
  if (warp == 1 && payload_executor) {
    bool first_return_put = true;
    uint64_t source_base = 0;
    for (uint32_t source_pe = 0;
         source_pe < config.num_pes;
         ++source_pe) {
      const uint32_t rows = receive_batches[source_pe].active_rows;
      if (rows == 0 || rows > config.token_capacity)
        continue;
      uint32_t row_begin = 0;
      uint32_t row_end = 0;
      pool_slice_weighted_source_range(
          rows,
          source_base,
          flat_row_begin,
          flat_row_end,
          &row_begin,
          &row_end);
      source_base += rows;
      if (row_begin == row_end)
        continue;
      const uint8_t* source =
          partial_staging +
          (static_cast<uint64_t>(source_pe) * config.token_capacity +
           row_begin) *
              config.row_bytes;
      uint8_t* destination =
          return_inbox +
          (static_cast<uint64_t>(config.my_pe) * config.token_capacity +
           row_begin) *
              config.row_bytes;
      const uint64_t bytes =
          static_cast<uint64_t>(row_end - row_begin) * config.row_bytes;
      if (first_return_put) {
        if (lane == 0 && g_events != nullptr) {
          g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
                   poolSliceProfileFirstReturnPut] =
              cuda::ptx::get_sreg_globaltimer();
        }
        first_return_put = false;
        __syncwarp();
      }
      pool_slice_put_nbi_warp(
          destination,
          source,
          bytes,
          source_pe,
          config.my_pe,
          lane);
    }
  }
  __syncthreads();
  pool_slice_quiet_block();
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
        (config.flags & POOL_SLICE_FLAGS_PUT_PHASE_WORDS) != 0,
        nullptr,
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
    if (lane == 0)
      dae_atomic_store_release_gpu(
          control + poolSliceControlScatterStart, sequence);
    __syncwarp();
  }

  pool_slice_wait_value_warp(
      control + poolSliceControlScatterStart, sequence, lane);
  const uint32_t global_warp = config.pool_rank * total_warps + warp;
  const uint32_t global_warps = config.pool_count * total_warps;
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

// Dense pool-gather return pipeline. Each (target reader, source PE) batch
// carries a named completion signal in the same NVSHMEM operation as its
// payload. Source scatter warps consume that contiguous route range as soon as
// the signal arrives, while other experts and outgoing batches remain active.
// The return-ready words start at group_ready[1]; group_ready[0] retains the
// public dispatch-generation telemetry contract.
static __device__ __noinline__ void pool_slice_return_scatter_pipelined(
    const PoolSliceConfig& config,
    int* bars,
    uint64_t* g_events,
    uint32_t compute_barrier_base,
    uint32_t total_warps,
    uint32_t thread_id,
    uint64_t sequence,
    uint32_t* shared_status) {
  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  auto* return_ready =
      reinterpret_cast<uint64_t*>(config.group_ready_address) + 1;
  const auto* receive_routes =
      reinterpret_cast<const PoolSliceReceiveBatch*>(
          config.receive_routes_address);
  const auto* send_offsets =
      reinterpret_cast<const uint32_t*>(config.send_offsets_address);
  const auto* origins =
      reinterpret_cast<const uint32_t*>(config.send_origin_rows_address);
  const auto* expert_output =
      reinterpret_cast<const uint8_t*>(config.expert_output_address);
  auto* return_inbox =
      reinterpret_cast<uint8_t*>(config.return_inbox_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);

  const bool payload_executor =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0 ||
      config.pool_rank != 0;
  const uint32_t payload_pool_count =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0
      ? config.pool_count
      : config.pool_count - 1;
  const uint32_t payload_pool_rank =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0
      ? config.pool_rank
      : config.pool_rank - 1;

  if (warp != 0 && payload_executor) {
    const uint32_t worker_warps = total_warps - 1;
    const uint32_t worker_slot = warp - 1;
    const uint32_t num_batches = config.local_readers * config.num_pes;
    for (uint32_t task =
             payload_pool_rank + worker_slot * payload_pool_count;
         task < num_batches;
         task += worker_warps * payload_pool_count) {
      const uint32_t reader = task / config.num_pes;
      const uint32_t source_pe = task % config.num_pes;
      if ((config.flags & POOL_SLICE_FLAGS_READER_PIPELINE) != 0) {
        pool_slice_wait_value_warp(
            control + poolSliceControlReaderReady + reader,
            sequence,
            lane);
      }
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
      const bool valid = route.sequence == sequence &&
          route.source_pe == source_pe &&
          route.local_reader == reader &&
          route.flags == POOL_SLICE_BATCH_FLAGS_NONE &&
          route.source_begin <= config.route_capacity &&
          route.row_count <= config.route_capacity - route.source_begin &&
          route.base_row <= config.expert_capacity_rows &&
          route.row_count <= config.expert_capacity_rows - route.base_row;
      if (!valid && lane == 0) {
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
      }

      const uint32_t global_reader =
          config.my_pe * config.local_readers + reader;
      uint64_t* ready_address = return_ready + global_reader;
      const uint64_t bytes = valid
          ? static_cast<uint64_t>(route.row_count) * config.row_bytes
          : 0;
      if (source_pe == config.my_pe) {
        // The source and expert pool slice share this HBM. The compute
        // barrier already orders expert_output, so publish the named batch
        // dependency without copying through return_inbox.
        __syncwarp();
        if (lane == 0)
          dae_atomic_store_release_gpu(ready_address, sequence);
        __syncwarp();
      } else if (bytes != 0) {
        nvshmemx_putmem_signal_nbi_warp(
            return_inbox +
                static_cast<uint64_t>(route.source_begin) * config.row_bytes,
            expert_output +
                static_cast<uint64_t>(reader) * config.expert_stride +
                route.base_row * config.expert_row_stride,
            static_cast<size_t>(bytes),
            ready_address,
            sequence,
            NVSHMEM_SIGNAL_SET,
            source_pe);
      } else {
        __syncwarp();
        if (lane == 0)
          nvshmemx_signal_op(
              ready_address,
              sequence,
              NVSHMEM_SIGNAL_SET,
              source_pe);
        __syncwarp();
      }
    }
  }

  if (config.pool_rank == 0 && warp == 0) {
    bool compute_ready = lane >= config.local_readers;
    while (__ballot_sync(0xffffffffU, compute_ready) != 0xffffffffU) {
      if (!compute_ready)
        compute_ready =
            pool_signal_ready(bars + compute_barrier_base + lane);
      if (__ballot_sync(0xffffffffU, compute_ready) != 0xffffffffU)
        __nanosleep(barrierPollSleepCycles);
    }
    pool_slice_record_profile(
        g_events, poolSliceProfileComputeReady, lane);
  }
  // Every NBI fused return posts its payload and signal WQEs immediately.
  // Completion remains asynchronous and is named by the receiver-visible
  // batch signal below. Do not place a CTA barrier here: warps whose expert
  // batch is ready must be able to post/consume it while another worker in
  // the same PoolInst CTA is still waiting on a different expert.

  if (config.pool_rank == 0 && warp == 0) {
    const uint32_t num_readers = config.num_pes * config.local_readers;
    for (uint32_t global_reader = lane;
         global_reader < num_readers;
         global_reader += 32) {
      const uint32_t target_pe = global_reader / config.local_readers;
      while (pool_slice_return_batch_fetch(
                 return_ready + global_reader,
                 target_pe == config.my_pe) < sequence)
        __nanosleep(barrierPollSleepCycles);
    }
    __syncwarp();
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnPayloadDone, lane);
    pool_slice_record_profile(
        g_events, poolSliceProfileReturnSignalsClosed, lane);
  }

  const uint32_t global_warp = config.pool_rank * total_warps + warp;
  const uint32_t global_warps = config.pool_count * total_warps;
  const uint32_t num_readers = config.num_pes * config.local_readers;
  const uint32_t reader_groups =
      global_warps < num_readers ? global_warps : num_readers;
  const uint32_t reader_group = global_warp % reader_groups;
  const uint32_t group_warp = global_warp / reader_groups;
  const uint32_t group_warps =
      (global_warps + reader_groups - 1 - reader_group) / reader_groups;
  for (uint32_t global_reader = reader_group;
       global_reader < num_readers;
       global_reader += reader_groups) {
    const uint32_t target_pe = global_reader / config.local_readers;
    pool_slice_wait_return_batch_warp(
        return_ready + global_reader,
        sequence,
        target_pe == config.my_pe,
        lane);
    const uint32_t route_begin = send_offsets[global_reader];
    const uint32_t route_end = send_offsets[global_reader + 1];
    const uint32_t reader = global_reader % config.local_readers;
    const PoolSliceReceiveBatch local_route = target_pe == config.my_pe
        ? receive_routes[
              static_cast<uint64_t>(reader) * config.num_pes + config.my_pe]
        : PoolSliceReceiveBatch{};
    if (target_pe == config.my_pe &&
        (local_route.sequence != sequence ||
         local_route.source_pe != config.my_pe ||
         local_route.local_reader != reader ||
         local_route.source_begin != route_begin ||
         local_route.row_count != route_end - route_begin)) {
      if (lane == 0) {
        atomicCAS(
            shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
      }
      continue;
    }
    for (uint32_t route = route_begin + group_warp;
         route < route_end;
         route += group_warps) {
      uint32_t origin = 0;
      if (lane == 0)
        origin = origins[route];
      origin = __shfl_sync(0xffffffffU, origin, 0);
      if (origin >= config.return_capacity_rows) {
        if (lane == 0) {
          atomicCAS(
              shared_status,
              static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
              static_cast<uint32_t>(POOL_SLICE_STATUS_ROUTE_RANGE));
        }
        continue;
      }
      const uint8_t* source = target_pe == config.my_pe
          ? expert_output +
                static_cast<uint64_t>(reader) * config.expert_stride +
                (local_route.base_row + route - route_begin) *
                    config.expert_row_stride
          : return_inbox + static_cast<uint64_t>(route) * config.row_bytes;
      pool_slice_copy_warp(
          returned + static_cast<uint64_t>(origin) * config.return_stride,
          source,
          config.row_bytes,
          lane);
    }
  }
  __syncthreads();

  // Keep expert-output reuse correct even when incoming scatter finishes
  // before this CTA's outgoing NBI return batches. The quiet is post-scatter,
  // so it does not gate peer consumption.
  pool_slice_quiet_block();
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

// Low-latency slot execution. Every PoolInst CTA is a peer executor:
// block zero owns metadata/signals, while all CTAs shard expert groups, quiet
// their own IBGDA work, and publish single-writer generation slots. No CTA
// waits on a global fence and no counter must be reset between sequences.
static __device__ __noinline__ void pool_slice_exchange_compact(
    const PoolSliceConfig& config,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t compute_barrier_base,
    uint32_t total_warps,
    uint32_t thread_id,
    uint64_t sequence,
    uint64_t metadata_value,
    uint64_t data_value,
    uint64_t return_value) {
  __shared__ uint32_t shared_status;
  __shared__ uint32_t shared_first_payload;
  __shared__ uint32_t shared_payload_sources;
  __shared__ uint32_t shared_dispatch_batches;
  __shared__ unsigned long long
      shared_reader_tails[poolSliceMaxLocalReaders];

  const uint32_t lane = thread_id & 31U;
  const uint32_t warp = thread_id >> 5;
  auto* control = reinterpret_cast<uint64_t*>(config.control_address);
  auto* group_ready = reinterpret_cast<uint64_t*>(config.group_ready_address);
  auto* send_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.send_batches_address);
  auto* receive_batches = reinterpret_cast<PoolSlicePublishBatch*>(
      config.receive_batches_address);
  auto* receive_rows = reinterpret_cast<uint64_t*>(
      config.receive_rows_address);
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
  auto* expert_input = reinterpret_cast<uint8_t*>(
      config.expert_input_address);
  auto* delivery_pool = reinterpret_cast<uint8_t*>(
      config.delivery_pool_address);

  if (thread_id == 0) {
    shared_status = POOL_SLICE_STATUS_OK;
    shared_first_payload = 0;
    shared_payload_sources = 0;
    shared_dispatch_batches = 0;
  }
  for (uint32_t reader = thread_id;
       reader < poolSliceMaxLocalReaders;
       reader += blockDim.x)
    shared_reader_tails[reader] = 0;

  if (config.pool_rank == 0) {
    for (uint32_t index = thread_id;
         index < 8;
         index += blockDim.x)
      control[index] = 0;
    for (uint32_t reader = thread_id;
         reader < config.local_readers;
         reader += blockDim.x)
      control[poolSliceControlReaderGatherCount + reader] = 0;
    for (uint32_t reader = thread_id;
         reader < config.local_readers;
         reader += blockDim.x)
      control[poolSliceControlReaderRowCount + reader] = 0;
    if ((config.flags & POOL_SLICE_FLAGS_WEIGHTED_RETURN) != 0) {
      const uint64_t combine_count =
          static_cast<uint64_t>(config.local_readers) * config.num_pes *
          config.token_capacity;
      for (uint64_t index = thread_id;
           index < combine_count;
           index += blockDim.x)
        combine_rows[index] = UINT64_MAX;
    }
    if (thread_id == 0)
      *group_ready = 0;
    if (g_events != nullptr) {
      uint64_t* block_events =
          g_events + static_cast<uint64_t>(blockIdx.x) * numProfileEvents;
      for (uint32_t event = poolSliceProfileStart + thread_id;
           event <= poolSliceProfileScatterDone;
           event += blockDim.x)
        block_events[event] = 0;
    }
  }
  __syncthreads();
  if (config.pool_rank == 0 && thread_id == 0) {
    if (g_events != nullptr) {
      g_events[static_cast<uint64_t>(blockIdx.x) * numProfileEvents +
               poolSliceProfileStart] = cuda::ptx::get_sreg_globaltimer();
    }
    dae_atomic_store_release_gpu(
        control + poolSliceControlStart, sequence);
  }
  __syncthreads();
  if (config.pool_rank != 0)
    pool_slice_wait_value_warp(
        control + poolSliceControlStart, sequence, lane);

  if (config.pool_rank == 0) {
    for (uint32_t target_pe = thread_id;
         target_pe < config.num_pes;
         target_pe += blockDim.x) {
      PoolSlicePublishBatch& batch = send_batches[target_pe];
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
    }
    __syncthreads();

    if (warp == 0) {
      if (config.num_pes <= 2) {
        for (uint32_t index = 0; index < config.num_pes; ++index) {
          const uint32_t target_pe = pool_slice_remote_first_pe(
              index, config.my_pe, config.num_pes);
          PoolSlicePublishBatch* destination =
              receive_batches + config.my_pe;
          const PoolSlicePublishBatch* source = send_batches + target_pe;
          if (target_pe == config.my_pe) {
            if (source->route_begin != source->route_end) {
              pool_slice_copy_warp(
                  receive_rows +
                      static_cast<uint64_t>(config.my_pe) *
                          config.route_capacity + source->route_begin,
                  send_rows + source->route_begin,
                  static_cast<uint64_t>(
                      source->route_end - source->route_begin) *
                      sizeof(uint64_t),
                  lane);
              __syncwarp();
            }
            pool_slice_copy_warp(
                destination, source, sizeof(PoolSlicePublishBatch), lane);
            __syncwarp();
            if (lane == 0)
              pool_slice_signal_release_local(
                  signal_array + config.signal_base + config.my_pe,
                  source->active_rows == 0 &&
                          source->flags == POOL_SLICE_BATCH_FLAGS_NONE
                      ? data_value
                      : metadata_value);
            __syncwarp();
          } else {
            if (source->route_begin != source->route_end) {
              nvshmemx_putmem_nbi_warp(
                  receive_rows +
                      static_cast<uint64_t>(config.my_pe) *
                          config.route_capacity + source->route_begin,
                  send_rows + source->route_begin,
                  static_cast<size_t>(
                      source->route_end - source->route_begin) *
                      sizeof(uint64_t),
                  target_pe);
            }
            nvshmemx_putmem_signal_warp(
                destination,
                source,
                sizeof(PoolSlicePublishBatch),
                signal_array + config.signal_base + config.my_pe,
                source->active_rows == 0 &&
                        source->flags == POOL_SLICE_BATCH_FLAGS_NONE
                    ? data_value
                    : metadata_value,
                NVSHMEM_SIGNAL_SET,
                target_pe);
          }
        }
      } else if (lane < config.num_pes) {
        const uint32_t target_pe = pool_slice_remote_first_pe(
            lane, config.my_pe, config.num_pes);
        PoolSlicePublishBatch* destination =
            receive_batches + config.my_pe;
        const PoolSlicePublishBatch* source = send_batches + target_pe;
        if (target_pe == config.my_pe) {
          if (source->route_begin != source->route_end) {
            uint64_t* destination_rows =
                receive_rows +
                static_cast<uint64_t>(config.my_pe) * config.route_capacity;
            for (uint32_t row = source->route_begin;
                 row < source->route_end;
                 ++row)
              destination_rows[row] = send_rows[row];
          }
          *destination = *source;
          pool_slice_signal_release_local(
              signal_array + config.signal_base + config.my_pe,
              source->active_rows == 0 &&
                      source->flags == POOL_SLICE_BATCH_FLAGS_NONE
                  ? data_value
                  : metadata_value);
        } else {
          if (source->route_begin != source->route_end) {
            nvshmem_putmem_nbi(
                receive_rows +
                    static_cast<uint64_t>(config.my_pe) *
                        config.route_capacity + source->route_begin,
                send_rows + source->route_begin,
                static_cast<size_t>(
                    source->route_end - source->route_begin) *
                    sizeof(uint64_t),
                target_pe);
          }
          nvshmem_putmem_signal(
              destination,
              source,
              sizeof(PoolSlicePublishBatch),
              signal_array + config.signal_base + config.my_pe,
              source->active_rows == 0 &&
                      source->flags == POOL_SLICE_BATCH_FLAGS_NONE
                  ? data_value
                  : metadata_value,
              NVSHMEM_SIGNAL_SET,
              target_pe);
        }
      }
      __syncwarp();
      // The local source route is known before any remote PE closes its data
      // phase. Materialize it once so pool workers can gather the self slice
      // as soon as all local replication CTAs publish their generations.
      // The coordinator deliberately does not rewrite this record below:
      // local workers may already be consuming it while remote metadata is
      // still arriving.
      if (lane < config.local_readers &&
          receive_batches[config.my_pe].flags ==
              POOL_SLICE_BATCH_FLAGS_NONE) {
        const uint32_t global_reader =
            config.my_pe * config.local_readers + lane;
        const uint32_t source_begin = send_offsets[global_reader];
        const uint32_t row_count =
            send_offsets[global_reader + 1] - source_begin;
        PoolSliceReceiveBatch& route = receive_routes[
            static_cast<uint64_t>(lane) * config.num_pes + config.my_pe];
        route.sequence = sequence;
        // The local source is first in every target-local expert block. This
        // lets its gather overlap network progress while still producing one
        // contiguous dynamic-reader input once remote prefixes are known.
        route.base_row = 0;
        route.source_begin = source_begin;
        route.row_count = row_count;
        route.source_pe = config.my_pe;
        route.local_reader = lane;
        route.flags = POOL_SLICE_BATCH_FLAGS_NONE;
        route.reserved_u32[0] = 0;
        route.reserved_u32[1] = 0;
        route.reserved_u32[2] = 0;
      }
      __syncwarp();
    }
  }

  // Spread reader groups round-robin across CTAs before assigning the next
  // warp. With multiple blocks, rank zero is a control-plane CTA so metadata
  // and merged phase signals never queue behind payload work on the same QP.
  const bool payload_executor =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0 ||
      config.pool_rank != 0;
  const uint32_t payload_pool_count =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0
      ? config.pool_count
      : config.pool_count - 1;
  const uint32_t payload_pool_rank =
      (config.flags & POOL_SLICE_FLAGS_DEDICATED_COORDINATOR) == 0
      ? config.pool_rank
      : (config.pool_rank == 0 ? 0 : config.pool_rank - 1);
  if (warp != 0 && payload_executor) {
    const uint32_t worker_warps = total_warps - 1;
    const uint32_t worker_slot = warp - 1;
    const uint32_t token_shards = config.pack_warps;
    const uint32_t num_tasks = config.num_pes * token_shards;
    for (uint32_t task =
             payload_pool_rank + worker_slot * payload_pool_count;
         task < num_tasks;
         task += worker_warps * payload_pool_count) {
      pool_slice_replicate_target_shard(
          task / token_shards,
          task % token_shards,
          token_shards,
          config,
          token_pool,
          delivery_pool,
          send_token_rows,
          send_token_counts,
          bars,
          write_barrier,
          &shared_status,
          config.pool_rank == 0 ? g_events : nullptr,
          &shared_first_payload,
          lane);
    }
  }
  __syncthreads();
  pool_slice_quiet_block();
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
    if (lane == 0)
      dae_atomic_store_release_gpu(
          control + poolSliceControlLocalGatherStart, sequence);
    __syncwarp();
    pool_slice_publish_phase_parallel(
        signal_array,
        config.signal_base + config.my_pe,
        data_value,
        config.my_pe,
        config.num_pes,
        lane,
        (config.flags & POOL_SLICE_FLAGS_PUT_PHASE_WORDS) != 0,
        send_batches,
        sequence);
    pool_slice_record_profile(
        g_events, poolSliceProfileFirstDataPublished, lane);
    pool_slice_record_profile(
        g_events, poolSliceProfileDataPublished, lane);

    const uint32_t expected_mask = pool_slice_pe_mask(config.num_pes);
    bool metadata_recorded = false;
    uint32_t data_mask = 0;
    while (data_mask != expected_mask) {
      const uint64_t observed = lane < config.num_pes
          ? pool_slice_signal_fetch(
                signal_array + config.signal_base + lane,
                lane == config.my_pe)
          : UINT64_MAX;
      const uint32_t metadata_mask =
          __ballot_sync(0xffffffffU, observed >= metadata_value) & expected_mask;
      data_mask =
          __ballot_sync(0xffffffffU, observed >= data_value) & expected_mask;
      if (!metadata_recorded && metadata_mask == expected_mask) {
        pool_slice_record_profile(
            g_events, poolSliceProfileMetadataClosed, lane);
        metadata_recorded = true;
      }
      if (data_mask != expected_mask)
        __nanosleep(barrierPollSleepCycles);
    }

    if (lane < config.num_pes) {
      const uint32_t source_pe = lane;
      const PoolSlicePublishBatch batch = receive_batches[source_pe];
      uint32_t batch_valid = batch.sequence == sequence &&
          batch.source_pe == source_pe &&
          batch.target_pe == config.my_pe &&
          batch.active_rows <= config.token_capacity &&
          batch.route_begin <= batch.route_end &&
          batch.route_end <= config.route_capacity &&
          batch.flags == POOL_SLICE_BATCH_FLAGS_NONE;
      uint32_t source_cursor = batch.route_begin;
      uint32_t source_rows = 0;
      uint32_t source_batches = 0;
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader) {
        const uint32_t count = batch.reader_counts[reader];
        if (count > config.token_capacity ||
            source_cursor > batch.route_end ||
            count > batch.route_end - source_cursor)
          batch_valid = 0;
        // The self route was published before the remote phase began so its
        // gather can overlap this coordinator's network wait. Avoid racing an
        // identical multiword rewrite against those local readers.
        if (source_pe != config.my_pe) {
          uint64_t base_row =
              receive_batches[config.my_pe].reader_counts[reader];
          for (uint32_t source_index = 0;
               source_index + 1 < config.num_pes;
               ++source_index) {
            const uint32_t preceding_source = pool_slice_remote_first_pe(
                source_index, config.my_pe, config.num_pes);
            if (preceding_source == source_pe)
              break;
            base_row +=
                receive_batches[preceding_source].reader_counts[reader];
          }
          if (base_row > config.expert_capacity_rows ||
              count > config.expert_capacity_rows - base_row)
            batch_valid = 0;
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
        }
        source_cursor += count;
        source_rows += count;
        source_batches += count != 0;
        atomicAdd(
            shared_reader_tails + reader,
            static_cast<unsigned long long>(count));
      }
      if (source_cursor != batch.route_end)
        batch_valid = 0;
      if (!batch_valid) {
        atomicCAS(
            &shared_status,
            static_cast<uint32_t>(POOL_SLICE_STATUS_OK),
            static_cast<uint32_t>(POOL_SLICE_STATUS_BATCH));
      } else {
        atomicAdd(&shared_payload_sources, source_rows != 0);
        atomicAdd(&shared_dispatch_batches, source_batches);
        if (source_rows == 0) {
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
    }
    __syncwarp();
    if (lane == 0) {
      uint64_t received_rows = 0;
      for (uint32_t reader = 0;
           reader < config.local_readers;
           ++reader) {
        control[poolSliceControlReaderRowCount + reader] =
            shared_reader_tails[reader];
        received_rows += shared_reader_tails[reader];
      }
      dae_atomic_store_release_gpu(group_ready, sequence);
      control[1] = config.num_pes;
      control[2] = received_rows;
      control[4] = sequence;
      control[5] = shared_payload_sources;
      control[6] = shared_dispatch_batches;
      control[7] = payload_pool_count;
      if (shared_status != POOL_SLICE_STATUS_OK)
        pool_slice_set_status(
            config, static_cast<PoolSliceStatus>(shared_status));
      dae_atomic_store_release_gpu(
          control + poolSliceControlGatherStart, sequence);
    }
    __syncwarp();
  }

  {
    // Local-source rows are already resident after the per-CTA dispatch
    // generations close. Gather them while rank-zero's coordinator warp is
    // waiting for remote data/metadata phase signals. This is a GPU-local
    // dependency: no system fence and no NVSHMEM signal is required.
    pool_slice_wait_value_warp(
        control + poolSliceControlLocalGatherStart, sequence, lane);
    const bool local_overlap_executor =
        payload_executor && config.pool_rank != 0;
    if (warp != 0 && local_overlap_executor) {
      const uint32_t worker_warps = total_warps - 1;
      const uint32_t worker_slot = warp - 1;
      const uint32_t route_shards = config.pack_warps;
      // Start only one interleaved shard per reader before remote closure.
      // A full local gather delayed IBGDA signal visibility more than the
      // HBM work it hid at the dense 128x7168 shape. This bounded tranche is
      // enough to overlap useful work without flooding the pool/NIC path.
      const uint32_t overlap_shards = config.pool_count > 1 ? 1 : 0;
      const uint32_t num_tasks = config.local_readers * overlap_shards;
      for (uint32_t task =
               (config.pool_rank - 1) +
                   worker_slot * (config.pool_count - 1);
           task < num_tasks;
           task += worker_warps * (config.pool_count - 1)) {
        const uint32_t local_reader = task / overlap_shards;
        pool_slice_gather_reader_group(
            local_reader,
            config.my_pe,
            task % overlap_shards,
            route_shards,
            config,
            receive_routes,
            receive_rows,
            combine_rows,
            send_token_rows,
            token_pool,
            delivery_pool,
            bars,
            write_barrier,
            expert_input,
            &shared_status,
            lane);
        if ((config.flags & POOL_SLICE_FLAGS_READER_PIPELINE) != 0) {
          pool_slice_complete_reader_shard(
              config,
              control,
              bars,
              dispatch_barrier_base,
              local_reader,
              config.num_pes * route_shards,
              sequence,
              &shared_status,
              lane);
        }
      }
    }
    __syncthreads();

    pool_slice_wait_value_warp(
        control + poolSliceControlGatherStart, sequence, lane);
    if (warp != 0 && payload_executor) {
      const uint32_t worker_warps = total_warps - 1;
      const uint32_t worker_slot = warp - 1;
      const uint32_t route_shards = config.pack_warps;
      const uint32_t remote_sources = config.num_pes - 1;
      const uint32_t num_batches = config.local_readers * remote_sources;
      const uint32_t remote_tasks = num_batches * route_shards;
      const uint32_t overlap_shards = config.pool_count > 1 ? 1 : 0;
      const uint32_t local_tail_shards = route_shards - overlap_shards;
      const uint32_t local_tail_tasks =
          config.local_readers * local_tail_shards;
      const uint32_t num_tasks = remote_tasks + local_tail_tasks;
      for (uint32_t task =
               payload_pool_rank + worker_slot * payload_pool_count;
           task < num_tasks;
           task += worker_warps * payload_pool_count) {
        uint32_t local_reader = 0;
        uint32_t source_pe = config.my_pe;
        uint32_t route_shard = 0;
        if (task < remote_tasks) {
          const uint32_t batch = task / route_shards;
          const uint32_t source_index = batch % remote_sources;
          source_pe =
              source_index < config.my_pe ? source_index : source_index + 1;
          local_reader = batch / remote_sources;
          route_shard = task % route_shards;
        } else {
          const uint32_t local_task = task - remote_tasks;
          local_reader = local_task / local_tail_shards;
          route_shard =
              overlap_shards + local_task % local_tail_shards;
        }
        pool_slice_gather_reader_group(
            local_reader,
            source_pe,
            route_shard,
            route_shards,
            config,
            receive_routes,
            receive_rows,
            combine_rows,
            send_token_rows,
            token_pool,
            delivery_pool,
            bars,
            write_barrier,
            expert_input,
            &shared_status,
            lane);
        if ((config.flags & POOL_SLICE_FLAGS_READER_PIPELINE) != 0) {
          pool_slice_complete_reader_shard(
              config,
              control,
              bars,
              dispatch_barrier_base,
              local_reader,
              config.num_pes * route_shards,
              sequence,
              &shared_status,
              lane);
        }
      }
    }
    __syncthreads();
    if (thread_id == 0) {
      if (shared_status != POOL_SLICE_STATUS_OK)
        pool_slice_set_status(
            config, static_cast<PoolSliceStatus>(shared_status));
      dae_atomic_store_release_gpu(
          control + poolSliceControlGatherGeneration + config.pool_rank,
          sequence);
    }
    __syncthreads();

    if (config.pool_rank == 0 && warp == 0) {
      pool_slice_wait_generation_warp(
          control + poolSliceControlGatherGeneration,
          config.pool_count,
          sequence,
          lane);
      pool_slice_record_profile(
          g_events, poolSliceProfilePayloadDone, lane);
      if (lane == 0) {
        if ((config.flags & POOL_SLICE_FLAGS_READER_PIPELINE) == 0) {
          for (uint32_t reader = 0;
               reader < config.local_readers;
               ++reader)
            pool_signal_release(bars + dispatch_barrier_base + reader);
        }
        dae_atomic_store_release_gpu(
            control + poolSliceControlDispatchReady, sequence);
      }
      __syncwarp();
      pool_slice_record_profile(g_events, poolSliceProfileGatherReady, lane);
    }
  }

  if ((config.flags & POOL_SLICE_FLAGS_READER_PIPELINE) == 0) {
    pool_slice_wait_value_warp(
        control + poolSliceControlDispatchReady, sequence, lane);
  }

  if ((config.flags & POOL_SLICE_FLAGS_WEIGHTED_RETURN) != 0) {
    pool_slice_return_weighted(
        config,
        bars,
        signal_array,
        g_events,
        compute_barrier_base,
        total_warps,
        thread_id,
        sequence,
        return_value,
        &shared_status);
    return;
  }

  if ((config.flags & POOL_SLICE_FLAGS_PIPELINED_RETURN) != 0) {
    pool_slice_return_scatter_pipelined(
        config,
        bars,
        g_events,
        compute_barrier_base,
        total_warps,
        thread_id,
        sequence,
        &shared_status);
    return;
  }

  const auto* expert_output = reinterpret_cast<const uint8_t*>(
      config.expert_output_address);
  auto* return_inbox = reinterpret_cast<uint8_t*>(
      config.return_inbox_address);
  if (warp != 0 && payload_executor) {
    const uint32_t worker_warps = total_warps - 1;
    const uint32_t worker_slot = warp - 1;
    const uint32_t num_batches = config.local_readers * config.num_pes;
    for (uint32_t task =
             payload_pool_rank + worker_slot * payload_pool_count;
         task < num_batches;
         task += worker_warps * payload_pool_count) {
      const uint32_t reader = task / config.num_pes;
      const uint32_t source_pe = task % config.num_pes;
      if ((config.flags & POOL_SLICE_FLAGS_READER_PIPELINE) != 0) {
        pool_slice_wait_value_warp(
            control + poolSliceControlReaderReady + reader,
            sequence,
            lane);
      }
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
    if (shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(shared_status));
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
        (config.flags & POOL_SLICE_FLAGS_PUT_PHASE_WORDS) != 0,
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
    if (lane == 0)
      dae_atomic_store_release_gpu(
          control + poolSliceControlScatterStart, sequence);
    __syncwarp();
  }

  pool_slice_wait_value_warp(
      control + poolSliceControlScatterStart, sequence, lane);
  const auto* origins = reinterpret_cast<const uint32_t*>(
      config.send_origin_rows_address);
  auto* returned = reinterpret_cast<uint8_t*>(config.returned_address);
  const auto* return_rows = reinterpret_cast<const uint8_t*>(
      config.return_inbox_address);
  const uint32_t global_warp = config.pool_rank * total_warps + warp;
  const uint32_t global_warps = config.pool_count * total_warps;
  for (uint32_t route = global_warp;
       route < config.active_rows;
       route += global_warps) {
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
        return_rows + static_cast<uint64_t>(route) * config.row_bytes,
        config.row_bytes,
        lane);
  }
  __syncthreads();
  if (thread_id == 0) {
    if (shared_status != POOL_SLICE_STATUS_OK)
      pool_slice_set_status(
          config, static_cast<PoolSliceStatus>(shared_status));
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
  __syncwarp();
}

// A PoolInst macro operation invoked by every PoolSliceExchangeExecuteWarp
// thread. PoolInst is separate from CommInst, and this compact protocol has one
// compile-time implementation: unique-token pool gather plus dependent return.
static __device__ __noinline__ void pool_slice_exchange(
    const PoolSliceConfig* config_pointer,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events,
    uint32_t write_barrier,
    uint32_t dispatch_barrier_base,
    uint32_t compute_barrier_base,
    uint32_t total_warps,
    uint32_t thread_id) {
  __shared__ PoolSliceConfig shared_config;

  if (config_pointer == nullptr || bars == nullptr || signal_array == nullptr)
    return;

  if (thread_id == 0)
    shared_config = *config_pointer;
  __syncthreads();

  const PoolSliceConfig& config = shared_config;
  if (!pool_slice_valid_config(config, total_warps) ||
      total_warps != static_cast<uint32_t>(blockDim.x / 32)) {
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

  pool_slice_exchange_compact(
      config,
      bars,
      signal_array,
      g_events,
      write_barrier,
      dispatch_barrier_base,
      compute_barrier_base,
      total_warps,
      thread_id,
      sequence,
      pool_slice_signal_value(sequence, POOL_SLICE_SIGNAL_METADATA),
      pool_slice_signal_value(sequence, POOL_SLICE_SIGNAL_DATA),
      pool_slice_signal_value(sequence, POOL_SLICE_SIGNAL_RETURN));
}
