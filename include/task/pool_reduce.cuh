#pragma once

#include "context.cuh"
#include "pool_slice_abi.cuh"
#include "virtualcore.cuh"

#include <cuda_bf16.h>

#include <cstdint>

static __device__ __forceinline__ void pool_reduce_record_event(
    uint64_t* events, int sm_id, uint32_t event) {
  if (__compute_tid() == 0 && events != nullptr)
    events[static_cast<uint64_t>(sm_id) * numProfileEvents + event] =
        cuda::ptx::get_sreg_globaltimer();
}

static __device__ __forceinline__ void pool_reduce_set_status(
    const PoolSliceConfig* config, PoolSliceStatus status) {
  if (__compute_tid() != 0 || config == nullptr ||
      config->control_address == 0)
    return;
  auto* control = reinterpret_cast<unsigned long long*>(
      config->control_address);
  atomicCAS(
      control,
      static_cast<unsigned long long>(POOL_SLICE_STATUS_OK),
      static_cast<unsigned long long>(status));
}

static __device__ __forceinline__ float pool_reduce_route_weight(
    uint64_t route_word) {
  union {
    uint16_t bits;
    __nv_bfloat16 value;
  } weight;
  weight.bits = static_cast<uint16_t>(route_word >> 32);
  return __bfloat162float(weight.value);
}

// Zero the private half of return_inbox while PoolInst dispatches activations.
// The PoolRawAddress completion that follows this operator publishes one
// device-scope release; expert reducers acquire it before issuing atomics.
template <typename M2CQueue, typename C2MQueue>
static __device__ __forceinline__ void task_pool_zero_weighted_return(
    int sm_id,
    const MInst* st_insts,
    M2CQueue& m2c,
    C2MQueue& c2m,
    uint64_t* g_events) {
  constexpr int compute_threads = 128;
  __activate_compute_group(compute_threads);

  const int config_slot = m2c.template pop<0>();
  const auto* config = static_cast<const PoolSliceConfig*>(
      slot_2_glob_ptr(st_insts, config_slot));
  const int completion_slot = m2c.template pop<0>();
  pool_reduce_record_event(
      g_events, sm_id, poolSliceProfileExternalZeroStart);

  const bool valid = config != nullptr &&
      (config->flags & POOL_SLICE_FLAGS_WEIGHTED_RETURN) != 0 &&
      (config->flags & POOL_SLICE_FLAGS_EXTERNAL_WEIGHTED_REDUCER) != 0 &&
      config->return_inbox_address != 0 && config->row_bytes != 0 &&
      config->num_pes != 0 && config->token_capacity != 0;
  if (!valid) {
    pool_reduce_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
  } else {
    const uint64_t rows =
        static_cast<uint64_t>(config->num_pes) * config->token_capacity;
    const uint64_t bytes = rows * config->row_bytes;
    auto* staging = reinterpret_cast<uint4*>(
        config->return_inbox_address + bytes);
    for (uint64_t vector = __compute_tid();
         vector < bytes / sizeof(uint4);
         vector += compute_threads)
      staging[vector] = make_uint4(0, 0, 0, 0);
  }
  __sync_compute_group(compute_threads);
  pool_reduce_record_event(
      g_events, sm_id, poolSliceProfileExternalZeroDone);
  c2m.template push<31, true, false>(
      __compute_tid(), special_slot_completion(completion_slot));
}

// One ordinary VDCores compute+memory block owns one local expert. It starts
// after that expert's gathered input (and optional expert compute) becomes
// visible, then atomically contributes weighted BF16x2 values to token-major
// pool staging. Different expert blocks may update the same token concurrently;
// native atomicAdd names exactly that dependency without a system-wide fence.
template <typename M2CQueue, typename C2MQueue>
static __device__ __forceinline__ void task_pool_expert_atomic_reduce_bf16(
    int sm_id,
    uint32_t local_reader,
    const MInst* st_insts,
    M2CQueue& m2c,
    C2MQueue& c2m,
    uint64_t* g_events) {
  constexpr int compute_threads = 128;
  __activate_compute_group(compute_threads);

  const int config_slot = m2c.template pop<0>();
  const auto* config = static_cast<const PoolSliceConfig*>(
      slot_2_glob_ptr(st_insts, config_slot));
  const int completion_slot = m2c.template pop<0>();
  pool_reduce_record_event(
      g_events, sm_id, poolSliceProfileExternalReduceStart);

  const bool valid = config != nullptr &&
      local_reader < config->local_readers &&
      (config->flags & POOL_SLICE_FLAGS_WEIGHTED_RETURN) != 0 &&
      (config->flags & POOL_SLICE_FLAGS_EXTERNAL_WEIGHTED_REDUCER) != 0 &&
      config->receive_routes_address != 0 &&
      config->receive_rows_address != 0 &&
      config->expert_output_address != 0 &&
      config->return_inbox_address != 0 &&
      config->sequence_address != 0 &&
      config->row_bytes != 0 &&
      config->row_bytes % sizeof(__nv_bfloat162) == 0;
  if (!valid) {
    pool_reduce_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
  } else {
    const auto* receive_routes =
        reinterpret_cast<const PoolSliceReceiveBatch*>(
            config->receive_routes_address);
    const auto* receive_rows = reinterpret_cast<const uint64_t*>(
        config->receive_rows_address);
    const auto* expert_output = reinterpret_cast<const uint8_t*>(
        config->expert_output_address);
    const uint64_t sequence = *reinterpret_cast<const uint64_t*>(
        config->sequence_address);
    const uint64_t receive_capacity =
        static_cast<uint64_t>(config->num_pes) * config->token_capacity;
    auto* partial_output = reinterpret_cast<__nv_bfloat162*>(
        config->return_inbox_address +
        receive_capacity * config->row_bytes);
    const uint32_t vector_elements =
        config->row_bytes / sizeof(__nv_bfloat162);

    for (uint32_t source_pe = 0;
         source_pe < config->num_pes;
         ++source_pe) {
      const PoolSliceReceiveBatch route = receive_routes[
          static_cast<uint64_t>(local_reader) * config->num_pes + source_pe];
      const bool route_valid =
          route.sequence == sequence &&
          route.source_begin <= config->route_capacity &&
          route.row_count <= config->route_capacity - route.source_begin &&
          route.base_row <= config->expert_capacity_rows &&
          route.row_count <= config->expert_capacity_rows - route.base_row &&
          route.source_pe == source_pe &&
          route.local_reader == local_reader &&
          route.flags == POOL_SLICE_BATCH_FLAGS_NONE;
      if (!route_valid) {
        pool_reduce_set_status(config, POOL_SLICE_STATUS_BATCH);
        continue;
      }

      const auto* source_routes =
          receive_rows +
          static_cast<uint64_t>(source_pe) * config->route_capacity;
      for (uint32_t relative = 0;
           relative < route.row_count;
           ++relative) {
        const uint64_t route_word =
            source_routes[route.source_begin + relative];
        const uint32_t packed_row = static_cast<uint32_t>(route_word);
        if (packed_row >= config->token_capacity) {
          pool_reduce_set_status(config, POOL_SLICE_STATUS_ROUTE_RANGE);
          continue;
        }
        const float weight = pool_reduce_route_weight(route_word);
        const auto* source = reinterpret_cast<const __nv_bfloat162*>(
            expert_output +
            static_cast<uint64_t>(local_reader) * config->expert_stride +
            (route.base_row + relative) * config->expert_row_stride);
        auto* destination = partial_output +
            (static_cast<uint64_t>(source_pe) * config->token_capacity +
             packed_row) *
                vector_elements;
        for (uint32_t element = __compute_tid();
             element < vector_elements;
             element += compute_threads) {
          const float2 value = __bfloat1622float2(source[element]);
          atomicAdd(
              destination + element,
              __floats2bfloat162_rn(
                  value.x * weight, value.y * weight));
        }
      }
    }
  }
  __sync_compute_group(compute_threads);
  pool_reduce_record_event(
      g_events, sm_id, poolSliceProfileExternalReduceDone);
  c2m.template push<31, true, false>(
      __compute_tid(), special_slot_completion(completion_slot));
}

// A lower-overhead external alternative. Reducer blocks own disjoint compact
// token rows, wait until every local expert is visible, and use ordinary
// vector loads plus FP32 registers instead of global atomics. It is the same
// weighted reduction algebra as PoolInst, executed by configurable ordinary
// compute+memory VDCores SMs.
template <typename M2CQueue, typename C2MQueue>
static __device__ __forceinline__ void task_pool_token_reduce_bf16(
    int sm_id,
    uint32_t reducer_rank,
    uint32_t reducer_count,
    const MInst* st_insts,
    M2CQueue& m2c,
    C2MQueue& c2m,
    uint64_t* g_events) {
  constexpr int compute_threads = 128;
  __activate_compute_group(compute_threads);

  const int config_slot = m2c.template pop<0>();
  const auto* config = static_cast<const PoolSliceConfig*>(
      slot_2_glob_ptr(st_insts, config_slot));
  const int completion_slot = m2c.template pop<0>();
  pool_reduce_record_event(
      g_events, sm_id, poolSliceProfileExternalReduceStart);

  const bool valid = config != nullptr && reducer_count != 0 &&
      reducer_rank < reducer_count &&
      (config->flags & POOL_SLICE_FLAGS_EXTERNAL_TOKEN_REDUCER) != 0 &&
      config->combine_rows_address != 0 &&
      config->receive_batches_address != 0 &&
      config->expert_output_address != 0 &&
      config->return_inbox_address != 0 &&
      config->row_bytes % sizeof(__nv_bfloat162) == 0;
  if (!valid) {
    pool_reduce_set_status(config, POOL_SLICE_STATUS_BAD_CONFIG);
  } else {
    const auto* batches = reinterpret_cast<const PoolSlicePublishBatch*>(
        config->receive_batches_address);
    const auto* combine_rows = reinterpret_cast<const uint64_t*>(
        config->combine_rows_address);
    const auto* expert_output = reinterpret_cast<const uint8_t*>(
        config->expert_output_address);
    const uint64_t receive_capacity =
        static_cast<uint64_t>(config->num_pes) * config->token_capacity;
    auto* partial_output = reinterpret_cast<__nv_bfloat162*>(
        config->return_inbox_address +
        receive_capacity * config->row_bytes);
    const uint32_t vector_elements =
        config->row_bytes / sizeof(__nv_bfloat162);
    const uint64_t token_tasks = receive_capacity;

    for (uint64_t task = reducer_rank;
         task < token_tasks;
         task += reducer_count) {
      const uint32_t source_pe =
          static_cast<uint32_t>(task / config->token_capacity);
      const uint32_t packed_row =
          static_cast<uint32_t>(task % config->token_capacity);
      if (packed_row >= batches[source_pe].active_rows)
        continue;
      auto* destination = partial_output + task * vector_elements;
      for (uint32_t element = __compute_tid();
           element < vector_elements;
           element += compute_threads) {
        float2 sums[4] = {
            make_float2(0.0f, 0.0f),
            make_float2(0.0f, 0.0f),
            make_float2(0.0f, 0.0f),
            make_float2(0.0f, 0.0f),
        };
#pragma unroll
        for (uint32_t reader = 0;
             reader < poolSliceMaxLocalReaders;
             ++reader) {
          if (reader >= config->local_readers)
            continue;
          const uint64_t word = combine_rows[
              (static_cast<uint64_t>(reader) * config->num_pes + source_pe) *
                  config->token_capacity +
              packed_row];
          if (word == UINT64_MAX)
            continue;
          const uint32_t expert_row = static_cast<uint32_t>(word);
          if (expert_row >= config->expert_capacity_rows) {
            pool_reduce_set_status(config, POOL_SLICE_STATUS_BATCH);
            continue;
          }
          const auto* source = reinterpret_cast<const __nv_bfloat162*>(
              expert_output +
              static_cast<uint64_t>(reader) * config->expert_stride +
              static_cast<uint64_t>(expert_row) *
                  config->expert_row_stride);
          const float2 value = __bfloat1622float2(source[element]);
          const float weight = pool_reduce_route_weight(word);
          sums[reader & 3U].x += value.x * weight;
          sums[reader & 3U].y += value.y * weight;
        }
        destination[element] = __floats2bfloat162_rn(
            sums[0].x + sums[1].x + sums[2].x + sums[3].x,
            sums[0].y + sums[1].y + sums[2].y + sums[3].y);
      }
    }
  }
  __sync_compute_group(compute_threads);
  pool_reduce_record_event(
      g_events, sm_id, poolSliceProfileExternalReduceDone);
  c2m.template push<31, true, false>(
      __compute_tid(), special_slot_completion(completion_slot));
}
