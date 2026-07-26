#pragma once

#include <cstdint>

static constexpr uint32_t epPoolMaxPes = 32;
static constexpr uint32_t epPoolMaxExperts = 132;
static constexpr uint32_t epPoolMinimumRowBytes = 1024;
static constexpr uint32_t epPoolAlignmentBytes = 16;

enum EpPoolStatus : uint64_t {
  EP_POOL_STATUS_OK = 0,
  EP_POOL_STATUS_BAD_CONFIG = 1,
  EP_POOL_STATUS_ROUTE_RANGE = 2,
  EP_POOL_STATUS_CAPACITY = 3,
  EP_POOL_STATUS_SEQUENCE = 4,
  EP_POOL_STATUS_BATCH = 5,
  EP_POOL_STATUS_SIGNAL_RANGE = 6,
};

enum EpPoolBatchFlags : uint32_t {
  EP_POOL_BATCH_FLAGS_NONE = 0,
  EP_POOL_BATCH_FLAGS_ERROR = 1U << 0,
};

// One record is published for each (source PE, local expert) pair. The target
// stores records as [local_expert][source_pe].
struct alignas(16) EpPoolBatch {
  uint64_t sequence;
  uint64_t base_row;
  uint32_t source_base;
  uint32_t row_count;
  uint32_t source_pe;
  uint32_t local_expert;
  uint32_t flags;
  uint32_t reserved_u32[3];
};
static_assert(sizeof(EpPoolBatch) == 48, "EpPoolBatch ABI changed");

// Python packs the same stable 192-byte ABI in python/dae/ep_pool.py.
struct alignas(16) EpPoolConfig {
  uint64_t source_address;
  uint64_t packed_source_address;
  uint64_t expert_input_address;
  uint64_t expert_output_address;
  uint64_t return_inbox_address;
  uint64_t returned_address;
  uint64_t send_offsets_address;
  uint64_t send_rows_address;
  uint64_t send_origin_rows_address;
  uint64_t send_batches_address;
  uint64_t receive_batches_address;
  uint64_t expert_tails_address;
  uint64_t sequence_address;
  uint64_t control_address;

  uint32_t row_bytes;
  uint32_t source_stride;
  uint32_t expert_row_stride;
  uint32_t return_stride;
  uint32_t expert_stride;
  uint32_t active_rows;
  uint32_t route_capacity;
  uint32_t expert_capacity_rows;
  uint32_t num_experts;
  uint32_t experts_per_pe;
  uint32_t num_pes;
  uint32_t my_pe;
  uint32_t dispatch_signal_base;
  uint32_t return_signal_base;
  uint32_t reset_signal_base;
  uint32_t signal_count;
  uint32_t flags;
  uint32_t source_capacity_rows;
  uint32_t return_capacity_rows;
  uint32_t reserved_u32[1];
};
static_assert(sizeof(EpPoolConfig) == 192, "EpPoolConfig ABI changed");
