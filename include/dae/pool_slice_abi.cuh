#pragma once

#include <cstdint>

static constexpr uint32_t poolSliceMaxPes = 32;
static constexpr uint32_t poolSliceMinimumRowBytes = 1024;
static constexpr uint32_t poolSliceAlignmentBytes = 16;
static constexpr uint32_t poolSliceControlWords = 8;
static constexpr uint32_t poolSliceProfileDataPublished = 8;
static constexpr uint32_t poolSliceProfileFirstPayload = 9;
static constexpr uint32_t poolSliceProfileMetadataClosed = 10;
static constexpr uint32_t poolSliceProfilePayloadDone = 11;
static constexpr uint32_t poolSliceProfileFirstDataPublished = 12;

enum PoolSliceStatus : uint64_t {
  POOL_SLICE_STATUS_OK = 0,
  POOL_SLICE_STATUS_BAD_CONFIG = 1,
  POOL_SLICE_STATUS_SEQUENCE = 2,
  POOL_SLICE_STATUS_BATCH = 3,
  POOL_SLICE_STATUS_ROUTE_RANGE = 4,
  POOL_SLICE_STATUS_CAPACITY = 5,
  POOL_SLICE_STATUS_SIGNAL_RANGE = 6,
};

enum PoolSliceBatchFlags : uint32_t {
  POOL_SLICE_BATCH_FLAGS_NONE = 0,
  POOL_SLICE_BATCH_FLAGS_ERROR = 1U << 0,
};

enum PoolSliceFlags : uint32_t {
  POOL_SLICE_FLAGS_NONE = 0,
  POOL_SLICE_FLAGS_STREAMING_GATHER = 1U << 0,
};

// One transport descriptor is published by every source to every pool slice.
// route_begin/end cover that target slice's grouped route span; they fully
// describe the common one-reader case. A zero-route descriptor is still
// published, so consuming all source descriptors closes the shared sender
// group without a reader-specific end message.
struct alignas(16) PoolSlicePublishBatch {
  uint64_t sequence;
  uint32_t source_pe;
  uint32_t target_pe;
  uint32_t active_rows;
  uint32_t flags;
  uint32_t route_begin;
  uint32_t route_end;
};
static_assert(
    sizeof(PoolSlicePublishBatch) == 32,
    "PoolSlicePublishBatch ABI changed");

// The target pool records the local contiguous range assigned to each
// (reader, source) pair. The same record is later consumed by the pool-owned
// return path.
struct alignas(16) PoolSliceReceiveBatch {
  uint64_t sequence;
  uint64_t base_row;
  uint32_t source_begin;
  uint32_t row_count;
  uint32_t source_pe;
  uint32_t local_reader;
  uint32_t flags;
  uint32_t reserved_u32[3];
};
static_assert(
    sizeof(PoolSliceReceiveBatch) == 48,
    "PoolSliceReceiveBatch ABI changed");

// Python packs this stable 240-byte ABI in python/dae/pool_slice.py. Every
// remotely addressed pointer names a same-order NVSHMEM symmetric allocation.
// The config object itself is local CUDA memory and is only read by its pool
// communication warp.
struct alignas(16) PoolSliceConfig {
  uint64_t source_address;
  uint64_t token_pool_address;
  uint64_t expert_input_address;
  uint64_t expert_output_address;
  uint64_t return_inbox_address;
  uint64_t returned_address;
  uint64_t send_offsets_address;
  uint64_t send_rows_address;
  uint64_t send_origin_rows_address;
  uint64_t send_batches_address;
  uint64_t receive_batches_address;
  uint64_t offsets_inbox_address;
  uint64_t rows_inbox_address;
  uint64_t receive_routes_address;
  uint64_t reader_tails_address;
  uint64_t sequence_address;
  uint64_t group_ready_address;
  uint64_t control_address;

  uint32_t row_bytes;
  uint32_t source_stride;
  uint32_t pool_stride;
  uint32_t expert_row_stride;
  uint32_t return_stride;
  uint32_t expert_stride;
  uint32_t active_rows;
  uint32_t token_capacity;
  uint32_t route_capacity;
  uint32_t expert_capacity_rows;
  uint32_t local_readers;
  uint32_t num_pes;
  uint32_t my_pe;
  uint32_t queue_signal_base;
  uint32_t data_signal_base;
  uint32_t return_signal_base;
  uint32_t signal_count;
  uint32_t return_capacity_rows;
  uint32_t flags;
  uint32_t data_stages;
  uint32_t early_ready_rows;
  uint32_t reserved_u32[1];
};
static_assert(sizeof(PoolSliceConfig) == 240, "PoolSliceConfig ABI changed");
