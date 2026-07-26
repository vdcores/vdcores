#pragma once

#include <cstdint>

static constexpr uint32_t poolSliceMaxPes = 32;
static constexpr uint32_t poolSliceMaxLocalReaders = 8;
static constexpr uint32_t poolSliceMinimumRowBytes = 1024;
static constexpr uint32_t poolSliceAlignmentBytes = 16;
static constexpr uint32_t poolSliceControlWords = 8;
static constexpr uint32_t poolSliceSignalPhases = 3;
static constexpr uint32_t poolSliceProfileStart = 5;
static constexpr uint32_t poolSliceProfileGatherReady = 6;
static constexpr uint32_t poolSliceProfileDone = 7;
static constexpr uint32_t poolSliceProfileDataPublished = 8;
static constexpr uint32_t poolSliceProfileFirstPayload = 9;
static constexpr uint32_t poolSliceProfileMetadataClosed = 10;
static constexpr uint32_t poolSliceProfilePayloadDone = 11;
static constexpr uint32_t poolSliceProfileFirstDataPublished = 12;
static constexpr uint32_t poolSliceProfileComputeReady = 13;
static constexpr uint32_t poolSliceProfileReturnPayloadDone = 14;
static constexpr uint32_t poolSliceProfileReturnSignalsClosed = 15;
static constexpr uint32_t poolSliceProfileScatterDone = 16;

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

enum PoolSliceSignalPhase : uint32_t {
  POOL_SLICE_SIGNAL_METADATA = 1,
  POOL_SLICE_SIGNAL_DATA = 2,
  POOL_SLICE_SIGNAL_RETURN = 3,
};

// One descriptor is published by every source pool core to every target pool
// slice, including targets with zero rows. Reader counts reconstruct the
// target's complete grouped offset span without a metadata GET. Keeping the
// common metadata in one cache line is deliberate: the pool protocol supports
// at most eight local readers and uses larger logical fanout by adding slices.
struct alignas(16) PoolSlicePublishBatch {
  uint64_t sequence;
  uint32_t source_pe;
  uint32_t target_pe;
  uint32_t active_rows;
  uint32_t flags;
  uint32_t route_begin;
  uint32_t route_end;
  uint32_t reader_counts[poolSliceMaxLocalReaders];
};
static_assert(
    sizeof(PoolSlicePublishBatch) == 64,
    "PoolSlicePublishBatch ABI changed");

// The target records the local contiguous range assigned to each
// (reader, source) batch. The same record drives the pool-owned return path.
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

// Python packs this stable 208-byte ABI in python/dae/pool_slice.py. Every
// remotely addressed pointer names a same-order NVSHMEM symmetric allocation.
// The config object itself is local CUDA memory and is read only by the
// communication-specialized VDCores block.
struct alignas(16) PoolSliceConfig {
  uint64_t source_address;
  uint64_t token_pool_address;
  uint64_t delivery_pool_address;
  uint64_t expert_input_address;
  uint64_t expert_output_address;
  uint64_t return_inbox_address;
  uint64_t returned_address;
  uint64_t send_offsets_address;
  uint64_t send_rows_address;
  uint64_t send_origin_rows_address;
  uint64_t send_batches_address;
  uint64_t receive_batches_address;
  uint64_t receive_routes_address;
  uint64_t sequence_address;
  uint64_t group_ready_address;
  uint64_t control_address;

  uint32_t row_bytes;
  uint32_t source_stride;
  uint32_t pool_stride;
  uint32_t delivery_stride;
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
  uint32_t signal_base;
  uint32_t signal_count;
  uint32_t return_capacity_rows;
  uint32_t pack_warps;
  uint32_t write_chunks;
  uint32_t write_chunk_rows;
};
static_assert(sizeof(PoolSliceConfig) == 208, "PoolSliceConfig ABI changed");
