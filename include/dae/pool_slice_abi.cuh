#pragma once

#include <cstdint>

static constexpr uint32_t poolSliceMaxPes = 32;
static constexpr uint32_t poolSliceMaxLocalReaders = 8;
static constexpr uint32_t poolSliceMaxPoolBlocks = 32;
static constexpr uint32_t poolSliceMaxExternalReducers = 32;
static constexpr uint32_t poolSliceMinimumRowBytes = 1024;
static constexpr uint32_t poolSliceAlignmentBytes = 16;
// The first eight words are user-visible telemetry. The remaining words are
// single-writer generation slots used to coordinate independently scheduled
// PoolInst CTAs without a device-wide fence or reset race.
static constexpr uint32_t poolSliceControlWords = 166;
static constexpr uint32_t poolSliceControlDispatchGeneration = 8;
static constexpr uint32_t poolSliceControlReturnGeneration = 40;
static constexpr uint32_t poolSliceControlScatterGeneration = 72;
static constexpr uint32_t poolSliceControlGatherGeneration = 104;
static constexpr uint32_t poolSliceControlStart = 136;
static constexpr uint32_t poolSliceControlPackedVisibility = 137;
static constexpr uint32_t poolSliceControlGatherStart = 138;
static constexpr uint32_t poolSliceControlDispatchReady = 139;
static constexpr uint32_t poolSliceControlScatterStart = 140;
static constexpr uint32_t poolSliceControlLocalGatherStart = 141;
static constexpr uint32_t poolSliceControlReaderGatherCount = 142;
static constexpr uint32_t poolSliceControlReaderReady = 150;
static constexpr uint32_t poolSliceControlReaderRowCount = 158;
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
static constexpr uint32_t poolSliceProfileExternalReduceStart = 17;
static constexpr uint32_t poolSliceProfileExternalReduceDone = 18;
static constexpr uint32_t poolSliceProfileExternalZeroStart = 19;
static constexpr uint32_t poolSliceProfileExternalZeroDone = 20;
// Per-PoolInst-CTA return telemetry. These events expose whether destination
// reduction, network posting, or CTA-local transport completion is the
// serialized portion of weighted return without introducing a helper kernel.
static constexpr uint32_t poolSliceProfileReturnReduceStart = 21;
static constexpr uint32_t poolSliceProfileReturnReduceDone = 22;
static constexpr uint32_t poolSliceProfileFirstReturnPut = 23;
static constexpr uint32_t poolSliceProfileReturnCtaDone = 24;

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

enum PoolSliceDispatchMode : uint32_t {
  // Pack each distinct (source activation, destination pool slice) once, then
  // resolve expert fanout with local HBM gathered reads. Route metadata maps
  // every expert route to its target's compact row index.
  POOL_SLICE_DISPATCH_POOL_GATHER = 0,
};

enum PoolSliceFlags : uint32_t {
  POOL_SLICE_FLAGS_NONE = 0,
  // Reserve PoolInst rank zero for metadata and phase signals. This is useful
  // for latency-bound sparse traffic; dense traffic can let every CTA carry
  // payload by leaving the flag clear.
  POOL_SLICE_FLAGS_DEDICATED_COORDINATOR = 1U << 0,
  // Publish data/return phase generations with an 8-byte RDMA write instead
  // of a remote atomic signal operation. Payload QPs are already quiet, so
  // the phase word needs visibility but not remote read-modify-write.
  POOL_SLICE_FLAGS_PUT_PHASE_WORDS = 1U << 1,
  // Compact pool-gather returns publish one fused payload signal per global
  // reader so source scatter can overlap outstanding expert/return work.
  POOL_SLICE_FLAGS_PIPELINED_RETURN = 1U << 2,
  // Release each reader when its own source/shard fan-in closes.
  POOL_SLICE_FLAGS_READER_PIPELINE = 1U << 3,
  // Apply the BF16 route weight at the destination pool slice, reduce all
  // local-reader contributions for a compact source token, and return one
  // partial row per (source token, pool slice). The source PoolInst then sums
  // the slice partials into token-major output. This is the production EP
  // return path; it trades one small weight field in route metadata for up to
  // local_readers-times less return traffic.
  POOL_SLICE_FLAGS_WEIGHTED_RETURN = 1U << 4,
  // Ordinary compute+memory VDCores reduce one expert at a time into a
  // token-major staging area. Each expert core can start as soon as its own
  // dynamic-read/compute dependency closes; PoolInst only owns the batched
  // network return and source-side final sum.
  POOL_SLICE_FLAGS_EXTERNAL_WEIGHTED_REDUCER = 1U << 5,
  // External reducers own disjoint compact-token rows and accumulate all
  // local experts without atomics. This gives up per-expert early start in
  // exchange for lower synchronization and HBM-atomic overhead.
  POOL_SLICE_FLAGS_EXTERNAL_TOKEN_REDUCER = 1U << 6,
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

// Python packs this stable 256-byte ABI in python/dae/pool_slice.py. Every
// remotely addressed pointer names a same-order NVSHMEM symmetric allocation.
// The config object itself is local CUDA memory and is read only by the
// PoolInst-specialized VDCores block.
struct alignas(16) PoolSliceConfig {
  // Target-local reverse route map used by weighted return. The writer source
  // tensor is a VDCores memory-operator operand and is not consumed by
  // PoolInst, so this first pointer names pool-owned metadata instead.
  uint64_t combine_rows_address;
  uint64_t token_pool_address;
  uint64_t delivery_pool_address;
  uint64_t expert_input_address;
  uint64_t expert_output_address;
  uint64_t return_inbox_address;
  uint64_t returned_address;
  uint64_t send_offsets_address;
  uint64_t send_rows_address;
  uint64_t send_origin_rows_address;
  uint64_t send_token_rows_address;
  uint64_t send_token_counts_address;
  uint64_t send_batches_address;
  uint64_t receive_batches_address;
  uint64_t receive_rows_address;
  uint64_t receive_routes_address;
  uint64_t sequence_address;
  uint64_t group_ready_address;
  uint64_t control_address;

  uint32_t row_bytes;
  // Weighted-return completion signals beginning at the PoolInst instruction's
  // compute_barrier_base. Inline and expert-atomic modes use local_readers;
  // token-sharded ordinary VDCores may use up to one full warp of reducers.
  uint32_t reducer_count;
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
  uint32_t pool_rank;
  uint32_t pool_count;
  uint32_t dispatch_mode;
  uint32_t flags;
};
static_assert(sizeof(PoolSliceConfig) == 256, "PoolSliceConfig ABI changed");
