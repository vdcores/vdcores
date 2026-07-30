#pragma once

#include <cstdint>

#include "pool_host_abi.h"

#ifndef DAE_POOL_SLICE_WARPS
#define DAE_POOL_SLICE_WARPS 8
#endif
#ifndef DAE_POOL_SLICE_WARP_QP_COMPLETION
#define DAE_POOL_SLICE_WARP_QP_COMPLETION 0
#endif
#ifndef DAE_POOL_SLICE_RAW_SGL
#define DAE_POOL_SLICE_RAW_SGL 0
#endif
#ifndef DAE_POOL_SLICE_RAW_SGL_WIDTH
#define DAE_POOL_SLICE_RAW_SGL_WIDTH 8
#endif

static constexpr uint32_t poolSliceMaxPes = 32;
static constexpr uint32_t poolSliceMaxLocalReaders = 8;
// A fixed all-PoolInst assembly may occupy every SM exposed by the VDCores
// launcher. Mixed assemblies are checked against the device SM count after
// adding their writer and reader cores.
static constexpr uint32_t poolSliceMaxPoolBlocks = 132;
// Dynamic grouping is a message-granularity choice, not a PoolInst-CTA limit.
// Thirty-two groups cover the intended <= 1K-token inference fast path while
// keeping the two ordered queues and their readiness state compact.
static constexpr uint32_t poolSliceMaxDataGroups = 32;
// A CTA-mapped-QP assembly needs one ordered generation per payload group.
// A separately compiled warp-mapped-QP assembly names every sender warp so
// the receiver can join the independent transport contexts exactly.
static constexpr uint32_t poolSlicePayloadWarps = DAE_POOL_SLICE_WARPS;
static_assert(
    poolSlicePayloadWarps >= 3 && poolSlicePayloadWarps <= 32,
    "PoolInst payload warp count is outside the supported range");
static_assert(
    DAE_POOL_SLICE_WARP_QP_COMPLETION == 0 ||
        DAE_POOL_SLICE_WARP_QP_COMPLETION == 1,
    "PoolInst QP completion scope must be CTA (0) or warp (1)");
static constexpr bool poolSliceWarpQpCompletion =
    DAE_POOL_SLICE_WARP_QP_COMPLETION != 0;
static constexpr uint32_t poolSliceCompletionSlots =
    poolSliceWarpQpCompletion ? poolSlicePayloadWarps : 1;
static_assert(
    DAE_POOL_SLICE_RAW_SGL == 0 || DAE_POOL_SLICE_RAW_SGL == 1,
    "PoolInst raw SGL selection must be disabled (0) or enabled (1)");
static constexpr bool poolSliceRawSgl = DAE_POOL_SLICE_RAW_SGL != 0;
static constexpr uint32_t poolSliceRawSglWidth =
    DAE_POOL_SLICE_RAW_SGL_WIDTH;
// A raw-SGL payload word carries both the invocation generation and the
// number of contiguous SGL segments already visible at the destination.  The
// route encoding limits one source envelope to 2^16 rows, so a 17-bit stride
// leaves every legal final value strictly below the next generation.
static constexpr uint64_t poolSliceRawSglProgressStride = 1ULL << 17;
static_assert(
    !poolSliceRawSgl ||
        (poolSliceRawSglWidth >= 1 && poolSliceRawSglWidth <= 30),
    "PoolInst raw RC SGL width must be in [1, 30]");
static_assert(poolSliceRawSglProgressStride > (1ULL << 16));
static_assert(
    !poolSliceRawSgl || !poolSliceWarpQpCompletion,
    "PoolInst raw RC SGL requires the CTA-mapped completion build");
// Dense LLM activations up through 8192 BF16 elements fit in one shared tile.
// PoolInst uses the tile only when every compiled warp owns one local expert;
// wider or sparse shapes retain the ordinary per-reader gather.
// Maximum source-owned reduction/return transport groups per destination.
// Runtime coalescing targets roughly 256 KiB while this bound keeps the
// dependency set compact.
static constexpr uint32_t poolSliceReturnGroupsPerSource = 4;
static constexpr uint32_t poolSliceMaxReturnReady =
    poolSliceMaxPes * poolSliceReturnGroupsPerSource;
// Every source exposes exactly two ordered queues. Keeping this compile-time
// shape small avoids a runtime queue-mode branch without recreating a
// (source, group) scan.
static constexpr uint32_t poolSliceMaxStreamQueues = 2;
static constexpr uint32_t poolSliceStreamQueueDepth =
    2 + (poolSliceMaxDataGroups + 1) / 2;
static_assert(poolSliceMaxPes * poolSliceMaxStreamQueues <= 64);
// The first five words are user-visible telemetry. The remaining words are
// single-writer generations and narrowly scoped counters used to coordinate
// independently scheduled PoolInst CTAs without a device-wide fence or reset
// race. Streaming COPY_ROWS messages name one statically scoped completion
// set, but consumers inspect only the head of each small ordered queue.
static constexpr uint32_t poolSliceControlDispatchGeneration = 5;
static constexpr uint32_t poolSliceControlReturnGeneration =
    poolSliceControlDispatchGeneration + poolSliceMaxPoolBlocks;
static constexpr uint32_t poolSliceControlScatterGeneration =
    poolSliceControlReturnGeneration + poolSliceMaxPoolBlocks;
static constexpr uint32_t poolSliceControlStart =
    poolSliceControlScatterGeneration + poolSliceMaxPoolBlocks;
static constexpr uint32_t poolSliceControlDispatchReady =
    poolSliceControlStart + 1;
static constexpr uint32_t poolSliceControlScatterStart =
    poolSliceControlDispatchReady + 1;
static constexpr uint32_t poolSliceControlReaderRowCount =
    poolSliceControlScatterStart + 1;
static constexpr uint32_t poolSliceControlStreamSendTotal =
    poolSliceControlReaderRowCount + poolSliceMaxLocalReaders;
static constexpr uint32_t poolSliceControlStreamSendDone =
    poolSliceControlStreamSendTotal + 1;
static constexpr uint32_t poolSliceControlStreamQueueRetiredMask =
    poolSliceControlStreamSendDone + 1;
// One put-with-signal publishes a standalone runtime-sized route/queue
// metadata packet into this monotonic per-source transport generation.
static constexpr uint32_t poolSliceControlStreamMetadataTransportReady =
    poolSliceControlStreamQueueRetiredMask + 1;
static constexpr uint32_t poolSliceControlStreamMetadataReady =
    poolSliceControlStreamMetadataTransportReady + poolSliceMaxPes;
static constexpr uint32_t poolSliceControlStreamRouteReady =
    poolSliceControlStreamMetadataReady + poolSliceMaxPes;
static constexpr uint32_t poolSliceControlStreamDataReady =
    poolSliceControlStreamRouteReady + poolSliceMaxPes;
static constexpr uint32_t poolSliceControlStreamQueueHead =
    poolSliceControlStreamDataReady +
    poolSliceMaxPes * poolSliceMaxDataGroups *
        poolSliceCompletionSlots;
static constexpr uint32_t poolSliceControlStreamQueueClaim =
    poolSliceControlStreamQueueHead +
    poolSliceMaxPes * poolSliceMaxStreamQueues;
static constexpr uint32_t poolSliceControlReturnReady =
    poolSliceControlStreamQueueClaim +
    poolSliceMaxPes * poolSliceMaxStreamQueues;
static constexpr uint32_t poolSliceControlReturnGroupCount =
    poolSliceControlReturnReady +
    poolSliceMaxPes * poolSliceMaxReturnReady;
// Every source publishes exactly one metadata envelope to every destination
// per invocation, including an empty envelope. Remember the last source
// sequence so its fused packet can advance one monotonic ADD generation.
static constexpr uint32_t poolSliceControlStreamMetadataSourceSequence =
    poolSliceControlReturnGroupCount +
    poolSliceMaxPes * poolSliceReturnGroupsPerSource;
static constexpr uint32_t poolSliceControlStreamMetadataSignalDelta =
    poolSliceControlStreamMetadataSourceSequence + 1;
static constexpr uint32_t poolSliceControlWords =
    poolSliceControlStreamMetadataSignalDelta + 1;
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
static constexpr uint32_t poolSliceProfileFirstGather = 17;
// Streaming-dispatch-only boundary: all destination copy groups have retired.
// The source transport boundary remains poolSliceProfileDataPublished.
static constexpr uint32_t poolSliceProfileStreamGatherDone = 18;
// Weighted-executor-only telemetry. These events expose whether destination
// reduction, network posting, or CTA-local transport completion is serialized
// without introducing a helper kernel.
static constexpr uint32_t poolSliceProfileReturnReduceStart = 19;
static constexpr uint32_t poolSliceProfileReturnReduceDone = 20;
static constexpr uint32_t poolSliceProfileFirstReturnPut = 21;
static constexpr uint32_t poolSliceProfileReturnCtaDone = 22;
enum PoolSliceStatus : uint64_t {
  POOL_SLICE_STATUS_OK = 0,
  POOL_SLICE_STATUS_BATCH = 1,
  POOL_SLICE_STATUS_ROUTE_RANGE = 2,
};

enum PoolSliceBatchFlags : uint32_t {
  POOL_SLICE_BATCH_FLAGS_NONE = 0,
  POOL_SLICE_BATCH_FLAGS_ERROR = 1U << 0,
};

// Immutable instructions in each source-owned destination queue. Queue zero
// starts with one RESERVE_ROUTES macro that consumes the 64-byte source
// envelope. COPY_ROWS instructions are striped over exactly two queues; each
// queue terminates with END. Data readiness lives in separate named static-QP
// slots so an early payload signal can never race a metadata write or an
// independently mapped transport context.
enum PoolSliceQueueOpcode : uint32_t {
  POOL_SLICE_QUEUE_RESERVE_ROUTES = 1,
  POOL_SLICE_QUEUE_COPY_ROWS = 2,
  POOL_SLICE_QUEUE_END = 3,
};

struct alignas(16) PoolSliceQueueEntry {
  uint64_t sequence;
  uint32_t message_index;
  uint32_t opcode;
  uint32_t row_begin;
  uint32_t row_end;
  uint32_t ready_slot;
  uint32_t flags;
};
static_assert(
    sizeof(PoolSliceQueueEntry) == 32,
    "PoolSliceQueueEntry ABI changed");

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

// Metadata is laid out per peer with the two queues interleaved by slot. The
// runtime-sized 32-bit row16/BF16 route words follow the last live queue slot
// round in the same
// packet, so one payload-coupled generation protects the complete metadata
// message without sending the unused queue tail.
struct alignas(16) PoolSliceMetadataEnvelope {
  PoolSlicePublishBatch batch;
  PoolSliceQueueEntry
      queues[poolSliceStreamQueueDepth][poolSliceMaxStreamQueues];
};
static_assert(
    sizeof(PoolSliceMetadataEnvelope) ==
        sizeof(PoolSlicePublishBatch) +
            poolSliceMaxStreamQueues * poolSliceStreamQueueDepth *
                sizeof(PoolSliceQueueEntry),
    "PoolSliceMetadataEnvelope ABI changed");

// The target records the local contiguous range assigned to each
// (reader, source) batch. The same record drives the pool-owned return path.
struct alignas(16) PoolSliceReceiveBatch {
  uint64_t sequence;
  uint32_t base_row;
  uint32_t source_begin;
  uint32_t row_count;
  uint32_t source_pe;
  uint32_t local_reader;
  uint32_t flags;
};
static_assert(
    sizeof(PoolSliceReceiveBatch) == 32,
    "PoolSliceReceiveBatch ABI changed");

// Python packs this 192-byte ABI in python/dae/pool_slice.py. Every
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
  uint64_t receive_routes_address;
  uint64_t sequence_address;
  uint64_t control_address;

  uint32_t row_bytes;
  uint32_t active_rows;
  uint32_t token_capacity;
  uint32_t route_capacity;
  uint32_t expert_capacity_rows;
  uint32_t local_readers;
  uint32_t num_pes;
  uint32_t my_pe;
  uint32_t signal_base;
  uint32_t group_limit;
  uint32_t write_chunks;
  uint32_t write_chunk_rows;
  uint32_t pool_rank;
  uint32_t pool_count;
};
static_assert(sizeof(PoolSliceConfig) == 192, "PoolSliceConfig ABI changed");

// Static per-peer host data-plane state. Metadata continues to use the base
// PoolSliceConfig/NVSHMEM path; these addresses are consulted only where the
// ordinary executor would deliver payload bytes and its ready generation.
struct PoolSliceHostPeer {
  uint64_t ring_memory;
  uint64_t remote_delivery_address;
  uint64_t remote_return_inbox_address;
  uint64_t remote_control_address;
  uint64_t remote_rkey;
};
static_assert(sizeof(PoolSliceHostPeer) == 40,
              "PoolSliceHostPeer ABI changed");

struct alignas(16) PoolSliceHostConfig {
  PoolSliceConfig pool;
  uint64_t peers_address;
  uint64_t producer_generations_address;
  uint32_t local_lkey;
  uint32_t reserved;
};
static_assert(sizeof(PoolSliceHostConfig) == 224,
              "PoolSliceHostConfig ABI changed");
