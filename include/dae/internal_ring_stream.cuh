#pragma once

#include <cstdint>

// Descriptor-driven allocator-owned TMA ring.  The memory opcode owns the
// lease and persistent full/empty barriers; consumers see one M2C lease and
// fixed stage offsets.  A plan can assign independent descriptor streams to
// both LDU ports without changing the memory instruction sequence.
namespace dae_internal_ring {

static constexpr int kMaxStages = 2;
static constexpr int kPorts = 2;
// Four coordinates cover every descriptor currently used by the resident
// runtime while keeping each per-port plan exactly one 64-byte cache line.
static constexpr int kMaxRank = 4;

enum PlanFlags : uint32_t {
  kCacheEvictFirst = 1U << 0,
  kCacheEvictLast = 1U << 1,
};

struct alignas(16) TmaLanePlan {
  uint16_t descriptor_index;
  uint8_t rank;
  uint8_t issue_count;
  uint32_t transaction_bytes;
  uint32_t destination_offset;
  uint32_t destination_issue_stride;
  int32_t coordinates[kMaxRank];
  int32_t iteration_delta[kMaxRank];
  int32_t issue_delta[kMaxRank];
};

struct alignas(16) TmaPlan {
  uint32_t stage_bytes;
  uint32_t flags;
  uint32_t reserved0;
  uint32_t reserved1;
  TmaLanePlan lanes[kPorts];
};

static_assert(sizeof(TmaLanePlan) == 64);
static_assert(sizeof(TmaPlan) == 144);

}  // namespace dae_internal_ring
