#pragma once

#include <cstddef>
#include <cstdint>

static constexpr uint32_t hostSglRingCapacity = 64;
static constexpr uint32_t hostSglRingMaxRows = 32;
static constexpr uint32_t hostSglRingMaxMessages = 32;
static constexpr uint32_t hostSglRingBatch = 16;

extern "C" {

struct HostSglEndpoint {
  uint32_t qp_num;
  uint32_t psn;
  uint16_t lid;
  uint8_t port_num;
  uint8_t gid_index;
  uint8_t active_mtu;
  uint8_t link_layer;
  uint8_t reserved[2];
  uint8_t gid[16];
};

struct HostSglRequest {
  uint32_t local_lkey;
  uint32_t remote_rkey;
  uint64_t source_base;
  uint64_t source_stride;
  uint32_t row_bytes;
  uint32_t row_count;
  const uint32_t* row_indices;
  uint64_t remote_data;
  uint64_t remote_signal;
  uint64_t sequence;
};

// Ordinary Grace memory shared coherently with a GPU producer. Slots and row
// indices live in the same aligned malloc allocation and are named explicitly
// so neither side needs a translated or pinned mirror.
struct alignas(64) HostSglRingMemory {
  uint32_t abi_version;
  uint32_t capacity;
  uint32_t max_rows;
  uint32_t reserved;
  uint64_t slots_address;
  uint64_t indices_address;
  uint64_t submitted_generation;
  uint64_t completed_generation;
  uint64_t error_generation;
  uint64_t reserved_words[1];
};

struct alignas(128) HostSglRingSlot {
  HostSglRequest request;
  uint64_t ready_generation;
  uint64_t consumed_generation;
  uint8_t reserved[48];
};

}  // extern "C"

static_assert(sizeof(HostSglEndpoint) == 32,
              "HostSglEndpoint is part of the ctypes ABI");
static_assert(sizeof(HostSglRequest) == 64,
              "HostSglRequest is part of the ctypes ABI");
static_assert(sizeof(HostSglRingMemory) == 64,
              "HostSglRingMemory ABI changed");
static_assert(sizeof(HostSglRingSlot) == 128,
              "HostSglRingSlot ABI changed");
