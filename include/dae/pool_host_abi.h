#pragma once

#include <cstddef>
#include <cstdint>

// Coherent Grace/GPU request ring used only by the host data-plane PoolInst.
// A zero-row request is the ordered per-peer epoch terminator; data requests
// retain the ordinary payload-coupled HBM readiness sequence.
static constexpr uint32_t hostSglRingCapacity = 64;
// One request may cover a full inference token batch when the destination
// group limit is smaller than ceil(tokens / 32). Keep the bound static so the
// producer has no fragmented-row fallback in its hot path.
static constexpr uint32_t hostSglRingMaxRows = 512;
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

struct HostSglRouteGroup {
  uint32_t source_row_begin;
  uint32_t row_count;
  uint32_t remote_row_begin;
  uint32_t peer_index;
};

struct HostSglPeerRoute {
  uint64_t ring_memory;
  uint64_t remote_data_base;
  uint64_t remote_signal_base;
  uint64_t remote_rkey;
  uint64_t group_begin;
  uint64_t group_count;
};

struct alignas(64) HostSglRingMemory {
  uint32_t abi_version;
  uint32_t capacity;
  uint32_t max_rows;
  uint32_t reserved;
  uint64_t slots_address;
  uint64_t indices_address;
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
static_assert(sizeof(HostSglRouteGroup) == 16,
              "HostSglRouteGroup is part of the CUDA ABI");
static_assert(sizeof(HostSglPeerRoute) == 48,
              "HostSglPeerRoute is part of the CUDA ABI");
static_assert(sizeof(HostSglRingMemory) == 64,
              "HostSglRingMemory ABI changed");
static_assert(sizeof(HostSglRingSlot) == 128,
              "HostSglRingSlot ABI changed");
