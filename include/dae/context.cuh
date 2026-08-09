#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>

// features
constexpr bool dae2EnableLooping = true;
constexpr bool dae2EnableGroup = true;
constexpr bool dae2BlockingStore = false;
constexpr bool dae2LoadInstructions = true;

#ifndef DAE_M2C_OBSERVER_WAIT
#define DAE_M2C_OBSERVER_WAIT 1
#endif
// The load VCore owns each M2C barrier phase. Compute threads only observe the
// phase transition, avoiding 128 redundant arrivals per loaded operand.
constexpr bool dae2M2CObserverWait = DAE_M2C_OBSERVER_WAIT != 0;

static constexpr int slotSizeKb = 8;
#ifndef DAE_NUM_SLOTS
#define DAE_NUM_SLOTS 24
#endif
static constexpr int numSlots = DAE_NUM_SLOTS;
#ifndef DAE_NUM_INSTS
#define DAE_NUM_INSTS 512
#endif
#ifndef DAE_DYNAMIC_SMEM_KB
#define DAE_DYNAMIC_SMEM_KB 212
#endif
static constexpr int dynamicSmemBytes = DAE_DYNAMIC_SMEM_KB * 1024;
// Attention scratch is a physical dynamic-shared-memory region, not an
// allocator slot.  The packed swapped-attention build places its compact
// scratch after the allocator arena; other attention paths retain slot 24.
#if defined(DAE_PACKED_SWAP_ATTENTION_SCRATCH)
static constexpr int attentionScratchSlot = numSlots;
#else
static constexpr int attentionScratchSlot = 24;
#endif
static constexpr int numInsts = dae2LoadInstructions ? DAE_NUM_INSTS : 4096;
static constexpr int numTmas = 1024;
static constexpr int numBars = 1024;

static constexpr int numSpecialSlots = 9;

static_assert(numSlots + numSpecialSlots <= ((2<<6) - 1), "Total number of slots must be less than or equal to 32");

static constexpr int numComputeWarps = 4;
static constexpr int numMemoryWarps = 4;

static constexpr int numThreadsPerWarp = 32;
static constexpr int numThreads = numThreadsPerWarp * (numComputeWarps + numMemoryWarps);
// one warpgroup + 1 memory warp
static constexpr int numProfileEvents = 128;
static constexpr int numComputeLoopCounters = 4;

#if defined(DAE_TRACK_PROFILE)
// Diagnostic-only, per-SM aggregate counters.  Keep these at the high end of
// the existing profile row so schedule-level OP_PROFILE_EVENT markers can use
// the low IDs on the same global-timer timeline.
enum DAETrackProfileEvent : int {
  DAE_TRACK_COMPUTE_M2C_WAIT_NS = 96,
  DAE_TRACK_COMPUTE_M2C_WAIT_CALLS = 97,
  DAE_TRACK_COMPUTE_M2C_CONTENDED = 98,
  DAE_TRACK_ALLOC_SLOT_STALL_NS = 99,
  DAE_TRACK_ALLOC_SLOT_STALL_EVENTS = 100,
  DAE_TRACK_ALLOC_SLOT_RETRIES = 101,
  DAE_TRACK_ALLOC_ISSUE_BARRIER_NS = 102,
  DAE_TRACK_ALLOC_ISSUE_BARRIER_CONTENDED = 103,
  DAE_TRACK_ALLOC_INSTRUCTIONS = 104,
  DAE_TRACK_LDU0_QUEUE_WAIT_NS = 105,
  DAE_TRACK_LDU0_QUEUE_WAIT_CALLS = 106,
  DAE_TRACK_LDU0_DEPENDENCY_WAIT_NS = 107,
  DAE_TRACK_LDU0_DEPENDENCY_CONTENDED = 108,
  DAE_TRACK_LDU0_COMMANDS = 109,
  DAE_TRACK_LDU1_QUEUE_WAIT_NS = 110,
  DAE_TRACK_LDU1_QUEUE_WAIT_CALLS = 111,
  DAE_TRACK_LDU1_DEPENDENCY_WAIT_NS = 112,
  DAE_TRACK_LDU1_DEPENDENCY_CONTENDED = 113,
  DAE_TRACK_LDU1_COMMANDS = 114,
  DAE_TRACK_STORE_QUEUE_WAIT_NS = 115,
  DAE_TRACK_STORE_QUEUE_WAIT_CALLS = 116,
  DAE_TRACK_STORE_SERVICE_NS = 117,
  DAE_TRACK_STORE_BARRIER_SERVICE_NS = 118,
  DAE_TRACK_STORE_COMMANDS = 119,
  DAE_TRACK_STORE_BARRIER_COMMANDS = 120,
  DAE_TRACK_MAGIC = 127,
};
static constexpr uint64_t daeTrackProfileMagic = 0x4454524b50524631ULL;
#endif

struct alignas(16) LoopCounters {
  uint32_t values[numComputeLoopCounters] = {};
};

// barrier configurations
static constexpr int numThreadsM2CBarrier =
    dae2M2CObserverWait ? 1 : numComputeWarps * numThreadsPerWarp + 1;
static constexpr int numThreadsC2MBarrier = numComputeWarps * numThreadsPerWarp + 1;
static constexpr int numThreadsLDBarrier = 2;

// Polling backoff for the memory core hot loops.
static constexpr int allocRetrySleepCycles = 16;
static constexpr int barrierPollSleepCycles = 16;

// Allocwarp instruction prefetch policy.
static constexpr int allocwarpInstructionPrefetchDistance = 2;
static constexpr int allocwarpInstructionSeedCount = 2;
static constexpr int allocwarpInstructionTargetSpan = 2;

constexpr int flagBits = 6;
constexpr int slotBits = 6;
static_assert(numSlots <= (1 << slotBits), "numSlots exceeds slotBits capacity");

// definition of instruction formats
struct alignas(8) CInst {
  uint16_t opcode;
  uint16_t args[3];
};


// we reserve the lower 6 bit of opcode as decode bits
enum InstOpDecode : uint16_t {
  MEM_OP_FLAGS_NONE = 0x0,
  MEM_OP_FLAGS_ALLOCATE = 0x1,
  MEM_OP_FLAGS_WRITEBACK = 0x2,
  MEM_OP_FLAGS_GROUP = 0x4,
  MEM_OP_FLAGS_JUMP = 0x8,
  MEM_OP_FLAGS_BARRIER = 0x10,
  MEM_OP_FLAGS_PORT = 0x20,
};

enum InstOpDecodeMask : uint16_t {
  MEM_OP_MASK_FLAGS = (1U << flagBits) - 1,
  MEM_OP_MASK_PENDING = 0x0003,
};

static __device__ __host__ __forceinline__ constexpr uint16_t rmask(const uint16_t mask) {
  return (uint16_t)(~mask);
}

#define MK_MOP(opcode, flags) \
    ((uint16_t)(((opcode) << flagBits) | ((flags) & ((1U << flagBits) - 1))))
    
enum InstOpcode : uint16_t {
  #define DAE_OP(name, value) name = value,
    #include "dae/opcode.cuh.inc"
  #undef DAE_OP
};

// TODO(zhiyuang): load128
struct alignas(16) MInst {
  uint16_t opcode; // 12 bits opcode + 4 bits flags
  uint16_t size;
  union {
    struct {
      uint16_t num_slots;
      uint16_t arg;
    };
    uint32_t shifter; // for shifting the address or arg field
  };

  union {
    uint64_t address;     // For other purpose
    uint16_t coords[4];   // For up to 4D TMA coordinates
  };

  __device__ __forceinline__ uint16_t flag(const uint16_t f) const {
    return opcode & f;
  }
  __device__ __forceinline__ uint16_t nslot() const {
    constexpr uint16_t slotMask = (1U << slotBits) - 1;
    return num_slots & slotMask;
  }
  __device__ __forceinline__ uint16_t bar() const {
    return num_slots >> slotBits;
  }
};

// helpers for building opcode
static __device__ __host__ constexpr uint16_t op(const uint16_t opcode) {
  return opcode >> flagBits;
}

static __device__ __host__ constexpr uint16_t jump(const uint16_t opcode) {
  return opcode | MEM_OP_FLAGS_JUMP;
}
