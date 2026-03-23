#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>

// features
constexpr bool dae2EnableLooping = true;
constexpr bool dae2EnableGroup = true;
constexpr bool dae2BlockingStore = false;
constexpr bool dae2LoadInstructions = false;

static constexpr int slotSizeKb = 8;
static constexpr int numSlots = 24;
static constexpr int numInsts = dae2LoadInstructions ? 512 : 4096;
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

// barrier configurations
static constexpr int numThreadsM2CBarrier = numComputeWarps * numThreadsPerWarp + 1;
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
static constexpr uint16_t tmaDescBits = 10;
static constexpr uint16_t tmaDescMask = (1U << tmaDescBits) - 1;
static constexpr uint16_t tmaIndexShift = tmaDescBits;
static constexpr uint16_t tmaIndexMask = 0x3U << tmaIndexShift;
static_assert(numSlots <= (1 << slotBits), "numSlots exceeds slotBits capacity");

enum TmaIndexedRestMode : uint16_t {
  TMA_INDEX_NONE = 0,
  TMA_INDEX_LAYER = 1,
  TMA_INDEX_LAYER_EXPERT = 2,
};

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

static __device__ __host__ constexpr uint16_t encode_tma_arg(
    uint16_t desc_id, TmaIndexedRestMode mode = TMA_INDEX_NONE) {
  return (desc_id & tmaDescMask) | (static_cast<uint16_t>(mode) << tmaIndexShift);
}

static __device__ __host__ constexpr uint16_t decode_tma_desc_id(uint16_t arg) {
  return arg & tmaDescMask;
}

static __device__ __host__ constexpr TmaIndexedRestMode decode_tma_index_mode(uint16_t arg) {
  return static_cast<TmaIndexedRestMode>((arg & tmaIndexMask) >> tmaIndexShift);
}
