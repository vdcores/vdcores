#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/ptx>

// features
constexpr bool dae2EnableLooping = true;
constexpr bool dae2EnableGroup = true;
constexpr bool dae2BlockingStore = false;
#ifndef DAE_LOAD_INSTRUCTIONS
#define DAE_LOAD_INSTRUCTIONS 1
#endif
constexpr bool dae2LoadInstructions = DAE_LOAD_INSTRUCTIONS != 0;

#ifndef DAE_M2C_OBSERVER_WAIT
#define DAE_M2C_OBSERVER_WAIT 1
#endif
// The load VCore owns each M2C barrier phase. Compute threads only observe the
// phase transition, avoiding 128 redundant arrivals per loaded operand.
constexpr bool dae2M2CObserverWait = DAE_M2C_OBSERVER_WAIT != 0;
#ifndef DAE_M2C_POLL_SLEEP_CYCLES
#define DAE_M2C_POLL_SLEEP_CYCLES 0
#endif
static constexpr int m2cPollSleepCycles = DAE_M2C_POLL_SLEEP_CYCLES;
static_assert(m2cPollSleepCycles >= 0);
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

// The resident SM100 runtime allocates TMEM once and shares a small bank of
// completion barriers across sequential compute tasks. Barrier zero is the
// ordinary single-UMMA completion barrier. Grouped BF16 GEMV owns barriers
// 1..8. Native FP8 and NVFP4 use disjoint four-stage full/empty rings so their
// issuer and retire warps can overlap without perturbing legacy phase state or
// each other's persistent barrier parity.
static constexpr int tmemGroupedBarrierBase = 1;
static constexpr int tmemGroupedBarrierCount = 8;
static constexpr int fp8UmmaPipelineStages = 4;
static constexpr int fp8UmmaPipelineBarrierBase =
    tmemGroupedBarrierBase + tmemGroupedBarrierCount;
static constexpr int fp8UmmaPipelineBarrierCount =
    fp8UmmaPipelineStages * 2;
#ifndef DAE_NVFP4_UMMA_PIPELINE_STAGES
#define DAE_NVFP4_UMMA_PIPELINE_STAGES 4
#endif
static constexpr int nvfp4UmmaPipelineStages =
    DAE_NVFP4_UMMA_PIPELINE_STAGES;
static constexpr int nvfp4UmmaPipelineBarrierBase =
    fp8UmmaPipelineBarrierBase + fp8UmmaPipelineBarrierCount;
static constexpr int nvfp4UmmaPipelineBarrierCount =
    nvfp4UmmaPipelineStages * 2;
#ifndef DAE_NVFP4_SCALE_COPY_STAGES
#define DAE_NVFP4_SCALE_COPY_STAGES 2
#endif
static constexpr int nvfp4ScaleCopyBarrierCount =
    DAE_NVFP4_SCALE_COPY_STAGES;
static constexpr int nvfp4ScaleCopyBarrierBase =
    nvfp4UmmaPipelineBarrierBase + nvfp4UmmaPipelineBarrierCount;
// Direct MXFP4/MXFP8 TMA scales use a dedicated tail ring of 4-KiB stages.
// Each stage barrier is initialized empty, observed by both LDU scale streams,
// and released once by the UMMA completion warp.
#ifndef DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA
#define DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA 0
#endif
static constexpr bool mxfp4Mxfp8DirectTmaEnabled =
    DAE_ENABLE_MXFP4_MXFP8_DIRECT_TMA != 0;
#ifndef DAE_MXFP4_MXFP8_TMA_SCALE_STAGES
#define DAE_MXFP4_MXFP8_TMA_SCALE_STAGES 2
#endif
static constexpr int mxfp4Mxfp8TmaScaleStages =
    DAE_MXFP4_MXFP8_TMA_SCALE_STAGES;
static_assert(
    mxfp4Mxfp8TmaScaleStages == 2 || mxfp4Mxfp8TmaScaleStages == 3,
    "direct MXFP4/MXFP8 TMA scale ring supports two or three stages");
static constexpr int mxfp4Mxfp8TmaScaleBarrierBase =
    nvfp4ScaleCopyBarrierBase + nvfp4ScaleCopyBarrierCount;
static constexpr int mxfp4Mxfp8TmaScaleBarrierCount =
    mxfp4Mxfp8DirectTmaEnabled ? mxfp4Mxfp8TmaScaleStages : 0;

// The production resident FFN uses one fixed two-stage pipeline per operand
// family. Weight and scale transactions share one full barrier; activation/SFB
// has an independent Down full barrier so its producer dependency never gates
// the weight stream. Python initializes the BF16 reduction destination.
static constexpr int mxfpResidentLinear1Stages = 2;
static constexpr int mxfpResidentLinear1FullBarrierBase =
    mxfp4Mxfp8TmaScaleBarrierBase + mxfp4Mxfp8TmaScaleBarrierCount;
static constexpr int mxfpResidentLinear1EmptyBarrierBase =
    mxfpResidentLinear1FullBarrierBase + mxfpResidentLinear1Stages;
static constexpr int mxfpResidentDownStages = 2;
static constexpr int mxfpResidentDownWeightFullBarrierBase =
    mxfpResidentLinear1EmptyBarrierBase + mxfpResidentLinear1Stages;
static constexpr int mxfpResidentDownEmptyBarrierBase =
    mxfpResidentDownWeightFullBarrierBase + mxfpResidentDownStages;
static constexpr int mxfpDownResidentOperandFullBarrierBase =
    mxfpResidentDownEmptyBarrierBase + mxfpResidentDownStages;
static constexpr int mxfpDownResidentOperandFullBarrierCount =
    mxfpResidentDownStages;
// LDU1 resolves the two device-scope reduction-destination dependencies while
// compute is still in Linear-1/UMMA. The Down epilogues consume these one-shot
// CTA-local tokens instead of loading the global readiness word on the tail.
static constexpr int mxfpDownResidentReductionReadyBarrierBase =
    mxfpDownResidentOperandFullBarrierBase +
    mxfpDownResidentOperandFullBarrierCount;
static constexpr int mxfpDownResidentReductionReadyBarrierCount = 3;
static constexpr int mxfpDownResidentLdu1PollStartBarrier =
    mxfpDownResidentReductionReadyBarrierBase + 2;

// The allocator-owned common MXFP8 stream uses independent weight,
// activation, UMMA-completion, and empty barriers for each of its two ring
// stages. These barriers persist across sequential generic projection tasks;
// Python supplies each task's cumulative K-pair phase base.
static constexpr int mxfp8CoupledStages = 2;
static constexpr int mxfp8CoupledWeightFullBarrierBase =
    mxfpDownResidentReductionReadyBarrierBase +
    mxfpDownResidentReductionReadyBarrierCount;
static constexpr int mxfp8CoupledActivationFullBarrierBase =
    mxfp8CoupledWeightFullBarrierBase + mxfp8CoupledStages;
static constexpr int mxfp8CoupledUmmaFullBarrierBase =
    mxfp8CoupledActivationFullBarrierBase + mxfp8CoupledStages;
static constexpr int mxfp8CoupledEmptyBarrierBase =
    mxfp8CoupledUmmaFullBarrierBase + mxfp8CoupledStages;

// Generic descriptor-driven internal ring.  Each port owns an independent
// transaction-completion barrier per stage; both observe the same consumer-
// released empty barrier.  Port-local command counters and the compute-side
// consumer counter preserve parity across sequential commands.
static constexpr int internalRingStages = 2;
static constexpr int internalRingFullBarrierBase =
    mxfp8CoupledEmptyBarrierBase + mxfp8CoupledStages;
static constexpr int internalRingFullBarrierCount =
    2 * internalRingStages;
static constexpr int internalRingEmptyBarrierBase =
    internalRingFullBarrierBase + internalRingFullBarrierCount;

static constexpr int tmemMmaBarrierCount =
    internalRingEmptyBarrierBase + internalRingStages;

static constexpr int numThreadsPerWarp = 32;
static constexpr int numThreads = numThreadsPerWarp * (numComputeWarps + numMemoryWarps);
// Control mailboxes are owned by allocator lane zero and consumed by one lane
// in each LDU. The remaining allocator lanes never access the published
// metadata and therefore are not participants in its lifetime barrier.
static constexpr int numThreadsLduControlPublishBarrier = 3;
// one warpgroup + 1 memory warp
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
static constexpr int numProfileEvents = 160;
#elif defined(DAE_ATTENTION_DETAIL_PROFILE)
static constexpr int numProfileEvents = 160;
#else
static constexpr int numProfileEvents = 128;
#endif
static constexpr int layerProfileEventBase = 2;
static constexpr int reloadProfileEventBase = 64;
static constexpr int trackProfileEventBase = 96;
static constexpr int detailProfileEventBase = 128;
static_assert(layerProfileEventBase < reloadProfileEventBase);
static_assert(reloadProfileEventBase < trackProfileEventBase);
static_assert(trackProfileEventBase < numProfileEvents);
#if defined(DAE_STU_HISTORY_PROFILE)
// Track the four store commands immediately ahead of one profiled RawAddress
// completion. Reload timing is disabled in this diagnostic build, leaving the
// upper part of its event range available without enlarging the normal buffer.
static constexpr int stuRawPopBeginEvent = 64;
static constexpr int stuRawServiceIdentityEvent = 65;
static constexpr int stuRawOutputTokenEvent = 66;
static constexpr int stuRawPtrMatchEventBase = 67;
static constexpr int stuRawPostEventBase = 71;
static constexpr int stuRawPtrEventBase = 75;
static constexpr int stuRawArrivalEventBase = 79;
static constexpr int stuHistoryEventBase = 83;
static constexpr int stuHistoryCommands = 4;
static_assert(stuRawPopBeginEvent + 1 <= stuRawServiceIdentityEvent);
static_assert(stuRawOutputTokenEvent + 1 <= stuRawPtrMatchEventBase);
static_assert(
    stuRawPtrMatchEventBase + numComputeWarps <= stuRawPostEventBase);
static_assert(stuRawPostEventBase + numComputeWarps <= stuRawPtrEventBase);
static_assert(stuRawPtrEventBase + numComputeWarps <= stuRawArrivalEventBase);
static_assert(
    stuRawArrivalEventBase + numComputeWarps <= stuHistoryEventBase);
static_assert(
    stuHistoryEventBase + 3 * stuHistoryCommands + 1 <=
    trackProfileEventBase);
#endif
#if defined(DAE_ATTENTION_DETAIL_PROFILE)
static_assert(detailProfileEventBase + 30 <= numProfileEvents);
#endif
#if defined(DAE_FP8_COUPLED_DETAIL_PROFILE)
static constexpr int fp8CoupledDetailLduEventBase = detailProfileEventBase;
static constexpr int fp8CoupledDetailCommands = 8;
static constexpr int fp8CoupledDetailWaitEventBase =
    fp8CoupledDetailLduEventBase + 2 * fp8CoupledDetailCommands;
static constexpr int fp8CoupledDetailSourceEventBase =
    fp8CoupledDetailWaitEventBase + 6;
static constexpr int fp8CoupledQuantStoreEvent =
    fp8CoupledDetailSourceEventBase + 2;
static constexpr int fp8CoupledResetAllocationEventBase =
    fp8CoupledQuantStoreEvent + 2;
static constexpr int fp8CoupledReloadEndEvent =
    fp8CoupledResetAllocationEventBase + 3;
static_assert(fp8CoupledReloadEndEvent + 1 <= numProfileEvents);
#endif
#if defined(DAE_MXFP_FFN_DETAIL_PROFILE)
static constexpr int mxfpFfnDetailEventBase = reloadProfileEventBase;
static constexpr int mxfpFfnDetailAllocatorLinear1 =
    mxfpFfnDetailEventBase + 0;
static constexpr int mxfpFfnDetailAllocatorDownWeight =
    mxfpFfnDetailEventBase + 1;
static constexpr int mxfpFfnDetailAllocatorDownActivation =
    mxfpFfnDetailEventBase + 2;
static constexpr int mxfpFfnDetailLdu0Linear1Begin =
    mxfpFfnDetailEventBase + 3;
static constexpr int mxfpFfnDetailLdu0Linear1End =
    mxfpFfnDetailEventBase + 4;
static constexpr int mxfpFfnDetailLdu0DownBegin =
    mxfpFfnDetailEventBase + 5;
static constexpr int mxfpFfnDetailLdu0DownReady =
    mxfpFfnDetailEventBase + 6;
static constexpr int mxfpFfnDetailLdu0DownEnd =
    mxfpFfnDetailEventBase + 7;
static constexpr int mxfpFfnDetailLdu1ActivationBegin =
    mxfpFfnDetailEventBase + 8;
static constexpr int mxfpFfnDetailLdu1PollReady =
    mxfpFfnDetailEventBase + 9;
static constexpr int mxfpFfnDetailLdu1ActivationEnd =
    mxfpFfnDetailEventBase + 10;
static constexpr int mxfpFfnDetailComputeBegin =
    mxfpFfnDetailEventBase + 11;
static constexpr int mxfpFfnDetailComputeLinear1End =
    mxfpFfnDetailEventBase + 12;
static constexpr int mxfpFfnDetailComputeEnd =
    mxfpFfnDetailEventBase + 13;
static constexpr int mxfpFfnDetailLdu0PreviousBegin =
    mxfpFfnDetailEventBase + 14;
static constexpr int mxfpFfnDetailLdu0PreviousEnd =
    mxfpFfnDetailEventBase + 15;
static constexpr int mxfpFfnDetailLdu0PreviousOpcode =
    mxfpFfnDetailEventBase + 16;
static constexpr int mxfpFfnDetailLdu0Linear1PrologueNs =
    mxfpFfnDetailEventBase + 17;
static constexpr int mxfpFfnDetailLdu0Linear1EmptyWaitNs =
    mxfpFfnDetailEventBase + 18;
static constexpr int mxfpFfnDetailComputeLinear1WeightWaitNs =
    mxfpFfnDetailEventBase + 19;
static constexpr int mxfpFfnDetailComputeLinear1UmmaWaitNs =
    mxfpFfnDetailEventBase + 20;
static constexpr int mxfpFfnDetailLdu0Linear1AfterDependency =
    mxfpFfnDetailEventBase + 21;
static constexpr int mxfpFfnDetailLdu0DownEmptyWaitNs =
    mxfpFfnDetailEventBase + 22;
static constexpr int mxfpFfnDetailLdu1DownEmptyWaitNs =
    mxfpFfnDetailEventBase + 23;
static constexpr int mxfpFfnDetailComputeDownWeightWaitNs =
    mxfpFfnDetailEventBase + 24;
static constexpr int mxfpFfnDetailComputeDownOperandWaitNs =
    mxfpFfnDetailEventBase + 25;
static constexpr int mxfpFfnDetailComputeDownUmmaWaitNs =
    mxfpFfnDetailEventBase + 26;
static constexpr int mxfpFfnDetailComputeDownToUmmaDoneNs =
    mxfpFfnDetailEventBase + 27;
static constexpr int mxfpFfnDetailComputeDownEpilogueNs =
    mxfpFfnDetailEventBase + 28;
static constexpr int mxfpFfnDetailComputeDownReductionWaitNs =
    mxfpFfnDetailEventBase + 29;
static constexpr int mxfpFfnDetailComputeDownOutputTmaNs =
    mxfpFfnDetailEventBase + 30;
// Events 27--30 pack task zero in the low uint32 and task one in the high
// uint32. Event 31 packs their begin offsets from ComputeLinear1End. This
// preserves the production 128-event profile-row stride.
static constexpr int mxfpFfnDetailComputeDownTaskBeginOffsetPacked =
    mxfpFfnDetailEventBase + 31;
static_assert(
    mxfpFfnDetailComputeDownTaskBeginOffsetPacked < trackProfileEventBase);
#endif
static constexpr int numComputeLoopCounters = 4;
static constexpr int lduBarrierReloadArrival = numBars - 2;
static constexpr int lduBarrierReloadDone = numBars - 1;

#if defined(DAE_TRACK_PROFILE)
// Diagnostic-only, per-SM aggregate counters.  Keep these at the high end of
// the existing profile row so layer and reload frontiers can use low IDs on
// the same global-timer timeline.
enum DAETrackProfileEvent : int {
  DAE_TRACK_COMPUTE_M2C_WAIT_NS = trackProfileEventBase,
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
  DAE_TRACK_PHYSICAL_SM_ID = 121,
  DAE_TRACK_SM_CLOCK_START = 122,
  DAE_TRACK_SM_CLOCK_END = 123,
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
static constexpr bool dae2C2MPollWait = true;
static constexpr int numThreadsC2MBarrier =
    numComputeWarps * numThreadsPerWarp + 1;
static constexpr int numThreadsLDBarrier = 2;

// Polling backoff for the memory core hot loops.
static constexpr int allocRetrySleepCycles = 16;
static constexpr int barrierPollSleepCycles = 16;
#ifndef DAE_RELOAD_POLL_SLEEP_CYCLES
#define DAE_RELOAD_POLL_SLEEP_CYCLES 16
#endif
static constexpr int reloadBarrierPollSleepCycles =
    DAE_RELOAD_POLL_SLEEP_CYCLES;
static_assert(reloadBarrierPollSleepCycles >= 0);
#ifndef DAE_RELOAD_ATOMIC_ADD
#define DAE_RELOAD_ATOMIC_ADD 0
#endif
static constexpr bool reloadBarrierUseAtomicAdd =
    DAE_RELOAD_ATOMIC_ADD != 0;
#ifndef DAE_ASYNC_BARRIER_RELOAD
#define DAE_ASYNC_BARRIER_RELOAD 1
#endif
static constexpr bool dae2AsyncBarrierReload =
    DAE_ASYNC_BARRIER_RELOAD != 0;
#ifndef DAE_ASYNC_BARRIER_RELOAD_WORKERS
#define DAE_ASYNC_BARRIER_RELOAD_WORKERS 16
#endif
static constexpr int asyncBarrierReloadWorkers =
    DAE_ASYNC_BARRIER_RELOAD_WORKERS;
static_assert(
    asyncBarrierReloadWorkers > 0 &&
    asyncBarrierReloadWorkers <= numThreadsPerWarp * numComputeWarps);

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
