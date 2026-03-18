#pragma once

#include "context.cuh"

template <unsigned numSlots>
struct SharedMemoryAllocator {
  static_assert(numSlots <= 32, "MAX_SLOTS must be less than or equal to 32");

  __device__ __forceinline__ int allocate(int lane_id, int *slot_avail, uint16_t req, int &alloc) {
    // special handling for special allocations
    if (req >= numSlots) {
      // TODO (zijian): rawaddress uses this path but alloc is not set
      alloc = req;
      return req;
    }

    // find available slots, should have no flag set (ALLOCATE/WRITEBACK)
    // All threads participate in ballot, then mask to only consider first numSlots
    // see dae2.cuh, the slots are inited to 1 for non-allocating lanes, so no need to mask here
    uint32_t availbility = *(volatile uint32_t *)slot_avail;
    uint32_t mask = ((1U << req) - 1) << lane_id;
    // TODO(zhiyuang): check this mod. can we NOT mask availability here?
    uint32_t candidate = __ballot_sync(ALL_THREADS, (availbility & mask) == mask);
    // ActiveMask ensures bits >= numSlots are 0, so no out-of-bounds allocation
    int slot = __ffs(candidate) - 1;
    if (candidate) {
      if (lane_id == slot) {
        atomicAnd(slot_avail, ~mask); // mark these slots as occupied.
      }
      alloc = __shfl_sync(ALL_THREADS, mask, slot);
    }


    // set the stall flag inside the allocator
    __mprint("[Alloc] request %d slots: avail=0x%08x mask=0x%x slot=%d, alloc=0x%x avail_after=0x%08x",
            (int)req, availbility, mask, slot, alloc, *(volatile uint32_t *)slot_avail);
    return slot;
  }
};

template <unsigned numSlots>
struct CollectionMemoryAllocator {
  static_assert(numSlots <= 32, "MAX_SLOTS must be less than or equal to 32");

  // registers providing the initial value
  uint16_t cur_lead = 0xFFFFU, opcode = 0;

  __device__ __forceinline__ uint32_t ballot(const uint16_t mask) {
    return __ballot_sync(ALL_THREADS, opcode & mask);
  }

  __device__ __forceinline__ int allocate(int lane_id, MInst &inst, bool &stall) {
    constexpr unsigned ActiveMask = numSlots == 32 ? 0xFFFFFFFFU : (1U << numSlots) - 1;
    uint16_t req = inst.nslot();

    // find available slots, should have no flag set (ALLOCATE/WRITEBACK)
    // All threads participate in ballot, then mask to only consider first numSlots
    uint32_t availbility = __ballot_sync(ALL_THREADS, !(opcode & MEM_OP_MASK_PENDING)) & ActiveMask;
    uint32_t mask = ((1U << req) - 1) << __memory_tid();
    uint32_t candidate = __ballot_sync(ALL_THREADS, (availbility & mask) == mask) & ActiveMask;
    // ActiveMask ensures bits >= numSlots are 0, so no out-of-bounds allocation
    int slot_id = __ffs(candidate) - 1; // return -1 if no slot available
    if (slot_id >= 0) {
      if (lane_id >= slot_id && lane_id < slot_id + req) {
        opcode = inst.opcode; // mainly to get the flags
        cur_lead = slot_id;
      }
    }
    // set the stall flag inside the allocator
    stall = (slot_id < 0);
    __mprint("[Alloc] request %d slots: addr=0x%lx avail=0x%x mask=0x%x candidate=0x%x slot_id=%d"
             " Myslot: status=%04x cur_lead=%d",
            (int)req, inst.address, availbility, mask, candidate, slot_id, opcode, cur_lead);
    return slot_id;
  }

  __device__ __forceinline__ void collect_committed() {
    // If the is satisfied, we clear the ALLOCATE flag
    if ((opcode & MEM_OP_FLAGS_ALLOCATE) == 0)
      opcode &= rmask(MEM_OP_FLAGS_WRITEBACK);
  }

  __device__ __forceinline__ void mark_used(int slot_id) {
    // it's ok to not clear the slot_id, as long as we clear the flag.
    // only free will match slot_id; double free is OK
    __mprint("[Alloc][Mark] mark slot %d as used, status %04x", slot_id, (int)opcode);
    constexpr uint16_t clearMask =
      dae2BlockingStore ?
        rmask(MEM_OP_FLAGS_ALLOCATE | MEM_OP_FLAGS_WRITEBACK) :
        rmask(MEM_OP_FLAGS_ALLOCATE);

    if (cur_lead == slot_id)
      opcode &= clearMask;
  }
};

// This class is meant to be used as register only
template<unsigned numSlots>
struct WarpParallelMemoryAllocator {
  static_assert(numSlots <= 32, "MAX_SLOTS must be less than or equal to 32");

  // register
  MInst slot;
  uint16_t cur_lead;

  __device__ __forceinline__ void init() {
      slot.opcode = MEM_OP_FLAGS_NONE;
      cur_lead = 0xFFFFU;
  }

  __device__ __forceinline__ uint32_t availability() {
    constexpr unsigned ActiveMask = numSlots == 32 ? 0xFFFFFFFFU : (1U << numSlots) - 1;
    // find available slots, should have no flag set (ALLOCATE/WRITEBACK)
    // All threads participate in ballot, then mask to only consider first numSlots
    uint32_t availbility = __ballot_sync(ALL_THREADS, !slot.flag(MEM_OP_MASK_PENDING)) & ActiveMask;
    return availbility;
  }

  __device__ __forceinline__ int allocate(MInst inst) {
    constexpr unsigned ActiveMask = numSlots == 32 ? 0xFFFFFFFFU : (1U << numSlots) - 1;
    uint16_t req = inst.nslot();

    // find available slots, should have no flag set (ALLOCATE/WRITEBACK)
    // All threads participate in ballot, then mask to only consider first numSlots
    uint32_t availbility = __ballot_sync(ALL_THREADS, !slot.flag(MEM_OP_MASK_PENDING)) & ActiveMask;
    uint32_t mask = ((1U << req) - 1) << __memory_tid();
    uint32_t candidate = __ballot_sync(ALL_THREADS, (availbility & mask) == mask) & ActiveMask;
    // ActiveMask ensures bits >= numSlots are 0, so no out-of-bounds allocation
    int slot_id = __ffs(candidate) - 1; // return -1 if no slot available
    if (slot_id >= 0) {
      if (__memory_tid() >= slot_id && __memory_tid() < slot_id + req) {
        slot = inst;
        cur_lead = slot_id;
      }
    }
    __mprint("[Alloc] request %d slots: addr=0x%lx avail=0x%x mask=0x%x candidate=0x%x slot_id=%d"
             " Myslot: status=%04x",
            (int)req, inst.address, availbility, mask, candidate, slot_id, slot.opcode);
    return slot_id;
  }

  __device__ __forceinline__ void collect_committed() {
    // If the is satisfied, we clear the ALLOCATE flag
    if ((slot.opcode & MEM_OP_FLAGS_ALLOCATE) == 0)
      slot.opcode &= rmask(MEM_OP_FLAGS_WRITEBACK);
  }

  __device__ __forceinline__ bool mark_used(int slot_id) {
    if (cur_lead == slot_id) {
      // TODO(zhiyuang): if any bug check this
      // TODO(zhiyuang): clear for a safe link function. check needed?
      // cur_lead = 0xFFFFU;
      slot.opcode &= rmask(MEM_OP_FLAGS_ALLOCATE);
    }
    uint16_t status = __shfl_sync(ALL_THREADS, slot.opcode, slot_id);
    __mprint("[Alloc][Mark] mark slot %d as used, status %d", slot_id, (int)status);
    return status & MEM_OP_FLAGS_WRITEBACK;
  }
  
  __device__ __forceinline__ MInst get(int slot_id) {
    MInst s;
    *(uint64_t*)(&s) = __shfl_sync(ALL_THREADS, *(const uint64_t*)(&slot), slot_id);
    *((uint64_t*)(&s) + 1) = __shfl_sync(ALL_THREADS, *((const uint64_t*)(&slot) + 1), slot_id);

    return s;
  }
};