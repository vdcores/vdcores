// runtime structures used for computation and memory virtual cores

#pragma once

#include "context.cuh"

// macros for runtime
template<int BarrierID, int Count>
__device__ __forceinline__ void __sync_barrier() {
    asm volatile(
        "bar.sync %0, %1;"
        :
        : "n"(BarrierID), "n"(Count)
        : "memory"
    );
}

template<int Count>
__device__ __forceinline__ void __sync_barrier(int BarrierID) {
    asm volatile(
        "bar.sync %0, %1;"
        :
        : "r"(BarrierID), "n"(Count)
        : "memory"
    );
}

__device__ __forceinline__ int load_l2(const int* addr) {
    int val;
    asm volatile(
        "ld.global.cg.u32 %0, [%1];"
        : "=r"(val) : "l"(addr));
    return val;
}

template<typename T>
__device__ __forceinline__ void prefetch_l1(const T* addr) {
    asm volatile(
        "prefetch.global.L1 [%0];"
        :
        : "l"(addr));
}

#define __compute_tid() (threadIdx.x)
#define __memory_tid() cuda::ptx::get_sreg_laneid()

static constexpr int __bar_cgroup = 8;
#define __sync_compute_group(N) __sync_barrier<__bar_cgroup, N>()
#define __activate_compute_group(N) if ((__compute_tid() >= N)) return

#define __compute__ __device__
#define __memory__ __device__

#define __print(tid, s, ...) \
  if (tid == 0) \
    printf("[%d] " s "\n", cuda::ptx::get_sreg_clusterid_x(),  ##__VA_ARGS__)

#define __smprint(sm, tid, s, ...) \
  if (tid == 0 && sm == cuda::ptx::get_sreg_clusterid_x()) \
    printf("[%d] " s "\n", cuda::ptx::get_sreg_clusterid_x(),  ##__VA_ARGS__)

#ifdef DAE_DEBUG_PRINT
  #if DAE_DEBUG_PRINT < 132
    #define __dae_print(tid, lbl, s, ...) \
      if (tid == 0 && cuda::ptx::get_sreg_clusterid_x() == DAE_DEBUG_PRINT) \
        printf("[%d][" lbl "] " s "\n", cuda::ptx::get_sreg_clusterid_x(),  ##__VA_ARGS__)
  #else
    #define __dae_print(tid, lbl, s, ...) \
      if (tid == 0) \
        printf("[%d][" lbl "] " s "\n", cuda::ptx::get_sreg_clusterid_x(),  ##__VA_ARGS__)
  #endif // DAE_DEBUG_PRINT < 132
#else // DAE_DEBUG_PRINT
  #define __dae_print(tid, lbl, s, ...)
#endif // DAE_DEBUG_PRINT

#define __kprint(s, ...) \
  __dae_print(__compute_tid(), "Kernel", s, ##__VA_ARGS__)
#define __cprint(s, ...) \
  __dae_print(__compute_tid(), "Compute", s, ##__VA_ARGS__)
#define __mprint(s, ...) \
  __dae_print(__memory_tid(), "CFU", s, ##__VA_ARGS__)
#define __ldprint(s, ...) \
  __dae_print(__memory_tid(), "LDU", s, ##__VA_ARGS__)
#define __stprint(s, ...) \
  __dae_print(__memory_tid(), "STU", s, ##__VA_ARGS__)

static constexpr unsigned ALL_THREADS = 0xFFFFFFFFU;
static constexpr unsigned SLOT_END = 0xFFU;

enum MemoryVirtualCoreGpr32 : int {
  MVC_GPR32_LOOP_COUNTER = 0,
  MVC_GPR32_JMP_CNT = 1,
  MVC_GPR32_LOOP_START_PC = 2,
  MVC_GPR32_BASE_REG = 3,
};

enum MemoryVirtualCoreGpr : int {
  MVC_GPR_DELTA = 0,
  MVC_GPR_ACC = 1,
};

static __device__ __forceinline__ void * get_slot_address(const void *base, uint8_t slot) {
  return (void *)((const unsigned char*)base + slot * slotSizeKb * 1024);
}

static __device__ __forceinline__ void * slot_2_glob_ptr(const MInst *st_insts, uint8_t slot) {
  uint64_t addr_v = st_insts[slot].address;
  void * glob_ptr = reinterpret_cast<void*>(addr_v);
  return glob_ptr;
}

// TODO(zhiyuang): do we want to put barrier index in the command?
// TODO(zhiyuang): do we want to do int32 or 64?
union LdCmd {
  struct {
    uint8_t slot;
    uint8_t bar;
    uint16_t opcode;
  };
  int raw;

  __device__ __forceinline__ void init(uint8_t s, uint8_t b, uint16_t op) {
    opcode = op;
    slot = s;
    bar = b;
  }
};

// definition of memory virtual core
struct MemoryVirtualCore {
  uint64_t gpr[2]; // 0 for delta and 1 for accumulator

  // control flow structures
  // 0 is special;
  int gpr_32[6];

  // runtime allocation state
  int slot_alloc;
  int port;
  // registers
  // TODO(zhiyuang): try better way to predict these? e.g, flag bits?
  bool pred_stall;
  bool pred_continue;
  bool pred_jump;
  bool pred_allocate;

  __device__ __forceinline__ void init() {
    pred_stall = false;
    pred_continue = true;
    pred_jump = false;
    pred_allocate = false;

    gpr_32[MVC_GPR32_LOOP_COUNTER] = 0;
    gpr_32[MVC_GPR32_JMP_CNT] = 0;
    gpr_32[MVC_GPR32_LOOP_START_PC] = 0;
    gpr_32[MVC_GPR32_BASE_REG] = 0;
    slot_alloc = -1;
  }

  template<typename T>
  __device__ __forceinline__ void reg_write(T &reg, T val, uint16_t lane) {
    if (__memory_tid() == lane)
      reg = val;
  }
  template<typename T>
  __device__ __forceinline__ uint64_t reg_read(T &reg, uint16_t lane) const {
    return __shfl_sync(ALL_THREADS, reg, lane);
  }

  // methods
  __device__ __forceinline__ void inst_decode(MInst &inst) {
    // decode predicates
    pred_allocate = inst.opcode & MEM_OP_FLAGS_ALLOCATE;
    pred_jump = inst.opcode & MEM_OP_FLAGS_JUMP;
    port = inst.opcode & MEM_OP_FLAGS_PORT ? 1 : 0;
    // pred_stall = false;
  }

  __device__ __forceinline__ bool id_repeat() const {
    return gpr_32[MVC_GPR32_LOOP_COUNTER];
  }
};

__device__ __forceinline__ uint32_t mkSlotMask(uint8_t slot, uint8_t nslot) {
  return ((1U << slot) - 1) ^ ((1U << (slot + nslot)) - 1);
}

__device__ __forceinline__ bool lane_in_half_open_range(int lane, int start, int end) {
  return (unsigned)(lane - start) < (unsigned)(end - start);
}

__device__ __forceinline__ bool lane_in_closed_range(int lane, int start, int end) {
  return (unsigned)(lane - start) <= (unsigned)(end - start);
}

template<int N, typename SrcT, typename DstT>
__device__ __forceinline__ void parallel_copy(int lane_id, SrcT *srcp, DstT *dstp) {
  static_assert(N % 4 == 0, "parallel_copy only supports copying multiple of 4 bytes");
  static_assert(N / 4 <= 32, "parallel_copy can copy at most 128 bytes");

  int32_t *src = reinterpret_cast<int32_t*>(srcp);
  int32_t *dst = reinterpret_cast<int32_t*>(dstp);

  if (lane_id < N / 4)
    dst[lane_id] = src[lane_id];
}
