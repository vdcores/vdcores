#pragma once

#include "virtualcore.cuh"
#include "allocator.cuh"
#include "queue.cuh"

union CompiledLdCmd {
  struct {
    uint8_t slot;
    uint8_t m2c_slot;
    uint16_t reserved;
  };
  int raw;

  __device__ __forceinline__ void init(uint8_t s, uint8_t mslot) {
    slot = s;
    m2c_slot = mslot;
    reserved = 0;
  }

  static __device__ __forceinline__ CompiledLdCmd end() {
    CompiledLdCmd cmd {};
    cmd.slot = SLOT_END;
    return cmd;
  }
};

static __device__ __forceinline__ CInst dae_make_compiled_cinst(
  uint16_t opcode,
  uint16_t arg0 = 0,
  uint16_t arg1 = 0,
  uint16_t arg2 = 0
) {
  CInst inst {};
  inst.opcode = opcode;
  inst.args[0] = arg0;
  inst.args[1] = arg1;
  inst.args[2] = arg2;
  return inst;
}

static __device__ __forceinline__ MInst dae_make_compiled_minst_address(
  uint16_t opcode,
  uint16_t size,
  uint16_t num_slots,
  uint16_t arg,
  uint64_t address
) {
  MInst inst {};
  inst.opcode = opcode;
  inst.size = size;
  inst.num_slots = num_slots;
  inst.arg = arg;
  inst.address = address;
  return inst;
}

static __device__ __forceinline__ MInst dae_make_compiled_minst_coords(
  uint16_t opcode,
  uint16_t size,
  uint16_t num_slots,
  uint16_t arg,
  uint16_t c0,
  uint16_t c1,
  uint16_t c2,
  uint16_t c3
) {
  MInst inst {};
  inst.opcode = opcode;
  inst.size = size;
  inst.num_slots = num_slots;
  inst.arg = arg;
  inst.coords[0] = c0;
  inst.coords[1] = c1;
  inst.coords[2] = c2;
  inst.coords[3] = c3;
  return inst;
}

#if __has_include("dae/compiled_program.inc")
  #include "dae/compiled_program.inc"
#else
static constexpr bool daeCompiledProgramEnabled = false;
static constexpr const char *daeCompiledProgramHash = "";
static constexpr int daeCompiledProgramNumSms = 0;
static constexpr int daeCompiledProgramLiveValueCount = 0;

static __device__ __forceinline__ int dae_compiled_live_offset_for_sm(int) {
  return 0;
}

template <typename... Args>
static __device__ __forceinline__ void dae_compiled_compute_execute(Args&&...) {
  assert(false && "compiled mode was not built into this runtime");
}

template <typename... Args>
static __device__ __forceinline__ void dae_compiled_alloc_execute(Args&&...) {
  assert(false && "compiled mode was not built into this runtime");
}

template <typename... Args>
static __device__ __forceinline__ void dae_compiled_ld_execute(Args&&...) {
  assert(false && "compiled mode was not built into this runtime");
}

template <typename... Args>
static __device__ __forceinline__ void dae_compiled_st_execute(Args&&...) {
  assert(false && "compiled mode was not built into this runtime");
}
#endif
