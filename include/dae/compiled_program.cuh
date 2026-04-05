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

#if __has_include("dae/compiled_program.inc")
  #include "dae/compiled_program.inc"
#else
static constexpr bool daeCompiledProgramEnabled = false;
static constexpr const char *daeCompiledProgramHash = "";
static constexpr int daeCompiledProgramNumSms = 0;

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
