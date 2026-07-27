#pragma once

#include <cstdint>

// Pool-local dependencies have exactly one producer and one or more polling
// consumers. A device-scope release/acquire flag is sufficient; the generic
// countdown barrier keeps atomicSub for arbitrary multi-producer operators.
static __device__ __forceinline__ void pool_signal_release(int* signal) {
  constexpr uint32_t ready = 0;
  asm volatile(
      "st.release.gpu.global.u32 [%0], %1;"
      :
      : "l"(signal), "r"(ready)
      : "memory");
}

static __device__ __forceinline__ bool pool_signal_ready(const int* signal) {
  uint32_t value;
  asm volatile(
      "ld.acquire.gpu.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(signal)
      : "memory");
  return value == 0;
}

