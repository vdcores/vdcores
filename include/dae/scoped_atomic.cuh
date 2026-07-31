#pragma once

#include <cstdint>

// Small HBM synchronization primitives for PoolInst implementations. These
// map directly to scoped PTX instead of instantiating a C++ atomic wrapper.
// Remote visibility stays in NVSHMEM primitives, so this file exposes only
// GPU-scope dependencies and bookkeeping atomics.
static __device__ __forceinline__ void dae_atomic_add_release_gpu(
    uint64_t* address, uint64_t value) {
  asm volatile(
      "red.release.gpu.global.add.u64 [%0], %1;"
      :
      : "l"(address), "l"(value)
      : "memory");
}

static __device__ __forceinline__ uint32_t
dae_atomic_fetch_add_acq_rel_gpu(uint32_t* address, uint32_t value) {
  uint32_t previous;
  asm volatile(
      "atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
      : "=r"(previous)
      : "l"(address), "r"(value)
      : "memory");
  return previous;
}

static __device__ __forceinline__ uint64_t dae_atomic_fetch_or_acq_rel_gpu(
    uint64_t* address, uint64_t value) {
  uint64_t previous;
  asm volatile(
      "atom.acq_rel.gpu.global.or.b64 %0, [%1], %2;"
      : "=l"(previous)
      : "l"(address), "l"(value)
      : "memory");
  return previous;
}

static __device__ __forceinline__ void dae_atomic_or_release_gpu(
    uint64_t* address, uint64_t value) {
  asm volatile(
      "red.release.gpu.global.or.b64 [%0], %1;"
      :
      : "l"(address), "l"(value)
      : "memory");
}

static __device__ __forceinline__ uint64_t
dae_atomic_compare_exchange_acquire_gpu(
    uint64_t* address, uint64_t expected, uint64_t desired) {
  uint64_t previous;
  asm volatile(
      "atom.acquire.gpu.global.cas.b64 %0, [%1], %2, %3;"
      : "=l"(previous)
      : "l"(address), "l"(expected), "l"(desired)
      : "memory");
  return previous;
}

static __device__ __forceinline__ uint64_t dae_atomic_load_acquire_gpu(
    const uint64_t* address) {
  uint64_t value;
  asm volatile(
      "ld.acquire.gpu.global.u64 %0, [%1];"
      : "=l"(value)
      : "l"(address)
      : "memory");
  return value;
}

static __device__ __forceinline__ uint32_t dae_atomic_load_relaxed_gpu(
    const uint32_t* address) {
  uint32_t value;
  asm volatile(
      "ld.relaxed.gpu.global.u32 %0, [%1];"
      : "=r"(value)
      : "l"(address)
      : "memory");
  return value;
}

static __device__ __forceinline__ void dae_atomic_store_release_gpu(
    uint64_t* address, uint64_t value) {
  asm volatile(
      "st.release.gpu.global.u64 [%0], %1;"
      :
      : "l"(address), "l"(value)
      : "memory");
}

// Sequence words carry no ordering. CUDA's native atomics are preferable to
// a general C++ atomic wrapper for this bookkeeping path.
static __device__ __forceinline__ uint64_t dae_atomic_load_relaxed_gpu(
    uint64_t* address) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  return atomicAdd(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(0));
}

static __device__ __forceinline__ void dae_atomic_store_relaxed_gpu(
    uint64_t* address, uint64_t value) {
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long));
  atomicExch(
      reinterpret_cast<unsigned long long*>(address),
      static_cast<unsigned long long>(value));
}
