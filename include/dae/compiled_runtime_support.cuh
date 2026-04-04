#pragma once

#include "allocator.cuh"
#include "queue.cuh"
#include "virtualcore.cuh"

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/std/atomic>

struct CompiledLdCmd {
  int slot_mask;
  int bar;
};

enum class CompiledAllocOpKind : int {
  Load = 0,
  Store = 1,
  Ready = 2,
};

static __device__ __forceinline__ void *compiled_align_to(void *ptr, size_t align) {
  uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t aligned = (addr + align - 1) & ~(align - 1);
  return reinterpret_cast<void *>(aligned);
}

static __device__ __forceinline__ void compiled_wait_global_barrier(
    int *bars,
    int bar_id) {
  volatile int *bar = bars + bar_id;
  while (*bar != 0) {
    __nanosleep(barrierPollSleepCycles);
  }
}

static __device__ __forceinline__ void compiled_arrive_global_barrier(
    int *bars,
    int bar_id) {
  cuda::std::atomic_ref<int> bar{bars[bar_id]};
  (void)bar.fetch_sub(1, cuda::std::memory_order_release);
}

template <typename M2CQueue, typename LDQueue>
__device__ __forceinline__ int compiled_alloc_load(
    int lane_id,
    SharedMemoryAllocator<numSlots> &alloc,
    int *slot_avail,
    uint16_t req_slots,
    int port_id,
    M2CQueue &m2c,
    LDQueue ldq[2]) {
  int slot_mask = 0;
  int slot = -1;
  while (slot < 0) {
    slot = alloc.allocate(lane_id, slot_avail, req_slots, slot_mask);
    if (slot < 0) {
      __nanosleep(allocRetrySleepCycles);
    }
  }

  if (lane_id == 0) {
    const int bar = m2c.ptr;
    m2c.put(slot_mask);
    ldq[port_id].put(CompiledLdCmd{slot_mask, bar});
    m2c.advance();
    ldq[port_id].commit();
    ldq[port_id].advance();
  }
  return slot;
}

template <typename M2CQueue>
__device__ __forceinline__ int compiled_alloc_store(
    int lane_id,
    SharedMemoryAllocator<numSlots> &alloc,
    int *slot_avail,
    uint16_t req_slots,
    M2CQueue &m2c) {
  int slot_mask = 0;
  int slot = -1;
  while (slot < 0) {
    slot = alloc.allocate(lane_id, slot_avail, req_slots, slot_mask);
    if (slot < 0) {
      __nanosleep(allocRetrySleepCycles);
    }
  }

  if (lane_id == 0) {
    m2c.put(slot_mask);
    m2c.commit();
    m2c.advance();
  }
  return slot;
}

template <typename M2CQueue, typename LDQueue>
__device__ __forceinline__ void compiled_run_alloc_op(
    int lane_id,
    SharedMemoryAllocator<numSlots> &alloc,
    int *slot_avail,
    CompiledAllocOpKind kind,
    uint16_t req_slots,
    int port_id,
    int repeat_count,
    M2CQueue &m2c,
    LDQueue ldq[2]) {
  for (int repeat_idx = 0; repeat_idx < repeat_count; ++repeat_idx) {
    switch (kind) {
      case CompiledAllocOpKind::Load:
        (void)compiled_alloc_load(lane_id, alloc, slot_avail, req_slots, port_id, m2c, ldq);
        break;
      case CompiledAllocOpKind::Store:
        (void)compiled_alloc_store(lane_id, alloc, slot_avail, req_slots, m2c);
        break;
      case CompiledAllocOpKind::Ready:
        if (lane_id == 0) {
          m2c.put(req_slots);
          ldq[port_id].put(CompiledLdCmd{req_slots, static_cast<int>(m2c.ptr)});
          m2c.advance();
          ldq[port_id].commit();
          ldq[port_id].advance();
        }
        break;
    }
  }
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_load_1d(
    const CompiledLdCmd &cmd,
    uint64_t address,
    int size,
    const void *smem_base,
    M2CQueue &m2c) {
  int slot = extract(cmd.slot_mask);
  cuda::device::memcpy_async_tx(
      (char *)(get_slot_address(smem_base, slot)),
      reinterpret_cast<const char *>(address),
      cuda::aligned_size_t<16>(size),
      m2c.barriers[cmd.bar]);
  cuda::device::barrier_expect_tx(
      m2c.barriers[cmd.bar],
      cuda::aligned_size_t<16>(size));
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_tma_load_tensor_1d(
    const CompiledLdCmd &cmd,
    const CUtensorMap *tma_descs,
    int desc_idx,
    uint64_t address,
    int size,
    const void *smem_base,
    M2CQueue &m2c) {
  int slot = extract(cmd.slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
      "[%0], [%1, {%2}], [%3];\n"
      :
      : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
        "l"((void *)(tma_descs + desc_idx)),
        "r"((uint32_t)address),
        "r"((uint32_t)__cvta_generic_to_shared(
            m2c.native_bar(cmd.bar)))
      : "memory");
  cuda::device::barrier_expect_tx(
      m2c.barriers[cmd.bar],
      cuda::aligned_size_t<16>(size));
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_tma_load_2d(
    const CompiledLdCmd &cmd,
    const CUtensorMap *tma_descs,
    int desc_idx,
    int size,
    uint16_t c0,
    uint16_t c1,
    const void *smem_base,
    M2CQueue &m2c) {
  int slot = extract(cmd.slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      "[%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
        "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((uint32_t)__cvta_generic_to_shared(
            m2c.native_bar(cmd.bar)))
      : "memory");
  cuda::device::barrier_expect_tx(
      m2c.barriers[cmd.bar],
      cuda::aligned_size_t<16>(size));
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_tma_load_3d(
    const CompiledLdCmd &cmd,
    const CUtensorMap *tma_descs,
    int desc_idx,
    int size,
    uint16_t c0,
    uint16_t c1,
    uint16_t c2,
    const void *smem_base,
    M2CQueue &m2c) {
  int slot = extract(cmd.slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
      "[%0], [%1, {%2, %3, %4}], [%5];\n"
      :
      : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
        "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((int)c2),
        "r"((uint32_t)__cvta_generic_to_shared(
            m2c.native_bar(cmd.bar)))
      : "memory");
  cuda::device::barrier_expect_tx(
      m2c.barriers[cmd.bar],
      cuda::aligned_size_t<16>(size));
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_tma_load_4d(
    const CompiledLdCmd &cmd,
    const CUtensorMap *tma_descs,
    int desc_idx,
    int size,
    uint16_t c0,
    uint16_t c1,
    uint16_t c2,
    uint16_t c3,
    const void *smem_base,
    M2CQueue &m2c) {
  int slot = extract(cmd.slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
      :
      : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
        "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((int)c2),
        "r"((int)c3),
        "r"((uint32_t)__cvta_generic_to_shared(
            m2c.native_bar(cmd.bar)))
      : "memory");
  cuda::device::barrier_expect_tx(
      m2c.barriers[cmd.bar],
      cuda::aligned_size_t<16>(size));
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_tma_load_5d_fix0(
    const CompiledLdCmd &cmd,
    const CUtensorMap *tma_descs,
    int desc_idx,
    int size,
    uint16_t c0,
    uint16_t c1,
    uint16_t c2,
    uint16_t c3,
    const void *smem_base,
    M2CQueue &m2c) {
  int slot = extract(cmd.slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
      "[%0], [%1, {0, %2, %3, %4, %5}], [%6];\n"
      :
      : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
        "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((int)c2),
        "r"((int)c3),
        "r"((uint32_t)__cvta_generic_to_shared(
            m2c.native_bar(cmd.bar)))
      : "memory");
  cuda::device::barrier_expect_tx(
      m2c.barriers[cmd.bar],
      cuda::aligned_size_t<16>(size));
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_raw_address_ready(
    const CompiledLdCmd &cmd,
    uint64_t address,
    MInst *st_insts,
    M2CQueue &m2c) {
  st_insts[cmd.slot_mask].address = address;
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_reg_store_ready(
    const CompiledLdCmd &cmd,
    int reg_id,
    int reg_file[32],
    M2CQueue &m2c) {
  reg_file[reg_id] = cmd.slot_mask;
  m2c.data[cmd.bar] = cmd.slot_mask | 0x80000000U;
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename M2CQueue>
__device__ __forceinline__ void compiled_reg_load_ready(
    const CompiledLdCmd &cmd,
    int reg_id,
    int reg_file[32],
    M2CQueue &m2c) {
  (void)cmd;
  m2c.data[cmd.bar] = reg_file[reg_id];
  (void)m2c.barriers[cmd.bar].arrive();
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_store_1d(
    C2MQueue &c2m,
    uint64_t address,
    int size,
    const void *smem_base) {
  int slot_mask = c2m.pop();
  int slot = extract(slot_mask);
  cuda::ptx::cp_async_bulk(
      cuda::ptx::space_global,
      cuda::ptx::space_shared,
      reinterpret_cast<void *>(address),
      (const void *)(get_slot_address(smem_base, slot)),
      size);
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  c2m.reset(slot_mask);
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_raw_address_writeback(
    C2MQueue &c2m,
    int *bars,
    int bar_id) {
  int raw_slot = c2m.pop();
  (void)raw_slot;
  if (bar_id >= 0) {
    compiled_arrive_global_barrier(bars, bar_id);
  }
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_tma_store_2d(
    C2MQueue &c2m,
    const CUtensorMap *tma_descs,
    int desc_idx,
    uint16_t c0,
    uint16_t c1,
    const void *smem_base) {
  int slot_mask = c2m.pop();
  int slot = extract(slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
      "[%0, {%1, %2}], [%3];\n"
      :
      : "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
      : "memory");
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  c2m.reset(slot_mask);
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_tma_store_3d(
    C2MQueue &c2m,
    const CUtensorMap *tma_descs,
    int desc_idx,
    uint16_t c0,
    uint16_t c1,
    uint16_t c2,
    const void *smem_base) {
  int slot_mask = c2m.pop();
  int slot = extract(slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
      "[%0, {%1, %2, %3}], [%4];\n"
      :
      : "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((int)c2),
        "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
      : "memory");
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  c2m.reset(slot_mask);
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_tma_store_4d(
    C2MQueue &c2m,
    const CUtensorMap *tma_descs,
    int desc_idx,
    uint16_t c0,
    uint16_t c1,
    uint16_t c2,
    uint16_t c3,
    const void *smem_base) {
  int slot_mask = c2m.pop();
  int slot = extract(slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
      "[%0, {%1, %2, %3, %4}], [%5];\n"
      :
      : "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((int)c2),
        "r"((int)c3),
        "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
      : "memory");
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  c2m.reset(slot_mask);
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_tma_store_5d_fix0(
    C2MQueue &c2m,
    const CUtensorMap *tma_descs,
    int desc_idx,
    uint16_t c0,
    uint16_t c1,
    uint16_t c2,
    uint16_t c3,
    const void *smem_base) {
  int slot_mask = c2m.pop();
  int slot = extract(slot_mask);
  asm volatile(
      "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
      "[%0, {0, %1, %2, %3, %4}], [%5];\n"
      :
      : "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((int)c2),
        "r"((int)c3),
        "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
      : "memory");
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  c2m.reset(slot_mask);
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_tma_reduce_add_2d(
    C2MQueue &c2m,
    const CUtensorMap *tma_descs,
    int desc_idx,
    uint16_t c0,
    uint16_t c1,
    const void *smem_base) {
  int slot_mask = c2m.pop();
  int slot = extract(slot_mask);
  asm volatile(
      "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "
      "[%0, {%1, %2}], [%3];\n"
      :
      : "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
      : "memory");
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  c2m.reset(slot_mask);
}

template <typename C2MQueue>
__device__ __forceinline__ void compiled_tma_reduce_add_3d(
    C2MQueue &c2m,
    const CUtensorMap *tma_descs,
    int desc_idx,
    uint16_t c0,
    uint16_t c1,
    uint16_t c2,
    const void *smem_base) {
  int slot_mask = c2m.pop();
  int slot = extract(slot_mask);
  asm volatile(
      "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group "
      "[%0, {%1, %2, %3}], [%4];\n"
      :
      : "l"((void *)(tma_descs + desc_idx)),
        "r"((int)c0),
        "r"((int)c1),
        "r"((int)c2),
        "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
      : "memory");
  cuda::ptx::cp_async_bulk_commit_group();
  cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
  c2m.reset(slot_mask);
}
