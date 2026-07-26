#pragma once

#ifndef DAE_ENABLE_NVSHMEM
#error "commwarp.cuh requires DAE_ENABLE_NVSHMEM"
#endif

#include "ep_pool.cuh"
#include "memory_pool.cuh"
#include "pool_slice.cuh"
#include "virtualcore.cuh"

#include <nvshmem.h>
#include <nvshmemx.h>

static __device__ __forceinline__ void commwarp_complete(uint32_t lane) {
  __syncwarp();
  if (lane == 0)
    nvshmem_quiet();
  __syncwarp();
}

static __device__ __noinline__ void communicationwarp_execute(
    uint32_t lane,
    const CommInst* instructions,
    int* bars,
    uint64_t* signal_array,
    uint64_t* g_events) {
  uint32_t pc = 0;
  bool running = true;
  while (running) {
    const CommInst inst = instructions[pc++ % numCommInsts];
    switch (inst.opcode) {
      case COMM_TERMINATE:
        running = false;
        break;

      case COMM_WAIT_BARRIER:
        if (lane == 0) {
          volatile int* bar = bars + inst.size;
          while (*bar != 0)
            __nanosleep(barrierPollSleepCycles);
        }
        __syncwarp();
        break;

      case COMM_NVSHMEM_PUT: {
        const uint32_t nbytes = static_cast<uint32_t>(inst.size) |
            (static_cast<uint32_t>(inst.arg0) << 16);
        const uint32_t target_pe = inst.arg1 & 0x00ffU;
        const uint32_t signal_id = inst.arg1 >> 8;
        void* address = reinterpret_cast<void*>(inst.address);
        nvshmemx_putmem_signal_nbi_warp(
            address,
            address,
            nbytes,
            signal_array + signal_id,
            1,
            NVSHMEM_SIGNAL_SET,
            target_pe);
        commwarp_complete(lane);
        break;
      }

      case COMM_NVSHMEM_WAIT:
        if (lane == 0) {
          nvshmem_signal_wait_until(
              signal_array + inst.size,
              NVSHMEM_CMP_GE,
              inst.address);
        }
        __syncwarp();
        break;

      case COMM_MEMORY_POOL_SUBMIT: {
        const auto* request = reinterpret_cast<const MemoryPoolRequest*>(
            inst.address);
        nvshmemx_putmem_signal_nbi_warp(
            const_cast<MemoryPoolRequest*>(request),
            request,
            sizeof(MemoryPoolRequest),
            signal_array + inst.size,
            request->sequence,
            NVSHMEM_SIGNAL_SET,
            inst.arg0);
        commwarp_complete(lane);
        break;
      }

      case COMM_MEMORY_POOL_WAIT:
        if (lane == 0) {
          const auto* request = reinterpret_cast<const MemoryPoolRequest*>(
              inst.address);
          nvshmem_signal_wait_until(
              signal_array + request->completion_signal,
              NVSHMEM_CMP_GE,
              request->sequence);
        }
        __syncwarp();
        break;

      case COMM_MEMORY_POOL_RUN: {
        const uint32_t expected_requests = static_cast<uint32_t>(inst.size) |
            (static_cast<uint32_t>(inst.arg0) << 16);
        memory_pool_run_warp(
            reinterpret_cast<const MemoryPoolConfig*>(inst.address),
            signal_array,
            expected_requests,
            lane);
        __syncwarp();
        break;
      }

      case COMM_EXPERT_POOL_RESET:
        ep_pool_reset(
            *reinterpret_cast<const EpPoolConfig*>(inst.address),
            bars,
            signal_array,
            inst.arg0,
            lane);
        break;

      case COMM_EXPERT_POOL_DISPATCH:
        ep_pool_dispatch_expert(
            *reinterpret_cast<const EpPoolConfig*>(inst.address),
            bars,
            signal_array,
            inst.size,
            inst.arg0,
            lane);
        break;

      case COMM_EXPERT_POOL_RETURN:
        ep_pool_return_expert(
            *reinterpret_cast<const EpPoolConfig*>(inst.address),
            bars,
            signal_array,
            inst.size,
            inst.arg0,
            lane);
        break;

      case COMM_POOL_SLICE_PUBLISH:
        pool_slice_publish(
            reinterpret_cast<const PoolSliceConfig*>(inst.address),
            signal_array,
            lane);
        break;

      case COMM_POOL_SLICE_GATHER: {
        const auto* config = reinterpret_cast<const PoolSliceConfig*>(
            inst.address);
        if (config != nullptr &&
            (config->flags & POOL_SLICE_FLAGS_STREAMING_GATHER) != 0) {
          pool_slice_gather_streaming(
              config,
              bars,
              signal_array,
              g_events,
              inst.size,
              inst.arg0,
              lane);
        } else {
          pool_slice_gather(
              config,
              bars,
              signal_array,
              inst.size,
              inst.arg0,
              lane);
        }
        break;
      }

      case COMM_POOL_SLICE_RETURN:
        pool_slice_return(
            reinterpret_cast<const PoolSliceConfig*>(inst.address),
            bars,
            signal_array,
            inst.size,
            lane);
        break;

      case COMM_RECORD_EVENT:
        if (lane == 0) {
          g_events[
              static_cast<uint64_t>(blockIdx.x) * numProfileEvents + inst.size] =
              cuda::ptx::get_sreg_globaltimer();
        }
        __syncwarp();
        break;

      default:
        if (lane == 0)
          asm volatile("trap;");
        running = false;
        break;
    }
  }
}
