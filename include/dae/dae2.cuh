#pragma once

#include "virtualcore.cuh"

#include "allocator.cuh"
#include "queue.cuh"

#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <bit>

// pipeline stages
#include "pipeline/allocwarp.cuh"
#include "pipeline/ldwarp.cuh"
#include "pipeline/stwarp.cuh"

// tasks
#include "task/gemv.cuh"
#include "task/wgmma.cuh"
#include "task/rms_norm.cuh"
#include "task/attention.cuh"
#include "task/silu.cuh"
#include "task/argmax.cuh"

static __device__ __forceinline__ void * align_to(void *ptr, size_t align) {
  uintptr_t addr = (uintptr_t)ptr;
  uintptr_t aligned = (addr + align - 1) & ~(align - 1);
  return (void*)aligned;
}

// TODO(zhiyuang): decide this maxnreg size.
// Also with setnreg for computation and memory separately?
static __global__
void dae2(
  const CInst* __restrict__ compute_instructions,
  const MInst* __restrict__ memory_instructions,
  const CUtensorMap* __restrict__ tma_descs,
  int * __restrict__ bars,
  uint64_t *  __restrict__ g_events
) {

  int sm_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int warp_id = (thread_id % 128) / 32;
  int lane_id = thread_id % 32;


  __kprint("[DAE2 SM %d] Kernel launched with %d threads (%d warps)\n", sm_id, blockDim.x, blockDim.x / 32);


  const CInst* __restrict__ cinsts;
  const MInst* __restrict__ minsts;

  // local datastructures
  if constexpr (dae2LoadInstructions) {
    __shared__ CInst smem_cinsts[numInsts];
    __shared__ MInst smem_minsts[numInsts];

    for (int i = thread_id; i < numInsts; i += blockDim.x) {
      smem_cinsts[i] = compute_instructions[sm_id * numInsts + i];
      smem_minsts[i] = memory_instructions[sm_id * numInsts + i];
    }

    cinsts = smem_cinsts;
    minsts = smem_minsts;
  } else {
    cinsts = compute_instructions + sm_id * numInsts;
    minsts = memory_instructions + sm_id * numInsts;
  }

  // intermidate insts
  constexpr int numQueueElements = 32;
  __shared__ MInst st_insts[numSlots + numSpecialSlots]; // we can have some special slots that don't go through the allocator, for special purposes like reduction output, argmax output, etc. these are indexed from numSlots and above.

  // allocator
  // TODO(zhiyuang): align this to lane 31 to avoid bank conflict?
  __shared__ int slot_avail;
  if (thread_id == 0)
    slot_avail = (1U << numSlots) - 1; // all slots are available at the beginning. each bit represents a slot. 1 means available, 0 means occupied.

  // Init the queues
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ cuda::barrier<cuda::thread_scope_block> barriers[4][numQueueElements];
  assert(numQueueElements <= blockDim.x && "Too many slots for barriers");
  if (threadIdx.x < numQueueElements) {
    init(&barriers[0][threadIdx.x], numThreadsM2CBarrier);
    init(&barriers[1][threadIdx.x], numThreadsC2MBarrier);
    init(&barriers[2][threadIdx.x], numThreadsLDBarrier);
    init(&barriers[3][threadIdx.x], numThreadsLDBarrier);
  }

  __shared__ int m2c_data[numQueueElements];
  __shared__ int c2m_data[numQueueElements];
  __shared__ int m2ld_data[2][numQueueElements];

  SizeBoundedBarrierQueue<int, numQueueElements> m2c {
    .barriers = barriers[0], .data = m2c_data, .ptr = 0
  };
  SizeBoundedBarrierAllocQueue<numQueueElements> c2m {
    barriers[1], c2m_data, 0, &slot_avail
  };
  SizeBoundedBarrierQueue<int, numQueueElements> m2ld[2] = {
    { .barriers = barriers[2], .data = m2ld_data[0], .ptr = 0 },
    { .barriers = barriers[3], .data = m2ld_data[1], .ptr = 0 }
  };

  // init the slots
  extern __shared__ uint8_t shared_mem[];
  void * smem_base = align_to((void*)shared_mem, 1024); // align to 1KB

  // alloc a small scratch space for temporary data
  // argmax uses this
  __shared__ uint64_t scratch_space[32]; // 8-bytes aligned

  if (threadIdx.x == 0) {
    int event_base = sm_id * numProfileEvents;
    g_events[event_base + 0] = cuda::ptx::get_sreg_globaltimer();
  }

  __syncthreads();

  // start memory and computation execution
  if (threadIdx.x < numComputeWarps * 32) {
    CInst inst;
    uint32_t pc = 0;
    uint32_t count = 0;
    bool finish = false;

    while (!finish) {
      inst = cinsts[(pc++) % numInsts];
    
      __cprint("Executing instruction at PC %d: opcode=%04x", pc - 1, inst.opcode);

      switch (inst.opcode) {
        case OP_DUMMY:
          for (int i = 0; i < inst.args[0]; i++) {
            __cprint("[Dummy][i=%d] before wait", i);
            auto slot_id = m2c.pop();
            __nanosleep(inst.args[1]); // simulate some compute
            __cprint("[Dummy][i=%d] after pop slot_id=%d", i, slot_id);
            c2m.push<0>(thread_id, slot_id);
          }
          break;
        // TODO(zhiyuang): consider a "link" style implementation of COPY
        case OP_COPY: {
          for (int i = 0; i < inst.args[0]; i++) {
            __cprint("[Copy][i=%d] before wait", i);
            auto read_slot = m2c.pop();
            uint32_t *read_data = (uint32_t*)get_slot_address(smem_base, extract(read_slot));
            auto write_slot = m2c.pop();
            uint32_t *write_data = (uint32_t*)get_slot_address(smem_base, extract(write_slot));

            __cprint("[Copy][i=%d] after pop read_slot=%d, write_slot=%d", i, read_slot, write_slot);
            for (int i = thread_id; i < inst.args[1]; i += 128)
              write_data[i] = read_data[i];

            c2m.push<0, true>(thread_id, write_slot);
            c2m.push(thread_id, read_slot); // push the read slot back to indicate it's done and can be reused
          }
          break;
        }
        case OP_GEMV_M64N8: {
          using gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
          task_gemv<gemv_atom, 64, 256, 4, false>(inst.args[0], inst.args[1], smem_base, m2c, c2m);
          }
          break;
        case OP_GEMV_M64N8K64: {
          using gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
          task_gemv<gemv_atom, 64, 64, 1, false>(inst.args[0], inst.args[1], smem_base, m2c, c2m);
          }
          break;
        case OP_GEMV_M64N8_MMA: {
          task_gemv_mma<64, 8, 256>(inst.args[0], smem_base, m2c, c2m);
          }
          break;
        // case OP_GEMV_M128N8: {
        //   using gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
        //   task_gemv<gemv_atom, 128, 128, 4, false>(inst.args[0], inst.args[1], smem_base, m2c, c2m);
        //   }
        //   break;
        case OP_GEMM_M64N64: {
          using gemm_atom = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
          task_gemm<gemm_atom, 64, 64, 128, 1, false>(inst.args[0], smem_base, m2c, c2m);
          }
          break;
        case OP_GEMM_M64N128K64: {
          using gemm_atom = cute::SM90_64x128x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
          task_gemm<gemm_atom, 64, 128, 64, 1, false>(inst.args[0], smem_base, m2c, c2m);
          }
          break;
        case OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim: {
          using kernel_QK = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
          using kernel_PV = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
          const bool need_norm = inst.args[2] & 0x1;
          const bool need_rope = inst.args[2] & 0x2;
          task_attention_fwd_flash3_grouped<128, 64, 64, false, 0, false, false, kernel_QK, kernel_PV>(inst.args[0], 0, 64, inst.args[1], 0, need_norm, need_rope, smem_base, (float*)scratch_space, st_insts, m2c, c2m);
        }
          break;
        case OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split: {
          using kernel_QK = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
          using kernel_PV = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
          const int num_kv_blocks = inst.args[0] & 0xFF;
          const int split_idx = (inst.args[0] >> 8) & 0xFF;
          const int num_active_q = inst.args[1] & 0xFF;
          const int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
          const int kv_start_idx = inst.args[2];
          task_attention_fwd_flash3_grouped<128, 64, 64, true, 16, false, false, kernel_QK, kernel_PV>(num_kv_blocks, split_idx, num_active_q, last_kv_active_token_len, kv_start_idx, false, false, smem_base, (float*)scratch_space, st_insts, m2c, c2m);
        }
          break;
        case OP_ATTN_SPLIT_POST_REDUCE: {
          task_split_post_reduce<128, 4, 64, 16, 32>(inst.args[0], smem_base, (float*)scratch_space, st_insts, m2c, c2m);
        }
          break;
        case OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64: {
          using kernel_QK = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
          using kernel_PV = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
          const bool need_norm = inst.args[2] & 0x1;
          const bool need_rope = inst.args[2] & 0x2;
          task_attention_fwd_flash3_grouped<64, 64, 64, false, 0, false, false, kernel_QK, kernel_PV>(inst.args[0], 0, 64, inst.args[1], 0, need_norm, need_rope, smem_base, (float*)scratch_space, st_insts, m2c, c2m);
        }
          break;
        case OP_SILU_MUL_SHARED_BF16_K_4096_INTER: {
          const int num_token = inst.args[0];
          task_silu_smem_1D<6144>(num_token, smem_base, m2c, c2m);
          }
          break;
        case OP_SILU_MUL_SHARED_BF16_K_64_SW128: {
          const int num_token = inst.args[0];
          auto layout_sV = tile_to_shape(
              GMMA::Layout_MN_SW128_Atom<__nv_bfloat162>{},
              make_shape(Int<32>{},num_token));
          task_silu_smem<64>(num_token, layout_sV, smem_base, m2c, c2m);
          }
          break;
        case OP_RMS_NORM_F16_K_4096_SMEM: {
          task_rms_norm_f16_from_smem<4096,__nv_bfloat16>(smem_base, inst.args[0], *reinterpret_cast<__nv_bfloat16*>(inst.args+1), (float*)scratch_space, m2c, c2m);
          }
          break;
        case OP_RMS_NORM_F16_K_2048_SMEM: {
          task_rms_norm_f16_from_smem<2048,__nv_bfloat16>(smem_base, inst.args[0], *reinterpret_cast<__nv_bfloat16*>(inst.args+1), (float*)scratch_space, m2c, c2m);
          }
          break;
        case OP_RMS_NORM_F16_K_128_SMEM: {
          task_rms_norm_f16_from_smem<128,__nv_bfloat16>(smem_base, inst.args[0], *reinterpret_cast<__nv_bfloat16*>(inst.args[1]), (float*)scratch_space, m2c, c2m);
          }
          break;
        case OP_ARGMAX_PARTIAL_bf16_1152_50688_132: {
          task_argmax_partial<1152, 50688, 132, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
          }
          break;
        case OP_ARGMAX_REDUCE_bf16_1152_132: {
          task_argmax_reduce_kernel<1152, 132, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
          }
          break;
        case OP_ARGMAX_PARTIAL_bf16_1024_65536_128: {
          task_argmax_partial<1024, 65536, 128, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
          }
          break;
        case OP_ARGMAX_REDUCE_bf16_1024_128: {
          task_argmax_reduce_kernel<1024, 128, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
          }
          break;
        case OP_ROPE_INTERLEAVE_512: {
          task_rope_interleaved<512>(smem_base, m2c, c2m);
          break;
        }
        case OP_LOOPC: {
          if (++count < inst.args[0]) {
            pc = inst.args[1]; // jump back to the beginning of the loop
            __cprint("LOOPC back to PC %d, count=%d", pc, count);
          } else {
            count = 0; // reset count for potential future loops
            __cprint("LOOPC finished, count=%d", count);
          }
          // TODO(zhiyuang): this compute group is hardcoded
          __sync_compute_group(128);
          break;
        }
        case OP_TERMINATEC: {
          finish = true;

          // send the terminaion to st warp
          c2m.push<0,true>(thread_id, 0);

          if (thread_id == 0) {
            int event_base = sm_id * numProfileEvents;
            g_events[event_base + 1] = cuda::ptx::get_sreg_globaltimer();
          }
          __cprint("TERMINATE from comptue: c2m.ptr=%d", c2m.ptr);
        }
        break;
        default:
          // unknown opcode
          __cprint("Unknown compute opcode: %d\n", inst.opcode);
          assert(false && "Unknown compute opcode");
      }
      // if (blockIdx.x == 0 && threadIdx.x == 0) {
      //   printf("[COMP] after execution: pc=%d, opcode=%04x\n", pc-1, inst.opcode);
      // }
    }
    __cprint("Finished execution pc=%d", pc-1);
  } else { // memory warp group
    // TODO(zhiyuang): reduce the register usage in memory warps
    // cuda::ptx::set_max_nreg();

    // TODO(zhiyuang): change this to threadIdx.x predicates. will be faster than lane_id based?
    if (warp_id == 0) {
      allocwarp_execute(
        lane_id,
        m2c, m2ld, minsts, &slot_avail,
        st_insts, smem_base, tma_descs, bars
      );
    } else if (warp_id == 1) {
      if (lane_id == 0) {
        stwarp_execute_singlethread(
          c2m, st_insts,
          smem_base, tma_descs, bars
        );
      }
    } else if (warp_id >= 2) { // LD Warps 0-1
      if (lane_id == 0) {
        int port_id = warp_id - 2;
        ldwarp_execute_singlethread(
          m2ld[port_id], m2c,
          st_insts,
          smem_base, tma_descs, bars
        );
      }
    } // End of warps
  } // End of memory warp group

  // end of megakernel
}
