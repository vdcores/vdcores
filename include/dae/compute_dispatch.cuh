#pragma once

#include "context.cuh"

#include "task/argmax.cuh"
#include "task/attention.cuh"
#include "task/deepseek_v4.cuh"
#include "task/fp8.cuh"
#include "task/gemv.cuh"
#include "task/nvfp4.cuh"
#include "task/nvfp4_umma.cuh"
#include "task/rms_norm.cuh"
#include "task/silu.cuh"
#include "task/wgmma.cuh"

#include <type_traits>

#define DAE_COMPUTE_OP_PARAMS \
  int sm_id, \
  int thread_id, \
  uint32_t &pc, \
  uint32_t *count, \
  bool &finish, \
  const CInst &inst, \
  void *smem_base, \
  uint32_t tmem_base_ptr, \
  uint64_t *tmem_mma_barrier, \
  uint32_t &tmem_mma_phase, \
  uint64_t *scratch_space, \
  MInst *st_insts, \
  M2CQueue &m2c, \
  C2MQueue &c2m, \
  uint64_t *g_events

#define DAE_COMPUTE_HANDLER_NAME(opname) dae_compute_handler_##opname

#define DAE_COMPUTE_OP_HANDLER(opname) \
  template <typename M2CQueue, typename C2MQueue> \
  static __device__ __forceinline__ void DAE_COMPUTE_HANDLER_NAME(opname)(DAE_COMPUTE_OP_PARAMS)

template <typename... Args>
static __device__ __forceinline__ void dae_ignore(Args const &...) {}

#define DAE_UNUSED(...) dae_ignore(__VA_ARGS__)

DAE_COMPUTE_OP_HANDLER(OP_DUMMY) {
  DAE_UNUSED(sm_id, pc, count, finish, smem_base, scratch_space, st_insts, g_events);
  for (int i = 0; i < inst.args[0]; i++) {
    __cprint("[Dummy][i=%d] before wait", i);
    auto slot_id = m2c.pop();
    __nanosleep(inst.args[1]);
    __cprint("[Dummy][i=%d] after pop slot_id=%d", i, slot_id);
    c2m.template push<0>(thread_id, slot_id);
  }
}

DAE_COMPUTE_OP_HANDLER(OP_COPY) {
  DAE_UNUSED(sm_id, pc, count, finish, scratch_space, st_insts, g_events);
  for (int i = 0; i < inst.args[0]; i++) {
    __cprint("[Copy][i=%d] before wait", i);
    auto read_slot = m2c.pop();
    uint32_t *read_data = (uint32_t *)get_slot_address(smem_base, extract(read_slot));
    auto write_slot = m2c.pop();
    uint32_t *write_data = (uint32_t *)get_slot_address(smem_base, extract(write_slot));

    __cprint("[Copy][i=%d] after pop read_slot=%d, write_slot=%d", i, read_slot, write_slot);
    for (int j = thread_id; j < inst.args[1]; j += 128) {
      write_data[j] = read_data[j];
    }

    c2m.template push<0, true>(thread_id, write_slot);
    c2m.push(thread_id, read_slot);
  }
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_M64N8_MMA) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
  task_gemv_mma<64, 8, 256>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_M64N8_ROPE_128) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_rope<64, 8, 256, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m, st_insts, inst.args[2]);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_M128N8_ROPE_128) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_rope<128, 8, 128, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m, st_insts, inst.args[2]);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M64N8_UP_SILU) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_up_silu_sm100<64, 8, 256, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M64N8_ISSUER_ONLY) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_issuer_only<64, 8, 256, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_NVFP4_GEMV_SM100) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, smem_base, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_nvfp4_gemv_sm100(inst.args[0], inst.args[1], st_insts, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_NVFP4_GEMV_UMMA_SM100) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_nvfp4_gemv_umma_sm100(
      inst.args[0], inst.args[1], inst.args[2], smem_base, tmem_base_ptr,
      tmem_mma_barrier, tmem_mma_phase, st_insts, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_FP8_BLOCK128_GEMV_SM100) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, smem_base, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_fp8_block128_gemv_sm100(inst.args[0], inst.args[1], inst.args[2],
                               st_insts, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_ROPE_512_64) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, smem_base, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_rope_64<512>(
      inst.args[0], inst.args[1] != 0, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_ROPE_128_64) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, smem_base, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_rope_64<128>(
      inst.args[0], inst.args[1] != 0, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_SPARSE_ATTENTION_512) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_sparse_attention_512(
      inst.args[0], inst.args[1], smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_ROUTE_TOP6) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_route_top6(
      inst.args[0] != 0,
      __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1)),
      smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_EXPERT_REDUCE) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, inst, smem_base,
             tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase, scratch_space,
             g_events);
  task_dsv4_expert_reduce(st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_FP32_BF16_GEMV) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_fp32_bf16_gemv(
      inst.args[0], inst.args[1], smem_base,
      st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_BF16_GEMV) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_bf16_gemv(
      inst.args[0], inst.args[1], smem_base,
      st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_HC_PRE) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_hc_pre(
      inst.args[0],
      __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1)),
      smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_HC_POST) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, inst, smem_base,
             tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase, scratch_space,
             g_events);
  task_dsv4_hc_post(st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_SILU_CLAMP_MUL_2048) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts,
             tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase, g_events);
  task_silu_clamp_smem_1D<2048>(
      inst.args[0],
      __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1)),
      smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_HADAMARD) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_hadamard(
      inst.args[0], inst.args[1], smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_GATED_POOL) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, smem_base, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_gated_pool(
      inst.args[0], inst.args[1], st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_INDEX_SCORE) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_index_score(
      inst.args[0], smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_TOPK_512) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_topk512(
      inst.args[0], inst.args[1], inst.args[2], smem_base,
      st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_HC_HEAD) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_hc_head(
      __bfloat162float(*reinterpret_cast<const __nv_bfloat16 *>(inst.args)),
      smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_FP8_QUANT_128) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_fp8_quant128(
      inst.args[0], smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_DSV4_NVFP4_QUANT_16) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, g_events);
  task_dsv4_nvfp4_quant16(
      inst.args[0], smem_base, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M128N8_DIRECT4) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_direct_grouped<128, 8, 128, 4, 4, false>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m, st_insts, inst.args[1] * 128,
      inst.args[2] * 128);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M128N8_ARGMAX4) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_direct_grouped<128, 8, 128, 4, 4, true>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m, st_insts, 0, inst.args[1] * 128,
      (void *)scratch_space, inst.args[2] * 128, 256);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M128N8_GROUP4_B2) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_grouped_reduce<128, 8, 128, 2, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M128N8_GROUP4_B3) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_grouped_reduce<128, 8, 128, 3, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M128N8_GROUP4_B4) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_grouped_reduce<128, 8, 128, 4, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMV_SM100_M128N8_GROUP4_B7) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  task_gemv_sm100_grouped_reduce<128, 8, 128, 7, 4>(
      inst.args[0], tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_GEMM_M64N64) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
  using gemm_atom = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemm<gemm_atom, 64, 64, 128, 1, false>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_GEMM_M64N64K64) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
  using gemm_atom = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemm<gemm_atom, 64, 64, 64, 1, false>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_GEMM_M64N128K64) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
  using gemm_atom = cute::SM90_64x128x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemm<gemm_atom, 64, 128, 64, 1, false>(inst.args[0], smem_base, m2c, c2m);
}

template <int HeadDim, bool SplitKv, typename KernelQK, typename KernelPV, typename M2CQueue, typename C2MQueue>
static __device__ __forceinline__ void handle_attention_common(
  const CInst &inst,
  const uint32_t *count,
  void *smem_base,
  uint64_t *scratch_space,
  MInst *st_insts,
  M2CQueue &m2c,
  C2MQueue &c2m
) {
  if constexpr (SplitKv) {
    const int num_kv_blocks = inst.args[0] & 0xFFF;
    const int split_idx = (inst.args[0] >> 12) & 0xF;
    const int num_active_q = inst.args[1] & 0xFF;
    const int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
    const int kv_start_idx = inst.args[2];
    if constexpr (std::is_same_v<KernelQK, cute::SM80_16x8x16_F32BF16BF16F32_TN>) {
      task_attention_fwd_flash3_grouped_mma<HeadDim, 64, 64, true, 16, false, false, KernelQK, KernelPV>(
        num_kv_blocks,
        split_idx,
        num_active_q,
        last_kv_active_token_len,
        kv_start_idx,
        false,
        false,
        smem_base,
        (float *)scratch_space,
        st_insts,
        m2c,
        c2m
      );
    } else {
      task_attention_fwd_flash3_grouped<HeadDim, 64, 64, true, 16, false, false, KernelQK, KernelPV>(
        num_kv_blocks,
        split_idx,
        num_active_q,
        last_kv_active_token_len,
        kv_start_idx,
        false,
        false,
        smem_base,
        (float *)scratch_space,
        st_insts,
        m2c,
        c2m
      );
    }
    return;
  }

  const int num_active_q = inst.args[1] & 0xFF;
  int num_kv_blocks = inst.args[0] & 0xFF;
  const int outer_seq_stride = (inst.args[0] >> 8) & 0xFF;
  int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  const bool need_norm = inst.args[2] & 0x1;
  const bool need_rope = inst.args[2] & 0x2;
  if (inst.args[2] & 0x8) {
    const int counter_reg = (inst.args[2] >> 4) & 0xF;
    num_kv_blocks += count[counter_reg];
  }
  if (outer_seq_stride > 0) {
    const int counter_reg = (inst.args[2] >> 12) & 0xF;
    last_kv_active_token_len += count[counter_reg] * outer_seq_stride;
    last_kv_active_token_len = (last_kv_active_token_len - 1) % 64 + 1;
  }
  if (inst.args[2] & 0x4) {
    const int counter_reg = (inst.args[2] >> 8) & 0xF;
    last_kv_active_token_len += count[counter_reg];
  }
  if constexpr (std::is_same_v<KernelQK, cute::SM80_16x8x16_F32BF16BF16F32_TN>) {
    task_attention_fwd_flash3_grouped_mma<HeadDim, 64, 64, false, 0, false, false, KernelQK, KernelPV>(
      num_kv_blocks,
      0,
      num_active_q,
      last_kv_active_token_len,
      0,
      need_norm,
      need_rope,
      smem_base,
      (float *)scratch_space,
      st_insts,
      m2c,
      c2m
    );
  } else {
    task_attention_fwd_flash3_grouped<HeadDim, 64, 64, false, 0, false, false, KernelQK, KernelPV>(
      num_kv_blocks,
      0,
      num_active_q,
      last_kv_active_token_len,
      0,
      need_norm,
      need_rope,
      smem_base,
      (float *)scratch_space,
      st_insts,
      m2c,
      c2m
    );
  }
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  const int encoded_active_q = inst.args[1] & 0xFF;
  const int num_active_q = encoded_active_q & 0x7F;
  const bool use_kv128 = encoded_active_q & 0x80;
  int num_kv_blocks = inst.args[0] & 0xFF;
  const int outer_seq_stride = (inst.args[0] >> 8) & 0xFF;
  int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  const bool need_norm = inst.args[2] & 0x1;
  const bool need_rope = inst.args[2] & 0x2;
  if (inst.args[2] & 0x8) {
    const int counter_reg = (inst.args[2] >> 4) & 0xF;
    num_kv_blocks += count[counter_reg];
  }
  if (outer_seq_stride > 0) {
    const int counter_reg = (inst.args[2] >> 12) & 0xF;
    const int kv_block_size = use_kv128 ? 128 : 64;
    last_kv_active_token_len += count[counter_reg] * outer_seq_stride;
    last_kv_active_token_len = (last_kv_active_token_len - 1) % kv_block_size + 1;
  }
  if (inst.args[2] & 0x4) {
    const int counter_reg = (inst.args[2] >> 8) & 0xF;
    last_kv_active_token_len += count[counter_reg];
  }
  if (use_kv128) {
    task_attention_fwd_sm100_decode<128, 128>(
      num_kv_blocks, 0, num_active_q, last_kv_active_token_len,
      need_norm, need_rope, tmem_base_ptr, tmem_mma_barrier,
      tmem_mma_phase, smem_base, (float *)scratch_space, st_insts,
      m2c, c2m);
  } else {
    task_attention_fwd_sm100_decode<128, 64>(
      num_kv_blocks, 0, num_active_q, last_kv_active_token_len,
      need_norm, need_rope, tmem_base_ptr, tmem_mma_barrier,
      tmem_mma_phase, smem_base, (float *)scratch_space, st_insts,
      m2c, c2m);
  }
#else
  using kernel_qk = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using kernel_pv = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
  handle_attention_common<128, false, kernel_qk, kernel_pv>(inst, count, smem_base, scratch_space, st_insts, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_SM100_BF16_HDIM128_DIRECT) {
  DAE_UNUSED(sm_id, thread_id, pc, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  const int encoded_active_q = inst.args[1] & 0xFF;
  const int num_active_q = encoded_active_q & 0x7F;
  const bool use_kv128 = encoded_active_q & 0x80;
  int num_kv_blocks = inst.args[0] & 0xFF;
  const int outer_seq_stride = (inst.args[0] >> 8) & 0xFF;
  int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  if (inst.args[2] & 0x8) {
    const int counter_reg = (inst.args[2] >> 4) & 0xF;
    num_kv_blocks += count[counter_reg];
  }
  if (outer_seq_stride > 0) {
    const int counter_reg = (inst.args[2] >> 12) & 0xF;
    const int kv_block_size = use_kv128 ? 128 : 64;
    last_kv_active_token_len += count[counter_reg] * outer_seq_stride;
    last_kv_active_token_len =
        (last_kv_active_token_len - 1) % kv_block_size + 1;
  }
  if (inst.args[2] & 0x4) {
    const int counter_reg = (inst.args[2] >> 8) & 0xF;
    last_kv_active_token_len += count[counter_reg];
  }
  if (use_kv128) {
    task_attention_fwd_sm100_decode<128, 128, false, 16, true, true>(
      num_kv_blocks, 0, num_active_q, last_kv_active_token_len,
      false, false, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  } else {
    task_attention_fwd_sm100_decode<128, 64, false, 16, true, true>(
      num_kv_blocks, 0, num_active_q, last_kv_active_token_len,
      false, false, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  }
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_SM100_BF16_HDIM128_SPLIT_DIRECT) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  const int num_kv_blocks = inst.args[0] & 0xFFF;
  const int split_idx = (inst.args[0] >> 12) & 0xF;
  const int num_active_q = inst.args[1] & 0x7F;
  const bool use_kv128 = inst.args[1] & 0x80;
  const int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  if (use_kv128) {
    task_attention_fwd_sm100_decode<128, 128, true, 16, true, true>(
      num_kv_blocks, split_idx, num_active_q, last_kv_active_token_len,
      false, false, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  } else {
    task_attention_fwd_sm100_decode<128, 64, true, 16, true, true>(
      num_kv_blocks, split_idx, num_active_q, last_kv_active_token_len,
      false, false, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  }
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT) {
  DAE_UNUSED(sm_id, thread_id, pc, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  const int encoded_active_q = inst.args[1] & 0xFF;
  const int num_active_q = encoded_active_q & 0x7F;
  const bool use_kv128 = encoded_active_q & 0x80;
  int num_kv_blocks = inst.args[0] & 0xFF;
  const int outer_seq_stride = (inst.args[0] >> 8) & 0xFF;
  int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  if (inst.args[2] & 0x8) {
    const int counter_reg = (inst.args[2] >> 4) & 0xF;
    num_kv_blocks += count[counter_reg];
  }
  if (outer_seq_stride > 0) {
    const int counter_reg = (inst.args[2] >> 12) & 0xF;
    last_kv_active_token_len += count[counter_reg] * outer_seq_stride;
    last_kv_active_token_len =
        (last_kv_active_token_len - 1) % 128 + 1;
  }
  if (inst.args[2] & 0x4) {
    const int counter_reg = (inst.args[2] >> 8) & 0xF;
    last_kv_active_token_len += count[counter_reg];
  }
  if (use_kv128) {
    task_attention_fwd_sm100_decode_swap<false, 16>(
      num_kv_blocks, 0, num_active_q, last_kv_active_token_len,
      tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  }
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  const int num_kv_blocks = inst.args[0] & 0xFFF;
  const int split_idx = (inst.args[0] >> 12) & 0xF;
  const int num_active_q = inst.args[1] & 0x7F;
  const bool use_kv128 = inst.args[1] & 0x80;
  const int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  if (use_kv128) {
    task_attention_fwd_sm100_decode_swap<true, 16>(
      num_kv_blocks, split_idx, num_active_q, last_kv_active_token_len,
      tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  }
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
  const int num_kv_blocks = inst.args[0] & 0xFFF;
  const int split_idx = (inst.args[0] >> 12) & 0xF;
  const int encoded_active_q = inst.args[1] & 0xFF;
  const int num_active_q = encoded_active_q & 0x7F;
  const bool use_kv128 = encoded_active_q & 0x80;
  const int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  if (use_kv128) {
    task_attention_fwd_sm100_decode<128, 128, true, 16>(
      num_kv_blocks, split_idx, num_active_q, last_kv_active_token_len,
      false, false, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  } else {
    task_attention_fwd_sm100_decode<128, 64, true, 16>(
      num_kv_blocks, split_idx, num_active_q, last_kv_active_token_len,
      false, false, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
      smem_base, (float *)scratch_space, st_insts, m2c, c2m);
  }
#else
  using kernel_qk = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using kernel_pv = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
  handle_attention_common<128, true, kernel_qk, kernel_pv>(inst, count, smem_base, scratch_space, st_insts, m2c, c2m);
#endif
}

DAE_COMPUTE_OP_HANDLER(OP_ATTN_SPLIT_POST_REDUCE) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  if ((inst.args[1] & 0x3) == 0x3 && inst.args[0] == 2) {
    task_split_post_reduce2_raw_direct<128, 4, 16>(
      smem_base, st_insts, m2c, c2m);
  } else if (inst.args[1] & 0x2) {
    task_split_post_reduce<128, 4, 64, 16, 32, true, true>(
      inst.args[0], smem_base, (float *)scratch_space,
      st_insts, m2c, c2m);
  } else if (inst.args[1] & 0x1) {
    task_split_post_reduce<128, 4, 64, 16, 32, true, false>(
      inst.args[0], smem_base, (float *)scratch_space,
      st_insts, m2c, c2m);
  } else {
    task_split_post_reduce<128, 4, 64, 16, 32, false, false>(
      inst.args[0], smem_base, (float *)scratch_space,
      st_insts, m2c, c2m);
  }
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  using kernel_qk = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using kernel_pv = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
  handle_attention_common<64, false, kernel_qk, kernel_pv>(inst, count, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  using kernel_qk = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  using kernel_pv = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  handle_attention_common<128, false, kernel_qk, kernel_pv>(inst, count, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  using kernel_qk = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  using kernel_pv = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  handle_attention_common<128, true, kernel_qk, kernel_pv>(inst, count, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  using kernel_qk = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  using kernel_pv = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  const int num_active_q = inst.args[1] & 0xFF;
  int num_kv_blocks = inst.args[0];
  int last_kv_active_token_len = (inst.args[1] >> 8) & 0xFF;
  const bool need_norm = inst.args[2] & 0x1;
  const bool need_rope = inst.args[2] & 0x2;
  if (inst.args[2] & 0x8) {
    const int counter_reg = (inst.args[2] >> 4) & 0xF;
    num_kv_blocks += count[counter_reg];
  }
  if (inst.args[2] & 0x4) {
    const int counter_reg = (inst.args[2] >> 8) & 0xFF;
    last_kv_active_token_len += count[counter_reg];
  }
  task_attention_fwd_flash3_grouped_mma<64, 64, 16, false, 0, false, false, kernel_qk, kernel_pv>(
    num_kv_blocks,
    0,
    num_active_q,
    last_kv_active_token_len,
    0,
    need_norm,
    need_rope,
    smem_base,
    (float *)scratch_space,
    st_insts,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP_HANDLER(OP_SILU_MUL_SHARED_BF16_K_4096_INTER) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
  task_silu_smem_1D<6144>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_SILU_MUL_SHARED_BF16_K_2048_INTER) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
  task_silu_smem_1D<2048>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_SILU_MUL_SHARED_BF16_K_64_SW128) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, scratch_space, st_insts, g_events);
  const int num_token = inst.args[0];
  auto layout_sv = tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<__nv_bfloat162>{},
    make_shape(Int<32>{}, num_token)
  );
  task_silu_smem<64>(num_token, layout_sv, smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_RMS_NORM_F16_K_4096_SMEM) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, st_insts, g_events);
  task_rms_norm_f16_from_smem<4096, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP_HANDLER(OP_RMS_NORM_F16_K_2048_SMEM) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, st_insts, g_events);
  task_rms_norm_f16_from_smem<2048, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP_HANDLER(OP_RMS_NORM_F16_K_512_SMEM) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, st_insts, g_events);
  task_rms_norm_f16_from_smem<512, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP_HANDLER(OP_RMS_NORM_F16_K_1024_SMEM) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, st_insts, g_events);
  task_rms_norm_f16_from_smem<1024, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP_HANDLER(OP_RMS_NORM_F16_K_5120_SMEM) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, st_insts, g_events);
  task_rms_norm_f16_from_smem<5120, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP_HANDLER(OP_RMS_NORM_F16_K_128_SMEM) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, st_insts, g_events);
  task_rms_norm_f16_from_smem<128, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP_HANDLER(OP_ARGMAX_PARTIAL_bf16_1152_50688_132) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  task_argmax_partial<1152, 50688, 132, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ARGMAX_REDUCE_bf16_1152_132) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  task_argmax_reduce_kernel<1152, 132, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ARGMAX_PARTIAL_bf16_1024_65536_128) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  task_argmax_partial<1024, 65536, 128, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ARGMAX_REDUCE_bf16_1024_128) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  task_argmax_reduce_kernel<1024, 128, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ARGMAX_REDUCE_GLOBAL_bf16_256) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, g_events);
  task_argmax_reduce_global_kernel<256, __nv_bfloat16>(
      inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_ROPE_INTERLEAVE_512) {
  DAE_UNUSED(sm_id, thread_id, pc, count, finish, inst, scratch_space, st_insts, g_events);
  task_rope_interleaved<512>(smem_base, m2c, c2m);
}

DAE_COMPUTE_OP_HANDLER(OP_PROFILE_EVENT) {
  DAE_UNUSED(pc, count, finish, smem_base, tmem_base_ptr,
             tmem_mma_barrier, tmem_mma_phase, scratch_space, st_insts, m2c,
             c2m);
  // Diagnostic-only schedule marker.  Synchronize only the four compute
  // warps: the memory warps are independent VM roles and never join this
  // barrier.  Production images do not select this opcode.
  __sync_compute_group(128);
  if (thread_id == 0) {
    const int event_id = inst.args[0];
    if (event_id >= 2 && event_id < numProfileEvents) {
      g_events[sm_id * numProfileEvents + event_id] =
          cuda::ptx::get_sreg_globaltimer();
    }
  }
}

DAE_COMPUTE_OP_HANDLER(OP_LOOPC) {
  DAE_UNUSED(sm_id, thread_id, finish, smem_base, scratch_space, st_insts, m2c, c2m, g_events);
  const int counter_reg = inst.args[2];
  if (++count[counter_reg] < inst.args[0]) {
    pc = inst.args[1];
    __cprint("LOOPC back to PC %d, reg=%d count=%d", pc, counter_reg, count[counter_reg]);
  } else {
    count[counter_reg] = 0;
    __cprint("LOOPC finished, reg=%d count=%d", counter_reg, count[counter_reg]);
  }
  __sync_compute_group(128);
}

DAE_COMPUTE_OP_HANDLER(OP_TERMINATEC) {
  DAE_UNUSED(pc, count, inst, smem_base, scratch_space, st_insts);
  finish = true;
  c2m.template push<0, true>(thread_id, 0);
  if (thread_id == 0) {
    int event_base = sm_id * numProfileEvents;
    g_events[event_base + 1] = cuda::ptx::get_sreg_globaltimer();
#if defined(DAE_TRACK_PROFILE)
    g_events[event_base + DAE_TRACK_COMPUTE_M2C_WAIT_NS] = m2c.track_wait_ns;
    g_events[event_base + DAE_TRACK_COMPUTE_M2C_WAIT_CALLS] =
        m2c.track_wait_calls;
    g_events[event_base + DAE_TRACK_COMPUTE_M2C_CONTENDED] =
        m2c.track_contended_waits;
    g_events[event_base + DAE_TRACK_MAGIC] = daeTrackProfileMagic;
#endif
  }
  __cprint("TERMINATE from comptue: c2m.ptr=%d", c2m.ptr);
}

#if __has_include("dae/dynamic_compute_handlers.inc")
  #include "dae/dynamic_compute_handlers.inc"
#endif

#undef DAE_COMPUTE_OP_HANDLER

template <typename M2CQueue, typename C2MQueue>
static __device__ __forceinline__ void dispatch_compute_instruction(
  int sm_id,
  int thread_id,
  uint32_t &pc,
  uint32_t *count,
  bool &finish,
  const CInst &inst,
  void *smem_base,
  uint32_t tmem_base_ptr,
  uint64_t *tmem_mma_barrier,
  uint32_t &tmem_mma_phase,
  uint64_t *scratch_space,
  MInst *st_insts,
  M2CQueue &m2c,
  C2MQueue &c2m,
  uint64_t *g_events
) {
  switch (inst.opcode) {
    #define DAE_COMPUTE_OP(name) \
      case name: \
        DAE_COMPUTE_HANDLER_NAME(name)(sm_id, thread_id, pc, count, finish, inst, smem_base, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase, scratch_space, st_insts, m2c, c2m, g_events); \
        break;
      #include "dae/selected_compute_ops.inc"
    #undef DAE_COMPUTE_OP
    default:
      __cprint("Unknown compute opcode: %d\n", inst.opcode);
      assert(false && "Unknown compute opcode");
  }
}

#undef DAE_UNUSED
#undef DAE_COMPUTE_HANDLER_NAME
#undef DAE_COMPUTE_OP_PARAMS
