#pragma once

#include "context.cuh"

#include "task/argmax.cuh"
#include "task/attention.cuh"
#include "task/gemv.cuh"
#include "task/rms_norm.cuh"
#include "task/silu.cuh"
#include "task/wgmma.cuh"

#include <type_traits>

#define DAE_COMPUTE_OP_PARAMS \
  int sm_id, \
  int thread_id, \
  uint32_t &pc, \
  uint32_t &count, \
  bool &finish, \
  const CInst &inst, \
  void *smem_base, \
  uint64_t *scratch_space, \
  MInst *st_insts, \
  M2CQueue &m2c, \
  C2MQueue &c2m, \
  uint64_t *g_events

#define DAE_COMPUTE_OP(name) \
  template <typename M2CQueue, typename C2MQueue> \
  static __device__ __forceinline__ void name(DAE_COMPUTE_OP_PARAMS)

template <typename... Args>
static __device__ __forceinline__ void dae_ignore(Args const &...) {}

DAE_COMPUTE_OP(handle_dummy) {
  (void)sm_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)smem_base;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  for (int i = 0; i < inst.args[0]; i++) {
    __cprint("[Dummy][i=%d] before wait", i);
    auto slot_id = m2c.pop();
    __nanosleep(inst.args[1]);
    __cprint("[Dummy][i=%d] after pop slot_id=%d", i, slot_id);
    c2m.template push<0>(thread_id, slot_id);
  }
}

DAE_COMPUTE_OP(handle_copy) {
  (void)sm_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
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

DAE_COMPUTE_OP(handle_gemv_m64n8) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  using gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemv<gemv_atom, 64, 256, 4, false>(inst.args[0], inst.args[1], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_gemv_m64n8k64) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  using gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemv<gemv_atom, 64, 64, 1, false>(inst.args[0], inst.args[1], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_gemv_m64n8b2) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  using gemv_atom = cute::SM90_64x8x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemv<gemv_atom, 64, 256, 2, false>(inst.args[0], inst.args[1], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_gemv_m64n8_mma) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  task_gemv_mma<64, 8, 256>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_gemm_m64n64) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  using gemm_atom = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemm<gemm_atom, 64, 64, 128, 1, false>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_gemm_m64n64k64) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  using gemm_atom = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemm<gemm_atom, 64, 64, 64, 1, false>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_gemm_m64n128k64) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  using gemm_atom = cute::SM90_64x128x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  task_gemm<gemm_atom, 64, 128, 64, 1, false>(inst.args[0], smem_base, m2c, c2m);
}

template <int HeadDim, bool SplitKv, typename KernelQK, typename KernelPV, typename M2CQueue, typename C2MQueue>
static __device__ __forceinline__ void handle_attention_common(
  const CInst &inst,
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

  const bool need_norm = inst.args[2] & 0x1;
  const bool need_rope = inst.args[2] & 0x2;
  if constexpr (std::is_same_v<KernelQK, cute::SM80_16x8x16_F32BF16BF16F32_TN>) {
    task_attention_fwd_flash3_grouped_mma<HeadDim, 64, 64, false, 0, false, false, KernelQK, KernelPV>(
      inst.args[0],
      0,
      64,
      inst.args[1],
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
      inst.args[0],
      0,
      64,
      inst.args[1],
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

DAE_COMPUTE_OP(handle_attention_hdim128) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  using kernel_qk = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using kernel_pv = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
  handle_attention_common<128, false, kernel_qk, kernel_pv>(inst, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP(handle_attention_hdim128_split) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  using kernel_qk = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using kernel_pv = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
  handle_attention_common<128, true, kernel_qk, kernel_pv>(inst, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP(handle_attention_split_post_reduce) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  task_split_post_reduce<128, 4, 64, 16, 32>(inst.args[0], smem_base, (float *)scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP(handle_attention_hdim64) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  using kernel_qk = cute::SM90_64x64x16_F32BF16BF16_SS<cute::GMMA::Major::K, cute::GMMA::Major::K>;
  using kernel_pv = cute::SM90_64x64x16_F32BF16BF16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
  handle_attention_common<64, false, kernel_qk, kernel_pv>(inst, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP(handle_attention_hdim128_mma) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  using kernel_qk = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  using kernel_pv = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  handle_attention_common<128, false, kernel_qk, kernel_pv>(inst, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP(handle_attention_hdim128_split_mma) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  using kernel_qk = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  using kernel_pv = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  handle_attention_common<128, true, kernel_qk, kernel_pv>(inst, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP(handle_attention_hdim64_mma) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  using kernel_qk = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  using kernel_pv = cute::SM80_16x8x16_F32BF16BF16F32_TN;
  handle_attention_common<64, false, kernel_qk, kernel_pv>(inst, smem_base, scratch_space, st_insts, m2c, c2m);
}

DAE_COMPUTE_OP(handle_silu_interleaved) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  task_silu_smem_1D<6144>(inst.args[0], smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_silu_sw128) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  const int num_token = inst.args[0];
  auto layout_sv = tile_to_shape(
    GMMA::Layout_MN_SW128_Atom<__nv_bfloat162>{},
    make_shape(Int<32>{}, num_token)
  );
  task_silu_smem<64>(num_token, layout_sv, smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_rms_norm_4096_smem) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)st_insts;
  (void)g_events;
  task_rms_norm_f16_from_smem<4096, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP(handle_rms_norm_2048_smem) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)st_insts;
  (void)g_events;
  task_rms_norm_f16_from_smem<2048, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP(handle_rms_norm_128_smem) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)st_insts;
  (void)g_events;
  task_rms_norm_f16_from_smem<128, __nv_bfloat16>(
    smem_base,
    inst.args[0],
    *reinterpret_cast<const __nv_bfloat16 *>(inst.args + 1),
    (float *)scratch_space,
    m2c,
    c2m
  );
}

DAE_COMPUTE_OP(handle_argmax_partial_1152) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  task_argmax_partial<1152, 50688, 132, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP(handle_argmax_reduce_1152) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  task_argmax_reduce_kernel<1152, 132, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP(handle_argmax_partial_1024) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  task_argmax_partial<1024, 65536, 128, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP(handle_argmax_reduce_1024) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)g_events;
  task_argmax_reduce_kernel<1024, 128, __nv_bfloat16>(inst.args[0], smem_base, st_insts, (void *)scratch_space, m2c, c2m);
}

DAE_COMPUTE_OP(handle_rope_interleave_512) {
  (void)sm_id;
  (void)thread_id;
  (void)pc;
  (void)count;
  (void)finish;
  (void)inst;
  (void)scratch_space;
  (void)st_insts;
  (void)g_events;
  task_rope_interleaved<512>(smem_base, m2c, c2m);
}

DAE_COMPUTE_OP(handle_loopc) {
  (void)sm_id;
  (void)thread_id;
  (void)finish;
  (void)smem_base;
  (void)scratch_space;
  (void)st_insts;
  (void)m2c;
  (void)c2m;
  (void)g_events;
  if (++count < inst.args[0]) {
    pc = inst.args[1];
    __cprint("LOOPC back to PC %d, count=%d", pc, count);
  } else {
    count = 0;
    __cprint("LOOPC finished, count=%d", count);
  }
  __sync_compute_group(128);
}

DAE_COMPUTE_OP(handle_terminatec) {
  (void)pc;
  (void)count;
  (void)inst;
  (void)smem_base;
  (void)scratch_space;
  (void)st_insts;
  (void)m2c;
  finish = true;
  c2m.template push<0, true>(thread_id, 0);
  if (thread_id == 0) {
    int event_base = sm_id * numProfileEvents;
    g_events[event_base + 1] = cuda::ptx::get_sreg_globaltimer();
  }
  __cprint("TERMINATE from comptue: c2m.ptr=%d", c2m.ptr);
}

#undef DAE_COMPUTE_OP

template <typename M2CQueue, typename C2MQueue>
static __device__ __forceinline__ void dispatch_compute_instruction(
  int sm_id,
  int thread_id,
  uint32_t &pc,
  uint32_t &count,
  bool &finish,
  const CInst &inst,
  void *smem_base,
  uint64_t *scratch_space,
  MInst *st_insts,
  M2CQueue &m2c,
  C2MQueue &c2m,
  uint64_t *g_events
) {
  switch (inst.opcode) {
    #define DAE_COMPUTE_OP(name, handler) \
      case name: \
        handler(sm_id, thread_id, pc, count, finish, inst, smem_base, scratch_space, st_insts, m2c, c2m, g_events); \
        break;
      #include "dae/selected_compute_ops.inc"
    #undef DAE_COMPUTE_OP
    default:
      __cprint("Unknown compute opcode: %d\n", inst.opcode);
      assert(false && "Unknown compute opcode");
  }
}

#undef DAE_COMPUTE_OP
#undef DAE_COMPUTE_OP_PARAMS
