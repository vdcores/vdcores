#pragma once

#include "virtualcore.cuh"

template<typename M2LD_Type, typename M2C_Type>
__device__ __forceinline__ void ldwarp_execute_singlethread(
    M2LD_Type &m2ld, M2C_Type &m2c,
    const MInst *st_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars) {

  __ldprint("[LD Warp] Start LD warp execution");

  int regFile[4];
  m2ld.wait();
  LdCmd cmd { .raw = m2ld.data[m2ld.ptr] };

  while (cmd.slot != SLOT_END) {
    auto &slot = cmd.slot;
    auto inst = st_insts[slot];

    m2ld.advance();

    auto &opcode = cmd.opcode;
    auto &bar = cmd.bar;

    __ldprint("Receive LD cmd: slot=%d bar=%d opcode=%d", slot, bar, op(opcode));

    // If its a readbar, we do the readbar
    // TODO(zhiyuang): wait bar here if bar is set
    if ((opcode & MEM_OP_FLAGS_BARRIER) && !(opcode & MEM_OP_FLAGS_WRITEBACK)) {
      volatile int *bar = bars + inst.bar();
      // bool first_wait = true;
      // if (blockIdx.x == 0 && first_wait) {
      //   printf("[LD][sm=%d] check bar=%d bars[bar]=%d\n", blockIdx.x, inst.bar(), *bar);
      // }
      while (*bar != 0) {
        // busy wait
        __nanosleep(barrierPollSleepCycles);
        // if (blockIdx.x == 0 && first_wait) {
        //   printf("[LD][sm=%d] waiting bar=%d bars[bar]=%d\n", blockIdx.x, inst.bar(), *bar);
        //   first_wait = false;
        // }
      }
      __ldprint("wait for global barrier before load: bar=%d", inst.bar());
    };

    // TODO(zhiyuang): change location?
    switch(op(opcode)) {
      case op(OP_PAGED_LOAD_1D): {
        // Inst.size = token_id, Inst.arg = session_id
        // All exits reach the common barrier arrival; Dummy frees the slots.
        if (inst.address == 0) {
          printf("[PAGED] missing descriptor\n");
          break;
        }
        const auto& cache = *reinterpret_cast<const PagedLoadConfig*>(inst.address);
        if (!cache.page_table || !cache.context_table) {
          printf("[PAGED] missing tables\n");
          break;
        }

        // Is there a way to make it clearer that we are refering to token_id and not size? Maybe copying to another variable? 
        const uint32_t page_index = inst.size / cache.tokens_per_page;
        const uint32_t page_offset = inst.size % cache.tokens_per_page;

        const uint32_t pages_per_session = cache.kv_seq_len / cache.tokens_per_page;
        
        // Can we make it clearer that arg is the session_id
        const int64_t page_id = cache.context_table[
            uint64_t(inst.arg) * pages_per_session + page_index];
        if (page_id < 0) {
          printf("[PAGED] session=%u token=%u unmapped\n",
                 unsigned(inst.arg), unsigned(inst.size));
          break;
        }

        // Positive physical page IDs and session IDs must be valid.
        const uint64_t page_addr = cache.page_table[page_id];
        if (!page_addr) {
          printf("[PAGED] null page address\n");
          break;
        }

        // Assumes bf16 dtype
        const uint64_t head_stride_bytes = uint64_t(cache.head_dim) * 2 * 2;
        const uint64_t layer_stride_bytes = cache.num_kv_heads * head_stride_bytes;
        const uint64_t token_stride_bytes = cache.layers * layer_stride_bytes;

        const uint64_t token_addr = page_addr + page_offset * token_stride_bytes;
        
        for (uint32_t layer = 0; layer < cache.layers; ++layer) {
          for (uint32_t head = 0; head < cache.num_kv_heads; ++head) {
            const uint64_t key_addr = token_addr + layer * layer_stride_bytes
                + head * head_stride_bytes;
            printf("[PAGED] session=%u token=%u layer=%u head=%u K=0x%llx V=0x%llx\n",
                   unsigned(inst.arg), unsigned(inst.size), layer, head,
                   static_cast<unsigned long long>(key_addr),
                   static_cast<unsigned long long>(key_addr + uint64_t(cache.head_dim) * 2));
          }
        }
        break;
      }
      case op(OP_ALLOC_TMA_LOAD_1D): {
        __ldprint("TMA 1D Load: size=%d", inst.size);
        // We need to get a slot ID first, as we will use its barrier
        cuda::device::memcpy_async_tx(
            (char *)(get_slot_address(smem_base, slot)),
            (char *)(inst.address),
            cuda::aligned_size_t<16>(inst.size),
            m2c.barriers[bar]
        );
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_TENSOR_1D): {
        __ldprint("TMA Tensor 1D Load: size=%d", inst.size);
        asm volatile(
          "cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2}], [%3];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((uint32_t)inst.address),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_2D): {
        const uint16_t *cord = inst.coords;
        __ldprint("TMA 2D Load: desc_idx=%d size=%d cord=(%d,%d)", inst.arg, inst.size, cord[0], cord[1]);
        asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3}], [%4];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_3D): {
        const uint16_t *cord = inst.coords;
        __ldprint("TMA 3D Load: desc_idx=%d size=%d cord=(%d,%d,%d)", inst.arg, inst.size, cord[0], cord[1], cord[2]);
        asm volatile(
          "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3, %4}], [%5];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((int)cord[2]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_4D): {
        const uint16_t *cord = inst.coords;
        __ldprint("TMA 4D Load: desc_idx=%d size=%d cord=(%d,%d,%d,%d)",
          inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
        asm volatile(
          "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {%2, %3, %4, %5}], [%6];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((int)cord[2]),
            "r"((int)cord[3]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_TMA_LOAD_5D_FIX0): {
        const uint16_t *cord = inst.coords;
        // hardcode first coord to be 0
        __ldprint("TMA 5D Load: desc_idx=%d size=%d cord=(0,%d,%d,%d,%d)",
          inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
        asm volatile(
          "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes"
          "[%0], [%1, {0, %2, %3, %4, %5}], [%6];\n"
          :
          : "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot))),
            "l"((void *)(tma_descs + inst.arg)),
            "r"((int)cord[0]),
            "r"((int)cord[1]),
            "r"((int)cord[2]),
            "r"((int)cord[3]),
            "r"((uint32_t)__cvta_generic_to_shared(
              m2c.native_bar(bar)
            ))
          : "memory");
        cuda::device::barrier_expect_tx(
          m2c.barriers[bar],
          cuda::aligned_size_t<16>(inst.size)
        );
        break; }
      case op(OP_ALLOC_WB_REG_STORE): {
        // TODO(zhiyuang): recalculate the mask or read from smem?
        int slotMask = mkSlotMask(slot, inst.nslot());
        m2c.data[bar] = slotMask | 0x80000000U; // set high bit to invalidate the writeback
        regFile[inst.size] = slotMask;
        __ldprint("[REG] store: reg_id=%d slot=%d nslot=%d bar=%d slotMask=0x%X",
          inst.size, slot, inst.nslot(), bar, slotMask);
        break;
      }
      case op(OP_ALLOC_REG_LOAD): {
        m2c.data[bar] = regFile[inst.size];
        __ldprint("[REG] load: reg_id=%d bar=%d slotMask=0x%X", inst.size, bar, regFile[inst.size]);
        break;
      }
    }

    // m2c data should be prepared in the CFU
    (void)m2c.barriers[bar].arrive();

    m2ld.wait();
    cmd.raw = m2ld.data[m2ld.ptr];
  } // End of LD warp loop

  __ldprint("End of LD warp execution");
  // __print(0, "End of LD warp execution");
}
