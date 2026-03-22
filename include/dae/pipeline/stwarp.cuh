#pragma once

#include "virtualcore.cuh"

// TODO(zhiyuang): attach bars to the writeback
template<typename C2M_Type>
__device__ __forceinline__ void stwarp_execute_singlethread(
    C2M_Type &c2m, const MInst* slot_insts,
    const void *smem_base, const CUtensorMap *tma_descs, int *bars) {

  __stprint("[ST Warp] Start ST warp execution");
    
  int slot_mask = c2m.pop();
  while (slot_mask) {
  
    auto slot = extract(slot_mask);
    bool do_free = true;

    __stprint("Receive ST slot: slot=%d", slot);

    auto &inst = slot_insts[slot];
    uint16_t opcode = inst.opcode;
    // all ops are writeback ops

    switch(op(opcode)) {
      case op(OP_ALLOC_WB_TMA_STORE_1D):
      {
        cuda::ptx::cp_async_bulk(
          cuda::ptx::space_global,
          cuda::ptx::space_shared,
          (void*)(inst.address),
          (const void *)(get_slot_address(smem_base, slot)),
          inst.size
        );
        cuda::ptx::cp_async_bulk_commit_group();
      } 
        break;
      case op(OP_ALLOC_WB_TMA_STORE_2D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 2D Store: desc_idx=%d size=%d cord=(%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1]);
          asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
            "[%0, {%1, %2}], [%3];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_4D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 4D Store: desc_idx=%d size=%d cord=(%d,%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
          asm volatile(
            "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
            "[%0, {%1, %2, %3, %4}], [%5];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((int)cord[3]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_3D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 3D Store: desc_idx=%d size=%d cord=(%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2]);
          asm volatile(
            "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
            "[%0, {%1, %2, %3}], [%4];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_STORE_5D_FIX0):
        {
          const uint16_t *cord = inst.coords;
          // harcode first coord to be 0
          __stprint("TMA 5D Store: desc_idx=%d size=%d cord=(0,%d,%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2], cord[3]);
          asm volatile(
            "cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
            "[%0, {0, %1, %2, %3, %4}], [%5];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((int)cord[3]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_REDUCE_ADD_2D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 2D Reduce-Add: desc_idx=%d size=%d cord=(%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1]);
          asm volatile(
            "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group "
            "[%0, {%1, %2}], [%3];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      case op(OP_ALLOC_WB_TMA_REDUCE_ADD_3D):
        {
          const uint16_t *cord = inst.coords;
          __stprint("TMA 3D Reduce-Add: desc_idx=%d size=%d cord=(%d,%d,%d)",
                    inst.arg, inst.size, cord[0], cord[1], cord[2]);
          asm volatile(
            "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.bulk_group "
            "[%0, {%1, %2, %3}], [%4];\n"
            :
            : "l"((void *)(tma_descs + inst.arg)),
              "r"((int)cord[0]),
              "r"((int)cord[1]),
              "r"((int)cord[2]),
              "r"((uint32_t)__cvta_generic_to_shared(get_slot_address(smem_base, slot)))
              : "memory");
          cuda::ptx::cp_async_bulk_commit_group();
        }
        break;
      default:
        // unknown opcode
        __stprint("Unknown mem wb opcode: slot_mask=%x slot=%d op=%d opcode=%04x\n", slot_mask, slot, op(inst.opcode), inst.opcode);
        do_free = false;
        break;
    }

    // do bar for all instructions all at once
    if (opcode & MEM_OP_FLAGS_BARRIER) {
      cuda::std::atomic_ref<int> bar {bars[inst.bar()]};
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      int current_cnt = bar.fetch_sub(1, cuda::std::memory_order_release);
      __stprint("Arrive for barrier %d, remaining count=%d", inst.bar(), current_cnt - 1);
      // if (inst.bar() == 0)
      //   printf("[sm=%d] Arrive for barrier %d, remaining count=%d\n", blockIdx.x, inst.bar(), current_cnt - 1);
    } else {
      cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>{});
    }

    // write back to free the slot
    __stprint("finish slot=%d op=%d flags=%02x",
      slot, op(inst.opcode), opcode & ((1 << flagBits) - 1));

    if (do_free)
      c2m.reset(slot_mask);
    slot_mask = c2m.pop();
  }

  __stprint("End of ST warp execution");
}