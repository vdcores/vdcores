#pragma once

#include "context.cuh"
#include "type.cuh"
#include "virtualcore.cuh"

#include <cute/algorithm/gemm.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

// Decode-only BF16 projection task.  Four consecutive M128 outputs share
// every K128 activation load and remain in disjoint TMEM columns.  A compact
// FP32 epilogue exposes only column zero for one TMA reduce-add, avoiding the
// replicated N8 global intermediate used by generic batched GEMV schedules.
template <typename M2CQueue, typename C2MQueue>
__device__ __forceinline__ void task_dsv4_bf16_gemv_group4_splitk_sm100(
    int num_k_tiles,
    void *smem_base,
    void *task_scratch,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    M2CQueue &m2c,
    C2MQueue &c2m) {
  using namespace cute;
  using Data = cutlass::bfloat16_t;
  using Accum = float;

  constexpr int kTileM = 128;
  constexpr int kTileN = 8;
  constexpr int kTileK = 128;
  constexpr int kOutputGroups = 4;
  constexpr int kActivationTilesPerChunk = 4;
  using Atom = SM100_MMA_F16BF16_SS<
      Data, Data, Accum, kTileM, kTileN,
      UMMA::Major::K, UMMA::Major::K>;
  using TiledMma = decltype(make_tiled_mma(Atom{}));

  if (num_k_tiles <= 0 ||
      (num_k_tiles != 2 && num_k_tiles % kActivationTilesPerChunk)) {
    asm volatile("trap;");
  }

  TiledMma tiled_mma;
  auto cta_mma = tiled_mma.get_slice(0);
  auto mma_shape_a = partition_shape_A(
      tiled_mma, make_shape(Int<kTileM>{}, Int<kTileK>{}));
  auto mma_shape_b = partition_shape_B(
      tiled_mma, make_shape(Int<kTileN>{}, Int<kTileK>{}));
  auto layout_sA = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Data>{}, mma_shape_a);
  auto layout_sB = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<Data>{}, mma_shape_b);
  auto logical_layout_sB = tile_to_shape(
      UMMA::Layout_K_SW128_Atom<Data>{},
      make_shape(Int<kTileN>{}, Int<kTileK>{}));
  static_assert(cosize_v<decltype(layout_sB)> == kTileN * kTileK);

  auto logical_c = make_tensor(
      make_smem_ptr(static_cast<Accum *>(nullptr)),
      make_layout(
          make_shape(Int<kTileM>{}, Int<kTileN>{}),
          make_stride(Int<kTileN>{}, Int<1>{})));
  auto cta_c = cta_mma.partition_C(logical_c);

  const auto scratch_address = reinterpret_cast<uintptr_t>(task_scratch);
  auto *b_storage = reinterpret_cast<Data *>(
      (scratch_address + 127U) & ~uintptr_t(127U));
  auto sB = make_tensor(make_smem_ptr(b_storage), layout_sB);
  auto logical_sB = make_tensor(
      make_smem_ptr(b_storage), logical_layout_sB);
  const int tid = __compute_tid();

  int logical_tile = 0;
  for (int chunk_start = 0; chunk_start < num_k_tiles;
       chunk_start += kActivationTilesPerChunk) {
    const int activation_slots = m2c.template pop<0>();
    const auto *activation = static_cast<const __nv_bfloat16 *>(
        get_slot_address(smem_base, extract(activation_slots)));

    const int tiles_in_chunk = min(
        kActivationTilesPerChunk, num_k_tiles - chunk_start);
    for (int tile_in_chunk = 0;
         tile_in_chunk < tiles_in_chunk;
         ++tile_in_chunk, ++logical_tile) {
      const Data value = Data(
          __bfloat162float(
              activation[tile_in_chunk * kTileK + tid]));
#pragma unroll
      for (int row = 0; row < kTileN; ++row) {
        logical_sB(row, tid) = value;
      }
      __sync_compute_group(128);

#pragma unroll
      for (int output_group = 0; output_group < kOutputGroups;
           ++output_group) {
        const int weight_slots = m2c.template pop<0>();
        auto *weight = static_cast<Data *>(
            get_slot_address(smem_base, extract(weight_slots)));
        auto sA = make_tensor(make_smem_ptr(weight), layout_sA);
        auto frag_a = cta_mma.make_fragment_A(sA);
        auto frag_b = cta_mma.make_fragment_B(sB);
        auto group_acc = cta_mma.make_fragment_C(cta_c);
        group_acc.data() = tmem_base_ptr + output_group * kTileN;
        tiled_mma.accumulate_ = logical_tile == 0
            ? UMMA::ScaleOut::Zero
            : UMMA::ScaleOut::One;

        if (tid < 32) {
#pragma unroll
          for (int k_block = 0; k_block < size<2>(frag_a); ++k_block) {
            gemm(
                tiled_mma,
                frag_a(_, _, k_block),
                frag_b(_, _, k_block),
                group_acc);
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;
        c2m.push(tid, weight_slots);
      }
      __sync_compute_group(128);
    }
    c2m.push(tid, activation_slots);
  }

  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __sync_compute_group(128);
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  const int output_slots = m2c.template pop<0>();
  auto *output = static_cast<float *>(
      get_slot_address(smem_base, extract(output_slots)));
  auto coord_c = make_identity_tensor(
      make_shape(Int<kTileM>{}, Int<kTileN>{}));
  auto cta_coord_c = cta_mma.partition_C(coord_c);
  using TmemLoad = SM100_TMEM_LOAD_32dp32b1x;
#pragma unroll
  for (int output_group = 0; output_group < kOutputGroups; ++output_group) {
    auto group_acc = cta_mma.make_fragment_C(cta_c);
    group_acc.data() = tmem_base_ptr + output_group * kTileN;
    auto t_acc = group_acc(make_coord(_, _), _0{}, _0{});
    auto c_acc = cta_coord_c(make_coord(_, _), _0{}, _0{});
    auto tiled_t2r = make_tmem_copy(TmemLoad{}, t_acc);
    const int thread_idx = tid % size(tiled_t2r);
    auto thread_t2r = tiled_t2r.get_slice(thread_idx);
    auto thread_tmem = thread_t2r.partition_S(t_acc);
    auto thread_coord = thread_t2r.partition_D(c_acc);
    auto registers = make_tensor<Accum>(shape(thread_coord));
    copy(tiled_t2r, thread_tmem, registers);
    for (int index = 0; index < size(registers); ++index) {
      const int row = int(get<0>(thread_coord(index)));
      const int column = int(get<1>(thread_coord(index)));
      if (row < kTileM && column == 0) {
        output[output_group * kTileM + row] = registers(index);
      }
    }
  }

  __sync_compute_group(128);
  c2m.template push<0, true>(tid, output_slots);
}
