#pragma once

#include "virtualcore.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm

template<typename Atom, int M, int N, int K,
         int b_load_interval, bool residual,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemm(
    const int nKTiles, 
    void *base, 
    M2C_Type& m2c, 
    C2M_Type& c2m
) {
    using namespace cute;
    using AtomTrait = MMA_Traits<Atom>;

    using data_t = typename AtomTrait::ValTypeA;
    using accum_t = typename AtomTrait::ValTypeC;

    constexpr int MMA_M = shape<0>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_N = shape<1>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_K = shape<2>(typename AtomTrait::Shape_MNK{});

    static_assert(M % MMA_M == 0, "Only M multiple of 64 is supported");
    static_assert(N % MMA_N == 0, "Only N multiple of 64 is supported");
    static_assert(K % MMA_K == 0, "Only K multiple of 16 is supported");

    int thread_id = __compute_tid();

    // Both A and B are in shared memory, C in register 
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<Atom>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );
    auto thr_mma = tiled_mma.get_slice(thread_id);

    // this layout should match the TMA load layout
    auto layout_sA = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<N>{},Int<K>{}));
    auto layout_sC = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<N>{}));

    // Load residual before the main loop
    data_t *sC = nullptr;
    if constexpr (residual) {
        int slot_c = m2c.template pop<0>();
        sC = (data_t *)get_slot_address(base, extract(slot_c));
    }
    Tensor t_dummyC = make_tensor(
        make_smem_ptr((accum_t*)sC),
        make_shape(Int<M>{}, Int<N>{})
    );
    auto frag_C = thr_mma.partition_fragment_C(t_dummyC);
    if constexpr (residual)
        copy(thr_mma.partition_C(t_dummyC), frag_C);
    else
        clear(frag_C);

    // loop over tiles of matrix, each tile is of size (M x N)
    int old_slots;
    constexpr int b_tile_offset = N * K; // offset of B tile in smem in elements
    // TODO(zhiyuang): add inflight for wait
    for (int i = 0; i < nKTiles; i++) {
        int slot_a, slot_b;
        data_t *sa, *sb;
        // load A and B from shared memory
        if (i % b_load_interval == 0) {
            slot_b = m2c.template pop<0>();
            sb = (data_t *)get_slot_address(base, extract(slot_b));
        } else {
            sb += b_tile_offset;
        }
        slot_a = m2c.template pop<0>();
        sa = (data_t *)get_slot_address(base, slot_a);

        if (i > 0) {
            warpgroup_wait<0>();
            c2m.push(thread_id, old_slots);
        }

        auto sA = make_tensor(make_smem_ptr(sa), layout_sA);
        auto sB = make_tensor(make_smem_ptr(sb), layout_sB);

        auto frag_A = thr_mma.partition_fragment_A(sA);
        auto frag_B = thr_mma.partition_fragment_B(sB);

        warpgroup_arrive();
        gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);   // C = A*B + C
        warpgroup_commit_batch();

        if (thread_id == 0) {
            old_slots = slot_a;
            if (i % b_load_interval == b_load_interval - 1)
                old_slots |= slot_b;
        }
    }

    int slot_c = m2c.template pop<0>();
    data_t* sc = (data_t *)get_slot_address(base, extract(slot_c));
    auto t_sC = make_tensor(make_smem_ptr(sc), layout_sC);
    auto thr_sC = thr_mma.partition_C(t_sC);
    
    // clean last batch
    warpgroup_wait<0>();
    c2m.push(thread_id, old_slots);

    copy(frag_C, thr_sC);
    c2m.template push<0, true>(thread_id, slot_c); // commit the output tile
}
