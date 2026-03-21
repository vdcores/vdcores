#pragma once

#include "virtualcore.cuh"
#include "rope.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm

// TODO(zhiyuang): this is a gemv style wgmma, not tile overN but prefetch K tiles
template<typename Atom, int M, int K,
         int b_load_interval, bool residual,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv(
    const int nKTiles, 
    const int prefetch,
    const void *base, 
    M2C_Type& m2c, 
    C2M_Type& c2m
) {
    using namespace cute;
    using AtomTrait = MMA_Traits<Atom>;

    using data_t = typename AtomTrait::ValTypeA;
    using accum_t = typename AtomTrait::ValTypeC;

    constexpr int MMA_K = shape<2>(typename AtomTrait::Shape_MNK{});
    constexpr int N = shape<1>(typename AtomTrait::Shape_MNK{});
    static_assert(K % MMA_K == 0, "Only K multiple of 16 is supported");

    int thread_id = threadIdx.x;

    // Both A and B are in shared memory, C in register 
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<Atom>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );
    auto thr_mma = tiled_mma.get_slice(thread_id);

    auto layout_sA = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<N>{},Int<K>{}));
    // TODO(zhiyuang): for swizzle 128 we use MN layout
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<accum_t>{},
        make_shape(Int<M>{},Int<N>{}));
    auto layout_output_C = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
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

    int old_slots;
    constexpr int b_tile_offset = N * K; // offset of B tile in smem in elements
    // TODO(zhiyuang): batch load B (vector)?
    for (int i = 0; i < nKTiles; i++) {
        int slot_a, slot_b;
        data_t *sa, *sb;
        
        if (i % b_load_interval == 0) {
            slot_b = m2c.template pop<0>();
            sb = (data_t *)get_slot_address(base, extract(slot_b));
        } else
            sb += b_tile_offset;

        slot_a = m2c.template pop<0>();
        sa = (data_t *)get_slot_address(base, extract(slot_a));

        // TODO(zhiyuang): move this before or after the commit?
        // currently putting here is better
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
            if (i % b_load_interval == 3)
                old_slots |= slot_b;
        }
    }

    auto slot_c = m2c.pop();
    auto t_sC = make_tensor(
        make_smem_ptr((data_t*)get_slot_address(base, extract(slot_c))),
        layout_output_C);
    auto partition_sC = thr_mma.partition_C(t_sC);

    warpgroup_wait<0>();

    c2m.push(thread_id, old_slots);

    copy(frag_C, partition_sC);
    // TODO: do we need synchronize warpgroup before returning slot?
    c2m.template push<0, true>(thread_id, slot_c);
}

// This function uses mma instead of wgmma, so it works on sm >= 89
template<int M, int N, int K, typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv_mma(const int nTiles, void *base, M2C_Type& m2c, C2M_Type& c2m) {
    using namespace cute;

    static_assert(N == 8, "Only support N=8 for now");

    using Atom = SM80_16x8x16_F32BF16BF16F32_TN;
    using AtomTrait = MMA_Traits<Atom>;
    using data_t = typename AtomTrait::ValTypeA;
    using accum_t = typename AtomTrait::ValTypeC;

    constexpr int MMA_M = shape<0>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_N = shape<1>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_K = shape<2>(typename AtomTrait::Shape_MNK{});
    constexpr int numThreads = 32 * (M / MMA_M);

    static_assert(M % MMA_M == 0, "M must be multiple of MMA_M");
    static_assert(K % MMA_K == 0, "K must be multiple of MMA_K");
    static_assert(numThreads == 128, "Only support a 128-thread compute group for now");

    int tid = __compute_tid();

    auto tiled_mma = make_tiled_mma(
        MMA_Atom<Atom>{},
        make_layout(make_shape(Int<M / MMA_M>{}, Int<1>{}, Int<1>{})), // atom replication
        make_tile(Int<M>{}, Int<MMA_N>{}, Int<K>{}) // final target MNK
    );
    auto thr_mma  = tiled_mma.get_slice(tid);

    auto layout_sA = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<M>{}, Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<MMA_N>{}, Int<K>{}));
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<M>{}, Int<MMA_N>{}));

    auto t_dummyA = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sA);
    auto t_dummyB = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sB);
    auto t_dummyC = make_tensor(make_smem_ptr(static_cast<accum_t*>(nullptr)),
        Layout<Shape<Int<M>, Int<MMA_N>>, Stride<Int<1>, Int<M>>>());

    auto frag_A = thr_mma.partition_fragment_A(t_dummyA);
    auto frag_B = thr_mma.partition_fragment_B(t_dummyB);
    auto frag_C = thr_mma.partition_fragment_C(t_dummyC);

    clear(frag_C);

    for (int i = 0; i < nTiles; i++) {
        int slot_b = m2c.template pop<0>();
        data_t* sb = (data_t *)get_slot_address(base, extract(slot_b));
        auto sB = make_tensor(make_smem_ptr(sb), layout_sB);
        copy(thr_mma.partition_B(sB), frag_B);

        int slot_a = m2c.template pop<0>();
        data_t* sa = (data_t *)get_slot_address(base, extract(slot_a));
        auto sA = make_tensor(make_smem_ptr(sa), layout_sA);
        copy(thr_mma.partition_A(sA), frag_A);

        gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);

        __sync_compute_group(numThreads);
        c2m.push(tid, slot_a | slot_b);
    }

    int slot_c = m2c.template pop<0>();
    data_t* sc = (data_t *)get_slot_address(base, extract(slot_c));
    auto t_sC = make_tensor(make_smem_ptr(sc), layout_sC);

    copy(frag_C, thr_mma.partition_C(t_sC));
    c2m.template push<0, true>(tid, slot_c);
}
