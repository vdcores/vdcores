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
template<typename T, int M, int N, int K, typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv_mma(const int nTiles, void *base, M2C_Type& m2c, C2M_Type& c2m) {
    using namespace cute;

    static_assert(N == 8, "Only support N=8 for now");
    static_assert(sizeof(T) == 2, "Only support fp16/u16/bf16 types");

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
    static_assert(numThreads <= 128, "At least 64 threads are required for queue operations");

    // TODO(zhiyuang): calculate arrival count for each thread
    const int arrival_count = 128 / numThreads + ((128 % numThreads) < __compute_tid());
    // now we just use 
    __activate_compute_group(numThreads);

    // One warp, so 32 threads
    int tid = __compute_tid();

    auto tiled_mma = make_tiled_mma(
        MMA_Atom<Atom>{},
        make_layout(make_shape(Int<M / MMA_M>{}, Int<1>{}, Int<1>{})), // atom replication
        make_tile(Int<M>{}, Int<MMA_N>{}, Int<K>{}) // final target MNK
    );
    auto thr_mma  = tiled_mma.get_slice(threadIdx.x);

    // auto tiled_copy = make_tiled_copy_A(
    //     Copy_Atom<SM75_U16x8_LDSM_T, T>{},
    //     tiled_mma
    // );
    // auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    //
    // 4. Partition SMEM per-thread, then make register fragments
    //
    // Per-thread views into SMEM tiles
    auto layout_sA = Layout<Shape<Int<M>, Int<K>>, Stride<_1, Int<M>>>{};
    auto layout_sB = Layout<Shape<Int<MMA_N>, Int<K>>,Stride<_1, Int<MMA_N>>>{};
    auto layout_sC = Layout<Shape<Int<M>, Int<MMA_N>>, Stride<Int<MMA_N>, _1>>{};

    auto t_dummyA = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sA);
    auto t_dummyB = make_tensor(make_smem_ptr(static_cast<data_t*>(nullptr)), layout_sB);
    auto t_dummyC = make_tensor(make_smem_ptr(static_cast<accum_t*>(nullptr)), layout_sC);
    
    // TODO(zhiyuang): tile A or slice A
    auto frag_A = thr_mma.partition_fragment_A(t_dummyA);
    auto frag_B = thr_mma.partition_fragment_B(t_dummyB);   // (thr, mma_n, mma_k)
    auto frag_C = thr_mma.partition_fragment_C(t_dummyC);   // (thr, mma_m, mma_n)

    clear(frag_C);                                      // C = 0

    for (int i = 0; i < nTiles; i++) {
        // load fragB, vectorized as bfloat162
        int slot_b = m2c.pop();
        data_t* sb = (data_t *)get_slot_address(base, extract(slot_b));
        auto t_sB = make_tensor(make_smem_ptr(sb), layout_sB);
        copy(thr_mma.partition_B(t_sB), frag_B);

        // load fragA first as a could be just ready
        int slot_a = m2c.pop();
        data_t* sa = (data_t *)get_slot_address(base, extract(slot_a));
        auto t_sA = make_tensor(make_smem_ptr(sa), layout_sA);
        copy(thr_mma.partition_A(t_sA), frag_A);

        // copy(tiled_copy,
        //     thr_copy.partition_S(t_sA),
        //     thr_copy.retile_D(frag_A)); 

        // 5. Issue the m16n8k16 MMA atom through cute::gemm
        gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);   // C = A*B + C

        // TODO(zhiyuang): do we need sync here
        __sync_compute_group(128);
        c2m.push(slot_a | slot_b);
    }

    // 6. Write back the C fragments to SMEM
    int slot_c = m2c.pop();
    data_t* sc = (data_t *)get_slot_address(base, extract(slot_c));
    auto t_sC = make_tensor(make_smem_ptr(sc), layout_sC);

    // we can directly store the accumulator fragment to shared memory
    copy(frag_C, thr_mma.partition_C(t_sC));

    c2m.template push<0, true>(slot_c);
}
