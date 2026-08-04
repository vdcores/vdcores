#pragma once

#include "virtualcore.cuh"
#include "rope.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/arch/mma_sm100.hpp>
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

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
            if (i % b_load_interval == b_load_interval - 1) 
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

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
// Blackwell-native BF16 GEMV/GEMM tile.  UMMA reads A/B from shared memory,
// accumulates F32 in TMEM, then pipelines the completed TMEM tile through
// registers into the existing VDCores shared-memory writeback slot.
template<int M, int N, int K, int BLoadInterval, bool Residual,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv_sm100(
    const int n_k_tiles,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    const void *base,
    M2C_Type &m2c,
    C2M_Type &c2m) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    using Atom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, N, UMMA::Major::K, UMMA::Major::K>;

    static_assert(M == 64 || M == 128, "SM100 UMMA GEMV requires M=64 or M=128");
    static_assert(N % 8 == 0 && N >= 8 && N <= 256, "Unsupported SM100 UMMA N tile");
    static_assert(K % 16 == 0, "SM100 BF16 UMMA requires K to be a multiple of 16");
    static_assert(BLoadInterval > 0, "BLoadInterval must be positive");

    const int tid = __compute_tid();
    auto tiled_mma = make_tiled_mma(Atom{});
    auto cta_mma = tiled_mma.get_slice(0);

    auto mma_shape_a = partition_shape_A(tiled_mma, make_shape(Int<M>{}, Int<K>{}));
    auto mma_shape_b = partition_shape_B(tiled_mma, make_shape(Int<N>{}, Int<K>{}));
    auto layout_sA = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<data_t>{}, mma_shape_a);
    auto layout_sB = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<data_t>{}, mma_shape_b);

    auto logical_c = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<N>{}), make_stride(Int<N>{}, Int<1>{})));
    auto cta_c = cta_mma.partition_C(logical_c);
    auto tmem_acc = cta_mma.make_fragment_C(cta_c);
    tmem_acc.data() = tmem_base_ptr;

    int residual_slot = 0;
    data_t *residual_ptr = nullptr;
    if constexpr (Residual) {
        residual_slot = m2c.template pop<0>();
        residual_ptr = static_cast<data_t *>(get_slot_address(base, extract(residual_slot)));
    }

    tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    int live_b_slot = 0;
    constexpr int b_tile_elements = N * K;

    for (int tile_idx = 0; tile_idx < n_k_tiles; ++tile_idx) {
        data_t *sB_ptr;
        if (tile_idx % BLoadInterval == 0) {
            live_b_slot = m2c.template pop<0>();
            sB_ptr = static_cast<data_t *>(get_slot_address(base, extract(live_b_slot)));
        } else {
            sB_ptr = static_cast<data_t *>(get_slot_address(base, extract(live_b_slot)))
                     + (tile_idx % BLoadInterval) * b_tile_elements;
        }

        const int slot_a = m2c.template pop<0>();
        data_t *sA_ptr = static_cast<data_t *>(get_slot_address(base, extract(slot_a)));
        auto sA = make_tensor(make_smem_ptr(sA_ptr), layout_sA);
        auto sB = make_tensor(make_smem_ptr(sB_ptr), layout_sB);
        auto frag_a = cta_mma.make_fragment_A(sA);
        auto frag_b = cta_mma.make_fragment_B(sB);

        if (tid < numThreadsPerWarp) {
            #pragma unroll
            for (int k_block = 0; k_block < size<2>(frag_a); ++k_block) {
                gemm(tiled_mma, frag_a(_, _, k_block), frag_b(_, _, k_block), tmem_acc);
                tiled_mma.accumulate_ = UMMA::ScaleOut::One;
            }
            cutlass::arch::umma_arrive(tmem_mma_barrier);
        }
        cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
        tmem_mma_phase ^= 1;

        int release_mask = slot_a;
        if ((tile_idx + 1) % BLoadInterval == 0 || tile_idx + 1 == n_k_tiles) {
            release_mask |= live_b_slot;
        }
        c2m.push(tid, release_mask);
    }

    const int output_slot = m2c.template pop<0>();
    data_t *output_ptr = static_cast<data_t *>(get_slot_address(base, extract(output_slot)));
    auto layout_output = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{}, make_shape(Int<M>{}, Int<N>{}));
    auto s_output = make_tensor(make_smem_ptr(output_ptr), layout_output);
    auto cta_output = cta_mma.partition_C(s_output);

    using TmemLoad = std::conditional_t<
        M == 64, SM100_TMEM_LOAD_16dp32b1x, SM100_TMEM_LOAD_32dp32b1x>;
    TiledCopy tiled_t2r = make_tmem_copy(TmemLoad{}, tmem_acc);
    ThrCopy thr_t2r = tiled_t2r.get_slice(tid);
    auto thread_tmem = thr_t2r.partition_S(tmem_acc);
    auto thread_output = thr_t2r.partition_D(cta_output);
    auto r_acc = make_tensor<accum_t>(shape(thread_output));
    copy(tiled_t2r, thread_tmem, r_acc);

    auto r_output = make_fragment_like(thread_output);
    if constexpr (Residual) {
        auto s_residual = make_tensor(make_smem_ptr(residual_ptr), layout_output);
        auto thread_residual = thr_t2r.partition_D(cta_mma.partition_C(s_residual));
        auto r_residual = make_fragment_like(thread_residual);
        copy(thread_residual, r_residual);
        #pragma unroll
        for (int i = 0; i < size(r_output); ++i) {
            r_output(i) = data_t(r_acc(i) + static_cast<float>(r_residual(i)));
        }
    } else {
        #pragma unroll
        for (int i = 0; i < size(r_output); ++i) {
            r_output(i) = data_t(r_acc(i));
        }
    }
    copy(r_output, thread_output);

    if constexpr (Residual) {
        c2m.push(tid, residual_slot);
    }
    c2m.template push<0, true>(tid, output_slot);
}
#endif
