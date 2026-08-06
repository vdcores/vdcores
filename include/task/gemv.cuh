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
// Generic Blackwell BF16 GEMV tile. Keep this path byte-for-byte equivalent
// to the shared-slot implementation used by the fused projection schedules;
// grouped direct output is isolated below so it cannot perturb RegStore or
// TMA-reduction layouts.
template<int M, int N, int K, int BLoadInterval, bool Residual, bool ApplyRope,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv_sm100_impl(
    const int n_k_tiles,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    const void *base,
    M2C_Type &m2c,
    C2M_Type &c2m,
    const MInst *st_insts,
    const int rope_head_offset) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    using Atom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, N, UMMA::Major::K, UMMA::Major::K>;

    static_assert(M == 64 || M == 128, "SM100 UMMA GEMV requires M=64 or M=128");
    static_assert(N % 8 == 0 && N >= 8 && N <= 256, "Unsupported SM100 UMMA N tile");
    static_assert(K % 16 == 0, "SM100 BF16 UMMA requires K to be a multiple of 16");
    static_assert(BLoadInterval > 0, "BLoadInterval must be positive");
    static_assert(!ApplyRope || (M == 64 && N == 8 && !Residual),
                  "The fused RoPE epilogue requires non-residual M64N8 output");

    const int tid = __compute_tid();
    const data_t *rope_row = nullptr;
    if constexpr (ApplyRope) {
        const int rope_slot = m2c.template pop<0>();
        const auto *volatile_st_insts =
            reinterpret_cast<const volatile MInst *>(st_insts);
        const uint64_t rope_address = volatile_st_insts[rope_slot].address;
        rope_row = reinterpret_cast<const data_t *>(rope_address);
    }
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

    // CUTLASS's SM100 epilogue policy uses the wide 16-datapath ownership
    // pattern for FP32 accumulators narrowed to 16-bit output.  It aligns the
    // M64 fragment with an stmatrix transpose store instead of scalar shared
    // writes; M128 retains its native 32-datapath mapping.
    using TmemLoad = std::conditional_t<
        M == 64, SM100_TMEM_LOAD_16dp256b1x, SM100_TMEM_LOAD_32dp32b1x>;
    TiledCopy tiled_t2r = make_tmem_copy(TmemLoad{}, tmem_acc);
    ThrCopy thr_t2r = tiled_t2r.get_slice(tid);
    auto thread_tmem = thr_t2r.partition_S(tmem_acc);
    auto thread_output = thr_t2r.partition_D(cta_output);
    auto r_acc = make_tensor<accum_t>(shape(thread_output));
    copy(tiled_t2r, thread_tmem, r_acc);

    auto r_output = make_tensor<data_t>(shape(thread_output));
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
    if constexpr (M == 64) {
        TiledCopy tiled_r2s = make_tiled_copy_D(
            Copy_Atom<SM90_U16x4_STSM_T, data_t>{}, tiled_t2r);
        ThrCopy thr_r2s = tiled_r2s.get_slice(tid);
        auto thread_r2s_output = thr_r2s.partition_D(cta_output);
        auto r2s_output = thr_r2s.retile_S(r_output);
        copy(tiled_r2s, r2s_output, thread_r2s_output);
    } else {
        copy(r_output, thread_output);
    }

    if constexpr (ApplyRope) {
        // RoPE is linear, so rotate each K-fold partial before the output TMA
        // reduction. Keep the UMMA result on-SM: TMEM -> shared epilogue ->
        // one final reduction/store, with no materialized projection round-trip.
        __sync_compute_group(128);
        #pragma unroll
        for (int pair_idx = tid; pair_idx < M * N / 2; pair_idx += 128) {
            const int batch = pair_idx / (M / 2);
            const int pair = pair_idx % (M / 2);
            const int even_row = pair * 2;
            const float even = static_cast<float>(s_output(even_row, batch));
            const float odd = static_cast<float>(s_output(even_row + 1, batch));
            const float cosine = static_cast<float>(
                rope_row[rope_head_offset + even_row]);
            const float sine = static_cast<float>(
                rope_row[rope_head_offset + even_row + 1]);
            s_output(even_row, batch) = data_t(even * cosine - odd * sine);
            s_output(even_row + 1, batch) = data_t(even * sine + odd * cosine);
        }
        __sync_compute_group(128);
    }

    if constexpr (Residual) {
        c2m.push(tid, residual_slot);
    }
    c2m.template push<0, true>(tid, output_slot);
}

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
    task_gemv_sm100_impl<M, N, K, BLoadInterval, Residual, false>(
        n_k_tiles, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
        base, m2c, c2m, nullptr, 0);
}

template<int M, int N, int K, int BLoadInterval,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv_sm100_rope(
    const int n_k_tiles,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    const void *base,
    M2C_Type &m2c,
    C2M_Type &c2m,
    const MInst *st_insts,
    const int rope_head_offset) {
    task_gemv_sm100_impl<M, N, K, BLoadInterval, false, true>(
        n_k_tiles, tmem_base_ptr, tmem_mma_barrier, tmem_mma_phase,
        base, m2c, c2m, st_insts, rope_head_offset);
}

// LM-head specialization: reuse each eight-token B tile across four disjoint
// output tiles and retain the four F32 accumulators in TMEM.  The epilogue
// either drains BF16 logits directly to global memory or emits compact argmax
// records without materializing logits.
template<int M, int N, int K, int BLoadInterval, int OutputGroups,
         bool FuseArgmax,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv_sm100_direct_grouped(
    const int n_k_tiles,
    uint32_t tmem_base_ptr,
    uint64_t *tmem_mma_barrier,
    uint32_t &tmem_mma_phase,
    const void *base,
    M2C_Type &m2c,
    C2M_Type &c2m,
    MInst *st_insts,
    const int output_stride,
    const int output_group_stride,
    void *reduction_scratch = nullptr,
    const int vocabulary_base = 0,
    const int partial_stride = 0) {
    using namespace cute;
    using data_t = cutlass::bfloat16_t;
    using accum_t = float;
    using Atom = SM100_MMA_F16BF16_SS<
        data_t, data_t, accum_t, M, N, UMMA::Major::K, UMMA::Major::K>;

    static_assert(M == 128 && N == 8,
                  "Grouped direct GEMV is specialized for M128N8");
    static_assert(K % 16 == 0, "SM100 BF16 UMMA requires K to be a multiple of 16");
    static_assert(BLoadInterval > 0, "BLoadInterval must be positive");
    static_assert(OutputGroups > 0, "SM100 GEMV needs at least one output group");

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

    int live_b_slot = 0;
    constexpr int b_tile_elements = N * K;

    for (int tile_idx = 0; tile_idx < n_k_tiles; ++tile_idx) {
        data_t *sB_ptr;
        if (tile_idx % BLoadInterval == 0) {
            live_b_slot = m2c.template pop<0>();
            sB_ptr = static_cast<data_t *>(
                get_slot_address(base, extract(live_b_slot)));
        } else {
            sB_ptr = static_cast<data_t *>(
                get_slot_address(base, extract(live_b_slot)))
                + (tile_idx % BLoadInterval) * b_tile_elements;
        }

        #pragma unroll
        for (int output_group = 0; output_group < OutputGroups;
             ++output_group) {
            const int slot_a = m2c.template pop<0>();
            data_t *sA_ptr = static_cast<data_t *>(
                get_slot_address(base, extract(slot_a)));
            auto sA = make_tensor(make_smem_ptr(sA_ptr), layout_sA);
            auto sB = make_tensor(make_smem_ptr(sB_ptr), layout_sB);
            auto frag_a = cta_mma.make_fragment_A(sA);
            auto frag_b = cta_mma.make_fragment_B(sB);
            auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
            group_tmem_acc.data() = tmem_base_ptr + output_group * N;
            tiled_mma.accumulate_ = tile_idx == 0
                                  ? UMMA::ScaleOut::Zero
                                  : UMMA::ScaleOut::One;

            if (tid < numThreadsPerWarp) {
                #pragma unroll
                for (int k_block = 0; k_block < size<2>(frag_a); ++k_block) {
                    gemm(tiled_mma, frag_a(_, _, k_block),
                         frag_b(_, _, k_block), group_tmem_acc);
                    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                }
                cutlass::arch::umma_arrive(tmem_mma_barrier);
            }
            cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
            tmem_mma_phase ^= 1;

            int release_mask = slot_a;
            if (output_group + 1 == OutputGroups
                && ((tile_idx + 1) % BLoadInterval == 0
                    || tile_idx + 1 == n_k_tiles)) {
                release_mask |= live_b_slot;
            }
            c2m.push(tid, release_mask);
        }
    }

    auto coord_c = make_identity_tensor(make_shape(Int<M>{}, Int<N>{}));
    auto cta_coord_c = cta_mma.partition_C(coord_c);
    using TmemLoad = SM100_TMEM_LOAD_32dp32b4x;
    data_t local_max[N];
    long long local_idx[N];
    if constexpr (FuseArgmax) {
        static_assert(N == 8, "fused argmax expects the eight-token decode tile");
        #pragma unroll
        for (int col = 0; col < N; ++col) {
            local_max[col] = data_t(-FLT_MAX);
            local_idx[col] = -1;
        }
    }

    int output_slot = -1;
    data_t *output_ptr = nullptr;
    if constexpr (!FuseArgmax) {
        output_slot = m2c.template pop<0>();
        output_ptr = static_cast<data_t *>(
            slot_2_glob_ptr(st_insts, output_slot));
    }
    #pragma unroll
    for (int output_group = 0; output_group < OutputGroups; ++output_group) {
        auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
        group_tmem_acc.data() = tmem_base_ptr + output_group * N;
        TiledCopy tiled_t2r = make_tmem_copy(TmemLoad{}, group_tmem_acc);
        ThrCopy thr_t2r = tiled_t2r.get_slice(tid);
        auto thread_tmem = thr_t2r.partition_S(group_tmem_acc);
        auto thread_coord = thr_t2r.partition_D(cta_coord_c);
        auto r_acc = make_tensor<accum_t>(shape(thread_coord));
        copy(tiled_t2r, thread_tmem, r_acc);

        #pragma unroll
        for (int i = 0; i < size(r_acc); ++i) {
            const int row = int(get<0>(thread_coord(i)));
            const int col = int(get<1>(thread_coord(i)));
            const data_t candidate = data_t(r_acc(i));
            if constexpr (FuseArgmax) {
                if (candidate > local_max[col]) {
                    local_max[col] = candidate;
                    local_idx[col] = vocabulary_base
                                   + output_group * output_group_stride + row;
                }
            } else {
                output_ptr[col * output_stride
                           + output_group * output_group_stride + row]
                    = candidate;
            }
        }
    }

    if constexpr (FuseArgmax) {
        const int partial_slot = m2c.template pop<0>();
        FusedArgmaxRecord *partials = static_cast<FusedArgmaxRecord *>(
            slot_2_glob_ptr(st_insts, partial_slot));
        WarpArgmaxRecord *warp_partials =
            static_cast<WarpArgmaxRecord *>(reduction_scratch);
        constexpr int n_warps = 4;
        const int lane_id = tid % numThreadsPerWarp;
        const int warp_id = tid / numThreadsPerWarp;

        // Fill the complete 4-warp x 8-token scratch tile before any warp
        // reuses it.  At 8 bytes/record this exactly fits the 256-byte runtime
        // scratch allocation.
        #pragma unroll
        for (int col = 0; col < N; ++col) {
            warp_reduce_max_idx(local_max[col], local_idx[col]);
            if (lane_id == 0) {
                warp_partials[col * n_warps + warp_id] = {
                    float(local_max[col]), int(local_idx[col])};
            }
        }
        __sync_compute_group(128);

        if (tid == 0) {
            #pragma unroll
            for (int col = 0; col < N; ++col) {
                data_t block_max = data_t(-FLT_MAX);
                long long block_idx = -1;
                #pragma unroll
                for (int warp = 0; warp < n_warps; ++warp) {
                    const WarpArgmaxRecord candidate =
                        warp_partials[col * n_warps + warp];
                    if (data_t(candidate.value) > block_max) {
                        block_max = data_t(candidate.value);
                        block_idx = candidate.index;
                    }
                }
                partials[col * partial_stride].value = block_max;
                partials[col * partial_stride].index = block_idx;
            }
        }

        // Join only the compute threads.  The memory warp observes completion
        // through C2M after all records are globally stored.
        __sync_compute_group(128);
        c2m.template push<31, true, false>(tid, 1U << partial_slot);
    } else {
        c2m.template push<31, true, false>(tid, 1U << output_slot);
    }
}

// Projection specialization. Each task reuses B across four M128 output tiles
// and packs all four epilogues into one 8 KiB shared slot for a single
// strided rank-4 TMA reduction.
template<int M, int N, int K, int BLoadInterval, int OutputGroups,
         typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_gemv_sm100_grouped_reduce(
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

    static_assert(M == 128 && N == 8,
                  "Grouped reduction GEMV is specialized for M128N8");
    static_assert(OutputGroups == 4,
                  "Grouped reduction GEMV is specialized for four outputs");

    const int tid = __compute_tid();
    auto tiled_mma = make_tiled_mma(Atom{});
    auto cta_mma = tiled_mma.get_slice(0);
    auto mma_shape_a = partition_shape_A(
        tiled_mma, make_shape(Int<M>{}, Int<K>{}));
    auto mma_shape_b = partition_shape_B(
        tiled_mma, make_shape(Int<N>{}, Int<K>{}));
    auto layout_sA = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, mma_shape_a);
    auto layout_sB = UMMA::tile_to_mma_shape(
        UMMA::Layout_K_SW128_Atom<data_t>{}, mma_shape_b);

    auto logical_c = make_tensor(
        make_smem_ptr(static_cast<accum_t *>(nullptr)),
        make_layout(make_shape(Int<M>{}, Int<N>{}),
                    make_stride(Int<N>{}, Int<1>{})));
    auto cta_c = cta_mma.partition_C(logical_c);

    int live_b_slot = 0;
    constexpr int b_tile_elements = N * K;
    for (int tile_idx = 0; tile_idx < n_k_tiles; ++tile_idx) {
        data_t *sB_ptr;
        if (tile_idx % BLoadInterval == 0) {
            live_b_slot = m2c.template pop<0>();
            sB_ptr = static_cast<data_t *>(
                get_slot_address(base, extract(live_b_slot)));
        } else {
            sB_ptr = static_cast<data_t *>(
                get_slot_address(base, extract(live_b_slot)))
                + (tile_idx % BLoadInterval) * b_tile_elements;
        }

        #pragma unroll
        for (int output_group = 0; output_group < OutputGroups;
             ++output_group) {
            const int slot_a = m2c.template pop<0>();
            data_t *sA_ptr = static_cast<data_t *>(
                get_slot_address(base, extract(slot_a)));
            auto sA = make_tensor(make_smem_ptr(sA_ptr), layout_sA);
            auto sB = make_tensor(make_smem_ptr(sB_ptr), layout_sB);
            auto frag_a = cta_mma.make_fragment_A(sA);
            auto frag_b = cta_mma.make_fragment_B(sB);
            auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
            group_tmem_acc.data() = tmem_base_ptr + output_group * N;
            tiled_mma.accumulate_ = tile_idx == 0
                                  ? UMMA::ScaleOut::Zero
                                  : UMMA::ScaleOut::One;

            if (tid < numThreadsPerWarp) {
                #pragma unroll
                for (int k_block = 0; k_block < size<2>(frag_a); ++k_block) {
                    gemm(tiled_mma, frag_a(_, _, k_block),
                         frag_b(_, _, k_block), group_tmem_acc);
                    tiled_mma.accumulate_ = UMMA::ScaleOut::One;
                }
                cutlass::arch::umma_arrive(tmem_mma_barrier);
            }
            cute::wait_barrier(*tmem_mma_barrier, tmem_mma_phase);
            tmem_mma_phase ^= 1;

            int release_mask = slot_a;
            if (output_group + 1 == OutputGroups
                && ((tile_idx + 1) % BLoadInterval == 0
                    || tile_idx + 1 == n_k_tiles)) {
                release_mask |= live_b_slot;
            }
            c2m.push(tid, release_mask);
        }
    }

    auto layout_output = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<M>{}, Int<N>{}));
    // Drain a complete M128N8 fragment per pass.  The narrower x1 form needs
    // four times as many TMEM load instructions for every grouped epilogue.
    using TmemLoad = SM100_TMEM_LOAD_32dp32b4x;
    const int output_slot = m2c.template pop<0>();
    data_t *output_base = static_cast<data_t *>(
        get_slot_address(base, extract(output_slot)));
    #pragma unroll
    for (int output_group = 0; output_group < OutputGroups; ++output_group) {
        data_t *output_ptr = output_base + output_group * M * N;
        auto s_output = make_tensor(make_smem_ptr(output_ptr), layout_output);
        auto cta_output = cta_mma.partition_C(s_output);
        auto group_tmem_acc = cta_mma.make_fragment_C(cta_c);
        group_tmem_acc.data() = tmem_base_ptr + output_group * N;
        TiledCopy tiled_t2r = make_tmem_copy(TmemLoad{}, group_tmem_acc);
        ThrCopy thr_t2r = tiled_t2r.get_slice(tid);
        auto thread_tmem = thr_t2r.partition_S(group_tmem_acc);
        auto thread_output = thr_t2r.partition_D(cta_output);
        auto r_acc = make_tensor<accum_t>(shape(thread_output));
        copy(tiled_t2r, thread_tmem, r_acc);
        auto r_output = make_fragment_like(thread_output);
        #pragma unroll
        for (int i = 0; i < size(r_output); ++i) {
            r_output(i) = data_t(r_acc(i));
        }
        copy(r_output, thread_output);
    }
    c2m.template push<0, true>(tid, output_slot);
}
#endif
