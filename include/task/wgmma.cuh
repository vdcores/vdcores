#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm

#include "context.cuh"

// TODO(zhiyuang): this is a gemv style wgmma, not tile overN but prefetch K tiles
template<int M, int N, int K, typename Atom, typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_wgmma_prefetch(
    const int nPrefetch,
    const int nKTiles, 
    const bool addResidual,
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

    assert(blockDim.x >= 128 && "At least 128 threads are required for wgmma_m64n256k16");

    // tile size: M = 64 * 16 * sizeof(half_t) = 2KB
    //            N = 256 * 16 * sizeof(half_t) = 8KB
    //            output C = 64 * 256 * sizeof(half_t) = 32KB 

    static_assert(M % MMA_M == 0, "Only M multiple of 64 is supported");
    static_assert(N % MMA_N == 0, "Only N multiple of 256 is supported");
    static_assert(K % MMA_K == 0, "Only K multiple of 16 is supported");

    int thread_id = threadIdx.x;

    // Both A and B are in shared memory, C in register 
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<Atom>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );
    auto thr_mma_qk   = tiled_mma.get_slice(thread_id);

    // this layout should match the TMA load layout
    auto layout_sA = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<N>{},Int<K>{}));
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<accum_t>{},
        make_shape(Int<M>{},Int<N>{}));

    // (M, N), M major
    Tensor t_dummyC = make_tensor(make_smem_ptr((accum_t*)base), Shape<Int<M>, Int<N>>{});

    // TODO(zhiyuang): do we need to allocate this earlier?
    // No since a and b are getting freed
    auto frag_C = thr_mma_qk.partition_fragment_C(t_dummyC);
    clear(frag_C);                                      // C = 0

    // prefetch nPrefetch tiles on matrix side, which is A
    constexpr int maxPrefetch = 16;
    assert(nPrefetch <= maxPrefetch && "nPrefetch exceeds maxPrefetch");
    assert(nPrefetch > 0 && "nPrefetch should be greater than 0");
    short prefetch_slots[maxPrefetch];
    for (int i = 0; i < nPrefetch; i++) {
        prefetch_slots[i] = m2c.template pop<0>();
    }

    int slot_a = prefetch_slots[0];
    data_t* sa = (data_t *)get_slot_address(base, slot_a);
    int slot_b = m2c.template pop<0>();
    data_t* sb = (data_t *)get_slot_address(base, slot_b);

    // TODO(zhiyuang): add inflight for wait
    for (int i = 0; i < nKTiles; i++) {
        // load A and B from shared memory
        int old_slot_a = slot_a;
        int old_slot_b = slot_b;

        auto sA = make_tensor(make_smem_ptr(sa), layout_sA);
        auto sB = make_tensor(make_smem_ptr(sb), layout_sB);

        auto frag_A = thr_mma_qk.partition_fragment_A(sA);
        auto frag_B = thr_mma_qk.partition_fragment_B(sB);

        warpgroup_arrive();
        gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);   // C = A*B + C
        // TODO(zhiyuang): add fence here?
        cuda::ptx::fence_proxy_async();
        warpgroup_commit_batch();

        // async load next tiles
        if (i < nKTiles - 1) {
            if (i + 1 < nPrefetch)
                slot_a = prefetch_slots[i + 1];
            else 
                slot_a = m2c.template pop<0>();

            slot_b = m2c.template pop<0>();
            sa = (data_t *)get_slot_address(base, slot_a);
            sb = (data_t *)get_slot_address(base, slot_b);
        }

        // TODO(zhiyuang): keep 1 wgmma inflight
        warpgroup_wait<0>();

        c2m.push(thread_id, old_slot_a);
        c2m.push(thread_id, old_slot_b);
    }

    // load residual from shared memory
    // assume we always have back to a data_t
    int slot_r = m2c.template pop<0>();
    data_t* sr = (data_t *)get_slot_address(base, slot_r);

    // TODO(zhiyuang): verify the layout here should we do menual tiling?
    // TODO(zhiyuang): this causes slow down when Ntiles increasing, why?
    Tensor t_sR = make_tensor(
        make_smem_ptr((data_t*)sr),
        tile_to_shape(
            GMMA::Layout_MN_SW128_Atom<data_t>{},
            make_shape(Int<M>{},Int<N>{}))
    );
    auto thr_sR =  thr_mma_qk.partition_C(t_sR);

    // TODO(zhiyuang): vectorized add based on the layout
    if (addResidual) {
        #pragma unroll
        for (int i = 0; i < size(frag_C); i++) {
            frag_C[i] = frag_C[i] + (accum_t)thr_sR[i];
        }
    }

    // copy back to shared memory
    // TODO(zhiyuang): add fence after this?
    copy(frag_C, thr_sR);
    // TODO: do we need synchronize warpgroup before returning slot?
    __sync_compute_group(numThreads);
    // TODO(zhiyuang) necessary barrier/synchronization?
    c2m.push(thread_id, slot_r);

    __cprint("[WGMMA Prefetch] Completed nPrefetch=%d", nPrefetch);
}

// TODO(zhiyuang): make this kernel tile unrolled over M
template<int M, int N, int K, typename Atom, typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_wgmma_residual(
    const int nOutputTiles,
    const int nKTiles, 
    const bool addResidual,
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

    assert(blockDim.x >= 128 && "At least 128 threads are required for wgmma_m64n256k16");

    // tile size: M = 64 * 16 * sizeof(half_t) = 2KB
    //            N = 256 * 16 * sizeof(half_t) = 8KB
    //            output C = 64 * 256 * sizeof(half_t) = 32KB 

    static_assert(M % MMA_M == 0, "Only M multiple of 64 is supported");
    static_assert(N % MMA_N == 0, "Only N multiple of 256 is supported");
    static_assert(K % MMA_K == 0, "Only K multiple of 16 is supported");

    int thread_id = threadIdx.x;

    // Both A and B are in shared memory, C in register 
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<Atom>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );
    auto thr_mma_qk   = tiled_mma.get_slice(threadIdx.x);

    // this layout should match the TMA load layout
    auto layout_sA = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<N>{},Int<K>{}));
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<accum_t>{},
        make_shape(Int<M>{},Int<N>{}));

    // (M, N), M major
    Tensor t_dummyC = make_tensor(make_smem_ptr((accum_t*)base), Shape<Int<M>, Int<N>>{});

    // TODO(zhiyuang): do we need to allocate this earlier?
    // No since a and b are getting freed
    auto frag_C = thr_mma_qk.partition_fragment_C(t_dummyC);
    for (int tile = 0; tile < nOutputTiles; tile++) {
        clear(frag_C);                                      // C = 0

        // loop over tiles of matrix, each tile is of size (M x N)
        int slot_a = m2c.template pop<0>();
        data_t* sa = (data_t *)get_slot_address(base, slot_a);
        int slot_b = m2c.template pop<0>();
        data_t* sb = (data_t *)get_slot_address(base, slot_b);

        // TODO(zhiyuang): add inflight for wait
        for (int i = 0; i < nKTiles; i++) {
            // load A and B from shared memory
            int old_slot_a = slot_a;
            int old_slot_b = slot_b;

            auto sA = make_tensor(make_smem_ptr(sa), layout_sA);
            auto sB = make_tensor(make_smem_ptr(sb), layout_sB);

            auto frag_A = thr_mma_qk.partition_fragment_A(sA);
            auto frag_B = thr_mma_qk.partition_fragment_B(sB);

            warpgroup_arrive();
            gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);   // C = A*B + C
            // TODO(zhiyuang): add fence here?
            cuda::ptx::fence_proxy_async();
            warpgroup_commit_batch();

            // async load next tiles
            if (i < nKTiles - 1) {
                slot_a = m2c.template pop<0>();
                slot_b = m2c.template pop<0>();
                sa = (data_t *)get_slot_address(base, slot_a);
                sb = (data_t *)get_slot_address(base, slot_b);
            }

            warpgroup_wait<0>();

            c2m.push(thread_id, old_slot_a);
            c2m.push(thread_id, old_slot_b);
        }

        // load residual from shared memory
        // assume we always have back to a data_t
        int slot_r = m2c.template pop<0>();
        data_t* sr = (data_t *)get_slot_address(base, slot_r);

        // TODO(zhiyuang): verify the layout here should we do menual tiling?
        // TODO(zhiyuang): this causes slow down when Ntiles increasing, why?
        Tensor t_sR = make_tensor(
            make_smem_ptr((data_t*)sr),
            tile_to_shape(
                GMMA::Layout_MN_SW128_Atom<data_t>{},
                make_shape(Int<M>{},Int<N>{}))
        );
        auto thr_sR =  thr_mma_qk.partition_C(t_sR);

        // TODO(zhiyuang): vectorized add based on the layout
        if (addResidual) {
          #pragma unroll
          for (int i = 0; i < size(frag_C); i++) {
              frag_C[i] = frag_C[i] + (accum_t)thr_sR[i];
          }
        }

        // copy back to shared memory
        // TODO(zhiyuang): add fence after this?
        copy(frag_C, thr_sR);
        // TODO: do we need synchronize warpgroup before returning slot?
        __sync_compute_group(numThreads);
        // TODO(zhiyuang) necessary barrier/synchronization?
        c2m.push(thread_id, slot_r);
    }
    __cprint("[WGMMA] Completed all %d output tiles", nOutputTiles);
}


// TODO(zhiyuang): make this kernel tile unrolled over M
template<int M, int N, int K, typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_wgmma_m64n256k16(
    const int nOutputTiles,
    const int nKTiles, 
    void *base, 
    M2C_Type& m2c, 
    C2M_Type& c2m
) {
    using namespace cute;

    constexpr int MMA_M = 64, MMA_N = 256, MMA_K = 16;
    assert(blockDim.x >= 128 && "At least 128 threads are required for wgmma_m64n256k16");

    // tile size: M = 64 * 16 * sizeof(half_t) = 2KB
    //            N = 256 * 16 * sizeof(half_t) = 8KB
    //            output C = 64 * 256 * sizeof(half_t) = 32KB 

    static_assert(M % MMA_M == 0, "Only M multiple of 64 is supported");
    static_assert(N % MMA_N == 0, "Only N multiple of 256 is supported");
    static_assert(K % MMA_K == 0, "Only K multiple of 16 is supported");

    int thread_id = threadIdx.x;

    // Both A and B are in shared memory, C in register 
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<
          SM90_64x256x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>
        >{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );
    auto thr_mma_qk   = tiled_mma.get_slice(threadIdx.x);

    // this layout should match the TMA load layout
    auto layout_sA = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<N>{},Int<K>{}));
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<half_t>{},
        make_shape(Int<M>{},Int<N>{}));

    // (M, N), M major
    Tensor t_dummyC = make_tensor(
        make_smem_ptr((half_t*)base),
        layout_sC
    );

    // TODO(zhiyuang): do we need to allocate this earlier?
    // No since a and b are getting freed
    auto frag_C = thr_mma_qk.partition_fragment_C(t_dummyC);
    for (int tile = 0; tile < nOutputTiles; tile++) {
        clear(frag_C);                                      // C = 0

        // loop over tiles of matrix, each tile is of size (M x N)
        int slot_a = m2c.template pop<0>();
        half_t* sa = (half_t *)get_slot_address(base, slot_a);
        int slot_b = m2c.template pop<0>();
        half_t* sb = (half_t *)get_slot_address(base, slot_b);

        // TODO(zhiyuang): add inflight for wait
        for (int i = 0; i < nKTiles; i++) {
            // load A and B from shared memory
            int old_slot_a = slot_a;
            int old_slot_b = slot_b;

            auto sA = make_tensor(make_smem_ptr(sa), layout_sA);
            auto sB = make_tensor(make_smem_ptr(sb), layout_sB);

            auto frag_A = thr_mma_qk.partition_fragment_A(sA);
            auto frag_B = thr_mma_qk.partition_fragment_B(sB);

            warpgroup_arrive();
            gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);   // C = A*B + C
            // TODO(zhiyuang): add fence here?
            cuda::ptx::fence_proxy_async();
            warpgroup_commit_batch();

            // async load next tiles
            if (i < nKTiles - 1) {
                slot_a = m2c.template pop<0>();
                slot_b = m2c.template pop<0>();
                sa = (half_t *)get_slot_address(base, slot_a);
                sb = (half_t *)get_slot_address(base, slot_b);
            }

            warpgroup_wait<0>();

            c2m.push(thread_id, old_slot_a);
            c2m.push(thread_id, old_slot_b);
        }

        // TODO: can we wait this eariler in the last batch?
        int slot_c = m2c.template pop<0>();
        half_t* sc = (half_t *)get_slot_address(base, slot_c);
        // (M, N), M major
        Tensor t_sC = make_tensor(make_smem_ptr(sc), layout_sC);
        auto thr_sC = thr_mma_qk.partition_C(t_sC);
        // copy back to shared memory
        // TODO(zhiyuang): add fence here
        copy(frag_C, thr_sC);
        // TODO: do we need synchronize warpgroup before returning slot?
        __sync_compute_group(numThreads);
        // TODO(zhiyuang) necessary barrier/synchronization?
        c2m.push(thread_id, slot_c);
    }
    __cprint("[WGMMA] Completed all %d output tiles", nOutputTiles);
}


template<int M, int N, int K, typename M2C_Type, typename C2M_Type>
__device__ __forceinline__ void task_wgmma_k(
    const int nOutputTiles,
    const int nKTiles, 
    void *base, 
    M2C_Type& m2c, 
    C2M_Type& c2m
) {
    using namespace cute;

    constexpr int MMA_M = 64, MMA_N = 256, MMA_K = 16;
    assert(blockDim.x >= 128 && "At least 128 threads are required for wgmma_m64n256k16");

    // tile size: M = 64 * 16 * sizeof(half_t) = 2KB
    //            N = 256 * 16 * sizeof(half_t) = 8KB
    //            output C = 64 * 256 * sizeof(half_t) = 32KB 

    static_assert(M % MMA_M == 0, "Only M multiple of 64 is supported");
    static_assert(N % MMA_N == 0, "Only N multiple of 256 is supported");
    static_assert(K % MMA_K == 0, "Only K multiple of 16 is supported");

    int thread_id = threadIdx.x;

    // Both A and B are in shared memory, C in register 
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<
          SM90_64x256x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>
        >{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );
    auto thr_mma_qk   = tiled_mma.get_slice(threadIdx.x);

    // this layout should match the TMA load layout
    auto layout_sA = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<N>{},Int<K>{}));
    auto layout_sC = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<half_t>{},
        make_shape(Int<M>{},Int<N>{}));

    // (M, N), M major
    Tensor t_dummyC = make_tensor(
        make_smem_ptr((half_t*)base),
        layout_sC
    );

    // TODO(zhiyuang): do we need to allocate this earlier?
    // No since a and b are getting freed
    auto frag_C = thr_mma_qk.partition_fragment_C(t_dummyC);
    for (int tile = 0; tile < nOutputTiles; tile++) {
        clear(frag_C);                                      // C = 0

        // loop over tiles of matrix, each tile is of size (M x N)
        int slot_a = m2c.template pop<0>();
        half_t* sa = (half_t *)get_slot_address(base, slot_a);
        int slot_b = m2c.template pop<0>();
        half_t* sb = (half_t *)get_slot_address(base, slot_b);

        // TODO(zhiyuang): add inflight for wait
        for (int i = 0; i < nKTiles; i++) {
            // load A and B from shared memory
            int old_slot_a = slot_a;
            int old_slot_b = slot_b;

            auto sA = make_tensor(make_smem_ptr(sa), layout_sA);
            auto sB = make_tensor(make_smem_ptr(sb), layout_sB);

            auto frag_A = thr_mma_qk.partition_fragment_A(sA);
            auto frag_B = thr_mma_qk.partition_fragment_B(sB);

            warpgroup_arrive();
            gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);   // C = A*B + C
            // TODO(zhiyuang): add fence here?
            cuda::ptx::fence_proxy_async();
            warpgroup_commit_batch();

            // async load next tiles
            if (i < nKTiles - 1) {
                slot_a = m2c.template pop<0>();
                slot_b = m2c.template pop<0>();
                sa = (half_t *)get_slot_address(base, slot_a);
                sb = (half_t *)get_slot_address(base, slot_b);
            }

            warpgroup_wait<0>();

            c2m.push(thread_id, old_slot_a);
            c2m.push(thread_id, old_slot_b);
        }

        // TODO: can we wait this eariler in the last batch?
        int slot_c = m2c.template pop<0>();
        half_t* sc = (half_t *)get_slot_address(base, slot_c);
        // (M, N), M major
        Tensor t_sC = make_tensor(make_smem_ptr(sc), layout_sC);
        auto thr_sC = thr_mma_qk.partition_C(t_sC);
        // copy back to shared memory
        // TODO(zhiyuang): add fence here
        copy(frag_C, thr_sC);
        // TODO: do we need synchronize warpgroup before returning slot?
        __sync_compute_group(numThreads);
        // TODO(zhiyuang) necessary barrier/synchronization?
        c2m.push(thread_id, slot_c);
    }
    __cprint("[WGMMA] Completed all %d output tiles", nOutputTiles);
}
