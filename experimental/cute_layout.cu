#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm
#include <cute/layout.hpp>

using namespace cute;

template<typename Layout>
__host__ void playout(const char* name, const Layout& layout) {
    printf("Layout %s:\n", name);
    print(layout);
    printf("\n");
}

void test_k16_layout() {
    using AtomRS = SM90_64x16x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
    using AtomTraitRS = MMA_Traits<AtomRS>;
    // This atom is for P @ V, which means (QT, KVT) @ (KVT, HDIM) = (QT, HDIM)
    constexpr int M = 64, N = 16, K = 128;
    auto tiled_mma_rs = make_tiled_mma(
        MMA_Atom<AtomRS>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );

    auto layout_sA = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<typename AtomTraitRS::ValTypeC>{},
        Shape<Int<M>, Int<K>>{});
    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<typename AtomTraitRS::ValTypeC>{},
        Shape<Int<N>, Int<K>>{});
    auto layout_sC = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<typename AtomTraitRS::ValTypeC>{},
        Shape<Int<M>, Int<N>>{});

    playout("layout_sA", layout_sA);
    playout("layout_sB", layout_sB);
    playout("layout_sC", layout_sC);
}

int main() {
    using namespace cute;
    constexpr int M = 128, N = 64, K = 128;
    using Atom = SM90_64x64x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>;

    using AtomTrait = MMA_Traits<Atom>;

    using data_t = typename AtomTrait::ValTypeA;
    using accum_t = typename AtomTrait::ValTypeC;

    constexpr int MMA_M = shape<0>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_N = shape<1>(typename AtomTrait::Shape_MNK{});
    constexpr int MMA_K = shape<2>(typename AtomTrait::Shape_MNK{});

    static_assert(M % MMA_M == 0, "Only M multiple of 64 is supported");
    static_assert(N % MMA_N == 0, "Only N multiple of 256 is supported");
    static_assert(K % MMA_K == 0, "Only K multiple of 16 is supported");

    // Both A and B are in shared memory, C in register 
    auto tiled_mma = make_tiled_mma(
        MMA_Atom<Atom>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<M>{}, Int<N>{}, Int<K>{}) // tile along the M, N dims
    );
    auto thr_mma   = tiled_mma.get_slice(0);

    // this layout should match the TMA load layout
    auto layout_sA = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        make_shape(Int<N>{},Int<K>{}));
    auto layout_svB = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<__nv_bfloat162>{},
        make_shape(N,Int<K/2>{}));
    auto layout_output_C = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<N>{}));
    auto layout_sC_vec = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<__nv_bfloat162>{},
        make_shape(Int<M/2>{},Int<N>{}));

    printf("\n layout_sA:\n");
    print(layout_sA);
    printf("\n layout_sB:\n");
    print(layout_sB);
    printf("\n layout_svB:\n");
    print(layout_svB);
    printf("\n layout_sB(15):\n");
    print(layout_sB(15));
    printf("\n layout_output_C:\n");
    print(layout_output_C);
    printf("\n layout_sC_vec:\n");
    print(layout_sC_vec);
    auto swizzle_mode = Swizzle<3,4,3>{};
    printf("\n swiz128[128] \n");
    print(swizzle_mode(128));
    
#if 0
    accum_t *base;
    Tensor t_sC = make_tensor(make_smem_ptr(base), make_shape(Int<M>{}, Int<N>{}));

    auto frag_C = thr_mma.partition_fragment_C(t_sC);

    auto c_tv = tiled_mma.get_layoutC_TV();
    auto layout_m = make_layout(
        make_shape(Int<M>{}, Int<N>{}),
        make_stride(Int<1>{}, Int<0>{}));
    
    auto tv2m = composition(layout_m, c_tv);
    auto reg2m = coalesce(select<1>(tv2m));
    auto ns = nullspace(reg2m); // get the "kern" of the mapping

    auto c_fragC = coalesce(frag_C);
    auto div_c = logical_divide(c_fragC, ns);

    auto reg2m_coal = coalesce(filter_zeros(reg2m));
    auto tv2m_coal = make_layout(
        coalesce(filter_zeros(select<0>(tv2m))),
        reg2m_coal
    );

    // usage example:
    // data_t m[M]; // could be smem pointer from pop
    // buf_max = make_tensor(make_smem_ptr(m), tv2m_coal);
    // // in each thread
    // copy(row_max, buf_max(thread_idx / 4, _));
    
    printf("\n c_tv:\n");
    print(c_tv);
    printf("\n tv2m:\n");
    print(tv2m);
    printf("\n reg2m:\n");
    print(reg2m);
    printf("\n reg2m_coal:\n");
    print(reg2m_coal);
    printf("\n tv2m_coal:\n");
    print(tv2m_coal);
    printf("\n nullspace(reg2m):\n");
    print(ns);
    printf("\n frag_C:\n");
    print(frag_C);
    printf("\n tiled:\n");
    print(div_c);
    printf("\n nrow tiled:\n");
    print(size<1>(div_c));
    printf("\n tiled 0:\n");
    print(div_c(_, 0));
    printf("\n tiled 1:\n");
    print(div_c(_, 1));
    printf("\n tiled (0, _), 1:\n");
    print(div_c(16, 0));
    printf("\n compl:\n");
    auto coset = complement(ns, c_fragC);
    print(coset);
    printf("\n");
    exit(0);

    // example code to reduce within layout

    // constexpr int num_row_per_thread = size(coset);
    // half_t nm[num_row_per_thread];
    // for (int i = 0; i < num_row_per_thread; i++) {
    //     auto t_i = div_c(_, i);
    //     nm[i] = reduce(t_i, 0, cute::max);
    // }

    // part2: K-major layouts
    print("\npart 2: K-major layouts:\n");
    constexpr int HEAD_DIM = 128;
    constexpr int Q_TILE_SIZE = 64;
    constexpr int KV_BLOCK_SIZE = 64;
    auto layout_sQ = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        Shape<Int<Q_TILE_SIZE>, Int<HEAD_DIM>>{});
    auto layout_sK = tile_to_shape(
        GMMA::Layout_K_SW128_Atom<data_t>{},
        Shape<Int<Q_TILE_SIZE>, Int<HEAD_DIM>>{});

    // note that the output layout is in MN major
    printf("\n layout_sQ:\n"); // Sw<3,4,3> o smem_ptr[16b](unset) o (_64,_64,_2):(_64,_1,_4096)
    print(coalesce(layout_sQ));
    printf("\n layout_sK:\n");
    print(coalesce(layout_sK));
    printf("\n");

    // example python TMA parameter 
    // (headdim,qtile)=(0,0) -> layout(0,0) = smem[0]
    // (headdim,qtile)=(0,1) -> layout(0,1) = smem[1] 

    // tensor = torch.rand(NUM_REQ * NUM_HEAD, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=gpu) - 0.5
    // global_dims = [64, SEQ_LEN, HEAD_DIM / 64, NUM_HEAD * NUM_REQ]
    // global_strides = [HEAD_DIM, 64, HEAD_DIM *SEQ_LEN] * elsize
    // box_dims = [64, Q_TILE_SIZE, HEAD_DIM / 64]
    // box_strides = [1, 1, 1, 1]

    // part 3: the output off a _SS MMA and the input of a _RS MMA
    print("\npart 3: MMA_SS output / MMA_RS input kv layout:\n");
    using AtomQK = SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::K>;
    auto tiled_mma_qk = make_tiled_mma(
        MMA_Atom<AtomQK>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<Q_TILE_SIZE>{}, Int<KV_BLOCK_SIZE>{}, Int<HEAD_DIM>{}) // tile along the M, N dims
    );
    print("\n tiled QK Layout:\n");
    print(tiled_mma_qk);
    printf("\n");
    auto layout_qk_result = coalesce(tiled_mma_qk.get_layoutC_TV(), Step<_1,_1>{});
    auto layout_qk_result_mn2tv = left_inverse(layout_qk_result);
    print("\n layout_qk_result_mn2tv (M=%d, N=%d):\n", Q_TILE_SIZE, KV_BLOCK_SIZE);
    print(layout_qk_result_mn2tv);
    printf("\n");


    using AtomRS = SM90_64x64x16_F16F16F16_RS<GMMA::Major::K, GMMA::Major::MN>;
    using AtomTraitRS = MMA_Traits<AtomRS>;
    // This atom is for P @ V, which means (QT, KVT) @ (KVT, HDIM) = (QT, HDIM)
    auto tiled_mma_rs = make_tiled_mma(
        MMA_Atom<AtomRS>{},
        make_layout(make_shape(Int<1>{}, Int<1>{}, Int<1>{})), // only one warp group
        make_tile(Int<Q_TILE_SIZE>{}, Int<HEAD_DIM>{}, Int<KV_BLOCK_SIZE>{}) // tile along the M, N dims
    );
    print("\n tiled PV Layout:\n");
    print(tiled_mma_rs);
    printf("\n");
    auto layout_pv_A = coalesce(tiled_mma_rs.get_layoutA_TV(), Step<_1,_1>{});
    auto layout_pv_A_mn2tv = left_inverse(layout_pv_A);
    print("\n layout_pv_A_mn2tv (M=%d, K=%d):\n", Q_TILE_SIZE, KV_BLOCK_SIZE);
    print(layout_pv_A_mn2tv);
    printf("\n");
    
    Tensor t_dummyS = make_tensor(
        make_smem_ptr((accum_t*)nullptr),
        make_shape(Int<Q_TILE_SIZE>{}, Int<KV_BLOCK_SIZE>{})
    );
    printf("\nSS output layout:\n");
    auto frag_P = tiled_mma_qk.get_slice(0).partition_fragment_C(t_dummyS);
    print(frag_P);
    printf("\nRS slice A layout:\n");
    auto frag_S = tiled_mma_rs.get_slice(0).partition_fragment_A(t_dummyS);
    print(frag_S);
    printf("\n");

    printf("\n");
    test_k16_layout();
#endif
}