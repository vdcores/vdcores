#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm
#include <cute/layout.hpp>

int main() {
    using namespace cute;
    constexpr int M = 128, N = 128, K = 128;
    using Atom = SM90_64x128x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>;

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
    GMMA::Layout_K_SW128_Atom<data_t> gmma_atom_layout;
    auto layout_sA = tile_to_shape(
        gmma_atom_layout,
        make_shape(Int<512>{},Int<256>{}));
    auto layout_sB = tile_to_shape(
        gmma_atom_layout,
        make_shape(Int<512>{},Int<256>{}));
    
    printf("\n GMMA layouts:\n");
    print(gmma_atom_layout);
    printf("\n layout_sA:\n");
    print(layout_sA);
    printf("\n c layout_sA:\n");
    print(coalesce(layout_sA));
    printf("\n layout_sB:\n");
    print(layout_sB);
    printf("\n c layout_sB:\n");
    print(coalesce(layout_sB));
    printf("\n");

    using kernel_PV = cute::SM90_64x64x16_F32F16F16_RS<cute::GMMA::Major::K, cute::GMMA::Major::MN>;
    using AtomTrait_PV = MMA_Traits<kernel_PV>;
    using data_t_PV = typename AtomTrait_PV::ValTypeA;
    using accum_t_PV = typename AtomTrait_PV::ValTypeC;

    auto layout_sV = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<accum_t_PV>{},
        make_shape(Int<64>{},Int<128>{}));
    void *base;
    auto sV = make_tensor(make_smem_ptr(()), layout_sV);
    auto frag_V = thr_mma_pv.partition_fragment_B(sV);
    return 0;
}