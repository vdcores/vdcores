#include <cute/tensor.hpp>
#include <cute/arch/mma_sm90.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm
#include <cute/layout.hpp>

using data_t = half;
static const auto data2float = __half2float;
static const auto float2data = __float2half;
#define data_add __hadd
#define data_mul __hmul


int main() {
    using namespace cute;
    constexpr int M = 64, N = 64, K = 16;
    using Atom = SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>;

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
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<M>{},Int<K>{}));
    auto layout_sB = tile_to_shape(
        GMMA::Layout_MN_SW128_Atom<data_t>{},
        make_shape(Int<N>{},Int<K>{}));
    auto layout_sC = make_layout(make_shape(Int<M>{}, Int<N>{}));

    DAELauncher dae {numSMs};
    auto &out_buf = dae.data(M * N * sizeof(data_t));
    data_t *out = out_buf.h_get<data_t>();

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            out[i + j * M] = float2data(static_cast<float>(i * N + j));
        }
    }

    Tensor t_sC = make_tensor(make_gmem_ptr(out_buf.d_get<data_t>()), make_shape(Int<M>{}, Int<N>{}));
    Tensor row_max = make_tensor<accum_t>(Shape<Int<M>>{});
    row_max.fill(-Float32.inf);

    for (int r = 0; r < M; ++ r) {
        
    }

}