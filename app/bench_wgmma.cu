#include "launcher.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>

using data_t = half;
// static const auto data2float = __half2float;
static const auto float2data = __float2half;
static void init_matrix(data_t* mat, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            mat[i + j * M] = float2data(static_cast<float>(i % 1000) / 1024.0f);
        }
    }
}

// 3342G for TK=4 on GH200 - could run longer for more accurate measurement
int main(int argc, char** argv) {

    constexpr int numSMs = 128;        // Just 1 SM for easy debugging

    constexpr uint32_t M = 64 * 128, N = 256, K = 64 * 96;

    constexpr uint32_t TileM = 64, TileN = 256, TileK = 64;
    constexpr uint32_t loadBytesA = TileM * TileK * sizeof(half);
    constexpr uint32_t loadBytesB = TileK * TileN * sizeof(half);
    constexpr uint32_t loadBytesC = TileM * TileN * sizeof(half);
    constexpr uint32_t blockNSize = 64; // each block handles 64 columns of N

    static_assert(numSMs * TileM <= M, "Assumes one block per SM");
    static_assert(M % TileM == 0, "M must be multiple of TileM");
    static_assert(N % TileN == 0, "N must be multiple of TileN");
    static_assert(K % TileK == 0, "K must be multiple of TileK");
    constexpr int nTiles = K / TileK;
    static_assert(TileN % blockNSize == 0, "N must be multiple of 64 for TMA");
    auto blockPerTile = TileN / blockNSize;

    // blocksize = 16K
    DAELauncher dae {numSMs};
    auto &gA = dae.data(M * K * sizeof(half));
    init_matrix(gA.h_get<data_t>(), M, K);
    auto &gB = dae.data(K * N * sizeof(half));
    init_matrix(gB.h_get<data_t>(), K, N);
    auto &out = dae.data(M * N * sizeof(half));

    auto &cbuilder = dae.comp;
    // cbuilder.add(cinst(OP_DUMMY, { nTiles * 2 + 1 }));
    cbuilder.add(cinst(OP_WGMMA_M64N256K16_F16, { 1, nTiles }));
    cbuilder.add(cinst(OP_TERMINATE));

    auto descA = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        gA.d_get(),
        { M, K },
        { TileM, TileK },
        CU_TENSOR_MAP_SWIZZLE_128B
    );
    auto descB = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        gB.d_get(),
        { N, K, blockPerTile },
        { 64, TileK, blockPerTile },
        CU_TENSOR_MAP_SWIZZLE_128B,
        { N * sizeof(half), 64 * sizeof(half) }
    );
    
    auto &mbuilder = dae.mem;

    mbuilder.add(minst(
        OP_REPEAT,
        0,
        make_cord({0, TileK}),
        nTiles 
    ));
    mbuilder.add(minst(
        OP_REPEAT,
        1, // MUST BE 0,1,... match the instructions later
        make_cord({0, TileK}),
        nTiles // For continuous REPEAT, only last one need count
    ));
    for (int sm = 0; sm < numSMs; sm++) {
      // load M
      mbuilder.add(sm,minst(
          OP_ALLOC_TMA_LOAD_2D,
          nslot(loadBytesA),
          make_cord({sm * TileM, 0}),
          loadBytesA,
          descA
      ));
      // load N & jump!
      mbuilder.add(sm,minst(
          jump(OP_ALLOC_TMA_LOAD_3D_2CORD),
          nslot(loadBytesB),
          make_cord({0, 0}),
          loadBytesB,
          descB
      ));
      // Store C
      mbuilder.add(sm,minst(
          OP_ALLOC_WB_TMA_STORE_1D,
          nslot(loadBytesC),
          (uint64_t)(out.d_get() + sm * TileM * N * sizeof(data_t)),
          loadBytesC
      ));
    }
    mbuilder.add(minst(OP_TERMINATE));

    size_t loadBytes = loadBytesB + loadBytesA;
    size_t total_bytes = (size_t)numSMs * nTiles * loadBytes;

    printf("Launching debug_gemm kernel M=%d, N=%d, K=%d nTiles=%d loadBytes=%dK totalBytes=%.2fMB\n",
        M, N, K, nTiles, loadBytes / 1024, total_bytes / (1024.0f * 1024.0f));
    dae.launch_bench(total_bytes);
}
