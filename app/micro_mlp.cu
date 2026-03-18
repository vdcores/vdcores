#include "dae2.cuh"
#include "launcher.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>

// bench = ~3.5T on GH200
int main(int argc, char** argv) {
    constexpr int numSMs = 128;
    constexpr uint64_t M = 4096, N = 2048;
    // blocksize = 16K
    constexpr uint32_t TileM = 32, TileN = 256;
    constexpr size_t matBytes = M * N * sizeof(__nv_bfloat16);
    constexpr uint16_t matTileBytes = TileM * TileN * sizeof(__nv_bfloat16);

    constexpr int numTiles = N / TileN;

    constexpr uint16_t vectorBytes = N * sizeof(__nv_bfloat16);
    constexpr uint16_t vectorSlots = 1;

    // static_assert(numSMs == M / TileM, "Assumes one block per SM");

    DAELauncher dae {numSMs};
    auto &mat = dae.data(matBytes); // Data block per SM
    auto &vec = dae.data(vectorBytes);
    uint16_t bar_id = 0;

    // Initialize with simple test values
    for (int i = 0; i < M * N; i++)
        mat.h_get<half>()[i] = __float2half(float(i) * 0.01);
    // vector-loading
    for (int i = 0; i < N; i++)
        vec.h_get<half>()[i] = __float2half(float(i) * 0.01);

    auto tma_mat = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        mat.d_get(),
        {N, M},
        {TileN, TileM}
    );

    auto &cbuilder = dae.comp;
    cbuilder.add(cinst(OP_GEMV_BF16_32X256, { numTiles }));
    cbuilder.add(cinst(OP_GEMV_BF16_32X256, { numTiles }));

    // in dummy mode, +2 for the two vector ops
    // cbuilder.add(cinst(OP_DUMMY, { numTiles + 2 }));
    // cbuilder.add(cinst(OP_DUMMY, { numTiles + 2 }));
    cbuilder.add(cinst(OP_TERMINATE));

    auto &mbuilder = dae.mem;

    // load vector
    mbuilder.add(minst(
        OP_ALLOC_TMA_LOAD_1D,
        vectorSlots,
        (uint64_t)vec.d_get(),
        vectorBytes
    ));
    // load mat tiles
    for (int sm = 0; sm < numSMs; sm++) {
      for (int i = 0; i < numTiles; i++) {
        uint32_t cord[2] = { i * TileN, sm * TileM };
        uint64_t cords = (uint64_t)cord[0] | ((uint64_t)cord[1] << 32);
        mbuilder.add(sm, minst(
          OP_ALLOC_TMA_LOAD_2D,
          2,
          cords,
          matTileBytes,
          tma_mat
        ));
      }
    }
    // store result vector
    mbuilder.add(minst(
        OP_ALLOC_WB_BAR_TMA_STORE_1D,
        1,
        (uint64_t)vec.d_get(),
        TileM * sizeof(half),
        bar_id
    ));

    // finish of first mat

    // load mat tiles
    for (int sm = 0; sm < numSMs; sm++) {
      for (int i = 0; i < numTiles; i++) {
        uint32_t cord[2] = { i * TileN, sm * TileM };
        uint64_t cords = (uint64_t)cord[0] | ((uint64_t)cord[1] << 32);
        mbuilder.add(sm, minst(
          OP_ALLOC_TMA_LOAD_2D,
          2,
          cords,
          matTileBytes,
          tma_mat
        ));
      }
    }
    // barrier for vector
    mbuilder.add(minst(OP_BAR_WAIT, 0, 0, numSMs, bar_id));
    // load result vector
    mbuilder.add(minst(
        OP_ALLOC_TMA_LOAD_1D,
        vectorSlots,
        (uint64_t)vec.d_get(),
        vectorBytes,
        bar_id
    ));
    mbuilder.add(minst(
        OP_ALLOC_WB_TMA_STORE_1D,
        1,
        (uint64_t)vec.d_get(),
        TileM * sizeof(half)
    ));

    mbuilder.add(minst(OP_TERMINATE));

    float total_bytes = (float)matBytes * 2 + (M + N) *sizeof(__nv_bfloat16) * numSMs;
    dae.launch_bench(total_bytes);
}
