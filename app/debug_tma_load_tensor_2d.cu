#include "dae2.cuh"
#include "launcher.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>

// Minimal test case for debugging TMA stride calculation
int main(int argc, char** argv) {
    constexpr int numSMs = 1;        // Just 1 SM for easy debugging
    constexpr uint64_t M = 128, N = 256;  // Small matrix: 128 rows × 256 cols
    // blocksize = 16K
    constexpr uint32_t TileM = 8, TileN = 16;  // Small tiles: 8 rows × 16 cols
    constexpr size_t totalBytes = M * N * sizeof(__nv_bfloat16);
    constexpr uint16_t loadBytes = TileM * TileN * sizeof(__nv_bfloat16);

    // constexpr int numLoads = N / TileN;  // 256/64 = 4 loads
    constexpr int numLoads = 1;

    // static_assert(numSMs == M / TileM, "Assumes one block per SM");

    using bf16 = __nv_bfloat16;

    printf("=== Minimal TMA 2D Load Test ===\n");
    printf("Matrix: M=%lu rows, N=%lu cols (stored as mat[M][N])\n", M, N);
    printf("Tiles: TileM=%u, TileN=%u\n", TileM, TileN);
    printf("Element size: %lu bytes\n", sizeof(bf16));
    printf("\nTMA Descriptor expects:\n");
    printf("  global_dims = {N=%lu, M=%lu}\n", N, M);
    printf("  global_strides[0] = N × sizeof(bf16) = %lu × %lu = %lu bytes\n",
           N, sizeof(bf16), N * sizeof(bf16));
    printf("  (This is the stride to move from row i to row i+1)\n");
    printf("\nEach SM will load %d tiles of size %u bytes\n", numLoads, loadBytes);
    printf("================================================\n\n");

    DAELauncher dae {numSMs};
    auto &mat = dae.sm_data(totalBytes / numSMs); // Data block per SM
    bool row_major = false;

    // Initialize matrix with recognizable pattern
    // mat[row][col] = row * N + col
    bf16 *h_mat = mat.h_get<bf16>();
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            // row major
            if (row_major)
                h_mat[row * N + col] = __float2bfloat16_rn(float((row * N + col) / 1000.0));
            // column major
            if (!row_major)
                h_mat[col * M + row] = __float2bfloat16_rn(float((row * N + col) / 1000.0));
        }
    }
    for (int ei = 0; ei < TileM * TileN; ei ++) {
        uint64_t offset = row_major ?
            ei / TileN * N + ei % TileN:
            ei / TileM * M + ei % TileM;
        printf("[Host] tile 0 element[%d]=%f\n", ei,
            __bfloat162float(h_mat[offset]));
    }

    auto tma_id = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        mat.d_get(),
        row_major ? std::array<uint64_t,5>{N, M, 1, 1, 1} : std::array<uint64_t,5>{M, N, 1, 1, 1},
        row_major ? std::array<uint32_t,5>{TileN, TileM, 1, 1, 1} : std::array<uint32_t,5>{TileM, TileN, 1, 1, 1}
    );

    auto &cbuilder = dae.comp;
    cbuilder.add(cinst(OP_DUMMY, { numLoads }));
    cbuilder.add(cinst(OP_TERMINATE));

    auto &mbuilder = dae.mem;
    for (int sm = 0; sm < numSMs; sm++) {
        for (int i = 0; i < numLoads; i++) {
            uint32_t cord[2] = {
                row_major ? (i * TileN) : (sm * TileM),
                row_major ? (sm * TileM) : (i * TileN)
            };
            uint64_t cords = (uint64_t)cord[0] | ((uint64_t)cord[1] << 32);
            // print this tile from host mem in first SM
            // if (sm == 0) {
            //     for (int ei = 0; ei < TileM * TileN; ei ++) {
            //         uint64_t offset = row_major ?
            //             cord[1] * N + cord[0] + ei / TileN * N + ei % TileN :
            //             cord[0] * M + cord[1] + ei / TileM * M + ei % TileM;
            //         printf("[Host] tile %d element[%d]=%f\n", i, ei,
            //             __bfloat162float(h_mat[offset]));
            //     }
            // }
            mbuilder.add(sm, minst(
                OP_ALLOC_TMA_LOAD_2D,
                2,
                cords,
                loadBytes,
                tma_id
            ));
        }
    }
    mbuilder.add(minst(OP_TERMINATE));

    float total_bytes = (float)numLoads * loadBytes * numSMs;
    dae.launch_bench(total_bytes, 1);
}
