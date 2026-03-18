#include "dae2.cuh"
#include "launcher.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>

// Round-trip test: TMA 2D Load → Compute (passthrough) → TMA 2D Store
int main(int argc, char** argv) {
    constexpr int numSMs = 1;        // Just 1 SM for easy debugging
    constexpr uint64_t M = 128, N = 256;  // Small matrix: 128 rows × 256 cols
    // blocksize = 16K
    constexpr uint32_t TileM = 8, TileN = 16;  // Small tiles: 8 rows × 16 cols
    constexpr size_t totalBytes = M * N * sizeof(__nv_bfloat16);
    constexpr uint16_t loadBytes = TileM * TileN * sizeof(__nv_bfloat16);

    constexpr int numLoads = 1;  // Test 1 tile for verification

    // static_assert(numSMs == M / TileM, "Assumes one block per SM");

    using bf16 = __nv_bfloat16;

    printf("=== TMA 2D Load/Store Round-trip Test ===\n");
    printf("Matrix: M=%lu rows, N=%lu cols\n", M, N);
    printf("Tiles: TileM=%u, TileN=%u\n", TileM, TileN);
    printf("Testing %d tile(s)\n", numLoads);
    printf("================================================\n\n");

    DAELauncher dae {numSMs};

    // Input matrix - initialize with recognizable pattern
    auto &mat_in = dae.sm_data(totalBytes / numSMs);
    bool row_major = true;

    // Initialize input matrix with pattern: mat[row][col] = row * N + col
    bf16 *h_mat_in = mat_in.h_get<bf16>();
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            // row major
            if (row_major)
                h_mat_in[row * N + col] = __float2bfloat16_rn(float((row * N + col) / 1000.0));
            // column major
            if (!row_major)
                h_mat_in[col * M + row] = __float2bfloat16_rn(float((row * N + col) / 1000.0));
        }
    }

    // Print first tile from input
    printf("[Host] Input tile (first 16 elements):\n");
    for (int ei = 0; ei < TileM * TileN && ei < 16; ei++) {
        uint64_t offset = row_major ?
            ei / TileN * N + ei % TileN:
            ei / TileM * M + ei % TileM;
        printf("  element[%d] = %f\n", ei, __bfloat162float(h_mat_in[offset]));
    }
    printf("\n");

    // Output matrix - initialize to zeros
    auto &mat_out = dae.sm_data(totalBytes / numSMs);
    bf16 *h_mat_out = mat_out.h_get<bf16>();
    for (int i = 0; i < M * N; i++) {
        h_mat_out[i] = __float2bfloat16_rn(0.0f);
    }

    // TMA descriptors for input and output (same dimensions)
    auto tma_in = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        mat_in.d_get(),
        row_major ? std::array<uint64_t,5>{N, M, 1, 1, 1} : std::array<uint64_t,5>{M, N, 1, 1, 1},
        row_major ? std::array<uint32_t,5>{TileN, TileM, 1, 1, 1} : std::array<uint32_t,5>{TileM, TileN, 1, 1, 1}
    );

    auto tma_out = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        mat_out.d_get(),
        row_major ? std::array<uint64_t,5>{N, M, 1, 1, 1} : std::array<uint64_t,5>{M, N, 1, 1, 1},
        row_major ? std::array<uint32_t,5>{TileN, TileM, 1, 1, 1} : std::array<uint32_t,5>{TileM, TileN, 1, 1, 1}
    );

    // Compute instructions: handle 2 slots per iteration (input + output)
    auto &cbuilder = dae.comp;
    cbuilder.add(cinst(OP_DUMMY, { numLoads }));  // same slot for both load and store
    cbuilder.add(cinst(OP_TERMINATE));

    // Memory instructions: load from input, allocate for output with writeback
    auto &mbuilder = dae.mem;
    for (int sm = 0; sm < numSMs; sm++) {
        for (int i = 0; i < numLoads; i++) {
            uint32_t cord[2] = {
                row_major ? (i * TileN) : (sm * TileM),
                row_major ? (sm * TileM) : (i * TileN)
            };
            uint64_t cords = (uint64_t)cord[0] | ((uint64_t)cord[1] << 32);

            // Load from input matrix
            mbuilder.add(sm, minst(
                OP_ALLOC_TMA_LOAD_2D,
                2,
                cords,
                loadBytes,
                tma_in
            ));

            // Allocate for output matrix with writeback
            mbuilder.add(sm, minst(
                linked(OP_ALLOC_WB_TMA_STORE_2D),
                0,
                cords,
                loadBytes,
                tma_out
            ));
        }
    }
    mbuilder.add(minst(OP_WRITE_BARRIER));
    mbuilder.add(minst(OP_TERMINATE));

    float total_bytes = (float)numLoads * loadBytes * numSMs;
    dae.launch_bench(total_bytes, 1);

    // Copy output back from device
    cudaMemcpy(h_mat_out, mat_out.d_get(), totalBytes / numSMs, cudaMemcpyDeviceToHost);

    // Verify results
    printf("\n[Host] Output tile (first 16 elements):\n");
    int errors = 0;
    for (int ei = 0; ei < TileM * TileN && ei < 16; ei++) {
        uint64_t offset = row_major ?
            ei / TileN * N + ei % TileN:
            ei / TileM * M + ei % TileM;
        float expected = __bfloat162float(h_mat_in[offset]);
        float actual = __bfloat162float(h_mat_out[offset]);
        printf("  element[%d] = %f (expected %f) %s\n",
               ei, actual, expected,
               (actual == expected) ? "✓" : "✗");
        if (actual != expected) errors++;
    }

    errors = 0;
    printf("\nFull tile verification:\n");
    int total_checked = 0;
    for (int ei = 0; ei < TileM * TileN; ei++) {
        uint64_t offset = row_major ?
            ei / TileN * N + ei % TileN:
            ei / TileM * M + ei % TileM;
        float expected = __bfloat162float(h_mat_in[offset]);
        float actual = __bfloat162float(h_mat_out[offset]);
        if (actual != expected) errors++;
        total_checked++;
    }

    if (errors == 0) {
        printf("  ✓ All %d elements match! Round-trip test PASSED.\n", total_checked);
    } else {
        printf("  ✗ %d/%d elements incorrect! Round-trip test FAILED.\n", errors, total_checked);
    }
}
