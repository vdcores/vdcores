#include "launcher.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>

using data_t = half;
static const auto data2float = __half2float;
static const auto float2data = __float2half;
#define data_add __hadd
#define data_mul __hmul

static void init_matrix(data_t* mat, uint32_t M, uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            mat[i + j * M] = float2data((static_cast<float>(i % 1000)) / 256.0f);
        }
    }
}

static void build_reference_matrix(data_t* ref, const data_t* A, const data_t* B, uint32_t M, uint32_t N, uint32_t K) {
    // Both A and B are MN major
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            data_t sum = float2data(0.0f);
            for (uint32_t k = 0; k < K; k++) {
                sum = data_add(sum, data_mul(A[i + k * M], B[j + k * N]));
            }
            ref[i + j * M] = sum;
        }
    }
}

// Minimal test case for debugging TMA stride calculation
int main(int argc, char** argv) {

    constexpr int numSMs = 1;        // Just 1 SM for easy debugging
    constexpr uint32_t M = 64, N = 256, K = 128;

    constexpr uint32_t TileM = 64, TileN = 256, TileK = 64;
    constexpr uint32_t loadBytesA = TileM * TileK * sizeof(half);
    constexpr uint32_t loadBytesB = TileK * TileN * sizeof(half);
    constexpr uint32_t loadBytesC = TileM * TileN * sizeof(half);

    static_assert(M % TileM == 0, "M must be multiple of TileM");
    static_assert(N % TileN == 0, "N must be multiple of TileN");
    constexpr int nOutput = N / TileN;
    static_assert(K % TileK == 0, "K must be multiple of TileK");
    constexpr int nTiles = K / TileK;
    static_assert(TileN % 64 == 0, "N must be multiple of 64 for TMA");

    // blocksize = 16K
    DAELauncher dae {numSMs};
    auto &gA = dae.data(M * K * sizeof(half));
    init_matrix(gA.h_get<data_t>(), M, K);
    auto &gB = dae.data(K * N * sizeof(half));
    init_matrix(gB.h_get<data_t>(), K, N);
    auto &out = dae.data(M * N * sizeof(half));
    auto &ref = dae.data(M * N * sizeof(half));
    build_reference_matrix(ref.h_get<data_t>(), gA.h_get<data_t>(), gB.h_get<data_t>(), M, N, K);

    auto &cbuilder = dae.comp;
    // cbuilder.add(cinst(OP_DUMMY, { 1 }));
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

    uint32_t blockSize = 64;
    auto descB = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        4,
        gB.d_get(),
        { blockSize, 16, N / blockSize, K / 16 },
        { blockSize, 16, TileN / blockSize, TileK / 16 },
        CU_TENSOR_MAP_SWIZZLE_128B,
        { N * sizeof(half), blockSize * sizeof(half), N * 16 * sizeof(half) }
    );
    auto descC = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        out.d_get(),
        { M, N },
        { TileM, TileN },
        CU_TENSOR_MAP_SWIZZLE_128B
    );
    
    auto &mbuilder = dae.mem;
    mbuilder.add(minst(
        OP_REPEAT,
        0,
        make_cord(0, TileK),
        0
    ));
    // repeats. only last repeat start the repeat with size > 0
    mbuilder.add(minst(
        OP_REPEAT,
        1,
        make_cord(0, 0, 0, TileK / 16),
        nTiles
    ));
    mbuilder.add(minst(
        OP_ALLOC_TMA_LOAD_2D,
        nslot(loadBytesA),
        make_cord(0, 0),
        loadBytesA,
        descA
    ));
    mbuilder.add(minst(
        jump(OP_ALLOC_TMA_LOAD_4D),
        nslot(loadBytesB),
        make_cord(0, 0, 0, 0),
        loadBytesB,
        descB
    ));
    mbuilder.add(minst(
        OP_ALLOC_WB_TMA_STORE_2D,
        nslot(loadBytesC),
        make_cord(0, 0),
        loadBytesC,
        descC
    ));

    mbuilder.add(minst(OP_WRITE_BARRIER));
    mbuilder.add(minst(OP_TERMINATE));

    printf("Launching debug_gemm kernel M=%d, N=%d, K=%d nTiles=%d...\n", M, N, K, nTiles);
    dae.launch();

    // copy back to host
    out.copy_to_host();
    // verify
    data_t *out_h = out.h_get<data_t>();
    data_t *ref_h = ref.h_get<data_t>();

    int error_count = 0;
    for (uint32_t j = 0; j < N; j++) {
        for (uint32_t i = 0; i < M; i++) {
            float out_f = data2float(out_h[i + j * M]);
            float ref_f = data2float(ref_h[i + j * M]);
            float diff = fabs(out_f - ref_f);
            // TODO(zhiyuang): this esp seems still large
            if (diff > 0.01f) {
                if (error_count < 16)
                    printf("Mismatch at (%d,%d): got %f, expected %f\n", i, j, out_f, ref_f);
                error_count++;
            }
        }
    }
    printf("Test completed with %d errors\n", error_count);
}
