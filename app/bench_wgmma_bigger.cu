#include "launcher.cuh"
#include <cublas_v2.h>

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
            mat[i + j * M] = float2data(static_cast<float>(i % 1000) / 1024.0f);
        }
    }
}

static void build_reference_matrix_cpu(data_t* ref, const data_t* A, const data_t* B, uint32_t M, uint32_t N, uint32_t K) {
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

static void build_reference_matrix_cublas(data_t* ref, const data_t* A, const data_t* B, uint32_t M, uint32_t N, uint32_t K) {
    // Both A and B are column major
    // C is column major
    // Use cuBLAS GPU GEMM for fast reference computation

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Allocate device memory
    data_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(data_t));
    cudaMalloc(&d_B, K * N * sizeof(data_t));
    cudaMalloc(&d_C, M * N * sizeof(data_t));

    // Copy inputs to device
    cudaMemcpy(d_A, A, M * K * sizeof(data_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(data_t), cudaMemcpyHostToDevice);

    // Run GEMM: C = A * B
    // A: M×K, column-major, LD=M
    // B: stored as N×K (accessed as B[j + k*N] in reference), so we transpose
    // C = A * B^T where B^T has dimensions K×N
    const data_t alpha = float2data(1.0f);
    const data_t beta = float2data(0.0f);

    cublasStatus_t status = cublasHgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,  // Transpose B to match reference
        M, N, K,                    // m, n, k dimensions
        &alpha,
        d_A, M,                     // A matrix, leading dimension M
        d_B, N,                     // B matrix (N×K), leading dimension N
        &beta,
        d_C, M                      // C matrix, leading dimension M
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS GEMM failed with status %d\n", int(status));
    }

    // Copy result back to host
    cudaMemcpy(ref, d_C, M * N * sizeof(data_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);
}

// 3342G for TK=4 on GH200 - could run longer for more accurate measurement
int main(int argc, char** argv) {

    constexpr int numSMs = 128;        // Just 1 SM for easy debugging

    constexpr uint32_t M = 512, N = 4096 * 3, K = 4096;

    constexpr uint32_t TileM = 64, TileN = 256, TileK = 64;
    constexpr uint32_t loadBytesA = TileM * TileK * sizeof(half);
    constexpr uint32_t loadBytesB = TileK * TileN * sizeof(half);
    constexpr uint32_t loadBytesC = TileM * TileN * sizeof(half);
    constexpr uint32_t blockNSize = 64; // each block handles 64 columns of N

    // static_assert(numSMs * TileM == M, "Assumes one block per SM");
    static_assert(M % TileM == 0, "M must be multiple of TileM");
    static_assert(N % TileN == 0, "N must be multiple of TileN");
    static_assert(K % TileK == 0, "K must be multiple of TileK");
    constexpr int nKTiles = K / TileK;
    constexpr int nMTiles = M / TileM;
    constexpr int nNTiles = N / TileN;
    static_assert(TileN % blockNSize == 0, "N must be multiple of 64 for TMA");

    static_assert(nMTiles * nNTiles % numSMs == 0, "Total number of MxN tiles must be divisible by number of SMs");
    constexpr int nOutputTilesPerSM = nMTiles * nNTiles / numSMs;
    auto blockPerTile = TileN / blockNSize;

    // blocksize = 16K
    DAELauncher dae {numSMs};
    auto &gA = dae.data(M * K * sizeof(half));
    init_matrix(gA.h_get<data_t>(), M, K);
    auto &gB = dae.data(K * N * sizeof(half));
    init_matrix(gB.h_get<data_t>(), K, N);
    auto &out = dae.data(M * N * sizeof(half));
    auto &ref = dae.data(M * N * sizeof(half));
    build_reference_matrix_cublas(ref.h_get<data_t>(), gA.h_get<data_t>(), gB.h_get<data_t>(), M, N, K);

    auto &cbuilder = dae.comp;
    // cbuilder.add(cinst(OP_DUMMY, { nTiles * 2 + 1 }));
    cbuilder.add(cinst(OP_WGMMA_M64N256K16_F16, { nOutputTilesPerSM, nKTiles }));
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
        // { N / blockPerTile, K, blockPerTile },
        { 64, K, N / 64},
        { 64, TileK, blockPerTile },
        CU_TENSOR_MAP_SWIZZLE_128B,
        { N * sizeof(half), 64 * sizeof(half) }
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

    printf("nMTiles=%d, nNTiles=%d, nOutputTilesPerSM=%d\n", nMTiles, nNTiles, nOutputTilesPerSM);
    for (int sm = 0; sm < numSMs; sm++) {
        for (int otile = sm * nOutputTilesPerSM; otile < (sm + 1) * nOutputTilesPerSM; otile++) {
            int mTile = otile / nNTiles;
            int nTile = otile % nNTiles;
            for (int i = 0; i < nKTiles; i++) {
                // load M
                mbuilder.add(sm,minst(
                    OP_ALLOC_TMA_LOAD_2D,
                    nslot(loadBytesA),
                    make_cord(mTile * TileM, TileK * i),
                    loadBytesA,
                    descA
                ));
                // load N
                mbuilder.add(sm,minst(
                    OP_ALLOC_TMA_LOAD_3D,
                    nslot(loadBytesB),
                    make_cord(nTile * TileN % 64, TileK * i, nTile * TileN / 64),
                    loadBytesB,
                    descB
                ));
            }
            mbuilder.add(sm,minst(
                OP_ALLOC_WB_TMA_STORE_2D,
                nslot(loadBytesC),
                make_cord(mTile * TileM, nTile * TileN),
                loadBytesC,
                descC
            ));
        }
    }
    mbuilder.add(minst(OP_WRITE_BARRIER));
    mbuilder.add(minst(OP_TERMINATE));

    size_t loadBytes = loadBytesB + loadBytesA;
    size_t total_bytes = (size_t)numSMs * nOutputTilesPerSM * nKTiles * loadBytes + (size_t)numSMs * nOutputTilesPerSM * loadBytesC;

    printf("Launching debug_gemm kernel M=%d, N=%d, K=%d nTiles=%d loadBytes=%dK totalBytes=%.2fMB\n",
        M, N, K, nKTiles, loadBytes / 1024, total_bytes / (1024.0f * 1024.0f));
    // dae.launch_bench(total_bytes, 1);
    dae.launch();

    // copy back to host
    out.copy_to_host();
    // verify
    data_t *out_h = out.h_get<data_t>();
    data_t *ref_h = ref.h_get<data_t>();

    int error_count = 0;
    float ref_check_sum = 0.0;
    float my_check_sum = 0.0;
    for (uint32_t j = 0; j < N; j++) {
        for (uint32_t i = 0; i < M; i++) {
            float out_f = data2float(out_h[i + j * M]);
            float ref_f = data2float(ref_h[i + j * M]);
            float diff = fabs(out_f - ref_f);
            // TODO(zhiyuang): this esp seems still large
            if (diff > 0.04f) {
                if (error_count < 16)
                    printf("Mismatch at (%d,%d): got %f, expected %f\n", i, j, out_f, ref_f);
                error_count++;
            }
            ref_check_sum += ref_f;
            my_check_sum += out_f;
        }
    }
    printf("Test completed with %d errors\n", error_count);
    printf("Reference checksum = %f, my checksum = %f\n", ref_check_sum, my_check_sum);
}
