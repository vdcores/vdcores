#include <cuda_fp16.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>      // SM80_16x8x16_F16F16F16F16_TN
#include <cute/atom/mma_atom.hpp>      // MMA_Atom / make_tiled_mma
#include <cute/algorithm/gemm.hpp>     // cute::gemm

#include <cuda/ptx>
#include <cuda/barrier>

using namespace cute;

constexpr int M = 48, V = 256;
constexpr int MMA_M = 16, MMA_K = 16;
constexpr int numTimestampPerKernel = 4;


static_assert(M % MMA_M == 0, "M must be multiple of MMA_M");
static_assert(V % MMA_K == 0, "V must be multiple of MMA_K");

constexpr int numThreads = 32 * (M / MMA_M); // one warp per 16 rows of M

__global__ void simt_kernel(half const* __restrict__ A,
                            half const* __restrict__ B,
                            half*       __restrict__ C,
                            uint64_t*   __restrict__ g_timestamp)
{
    assert(blockDim.x == 32 && "Expected blockDim.x to be 32");
    assert(M == 32 && "Expected M to be 32");

    // One warp, so 32 threads
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Shared memory for one 16x16 + one 16x8 + one 16x8 tile
    __shared__ uint64_t timestamp[numTimestampPerKernel];

    if (tid == 0) timestamp[0] = cuda::ptx::get_sreg_clock64();

    extern __shared__ half smem[];
    half* sA = smem;                    // 16*16
    half* sB = sA + M * V;              // 16*8

    // 1. Naive GMEM -> SMEM loads (just to have data somewhere)
    for (int i = tid; i < M * V; i += blockDim.x) sA[i] = A[i];
    for (int i = tid; i < V;  i += blockDim.x) sB[i] = B[i];
    __syncwarp();

    if (tid == 0) timestamp[1] = cuda::ptx::get_sreg_clock64();

    __syncwarp();
    // 2. Simple SIMT matrix-vector multiply
    // Each warp computes 16 rows of the output vector C
    half rC;

    #pragma unroll 8
    for (int v = 0; v < V; ++v) {
        rC = __hfma(sA[v * M + lane_id], sB[v], rC);
    }

    if (tid == 0) timestamp[2] = cuda::ptx::get_sreg_clock64();

    // 3. Store result back to GMEM
    C[lane_id] = rC;

    // 4. Record end timestamp
    if (tid == 0) {
        timestamp[3] = cuda::ptx::get_sreg_clock64();
        for (int i = 0; i < numTimestampPerKernel; i++) {
            g_timestamp[i] = timestamp[i]; 
        }
    }
}

// SM80 warp-level 16x8x16 FP16 MMA (A,B,C,D all fp16 here)
__global__ void mma16x8x16_kernel(half const* __restrict__ A,
                                  half const* __restrict__ B,
                                  half*       __restrict__ C,
                                  uint64_t*   __restrict__ g_timestamp)
{
    assert(blockDim.x % 32 == 0 && "Expected blockDim.x multiple of 32");
    assert(blockDim.x / 32 == M / 16 && "Expected warps per block to match M dimension");

    // One warp, so 32 threads
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    if (tid == 0) init(&bar, blockDim.x);
    __syncthreads();

    // Shared memory for one 16x16 + one 16x8 + one 16x8 tile
    __shared__ uint64_t timestamp[numTimestampPerKernel];

    if (tid == 0) timestamp[0] = cuda::ptx::get_sreg_clock64();

    extern __shared__ half smem[];
    half* sA = smem;                    // 16*16
    half* sB = sA + M * V;              // 16*8
    half* sC = sB + V;                  // 16*8 (also used as D buffer)

    // 1. Naive GMEM -> SMEM loads (just to have data somewhere)
    if (tid == 0) {
        cuda::device::barrier_expect_tx(bar, (M * V + V) * sizeof(half));
        cuda::device::memcpy_async_tx(sA, A, cuda::aligned_size_t<16>(M * V * sizeof(half)), bar);
        cuda::device::memcpy_async_tx(sB, B, cuda::aligned_size_t<16>(V * sizeof(half)), bar);
    }
    bar.arrive_and_wait();
    __syncthreads();

    if (tid == 0) timestamp[1] = cuda::ptx::get_sreg_clock64();

    // 2. Describe SMEM tiles as CuTe tensors

    // B: (K=16,N=8), col-major from MMA's point-of-view (TN / row.col)
    // Note B is dummy here: anyway to bypass this limitation?
    Tensor t_sB = make_tensor(
        make_smem_ptr(sB),
        make_shape(Int<8>{}, Int<16>{}),      // (K,N)
        make_stride(Int<1>{}, Int<8>{}));    // col-major in (K,N)

    // C (and later D): (M=16,N=8), row-major
    Tensor t_sC = make_tensor(
        make_smem_ptr(sC),
        make_shape(Int<M>{}, Int<8>{}),      // (M,N)
        make_stride(Int<8>{}, Int<1>{})
    );

    //
    // 3. Build the MMA atom (single 16x8x16 warp-level MMA)
    //

    auto tiled_mma = make_tiled_mma(
        MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>{},
        make_layout(
            make_shape(Int<M / 16>{}, Int<1>{}, Int<1>{})
        ) // parallel: (mma_m, mma_n, mma_k)
    );
    auto thr_mma_qk   = tiled_mma.get_slice(threadIdx.x);

    auto tiled_copy = make_tiled_copy_A(
        // Copy_Atom<SM75_U32x4_LDSM_N, half>{},
        Copy_Atom<SM75_U16x8_LDSM_T, half>{},
        tiled_mma
    );

    auto thr_copy = tiled_copy.get_slice(threadIdx.x);

    //
    // 4. Partition SMEM per-thread, then make register fragments
    //
    // Per-thread views into SMEM tiles
    
    // TODO(zhiyuang): tile A or slice A
    auto frag_B = thr_mma_qk.partition_fragment_B(t_sB);   // (thr, mma_n, mma_k)
    auto frag_C = thr_mma_qk.partition_fragment_C(t_sC);   // (thr, mma_m, mma_n)

    clear(frag_B);
    clear(frag_C);                                      // C = 0

    constexpr int numTiles = V / MMA_K; // tile along K
    #pragma unroll
    for (int t = 0; t < numTiles; t++) {
        // Load SMEM -> registers
        // Non-transposed layout
        // Tensor t_sA = make_tensor(
        //     make_smem_ptr(sA + t * M * MMA_K),
        //     make_shape(Int<M>{}, Int<MMA_K>{}),     // (M,K)
        //     make_stride(Int<MMA_K>{}, Int<1>{}));   // row-major compact
        // Non-transposed Layout, strided
        // Tensor t_sA = make_tensor(
        //     make_smem_ptr(sA + t * MMA_K),
        //     make_shape(Int<M>{}, Int<MMA_K>{}),     // (M,K)
        //     make_stride(Int<V>{}, Int<1>{}));   // row-major compact
        // Transposed Layout
        Tensor t_sA = make_tensor(
            make_smem_ptr(sA + t * MMA_K * M),
            make_shape(Int<M>{}, Int<MMA_K>{}),     // (M,K)
            make_stride(Int<1>{}, Int<M>{}));       // col-major

        auto frag_A = thr_mma.partition_fragment_A(t_sA);

        // copy of vector
        if (lane_id < 4) {
            uint32_t *ptr_regs = reinterpret_cast<uint32_t*>(frag_B.data());
            uint32_t *ptr_smem = reinterpret_cast<uint32_t*>(sB + t * MMA_K);
            ptr_regs[0] = ptr_smem[lane_id];
            ptr_regs[1] = ptr_smem[lane_id + 4];
        }

        copy(tiled_copy,
             thr_copy.partition_S(t_sA),
             thr_copy.retile_D(frag_A)); 

        // copy(thr_mma.partition_A(t_sA), frag_A);

        // 5. Issue the m16n8k16 MMA atom through cute::gemm
        gemm(tiled_mma, frag_C, frag_A, frag_B, frag_C);   // C = A*B + C
    }

    if (tid == 0) timestamp[2] = cuda::ptx::get_sreg_clock64();

    // 6. Store D fragment back to SMEM and then to GMEM
    if (lane_id % 4 == 0 && lane_id / 4 < 8) {
        auto regs = frag_C.data();
        C[warp_id * MMA_M + lane_id / 4] = regs[0];
        C[warp_id * MMA_M + lane_id / 4 + 8] = regs[2];
    }

    if (tid == 0) {
        timestamp[3] = cuda::ptx::get_sreg_clock64();
        for (int i = 0; i < numTimestampPerKernel; i++) {
            g_timestamp[i] = timestamp[i];
        }
    }
}

auto constexpr dut_kernel = mma16x8x16_kernel;
int constexpr numThreads_dut = numThreads;

int main() {
    // Benchmark parameters
    constexpr int warmup_iters = 100;
    constexpr int bench_iters = 1000;

    // Allocate host memory
    uint64_t* h_timestamp = new uint64_t[numTimestampPerKernel * bench_iters];
    half* h_A = new half[M * V];
    half* h_B = new half[V];
    half* h_C = new half[M];

    // Initialize with simple test values
    for (int i = 0; i < M * V; i++) {
        h_A[i] = __float2half(float(i) * 0.01);
    }

    // vector-loading
    for (int i = 0; i < V; i++) {
        h_B[i] = __float2half(float(i) * 0.01);
    }

    // Allocate device memory
    uint64_t* d_timestamp;
    half *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * V * sizeof(half));
    cudaMalloc(&d_B, V * sizeof(half));
    cudaMalloc(&d_C, M * sizeof(half));
    cudaMalloc(&d_timestamp, numTimestampPerKernel * sizeof(uint64_t));

    // Copy data to device
    cudaMemcpy(d_A, h_A, M * V * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, V * sizeof(half), cudaMemcpyHostToDevice);

    constexpr int smem_size = (M * V + V + M) * sizeof(half) + 128;
    static_assert(smem_size <= 196 * 1024, "Not enough shared memory allocated");
    printf("Dynamic shared memory size: %d bytes\n", smem_size);
    auto err = cudaFuncSetAttribute(dut_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    if (err != cudaSuccess) {
        printf("Failed to set max dynamic shared memory: %s\n", 
            cudaGetErrorString(err));
        return -1;
    }

    // Warmup
    printf("Running warmup (%d iterations)...\n", warmup_iters);
    for (int i = 0; i < warmup_iters; i++) {
        dut_kernel<<<1, numThreads_dut, smem_size>>>(d_A, d_B, d_C, d_timestamp);
    }
    cudaDeviceSynchronize();

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Benchmark
    printf("Running benchmark (%d iterations)...\n", bench_iters);
    for (int i = 0; i < bench_iters; i++) {
        dut_kernel<<<1, numThreads_dut, smem_size>>>(d_A, d_B, d_C, d_timestamp);
        cudaMemcpy(h_timestamp + i * numTimestampPerKernel, d_timestamp, numTimestampPerKernel * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    }

    // Copy result back to host for verification
    cudaMemcpy(h_C, d_C, M * sizeof(half), cudaMemcpyDeviceToHost);

    // Calculate average time between timestamps
    double avg_stage1 = 0.0; // Start to data loaded
    double avg_stage2 = 0.0; // Data loaded to compute done
    double avg_stage3 = 0.0; // Compute done to store done
    double avg_total = 0.0;  // Start to end

    for (int i = 0; i < bench_iters; i++) {
        uint64_t* ts = h_timestamp + i * numTimestampPerKernel;
        avg_stage1 += (ts[1] - ts[0]);
        avg_stage2 += (ts[2] - ts[1]);
        avg_stage3 += (ts[3] - ts[2]);
        avg_total += (ts[3] - ts[0]);
    }

    avg_stage1 /= bench_iters;
    avg_stage2 /= bench_iters;
    avg_stage3 /= bench_iters;
    avg_total /= bench_iters;

    // Print benchmark results
    printf("\n=== Benchmark Results ===\n");
    printf("Matrix size: %dx%d @ %dx1\n", M, V, V);
    printf("Total iterations: %d\n", bench_iters);
    
    printf("\n--- Cycle Counts (Average) ---\n");
    printf("Stage 1 (Start -> Data Loaded):     %.2f cycles\n", avg_stage1);
    printf("Stage 2 (Data Loaded -> MMA Done):  %.2f cycles\n", avg_stage2);
    printf("Stage 3 (MMA Done -> Store Done):   %.2f cycles\n", avg_stage3);
    printf("Total (Start -> End):               %.2f cycles\n", avg_total);
    
    // Assume GPU clock frequency (adjust based on your GPU)
    // For H100: ~1.8 GHz, A100: ~1.4 GHz, V100: ~1.3 GHz
    double gpu_freq_ghz = 1.98; // Adjust this for your GPU
    printf("\n--- Time Estimates (assuming %.2f GHz clock) ---\n", gpu_freq_ghz);
    printf("Stage 1: %.3f ns\n", avg_stage1 / gpu_freq_ghz);
    printf("Stage 2: %.3f ns\n", avg_stage2 / gpu_freq_ghz);
    printf("Stage 3: %.3f ns\n", avg_stage3 / gpu_freq_ghz);
    printf("Total:   %.3f ns\n", avg_total / gpu_freq_ghz);
    
    // Calculate FLOPS based on cycle count
    double flops_per_kernel = (double)M * V * 2.0;
    double gflops = (flops_per_kernel / (avg_stage2 / gpu_freq_ghz));
    printf("\nPerformance: %.2f GFLOPS\n", gflops);
    printf("\nN per microsecond: %.2f\n", (double)V / (avg_stage2 / gpu_freq_ghz) * 1000 );

    // Cleanup
    delete[] h_timestamp;
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_timestamp);

    return 0;
}
