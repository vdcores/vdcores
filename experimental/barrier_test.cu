#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test_barrier_correct(char* dst, char* src) {
    const int memsize = 16;
    // Use a block-scoped barrier allocated in shared memory so the copy engine
    // and threads in the block share the same barrier object.
    __shared__ cuda::barrier<cuda::thread_scope_block> bar;
    alignas(16) __shared__ char shared_buffer[memsize];

    if (threadIdx.x == 0) {
        init(&bar, 1);  // 4 memcpy_async + 1 arrive_and_wait

        cuda::device::memcpy_async_tx(shared_buffer, src, cuda::aligned_size_t<16>(memsize), bar);

        auto token = cuda::device::barrier_arrive_tx(bar, 1, memsize);
        bar.wait(std::move(token));

        for (int i = 0; i < memsize; i++) {
            printf("%c ", shared_buffer[i]);
        }
        
        printf("\nCorrect version completed!\n");
    }
}

__global__ void test_barrier_broken(char* dst, char* src) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Broken: Initialize barrier with count = 1 (will deadlock!)
        cuda::barrier<cuda::thread_scope_system> bar;
        init(&bar, 1);

        cuda::memcpy_async(dst,     src,      1, bar);
        cuda::memcpy_async(dst + 1, src + 8,  1, bar);
        cuda::memcpy_async(dst + 2, src + 16, 1, bar);
        cuda::memcpy_async(dst + 3, src + 24, 1, bar);

        bar.arrive_and_wait();
        
        printf("Broken version completed (this should never print)!\n");
    }
}

int main() {
    const int SIZE = 32;
    char *h_src, *h_dst;
    char *d_src, *d_dst;

    // Allocate host memory
    h_src = new char[SIZE];
    h_dst = new char[SIZE];

    // Initialize source data
    for (int i = 0; i < SIZE; i++) {
        h_src[i] = 'A' + (i % 26);
    }
    memset(h_dst, 0, SIZE);

    // Allocate device memory
    cudaMalloc(&d_src, SIZE);
    cudaMalloc(&d_dst, SIZE);

    // Copy source to device
    cudaMemcpy(d_src, h_src, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dst, h_dst, SIZE, cudaMemcpyHostToDevice);

    printf("Testing CORRECT barrier usage...\n");
    test_barrier_broken<<<1, 1>>>(d_dst, d_src);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    // Copy result back
    cudaMemcpy(h_dst, d_dst, SIZE, cudaMemcpyDeviceToHost);
    printf("Result: ");
    for (int i = 0; i < 4; i++) {
        printf("%c ", h_dst[i]);
    }
    printf("\n");

    // Reset destination
    memset(h_dst, 0, SIZE);
    cudaMemcpy(d_dst, h_dst, SIZE, cudaMemcpyHostToDevice);

    printf("\nTesting BROKEN barrier usage (will hang)...\n");
    printf("This will timeout or hang - press Ctrl+C if needed\n");
    
    test_barrier_correct<<<1, 1>>>(d_dst, d_src);
    
    // Set a timeout for synchronization
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error (expected): %s\n", cudaGetErrorString(err));
    } else {
        printf("Unexpectedly completed!\n");
    }

    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);
    delete[] h_src;
    delete[] h_dst;

    return 0;
}