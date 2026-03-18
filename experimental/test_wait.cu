#include <cstdio>
#include <cuda/ptx>
#include <cuda/barrier>

using namespace cuda::ptx;

__global__ void test_mbarrier() {
    // One mbarrier in shared memory
    __shared__ cuda::barrier<cuda::thread_scope_block> barrier_obj;

    int tid = threadIdx.x;

    // Init: thread 0 initializes barrier with expected arrivals = 1
    if (tid == 0) {
        init(&barrier_obj, 1);
        printf("[T0] Barrier initialized\n");
    }

    __syncthreads();

    cuda::barrier<cuda::thread_scope_block>::arrival_token token[4];


    if (tid == 1) {
        // Thread 1 performs arrival (complete)
        printf("[T1] Doing mbarrier_arrive\n");
        token[0] = barrier_obj.arrive();
    }

    __syncthreads();
    token[0] = __shfl_sync(0xFFFFFFFF, token[0], 1);
    __syncthreads();

    if (tid == 0) {
        // Thread 0 waits
        unsigned parity = 0;

        printf("[T0] Waiting with mbarrier_test_wait...\n");

        // Wait until arrival from T1
        bool completed = cuda::ptx::mbarrier_test_wait_parity(
            cuda::device::barrier_native_handle(barrier_obj),
            parity
        );

        printf("[T0] Wait completed, result = %d\n", completed);
    }
}

int main() {
    test_mbarrier<<<1, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}
