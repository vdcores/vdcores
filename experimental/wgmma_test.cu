#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda/barrier>

using bf16 = __nv_bfloat16;

__device__ __forceinline__ void wgmma_8(uint64_t const &desc_a, uint64_t const &desc_b, float d[4]) {

    constexpr int32_t ScaleD = 1;
    constexpr int32_t ScaleA = 1;
    constexpr int32_t ScaleB = 1;
    constexpr int32_t TransA = 0;
    constexpr int32_t TransB = 0;

    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{%0,   %1,   %2,   %3},"
        " %4,"
        " %5,"
        " %6, %7, %8, %9, %10;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
          "n"(int32_t(ScaleB)), "n"(int32_t(TransA)), "n"(int32_t(TransB)));
}
__device__ __forceinline__ void warpgroup_fence_operand(uint32_t &reg) {
    asm volatile("" : "+r"(reg)::"memory");
}

__device__ __forceinline__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ __forceinline__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}
__device__
uint64_t insert_bit(uint32_t start_bit, uint64_t target, uint64_t val)
{
    return target | (val << start_bit);
}

template <class PointerType>
__device__ uint64_t make_desc(PointerType smem_ptr, int sbo, int lbo) {

  uint64_t desc = 0;
  uint32_t base_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint32_t start_address = base_ptr >> 4;
  uint64_t swizzle = 0;
  uint64_t offset = 0;

  desc = insert_bit(62, desc, swizzle);
  desc = insert_bit(49, desc, offset);
  desc = insert_bit(32, desc, (uint64_t)sbo);
  desc = insert_bit(16, desc, (uint64_t)lbo);
  desc = insert_bit(0, desc, start_address);
  return desc;
}
//  A is row major and B is column major
__global__ void test_wgmma_8(float* C) {

    __shared__ bf16 A[64*16];
    __shared__ bf16 B[16*8];

    int tid = threadIdx.x;
    int wid = tid / 32;
    int lid = tid % 32;

    for (int i = 0; i < 8; i++)
    {
        int ind = ((tid % 2) * 512) + ((tid / 2) * 8 + i);
        float val = (tid * 8) + i;  
        A[ind] = (bf16)(val);
    }

    int xtid = tid % 64;
    float val = (tid / 64) * 64 + ((xtid % 8) * 8) + (xtid / 8);    
    B[tid] = (bf16)val;

    __syncthreads();

    uint64_t desc_a = make_desc(A, 8, 64);
    uint64_t desc_b = make_desc(B, 1, 8);

    float d[4];
    memset(d, 0, sizeof(d));

    warpgroup_arrive();
    wgmma_8(desc_a, desc_b, d);

    warpgroup_commit_batch();
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");

    C[wid * 128 + (lid * 2)] = d[0];
    C[wid * 128 + (lid * 2) + 1] = d[1];
    C[wid * 128 + 64 + (lid * 2) ] = d[2];
    C[wid * 128 + 64 + (lid * 2) + 1] = d[3];
}

int main() {
    dim3 grid(1);
    dim3 thread(128);

    int size = 64*8*sizeof(float);
    float* hC = (float*)malloc(size);
    float* dC;
    cudaMalloc((void**)&dC, size);
    test_wgmma_8<<<grid, thread>>>(dC);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    printf("---------res--------\n");
    for(int i = 0; i < 64; i++)
    {
        for (int j = 0; j < 8; j++)
        {
            printf("%f ",hC[i * 8 + j]);
        }
        printf("\n");
    }
}