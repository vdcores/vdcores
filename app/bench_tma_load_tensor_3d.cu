#include "dae2.cuh"
#include "launcher.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>

// bench = ~3.5T on GH200
int main(int argc, char** argv) {
    constexpr int numSMs = 128;
    constexpr uint64_t M = 4096, N = 4096*3;
    // blocksize = 16K
    constexpr uint32_t TileM = 32, TileN = 256;
    constexpr size_t totalBytes = M * N * sizeof(__nv_bfloat16);
    constexpr uint16_t loadBytes = TileM * TileN * sizeof(__nv_bfloat16);

    constexpr int numLoads = N / TileN;

    static_assert(numSMs == M / TileM, "Assumes one block per SM");

    using bf16 = __nv_bfloat16;
    
    DAELauncher dae {numSMs};
    auto &mat = dae.sm_data(totalBytes / numSMs); // Data block per SM

    auto tma_id = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        3,
        mat.d_get(),
        {N / 4, M, 4},
        {64, TileM, 4},
        CU_TENSOR_MAP_SWIZZLE_128B,
        {N * sizeof(bf16), 64 * sizeof(bf16)}
    );

    auto &cbuilder = dae.comp;
    cbuilder.add(cinst(OP_DUMMY, { numLoads }));
    cbuilder.add(cinst(OP_TERMINATE));

    auto &mbuilder = dae.mem;
    for (int sm = 0; sm < numSMs; sm++) {
        for (int i = 0; i < numLoads; i++) {
            mbuilder.add(sm, minst(
                OP_ALLOC_TMA_LOAD_3D,
                nslot(loadBytes),
                make_cord(i * TileN / 4, sm * TileM),
                loadBytes,
                tma_id
            ));
        }
    }
    mbuilder.add(minst(OP_TERMINATE));

    float total_bytes = (float)numLoads * loadBytes * numSMs;
    printf("Launching bench_tma_load_tensor_3d M=%lu, N=%lu numLoads=%d loadBytes=%uK totalBytes=%.2fMB\n",
            M, N, numLoads, loadBytes / 1024, total_bytes / (1024.0f * 1024.0f));
    dae.launch_bench(total_bytes);
}
