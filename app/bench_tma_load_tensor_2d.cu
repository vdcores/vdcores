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

    // static_assert(numSMs == M / TileM, "Assumes one block per SM");

    using bf16 = __nv_bfloat16;
    
    DAELauncher dae {numSMs};
    auto &mat = dae.sm_data(totalBytes / numSMs); // Data block per SM

    auto tma_id = dae.tma.add(
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        mat.d_get(),
        {N, M},
        {TileN, TileM}
    );

    auto &cbuilder = dae.comp;
    cbuilder.add(cinst(OP_DUMMY, { numLoads }));
    cbuilder.add(cinst(OP_TERMINATE));

    auto &mbuilder = dae.mem;
    for (int sm = 0; sm < numSMs; sm++) {
        for (int i = 0; i < numLoads; i++) {
            uint32_t cord[2] = { i * TileN, sm * TileM };
            uint64_t cords = (uint64_t)cord[0] | ((uint64_t)cord[1] << 32);
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
    dae.launch_bench(total_bytes);
}
