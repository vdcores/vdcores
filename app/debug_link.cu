#include "dae2.cuh"
#include "launcher.cuh"

#include <cuda.h>
#include <vector>
#include <cstdio>

// Minimal test case for debugging TMA stride calculation
int main(int argc, char** argv) {
    constexpr int numSMs = 1;        // Just 1 SM for easy debugging
    constexpr size_t totalBytes = 1024 * 1024 * 64; // 64 MB
    constexpr uint16_t loadBytes = 16 * 1024;

    // blocksize = 16K
    DAELauncher dae {numSMs};
    auto &in = dae.data(totalBytes);
    auto &out = dae.data(totalBytes);

    auto &cbuilder = dae.comp;
    cbuilder.add(cinst(OP_DUMMY, { 1 }));
    cbuilder.add(cinst(OP_TERMINATE));
    
    auto &mbuilder = dae.mem;
    mbuilder.add(minst(
        OP_ALLOC_TMA_LOAD_1D,
        2,
        (uint64_t)in.d_get(),
        loadBytes
    ));
    mbuilder.add(minst(
        linked(OP_ALLOC_WB_TMA_STORE_1D),
        0,
        (uint64_t)out.d_get(),
        loadBytes
    ));
    mbuilder.add(minst(OP_TERMINATE));

    printf("Launching debug_link kernel...\n");
    dae.launch();
}
