#include "dae2.cuh"
#include "launcher.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#ifdef DAE_DEBUG_PRINT
  #define DAE_DEBUG
#endif

int main(int argc, char** argv) {
  constexpr int numSMs = 128;
  size_t dataSlots = 2;
  uint16_t dataBytes = 1024 * slotSizeKb * dataSlots;
  uint16_t numLoads = 96;

  DAELauncher dae {numSMs};
  auto &dblock = dae.sm_data(128 * 1024 * 1024); // 128 MB per SM

  auto &cbuilder = dae.comp;
  // Create compute instructions (1 OP_DUMMY + 1 OP_TERMINATE)
  cbuilder.add(cinst(OP_DUMMY, { numLoads }));
  cbuilder.add(cinst(OP_TERMINATE));

  // Create memory instructions (1 OP_ALLOC_TMA_LOAD_1D + 1 OP_TERMINATE)
  auto &mbuilder = dae.mem;
  // Allocate TMA load 1D instructions
  for (int sm = 0; sm < numSMs; sm++) {
    for (int i = 0; i < numLoads; i++) {
      mbuilder.add(sm, minst(
        OP_ALLOC_WB_TMA_STORE_1D,
        dataSlots,
        (uint64_t)dblock.d_get(sm, i * dataBytes),
        dataBytes
      ));
    }
  }
  mbuilder.add(minst(OP_TERMINATE));

  // Launch kernel
  float total_bytes = (float)numLoads * dataBytes * numSMs;
  dae.launch_bench(total_bytes);

  return 0;
}