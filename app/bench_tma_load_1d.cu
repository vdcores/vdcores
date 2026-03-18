#include "launcher.cuh"

#include <iostream>
#include <cuda_runtime.h>
#include <vector>

#ifdef DAE_DEBUG_PRINT
  #define DAE_DEBUG
#endif

int main(int argc, char** argv) {
  constexpr int numSMs = 128;
  constexpr size_t loadDataSlots = 2;
  constexpr uint16_t loadDataSize = 1024 * slotSizeKb * loadDataSlots;
  constexpr uint16_t numLoads = 512;

  constexpr size_t dataBytes = 128 * 1024 * 1024; // 128 MB per SM

  static_assert(dataBytes >= loadDataSize * numLoads, "Not enough data per SM");

  DAELauncher dae {numSMs};
  auto &dblock = dae.sm_data(128 * 1024 * 1024); // 128 MB per SM

  auto &cbuilder = dae.comp;
  // Create compute instructions (1 OP_DUMMY + 1 OP_TERMINATE)
  cbuilder.add(cinst(OP_DUMMY, { numLoads }));
  cbuilder.add(cinst(OP_TERMINATE));

  // Create memory instructions (1 OP_ALLOC_TMA_LOAD_1D + 1 OP_TERMINATE)
  auto &mbuilder = dae.mem;
  // Allocate TMA load 1D instructions
  mbuilder.add(minst(
    OP_REPEAT,
    0, // gpr to use. must TARGET_INST_PC - LOOP_START_PC
    // Note loop start PC is PC + 1
    loadDataSize,
    numLoads 
  ));
  // new lambda form to avoid manual for loop over SMs
  mbuilder.add([&](int sm) {
    return minst(
      jump(OP_ALLOC_TMA_LOAD_1D),
      loadDataSlots,
      (uint64_t)dblock.d_get(sm),
      loadDataSize);
  });
  mbuilder.add(minst(OP_TERMINATE));

  // Launch kernel
  float total_bytes = (float)numLoads * loadDataSize * numSMs;
  dae.launch_bench(total_bytes);

  std::cout << "DAE2 kernel executed successfully" << std::endl;
  return 0;
}

