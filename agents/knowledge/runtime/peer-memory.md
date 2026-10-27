# Peer memory and NVLink

- `Launcher(..., device="cuda:0", num_gpus=2)` explicitly enables all ordered
  peer mappings among the first two **visible** CUDA devices. The default of one
  GPU rejects peer tensors. A one-GPU launcher can still target any visible GPU.
- `src/torch_runtime.cu` uses Torch's `CUDACachingAllocator::enablePeerAccess`
  hook, which also maps expandable segments and async allocator pools. Allocate
  tensors normally with Torch on their owning device; no separate allocator,
  NVSHMEM, or extra CUDA link library is needed. Hardware chooses NVLink when
  available; peer capability alone does not prove an NVLink connection.
- Existing `TmaLoad1D`, `TmaStore1D`, and `RawAddress` instructions carry peer UVA
  pointers unchanged. Memory instructions retain their backing tensors,
  including through `copy()`. VM slots stay local to each executing GPU.
- Multi-GPU launchers share one barrier tensor and ID namespace on GPU 0.
  The `bars_src` reset template used by model schedules is shared on GPU 0 too.
  Declare/bind all barrier counts before launching. The first launch initializes
  the table; later GPU launches use it without resetting live counters. After all
  GPUs finish a run, call `launcher.reset_barriers()` once before the next run.
  Single-GPU launchers keep local barriers and reset them on each launch.
- Finish producers on the owning GPU before consumers on another GPU, using
  CUDA events, shared schedule barriers, or explicit synchronization. Construct
  one launcher per executing GPU and keep concurrent data writes disjoint.
- Launch uses the launcher's device and that device's current stream, restores
  the caller's device. Barrier waits/decrements use system-scope acquire/release
  atomics. Synchronous launch waits only on its stream. The C++ launcher checks
  device placement and reports launch/execution errors; the default remains
  synchronous.
- The native `runtime.launch_dae(..., synchronize=False)` can enqueue independent
  kernels on multiple GPUs; callers must keep buffers alive until completion.
  Upload and timing helpers stay inside `benchmark/nvlink.py`.
- Rebuild with `make pyext`, then run
  `na run -n=2 python -m unittest discover -s tests -p test_nvlink.py`.
