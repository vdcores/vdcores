# Multi-GPU Launcher Notes

These notes summarize the minimal multi-GPU runtime path added for NVLink-connected GPUs.

## Public Surface

- `Launcher(..., gpu_ids=[physical_gpu_ids...])` enables multi-GPU mode.
- `Schedule.place(num_sms, base_sm=0, gpu=virtual_gpu_id)` targets a virtual GPU index, starting at `0`.
- When `gpu=None`, placement still uses the launcher SM space; in multi-GPU mode that is the flattened default launcher allocation over `gpu_ids`.

## Runtime Shape

- One logical launcher now owns one per-GPU runtime context.
- Each selected GPU gets its own:
  - instruction buffers
  - TMA descriptor copy
  - barrier storage
  - profile buffer
- The Torch extension now has device-aware launch/setup entrypoints and a peer-access helper.

## Verification Harness

- Keep the original single-GPU baseline in [`app/python/tmacopy.py`](/root/vdcores/app/python/tmacopy.py).
- Use [`app/python/tmacopy_multigpu.py`](/root/vdcores/app/python/tmacopy_multigpu.py) for cross-GPU copy checks.
- The new harness covers:
  - explicit virtual-GPU placement
  - flattened global-SM placement
  - forward and reverse source/destination direction
  - simultaneous bidirectional peer copies with per-direction and aggregate bandwidth reporting
- The harness mirrors the known-good single-GPU copy pattern with `Copy(num_loads, load_bytes)` plus `RepeatM.on(...)` and per-step `TmaLoad1D(..., bytes=load_bytes)` / `TmaStore1D(..., bytes=load_bytes)` calls; treating the full `[num_loads, load_bytes]` tensor as one `SchedCopy` overflows the 16-bit memory-instruction size field.
- For explicit-only multi-GPU launches, `Launcher(num_sms=0, gpu_ids=[...])` avoids allocating any default flattened SM placement and lets the explicit `place(..., gpu=...)` schedules define all launched SM slots.

## Verification Preconditions

- For an NVLink-specific validation, confirm the chosen GPU pair reports an `NV#` topology in `nvidia-smi topo -m`.
- CUDA peer access can still report `True` on non-`NV#` topologies, so peer-access capability alone is not a sufficient NVLink verification check.

## Practical Caveat

- In this repository state, multi-GPU explicit placement is routed in the launcher before instruction build so the existing schedule instruction model stays intact.
- Raw broadcast instructions such as `TerminateC()` / `TerminateM()` are still the expected way to close out launched SM slots, including the extra per-GPU launch prefixes introduced by explicit placement.

## Performance Debugging Notes

- For multi-GPU runs, do not compute one cross-device execution span from the flattened `dae.profile` view. The per-SM timestamps come from each GPU's local `globaltimer`, so an aggregate `max(end) - min(start)` across both devices is not meaningful.
- For multi-GPU timing, read each context separately through `dae.contexts[vgpu].profile[:launch_sms]` and compare per-device spans instead.
- Keep benchmark-specific measurement logic in app-side harnesses instead of leaving launcher-side caching experiments in `python/dae/launcher.py`.
- Use [`app/python/tmacopy_multigpu_bench.py`](/root/vdcores/app/python/tmacopy_multigpu_bench.py) as the current local benchmark helper for:
  - single-GPU `tma1d`-style profile-vs-event timing
  - bidirectional DAE peer-copy timing
  - plain PyTorch peer-copy comparison
- On this host, plain PyTorch peer copies are still much faster than the current DAE peer-copy harness, so the remaining gap is not a system-wide NVLink limitation.
