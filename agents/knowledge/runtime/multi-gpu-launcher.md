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

- Keep the original single-GPU baseline in [`app/python/tmacopy.py`](/workspace/vdcores/app/python/tmacopy.py).
- Use [`app/python/tmacopy_multigpu.py`](/workspace/vdcores/app/python/tmacopy_multigpu.py) for cross-GPU copy checks.
- The new harness covers:
  - explicit virtual-GPU placement
  - flattened global-SM placement
  - forward and reverse source/destination direction

## Practical Caveat

- In this repository state, multi-GPU explicit placement is routed in the launcher before instruction build so the existing schedule instruction model stays intact.
- Raw broadcast instructions such as `TerminateC()` / `TerminateM()` are still the expected way to close out launched SM slots, including the extra per-GPU launch prefixes introduced by explicit placement.
