# Tracking `app/python/gemv_out.py` → runtime GEMV operator

This note documents the conversion path from the small GEMV harness in `app/python/gemv_out.py` to the concrete compute operator name that must exist in the built Torch extension.

The intended outcome is that you can:

- identify the canonical compute-op string(s) the harness needs
- generate a minimal compute-op selection file
- rebuild the extension with only those compute ops
- re-run the harness successfully against that selective build

## 1) Harness entry point

`app/python/gemv_out.py` creates a `Launcher`, builds a `GemvLayer`, places the layer schedule on `num_sms`, and executes via `dae_app(...)`.

Key choices:

- the compute atom is `Gemv_M64N8`
- the schedule is provided by `GemvLayer.schedule()` (a `SchedGemv` instance)

## 2) Layer → schedule (Python)

`GemvLayer` lives in `python/dae/model.py`.

It:

- validates `A/B/C` shapes
- derives logical `MNK`
- builds 3 `TmaTensor` objects (`loadA`, `loadB`, `storeC/reduceC`)
- returns a `SchedGemv` schedule through `GemvLayerBase.schedule()`

`SchedGemv` lives in `python/dae/schedule.py`.

It:

- maps each SM to a `(m, k)` tile slice (with fold across K)
- emits one compute instruction per SM: `self.Atom(kTiles, ...)`
- emits the memory-side `RepeatM.onSync(...)` sequence of `TmaTensor.cord(...)` loads
- emits a final `storeC.cord(...)` (store or reduce depending on fold/reduction mode)

## 3) The GEMV compute operator identity (canonical name)

The compute instruction wrapper `Gemv_M64N8` is defined in `python/dae/instructions.py`.

Unlike older static compute ops (e.g. `OP_GEMM_M64N64`), GEMV uses a compute-family reference:

- `Gemv_M64N8` stores `opcode = family_ref("GEMV_WGMMA", M=64, N=8, K=256, BLOAD=4, RESIDUAL=...)`
- the *canonical operator name* is derived from that family reference and later resolved to a numeric opcode from `dae.runtime.opcode` at serialization time

Compute families are declared in `include/dae/opcode.cuh.inc` via `DAE_DEFINE_COMP_FAMILY(...)`, including:

- `GEMV_WGMMA` (fields include `M`, `N`, `K`, `BLOAD`, `RESIDUAL`)
- `GEMV_MMA`

The CUDA task implementation for GEMV is in `include/task/gemv.cuh` (for example `task_gemv(...)`).

## 4) Extract the required compute-ops for `gemv_out.py`

The app-level helper `dae_app(...)` exposes `--write-compute-ops` (see `python/dae/util.py`).

From a working environment (one where the extension can load and the harness can build its schedule), run:

```bash
python app/python/gemv_out.py --write-compute-ops gemv.compute_ops.txt
```

This writes one canonical operator name per line. For GEMV-family ops, expect names of the form:

- `OP_GEMV_WGMMA__M_...__N_...__K_...__BLOAD_...__RESIDUAL_...`

## 5) Rebuild with a minimal compute-op set

The selective-build generator (`tools/generate_selected_compute_ops.py`) prefers:

1. `DAE_COMPUTE_OPS` (comma-separated op names)
2. `DAE_COMPUTE_OPS_FILE` (one op name per line)
3. repo-root `dae_compute_ops.vdcore.build`

To build using the harness-generated op list:

```bash
DAE_COMPUTE_OPS_FILE=gemv.compute_ops.txt make pyext
```

## 6) Re-run (smoke test)

```bash
python app/python/gemv_out.py --launch
```

If the extension was built without the required compute op(s), the launcher should reject the schedule before launch (see notes in `agents/knowledge/project-map.md` about `Launcher.launch()` verifying `runtime.supported_compute_ops`).

