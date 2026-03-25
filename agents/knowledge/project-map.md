# Project Map

This note summarizes the stable structure confirmed during repository initialization.

## Main Entry Points

- `README.md`: high-level project overview, setup path, and demo commands.
- `Makefile`: builds `runtime.o` from `src/runtime.cu` and installs the Python extension with `make pyext`.
- `setup.py`: packages the `dae` Python module and compiles `src/torch_runtime.cu` linked against `runtime.o`.
- `setup.sh`: reference environment bootstrap for Conda, CUDA Toolkit 13.0.2, and Python dependencies.

## Code Areas

- `app/python/`: runnable experiments and schedule prototypes.
- `app/python/llama3/sched.py`: primary end-to-end decoding demo for `meta-llama/Llama-3.1-8B-Instruct`.
- `app/python/llama32_1b/sched.py`: isolated Llama 3.2 1B scheduling path with a `--dry-build` mode that validates the Python schedule before the remaining low-level runtime support is added.
- `app/python/mistral_small_24b/`: Mistral Small 24B single-token decode port with manual RoPE-table construction, `QW != hidden_size` attention wiring, and a 132-SM logits/argmax path.
- `app/python/llama3/reference.py` and `app/python/llama3/llama_attention_reference.py`: local reference helpers worth checking before re-deriving model math.
- `app/python/attention_simple_decoding.py`: dedicated isolated GQA decode-attention harness for validating the shared attention opcode path and collecting quick single-kernel timing.
- `app/python/gemv_mma_out.py`: dedicated correctness harness for the isolated `N=8` MMA GEMV operator path.
- `app/python/qwen3/`: Qwen 3 client, layer, utilities, and schedule variants.
  The current decode path is split across `sched.py` (graph/TMA/instruction scheduling), `runtime_context.py` (HF model load, tensor materialization, packed side-input prep, KV bootstrap), `correctness.py` (reference comparisons), and `cli.py` (prefiltered app args).
- `python/dae/launcher.py`: launcher/resource-management entry point and public compatibility surface for legacy `from dae.launcher import *` usage.
- `python/dae/instructions.py`: serialized instruction types, compute operation definitions, memory-side instruction helpers, and TMA instruction wrappers used by `launcher.py`.
- `python/dae/op_families.py`: minimal dynamic compute-op family registry; it now loads the declarative family definitions exported by `src/torch_runtime.cu` as `dae.runtime.compute_family_specs`, builds canonical family-backed op refs through a generic `family_ref(...)` helper, validates canonical dynamic names such as `OP_GEMV_WGMMA__...` against that runtime-exported source, and leaves concrete opcode instances to generated build artifacts.
- `python/dae/op_family_specs.py`: shared parser helpers for declarative compute-family definitions. Runtime Python parses the extension-exported spec objects through it, while the build-time generator reuses the same parsing rules against `include/dae/opcode.cuh.inc`.
- `python/dae/instruction_utils.py`: small opcode/packing helpers shared by the instruction and op modules.
- `python/dae/instruction_utils.py`: small opcode/packing helpers shared by the instruction and op modules; it now also owns compute-op family normalization, lazy runtime-opcode resolution, and compute-instruction tensor packing so `python/dae/instructions.py` stays mostly declarative.
- `python/dae/util.py`: CLI helpers including instruction dumps, profiling output, and `--write-compute-ops` generation of a default `dae_compute_ops.vdcore.build` build-selection file from a built launcher.
- `python/dae/schedule.py`: scheduling interface and composition layer.
- `python/dae/model.py`: model-side Python support code.
- `include/dae/`: runtime abstractions such as allocator, launcher, queues, runtime, and virtual cores.
  The compute warp dispatch now lives in `include/dae/compute_dispatch.cuh`, and supported selective-build ops are discovered from `DAE_COMPUTE_OP_HANDLER(OP_...)` declarations in that file.
- `include/task/`: CUDA task building blocks including attention, GEMV, RMSNorm, RoPE, SiLU, WGMMA, and argmax.
- `src/runtime.cu`: runtime implementation compiled to `runtime.o`.
- `src/torch_runtime.cu`: Torch extension binding source.
- `tools/generate_selected_compute_ops.py`: build-time helper that prefers `DAE_COMPUTE_OPS`, then `DAE_COMPUTE_OPS_FILE`, then a repo-root `dae_compute_ops.vdcore.build`, and emits both `build/generated/dae/selected_compute_ops.inc` and `build/generated/dae/compute_opcode_order.inc` for the selective-build flow.
  It also emits `build/generated/dae/dynamic_compute_handlers.inc` for any selected dynamic op-family handlers.

## Operational Notes

- The repository currently includes built extension artifacts under `python/dae/`.
- Full runtime verification may require Hopper-class CUDA hardware.
- For Python-only edits, start with light checks such as `python -m py_compile`.
- `app/python/llama3/sched.py` now includes a `--correctness` mode for a single-token, single-decoding-step validation against `app/python/llama3/reference.py`.
- `python/dae/schedule.py` now treats SM-count placement as a post-construction concern across the main scheduler classes, including `SchedArgmax`.
- `python/dae/launcher.py` and `app/python/llama3/sched.py` now support late-bound barrier counts: barrier ids are still built early, and the llama path now binds selected placement-dependent layer/system barriers from a generic scan of the placed schedule bundle's barrier-releasing memory instructions rather than from a handwritten per-bar table.
- `python/dae/launcher.py` now exposes `extract_compute_operator_names(...)` and `Launcher.compute_operator_names()`, and `Launcher.launch()` now rejects schedules whose required compute ops are not present in the built extension's `runtime.supported_compute_ops`.
- When `build/generated/dae/compute_opcode_order.inc` is present, `include/dae/opcode.cuh.inc` uses it to renumber compute opcodes dynamically: selected build ops receive the first contiguous values, and remaining compute ops are appended afterward so `dae.runtime.opcode` still exposes the full compute-op namespace.
- `python/dae/instructions.py` now keeps fixed compute instructions numeric-first, but allows registered op-family instructions to store a canonical family string and resolve the numeric opcode lazily from `dae.runtime.opcode` during tensor serialization.
- `include/dae/opcode.cuh.inc` is now the source of truth for compute-family declarations through `DAE_DEFINE_COMP_FAMILY(...)` entries. `src/torch_runtime.cu` exports those declarations to Python as `runtime.compute_family_specs`, so runtime Python no longer reads the file directly, while concrete opcode instances are still generated only for the selected Python-requested names.
- `Makefile` now routes compute-op generation through a single generated stamp file so the generator runs once per build even though `runtime.o` depends on multiple generated include outputs.
- `python/dae/instructions.py` and `include/dae/pipeline/allocwarp.cuh` now support non-power-of-two embedding row widths through `OP_CC0_ROW_BYTES`, so `CC0(...)` is no longer limited to power-of-two row-byte sizes.
- `src/torch_runtime.cu` now clamps cache-policy windows to the device `accessPolicyMaxWindowSize`, which avoids `cudaStreamSetAttribute(... invalid argument)` on larger model weights.
- The llama/qwen shared-memory SiLU stages are no longer only inline callables in the app scripts; they now have dedicated schedule classes in `python/dae/schedule.py` for the interleaved phase and the fused register-backed phase.
- `app/python/llama3/sched.py` now follows the newer schedule-construction style: build dependency-only schedules first, attach mostly-static bars immediately after construction, then apply `place(...)` in a grouped step before submission to `dae.i(...)`.
