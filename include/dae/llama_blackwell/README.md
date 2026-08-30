# Lean Llama-8B Blackwell runtime profile

This directory snapshots the resident VM portion of the qualified full
Blackwell port at `vdcores-blackwell-tasks` commit
`992c390443c31e3aff6c15f37105227f09e78388`. It deliberately contains only
the VM dispatcher and memory-pipeline headers. Llama compute kernels continue
to resolve from the current tree's `include/task/`, so the profile uses the
latest BF16 UMMA tasks without carrying the DeepSeek-only fixed-ring,
barrier-reload, and coupled-stream state in every resident CTA.

The general DeepSeek runtime remains the default. Build the isolated 152-SM
Llama image with:

```bash
make PYTHON=/path/to/python llama8b-blackwell-pyext
```

The target always cleans first because Make does not otherwise treat a change
in runtime-profile variables as an `runtime.o` dependency. The profile must
be used with `benchmarks/blackwell_llama8b_fused_argmax.ops`; do not add
DeepSeek compute or memory instructions to that manifest.
