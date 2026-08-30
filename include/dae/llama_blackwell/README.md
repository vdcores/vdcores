# Lean dense-model Blackwell runtime profile

This directory snapshots the resident VM portion of the qualified full
Blackwell Llama-8B port at `vdcores-blackwell-tasks` commit
`992c390443c31e3aff6c15f37105227f09e78388`. It deliberately contains only
the VM dispatcher and memory-pipeline headers. Llama compute kernels continue
to resolve from the current tree's `include/task/`, so the profile uses the
latest BF16 UMMA tasks without carrying the DeepSeek-only fixed-ring,
barrier-reload, and coupled-stream state in every resident CTA.

The same lean VM now hosts the Llama-3.2-1B, Qwen3-8B, and Qwen3-1.7B ports.
Their model-specific operator manifests keep unsupported or unused DeepSeek
instructions out of each image.

The general DeepSeek runtime remains the default. Build the isolated 152-SM
Llama image with:

```bash
make PYTHON=/path/to/python llama8b-blackwell-pyext
make PYTHON=/path/to/python llama1b-blackwell-pyext
make PYTHON=/path/to/python qwen3-8b-blackwell-pyext
make PYTHON=/path/to/python qwen3-1b-blackwell-pyext
```

Each target always cleans first because Make does not otherwise treat a change
in runtime-profile variables or operator manifests as a `runtime.o`
dependency. Do not add DeepSeek compute or memory instructions to these dense
model manifests.
