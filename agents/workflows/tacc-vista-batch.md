# TACC Vista Batch Workflow

Use this workflow while the project has a limited Vista allocation. The August
2026 project guidance supersedes older `idev` examples in the shared guide and
in `experimental/nvshmem/README.md`.

## Rules

- Edit, inspect, and use Git on the Vista login node.
- Compile and run GPU code only through `sbatch`.
- Do not use `idev`, SSH into an allocated node for development, or submit a
  sleeping job to hold a node.
- Keep jobs short and run one GPU experiment at a time.

## Initial Setup

Clone the repository on the login node, then submit the one-time setup job from
the repository root:

```bash
cd "$HOME"
git clone https://github.com/vdcores/vdcores.git
cd vdcores
sbatch -A <allocation> tools/tacc/vista_setup.sbatch
```

The setup job installs Miniconda only when `$HOME/miniconda3` is absent and
installs the CUDA 13 Python environment on a GH200 compute node. The focused
smoke job below performs the first build, avoiding a duplicate compilation. Do
not run `setup.sh` directly from the login node.

## Split-Attention Smoke Test

Submit the focused build and correctness check from the repository root:

```bash
sbatch -A <allocation> tools/tacc/vista_attention_smoke.sbatch
```

Vista's site `sbatch` wrapper prints validation text before the job id, even
with `--parsable`. Do not capture its full output with `JOB=$(sbatch ...)`.
Read the final numeric job id from the submission output, then assign it
separately when needed, for example `JOB=123456`.

The job builds only the two compute operators needed by
`attention_split_kv.py`, launches the example with a timeout, and compares the
result with its PyTorch reference. To reuse an already compatible extension,
let `sbatch` inherit the variable from the submitting shell. Vista's
documentation recommends avoiding the `--export` option:

```bash
SKIP_BUILD=1 sbatch -A <allocation> tools/tacc/vista_attention_smoke.sbatch
```

Use `SKIP_BUILD=1` only when neither CUDA/C++ code nor the selected operation
set has changed.

## Monitoring

Before each submission, confirm the current node role, live partition capacity,
and existing jobs:

```bash
hostname -f
qlimits
sinfo -p gh,gh-dev -S+P -o "%18P %8a %20F"
squeue -p gh,gh-dev
squeue -u "$USER"
```

After submission, monitor the selected job sparingly:

```bash
squeue -j <job-id>
sacct -j <job-id> --format=JobID,State,Elapsed,ExitCode
less vdcores-attn-<job-id>.out
less vdcores-attn-<job-id>.err
```

Record the job id, Git commit, module versions, CUDA version, PyTorch version,
and correctness result before changing the attention schedule.
