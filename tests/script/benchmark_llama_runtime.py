"""Benchmark unchanged Llama schedules with restored inputs between launches.

Examples (after building the required compute operators):
  na run python tests/script/benchmark_llama_runtime.py 8b --checkpoint /model/path
  na run python tests/script/benchmark_llama_runtime.py 8b --checkpoint /model/path --correctness

The checkpoint override affects only Hugging Face loading. Device timing is
max(compute end) - min(compute start), matching Launcher.bench; host timing
includes the synchronous Launcher.launch call. Neither includes buffer reset.
Each invocation should run in a fresh process. The apps use eight request lanes
and one decode step, not a batch-one serving workload.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("1b", "8b"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=101)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--tensor-check", action="store_true",
                        help="Use the 8B app's unrolled single-step tensor check")
    parser.add_argument("--write-compute-ops", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error("samples must be positive and warmup nonnegative")
    if not args.checkpoint.is_dir():
        parser.error("checkpoint must be a local Hugging Face model directory")

    import numpy as np
    import torch
    import transformers
    import dae.runtime
    import dae.util

    torch.manual_seed(1234)
    torch.set_num_threads(8)
    os.environ.setdefault("HF_TOKEN", "")
    checkpoint = str(args.checkpoint.resolve())
    print(f"[checkpoint] {checkpoint}", flush=True)
    for cls in (transformers.AutoModelForCausalLM, transformers.AutoConfig,
                transformers.AutoTokenizer):
        original = cls.from_pretrained

        def local_pretrained(_name, *pos, _original=original, **kw):
            kw.pop("token", None)
            kw["local_files_only"] = True
            return _original(checkpoint, *pos, **kw)

        cls.from_pretrained = local_pretrained

    repo = Path(__file__).resolve().parents[2]
    app = repo / "app/python" / ("llama32_1b" if args.model == "1b" else "llama3") / "sched.py"
    sys.path.insert(0, str(app.parent))
    sys.argv = [str(app), "-N", "1"]
    if args.correctness or args.tensor_check:
        sys.argv += ["--correctness"]
        if args.tensor_check and args.model == "8b":
            sys.argv += ["--no-control-flow"]
        runpy.run_path(str(app), run_name="__main__")
        return

    # Let the original app construct its schedule, then benchmark that launcher.
    dae.util.dae_app = lambda *a, **kw: None
    state = runpy.run_path(str(app), run_name="__main__")
    launcher = state["dae"]
    if args.write_compute_ops:
        args.write_compute_ops.write_text("\n".join(launcher.compute_operator_names()) + "\n")
        return

    # All writable intermediates in these two apps; parameters stay untouched.
    names = ("matTokens", "matHidden", "matRMSHidden", "attnQs", "attnKs", "attnVs",
             "attnO", "matInterm", "matGateOut", "matSiLUOut", "matLogits",
             "matArgmaxIdx", "matArgmaxVal", "matArgmaxOut")
    buffers = []
    for name in names:
        value = state.get(name)
        for tensor in value if isinstance(value, list) else [value]:
            if isinstance(tensor, torch.Tensor):
                buffers.append((tensor, tensor.clone()))

    spans, wall, tokens = [], [], []
    print(f"[launch] warmup={args.warmup} samples={args.samples}", flush=True)
    for i in range(args.warmup + args.samples):
        for dst, src in buffers:
            dst.copy_(src)
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        launcher.launch()
        torch.cuda.synchronize()
        elapsed = time.perf_counter_ns() - start
        data = launcher.profile[:, :2].cpu().numpy()
        span = int(data[:, 1].max()) - int(data[:, 0].min())
        tokens.append(int(state["matTokens"][0, 1].item()))
        if i >= args.warmup:
            spans.append(span)
            wall.append(elapsed)
        if (i + 1) % 20 == 0:
            print(f"[sample] {i + 1}/{args.warmup + args.samples}: {span / 1e6:.6f} ms", flush=True)

    result = {
        "model": args.model, "checkpoint": checkpoint,
        "batch": state["N"], "decode_steps": 1,
        "input_token_and_position": state["input_token_id_and_pos"],
        "samples_ns": spans, "host_launch_ns": wall,
        "tokens": dict(Counter(tokens)), "warmup": args.warmup,
        "median_ms": float(np.median(spans) / 1e6),
        "p10_ms": float(np.percentile(spans, 10) / 1e6),
        "p90_ms": float(np.percentile(spans, 90) / 1e6),
        "host_median_ms": float(np.median(wall) / 1e6),
        "gpu": str(torch.cuda.get_device_properties(0)),
        "na_job": os.environ.get("NA_JOB"),
        "runtime_sha256": hashlib.sha256(Path(dae.runtime.__file__).read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(dae.launcher.__file__).read_bytes()).hexdigest(),
        "compute_ops": launcher.compute_operator_names(),
    }
    print("[result]", json.dumps({k: v for k, v in result.items()
                                  if k not in ("samples_ns", "host_launch_ns")}), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
