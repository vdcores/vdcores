"""Run an embedded-loop experiment matrix sequentially on one machine."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def integers(value):
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or len(set(values)) != len(values) or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("expected unique nonnegative integers")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--work-items", type=integers, default=[64, 4096, 65536])
    parser.add_argument("--lanes", type=integers, default=[1, 4])
    parser.add_argument("--workers", type=integers, default=[2])
    parser.add_argument("--poll-us", type=integers, default=[0, 50])
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--deadline-ms", type=float, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any(value <= 0 for value in args.work_items + args.lanes + args.workers):
        parser.error("work-items, lanes, and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for work, lanes, workers, poll in itertools.product(args.work_items, args.lanes, args.workers, args.poll_us):
        path = args.output_dir / f"n{work}-lanes{lanes}-workers{workers}-poll{poll}.json"
        subprocess.run([
            sys.executable, str(ROOT / "tests/script/run_embedded_loop.py"),
            "--backend", args.backend, "--work-items", str(work),
            "--lanes", str(lanes), "--workers", str(workers), "--poll-us", str(poll),
            "--iterations", str(args.iterations), "--warmup", str(args.warmup),
            "--repeats", str(args.repeats), "--deadline-ms", str(args.deadline_ms),
            "--output", str(path)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
