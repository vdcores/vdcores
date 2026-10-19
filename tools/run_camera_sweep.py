"""Run the camera graph evaluation sequentially; never overlap GPU experiments."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--tail-iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("choose a fresh output directory")
    cases = [("small-1", 128, 192, 1, []), ("small-4", 128, 192, 4, []),
             ("large-1", 480, 640, 1, []), ("large-4", 480, 640, 4, []),
             ("large-4-cpu-load", 480, 640, 4, ["--cpu-load-workers", "2"]),
             ("large-4-gpu-load", 480, 640, 4, ["--gpu-load"]),
             ("large-4-period10ms", 480, 640, 4, ["--period-ms", "10"]),
             ("large-4-period2ms", 480, 640, 4, ["--period-ms", "2"]),
             ("large-4-tail", 480, 640, 4, [])]
    for name, h, w, lanes, extra in cases:
        print(f"Starting {name}", flush=True)
        command = [sys.executable, str(ROOT / "tools/run_camera_control.py"),
                   "--height", str(h), "--width", str(w), "--lanes", str(lanes),
                   "--iterations", str(args.tail_iterations if name.endswith("tail") else args.iterations),
                   "--warmup", "20", "--repeats", str(args.repeats), "--deadline-ms", "10",
                   "--output", str(args.output_dir / f"{name}.json.gz"), *extra]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
