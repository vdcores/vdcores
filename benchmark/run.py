"""Run selected benchmark suites from the repository root."""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description="VDCores benchmarks")
    parser.add_argument("suites", nargs="+", choices=["nvlink"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--num-sms", type=int, default=132)
    parser.add_argument("--sizes-mib", type=int, nargs="+", default=[64, 256])
    parser.add_argument("-b", "--bench", type=int, nargs="?", const=20, default=20)
    args = parser.parse_args()
    for suite in args.suites:
        importlib.import_module(f"benchmark.{suite}").run(args)

if __name__ == "__main__":
    main()
