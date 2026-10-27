"""CLI and suite discovery; GPU imports belong in each suite's run function."""

import argparse
import importlib
import json
import pkgutil
from pathlib import Path

from . import suites


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main(argv=None):
    names = sorted(module.name for module in pkgutil.iter_modules(suites.__path__) if not module.name.startswith("_"))
    parser = argparse.ArgumentParser(description="Compare VDCores with raw systems; suites run sequentially.")
    parser.add_argument("suites", nargs="*", metavar="SUITE", help=f"selected suites: {', '.join(names)}; or all")
    parser.add_argument("--list", action="store_true", help="list available suites without initializing CUDA")
    parser.add_argument("--num-gpus", type=positive_int, default=1, help="explicit number of visible GPUs to use (default: 1)")
    parser.add_argument("-b", "--bench", type=positive_int, nargs="?", const=20, default=20, metavar="ITERATIONS")
    parser.add_argument("--warmup", type=positive_int, default=3)
    parser.add_argument("--sizes-mib", type=positive_int, nargs="+", default=[16, 64, 256], help="buffer sizes in MiB")
    parser.add_argument("--num-sms", type=positive_int, help="VDCores SM count (default: all available SMs)")
    parser.add_argument("--json", type=Path, metavar="PATH", help="write structured measurements")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(names))
        return
    selected = names if args.suites == ["all"] else args.suites
    if not selected:
        parser.error("select a suite or use --list")
    unknown = set(selected) - set(names)
    if unknown:
        parser.error(f"unknown suites: {', '.join(sorted(unknown))}")
    results = []
    for name in dict.fromkeys(selected):
        module = importlib.import_module(f"{suites.__name__}.{name}")
        try:
            results.append({"suite": name, **module.run(args)})
        except (ValueError, RuntimeError) as error:
            parser.exit(1, f"{name}: {error}\n")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
