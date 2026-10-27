"""Allow `python benchmark/run.py ...` as well as `python -m benchmark ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.runner import main

if __name__ == "__main__":
    main()
