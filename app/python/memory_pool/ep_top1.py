"""Compatibility entry point for the one-launch VDCores expert-pool demo.

The maintained implementation is :mod:`ep_pool_top1`.  Keeping this filename
avoids breaking older commands while ensuring they no longer execute the
retired two-phase generic-pool/auxiliary-compute harness.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        Path(__file__).with_name("ep_pool_top1.py"),
        run_name="__main__",
    )
