"""Opt-in singleton-NVSHMEM GPU integration tests.

Set ``DAE_RUN_NVSHMEM_GPU_TESTS=1`` after ``make nvshmem-pyext``. Multi-PE
coverage remains an ``ibrun`` application test on Vista.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


RUN_GPU = os.environ.get("DAE_RUN_NVSHMEM_GPU_TESTS") == "1"
ROOT = Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> str:
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return completed.stdout


@pytest.mark.skipif(not RUN_GPU, reason="set DAE_RUN_NVSHMEM_GPU_TESTS=1")
def test_single_pe_sixteen_write_dependency_fan_in():
    output = _run(
        "app/python/memory_pool/dependent_rw.py",
        "--writes-per-pe",
        "16",
        "--elements",
        "256",
    )
    assert "dependent RW PASS writes=16 value=136" in output


@pytest.mark.skipif(not RUN_GPU, reason="set DAE_RUN_NVSHMEM_GPU_TESTS=1")
def test_single_pe_top1_ep_scatter_compute_gather():
    output = _run(
        "app/python/memory_pool/ep_top1.py",
        "--tokens-per-pe",
        "8",
        "--hidden-size",
        "32",
    )
    assert "top-1 EP PASS tokens=8 hidden=32" in output
