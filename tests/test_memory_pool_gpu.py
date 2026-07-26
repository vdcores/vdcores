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
        "512",
    )
    assert "sharded EP PASS tokens=8 hidden=512" in output


@pytest.mark.skipif(not RUN_GPU, reason="set DAE_RUN_NVSHMEM_GPU_TESTS=1")
def test_single_pe_sharded_pool_topk_pipeline_with_aligned_rows():
    output = _run(
        "app/python/memory_pool/ep_pool_top1.py",
        "--tokens-per-pe",
        "13",
        "--hidden-size",
        "512",
        "--experts-per-pe",
        "2",
        "--top-k",
        "2",
    )
    assert (
        "sharded EP PASS tokens=13 hidden=512 experts=2 received=26 top_k=2"
        in output
    )


@pytest.mark.skipif(not RUN_GPU, reason="set DAE_RUN_NVSHMEM_GPU_TESTS=1")
def test_single_pe_pool_owned_dynamic_read():
    output = _run(
        "app/python/memory_pool/pool_slice_dynamic_read.py",
        "--tokens-per-pe",
        "13",
        "--hidden-size",
        "512",
        "--readers-per-pe",
        "1",
        "--top-k",
        "1",
        "--warmup",
        "0",
        "--iterations",
        "1",
    )
    assert (
        "pool-slice dynamic-read PASS tokens=13 hidden=512 readers=1 top_k=1"
        in output
    )
