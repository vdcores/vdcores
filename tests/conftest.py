from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))


def pytest_addoption(parser):
    parser.addoption(
        "--run-build",
        action="store_true",
        default=False,
        help="run build tests that invoke make pyext",
    )
    parser.addoption(
        "--run-perf",
        action="store_true",
        default=False,
        help="run GPU performance smoke tests",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-perf"):
        return

    skip_perf = pytest.mark.skip(reason="performance tests require --run-perf")
    for item in items:
        if "perf" in item.keywords:
            item.add_marker(skip_perf)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def dae_runtime():
    return pytest.importorskip("dae.runtime")


@pytest.fixture(scope="session")
def cuda_device(dae_runtime):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        pytest.skip(
            "VDCores tests require a Hopper-class CUDA device "
            f"(got capability {capability})"
        )
    return torch.device("cuda")
