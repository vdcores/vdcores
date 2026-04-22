from __future__ import annotations

import subprocess

import pytest


class TestBuild:
    @pytest.mark.build
    def test_make_pyext_target_is_wired(self, repo_root):
        result = subprocess.run(
            ["make", "-n", "pyext"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "runtime.o" in result.stdout
        assert "pip install -e ." in result.stdout

    @pytest.mark.build
    def test_make_pyext_builds_when_requested(self, repo_root, request):
        if not request.config.getoption("--run-build"):
            pytest.skip("full extension rebuild requires --run-build")
        subprocess.run(["make", "pyext"], cwd=repo_root, check=True)
