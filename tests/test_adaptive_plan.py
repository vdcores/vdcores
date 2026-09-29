from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tests" / "script" / "run_adaptive_attention_handoff.py"


def calibration() -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment": "synthetic_test_fixture",
        "rows": [
            {
                "work_items": 64,
                "cpu": {"end_to_end_ns": {"median_ns": 100_000}},
                "gpu": {"end_to_end_ns": {"median_ns": 180_000}},
            },
            {
                "work_items": 4096,
                "cpu": {"end_to_end_ns": {"median_ns": 600_000}},
                "gpu": {"end_to_end_ns": {"median_ns": 200_000}},
            },
        ],
    }


class AdaptivePlanCliTest(unittest.TestCase):
    def run_plan(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            policy_path = Path(directory) / "policy.json"
            policy_path.write_text(json.dumps(calibration()), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--policy",
                    str(policy_path),
                    "--plan-only",
                    *arguments,
                ],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

    def test_plan_only_selects_cpu_without_importing_cuda(self) -> None:
        completed = self.run_plan("--work-items", "64")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["mode"], "plan_only")
        self.assertEqual(result["decision"]["device"], "cpu")
        self.assertEqual(
            [stage["device"] for stage in result["plan"]],
            ["gpu", "cpu", "gpu"],
        )

    def test_plan_only_selects_gpu_at_larger_measured_size(self) -> None:
        completed = self.run_plan("--work-items", "4096")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["decision"]["device"], "gpu")

    def test_gpu_queue_delay_can_change_the_adaptive_plan(self) -> None:
        completed = self.run_plan(
            "--work-items", "4096", "--gpu-queue-us", "500"
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["decision"]["device"], "cpu")

    def test_unmeasured_size_is_rejected_by_default(self) -> None:
        completed = self.run_plan("--work-items", "8192")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("outside measured range", completed.stderr)


if __name__ == "__main__":
    unittest.main()
