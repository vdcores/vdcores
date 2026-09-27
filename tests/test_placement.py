from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "python" / "dae" / "placement.py"
SPEC = importlib.util.spec_from_file_location("dae_placement_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {MODULE_PATH}")
PLACEMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLACEMENT
SPEC.loader.exec_module(PLACEMENT)

MeasuredCrossoverPolicy = PLACEMENT.MeasuredCrossoverPolicy
PlacementSample = PLACEMENT.PlacementSample


class MeasuredCrossoverPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = MeasuredCrossoverPolicy(
            [
                PlacementSample(100, cpu_latency_ns=100, gpu_latency_ns=200),
                PlacementSample(400, cpu_latency_ns=400, gpu_latency_ns=100),
            ]
        )

    def test_exact_measurements_choose_the_faster_device(self) -> None:
        self.assertEqual(self.policy.choose(100).device, "cpu")
        self.assertEqual(self.policy.choose(400).device, "gpu")

    def test_interpolates_in_log_work_size(self) -> None:
        decision = self.policy.choose(200)
        self.assertEqual(decision.cpu_estimate_ns, 250)
        self.assertEqual(decision.gpu_estimate_ns, 150)
        self.assertEqual(decision.device, "gpu")

    def test_queue_delay_can_flip_the_choice(self) -> None:
        decision = self.policy.choose(200, gpu_queue_ns=200)
        self.assertEqual(decision.device, "cpu")
        self.assertEqual(decision.cpu_total_ns, 250)
        self.assertEqual(decision.gpu_total_ns, 350)

    def test_deadline_overrides_preference_when_only_one_device_meets_it(self) -> None:
        policy = MeasuredCrossoverPolicy(
            [PlacementSample(100, cpu_latency_ns=100, gpu_latency_ns=80)],
            preferred_device="cpu",
            switch_margin=0.5,
        )
        decision = policy.choose(100, deadline_ns=90)
        self.assertEqual(decision.device, "gpu")
        self.assertIn("deadline", decision.reason)

    def test_switch_margin_avoids_a_small_device_flip(self) -> None:
        policy = MeasuredCrossoverPolicy(
            [PlacementSample(100, cpu_latency_ns=100, gpu_latency_ns=95)],
            preferred_device="cpu",
            switch_margin=0.1,
        )
        self.assertEqual(policy.choose(100).device, "cpu")

    def test_out_of_range_request_uses_nearest_measurement(self) -> None:
        low = self.policy.choose(10)
        high = self.policy.choose(1000)
        self.assertTrue(low.clamped_to_measured_range)
        self.assertTrue(high.clamped_to_measured_range)
        self.assertEqual((low.cpu_estimate_ns, low.gpu_estimate_ns), (100, 200))
        self.assertEqual((high.cpu_estimate_ns, high.gpu_estimate_ns), (400, 100))

    def test_builds_from_benchmark_json(self) -> None:
        result = {
            "rows": [
                {
                    "work_items": 64,
                    "cpu": {"end_to_end_ns": {"median_ns": 120}},
                    "gpu": {"end_to_end_ns": {"median_ns": 90}},
                }
            ]
        }
        policy = MeasuredCrossoverPolicy.from_benchmark(result)
        self.assertEqual(policy.measured_range, (64, 64))
        self.assertEqual(policy.choose(64).device, "gpu")

    def test_rejects_duplicate_work_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            MeasuredCrossoverPolicy(
                [
                    PlacementSample(100, 100, 200),
                    PlacementSample(100, 90, 180),
                ]
            )


if __name__ == "__main__":
    unittest.main()
