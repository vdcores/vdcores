from __future__ import annotations

from pathlib import Path
import sys
import unittest


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

import dae  # noqa: E402
from dae.heterogeneous import (  # noqa: E402
    HeterogeneousScheduler,
    OperationSpec,
    OperationState,
)
from dae.placement import MeasuredCrossoverPolicy, PlacementSample  # noqa: E402


def placement_policy() -> MeasuredCrossoverPolicy:
    return MeasuredCrossoverPolicy(
        [
            PlacementSample(64, cpu_latency_ns=100, gpu_latency_ns=180),
            PlacementSample(4096, cpu_latency_ns=600, gpu_latency_ns=200),
        ],
        switch_margin=0.05,
    )


def attention_chain(
    work_items: int, *, policy: MeasuredCrossoverPolicy | None = None
) -> HeterogeneousScheduler:
    scheduler = HeterogeneousScheduler(placement_policy=policy)
    scheduler.add(OperationSpec("attention", 32768, devices=("gpu",)))
    scheduler.add(OperationSpec("flex_dot", work_items, dependencies=("attention",)))
    scheduler.add(
        OperationSpec("gpu_successor", 1, dependencies=("flex_dot",), devices=("gpu",))
    )
    scheduler.seal()
    return scheduler


class HeterogeneousSchedulerTest(unittest.TestCase):
    def test_pure_planning_import_does_not_require_cuda_extension(self) -> None:
        self.assertNotIn("dae.runtime", sys.modules)
        self.assertNotIn("launcher", dae.__dict__)

    def test_dependency_chain_releases_one_stage_at_a_time(self) -> None:
        scheduler = attention_chain(64, policy=placement_policy())
        self.assertEqual([operation.name for operation in scheduler.ready()], ["attention"])

        attention = scheduler.dispatch("attention")
        self.assertEqual(attention.device, "gpu")
        scheduler.complete("attention")
        self.assertEqual([operation.name for operation in scheduler.ready()], ["flex_dot"])

        middle = scheduler.dispatch("flex_dot")
        self.assertEqual(middle.device, "cpu")
        scheduler.complete("flex_dot")
        self.assertEqual(
            [operation.name for operation in scheduler.ready()], ["gpu_successor"]
        )

        successor = scheduler.dispatch("gpu_successor")
        self.assertEqual(successor.device, "gpu")
        scheduler.complete("gpu_successor")
        self.assertTrue(scheduler.is_complete)

    def test_work_size_can_move_flexible_stage_to_gpu(self) -> None:
        scheduler = attention_chain(4096, policy=placement_policy())
        scheduler.dispatch("attention")
        scheduler.complete("attention")
        self.assertEqual(scheduler.dispatch("flex_dot").device, "gpu")

    def test_queue_delay_can_move_flexible_stage_back_to_cpu(self) -> None:
        scheduler = attention_chain(4096, policy=placement_policy())
        scheduler.dispatch("attention")
        scheduler.complete("attention")
        decision = scheduler.dispatch("flex_dot", gpu_queue_ns=500)
        self.assertEqual(decision.device, "cpu")
        self.assertIsNotNone(decision.placement)

    def test_flexible_operation_requires_a_policy(self) -> None:
        scheduler = attention_chain(64)
        scheduler.dispatch("attention")
        scheduler.complete("attention")
        with self.assertRaisesRegex(RuntimeError, "no placement policy"):
            scheduler.dispatch("flex_dot")

    def test_operation_cannot_dispatch_before_dependencies_complete(self) -> None:
        scheduler = attention_chain(64, policy=placement_policy())
        self.assertEqual(scheduler.state("flex_dot"), OperationState.PENDING)
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            scheduler.dispatch("flex_dot")

    def test_unknown_dependency_is_rejected(self) -> None:
        scheduler = HeterogeneousScheduler(
            [OperationSpec("orphan", 1, dependencies=("missing",))]
        )
        with self.assertRaisesRegex(ValueError, "unknown dependencies"):
            scheduler.seal()

    def test_dependency_cycle_is_rejected(self) -> None:
        scheduler = HeterogeneousScheduler(
            [
                OperationSpec("a", 1, dependencies=("b",)),
                OperationSpec("b", 1, dependencies=("a",)),
            ]
        )
        with self.assertRaisesRegex(ValueError, "cycle"):
            scheduler.seal()

    def test_only_running_operations_can_complete(self) -> None:
        scheduler = attention_chain(64, policy=placement_policy())
        with self.assertRaisesRegex(RuntimeError, "not running"):
            scheduler.complete("attention")


if __name__ == "__main__":
    unittest.main()
