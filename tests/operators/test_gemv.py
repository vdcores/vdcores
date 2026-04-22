from __future__ import annotations

import pytest

from operator_cases import (
    assert_mean_relative_close,
    benchmark_case,
    build_gemv_codegen_case,
    build_gemv_out_case,
    launch_and_sync,
)


class TestGemvOperator:
    @pytest.mark.no_gpu
    def test_codegen_without_gpu(self, dae_runtime):
        codegen = build_gemv_codegen_case()
        assert codegen.compute_ops
        assert codegen.avg_compute_insts > 0
        assert codegen.avg_memory_insts > 0

    @pytest.mark.gpu
    def test_correctness_gpu(self, cuda_device):
        case = build_gemv_out_case(cuda_device)
        launch_and_sync(case)
        assert_mean_relative_close(
            case.reference(),
            case.result(),
            max_relative_diff=0.015,
        )

    @pytest.mark.gpu
    @pytest.mark.perf
    def test_performance_smoke_gpu(self, cuda_device):
        stats = benchmark_case(build_gemv_out_case(cuda_device))
        assert stats["mean_ns"] > 0
        assert stats["max_ns"] >= stats["min_ns"] > 0
