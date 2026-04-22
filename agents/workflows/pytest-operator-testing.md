# Pytest Operator Testing

Use `pytest` for operator-level coverage. The first test layer is CPU-only
instruction generation; it should use synthetic `MemoryInstruction` inputs when
real TMA descriptors would require CUDA tensors. GPU correctness tests should
launch the real `Launcher` schedules and compare against a PyTorch reference.
Keep operator tests in per-operator files under `tests/operators/`, with shared
builders in non-collected helper modules such as `tests/operator_cases.py`.

Recommended commands:

```bash
pytest -m "build or no_gpu"
pytest -m "gpu and not perf"
pytest -m perf --run-perf
pytest tests/operators/test_gemv.py
```

Performance tests are smoke tests by default: assert that launcher profile
timing is positive and leave threshold-based regressions for later baselines.
Run them sequentially; do not benchmark multiple GPU jobs in parallel.
