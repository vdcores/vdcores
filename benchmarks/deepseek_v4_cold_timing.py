"""Shared cold-data CUDA-graph timing helpers for DSV4 microbenchmarks."""

from __future__ import annotations

import torch


def percentile_us(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def cold_graph_timings_us(
    run,
    *,
    stream: torch.cuda.Stream,
    warmup: int,
    samples: int,
    l2_scrub_mib: int,
) -> list[float]:
    """Time one graph invocation after an out-of-interval L2 scrub.

    Code, CUDA modules, and the graph are warm.  Each measured graph contains
    exactly one ``run`` invocation.  A read/modify/write traversal of a buffer
    larger than twice GB200's L2 is ordered immediately before the start event,
    so the returned interval measures cold data rather than scrub time.
    """

    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if samples <= 0:
        raise ValueError("samples must be positive")
    if l2_scrub_mib <= 0:
        raise ValueError("l2_scrub_mib must be positive")

    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run()

    with torch.cuda.stream(stream):
        for _ in range(warmup):
            graph.replay()
    current.wait_stream(stream)
    torch.cuda.synchronize()

    scrub = torch.zeros(
        l2_scrub_mib * 1024 * 1024,
        dtype=torch.uint8,
        device="cuda",
    )
    stream.wait_stream(current)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(samples)]
    with torch.cuda.stream(stream):
        for start, stop in zip(starts, stops):
            scrub.add_(1)
            start.record(stream)
            graph.replay()
            stop.record(stream)
    current.wait_stream(stream)
    torch.cuda.synchronize()
    return [start.elapsed_time(stop) * 1.0e3 for start, stop in zip(starts, stops)]
