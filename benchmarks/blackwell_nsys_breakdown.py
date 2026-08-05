#!/usr/bin/env python3
"""Summarize kernel clusters in an Nsight Systems SQLite export.

The fixed-context vLLM profiler captures one prefill followed by one decode.
This helper keeps the large ``.nsys-rep``/SQLite files worker-local and emits a
small text breakdown that is convenient to retain with benchmark results.
"""

from __future__ import annotations

import argparse
import collections
import sqlite3


def short_name(name: str) -> str:
    if name.startswith("nvjet_sm100_"):
        base, launch_kind = name.split("_bz_", 1)
        if launch_kind.startswith("splitK"):
            return f"{base}_splitK"
        return base
    if name.startswith("fmhaSm100"):
        if "ForGen" in name:
            return "flashinfer_decode"
        return "flashinfer_prefill"
    if "reshape_and_cache_flash_kernel" in name:
        return "kv_cache_store"
    if "fused_add_rms_norm" in name:
        return "fused_add_rms_norm"
    if "fused_mul_silu" in name:
        return "silu_mul"
    if name.startswith("triton_poi_fused_3"):
        return "rope"
    if "splitKreduce_kernel" in name:
        return "split_k_reduce"
    return name.split("<", 1)[0].removeprefix("void ")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", help="worker-local Nsight Systems SQLite export")
    parser.add_argument(
        "--gap-us",
        type=float,
        default=50.0,
        help="idle gap that separates GPU kernel clusters",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument(
        "--all-clusters",
        action="store_true",
        help="print every GPU cluster instead of only the largest decode cluster",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(args.sqlite)
    strings = dict(connection.execute("select id, value from StringIds"))
    ranges = []
    for start, end, text, text_id in connection.execute(
        "select start, end, text, textId from NVTX_EVENTS "
        "where end is not null order by start"
    ):
        ranges.append((start, end, text or strings.get(text_id, "")))

    kernels = []
    for row in connection.execute(
        "select start, end, shortName, correlationId, "
        "gridX, gridY, gridZ, blockX, blockY, blockZ "
        "from CUPTI_ACTIVITY_KIND_KERNEL order by start"
    ):
        start, end, name_id, correlation_id, *launch = row
        kernels.append(
            (start, end, strings[name_id], correlation_id, tuple(launch))
        )

    print("NSYS_NVTX_RANGES")
    for start, end, name in ranges:
        print(f"  cpu_us={(end - start) / 1e3:.3f} name={name}")

    if not kernels:
        raise RuntimeError("Nsight export contains no CUDA kernels")

    gap_ns = int(args.gap_us * 1e3)
    clusters: list[list[tuple]] = [[kernels[0]]]
    cluster_end = kernels[0][1]
    for kernel in kernels[1:]:
        if kernel[0] - cluster_end > gap_ns:
            clusters.append([])
        clusters[-1].append(kernel)
        cluster_end = max(cluster_end if clusters[-1][:-1] else 0, kernel[1])

    selected_clusters = list(enumerate(clusters))
    selection = "all_idle_gap_clusters"
    if not args.all_clusters:
        graph_launch_ids = {
            correlation_id
            for (correlation_id,) in connection.execute(
                "select r.correlationId from CUPTI_ACTIVITY_KIND_RUNTIME r "
                "join StringIds s on s.id = r.nameId "
                "where s.value like 'cudaGraphLaunch%'"
            )
        }
        graph_counts = collections.Counter(
            correlation_id
            for _, _, _, correlation_id, _ in kernels
            if correlation_id in graph_launch_ids
        )
        if graph_counts:
            # vLLM's B8 decode model is one full CUDA graph.  Start at that
            # graph's first kernel and retain the immediately following sampler
            # kernels.  This remains valid when a long prefill and decode have
            # no 50-us idle gap between them.
            graph_id, _ = graph_counts.most_common(1)[0]
            graph_kernels = [
                kernel for kernel in kernels if kernel[3] == graph_id
            ]
            graph_start = min(kernel[0] for kernel in graph_kernels)
            decode_kernels = []
            decode_end = graph_start
            for kernel in kernels:
                if kernel[0] < graph_start:
                    continue
                if decode_kernels and kernel[0] - decode_end > gap_ns:
                    break
                decode_kernels.append(kernel)
                decode_end = max(decode_end, kernel[1])
            selected_clusters = [(-1, decode_kernels)]
            selection = f"decode_cuda_graph correlation={graph_id}"
        else:
            # Older traces may not expose graph correlation.  Their prefill is
            # split into short layer clusters while decode has the most nodes.
            selected_clusters = [
                max(selected_clusters, key=lambda item: len(item[1]))
            ]
            selection = "largest_idle_gap_cluster"

    print(f"NSYS_KERNEL_CLUSTERS selection={selection}")
    for cluster_index, cluster in selected_clusters:
        first_start = min(kernel[0] for kernel in cluster)
        last_end = max(kernel[1] for kernel in cluster)
        durations = collections.Counter()
        counts = collections.Counter()
        launches = {}
        for start, end, name, _, launch in cluster:
            key = short_name(name)
            durations[key] += end - start
            counts[key] += 1
            launches.setdefault(key, launch)
        total_kernel_ns = sum(durations.values())
        span_ns = last_end - first_start
        overlap_ns = max(0, total_kernel_ns - span_ns)
        print(
            f"cluster={cluster_index} kernels={len(cluster)} "
            f"span_us={span_ns / 1e3:.3f} "
            f"sum_kernel_us={total_kernel_ns / 1e3:.3f} "
            f"overlap_us={overlap_ns / 1e3:.3f} "
            f"overlap_percent={100.0 * overlap_ns / total_kernel_ns:.3f}"
        )
        for name, duration_ns in durations.most_common(args.top):
            grid = launches[name][:3]
            block = launches[name][3:]
            print(
                f"  total_us={duration_ns / 1e3:.3f} calls={counts[name]} "
                f"grid={grid} block={block} name={name}"
            )


if __name__ == "__main__":
    main()
