"""Sustained pairwise NVLink payload bandwidth, including simultaneous duplex."""

from pathlib import Path
import statistics
import subprocess


def _topology(torch):
    """Resolve CUDA-visible UUIDs instead of assuming physical GPU numbering."""
    try:
        inventory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True,
        )
        topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("nvidia-smi is required to verify the NVLink topology") from error
    physical = {uuid.strip().lower().removeprefix("gpu-"): f"GPU{index.strip()}"
                for index, uuid in (line.split(",") for line in inventory.splitlines())}
    try:
        devices = [physical[str(torch.cuda.get_device_properties(i).uuid).lower().removeprefix("gpu-")]
                   for i in range(2)]
        lines = [line.split() for line in topology.splitlines() if line.strip()]
        header = lines[0]
        rows = {line[0]: line[1:] for line in lines[1:] if line[0].startswith("GPU")}
        link = rows[devices[0]][header.index(devices[1])]
    except (KeyError, ValueError, IndexError, AttributeError) as error:
        raise RuntimeError("could not map allocated CUDA GPUs to the NVLink topology") from error
    if not link.startswith("NV") or not link[2:].isdigit():
        raise RuntimeError(f"allocated GPUs use {link}, not NVLink; refusing a PCIe bandwidth result")
    return {"physical_devices": devices, "link": link}


def _schedule(torch, source, target, sms, tile):
    from dae import runtime
    from dae.instructions import Copy, RepeatM, TerminateC, TerminateM, TmaLoad1D, TmaStore1D
    from dae.launcher import Launcher

    tiles = source.numel() // tile
    quotient, remainder = divmod(tiles, sms)
    if quotient + bool(remainder) >= 2**16:
        raise ValueError("payload exceeds the 16-bit repeat count; use a smaller size")
    dae = Launcher(sms, device=target.device, num_gpus=2)

    def work(sm):
        count = quotient + (sm < remainder)
        if not count:
            return [TerminateC(), TerminateM()]
        offset = (sm * quotient + min(sm, remainder)) * tile
        return [
            Copy(count, tile),
            *RepeatM.on(count,
                        [TmaLoad1D(source[offset:offset + tile]), tile],
                        [TmaStore1D(target[offset:offset + tile]), tile]),
            TerminateC(), TerminateM(),
        ]

    dae.i(work)
    dae.build_instructions()
    # Benchmark-only upload: no new runtime compilation or graph API is needed.
    cinsts = dae.cinsts.to(dae.device).view(-1, 8)
    minsts = dae.minsts.to(dae.device).view(-1, 16)
    tma = torch.empty((4, 128), dtype=torch.uint8, device=dae.device)
    profile = dae.profile.view(torch.uint8).view(-1, 8)

    def launch():
        runtime.launch_dae(sms, dae.smem_size, cinsts, minsts, tma, dae.bars, profile,
                           torch.cuda.current_stream(dae.device).cuda_stream, False)

    return launch


def _measure(torch, transfers, warmup, iterations):
    """Graph batches remove Python dispatch from timings; events share a GPU clock.

    A control stream releases both GPUs with one event, then waits on completion
    events from both. Only same-device start/end events are used for elapsed time.
    """
    batch = 8
    graphs, streams = [], []
    for device, operation in transfers:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(stream):
                for _ in range(warmup):
                    operation()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(batch):
                    operation()
            graphs.append(graph)
            streams.append(stream)

    control = torch.cuda.Stream(device=transfers[0][0])
    with torch.cuda.device(control.device):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        done = [torch.cuda.Event() for _ in transfers]
        # Warm graph executables before sampling.
        for graph, stream in zip(graphs, streams):
            with torch.cuda.stream(stream):
                graph.replay()
            stream.synchronize()
        samples = []
        for _ in range(iterations):
            start.record(control)
            for graph, stream, completed in zip(graphs, streams, done):
                with torch.cuda.stream(stream):
                    stream.wait_event(start)
                    graph.replay()
                    completed.record(stream)
                control.wait_event(completed)
            end.record(control)
            end.synchronize()
            samples.append(start.elapsed_time(end) / batch)
    return samples


def run(args):
    if args.num_gpus != 2:
        raise ValueError("this pairwise suite requires explicit --num-gpus 2")
    import torch
    from torch.utils.cpp_extension import load
    from dae import runtime

    if torch.cuda.device_count() < 2:
        raise ValueError("two visible GPUs are required; allocate with na run -n=2")
    properties = [torch.cuda.get_device_properties(i) for i in range(2)]
    if any((prop.major, prop.minor) != (9, 0) for prop in properties):
        raise ValueError("the current VDCores build requires Hopper (sm_90a) GPUs")
    sms = args.num_sms or min(prop.multi_processor_count for prop in properties)
    if sms > min(prop.multi_processor_count for prop in properties):
        raise ValueError("--num-sms exceeds the allocated GPU's SM count")
    topology = _topology(torch)
    missing = {"OP_COPY", "OP_TERMINATEC"} - set(runtime.supported_compute_ops)
    if missing:
        raise ValueError(f"missing compute ops {sorted(missing)}; rebuild with make pyext DAE_COMPUTE_OPS=OP_COPY,OP_TERMINATEC")
    runtime.enable_peer_access(2)
    # Compile outside all timed regions. C++ baseline has no Torch tensor math.
    raw = load(name="vdcores_benchmark_nvlink", sources=[str(Path(__file__).resolve().parents[1] / "cuda/nvlink.cu")],
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    print(f"NVLink {topology['link']}: {properties[0].name} <-> {properties[1].name}; {sms} VDCores SMs", flush=True)
    print("Payload GB/s (decimal); duplex sums both directions. Graph batch=8; correctness checked outside timing.")
    print(f"{'MiB/dir':>8} {'direction':>12} {'implementation':>20} {'tile KiB':>8} {'median GB/s':>12} {'peak GB/s':>10}")
    results = []
    for mib in args.sizes_mib:
        size = mib * 1024**2
        source = [torch.randint(0, 256, (size,), device=i, dtype=torch.uint8) for i in range(2)]
        target = [torch.empty_like(source[i]) for i in range(2)]
        for device in range(2):
            torch.cuda.synchronize(device)
        for direction, destinations in (("0->1", [1]), ("1->0", [0]), ("duplex", [0, 1])):
            variants = [("cuda_kernel", None), ("cuda_memcpy_async", None)] + [("vdcores", tile) for tile in (8192, 16384, 32768)]
            for implementation, tile in variants:
                transfers = []
                for dst in destinations:
                    src = 1 - dst
                    target[dst].fill_(0xA5)
                    if implementation == "vdcores":
                        operation = _schedule(torch, source[src], target[dst], sms, tile)
                    elif implementation == "cuda_kernel":
                        operation = lambda dst=dst, src=src: raw.copy_kernel(
                            target[dst].data_ptr(), source[src].data_ptr(), size,
                            properties[dst].multi_processor_count * 4, torch.cuda.current_stream(dst).cuda_stream)
                    else:
                        operation = lambda dst=dst, src=src: raw.copy_engine(
                            target[dst].data_ptr(), source[src].data_ptr(),
                            size, torch.cuda.current_stream(dst).cuda_stream)
                    transfers.append((dst, operation))
                for device in range(2):
                    torch.cuda.synchronize(device)
                samples = _measure(torch, transfers, args.warmup, args.bench)
                for dst in destinations:
                    # Compare on the source GPU; this checking copy is untimed.
                    if not torch.equal(target[dst].to(source[1 - dst].device), source[1 - dst]):
                        raise RuntimeError(f"incorrect {implementation} transfer ({direction}, {mib} MiB)")
                payload = size * len(destinations)  # Count only bytes crossing NVLink.
                median_ms, best_ms = statistics.median(samples), min(samples)
                row = {"size_bytes_per_direction": size, "direction": direction,
                       "implementation": implementation, "tile_bytes": tile,
                       "median_ms": median_ms, "min_ms": best_ms,
                       "median_gbps": payload / median_ms / 1e6,
                       "peak_gbps": payload / best_ms / 1e6, "samples_ms": samples}
                results.append(row)
                tile_label = str(tile // 1024) if tile else "-"
                print(f"{mib:8} {direction:>12} {implementation:>20} {tile_label:>8} {row['median_gbps']:12.2f} {row['peak_gbps']:10.2f}", flush=True)
        del source, target, transfers
    peaks = []
    for direction in ("0->1", "1->0", "duplex"):
        for implementation in ("cuda_kernel", "cuda_memcpy_async", "vdcores"):
            best = max((row for row in results if row["direction"] == direction and row["implementation"] == implementation), key=lambda row: row["peak_gbps"])
            peaks.append(best)
            print(f"Peak measured {direction} {implementation}: {best['peak_gbps']:.2f} GB/s ({best['size_bytes_per_direction'] // 1024**2} MiB/direction, tile={best['tile_bytes']})")
    return {"devices": [{"name": prop.name, "uuid": str(prop.uuid)} for prop in properties],
            "topology": topology, "num_gpus": 2, "num_sms": sms, "iterations": args.bench,
            "warmup": args.warmup, "graph_batch": 8, "bandwidth_unit": "GB/s (10^9 bytes/s)",
            "results": results, "peaks": peaks}
