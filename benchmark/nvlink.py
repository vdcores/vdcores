"""Peer-memory roundtrip bandwidth: raw CUDA versus VDCores."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from dae import runtime
from dae.instructions import Dummy, RepeatM, TerminateC, TerminateM, TmaLoad1D
from dae.launcher import Launcher


def _schedule(peer, sms):
    tile = 8192
    count, extra = divmod(peer.numel() // tile, sms)
    dae = Launcher(sms, device="cuda:0", num_gpus=2)

    def work(sm):
        n = count + (sm < extra)
        offset = (sm * count + min(sm, extra)) * tile
        if not n:
            return [TerminateC(), TerminateM()]
        return [Dummy(n, writeback=True),
                *RepeatM.on(n, [TmaLoad1D(peer[offset:offset + tile]).writeback(), tile]),
                TerminateC(), TerminateM()]

    dae.i(work)
    dae.build_instructions()
    cinsts = dae.cinsts.to(dae.device).view(-1, 8)
    minsts = dae.minsts.to(dae.device).view(-1, 16)
    tma = torch.empty((4, 128), dtype=torch.uint8, device=dae.device)
    profile = dae.profile.view(torch.uint8).view(-1, 8)
    return lambda: runtime.launch_dae(
        sms, dae.smem_size, cinsts, minsts, tma, dae.bars, profile,
        torch.cuda.current_stream().cuda_stream, False)


def _seconds(operation, iterations):
    for _ in range(3):
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(8):
            operation()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iterations):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / 8 / 1000)
    return min(samples)


def run(args):
    assert args.num_gpus == 2, "NVLink benchmark requires --num-gpus 2"
    raw = load(name="vdcores_benchmark_nvlink",
               sources=[str(Path(__file__).with_name("cuda") / "nvlink.cu")],
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
    print("GPU 0 loads from GPU 1 and stores back; GB/s counts both transfers.")
    print(f"{'MiB':>6} {'CUDA GB/s':>12} {'VDCores GB/s':>14}")
    with torch.cuda.device(0):
        for mib in args.sizes_mib:
            peer = torch.randint(256, (mib * 1024**2,), dtype=torch.uint8, device="cuda:1")
            reference = peer.clone()
            vdcores = _schedule(peer, args.num_sms)
            torch.cuda.synchronize(1)
            cuda = lambda: raw.roundtrip_kernel(
                peer.data_ptr(), peer.data_ptr(), peer.numel(), args.num_sms * 4,
                torch.cuda.current_stream().cuda_stream)
            bandwidth = []
            for operation in (cuda, vdcores):
                seconds = _seconds(operation, args.bench)
                assert torch.equal(peer, reference)
                bandwidth.append(2 * peer.numel() / seconds / 1e9)
            print(f"{mib:6} {bandwidth[0]:12.2f} {bandwidth[1]:14.2f}", flush=True)
