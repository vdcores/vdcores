import argparse
from statistics import mean

import torch

from dae.launcher import *
from dae.schedule import ListSchedule


def build_single_gpu_load_schedule(src: torch.Tensor, num_loads: int, load_bytes: int):
    def repeat_load(sm: int):
        return RepeatM.on(
            num_loads,
            [TmaLoad1D(src[sm, 0, ...], bytes=load_bytes), load_bytes],
        )

    return ListSchedule([repeat_load])


def build_per_sm_copy_schedule(src: torch.Tensor, dst: torch.Tensor, num_loads: int, load_bytes: int):
    def repeat_copy(sm: int):
        return RepeatM.on(
            num_loads,
            [TmaLoad1D(src[sm, 0, ...], bytes=load_bytes), load_bytes],
            [TmaStore1D(dst[sm, 0, ...], bytes=load_bytes), load_bytes],
        )

    return ListSchedule([Copy(num_loads, load_bytes), repeat_copy])


def build_per_sm_read_schedule(src: torch.Tensor, num_loads: int, load_bytes: int):
    def repeat_read(sm: int):
        return RepeatM.on(
            num_loads,
            [TmaLoad1D(src[sm, 0, ...], bytes=load_bytes), load_bytes],
        )

    return ListSchedule([repeat_read])


def bandwidth_gib_per_s(total_bytes: int, duration_s: float) -> float:
    return total_bytes / duration_s / (1024 ** 3)


def bench_single_gpu_tma1d(device: torch.device, sms: int, num_loads: int, load_bytes: int, warmup: int, iterations: int):
    src = torch.rand(sms, num_loads, load_bytes // 4, dtype=torch.float32, device=device)
    dae = Launcher(sms, device=device)
    dae.i(build_single_gpu_load_schedule(src, num_loads, load_bytes).place(sms))
    dae.i(TerminateM(), Dummy(num_loads), TerminateC())

    stream = torch.cuda.current_stream(device=device)
    for _ in range(warmup):
        dae.launch()
    torch.cuda.synchronize(device)

    event_ms = []
    profile_ns = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        dae.launch()
        end.record(stream)
        end.synchronize()
        event_ms.append(start.elapsed_time(end))

        prof = dae.profile[:sms].cpu().numpy()
        profile_ns.append(prof[:, 1].max() - prof[:, 0].min())

    total_bytes = sms * num_loads * load_bytes
    print(
        f"[single-gpu] sms={sms} load_bytes={load_bytes} num_loads={num_loads} "
        f"event={bandwidth_gib_per_s(total_bytes, mean(event_ms) / 1e3):.2f} GiB/s "
        f"profile={bandwidth_gib_per_s(total_bytes, mean(profile_ns) / 1e9):.2f} GiB/s "
        f"(event_ms={mean(event_ms):.3f}, profile_ms={mean(profile_ns)/1e6:.3f})"
    )


def bench_bidirectional_dae(
    gpu_ids: list[int],
    sms_per_gpu: int,
    num_loads: int,
    load_bytes: int,
    warmup: int,
    iterations: int,
):
    device0 = torch.device(f"cuda:{gpu_ids[0]}")
    device1 = torch.device(f"cuda:{gpu_ids[1]}")

    src_0_to_1 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device0)
    dst_0_to_1 = torch.zeros_like(src_0_to_1, device=device1)
    src_1_to_0 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device1)
    dst_1_to_0 = torch.zeros_like(src_1_to_0, device=device0)

    dae = Launcher(num_sms=0, gpu_ids=gpu_ids)
    dae.i(
        build_per_sm_copy_schedule(src_1_to_0, dst_1_to_0, num_loads, load_bytes).place(sms_per_gpu, gpu=0),
        build_per_sm_copy_schedule(src_0_to_1, dst_0_to_1, num_loads, load_bytes).place(sms_per_gpu, gpu=1),
        TerminateM(),
        TerminateC(),
    )

    stream0 = torch.cuda.current_stream(device=device0)
    stream1 = torch.cuda.current_stream(device=device1)

    for _ in range(warmup):
        dae.launch()
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

    event_ms = []
    profile_ns = []
    for _ in range(iterations):
        start0 = torch.cuda.Event(enable_timing=True)
        end0 = torch.cuda.Event(enable_timing=True)
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)

        start0.record(stream0)
        start1.record(stream1)
        dae.launch()
        end0.record(stream0)
        end1.record(stream1)

        end0.synchronize()
        end1.synchronize()
        event_ms.append(max(start0.elapsed_time(end0), start1.elapsed_time(end1)))

        prof0 = dae.contexts[0].profile[:sms_per_gpu].cpu().numpy()
        prof1 = dae.contexts[1].profile[:sms_per_gpu].cpu().numpy()
        profile_ns.append(max(prof0[:, 1].max() - prof0[:, 0].min(), prof1[:, 1].max() - prof1[:, 0].min()))

    assert torch.equal(dst_0_to_1.cpu(), src_0_to_1.cpu())
    assert torch.equal(dst_1_to_0.cpu(), src_1_to_0.cpu())

    bytes_per_direction = sms_per_gpu * num_loads * load_bytes
    print(
        f"[peer-dae] sms={sms_per_gpu} load_bytes={load_bytes} num_loads={num_loads} "
        f"event={bandwidth_gib_per_s(2 * bytes_per_direction, mean(event_ms) / 1e3):.2f} GiB/s "
        f"profile={bandwidth_gib_per_s(2 * bytes_per_direction, mean(profile_ns) / 1e9):.2f} GiB/s "
        f"(event_ms={mean(event_ms):.3f}, profile_ms={mean(profile_ns)/1e6:.3f})"
    )


def bench_bidirectional_dae_read(
    gpu_ids: list[int],
    sms_per_gpu: int,
    num_loads: int,
    load_bytes: int,
    warmup: int,
    iterations: int,
):
    device0 = torch.device(f"cuda:{gpu_ids[0]}")
    device1 = torch.device(f"cuda:{gpu_ids[1]}")

    src_0_to_1 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device0)
    src_1_to_0 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device1)

    dae = Launcher(num_sms=0, gpu_ids=gpu_ids)
    dae.i(
        build_per_sm_read_schedule(src_1_to_0, num_loads, load_bytes).place(sms_per_gpu, gpu=0),
        build_per_sm_read_schedule(src_0_to_1, num_loads, load_bytes).place(sms_per_gpu, gpu=1),
        TerminateM(),
        Dummy(num_loads),
        TerminateC(),
    )

    stream0 = torch.cuda.current_stream(device=device0)
    stream1 = torch.cuda.current_stream(device=device1)

    for _ in range(warmup):
        dae.launch()
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

    event_ms = []
    profile_ns = []
    for _ in range(iterations):
        start0 = torch.cuda.Event(enable_timing=True)
        end0 = torch.cuda.Event(enable_timing=True)
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)

        start0.record(stream0)
        start1.record(stream1)
        dae.launch()
        end0.record(stream0)
        end1.record(stream1)

        end0.synchronize()
        end1.synchronize()
        event_ms.append(max(start0.elapsed_time(end0), start1.elapsed_time(end1)))

        prof0 = dae.contexts[0].profile[:sms_per_gpu].cpu().numpy()
        prof1 = dae.contexts[1].profile[:sms_per_gpu].cpu().numpy()
        profile_ns.append(max(prof0[:, 1].max() - prof0[:, 0].min(), prof1[:, 1].max() - prof1[:, 0].min()))

    bytes_per_direction = sms_per_gpu * num_loads * load_bytes
    print(
        f"[peer-dae-read] sms={sms_per_gpu} load_bytes={load_bytes} num_loads={num_loads} "
        f"event={bandwidth_gib_per_s(2 * bytes_per_direction, mean(event_ms) / 1e3):.2f} GiB/s "
        f"profile={bandwidth_gib_per_s(2 * bytes_per_direction, mean(profile_ns) / 1e9):.2f} GiB/s "
        f"(event_ms={mean(event_ms):.3f}, profile_ms={mean(profile_ns)/1e6:.3f})"
    )


def bench_bidirectional_dae_write(
    gpu_ids: list[int],
    sms_per_gpu: int,
    num_loads: int,
    load_bytes: int,
    warmup: int,
    iterations: int,
):
    device0 = torch.device(f"cuda:{gpu_ids[0]}")
    device1 = torch.device(f"cuda:{gpu_ids[1]}")

    src_0_to_1 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device0)
    dst_0_to_1 = torch.zeros_like(src_0_to_1, device=device1)
    src_1_to_0 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device1)
    dst_1_to_0 = torch.zeros_like(src_1_to_0, device=device0)

    dae = Launcher(num_sms=0, gpu_ids=gpu_ids)
    dae.i(
        build_per_sm_copy_schedule(src_1_to_0, dst_1_to_0, num_loads, load_bytes).place(sms_per_gpu, gpu=0),
        build_per_sm_copy_schedule(src_0_to_1, dst_0_to_1, num_loads, load_bytes).place(sms_per_gpu, gpu=1),
        TerminateM(),
        TerminateC(),
    )

    stream0 = torch.cuda.current_stream(device=device0)
    stream1 = torch.cuda.current_stream(device=device1)

    for _ in range(warmup):
        dae.launch()
    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)

    event_ms = []
    profile_ns = []
    for _ in range(iterations):
        start0 = torch.cuda.Event(enable_timing=True)
        end0 = torch.cuda.Event(enable_timing=True)
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)

        start0.record(stream0)
        start1.record(stream1)
        dae.launch()
        end0.record(stream0)
        end1.record(stream1)

        end0.synchronize()
        end1.synchronize()
        event_ms.append(max(start0.elapsed_time(end0), start1.elapsed_time(end1)))

        prof0 = dae.contexts[0].profile[:sms_per_gpu].cpu().numpy()
        prof1 = dae.contexts[1].profile[:sms_per_gpu].cpu().numpy()
        profile_ns.append(max(prof0[:, 1].max() - prof0[:, 0].min(), prof1[:, 1].max() - prof1[:, 0].min()))

    bytes_per_direction = sms_per_gpu * num_loads * load_bytes
    print(
        f"[peer-dae-write] sms={sms_per_gpu} load_bytes={load_bytes} num_loads={num_loads} "
        f"event={bandwidth_gib_per_s(2 * bytes_per_direction, mean(event_ms) / 1e3):.2f} GiB/s "
        f"profile={bandwidth_gib_per_s(2 * bytes_per_direction, mean(profile_ns) / 1e9):.2f} GiB/s "
        f"(event_ms={mean(event_ms):.3f}, profile_ms={mean(profile_ns)/1e6:.3f})"
    )


def bench_bidirectional_torch(
    gpu_ids: list[int],
    sms_per_gpu: int,
    num_loads: int,
    load_bytes: int,
    warmup: int,
    iterations: int,
):
    device0 = torch.device(f"cuda:{gpu_ids[0]}")
    device1 = torch.device(f"cuda:{gpu_ids[1]}")

    src_0_to_1 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device0)
    dst_0_to_1 = torch.zeros_like(src_0_to_1, device=device1)
    src_1_to_0 = torch.rand(sms_per_gpu, num_loads, load_bytes // 4, dtype=torch.float32, device=device1)
    dst_1_to_0 = torch.zeros_like(src_1_to_0, device=device0)

    stream0 = torch.cuda.Stream(device=device0)
    stream1 = torch.cuda.Stream(device=device1)

    for _ in range(warmup):
        with torch.cuda.stream(stream0):
            dst_1_to_0.copy_(src_1_to_0, non_blocking=True)
        with torch.cuda.stream(stream1):
            dst_0_to_1.copy_(src_0_to_1, non_blocking=True)
    stream0.synchronize()
    stream1.synchronize()

    event_ms = []
    for _ in range(iterations):
        start0 = torch.cuda.Event(enable_timing=True)
        end0 = torch.cuda.Event(enable_timing=True)
        start1 = torch.cuda.Event(enable_timing=True)
        end1 = torch.cuda.Event(enable_timing=True)

        start0.record(stream0)
        start1.record(stream1)
        with torch.cuda.stream(stream0):
            dst_1_to_0.copy_(src_1_to_0, non_blocking=True)
        with torch.cuda.stream(stream1):
            dst_0_to_1.copy_(src_0_to_1, non_blocking=True)
        end0.record(stream0)
        end1.record(stream1)

        end0.synchronize()
        end1.synchronize()
        event_ms.append(max(start0.elapsed_time(end0), start1.elapsed_time(end1)))

    assert torch.equal(dst_0_to_1.cpu(), src_0_to_1.cpu())
    assert torch.equal(dst_1_to_0.cpu(), src_1_to_0.cpu())

    bytes_per_direction = sms_per_gpu * num_loads * load_bytes
    print(
        f"[peer-torch] sms={sms_per_gpu} load_bytes={load_bytes} num_loads={num_loads} "
        f"event={bandwidth_gib_per_s(2 * bytes_per_direction, mean(event_ms) / 1e3):.2f} GiB/s "
        f"(event_ms={mean(event_ms):.3f})"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark TMA copy paths with internal DAE timing")
    parser.add_argument("--gpu-ids", type=int, nargs="+", default=[0, 1], help="Physical GPU ids for multi-GPU tests")
    parser.add_argument("--sms", type=int, default=132, help="SMs for single-GPU tests")
    parser.add_argument("--sms-per-gpu", type=int, default=132, help="SMs per GPU for peer tests")
    parser.add_argument("--num-loads", type=int, default=1000, help="Sequential length / number of repeated TMA ops")
    parser.add_argument("--load-bytes", type=int, default=16384, help="Bytes per TMA1D operation")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--mode",
        choices=["single", "peer-dae", "peer-dae-read", "peer-dae-write", "peer-torch", "compare", "sweep"],
        default="compare",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.gpu_ids) < 2:
        raise ValueError("Need two gpu ids for peer benchmarks")

    if args.mode == "single":
        bench_single_gpu_tma1d(torch.device(f"cuda:{args.gpu_ids[0]}"), args.sms, args.num_loads, args.load_bytes, args.warmup, args.iters)
        return
    if args.mode == "peer-dae":
        bench_bidirectional_dae(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        return
    if args.mode == "peer-dae-read":
        bench_bidirectional_dae_read(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        return
    if args.mode == "peer-dae-write":
        bench_bidirectional_dae_write(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        return
    if args.mode == "peer-torch":
        bench_bidirectional_torch(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        return
    if args.mode == "compare":
        bench_single_gpu_tma1d(torch.device(f"cuda:{args.gpu_ids[0]}"), args.sms, args.num_loads, args.load_bytes, args.warmup, args.iters)
        bench_bidirectional_dae(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        bench_bidirectional_dae_read(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        bench_bidirectional_dae_write(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        bench_bidirectional_torch(args.gpu_ids[:2], args.sms_per_gpu, args.num_loads, args.load_bytes, args.warmup, args.iters)
        return

    for sms_per_gpu in [32, 64, 96, 132]:
        for num_loads in [512, 1000, 2048]:
            for load_bytes in [4096, 8192, 16384]:
                bench_bidirectional_dae(args.gpu_ids[:2], sms_per_gpu, num_loads, load_bytes, args.warmup, args.iters)


if __name__ == "__main__":
    main()
