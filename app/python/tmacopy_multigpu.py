import argparse
from statistics import mean

import torch

from dae.launcher import *
from dae.schedule import ListSchedule


def build_copy_schedule(src: torch.Tensor, dst: torch.Tensor, num_loads: int, load_bytes: int):
    def repeat_copy(sm: int):
        return RepeatM.on(
            num_loads,
            [TmaLoad1D(src[0, ...], bytes=load_bytes), load_bytes],
            [TmaStore1D(dst[0, ...], bytes=load_bytes), load_bytes],
        )

    return ListSchedule([Copy(num_loads, load_bytes), repeat_copy])


def build_per_sm_copy_schedule(src: torch.Tensor, dst: torch.Tensor, num_loads: int, load_bytes: int):
    def repeat_copy(sm: int):
        return RepeatM.on(
            num_loads,
            [TmaLoad1D(src[sm, 0, ...], bytes=load_bytes), load_bytes],
            [TmaStore1D(dst[sm, 0, ...], bytes=load_bytes), load_bytes],
        )

    return ListSchedule([Copy(num_loads, load_bytes), repeat_copy])


def run_explicit_virtual_gpu_copy(gpu_ids: list[int], src_virtual_gpu: int, dst_virtual_gpu: int, num_loads: int, load_bytes: int):
    src_device = torch.device(f"cuda:{gpu_ids[src_virtual_gpu]}")
    dst_device = torch.device(f"cuda:{gpu_ids[dst_virtual_gpu]}")

    src = torch.rand(num_loads, load_bytes // 4, dtype=torch.float32, device=src_device)
    dst = torch.zeros_like(src, device=dst_device)

    dae = Launcher(num_sms=1, gpu_ids=gpu_ids)
    dae.i(
        build_copy_schedule(src, dst, num_loads, load_bytes).place(1, gpu=dst_virtual_gpu),
        TerminateM(),
        TerminateC(),
    )
    dae.launch()

    torch.cuda.synchronize(src_device)
    torch.cuda.synchronize(dst_device)
    assert torch.equal(dst.cpu(), src.cpu()), (
        f"Explicit virtual-gpu copy failed: src gpu={src_virtual_gpu}, dst gpu={dst_virtual_gpu}"
    )
    print(f"[ok] explicit gpu copy src=v{src_virtual_gpu}->dst=v{dst_virtual_gpu}")


def run_flattened_copy(gpu_ids: list[int], src_virtual_gpu: int, dst_virtual_gpu: int, num_loads: int, load_bytes: int):
    if dst_virtual_gpu != 1:
        raise ValueError("Flattened placement check expects the destination virtual gpu to be index 1")

    src_device = torch.device(f"cuda:{gpu_ids[src_virtual_gpu]}")
    dst_device = torch.device(f"cuda:{gpu_ids[dst_virtual_gpu]}")

    src = torch.rand(num_loads, load_bytes // 4, dtype=torch.float32, device=src_device)
    dst = torch.zeros_like(src, device=dst_device)

    first_gpu_sms = torch.cuda.get_device_properties(gpu_ids[0]).multi_processor_count
    total_sms = first_gpu_sms + 1

    dae = Launcher(num_sms=total_sms, gpu_ids=gpu_ids)
    dae.i(
        build_copy_schedule(src, dst, num_loads, load_bytes).place(1, base_sm=first_gpu_sms),
        TerminateM(),
        TerminateC(),
    )
    dae.launch()

    torch.cuda.synchronize(src_device)
    torch.cuda.synchronize(dst_device)
    assert torch.equal(dst.cpu(), src.cpu()), (
        f"Flattened global-sm copy failed: src gpu={src_virtual_gpu}, dst gpu={dst_virtual_gpu}"
    )
    print(
        f"[ok] flattened global-sm copy src=v{src_virtual_gpu}->dst=v{dst_virtual_gpu} "
        f"(base_sm={first_gpu_sms})"
    )


def run_bidirectional_peer_bandwidth(
    gpu_ids: list[int],
    num_loads: int,
    load_bytes: int,
    iterations: int,
    warmup: int,
    sms_per_gpu: int,
):
    device0 = torch.device(f"cuda:{gpu_ids[0]}")
    device1 = torch.device(f"cuda:{gpu_ids[1]}")
    sms_per_gpu = min(
        sms_per_gpu,
        torch.cuda.get_device_properties(device0).multi_processor_count,
        torch.cuda.get_device_properties(device1).multi_processor_count,
    )

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

    elapsed_ms_0_to_1 = []
    elapsed_ms_1_to_0 = []
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
        elapsed_ms_1_to_0.append(start0.elapsed_time(end0))
        elapsed_ms_0_to_1.append(start1.elapsed_time(end1))

    torch.cuda.synchronize(device0)
    torch.cuda.synchronize(device1)
    assert torch.equal(dst_0_to_1.cpu(), src_0_to_1.cpu()), "Bidirectional copy failed for v0->v1"
    assert torch.equal(dst_1_to_0.cpu(), src_1_to_0.cpu()), "Bidirectional copy failed for v1->v0"

    bytes_per_direction = sms_per_gpu * num_loads * load_bytes
    avg_ms_0_to_1 = mean(elapsed_ms_0_to_1)
    avg_ms_1_to_0 = mean(elapsed_ms_1_to_0)
    bidirectional_ms = max(avg_ms_0_to_1, avg_ms_1_to_0)
    gib = 1024 ** 3

    bw_0_to_1 = bytes_per_direction / (avg_ms_0_to_1 / 1e3) / gib
    bw_1_to_0 = bytes_per_direction / (avg_ms_1_to_0 / 1e3) / gib
    bidirectional_bw = (2 * bytes_per_direction) / (bidirectional_ms / 1e3) / gib

    print(
        "[ok] bidirectional peer copy "
        f"v0->v1={bw_0_to_1:.2f} GiB/s, "
        f"v1->v0={bw_1_to_0:.2f} GiB/s, "
        f"aggregate={bidirectional_bw:.2f} GiB/s "
        f"(sms_per_gpu={sms_per_gpu}, iters={iterations}, warmup={warmup})"
    )


def main():
    parser = argparse.ArgumentParser(description="Verify cross-GPU TMA copy over NVLink")
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True, help="Physical GPU ids")
    parser.add_argument("--num-loads", type=int, default=1024)
    parser.add_argument("--load-bytes", type=int, default=1024 * 8)
    parser.add_argument("--bandwidth-iters", type=int, default=10, help="Measured iterations for the bidirectional bandwidth example")
    parser.add_argument("--bandwidth-warmup", type=int, default=2, help="Warmup iterations before timing the bidirectional bandwidth example")
    parser.add_argument("--bandwidth-sms-per-gpu", type=int, default=16, help="SMs per GPU for the bidirectional bandwidth example")
    args = parser.parse_args()

    if len(args.gpu_ids) < 2:
        raise ValueError("Need at least two physical GPU ids for multi-GPU copy verification")

    gpu_ids = args.gpu_ids[:2]
    print(f"[config] physical gpu ids={gpu_ids}")

    run_explicit_virtual_gpu_copy(gpu_ids, src_virtual_gpu=0, dst_virtual_gpu=1, num_loads=args.num_loads, load_bytes=args.load_bytes)
    run_explicit_virtual_gpu_copy(gpu_ids, src_virtual_gpu=1, dst_virtual_gpu=0, num_loads=args.num_loads, load_bytes=args.load_bytes)
    run_flattened_copy(gpu_ids, src_virtual_gpu=0, dst_virtual_gpu=1, num_loads=args.num_loads, load_bytes=args.load_bytes)
    run_flattened_copy(gpu_ids[::-1], src_virtual_gpu=0, dst_virtual_gpu=1, num_loads=args.num_loads, load_bytes=args.load_bytes)
    run_bidirectional_peer_bandwidth(
        gpu_ids,
        num_loads=args.num_loads,
        load_bytes=args.load_bytes,
        iterations=args.bandwidth_iters,
        warmup=args.bandwidth_warmup,
        sms_per_gpu=args.bandwidth_sms_per_gpu,
    )


if __name__ == "__main__":
    main()
