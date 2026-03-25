import argparse

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

    return ListSchedule([repeat_copy])


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


def main():
    parser = argparse.ArgumentParser(description="Verify cross-GPU TMA copy over NVLink")
    parser.add_argument("--gpu-ids", type=int, nargs="+", required=True, help="Physical GPU ids")
    parser.add_argument("--num-loads", type=int, default=1024)
    parser.add_argument("--load-bytes", type=int, default=1024 * 8)
    args = parser.parse_args()

    if len(args.gpu_ids) < 2:
        raise ValueError("Need at least two physical GPU ids for multi-GPU copy verification")

    gpu_ids = args.gpu_ids[:2]
    print(f"[config] physical gpu ids={gpu_ids}")

    run_explicit_virtual_gpu_copy(gpu_ids, src_virtual_gpu=0, dst_virtual_gpu=1, num_loads=args.num_loads, load_bytes=args.load_bytes)
    run_explicit_virtual_gpu_copy(gpu_ids, src_virtual_gpu=1, dst_virtual_gpu=0, num_loads=args.num_loads, load_bytes=args.load_bytes)
    run_flattened_copy(gpu_ids, src_virtual_gpu=0, dst_virtual_gpu=1, num_loads=args.num_loads, load_bytes=args.load_bytes)
    run_flattened_copy(gpu_ids[::-1], src_virtual_gpu=0, dst_virtual_gpu=1, num_loads=args.num_loads, load_bytes=args.load_bytes)


if __name__ == "__main__":
    main()
