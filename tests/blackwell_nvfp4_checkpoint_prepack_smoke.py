#!/usr/bin/env python3
"""Exact CUDA/resident prepack comparison against the queued VDCores oracle."""

import json
import tempfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from dae import runtime
from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.deepseek_v4_checkpoint import (
    DeepSeekV4Checkpoint,
    DeepSeekV4ResidentCheckpoint,
)
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import SchedNvfp4UmmaPrepack
from dae.tma_utils import Major


def main() -> None:
    device = torch.device("cuda")
    rows, k = 256, 512
    generator = torch.Generator(device=device).manual_seed(20260811)
    weight = torch.randint(
        0,
        256,
        (rows, k // 2),
        dtype=torch.uint8,
        generator=generator,
        device=device,
    )
    exponents = torch.randint(
        -4,
        5,
        (rows, k // 16),
        generator=generator,
        device=device,
    )
    scale = torch.pow(2.0, exponents.float()).to(torch.float8_e4m3fn)

    m_tiles = rows // SchedNvfp4UmmaPrepack.TILE_M
    k_tiles = k // SchedNvfp4UmmaPrepack.TILE_K
    shape = (
        m_tiles,
        k_tiles,
        SchedNvfp4UmmaPrepack.WEIGHT_TILE_BYTES,
    )
    expected = torch.empty(shape, dtype=torch.uint8, device=device)
    actual = torch.empty_like(expected)
    scale_tiles = (
        scale.view(m_tiles, 128, k_tiles, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
    )

    launcher = Launcher(m_tiles, device=device)
    data_tma = TmaTensor(launcher, weight).wgmma_load(128, 128, Major.K)
    launcher.s(
        SchedNvfp4UmmaPrepack(
            SchedNvfp4UmmaPrepack.WEIGHT,
            weight,
            scale_tiles,
            expected,
            data_tma,
        ).place(m_tiles)
    )
    launcher.launch()
    runtime.prepack_nvfp4_checkpoint(weight, scale, actual)
    torch.cuda.synchronize(device)

    if not torch.equal(actual, expected):
        mismatches = int(torch.count_nonzero(actual != expected))
        raise AssertionError(f"native checkpoint prepack mismatches={mismatches}")

    prefix = "layers.0.ffn.experts.0.w1"
    host_tensors = {
        f"{prefix}.weight": weight.cpu(),
        f"{prefix}.weight_scale": scale.cpu(),
        f"{prefix}.weight_scale_2": torch.tensor(0.25),
        f"{prefix}.input_scale": torch.tensor(0.5),
    }
    with tempfile.TemporaryDirectory(prefix="dsv4-native-prepack-") as temp:
        root = Path(temp)
        filename = "model-00001-of-00001.safetensors"
        save_file(host_tensors, root / filename)
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": sum(
                        tensor.numel() * tensor.element_size()
                        for tensor in host_tensors.values()
                    )},
                    "weight_map": {name: filename for name in host_tensors},
                }
            )
        )
        resident = DeepSeekV4ResidentCheckpoint.from_checkpoint(
            DeepSeekV4Checkpoint(root, DeepSeekV4FlashConfig()),
            device=device,
            names=host_tensors,
            native_nvfp4=True,
        )
        native = resident.load_native_nvfp4_linear(prefix, device=device)
        if not torch.equal(native.weight_tiles, expected):
            mismatches = int(torch.count_nonzero(native.weight_tiles != expected))
            raise AssertionError(f"resident native prepack mismatches={mismatches}")
        torch.testing.assert_close(native.alpha, torch.tensor([0.125], device=device))
    print(
        "BLACKWELL_NVFP4_CHECKPOINT_PREPACK status=PASS "
        f"shape={rows}x{k} bytes={actual.numel()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
