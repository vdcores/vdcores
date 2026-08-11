#!/usr/bin/env python3
"""Exact native FP8 converter and resident replacement smoke."""

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
from dae.deepseek_v4_quant import quantize_fp8_block128
from dae.instructions import TmaTensor
from dae.launcher import Launcher
from dae.schedule import SchedFp8UmmaPrepack
from dae.tma_utils import Major


def main() -> None:
    device = torch.device("cuda")
    rows, k = 256, 256
    generator = torch.Generator(device=device).manual_seed(20260811)
    source = torch.randn(
        (rows, k), generator=generator, dtype=torch.bfloat16, device=device
    )
    weight, scale = quantize_fp8_block128(source)
    m_tiles = rows // SchedFp8UmmaPrepack.TILE_M
    k_tiles = k // SchedFp8UmmaPrepack.TILE_K
    shape = (
        m_tiles,
        k_tiles,
        SchedFp8UmmaPrepack.WEIGHT_TILE_BYTES,
    )
    expected = torch.empty(shape, dtype=torch.uint8, device=device)
    actual = torch.empty_like(expected)

    launcher = Launcher(m_tiles, device=device)
    data_tma = TmaTensor(
        launcher, weight.view(torch.uint8)
    ).wgmma_load(128, 128, Major.K)
    launcher.s(
        SchedFp8UmmaPrepack(
            SchedFp8UmmaPrepack.WEIGHT,
            weight,
            scale,
            expected,
            data_tma,
        ).place(m_tiles)
    )
    launcher.launch()
    runtime.prepack_fp8_checkpoint(weight, scale, actual)
    torch.cuda.synchronize(device)
    if not torch.equal(actual, expected):
        mismatches = int(torch.count_nonzero(actual != expected))
        raise AssertionError(f"native FP8 converter mismatches={mismatches}")

    prefix = "layers.0.attn.wq_b"
    host_tensors = {
        f"{prefix}.weight": weight.cpu(),
        f"{prefix}.scale": scale.cpu(),
    }
    with tempfile.TemporaryDirectory(prefix="dsv4-fp8-prepack-") as temp:
        root = Path(temp)
        filename = "model-00001-of-00001.safetensors"
        save_file(host_tensors, root / filename)
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "total_size": sum(
                            tensor.numel() * tensor.element_size()
                            for tensor in host_tensors.values()
                        )
                    },
                    "weight_map": {
                        name: filename for name in host_tensors
                    },
                }
            )
        )
        resident = DeepSeekV4ResidentCheckpoint.from_checkpoint(
            DeepSeekV4Checkpoint(root, DeepSeekV4FlashConfig()),
            device=device,
            names=host_tensors,
            native_fp8_prefixes=(prefix,),
        )
        native = resident.load_native_fp8_linear(prefix, device=device)
        if not torch.equal(native.weight_tiles, expected):
            mismatches = int(
                torch.count_nonzero(native.weight_tiles != expected)
            )
            raise AssertionError(
                f"resident native FP8 replacement mismatches={mismatches}"
            )
        if resident.storage_bytes < expected.numel():
            raise AssertionError("resident FP8 span did not reserve native bytes")

    print(
        "BLACKWELL_FP8_CHECKPOINT_PREPACK status=PASS "
        f"shape={rows}x{k} bytes={actual.numel()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
