#!/usr/bin/env python3
"""Convert DeepSeek-V4 FFN weights into the resident MXFP4 stream image.

This is an offline-only utility.  Routed ModelOpt NVFP4 and shared block-FP8
checkpoint tensors are dequantized, requantized to group-32 MXFP4, and written
in the exact task-major TMA/SFA layout used by the persistent VDCores FFN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.deepseek_v4_checkpoint import DeepSeekV4Checkpoint
from dae.deepseek_v4_mxfp_checkpoint import (
    CHECKPOINT_EXPERTS,
    DOWN_SLICES,
    FORMAT_VERSION,
    LINEAR1_SLICES,
    STREAM_EXPERTS,
    default_mxfp_ffn_directory,
    mxfp_ffn_layer_path,
    pack_mxfp4_data,
    pack_mxfp4_scales,
)
from dae.deepseek_v4_quant import dequantize_fp8_block128, dequantize_nvfp4


def _parse_layers(value: str, count: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(count))
    selected = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = (int(part) for part in item.split("-", 1))
            selected.extend(range(first, last + 1))
        else:
            selected.append(int(item))
    result = tuple(dict.fromkeys(selected))
    if not result or any(layer < 0 or layer >= count for layer in result):
        raise ValueError(f"layers must be selected from [0,{count})")
    return result


def _source_digest(checkpoint: Path) -> str:
    payload = (checkpoint / "model.safetensors.index.json").read_bytes()
    return hashlib.sha256(payload).hexdigest()


def _quantize_and_pack(source: torch.Tensor, *, tile_k: int):
    import flashinfer
    from flashinfer import SfLayout

    source = source.to(torch.bfloat16)
    packed, scales = flashinfer.mxfp4_quantize(
        source,
        sfLayout=SfLayout.layout_128x4,
    )
    data = pack_mxfp4_data(packed, tile_k=tile_k)
    native_scales = pack_mxfp4_scales(
        scales,
        rows=source.shape[0],
        k=source.shape[1],
        tile_k=tile_k,
    )
    return data, native_scales


def _allocate_host_layer() -> dict[str, torch.Tensor]:
    return {
        "linear1_weights": torch.empty(
            (STREAM_EXPERTS * LINEAR1_SLICES, 16, 4, 128, 64),
            dtype=torch.uint8,
        ),
        "linear1_scales": torch.empty(
            (STREAM_EXPERTS * LINEAR1_SLICES, 16, 2048),
            dtype=torch.uint8,
        ),
        "down_weights": torch.empty(
            (STREAM_EXPERTS * DOWN_SLICES, 8, 2, 128, 64),
            dtype=torch.uint8,
        ),
        "down_scales": torch.empty(
            (STREAM_EXPERTS * DOWN_SLICES, 8, 1024),
            dtype=torch.uint8,
        ),
    }


def _store_linear1(
    output: dict[str, torch.Tensor],
    stream_expert: int,
    gate_source: torch.Tensor,
    up_source: torch.Tensor,
) -> None:
    gate_data, gate_scales = _quantize_and_pack(gate_source, tile_k=512)
    up_data, up_scales = _quantize_and_pack(up_source, tile_k=512)
    begin = stream_expert * LINEAR1_SLICES
    end = begin + LINEAR1_SLICES
    output["linear1_weights"][begin:end].copy_(
        torch.cat((gate_data, up_data), dim=1).cpu()
    )
    output["linear1_scales"][begin:end].copy_(
        torch.cat((gate_scales, up_scales), dim=1).cpu()
    )


def _store_down(
    output: dict[str, torch.Tensor],
    stream_expert: int,
    source: torch.Tensor,
) -> None:
    data, scales = _quantize_and_pack(source, tile_k=256)
    begin = stream_expert * DOWN_SLICES
    end = begin + DOWN_SLICES
    output["down_weights"][begin:end].copy_(data.cpu())
    output["down_scales"][begin:end].copy_(scales.cpu())


def _convert_shared(
    checkpoint: DeepSeekV4Checkpoint,
    layer_id: int,
    output: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    prefixes = tuple(
        f"layers.{layer_id}.ffn.shared_experts.{tag}"
        for tag in ("w1", "w3", "w2")
    )
    names = tuple(
        name
        for prefix in prefixes
        for name in (f"{prefix}.weight", f"{prefix}.scale")
    )
    tensors = checkpoint.load_tensors(names, device=str(device))
    sources = []
    for prefix in prefixes:
        sources.append(
            dequantize_fp8_block128(
                tensors[f"{prefix}.weight"], tensors[f"{prefix}.scale"]
            )
        )
    _store_linear1(output, 0, sources[0], sources[1])
    _store_down(output, 0, sources[2])


def _convert_routed_batch(
    checkpoint: DeepSeekV4Checkpoint,
    layer_id: int,
    experts: range,
    output: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    prefixes = tuple(
        f"layers.{layer_id}.ffn.experts.{expert}.{tag}"
        for expert in experts
        for tag in ("w1", "w3", "w2")
    )
    names = tuple(
        name
        for prefix in prefixes
        for name in (
            f"{prefix}.weight",
            f"{prefix}.weight_scale",
            f"{prefix}.weight_scale_2",
        )
    )
    tensors = checkpoint.load_tensors(names, device=str(device))

    def source(prefix: str) -> torch.Tensor:
        return dequantize_nvfp4(
            tensors[f"{prefix}.weight"],
            tensors[f"{prefix}.weight_scale"],
            tensors[f"{prefix}.weight_scale_2"],
        )

    for expert in experts:
        prefix = f"layers.{layer_id}.ffn.experts.{expert}"
        gate = source(f"{prefix}.w1")
        up = source(f"{prefix}.w3")
        down = source(f"{prefix}.w2")
        _store_linear1(output, expert + 1, gate, up)
        _store_down(output, expert + 1, down)


def _write_manifest(
    output_root: Path,
    checkpoint: Path,
    source_digest: str,
    layers: tuple[int, ...],
) -> None:
    manifest = {
        "format": "vdcores-deepseek-v4-mxfp-ffn",
        "version": FORMAT_VERSION,
        "source_index_sha256": source_digest,
        "source_checkpoint": str(checkpoint),
        "layers": list(layers),
        "stream_experts": STREAM_EXPERTS,
        "shared_stream_index": 0,
        "routed_stream_offset": 1,
        "linear1_tile_k": 512,
        "down_tile_k": 256,
    }
    temporary = output_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_root / "manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--expert-batch", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.expert_batch <= 0:
        parser.error("expert-batch must be positive")

    config = DeepSeekV4FlashConfig()
    try:
        layers = _parse_layers(args.layers, config.num_layers)
    except ValueError as error:
        parser.error(str(error))
    checkpoint_root = args.checkpoint.resolve()
    output_root = (
        default_mxfp_ffn_directory(checkpoint_root)
        if args.output is None
        else args.output.resolve()
    )
    output_root.mkdir(parents=True, exist_ok=True)
    digest = _source_digest(checkpoint_root)
    checkpoint = DeepSeekV4Checkpoint(checkpoint_root, config)
    device = torch.device("cuda")

    from safetensors.torch import save_file

    completed = []
    for layer_id in layers:
        destination = mxfp_ffn_layer_path(output_root, layer_id)
        if destination.exists() and not args.overwrite:
            print(
                "DSV4_MXFP_OFFLINE_LAYER status=SKIP "
                f"layer={layer_id} path={destination}",
                flush=True,
            )
            completed.append(layer_id)
            continue
        started = time.monotonic()
        print(
            "DSV4_MXFP_OFFLINE_LAYER status=START "
            f"layer={layer_id} path={destination}",
            flush=True,
        )
        output = _allocate_host_layer()
        _convert_shared(checkpoint, layer_id, output, device)
        for first in range(0, CHECKPOINT_EXPERTS, args.expert_batch):
            last = min(first + args.expert_batch, CHECKPOINT_EXPERTS)
            _convert_routed_batch(
                checkpoint,
                layer_id,
                range(first, last),
                output,
                device,
            )
            print(
                "DSV4_MXFP_OFFLINE_BATCH status=PASS "
                f"layer={layer_id} experts={first}-{last - 1}",
                flush=True,
            )
        temporary = destination.with_suffix(".safetensors.tmp")
        save_file(
            output,
            str(temporary),
            metadata={
                "format": "vdcores-deepseek-v4-mxfp-ffn",
                "version": str(FORMAT_VERSION),
                "layer": str(layer_id),
                "source_index_sha256": digest,
            },
        )
        os.replace(temporary, destination)
        completed.append(layer_id)
        print(
            "DSV4_MXFP_OFFLINE_LAYER status=PASS "
            f"layer={layer_id} gib={destination.stat().st_size / (1 << 30):.3f} "
            f"elapsed_s={time.monotonic() - started:.3f}",
            flush=True,
        )
    _write_manifest(output_root, checkpoint_root, digest, tuple(completed))
    print(
        "DSV4_MXFP_OFFLINE status=PASS "
        f"layers={','.join(str(layer) for layer in completed)} "
        f"output={output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
