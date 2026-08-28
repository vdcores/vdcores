#!/usr/bin/env python3
"""Check retained resident-FFN boundaries against the offline MXFP image.

The production launch writes all inputs and intermediate records needed by
this checker into a host-side diagnostic dump.  This script performs no DAE
launch: it reconstructs the exact offline MXFP4 matrices and checks one FFN
boundary at a time with an independent FlashInfer dequantizer plus PyTorch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.launcher import Launcher
from dae.schedule import SchedDsv4HcPost


FP8_MAX = 448.0
LINEAR1_SLICES = 16
DOWN_SLICES = 32


def _load_layer(root: Path, layer_id: int) -> dict[str, torch.Tensor]:
    path = root / f"layer-{layer_id:03d}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: handle.get_tensor(name) for name in handle.keys()}


def _unpack_mxfp4_data(data: torch.Tensor) -> torch.Tensor:
    """Invert pack_mxfp4_data without changing any FP4 nibbles."""
    if data.ndim != 5 or tuple(data.shape[-2:]) != (128, 64):
        raise ValueError(f"unexpected resident MXFP4 data shape {data.shape}")
    return (
        data.permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(data.shape[0] * 128, -1)
    )


def _dequantize_mxfp4(
    data: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    import flashinfer
    from flashinfer import SfLayout

    packed = _unpack_mxfp4_data(data)
    return flashinfer.mxfp4_dequantize(
        packed,
        scales.contiguous().reshape(-1),
        sfLayout=SfLayout.layout_128x4,
    )


def _ue8m0_values(raw: torch.Tensor) -> torch.Tensor:
    return torch.exp2(raw.to(torch.int16).float() - 127.0)


def _decode_input_row(dump: dict[str, torch.Tensor]) -> torch.Tensor:
    data = dump["mxfp_activation_data"].reshape(8, 4, 8, 8, 16)
    scales = dump["mxfp_activation_scales"].reshape(8, 4, 512)
    values = []
    for record in range(8):
        for subtile in range(4):
            # Row zero has destination_chunk == source_chunk in the native
            # Layout_K_SW128 image, so its logical bytes are contiguous here.
            quantized = data[record, subtile, 0].reshape(4, 32)
            scale = _ue8m0_values(scales[record, subtile, :4])
            values.append(
                quantized.view(torch.float8_e4m3fn).float()
                * scale[:, None]
            )
    return torch.cat([value.reshape(-1) for value in values])


def _middle_row_payload(
    records: torch.Tensor, row: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    if tuple(records.shape) != (LINEAR1_SLICES, 1536):
        raise ValueError(f"unexpected middle-record shape {records.shape}")
    quantized_parts = []
    scale_parts = []
    for output_slice in range(LINEAR1_SLICES):
        data = records[output_slice, :1024].reshape(8, 8, 16)
        logical_chunks = torch.empty((8, 16), dtype=torch.uint8)
        for source_chunk in range(8):
            logical_chunks[source_chunk].copy_(
                data[row, source_chunk ^ row]
            )
        quantized_parts.append(logical_chunks.reshape(-1))
        scale_payload = records[output_slice, 1024:]
        scale_parts.append(
            scale_payload[row * 16 : row * 16 + 4].clone()
        )
    return torch.cat(quantized_parts), torch.cat(scale_parts)


def _decode_middle_row(records: torch.Tensor, row: int = 0) -> torch.Tensor:
    quantized, raw_scales = _middle_row_payload(records, row)
    return (
        quantized.view(torch.float8_e4m3fn).float().reshape(-1, 32)
        * _ue8m0_values(raw_scales)[:, None]
    ).reshape(-1)


def _quantize_middle_reference(
    source: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    groups = source.float().reshape(-1, 32)
    requested = (groups.abs().amax(dim=-1) / FP8_MAX).clamp_min(
        2.0**-127
    )
    exponents = torch.ceil(torch.log2(requested)).clamp(-127, 127)
    scales = torch.exp2(exponents)
    quantized = (
        (groups / scales[:, None])
        .clamp(-FP8_MAX, FP8_MAX)
        .to(torch.float8_e4m3fn)
    )
    raw_scales = (exponents.to(torch.int16) + 127).to(torch.uint8)
    return (
        quantized.view(torch.uint8).reshape(-1),
        raw_scales,
        (quantized.float() * scales[:, None]).reshape(-1),
    )


def _report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    delta = actual - expected
    relative = float(delta.norm() / expected.norm().clamp_min(1.0e-30))
    cosine = float(
        torch.nn.functional.cosine_similarity(actual, expected, dim=0)
    )
    print(
        "DSV4_FFN_STAGE_CHECK "
        f"stage={name} rel_l2={relative:.9f} cosine={cosine:.9f} "
        f"mean_abs={float(delta.abs().mean()):.9f} "
        f"max_abs={float(delta.abs().max()):.9f} "
        f"actual_nonfinite={int((~torch.isfinite(actual)).sum())} "
        f"expected_nonfinite={int((~torch.isfinite(expected)).sum())}",
        flush=True,
    )


def _report_exact(
    name: str, actual: torch.Tensor, expected: torch.Tensor
) -> None:
    actual = actual.reshape(-1)
    expected = expected.reshape(-1)
    mismatches = int((actual != expected).sum())
    print(
        "DSV4_FFN_STAGE_CHECK "
        f"stage={name} kind=exact mismatches={mismatches} "
        f"elements={actual.numel()} "
        f"fraction={mismatches / actual.numel():.9f}",
        flush=True,
    )


def _stream_experts(dump: dict[str, torch.Tensor]) -> list[int]:
    route_indices = dump["route_record"][:32].view(torch.int32)[:6]
    return [0, *[int(expert) + 1 for expert in route_indices]]


def check_linear1(
    dump: dict[str, torch.Tensor], image: dict[str, torch.Tensor]
) -> None:
    device = torch.device("cuda")
    activation = _decode_input_row(dump).to(device)
    middle_records = dump["mxfp_middle_records"]
    stream_experts = _stream_experts(dump)
    print(
        "DSV4_FFN_STAGE_INPUT "
        f"stage=linear1 stream_experts={stream_experts} "
        f"activation_norm={float(activation.norm()):.9f}",
        flush=True,
    )
    for physical_expert, stream_expert in enumerate(stream_experts):
        begin = stream_expert * LINEAR1_SLICES
        end = begin + LINEAR1_SLICES
        weights = image["linear1_weights"][begin:end]
        scales = image["linear1_scales"][begin:end]
        gate_weight = _dequantize_mxfp4(
            weights[:, :8], scales[:, :8]
        ).to(device)
        up_weight = _dequantize_mxfp4(
            weights[:, 8:], scales[:, 8:]
        ).to(device)
        gate = gate_weight @ activation
        up = up_weight @ activation
        expected = torch.nn.functional.silu(gate) * up
        bounded = torch.nn.functional.silu(gate.clamp(max=10.0)) * up.clamp(
            min=-10.0, max=10.0
        )
        (
            expected_quantized,
            expected_scales,
            expected_dequantized,
        ) = _quantize_middle_reference(expected)
        (
            bounded_quantized,
            bounded_scales,
            bounded_dequantized,
        ) = _quantize_middle_reference(bounded)
        actual_rows = torch.stack(
            [
                _decode_middle_row(
                    middle_records[physical_expert], row=row
                )
                for row in range(8)
            ]
        ).to(device)
        _report(
            f"linear1_expert_{physical_expert}_stream_{stream_expert}",
            actual_rows[0],
            expected,
        )
        _report(
            f"linear1_quantized_expert_{physical_expert}",
            actual_rows[0],
            expected_dequantized,
        )
        _report(
            f"linear1_model_bounded_expert_{physical_expert}",
            actual_rows[0],
            bounded_dequantized,
        )
        print(
            "DSV4_FFN_LINEAR1_LIMIT "
            f"physical_expert={physical_expert} "
            f"gate_above={int((gate > 10.0).sum())} "
            f"up_below={int((up < -10.0).sum())} "
            f"up_above={int((up > 10.0).sum())} "
            f"middle_changed={int((expected != bounded).sum())}",
            flush=True,
        )
        actual_quantized, actual_scales = _middle_row_payload(
            middle_records[physical_expert], row=0
        )
        _report_exact(
            f"linear1_data_expert_{physical_expert}",
            actual_quantized,
            bounded_quantized.cpu(),
        )
        _report_exact(
            f"linear1_scales_expert_{physical_expert}",
            actual_scales,
            bounded_scales.cpu(),
        )
        _report(
            f"linear1_row_replication_{physical_expert}",
            actual_rows[1:],
            actual_rows[0].expand_as(actual_rows[1:]),
        )
        del (
            gate_weight,
            up_weight,
            gate,
            up,
            expected,
            bounded,
            expected_dequantized,
            bounded_dequantized,
            actual_rows,
        )


def check_down(
    dump: dict[str, torch.Tensor], image: dict[str, torch.Tensor]
) -> None:
    device = torch.device("cuda")
    middle_records = dump["mxfp_middle_records"]
    stream_experts = _stream_experts(dump)
    route_weights = dump["route_record"][32:64].view(torch.float32)[:6]
    route_scales = torch.cat(
        (torch.ones(1, dtype=torch.float32), route_weights.float())
    ).to(device)
    expected = torch.zeros(4096, dtype=torch.float32, device=device)
    rounded_contributions = []
    print(
        "DSV4_FFN_STAGE_INPUT "
        f"stage=down stream_experts={stream_experts} "
        f"route_scales={route_scales.cpu().tolist()}",
        flush=True,
    )
    for physical_expert, stream_expert in enumerate(stream_experts):
        begin = stream_expert * DOWN_SLICES
        end = begin + DOWN_SLICES
        down_weight = _dequantize_mxfp4(
            image["down_weights"][begin:end],
            image["down_scales"][begin:end],
        ).to(device)
        middle = _decode_middle_row(
            middle_records[physical_expert], row=0
        ).to(device)
        contribution = down_weight @ middle
        contribution *= route_scales[physical_expert]
        # The production epilogue converts each TMA reduction producer to
        # BF16 before publishing it.  Sum those exact producer values in FP32
        # for the independent reference; only the subsequent reduction order
        # may introduce another BF16-rounding difference.
        rounded = contribution.to(torch.bfloat16).float()
        rounded_contributions.append(rounded)
        expected += rounded
        print(
            "DSV4_FFN_DOWN_CONTRIBUTION "
            f"physical_expert={physical_expert} "
            f"stream_expert={stream_expert} "
            f"route_scale={float(route_scales[physical_expert]):.9f} "
            f"norm={float(rounded.norm()):.9f}",
            flush=True,
        )
        del down_weight, middle, contribution

    actual = dump["mxfp_ffn_output"].float().to(device)
    _report("down_routed_reduce", actual[0], expected)
    _report(
        "down_output_row_replication",
        actual[1:],
        actual[0].expand_as(actual[1:]),
    )

    tile_errors = (
        (actual[0] - expected)
        .reshape(DOWN_SLICES, 128)
        .norm(dim=1)
        / expected.reshape(DOWN_SLICES, 128)
        .norm(dim=1)
        .clamp_min(1.0e-30)
    )
    for error, tile in zip(*torch.topk(tile_errors, 8)):
        print(
            "DSV4_FFN_DOWN_TILE "
            f"tile={int(tile)} rel_l2={float(error):.9f}",
            flush=True,
        )

    # Bound expected variation from BF16 reduction ordering.  This is not
    # used as the primary reference; it separates reduction rounding from a
    # wrong weight, activation, route, or output address.
    ordered = rounded_contributions[0].to(torch.bfloat16)
    for contribution in rounded_contributions[1:]:
        ordered = (ordered.float() + contribution).to(torch.bfloat16)
    _report("down_ordered_bf16_bound", actual[0], ordered.float())


def check_hc_post(dump: dict[str, torch.Tensor]) -> None:
    layer_id = int(dump["layer_ids"].item())
    # Match ResidentOneLaunchDecode._mxfp_output_set: SWA/CSA publish into
    # direction one and HCA publishes into direction zero.
    output_set = int(
        DeepSeekV4FlashConfig().attention_kind(layer_id) != "hca"
    )
    device = torch.device("cuda")
    metadata = dump["mhc_output_metadatas"][output_set].float().to(device)
    post = metadata[:4]
    comb = metadata[4:20].reshape(4, 4)
    branch = dump["mxfp_ffn_output"][0].float().to(device)
    input_residual = dump["next_residual"].float().to(device)
    # Match dsv4_hc_post_value's defined operation order: one rounded FP32
    # multiply followed by four IEEE FP32 fmaf operations.  PyTorch's
    # addcmul lowering is not required to fuse, so evaluate each product-plus-
    # add exactly in FP64 and round once to FP32 to model fmaf.
    expected = post[:, None] * branch[None, :]
    for residual_index in range(4):
        expected = (
            comb[residual_index, :, None].double()
            * input_residual[residual_index, None, :].double()
            + expected.double()
        ).float()
    expected_bf16 = expected.to(torch.bfloat16)
    actual = dump["residual"].to(device)
    print(
        "DSV4_FFN_STAGE_INPUT "
        f"stage=hc_post layer={layer_id} output_set={output_set} "
        f"branch_norm={float(branch.norm()):.9f} "
        f"residual_norm={float(input_residual.norm()):.9f}",
        flush=True,
    )
    _report("ffn_hc_post", actual, expected_bf16)
    _report_exact("ffn_hc_post_bf16", actual, expected_bf16)


def check_hc_post_isolated(dump: dict[str, torch.Tensor]) -> None:
    layer_id = int(dump["layer_ids"].item())
    output_set = int(
        DeepSeekV4FlashConfig().attention_kind(layer_id) != "hca"
    )
    device = torch.device("cuda")
    branch = dump["mxfp_ffn_output"][0].to(device).contiguous()
    residual = dump["next_residual"].to(device).contiguous()
    coefficients = (
        dump["mhc_output_metadatas"][output_set].to(device).contiguous()
    )
    post = coefficients[:4]
    comb = coefficients[4:].view(4, 4)
    output = torch.empty_like(residual)
    launcher = Launcher(32, device=device)
    launcher.s(
        SchedDsv4HcPost(
            branch,
            residual,
            post,
            comb,
            output,
            launcher=launcher,
            packed_coefficients=coefficients,
        ).place(32)
    )
    launcher.launch()
    torch.cuda.synchronize(device)
    integrated = dump["residual"].to(device)
    print(
        "DSV4_FFN_STAGE_INPUT "
        f"stage=hc_post_isolated layer={layer_id} output_set={output_set}",
        flush=True,
    )
    _report("hc_post_isolated_vs_integrated", output, integrated)
    _report_exact("hc_post_isolated_vs_integrated", output, integrated)

    expected = post[:, None].float() * branch[None, :].float()
    for residual_index in range(4):
        expected = (
            comb[residual_index, :, None].double()
            * residual[residual_index, None, :].double()
            + expected.double()
        ).float()
    expected = expected.to(torch.bfloat16)
    _report("hc_post_isolated_reference", output, expected)
    _report_exact("hc_post_isolated_reference", output, expected)


def check_reference_ffn(
    dump: dict[str, torch.Tensor], image: dict[str, torch.Tensor]
) -> None:
    """Evaluate the production MXFP4-weight/MXFP8-activation FFN in PyTorch."""
    required = (
        "reference_attention_post",
        "reference_ffn_normalized",
        "reference_ffn_post",
        "reference_ffn_comb",
        "reference_route_indices",
        "reference_route_weights",
    )
    missing = [name for name in required if name not in dump]
    if missing:
        raise ValueError(f"diagnostic dump misses reference fields {missing}")

    device = torch.device("cuda")
    normalized = dump["reference_ffn_normalized"].to(device)
    _, _, activation = _quantize_middle_reference(normalized)
    route_indices = dump["reference_route_indices"].to(torch.int32)
    route_weights = dump["reference_route_weights"].float().to(device)
    stream_experts = [0, *[int(expert) + 1 for expert in route_indices]]
    contributions = []
    limit_counts = []
    for physical_expert, stream_expert in enumerate(stream_experts):
        linear1_begin = stream_expert * LINEAR1_SLICES
        linear1_end = linear1_begin + LINEAR1_SLICES
        linear1_weights = image["linear1_weights"][
            linear1_begin:linear1_end
        ]
        linear1_scales = image["linear1_scales"][
            linear1_begin:linear1_end
        ]
        gate_weight = _dequantize_mxfp4(
            linear1_weights[:, :8], linear1_scales[:, :8]
        ).to(device)
        up_weight = _dequantize_mxfp4(
            linear1_weights[:, 8:], linear1_scales[:, 8:]
        ).to(device)
        gate = gate_weight @ activation
        up = up_weight @ activation
        limit_counts.append(
            (
                int((gate > 10.0).sum()),
                int((up < -10.0).sum()),
                int((up > 10.0).sum()),
            )
        )
        middle = torch.nn.functional.silu(gate.clamp(max=10.0))
        middle *= up.clamp(min=-10.0, max=10.0)
        _, _, middle = _quantize_middle_reference(middle)

        down_begin = stream_expert * DOWN_SLICES
        down_end = down_begin + DOWN_SLICES
        down_weight = _dequantize_mxfp4(
            image["down_weights"][down_begin:down_end],
            image["down_scales"][down_begin:down_end],
        ).to(device)
        route_scale = (
            torch.tensor(1.0, device=device)
            if physical_expert == 0
            else route_weights[physical_expert - 1]
        )
        contribution = (down_weight @ middle) * route_scale
        contributions.append(contribution.to(torch.bfloat16).float())
        del (
            gate_weight,
            up_weight,
            gate,
            up,
            middle,
            down_weight,
            contribution,
        )

    reference_branch = torch.stack(contributions).sum(dim=0).to(
        torch.bfloat16
    )
    actual_branch = dump["mxfp_ffn_output"][0].to(device)
    actual_normalized = dump["ffn_normalized"].to(device)
    actual_route_indices = dump["route_record"][:32].view(torch.int32)[:6]
    actual_route_weights = dump["route_record"][32:64].view(torch.float32)[:6]
    _report("production_ffn_input", actual_normalized, normalized)
    _report_exact(
        "production_ffn_route_indices",
        actual_route_indices,
        route_indices,
    )
    _report(
        "production_ffn_route_weights",
        actual_route_weights,
        route_weights.cpu(),
    )
    _report("production_ffn_branch_e2e", actual_branch, reference_branch)

    reference_post = dump["reference_ffn_post"].float().to(device)
    reference_comb = dump["reference_ffn_comb"].float().to(device)
    reference_residual = dump["reference_attention_post"].float().to(device)
    reference_output = (
        reference_post[:, None] * reference_branch.float()[None, :]
    )
    reference_output += torch.einsum(
        "ij,id->jd", reference_comb, reference_residual
    )
    reference_output = reference_output.to(torch.bfloat16)
    _report(
        "production_layer_output_e2e",
        dump["residual"].to(device),
        reference_output,
    )
    print(
        "DSV4_FFN_REFERENCE_LIMITS "
        f"counts={limit_counts} stream_experts={stream_experts}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path)
    parser.add_argument("--mxfp-ffn-root", type=Path, required=True)
    parser.add_argument(
        "--layer-id",
        type=int,
        help="select the terminal layer in a production-prefix dump",
    )
    parser.add_argument(
        "--route-bank",
        type=int,
        choices=(0, 1),
        help="select the active route-record bank in a production-prefix dump",
    )
    parser.add_argument(
        "--stage",
        choices=(
            "linear1",
            "down",
            "hc-post",
            "hc-post-isolated",
            "reference-ffn",
        ),
        default="linear1",
    )
    args = parser.parse_args()

    dump = torch.load(args.dump, map_location="cpu", weights_only=True)
    if args.layer_id is not None:
        if args.route_bank is None:
            parser.error("--layer-id requires --route-bank")
        dump["layer_ids"] = torch.tensor([args.layer_id], dtype=torch.int64)
        dump["route_record"] = dump["route_records"][args.route_bank]
    layer_ids = dump.get("layer_ids")
    if layer_ids is None or layer_ids.numel() != 1:
        raise ValueError("the FFN stage checker requires a one-layer dump")
    layer_id = int(layer_ids.item())
    image = _load_layer(args.mxfp_ffn_root, layer_id)
    if args.stage == "linear1":
        check_linear1(dump, image)
    elif args.stage == "down":
        check_down(dump, image)
    elif args.stage == "hc-post":
        check_hc_post(dump)
    elif args.stage == "hc-post-isolated":
        check_hc_post_isolated(dump)
    elif args.stage == "reference-ffn":
        check_reference_ffn(dump, image)


if __name__ == "__main__":
    main()
