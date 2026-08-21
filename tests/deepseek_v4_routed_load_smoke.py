#!/usr/bin/env python3
"""One-launch route-to-LDU-to-NVFP4 correctness smoke."""

from __future__ import annotations

import torch

from dae.deepseek_v4 import route_top6_reference
from dae.deepseek_v4_quant import dequantize_nvfp4, quantize_nvfp4
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.schedule import LayeredSchedule, SchedDsv4RouteTop6, SchedRoutedNvfp4Gemv


def main() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810)
    rows, k, num_sms = 128, 4096, 2
    selected_expert = 37

    default_source = torch.randn(
        (rows, k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.03
    selected_source = torch.randn(
        (rows, k), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.07
    activation_source = torch.randn(
        (k,), generator=generator, dtype=torch.bfloat16, device=device
    ) * 0.1
    default_weight, default_scale, default_scale2 = quantize_nvfp4(
        default_source
    )
    selected_weight, selected_scale, selected_scale2 = quantize_nvfp4(
        selected_source
    )
    activation, activation_scale, input_scale = quantize_nvfp4(
        activation_source
    )
    default_alpha = torch.zeros((4,), dtype=torch.float32, device=device)
    selected_alpha = torch.zeros((4,), dtype=torch.float32, device=device)
    default_alpha[0] = default_scale2 * input_scale
    selected_alpha[0] = selected_scale2 * input_scale
    output_w1 = torch.empty((rows,), dtype=torch.bfloat16, device=device)
    output_w3 = torch.empty((rows,), dtype=torch.bfloat16, device=device)

    def expert_column(default, selected):
        column = [default] * 256
        column[selected_expert] = selected
        return column

    columns = {
        "alpha": expert_column(default_alpha, selected_alpha),
    }
    weight_fields = []
    weight_scale_fields = []
    rows_per_sm = rows // num_sms
    tile_rows = (65520 // (k // 2) // 8) * 8
    for sm in range(num_sms):
        row_start = sm * rows_per_sm
        row_stop = row_start + rows_per_sm
        sm_weight_fields = []
        sm_scale_fields = []
        for tile_index, tile_start in enumerate(
            range(row_start, row_stop, tile_rows)
        ):
            tile_stop = min(row_stop, tile_start + tile_rows)
            weight_name = f"weight_sm{sm}_tile{tile_index}"
            scale_name = f"weight_scale_sm{sm}_tile{tile_index}"
            columns[weight_name] = expert_column(
                default_weight[tile_start:tile_stop],
                selected_weight[tile_start:tile_stop],
            )
            columns[scale_name] = expert_column(
                default_scale[tile_start:tile_stop],
                selected_scale[tile_start:tile_stop],
            )
            sm_weight_fields.append(weight_name)
            sm_scale_fields.append(scale_name)
        weight_fields.append(tuple(sm_weight_fields))
        weight_scale_fields.append(tuple(sm_scale_fields))
    owners = tuple(target for column in columns.values() for target in column)
    table = RoutedAddressTable.from_pointer_columns(
        {
            name: [target.data_ptr() for target in column]
            for name, column in columns.items()
        },
        device=device,
        owners=owners,
    )

    logits = torch.linspace(
        -1.0, 1.0, 256, dtype=torch.float32, device=device
    )
    bias = torch.zeros((256,), dtype=torch.float32, device=device)
    hash_indices = torch.zeros((8,), dtype=torch.int32, device=device)
    hash_indices[:6] = torch.tensor(
        [selected_expert, 0, 1, 2, 3, 4], dtype=torch.int32, device=device
    )
    route_weights = torch.empty((8,), dtype=torch.float32, device=device)
    route_indices = torch.empty((8,), dtype=torch.int32, device=device)

    launcher = Launcher(num_sms, device=device)
    route_bar = launcher.new_bar(1)
    output_bar = launcher.new_bar(num_sms)
    route = SchedDsv4RouteTop6(
        logits,
        bias,
        hash_indices,
        route_indices,
        route_weights,
        hash_routing=True,
    ).bar("output", route_bar).place(1)
    expert_w1_inner = SchedRoutedNvfp4Gemv(
        table.state,
        route_rank=0,
        weight_fields=[
            tuple(table.field(name) for name in fields)
            for fields in weight_fields
        ],
        weight_scale_fields=[
            tuple(table.field(name) for name in fields)
            for fields in weight_scale_fields
        ],
        alpha_field=table.field("alpha"),
        rows=rows,
        k=k,
        activation=activation,
        activation_scale=activation_scale,
        output=output_w1,
        activation_mode="retain",
    )
    expert_w1 = (
        LayeredSchedule(
            expert_w1_inner,
            ((table.state, (table.state,)),),
            route_indices=route_indices,
        )
        .bar("route", route_bar)
        .place(num_sms)
    )
    expert_w3_inner = SchedRoutedNvfp4Gemv(
        table.state,
        route_rank=0,
        weight_fields=[
            tuple(table.field(name) for name in fields)
            for fields in weight_fields
        ],
        weight_scale_fields=[
            tuple(table.field(name) for name in fields)
            for fields in weight_scale_fields
        ],
        alpha_field=table.field("alpha"),
        rows=rows,
        k=k,
        activation=activation,
        activation_scale=activation_scale,
        output=output_w3,
        route_ready=True,
        activation_mode="reuse",
    )
    expert_w3 = (
        LayeredSchedule(
            expert_w3_inner,
            ((table.state, (table.state,)),),
            route_indices=route_indices,
        )
        .bar("output", output_bar)
        .place(num_sms)
    )
    launcher.s(route, expert_w1, expert_w3)
    launcher.launch()

    reference = (
        dequantize_nvfp4(selected_weight, selected_scale, selected_scale2)
        @ dequantize_nvfp4(activation, activation_scale, input_scale)
    ).to(torch.bfloat16)
    torch.testing.assert_close(output_w1, reference, rtol=5.0e-2, atol=5.0e-2)
    torch.testing.assert_close(output_w3, reference, rtol=5.0e-2, atol=5.0e-2)
    expected_route_weights, _ = route_top6_reference(
        logits, bias, hash_indices=hash_indices[:6]
    )
    torch.testing.assert_close(
        route_weights[:6], expected_route_weights, rtol=1.0e-5, atol=1.0e-5
    )
    assert route_indices.tolist()[0] == selected_expert
    assert launcher.bars.view(torch.int32)[output_bar].item() == 0
    max_abs = max(
        (output_w1.float() - reference.float()).abs().max().item(),
        (output_w3.float() - reference.float()).abs().max().item(),
    )
    print(
        "DSV4_ROUTED_LOAD status=PASS indirect=1 adaptive_fusion=1 "
        f"tiled=1 launches=1 selected_expert={selected_expert} sms={num_sms} "
        f"max_abs={max_abs:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
