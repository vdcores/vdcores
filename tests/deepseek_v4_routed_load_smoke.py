#!/usr/bin/env python3
"""One-launch route-to-LDU-to-NVFP4 correctness smoke."""

from __future__ import annotations

import torch

from dae.deepseek_v4_quant import dequantize_nvfp4, quantize_nvfp4
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.schedule import SchedDsv4RouteTop6, SchedRoutedNvfp4Gemv


def main() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810)
    rows, k, num_sms = 128, 256, 4
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
    default_alpha = (default_scale2 * input_scale).reshape(1)
    selected_alpha = (selected_scale2 * input_scale).reshape(1)
    output = torch.empty((rows,), dtype=torch.bfloat16, device=device)

    def expert_column(default, selected):
        column = [default] * 256
        column[selected_expert] = selected
        return column

    table = RoutedAddressTable(
        {
            "weight": expert_column(default_weight, selected_weight),
            "weight_scale": expert_column(default_scale, selected_scale),
            "activation": [activation] * 256,
            "activation_scale": [activation_scale] * 256,
            "alpha": expert_column(default_alpha, selected_alpha),
            "output": [output] * 256,
        }
    )

    logits = torch.zeros((256,), dtype=torch.bfloat16, device=device)
    bias = torch.zeros((256,), dtype=torch.float32, device=device)
    hash_indices = torch.tensor(
        [selected_expert, 0, 1, 2, 3, 4], dtype=torch.int32, device=device
    )
    route_weights = torch.empty((6,), dtype=torch.float32, device=device)

    launcher = Launcher(num_sms, device=device)
    route_bar = launcher.new_bar(1)
    output_bar = launcher.new_bar(num_sms)
    route = SchedDsv4RouteTop6(
        logits,
        bias,
        hash_indices,
        table.route_indices,
        route_weights,
        hash_routing=True,
    ).bar("output", route_bar).place(1)
    expert = SchedRoutedNvfp4Gemv(
        table.state,
        route_rank=0,
        weight_field=table.field("weight"),
        weight_scale_field=table.field("weight_scale"),
        activation_field=table.field("activation"),
        activation_scale_field=table.field("activation_scale"),
        alpha_field=table.field("alpha"),
        output_field=table.field("output"),
        rows=rows,
        k=k,
        activation=activation,
        activation_scale=activation_scale,
        output=output,
    ).bar("route", route_bar).bar("output", output_bar).place(num_sms)
    launcher.s(route, expert)
    launcher.launch()

    reference = (
        dequantize_nvfp4(selected_weight, selected_scale, selected_scale2)
        @ dequantize_nvfp4(activation, activation_scale, input_scale)
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, reference, rtol=5.0e-2, atol=5.0e-2)
    assert table.route_indices.tolist()[0] == selected_expert
    assert launcher.bars.view(torch.int32)[output_bar].item() == 0
    max_abs = (output.float() - reference.float()).abs().max().item()
    print(
        "DSV4_ROUTED_LOAD status=PASS "
        f"launches=1 selected_expert={selected_expert} sms={num_sms} "
        f"max_abs={max_abs:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
