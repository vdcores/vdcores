#!/usr/bin/env python3
"""Correctness and latency benchmark for the DeepSeek-V4 NVFP4 GEMV task."""

from __future__ import annotations

import argparse
import statistics

import torch

from dae.deepseek_v4_schedule import DeepSeekV4ShapePolicy
from dae.deepseek_v4_quant import dequantize_nvfp4, quantize_nvfp4
from dae.instructions import ProfileEvent, TmaTensor
from dae.launcher import Launcher
from dae.routing import RoutedAddressTable
from dae.schedule import (
    SchedDsv4Nvfp4QuantUmmaB,
    SchedNvfp4Gemv,
    SchedNvfp4GemvUmma,
    SchedNvfp4GemvUmmaStream,
    SchedNvfp4UmmaPrepack,
    SchedRoutedNvfp4Gemv,
    SchedRoutedNvfp4GemvUmmaStream,
)
from dae.tma_utils import Major


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--sms", type=int, default=0)
    parser.add_argument(
        "--implementation",
        choices=("cuda", "umma", "umma_stream"),
        default="cuda",
    )
    parser.add_argument(
        "--routed-tiles",
        action="store_true",
        help=(
            "use the production routed/TMA tiling path; this permits the "
            "25-SM expert partition whose full static shard exceeds uint16"
        ),
    )
    parser.add_argument(
        "--routed-native",
        action="store_true",
        help="resolve combined native weight tiles through the routing table",
    )
    parser.add_argument(
        "--sms-list",
        default="",
        help="comma-separated SM counts; reuses one quantized input for a sweep",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--native-activation-quant",
        action="store_true",
        help=(
            "emit token-dependent activation data/scales directly in the "
            "combined native UMMA layout and verify it against setup prepack"
        ),
    )
    parser.add_argument(
        "--bulk-activation",
        action="store_true",
        help="load all native activation K tiles in one shared allocation",
    )
    parser.add_argument(
        "--reuse-activation-pair",
        action="store_true",
        help="benchmark W1/W3-style retain then RegLoad reuse on the same SMs",
    )
    parser.add_argument("--trace-stages", action="store_true")
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="print the first launch's vectors and error distribution",
    )
    parser.add_argument(
        "--unit-scales",
        action="store_true",
        help="use exact E2M1 sources with every NVFP4 block scale equal to one",
    )
    parser.add_argument(
        "--dump-columns",
        action="store_true",
        help="drain and verify all eight native UMMA output columns",
    )
    parser.add_argument(
        "--identity-weight",
        action="store_true",
        help="use an exact packed FP4 identity matrix (requires M=K)",
    )
    parser.add_argument(
        "--indexed-activation",
        action="store_true",
        help="use packed codes k mod 16 with unit block/global scales",
    )
    parser.add_argument(
        "--block-indexed-activation",
        action="store_true",
        help="use one distinct FP4 code per 16-value scale block",
    )
    parser.add_argument(
        "--indexed-scales",
        action="store_true",
        help="use constant +1 FP4 data and one distinct E4M3 scale per block",
    )
    args = parser.parse_args()
    if args.routed_tiles and args.implementation != "cuda":
        parser.error("--routed-tiles currently supports only --implementation cuda")
    if args.routed_native and args.implementation != "umma_stream":
        parser.error("--routed-native requires --implementation umma_stream")
    if args.routed_native and args.reuse_activation_pair:
        parser.error("--routed-native does not use the static paired benchmark")
    if args.native_activation_quant and args.implementation != "umma_stream":
        parser.error("--native-activation-quant requires --implementation umma_stream")
    if args.bulk_activation and args.implementation != "umma_stream":
        parser.error("--bulk-activation requires --implementation umma_stream")
    if args.reuse_activation_pair and args.implementation != "umma_stream":
        parser.error("--reuse-activation-pair requires --implementation umma_stream")
    if args.native_activation_quant and (
        args.indexed_activation
        or args.block_indexed_activation
        or args.indexed_scales
    ):
        parser.error("native activation quant requires an unmodified BF16 source")

    def stage(name: str) -> None:
        if args.trace_stages:
            torch.cuda.synchronize()
            print(f"DSV4_NVFP4_STAGE {name}", flush=True)

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260810)
    if args.unit_scales:
        codebook = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            dtype=torch.bfloat16,
            device=device,
        )
        weight_source = codebook[
            torch.randint(
                0, 16, (args.m, args.k), generator=generator, device=device
            )
        ]
        # Constant activation data and scales isolate A-layout/output mapping;
        # every logical B row staged for the N=8 UMMA tile is identical.
        input_source = torch.ones(
            (args.k,), dtype=torch.bfloat16, device=device
        )
        weight_source.reshape(args.m, -1, 16)[:, :, 0] = 6.0
        forced_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        weight_source = torch.randn(
            (args.m, args.k), generator=generator,
            dtype=torch.bfloat16, device=device
        ) * 0.05
        input_source = torch.randn(
            (args.k,), generator=generator, dtype=torch.bfloat16, device=device
        ) * 0.1
        forced_scale = None
    stage("sources_ready")
    weight, weight_sf, weight_scale2 = quantize_nvfp4(
        weight_source, forced_scale
    )
    if args.identity_weight:
        if args.m != args.k:
            raise ValueError("--identity-weight requires M=K")
        weight = torch.zeros(
            (args.m, args.k // 2), dtype=torch.uint8, device=device
        )
        diagonal = torch.arange(args.m, device=device)
        identity_codes = torch.where(
            (diagonal & 1) == 0,
            torch.full_like(diagonal, 2),
            torch.full_like(diagonal, 2 << 4),
        ).to(torch.uint8)
        weight[diagonal, diagonal // 2] = identity_codes
        weight_sf = torch.ones(
            (args.m, args.k // 16),
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        weight_scale2 = torch.ones((), dtype=torch.float32, device=device)
    stage("weight_quantized")
    activation, activation_sf, input_scale = quantize_nvfp4(
        input_source, forced_scale
    )
    if args.indexed_activation or args.block_indexed_activation:
        activation_indices = torch.arange(
            args.k, device=device, dtype=torch.int64
        )
        if args.block_indexed_activation:
            activation_codes = ((activation_indices // 16) % 16).to(
                torch.uint8
            )
        else:
            activation_codes = (activation_indices % 16).to(torch.uint8)
        activation = (
            activation_codes[0::2] | (activation_codes[1::2] << 4)
        ).contiguous()
        activation_sf = torch.ones(
            (args.k // 16,), dtype=torch.float8_e4m3fn, device=device
        )
        input_scale = torch.ones((), dtype=torch.float32, device=device)
    if args.indexed_scales:
        scale_count = args.k // 16
        if scale_count > 32:
            raise ValueError("--indexed-scales supports at most 32 blocks")
        activation = torch.full(
            (args.k // 2,), 0x22, dtype=torch.uint8, device=device
        )
        activation_sf = torch.arange(
            48, 48 + scale_count, dtype=torch.uint8, device=device
        ).view(torch.float8_e4m3fn)
        input_scale = torch.ones((), dtype=torch.float32, device=device)
    if args.unit_scales:
        assert bool((weight_sf.float() == 1.0).all().item())
        assert activation_sf.float().unique().numel() == 1
    stage("activation_quantized")
    alpha = (weight_scale2 * input_scale).reshape(1)
    reference = (
        dequantize_nvfp4(weight, weight_sf, weight_scale2)
        @ dequantize_nvfp4(activation, activation_sf, input_scale)
    ).to(torch.bfloat16)
    stage("reference_ready")

    device_sms = torch.cuda.get_device_properties(device).multi_processor_count
    if args.sms_list:
        sms_values = [int(value) for value in args.sms_list.split(",")]
    else:
        default_sms = (
            min(args.m, device_sms)
            if args.implementation == "cuda"
            else (args.m + 127) // 128
        )
        sms_values = [args.sms or default_sms]
    if any(value <= 0 or value > device_sms for value in sms_values):
        raise ValueError(f"SM counts must be in [1, {device_sms}]")

    for num_sms in sms_values:
        output_columns = 8 if args.dump_columns else 1
        if args.dump_columns and args.implementation != "umma":
            raise ValueError("--dump-columns requires --implementation umma")
        output = torch.empty(
            (args.m * output_columns,), dtype=torch.bfloat16, device=device
        )
        launcher = Launcher(num_sms, device=device)
        if args.routed_tiles:
            assignment = DeepSeekV4ShapePolicy(num_sms).nvfp4_gemv(
                args.m, args.k
            )
            columns: dict[str, list[int]] = {}
            weight_field_names: list[tuple[str, ...]] = []
            scale_field_names: list[tuple[str, ...]] = []
            for sm in range(assignment.num_sms):
                row_start, row_count = assignment.shard(sm)
                sm_weight_fields = []
                sm_scale_fields = []
                for tile_index, tile_start in enumerate(
                    range(
                        row_start,
                        row_start + row_count,
                        assignment.tile_rows,
                    )
                ):
                    weight_name = f"weight.sm{sm}.tile{tile_index}"
                    scale_name = f"weight_scale.sm{sm}.tile{tile_index}"
                    weight_address = (
                        weight.data_ptr()
                        + tile_start * weight.stride(0) * weight.element_size()
                    )
                    scale_address = (
                        weight_sf.data_ptr()
                        + tile_start
                        * weight_sf.stride(0)
                        * weight_sf.element_size()
                    )
                    columns[weight_name] = [weight_address] * 256
                    columns[scale_name] = [scale_address] * 256
                    sm_weight_fields.append(weight_name)
                    sm_scale_fields.append(scale_name)
                weight_field_names.append(tuple(sm_weight_fields))
                scale_field_names.append(tuple(sm_scale_fields))
            alpha_storage = torch.zeros(
                (4,), dtype=torch.float32, device=device
            )
            alpha_storage[0].copy_(alpha.reshape(-1)[0])
            columns["alpha"] = [alpha_storage.data_ptr()] * 256
            table = RoutedAddressTable.from_pointer_columns(
                columns,
                device=device,
                owners=(weight, weight_sf, alpha_storage),
            )
            table.route_indices[0] = 0
            schedule = SchedRoutedNvfp4Gemv(
                table.state,
                route_rank=0,
                weight_fields=[
                    tuple(table.field(name) for name in names)
                    for names in weight_field_names
                ],
                weight_scale_fields=[
                    tuple(table.field(name) for name in names)
                    for names in scale_field_names
                ],
                alpha_field=table.field("alpha"),
                rows=args.m,
                k=args.k,
                activation=activation,
                activation_scale=activation_sf,
                output=output,
                route_ready=True,
            ).place(num_sms)
        elif args.implementation == "umma_stream":
            if args.m % 128 or args.k % 256:
                raise ValueError("streaming UMMA requires M128 and K256 alignment")
            m_tiles = args.m // 128
            k_tiles = args.k // 256
            if num_sms != m_tiles and not args.routed_native:
                raise ValueError(
                    "streaming UMMA requires one SM per M128 output tile"
                )
            weight_scale_tiles = (
                weight_sf.view(m_tiles, 128, k_tiles, 16)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            activation_rows = (
                activation.reshape(1, -1).expand(8, -1).contiguous()
            )
            activation_scale_tiles = activation_sf.view(k_tiles, 16)
            alpha_storage = torch.zeros(
                (4,), dtype=torch.float32, device=device
            )
            alpha_storage[0].copy_(alpha.reshape(-1)[0])

            # This is immutable/setup work, deliberately outside the measured
            # token-time launcher.  The resulting native tiles place data and
            # scales in one byte range so each operand needs only one LDU/TMA
            # transaction in the streaming task.
            packed_weight = torch.empty(
                (
                    m_tiles,
                    k_tiles,
                    SchedNvfp4UmmaPrepack.WEIGHT_TILE_BYTES,
                ),
                dtype=torch.uint8,
                device=device,
            )
            packed_activation = torch.empty(
                (
                    k_tiles,
                    SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES,
                ),
                dtype=torch.uint8,
                device=device,
            )
            weight_prepack_launcher = Launcher(m_tiles, device=device)
            weight_tma = TmaTensor(weight_prepack_launcher, weight).wgmma_load(
                128, 128, Major.K
            )
            weight_prepack_launcher.s(
                SchedNvfp4UmmaPrepack(
                    SchedNvfp4UmmaPrepack.WEIGHT,
                    weight,
                    weight_scale_tiles,
                    packed_weight,
                    weight_tma,
                ).place(m_tiles)
            )
            weight_prepack_launcher.launch()

            activation_prepack_launcher = Launcher(1, device=device)
            activation_tma = TmaTensor(
                activation_prepack_launcher, activation_rows
            ).wgmma_load(8, 128, Major.K)
            activation_prepack_launcher.s(
                SchedNvfp4UmmaPrepack(
                    SchedNvfp4UmmaPrepack.ACTIVATION,
                    activation_rows,
                    activation_scale_tiles,
                    packed_activation,
                    activation_tma,
                ).place(1)
            )
            activation_prepack_launcher.launch()
            torch.cuda.synchronize()
            stage(f"native_prepack_complete_{num_sms}")

            if args.native_activation_quant:
                prepacked_activation = packed_activation
                packed_activation = torch.empty_like(prepacked_activation)
                native_quant_launcher = Launcher(k_tiles, device=device)
                native_quant_schedule = SchedDsv4Nvfp4QuantUmmaB(
                    input_source,
                    input_scale.reshape(-1),
                    packed_activation,
                ).place(k_tiles)
                native_quant_launcher.s(
                    ProfileEvent(2),
                    native_quant_schedule,
                    ProfileEvent(3),
                )
                native_quant_launcher.launch()
                torch.cuda.synchronize()
                if not torch.equal(packed_activation, prepacked_activation):
                    mismatch = packed_activation != prepacked_activation
                    data_mismatch = mismatch[:, :1024].sum().item()
                    scale_mismatch = mismatch[:, 1024:].sum().item()
                    first = mismatch.nonzero()[0]
                    tile_index, byte_index = (int(value) for value in first)
                    scale_indices = mismatch[:, 1024:].nonzero()
                    if scale_indices.numel():
                        scale_tile, scale_offset = (
                            int(value) for value in scale_indices[0]
                        )
                        scale_byte = scale_offset + 1024
                        scale_detail = (
                            f"{scale_tile}:{scale_byte}:"
                            f"{int(packed_activation[scale_tile, scale_byte])}:"
                            f"{int(prepacked_activation[scale_tile, scale_byte])}"
                        )
                    else:
                        scale_detail = "none"
                    print(
                        "DSV4_NVFP4_NATIVE_LAYOUT_MISMATCH "
                        f"data_bytes={data_mismatch} "
                        f"scale_bytes={scale_mismatch} "
                        f"per_tile={mismatch.sum(dim=1).cpu().tolist()} "
                        f"first={tile_index}:{byte_index} "
                        f"actual={int(packed_activation[tile_index, byte_index])} "
                        f"expected={int(prepacked_activation[tile_index, byte_index])} "
                        f"scale_first_actual_expected={scale_detail}",
                        flush=True,
                    )
                    if args.k == 256:
                        source_chunks = activation.reshape(8, 16).cpu()

                        def chunk_map(tensor):
                            chunks = tensor[0, :1024].reshape(64, 16).cpu()
                            return [
                                next(
                                    (
                                        index
                                        for index in range(8)
                                        if torch.equal(chunk, source_chunks[index])
                                    ),
                                    -1,
                                )
                                for chunk in chunks
                            ]

                        print(
                            "DSV4_NVFP4_NATIVE_CHUNK_MAP "
                            f"actual={chunk_map(packed_activation)} "
                            f"expected={chunk_map(prepacked_activation)}",
                            flush=True,
                        )
                torch.testing.assert_close(
                    packed_activation,
                    prepacked_activation,
                    rtol=0,
                    atol=0,
                )
                for _ in range(args.warmup):
                    native_quant_launcher.launch()
                torch.cuda.synchronize()
                native_quant_timings = []
                for _ in range(args.iterations):
                    native_quant_launcher.launch()
                    native_profile = (
                        native_quant_launcher.profile[:, :4].cpu().numpy()
                    )
                    native_quant_timings.append(
                        (
                            native_profile[:, 3].max()
                            - native_profile[:, 2].min()
                        )
                        / 1.0e3
                    )
                print(
                    "DSV4_NVFP4_QUANT_UMMA_RESULT "
                    f"shape=1x{args.k} sms={k_tiles} "
                    f"task_min_us={min(native_quant_timings):.6f} "
                    f"task_median_us={statistics.median(native_quant_timings):.6f} "
                    f"task_max_us={max(native_quant_timings):.6f} "
                    "layout_exact=true",
                    flush=True,
                )

            if args.routed_native:
                columns = {}
                native_field_names = []
                for m_tile in range(m_tiles):
                    name = f"native.weight.m{m_tile}"
                    columns[name] = [packed_weight[m_tile, 0].data_ptr()]
                    native_field_names.append(name)
                columns["native.alpha"] = [alpha_storage.data_ptr()]
                native_table = RoutedAddressTable.from_pointer_columns(
                    columns,
                    device=device,
                    owners=(packed_weight, alpha_storage),
                )
                native_table.route_indices[0] = 0
                schedule = SchedRoutedNvfp4GemvUmmaStream(
                    native_table.state,
                    0,
                    tuple(
                        native_table.field(name) for name in native_field_names
                    ),
                    native_table.field("native.alpha"),
                    packed_activation,
                    output,
                    route_ready=True,
                    activation_mode=(
                        "load" if args.bulk_activation else "stream"
                    ),
                ).place(num_sms)
            elif args.reuse_activation_pair:
                reuse_output = torch.empty_like(output)
                schedule = (
                    SchedNvfp4GemvUmmaStream(
                        packed_weight,
                        packed_activation,
                        alpha_storage,
                        output,
                        activation_mode="retain",
                    ).place(num_sms),
                    SchedNvfp4GemvUmmaStream(
                        packed_weight,
                        packed_activation,
                        alpha_storage,
                        reuse_output,
                        activation_mode="reuse",
                    ).place(num_sms),
                )
            else:
                schedule = SchedNvfp4GemvUmmaStream(
                    packed_weight,
                    packed_activation,
                    alpha_storage,
                    output,
                    activation_mode=(
                        "load" if args.bulk_activation else "stream"
                    ),
                ).place(num_sms)
        else:
            schedule_cls = (
                SchedNvfp4Gemv
                if args.implementation == "cuda"
                else SchedNvfp4GemvUmma
            )
            schedule = schedule_cls(
                weight,
                weight_sf,
                activation,
                activation_sf,
                alpha,
                output,
                **(
                    {"output_columns": output_columns}
                    if args.implementation == "umma"
                    else {}
                ),
            ).place(num_sms)
        launcher.s(
            ProfileEvent(2),
            *(schedule if isinstance(schedule, tuple) else (schedule,)),
            ProfileEvent(3),
        )
        stage(f"launcher_ready_{num_sms}")
        launcher.launch()
        torch.cuda.synchronize()
        stage(f"first_launch_complete_{num_sms}")

        output_matrix = output.reshape(args.m, output_columns)
        comparison = output_matrix[:, 0]
        expected = reference.reshape(args.m, 1).expand_as(output_matrix)
        max_abs = (comparison.float() - reference.float()).abs().max().item()
        mean_rel = (
            (comparison.float() - reference.float()).abs().mean()
            / reference.float().abs().mean().clamp_min(1.0e-8)
        ).item()
        if args.diagnostic:
            difference = (comparison.float() - reference.float()).abs()
            print(f"output={comparison.float().cpu().tolist()}", flush=True)
            print(f"reference={reference.float().cpu().tolist()}", flush=True)
            print(
                "diagnostic "
                f"bad_indices={(difference > 5e-2).nonzero().flatten().cpu().tolist()} "
                f"cosine={torch.nn.functional.cosine_similarity(comparison.float(), reference.float(), dim=0).item():.8f}",
                flush=True,
            )
        if args.dump_columns:
            all_errors = (output_matrix.float() - expected.float()).abs()
            column_errors = all_errors.amax(dim=0)
            tolerance = 5e-2 + 2e-2 * expected.float().abs()
            good = all_errors <= tolerance
            good_masks = sum(
                good[:, column].to(torch.int32) << column
                for column in range(output_columns)
            )
            unique_masks, mask_counts = torch.unique(
                good_masks, return_counts=True
            )
            mask_histogram = {
                int(mask): int(count)
                for mask, count in zip(
                    unique_masks.cpu().tolist(), mask_counts.cpu().tolist()
                )
            }
            print(
                f"column_max_abs={column_errors.cpu().tolist()} "
                f"row_good_mask_histogram={mask_histogram}", flush=True
            )
        torch.testing.assert_close(
            output_matrix, expected, rtol=2e-2, atol=5e-2
        )
        if args.reuse_activation_pair:
            torch.testing.assert_close(
                reuse_output.reshape_as(expected),
                expected,
                rtol=2e-2,
                atol=5e-2,
            )

        for _ in range(args.warmup):
            launcher.launch()
        torch.cuda.synchronize()
        kernel_timings = []
        task_timings = []
        for _ in range(args.iterations):
            launcher.launch()
            profile = launcher.profile[:, :4].cpu().numpy()
            kernel_timings.append(
                (profile[:, 1].max() - profile[:, 0].min()) / 1.0e3
            )
            task_timings.append(
                (profile[:, 3].max() - profile[:, 2].min()) / 1.0e3
            )

        print(
            "DSV4_NVFP4_GEMV_RESULT "
            f"implementation={args.implementation} "
            f"addressing={('routed_tiled' if args.routed_tiles else ('routed_native' if args.routed_native else 'static'))} "
            f"shape={args.m}x1x{args.k} sms={num_sms} "
            f"passes={2 if args.reuse_activation_pair else 1} "
            f"task_min_us={min(task_timings):.6f} "
            f"task_median_us={statistics.median(task_timings):.6f} "
            f"task_max_us={max(task_timings):.6f} "
            f"kernel_median_us={statistics.median(kernel_timings):.6f} "
            f"max_abs={max_abs:.6f} "
            f"mean_relative={mean_rel:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
