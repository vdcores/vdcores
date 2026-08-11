import torch

from dae.instructions import ArgmaxSmemPartialBf16, ArgmaxSmemReduceBf16

from dae.deepseek_v4 import (
    DeepSeekV4FlashConfig,
    apply_partial_rope_128_64,
    apply_partial_rope_512_64,
    bounded_swiglu,
    decode_compressed_indices,
    decode_window_indices,
    deepseek_v4_rope_table,
    gated_pool_reference,
    hadamard_reference,
    hc_head_reference,
    hc_post_reference,
    hc_pre_reference,
    index_score_reference,
    route_top6_reference,
    sparse_attention_512_reference,
)
from dae.deepseek_v4_flow import build_decode_plan, build_layer_decode_plan
from dae.runtime import opcode
from dae.schedule import (
    SchedArgmaxSmemPartial,
    SchedArgmaxSmemReduce,
    SchedDsv4Bf16Gemv,
    SchedDsv4Fp32Bf16Gemv,
    SchedDsv4Fp8Quant128,
    SchedDsv4Nvfp4Quant16,
    SchedFp8Block128Gemv,
)


def test_shared_argmax_instructions_and_shape_sharding(monkeypatch):
    class FakeTransfer:
        def __init__(self, tensor):
            self.tensor = tensor

        def bar(self, _):
            return self

    globals_ = SchedArgmaxSmemPartial.schedule.__globals__
    monkeypatch.setitem(globals_, "_shared_load_1d", FakeTransfer)
    monkeypatch.setitem(globals_, "_shared_store_1d", FakeTransfer)

    logits = torch.empty((129280,), dtype=torch.bfloat16)
    partials = torch.empty((152, 16), dtype=torch.uint8)
    partial = SchedArgmaxSmemPartial(logits, partials).place(152)
    first = partial.schedule(0)
    last = partial.schedule(151)

    assert isinstance(first[0], ArgmaxSmemPartialBf16)
    assert first[0].args == [856, 0, 0]
    assert isinstance(last[0], ArgmaxSmemPartialBf16)
    assert last[0].args == [848, 62896, 1]

    output = torch.empty((1,), dtype=torch.int64)
    reduce = SchedArgmaxSmemReduce(partials, output).place(1).schedule(0)
    assert isinstance(reduce[0], ArgmaxSmemReduceBf16)
    assert reduce[0].args == [152]


def test_deepseek_v4_flash_config_covers_transformer_and_mtp():
    config = DeepSeekV4FlashConfig()

    assert len(config.compress_ratios) == config.num_layers + 1
    assert [config.attention_kind(layer) for layer in range(6)] == [
        "swa", "swa", "csa", "hca", "csa", "hca"
    ]
    assert config.num_heads * config.head_dim == 32768
    assert config.num_experts == 256
    assert config.experts_per_token == 6


def test_decode_plan_covers_all_attention_families_and_model_stages():
    plan = build_decode_plan(127)

    assert len(plan) == 43
    assert sum(layer.attention_kind == "swa" for layer in plan) == 2
    assert sum(layer.attention_kind == "csa" for layer in plan) == 21
    assert sum(layer.attention_kind == "hca" for layer in plan) == 20
    assert sum(layer.hash_routing for layer in plan) == 3
    assert all(layer.should_compress for layer in plan if layer.compress_ratio)
    assert plan[0].attention_candidates == 128
    assert plan[2].attention_candidates == 160
    assert plan[3].attention_candidates == 129
    assert "index_topk" in plan[2].stages
    assert "index_topk" not in plan[3].stages
    assert not any(
        stage.endswith("cache_store")
        for layer in plan
        for stage in layer.stages
    )
    assert plan[0].stages[-4:] == (
        "routed_expert_nvfp4",
        "shared_expert_fp8",
        "expert_reduce",
        "hc_ffn_post",
    )


def test_checkpoint_rope_tables_cover_main_and_yarn_compressor_frequencies():
    identity = deepseek_v4_rope_table(0)
    main = deepseek_v4_rope_table(1)
    compressed = deepseek_v4_rope_table(1, compressed=True)

    assert identity.shape == (32, 2)
    torch.testing.assert_close(identity[:, 0], torch.ones(32))
    torch.testing.assert_close(identity[:, 1], torch.zeros(32))
    torch.testing.assert_close(main[0], torch.tensor([0.5403023, 0.8414710]))
    torch.testing.assert_close(
        compressed[-1], torch.tensor([1.0, 5.6805294e-7]), atol=1.0e-7, rtol=1.0e-5
    )
    assert not torch.equal(main, compressed)


def test_long_context_plan_caps_only_csa_compressed_selection():
    csa = build_layer_decode_plan(2, 4095)
    hca = build_layer_decode_plan(3, 4095)

    assert csa.compressed_rows == 1024
    assert csa.compressed_selected == 512
    assert csa.attention_candidates == 640
    assert hca.compressed_rows == 32
    assert hca.compressed_selected == 32
    assert hca.attention_candidates == 160


def test_early_csa_plan_uses_only_the_available_window():
    csa = build_layer_decode_plan(2, 0)

    assert csa.compressed_rows == 0
    assert csa.compressed_selected == 0
    assert csa.attention_candidates == 1
    assert not csa.should_compress
    assert "index_score" not in csa.stages
    assert "index_topk" not in csa.stages


def test_partial_rope_preserves_prefix_and_supports_inverse():
    source = torch.linspace(-1, 1, 2 * 512, dtype=torch.float32).reshape(2, 512)
    angles = torch.linspace(-0.7, 0.7, 32)
    table = torch.stack((angles.cos(), angles.sin()), dim=1)

    rotated = apply_partial_rope_512_64(source, table)
    restored = apply_partial_rope_512_64(rotated, table, inverse=True)

    torch.testing.assert_close(rotated[:, :-64], source[:, :-64], rtol=0, atol=0)
    torch.testing.assert_close(restored, source, rtol=1.0e-6, atol=1.0e-6)

    index_source = source[:, :128].clone()
    index_rotated = apply_partial_rope_128_64(index_source, table)
    index_restored = apply_partial_rope_128_64(
        index_rotated, table, inverse=True
    )
    torch.testing.assert_close(
        index_rotated[:, :-64], index_source[:, :-64], rtol=0, atol=0
    )
    torch.testing.assert_close(
        index_restored, index_source, rtol=1.0e-6, atol=1.0e-6
    )


def test_sparse_attention_sink_is_denominator_only():
    q = torch.ones((2, 512), dtype=torch.bfloat16)
    kv = torch.stack(
        (
            torch.ones(512, dtype=torch.bfloat16),
            -torch.ones(512, dtype=torch.bfloat16),
        )
    )
    indices = torch.tensor([0, 1, -1], dtype=torch.int32)
    no_sink = sparse_attention_512_reference(
        q, kv, indices, torch.full((2,), -100.0)
    )
    dominant_sink = sparse_attention_512_reference(
        q, kv, indices, torch.full((2,), 100.0)
    )

    assert dominant_sink.abs().max() < no_sink.abs().max()


def test_router_bias_changes_selection_but_not_selected_weight_values():
    logits = torch.linspace(-2, 2, 256, dtype=torch.bfloat16)
    bias = torch.zeros(256)
    bias[0] = 100.0
    weights, indices = route_top6_reference(logits, bias)

    assert indices[0].item() == 0
    scores = torch.nn.functional.softplus(logits.float()).sqrt()
    expected = scores[indices.long()]
    expected = expected / expected.sum() * 1.5
    torch.testing.assert_close(weights, expected)


def test_bounded_swiglu_clamps_gate_and_up_independently():
    gate = torch.tensor([-20.0, 20.0], dtype=torch.bfloat16)
    up = torch.tensor([-20.0, 20.0], dtype=torch.bfloat16)
    actual = bounded_swiglu(gate, up)
    expected = torch.nn.functional.silu(gate.float().clamp(max=10.0))
    expected *= up.float().clamp(-10.0, 10.0)

    torch.testing.assert_close(actual, expected.to(torch.bfloat16))


def test_hc_reference_shapes_and_sinkhorn_column_sums():
    generator = torch.Generator().manual_seed(7)
    residual = torch.randn((4, 4096), generator=generator).to(torch.bfloat16)
    mixes = torch.randn((24,), generator=generator)
    scale = torch.ones((3,))
    base = torch.zeros((24,))

    hidden, post, comb = hc_pre_reference(residual, mixes, scale, base)

    assert hidden.shape == (4096,)
    assert post.shape == (4,)
    assert comb.shape == (4, 4)
    torch.testing.assert_close(comb.sum(dim=0), torch.ones(4), rtol=2.0e-5, atol=2.0e-5)


def test_hc_post_consumes_the_transposed_sinkhorn_matrix():
    residual = torch.eye(4, dtype=torch.bfloat16)
    branch = torch.zeros(4, dtype=torch.bfloat16)
    post = torch.zeros(4)
    comb = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [8.0, 9.0, 10.0, 11.0],
            [12.0, 13.0, 14.0, 15.0],
        ]
    )

    actual = hc_post_reference(branch, residual, post, comb)

    torch.testing.assert_close(actual, comb.T.to(torch.bfloat16))


def test_hadamard_reference_is_normalized_and_self_inverse():
    source = torch.arange(128, dtype=torch.float32).reshape(1, 128) / 128
    transformed = hadamard_reference(source)
    restored = hadamard_reference(transformed)

    torch.testing.assert_close(transformed.square().sum(), source.square().sum())
    torch.testing.assert_close(restored, source, rtol=1.0e-5, atol=1.0e-5)


def test_gated_pool_and_index_score_references_follow_model_axes():
    values = torch.arange(24, dtype=torch.float32).reshape(3, 8)
    scores = torch.zeros_like(values)
    pooled = gated_pool_reference(values, scores)
    torch.testing.assert_close(pooled, values.mean(dim=0))

    q = torch.ones((64, 128), dtype=torch.bfloat16)
    kv = torch.stack(
        (torch.ones(128), -torch.ones(128), torch.zeros(128))
    ).to(torch.bfloat16)
    weights = torch.ones(64)
    index_scores = index_score_reference(q, kv, weights)
    torch.testing.assert_close(index_scores, torch.tensor([8192.0, 0.0, 0.0]))


def test_decode_cache_indices_cover_window_and_compressed_prefix():
    early = decode_window_indices(3)
    wrapped = decode_window_indices(130)
    compressed = decode_compressed_indices(511, 128)

    assert early[:4].tolist() == [0, 1, 2, 3]
    assert (early[4:] == -1).all()
    assert wrapped.tolist() == list(range(3, 128)) + [0, 1, 2]
    assert compressed.tolist() == [128, 129, 130, 131]


def test_hc_head_reference_reduces_four_streams():
    residual = torch.arange(4, dtype=torch.float32)[:, None].expand(4, 4096)
    mixes = torch.zeros(4)
    scale = torch.ones(1)
    base = torch.zeros(4)
    output = hc_head_reference(residual.to(torch.bfloat16), mixes, scale, base)

    expected = residual.sum(dim=0) * (0.5 + 1.0e-6)
    torch.testing.assert_close(output, expected.to(torch.bfloat16))


def test_linear_schedules_address_rows_above_uint16_limit(monkeypatch):
    class FakeTransfer:
        def __init__(self, tensor):
            self.tensor = tensor

        def bar(self, _):
            return self

    globals_ = SchedFp8Block128Gemv.schedule.__globals__
    monkeypatch.setitem(globals_, "_shared_load_1d", FakeTransfer)
    monkeypatch.setitem(globals_, "_shared_store_1d", FakeTransfer)
    rows, k, num_sms = 65544, 128, 65544
    weight = torch.empty((rows, k), dtype=torch.float8_e4m3fn)
    weight_scale = torch.empty(
        ((rows + 127) // 128, k // 128), dtype=torch.float8_e8m0fnu
    )
    activation = torch.empty((k,), dtype=torch.float8_e4m3fn)
    activation_scale = torch.empty((k // 128,), dtype=torch.float8_e8m0fnu)
    output = torch.empty((rows,), dtype=torch.bfloat16)
    schedule = SchedFp8Block128Gemv(
        weight, weight_scale, activation, activation_scale, output
    ).place(num_sms)

    sm = rows - 1
    rows_per_sm, extra = divmod(rows, num_sms)
    row_start = sm * rows_per_sm + min(sm, extra)
    row_count = rows_per_sm + (1 if sm < extra else 0)
    instructions = schedule.schedule(sm)
    assert row_start > 0xFFFF
    assert instructions[0].opcode == opcode.OP_FP8_BLOCK128_GEMV_SM100
    assert instructions[0].args == [1, k, row_start % 128]
    assert instructions[1].tensor.data_ptr() == weight[row_start].data_ptr()
    assert instructions[2].tensor.data_ptr() == weight_scale[row_start // 128].data_ptr()
    assert instructions[5].tensor.data_ptr() == output[row_start].data_ptr()

    fp32_weight = torch.empty((rows, 1), dtype=torch.float32)
    fp32_input = torch.empty((1,), dtype=torch.bfloat16)
    fp32_output = torch.empty((rows,), dtype=torch.float32)
    fp32_schedule = SchedDsv4Fp32Bf16Gemv(
        fp32_weight, fp32_input, fp32_output
    ).place(num_sms)
    fp32_instructions = fp32_schedule.schedule(sm)
    assert fp32_instructions[0].args == [1, 8192]
    assert fp32_instructions[1].tensor.data_ptr() == fp32_weight[row_start].data_ptr()
    assert fp32_instructions[3].tensor.data_ptr() == fp32_output[row_start].data_ptr()

    bf16_weight = torch.empty((rows, 1), dtype=torch.bfloat16)
    bf16_output = torch.empty((rows,), dtype=torch.bfloat16)
    bf16_schedule = SchedDsv4Bf16Gemv(
        bf16_weight, fp32_input, bf16_output
    ).place(num_sms)
    bf16_instructions = bf16_schedule.schedule(sm)
    assert bf16_instructions[0].args == [1, 16384, 0]
    assert bf16_instructions[1].tensor.data_ptr() == bf16_weight[row_start].data_ptr()
    assert bf16_instructions[3].tensor.data_ptr() == bf16_output[row_start].data_ptr()

    fp32_from_bf16 = torch.empty((rows,), dtype=torch.float32)
    fp32_from_bf16_schedule = SchedDsv4Bf16Gemv(
        bf16_weight, fp32_input, fp32_from_bf16
    ).place(num_sms)
    assert fp32_from_bf16_schedule.schedule(sm)[0].args == [1, 16384, 1]


def test_activation_quant_schedules_shard_whole_scale_blocks(monkeypatch):
    class FakeTransfer:
        def __init__(self, tensor):
            self.tensor = tensor

        def bar(self, _):
            return self

    globals_ = SchedDsv4Fp8Quant128.schedule.__globals__
    monkeypatch.setitem(globals_, "_shared_load_1d", FakeTransfer)
    monkeypatch.setitem(globals_, "_shared_store_1d", FakeTransfer)
    source = torch.empty((4096,), dtype=torch.bfloat16)

    fp8_output = torch.empty_like(source, dtype=torch.float8_e4m3fn)
    fp8_scale = torch.empty((32,), dtype=torch.float8_e8m0fnu)
    fp8_schedule = SchedDsv4Fp8Quant128(
        source, fp8_output, fp8_scale
    ).place(3)
    fp8_instructions = fp8_schedule.schedule(2)
    assert fp8_instructions[0].args == [10 * 128]
    assert fp8_instructions[1].tensor.data_ptr() == source[22 * 128:].data_ptr()
    assert fp8_instructions[2].tensor.data_ptr() == fp8_output[22 * 128:].data_ptr()
    assert fp8_instructions[3].tensor.data_ptr() == fp8_scale[22:].data_ptr()

    global_scale = torch.ones((1,), dtype=torch.float32)
    fp4_output = torch.empty((2048,), dtype=torch.uint8)
    fp4_scale = torch.empty((256,), dtype=torch.float8_e4m3fn)
    fp4_schedule = SchedDsv4Nvfp4Quant16(
        source, global_scale, fp4_output, fp4_scale
    ).place(3)
    fp4_instructions = fp4_schedule.schedule(2)
    assert fp4_instructions[0].args == [85 * 16]
    assert fp4_instructions[1].tensor.data_ptr() == source[171 * 16:].data_ptr()
    assert fp4_instructions[3].tensor.data_ptr() == fp4_output[171 * 8:].data_ptr()
    assert fp4_instructions[4].tensor.data_ptr() == fp4_scale[171:].data_ptr()
