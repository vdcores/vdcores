import torch

from dae.deepseek_v4 import (
    DeepSeekV4FlashConfig,
    apply_partial_rope_512_64,
    bounded_swiglu,
    decode_compressed_indices,
    decode_window_indices,
    gated_pool_reference,
    hadamard_reference,
    hc_head_reference,
    hc_pre_reference,
    index_score_reference,
    route_top6_reference,
    sparse_attention_512_reference,
)
from dae.runtime import opcode
from dae.schedule import SchedDsv4Fp32Bf16Gemv, SchedFp8Block128Gemv


def test_deepseek_v4_flash_config_covers_transformer_and_mtp():
    config = DeepSeekV4FlashConfig()

    assert len(config.compress_ratios) == config.num_layers + 1
    assert [config.attention_kind(layer) for layer in range(6)] == [
        "swa", "swa", "csa", "hca", "csa", "hca"
    ]
    assert config.num_heads * config.head_dim == 32768
    assert config.num_experts == 256
    assert config.experts_per_token == 6


def test_partial_rope_preserves_prefix_and_supports_inverse():
    source = torch.linspace(-1, 1, 2 * 512, dtype=torch.float32).reshape(2, 512)
    angles = torch.linspace(-0.7, 0.7, 32)
    table = torch.stack((angles.cos(), angles.sin()), dim=1)

    rotated = apply_partial_rope_512_64(source, table)
    restored = apply_partial_rope_512_64(rotated, table, inverse=True)

    torch.testing.assert_close(rotated[:, :-64], source[:, :-64], rtol=0, atol=0)
    torch.testing.assert_close(restored, source, rtol=1.0e-6, atol=1.0e-6)


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
    class FakeRawAddress:
        def __init__(self, tensor, slot):
            self.tensor = tensor
            self.slot = slot

        def bar(self, _):
            return self

        def writeback(self):
            return self

    monkeypatch.setitem(
        SchedFp8Block128Gemv.schedule.__globals__,
        "RawAddress",
        FakeRawAddress,
    )
    rows, k, num_sms = 129280, 128, 3
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

    sm = 2
    rows_per_sm, extra = divmod(rows, num_sms)
    row_start = sm * rows_per_sm + min(sm, extra)
    row_count = rows_per_sm + (1 if sm < extra else 0)
    instructions = schedule.schedule(sm)
    assert row_start > 0xFFFF
    assert instructions[0].opcode == opcode.OP_FP8_BLOCK128_GEMV_SM100
    assert instructions[0].args == [row_count, k, row_start % 128]
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
    assert fp32_instructions[0].args == [row_count, 1]
    assert fp32_instructions[1].tensor.data_ptr() == fp32_weight[row_start].data_ptr()
    assert fp32_instructions[3].tensor.data_ptr() == fp32_output[row_start].data_ptr()
