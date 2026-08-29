from types import SimpleNamespace

import torch

from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.deepseek_v4_live import DeepSeekV4LiveDecodeState


def _compressor(rows: int, width: int, offset: float):
    values = torch.arange(rows * 2 * width, dtype=torch.float32).reshape(
        1, rows, 2 * width
    )
    values += offset
    return SimpleNamespace(kv_state=values, score_state=values + 10_000)


def test_import_pytorch_prefill_maps_caches_and_incremental_pools():
    config = DeepSeekV4FlashConfig()
    prefix = 5
    state = DeepSeekV4LiveDecodeState(16, device="cpu", config=config)
    layers = []
    for layer_id in range(config.num_layers):
        kind = config.attention_kind(layer_id)
        ratio = config.compress_ratios[layer_id]
        compressed = prefix // ratio if ratio else 0
        cache = torch.full(
            (1, config.sliding_window + compressed, config.head_dim),
            float(layer_id + 1),
            dtype=torch.bfloat16,
        )
        attention = SimpleNamespace(kv_cache=cache)
        if kind == "csa":
            attention.compressor = _compressor(
                8, config.head_dim, layer_id * 100_000
            )
            attention.indexer = SimpleNamespace(
                kv_cache=torch.full(
                    (1, compressed, config.index_head_dim),
                    float(layer_id + 101),
                    dtype=torch.bfloat16,
                ),
                compressor=_compressor(
                    8, config.index_head_dim, layer_id * 200_000
                ),
            )
        elif kind == "hca":
            attention.compressor = _compressor(
                128, config.head_dim, layer_id * 100_000
            )
        layers.append(SimpleNamespace(attn=attention))

    state.import_pytorch_prefill(SimpleNamespace(layers=layers), prefix)

    for layer_id in range(config.num_layers):
        kind = config.attention_kind(layer_id)
        ratio = config.compress_ratios[layer_id]
        compressed = prefix // ratio if ratio else 0
        cache = state.attention_cache(layer_id, compressed)
        assert torch.all(cache[:prefix] == layer_id + 1)
        if compressed:
            assert torch.all(
                cache[config.sliding_window :] == layer_id + 1
            )

    csa_layer = state.layers_by_kind["csa"][0]
    _, csa_offset = state._offset(csa_layer, "csa")
    csa_source = layers[csa_layer].attn.compressor
    # Five prefetched tokens mean group one: rows 0..3 are the preceding
    # overlap half, row 4 is the current ordinary half, and the current first
    # half has already seeded row zero of the opposite bank.
    assert torch.equal(
        state.csa_pool_values[csa_offset, 1, :4],
        csa_source.kv_state[0, :4, : config.head_dim],
    )
    assert torch.equal(
        state.csa_pool_values[csa_offset, 1, 4],
        csa_source.kv_state[0, 4, config.head_dim :],
    )
    assert torch.equal(
        state.csa_pool_values[csa_offset, 0, 0],
        csa_source.kv_state[0, 4, : config.head_dim],
    )
    assert torch.all(
        state.index_cache(csa_layer, 1) == csa_layer + 101
    )

    hca_layer = state.layers_by_kind["hca"][0]
    _, hca_offset = state._offset(hca_layer, "hca")
    hca_source = layers[hca_layer].attn.compressor
    assert torch.equal(
        state.hca_pool_values[hca_offset, :prefix],
        hca_source.kv_state[0, :prefix, : config.head_dim],
    )
