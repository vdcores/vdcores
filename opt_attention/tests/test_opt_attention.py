import math
import os

import pytest
import torch
from transformers import OPTConfig, StaticCache
from transformers.models.opt.modeling_opt import OPTAttention, eager_attention_forward

import opt_attention


def _make_module(hidden_size: int, num_heads: int, device: torch.device):
    config = OPTConfig(hidden_size=hidden_size, num_attention_heads=num_heads, num_hidden_layers=1)
    module = OPTAttention(config, layer_idx=0).to(device=device, dtype=torch.float16)
    module.eval()
    return module


def _compare_decode(batch: int, heads: int, seq: int, dtype: torch.dtype, masked: bool):
    device = torch.device("cuda")
    torch.manual_seed(1234)
    query = torch.randn(batch, heads, 1, 128, device=device, dtype=dtype) / math.sqrt(128.0)
    key = torch.randn(batch, heads, seq, 128, device=device, dtype=dtype)
    value = torch.randn(batch, heads, seq, 128, device=device, dtype=dtype)
    mask = None
    if masked:
        mask = torch.zeros(batch, 1, 1, seq, device=device, dtype=torch.float32)
        mask[..., seq // 2 :] = torch.finfo(torch.float32).min

    module = _make_module(heads * 128, heads, device)
    got, got_weights = opt_attention.opt_attention_forward(
        module, query, key, value, mask, dropout=0.0, scaling=1.0
    )
    ref, _ = eager_attention_forward(module, query, key, value, mask, dropout=0.0, scaling=1.0)

    assert got_weights is None
    torch.testing.assert_close(got, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("masked", [False, True])
def test_small_decode(dtype, masked):
    _compare_decode(batch=2, heads=4, seq=64, dtype=dtype, masked=masked)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_static_cache_integration():
    device = torch.device("cuda")
    dtype = torch.float16
    batch, heads, seq = 2, 4, 64
    config = OPTConfig(hidden_size=heads * 128, num_attention_heads=heads, num_hidden_layers=1)
    module = OPTAttention(config, layer_idx=0).to(device=device, dtype=dtype)
    module.eval()
    cache = StaticCache(config=config, max_cache_len=seq)

    key_token = torch.randn(batch, heads, 1, 128, device=device, dtype=dtype)
    value_token = torch.randn(batch, heads, 1, 128, device=device, dtype=dtype)
    cache_position = torch.tensor([0], device=device)
    key, value = cache.update(key_token, value_token, 0, {"cache_position": cache_position})
    query = torch.randn(batch, heads, 1, 128, device=device, dtype=dtype) / math.sqrt(128.0)
    mask = torch.full((batch, 1, 1, seq), torch.finfo(torch.float32).min, device=device)
    mask[..., 0] = 0.0

    opt_attention.register()
    got, _ = opt_attention.opt_attention_forward(module, query, key, value, mask, dropout=0.0, scaling=1.0)
    ref, _ = eager_attention_forward(module, query, key, value, mask, dropout=0.0, scaling=1.0)
    torch.testing.assert_close(got, ref, rtol=4e-2, atol=4e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(os.environ.get("RUN_LARGE_OPT_ATTENTION") != "1", reason="large shape test is opt-in")
@pytest.mark.parametrize("heads", [32, 56])
def test_large_opt_shapes(heads):
    _compare_decode(batch=512, heads=heads, seq=64, dtype=torch.float16, masked=True)
