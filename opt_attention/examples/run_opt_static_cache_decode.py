import math

import torch
from transformers import OPTConfig, StaticCache
from transformers.models.opt.modeling_opt import OPTAttention, eager_attention_forward

import opt_attention


def main():
    device = torch.device("cuda")
    dtype = torch.float16
    batch, heads, seq = 512, 32, 64
    config = OPTConfig(hidden_size=heads * 128, num_attention_heads=heads, num_hidden_layers=1)
    module = OPTAttention(config, layer_idx=0).to(device=device, dtype=dtype)
    module.eval()

    cache = StaticCache(config=config, max_cache_len=seq)
    key = torch.randn(batch, heads, 1, 128, device=device, dtype=dtype)
    value = torch.randn(batch, heads, 1, 128, device=device, dtype=dtype)
    key_cache, value_cache = cache.update(key, value, 0, {"cache_position": torch.tensor([0], device=device)})

    query = torch.randn(batch, heads, 1, 128, device=device, dtype=dtype) / math.sqrt(128.0)
    mask = torch.full((batch, 1, 1, seq), torch.finfo(torch.float32).min, device=device)
    mask[..., 0] = 0.0

    opt_attention.register()
    got, _ = opt_attention.opt_attention_forward(module, query, key_cache, value_cache, mask, scaling=1.0)
    ref, _ = eager_attention_forward(module, query, key_cache, value_cache, mask, dropout=0.0, scaling=1.0)
    print(torch.max(torch.abs(got - ref)).item())


if __name__ == "__main__":
    main()
