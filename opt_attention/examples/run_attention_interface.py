import math

import torch
from transformers import OPTConfig, StaticCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.opt.modeling_opt import OPTAttention, eager_attention_forward

import opt_attention


def main():
    device = torch.device("cuda")
    dtype = torch.float16
    batch, heads, seq_len, head_dim = 2, 4, 64, 128

    opt_attention.register()

    config = OPTConfig(
        hidden_size=heads * head_dim,
        num_attention_heads=heads,
        num_hidden_layers=1,
    )
    config._attn_implementation = "vdcores_opt"
    module = OPTAttention(config, layer_idx=0).to(device=device, dtype=dtype)
    module.eval()

    query_states = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype) / math.sqrt(head_dim)
    next_key = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)
    next_value = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype)

    static_cache = StaticCache(config=config, max_cache_len=seq_len)
    key_states, value_states = static_cache.update(
        next_key,
        next_value,
        layer_idx=0,
        cache_kwargs={"cache_position": torch.tensor([0], device=device)},
    )

    attention_mask = torch.full(
        (batch, 1, 1, seq_len),
        torch.finfo(torch.float32).min,
        device=device,
        dtype=torch.float32,
    )
    attention_mask[..., 0] = 0.0

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        module.config._attn_implementation,
        eager_attention_forward,
    )
    attn_output, attn_weights = attention_interface(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=1.0,
    )

    ref_output, _ = eager_attention_forward(
        module,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0,
        scaling=1.0,
    )
    torch.cuda.synchronize()

    print("implementation:", module.config._attn_implementation)
    print("output:", tuple(attn_output.shape), attn_output.dtype)
    print("weights:", attn_weights)
    print("max_error:", torch.max(torch.abs(attn_output - ref_output)).item())


if __name__ == "__main__":
    main()
