import math
import os
from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR = "/tmp/huggingface_cache"


@dataclass
class LayerAttentionTensors:
    query_states: torch.Tensor
    key_states: torch.Tensor
    value_states: torch.Tensor
    o_proj_weight: torch.Tensor
    o_proj_bias: torch.Tensor | None
    num_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_dropout: float
    rope_theta: float


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rotary_pos_emb(
    qk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    qk_embed = (qk * cos) + (rotate_half(qk) * sin)
    return qk_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    expanded = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return expanded.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def build_rope_cache(
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def attention_reference(
    tensors: LayerAttentionTensors,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    k_position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    attn_output = attention_reference_pre_o_proj(
        tensors=tensors,
        position_ids=position_ids,
        attention_mask=attention_mask,
        k_position_ids=k_position_ids,
    )
    return out_proj_reference(
        attn_output,
        o_proj_weight=tensors.o_proj_weight,
        o_proj_bias=tensors.o_proj_bias,
    )


def attention_reference_pre_o_proj(
    tensors: LayerAttentionTensors,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    k_position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size, seq_len, _ = tensors.query_states.shape
    query_states = tensors.query_states
    key_states = tensors.key_states
    value_states = tensors.value_states
    query_states = query_states.view(batch_size, seq_len, tensors.num_heads, tensors.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, tensors.num_key_value_heads, tensors.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_len, tensors.num_key_value_heads, tensors.head_dim).transpose(1, 2)

    cos, sin = build_rope_cache(
        position_ids=position_ids,
        head_dim=tensors.head_dim,
        rope_theta=tensors.rope_theta,
        device=query_states.device,
        dtype=query_states.dtype,
    )
    query_states = apply_rotary_pos_emb(query_states, cos, sin)
    if k_position_ids is not None:
        cos, sin = build_rope_cache(
            position_ids=k_position_ids,
            head_dim=tensors.head_dim,
            rope_theta=tensors.rope_theta,
            device=key_states.device,
            dtype=key_states.dtype,
        )
    key_states = apply_rotary_pos_emb(key_states, cos, sin)

    num_key_value_groups = tensors.num_heads // tensors.num_key_value_heads
    key_states = repeat_kv(key_states, num_key_value_groups)
    value_states = repeat_kv(value_states, num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(tensors.head_dim)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

def attention_ref_rope_embed(
    tensors: LayerAttentionTensors,
    position_ids: torch.Tensor,
    k_position_ids: torch.Tensor | None = None,
):
    batch_size, seq_len, _ = tensors.query_states.shape
    query_states = tensors.query_states
    key_states = tensors.key_states
    value_states = tensors.value_states
    query_states = query_states.view(batch_size, seq_len, tensors.num_heads, tensors.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_len, tensors.num_key_value_heads, tensors.head_dim).transpose(1, 2)

    cos, sin = build_rope_cache(
        position_ids=position_ids,
        head_dim=tensors.head_dim,
        rope_theta=tensors.rope_theta,
        device=query_states.device,
        dtype=query_states.dtype,
    )
    query_states = apply_rotary_pos_emb(query_states, cos, sin)
    if k_position_ids is not None:
        cos, sin = build_rope_cache(
            position_ids=k_position_ids,
            head_dim=tensors.head_dim,
            rope_theta=tensors.rope_theta,
            device=key_states.device,
            dtype=key_states.dtype,
        )
    key_states = apply_rotary_pos_emb(key_states, cos, sin)
    return query_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1), key_states.transpose(1, 2).contiguous().view(batch_size, seq_len, -1), cos, sin


def out_proj_reference(
    attn_output: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    projected = torch.matmul(attn_output, o_proj_weight.transpose(0, 1))
    if o_proj_bias is not None:
        projected = projected + o_proj_bias
    return projected


def load_model_and_config():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    config = AutoConfig.from_pretrained(MODEL_NAME)
    return model, config
