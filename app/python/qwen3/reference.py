import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM

from dae.util import tensor_diff


DEFAULT_MODEL_NAME = "Qwen/Qwen3-8B"


def mean_relative_diff_pct(t1: torch.Tensor, t2: torch.Tensor, ref: torch.Tensor | None = None):
    if ref is None:
        ref = t1
    denom = ref.abs().float().mean().item()
    if denom == 0:
        denom = 1.0
    return (t1 - t2).abs().float().mean().item() / denom * 100.0


def check_tensor_threshold(
    name: str,
    expected: torch.Tensor,
    actual: torch.Tensor,
    threshold_pct: float,
    ref: torch.Tensor | None = None,
):
    diff = mean_relative_diff_pct(expected, actual, ref=ref)
    tensor_diff(name, expected, actual, ref=ref)
    passed = diff <= threshold_pct
    status = "PASS" if passed else "FAIL"
    print(f"[correctness] {status} {name}: {diff:.3f}% <= {threshold_pct:.3f}%")
    return passed, diff


def reference_pass(
    model,
    inputs,
    verbose=False,
    *,
    use_cache=False,
    past_key_values=None,
    cache_position=None,
):
    captured_data = defaultdict(dict)

    def register_qwen_hooks(model):
        handles = []

        def get_pre_hook(layer_idx, name):
            def hook(module, inputs):
                captured_data[layer_idx][name] = inputs[0].detach()
            return hook

        def get_hook(layer_idx, name):
            def hook(module, inputs, output):
                data = output[0] if isinstance(output, tuple) else output
                captured_data[layer_idx][name] = data.detach()
            return hook

        for i, layer in enumerate(model.model.layers):
            handles.append(layer.register_forward_pre_hook(get_pre_hook(i, "hidden_state_in")))
            handles.append(layer.input_layernorm.register_forward_hook(get_hook(i, "input_rms")))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(get_pre_hook(i, "post_attn_residual")))
            handles.append(layer.post_attention_layernorm.register_forward_hook(get_hook(i, "post_attn_rms")))

            attn = layer.self_attn
            handles.append(attn.q_proj.register_forward_hook(get_hook(i, "q_proj")))
            handles.append(attn.k_proj.register_forward_hook(get_hook(i, "k_proj")))
            handles.append(attn.v_proj.register_forward_hook(get_hook(i, "v_proj")))
            handles.append(attn.q_norm.register_forward_hook(get_hook(i, "q_norm")))
            handles.append(attn.k_norm.register_forward_hook(get_hook(i, "k_norm")))
            handles.append(attn.o_proj.register_forward_pre_hook(get_pre_hook(i, "attn_context")))
            handles.append(attn.o_proj.register_forward_hook(get_hook(i, "o_proj")))
            handles.append(attn.register_forward_hook(get_hook(i, "attn_block_out")))

            mlp = layer.mlp
            handles.append(mlp.gate_proj.register_forward_hook(get_hook(i, "gate_proj")))
            handles.append(mlp.up_proj.register_forward_hook(get_hook(i, "up_proj")))
            handles.append(mlp.down_proj.register_forward_hook(get_hook(i, "down_proj")))
            handles.append(layer.register_forward_hook(get_hook(i, "hidden_state_out")))

        handles.append(model.model.norm.register_forward_pre_hook(get_pre_hook("final", "final_rms_in")))
        handles.append(model.model.norm.register_forward_hook(get_hook("final", "final_rms")))
        handles.append(model.lm_head.register_forward_hook(get_hook("final", "lm_head")))
        return handles

    hooks = register_qwen_hooks(model)
    with torch.no_grad():
        output = model(
            **inputs,
            use_cache=use_cache,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        if verbose:
            print(output)

    for handle in hooks:
        handle.remove()

    return captured_data, output


def cached_reference_pass(
    model,
    prefix_token_ids,
    current_token_id: int,
    *,
    prefix_position: int = 0,
    current_position: int | None = None,
):
    if isinstance(prefix_token_ids, int):
        prefix_token_ids = [prefix_token_ids]
    prefix_token_ids = list(prefix_token_ids)
    if len(prefix_token_ids) == 0:
        raise ValueError("cached_reference_pass requires at least one prefix token")
    if current_position is None:
        current_position = prefix_position + len(prefix_token_ids)

    prefix_inputs = input_batch1(
        *prefix_token_ids,
        positions=list(range(prefix_position, prefix_position + len(prefix_token_ids))),
        attention_len=prefix_position + len(prefix_token_ids),
    )
    prefix_captured, prefix_output = reference_pass(
        model,
        prefix_inputs,
        use_cache=True,
        cache_position=torch.arange(prefix_position, prefix_position + len(prefix_token_ids), device="cuda"),
    )

    current_inputs = input_batch1(
        current_token_id,
        positions=[current_position],
        attention_len=current_position + 1,
    )
    current_captured, current_output = reference_pass(
        model,
        current_inputs,
        use_cache=True,
        past_key_values=prefix_output.past_key_values,
        cache_position=torch.arange(current_position, current_position + 1, device="cuda"),
    )

    return {
        "prefix_captured": prefix_captured,
        "prefix_output": prefix_output,
        "current_captured": current_captured,
        "current_output": current_output,
    }


def input_batch1(*tokens, mat=None, positions=None, attention_len=None):
    seq_len = len(tokens)
    if mat is not None:
        for i in range(seq_len):
            mat[i] = tokens[i]
    if positions is not None:
        assert len(positions) == seq_len
        position_ids = torch.tensor(positions, dtype=torch.long, device="cuda").unsqueeze(0)
    else:
        position_ids = torch.zeros((1, seq_len), dtype=torch.long, device="cuda")
    if attention_len is None:
        attention_len = seq_len

    return {
        "input_ids": torch.tensor([tokens], device="cuda"),
        "attention_mask": torch.ones((1, attention_len), dtype=torch.long, device="cuda"),
        "position_ids": position_ids,
    }


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_NAME,
        cache_dir="/tmp/huggingface_cache",
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ["HF_TOKEN"],
    )
    captured_data, _ = reference_pass(model, input_batch1(791))
    print(f"captured {len(captured_data)} hook groups")
