from collections import defaultdict
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dae.util import tensor_diff


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


def permute_rope_activation(activation: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    return (
        activation.view(*activation.shape[:-1], num_heads, 2, head_dim // 2)
        .transpose(-2, -1)
        .reshape(*activation.shape[:-1], num_heads * head_dim)
        .contiguous()
    )


def permute_rope_head_weight(weight: torch.Tensor) -> torch.Tensor:
    head_dim = weight.shape[-1]
    return (
        weight.view(2, head_dim // 2)
        .transpose(0, 1)
        .reshape_as(weight)
        .contiguous()
    )


def apply_rms_affine(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    hidden_states_f32 = hidden_states.float()
    variance = hidden_states_f32.pow(2).mean(dim=-1, keepdim=True)
    hidden_states_f32 = hidden_states_f32 * torch.rsqrt(variance + eps)
    return (hidden_states_f32 * weight.float()).to(hidden_states.dtype)


def build_interleaved_rope(position_ids: torch.Tensor, head_dim: int, rope_theta: float, device, dtype):
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.einsum("bs,d->bsd", position_ids.float(), inv_freq)
    rope = torch.empty(*freqs.shape[:-1], head_dim, device=device, dtype=dtype)
    rope[..., 0::2] = freqs.cos().to(dtype=dtype)
    rope[..., 1::2] = freqs.sin().to(dtype=dtype)
    return rope


def apply_interleaved_rope(hidden_states: torch.Tensor, rope: torch.Tensor) -> torch.Tensor:
    hidden_states_f32 = hidden_states.float()
    cos = rope[..., 0::2].float()
    sin = rope[..., 1::2].float()
    even = hidden_states_f32[..., 0::2]
    odd = hidden_states_f32[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos),
        dim=-1,
    ).flatten(-2)
    return rotated.to(hidden_states.dtype)


def reference_pass(model, inputs, rope_theta: float, verbose: bool = False):
    captured_data = defaultdict(dict)

    def register_qwen_hooks(model):
        handles = []

        def get_pre_hook(layer_idx, name):
            def hook(module, hook_inputs):
                captured_data[layer_idx][name] = hook_inputs[0].detach()

            return hook

        def get_hook(layer_idx, name):
            def hook(module, hook_inputs, output):
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
            handles.append(attn.o_proj.register_forward_hook(get_hook(i, "o_proj")))
            handles.append(attn.register_forward_hook(get_hook(i, "attn_block_out")))

            mlp = layer.mlp
            handles.append(mlp.gate.register_forward_hook(get_hook(i, "router_logits")))
            handles.append(mlp.register_forward_hook(get_hook(i, "mlp_out")))
            handles.append(layer.register_forward_hook(get_hook(i, "hidden_state_out")))

        handles.append(model.model.norm.register_forward_hook(get_hook("final", "final_rms")))
        handles.append(model.lm_head.register_forward_hook(get_hook("final", "lm_head")))
        return handles

    hooks = register_qwen_hooks(model)
    with torch.no_grad():
        output = model(**inputs, use_cache=False)
        if verbose:
            print(output)
    for hook in hooks:
        hook.remove()

    position_ids = inputs.get("position_ids")
    if position_ids is None:
        position_ids = torch.zeros((1, inputs["input_ids"].shape[1]), device=inputs["input_ids"].device, dtype=torch.long)

    rope = build_interleaved_rope(
        position_ids=position_ids,
        head_dim=getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
        rope_theta=rope_theta,
        device=position_ids.device,
        dtype=model.dtype,
    )

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        q_heads = attn.num_key_value_groups * model.config.num_key_value_heads
        kv_heads = model.config.num_key_value_heads
        head_dim = attn.head_dim
        q_norm_weight = permute_rope_head_weight(attn.q_norm.weight.detach())
        k_norm_weight = permute_rope_head_weight(attn.k_norm.weight.detach())

        q_proj = permute_rope_activation(captured_data[i]["q_proj"], q_heads, head_dim)
        k_proj = permute_rope_activation(captured_data[i]["k_proj"], kv_heads, head_dim)

        q_norm = apply_rms_affine(
            q_proj.view(*q_proj.shape[:-1], q_heads, head_dim),
            q_norm_weight.view(1, 1, 1, head_dim),
            model.config.rms_norm_eps,
        )
        k_norm = apply_rms_affine(
            k_proj.view(*k_proj.shape[:-1], kv_heads, head_dim),
            k_norm_weight.view(1, 1, 1, head_dim),
            model.config.rms_norm_eps,
        )

        captured_data[i]["q_proj_interleaved"] = q_proj
        captured_data[i]["k_proj_interleaved"] = k_proj
        captured_data[i]["q_norm_interleaved"] = q_norm.reshape_as(q_proj)
        captured_data[i]["k_norm_interleaved"] = k_norm.reshape_as(k_proj)
        captured_data[i]["q_rope_interleaved"] = apply_interleaved_rope(
            q_norm,
            rope[:, :, None, :],
        ).reshape_as(q_proj)
        captured_data[i]["k_rope_interleaved"] = apply_interleaved_rope(
            k_norm,
            rope[:, :, None, :],
        ).reshape_as(k_proj)

        router_logits = captured_data[i]["router_logits"].float()
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_weight, topk_idx = torch.topk(router_probs, model.config.num_experts_per_tok, dim=-1)
        if getattr(model.config, "norm_topk_prob", False):
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
        captured_data[i]["router_topk_idx"] = topk_idx.detach()
        captured_data[i]["router_topk_weight"] = topk_weight.to(dtype=model.dtype).detach()

    return captured_data, output


def reference_pass_local(ctx, inputs, verbose: bool = False):
    captured_data = defaultdict(dict)
    token_index = 0
    hidden = ctx.matEmbed[inputs["input_ids"][0, token_index]].unsqueeze(0).to(dtype=ctx.dtype)
    rope_row = ctx.matRope[inputs["position_ids"][0, token_index]].view(1, 1, ctx.HEAD_DIM)

    for i in range(ctx.num_layers):
        layer_in = hidden
        rms_in_w = ctx.matRMSInputW0 if i == 0 else ctx.matRMSInputWLoop[i - 1]
        input_rms = apply_rms_affine(layer_in, rms_in_w, ctx.eps)
        q_proj = F.linear(input_rms, ctx.matqWs[i])
        k_proj = F.linear(input_rms, ctx.matkWs[i])
        v_proj = F.linear(input_rms, ctx.matvWs[i])

        q_heads = q_proj.view(1, 1, ctx.NUM_Q_HEAD, ctx.HEAD_DIM)
        k_heads = k_proj.view(1, 1, ctx.NUM_KV_HEAD, ctx.HEAD_DIM)
        q_norm = apply_rms_affine(q_heads, ctx.matQNormWs[i].view(1, 1, 1, ctx.HEAD_DIM), ctx.eps)
        k_norm = apply_rms_affine(k_heads, ctx.matKNormWs[i].view(1, 1, 1, ctx.HEAD_DIM), ctx.eps)
        q_rope = apply_interleaved_rope(q_norm, rope_row)
        k_rope = apply_interleaved_rope(k_norm, rope_row)

        v_heads = v_proj.view(1, ctx.NUM_KV_HEAD, ctx.HEAD_DIM)
        attn_out = (
            v_heads[:, :, None, :]
            .expand(1, ctx.NUM_KV_HEAD, ctx.HEAD_GROUP_SIZE, ctx.HEAD_DIM)
            .reshape(1, ctx.QW)
        )
        attn_block_out = F.linear(attn_out, ctx.matOutWs[i])
        post_attn_residual = layer_in + attn_block_out
        post_attn_rms = apply_rms_affine(post_attn_residual, ctx.matRMSPostAttnW[i], ctx.eps)

        router_logits = F.linear(post_attn_rms, ctx.matRouterWs[i])
        router_probs = torch.softmax(router_logits.float(), dim=-1)
        topk_weight, topk_idx = torch.topk(router_probs, ctx.TOP_K, dim=-1)
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        mlp_out = torch.zeros_like(post_attn_residual)
        for slot in range(ctx.TOP_K):
            expert = topk_idx[0, slot].item()
            gate = F.linear(post_attn_rms, ctx.matExpertGateWs[i, expert])
            up = F.linear(post_attn_rms, ctx.matExpertUpWs[i, expert])
            act = F.silu(gate) * up
            down = F.linear(act, ctx.matExpertDownWs[i, expert])
            mlp_out = mlp_out + down * topk_weight[0, slot].to(dtype=ctx.dtype)

        hidden = post_attn_residual + mlp_out

        captured_data[i]["hidden_state_in"] = layer_in.unsqueeze(1)
        captured_data[i]["input_rms"] = input_rms.unsqueeze(1)
        captured_data[i]["q_proj"] = q_proj.unsqueeze(1)
        captured_data[i]["k_proj"] = k_proj.unsqueeze(1)
        captured_data[i]["v_proj"] = v_proj.unsqueeze(1)
        captured_data[i]["q_proj_interleaved"] = q_proj.unsqueeze(1)
        captured_data[i]["k_proj_interleaved"] = k_proj.unsqueeze(1)
        captured_data[i]["q_norm_interleaved"] = q_norm.reshape(1, 1, -1)
        captured_data[i]["k_norm_interleaved"] = k_norm.reshape(1, 1, -1)
        captured_data[i]["q_rope_interleaved"] = q_rope.reshape(1, 1, -1)
        captured_data[i]["k_rope_interleaved"] = k_rope.reshape(1, 1, -1)
        captured_data[i]["attn_block_out"] = attn_block_out.unsqueeze(1)
        captured_data[i]["post_attn_residual"] = post_attn_residual.unsqueeze(1)
        captured_data[i]["post_attn_rms"] = post_attn_rms.unsqueeze(1)
        captured_data[i]["router_logits"] = router_logits.unsqueeze(1)
        captured_data[i]["router_topk_idx"] = topk_idx.unsqueeze(1)
        captured_data[i]["router_topk_weight"] = topk_weight.to(dtype=ctx.dtype).unsqueeze(1)
        captured_data[i]["mlp_out"] = mlp_out.unsqueeze(1)
        captured_data[i]["hidden_state_out"] = hidden.unsqueeze(1)

    final_rms_w = ctx.matRMSInputWLoop[ctx.num_layers - 1]
    final_rms = apply_rms_affine(hidden, final_rms_w, ctx.eps)
    lm_head = F.linear(final_rms, ctx.matLmHeadWFull)
    captured_data["final"]["final_rms"] = final_rms.unsqueeze(1)
    captured_data["final"]["lm_head"] = lm_head.unsqueeze(1)
    return captured_data, SimpleNamespace(logits=lm_head.unsqueeze(1))


def input_batch1(*tokens, mat=None, positions=None):
    seq_len = len(tokens)
    if mat is not None:
        for i in range(seq_len):
            mat[i] = tokens[i]
    if positions is not None:
        assert len(positions) == seq_len
        position_ids = torch.tensor(positions, dtype=torch.long, device="cuda").unsqueeze(0)
    else:
        position_ids = torch.zeros((1, seq_len), dtype=torch.long, device="cuda")

    return {
        "input_ids": torch.tensor([tokens], device="cuda"),
        "attention_mask": torch.ones((1, seq_len), dtype=torch.long, device="cuda"),
        "position_ids": position_ids,
    }
