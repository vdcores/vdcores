import torch

from reference import check_tensor_threshold, input_batch1, reference_pass, reference_pass_local
from runtime_context import QwenScheduleContext, apply_rms_affine_rope_heads


def run_correctness_check(ctx: QwenScheduleContext):
    final_hidden_threshold = 15.0
    final_rms_threshold = 12.0
    router_weight_threshold = 5.0

    print("[correctness] running single-token position-0 reference capture...")
    token_index = 0
    token_pos = ctx.input_token_id_and_pos[0][1]
    inputs = input_batch1(
        *(token for token, _ in ctx.input_token_id_and_pos),
        mat=ctx.matTokens[0],
        positions=[pos for _, pos in ctx.input_token_id_and_pos],
    )

    if ctx.use_local_generated_weights:
        captured, _ = reference_pass_local(ctx, inputs)
    else:
        captured, _ = reference_pass(ctx.model, inputs, rope_theta=ctx.rope_theta)
    rope_row = ctx.matRope[token_pos]
    all_ok = True

    layer = captured[ctx.num_layers - 1]
    dae_q_rope = apply_rms_affine_rope_heads(
        ctx.attnQs[0].view(ctx.NUM_Q_HEAD, ctx.HEAD_DIM),
        ctx.matQNormWs[-1],
        rope_row,
        ctx.eps,
    ).reshape(-1)
    print(f"[correctness] Checking Layer {ctx.num_layers - 1}:")
    final_checks = [
        check_tensor_threshold("v_proj", layer["v_proj"][0, token_index], ctx.attnVs[0, token_pos], 5.0),
        check_tensor_threshold("q_proj_interleaved", layer["q_proj_interleaved"][0, token_index], ctx.attnQs[0], 5.0),
        check_tensor_threshold("q_rope_interleaved", layer["q_rope_interleaved"][0, token_index], dae_q_rope, 5.0),
        check_tensor_threshold("k_rope_interleaved", layer["k_rope_interleaved"][0, token_index], ctx.attnKs[0, token_pos], 5.0),
        check_tensor_threshold(
            "router_topk_weight",
            layer["router_topk_weight"][0, token_index],
            ctx.matRouterTopKWeight[:, 0],
            router_weight_threshold,
        ),
        check_tensor_threshold("final_hidden", layer["hidden_state_out"][0, token_index], ctx.matHidden[0], final_hidden_threshold),
        check_tensor_threshold("final_rms", captured["final"]["final_rms"][0, token_index], ctx.matRMSHidden[0], final_rms_threshold),
    ]
    router_idx_ok = torch.equal(layer["router_topk_idx"][0, token_index].cpu(), ctx.matRouterTopKIdx[0].cpu())
    print(
        f"[correctness] {'PASS' if router_idx_ok else 'FAIL'} router_topk_idx:"
        f" ref={layer['router_topk_idx'][0, token_index].tolist()},"
        f" dae={ctx.matRouterTopKIdx[0].tolist()}"
    )
    all_ok = all_ok and router_idx_ok

    for i in range(ctx.logits_epoch):
        start = i * ctx.logits_slice
        end = min((i + 1) * ctx.logits_slice, ctx.vocab_size)
        final_checks.append(
            check_tensor_threshold(
                f"logits_{i}",
                captured["final"]["lm_head"][0, token_index, start:end],
                ctx.matLogits[i][0, : end - start],
                10.0,
            )
        )

    all_ok = all_ok and all(passed for passed, _ in final_checks)

    ref_idx = torch.argmax(captured["final"]["lm_head"], dim=-1)
    dae_idx = ctx.matTokens[0, token_index + 1].item()
    ref_token = ref_idx[0, token_index].item()
    token_ok = ref_token == dae_idx
    print(f"[correctness] {'PASS' if token_ok else 'FAIL'} final_token: ref={ref_token}, dae={dae_idx}")
    all_ok = all_ok and token_ok

    if not all_ok:
        raise RuntimeError("Correctness check failed")
    print("[correctness] all checks passed")
