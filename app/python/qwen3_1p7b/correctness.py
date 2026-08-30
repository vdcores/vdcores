import torch
import torch.nn.functional as F

from reference import check_tensor_threshold, input_batch1, reference_pass
from runtime_context import QwenScheduleContext, apply_rms_affine_rope_heads


def run_correctness_check(ctx: QwenScheduleContext):
    tensor_threshold = 5.0
    mlp_prefix = min(4096, ctx.INTERMIDIATE)

    print("[correctness] running prefill + single-decode reference capture...")
    token_index = len(ctx.prefill_token_id_and_pos)
    token_pos = ctx.input_token_id_and_pos[0][1]
    inputs = input_batch1(
        *(token for token, _ in ctx.prefill_token_id_and_pos),
        *(token for token, _ in ctx.input_token_id_and_pos),
        mat=ctx.matTokens[0],
        positions=[pos for _, pos in ctx.prefill_token_id_and_pos]
        + [pos for _, pos in ctx.input_token_id_and_pos],
    )

    captured, _ = reference_pass(ctx.model, inputs, rope_theta=ctx.rope_theta)
    rope_row = ctx.matRope[token_pos]
    all_ok = True

    for i in range(min(2, ctx.num_layers)):
        layer = captured[i]
        dae_q_rope = apply_rms_affine_rope_heads(
            ctx.attnQs[i][0].view(ctx.NUM_Q_HEAD, ctx.HEAD_DIM),
            ctx.matQNormWs[i],
            rope_row,
            ctx.eps,
        ).reshape(-1)
        print(f"[correctness] Layer {i}:")
        checks = [
            check_tensor_threshold(
                "v_proj",
                layer["v_proj"][0, token_index],
                ctx.attnVs[i][token_pos, 0],
                tensor_threshold,
            ),
            check_tensor_threshold(
                f"v_proj_req{ctx.BATCH - 1}",
                layer["v_proj"][0, token_index],
                ctx.attnVs[i][token_pos, ctx.BATCH - 1],
                tensor_threshold,
            ),
            check_tensor_threshold(
                "q_proj_interleaved",
                layer["q_proj_interleaved"][0, token_index],
                ctx.attnQs[i][0],
                tensor_threshold,
            ),
            check_tensor_threshold(
                "q_rope_interleaved",
                layer["q_rope_interleaved"][0, token_index],
                dae_q_rope,
                tensor_threshold,
            ),
            check_tensor_threshold(
                "k_rope_interleaved",
                layer["k_rope_interleaved"][0, token_index],
                ctx.attnKs[i][token_pos, 0],
                tensor_threshold,
            ),
            check_tensor_threshold(
                f"k_rope_interleaved_req{ctx.BATCH - 1}",
                layer["k_rope_interleaved"][0, token_index],
                ctx.attnKs[i][token_pos, ctx.BATCH - 1],
                tensor_threshold,
            ),
        ]
        all_ok = all_ok and all(passed for passed, _ in checks)

    print(f"[correctness] Checking Layer {ctx.num_layers - 1}:")
    layer = captured[ctx.num_layers - 1]
    silu_ref = F.silu(layer["gate_proj"][0, token_index]) * layer["up_proj"][0, token_index]
    q_ref = layer["q_rope_interleaved"][0, token_index].view(
        ctx.NUM_KV_HEAD, ctx.HEAD_GROUP_SIZE, ctx.HEAD_DIM
    ).float()
    k_ref = layer["k_rope_interleaved"][0, : token_index + 1].view(
        token_index + 1, ctx.NUM_KV_HEAD, ctx.HEAD_DIM
    ).float()
    v_ref = layer["v_proj"][0, : token_index + 1].view(
        token_index + 1, ctx.NUM_KV_HEAD, ctx.HEAD_DIM
    ).float()
    attention_probs = torch.softmax(
        torch.einsum("hgd,shd->hgs", q_ref, k_ref)
        * (ctx.HEAD_DIM ** -0.5),
        dim=-1,
    )
    attention_ref = torch.einsum(
        "hgs,shd->hgd", attention_probs, v_ref
    ).to(ctx.dtype).reshape(-1)
    final_checks = [
        check_tensor_threshold(
            "attention_out",
            attention_ref,
            ctx.attnO[0],
            tensor_threshold,
        ),
        check_tensor_threshold(
            f"attention_out_req{ctx.BATCH - 1}",
            attention_ref,
            ctx.attnO[ctx.BATCH - 1],
            tensor_threshold,
        ),
        check_tensor_threshold("gate_proj_prefix", layer["gate_proj"][0, token_index, :mlp_prefix], ctx.matGateOut[0, :mlp_prefix], tensor_threshold),
        check_tensor_threshold("up_proj_prefix", layer["up_proj"][0, token_index, :mlp_prefix], ctx.matInterm[0, :mlp_prefix], tensor_threshold),
        check_tensor_threshold("silu", silu_ref, ctx.matSiLUOut[0], tensor_threshold),
        check_tensor_threshold("final_hidden", layer["hidden_state_out"][0, token_index], ctx.matHidden[0], tensor_threshold),
        check_tensor_threshold("final_rms", captured["final"]["final_rms"][0, token_index], ctx.matRMSHidden[0], tensor_threshold),
    ]

    for i in range(ctx.logits_epoch):
        start = i * ctx.logits_slice
        end = min((i + 1) * ctx.logits_slice, ctx.vocab_size)
        final_checks.append(
            check_tensor_threshold(
                f"logits_{i}",
                captured["final"]["lm_head"][0, token_index, start:end],
                ctx.matLogits[i][0, : end - start],
                tensor_threshold,
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
