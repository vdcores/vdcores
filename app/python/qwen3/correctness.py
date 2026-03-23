import torch
import torch.nn.functional as F

from reference import check_tensor_threshold, input_batch1, reference_pass
from runtime_context import QwenScheduleContext, apply_rms_affine_rope_heads


def run_correctness_check(ctx: QwenScheduleContext):
    silu_threshold = 10.0
    final_hidden_threshold = 15.0
    final_rms_threshold = 12.0

    print("[correctness] running 1-prefill + 1-decode reference capture...")
    decode_index = len(ctx.prefill_token_id_and_pos)
    decode_pos = ctx.input_token_id_and_pos[0][1]
    inputs = input_batch1(
        *(token for token, _ in ctx.prefill_token_id_and_pos),
        *(token for token, _ in ctx.input_token_id_and_pos),
        mat=ctx.matTokens[0],
        positions=[pos for _, pos in ctx.prefill_token_id_and_pos] + [pos for _, pos in ctx.input_token_id_and_pos],
    )

    captured, _ = reference_pass(ctx.model, inputs, rope_theta=ctx.rope_theta)
    rope_row = ctx.matRope[decode_pos]
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
            check_tensor_threshold("v_proj", layer["v_proj"][0, decode_index], ctx.attnVs[i][0, decode_pos], 5.0),
            check_tensor_threshold(
                "q_proj_interleaved",
                layer["q_proj_interleaved"][0, decode_index],
                ctx.attnQs[i][0],
                5.0,
            ),
            check_tensor_threshold(
                "q_rope_interleaved",
                layer["q_rope_interleaved"][0, decode_index],
                dae_q_rope,
                5.0,
            ),
            check_tensor_threshold(
                "k_rope_interleaved",
                layer["k_rope_interleaved"][0, decode_index],
                ctx.attnKs[i][0, decode_pos],
                5.0,
            ),
        ]
        all_ok = all_ok and all(passed for passed, _ in checks)

    print(f"[correctness] Checking Layer {ctx.num_layers - 1}:")
    layer = captured[ctx.num_layers - 1]
    silu_ref = F.silu(layer["gate_proj"][0, decode_index]) * layer["up_proj"][0, decode_index]
    final_checks = [
        check_tensor_threshold("gate_proj_low", layer["gate_proj"][0, decode_index, :4096], ctx.matGateOut[0, :4096], 5.0),
        check_tensor_threshold("up_proj_low", layer["up_proj"][0, decode_index, :4096], ctx.matInterm[0, :4096], 5.0),
        check_tensor_threshold("silu", silu_ref, ctx.matSiLUOut[0], silu_threshold),
        check_tensor_threshold("final_hidden", layer["hidden_state_out"][0, decode_index], ctx.matHidden[0], final_hidden_threshold),
        check_tensor_threshold("final_rms", captured["final"]["final_rms"][0, decode_index], ctx.matRMSHidden[0], final_rms_threshold),
    ]

    for i in range(ctx.logits_epoch):
        start = i * ctx.logits_slice
        end = min((i + 1) * ctx.logits_slice, ctx.vocab_size)
        final_checks.append(
            check_tensor_threshold(
                f"logits_{i}",
                captured["final"]["lm_head"][0, decode_index, start:end],
                ctx.matLogits[i][0, :end - start],
                10.0,
            )
        )

    all_ok = all_ok and all(passed for passed, _ in final_checks)

    ref_idx = torch.argmax(captured["final"]["lm_head"], dim=-1)
    dae_idx = ctx.matTokens[0, decode_index + 1].item()
    ref_token = ref_idx[0, decode_index].item()
    token_ok = ref_token == dae_idx
    print(f"[correctness] {'PASS' if token_ok else 'FAIL'} final_token: ref={ref_token}, dae={dae_idx}")
    all_ok = all_ok and token_ok

    if not all_ok:
        raise RuntimeError("Correctness check failed")
    print("[correctness] all checks passed")
