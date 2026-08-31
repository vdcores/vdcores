import torch
import torch.nn.functional as F

from reference import check_tensor_threshold, input_batch1, reference_pass
from runtime_context import QwenScheduleContext, apply_rms_affine_rope_heads


def run_correctness_check(
    ctx: QwenScheduleContext,
    attn_q_snapshots: list[torch.Tensor] | None = None,
):
    silu_threshold = 5.0
    final_hidden_threshold = 5.0
    final_rms_threshold = 5.0

    print(
        "[correctness] running "
        f"{len(ctx.prefill_token_id_and_pos)}-prefill + 1-decode reference capture..."
    )
    decode_index = len(ctx.prefill_token_id_and_pos)
    decode_pos = ctx.input_token_id_and_pos[0][1]
    mlp_prefix = min(4096, ctx.INTERMIDIATE)
    inputs = input_batch1(
        *(token for token, _ in ctx.prefill_token_id_and_pos),
        *(token for token, _ in ctx.input_token_id_and_pos),
        mat=ctx.matTokens[0],
        positions=[pos for _, pos in ctx.prefill_token_id_and_pos] + [pos for _, pos in ctx.input_token_id_and_pos],
    )

    captured, _ = reference_pass(ctx.model, inputs, rope_theta=ctx.rope_theta)
    if attn_q_snapshots is None:
        raise ValueError("Qwen correctness requires pre-clear Q snapshots")
    rope_row = ctx.matRope[decode_pos]
    all_ok = True

    def check_batch_rows(name: str, rows: torch.Tensor):
        """Broadcast inputs must keep every live row numerically coherent."""
        if ctx.REQ == 1:
            return True, 0.0
        expected = rows[0:1].expand_as(rows)
        return check_tensor_threshold(
            f"{name}_batch_rows", expected, rows, 5.0
        )

    for i in range(min(2, ctx.num_layers)):
        layer = captured[i]
        dae_q_rope = apply_rms_affine_rope_heads(
            attn_q_snapshots[i][0].view(ctx.NUM_Q_HEAD, ctx.HEAD_DIM),
            ctx.matQNormWs[i],
            rope_row,
            ctx.eps,
        ).reshape(-1)
        print(f"[correctness] Layer {i}:")
        checks = [
            check_tensor_threshold("v_proj", layer["v_proj"][0, decode_index], ctx.attnVs[i][decode_pos, 0], 5.0),
            check_tensor_threshold(
                "q_proj_interleaved",
                layer["q_proj_interleaved"][0, decode_index],
                attn_q_snapshots[i][0],
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
                ctx.attnKs[i][decode_pos, 0],
                5.0,
            ),
            check_batch_rows("v_proj", ctx.attnVs[i][decode_pos, :ctx.REQ]),
            check_batch_rows("q_proj", attn_q_snapshots[i][:ctx.REQ]),
            check_batch_rows("k_rope", ctx.attnKs[i][decode_pos, :ctx.REQ]),
        ]
        all_ok = all_ok and all(passed for passed, _ in checks)

    print(f"[correctness] Checking Layer {ctx.num_layers - 1}:")
    layer = captured[ctx.num_layers - 1]
    silu_ref = F.silu(layer["gate_proj"][0, decode_index]) * layer["up_proj"][0, decode_index]
    final_checks = [
        check_tensor_threshold("gate_proj_low", layer["gate_proj"][0, decode_index, :mlp_prefix], ctx.matGateOut[0, :mlp_prefix], 5.0),
        check_tensor_threshold("up_proj_low", layer["up_proj"][0, decode_index, :mlp_prefix], ctx.matInterm[0, :mlp_prefix], 5.0),
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
                5.0,
            )
        )

    final_checks.extend([
        check_batch_rows("gate_proj_low", ctx.matGateOut[:ctx.REQ, :mlp_prefix]),
        check_batch_rows("up_proj_low", ctx.matInterm[:ctx.REQ, :mlp_prefix]),
        check_batch_rows("silu", ctx.matSiLUOut[:ctx.REQ]),
        check_batch_rows("final_hidden", ctx.matHidden[:ctx.REQ]),
        check_batch_rows("final_rms", ctx.matRMSHidden[:ctx.REQ]),
    ])
    for i in range(ctx.logits_epoch):
        final_checks.append(
            check_batch_rows(f"logits_{i}", ctx.matLogits[i][:ctx.REQ])
        )

    all_ok = all_ok and all(passed for passed, _ in final_checks)

    ref_idx = torch.argmax(captured["final"]["lm_head"], dim=-1)
    ref_token = ref_idx[0, decode_index].item()
    dae_tokens = ctx.matTokens[:ctx.REQ, decode_index + 1]
    token_ok = bool(torch.all(dae_tokens == ref_token))
    print(
        f"[correctness] {'PASS' if token_ok else 'FAIL'} final_token: "
        f"ref={ref_token}, dae={dae_tokens.tolist()}"
    )
    all_ok = all_ok and token_ok

    if not all_ok:
        raise RuntimeError("Correctness check failed")
    print("[correctness] all checks passed")
