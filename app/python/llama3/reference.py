import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import os

from dae.util import tensor_diff


def mean_relative_diff_pct(t1: torch.Tensor, t2: torch.Tensor, ref: torch.Tensor | None = None):
    if ref is None:
        ref = t1
    denom = ref.abs().float().mean().item()
    if denom == 0:
        denom = 1.0
    return (t1 - t2).abs().float().mean().item() / denom * 100.0


def check_tensor_threshold(name: str,
                           expected: torch.Tensor,
                           actual: torch.Tensor,
                           threshold_pct: float,
                           ref: torch.Tensor | None = None):
    diff = mean_relative_diff_pct(expected, actual, ref=ref)
    tensor_diff(name, expected, actual, ref=ref)
    passed = diff <= threshold_pct
    status = "PASS" if passed else "FAIL"
    print(f"[correctness] {status} {name}: {diff:.3f}% <= {threshold_pct:.3f}%")
    return passed, diff


def reference_pass(model, inputs, verbose=False):
    # Container for all captured tensors
    # structure: { layer_idx: { "q_proj": tensor, "rms_1": tensor, ... } }
    captured_data = defaultdict(dict)

    def register_llama_hooks(model):
        handles = []

        def get_pre_hook(layer_idx, name):
            def hook(module, inputs):
                captured_data[layer_idx][name] = inputs[0].detach()
            return hook

        def get_hook(layer_idx, name):
            def hook(module, input, output):
                # Handle cases where output is a tuple (like in Attention layers)
                data = output[0] if isinstance(output, tuple) else output
                captured_data[layer_idx][name] = data.detach()
            return hook

        for i, layer in enumerate(model.model.layers):
            # --- Layer Input (Hidden State) ---
            handles.append(layer.register_forward_pre_hook(get_pre_hook(i, "hidden_state_in")))
            
            # --- RMSNorm Outputs ---
            handles.append(layer.input_layernorm.register_forward_hook(get_hook(i, "input_rms")))
            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(get_pre_hook(i, "post_attn_residual")))
            handles.append(layer.post_attention_layernorm.register_forward_hook(get_hook(i, "post_attn_rms")))

            # --- Attention Projections ---
            attn = layer.self_attn
            handles.append(attn.q_proj.register_forward_hook(get_hook(i, "q_proj")))
            handles.append(attn.k_proj.register_forward_hook(get_hook(i, "k_proj")))
            handles.append(attn.v_proj.register_forward_hook(get_hook(i, "v_proj")))
            handles.append(attn.o_proj.register_forward_hook(get_hook(i, "o_proj")))
            
            # --- Full Attention Block Output (Post-O_proj + Softmax) ---
            handles.append(attn.register_forward_hook(get_hook(i, "attn_block_out")))

            # --- MLP Projections ---
            mlp = layer.mlp
            handles.append(mlp.gate_proj.register_forward_hook(get_hook(i, "gate_proj")))
            handles.append(mlp.up_proj.register_forward_hook(get_hook(i, "up_proj")))
            handles.append(mlp.down_proj.register_forward_hook(get_hook(i, "down_proj")))
            
            # --- Layer Output (Hidden State) ---
            handles.append(layer.register_forward_hook(get_hook(i, "hidden_state_out")))

        # --- 2. Final Model-Level Components ---
        # Final RMSNorm (applied after all layers)
        handles.append(model.model.norm.register_forward_hook(get_hook("final", "final_rms")))
        
        # LM Head (Logits generation)
        handles.append(model.lm_head.register_forward_hook(get_hook("final", "lm_head")))

        return handles

    # 2. Run Inference
    hooks = register_llama_hooks(model)

    with torch.no_grad():
        output = model(**inputs, use_cache=False)
        if verbose:
            print(output)

    # 3. Cleanup
    for h in hooks:
        h.remove()

    return captured_data, output

def input_batch1(*tokens, mat = None, positions=None):
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
        "position_ids": position_ids
    }

if __name__ == '__main__':
    # 1. Load Model (Using bfloat16 to match H100/A100 hardware targets)
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    from transformers import LlamaConfig
    import transformers.models.llama.modeling_llama as m

    config = LlamaConfig()
    config.rope_parameters = {
        "rope_theta": 1e30, # quick hack to disable rotary embed
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="/tmp/huggingface_cache",
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.environ['HF_TOKEN'],
        # config=config
    )

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # prompt = "T"
    # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    inputs = input_batch1(128000)

    # print(inputs)

    captured_data, output = reference_pass(model, inputs)
    # pkv = output.past_key_values
    # for k, v, _ in pkv:
    #     print(k.shape, v.shape)
