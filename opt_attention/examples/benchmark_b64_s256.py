import math
import os
import statistics

import torch
from transformers import OPTConfig
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.opt.modeling_opt import OPTAttention, eager_attention_forward

import opt_attention


def bench(label, fn, warmup=20, iters=100):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))
    print(
        f"{label}: mean_ms={statistics.mean(times):.4f} "
        f"median_ms={statistics.median(times):.4f} min_ms={min(times):.4f}"
    )


def run_case(label: str, heads: int, split_size: int = 256):
    device = torch.device("cuda")
    dtype = torch.float16
    batch, seq_len, head_dim = 64, 256, 128
    hidden = heads * head_dim
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    num_splits = math.ceil(seq_len / split_size)
    ctas = batch * heads * num_splits

    os.environ["OPT_ATTENTION_SPLIT_SIZE"] = str(split_size)
    print(
        f"case {label} batch={batch} heads={heads} hidden={hidden} seq={seq_len} "
        f"split_size={split_size} splits={num_splits} ctas={ctas} estimated_waves={math.ceil(ctas / sms)}"
    )

    config = OPTConfig(hidden_size=hidden, num_attention_heads=heads, num_hidden_layers=1)
    module = OPTAttention(config, layer_idx=0).to(device=device, dtype=dtype)
    module.eval()

    torch.manual_seed(123 + heads)
    query = torch.randn(batch, heads, 1, head_dim, device=device, dtype=dtype) / math.sqrt(head_dim)
    key = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)

    vd_iface = ALL_ATTENTION_FUNCTIONS.get_interface("vdcores_opt", eager_attention_forward)
    sdpa_iface = ALL_ATTENTION_FUNCTIONS.get_interface("sdpa", eager_attention_forward)

    with torch.no_grad():
        got, _ = vd_iface(module, query, key, value, None, dropout=0.0, scaling=1.0)
        ref, _ = sdpa_iface(module, query, key, value, None, dropout=0.0, scaling=1.0)
        torch.cuda.synchronize()

    diff = (got - ref).abs().float()
    print(
        f"correctness_vs_sdpa: max_err={diff.max().item():.8f} "
        f"mean_err={diff.mean().item():.8f} "
        f"allclose={torch.allclose(got, ref, rtol=4e-2, atol=4e-2)}"
    )
    bench("vdcores_opt_interface", lambda: vd_iface(module, query, key, value, None, dropout=0.0, scaling=1.0))
    bench("pytorch_sdpa_interface", lambda: sdpa_iface(module, query, key, value, None, dropout=0.0, scaling=1.0))


def main():
    opt_attention.register()
    split_size = int(os.environ.get("OPT_ATTENTION_SPLIT_SIZE", "256"))
    cases = (
        ("occupancy-reference", 1),
        ("OPT-6.7B", 32),
        ("OPT-30B", 56),
    )
    for label, heads in cases:
        run_case(label, heads, split_size=split_size)


if __name__ == "__main__":
    main()
