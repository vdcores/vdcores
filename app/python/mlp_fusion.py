import argparse
import sys

import torch
import torch.nn.functional as F
from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app
from qwen3.reference import check_tensor_threshold


HIDDEN = 4096
INTERMEDIATE = 12288
LOW_INTERMEDIATE = 4096
N = 8
FULL_SMS = 132
GEMM_SMS = 128
SILU_SMS = 4
DTYPE = torch.bfloat16


def parse_args():
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    parser.add_argument("--tokens", type=int, default=N)
    parsed_args, remaining_argv = parser.parse_known_args()
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args


def run_correctness_check(mat_rms_hidden, mat_gate_w, mat_up_w, mat_down_w, mat_gate_out, mat_interm, mat_silu_out, mat_hidden):
    with torch.no_grad():
        gate_ref = F.linear(mat_rms_hidden, mat_gate_w)
        up_ref = F.linear(mat_rms_hidden, mat_up_w)
        silu_ref = F.silu(gate_ref.float()).to(mat_rms_hidden.dtype) * up_ref
        down_ref = F.linear(silu_ref, mat_down_w)

    checks = [
        check_tensor_threshold("gate_proj", gate_ref, mat_gate_out, 5.0),
        check_tensor_threshold("up_proj", up_ref, mat_interm, 5.0),
        check_tensor_threshold("silu", silu_ref, mat_silu_out, 10.0),
        check_tensor_threshold("down_proj", down_ref, mat_hidden, 10.0),
    ]
    if not all(passed for passed, _ in checks):
        raise RuntimeError("MLP correctness check failed")
    print("[correctness] all checks passed")


parsed_args = parse_args()
torch.manual_seed(parsed_args.seed)

if parsed_args.intermediate < LOW_INTERMEDIATE:
    raise ValueError(
        f"Expected intermediate >= {LOW_INTERMEDIATE}, got {parsed_args.intermediate}"
    )
if (parsed_args.intermediate - LOW_INTERMEDIATE) % Gemv_M64N8.MNK[0] != 0:
    raise ValueError(
        "High-half intermediate width must be a multiple of the GEMV tile size "
        f"{Gemv_M64N8.MNK[0]}, got {parsed_args.intermediate - LOW_INTERMEDIATE}"
    )
if parsed_args.tokens % SILU_SMS != 0:
    raise ValueError(f"--tokens must be divisible by {SILU_SMS}, got {parsed_args.tokens}")

gpu = torch.device("cuda")
dae = Launcher(FULL_SMS, device=gpu)

mat_rms_hidden = torch.rand(parsed_args.tokens, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_gate_out = torch.zeros(parsed_args.tokens, parsed_args.intermediate, dtype=DTYPE, device=gpu)
mat_interm = torch.zeros(parsed_args.tokens, parsed_args.intermediate, dtype=DTYPE, device=gpu)
mat_silu_out = torch.zeros(parsed_args.tokens, parsed_args.intermediate, dtype=DTYPE, device=gpu)
mat_hidden = torch.zeros(parsed_args.tokens, parsed_args.hidden, dtype=DTYPE, device=gpu)

mat_gate_w = torch.rand(parsed_args.intermediate, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_up_w = torch.rand(parsed_args.intermediate, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_down_w = torch.rand(parsed_args.hidden, parsed_args.intermediate, dtype=DTYPE, device=gpu) - 0.5

dae.set_streaming(mat_gate_w, mat_up_w, mat_down_w)

bar_silu_in = dae.new_bar(128)
bar_silu_out1 = dae.new_bar(N)
bar_silu_out2 = dae.new_bar(GEMM_SMS)

tile_m, _, tile_k = Gemv_M64N8.MNK
t_rms_hidden = TmaTensor(dae, mat_rms_hidden).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_store_gate = TmaTensor(dae, mat_gate_out).wgmma_store(parsed_args.tokens, tile_m, Major.MN)
t_store_interm = TmaTensor(dae, mat_interm).wgmma_store(parsed_args.tokens, tile_m, Major.MN)
t_load_silu = TmaTensor(dae, mat_silu_out).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_store_silu = TmaTensor(dae, mat_silu_out).wgmma_store(parsed_args.tokens, tile_m, Major.MN)
t_reduce_hidden = TmaTensor(dae, mat_hidden).wgmma("reduce", parsed_args.tokens, tile_m, Major.MN)
t_load_gate = TmaTensor(dae, mat_gate_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_up = TmaTensor(dae, mat_up_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_down = TmaTensor(dae, mat_down_w).wgmma_load(tile_m, tile_k, Major.K)

gate_proj_low = SchedGemv(
    Gemv_M64N8,
    MNK=(LOW_INTERMEDIATE, parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_gate, t_rms_hidden, t_store_gate),
).bar("store", bar_silu_in)
up_proj_low = SchedGemv(
    Gemv_M64N8,
    MNK=(LOW_INTERMEDIATE, parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_up, t_rms_hidden, t_store_interm),
).bar("store", bar_silu_in)

silu_low = SchedSmemSiLUInterleaved(
    num_token=parsed_args.tokens,
    gate_glob=mat_gate_out[:, :LOW_INTERMEDIATE],
    up_glob=mat_interm[:, :LOW_INTERMEDIATE],
    out_glob=mat_silu_out[:, :LOW_INTERMEDIATE],
).bar("input", bar_silu_in).bar("output", bar_silu_out1)

reg_gate = 0
reg_up = 1
reg_store_gate = RegStore(reg_gate, mat_gate_out[:, 0:tile_m])
reg_store_up = RegStore(reg_up, mat_interm[:, 0:tile_m])

gate_proj_high = SchedGemv(
    Gemv_M64N8,
    MNK=((LOW_INTERMEDIATE, parsed_args.intermediate - LOW_INTERMEDIATE), parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_gate, t_rms_hidden, reg_store_gate),
)
up_proj_high = SchedGemv(
    Gemv_M64N8,
    MNK=((LOW_INTERMEDIATE, parsed_args.intermediate - LOW_INTERMEDIATE), parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_up, t_rms_hidden, reg_store_up),
)
silu_high = SchedRegSiLUFused(
    num_token=parsed_args.tokens,
    store_tma=t_store_silu,
    reg_gate=reg_gate,
    reg_up=reg_up,
    base_offset=LOW_INTERMEDIATE,
    stride=tile_m,
).bar("output", bar_silu_out2)

down_proj_low = SchedGemv(
    Gemv_M64N8,
    MNK=(parsed_args.hidden, parsed_args.tokens, LOW_INTERMEDIATE),
    tmas=(t_load_down, t_load_silu, t_reduce_hidden),
).bar("load", bar_silu_out1)
down_proj_high = SchedGemv(
    Gemv_M64N8,
    MNK=(parsed_args.hidden, parsed_args.tokens, (LOW_INTERMEDIATE, parsed_args.intermediate - LOW_INTERMEDIATE)),
    tmas=(t_load_down, t_load_silu, t_reduce_hidden),
).bar("load", bar_silu_out2)

gate_proj_low = gate_proj_low.place(64)
up_proj_low = up_proj_low.place(64, base_sm=64)
silu_low = silu_low.place(SILU_SMS, base_sm=GEMM_SMS)
gate_proj_high = gate_proj_high.place(GEMM_SMS)
up_proj_high = up_proj_high.place(GEMM_SMS)
silu_high = silu_high.place(GEMM_SMS)
down_proj_low = down_proj_low.place(GEMM_SMS)
down_proj_high = down_proj_high.place(GEMM_SMS)

dae.s(
    gate_proj_low,
    up_proj_low,
    silu_low,
    gate_proj_high,
    up_proj_high,
    silu_high,
    down_proj_low,
    down_proj_high,
)

print(
    "run qwen3 mlp-only "
    f"hidden={parsed_args.hidden} intermediate={parsed_args.intermediate} tokens={parsed_args.tokens}..."
)
dae_app(dae)
if parsed_args.correctness:
    run_correctness_check(
        mat_rms_hidden,
        mat_gate_w,
        mat_up_w,
        mat_down_w,
        mat_gate_out,
        mat_interm,
        mat_silu_out,
        mat_hidden,
    )
