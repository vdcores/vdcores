import argparse
import sys

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app, tensor_diff


HIDDEN = 4096
INTERMEDIATE = 14336
TOKENS = 8
FULL_SMS = 132
WAVE0_SMS = 132
WAVE1_SMS = 92
WAVE0_M = WAVE0_SMS * 64
WAVE1_M = WAVE1_SMS * 64
DOWN_SMS = 128
SILU_SMS = 4
DTYPE = torch.bfloat16


def parse_args():
    raw_argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    parser.add_argument("--intermediate", type=int, default=INTERMEDIATE)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parsed_args, remaining_argv = parser.parse_known_args()
    if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
        remaining_argv = [*remaining_argv, "--launch"]
    sys.argv = [sys.argv[0], *remaining_argv]
    return parsed_args


def run_correctness_check(
    mat_rms_hidden,
    mat_gate_w,
    mat_up_w,
    mat_down_w,
    mat_gate_out,
    mat_interm,
    mat_silu_out,
    mat_hidden,
):
    with torch.no_grad():
        gate_ref = F.linear(mat_rms_hidden, mat_gate_w)
        up_ref = F.linear(mat_rms_hidden, mat_up_w)
        silu_ref = F.silu(gate_ref.float()).to(mat_rms_hidden.dtype) * up_ref
        down_ref = F.linear(silu_ref, mat_down_w)

    tensor_diff("down_proj", down_ref, mat_hidden)


parsed_args = parse_args()
torch.manual_seed(parsed_args.seed)

if parsed_args.hidden != HIDDEN:
    raise ValueError(f"Expected hidden == {HIDDEN}, got {parsed_args.hidden}")
if parsed_args.intermediate != INTERMEDIATE:
    raise ValueError(
        "This Llama serial harness targets the Llama3 8B width exactly: "
        f"expected intermediate == {INTERMEDIATE}, got {parsed_args.intermediate}"
    )
if parsed_args.tokens % SILU_SMS != 0:
    raise ValueError(f"--tokens must be divisible by {SILU_SMS}, got {parsed_args.tokens}")
if WAVE0_M + WAVE1_M != parsed_args.intermediate:
    raise ValueError("Wave sizes must cover the full intermediate width")

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

# SiLU consumes both full gate and full up outputs, so it must wait for all
# four projection waves: gate(132 + 92) and up(132 + 92).
bar_up_done = dae.new_bar(2 * (WAVE0_SMS + WAVE1_SMS))
bar_silu_done = dae.new_bar(parsed_args.tokens)

tile_m, _, tile_k = Gemv_M64N8.MNK
t_rms_hidden = TmaTensor(dae, mat_rms_hidden).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_store_gate = TmaTensor(dae, mat_gate_out).wgmma_store(parsed_args.tokens, tile_m, Major.MN)
t_store_interm = TmaTensor(dae, mat_interm).wgmma_store(parsed_args.tokens, tile_m, Major.MN)
t_load_silu = TmaTensor(dae, mat_silu_out).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_reduce_hidden = TmaTensor(dae, mat_hidden).wgmma("reduce", parsed_args.tokens, tile_m, Major.MN)
t_load_gate = TmaTensor(dae, mat_gate_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_up = TmaTensor(dae, mat_up_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_down = TmaTensor(dae, mat_down_w).wgmma_load(tile_m, tile_k, Major.K)

gate_proj_wave0 = SchedGemv(
    Gemv_M64N8,
    MNK=(WAVE0_M, parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_gate, t_rms_hidden, t_store_gate),
).bar("store", bar_up_done).place(WAVE0_SMS)
gate_proj_wave1 = SchedGemv(
    Gemv_M64N8,
    MNK=((WAVE0_M, WAVE1_M), parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_gate, t_rms_hidden, t_store_gate),
).bar("store", bar_up_done).place(WAVE1_SMS)

up_proj_wave0 = SchedGemv(
    Gemv_M64N8,
    MNK=(WAVE0_M, parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_up, t_rms_hidden, t_store_interm),
).bar("store", bar_up_done).place(WAVE0_SMS)
up_proj_wave1 = SchedGemv(
    Gemv_M64N8,
    MNK=((WAVE0_M, WAVE1_M), parsed_args.tokens, parsed_args.hidden),
    tmas=(t_load_up, t_rms_hidden, t_store_interm),
).bar("store", bar_up_done).place(WAVE1_SMS)

silu = SchedSmemSiLUInterleaved(
    num_token=parsed_args.tokens,
    gate_glob=mat_gate_out,
    up_glob=mat_interm,
    out_glob=mat_silu_out,
).bar("input", bar_up_done).bar("output", bar_silu_done).place(SILU_SMS, base_sm=128)

down_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(parsed_args.hidden, parsed_args.tokens, parsed_args.intermediate),
    tmas=(t_load_down, t_load_silu, t_reduce_hidden),
).bar("load", bar_silu_done).place(DOWN_SMS)

dae.s(
    gate_proj_wave0,
    gate_proj_wave1,
    up_proj_wave0,
    up_proj_wave1,
    silu,
    down_proj,
)

print(
    "run llama3 mlp-serial "
    f"hidden={parsed_args.hidden} intermediate={parsed_args.intermediate} "
    f"tokens={parsed_args.tokens} gate_up_waves=({WAVE0_SMS},{WAVE1_SMS}) down_sms={DOWN_SMS}..."
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
