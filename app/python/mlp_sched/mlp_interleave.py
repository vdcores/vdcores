import argparse
import sys

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app, tensor_diff


HIDDEN = 4096
INTERMEDIATE = 12288
TOKENS = 8
FULL_SMS = 132
STAGE_SMS = 128
SILU_SMS = 4
SLICE_INTERMEDIATE = 4096
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

    checks = [
        tensor_diff("down_proj", down_ref, mat_hidden),
    ]
    if not all(passed for passed, _ in checks):
        raise RuntimeError("MLP correctness check failed")
    print("[correctness] all checks passed")


parsed_args = parse_args()
torch.manual_seed(parsed_args.seed)

if parsed_args.hidden != HIDDEN:
    raise ValueError(f"Expected hidden == {HIDDEN}, got {parsed_args.hidden}")
if parsed_args.intermediate != INTERMEDIATE:
    raise ValueError(
        "This interleaved harness currently targets the Qwen3 MLP width exactly: "
        f"expected intermediate == {INTERMEDIATE}, got {parsed_args.intermediate}"
    )

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

num_slices = parsed_args.intermediate // SLICE_INTERMEDIATE
bar_silu_in = [dae.new_bar(2 * STAGE_SMS) for _ in range(num_slices)]
bar_silu_out = [dae.new_bar(parsed_args.tokens) for _ in range(num_slices)]

tile_m, _, tile_k = Gemv_M64N8.MNK
t_rms_hidden = TmaTensor(dae, mat_rms_hidden).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_load_silu = TmaTensor(dae, mat_silu_out).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_reduce_hidden = TmaTensor(dae, mat_hidden).wgmma("reduce", parsed_args.tokens, tile_m, Major.MN)
t_load_gate = TmaTensor(dae, mat_gate_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_up = TmaTensor(dae, mat_up_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_down = TmaTensor(dae, mat_down_w).wgmma_load(tile_m, tile_k, Major.K)
t_reduce_gate = TmaTensor(dae, mat_gate_out).wgmma("reduce", parsed_args.tokens, tile_m, Major.MN)
t_reduce_interm = TmaTensor(dae, mat_interm).wgmma("reduce", parsed_args.tokens, tile_m, Major.MN)

gate_schedules = []
up_schedules = []
down_schedules = []
silu_schedules = []
stage_items = []
for slice_idx, base_offset in enumerate(range(0, parsed_args.intermediate, SLICE_INTERMEDIATE)):
    gate_proj = (
        SchedGemv(
            Gemv_M64N8,
            MNK=((base_offset, SLICE_INTERMEDIATE), parsed_args.tokens, parsed_args.hidden),
            tmas=(t_load_gate, t_rms_hidden, t_reduce_gate),
        ).bar("store", bar_silu_in[slice_idx]).place(STAGE_SMS)
    )
    up_proj = (
        SchedGemv(
            Gemv_M64N8,
            MNK=((base_offset, SLICE_INTERMEDIATE), parsed_args.tokens, parsed_args.hidden),
            tmas=(t_load_up, t_rms_hidden, t_reduce_interm),
        ).bar("store", bar_silu_in[slice_idx]).place(STAGE_SMS)
    )
    silu = (
        SchedSmemSiLUInterleaved(
            num_token=parsed_args.tokens,
            gate_glob=mat_gate_out[:, base_offset:base_offset + SLICE_INTERMEDIATE],
            up_glob=mat_interm[:, base_offset:base_offset + SLICE_INTERMEDIATE],
            out_glob=mat_silu_out[:, base_offset:base_offset + SLICE_INTERMEDIATE],
        ).bar("output", bar_silu_out[slice_idx]).place(SILU_SMS, base_sm=STAGE_SMS)
        # .bar("input", bar_silu_in[slice_idx])
    )
    down_proj = (
        SchedGemv(
            Gemv_M64N8,
            MNK=(parsed_args.hidden, parsed_args.tokens, (base_offset, SLICE_INTERMEDIATE)),
            tmas=(t_load_down, t_load_silu, t_reduce_hidden),
        ).place(STAGE_SMS)
        # .bar("load", bar_silu_out[slice_idx])
    )
    gate_schedules.append(gate_proj)
    up_schedules.append(up_proj)
    silu_schedules.append(silu)
    down_schedules.append(down_proj)
    stage_items.extend([gate_proj, up_proj, silu, down_proj])

dae.s(
    *stage_items,
)

print(
    "run qwen3 mlp-only interleaved "
    f"hidden={parsed_args.hidden} intermediate={parsed_args.intermediate} "
    f"tokens={parsed_args.tokens} stage_sms={STAGE_SMS} silu_sms={SILU_SMS}..."
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
