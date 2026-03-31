import argparse
import sys

import torch

from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app


HIDDEN = 4096
INTERMEDIATE = 12288
TOKENS = 8
FULL_SMS = 132
DOWN_SMS = 128
PROJ_SLICE = 8192
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

parsed_args = parse_args()
torch.manual_seed(parsed_args.seed)

if parsed_args.hidden != HIDDEN:
    raise ValueError(f"Expected hidden == {HIDDEN}, got {parsed_args.hidden}")
if parsed_args.intermediate != INTERMEDIATE:
    raise ValueError(
        "This serial harness currently targets the Qwen3 MLP width exactly: "
        f"expected intermediate == {INTERMEDIATE}, got {parsed_args.intermediate}"
    )

gpu = torch.device("cuda")
dae = Launcher(FULL_SMS, device=gpu)

mat_rms_hidden = torch.rand(parsed_args.tokens, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_proj_out = torch.zeros(parsed_args.tokens, parsed_args.intermediate * 2, dtype=DTYPE, device=gpu)
mat_silu_out = torch.zeros(parsed_args.tokens, parsed_args.intermediate, dtype=DTYPE, device=gpu)
mat_hidden = torch.zeros(parsed_args.tokens, parsed_args.hidden, dtype=DTYPE, device=gpu)

mat_gate_w = torch.rand(parsed_args.intermediate, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_up_w = torch.rand(parsed_args.intermediate, parsed_args.hidden, dtype=DTYPE, device=gpu) - 0.5
mat_down_w = torch.rand(parsed_args.hidden, parsed_args.intermediate, dtype=DTYPE, device=gpu) - 0.5
mat_proj_w = torch.empty(parsed_args.intermediate * 2, parsed_args.hidden, dtype=DTYPE, device=gpu)
for chunk_idx in range(3):
    gate_start = chunk_idx * 4096
    proj_start = chunk_idx * PROJ_SLICE
    mat_proj_w[proj_start:proj_start + 4096] = mat_gate_w[gate_start:gate_start + 4096]
    mat_proj_w[proj_start + 4096:proj_start + PROJ_SLICE] = mat_up_w[gate_start:gate_start + 4096]

dae.set_streaming(mat_proj_w, mat_down_w)

num_proj_slices = (parsed_args.intermediate * 2) // PROJ_SLICE
num_silu_slices = parsed_args.intermediate // 4096
bar_proj_done = dae.new_bar(num_proj_slices * 128)
bar_silu_done = dae.new_bar(num_silu_slices * parsed_args.tokens)

tile_m, _, tile_k = Gemv_M64N8.MNK
t_rms_hidden = TmaTensor(dae, mat_rms_hidden).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_load_silu = TmaTensor(dae, mat_silu_out).wgmma_load(parsed_args.tokens, tile_k * Gemv_M64N8.n_batch, Major.K)
t_reduce_hidden = TmaTensor(dae, mat_hidden).wgmma("reduce", parsed_args.tokens, tile_m, Major.MN)
t_load_proj = TmaTensor(dae, mat_proj_w).wgmma_load(tile_m, tile_k, Major.K)
t_load_down = TmaTensor(dae, mat_down_w).wgmma_load(tile_m, tile_k, Major.K)
t_store_proj = TmaTensor(dae, mat_proj_out).wgmma_store(parsed_args.tokens, tile_m, Major.MN)

proj_schedules = []
silu_schedules = []
for base_offset in range(0, parsed_args.intermediate * 2, PROJ_SLICE):
    proj_schedules.append(
        SchedGemv(
            Gemv_M64N8,
            MNK=((base_offset, PROJ_SLICE), parsed_args.tokens, parsed_args.hidden),
            tmas=(t_load_proj, t_rms_hidden, t_store_proj),
        ).bar("store", bar_proj_done).place(128)
    )

for silu_idx, base_offset in enumerate(range(0, parsed_args.intermediate, 4096)):
    proj_base = silu_idx * PROJ_SLICE
    silu_schedules.append(
        SchedSmemSiLUInterleaved(
            num_token=parsed_args.tokens,
            gate_glob=mat_proj_out[:, proj_base:proj_base + 4096],
            up_glob=mat_proj_out[:, proj_base + 4096:proj_base + PROJ_SLICE],
            out_glob=mat_silu_out[:, base_offset:base_offset + 4096],
        ).bar("input", bar_proj_done).bar("output", bar_silu_done).place(parsed_args.tokens)
    )

down_proj = SchedGemv(
    Gemv_M64N8,
    MNK=(parsed_args.hidden, parsed_args.tokens, parsed_args.intermediate),
    tmas=(t_load_down, t_load_silu, t_reduce_hidden),
).bar("load", bar_silu_done).place(DOWN_SMS)

dae.s(
    *proj_schedules,
    *silu_schedules,
    down_proj,
)

print(
    "run qwen3 mlp-serial "
    f"hidden={parsed_args.hidden} intermediate={parsed_args.intermediate} "
    f"tokens={parsed_args.tokens} proj_sms=128 proj_slices={num_proj_slices} down_sms={DOWN_SMS}..."
)
dae_app(dae)
