import torch

from dae.launcher import *
from dae.schedule import SchedGemm, SchedGemv
from dae.util import dae_app, tensor_diff


torch.manual_seed(0)

gpu = torch.device("cuda")
dtype = torch.bfloat16

SHRINK_GEMM = Gemm_M64N64
EXPAND_GEMM = Gemm_M64N128K64
EXPAND_GEMV = globals().get("Gemv_M64N8K64")

HIDDEN = 4096
LORA_RANK = 64
GROUP_SIZES = [64] * 2 + [8] * 8
NUM_SMS = 128

if EXPAND_GEMV is None:
    raise RuntimeError("Gemv_M64N8K64 must be added before running app/python/lora_fixed_rank_demo.py")

def make_group_tensors():
    xs = []
    a_weights = []
    b_weights = []
    shrink_outs = []
    expand_outs = []

    for token_count in GROUP_SIZES:
        xs.append(torch.rand(token_count, HIDDEN, dtype=dtype, device=gpu) - 0.5)
        a_weights.append(torch.rand(LORA_RANK, HIDDEN, dtype=dtype, device=gpu) - 0.5)
        b_weights.append(torch.rand(HIDDEN, LORA_RANK, dtype=dtype, device=gpu) - 0.5)
        shrink_outs.append(torch.zeros(token_count, LORA_RANK, dtype=dtype, device=gpu))
        expand_outs.append(torch.zeros(token_count, HIDDEN, dtype=dtype, device=gpu))

    return xs, a_weights, b_weights, shrink_outs, expand_outs

def build_reference(xs, a_weights, b_weights):
    shrink_refs = []
    expand_refs = []
    for x, a_weight, b_weight in zip(xs, a_weights, b_weights):
        shrink_ref = x.float() @ a_weight.t().float()
        expand_ref = shrink_ref @ b_weight.t().float()
        shrink_refs.append(shrink_ref.to(dtype))
        expand_refs.append(expand_ref.to(dtype))
    return shrink_refs, expand_refs

matX, matA, matB, matShrink, matOut = make_group_tensors()
refShrink, refOut = build_reference(matX, matA, matB)

dae = Launcher(NUM_SMS, device=gpu)
bars = [None] * len(GROUP_SIZES)

#---- Schedule Srhink GEMM ----#
# simple heuristic to dispatch to different kernels
shrink_base_sm = 0
shrink_insts = []
for group_id, token_count in enumerate(GROUP_SIZES):
    if token_count == 8:
        Atom = Gemv_M64N8
        Atom.n_batch = 2
        M, N, K = LORA_RANK, token_count, HIDDEN
    elif token_count == 64:
        Atom = Gemv_M64N64
        M, N, K = token_count, LORA_RANK, HIDDEN
    else:
        raise ValueError(f"Unsupported token count {token_count}")
    TileM, TileN, TileK = Atom.MNK
    if token_count == 8:
        loadA = TmaTensor(dae, matA[group_id]).wgmma_load(TileM, TileK, Major.K)
        loadB = TmaTensor(dae, matX[group_id]).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
        reduceC = TmaTensor(dae, matShrink[group_id]).wgmma("reduce", TileN, TileM, Major.MN)
    else:
        loadA = TmaTensor(dae, matX[group_id]).wgmma_load(TileM, TileK, Major.K)
        loadB = TmaTensor(dae, matA[group_id]).wgmma_load(TileN, TileK, Major.K)
        reduceC = TmaTensor(dae, matShrink[group_id]).wgmma("reduce", TileM, TileN, Major.K)

    num_sms = 8 if token_count == 8 else 32
    shrink_bar = dae.new_bar(num_sms)
    bars[group_id] = shrink_bar

    op_class = SchedGemv if token_count == 8 else SchedGemm
    shrink = op_class(
        Atom,
        MNK=(M, N, K),
        tmas=(loadA, loadB, reduceC),
    ).place(num_sms, shrink_base_sm).bar("store", shrink_bar)
    shrink_insts.append(shrink)

    shrink_base_sm += num_sms
    if shrink_base_sm >= NUM_SMS:
        shrink_base_sm = 0

dae.i(
    *shrink_insts,
    TerminateC(),
    TerminateM(),
)

print("LoRA fixed-rank mixed pipeline")
print(f"group sizes: {GROUP_SIZES}, sms: {NUM_SMS}")

dae_app(dae)

for group_id, token_count in enumerate(GROUP_SIZES):
    tensor_diff(f"group{group_id}_shrink_{token_count}", refShrink[group_id], matShrink[group_id])
    # tensor_diff(f"group{group_id}_expand_{token_count}", refOut[group_id], matOut[group_id])
