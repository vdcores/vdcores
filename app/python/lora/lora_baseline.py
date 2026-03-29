import torch

from dae.launcher import *
from dae.schedule import SchedGemv
from dae.util import dae_app, tensor_diff


torch.manual_seed(0)

gpu = torch.device("cuda")
dtype = torch.bfloat16

HIDDEN = 4096
LORA_RANK = 64
GROUP_SIZES = [128] + [64] * 4 + [8] * 11
NUM_SMS = 128

EXPAND_GEMV = globals().get("Gemv_M64N8K64")
if EXPAND_GEMV is None:
    raise RuntimeError("Gemv_M64N8K64 must be added before running app/python/lora_baseline.py")


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
bars = [None] * (len(GROUP_SIZES) + 1)

shrink_insts = []
expand_insts = []


def base_shrink_sched():
    shrink_base_sm = 0

    def split_n(base_sm, num_sm, token_count, atom):
        tile_m, tile_n, tile_k = atom.MNK
        insts = []
        loadA = TmaTensor(dae, matA[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        for i in range(token_count // tile_n):
            loadB = TmaTensor(dae, matX[group_id][i * tile_n:(i + 1) * tile_n]).wgmma_load(
                tile_n, tile_k * atom.n_batch, Major.K
            )
            reduceC = TmaTensor(dae, matShrink[group_id][i * tile_n:(i + 1) * tile_n]).wgmma(
                "reduce", tile_n, tile_m, Major.MN
            )

            inst = SchedGemv(
                atom,
                MNK=(LORA_RANK, tile_n, HIDDEN),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sm, base_sm).bar("store", bars[-1])
            insts.append(inst)
            base_sm = (base_sm + num_sm) % NUM_SMS
        return insts, base_sm

    bar_cnt = sum(token_count for token_count in GROUP_SIZES) // 8
    bars[-1] = dae.new_bar(bar_cnt)
    for group_id, token_count in enumerate(GROUP_SIZES):
        insts, shrink_base_sm = split_n(shrink_base_sm, 1, token_count, Gemv_M64N8)
        shrink_insts.extend(insts)


def base_expand_sched():
    expand_base_sm = 0

    def split_n_m(base_sm, num_sm, token_count, atom):
        tile_m, tile_n, tile_k = atom.MNK
        insts = []
        loadA = TmaTensor(dae, matB[group_id]).wgmma_load(tile_m, tile_k, Major.K)
        for i in range(token_count // tile_n):
            loadB = TmaTensor(dae, matShrink[group_id][i * tile_n:(i + 1) * tile_n]).wgmma_load(
                tile_n, tile_k * atom.n_batch, Major.K
            )
            reduceC = TmaTensor(dae, matOut[group_id][i * tile_n:(i + 1) * tile_n]).wgmma(
                "reduce", tile_n, tile_m, Major.MN
            )

            inst = SchedGemv(
                atom,
                MNK=(HIDDEN, tile_n, LORA_RANK),
                tmas=(loadA, loadB, reduceC),
            ).place(num_sm, base_sm).bar("load", bars[-1])
            insts.append(inst)
            base_sm = (base_sm + num_sm) % NUM_SMS
        return insts, base_sm

    for group_id, token_count in enumerate(GROUP_SIZES):
        num_sms = HIDDEN // 64
        insts, expand_base_sm = split_n_m(expand_base_sm, num_sms, token_count, Gemv_M64N8K64)
        expand_insts.extend(insts)


base_shrink_sched()
base_expand_sched()

dae.i(
    shrink_insts,
    expand_insts,
    TerminateC(),
    TerminateM(),
)

print("LoRA fixed-rank baseline")
print(f"group sizes: {GROUP_SIZES}, sms: {NUM_SMS}")

dae_app(dae)

# for group_id, token_count in enumerate(GROUP_SIZES):
#     tensor_diff(f"group{group_id}_shrink_{token_count}", refShrink[group_id], matShrink[group_id])
#     tensor_diff(f"group{group_id}_expand_{token_count}", refOut[group_id], matOut[group_id])
