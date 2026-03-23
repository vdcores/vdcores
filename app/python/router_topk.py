import torch
from dae.launcher import *
from dae.schedule import SchedRouterTopK
from dae.tma_utils import StaticCordAdapter
from dae.util import dae_app, tensor_diff


gpu = torch.device("cuda")
torch.manual_seed(0)

N = 8
NUM_EXPERTS = 128
TOP_K = 8

matLogits = torch.rand(N, NUM_EXPERTS, dtype=torch.bfloat16, device=gpu) - 0.5
matTopKIdx = torch.zeros(1, TOP_K, dtype=torch.int32, device=gpu)
matTopKWeight = torch.zeros(TOP_K, 32, dtype=torch.bfloat16, device=gpu)

dae = Launcher(1, device=gpu)
sched = SchedRouterTopK(
    num_token=N,
    logits_tma=StaticCordAdapter(TmaLoad1D(matLogits)),
    weight_tma=StaticCordAdapter(TmaStore1D(matTopKWeight)),
    idx_tma=StaticCordAdapter(RawAddress(matTopKIdx, 24)),
).place(1)

dae.s(sched)
dae_app(dae)

probs = torch.softmax(matLogits.float(), dim=-1)
ref_weight, ref_idx = torch.topk(probs, TOP_K, dim=-1)
ref_weight = ref_weight / ref_weight.sum(dim=-1, keepdim=True)

print("router_topk idx ref:", ref_idx[0].tolist())
print("router_topk idx dae:", matTopKIdx[0].tolist())
tensor_diff("router_topk_weight", ref_weight[0], matTopKWeight[:, 0].float())
