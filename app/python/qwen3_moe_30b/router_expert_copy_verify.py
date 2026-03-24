import torch

from dae.launcher import *
from dae.schedule import SchedRouterTopK
from dae.tma_utils import StaticCordAdapter, ToConvertedCordAdapter
from dae.util import dae_app, tensor_diff


class IndexedWeightCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, prefix):
        super().__init__(inner, lambda *cords: (*prefix, *cords))
        self.prefix = tuple(prefix)

    def cord2tma(self, *cords):
        return self.inner.cord2tma(*self.prefix, *cords)


gpu = torch.device("cuda")
torch.manual_seed(0)

N = 8
NUM_EXPERTS = 128
TOP_K = 8
TileM, _, TileK = Gemv_M64N8.MNK
M = 128
K = 512

matLogits = torch.rand(N, NUM_EXPERTS, dtype=torch.bfloat16, device=gpu) - 0.5
matTopKIdx = torch.zeros(1, TOP_K, dtype=torch.int32, device=gpu)
matTopKWeight = torch.zeros(TOP_K, 32, dtype=torch.bfloat16, device=gpu)
matExpert = torch.rand(1, NUM_EXPERTS, M, K, dtype=torch.bfloat16, device=gpu) - 0.5
matOut = torch.zeros(TileM, TileK, dtype=torch.bfloat16, device=gpu)

dae = Launcher(1, device=gpu)
group = dae.add_group("g", 1)
group.addBarrier("bar_router")
group.addTma("loadExpert", [matExpert], lambda t: t.indexed("layer_expert").wgmma_load(TileM, TileK, Major.K))
group.addTma("storeOut", [matOut], lambda t: t.wgmma_store(TileM, TileK, Major.K))
dae.build_groups()

router = SchedRouterTopK(
    num_token=N,
    logits_tma=StaticCordAdapter(TmaLoad1D(matLogits)),
    weight_tma=StaticCordAdapter(TmaStore1D(matTopKWeight)),
    idx_addr=StaticCordAdapter(RawAddress(matTopKIdx, 24)),
).bar("output", group["bar_router"]).place(1)

load_expert = IndexedWeightCordAdapter(group["loadExpert"], (0, 0)).cord(64, 256)
store_out = group["storeOut"].cord(0, 0)

def sm_task(sm: int):
    if sm != 0:
        return []
    return [
        Copy(1, size=group["loadExpert"].size),
        load_expert,
        store_out,
    ]

dae.i(
    router,
    SetLayerIndex(0),
    IssueBarrier(group["bar_router"]),
    LoadExpertIndex(matTopKIdx, 0),
    sm_task,
)
dae.bind_late_barrier_counts(router)
dae.s()
dae_app(dae)

probs = torch.softmax(matLogits.float(), dim=-1)
ref_idx = torch.topk(probs, TOP_K, dim=-1).indices[0, 0].item()
ref = matExpert[0, ref_idx, 64 : 64 + TileM, 256 : 256 + TileK]
print("router expert ref idx:", ref_idx)
print("router expert dae idx:", matTopKIdx[0, 0].item())
tensor_diff("router_expert_copy", ref, matOut)
