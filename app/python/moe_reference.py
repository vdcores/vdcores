import torch

from dae.launcher import *
from dae.util import dae_app, tensor_diff


def routing_gemm(sm: int):
    tile_m, tile_n, tile_k = Atom.MNK
    n_batch = Atom.n_batch
    m_tiles = M // tile_m
    n_tiles = N // tile_n
    tiles_per_fold = m_tiles * n_tiles
    total_workers = tiles_per_fold * fold
    k_per_fold = K // fold

    insts = []
    for worker in range(sm, total_workers, num_sms):
        tile_idx = worker % tiles_per_fold
        fold_idx = worker // tiles_per_fold
        m_tile = tile_idx % m_tiles
        n_tile = tile_idx // m_tiles
        m = m_tile * tile_m
        n = n_tile * tile_n
        k_start = fold_idx * k_per_fold

        insts += [Atom(k_per_fold // tile_k)]
        for k_group in range(k_per_fold // (tile_k * n_batch)):
            k = k_start + k_group * tile_k * n_batch
            insts += [loadRouterWeights.cord(n, k)]
            for i in range(n_batch):
                insts += [loadHidden.cord(m, k + i * tile_k)]
        insts += [reduceRouterScores.cord(m, n)]

    return insts

def moe_expert_gemm(sm: int):
    tile_m, tile_n, tile_k = Atom.MNK
    n_batch = Atom.n_batch
    m_tiles = M // tile_m
    k_per_fold = K // fold
    tiles_per_m = m_tiles
    total_workers = tiles_per_m * N * fold

    insts = []
    for worker in range(sm, total_workers, num_sms):
        tmp = worker
        m_tile = tmp % m_tiles
        tmp //= m_tiles
        expert = tmp % N
        fold_idx = tmp // N
        m = m_tile * tile_m
        k_start = fold_idx * k_per_fold

        insts += [Atom(k_per_fold // tile_k)]
        for k_group in range(k_per_fold // (tile_k * n_batch)):
            k = k_start + k_group * tile_k * n_batch
            insts += [
                loadHidden.cord(m, k),
                loadRouterWeights.cord(expert, k),
            ]
        insts += [reduceRouterScores.cord(m, expert)]

    return insts

gpu = torch.device("cuda")
torch.manual_seed(0)
dtype = torch.bfloat16

Atom = Gemm_M64N64K64
TileM, TileN, TileK = Atom.MNK
M = 8192
N = 64
K = 128
num_sms = 128
fold = 2

hidden_states = torch.rand(M, K, dtype=dtype, device=gpu) - 0.5
router_weights = torch.rand(N, K, dtype=dtype, device=gpu) - 0.5
router_scores = torch.zeros(M, N, dtype=dtype, device=gpu)
dae = Launcher(num_sms, device=gpu)
loadHidden = TmaTensor(dae, hidden_states).wgmma_load(TileM, TileK, Major.K)
loadRouterWeights = TmaTensor(dae, router_weights).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
reduceRouterScores = TmaTensor(dae, router_scores).wgmma("reduce", TileM, TileN, Major.K)

# routing
dae.i(routing_gemm, TerminateC(), TerminateM())
dae_app(dae)

ref = hidden_states @ router_weights.t()
res = router_scores
tensor_diff("routing_gemm", ref, res)

# top-k: pick best expert per token and normalize weights
TOP_K = 1
router_probs = torch.softmax(router_scores.float(), dim=1)
expert_weights, expert_ids = torch.topk(router_probs, TOP_K, dim=1)
expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)
expert_ids = expert_ids.to(torch.int32)

expert_hidden_dim = 256
expert_w1 = torch.rand(N, K, expert_hidden_dim, dtype=dtype, device=gpu) - 0.5
expert_w2 = torch.rand(N, expert_hidden_dim, K, dtype=dtype, device=gpu) - 0.5

# expert dispatch
moe_output = torch.zeros_like(hidden_states)
for expert in range(N):
    mask = (expert_ids == expert).any(dim=1)
    x = hidden_states[mask]
    if x.shape[0] == 0:
        continue
    h = torch.relu(x @ expert_w1[expert])
    y = h @ expert_w2[expert]
    tok_weights = expert_weights[mask]
    moe_output[mask] += (y * tok_weights).to(dtype)

# ref for checking
ref_moe = torch.zeros_like(hidden_states)
for expert in range(N):
    mask = (expert_ids == expert).any(dim=1)
    x = hidden_states[mask]
    if x.shape[0] == 0:
        continue
    h = torch.relu(x @ expert_w1[expert])
    y = h @ expert_w2[expert]
    tok_weights = expert_weights[mask]
    ref_moe[mask] += (y * tok_weights).to(dtype)

tensor_diff("moe_expert", ref_moe, moe_output)