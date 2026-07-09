import torch
import time
import matplotlib.pyplot as plt

from dae.launcher import *
from dae.util import dae_app


gpu = torch.device("cuda")
torch.manual_seed(0)
dtype = torch.bfloat16

Atom = Gemm_M64N64K64
TileM, TileN, TileK = Atom.MNK
M = 8192
N = 128
K = 2048
num_sms = 128
fold = 2

hidden_states = torch.rand(M, K, dtype=dtype, device=gpu) - 0.5
router_weights = torch.rand(N, K, dtype=dtype, device=gpu) - 0.5
router_scores = torch.zeros(M, N, dtype=dtype, device=gpu)
dae = Launcher(num_sms, device=gpu)
loadHidden = TmaTensor(dae, hidden_states).wgmma_load(TileM, TileK, Major.K)
loadRouterWeights = TmaTensor(dae, router_weights).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
reduceRouterScores = TmaTensor(dae, router_scores).wgmma("reduce", TileM, TileN, Major.K)

m_tiles = M // TileM
n_tiles = N // TileN
tiles_per_fold = m_tiles * n_tiles
total_workers = tiles_per_fold * fold
k_per_fold = K // fold

def routing_gemm(sm: int):
    insts = []
    for worker in range(sm, total_workers, num_sms):
        tile_idx = worker % tiles_per_fold
        fold_idx = worker // tiles_per_fold
        m_tile = tile_idx % m_tiles
        n_tile = tile_idx // m_tiles
        m = m_tile * TileM
        n = n_tile * TileN
        k_start = fold_idx * k_per_fold

        insts += [Atom(k_per_fold // TileK)]
        for k_group in range(k_per_fold // (TileK * Atom.n_batch)):
            k = k_start + k_group * TileK * Atom.n_batch
            insts += [loadRouterWeights.cord(n, k)]
            for i in range(Atom.n_batch):
                insts += [loadHidden.cord(m, k + i * TileK)]
        insts += [reduceRouterScores.cord(m, n)]

    return insts

# placeholder for expert gemm, uses same tensors as routing for now
# in full impl would use per-expert weights and only process assigned tokens
def moe_expert_gemm(sm: int):
    m_tiles = M // TileM
    k_per_fold = K // fold
    n_tiles_expert = N // TileN
    total_workers_expert = m_tiles * n_tiles_expert * fold
    insts = []

    for worker in range(sm, total_workers_expert, num_sms):
        tmp = worker
        m_tile = tmp % m_tiles
        tmp //= m_tiles
        n_tile = tmp % n_tiles_expert
        fold_idx = tmp // n_tiles_expert
        m = m_tile * TileM
        n = n_tile * TileN
        k_start = fold_idx * k_per_fold

        insts += [Atom(k_per_fold // TileK)]
        for k_group in range(k_per_fold // (TileK * Atom.n_batch)):
            k = k_start + k_group * TileK * Atom.n_batch
            insts += [loadHidden.cord(m, k), loadRouterWeights.cord(n, k)]
        insts += [reduceRouterScores.cord(m, n)]

    return insts

def moe_schedule_sequential(sm: int):
    return routing_gemm(sm) + moe_expert_gemm(sm)

def moe_schedule_interleaved(sm: int):
    insts = []
    for worker in range(sm, total_workers, num_sms):
        tile_idx = worker % tiles_per_fold
        fold_idx = worker // tiles_per_fold
        m_tile = tile_idx % m_tiles
        n_tile = tile_idx // m_tiles
        m = m_tile * TileM
        n = n_tile * TileN
        k_start = fold_idx * k_per_fold

        insts += [Atom(k_per_fold // TileK)]
        for k_group in range(k_per_fold // (TileK * Atom.n_batch)):
            k = k_start + k_group * TileK * Atom.n_batch
            insts += [loadRouterWeights.cord(n, k), loadHidden.cord(m, k)]
        insts += [reduceRouterScores.cord(m, n)]

        expert = n
        insts += [Atom(k_per_fold // TileK)]
        for k_group in range(k_per_fold // (TileK * Atom.n_batch)):
            k = k_start + k_group * TileK * Atom.n_batch
            insts += [loadHidden.cord(m, k), loadRouterWeights.cord(expert, k)]
        insts += [reduceRouterScores.cord(m, expert)]

    return insts

def moe_space_shared(sm: int):
    # stagger expert assignment across SMs so different experts run in parallel
    # instead of all SMs working on the same expert tile
    insts = []
    experts_per_sm_group = n_tiles
    sm_expert_offset = sm % experts_per_sm_group
    for worker in range(sm, total_workers, num_sms):
        tile_idx = worker % tiles_per_fold
        fold_idx = worker // tiles_per_fold
        m_tile = tile_idx % m_tiles
        n_tile = (tile_idx // m_tiles + sm_expert_offset) % n_tiles
        m = m_tile * TileM
        n = n_tile * TileN
        k_start = fold_idx * k_per_fold

        insts += [Atom(k_per_fold // TileK)]
        for k_group in range(k_per_fold // (TileK * Atom.n_batch)):
            k = k_start + k_group * TileK * Atom.n_batch
            insts += [loadRouterWeights.cord(n, k), loadHidden.cord(m, k)]
        insts += [reduceRouterScores.cord(m, n)]

    return insts

def moe_overlapped_store(sm: int):
    insts = []
    # first pass: all compute, no stores
    for worker in range(sm, total_workers, num_sms):
        tile_idx = worker % tiles_per_fold
        fold_idx = worker // tiles_per_fold
        m_tile = tile_idx % m_tiles
        n_tile = tile_idx // m_tiles
        m = m_tile * TileM
        n = n_tile * TileN
        k_start = fold_idx * k_per_fold
        insts += [Atom(k_per_fold // TileK)]
        for k_group in range(k_per_fold // (TileK * Atom.n_batch)):
            k = k_start + k_group * TileK * Atom.n_batch
            insts += [loadRouterWeights.cord(n, k), loadHidden.cord(m, k)]

    # second pass: all stores
    for worker in range(sm, total_workers, num_sms):
        tile_idx = worker % tiles_per_fold
        fold_idx = worker // tiles_per_fold
        m_tile = tile_idx % m_tiles
        n_tile = tile_idx // m_tiles
        m = m_tile * TileM
        n = n_tile * TileN
        insts += [reduceRouterScores.cord(m, n)]

    return insts

results = {}

def run_and_time(name, fn):
    torch.cuda.synchronize()
    t0 = time.time()
    dae.i(fn, TerminateC(), TerminateM())
    dae_app(dae)
    torch.cuda.synchronize()
    t1 = time.time()
    ms = (t1 - t0) * 1000
    results[name] = ms
    print(f"{name}: {ms:.2f} ms")

# schedule comparison
run_and_time("routing_only", routing_gemm)
run_and_time("moe_sequential", moe_schedule_sequential)
run_and_time("moe_interleaved", moe_schedule_interleaved)
run_and_time("moe_space_shared", moe_space_shared)
run_and_time("moe_overlapped_store", moe_overlapped_store)