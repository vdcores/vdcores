import torch
import dae.nvshmem as nvshmem

from dae.instructions import NvshmemPut, NvshmemWait
from dae.launcher import *
from dae.util import dae_app, tensor_diff
from mpi4py import MPI


def gemv(dae, A, x, out, M, N, K, num_sms):
    A_t = TmaTensor(dae, A).wgmma_load(M, K, Major.K)
    x_t = TmaTensor(dae, x).wgmma_load(1, K, Major.K)
    out_t = TmaTensor(dae, out).wgmma("reduce", 1, M, Major.MN)
    op = SchedGemv(
        Gemv,
        MNK=(M, 1, K),
        tmas=(A_t, x_t, out_t),
        exec=False,
    ).place(num_sms)
    return op

expert_hidden_dim = 256
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

expert_w1 = torch.rand(N, expert_hidden_dim, K, dtype=dtype, device=gpu) - 0.5
expert_gate = torch.rand(N, expert_hidden_dim, K, dtype=dtype, device=gpu) - 0.5
expert_w2 = torch.rand(N, K, expert_hidden_dim, dtype=dtype, device=gpu) - 0.5

runtime = nvshmem.init(symmetric_size="512M")

NUM_PES = runtime.num_pes
LOCAL_PE = runtime.pe
MAX_TOKENS_PER_EXPERT = M
NUM_SIGNALS = N * NUM_PES
DUMMY_SIGNAL_X = NUM_SIGNALS
DUMMY_SIGNAL_META = NUM_SIGNALS + 1
signals = nvshmem.init_signal_space(NUM_SIGNALS + 2)

recv_x = nvshmem.zeros(
    N * NUM_PES * MAX_TOKENS_PER_EXPERT * K,
    dtype=dtype,
)

recv_meta = nvshmem.zeros(
    N * NUM_PES * MAX_TOKENS_PER_EXPERT * 2,
    dtype=torch.int32,
)

recv_count = nvshmem.zeros(
    N * NUM_PES,
    dtype=torch.int32,
)

send_count = torch.zeros(
    N,
    dtype=torch.int32,
    device=gpu,
)

# fixed output buffer
combine_y = nvshmem.zeros(
    N * NUM_PES * MAX_TOKENS_PER_EXPERT * K,
    dtype=dtype,
)


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
            insts += [
                loadHidden.cord(m, k),
                loadRouterWeights.cord(n, k),
            ]
        insts += [reduceRouterScores.cord(m, n)]

    return insts

dae = Launcher(num_sms, device=gpu)
loadHidden = TmaTensor(dae, hidden_states).wgmma_load(TileM, TileK, Major.K)
loadRouterWeights = TmaTensor(dae, router_weights).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
reduceRouterScores = TmaTensor(dae, router_scores).wgmma("reduce", TileM, TileN, Major.K)
dae.i(routing_gemm, TerminateC(), TerminateM())
dae_app(dae)

router_scores = router_scores.float()
router_probs = torch.softmax(router_scores.float(), dim=1)
expert_weights, expert_ids = torch.topk(router_probs, 1, dim=1)


EXPERT_TO_PE = {
    e: e % NUM_PES
    for e in range(N)
}

expert_dispatch = {}

for e in range(N):
    token_mask = (expert_ids[:, 0] == e)

    if token_mask.sum() == 0:
        continue

    token_ids = torch.nonzero(
        token_mask,
        as_tuple=True,
    )[0]

    num_tokens = token_ids.numel()

    # reserve slots for this expert
    start_slot = send_count[e].item()
    end_slot = start_slot + num_tokens

    if end_slot > MAX_TOKENS_PER_EXPERT:
        raise RuntimeError(
            f"Too many tokens assigned to expert {e}"
        )

    send_count[e] = end_slot

    expert_dispatch[e] = {
        "tokens": token_ids,
        "slots": torch.arange(
            start_slot,
            end_slot,
            dtype=torch.int32,
            device=gpu,
        ),
        "source_pe": LOCAL_PE,
        "pe": EXPERT_TO_PE[e],
        "remote": EXPERT_TO_PE[e] != LOCAL_PE,
    }

print("\n--- expert dispatch ---")
for e, info in expert_dispatch.items():
    print(
        f"expert {e}: "
        f"num_tokens={len(info['tokens'])} "
        f"PE={info['pe']} "
        f"remote={info['remote']}"
    )

# separate local and remote experts
local_experts = {}
remote_experts = {}

for e, info in expert_dispatch.items():
    if info["remote"]:
        remote_experts[e] = info
    else:
        local_experts[e] = info


print("\n--- local experts ---")
for e, info in local_experts.items():
    print(
        f"expert {e}: "
        f"num_tokens={len(info['tokens'])}"
    )

print("\n--- remote experts ---")
for e, info in remote_experts.items():
    print(
        f"expert {e}: "
        f"num_tokens={len(info['tokens'])} "
        f"target PE={info['pe']}"
    )

ref_moe = torch.zeros_like(hidden_states)

# run local expert computation
for e, info in local_experts.items():
    token_ids = info["tokens"]
    x = hidden_states[token_ids]
    h = (
        torch.nn.functional.silu(x @ expert_w1[e].t())
        * (x @ expert_gate[e].t())
    )
    y = h @ expert_w2[e].t()
    ref_moe[token_ids] = (
        y * expert_weights[token_ids]
    ).to(dtype)


# send remote expert tokens
send_dae = Launcher(
    1,
    device=gpu,
    signal_array=signals,
)
send_dae.new_bar(1)

for e, info in remote_experts.items():
    token_ids = info["tokens"]
    target_pe = info["pe"]
    num_tokens = token_ids.numel()
    for slot, token_id in zip(info["slots"].tolist(), token_ids.tolist()):
        x_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
            + LOCAL_PE * MAX_TOKENS_PER_EXPERT * K
            + slot * K
        )
        meta_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * 2
            + LOCAL_PE * MAX_TOKENS_PER_EXPERT * 2
            + slot * 2
        )

        # pack token into receive buffer
        recv_x[
            x_offset:x_offset + K
        ].copy_(hidden_states[token_id])

        # store token metadata
        recv_meta[meta_offset] = token_id
        recv_meta[meta_offset + 1] = 0

    # get signal index
    signal_id = e * NUM_PES + LOCAL_PE

    # send token data
    x_offset = (
        e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
        + LOCAL_PE * MAX_TOKENS_PER_EXPERT * K
    )
    send_dae.i(
        NvshmemPut(
            address=recv_x.data_ptr()
            + x_offset * recv_x.element_size(),
            nbytes=num_tokens * K * recv_x.element_size(),
            target_pe=target_pe,
            signal_id=DUMMY_SIGNAL_X,
        )
    )

    # send token metadata
    meta_offset = (
        e * NUM_PES * MAX_TOKENS_PER_EXPERT * 2
        + LOCAL_PE * MAX_TOKENS_PER_EXPERT * 2
    )
    send_dae.i(
        NvshmemPut(
            address=recv_meta.data_ptr()
            + meta_offset * recv_meta.element_size(),
            nbytes=num_tokens * 2 * recv_meta.element_size(),
            target_pe=target_pe,
            signal_id=DUMMY_SIGNAL_META,
        )
    )

    # send token count
    count_offset = e * NUM_PES + LOCAL_PE
    recv_count[count_offset] = num_tokens
    send_dae.i(
        NvshmemPut(
            address=recv_count.data_ptr()
            + count_offset * recv_count.element_size(),
            nbytes=recv_count.element_size(),
            target_pe=target_pe,
            signal_id=signal_id,
        )
    )

send_dae.i(TerminateC(), TerminateM())
dae_app(send_dae)

# wait for remote tokens
wait_dae = Launcher(
    1,
    device=gpu,
    signal_array=signals,
)
wait_dae.new_bar(1)

for e in range(N):
    if EXPERT_TO_PE[e] != LOCAL_PE:
        continue

    for source_pe in range(NUM_PES):
        if source_pe == LOCAL_PE:
            continue

        signal_id = e * NUM_PES + source_pe
        wait_dae.i(
            NvshmemWait(signal_id=signal_id)
        )

wait_dae.i(TerminateC(), TerminateM())
dae_app(wait_dae)

# run remote expert computation
return_dae = Launcher(
    1,
    device=gpu,
    signal_array=signals,
)
return_dae.new_bar(1)

for e in range(N):
    if EXPERT_TO_PE[e] != LOCAL_PE:
        continue

    for source_pe in range(NUM_PES):
        if source_pe == LOCAL_PE:
            continue

        count_idx = e * NUM_PES + source_pe
        count = int(recv_count[count_idx].item())

        if count == 0:
            continue

        x_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
            + source_pe * MAX_TOKENS_PER_EXPERT * K
        )

        meta_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * 2
            + source_pe * MAX_TOKENS_PER_EXPERT * 2
        )

        x = recv_x[
            x_offset:x_offset + count * K
        ].view(count, K)

        meta = recv_meta[
            meta_offset:meta_offset + count * 2
        ].view(count, 2)

        token_ids = meta[:, 0]

        h = (
            torch.nn.functional.silu(
                x @ expert_w1[e].t()
            )
            * (x @ expert_gate[e].t())
        )

        y = h @ expert_w2[e].t()

        # write outputs to fixed output buffer
        combine_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
            + source_pe * MAX_TOKENS_PER_EXPERT * K
        )
        combine_y[
            combine_offset:combine_offset + count * K
        ].copy_(y.reshape(-1))

        # send outputs back to source PE
        signal_id = e * NUM_PES + source_pe

        return_dae.i(
            NvshmemPut(
                address=combine_y.data_ptr()
                + combine_offset * combine_y.element_size(),
                nbytes=count * K * combine_y.element_size(),
                target_pe=source_pe,
                signal_id=signal_id,
            )
        )

return_dae.i(TerminateC(), TerminateM())
dae_app(return_dae)

# wait for returned expert outputs
return_wait_dae = Launcher(
    1,
    device=gpu,
    signal_array=signals,
)
return_wait_dae.new_bar(1)

for e in remote_experts:
    signal_id = e * NUM_PES + LOCAL_PE
    return_wait_dae.i(
        NvshmemWait(signal_id=signal_id)
    )
return_wait_dae.i(TerminateC(), TerminateM())
dae_app(return_wait_dae)

# combine remote outputs
for e, info in remote_experts.items():
    token_ids = info["tokens"]
    count = len(token_ids)
    combine_offset = (
        e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
        + LOCAL_PE * MAX_TOKENS_PER_EXPERT * K
    )

    y_e = combine_y[
        combine_offset:combine_offset + count * K
    ].view(count, K)

    for i, token_id in enumerate(token_ids.tolist()):
        ref_moe[token_id] = (
            y_e[i] * expert_weights[token_id]
        ).to(dtype)

# compute full reference MoE
full_ref_moe = torch.zeros_like(hidden_states)

for e in range(N):
    token_ids = torch.nonzero(
        expert_ids[:, 0] == e,
        as_tuple=True,
    )[0]

    if token_ids.numel() == 0:
        continue

    x = hidden_states[token_ids]

    h = (
        torch.nn.functional.silu(x @ expert_w1[e].t())
        * (x @ expert_gate[e].t())
    )

    y = h @ expert_w2[e].t()

    full_ref_moe[token_ids] = (
        y * expert_weights[token_ids]
    ).to(dtype)

diff = tensor_diff("distributed MoE", ref_moe, full_ref_moe)
print(f"MoE output diff: {diff}", flush=True)

print("REACHED END OF PYTHON FILE", flush=True)