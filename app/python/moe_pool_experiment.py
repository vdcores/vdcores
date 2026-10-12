import torch
import dae.nvshmem as nvshmem

from dae.instructions import *
from dae.instructions import MetadataTmaLoadND
from dae.launcher import *
from dae.tma_utils import bytes2slots
from dae.runtime import build_tma_desc
from dae.schedule import SchedGemm, SchedSmemSiLUInterleaved, SchedGemv
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
M = 128
N = 64
K = 512
num_sms = 128
fold = 2

runtime = nvshmem.init(symmetric_size="512M")
NUM_PES = runtime.num_pes
LOCAL_PE = runtime.pe

hidden_states = nvshmem.zeros(M, K, dtype=dtype)
hidden_states.copy_(torch.rand_like(hidden_states) - 0.5)

router_weights = torch.rand(N, K, dtype=dtype, device=gpu) - 0.5
router_scores = torch.zeros(M, N, dtype=dtype, device=gpu)

expert_w1 = torch.rand(
    N, expert_hidden_dim, K, dtype=dtype, device=gpu
) - 0.5
expert_gate = torch.rand(
    N, expert_hidden_dim, K, dtype=dtype, device=gpu
) - 0.5
expert_w2 = torch.rand(
    N, K, expert_hidden_dim, dtype=dtype, device=gpu
) - 0.5


def _weight_desc(mat):
    # mat is [expert, rows, cols]. The load issues [expert, row, col].
    expert, rows, cols = mat.shape
    laid = torch.nn.functional.pad(mat.permute(2, 1, 0), (0, 7)).contiguous()
    esize = mat.element_size()
    expert_dim = laid.shape[-1]
    global_dims = [expert_dim, rows, cols]
    strides = [expert_dim * esize, expert_dim * rows * esize]
    box = [8, 1, 1]
    desc = build_tma_desc(laid, global_dims, strides, box, [1, 1, 1], 0, 0)
    nbytes = esize
    for dim in box:
        nbytes *= dim
    return {
        "desc": desc,
        "tensor": laid,
        "nbytes": nbytes,
        "slots": bytes2slots(nbytes),
    }


# One descriptor for each full weight tensor, created once. The load does not
# build these. TopK writes the selected expert ids into expert_metadata.
ffn_up_desc = _weight_desc(expert_w1)
ffn_gate_desc = _weight_desc(expert_gate)
ffn_down_desc = _weight_desc(expert_w2)

MAX_TOKENS_PER_EXPERT = M * NUM_PES
NUM_SIGNALS = N * NUM_PES
DUMMY_SIGNAL_X = 0
DUMMY_SIGNAL_META = 1
signals = nvshmem.init_signal_space(NUM_SIGNALS)

recv_x = nvshmem.zeros(
    N * NUM_PES * MAX_TOKENS_PER_EXPERT * K,
    dtype=dtype,
)

recv_meta = nvshmem.zeros(
    N * NUM_PES * MAX_TOKENS_PER_EXPERT * 2,
    dtype=torch.int32,
)

recv_count = nvshmem.zeros(N * NUM_PES, dtype=torch.int32)

send_count = torch.zeros(N, dtype=torch.int32, device=gpu)

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
        n = n_tile * tile_m
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

loadHidden = TmaTensor(dae, hidden_states).wgmma_load(
    TileM, TileK, Major.K
)
loadRouterWeights = TmaTensor(dae, router_weights).wgmma_load(
    TileN, TileK * Atom.n_batch, Major.K
)
reduceRouterScores = TmaTensor(dae, router_scores).wgmma(
    "reduce", TileM, TileN, Major.K
)

dae.i(routing_gemm, TerminateC(), TerminateM())
dae_app(dae)

router_scores = router_scores.float()
router_probs = torch.softmax(router_scores, dim=1)
expert_weights, expert_ids = torch.topk(router_probs, 1, dim=1)
# Selected expert ids. meta_idx picks an entry; the load does not store the id.
expert_metadata = torch.unique(expert_ids[:, 0].to(torch.int64), sorted=True).to(torch.uint32)

EXPERT_TO_PE = {e: e % NUM_PES for e in range(N)}

expert_dispatch = {}

for e in range(N):
    token_ids = torch.nonzero(
        expert_ids[:, 0] == e,
        as_tuple=True,
    )[0]

    num_tokens = token_ids.numel()
    start_slot = int(send_count[e].item())
    end_slot = start_slot + num_tokens

    if end_slot > MAX_TOKENS_PER_EXPERT:
        raise RuntimeError(f"Too many tokens assigned to expert {e}")

    send_count[e] = end_slot

    expert_dispatch[e] = {
        "tokens": token_ids,
        "slots": torch.arange(
            start_slot,
            end_slot,
            dtype=torch.int64,
            device=gpu,
        ),
        "source_pe": LOCAL_PE,
        "pe": EXPERT_TO_PE[e],
        "remote": EXPERT_TO_PE[e] != LOCAL_PE,
    }

local_experts = {}
remote_experts = {}

for e, info in expert_dispatch.items():
    if info["remote"]:
        remote_experts[e] = info
    else:
        local_experts[e] = info

remote_experts_nonzero = {
    e: info
    for e, info in remote_experts.items()
    if info["tokens"].numel() > 0
}

ref_moe = torch.zeros_like(hidden_states)

for e, info in local_experts.items():
    token_ids = info["tokens"]

    if token_ids.numel() == 0:
        continue

    x = hidden_states[token_ids]

    h = (
        torch.nn.functional.silu(x @ expert_w1[e].t())
        * (x @ expert_gate[e].t())
    )

    y = h @ expert_w2[e].t()

    ref_moe[token_ids] = (
        y * expert_weights[token_ids]
    ).to(dtype)

from dae.runtime import config as _runtime_config

MAX_COMM_INSTS = _runtime_config.max_comm_insts
PUTS_PER_EXPERT_SEND = 3
EXPERTS_PER_SEND_CHUNK = max(
    1,
    (MAX_COMM_INSTS - 1) // PUTS_PER_EXPERT_SEND,
)

remote_items = list(remote_experts.items())

for chunk_start in range(
    0,
    len(remote_items),
    EXPERTS_PER_SEND_CHUNK,
):
    chunk = remote_items[
        chunk_start:chunk_start + EXPERTS_PER_SEND_CHUNK
    ]

    send_dae = Launcher(
        1,
        device=gpu,
        signal_array=signals,
    )

    send_dae.new_bar(1)

    for e, info in chunk:
        token_ids = info["tokens"]
        target_pe = info["pe"]
        num_tokens = token_ids.numel()

        if num_tokens == 0:
            count_offset = e * NUM_PES + LOCAL_PE
            recv_count[count_offset] = 0

            send_dae.i(
                NvshmemPut(
                    address=recv_count.data_ptr()
                    + count_offset * recv_count.element_size(),
                    nbytes=recv_count.element_size(),
                    target_pe=target_pe,
                    signal_id=e * NUM_PES + LOCAL_PE,
                )
            )
            continue

        slots = info["slots"]

        x_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
            + LOCAL_PE * MAX_TOKENS_PER_EXPERT * K
        )

        meta_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * 2
            + LOCAL_PE * MAX_TOKENS_PER_EXPERT * 2
        )

        for i in range(num_tokens):
            slot = int(slots[i].item())
            token_id = int(token_ids[i].item())

            recv_x[
                x_offset + slot * K:
                x_offset + (slot + 1) * K
            ].copy_(hidden_states[token_id])

            recv_meta[meta_offset + slot * 2] = token_id
            recv_meta[meta_offset + slot * 2 + 1] = e

        send_dae.i(
            NvshmemPut(
                address=recv_x.data_ptr()
                + x_offset * recv_x.element_size(),
                nbytes=num_tokens * K * recv_x.element_size(),
                target_pe=target_pe,
                signal_id=DUMMY_SIGNAL_X,
            )
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

        count_offset = e * NUM_PES + LOCAL_PE
        recv_count[count_offset] = num_tokens

        send_dae.i(
            NvshmemPut(
                address=recv_count.data_ptr()
                + count_offset * recv_count.element_size(),
                nbytes=recv_count.element_size(),
                target_pe=target_pe,
                signal_id=e * NUM_PES + LOCAL_PE,
            )
        )

    send_dae.i(TerminateC(), TerminateM())
    dae_app(send_dae)

MPI.COMM_WORLD.Barrier()

wait_pairs = [
    (e, source_pe)
    for e in range(N)
    if EXPERT_TO_PE[e] == LOCAL_PE
    for source_pe in range(NUM_PES)
    if source_pe != LOCAL_PE
]

WAITS_PER_CHUNK = max(1, MAX_COMM_INSTS - 1)

for chunk_start in range(
    0,
    len(wait_pairs),
    WAITS_PER_CHUNK,
):
    chunk = wait_pairs[
        chunk_start:
        chunk_start + WAITS_PER_CHUNK
    ]

    wait_dae = Launcher(
        1,
        device=gpu,
        signal_array=signals,
    )

    wait_dae.new_bar(1)

    for e, source_pe in chunk:
        wait_dae.i(
            NvshmemWait(
                signal_id=e * NUM_PES + source_pe
            )
        )

    wait_dae.i(TerminateC(), TerminateM())
    dae_app(wait_dae)

torch.cuda.synchronize(gpu)
signals.zero_()
torch.cuda.synchronize(gpu)

MPI.COMM_WORLD.Barrier()


class _WeightDesc:
    def __init__(self, arg, size, num_slots):
        self.arg = arg
        self.size = size
        self.num_slots = num_slots


class ExpertWeightLoad(MetadataTmaLoadND):
    # The expert id is metadata_array[meta_idx] and is the first coordinate.
    def __init__(self, desc, meta_idx):
        super().__init__(desc, meta_idx, [0, 0])

    def cord(self, n, k):
        return super().cord(int(n), int(k))

    def cord2tma(self, n, k):
        return [int(n), int(k)]


def _bind_expert_weight(dae, weight_desc, meta_idx=0):
    return ExpertWeightLoad(
        _WeightDesc(
            dae.new_tma(weight_desc["desc"]),
            weight_desc["nbytes"],
            weight_desc["slots"],
        ),
        meta_idx,
    )


def run_vdcores_expert(x, e):
    real_count = x.shape[0]
    count = ((real_count + TileM - 1) // TileM) * TileM
    meta_idx = int((expert_metadata == int(e)).nonzero(as_tuple=False)[0].item())

    x_padded = torch.zeros(count, K, dtype=dtype, device=gpu)
    x_padded[:real_count].copy_(x)

    gate_out = torch.zeros(
        count, expert_hidden_dim, dtype=dtype, device=gpu
    )
    up_out = torch.zeros(
        count, expert_hidden_dim, dtype=dtype, device=gpu
    )
    silu_out = torch.zeros(
        count, expert_hidden_dim, dtype=dtype, device=gpu
    )
    out = torch.zeros(count, K, dtype=dtype, device=gpu)

    dae = Launcher(128, device=gpu, metadata=expert_metadata)

    load_x = TmaTensor(dae, x_padded).wgmma_load(
        TileM, TileK, Major.K
    )
    load_gate = _bind_expert_weight(dae, ffn_gate_desc, meta_idx)
    load_up = _bind_expert_weight(dae, ffn_up_desc, meta_idx)

    gate_reduce = TmaTensor(dae, gate_out).wgmma(
        "reduce", TileM, TileN, Major.K
    )
    up_reduce = TmaTensor(dae, up_out).wgmma(
        "reduce", TileM, TileN, Major.K
    )

    gate = SchedGemm(
        Atom,
        MNK=(count, expert_hidden_dim, K),
        tmas=(load_x, load_gate, gate_reduce),
        group=False,
    ).place(32)

    up = SchedGemm(
        Atom,
        MNK=(count, expert_hidden_dim, K),
        tmas=(load_x, load_up, up_reduce),
        group=False,
    ).place(32)

    dae.s([gate, up])
    dae.i(TerminateC(), TerminateM())
    dae_app(dae)
    torch.cuda.synchronize(gpu)

    silu_out = torch.nn.functional.silu(gate_out) * up_out

    dae = Launcher(128, device=gpu, metadata=expert_metadata)

    silu_load = TmaTensor(dae, silu_out).wgmma_load(
        TileM, TileK, Major.K
    )
    load_down = _bind_expert_weight(dae, ffn_down_desc, meta_idx)
    out_reduce = TmaTensor(dae, out).wgmma(
        "reduce", TileM, TileN, Major.K
    )

    down = SchedGemm(
        Atom,
        MNK=(count, K, expert_hidden_dim),
        tmas=(silu_load, load_down, out_reduce),
        group=False,
    ).place(32)

    dae.i(down)
    dae.i(TerminateC(), TerminateM())
    dae_app(dae)
    torch.cuda.synchronize(gpu)

    ref_h = torch.nn.functional.silu(
        gate_out[:real_count]
    ) * up_out[:real_count]

    ref_y = ref_h @ expert_w2[e].t()

    tensor_diff("expert", ref_y, out[:real_count])

    return out[:real_count]


pending_returns = []

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
            x_offset:
            x_offset + MAX_TOKENS_PER_EXPERT * K
        ].view(MAX_TOKENS_PER_EXPERT, K)

        y = run_vdcores_expert(x[:count], e)

        combine_offset = (
            e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
            + source_pe * MAX_TOKENS_PER_EXPERT * K
        )

        combine_y[
            combine_offset:
            combine_offset + count * K
        ].copy_(y.reshape(-1))

        pending_returns.append(
            (e, source_pe, count, combine_offset)
        )

torch.cuda.synchronize(gpu)

for chunk_start in range(
    0,
    len(pending_returns),
    RETURNS_PER_CHUNK if "RETURNS_PER_CHUNK" in globals() else max(1, MAX_COMM_INSTS - 1),
):
    chunk = pending_returns[
        chunk_start:
        chunk_start + (
            RETURNS_PER_CHUNK
            if "RETURNS_PER_CHUNK" in globals()
            else max(1, MAX_COMM_INSTS - 1)
        )
    ]

    return_dae = Launcher(
        1,
        device=gpu,
        signal_array=signals,
    )

    return_dae.new_bar(1)

    for e, source_pe, count, combine_offset in chunk:
        return_dae.i(
            NvshmemPut(
                address=combine_y.data_ptr()
                + combine_offset * combine_y.element_size(),
                nbytes=count * K * combine_y.element_size(),
                target_pe=source_pe,
                signal_id=e * NUM_PES + source_pe,
            )
        )

    return_dae.i(TerminateC(), TerminateM())
    dae_app(return_dae)

torch.cuda.synchronize(gpu)

MPI.COMM_WORLD.Barrier()

return_wait_pairs = [
    (e, LOCAL_PE)
    for e, info in remote_experts_nonzero.items()
]

RETURNS_PER_CHUNK = max(1, MAX_COMM_INSTS - 1)

for chunk_start in range(
    0,
    len(return_wait_pairs),
    RETURNS_PER_CHUNK,
):
    chunk = return_wait_pairs[
        chunk_start:
        chunk_start + RETURNS_PER_CHUNK
    ]

    wait_return_dae = Launcher(
        1,
        device=gpu,
        signal_array=signals,
    )

    wait_return_dae.new_bar(1)

    for e, source_pe in chunk:
        wait_return_dae.i(
            NvshmemWait(
                signal_id=e * NUM_PES + source_pe
            )
        )

    wait_return_dae.i(TerminateC(), TerminateM())
    dae_app(wait_return_dae)

torch.cuda.synchronize(gpu)

for e, info in remote_experts_nonzero.items():
    token_ids = info["tokens"]
    count = token_ids.numel()

    combine_offset = (
        e * NUM_PES * MAX_TOKENS_PER_EXPERT * K
        + LOCAL_PE * MAX_TOKENS_PER_EXPERT * K
    )

    y = combine_y[
        combine_offset:
        combine_offset + count * K
    ].view(count, K)

    for i, token_id in enumerate(token_ids.tolist()):
        ref_moe[token_id] = (
            y[i] * expert_weights[token_id]
        ).to(dtype)

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
        torch.nn.functional.silu(
            x @ expert_w1[e].t()
        )
        * (x @ expert_gate[e].t())
    )

    y = h @ expert_w2[e].t()

    full_ref_moe[token_ids] = (
        y * expert_weights[token_ids]
    ).to(dtype)

diff = tensor_diff(
    "distributed MoE",
    ref_moe,
    full_ref_moe,
)

print(f"MoE output diff: {diff}", flush=True)
print("end of file")