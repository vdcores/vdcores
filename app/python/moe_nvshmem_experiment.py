import torch

from dae.instructions import *
from dae.instructions import MetadataTmaLoadND
from dae.launcher import *
from dae.tma_utils import build_tma_wgmma_kmajor, bytes2slots
from dae.schedule import SchedGemm, SchedSmemSiLUInterleaved, SchedGemv
from dae.util import dae_app, tensor_diff


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
dtype = torch.bfloat16

Atom = Gemm_M64N64K64
TileM, TileN, TileK = Atom.MNK
M = 128
N = 64
K = 512
num_sms = 128
fold = 2

if __name__ == "__main__":
    import dae.nvshmem as nvshmem
    from mpi4py import MPI

    torch.manual_seed(0)
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
    """Pack a logical [expert, output, input] BF16 CUDA view once.

    Transpose [expert, input, output] storage before calling. Split fused
    up/gate weights into logical views first; no layout or ordering is inferred.
    Retain and reuse this descriptor; later edits to an unpacked source view
    are not reflected in its contiguous copy.
    """
    # Full [expert, rows, cols] tensor. The K-major map is
    # [64, rows, cols/64, expert], so one descriptor covers every expert.
    if mat.ndim != 3 or mat.dtype != torch.bfloat16 or mat.device.type != "cuda":
        raise ValueError("expert weights must be 3D BF16 CUDA tensors")
    if any(dim <= 0 for dim in mat.shape) or mat.shape[1] % TileN or mat.shape[2] % TileK:
        raise ValueError("expert weight dimensions must be positive full GEMM tiles")
    mat = mat.contiguous()
    rank, desc = build_tma_wgmma_kmajor(mat, TileK, TileN)
    assert rank == 4, rank
    nbytes = mat.element_size() * TileN * TileK
    return {
        "weight": mat,
        "desc": desc,
        "nbytes": nbytes,
        "slots": bytes2slots(nbytes),
    }


if __name__ == "__main__":
    # One descriptor for each full weight tensor, created once.
    ffn_up_desc = _weight_desc(expert_w1)
    ffn_gate_desc = _weight_desc(expert_gate)
    ffn_down_desc = _weight_desc(expert_w2)

    MAX_TOKENS_PER_EXPERT = M * NUM_PES
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


if __name__ == "__main__":
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


def _expert_tokens(expert_ids, expert_weights, expert):
    # One token can select this expert in only one top-k slot.
    token_ids = []
    token_weights = []
    for slot in range(expert_ids.shape[1]):
        chosen = torch.nonzero(expert_ids[:, slot] == expert, as_tuple=True)[0]
        if chosen.numel() == 0:
            continue
        token_ids.append(chosen)
        token_weights.append(expert_weights[chosen, slot])
    if not token_ids:
        empty_ids = torch.empty(0, dtype=torch.int64, device=expert_ids.device)
        empty_w = torch.empty(0, dtype=expert_weights.dtype, device=expert_weights.device)
        return empty_ids, empty_w
    return torch.cat(token_ids), torch.cat(token_weights)


# k=1 is the original route. 2 and 4 are in between. 8 is wide but still
# one selection per expert, and it fits in the per-expert recv buffer.
TOP_KS = (1, 2, 4, 8)
if __name__ == "__main__":
    EXPERT_TO_PE = {e: e % NUM_PES for e in range(N)}

class _WeightDesc:
    def __init__(self, arg, size, num_slots):
        self.arg = arg
        self.size = size
        self.num_slots = num_slots


class ExpertWeightLoad(MetadataTmaLoadND):
    # Static coordinates are the K-major tile. metadata[0] is appended and is
    # the expert coordinate of the full weight tensor.
    def __init__(self, desc):
        super().__init__(desc, 0, [0, 0, 0], metadata_last=True)

    def cord(self, n, k):
        return super().cord(0, int(n), int(k) // 64)

    def cord2tma(self, n, k):
        return [0, int(n), int(k) // 64]


def _bind_expert_weight(dae, weight_desc):
    return ExpertWeightLoad(
        _WeightDesc(
            dae.new_tma(weight_desc["desc"]),
            weight_desc["nbytes"],
            weight_desc["slots"],
        ),
    )


def _stage_expert(expert_id, expert_metadata):
    expert_metadata.copy_(
        torch.tensor([expert_id], dtype=torch.uint32, device=expert_metadata.device)
    )


def run_vdcores_expert(x, e, weight_descs=None, *, app=False):
    """Run silu(up) * gate, then down, for one routed expert.

    Pass descriptors from _weight_desc in (up, gate, down) order. Shapes are
    [E, I, D], [E, I, D], [E, O, I]; all widths must be multiples of 64.
    Omit descriptors only in the initialized experiment. app=True preserves
    the experiment's dae_app CLI; standalone calls launch each stage once.
    """
    # Build descriptors once with _weight_desc; reuse them for every routed expert.
    # The activation convention remains silu(up) * gate.
    if weight_descs is None and "ffn_up_desc" not in globals():
        raise ValueError("standalone calls require up, gate and down descriptors")
    up_desc, gate_desc, down_desc = (
        (ffn_up_desc, ffn_gate_desc, ffn_down_desc)
        if weight_descs is None else weight_descs
    )
    up_weight = up_desc["weight"]
    gate_weight = gate_desc["weight"]
    down_weight = down_desc["weight"]
    for weight in (up_weight, gate_weight, down_weight):
        if weight.ndim != 3 or weight.dtype != torch.bfloat16 or weight.device.type != "cuda":
            raise ValueError("expert weights must be 3D BF16 CUDA tensors")
        if weight.device != up_weight.device:
            raise ValueError("expert weights must be on the same device")
        if any(dim <= 0 for dim in weight.shape) or weight.shape[1] % TileN or weight.shape[2] % TileK:
            raise ValueError("expert weight dimensions must be positive full GEMM tiles")
    num_experts, hidden_dim, input_dim = up_weight.shape
    output_dim = down_weight.shape[1]
    if gate_weight.shape != up_weight.shape or down_weight.shape != (num_experts, output_dim, hidden_dim):
        raise ValueError("up, gate and down expert dimensions do not match")
    if x.ndim != 2 or x.shape[1] != input_dim or x.dtype != up_weight.dtype or x.device != up_weight.device:
        raise ValueError("input must match the expert weight width, dtype and device")
    if not isinstance(e, int) or not 0 <= e < num_experts:
        raise ValueError("expert id is out of range")
    if x.shape[0] == 0:
        return x.new_empty((0, output_dim))
    real_count = x.shape[0]
    count = ((real_count + TileM - 1) // TileM) * TileM
    # Each call owns its metadata, shared by all three projections.
    gpu = x.device
    dtype = x.dtype
    expert_metadata = torch.empty(1, dtype=torch.uint32, device=gpu)
    _stage_expert(e, expert_metadata)

    up_fold = max(1, 32 // ((count // TileM) * (hidden_dim // TileN)))
    while 32 % up_fold or input_dim % (up_fold * TileK * Atom.n_batch):
        up_fold -= 1
    down_fold = max(1, 32 // ((count // TileM) * (output_dim // TileN)))
    while 32 % down_fold or hidden_dim % (down_fold * TileK * Atom.n_batch):
        down_fold -= 1

    x_padded = torch.zeros(count, input_dim, dtype=dtype, device=gpu)
    x_padded[:real_count].copy_(x)

    gate_out = torch.zeros(
        count, hidden_dim, dtype=dtype, device=gpu
    )
    up_out = torch.zeros(
        count, hidden_dim, dtype=dtype, device=gpu
    )
    out = torch.zeros(count, output_dim, dtype=dtype, device=gpu)

    dae = Launcher(128, device=gpu, metadata=expert_metadata)

    load_x = TmaTensor(dae, x_padded).wgmma_load(
        TileM, TileK, Major.K
    )
    load_gate = _bind_expert_weight(dae, gate_desc)
    load_up = _bind_expert_weight(dae, up_desc)

    gate_reduce = TmaTensor(dae, gate_out).wgmma(
        "reduce", TileM, TileN, Major.K
    )
    up_reduce = TmaTensor(dae, up_out).wgmma(
        "reduce", TileM, TileN, Major.K
    )

    gate = SchedGemm(
        Atom,
        MNK=(count, hidden_dim, input_dim),
        tmas=(load_x, load_gate, gate_reduce),
        fold=up_fold,
        group=False,
    ).place(32)

    up = SchedGemm(
        Atom,
        MNK=(count, hidden_dim, input_dim),
        tmas=(load_x, load_up, up_reduce),
        fold=up_fold,
        group=False,
    ).place(32)

    dae.s([gate, up])
    with torch.cuda.device(gpu):
        if app:
            dae_app(dae)
        else:
            dae.launch()
    torch.cuda.synchronize(gpu)

    silu_out = torch.nn.functional.silu(up_out) * gate_out

    dae = Launcher(128, device=gpu, metadata=expert_metadata)

    silu_load = TmaTensor(dae, silu_out).wgmma_load(
        TileM, TileK, Major.K
    )
    load_down = _bind_expert_weight(dae, down_desc)
    out_reduce = TmaTensor(dae, out).wgmma(
        "reduce", TileM, TileN, Major.K
    )

    down = SchedGemm(
        Atom,
        MNK=(count, output_dim, hidden_dim),
        tmas=(silu_load, load_down, out_reduce),
        fold=down_fold,
        group=False,
    ).place(32)

    dae.i(down)
    dae.i(TerminateC(), TerminateM())
    with torch.cuda.device(gpu):
        if app:
            dae_app(dae)
        else:
            dae.launch()
    torch.cuda.synchronize(gpu)

    ref_h = (
        torch.nn.functional.silu(x @ up_weight[e].t())
        * (x @ gate_weight[e].t())
    )
    ref_y = ref_h @ down_weight[e].t()

    tensor_diff("expert", ref_y, out[:real_count])

    return out[:real_count]



def run_distributed(top_k):
    send_count.zero_()
    recv_count.zero_()
    combine_y.zero_()
    signals.zero_()
    torch.cuda.synchronize(gpu)
    MPI.COMM_WORLD.Barrier()
    print(f"pe={LOCAL_PE} top_k={top_k} start", flush=True)

    expert_weights, expert_ids = torch.topk(router_probs, top_k, dim=1)
    if expert_ids.shape[1] != top_k:
        raise RuntimeError(
            f"pe={LOCAL_PE} top_k={top_k} selected {expert_ids.shape[1]} experts"
        )
    expert_dispatch = {}

    for e in range(N):
        token_ids, token_weights = _expert_tokens(expert_ids, expert_weights, e)

        num_tokens = token_ids.numel()
        start_slot = int(send_count[e].item())
        end_slot = start_slot + num_tokens

        if end_slot > MAX_TOKENS_PER_EXPERT:
            raise RuntimeError(f"Too many tokens assigned to expert {e}")

        send_count[e] = end_slot

        expert_dispatch[e] = {
            "tokens": token_ids,
            "weights": token_weights,
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

    n_assign = sum(info["tokens"].numel() for info in expert_dispatch.values())
    if n_assign != M * top_k:
        raise RuntimeError(
            f"pe={LOCAL_PE} top_k={top_k} dispatched {n_assign} "
            f"assignments, expected {M * top_k}"
        )

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

        w = info["weights"].to(dtype).unsqueeze(1)
        ref_moe[token_ids] = (ref_moe[token_ids] + y * w).to(dtype)

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

            y = run_vdcores_expert(x[:count], e, app=True)

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

        w = info["weights"]
        for i, token_id in enumerate(token_ids.tolist()):
            ref_moe[token_id] = (ref_moe[token_id] + y[i] * w[i]).to(dtype)

    full_ref_moe = torch.zeros_like(hidden_states)

    for e in range(N):
        token_ids, token_weights = _expert_tokens(expert_ids, expert_weights, e)

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

        w = token_weights.to(dtype).unsqueeze(1)
        full_ref_moe[token_ids] = (full_ref_moe[token_ids] + y * w).to(dtype)

    tensor_diff(
        f"distributed MoE k={top_k}",
        ref_moe,
        full_ref_moe,
    )
    scale = ref_moe.abs().float().mean().item()
    pct = (ref_moe - full_ref_moe).abs().float().mean().item() / scale * 100
    MPI.COMM_WORLD.Barrier()
    print(
        f"pe={LOCAL_PE} top_k={top_k} distributed diff={pct:.4f}%",
        flush=True,
    )
    MPI.COMM_WORLD.Barrier()

if __name__ == "__main__":
    for top_k in TOP_KS:
        run_distributed(top_k)
    print("end of file")
