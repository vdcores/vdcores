import argparse
import sys

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app, tensor_diff

arg_parser = argparse.ArgumentParser(add_help=False)
arg_parser.add_argument("--correctness", action="store_true")
arg_parser.add_argument("--no-overlap", action="store_true",
                        help="Serialize logical MLP operations with explicit IssueBarrier waits")
arg_parser.add_argument("--tail-store", choices=("reg", "tma"), default="reg",
                        help="Store the MLP tail gate/up projections through registers or TMA")
arg_parser.add_argument("--seed", type=int, default=0)
parsed_args, remaining_argv = arg_parser.parse_known_args()
if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
    remaining_argv = [*remaining_argv, "--launch"]
sys.argv = [sys.argv[0], *remaining_argv]

torch.manual_seed(parsed_args.seed)

gpu = torch.device("cuda")
dtype = torch.bfloat16

N = 8
HIDDEN = 4096
INTERMEDIATE = 14336
MLP_LOW = 4096
MLP_SPLIT = 6144
MLP_TAIL = INTERMEDIATE - MLP_SPLIT
MLP_TAIL_CHUNK = 4096
use_tma_tail = parsed_args.tail_store == "tma"

num_sms = 128
full_sms = 132
dae = Launcher(full_sms, device=gpu)

TileM, _, TileK = Gemv_M64N8.MNK

matHidden = torch.randn(N, HIDDEN, dtype=dtype, device=gpu)
matRMSHidden = matHidden
matGate = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=gpu)
matUp = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=gpu)
matDown = torch.randn(HIDDEN, INTERMEDIATE, dtype=dtype, device=gpu)

matInterm = torch.zeros(N, INTERMEDIATE, dtype=dtype, device=gpu)
matGateOut = torch.zeros(N, INTERMEDIATE, dtype=dtype, device=gpu)
matSiLUOut = torch.zeros(N, INTERMEDIATE, dtype=dtype, device=gpu)
matOut = torch.zeros(N, HIDDEN, dtype=dtype, device=gpu)

defaultg = dae.get_group()
defaultg.addBarrier("bar_out")
if parsed_args.no_overlap:
    defaultg.addBarrier("bar_gate_done")
    defaultg.addBarrier("bar_up_done")
    defaultg.addBarrier("bar_silu_low_done")
    defaultg.addBarrier("bar_silu_tail_done")
else:
    defaultg.addBarrier("bar_silu_in")
    defaultg.addBarrier("bar_silu_out1")
    defaultg.addBarrier("bar_silu_out2")
    if use_tma_tail:
        defaultg.addBarrier("bar_silu_tail_in")

defaultg.addTma("loadRMSLayer", [matRMSHidden], lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
defaultg.addTma("loadSiluLayer", [matSiLUOut], lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
defaultg.addTma("storeInterm", [matInterm], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("storeGateOut", [matGateOut], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("reduceInterm", [matInterm], lambda t: t.wgmma("reduce", N, TileM, Major.MN))
defaultg.addTma("reduceGateOut", [matGateOut], lambda t: t.wgmma("reduce", N, TileM, Major.MN))
defaultg.addTma("reduceOut", [matOut], lambda t: t.wgmma("reduce", N, TileM, Major.MN))
defaultg.addTma("storeSiluLayer", [matSiLUOut], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("loadDown", [matDown], lambda t: t.wgmma_load(TileM, TileK, Major.K))
defaultg.addTma("loadUp", [matUp], lambda t: t.wgmma_load(TileM, TileK, Major.K))
defaultg.addTma("loadGate", [matGate], lambda t: t.wgmma_load(TileM, TileK, Major.K))

dae.set_persistent(matRMSHidden)
dae.set_streaming(matGate, matUp, matDown)

dae.build_groups()

reg_gate, reg_up = 0, 1
regStoreGate = RegStore(reg_gate, matGateOut[:, 0:TileM])
regStoreUp = RegStore(reg_up, matInterm[:, 0:TileM])

gate_proj_low = SchedGemv(
    Gemv_M64N8,
    MNK=(MLP_LOW, N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
)
gate_proj_high = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_LOW, MLP_SPLIT - MLP_LOW), N, HIDDEN),
    tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["reduceGateOut"]),
)
up_proj_low = SchedGemv(
    Gemv_M64N8,
    MNK=(MLP_LOW, N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
)
up_proj_high = SchedGemv(
    Gemv_M64N8,
    MNK=((MLP_LOW, MLP_SPLIT - MLP_LOW), N, HIDDEN),
    tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["reduceInterm"]),
)

silu1 = SchedSmemSiLUInterleaved(
    num_token=N,
    gate_glob=matGateOut[:, :MLP_SPLIT],
    up_glob=matInterm[:, :MLP_SPLIT],
    out_glob=matSiLUOut[:, :MLP_SPLIT],
)

if use_tma_tail:
    gate_proj_tail0 = SchedGemv(
        Gemv_M64N8,
        MNK=((MLP_SPLIT, MLP_TAIL_CHUNK), N, HIDDEN),
        tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
    )
    gate_proj_tail1 = SchedGemv(
        Gemv_M64N8,
        MNK=((MLP_SPLIT + MLP_TAIL_CHUNK, MLP_TAIL_CHUNK), N, HIDDEN),
        tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], defaultg["storeGateOut"]),
    )
    up_proj_tail0 = SchedGemv(
        Gemv_M64N8,
        MNK=((MLP_SPLIT, MLP_TAIL_CHUNK), N, HIDDEN),
        tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
    )
    up_proj_tail1 = SchedGemv(
        Gemv_M64N8,
        MNK=((MLP_SPLIT + MLP_TAIL_CHUNK, MLP_TAIL_CHUNK), N, HIDDEN),
        tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], defaultg["storeInterm"]),
    )
    silu_tail0 = SchedSmemSiLUInterleaved(
        num_token=N,
        gate_glob=matGateOut[:, MLP_SPLIT:MLP_SPLIT + MLP_TAIL_CHUNK],
        up_glob=matInterm[:, MLP_SPLIT:MLP_SPLIT + MLP_TAIL_CHUNK],
        out_glob=matSiLUOut[:, MLP_SPLIT:MLP_SPLIT + MLP_TAIL_CHUNK],
    )
    silu_tail1 = SchedSmemSiLUInterleaved(
        num_token=N,
        gate_glob=matGateOut[:, MLP_SPLIT + MLP_TAIL_CHUNK:INTERMEDIATE],
        up_glob=matInterm[:, MLP_SPLIT + MLP_TAIL_CHUNK:INTERMEDIATE],
        out_glob=matSiLUOut[:, MLP_SPLIT + MLP_TAIL_CHUNK:INTERMEDIATE],
    )
else:
    gate_proj_fused = SchedGemv(
        Gemv_M64N8,
        MNK=((MLP_SPLIT, MLP_TAIL), N, HIDDEN),
        tmas=(defaultg["loadGate"], defaultg["loadRMSLayer"], regStoreGate),
    )
    up_proj_fused = SchedGemv(
        Gemv_M64N8,
        MNK=((MLP_SPLIT, MLP_TAIL), N, HIDDEN),
        tmas=(defaultg["loadUp"], defaultg["loadRMSLayer"], regStoreUp),
    )
    silu_fused = SchedRegSiLUFused(
        num_token=N,
        store_tma=defaultg["storeSiluLayer"],
        reg_gate=reg_gate,
        reg_up=reg_up,
        base_offset=MLP_SPLIT,
        stride=TileM,
    )

down_proj_low = SchedGemv(
    Gemv_M64N8,
    MNK=(HIDDEN, N, MLP_SPLIT),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
)
down_proj_high = SchedGemv(
    Gemv_M64N8,
    MNK=(HIDDEN, N, (MLP_SPLIT, MLP_TAIL)),
    tmas=(defaultg["loadDown"], defaultg["loadSiluLayer"], defaultg["reduceOut"]),
).bar("store", defaultg["bar_out"])

if parsed_args.no_overlap:
    gate_proj_low.bar("store", defaultg["bar_gate_done"])
    gate_proj_high.bar("store", defaultg["bar_gate_done"])
    up_proj_low.bar("store", defaultg["bar_up_done"])
    up_proj_high.bar("store", defaultg["bar_up_done"])
    silu1.bar("output", defaultg["bar_silu_low_done"])
    if use_tma_tail:
        gate_proj_tail0.bar("store", defaultg["bar_gate_done"])
        gate_proj_tail1.bar("store", defaultg["bar_gate_done"])
        up_proj_tail0.bar("store", defaultg["bar_up_done"])
        up_proj_tail1.bar("store", defaultg["bar_up_done"])
        silu_tail0.bar("output", defaultg["bar_silu_tail_done"])
        silu_tail1.bar("output", defaultg["bar_silu_tail_done"])
    else:
        silu_fused.bar("output", defaultg["bar_silu_tail_done"])
else:
    gate_proj_high.bar("store", defaultg["bar_silu_in"])
    up_proj_high.bar("store", defaultg["bar_silu_in"])
    silu1.bar("input", defaultg["bar_silu_in"]).bar("output", defaultg["bar_silu_out1"])
    if use_tma_tail:
        gate_proj_tail0.bar("store", defaultg["bar_silu_tail_in"])
        gate_proj_tail1.bar("store", defaultg["bar_silu_tail_in"])
        up_proj_tail0.bar("store", defaultg["bar_silu_tail_in"])
        up_proj_tail1.bar("store", defaultg["bar_silu_tail_in"])
        silu_tail0.bar("input", defaultg["bar_silu_tail_in"]).bar("output", defaultg["bar_silu_out2"])
        silu_tail1.bar("input", defaultg["bar_silu_tail_in"]).bar("output", defaultg["bar_silu_out2"])
    else:
        silu_fused.bar("output", defaultg["bar_silu_out2"])
    down_proj_low.bar("load", defaultg["bar_silu_out1"])
    down_proj_high.bar("load", defaultg["bar_silu_out2"])

gate_proj_low = gate_proj_low.place(64)
gate_proj_high = gate_proj_high.place(64)
up_proj_low = up_proj_low.place(64, base_sm=64)
up_proj_high = up_proj_high.place(64, base_sm=64)
silu1 = silu1.place(4, base_sm=128)
if use_tma_tail:
    gate_proj_tail0 = gate_proj_tail0.place(64)
    gate_proj_tail1 = gate_proj_tail1.place(64)
    up_proj_tail0 = up_proj_tail0.place(64, base_sm=64)
    up_proj_tail1 = up_proj_tail1.place(64, base_sm=64)
    silu_tail0 = silu_tail0.place(4, base_sm=128)
    silu_tail1 = silu_tail1.place(4, base_sm=128)
    tail_schedules = [
        gate_proj_tail0,
        gate_proj_tail1,
        up_proj_tail0,
        up_proj_tail1,
        silu_tail0,
        silu_tail1,
    ]
else:
    gate_proj_fused = gate_proj_fused.place(num_sms)
    up_proj_fused = up_proj_fused.place(num_sms)
    silu_fused = silu_fused.place(num_sms)
    tail_schedules = [
        gate_proj_fused,
        up_proj_fused,
        silu_fused,
    ]
down_proj_low = down_proj_low.place(num_sms)
down_proj_high = down_proj_high.place(num_sms)

dae.bind_late_barrier_counts(
    gate_proj_low,
    gate_proj_high,
    up_proj_low,
    up_proj_high,
    silu1,
    tail_schedules,
    down_proj_low,
    down_proj_high,
)

if parsed_args.no_overlap:
    if use_tma_tail:
        dae.i(
            gate_proj_low,
            gate_proj_high,
            gate_proj_tail0,
            gate_proj_tail1,
            IssueBarrier(defaultg["bar_gate_done"]),

            up_proj_low,
            up_proj_high,
            up_proj_tail0,
            up_proj_tail1,
            IssueBarrier(defaultg["bar_up_done"]),

            silu1,
            silu_tail0,
            silu_tail1,
            IssueBarrier(defaultg["bar_silu_low_done"]),
            IssueBarrier(defaultg["bar_silu_tail_done"]),

            down_proj_low,
            down_proj_high,
        )
    else:
        dae.i(
            gate_proj_low,
            gate_proj_high,
            IssueBarrier(defaultg["bar_gate_done"]),

            up_proj_low,
            up_proj_high,
            IssueBarrier(defaultg["bar_up_done"]),

            silu1,
            IssueBarrier(defaultg["bar_silu_low_done"]),

            gate_proj_fused,
            up_proj_fused,
            silu_fused,
            IssueBarrier(defaultg["bar_silu_tail_done"]),

            down_proj_low,
            down_proj_high,
        )
else:
    dae.i(
        gate_proj_low,
        gate_proj_high,
        up_proj_low,
        up_proj_high,
        silu1,
        tail_schedules,
        down_proj_low,
        down_proj_high,
    )

mode = "no-overlap" if parsed_args.no_overlap else "overlapped"
print(f"Llama3 MLP standalone ({mode}, tail-store={parsed_args.tail_store}) on [N={N}, HIDDEN={HIDDEN}, INTERMEDIATE={INTERMEDIATE}], SMs={full_sms}")
dae.s()
dae_app(dae)

if parsed_args.correctness:
    ref = F.linear(F.silu(F.linear(matHidden, matGate)) * F.linear(matHidden, matUp), matDown)
    tensor_diff("mlp_out", ref, matOut)
