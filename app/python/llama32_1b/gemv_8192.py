import argparse
import sys

import torch
import torch.nn.functional as F

from dae.launcher import *
from dae.schedule import *
from dae.util import dae_app, tensor_diff


arg_parser = argparse.ArgumentParser(add_help=False)
arg_parser.add_argument("--correctness", action="store_true")
arg_parser.add_argument("--seed", type=int, default=0)
arg_parser.add_argument("--sms", type=int, default=128)
arg_parser.add_argument("--split-64-64", action="store_true",
                        help="Run M=8192 as two serialized M=4096 GEMVs on 64 SMs each")
arg_parser.add_argument("--stream-k-78", action="store_true",
                        help="Run a 78-SM stream-K style split: 39 M tiles with fold=2, repeated twice, then 50 M tiles")
arg_parser.add_argument("--tile-stream-k-78", action="store_true",
                        help="Distribute 64M x 1024K GEMV macro-tiles nearly evenly over 78 SMs")
parsed_args, remaining_argv = arg_parser.parse_known_args()
if parsed_args.correctness and not any(arg in ("-l", "--launch", "-b", "--bench") for arg in remaining_argv):
    remaining_argv = [*remaining_argv, "--launch"]
sys.argv = [sys.argv[0], *remaining_argv]

torch.manual_seed(parsed_args.seed)

gpu = torch.device("cuda")
dtype = torch.bfloat16

N = 8
HIDDEN = 4096
INTERMEDIATE = 8192
full_sms = 132

dae = Launcher(full_sms, device=gpu)
TileM, _, TileK = Gemv_M64N8.MNK

matInput = torch.randn(N, HIDDEN, dtype=dtype, device=gpu)
matWeight = torch.randn(INTERMEDIATE, HIDDEN, dtype=dtype, device=gpu)
matOut = torch.zeros(N, INTERMEDIATE, dtype=dtype, device=gpu)

defaultg = dae.get_group()
defaultg.addBarrier("bar_out")

defaultg.addTma("loadInput", [matInput], lambda t: t.wgmma_load(N, TileK * Gemv_M64N8.n_batch, Major.K))
defaultg.addTma("loadWeight", [matWeight], lambda t: t.wgmma_load(TileM, TileK, Major.K))
defaultg.addTma("storeOut", [matOut], lambda t: t.wgmma_store(N, TileM, Major.MN))
defaultg.addTma("reduceOut", [matOut], lambda t: t.wgmma("reduce", N, TileM, Major.MN))

dae.set_persistent(matInput)
dae.set_streaming(matWeight)

dae.build_groups()


class SchedGemvTileStreamK(Schedule):
    def __init__(self, Atom, MNK, tmas):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas

    def _on_place(self):
        TileM, TileN, TileK = self.Atom.MNK
        M, N_, K = self.MNK
        self.k_group = TileK * self.Atom.n_batch
        assert M % TileM == 0
        assert N_ % TileN == 0
        assert K % self.k_group == 0
        self.m_tiles = M // TileM
        self.k_groups = K // self.k_group
        self.total_groups = self.m_tiles * self.k_groups

    def schedule(self, sm):
        if sm < 0:
            return []

        TileM, _, TileK = self.Atom.MNK
        loadA, loadB, storeC = self.tmas
        groups_per_sm = self.total_groups // self.num_sms
        remainder = self.total_groups % self.num_sms
        start = sm * groups_per_sm + min(sm, remainder)
        count = groups_per_sm + (1 if sm < remainder else 0)

        insts = []
        for i in range(count):
            group_idx = start + i
            m_tile = group_idx // self.k_groups
            k_group = group_idx % self.k_groups
            m = m_tile * TileM
            k = k_group * self.k_group
            is_last = i == count - 1
            insts.extend([
                self.Atom(self.Atom.n_batch),
                RepeatM.onSync(
                    0,
                    None,
                    1,
                    (loadB.cord(0, k), loadB.cord2tma(0, self.k_group)),
                    *[
                        (loadA.cord(m, k + TileK * j), loadA.cord2tma(0, self.k_group))
                        for j in range(self.Atom.n_batch)
                    ],
                    asyncPort=True,
                ),
                storeC.cord(0, m).bar(self._bar("store") if is_last else None).group(is_last),
            ])
        return insts

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


if parsed_args.tile_stream_k_78:
    gemv = SchedGemvTileStreamK(
        Gemv_M64N8,
        MNK=(INTERMEDIATE, N, HIDDEN),
        tmas=(defaultg["loadWeight"], defaultg["loadInput"], defaultg["reduceOut"]),
    ).bar("store", defaultg["bar_out"]).place(78)
    schedules = [gemv]
elif parsed_args.stream_k_78:
    chunk_tiles = 39
    chunk_m = chunk_tiles * TileM
    tail_m = INTERMEDIATE - 2 * chunk_m
    gemv_0 = SchedGemv(
        Gemv_M64N8,
        MNK=(chunk_m, N, HIDDEN),
        tmas=(defaultg["loadWeight"], defaultg["loadInput"], defaultg["reduceOut"]),
        fold=2,
    ).place(78)
    gemv_1 = SchedGemv(
        Gemv_M64N8,
        MNK=((chunk_m, chunk_m), N, HIDDEN),
        tmas=(defaultg["loadWeight"], defaultg["loadInput"], defaultg["reduceOut"]),
        fold=2,
    ).place(78)
    gemv_2 = SchedGemv(
        Gemv_M64N8,
        MNK=((2 * chunk_m, tail_m), N, HIDDEN),
        tmas=(defaultg["loadWeight"], defaultg["loadInput"], defaultg["storeOut"]),
    ).bar("store", defaultg["bar_out"]).place(tail_m // TileM)
    schedules = [gemv_0, gemv_1, gemv_2]
elif parsed_args.split_64_64:
    gemv_0 = SchedGemv(
        Gemv_M64N8,
        MNK=(INTERMEDIATE // 2, N, HIDDEN),
        tmas=(defaultg["loadWeight"], defaultg["loadInput"], defaultg["storeOut"]),
    ).place(64)
    gemv_1 = SchedGemv(
        Gemv_M64N8,
        MNK=((INTERMEDIATE // 2, INTERMEDIATE // 2), N, HIDDEN),
        tmas=(defaultg["loadWeight"], defaultg["loadInput"], defaultg["storeOut"]),
    ).bar("store", defaultg["bar_out"]).place(64)
    schedules = [gemv_0, gemv_1]
else:
    gemv = SchedGemv(
        Gemv_M64N8,
        MNK=(INTERMEDIATE, N, HIDDEN),
        tmas=(defaultg["loadWeight"], defaultg["loadInput"], defaultg["storeOut"]),
    ).bar("store", defaultg["bar_out"]).place(parsed_args.sms)
    schedules = [gemv]

dae.bind_late_barrier_counts(schedules)
dae.i(schedules)

if parsed_args.tile_stream_k_78:
    mode = "tile-stream-k-78"
    sms_label = "78"
elif parsed_args.stream_k_78:
    mode = "stream-k-78"
    sms_label = "78+78+50"
elif parsed_args.split_64_64:
    mode = "split-64-64"
    sms_label = "64+64"
else:
    mode = "single"
    sms_label = str(parsed_args.sms)
print(
    f"Llama 3.2 1B standalone GEMV ({mode}) "
    f"on [M={INTERMEDIATE}, N={N}, K={HIDDEN}], SMs={sms_label}/{full_sms}"
)
dae.s()
dae_app(dae)

if parsed_args.correctness:
    ref = F.linear(matInput, matWeight)
    tensor_diff("gemv_out", ref, matOut)
