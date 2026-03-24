import torch

from dae.launcher import *
from dae.tma_utils import ToConvertedCordAdapter
from dae.util import tensor_diff


class IndexedWeightCordAdapter(ToConvertedCordAdapter):
    def __init__(self, inner, prefix):
        super().__init__(inner, lambda *cords: (*prefix, *cords))
        self.prefix = tuple(prefix)

    def cord2tma(self, *cords):
        return self.inner.cord2tma(*self.prefix, *cords)


def launch_copy(prefix_insts, load_inst, store_inst):
    dae = Launcher(1, device=load_inst.tensor().device if hasattr(load_inst, "tensor") else torch.device("cuda"))

    def sm_task(sm: int):
        if sm != 0:
            return []
        return [
            Copy(1, size=load_inst.size),
            load_inst,
            store_inst,
        ]

    dae.i(*prefix_insts, sm_task, TerminateC(), TerminateM())
    dae.launch()


def verify_dense_layer():
    gpu = torch.device("cuda")
    dtype = torch.bfloat16
    TileM, _, TileK = Gemv_M64N8.MNK

    num_layers = 3
    M = 128
    K = 512
    layer = 2
    m = 64
    k = 256

    src = torch.rand(num_layers, M, K, dtype=dtype, device=gpu) - 0.5
    dst = torch.zeros(TileM, TileK, dtype=dtype, device=gpu)

    dae = Launcher(1, device=gpu)
    src_tma = TmaTensor(dae, src).indexed("layer").wgmma_load(TileM, TileK, Major.K)
    dst_tma = TmaTensor(dae, dst).wgmma_store(TileM, TileK, Major.K)
    load_inst = IndexedWeightCordAdapter(src_tma, (0,)).cord(m, k)
    store_inst = dst_tma.cord(0, 0)

    def sm_task(sm: int):
        if sm != 0:
            return []
        return [
            Copy(1, size=src_tma.size),
            load_inst,
            store_inst,
        ]

    dae.i(SetLayerIndex(layer), sm_task, TerminateC(), TerminateM())
    dae.launch()

    ref = src[layer, m : m + TileM, k : k + TileK]
    print("[dense-layer] layer", layer, "m", m, "k", k)
    tensor_diff("dense_layer_copy", ref, dst)


def verify_router_layer():
    gpu = torch.device("cuda")
    dtype = torch.bfloat16
    TileM, _, TileK = Gemv_M128N8.MNK

    num_layers = 3
    M = 128
    K = 2048
    layer = 1
    m = 0
    k = 1536

    src = torch.rand(num_layers, M, K, dtype=dtype, device=gpu) - 0.5
    dst = torch.zeros(TileM, TileK, dtype=dtype, device=gpu)

    dae = Launcher(1, device=gpu)
    src_tma = TmaTensor(dae, src).indexed("layer").wgmma_load(TileM, TileK, Major.K)
    dst_tma = TmaTensor(dae, dst).wgmma_store(TileM, TileK, Major.K)
    load_inst = IndexedWeightCordAdapter(src_tma, (0,)).cord(m, k)
    store_inst = dst_tma.cord(0, 0)

    def sm_task(sm: int):
        if sm != 0:
            return []
        return [
            Copy(1, size=src_tma.size),
            load_inst,
            store_inst,
        ]

    dae.i(SetLayerIndex(layer), sm_task, TerminateC(), TerminateM())
    dae.launch()

    ref = src[layer, m : m + TileM, k : k + TileK]
    print("[router-layer] layer", layer, "m", m, "k", k)
    tensor_diff("router_layer_copy", ref, dst)


def verify_expert_layer():
    gpu = torch.device("cuda")
    dtype = torch.bfloat16
    TileM, _, TileK = Gemv_M64N8_MMA_SCALE.MNK

    num_layers = 3
    num_experts = 128
    M = 128
    K = 512
    layer = 1
    expert = 3
    m = 64
    k = 256

    src = torch.rand(num_layers, num_experts, M, K, dtype=dtype, device=gpu) - 0.5
    dst = torch.zeros(TileM, TileK, dtype=dtype, device=gpu)
    expert_idx = torch.tensor([[expert]], dtype=torch.int32, device=gpu)

    dae = Launcher(1, device=gpu)
    src_tma = TmaTensor(dae, src).indexed("layer_expert").wgmma_load(TileM, TileK, Major.K)
    dst_tma = TmaTensor(dae, dst).wgmma_store(TileM, TileK, Major.K)
    load_inst = IndexedWeightCordAdapter(src_tma, (0, 0)).cord(m, k)
    store_inst = dst_tma.cord(0, 0)

    def sm_task(sm: int):
        if sm != 0:
            return []
        return [
            Copy(1, size=src_tma.size),
            load_inst,
            store_inst,
        ]

    dae.i(SetLayerIndex(layer), LoadExpertIndex(expert_idx, 0), sm_task, TerminateC(), TerminateM())
    dae.launch()

    ref = src[layer, expert, m : m + TileM, k : k + TileK]
    print("[expert-layer] layer", layer, "expert", expert, "m", m, "k", k)
    tensor_diff("expert_layer_copy", ref, dst)


if __name__ == "__main__":
    verify_dense_layer()
    verify_router_layer()
    verify_expert_layer()
