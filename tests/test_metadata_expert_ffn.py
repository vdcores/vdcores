import importlib.util
from pathlib import Path

import pytest
import torch

from dae.instructions import (
    Gemm_M64N64K64,
    MetadataTmaLoadND,
    TerminateC,
    TerminateM,
    TmaTensor,
)
from dae.launcher import Launcher
from dae.runtime import opcode
from dae.schedule import SchedGemm
from dae.tma_utils import Major


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# Same 64x64 descriptor as moe_nvshmem_experiment.py.
# The expert id stays in metadata[0].
_EXPERTS = 8
_M = 64
_REL_TOL = 0.02


class _Desc:
    def __init__(self, arg, size, num_slots):
        self.arg = arg
        self.size = size
        self.num_slots = num_slots


# Import the real module: its script setup is guarded, so this needs neither
# MPI nor NVSHMEM and does not allocate weights or launch the experiment.
_source = Path(__file__).resolve().parents[1] / "app/python/moe_nvshmem_experiment.py"
_spec = importlib.util.spec_from_file_location("moe_nvshmem_experiment", _source)
experiment = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(experiment)
ExpertWeightLoad = experiment.ExpertWeightLoad
_weight_desc = experiment._weight_desc


def _bind_weight(launcher, weight):
    spec = _weight_desc(weight)
    load = experiment._bind_expert_weight(launcher, spec)
    # Preserve packed storage for the lifetime of the scheduled load.
    load.weight = spec["weight"]
    return load


def _reference(x, up, gate, down, expert_id):
    # same expert ffn as moe_reference.py. id stays on device.
    index = expert_id.reshape(1)
    up_w = up.index_select(0, index).squeeze(0)
    gate_w = gate.index_select(0, index).squeeze(0)
    down_w = down.index_select(0, index).squeeze(0)
    hidden = torch.nn.functional.silu(x @ up_w.t()) * (x @ gate_w.t())
    return hidden @ down_w.t()


def _assert_close(name, got, ref):
    assert got.shape == ref.shape, f"{name}: shape {got.shape} != {ref.shape}"
    assert torch.isfinite(got).all(), f"{name}: nonfinite GPU output"
    assert torch.isfinite(ref).all(), f"{name}: nonfinite reference"
    diff = (got.float() - ref.float()).abs()
    scale = ref.float().abs().mean().clamp_min(1e-3)
    relative = diff.mean() / scale
    print(
        f"{name}: mean relative {float(relative):.6f}, "
        f"max abs {float(diff.max()):.6f}"
    )
    assert float(relative) < _REL_TOL, (
        f"{name} mean relative diff {float(relative):.4f} exceeds {_REL_TOL}"
    )


def test_kmajor_repeat_keeps_appended_expert():
    desc = _Desc(3, 8192, 1)
    load = ExpertWeightLoad(desc)
    issued = load.cord(64, 128)
    assert issued.opcode == opcode.OP_ALLOC_TMA_LOAD_METADATA_LAST
    assert issued.cords == [0, 64, 2, 0xFFFF]
    issued.delta(load.cord2tma(0, 64))
    assert issued.opcode == opcode.OP_ALLOC_TMA_LOAD_METADATA_LAST
    assert issued.cords == [0, 64, 3, 0xFFFF]


def test_metadata_last_preserves_default_packing():
    desc = _Desc(7, 8192, 1)
    coords = [0, 4, 1]
    default = MetadataTmaLoadND(desc, 5, coords)
    last = MetadataTmaLoadND(desc, 5, coords, metadata_last=True)
    assert default.opcode == opcode.OP_ALLOC_TMA_LOAD_METADATA
    assert last.opcode == opcode.OP_ALLOC_TMA_LOAD_METADATA_LAST
    assert default.arg == last.arg
    assert default.size == last.size
    assert default.cords == last.cords
    issued = last.cord(0, 4, 1)
    assert issued.opcode == opcode.OP_ALLOC_TMA_LOAD_METADATA_LAST
    assert issued.cords == [0, 4, 1, 0xFFFF]


@cuda_only
def test_expert_ffn_default_shapes_and_validation(monkeypatch):
    # Empty dispatches exercise dimension validation without launching CUDA.
    up = {"weight": torch.empty(8, 192, 256, dtype=torch.bfloat16, device="cuda")}
    down = {"weight": torch.empty(8, 128, 192, dtype=torch.bfloat16, device="cuda")}
    monkeypatch.setattr(experiment, "ffn_up_desc", up, raising=False)
    monkeypatch.setattr(experiment, "ffn_gate_desc", up, raising=False)
    monkeypatch.setattr(experiment, "ffn_down_desc", down, raising=False)
    run = experiment.run_vdcores_expert
    x = torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
    assert run(x, 0).shape == (0, 128)
    with pytest.raises(ValueError, match="input"):
        run(x[:, :128], 0)
    with pytest.raises(ValueError, match="expert id"):
        run(x, 8)
    with pytest.raises(ValueError, match="dimensions"):
        run(x, 0, (up, down, down))


@cuda_only
@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("input_dim,hidden_dim,output_dim,layout", [
    (512, 256, 512, "contiguous"),
    (128, 128, 128, "contiguous"),
    (256, 192, 128, "transposed"),
    (128, 256, 192, "fused"),
])
def test_metadata_expert_ffn_follows_topk(k, input_dim, hidden_dim, output_dim, layout):
    device = torch.device("cuda")
    torch.manual_seed(0)
    dtype = torch.bfloat16

    x = (torch.randn(_M, input_dim, device=device, dtype=dtype) * 0.1).contiguous()
    router_w = torch.randn(_EXPERTS, input_dim, device=device, dtype=dtype)
    expert_up = (
        torch.randn(_EXPERTS, hidden_dim, input_dim, device=device, dtype=dtype) * 0.1
    ).contiguous()
    expert_gate = (
        torch.randn(_EXPERTS, hidden_dim, input_dim, device=device, dtype=dtype) * 0.1
    ).contiguous()
    expert_down = (
        torch.randn(_EXPERTS, output_dim, hidden_dim, device=device, dtype=dtype) * 0.1
    ).contiguous()

    if layout == "transposed":
        expert_up = expert_up.transpose(-1, -2).contiguous().transpose(-1, -2)
        expert_gate = expert_gate.transpose(-1, -2).contiguous().transpose(-1, -2)
        expert_down = expert_down.transpose(-1, -2).contiguous().transpose(-1, -2)
    elif layout == "fused":
        fused = torch.cat((expert_up, expert_gate), dim=1)
        expert_up, expert_gate = fused.split(hidden_dim, dim=1)

    # one descriptor for the whole tensor. topk picks the experts for token 0.
    router_probs = torch.softmax(x @ router_w.t(), dim=-1)
    weights, expert_ids = torch.topk(router_probs, k, dim=1)
    chosen = expert_ids[0]
    if k > 1:
        assert not torch.equal(chosen[0], chosen[1])

    metadata = torch.empty(1, dtype=torch.uint32, device=device)
    metadata.copy_(chosen[0].to(torch.uint32).reshape(1))

    up_out = torch.zeros(_M, hidden_dim, dtype=dtype, device=device)
    gate_out = torch.zeros(_M, hidden_dim, dtype=dtype, device=device)
    hidden = torch.empty(_M, hidden_dim, dtype=dtype, device=device)
    out = torch.zeros(_M, output_dim, dtype=dtype, device=device)

    atom = Gemm_M64N64K64
    tile_m, tile_n, tile_k = atom.MNK

    projections = Launcher(1, device=device, metadata=metadata)
    load_x = TmaTensor(projections, x).wgmma_load(tile_m, tile_k, Major.K)
    load_up = _bind_weight(projections, expert_up)
    load_gate = _bind_weight(projections, expert_gate)
    store_up = TmaTensor(projections, up_out).wgmma(
        "reduce", tile_m, tile_n, Major.K
    )
    store_gate = TmaTensor(projections, gate_out).wgmma(
        "reduce", tile_m, tile_n, Major.K
    )
    up = SchedGemm(
        atom,
        MNK=(_M, hidden_dim, input_dim),
        tmas=(load_x, load_up, store_up),
        fold=1,
        group=False,
    ).place(1)
    gate = SchedGemm(
        atom,
        MNK=(_M, hidden_dim, input_dim),
        tmas=(load_x, load_gate, store_gate),
        fold=1,
        group=False,
    ).place(1)
    projections.i(up, gate, TerminateC(), TerminateM())

    down_launch = Launcher(1, device=device, metadata=metadata)
    load_h = TmaTensor(down_launch, hidden).wgmma_load(tile_m, tile_k, Major.K)
    load_down = _bind_weight(down_launch, expert_down)
    store_out = TmaTensor(down_launch, out).wgmma(
        "reduce", tile_m, tile_n, Major.K
    )
    down = SchedGemm(
        atom,
        MNK=(_M, output_dim, hidden_dim),
        tmas=(load_h, load_down, store_out),
        fold=1,
        group=False,
    ).place(1)
    down_launch.i(down, TerminateC(), TerminateM())

    def run(expert_id):
        metadata.copy_(expert_id.to(torch.uint32).reshape(1))
        up_out.zero_()
        gate_out.zero_()
        out.zero_()
        projections.launch()
        torch.cuda.synchronize(device)
        hidden.copy_(torch.nn.functional.silu(up_out) * gate_out)
        down_launch.launch()
        torch.cuda.synchronize(device)
        return out

    mixed = torch.zeros(output_dim, dtype=torch.float32, device=device)
    ref_mixed = torch.zeros(output_dim, dtype=torch.float32, device=device)
    projection_packet = None
    down_packet = None
    previous = None
    for slot in range(k):
        expert_id = chosen[slot]
        got = run(expert_id).clone()
        if projection_packet is None:
            projection_packet = projections._launch_packet
            down_packet = down_launch._launch_packet
        else:
            assert projections._launch_packet is projection_packet
            assert down_launch._launch_packet is down_packet
        ref = _reference(x, expert_up, expert_gate, expert_down, expert_id)
        _assert_close(f"k={k} expert {slot}", got, ref)
        mixed += got[0].float() * weights[0, slot].float()
        ref_mixed += ref[0].float() * weights[0, slot].float()
        if previous is not None:
            assert not torch.allclose(previous, got)
        previous = got
    _assert_close(f"k={k} mix", mixed, ref_mixed)


@cuda_only
@pytest.mark.parametrize("k", [1, 2, 4, 8])
@pytest.mark.parametrize("tokens", [1, 64, 65])
@pytest.mark.parametrize("input_dim,hidden_dim,output_dim", [
    (512, 256, 512),
    (128, 128, 128),
    (256, 192, 128),
    (128, 256, 192),
])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "fused"])
def test_run_vdcores_expert(k, tokens, input_dim, hidden_dim, output_dim, layout, monkeypatch):
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    x = torch.randn(tokens, input_dim, device=device, dtype=dtype) * 0.1
    up = torch.randn(_EXPERTS, hidden_dim, input_dim, device=device, dtype=dtype) * 0.1
    gate = torch.randn_like(up) * 0.1
    down = torch.randn(_EXPERTS, output_dim, hidden_dim, device=device, dtype=dtype) * 0.1
    if layout == "transposed":
        up = up.transpose(-1, -2).contiguous().transpose(-1, -2)
        gate = gate.transpose(-1, -2).contiguous().transpose(-1, -2)
        down = down.transpose(-1, -2).contiguous().transpose(-1, -2)
    elif layout == "fused":
        fused = torch.cat((up, gate), dim=1)
        up, gate = fused.split(hidden_dim, dim=1)

    descriptors = tuple(_weight_desc(weight) for weight in (up, gate, down))
    pointers = tuple(desc["weight"].data_ptr() for desc in descriptors)
    for weight, desc in zip((up, gate, down), descriptors):
        assert desc["weight"].is_contiguous()
        torch.testing.assert_close(desc["weight"], weight, rtol=0, atol=0)

    def unexpected_pack(*args, **kwargs):
        pytest.fail("weights must not be repacked during expert execution")

    monkeypatch.setattr(experiment, "_weight_desc", unexpected_pack)
    router = torch.randn(_EXPERTS, input_dim, device=device, dtype=dtype)
    weights, ids = torch.topk(torch.softmax(x @ router.t(), dim=-1), k, dim=-1)
    mixed = torch.zeros(tokens, output_dim, device=device, dtype=torch.float32)
    reference = torch.zeros_like(mixed)
    # Exercise the real helper, including padding and split-K. Each row's
    # selected route weights are accumulated without changing Top-K semantics.
    for expert in range(_EXPERTS):
        rows, slots = torch.where(ids == expert)
        if rows.numel() == 0:
            continue
        expert_id = torch.tensor(expert, device=device)
        got = experiment.run_vdcores_expert(x, expert, descriptors)
        ref = _reference(x, up, gate, down, expert_id)
        _assert_close(f"helper expert {expert}", got, ref)
        route_weights = weights[rows, slots].float().unsqueeze(1)
        mixed[rows] += got[rows].float() * route_weights
        reference[rows] += ref[rows].float() * route_weights
    _assert_close("helper Top-K mix", mixed, reference)
    assert tuple(desc["weight"].data_ptr() for desc in descriptors) == pointers

    if (input_dim, hidden_dim, output_dim, layout) == (512, 256, 512, "contiguous"):
        for name, desc in zip(("ffn_up_desc", "ffn_gate_desc", "ffn_down_desc"), descriptors):
            monkeypatch.setattr(experiment, name, desc, raising=False)
        got = experiment.run_vdcores_expert(x, 0)
        ref = _reference(x, up, gate, down, torch.tensor(0, device=device))
        _assert_close("original default descriptors", got, ref)

    # Revisit the first and last expert with the same descriptors, including
    # experts that a narrow Top-K selection might not have selected.
    for expert in (0, _EXPERTS - 1, 0):
        got = experiment.run_vdcores_expert(x, expert, descriptors)
        ref = _reference(x, up, gate, down, torch.tensor(expert, device=device))
        _assert_close(f"helper repeated expert {expert}", got, ref)


@cuda_only
@pytest.mark.parametrize("shape", [(8, 128), (8, 0, 128), (8, 96, 128), (8, 128, 96)])
def test_expert_weight_rejects_unsupported_shapes(shape):
    with pytest.raises(ValueError):
        _weight_desc(torch.empty(shape, dtype=torch.bfloat16, device="cuda"))


def test_expert_requires_descriptors_when_imported():
    with pytest.raises(ValueError, match="standalone calls require"):
        experiment.run_vdcores_expert(torch.empty(0, 128), 0)


@cuda_only
def test_expert_rejects_mismatched_input_and_projection_shapes():
    up = _weight_desc(torch.zeros(8, 192, 256, dtype=torch.bfloat16, device="cuda"))
    down = _weight_desc(torch.zeros(8, 128, 192, dtype=torch.bfloat16, device="cuda"))
    x = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="dimensions"):
        experiment.run_vdcores_expert(x, 0, (up, down, down))
    with pytest.raises(ValueError, match="input"):
        experiment.run_vdcores_expert(x[:, :128], 0, (up, up, down))
    with pytest.raises(ValueError, match="input"):
        experiment.run_vdcores_expert(x.float(), 0, (up, up, down))
    with pytest.raises(ValueError, match="expert id"):
        experiment.run_vdcores_expert(x, -1, (up, up, down))
    with pytest.raises(ValueError, match="BF16"):
        _weight_desc(up["weight"].float())


@cuda_only
@pytest.mark.parametrize("input_dim,hidden_dim,output_dim,tokens,expected_folds", [
    (512, 256, 512, 1, (8, 8, 4)),
    (256, 192, 128, 65, (4, 4, 1)),
    (64, 64, 64, 1, (1, 1, 1)),
    (192, 64, 256, 129, (1, 1, 1)),
])
def test_expert_fold_boundaries(input_dim, hidden_dim, output_dim, tokens, expected_folds, monkeypatch):
    # Observe the real schedules without replacing their construction or GPU
    # execution. Cover split-K, fallback to one fold, and one-tile widths.
    folds = []

    def record_gemm(*args, **kwargs):
        folds.append(kwargs["fold"])
        return SchedGemm(*args, **kwargs)

    monkeypatch.setattr(experiment, "SchedGemm", record_gemm)
    torch.manual_seed(1)
    # A non-default stream also exercises ordering of metadata publication,
    # up/gate loads, activation, and down loads in the actual helper.
    with torch.cuda.stream(torch.cuda.Stream()):
        x = torch.randn(tokens, input_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        up = torch.randn(2, hidden_dim, input_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        gate = torch.randn_like(up) * 0.1
        down = torch.randn(2, output_dim, hidden_dim, device="cuda", dtype=torch.bfloat16) * 0.1
        descriptors = tuple(_weight_desc(weight) for weight in (up, gate, down))
        got = experiment.run_vdcores_expert(x, 1, descriptors)
        ref = _reference(x, up, gate, down, torch.tensor(1, device="cuda"))
        _assert_close("fold boundaries", got, ref)
        assert tuple(folds) == expected_folds
        empty = experiment.run_vdcores_expert(x[:0], 1, descriptors)
        assert empty.shape == (0, output_dim)
        assert empty.dtype == x.dtype and empty.device == x.device
        assert tuple(folds) == expected_folds, "empty input must not schedule GEMMs"
