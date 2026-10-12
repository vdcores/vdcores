import pytest
import torch

from dae.instructions import (
    Copy,
    MemoryInstruction,
    MetadataTmaLoadND,
    RepeatM,
    TerminateC,
    TerminateM,
    TmaStore1D,
    addr2cords,
    cords2addr,
)
from dae.launcher import Launcher
from dae.runtime import opcode
from dae.tma_utils import bytes2slots
from dae import runtime


cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# meta_idx 0 and 1. The value is the leading bf16 coordinate, so it has to be a
# multiple of 8; otherwise the tile address is not 16-byte aligned.
_EXPERT_IDS = (0, 8)
_LEADING = 16
_BOX0 = 8
_SENTINEL = 0xFFFF


def _decode(inst):
    desc_index = inst.arg & 0x3FF
    meta_idx = ((inst.size >> 12) & 0xF) | (((inst.arg >> 10) & 0x3F) << 4)
    nbytes = (inst.size & 0xFFF) << 4
    static = []
    for coord in inst.cords:
        if coord == _SENTINEL:
            break
        static.append(coord)
    return desc_index, meta_idx, nbytes, static


class _Desc:
    def __init__(self, arg, size, num_slots):
        self.arg = arg
        self.size = size
        self.num_slots = num_slots


@pytest.mark.parametrize(
    ("desc_index", "meta_idx", "nbytes", "coords"),
    [
        (0, 0, 16, [0]),
        (1023, 15, 16, [65534]),
        (0, 16, 65520, []),
        (1023, 1023, 65520, [0, 65534, 0, 65534]),
    ],
)
def test_metadata_tma_load_packing(desc_index, meta_idx, nbytes, coords):
    inst = MetadataTmaLoadND(_Desc(desc_index, nbytes, 1), meta_idx, coords)
    assert _decode(inst) == (desc_index, meta_idx, nbytes, coords)
    assert inst.cords[len(coords):] == [_SENTINEL] * (4 - len(coords))
    assert inst.rank == len(coords) + 1


def test_metadata_tma_load_rejects_out_of_range_fields():
    with pytest.raises(ValueError):
        MetadataTmaLoadND(_Desc(0, 16, 1), 1024, [0])
    with pytest.raises(ValueError):
        MetadataTmaLoadND(_Desc(0, 16, 1), 0, [65535])


def test_doc_example_three_descriptors_use_meta_idx():
    # meta_idx 0 and 1. One descriptor each for up, gate, and down.
    for meta_idx in (0, 1):
        for desc_index in (0, 1, 2):
            inst = MetadataTmaLoadND(_Desc(desc_index, 16, 1), meta_idx, [4, 8])
            got_desc, got_meta, nbytes, static = _decode(inst)
            assert (got_desc, got_meta, nbytes, static) == (desc_index, meta_idx, 16, [4, 8])
            assert inst.cords[2:] == [_SENTINEL, _SENTINEL]


def test_repeat_keeps_metadata_packing_and_expert_coordinate():
    # Static coordinates follow the expert id. The device inserts that id first.
    desc = _Desc(7, 8192, 1)
    load = MetadataTmaLoadND(desc, 1, [0, 0, 0])
    issued = load.cord(0, 64, 2).group(False)
    assert _decode(issued) == (7, 1, 8192, [0, 64, 2])
    assert issued.cords[3] == _SENTINEL
    assert (issued.opcode & 4) == 0

    original = issued.cords.copy()
    issued.delta([0, 0, 1])
    assert _decode(issued) == (7, 1, 8192, [0, 64, 3])
    assert issued.cords[3] == _SENTINEL

    advanced = addr2cords(cords2addr(original) + cords2addr([0, 0, 1, 0]))
    assert advanced == [0, 64, 3, _SENTINEL]

    repeated = RepeatM.on(2, (issued, [0, 0, 1]))
    bodies = [inst for inst in repeated if inst.arg == issued.arg]
    assert bodies
    for body in bodies:
        assert _decode(body)[0] == 7
        assert _decode(body)[1] == 1
        assert body.cords[3] == _SENTINEL


def _strides(global_dims, element_size):
    strides = []
    stride = global_dims[0] * element_size
    for dim in global_dims[1:]:
        strides.append(stride)
        stride *= dim
    return strides


def _weights(global_dims, bias=0):
    # global_dims[0] is the leading expert dimension and the contiguous TMA dim.
    physical_shape = tuple(reversed(global_dims))
    numel = 1
    for dim in global_dims:
        numel *= dim
    values = torch.arange(numel, device="cuda", dtype=torch.int32).to(torch.bfloat16)
    if bias:
        values = values + bias
    physical = values.reshape(physical_shape)
    logical = physical.permute(*reversed(range(len(global_dims))))
    return physical, logical


def _register(launcher, tensor, global_dims, box_dims):
    desc = runtime.build_tma_desc(
        tensor,
        list(global_dims),
        _strides(global_dims, tensor.element_size()),
        list(box_dims),
        [1] * len(global_dims),
        0,
        0,
    )
    nbytes = tensor.element_size()
    for dim in box_dims:
        nbytes *= dim
    return launcher.new_tma(desc), nbytes


def _load_case(n):
    # n static coordinates, so the descriptor rank is n + 1.
    # Dim 0 is the expert dimension. The metadata value is that leading coordinate.
    if n == 0:
        return (), (_LEADING,), (_BOX0,)
    static = (0,) if n == 1 else (0, 1) + (0,) * (n - 2)
    middle = tuple(3 if i == 0 else 2 for i in range(n))
    return static, (_LEADING,) + middle, (_BOX0,) + (1,) * n


def _expected(logical, expert, static):
    # The leading box is 8 bf16 values, the minimum legal box on TMA dim 0.
    index = (slice(expert, expert + _BOX0),) + tuple(static)
    return logical[index].contiguous().view(-1)


def _store_box(n):
    if n <= 1:
        return (_BOX0,)
    return (_BOX0,) + (1,) * (n - 1)


def _store(launcher, output, n, nbytes):
    if n <= 1:
        return TmaStore1D(output, bytes=nbytes)
    box = _store_box(n)
    arg, _ = _register(launcher, output, box, box)
    return MemoryInstruction(
        opcode=(
            opcode.OP_ALLOC_WB_TMA_STORE_2D,
            opcode.OP_ALLOC_WB_TMA_STORE_3D,
            opcode.OP_ALLOC_WB_TMA_STORE_4D,
        )[n - 2],
        num_slots=bytes2slots(nbytes),
        arg=arg,
        size=nbytes,
        cords=[0] * n,
    )


def _program(launcher, loads, stores, nbytes):
    insts = []
    for load, store in zip(loads, stores):
        insts.extend([Copy(1, size=nbytes), load, store])
    launcher.i(*insts, TerminateC(), TerminateM())


def _check(actual, expected):
    torch.cuda.synchronize()
    if not torch.equal(actual.view(-1), expected):
        raise AssertionError(
            f"output={actual.view(-1).to(torch.int32).tolist()} "
            f"expected={expected.to(torch.int32).tolist()}"
        )


def _run_case(n, meta_idx, metadata):
    static, global_dims, box = _load_case(n)
    physical, logical = _weights(global_dims)
    output = torch.empty(_store_box(n), dtype=torch.bfloat16, device="cuda")
    launcher = Launcher(1, device="cuda", metadata=metadata)
    arg, nbytes = _register(launcher, physical, global_dims, box)
    desc = _Desc(arg, nbytes, bytes2slots(nbytes))
    _program(
        launcher,
        [MetadataTmaLoadND(desc, meta_idx, static)],
        [_store(launcher, output, n, nbytes)],
        nbytes,
    )

    def run(expert):
        metadata[meta_idx] = expert
        output.fill_(0)
        launcher.launch()
        _check(output, _expected(logical, expert, static))

    return launcher, run


@cuda_only
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_metadata_tma_load_nd(n):
    metadata = torch.tensor(list(_EXPERT_IDS), dtype=torch.uint32, device="cuda")
    launcher, run = _run_case(n, 0, metadata)
    run(_EXPERT_IDS[0])
    packet = launcher._launch_packet
    run(_EXPERT_IDS[1])
    assert launcher._launch_packet is packet


@cuda_only
def test_metadata_tma_load_zero_static_coordinates():
    metadata = torch.tensor(list(_EXPERT_IDS), dtype=torch.uint32, device="cuda")
    launcher, run = _run_case(0, 0, metadata)
    run(_EXPERT_IDS[0])
    packet = launcher._launch_packet
    run(_EXPERT_IDS[1])
    assert launcher._launch_packet is packet


@cuda_only
def test_metadata_tma_load_index_at_least_16():
    metadata = torch.zeros(17, dtype=torch.uint32, device="cuda")
    launcher, run = _run_case(1, 16, metadata)
    run(_EXPERT_IDS[0])
    packet = launcher._launch_packet
    run(_EXPERT_IDS[1])
    assert launcher._launch_packet is packet


@cuda_only
def test_up_gate_down_use_separate_descriptors():
    # Up, gate, and down each have one descriptor. meta_idx 0 and 1 both use them.
    static, global_dims, box = _load_case(1)
    weights = {
        name: _weights(global_dims, bias=bias)
        for name, bias in (("up", 0), ("gate", 100), ("down", 200))
    }
    metadata = torch.tensor(list(_EXPERT_IDS), dtype=torch.uint32, device="cuda")
    launcher = Launcher(1, device="cuda", metadata=metadata)

    descs = {}
    nbytes = None
    for name, (physical, _) in weights.items():
        arg, nbytes = _register(launcher, physical, global_dims, box)
        descs[name] = _Desc(arg, nbytes, bytes2slots(nbytes))
    assert len({desc.arg for desc in descs.values()}) == 3

    loads = []
    stores = []
    outputs = {}
    for meta_idx in (0, 1):
        for name, desc in descs.items():
            output = torch.empty(_store_box(1), dtype=torch.bfloat16, device="cuda")
            outputs[(name, meta_idx)] = output
            loads.append(MetadataTmaLoadND(desc, meta_idx, static))
            stores.append(_store(launcher, output, 1, nbytes))
    _program(launcher, loads, stores, nbytes)

    launcher.launch()
    packet = launcher._launch_packet
    for meta_idx, expert in enumerate(_EXPERT_IDS):
        for name, (_, logical) in weights.items():
            _check(outputs[(name, meta_idx)], _expected(logical, expert, static))
    assert launcher._launch_packet is packet
