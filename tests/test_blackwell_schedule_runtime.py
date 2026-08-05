import pytest
import torch

from dae.instructions import (
    ATTENTION_M64N64K16_F16_F32_64_64_hdim,
    ATTENTION_SM100_BF16_HDIM128_DIRECT,
    Gemv_M128N8Direct4,
    MemoryInstruction,
    RepeatM,
)
from dae.runtime import opcode
from dae.launcher import Launcher
from dae.schedule import SchedSmemSiLUInterleaved
from dae.tma_utils import cords2addr, cord_func_2d_tile_major, pack_weight_tile_major


def test_tile_major_weight_pack_round_trip_and_coordinates():
    tile_m, tile_k = 64, 256
    weight = torch.arange(128 * 512, dtype=torch.int32).reshape(128, 512)
    packed = pack_weight_tile_major(weight, tile_m, tile_k)

    assert packed.shape == (2, 2, 4, 64, 64)
    recovered = packed.permute(1, 3, 0, 2, 4).reshape_as(weight)
    torch.testing.assert_close(recovered, weight, rtol=0, atol=0)

    cord = cord_func_2d_tile_major(packed, packed.dim())
    assert cord(64, 256) == [0, 0, 1, 1]


def test_dynamic_repeat_encodes_zero_count_skip_window():
    step = MemoryInstruction(opcode.OP_ALLOC_TMA_LOAD_1D, num_slots=1, arg=0, size=16)
    instructions = RepeatM.on(
        0,
        (step, [1]),
        count_counter_reg=3,
    )

    repeat = instructions[0]
    assert repeat.size == 0
    assert repeat.arg & RepeatM.COUNT_COUNTER_MODE_FLAG
    assert repeat.arg & RepeatM.COUNTER_REG_MASK == 3
    assert (repeat.arg >> RepeatM.SKIP_COUNT_SHIFT) & RepeatM.SKIP_COUNT_MASK == 1


def test_attention_runtime_counter_fields_are_disjoint():
    instruction = ATTENTION_M64N64K16_F16_F32_64_64_hdim(
        num_kv_block=1,
        num_active_q=4,
        last_kv_active_token_len=1,
        seq_len_counter_reg=1,
        num_kv_block_counter_reg=3,
        kv_block_size=128,
        outer_seq_len_counter_reg=2,
        outer_seq_len_counter_stride=1,
    )

    assert instruction.args[0] == 0x0101
    assert instruction.args[1] == 0x0184
    assert instruction.args[2] == 0x213F


def test_direct_attention_preserves_dynamic_decode_fields():
    instruction = ATTENTION_SM100_BF16_HDIM128_DIRECT(
        num_kv_block=1,
        num_active_q=4,
        last_kv_active_token_len=1,
        need_norm=False,
        need_rope=False,
        seq_len_counter_reg=1,
        num_kv_block_counter_reg=3,
        kv_block_size=128,
        outer_seq_len_counter_reg=2,
        outer_seq_len_counter_stride=1,
    )

    assert instruction.args[0] == 0x0101
    assert instruction.args[1] == 0x0184
    assert instruction.args[2] == 0x213C


def test_grouped_direct_gemv_encodes_element_strides_in_m128_units():
    instruction = Gemv_M128N8Direct4(
        kTiles=32,
        output_stride=65536,
        output_group_stride=16384,
    )

    assert instruction.opcode == opcode.OP_GEMV_SM100_M128N8_DIRECT4
    assert instruction.args == [32, 512, 128]
    with pytest.raises(ValueError, match="multiples of 128"):
        Gemv_M128N8Direct4(32, 65535, 16384)


def test_launcher_loop_counter_validation_without_allocating_gpu_state():
    launcher = Launcher.__new__(Launcher)
    launcher.loop_counters = [0, 0, 0, 0]

    launcher.set_loop_counter(3, 17)
    assert launcher.loop_counters == [0, 0, 0, 17]
    with pytest.raises(ValueError, match="register"):
        launcher.set_loop_counter(4, 1)
    with pytest.raises(ValueError, match="uint32"):
        launcher.set_loop_counter(0, -1)
    with pytest.raises(ValueError, match="uint32"):
        launcher.set_loop_counter(0, 1 << 32)


def test_three_way_silu_schedule_maps_one_2048_shard_per_aux_sm(monkeypatch):
    monkeypatch.setattr("dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr())
    gate = torch.empty((8, 6144), dtype=torch.bfloat16)
    up = torch.empty_like(gate)
    out = torch.empty_like(gate)
    schedule = (
        SchedSmemSiLUInterleaved(8, gate, up, out, shards_per_token=3)
        .bar("input", 4)
        .bar("output", 5)
        .place(24, base_sm=128)
    )

    first = schedule.schedule(0)
    last = schedule.schedule(23)
    assert first[0].opcode == opcode.OP_SILU_MUL_SHARED_BF16_K_2048_INTER
    assert first[1].size == last[1].size == 2048 * gate.element_size()
    assert cords2addr(first[2].cords) == gate[0, :2048].data_ptr()
    assert cords2addr(last[2].cords) == gate[7, 4096:].data_ptr()
    assert schedule.bar_release_count("output") == 24

    with pytest.raises(AssertionError, match="Three-way"):
        SchedSmemSiLUInterleaved(
            8, gate, up, out, shards_per_token=3
        ).place(8)


def test_launch_sequence_forwards_all_counter_snapshots(monkeypatch):
    launcher = Launcher.__new__(Launcher)
    launcher.num_sms = 152
    launcher.smem_size = 202 * 1024
    launcher.bars = torch.empty(1)
    buffers = tuple(torch.empty(1) for _ in range(4))
    launcher._prepare_runtime_launch = lambda reset_bars: (*buffers, 1234)

    captured = {}

    def fake_launch(*args):
        captured["args"] = args
        return 0

    monkeypatch.setattr("dae.launcher.runtime.launch_dae_sequence", fake_launch)
    counters = [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 2, 1]]
    launcher.launch_sequence(counters, synchronize=False, reset_bars=False)

    args = captured["args"]
    assert args[0:2] == (152, 202 * 1024)
    assert args[7] == counters
    assert args[8:] == (1234, False)


def test_launch_sequence_rejects_empty_plan_before_preparation():
    launcher = Launcher.__new__(Launcher)
    with pytest.raises(ValueError, match="must not be empty"):
        launcher.launch_sequence([])
