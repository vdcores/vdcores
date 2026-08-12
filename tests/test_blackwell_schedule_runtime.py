from pathlib import Path

import pytest
import torch

from dae.instructions import (
    AffineRoutedTmaLoad1D,
    AffineRoutedTmaLoadBase1D,
    ATTN_SPLIT_POST_REDUCE,
    ATTENTION_M64N64K16_F16_F32_64_64_hdim,
    ATTENTION_SM100_BF16_HDIM128_DIRECT,
    ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT,
    ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT,
    ARGMAX_REDUCE_GLOBAL_bf16_256,
    ComputeInstruction,
    Copy,
    Dsv4ContiguousAttention512UmmaSm100,
    Dsv4ContiguousAttention512UmmaTail32Sm100,
    Dsv4HcPost,
    Dsv4Fp8QuantUmmaBSm100,
    Dsv4Nvfp4QuantUmmaBSm100,
    Dsv4PreloadRopeTables,
    Dsv4Rope128_64,
    Dsv4Rope512_64,
    Dsv4SiluClampMul128,
    Gemv_M128N8Argmax4,
    Gemv_M128N8Direct4,
    Gemv_M128N8_ROPE_128,
    Gemv_M64N8_ROPE_128,
    Gemv_M64N8UpSiLU,
    LoopC,
    LoopM,
    MemoryInstruction,
    IndirectRoutedTmaLoadBase1D,
    IndirectRoutedTmaLoad1D,
    IndirectTmaLoad1D,
    LduProfileLayer,
    LduSetAffineExpertBase,
    Fp8GemvUmmaStreamSm100,
    Fp8UmmaPrepackSm100,
    Nvfp4GemvSm100,
    Nvfp4GemvUmmaStreamSm100,
    Nvfp4UmmaPrepackSm100,
    ProfileEvent,
    ProfileStep,
    RepeatM,
    ResetIndirectLayer,
    RoutedTmaLoad1D,
    RegLoad,
    TmaLoadAddressReg1D,
    TmaLoadReg1D,
    TmaTensor,
)
from dae.runtime import config, opcode
from dae.deepseek_v4_schedule import DeepSeekV4ShapePolicy
from dae.launcher import Launcher
from dae.schedule import (
    Schedule,
    SchedDsv4HcPost,
    SchedDsv4PreloadRopeTables,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedFp8GemvUmmaStream,
    SchedAttentionDecoding,
    SchedDsv4SwiGluShard128,
    SchedRoutedNvfp4GemvUmmaStream,
    SchedSmemSiLUInterleaved,
    SubgridSchedule,
)
from dae.sequential import (
    LoopedSequentialProgram,
    SequentialBlock,
    SequentialProgram,
    SequentialStage,
)
from dae.tma_utils import (
    cords2addr,
    cord_func_2d_kmajor,
    cord_func_2d_tile_major,
    cord_func_m128n8_grouped_output,
    cord_func_rowmajor_2d,
    pack_weight_tile_major,
)


def test_tile_major_weight_pack_round_trip_and_coordinates():
    tile_m, tile_k = 64, 256
    weight = torch.arange(128 * 512, dtype=torch.int32).reshape(128, 512)
    packed = pack_weight_tile_major(weight, tile_m, tile_k)

    assert packed.shape == (2, 2, 4, 64, 64)
    recovered = packed.permute(1, 3, 0, 2, 4).reshape_as(weight)
    torch.testing.assert_close(recovered, weight, rtol=0, atol=0)

    cord = cord_func_2d_tile_major(packed, packed.dim())
    assert cord(64, 256) == [0, 0, 1, 1]


def test_kmajor_uint8_coordinates_use_128_byte_blocks():
    packed_fp4 = torch.empty((8, 256), dtype=torch.uint8)
    cord = cord_func_2d_kmajor(packed_fp4, 3)

    assert cord(0, 128) == [0, 0, 1]


def test_dsv4_umma_attention_opcode_and_context_gate():
    instruction = Dsv4ContiguousAttention512UmmaSm100(128)
    tail = Dsv4ContiguousAttention512UmmaTail32Sm100(160)
    shard0 = Dsv4ContiguousAttention512UmmaSm100(128, output_tile=0)
    shard3 = Dsv4ContiguousAttention512UmmaTail32Sm100(
        160, output_tile=3
    )

    assert instruction.opcode == opcode.OP_DSV4_CONTIGUOUS_ATTENTION_512_UMMA_SM100
    assert instruction.args == [128]
    assert tail.opcode == (
        opcode.OP_DSV4_CONTIGUOUS_ATTENTION_512_UMMA_TAIL32_SM100
    )
    assert tail.args == [160]
    assert shard0.args == [128, 1]
    assert shard3.args == [160, 4]
    with pytest.raises(ValueError, match=r"\[1,128\]"):
        Dsv4ContiguousAttention512UmmaSm100(129)
    with pytest.raises(ValueError, match=r"\[129,160\]"):
        Dsv4ContiguousAttention512UmmaTail32Sm100(128)
    with pytest.raises(ValueError, match=r"\[0,3\]"):
        Dsv4ContiguousAttention512UmmaSm100(128, output_tile=4)


def test_rowmajor_2d_tma_coordinates_preserve_row_and_column():
    output = torch.empty((64, 512), dtype=torch.bfloat16)
    cord = cord_func_rowmajor_2d(output, 2)

    assert cord(0, 0) == [0, 0]
    assert cord(32, 384) == [384, 32]


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


def test_memory_instruction_accepts_maximum_uint16_address_chunk():
    instruction = MemoryInstruction(
        opcode.OP_RESET_INDIRECT_LAYER,
        num_slots=0,
        arg=0,
        size=0,
        cords=[0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF],
    )

    assert instruction.cords == [0xFFFF] * 4


def test_routed_tma_load_encodes_l2_lookup_and_shared_span():
    class FakeDevice:
        type = "cuda"

    class FakeRoutingState:
        device = FakeDevice()

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 6

        @staticmethod
        def element_size():
            return 8

        @staticmethod
        def data_ptr():
            return 0x123456789ABC

    instruction = RoutedTmaLoad1D(
        FakeRoutingState(), 5, 257, 16384
    ).bar(9)

    assert instruction.opcode == opcode.OP_ALLOC_ROUTED_TMA_LOAD_1D | 16
    assert instruction.num_slots == 2 | (9 << 6)
    assert instruction.arg == (257 << 3) | 5
    assert instruction.size == 16384
    assert cords2addr(instruction.cords) == FakeRoutingState.data_ptr()


def test_affine_routed_load_encodes_rank_refresh_and_layer_stride():
    load = AffineRoutedTmaLoad1D(
        0x123456780,
        5,
        16384,
        refresh_route=True,
    ).bar(9)
    base = AffineRoutedTmaLoadBase1D(
        0x123456780,
        5,
        16384,
    )

    assert load.opcode == opcode.OP_ALLOC_AFFINE_ROUTED_TMA_LOAD_1D | 16
    assert load.num_slots == 2 | (9 << 6)
    assert load.arg == 8 | 5
    assert cords2addr(load.cords) == 0x123456780
    assert base.opcode == opcode.OP_ALLOC_AFFINE_ROUTED_TMA_LOAD_BASE_1D
    assert base.arg == 5


def test_affine_expert_preload_encodes_fixed_base_and_geometry():
    class FakeDevice:
        type = "cuda"

    class FakeStorage:
        device = FakeDevice()
        dtype = torch.uint8

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def data_ptr():
            return 0x3456789ABC00

    instruction = LduSetAffineExpertBase(
        FakeStorage(), 14_156_032, 256, initial_layer=42, special_slot=2
    )

    assert instruction.opcode == opcode.OP_LDU_SET_AFFINE_EXPERT_BASE
    assert instruction.num_slots == config.num_slots + 2
    assert instruction.size == 14_156_032 // 256
    assert instruction.arg == (42 << 9) | 256
    assert cords2addr(instruction.cords) == FakeStorage.data_ptr()


def test_indirect_loads_keep_address_selection_in_ldu():
    class FakeDevice:
        type = "cuda"

    class FakePointerTable:
        device = FakeDevice()
        dtype = torch.int64

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 2

        @staticmethod
        def data_ptr():
            return 0x23456789ABC0

    direct = IndirectTmaLoad1D(FakePointerTable(), 4096)
    layered = IndirectTmaLoad1D(
        FakePointerTable(), 4096, layer_indexed=True
    )
    routed = IndirectRoutedTmaLoad1D(
        FakePointerTable(),
        4,
        511,
        16384,
        layer_indexed=True,
    )
    routed_base = IndirectRoutedTmaLoadBase1D(
        FakePointerTable(),
        4,
        511,
        16384,
        layer_indexed=True,
    )

    assert direct.opcode == opcode.OP_ALLOC_INDIRECT_TMA_LOAD_1D
    assert layered.opcode == opcode.OP_ALLOC_LAYER_TMA_LOAD_1D
    assert direct.num_slots == 1
    assert cords2addr(direct.cords) == FakePointerTable.data_ptr()
    assert routed.opcode == opcode.OP_ALLOC_LAYER_ROUTED_TMA_LOAD_1D
    assert routed.arg == (511 << 3) | 4
    assert routed.num_slots == 2
    assert routed_base.opcode == opcode.OP_ALLOC_LAYER_ROUTED_TMA_LOAD_BASE_1D
    assert routed_base.arg == routed.arg
    assert routed_base.num_slots == routed.num_slots


def test_sequential_program_uses_stu_release_and_gates_both_ldu_ports():
    class FakeLauncher:
        num_sms = 2
        num_bars = 0
        max_insts = 32

        def __init__(self):
            self.bar_values = {}

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            self.bar_values[bar_id] = count
            return bar_id

    class TwoPortStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ).port(1),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    launcher = FakeLauncher()
    program = SequentialProgram(
        launcher,
        [
            SequentialStage("producer", TwoPortStage(), 2),
            SequentialStage("consumer", TwoPortStage(), 2),
        ],
    )

    assert launcher.bar_values == {0: 2}
    assert len(program.placed_schedules) == 2
    for instructions in program.instructions:
        memory = [
            inst for inst in instructions if isinstance(inst, MemoryInstruction)
        ]
        producer_store = memory[2]
        consumer_load0, consumer_load1 = memory[3:5]
        assert producer_store.opcode & 0x10
        assert producer_store.num_slots >> 6 == 0
        assert consumer_load0.opcode & 0x10
        assert consumer_load1.opcode & 0x10
        assert consumer_load0.num_slots >> 6 == 0
        assert consumer_load1.num_slots >> 6 == 0


def test_sequential_program_balances_default_loads_by_size():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 32

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            return bar_id

    class ThreeLoadStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            loads = [
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=size,
                    address=0,
                )
                for size in (64, 16, 16)
            ]
            return [
                Copy(1, 16),
                *loads,
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    program = SequentialProgram(
        FakeLauncher(),
        [SequentialStage("balanced", ThreeLoadStage(), 1)],
        balance_load_ports=True,
    )
    loads = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
        and inst.opcode & 0x1
        and not inst.opcode & 0x2
    ]
    assert [bool(inst.opcode & 0x20) for inst in loads] == [False, True, True]


def test_sequential_program_preserves_fixed_load_ports():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 16

        def new_bar(self, count):
            self.num_bars += 1
            return self.num_bars - 1

    class FixedLoadStage(Schedule):
        def schedule(self, sm):
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_TMA_LOAD_REG_1D,
                    num_slots=1,
                    arg=0,
                    size=1024,
                    address=0,
                ).fixed_port(1),
                MemoryInstruction(
                    opcode.OP_ALLOC_TMA_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    program = SequentialProgram(
        FakeLauncher(),
        [SequentialStage("fixed", FixedLoadStage(), 1)],
        balance_load_ports=True,
    )
    loads = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
        and inst.opcode & 0x1
        and not inst.opcode & 0x2
    ]
    assert loads[0].annotation["fixed_port"] == 1
    assert [bool(inst.opcode & 0x20) for inst in loads] == [True, False]


def test_nvfp4_gemv_can_retain_activation_slots():
    instruction = Nvfp4GemvSm100(16, 4096, retain_activation=True)

    assert instruction.args == [16, 4096, 1]


def test_nvfp4_streaming_umma_encodes_k_tile_count():
    instruction = Nvfp4GemvUmmaStreamSm100(16)

    assert instruction.args == [16, 0, 0]


def test_nvfp4_streaming_umma_encodes_retained_bulk_activation():
    instruction = Nvfp4GemvUmmaStreamSm100(
        16, retain_activation=True, bulk_activation=True
    )

    assert instruction.args == [16, 1, 1]


def test_nvfp4_umma_prepack_encodes_kind_and_k_tile_count():
    instruction = Nvfp4UmmaPrepackSm100(
        Nvfp4UmmaPrepackSm100.ACTIVATION, 16
    )

    assert instruction.args == [1, 16]


def test_dsv4_nvfp4_native_quant_encodes_k_tile_count():
    instruction = Dsv4Nvfp4QuantUmmaBSm100(1)

    assert instruction.args == [1]


def test_fp8_streaming_umma_encodes_k_tiles():
    instruction = Fp8GemvUmmaStreamSm100(64)

    assert instruction.args == [64]


def test_fp8_umma_prepack_encodes_kind_and_k_tile_count():
    instruction = Fp8UmmaPrepackSm100(
        Fp8UmmaPrepackSm100.ACTIVATION, 32
    )

    assert instruction.args == [1, 32]


def test_dsv4_fp8_native_quant_encodes_k_tile_count():
    instruction = Dsv4Fp8QuantUmmaBSm100(1)

    assert instruction.args == [1]


def test_dsv4_hc_post_encodes_local_width():
    instruction = Dsv4HcPost(128)

    assert instruction.args == [128]


def test_dsv4_hc_post_uses_shape_aligned_shared_shards(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    schedule = SchedDsv4HcPost(
        torch.empty((4096,), dtype=torch.bfloat16),
        torch.empty((4, 4096), dtype=torch.bfloat16),
        torch.empty((4,), dtype=torch.float32),
        torch.empty((4, 4), dtype=torch.float32),
        torch.empty((4, 4096), dtype=torch.bfloat16),
    ).place(32)

    instructions = schedule.schedule(0)
    compute = [inst for inst in instructions if isinstance(inst, Dsv4HcPost)]
    memory = [
        inst for inst in instructions if isinstance(inst, MemoryInstruction)
    ]

    assert [inst.args for inst in compute] == [[128]]
    assert len(memory) == 11
    assert all(inst.num_slots == 1 for inst in memory)
    assignment = DeepSeekV4ShapePolicy(152).hc_post(4096, 4)
    assert assignment.num_sms == 32
    assert assignment.row_alignment == 128


def test_fp8_native_stream_bounds_activation_chunks_to_one_slot(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    schedule = SchedFp8GemvUmmaStream(
        torch.empty((3, 8, 16896), dtype=torch.uint8),
        torch.empty((8, 2048), dtype=torch.uint8),
        torch.empty((384,), dtype=torch.bfloat16),
    ).place(2)

    instructions = schedule.schedule(0)
    compute = [
        inst for inst in instructions
        if isinstance(inst, Fp8GemvUmmaStreamSm100)
    ]

    assert [inst.args for inst in compute] == [[8], [8]]
    assert not any(isinstance(inst, TmaLoadReg1D) for inst in instructions)
    assert not any(isinstance(inst, RegLoad) for inst in instructions)
    activation_loads = [
        inst
        for inst in instructions
        if isinstance(inst, MemoryInstruction) and inst.size == 4 * 2048
    ]
    weight_loads = [
        inst
        for inst in instructions
        if isinstance(inst, MemoryInstruction) and inst.size == 16896
    ]
    assert len(activation_loads) == 4
    assert all(inst.num_slots == 1 for inst in activation_loads)
    assert len(weight_loads) == 16
    assert all(inst.num_slots == 3 for inst in weight_loads)


def test_dsv4_shard_swiglu_encodes_bound_and_width():
    instruction = Dsv4SiluClampMul128(1, 10.0)

    assert instruction.opcode == opcode.OP_DSV4_SILU_CLAMP_MUL_128
    assert instruction.args[0] == 1


def test_dsv4_shard_swiglu_consumes_port_local_registers(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    instructions = SchedDsv4SwiGluShard128(
        1,
        0,
        1,
        1,
        torch.empty((256,), dtype=torch.bfloat16),
    ).place(2).schedule(0)

    assert instructions[0].opcode == opcode.OP_DSV4_SILU_CLAMP_MUL_128
    assert instructions[2].size == instructions[3].size == 1
    assert not instructions[2].opcode & 32
    assert instructions[3].opcode & 32


def test_routed_address_register_load_encodes_fixed_offset():
    instruction = TmaLoadAddressReg1D(0, 18432, 18432)

    assert instruction.opcode == opcode.OP_ALLOC_TMA_LOAD_ADDRESS_REG_1D
    assert instruction.num_slots == 3
    assert cords2addr(instruction.cords) == 18432


def test_routed_native_gemv_gates_both_ldu_ports_on_route(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )

    class FakeDevice:
        type = "cuda"

    class FakeRoutingState:
        device = FakeDevice()

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 6

        @staticmethod
        def element_size():
            return 8

        @staticmethod
        def data_ptr():
            return 0x123456789ABC

    instructions = (
        SchedRoutedNvfp4GemvUmmaStream(
            FakeRoutingState(),
            0,
            (1,),
            2,
            torch.empty((2, 3072), dtype=torch.uint8),
            torch.empty((128,), dtype=torch.bfloat16),
        )
        .bar("route", 7)
        .place(1)
        .schedule(0)
    )
    routed_loads = [
        inst
        for inst in instructions
        if isinstance(inst, MemoryInstruction)
        and (inst.opcode & ~0x3F)
        in (
            opcode.OP_ALLOC_ROUTED_TMA_LOAD_1D & ~0x3F,
            opcode.OP_ALLOC_ROUTED_TMA_LOAD_BASE_1D & ~0x3F,
        )
    ]

    assert len(routed_loads) == 2
    assert all(inst.opcode & 16 for inst in routed_loads)
    assert {inst.num_slots >> 6 for inst in routed_loads} == {7}
    assert {bool(inst.opcode & 32) for inst in routed_loads} == {False, True}


def test_subgrid_schedule_keeps_shape_placement_inside_global_stage():
    class Inner(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [ComputeInstruction(0x1234, [sm])]

    placed = SubgridSchedule(Inner(), 2, 3).place(8)

    assert [len(placed(sm)) for sm in range(8)] == [0, 0, 0, 1, 1, 0, 0, 0]
    assert placed(3)[0].args == [0]
    assert placed(4)[0].args == [1]


def test_sequential_program_elides_only_same_placement_independent_edge():
    class FakeLauncher:
        num_sms = 2
        num_bars = 0
        max_insts = 32

        def __init__(self):
            self.bar_values = {}

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            self.bar_values[bar_id] = count
            return bar_id

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    launcher = FakeLauncher()
    program = SequentialProgram(
        launcher,
        (
            SequentialStage("first", BasicStage(), 2),
            SequentialStage(
                "independent", BasicStage(), 2, wait_for_previous=False
            ),
            SequentialStage("join", BasicStage(), 2),
        ),
    )
    assert launcher.bar_values == {0: 2}
    memory = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
    ]
    first_store = memory[1]
    independent_store = memory[3]
    join_load = memory[4]
    assert not first_store.opcode & 0x10
    assert independent_store.opcode & 0x10
    assert join_load.opcode & 0x10

    with pytest.raises(ValueError, match="must match the previous stage placement"):
        SequentialProgram(
            FakeLauncher(),
            (
                SequentialStage("first", BasicStage(), 2),
                SequentialStage(
                    "bad", BasicStage(), 1, wait_for_previous=False
                ),
            ),
        )


def test_sequential_profile_markers_can_use_reserved_special_slots():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 32

        def __init__(self):
            self.bar_values = {}

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            self.bar_values[bar_id] = count
            return bar_id

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    program = SequentialProgram(
        FakeLauncher(),
        (
            SequentialStage("profiled", BasicStage(), 1, profile_after=True),
            SequentialStage("tail", BasicStage(), 1),
        ),
        profile_special_slot=7,
    )
    marker = next(
        inst
        for inst in program.instructions[0]
        if (inst.opcode & ~0x3F) == (opcode.OP_LDU_PROFILE_LAYER & ~0x3F)
    )

    assert marker.num_slots & 0x3F == config.num_slots + 7


def test_sequential_program_fans_out_and_joins_labeled_stage_groups():
    class FakeLauncher:
        num_sms = 4
        num_bars = 0
        max_insts = 32

        def __init__(self):
            self.bar_values = {}

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            self.bar_values[bar_id] = count
            return bar_id

        def set_bar(self, bar_id, count):
            self.bar_values[bar_id] = count

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    launcher = FakeLauncher()
    program = SequentialProgram(
        launcher,
        (
            SequentialStage(
                "producer", BasicStage(), 2, release_group="ready"
            ),
            SequentialStage(
                "branch0",
                BasicStage(),
                2,
                wait_group="ready",
                release_group="join",
            ),
            SequentialStage(
                "branch1",
                BasicStage(),
                2,
                base_sm=2,
                profile_after=True,
                wait_group="ready",
                release_group="join",
            ),
            SequentialStage(
                "consumer", BasicStage(), 4, wait_group="join"
            ),
        ),
    )

    assert launcher.bar_values == {0: 2, 1: 4}
    assert program.barriers == [0, 1]
    marker = next(
        inst
        for inst in program.instructions[0]
        if (inst.opcode & ~0x3F) == (opcode.OP_LDU_PROFILE_LAYER & ~0x3F)
    )
    assert marker.num_slots >> 6 == 1
    for sm in range(4):
        memory = [
            inst
            for inst in program.instructions[sm]
            if isinstance(inst, MemoryInstruction)
        ]
        if sm < 2:
            assert memory[1].num_slots >> 6 == 0
        assert memory[-2].num_slots >> 6 == 1


def test_sequential_program_rejects_wait_group_without_producer():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 8

    with pytest.raises(ValueError, match="groups with no producers"):
        SequentialProgram(
            FakeLauncher(),
            (
                SequentialStage(
                    "consumer", Schedule(), 1, wait_group="missing"
                ),
            ),
        )


def test_sequential_program_binds_model_specific_input_role_to_same_edge():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 32

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            return bar_id

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            load = MemoryInstruction(
                opcode.OP_ALLOC_LDU_LOAD_1D,
                num_slots=1,
                arg=0,
                size=16,
                address=0,
            )
            if self._bar("route") is not None:
                load.bar(self._bar("route"))
            return [
                Copy(1, 16),
                load,
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    program = SequentialProgram(
        FakeLauncher(),
        [
            SequentialStage("route", BasicStage(), 1),
            SequentialStage("expert", BasicStage(), 1, input_role="route"),
        ],
    )
    expert_load = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
    ][2]
    assert expert_load.opcode & 0x10
    assert expert_load.num_slots >> 6 == 0


def test_sequential_program_prefetches_static_load_before_explicit_input_gate():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 32

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            return bar_id

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            loads = []
            for address in (0x1000, 0x2000):
                loads.append(
                    MemoryInstruction(
                        opcode.OP_ALLOC_LDU_LOAD_1D,
                        num_slots=1,
                        arg=0,
                        size=16,
                        address=address,
                    )
                )
            if self._bar("activation") is not None:
                loads[1].bar(self._bar("activation"))
            return [
                Copy(1, 16),
                *loads,
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0x3000,
                ),
            ]

    program = SequentialProgram(
        FakeLauncher(),
        [
            SequentialStage("producer", BasicStage(), 1),
            SequentialStage(
                "consumer",
                BasicStage(),
                1,
                input_role="activation",
                prefetch_before_wait=True,
            ),
        ],
    )
    memory = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
    ]
    consumer_static, consumer_activation = memory[3:5]
    assert not consumer_static.opcode & 0x10
    assert consumer_activation.opcode & 0x10
    assert consumer_activation.num_slots >> 6 == 0


def test_looped_program_reloads_dependencies_in_ldu_without_issue_barrier():
    class FakeDevice:
        type = "cuda"

    class FakeBarrierSource:
        device = FakeDevice()

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 1 << 20

        @staticmethod
        def element_size():
            return 4

        @staticmethod
        def data_ptr():
            return 0x12340000

    class FakeLauncher:
        num_sms = 2
        num_bars = 0
        max_insts = 64
        bars_src = FakeBarrierSource()

        def __init__(self):
            self.bar_values = {}

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            self.bar_values[bar_id] = count
            return bar_id

        def copy_cptrs(self):
            return [0] * self.num_sms

        def copy_mptrs(self):
            return [0] * self.num_sms

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    launcher = FakeLauncher()
    program = LoopedSequentialProgram(
        launcher,
        (
            SequentialBlock(
                "body",
                (
                    SequentialStage("producer", BasicStage(), 2),
                    SequentialStage(
                        "consumer", BasicStage(), 2, profile_after=True
                    ),
                ),
                repeat=2,
                barrier_banks=2,
            ),
            SequentialBlock(
                "tail",
                (SequentialStage("tail", BasicStage(), 2),),
                reload_after=False,
            ),
        ),
    )

    assert launcher.bar_values == {0: 2, 1: 2, 2: 2, 3: 2}
    compute = [
        inst for inst in program.instructions[0]
        if isinstance(inst, ComputeInstruction)
    ]
    memory = [
        inst for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
    ]
    assert isinstance(compute[2], LoopC)
    assert compute[2].args == [2, 0, 0]
    reload = next(
        inst
        for inst in memory
        if (inst.opcode & ~0x3F) == (opcode.OP_LDU_RELOAD_BARRIERS & ~0x3F)
    )
    assert (reload.opcode & ~0x3F) == (
        opcode.OP_LDU_RELOAD_BARRIERS & ~0x3F
    )
    assert reload.opcode & 0x4
    assert reload.num_slots & 0x3F == config.num_slots + 2
    assert reload.num_slots >> 6 == 1
    assert reload.arg == 0
    assert reload.size == 2
    # Bank zero spans barriers [0, 2); the shifted bank spans [2, 4).
    assert (reload.num_slots >> 6) + 1 - reload.size == 0
    assert (reload.num_slots >> 6) + 2 + 1 - reload.size == 2
    profile_layer = next(
        inst
        for inst in memory
        if (inst.opcode & ~0x3F) == (opcode.OP_LDU_PROFILE_LAYER & ~0x3F)
    )
    assert profile_layer.opcode & 0x4
    assert profile_layer.num_slots >> 6 == 1
    assert profile_layer.arg == 2
    assert profile_layer.size == 2
    loop = next(inst for inst in memory if isinstance(inst, LoopM))
    assert loop.size == 2
    assert loop.cords[0] == 1
    assert loop.cords[1] == 1
    assert loop.cords[2] == 2 << 6
    assert memory.index(profile_layer) < memory.index(reload) < memory.index(loop)
    shifted_body = [
        inst
        for inst in memory
        if inst.opcode & 0x10 and (inst.num_slots >> 6) < 2
    ]
    assert shifted_body
    assert all(inst.opcode & 0x4 for inst in shifted_body)
    assert all((inst.opcode & ~0x10) != opcode.OP_ISSUE_BARRIER for inst in memory)


def test_looped_program_nests_two_barrier_banks_inside_ten_outer_iterations():
    class FakeDevice:
        type = "cuda"

    class FakeBarrierSource:
        device = FakeDevice()

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 1 << 20

        @staticmethod
        def element_size():
            return 4

        @staticmethod
        def data_ptr():
            return 0x34560000

    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 64
        bars_src = FakeBarrierSource()

        def __init__(self):
            self.bar_values = {}

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            self.bar_values[bar_id] = count
            return bar_id

        def copy_cptrs(self):
            return [0]

        def copy_mptrs(self):
            return [0]

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    program = LoopedSequentialProgram(
        FakeLauncher(),
        (
            SequentialBlock(
                "pairs",
                (SequentialStage("pair", BasicStage(), 1),),
                repeat=20,
                barrier_banks=2,
            ),
        ),
    )
    compute_loops = [
        inst for inst in program.instructions[0] if isinstance(inst, LoopC)
    ]
    memory_loops = [
        inst for inst in program.instructions[0] if isinstance(inst, LoopM)
    ]
    assert isinstance(program.instructions[0][0], ResetIndirectLayer)
    assert [inst.args for inst in compute_loops] == [[2, 0, 0], [10, 0, 1]]
    assert [(inst.size, inst.num_slots) for inst in memory_loops] == [(2, 0), (10, 1)]
    assert [inst.cords[1] for inst in memory_loops] == [1, 0]


def test_deepseek_shape_policy_assigns_complete_scale_and_row_tiles():
    policy = DeepSeekV4ShapePolicy(152)

    fp8 = policy.fp8_gemv(4096, 4096)
    fp4 = policy.nvfp4_gemv(2048, 4096)
    quant = policy.quantize(4096, 16)
    attention = policy.attention(64, 512)

    assert (fp8.num_sms, fp8.tile_rows, fp8.tile_k) == (152, 15, 128)
    assert (
        fp4.num_sms,
        fp4.row_alignment,
        fp4.tile_rows,
        fp4.tile_k,
    ) == (152, 8, 24, 256)
    assert [policy.parallel_partition(branch, 8) for branch in range(8)] == [
        (0, 19),
        (19, 19),
        (38, 19),
        (57, 19),
        (76, 19),
        (95, 19),
        (114, 19),
        (133, 19),
    ]
    assert [
        policy.uniform_parallel_partition(branch, 6)
        for branch in range(6)
    ] == [
        (0, 25),
        (25, 25),
        (50, 25),
        (75, 25),
        (100, 25),
        (125, 25),
    ]
    assert [
        policy.weighted_parallel_partition(branch, (1024, 512))
        for branch in range(2)
    ] == [(0, 101), (101, 51)]
    assert all(fp4.shard(sm)[1] % 8 == 0 for sm in range(fp4.num_sms))
    assert (quant.num_sms, quant.tile_rows) == (16, 16)
    assert attention.num_sms == 64


def test_nvfp4_gemv_encodes_shared_shard_shape():
    instruction = Nvfp4GemvSm100(32, 256)

    assert instruction.args == [32, 256, 0]


def test_deepseek_compute_tasks_cannot_escape_shared_memory():
    root = Path(__file__).resolve().parents[1]
    sources = [
        root / "include/task/deepseek_v4.cuh",
        root / "include/task/fp8.cuh",
        root / "include/task/nvfp4.cuh",
        root / "include/task/nvfp4_umma.cuh",
    ]
    combined = "\n".join(source.read_text() for source in sources)

    assert "slot_2_glob_ptr" not in combined
    assert "const MInst *st_insts" not in combined
    assert "__threadfence" not in combined


def test_resident_rope_tables_preload_once_and_fixed_tasks_skip_table_loads(
    monkeypatch,
):
    monkeypatch.setitem(
        SchedDsv4PreloadRopeTables.schedule.__globals__[
            "TmaLoad1D"
        ].__init__.__globals__,
        "get_tensor_address",
        lambda tensor: tensor.data_ptr(),
    )
    tables = tuple(
        torch.empty((32, 2), dtype=torch.float32) for _ in range(4)
    )
    preload = SchedDsv4PreloadRopeTables(tables).place(2)
    preload_instructions = preload.schedule(0)

    assert isinstance(preload_instructions[0], Dsv4PreloadRopeTables)
    assert preload_instructions[0].args == [4]
    assert len(preload_instructions) == 5
    assert all(
        instruction.opcode == opcode.OP_ALLOC_TMA_LOAD_1D
        for instruction in preload_instructions[1:]
    )

    input512 = torch.empty((1, 512), dtype=torch.bfloat16)
    output512 = torch.empty_like(input512)
    fixed512 = SchedDsv4Rope512_64(
        input512,
        tables[2],
        output512,
        inverse=True,
        fixed_table_id=2,
    ).place(1).schedule(0)
    assert isinstance(fixed512[0], Dsv4Rope512_64)
    assert fixed512[0].args == [1, 1, 3]
    assert len(fixed512) == 3

    input128 = torch.empty((1, 128), dtype=torch.bfloat16)
    output128 = torch.empty_like(input128)
    fixed128 = SchedDsv4Rope128_64(
        input128,
        tables[1],
        output128,
        fixed_table_id=1,
    ).place(1).schedule(0)
    assert isinstance(fixed128[0], Dsv4Rope128_64)
    assert fixed128[0].args == [1, 0, 2]
    assert len(fixed128) == 3

    dynamic = SchedDsv4Rope512_64(
        input512, tables[0], output512
    ).place(1).schedule(0)
    assert dynamic[0].args == [1, 0, 0]
    assert len(dynamic) == 4


def test_resident_rope_preload_barrier_is_compute_group_only():
    source = (
        Path(__file__).resolve().parents[1]
        / "include/task/deepseek_v4.cuh"
    ).read_text()
    handler = source.split(
        "task_dsv4_preload_rope_tables", 1
    )[1].split("task_dsv4_rope_64", 1)[0]

    assert handler.count("__sync_compute_group(128)") == 1
    assert "__threadfence" not in handler


def test_resident_loop_control_uses_only_compute_group_rendezvous():
    source = (
        Path(__file__).resolve().parents[1]
        / "include/dae/compute_dispatch.cuh"
    ).read_text()
    handler = source.split(
        "DAE_COMPUTE_OP_HANDLER(OP_LOOPC)", 1
    )[1].split("DAE_COMPUTE_OP_HANDLER(OP_TERMINATEC)", 1)[0]

    assert handler.count("__sync_compute_group(128)") == 1
    assert "__threadfence" not in handler


def test_profile_event_reserves_kernel_start_and_end_slots():
    instruction = ProfileEvent(17)

    assert instruction.opcode == opcode.OP_PROFILE_EVENT
    assert instruction.args == [17, 0]
    assert ProfileEvent(18, wait_for_memory=True).args == [18, 1]
    with pytest.raises(ValueError, match="slots 0/1"):
        ProfileEvent(1)


def test_profile_step_encodes_paired_counter_modes():
    assert ProfileStep(17, begin=True).args == [17, 2]
    assert ProfileStep(17, begin=False).args == [17, 3]
    with pytest.raises(ValueError, match="layer-profile range"):
        ProfileStep(config.reload_profile_event_base, begin=True)


def test_sequential_step_profile_does_not_add_a_dependency_barrier():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 32

        def new_bar(self, count):
            bar_id = self.num_bars
            self.num_bars += 1
            return bar_id

    class BasicStage(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ),
            ]

    launcher = FakeLauncher()
    program = SequentialProgram(
        launcher,
        (
            SequentialStage(
                "profiled",
                BasicStage(),
                1,
                profile_step_event=config.layer_profile_event_base,
            ),
        ),
    )
    compute = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, ComputeInstruction)
    ]
    assert [inst.args[1] for inst in compute if isinstance(inst, ProfileStep)] == [2, 3]
    assert launcher.num_bars == 0


def test_ldu_layer_profile_encodes_internal_counter_range():
    assert config.layer_profile_event_base == 2
    assert config.reload_profile_event_base == 64
    assert config.track_profile_event_base == 96
    instruction = LduProfileLayer(config.layer_profile_event_base, 43).bar(7)

    assert (instruction.opcode & ~0x3F) == (
        opcode.OP_LDU_PROFILE_LAYER & ~0x3F
    )
    assert instruction.num_slots & 0x3F == config.num_slots
    assert instruction.num_slots >> 6 == 7
    assert instruction.arg == config.layer_profile_event_base
    assert instruction.size == 43


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


def test_swapped_attention_preserves_runtime_fields_and_requires_kv128():
    instruction = ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT(
        num_kv_block=1,
        num_active_q=4,
        last_kv_active_token_len=1,
        seq_len_counter_reg=1,
        num_kv_block_counter_reg=3,
        outer_seq_len_counter_reg=2,
        outer_seq_len_counter_stride=1,
    )

    assert instruction.opcode == opcode.OP_ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT
    assert instruction.args == [0x0101, 0x0184, 0x213C]
    with pytest.raises(AssertionError, match="requires KV128"):
        ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT(1, 4, 1, kv_block_size=64)


def test_swapped_split_attention_and_raw_reducer_flags():
    instruction = ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT(
        num_kv_block=2,
        split_idx=3,
        num_active_q=4,
        last_kv_active_token_len=117,
        kv_start_idx=768,
    )
    reducer = ATTN_SPLIT_POST_REDUCE(
        4, raw_partial=True, direct_output=True
    )

    assert instruction.opcode == opcode.OP_ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT
    assert instruction.args == [0x3002, 0x7584, 768]
    assert reducer.args == [4, 0x3]
    with pytest.raises(AssertionError):
        ATTN_SPLIT_POST_REDUCE(4, direct_output=True)


def test_decode_schedule_selects_swapped_attention_without_changing_defaults():
    output = torch.empty((8, 8, 4, 128), dtype=torch.bfloat16)
    swapped = SchedAttentionDecoding(
        reqs=8,
        seq_len=128,
        KV_BLOCK_SIZE=128,
        NUM_KV_HEADS=8,
        matO=output,
        tmas=(None, None, None),
        num_active_q=4,
        swapped_qk_pv=True,
    )
    default = SchedAttentionDecoding(
        reqs=8,
        seq_len=128,
        KV_BLOCK_SIZE=128,
        NUM_KV_HEADS=8,
        matO=output,
        tmas=(None, None, None),
        num_active_q=4,
    )

    assert swapped.AttentionInst is ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT
    assert default.AttentionInst is ATTENTION_SM100_BF16_HDIM128_DIRECT
    with pytest.raises(ValueError, match="requires direct HDIM128 output and KV128"):
        SchedAttentionDecoding(
            reqs=8,
            seq_len=64,
            KV_BLOCK_SIZE=64,
            NUM_KV_HEADS=8,
            matO=output,
            tmas=(None, None, None),
            num_active_q=4,
            swapped_qk_pv=True,
        )


def test_decode_schedule_maps_head_major_workers_to_per_head_barriers():
    output = torch.empty((8, 8, 4, 128), dtype=torch.bfloat16)
    q_bars = list(range(10, 18))
    kv_bars = list(range(20, 28))
    o_bars = [30, 30, 31, 31, 32, 32, 32, 32]
    sharded = SchedAttentionDecoding(
        reqs=8,
        seq_len=128,
        KV_BLOCK_SIZE=128,
        NUM_KV_HEADS=8,
        matO=output,
        tmas=(None, None, None),
        num_active_q=4,
        swapped_qk_pv=True,
        q_head_bars=q_bars,
        kv_head_bars=kv_bars,
        o_head_bars=o_bars,
        head_major=True,
    )
    default = SchedAttentionDecoding(
        reqs=8,
        seq_len=128,
        KV_BLOCK_SIZE=128,
        NUM_KV_HEADS=8,
        matO=output,
        tmas=(None, None, None),
        num_active_q=4,
        swapped_qk_pv=True,
    )

    assert sharded._map_req_head(8) == (0, 1)
    assert default._map_req_head(8) == (1, 0)
    assert sharded.q_head_bars[1] == 11
    assert sharded.kv_head_bars[1] == 21
    assert sharded.o_head_bars[5] == 32
    with pytest.raises(ValueError, match="provided together"):
        SchedAttentionDecoding(
            reqs=8,
            seq_len=128,
            KV_BLOCK_SIZE=128,
            NUM_KV_HEADS=8,
            matO=output,
            tmas=(None, None, None),
            q_head_bars=q_bars,
        )
    with pytest.raises(ValueError, match="output head-barrier"):
        SchedAttentionDecoding(
            reqs=8,
            seq_len=128,
            KV_BLOCK_SIZE=128,
            NUM_KV_HEADS=8,
            matO=output,
            tmas=(None, None, None),
            o_head_bars=o_bars[:-1],
        )


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


def test_up_silu_gemv_selects_the_sm100_overlap_opcode():
    instruction = Gemv_M64N8UpSiLU(kTiles=16)

    assert instruction.opcode == opcode.OP_GEMV_SM100_M64N8_UP_SILU
    assert instruction.args == [16]


def test_fused_rope_gemv_encodes_position_and_head_half():
    instruction = Gemv_M64N8_ROPE_128(
        kTiles=8,
        hist_len=127,
        head_dim_ofst=64,
    )

    assert instruction.opcode == opcode.OP_GEMV_M64N8_ROPE_128
    assert instruction.args == [8, 127, 64]

    m128_instruction = Gemv_M128N8_ROPE_128(
        kTiles=8,
        hist_len=127,
        head_dim_ofst=0,
    )
    assert m128_instruction.opcode == opcode.OP_GEMV_M128N8_ROPE_128
    assert m128_instruction.args == [8, 127, 0]


def test_fused_lm_head_argmax_encodes_absolute_vocab_offsets():
    instruction = Gemv_M128N8Argmax4(
        kTiles=32,
        output_group_stride=16384,
        vocabulary_base=65536,
    )
    reducer = ARGMAX_REDUCE_GLOBAL_bf16_256(num_active_token=1)

    assert instruction.opcode == opcode.OP_GEMV_SM100_M128N8_ARGMAX4
    assert instruction.args == [32, 128, 512]
    assert reducer.opcode == opcode.OP_ARGMAX_REDUCE_GLOBAL_bf16_256
    assert reducer.args == [1]
    with pytest.raises(ValueError, match="multiples of 128"):
        Gemv_M128N8Argmax4(32, 16384, 1)


def test_grouped_m128_output_uses_one_rank4_reduction_tile():
    output = torch.empty((8, 4096), dtype=torch.bfloat16)
    cord = cord_func_m128n8_grouped_output(
        output, 4, output_groups=4
    )

    assert cord(0, 0) == [0, 0, 0, 0]
    assert cord(0, 896) == [0, 0, 14, 0]
    tma = TmaTensor.__new__(TmaTensor)
    assert tma._rank2opcode(4, "reduce") == opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_4D


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


def test_three_way_silu_schedule_accepts_shard_local_barriers(monkeypatch):
    monkeypatch.setattr("dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr())
    gate = torch.empty((8, 6144), dtype=torch.bfloat16)
    up = torch.empty_like(gate)
    out = torch.empty_like(gate)
    schedule = SchedSmemSiLUInterleaved(
        8, gate, up, out, shards_per_token=3
    )
    for shard_id in range(3):
        schedule.bar(f"input{shard_id}", 10 + shard_id)
        schedule.bar(f"output{shard_id}", 20 + shard_id)
    schedule = schedule.place(24, base_sm=128)

    for shard_id in range(3):
        insts = schedule.schedule(shard_id)
        assert insts[1].num_slots >> 6 == 20 + shard_id
        assert insts[2].num_slots >> 6 == 10 + shard_id
        assert schedule.bar_release_count(f"output{shard_id}") == 8
    assert schedule.bar_release_count("output") == 0


def test_three_way_silu_schedule_can_pack_one_shard_per_sm_group(monkeypatch):
    monkeypatch.setattr("dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr())
    gate = torch.empty((8, 6144), dtype=torch.bfloat16)
    up = torch.empty_like(gate)
    out = torch.empty_like(gate)
    schedule = (
        SchedSmemSiLUInterleaved(
            8, gate, up, out, shards_per_token=3, fixed_shard_id=1
        )
        .bar("input1", 11)
        .bar("output1", 21)
        .place(8, base_sm=128)
    )

    first = schedule.schedule(0)
    last = schedule.schedule(7)
    assert cords2addr(first[2].cords) == gate[0, 2048:4096].data_ptr()
    assert cords2addr(last[2].cords) == gate[7, 2048:4096].data_ptr()
    assert first[1].num_slots >> 6 == 21
    assert first[2].num_slots >> 6 == 11
    assert schedule.bar_release_count("output1") == 8

    with pytest.raises(AssertionError):
        SchedSmemSiLUInterleaved(
            8, gate, up, out, shards_per_token=3, fixed_shard_id=3
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
