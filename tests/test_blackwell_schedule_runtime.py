from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dae.instructions import (
    ATTN_SPLIT_POST_REDUCE,
    ATTENTION_M64N64K16_F16_F32_64_64_hdim,
    ATTENTION_SM100_BF16_HDIM128_DIRECT,
    ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT,
    ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT,
    ARGMAX_REDUCE_GLOBAL_bf16_256,
    ComputeInstruction,
    Copy,
    Dsv4AttentionSplit64UmmaSm100,
    Dsv4AttentionSplitReduceFp8Sm100,
    Dsv4ContiguousAttention512UmmaSm100,
    Dsv4ContiguousAttention512UmmaTail32Sm100,
    Dsv4HcPreRms,
    Dsv4HcHeadRms,
    Dsv4HcPost,
    Dsv4IndexScore,
    Dsv4Fp8QuantUmmaBSm100,
    Dsv4Mxfp8QuantFfnInputSm100,
    Dsv4Fp32SwiGluNvfp4QuantUmmaBSm100,
    Dsv4RmsFp8QuantUmmaBSm100,
    Dsv4Fp32ToBf16,
    Dsv4Nvfp4QuantUmmaBSm100,
    Dsv4PreloadRopeTables,
    Dsv4Rope64,
    Dsv4RmsRope512_64,
    Dsv4SiluClampMul,
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
    LduReloadBarriers,
    Fp8GemvUmmaSplitKSm100,
    Fp8GemvUmmaStreamSm100,
    Fp8GemvUmmaCoupledSm100,
    Fp8UmmaPrepackSm100,
    Mxfp4Mxfp8GemvUmmaK512MetaScaleFp32Sm100,
    Mxfp4Mxfp8GemvUmmaK512TmaScaleFp32Sm100,
    Mxfp4Mxfp8GateUpSiluFixedRingSm100,
    Mxfp4Mxfp8DownFixedRingSm100,
    TmaLoadMxfpCoupledStream,
    Nvfp4GemvSm100,
    Nvfp4GemvUmmaK512Fp32Sm100,
    Nvfp4GemvUmmaPipelineSm100,
    Nvfp4GemvUmmaPipelineFp32Sm100,
    Nvfp4GemvUmmaPipelineFp32Group2Sm100,
    Nvfp4GemvUmmaFp32Sm100,
    Nvfp4GemvUmmaStreamSm100,
    Nvfp4UmmaPrepackSm100,
    ProfileEvent,
    ProfileLayer,
    ProfileStep,
    RawAddress,
    RMS_NORM_F16_SMEM,
    RepeatM,
    ResetIndirectLayer,
    RoutedTmaLoad1D,
    RegLoad,
    RegStore,
    TmaLoadAddressReg1D,
    TmaLoad1D,
    TmaLoadMxfpScaleBase1D,
    TmaLoadMxfpScale1D,
    TmaLoadReg1D,
    TmaTensor,
    select_rms_smem_instruction,
)
from dae.runtime import config, opcode
from dae.deepseek_v4_schedule import DeepSeekV4ShapePolicy
from dae.launcher import Launcher, SMInstructionBuilder
from dae.schedule import (
    Schedule,
    SchedDsv4Fp8QuantUmmaB,
    SchedDsv4Mxfp8QuantFfnInput,
    SchedDsv4HcPreRms,
    SchedDsv4HcPost,
    SchedDsv4PreloadRopeTables,
    SchedDsv4Rope128_64,
    SchedDsv4Rope512_64,
    SchedDsv4RmsFp8QuantUmmaB,
    SchedDsv4RmsRope512_64,
    SchedFp8GemvUmmaStream,
    SchedFp8GemvUmmaSplitK,
    SchedFp8GemvUmmaCoupled,
    SchedMxfp4Mxfp8GemvUmmaK512,
    SchedMxfp4Mxfp8GateUpSiluFixedRing,
    SchedMxfp4Mxfp8DownFixedRing,
    SchedLayeredMxfp4Mxfp8RoutedResidentFfn,
    SchedDsv4ZeroFill,
    SchedDsv4Fp32ToBf16,
    SchedAttentionDecoding,
    SchedDsv4SwiGluShard128,
    SchedNvfp4GemvUmmaSplitK,
    SchedNvfp4GemvUmmaStream,
    SchedRoutedDsv4Fp32SwiGluNvfp4QuantUmmaB,
    SchedRoutedNvfp4ExpertGroupSplitK,
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


def test_m128n8_tma_descriptor_preserves_physical_n_stride(monkeypatch):
    descriptor_args = []

    def build_tma_desc(*args):
        descriptor_args.append(args)
        return torch.empty((128,), dtype=torch.uint8)

    monkeypatch.setattr("dae.runtime.build_tma_desc", build_tma_desc)

    class FakeLauncher:
        @staticmethod
        def new_tma(_descriptor):
            return 7

    hidden = 4096
    record_rows = 5
    arena = torch.empty((36, hidden), dtype=torch.bfloat16)
    output = torch.as_strided(
        arena,
        size=(8, hidden),
        stride=(record_rows * hidden, 1),
    )
    TmaTensor(FakeLauncher(), output).m128n8_output("reduce")

    assert len(descriptor_args) == 1
    assert descriptor_args[0][2] == [
        record_rows * hidden * output.element_size(),
        64 * output.element_size(),
    ]


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


def test_dynamic_repeat_encodes_counter_transform():
    step = MemoryInstruction(
        opcode.OP_ALLOC_TMA_LOAD_1D,
        num_slots=1,
        arg=0,
        size=16,
    )
    repeat = RepeatM.offsetByCounter(
        2,
        step,
        128,
        counter_shift=2,
        counter_mask_bits=1,
    )[0]

    assert repeat.arg & RepeatM.COUNTER_REG_MASK == 2
    assert (
        repeat.arg & RepeatM.COUNTER_SHIFT_MASK
    ) >> RepeatM.COUNTER_SHIFT_SHIFT == 2
    assert (
        repeat.arg & RepeatM.COUNTER_MASK_BITS_MASK
    ) >> RepeatM.COUNTER_MASK_BITS_SHIFT == 1


def test_dynamic_deepseek_attention_and_index_score_encode_position_counter():
    attention = Dsv4AttentionSplit64UmmaSm100(
        48,
        ring_port_mask=3,
        row_start=192,
        position_counter_reg=2,
        attention_kind="csa",
    )
    reducer = Dsv4AttentionSplitReduceFp8Sm100(
        6,
        17,
        1,
        position_counter_reg=2,
        attention_kind="hca",
    )
    score = Dsv4IndexScore(
        240,
        row_start=480,
        position_counter_reg=2,
    )

    assert attention.args[0] == 47 | (192 << 6)
    assert attention.args[1] == 3 | (1 << 2) | (2 << 4) | 0x8000
    assert reducer.args == [6 | 0x8000, 17, 1 | (2 << 2) | (2 << 4)]
    assert score.args == [240, 480, 3]


def test_counter_offset_window_preserves_retained_stream_adjacency():
    linear1 = TmaLoadMxfpCoupledStream(
        0x100000,
        kind=TmaLoadMxfpCoupledStream.LINEAR1,
        stages=2,
        area_slots=21,
        area_id=0,
        mailbox=4,
        port=0,
    )
    down = TmaLoadMxfpCoupledStream(
        0x100000,
        kind=TmaLoadMxfpCoupledStream.DOWN_WEIGHT,
        stages=2,
        area_slots=10,
        area_id=0,
        mailbox=6,
        port=0,
    )
    expanded = RepeatM.offsetWindowByCounters(
        ((0, 112 * 12 * 8), (1, 2 * 112 * 12 * 8)),
        linear1,
        down,
    )

    assert len(expanded) == 5
    assert all(isinstance(inst, RepeatM) for inst in expanded[:3])
    assert [inst.num_slots for inst in expanded[1:3]] == [0x0402, 0x0402]
    assert expanded[-2] is linear1
    assert expanded[-1] is down
    assert not linear1.opcode & 0x8
    assert down.opcode & 0x8

    builder = SMInstructionBuilder(sm_id=0)
    builder.add(expanded)
    builder.rewrite_coupled_stream_local_chains()
    assert builder.minsts[-2].arg & TmaLoadMxfpCoupledStream.LOCAL_CHAIN
    assert not builder.minsts[-1].arg & TmaLoadMxfpCoupledStream.LOCAL_CHAIN


def test_layered_mxfp_resident_rebases_encoded_plan_addresses():
    plans = torch.zeros((2, 112, 12), dtype=torch.int64)
    commands = [
        None,
        MemoryInstruction(opcode.OP_TMA_LOAD_MX_COUPLED_STREAM, 1, 0, 0, address=0x1000),
        MemoryInstruction(opcode.OP_TMA_LOAD_MX_COUPLED_STREAM, 1, 0, 0, address=0x2000),
        MemoryInstruction(opcode.OP_TMA_LOAD_MX_COUPLED_STREAM, 1, 0, 0, address=0x3000),
        MemoryInstruction(opcode.OP_ALLOC_WB_RAW_ADDRESS, 1, 0, 0, address=0x4000),
    ]
    layered = object.__new__(SchedLayeredMxfp4Mxfp8RoutedResidentFfn)
    layered.num_sms = 112
    layered.placed_resident = SimpleNamespace(
        schedule=lambda sm: commands,
    )
    layered.layered_plans = plans
    layered.plan_layer_bytes = plans.stride(0) * plans.element_size()
    layered.counter_strides = ((0, 1),)

    _, weight_window, down_activation, _ = layered.schedule(4)
    expected = plans[0, 4].data_ptr()
    assert cords2addr(weight_window[-2].cords) == expected
    assert cords2addr(weight_window[-1].cords) == expected
    assert cords2addr(down_activation.cords) == expected


def test_counter_offset_helpers_retire_their_allocator_windows():
    single = MemoryInstruction(
        opcode.OP_ALLOC_TMA_LOAD_1D,
        num_slots=1,
        arg=0,
        size=16,
        address=0x100000,
    )
    expanded_single = RepeatM.offsetByCounters(((0, 128),), single)
    assert expanded_single[-1] is single
    assert single.opcode & 0x8

    first = MemoryInstruction(
        opcode.OP_ALLOC_TMA_LOAD_1D,
        num_slots=1,
        arg=0,
        size=16,
        address=0x200000,
    )
    second = MemoryInstruction(
        opcode.OP_ALLOC_TMA_LOAD_1D,
        num_slots=1,
        arg=0,
        size=16,
        address=0x300000,
    )
    expanded_window = RepeatM.byCounter(
        0,
        (first, 128),
        (second, 256),
    )
    assert not first.opcode & 0x8
    assert expanded_window[-1] is second
    assert second.opcode & 0x8


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


def test_nvfp4_streaming_umma_encodes_normal_swiglu_epilogue():
    instruction = Nvfp4GemvUmmaStreamSm100(
        8, bulk_activation=True, swiglu_limit=10.0
    )

    assert instruction.args == [8, 320, 1]


def test_nvfp4_pipeline_encodes_activation_tiles_per_load():
    instruction = Nvfp4GemvUmmaPipelineSm100(
        16, activation_tiles_per_load=4
    )

    assert instruction.args == [16, 0, 4]
    with pytest.raises(ValueError, match="cover the full K operand"):
        Nvfp4GemvUmmaPipelineSm100(
            16,
            retain_activation=True,
            activation_tiles_per_load=4,
        )


def test_nvfp4_fp32_pipeline_encodes_activation_tiles_per_load():
    instruction = Nvfp4GemvUmmaPipelineFp32Sm100(
        8, activation_tiles_per_load=2
    )

    assert instruction.args == [8, 0, 2]


def test_nvfp4_paired_fp32_pipeline_encodes_activation_batch():
    instruction = Nvfp4GemvUmmaPipelineFp32Group2Sm100(
        8, activation_tiles_per_load=4
    )

    assert instruction.opcode == (
        opcode.OP_NVFP4_GEMV_UMMA_PIPELINE_FP32_GROUP2_SM100
    )
    assert instruction.args == [8, 0, 4]


def test_nvfp4_k512_pipeline_packs_activation_retention():
    ordinary = Nvfp4GemvUmmaK512Fp32Sm100(8)
    retained = Nvfp4GemvUmmaK512Fp32Sm100(
        8, retain_activation=True
    )

    assert ordinary.args == [8, 2, 1]
    assert retained.args == [8, 2, 0x101]


def test_dsv4_fp32_swiglu_native_quant_encodes_tile_count_and_bound():
    instruction = Dsv4Fp32SwiGluNvfp4QuantUmmaBSm100(2, 10.0)

    assert instruction.opcode == (
        opcode.OP_DSV4_FP32_SWIGLU_NVFP4_QUANT_UMMA_B_SM100
    )
    assert instruction.args[0] == 2



def test_nvfp4_pipeline_schedule_chunks_activation_by_parameter(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    instructions = SchedNvfp4GemvUmmaStream(
        torch.empty((1, 4, 18432), dtype=torch.uint8),
        torch.empty((4, 3072), dtype=torch.uint8),
        torch.empty((4,), dtype=torch.float32),
        torch.empty((128,), dtype=torch.bfloat16),
        pipeline=True,
        activation_tiles_per_load=2,
    ).place(1).schedule(0)

    assert instructions[0].args == [4, 0, 2]
    assert len(instructions) == 9

def test_routed_nvfp4_fp32_pipeline_keeps_partial_activation_batch(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )

    class FakeDevice:
        type = "cuda"

    class FakeRoutingState:
        device = FakeDevice()

        @staticmethod
        def data_ptr():
            return 0x123456789ABC

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 12

        @staticmethod
        def element_size():
            return 4

    instructions = SchedRoutedNvfp4GemvUmmaStream(
        FakeRoutingState(),
        0,
        (1, 2),
        3,
        torch.empty((4, 3072), dtype=torch.uint8),
        torch.empty((256,), dtype=torch.float32),
        output_mode="fp32_store",
        output_scale=torch.ones((1,), dtype=torch.float32),
        route_ready=True,
        pipeline=True,
        activation_tiles_per_load=2,
    ).place(1).schedule(0)

    compute = [
        inst
        for inst in instructions
        if isinstance(inst, Nvfp4GemvUmmaPipelineFp32Sm100)
    ]
    assert [inst.args for inst in compute] == [[4, 0, 2], [4, 0, 2]]


def test_routed_nvfp4_retains_activation_until_true_last_output(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )

    class FakeDevice:
        type = "cuda"

    class FakeRoutingState:
        device = FakeDevice()

        @staticmethod
        def data_ptr():
            return 0x123456789ABC

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 12

        @staticmethod
        def element_size():
            return 4

    instructions = SchedRoutedNvfp4GemvUmmaStream(
        FakeRoutingState(),
        0,
        (1, 2, 3),
        4,
        torch.empty((4, 3072), dtype=torch.uint8),
        torch.empty((384,), dtype=torch.float32),
        output_mode="fp32_store",
        output_scale=torch.ones((1,), dtype=torch.float32),
        route_ready=True,
    ).place(1).schedule(0)

    compute = [
        inst
        for inst in instructions
        if isinstance(inst, Nvfp4GemvUmmaFp32Sm100)
    ]
    assert [inst.args for inst in compute] == [
        [4, 1, 1],
        [4, 1, 1],
        [4, 0, 1],
    ]


def test_nvfp4_splitk_maps_k_shards_and_tma_reduces_fp32(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def new_tma(self, _desc):
            return 3

    accumulator = torch.empty((1, 256), dtype=torch.float32)
    output_reduce = TmaTensor(FakeLauncher(), accumulator).rowmajor_2d(
        "reduce", 1, 128
    )
    schedule = SchedNvfp4GemvUmmaSplitK(
        torch.empty((2, 4, 18432), dtype=torch.uint8),
        torch.empty((4, 3072), dtype=torch.uint8),
        torch.empty((4,), dtype=torch.float32),
        torch.ones((1,), dtype=torch.float32),
        output_reduce,
        split_k=2,
    ).place(4)

    first = schedule.schedule(0)
    second_split = schedule.schedule(2)
    assert isinstance(first[0], Nvfp4GemvUmmaFp32Sm100)
    assert first[0].args == second_split[0].args == [2, 0, 1]
    assert first[-1].opcode == opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D
    assert first[-1].size == 128 * accumulator.element_size()
    assert second_split[3].cords != first[3].cords


def test_grouped_routed_splitk_interleaves_routes_and_pairs_gate_up(
    monkeypatch,
):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeDevice:
        type = "cuda"

    class FakeRoutingState:
        device = FakeDevice()

        @staticmethod
        def data_ptr():
            return 0x123456789ABC

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 12

        @staticmethod
        def element_size():
            return 4

    class FakeLauncher:
        def new_tma(self, _desc):
            return 3

    gate = torch.empty((2, 256), dtype=torch.float32)
    up = torch.empty_like(gate)
    output_reduces = tuple(
        TmaTensor(FakeLauncher(), output).rowmajor_2d("reduce", 1, 128)
        for output in (gate, up)
    )
    schedule = SchedRoutedNvfp4ExpertGroupSplitK(
        FakeRoutingState(),
        (
            ((10, 11), (12, 13)),
            ((20, 21), (22, 23)),
        ),
        (30, 31),
        torch.empty((2, 4, 3072), dtype=torch.uint8),
        output_reduces,
        torch.ones((1,), dtype=torch.float32),
        split_k=2,
        route_ready=True,
        pipeline=True,
        activation_tiles_per_load=2,
    ).place(4)

    route_zero = schedule.schedule(0)
    route_one = schedule.schedule(1)
    compute = [
        inst
        for inst in route_zero
        if isinstance(inst, Nvfp4GemvUmmaPipelineFp32Group2Sm100)
    ]
    routed_zero = [
        inst
        for inst in route_zero
        if isinstance(inst, RoutedTmaLoad1D)
    ]
    routed_one = [
        inst
        for inst in route_one
        if isinstance(inst, RoutedTmaLoad1D)
    ]
    reductions = [
        inst
        for inst in route_zero
        if isinstance(inst, MemoryInstruction)
        and (inst.opcode & ~0x3F)
        == (opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D & ~0x3F)
    ]
    activation_loads = [
        inst
        for inst in route_zero
        if isinstance(inst, TmaLoad1D)
        and inst.size == 2 * 3072
    ]

    assert len(compute) == 2
    assert all(inst.args == [2, 0, 2] for inst in compute)
    assert {inst.arg & 7 for inst in routed_zero} == {0}
    assert {inst.arg & 7 for inst in routed_one} == {1}
    assert len(reductions) == 4
    assert len(activation_loads) == 2
    assert all(not inst.opcode & 32 for inst in activation_loads)


def test_grouped_fused_swiglu_stores_native_nvfp4_tiles(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )

    class FakeDevice:
        type = "cuda"

    class FakeRoutingState:
        device = FakeDevice()

        @staticmethod
        def data_ptr():
            return 0x123456789ABC

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def numel():
            return 12

        @staticmethod
        def element_size():
            return 4

    gate = torch.empty((2, 256), dtype=torch.float32)
    up = torch.empty_like(gate)
    output = torch.empty((2, 1, 3072), dtype=torch.uint8)
    schedule = SchedRoutedDsv4Fp32SwiGluNvfp4QuantUmmaB(
        FakeRoutingState(),
        42,
        gate,
        up,
        output,
        route_ready=True,
    ).place(2)

    route_zero = schedule.schedule(0)
    route_one = schedule.schedule(1)

    assert isinstance(
        route_zero[0], Dsv4Fp32SwiGluNvfp4QuantUmmaBSm100
    )
    assert route_zero[0].args[0] == 1
    assert route_zero[-1].size == route_one[-1].size == 3072
    assert route_zero[3].arg == (42 << 3)
    assert route_one[3].arg == (42 << 3) | 1


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
    assert instruction.compute_operator_name() == (
        "OP_FP8_GEMV_UMMA_STREAM_SM100__SCALE_PACK_1__OUTPUT_GROUPS_1"
    )

    packed = Fp8GemvUmmaStreamSm100(64, 4)
    assert packed.args == [64]
    assert packed.compute_operator_name() == (
        "OP_FP8_GEMV_UMMA_STREAM_SM100__SCALE_PACK_4__OUTPUT_GROUPS_1"
    )

    grouped = Fp8GemvUmmaStreamSm100(8, 2, 2)
    assert grouped.args == [8]
    assert grouped.compute_operator_name() == (
        "OP_FP8_GEMV_UMMA_STREAM_SM100__SCALE_PACK_2__OUTPUT_GROUPS_2"
    )


def test_fp8_splitk_umma_encodes_local_k_tiles():
    fp32 = Fp8GemvUmmaSplitKSm100(16)
    bf16 = Fp8GemvUmmaSplitKSm100(16, 2)

    assert fp32.args == [16]
    assert fp32.compute_operator_name() == (
        "OP_FP8_GEMV_UMMA_SPLITK_SM100__SCALE_PACK_1__"
        "OUTPUT_GROUPS_1__REDUCTION_BYTES_4"
    )
    assert bf16.args == [16]
    assert bf16.compute_operator_name() == (
        "OP_FP8_GEMV_UMMA_SPLITK_SM100__SCALE_PACK_1__"
        "OUTPUT_GROUPS_1__REDUCTION_BYTES_2"
    )

    grouped = Fp8GemvUmmaSplitKSm100(8, 2, 2, 2)
    assert grouped.compute_operator_name() == (
        "OP_FP8_GEMV_UMMA_SPLITK_SM100__SCALE_PACK_2__"
        "OUTPUT_GROUPS_2__REDUCTION_BYTES_2"
    )


def test_mxfp4_mxfp8_k512_instructions_encode_family_bload():
    tma = Mxfp4Mxfp8GemvUmmaK512TmaScaleFp32Sm100(4)
    metadata_address = 0x1234_5678_9AB0
    metadata = Mxfp4Mxfp8GemvUmmaK512MetaScaleFp32Sm100(
        metadata_address, 2
    )

    assert tma.args == []
    assert metadata.args == [
        metadata_address & 0xFFFF,
        (metadata_address >> 16) & 0xFFFF,
        (metadata_address >> 32) & 0xFFFF,
    ]
    assert tma.compute_operator_name() == (
        "OP_MXFP4_MXFP8_GEMV_UMMA_TMA_SCALE_FP32_SM100__"
        "K_512__BLOAD_4"
    )
    assert metadata.compute_operator_name() == (
        "OP_MXFP4_MXFP8_GEMV_UMMA_META_SCALE_FP32_SM100__"
        "K_512__BLOAD_2"
    )
    with pytest.raises(ValueError, match="1, 2, 4, or 8"):
        Mxfp4Mxfp8GemvUmmaK512TmaScaleFp32Sm100(3)
    with pytest.raises(ValueError, match="fit in 48 bits"):
        Mxfp4Mxfp8GemvUmmaK512MetaScaleFp32Sm100(1 << 48, 2)


def test_mxfp4_mxfp8_gate_up_silu_instructions_encode_selected_families():
    metadata_address = 0x1234_5678_9AB0
    fixed_k128 = Mxfp4Mxfp8GateUpSiluFixedRingSm100(
        metadata_address, tile_k=128
    )
    fixed_k512 = Mxfp4Mxfp8GateUpSiluFixedRingSm100(
        metadata_address, tile_k=512
    )

    encoded_pointer = [
        metadata_address & 0xFFFF,
        (metadata_address >> 16) & 0xFFFF,
        (metadata_address >> 32) & 0xFFFF,
    ]
    assert fixed_k128.args == encoded_pointer
    assert fixed_k512.args == encoded_pointer
    assert fixed_k128.compute_operator_name() == (
        "OP_MXFP4_MXFP8_GATE_UP_SILU_FIXED_RING_SM100__"
        "K_128__STAGES_10"
    )
    assert fixed_k512.compute_operator_name() == (
        "OP_MXFP4_MXFP8_GATE_UP_SILU_FIXED_RING_SM100__"
        "K_512__STAGES_2"
    )

    with pytest.raises(ValueError, match="K128 or K512"):
        Mxfp4Mxfp8GateUpSiluFixedRingSm100(
            metadata_address, tile_k=256
        )
    with pytest.raises(ValueError, match="fit in 48 bits"):
        Mxfp4Mxfp8GateUpSiluFixedRingSm100(1 << 48)


def test_mxfp4_mxfp8_gate_up_fixed_ring_shards_mixed_tasks(monkeypatch):
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def __init__(self):
            self.next_tma = 5

        def new_tma(self, _desc):
            result = self.next_tma
            self.next_tma += 1
            return result

    tasks = 3
    k_tiles = 8
    gate_weight = torch.empty(
        (tasks, k_tiles, 4, 128, 64), dtype=torch.uint8
    )
    up_weight = torch.empty_like(gate_weight)
    gate_scale = torch.empty((tasks, k_tiles, 2048), dtype=torch.uint8)
    up_scale = torch.empty_like(gate_scale)
    activation_data = torch.empty((k_tiles, 4096), dtype=torch.uint8)
    activation_scale = torch.empty((k_tiles, 2048), dtype=torch.uint8)
    output_record = torch.empty((tasks, 1536), dtype=torch.uint8)
    output_data = output_record[:, :1024]
    output_scale = output_record[:, 1024:]
    metadata = torch.empty((tasks, 128), dtype=torch.uint8)
    fake_launcher = FakeLauncher()
    gate_tma = TmaTensor(fake_launcher, gate_weight).mxfp4_load(512)
    up_tma = TmaTensor(fake_launcher, up_weight).mxfp4_load(512)

    schedule = SchedMxfp4Mxfp8GateUpSiluFixedRing(
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        activation_data,
        activation_scale,
        output_data,
        output_scale,
        gate_tma,
        up_tma,
        metadata,
        tile_k=512,
    ).place(2)
    assert schedule._tile_shard(0) == (0, 2)
    assert schedule._tile_shard(1) == (2, 1)

    direct = schedule.schedule(0)
    assert len(direct) == 2
    assert all(
        isinstance(inst, Mxfp4Mxfp8GateUpSiluFixedRingSm100)
        for inst in direct
    )
    assert len(schedule.schedule(1)) == 1

def test_mxfp4_mxfp8_coupled_stream_local_chain():
    plan_address = 0x1234
    linear1 = TmaLoadMxfpCoupledStream(
        plan_address,
        kind=TmaLoadMxfpCoupledStream.LINEAR1,
        stages=2,
        area_slots=21,
        area_id=0,
        mailbox=8,
        port=0,
    )
    down_weight = TmaLoadMxfpCoupledStream(
        plan_address,
        kind=TmaLoadMxfpCoupledStream.DOWN_WEIGHT,
        stages=2,
        area_slots=10,
        area_id=0,
        mailbox=6,
        port=0,
    )
    down_activation = TmaLoadMxfpCoupledStream(
        plan_address,
        kind=TmaLoadMxfpCoupledStream.DOWN_ACTIVATION,
        stages=2,
        area_slots=10,
        area_id=0,
        mailbox=7,
        port=1,
    )

    builder = SMInstructionBuilder(0)
    builder.add([linear1, down_weight, down_activation])
    builder.rewrite_coupled_stream_local_chains()

    rewritten_linear1, rewritten_down_weight, rewritten_activation = (
        builder.minsts
    )
    assert rewritten_linear1.arg & TmaLoadMxfpCoupledStream.LOCAL_CHAIN
    assert not (
        rewritten_down_weight.arg & TmaLoadMxfpCoupledStream.LOCAL_CHAIN
    )
    assert not (
        rewritten_activation.arg & TmaLoadMxfpCoupledStream.LOCAL_CHAIN
    )
    assert rewritten_linear1.annotation["coupled_stream_local_chain"] == "source"
    assert rewritten_down_weight.annotation["fixed_port"] == 0
    assert rewritten_activation.annotation["fixed_port"] == 1

    # SequentialProgram deliberately copies memory instructions into the
    # common base type. Local-chain lowering must therefore depend only on
    # the operator annotations, not a subclass-only helper method.
    copied_builder = SMInstructionBuilder(0)
    copied_builder.add([linear1.copy(), down_weight.copy()])
    copied_builder.rewrite_coupled_stream_local_chains()
    assert (
        copied_builder.minsts[0].arg
        & TmaLoadMxfpCoupledStream.LOCAL_CHAIN
    )
    assert (
        copied_builder.minsts[0].annotation["coupled_stream_local_chain"]
        == "source"
    )


def test_mxfp4_mxfp8_k512_schedule_separates_scale_delivery(monkeypatch):
    tma_load = SchedMxfp4Mxfp8GemvUmmaK512.schedule.__globals__["TmaLoad1D"]
    monkeypatch.setitem(
        tma_load.__init__.__globals__,
        "get_tensor_address",
        lambda tensor: tensor.data_ptr(),
    )
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def new_tma(self, _desc):
            return 7

    weight_data = torch.empty((1, 8, 4, 128, 64), dtype=torch.uint8)
    weight_tma = TmaTensor(FakeLauncher(), weight_data).mxfp4_k512_load()
    weight_scale = torch.empty((1, 8, 2048), dtype=torch.uint8)
    activation_data = torch.empty((8, 4096), dtype=torch.uint8)
    activation_scale = torch.empty((8, 2048), dtype=torch.uint8)
    metadata = torch.empty((1, 128), dtype=torch.uint8)
    output = torch.empty((128,), dtype=torch.float32)

    tma = SchedMxfp4Mxfp8GemvUmmaK512(
        weight_data,
        weight_scale,
        activation_data,
        activation_scale,
        output,
        weight_tma,
        scale_mode="tma",
        activation_tiles_per_load=4,
    ).place(1)
    tma_insts = tma.schedule(0)
    assert isinstance(
        tma_insts[0], Mxfp4Mxfp8GemvUmmaK512TmaScaleFp32Sm100
    )
    assert [inst.size for inst in tma_insts[1:-1]] == (
        [4 * 4096]
        + [0, 0, 32768] * 4
        + [4 * 4096]
        + [0, 0, 32768] * 4
    )
    assert isinstance(tma_insts[2], TmaLoadMxfpScaleBase1D)
    assert isinstance(tma_insts[3], TmaLoadMxfpScaleBase1D)
    assert tma_insts[2].num_slots == config.num_slots + 6
    assert tma_insts[3].num_slots == config.num_slots + 7
    assert isinstance(tma_insts[5], TmaLoadMxfpScale1D)
    assert isinstance(tma_insts[6], TmaLoadMxfpScale1D)
    assert tma_insts[5].num_slots == config.num_slots + 1
    assert tma_insts[6].num_slots == (
        config.num_slots + config.mxfp4_mxfp8_tma_scale_stages + 1
    )
    assert tma_insts[2].annotation["fixed_port"] == 0
    assert tma_insts[3].annotation["fixed_port"] == 1
    assert tma_insts[4].num_slots == 8
    assert len(tma_insts) == 28
    compact_scales = [
        inst for inst in tma_insts if isinstance(inst, TmaLoadMxfpScale1D)
    ]
    assert [inst.num_slots for inst in compact_scales] == [
        config.num_slots
        + operand * config.mxfp4_mxfp8_tma_scale_stages
        + tile % config.mxfp4_mxfp8_tma_scale_stages
        for tile in range(1, 8)
        for operand in (
            TmaLoadMxfpScale1D.WEIGHT,
            TmaLoadMxfpScale1D.ACTIVATION,
        )
    ]
    assert [inst.annotation["fixed_port"] for inst in compact_scales] == [
        port for _tile in range(1, 8) for port in (0, 1)
    ]

    with pytest.raises(ValueError, match="SFA on LDU0"):
        SchedMxfp4Mxfp8GemvUmmaK512(
            weight_data,
            weight_scale,
            activation_data,
            activation_scale,
            output,
            weight_tma,
            scale_mode="tma",
            activation_tiles_per_load=4,
            tma_scale_ports=(1, 1),
        ).place(1)

    direct = SchedMxfp4Mxfp8GemvUmmaK512(
        weight_data,
        weight_scale,
        activation_data,
        activation_scale,
        output,
        weight_tma,
        scale_mode="metadata",
        metadata=metadata,
        activation_tiles_per_load=4,
    ).place(1)
    direct_insts = direct.schedule(0)
    assert isinstance(
        direct_insts[0], Mxfp4Mxfp8GemvUmmaK512MetaScaleFp32Sm100
    )
    assert [inst.size for inst in direct_insts[1:4]] == [
        4 * 4096,
        32768,
        32768,
    ]
    assert len(direct_insts) == 12

    streamed = SchedMxfp4Mxfp8GemvUmmaK512(
        weight_data,
        weight_scale,
        activation_data,
        activation_scale,
        output,
        weight_tma,
        scale_mode="metadata",
        metadata=metadata,
        activation_tiles_per_load=1,
    ).place(1).schedule(0)
    assert [inst.size for inst in streamed[1:-1]] == [
        size for _ in range(8) for size in (4096, 32768)
    ]
    assert len(streamed) == 18

    full = SchedMxfp4Mxfp8GemvUmmaK512(
        weight_data,
        weight_scale,
        activation_data,
        activation_scale,
        output,
        weight_tma,
        scale_mode="metadata",
        metadata=metadata,
        activation_tiles_per_load=8,
    ).place(1).schedule(0)
    assert full[1].size == 8 * 4096
    assert [inst.size for inst in full[2:-1]] == [32768] * 8
    assert len(full) == 11


def test_dsv4_zero_fill_shards_contiguous_output(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    output = torch.empty((1024,), dtype=torch.bfloat16)
    gate = torch.zeros((1,), dtype=torch.uint32)
    schedule = (
        SchedDsv4ZeroFill(gate, output)
        .bar("gate", 8)
        .bar("output", 9)
        .place(8)
    )

    first = schedule.schedule(0)
    last = schedule.schedule(7)

    assert first[0].args == [64, 1]
    assert last[0].args == [64, 1]
    assert first[1].size == last[1].size == 4
    assert first[1].opcode & 16
    assert first[2].size == last[2].size == 256
    assert first[2].cords != last[2].cords
    assert first[2].opcode & 16
    assert schedule.bar_release_count("output") == 8


def test_dsv4_shape_policy_selects_validated_attention_splits():
    policy = DeepSeekV4ShapePolicy(152)

    assert policy.fp8_umma_split_k(1024, 4096) == (8, 64)
    assert policy.fp8_umma_split_k(512, 4096) == (8, 32)
    assert policy.fp8_umma_split_k(32768, 1024) == (1, 128)
    assert policy.fp8_umma_split_k(8192, 1024) == (2, 128)
    assert policy.fp8_umma_split_k(8192, 4096) == (2, 128)
    assert policy.fp8_umma_split_k(4096, 8192) == (4, 128)


def test_dsv4_fp32_to_bf16_finalizer_shards_m128_tiles(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    source = torch.empty((32768,), dtype=torch.float32)
    output = torch.empty((32768,), dtype=torch.bfloat16)
    schedule = SchedDsv4Fp32ToBf16(source, output).place(152)

    first = schedule.schedule(0)
    last = schedule.schedule(151)

    assert isinstance(first[0], Dsv4Fp32ToBf16)
    assert first[0].args == [256]
    assert last[0].args == [128]
    assert first[1].size == 256 * 4
    assert first[2].size == 256 * 2
    assert first[1].cords != last[1].cords


def test_fp8_umma_prepack_encodes_kind_and_k_tile_count():
    instruction = Fp8UmmaPrepackSm100(
        Fp8UmmaPrepackSm100.ACTIVATION, 32
    )

    assert instruction.args == [1, 32]


def test_dsv4_fp8_native_quant_encodes_k_tile_count():
    instruction = Dsv4Fp8QuantUmmaBSm100(1)

    assert instruction.args == [1]
    assert instruction.compute_operator_name() == (
        "OP_DSV4_FP8_QUANT_UMMA_B_SM100__SCALE_PACK_1"
    )


def test_dsv4_mxfp8_ffn_input_quant_encodes_k512_tiles():
    instruction = Dsv4Mxfp8QuantFfnInputSm100(1)

    assert instruction.args == [1]
    assert instruction.compute_operator_name() == (
        "OP_DSV4_MXFP8_QUANT_FFN_INPUT_SM100"
    )


def test_dsv4_cleanroom_preattention_fusions_encode_shape_shards(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    source = torch.empty((1024,), dtype=torch.bfloat16)
    weight = torch.empty_like(source)
    native = torch.empty((8, 2048), dtype=torch.uint8)
    quant = SchedDsv4RmsFp8QuantUmmaB(
        source, weight, native, 1.0e-6
    ).place(4)

    first = quant.schedule(0)
    last = quant.schedule(3)
    assert isinstance(first[0], Dsv4RmsFp8QuantUmmaBSm100)
    assert first[0].args[:2] == [8, 0x200]
    assert last[0].args[:2] == [8, 0x206]
    assert len(first) == 4
    assert first[-1].size == 2 * 2048

    rows = torch.empty((2, 512), dtype=torch.bfloat16)
    table = torch.empty((32, 2), dtype=torch.float32)
    fused = SchedDsv4RmsRope512_64(
        rows,
        table,
        torch.empty_like(rows),
        epsilon=1.0e-6,
        fixed_table_id=1,
    ).place(2)
    q_instructions = fused.schedule(0)
    assert isinstance(q_instructions[0], Dsv4RmsRope512_64)
    assert q_instructions[0].args[:2] == [0, 2]
    assert len(q_instructions) == 3

    weighted = SchedDsv4RmsRope512_64(
        rows[:1],
        table,
        torch.empty_like(rows[:1]),
        epsilon=1.0e-6,
        weight=torch.empty((512,), dtype=torch.bfloat16),
        fixed_table_id=0,
    ).place(1).schedule(0)
    assert weighted[0].args[:2] == [1, 1]
    assert len(weighted) == 4


def test_dsv4_hc_post_encodes_local_width():
    instruction = Dsv4HcPost(128)
    fp32_instruction = Dsv4HcPost(128, branch_fp32=True)

    assert instruction.args == [128, 0]
    assert fp32_instruction.args == [128, 1]


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

    assert [inst.args for inst in compute] == [[128, 0]]
    assert len(memory) == 11
    assert all(inst.num_slots == 1 for inst in memory)
    assignment = DeepSeekV4ShapePolicy(152).hc_post(4096, 4)
    assert assignment.num_sms == 32
    assert assignment.row_alignment == 128

    fp32 = SchedDsv4HcPost(
        torch.empty((4096,), dtype=torch.float32),
        torch.empty((4, 4096), dtype=torch.bfloat16),
        torch.empty((4,), dtype=torch.float32),
        torch.empty((4, 4), dtype=torch.float32),
        torch.empty((4, 4096), dtype=torch.bfloat16),
    ).place(32).schedule(0)
    assert isinstance(fp32[0], Dsv4HcPost)
    assert fp32[0].args == [128, 1]
    assert fp32[1].size == 128 * 4


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


def test_fp8_native_stream_packs_scale_records(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    schedule = SchedFp8GemvUmmaStream(
        torch.empty((1, 8, 16896), dtype=torch.uint8),
        torch.empty((8, 2048), dtype=torch.uint8),
        torch.empty((128,), dtype=torch.bfloat16),
        scale_pack=4,
    ).place(1)

    instructions = schedule.schedule(0)
    compute = next(
        inst for inst in instructions
        if isinstance(inst, Fp8GemvUmmaStreamSm100)
    )
    full_weights = [
        inst for inst in instructions
        if isinstance(inst, MemoryInstruction) and inst.size == 16896
    ]
    data_only_weights = [
        inst for inst in instructions
        if (
            isinstance(inst, MemoryInstruction)
            and inst.size == 16384
            and inst.annotation.get("fixed_port") == 0
        )
    ]

    assert compute.args == [8]
    assert len(full_weights) == 2
    assert all(inst.num_slots == 3 for inst in full_weights)
    assert len(data_only_weights) == 6
    assert all(inst.num_slots == 2 for inst in data_only_weights)


def test_dsv4_fp8_native_quant_shards_whole_scale_groups(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    schedule = SchedDsv4Fp8QuantUmmaB(
        torch.empty((1024,), dtype=torch.bfloat16),
        torch.empty((8, 2048), dtype=torch.uint8),
        scale_pack=4,
    ).place(2)

    first = schedule.schedule(0)
    last = schedule.schedule(1)

    assert first[0].args == last[0].args == [4]
    assert first[1].size == last[1].size == 4 * 128 * 2
    assert first[2].size == last[2].size == 4 * 2048
    assert first[1].cords != last[1].cords


def test_dsv4_mxfp8_ffn_input_quant_uses_one_packed_record_per_sm(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    schedule = SchedDsv4Mxfp8QuantFfnInput(
        torch.empty((4096,), dtype=torch.bfloat16),
        torch.empty((8, 6144), dtype=torch.uint8),
    ).bar("output", 7).place(8)

    first = schedule.schedule(0)
    last = schedule.schedule(7)

    assert isinstance(first[0], Dsv4Mxfp8QuantFfnInputSm100)
    assert first[0].args == last[0].args == [1]
    assert first[1].size == last[1].size == 512 * 2
    assert first[2].size == last[2].size == 6144
    assert first[1].num_slots == last[1].num_slots == 1
    assert first[2].num_slots & 0x3F == last[2].num_slots & 0x3F == 1
    assert first[2].num_slots >> 6 == 7
    assert schedule.bar_release_count("output") == 8
    assert first[1].cords != last[1].cords


def test_dsv4_hc_pre_rms_is_one_cleanroom_fused_task(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    monkeypatch.setattr(
        "dae.schedule.RawAddress",
        lambda tensor, slot_id: MemoryInstruction(
            opcode=opcode.OP_ALLOC_WB_RAW_ADDRESS,
            num_slots=slot_id,
            arg=slot_id,
            size=0,
            address=tensor.data_ptr(),
        ),
    )
    residual = torch.empty((4, 4096), dtype=torch.bfloat16)
    packed_output = torch.empty((4136,), dtype=torch.bfloat16)
    output_metadata = packed_output[4096:].view(torch.float32)
    schedule = SchedDsv4HcPreRms(
        residual,
        torch.empty((24,), dtype=torch.float32),
        torch.empty((3,), dtype=torch.float32),
        torch.empty((24,), dtype=torch.float32),
        torch.empty((4096,), dtype=torch.bfloat16),
        packed_output[:4096],
        output_metadata[:4],
        output_metadata[4:].view(4, 4),
        residual_square_sum=torch.empty((1,), dtype=torch.float32),
        packed_metadata=torch.empty((56,), dtype=torch.float32),
        packed_output=packed_output,
    ).bar("output", 7).place(1)

    instructions = schedule.schedule(0)
    compute = [
        inst for inst in instructions if isinstance(inst, ComputeInstruction)
    ]
    output_releases = [
        inst
        for inst in instructions
        if isinstance(inst, MemoryInstruction)
        and inst.num_slots >> 6 == 7
    ]

    assert [type(inst) for inst in compute] == [Dsv4HcPreRms]
    assert not any(
        isinstance(inst, (RegLoad, RegStore)) for inst in instructions
    )
    assert len(output_releases) == 1
    assert schedule.bar_release_count("output") == 1

    zeroed_residual = torch.empty((4, 4096), dtype=torch.bfloat16)
    zeroed_packed_output = torch.empty((4136,), dtype=torch.bfloat16)
    zeroed_output_metadata = zeroed_packed_output[4096:].view(torch.float32)
    zeroed = SchedDsv4HcPreRms(
        zeroed_residual,
        torch.empty((24,), dtype=torch.float32),
        torch.empty((3,), dtype=torch.float32),
        torch.empty((24,), dtype=torch.float32),
        torch.empty((4096,), dtype=torch.bfloat16),
        zeroed_packed_output[:4096],
        zeroed_output_metadata[:4],
        zeroed_output_metadata[4:].view(4, 4),
        residual_square_sum=torch.empty((1,), dtype=torch.float32),
        packed_metadata=torch.empty((56,), dtype=torch.float32),
        packed_output=zeroed_packed_output,
        zero_fp32_output=torch.empty((4096,), dtype=torch.float32),
    ).place(1).schedule(0)
    assert isinstance(zeroed[0], Dsv4HcPreRms)
    assert zeroed[0].args == [1, 0]
    assert zeroed[-1].size == 4096 * 4

    fp8_packed_output = torch.empty((4176,), dtype=torch.uint8)
    fp8_output = fp8_packed_output[:4096].view(torch.float8_e4m3fn)
    fp8_output_metadata = fp8_packed_output[4096:].view(torch.float32)
    fp8 = SchedDsv4HcPreRms(
        torch.empty((4, 4096), dtype=torch.bfloat16),
        torch.empty((24,), dtype=torch.float32),
        torch.empty((3,), dtype=torch.float32),
        torch.empty((24,), dtype=torch.float32),
        torch.empty((4096,), dtype=torch.bfloat16),
        None,
        fp8_output_metadata[:4],
        fp8_output_metadata[4:].view(4, 4),
        residual_square_sum=torch.empty((1,), dtype=torch.float32),
        packed_metadata=torch.empty((56,), dtype=torch.float32),
        packed_output=fp8_packed_output,
        fp8_output=fp8_output,
        fp8_scale=torch.empty((32,), dtype=torch.float8_e8m0fnu),
    ).bar("output", 9).place(1).schedule(0)
    assert isinstance(fp8[0], Dsv4HcPreRms)
    assert fp8[0].args == [Dsv4HcPreRms.OUTPUT_FP8, 0]
    assert fp8[3].size == 4096
    assert fp8[-1].size == 32
    assert fp8[-1].num_slots >> 6 == 9

    with pytest.raises(ValueError, match="either BF16 or FP8, not both"):
        SchedDsv4HcPreRms(
            residual,
            torch.empty((24,), dtype=torch.float32),
            torch.empty((3,), dtype=torch.float32),
            torch.empty((24,), dtype=torch.float32),
            torch.empty((4096,), dtype=torch.bfloat16),
            packed_output[:4096],
            output_metadata[:4],
            output_metadata[4:].view(4, 4),
            residual_square_sum=torch.empty((1,), dtype=torch.float32),
            packed_metadata=torch.empty((56,), dtype=torch.float32),
            packed_output=packed_output,
            fp8_output=torch.empty((4096,), dtype=torch.float8_e4m3fn),
            fp8_scale=torch.empty((32,), dtype=torch.float8_e8m0fnu),
        ).place(1)


def test_fp8_native_splitk_maps_k_shards_and_tma_reduces(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def new_tma(self, _desc):
            return 3

    weights = torch.empty((8, 32, 16896), dtype=torch.uint8)
    activations = torch.empty((32, 2048), dtype=torch.uint8)
    accumulator = torch.empty((1, 1024), dtype=torch.float32)
    output_reduce = TmaTensor(FakeLauncher(), accumulator).rowmajor_2d(
        "reduce", 1, 128
    )
    schedule = SchedFp8GemvUmmaSplitK(
        weights, activations, output_reduce, split_k=2
    ).place(16)

    first = schedule.schedule(0)
    second_split = schedule.schedule(8)
    first_compute = next(
        inst for inst in first if isinstance(inst, Fp8GemvUmmaSplitKSm100)
    )
    second_compute = next(
        inst
        for inst in second_split
        if isinstance(inst, Fp8GemvUmmaSplitKSm100)
    )
    first_store = first[-1]
    second_activation = next(
        inst
        for inst in second_split
        if isinstance(inst, MemoryInstruction)
        and inst.size == 4 * 2048
    )

    assert first_compute.args == [16]
    assert second_compute.args == [16]
    assert first_store.opcode == opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D
    assert first_store.size == 128 * accumulator.element_size()
    assert first_store.cords[:2] == [0, 0]
    assert second_activation.cords != first[1].cords
    assert schedule.bar_release_count("output") == 0


def test_fp8_native_splitk_balances_more_work_tiles_than_sms(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def new_tma(self, _desc):
            return 3

    weights = torch.empty((256, 8, 16896), dtype=torch.uint8)
    activations = torch.empty((8, 2048), dtype=torch.uint8)
    accumulator = torch.empty((1, 32768), dtype=torch.float32)
    output_reduce = TmaTensor(FakeLauncher(), accumulator).rowmajor_2d(
        "reduce", 1, 128
    )
    schedule = SchedFp8GemvUmmaSplitK(
        weights, activations, output_reduce, split_k=2
    ).bar("output", 7).place(152)

    first = schedule.schedule(0)
    last = schedule.schedule(151)
    first_compute = [
        inst for inst in first if isinstance(inst, Fp8GemvUmmaSplitKSm100)
    ]
    last_compute = [
        inst for inst in last if isinstance(inst, Fp8GemvUmmaSplitKSm100)
    ]
    first_stores = [
        inst
        for inst in first
        if (
            isinstance(inst, MemoryInstruction)
            and (inst.opcode & ~16) == opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D
        )
    ]

    assert len(first_compute) == 4
    assert len(last_compute) == 3
    assert all(inst.args == [4] for inst in first_compute + last_compute)
    assert len(first_stores) == 4
    assert sum(bool(inst.opcode & 16) for inst in first_stores) == 1
    assert first_stores[-1].opcode & 16
    assert schedule.bar_release_count("output") == 152


def test_fp8_coupled_stream_uses_one_allocator_lease_for_both_ldus(
    monkeypatch,
):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    weights = torch.empty((4, 8, 16896), dtype=torch.uint8)
    activations = torch.empty((8, 2048), dtype=torch.uint8)
    output = torch.empty((512,), dtype=torch.bfloat16)
    schedule = SchedFp8GemvUmmaCoupled(
        weights, activations, output
    ).bar("output", 7).place(2)

    instructions = schedule.schedule(0)
    compute = [
        inst for inst in instructions
        if isinstance(inst, Fp8GemvUmmaCoupledSm100)
    ]
    coupled = [
        inst for inst in instructions
        if isinstance(inst, TmaLoadMxfpCoupledStream)
    ]

    assert [inst.args for inst in compute] == [[4, 2, 0]]
    assert len(coupled) == 1
    assert coupled[0].opcode & 1
    assert coupled[0].num_slots == 17
    assert coupled[0].size == 4
    assert coupled[0].annotation["coupled_stream_dual_port"]
    assert schedule.weight_stream.shape == (1, 2, 4, 66560)
    assert not hasattr(schedule, "activation_stream")
    assert schedule.bar_release_count("output") == 2


def test_fp8_coupled_batch_flattens_independent_projection_work(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    weights = torch.empty((2, 2, 2, 16896), dtype=torch.uint8)
    activations = torch.empty((2, 2, 2048), dtype=torch.uint8)
    output = torch.empty((2, 256), dtype=torch.bfloat16)
    schedule = SchedFp8GemvUmmaCoupled(
        weights, activations, output
    ).place(2)

    first = schedule.schedule(0)
    second = schedule.schedule(1)
    first_load = next(
        inst for inst in first if isinstance(inst, TmaLoadMxfpCoupledStream)
    )
    second_load = next(
        inst for inst in second if isinstance(inst, TmaLoadMxfpCoupledStream)
    )

    assert schedule.weight_stream.shape == (2, 1, 1, 66560)
    assert schedule.work_tiles == 2
    first_address = sum(
        value << (16 * index) for index, value in enumerate(first_load.cords)
    )
    second_address = sum(
        value << (16 * index) for index, value in enumerate(second_load.cords)
    )
    assert first_address == schedule.stream_plans[0].data_ptr()
    assert second_address == schedule.stream_plans[1].data_ptr()


def test_fp8_coupled_layer_plans_use_the_resident_layer_index(monkeypatch):
    monkeypatch.setattr(
        "dae.instructions.get_tensor_address", lambda tensor: tensor.data_ptr()
    )
    first_weights = torch.empty((2, 2, 16896), dtype=torch.uint8)
    second_weights = torch.empty((2, 2, 16896), dtype=torch.uint8)
    activations = torch.empty((2, 2048), dtype=torch.uint8)
    output = torch.empty((256,), dtype=torch.bfloat16)
    schedule = SchedFp8GemvUmmaCoupled(
        first_weights,
        activations,
        output,
        weight_layers=(first_weights, second_weights),
    ).place(1)

    load = next(
        inst
        for inst in schedule.schedule(0)
        if isinstance(inst, TmaLoadMxfpCoupledStream)
    )
    address = sum(
        value << (16 * index) for index, value in enumerate(load.cords)
    )

    assert load.size == TmaLoadMxfpCoupledStream.LAYER_INDEXED_SIZE | 1
    assert load.annotation["coupled_stream_layer_indexed"]
    assert schedule.stream_plans.shape == (1, 2, 2)
    assert address == schedule.stream_plans[0, 0].data_ptr()
    assert schedule.stream_plans[0, 0, 0].item() == (
        schedule.weight_stream_layers[0][0, 0, 0].data_ptr()
    )
    assert schedule.stream_plans[0, 1, 0].item() == (
        schedule.weight_stream_layers[1][0, 0, 0].data_ptr()
    )
    assert schedule.stream_plans[0, 0, 1].item() == (
        schedule.stream_plans[0, 1, 1].item()
    )


def test_fp8_coupled_balanced_k_splits_only_the_placement_tail(monkeypatch):
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def new_tma(self, _desc):
            return 3

    weights = torch.empty((10, 8, 16896), dtype=torch.uint8)
    activations = torch.empty((8, 2048), dtype=torch.uint8)
    accumulator = torch.empty((10, 128), dtype=torch.bfloat16)
    output_reduce = TmaTensor(FakeLauncher(), accumulator).rowmajor_2d(
        "reduce", 1, 128
    )
    schedule = SchedFp8GemvUmmaCoupled(
        weights,
        activations,
        output_reduce,
        balanced_k=True,
    ).place(3)

    pair_loads = []
    task_counts = []
    for sm in range(3):
        compute = [
            inst
            for inst in schedule.schedule(sm)
            if isinstance(inst, Fp8GemvUmmaCoupledSm100)
        ]
        pair_loads.append(sum(inst.args[0] for inst in compute))
        task_counts.append(len(compute))

    assert schedule.work_tiles == 7
    assert sum(pair_loads) == 5 * 4
    assert max(pair_loads) == 7
    assert max(task_counts) == 3


def test_fp8_coupled_balanced_k_releases_only_work_owning_sms(monkeypatch):
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def new_tma(self, _desc):
            return 3

    weights = torch.empty((32, 16, 16896), dtype=torch.uint8)
    activations = torch.empty((16, 2048), dtype=torch.uint8)
    accumulator = torch.empty((32, 128), dtype=torch.float32)
    output_reduce = TmaTensor(FakeLauncher(), accumulator).rowmajor_2d(
        "reduce", 1, 128
    )
    schedule = (
        SchedFp8GemvUmmaCoupled(
            weights,
            activations,
            output_reduce,
            balanced_k=True,
        )
        .bar("output", 17)
        .place(56)
    )

    # M4096/K2048 has 128 M256/K256 products. Balancing its 16 output pairs
    # over 56 SMs creates 48 shards; only those work-owning SMs release the
    # dependency.
    assert schedule.work_tiles == 48
    assert schedule.active_sms == 48
    assert schedule.bar_release_count("output") == 48
    assert schedule.collect_barrier_release_counts() == {17: 48}


def test_fp8_coupled_splitk_keeps_one_common_compute_shape(monkeypatch):
    monkeypatch.setattr(
        "dae.runtime.build_tma_desc",
        lambda *args, **kwargs: torch.empty((128,), dtype=torch.uint8),
    )

    class FakeLauncher:
        def new_tma(self, _desc):
            return 3

    weights = torch.empty((8, 32, 16896), dtype=torch.uint8)
    activations = torch.empty((32, 2048), dtype=torch.uint8)
    accumulator = torch.empty((1, 1024), dtype=torch.float32)
    output_reduce = TmaTensor(FakeLauncher(), accumulator).rowmajor_2d(
        "reduce", 1, 128
    )
    schedule = SchedFp8GemvUmmaCoupled(
        weights, activations, output_reduce, split_k=2
    ).place(4)

    instructions = schedule.schedule(0)
    compute = [
        inst for inst in instructions
        if isinstance(inst, Fp8GemvUmmaCoupledSm100)
    ]
    coupled = [
        inst for inst in instructions
        if isinstance(inst, TmaLoadMxfpCoupledStream)
    ]
    stores = [
        inst for inst in instructions
        if isinstance(inst, MemoryInstruction) and inst.opcode & 2
    ]

    assert [inst.args for inst in compute] == [
        [8, 4, 0],
        [8, 4, 8],
    ]
    assert [inst.size for inst in coupled] == [8, 8]
    assert all(inst.num_slots == 17 for inst in coupled)
    assert [inst.arg & 0xFE00 for inst in coupled] == [
        0,
        8 << 9,
    ]
    assert len(stores) == 2
    assert all(
        (inst.opcode & ~16) == opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D
        for inst in stores
    )
    assert all(inst.size == 2 * 128 * 4 for inst in stores)


def test_sequential_program_carries_coupled_fp8_phase_between_stages():
    class FakeLauncher:
        num_sms = 1
        num_bars = 0
        max_insts = 16

    class CoupledStage(Schedule):
        def __init__(self, pair_count):
            super().__init__()
            self.pair_count = pair_count

        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Fp8GemvUmmaCoupledSm100(self.pair_count, 2, 0),
                TmaLoadMxfpCoupledStream(
                    0x1000,
                    kind=TmaLoadMxfpCoupledStream.FP8_GEMV,
                    stages=TmaLoadMxfpCoupledStream.FP8_STAGES,
                    area_slots=TmaLoadMxfpCoupledStream.FP8_AREA_SLOTS,
                    area_id=0,
                    stream_length=self.pair_count,
                ),
            ]

    program = SequentialProgram(
        FakeLauncher(),
        (
            SequentialStage("first", CoupledStage(2), 1),
            SequentialStage(
                "second",
                CoupledStage(4),
                1,
                wait_for_previous=False,
            ),
        ),
    )

    compute = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, Fp8GemvUmmaCoupledSm100)
    ]
    loads = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
        and inst.annotation.get("coupled_stream_kind")
        == TmaLoadMxfpCoupledStream.FP8_GEMV
    ]
    assert [inst.args[2] for inst in compute] == [0, 2]
    assert [
        (inst.arg >> TmaLoadMxfpCoupledStream.PHASE_BASE_SHIFT)
        & TmaLoadMxfpCoupledStream.MAX_PHASE_BASE
        for inst in loads
    ] == [0, 2]
    assert program.coupled_fp8_final_phases == (2,)


def test_dsv4_swiglu_encodes_bound_and_width():
    instruction = Dsv4SiluClampMul(1, 128, 10.0)

    assert instruction.opcode == opcode.OP_DSV4_SILU_CLAMP_MUL
    assert instruction.args[0] == 1
    assert instruction.args[1] == 128


def test_runtime_width_rms_instruction_replaces_small_shape_variants():
    instruction = select_rms_smem_instruction(512)(1, 1.0e-6)

    assert isinstance(instruction, RMS_NORM_F16_SMEM)
    assert instruction.opcode == opcode.OP_RMS_NORM_F16_SMEM
    assert instruction.args[:2] == [1, 512]


def test_fused_hc_head_rms_encodes_both_epsilons():
    instruction = Dsv4HcHeadRms(1.0e-5, 1.0e-6)

    assert instruction.opcode == opcode.OP_DSV4_HC_HEAD_RMS
    assert len(instruction.args) == 2


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

    assert instructions[0].opcode == opcode.OP_DSV4_SILU_CLAMP_MUL
    assert instructions[0].args[1] == 128
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

    fork = SequentialProgram(
        FakeLauncher(),
        (
            SequentialStage("left", BasicStage(), 2),
            SequentialStage(
                "right",
                BasicStage(),
                2,
                base_sm=2,
                wait_for_previous=False,
                parallel_with_previous=True,
            ),
        ),
    )
    assert fork.barriers == []
    assert all(fork.instructions[sm] for sm in range(4))

    with pytest.raises(ValueError, match="disjoint SM placement"):
        SequentialProgram(
            FakeLauncher(),
            (
                SequentialStage("left", BasicStage(), 2),
                SequentialStage(
                    "overlap",
                    BasicStage(),
                    2,
                    base_sm=1,
                    wait_for_previous=False,
                    parallel_with_previous=True,
                ),
            ),
        )


def test_sequential_profile_markers_use_the_compute_stream():
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
    )
    marker = next(
        inst
        for inst in program.instructions[0]
        if isinstance(inst, ProfileLayer)
    )

    assert marker.args == [config.layer_profile_event_base, 6, 1]


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
        if isinstance(inst, ProfileLayer)
    )
    assert marker.args == [config.layer_profile_event_base, 6, 1]
    for sm in range(4):
        memory = [
            inst
            for inst in program.instructions[sm]
            if isinstance(inst, MemoryInstruction)
        ]
        if sm < 2:
            assert memory[1].num_slots >> 6 == 0
        assert memory[-2].num_slots >> 6 == 1


def test_sequential_program_binds_multiple_group_roles_inside_stages():
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

    class RoleProducer(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ).bar(self._bar("output0")),
                MemoryInstruction(
                    opcode.OP_ALLOC_WB_STU_STORE_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ).bar(self._bar("output1")),
            ]

        def bar_release_count(self, role):
            if role not in ("output0", "output1"):
                return 0
            return self._bar_release_if_present(role, self.num_sms)

    class RoleConsumer(Schedule):
        def schedule(self, sm):
            if sm < 0:
                return []
            role = f"input{sm & 1}"
            return [
                Copy(1, 16),
                MemoryInstruction(
                    opcode.OP_ALLOC_LDU_LOAD_1D,
                    num_slots=1,
                    arg=0,
                    size=16,
                    address=0,
                ).bar(self._bar(role)),
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
                "producer",
                RoleProducer(),
                2,
                release_group_roles=(
                    ("ready0", "output0"),
                    ("ready1", "output1"),
                ),
            ),
            SequentialStage(
                "consumer",
                RoleConsumer(),
                4,
                wait_group_roles=(
                    ("ready0", "input0"),
                    ("ready1", "input1"),
                ),
            ),
        ),
    )

    assert launcher.bar_values == {0: 2, 1: 2}
    for sm, instructions in enumerate(program.instructions):
        memory = [
            inst for inst in instructions
            if isinstance(inst, MemoryInstruction)
        ]
        if sm < 2:
            assert memory[0].num_slots >> 6 == 0
            assert memory[1].num_slots >> 6 == 1
            consumer_load = memory[2]
        else:
            consumer_load = memory[0]
        assert consumer_load.num_slots >> 6 == (sm & 1)


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
    compute_loop = next(inst for inst in compute if isinstance(inst, LoopC))
    assert compute_loop.args == [2, 0, 0]
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
    profile_layer = next(inst for inst in compute if isinstance(inst, ProfileLayer))
    assert profile_layer.args == [config.layer_profile_event_base, 6, 2]
    loop = next(inst for inst in memory if isinstance(inst, LoopM))
    assert loop.size == 2
    assert loop.cords[0] == 1
    assert loop.cords[1] == 1
    assert loop.cords[2] == 2 << 6
    assert memory.index(reload) < memory.index(loop)
    shifted_body = [
        inst
        for inst in memory
        if inst.opcode & 0x10 and (inst.num_slots >> 6) < 2
    ]
    assert shifted_body
    assert all(inst.opcode & 0x4 for inst in shifted_body)
    assert all((inst.opcode & ~0x10) != opcode.OP_ISSUE_BARRIER for inst in memory)


def test_looped_program_reload_can_include_shared_task_barriers():
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
        max_insts = 64
        bars_src = FakeBarrierSource()

        def __init__(self):
            self.num_bars = 4
            self.bar_values = {bar: 1 for bar in range(4)}

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
                "shared-bars",
                (SequentialStage("work", BasicStage(), 1),),
                repeat=2,
                reload_barrier_start=0,
                reload_mxfp_resident=True,
            ),
        ),
    )
    reload = next(
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
        and (inst.opcode & ~0x3F)
        == (opcode.OP_LDU_RELOAD_BARRIERS & ~0x3F)
    )
    completion = reload.num_slots >> 6
    assert reload.arg == LduReloadBarriers.RESET_MXFP_RESIDENT
    assert reload.num_slots & 0x3F == config.num_slots + 2
    assert reload.size == completion + 1
    assert completion >= 4

    with pytest.raises(ValueError, match="shifted barrier banks"):
        LoopedSequentialProgram(
            FakeLauncher(),
            (
                SequentialBlock(
                    "bad",
                    (SequentialStage("work", BasicStage(), 1),),
                    repeat=2,
                    barrier_banks=2,
                    reload_barrier_start=0,
                ),
            ),
        )


def test_looped_program_carries_final_completion_directly_into_tail_loads():
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
        num_sms = 2
        max_insts = 64
        bars_src = FakeBarrierSource()

        def __init__(self):
            self.num_bars = 0
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

    program = LoopedSequentialProgram(
        FakeLauncher(),
        (
            SequentialBlock(
                "body",
                (SequentialStage("body", BasicStage(), 2),),
                repeat=3,
                reload_mxfp_resident=True,
                elide_terminal_reload=True,
            ),
            SequentialBlock(
                "tail",
                (SequentialStage("tail", BasicStage(), 2),),
                reload_after=False,
            ),
        ),
    )

    memory = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
    ]
    reload_index = next(
        index
        for index, inst in enumerate(memory)
        if (inst.opcode & ~0x3F)
        == (opcode.OP_LDU_RELOAD_BARRIERS & ~0x3F)
    )
    reload = memory[reload_index]
    completion = reload.num_slots >> 6
    assert reload.arg & LduReloadBarriers.SKIP_FINAL_LOOP
    assert reload.arg & LduReloadBarriers.RESET_MXFP_RESIDENT
    assert (
        reload.arg >> LduReloadBarriers.LOOP_REG_SHIFT
    ) & 0x3 == 0
    assert reload.size == completion + 1

    loop_index = next(
        index
        for index, inst in enumerate(memory)
        if index > reload_index and isinstance(inst, LoopM)
    )
    assert memory[loop_index].size == 3
    tail_loads = [
        inst
        for inst in memory[loop_index + 1 :]
        if inst.opcode & 0x1 and not inst.opcode & 0x2
    ]
    assert {int(bool(inst.opcode & 0x20)) for inst in tail_loads} == {0, 1}
    assert all(inst.opcode & 0x10 for inst in tail_loads)
    assert all(inst.num_slots >> 6 == completion for inst in tail_loads)
    assert all(
        (inst.opcode & ~0x10) != opcode.OP_ISSUE_BARRIER for inst in memory
    )

    # The terminal-loop metadata must not narrow the reload range: the full
    # model restores substantially more than 255 dependency counters.
    wide_reload = LduReloadBarriers(
        FakeBarrierSource(),
        first_bar=256,
        count=512,
        special_slot=2,
        reset_mxfp_resident=True,
        skip_final_loop_reg=3,
    )
    assert wide_reload.arg & LduReloadBarriers.FIRST_BAR_MASK == 256
    assert (
        wide_reload.arg >> LduReloadBarriers.LOOP_REG_SHIFT
    ) & 0x3 == 3
    assert wide_reload.size == 512


def test_sequential_program_resets_resident_state_between_two_ffns():
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
        max_insts = 64
        bars_src = FakeBarrierSource()

        def __init__(self):
            self.num_bars = 0
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
            SequentialStage(
                "linear1",
                BasicStage(),
                1,
                reset_mxfp_resident_after=True,
            ),
            SequentialStage("linear2", BasicStage(), 1),
        ),
    )
    memory = [
        inst
        for inst in program.instructions[0]
        if isinstance(inst, MemoryInstruction)
    ]
    reload_index = next(
        index
        for index, inst in enumerate(memory)
        if (inst.opcode & ~0x3F)
        == (opcode.OP_LDU_RELOAD_BARRIERS & ~0x3F)
    )
    reload = memory[reload_index]
    boundary = reload.num_slots >> 6
    assert reload.num_slots & 0x3F == config.num_slots + 7
    assert reload.arg == boundary | LduReloadBarriers.RESET_MXFP_RESIDENT
    assert reload.size == 1
    assert reload_index == 2
    assert memory[1].num_slots >> 6 == boundary
    assert not memory[3].opcode & 0x10


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


def test_generic_gemm_compute_tasks_cannot_escape_shared_memory():
    root = Path(__file__).resolve().parents[1]
    # DeepSeek's compact metadata tasks intentionally use the explicit raw-
    # address operand contract for tiny packed records.  Keep the generic
    # FP8/NVFP4 GEMM handlers allocator-owned and shared-memory confined.
    sources = [
        root / "include/task/fp8.cuh",
        root / "include/task/nvfp4.cuh",
        root / "include/task/nvfp4_umma.cuh",
    ]
    combined = "\n".join(source.read_text() for source in sources)

    assert "slot_2_glob_ptr" not in combined
    assert "const MInst *st_insts" not in combined
    assert "__threadfence" not in combined


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="resident raw-address metadata requires a CUDA tensor",
)
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
        torch.empty((32, 2), dtype=torch.float32, device="cuda")
        for _ in range(4)
    )
    preload = SchedDsv4PreloadRopeTables(tables).place(2)
    preload_instructions = preload.schedule(0)

    assert isinstance(preload_instructions[0], Dsv4PreloadRopeTables)
    assert preload_instructions[0].args == [4]
    assert len(preload_instructions) == 2
    assert isinstance(preload_instructions[1], RawAddress)
    assert preload_instructions[1].num_slots == config.num_slots
    assert preload.packed_tables.shape == (4, 32, 2)

    input512 = torch.empty((1, 512), dtype=torch.bfloat16)
    output512 = torch.empty_like(input512)
    fixed512 = SchedDsv4Rope512_64(
        input512,
        tables[2],
        output512,
        inverse=True,
        fixed_table_id=2,
    ).place(1).schedule(0)
    assert isinstance(fixed512[0], Dsv4Rope64)
    assert fixed512[0].args == [1, 512, 7]
    assert len(fixed512) == 3

    input128 = torch.empty((1, 128), dtype=torch.bfloat16)
    output128 = torch.empty_like(input128)
    fixed128 = SchedDsv4Rope128_64(
        input128,
        tables[1],
        output128,
        fixed_table_id=1,
    ).place(1).schedule(0)
    assert isinstance(fixed128[0], Dsv4Rope64)
    assert fixed128[0].args == [1, 128, 4]
    assert len(fixed128) == 3

    dynamic = SchedDsv4Rope512_64(
        input512, tables[0], output512
    ).place(1).schedule(0)
    assert dynamic[0].args == [1, 512, 0]
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
    assert instruction.num_slots & 0x3F == 0
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
