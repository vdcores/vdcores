from pathlib import Path

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
    Gemv_M128N8Argmax4,
    Gemv_M128N8Direct4,
    Gemv_M128N8_ROPE_128,
    Gemv_M64N8_ROPE_128,
    Gemv_M64N8UpSiLU,
    LoopC,
    LoopM,
    MemoryInstruction,
    IndirectRoutedTmaLoad1D,
    IndirectTmaLoad1D,
    LduProfileLayer,
    Nvfp4GemvSm100,
    ProfileEvent,
    RepeatM,
    ResetIndirectLayer,
    RoutedTmaLoad1D,
    TmaTensor,
)
from dae.runtime import config, opcode
from dae.deepseek_v4_schedule import DeepSeekV4ShapePolicy
from dae.launcher import Launcher
from dae.schedule import Schedule, SchedAttentionDecoding, SchedSmemSiLUInterleaved
from dae.sequential import (
    LoopedSequentialProgram,
    SequentialBlock,
    SequentialProgram,
    SequentialStage,
)
from dae.tma_utils import (
    cords2addr,
    cord_func_2d_tile_major,
    cord_func_m128n8_grouped_output,
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

    assert direct.opcode == opcode.OP_ALLOC_INDIRECT_TMA_LOAD_1D
    assert layered.opcode == opcode.OP_ALLOC_LAYER_TMA_LOAD_1D
    assert direct.num_slots == 1
    assert cords2addr(direct.cords) == FakePointerTable.data_ptr()
    assert routed.opcode == opcode.OP_ALLOC_LAYER_ROUTED_TMA_LOAD_1D
    assert routed.arg == (511 << 3) | 4
    assert routed.num_slots == 2


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
    assert (fp4.num_sms, fp4.row_alignment, fp4.tile_k) == (152, 8, 256)
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


def test_profile_event_reserves_kernel_start_and_end_slots():
    instruction = ProfileEvent(17)

    assert instruction.opcode == opcode.OP_PROFILE_EVENT
    assert instruction.args == [17]
    with pytest.raises(ValueError, match="slots 0/1"):
        ProfileEvent(1)


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
