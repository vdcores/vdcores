from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dae.compiler import CompileModeError, RepeatRegionIR, compile_builders
from dae.instructions import Gemv_M64N8, LoopC, LoopM, MemoryInstruction, RepeatM, TerminateC, TerminateM
from dae.launcher import SMInstructionBuilder
from dae.runtime import opcode


def _memory_inst(op, *, num_slots=1, arg=0, size=128, cords=None):
    if cords is None:
        cords = [0, 0, 0, 0]
    return MemoryInstruction(opcode=op, num_slots=num_slots, arg=arg, size=size, cords=cords)


class CompilerIRTests(unittest.TestCase):
    def test_compile_ir_roundtrip_collapses_repeat_region(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(Gemv_M64N8(16))

        load_b = _memory_inst(opcode.OP_ALLOC_TMA_LOAD_2D, num_slots=2, arg=7, cords=[0, 8])
        load_a = _memory_inst(opcode.OP_ALLOC_TMA_LOAD_3D, num_slots=4, arg=9, cords=[1, 2, 3]).group()
        store_c = _memory_inst(opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D, num_slots=3, arg=11, cords=[5, 6]).bar(3)

        original_memory = [
            *RepeatM.on(
                4,
                (load_b.copy(), [0, 16]),
                (load_a.copy(), [0, 0, 1]),
            ),
            store_c,
            TerminateM(),
        ]
        for inst in original_memory:
            builder.add_memory(inst)
        builder.add_compute(TerminateC())

        artifacts = compile_builders([builder], mode="compile_ir")
        normalized_memory = artifacts.normalized_program.sms[0].memory_ops
        self.assertIsInstance(normalized_memory[0], RepeatRegionIR)

        emitted_memory = [repr(inst) for inst in artifacts.emitted_memory[0]]
        self.assertEqual(emitted_memory, [repr(inst) for inst in original_memory])
        emitted_compute = [repr(inst) for inst in artifacts.emitted_compute[0]]
        self.assertEqual(emitted_compute, [repr(inst) for inst in builder.cinsts])

    def test_compile_ir_preserves_loop_nodes(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(Gemv_M64N8(8))
        builder.add_compute(LoopC(3, 1))
        builder.add_compute(TerminateC())
        builder.add_memory(LoopM(5, 2, reg=1, bar_shift=3, tma_shift=7))
        builder.add_memory(TerminateM())

        artifacts = compile_builders([builder], mode="compile_ir")
        compute_kinds = [node.kind for node in artifacts.normalized_program.sms[0].compute_ops]
        memory_kinds = [node.kind for node in artifacts.normalized_program.sms[0].memory_ops]
        self.assertIn("loopc", compute_kinds)
        self.assertIn("loopm", memory_kinds)

    def test_compile_mode_fails_loudly_on_unknown_opcode(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(TerminateC())
        builder.add_memory(MemoryInstruction(opcode=0xFFFF, num_slots=0, arg=0, size=0, cords=[0, 0, 0, 0]))

        with self.assertRaises(CompileModeError):
            compile_builders([builder], mode="compile_ir")

    def test_compile_cuda_emits_split_loop_source_for_supported_subset(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(Gemv_M64N8(16))
        builder.add_compute(TerminateC())
        builder.add_memory(
            _memory_inst(opcode.OP_ALLOC_TMA_LOAD_2D, num_slots=2, arg=3, cords=[1, 2]).group().bar(5)
        )
        builder.add_memory(
            _memory_inst(opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D, num_slots=3, arg=4, cords=[5, 6]).bar(7)
        )
        builder.add_memory(TerminateM())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled_gemv.cu"
            artifacts = compile_builders(
                [builder],
                mode="compile_cuda",
                cuda_output_path=output_path,
            )

            self.assertIsNotNone(artifacts.generated_cuda)
            assert artifacts.generated_cuda is not None
            self.assertTrue(artifacts.generated_cuda.path.exists())
            source = artifacts.generated_cuda.source
            self.assertIn("run_alloc_loop", source)
            self.assertIn("run_ldu_loop", source)
            self.assertIn("run_stu_loop", source)
            self.assertIn("run_compute_loop", source)
            self.assertIn("addr_gen=inside_ldu_loop", source)
            self.assertIn("addr_gen=inside_stu_loop", source)

    def test_compile_cuda_rejects_memory_control_ops_for_now(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(Gemv_M64N8(8))
        builder.add_compute(TerminateC())
        builder.add_memory(LoopM(2, 0, reg=0, bar_shift=1, tma_shift=1))
        builder.add_memory(TerminateM())

        with self.assertRaises(CompileModeError):
            compile_builders([builder], mode="compile_cuda")


if __name__ == "__main__":
    unittest.main()
