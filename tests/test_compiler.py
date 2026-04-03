from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))

from dae.compiler import CompileModeError, RepeatRegionIR, compile_builders
from dae.instructions import Copy, Dummy, Gemv_M64N8, LoopC, LoopM, MemoryInstruction, RepeatM, TerminateC, TerminateM
from dae.launcher import Launcher, SMInstructionBuilder
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
            self.assertIn("compiled_alloc_sm_0", source)
            self.assertIn("compiled_ldu_sm_0_0", source)
            self.assertIn("compiled_stu_sm_0", source)
            self.assertIn("compiled_compute_sm_0", source)
            self.assertIn("dae_compiled_sm_kernel_0", source)
            self.assertIn("no instruction tables are retained here", source)
            self.assertIn("cudaStreamCreateWithFlags", source)
            self.assertIn("cudaEventRecord(launch_ready, cuda_stream)", source)
            self.assertNotIn("cudaStreamSynchronize(cuda_stream);", source)
            self.assertNotIn("struct AllocOp", source)
            self.assertNotIn("kPrograms", source)
            self.assertIsNotNone(artifacts.generated_runtime)
            assert artifacts.generated_runtime is not None
            self.assertTrue(artifacts.generated_runtime.path.exists())
            self.assertIsNotNone(artifacts.split_unit_program)

    def test_compile_cuda_supports_dummy_copy_split_loops(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(Copy(2, 128))
        builder.add_compute(Dummy(4))
        builder.add_compute(TerminateC())
        repeated_memory = RepeatM.on(
            3,
            (_memory_inst(opcode.OP_ALLOC_TMA_LOAD_1D, num_slots=1, size=128, cords=[16]), 128),
            (_memory_inst(opcode.OP_ALLOC_WB_TMA_STORE_1D, num_slots=1, size=128, cords=[64]).bar(2), 128),
        )
        for inst in repeated_memory:
            builder.add_memory(inst)
        builder.add_memory(TerminateM())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled_copy.cu"
            artifacts = compile_builders([builder], mode="compile_cuda", cuda_output_path=output_path)
            self.assertIsNotNone(artifacts.split_unit_program)
            assert artifacts.split_unit_program is not None
            sm = artifacts.split_unit_program.sms[0]
            self.assertEqual(len(sm.alloc_spans), 1)
            self.assertEqual(sm.alloc_spans[0].trip_count, 3)
            self.assertEqual(len(sm.ldu_ops), 1)
            self.assertEqual(len(sm.stu_ops), 1)
            self.assertEqual(sm.compute_ops[0].opcode_name, "OP_COPY")
            source = artifacts.generated_cuda.source
            self.assertIn("for (int repeat_idx_0 = 0; repeat_idx_0 < 3; ++repeat_idx_0)", source)
            self.assertIn("compiled_alloc_load", source)
            self.assertIn("compiled_alloc_store", source)
            self.assertIn("compiled_load_1d", source)
            self.assertIn("compiled_store_1d", source)
            self.assertEqual(source.count("CompiledLdCmd cmd{};"), 1)
            self.assertNotIn("kAllocSpans", source)
            self.assertNotIn("kLduOps", source)
            self.assertNotIn("kStuOps", source)

    def test_compile_cuda_rejects_memory_control_ops_for_now(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(Gemv_M64N8(8))
        builder.add_compute(TerminateC())
        builder.add_memory(LoopM(2, 0, reg=0, bar_shift=1, tma_shift=1))
        builder.add_memory(TerminateM())

        with self.assertRaises(CompileModeError):
            compile_builders([builder], mode="compile_cuda")

    def test_launcher_compile_ir_matches_interp_encoding(self):
        interp = Launcher(1, device="cpu")
        interp.builder[0].add_compute(Dummy(2))
        interp.builder[0].add_compute(TerminateC())
        interp.builder[0].add_memory(_memory_inst(opcode.OP_ALLOC_TMA_LOAD_1D, num_slots=1, size=128, cords=[32]))
        interp.builder[0].add_memory(TerminateM())
        interp.build_instructions(mode="interp")

        compiled = Launcher(1, device="cpu")
        compiled.builder[0].add_compute(Dummy(2))
        compiled.builder[0].add_compute(TerminateC())
        compiled.builder[0].add_memory(_memory_inst(opcode.OP_ALLOC_TMA_LOAD_1D, num_slots=1, size=128, cords=[32]))
        compiled.builder[0].add_memory(TerminateM())
        compiled.build_instructions(mode="compile_ir")

        self.assertEqual(
            [repr(inst) for inst in interp.builder[0].built_cinsts],
            [repr(inst) for inst in compiled.builder[0].built_cinsts],
        )
        self.assertEqual(
            [repr(inst) for inst in interp.builder[0].built_minsts],
            [repr(inst) for inst in compiled.builder[0].built_minsts],
        )

    def test_compile_cuda_uses_direct_address_expressions(self):
        builder = SMInstructionBuilder(sm_id=0)
        builder.add_compute(Copy(1, 128))
        builder.add_compute(TerminateC())
        for inst in RepeatM.on(
            2,
            (_memory_inst(opcode.OP_ALLOC_TMA_LOAD_1D, num_slots=1, size=128, cords=[64]), 128),
            (_memory_inst(opcode.OP_ALLOC_WB_TMA_STORE_1D, num_slots=1, size=128, cords=[256]), 128),
        ):
            builder.add_memory(inst)
        builder.add_memory(TerminateM())

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "compiled_addr.cu"
            artifacts = compile_builders([builder], mode="compile_cuda", cuda_output_path=output_path)
            source = artifacts.generated_cuda.source
            self.assertIn("compiled_load_1d(cmd, (64ULL + static_cast<uint64_t>(repeat_idx_0) * 128ULL)", source)
            self.assertIn("compiled_store_1d(c2m, (256ULL + static_cast<uint64_t>(repeat_idx_0) * 128ULL)", source)


if __name__ == "__main__":
    unittest.main()
