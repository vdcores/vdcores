from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import pytest
import torch


__test__ = False


@dataclass
class OperatorCase:
    name: str
    dae: object
    reference: Callable[[], torch.Tensor]
    result: Callable[[], torch.Tensor]
    total_bytes: int | None = None
    total_flops: int | None = None


@dataclass
class CodegenResult:
    name: str
    dae: object
    compute_ops: list[str]
    avg_compute_insts: float
    avg_memory_insts: float


def mean_relative_diff(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected_f = expected.float()
    actual_f = actual.float()
    denom = expected_f.abs().mean().clamp_min(1e-7)
    return ((expected_f - actual_f).abs().mean() / denom).item()


def assert_mean_relative_close(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    max_relative_diff: float,
):
    diff = mean_relative_diff(expected, actual)
    assert diff <= max_relative_diff, (
        f"mean relative diff {diff:.6f} exceeded {max_relative_diff:.6f}"
    )


def require_compiled_ops(operator_names: list[str]):
    from dae import runtime

    supported = getattr(runtime, "supported_compute_ops", None)
    if supported is None:
        return

    supported = set(supported)
    missing = [name for name in operator_names if name not in supported]
    if missing:
        pytest.skip(
            "dae.runtime was not built with required compute operators: "
            + ", ".join(missing)
        )


def launch_and_sync(case: OperatorCase):
    case.dae.launch()
    torch.cuda.synchronize()


def benchmark_case(case: OperatorCase, iterations: int | None = None) -> dict[str, float]:
    if iterations is None:
        iterations = int(os.environ.get("DAE_TEST_PERF_ITERS", "1"))
    iterations = max(1, iterations)

    torch.cuda.synchronize()
    case.dae.launch()
    torch.cuda.synchronize()

    execution_times = []
    for _ in range(iterations):
        case.dae.launch()
        torch.cuda.synchronize()
        profile_data = case.dae.profile[:, 0:2].cpu().numpy()
        execution_times.append(float(profile_data[:, 1].max() - profile_data[:, 0].min()))

    mean_ns = sum(execution_times) / len(execution_times)
    stats = {
        "iterations": float(iterations),
        "mean_ns": mean_ns,
        "min_ns": min(execution_times),
        "max_ns": max(execution_times),
    }
    if case.total_bytes is not None and mean_ns > 0:
        stats["bandwidth_gib_s"] = case.total_bytes / (mean_ns / 1e9) / (1024**3)
    if case.total_flops is not None and mean_ns > 0:
        stats["gflops"] = case.total_flops / mean_ns / 1e6
    return stats


class FakeTma:
    def __init__(
        self,
        opcode_name: str,
        *,
        mode: str = "load",
        size: int = 8192,
        num_slots: int = 1,
        arg: int = 0,
    ):
        from dae import runtime

        self.opcode = getattr(runtime.opcode, opcode_name)
        self.mode = mode
        self.size = size
        self.num_slots = num_slots
        self.arg = arg

    def cord(self, *cords):
        from dae.instructions import MemoryInstruction

        return MemoryInstruction(
            opcode=self.opcode,
            num_slots=self.num_slots,
            arg=self.arg,
            size=self.size,
            cords=list(cords),
        )

    def cord2tma(self, *cords):
        return list(cords)

    def group(self, enable=True):
        return self.cord(0).group(enable)


def fake_raw_address(slot_id: int, address: int = 0x1000):
    from dae import runtime
    from dae.instructions import MemoryInstruction

    return MemoryInstruction(
        opcode=runtime.opcode.OP_ALLOC_WB_RAW_ADDRESS,
        num_slots=slot_id,
        arg=slot_id,
        size=0,
        address=address,
    )


def _finish_codegen(name: str, dae) -> CodegenResult:
    compute_ops = dae.compute_operator_names()
    require_compiled_ops(compute_ops)
    dae.build_instructions()
    total_compute = sum(len(builder.built_cinsts) for builder in dae.builder)
    total_memory = sum(len(builder.built_minsts) for builder in dae.builder)
    avg_compute = total_compute / dae.num_sms
    avg_memory = total_memory / dae.num_sms
    assert dae.cinsts.device.type == "cpu"
    assert dae.minsts.device.type == "cpu"
    return CodegenResult(name, dae, compute_ops, avg_compute, avg_memory)


def build_gemv_codegen_case() -> CodegenResult:
    from dae.instructions import Gemv_M64N8
    from dae.launcher import Launcher
    from dae.schedule import SchedGemv

    atom = Gemv_M64N8
    tile_m, tile_n, tile_k = atom.MNK
    dae = Launcher(1, device="cpu")
    sched = SchedGemv(
        atom,
        MNK=(tile_m, tile_n, tile_k * atom.n_batch),
        tmas=(
            FakeTma("OP_ALLOC_TMA_LOAD_2D", mode="load"),
            FakeTma("OP_ALLOC_TMA_LOAD_2D", mode="load"),
            FakeTma("OP_ALLOC_WB_TMA_REDUCE_ADD_2D", mode="reduce"),
        ),
    ).place(1)
    dae.s(sched)
    return _finish_codegen("gemv_out", dae)


def build_rmsnorm_codegen_case() -> CodegenResult:
    from dae.launcher import Launcher
    from dae.schedule import SchedRMSShared

    hidden_size = 4096
    dae = Launcher(2, device="cpu")
    sched = SchedRMSShared(
        num_token=2,
        epsilon=1.0,
        hidden_size=hidden_size,
        tmas=(
            FakeTma("OP_ALLOC_TMA_LOAD_1D", mode="load", size=hidden_size * 2),
            FakeTma("OP_ALLOC_TMA_LOAD_1D", mode="load", size=hidden_size * 2),
            FakeTma("OP_ALLOC_WB_TMA_STORE_1D", mode="store", size=hidden_size * 2),
        ),
    ).place(2)
    dae.s(sched)
    return _finish_codegen("rmsnorm", dae)


def build_silu_codegen_case() -> CodegenResult:
    from dae.instructions import (
        SILU_MUL_SHARED_BF16_K_4096_INTER,
        TerminateC,
        TerminateM,
    )
    from dae.launcher import Launcher

    num_sms = 4
    hidden_size = 4096
    row_bytes = hidden_size * 2
    load_gate = FakeTma("OP_ALLOC_TMA_LOAD_1D", mode="load", size=row_bytes)
    load_up = FakeTma("OP_ALLOC_TMA_LOAD_1D", mode="load", size=row_bytes)
    store = FakeTma("OP_ALLOC_WB_TMA_STORE_1D", mode="store", size=row_bytes)

    def task(sm: int):
        return [
            SILU_MUL_SHARED_BF16_K_4096_INTER(1),
            store.cord(sm * row_bytes),
            load_gate.cord(sm * row_bytes),
            load_up.cord(sm * row_bytes),
        ]

    dae = Launcher(num_sms, device="cpu")
    dae.i(task, TerminateC(), TerminateM())
    return _finish_codegen("silu_mul", dae)


def build_argmax_codegen_case() -> CodegenResult:
    from dae.instructions import (
        ARGMAX_PARTIAL_bf16_1024_65536_128,
        ARGMAX_REDUCE_bf16_1024_128,
        TerminateC,
        TerminateM,
    )
    from dae.launcher import Launcher

    num_sms = 128
    num_token = 8
    logits_slice = 8192 * 8
    num_slice = 2
    sm_per_slice = num_sms // num_slice
    c_per_sm = logits_slice // sm_per_slice

    def task(sm: int):
        slice_idx = sm // sm_per_slice
        slice_offset = (sm % sm_per_slice) * c_per_sm
        insts = [
            ARGMAX_PARTIAL_bf16_1024_65536_128(num_token),
            fake_raw_address(24, 0x1000 + slice_idx * 0x100000 + slice_offset * 2),
            fake_raw_address(25, 0x2000 + sm * 2).writeback(),
            fake_raw_address(26, 0x3000 + sm * 8).writeback(),
        ]
        if sm < num_token:
            insts += [
                ARGMAX_REDUCE_bf16_1024_128(1),
                fake_raw_address(27, 0x4000 + sm * num_sms * 2),
                fake_raw_address(28, 0x5000 + sm * num_sms * 8),
                fake_raw_address(29, 0x6000 + sm * 8).writeback(),
            ]
        return insts

    dae = Launcher(num_sms, device="cpu")
    dae.i(task, TerminateC(), TerminateM())
    return _finish_codegen("argmax", dae)


def build_gemv_out_case(device: torch.device) -> OperatorCase:
    from dae.instructions import Gemv_M64N8
    from dae.launcher import Launcher
    from dae.model import GemvLayer

    torch.manual_seed(0)
    dtype = torch.bfloat16
    atom = Gemv_M64N8
    m, n, k = 4096, atom.MNK[1], 4096
    num_sms = 128

    mat_a = torch.rand(m, k, dtype=dtype, device=device) - 0.5
    mat_b = torch.rand(n, k, dtype=dtype, device=device) - 0.5
    mat_c = torch.zeros(n, m, dtype=dtype, device=device)

    dae = Launcher(num_sms, device=device)
    layer = GemvLayer(dae, atom, "out_proj", (mat_a, mat_b, mat_c))
    dae.s(layer.schedule().place(num_sms))
    require_compiled_ops(dae.compute_operator_names())

    return OperatorCase(
        name="gemv_out",
        dae=dae,
        reference=lambda: mat_a @ mat_b.t(),
        result=lambda: mat_c.t(),
        total_bytes=mat_a.nbytes + mat_b.nbytes + mat_c.nbytes,
        total_flops=2 * m * n * k,
    )


def build_rmsnorm_case(device: torch.device) -> OperatorCase:
    from dae.instructions import TmaLoad1D, TmaStore1D, TmaTensor
    from dae.launcher import Launcher
    from dae.schedule import SchedRMSShared

    torch.manual_seed(1)
    dtype = torch.bfloat16
    num_token = 8
    hidden_size = 4096
    num_sms = 8
    epsilon = 1.0

    dae = Launcher(num_sms, device=device)
    weights = torch.rand((hidden_size,), dtype=dtype, device=device) - 0.5
    mat_in = torch.rand((num_token, hidden_size), dtype=dtype, device=device) - 0.5
    mat_out = torch.zeros_like(mat_in)

    load_weights = TmaTensor(dae, weights).tensor1d("load", hidden_size)
    rms = SchedRMSShared(
        num_token=num_token,
        epsilon=epsilon,
        tmas=(
            load_weights.cord(0),
            TmaLoad1D(mat_in, bytes=hidden_size * 2),
            TmaStore1D(mat_out, bytes=hidden_size * 2),
        ),
    ).place(num_sms)
    dae.s(rms)
    require_compiled_ops(dae.compute_operator_names())

    def reference():
        var = mat_in.pow(2).mean(dim=-1, keepdim=True)
        return mat_in * torch.rsqrt(var + epsilon) * weights

    return OperatorCase(
        name="rmsnorm",
        dae=dae,
        reference=reference,
        result=lambda: mat_out,
        total_bytes=weights.nbytes + mat_in.nbytes + mat_out.nbytes,
    )


def build_silu_case(device: torch.device) -> OperatorCase:
    import torch.nn.functional as F
    from dae.launcher import Launcher
    from dae.schedule import SchedSmemSiLUInterleaved

    torch.manual_seed(2)
    dtype = torch.bfloat16
    num_token = 8
    hidden_size = 4096
    num_sms = 4

    gate = torch.rand(num_token, hidden_size, dtype=dtype, device=device) - 0.5
    up = torch.rand(num_token, hidden_size, dtype=dtype, device=device) - 0.5
    out = torch.zeros(num_token, hidden_size, dtype=dtype, device=device)

    dae = Launcher(num_sms, device=device)
    dae.s(SchedSmemSiLUInterleaved(num_token, gate, up, out).place(num_sms))
    require_compiled_ops(dae.compute_operator_names())

    return OperatorCase(
        name="silu_mul",
        dae=dae,
        reference=lambda: F.silu(gate) * up,
        result=lambda: out,
        total_bytes=gate.nbytes + up.nbytes + out.nbytes,
    )


def build_argmax_case(device: torch.device) -> OperatorCase:
    from dae.instructions import (
        ARGMAX_PARTIAL_bf16_1024_65536_128,
        ARGMAX_REDUCE_bf16_1024_128,
    )
    from dae.launcher import Launcher
    from dae.schedule import SchedArgmax

    torch.manual_seed(2333)
    dtype = torch.bfloat16
    num_token = 8
    num_sms = 128
    logits_fold = 8
    logits_slice = 8192 * logits_fold
    logits_epoch = 2

    dae = Launcher(num_sms, device=device)
    logits = [
        torch.rand(num_token, logits_slice, dtype=dtype, device=device)
        for _ in range(logits_epoch)
    ]
    out_val = torch.zeros(num_token, num_sms, dtype=dtype, device=device)
    out_idx = torch.zeros(num_token, num_sms, dtype=torch.long, device=device)
    final_out = torch.zeros(num_token, dtype=torch.long, device=device)

    argmax = SchedArgmax(
        num_token=num_token,
        logits_slice=logits_slice,
        num_slice=logits_epoch,
        AtomPartial=ARGMAX_PARTIAL_bf16_1024_65536_128,
        AtomReduce=ARGMAX_REDUCE_bf16_1024_128,
        matLogits=logits,
        matOutVal=out_val,
        matOutIdx=out_idx,
        matFinalOut=final_out,
    ).place(num_sms)
    dae.s(argmax)
    require_compiled_ops(dae.compute_operator_names())

    def reference_values():
        mat_in = torch.cat(logits, dim=-1)
        ref_idx = torch.argmax(mat_in, dim=-1)
        return torch.gather(mat_in, 1, ref_idx.unsqueeze(1)).squeeze(1)

    def result_values():
        mat_in = torch.cat(logits, dim=-1)
        return torch.gather(mat_in, 1, final_out.unsqueeze(1)).squeeze(1)

    return OperatorCase(
        name="argmax",
        dae=dae,
        reference=reference_values,
        result=result_values,
        total_bytes=sum(t.nbytes for t in logits) + out_val.nbytes + out_idx.nbytes,
    )
