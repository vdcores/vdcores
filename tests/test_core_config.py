from __future__ import annotations

import pytest
import torch

from dae.core import CoreConfig, CoreKind, KernelVariant, parse_kernel_variant
from dae.instructions import (
    PoolInstruction,
    PoolSliceDynamicReadCopy,
    PoolSliceExchange,
)
from dae.launcher import Launcher, SMInstructionBuilder


def test_core_config_abi_and_variants():
    default = CoreConfig.compute_memory()
    assert default.pack() == bytes([CoreKind.COMPUTE_MEMORY, 4, 2, 0, 0, 0, 0, 0])

    compact = CoreConfig.compute_memory(load_warps=1)
    assert compact.pack() == bytes([CoreKind.COMPUTE_MEMORY, 4, 1, 0, 0, 0, 0, 0])

    pool = CoreConfig.pool()
    assert pool.pack(runtime_core_warps=8) == bytes(
        [CoreKind.POOL, 0, 0, 0, 0, 0, 0, 0]
    )
    assert CoreConfig.inactive().pack() == bytes(
        [CoreKind.INACTIVE, 0, 0, 0, 0, 0, 0, 0]
    )

    assert parse_kernel_variant("default") == KernelVariant.COMPUTE_MEMORY
    assert parse_kernel_variant("one_load") == KernelVariant.COMPUTE_MEMORY_ONE_LOAD
    assert parse_kernel_variant("pool") == KernelVariant.POOL
    with pytest.raises(ValueError, match="unknown VDCores kernel variant"):
        parse_kernel_variant("not-a-core")


def test_core_config_rejects_unsupported_role_shapes():
    with pytest.raises(ValueError, match="one or two load"):
        CoreConfig.compute_memory(load_warps=0)
    with pytest.raises(ValueError, match="four-warp"):
        CoreConfig(CoreKind.COMPUTE_MEMORY, 3, 2, 0).pack()
    with pytest.raises(ValueError, match="complete physical envelope"):
        CoreConfig.pool(pool_warps=7).pack(runtime_core_warps=8)


def test_launcher_infers_mixed_pool_inst_runtime_without_comm_warp():
    launcher = object.__new__(Launcher)
    launcher.num_sms = 2
    launcher.kernel_variant = KernelVariant.AUTO
    launcher.core_configs = [None, None]
    launcher.builder = [SMInstructionBuilder(0), SMInstructionBuilder(1)]
    pool_inst = PoolSliceExchange(
        1,
        write_barrier=0,
        dispatch_barrier_base=1,
        compute_barrier_base=2,
    )
    assert isinstance(pool_inst, PoolInstruction)
    launcher.builder[1].built_poolinsts.append(pool_inst)

    cores = launcher._resolve_core_configs()
    assert cores[0] == CoreConfig.compute_memory()
    assert cores[1] == CoreConfig.pool()
    assert launcher._select_kernel_variant(cores) == KernelVariant.RUNTIME


def test_launcher_selects_pool_inst_execute_warp_from_registry(monkeypatch):
    launcher = object.__new__(Launcher)
    launcher.builder = [SMInstructionBuilder(0)]
    pool_inst = PoolSliceExchange(
        1,
        write_barrier=0,
        dispatch_barrier_base=1,
        compute_barrier_base=2,
    )
    launcher.builder[0].built_poolinsts.append(pool_inst)
    launcher.builder[0].built_poolinsts.append(
        PoolSliceDynamicReadCopy(
            1,
            local_reader=0,
            write_barrier=0,
            dispatch_barrier_base=1,
        )
    )
    monkeypatch.setattr(
        "dae.launcher.runtime.pool_execute_warp_types",
        {pool_inst.opcode: "PoolSliceExchangeExecuteWarp"},
        raising=False,
    )
    assert launcher._resolve_pool_inst_opcode() == pool_inst.opcode


def test_launcher_rejects_multiple_pool_execute_warp_types():
    launcher = object.__new__(Launcher)
    launcher.builder = [SMInstructionBuilder(0), SMInstructionBuilder(1)]
    launcher.builder[0].built_poolinsts.append(PoolInstruction(1))
    launcher.builder[1].built_poolinsts.append(PoolInstruction(2))
    with pytest.raises(ValueError, match="only one PoolInst execute-warp type"):
        launcher._resolve_pool_inst_opcode()


def test_pool_instruction_storage_expands_only_for_pool_programs():
    launcher = object.__new__(Launcher)
    launcher.num_sms = 2
    launcher.max_pool_insts = 141
    launcher.poolinsts = torch.zeros((2, 1, 16), dtype=torch.uint8)
    launcher.builder = [SMInstructionBuilder(0), SMInstructionBuilder(1)]

    launcher._ensure_pool_instruction_storage()
    assert launcher.poolinsts.shape == (2, 1, 16)

    launcher.builder[1].poolinsts.append(
        PoolSliceDynamicReadCopy(
            1,
            local_reader=0,
            write_barrier=0,
            dispatch_barrier_base=1,
        )
    )
    launcher._ensure_pool_instruction_storage()
    assert launcher.poolinsts.shape == (2, 141, 16)


def test_launcher_cache_window_can_be_disabled():
    launcher = object.__new__(Launcher)
    launcher._cache_window_disabled = True
    assert launcher._select_launch_cache_window(
        bars=None,
        tma=None,
        cinsts=None,
        minsts=None,
        comminsts=None,
        poolinsts=None,
    ) is None
