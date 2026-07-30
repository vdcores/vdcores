from __future__ import annotations

import pytest
import torch

from dae.core import CoreConfig, KernelVariant
from dae.instructions import TerminateC, TerminateComm, TerminateM
from dae.launcher import Launcher
from dae.runtime import config


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _terminated_launcher(num_sms: int = 1, **kwargs) -> Launcher:
    launcher = Launcher(num_sms=num_sms, **kwargs)
    launcher.i(TerminateC(), TerminateM())
    return launcher


def test_fixed_default_compute_memory_core_launches():
    launcher = _terminated_launcher()
    assert launcher._select_kernel_variant(
        [CoreConfig.compute_memory()]
    ) == KernelVariant.COMPUTE_MEMORY
    launcher.launch()


def test_fixed_one_load_core_launches():
    launcher = _terminated_launcher()
    launcher.set_core(0, CoreConfig.compute_memory(load_warps=1))
    assert launcher._select_kernel_variant(
        [CoreConfig.compute_memory(load_warps=1)]
    ) == KernelVariant.COMPUTE_MEMORY_ONE_LOAD
    launcher.launch()


def test_launch_packet_is_reused_and_core_change_invalidates_it():
    launcher = _terminated_launcher()
    launcher.launch()
    packet = launcher._launch_packet
    assert packet is not None

    launcher.launch()
    assert launcher._launch_packet is packet

    launcher.set_core(0, CoreConfig.compute_memory(load_warps=1))
    assert launcher._launch_packet is None
    launcher.launch()
    assert launcher._launch_packet is not packet


def test_runtime_envelope_accepts_different_per_block_load_counts():
    launcher = _terminated_launcher(num_sms=2)
    launcher.set_core(0, CoreConfig.compute_memory(load_warps=1))
    launcher.set_core(1, CoreConfig.compute_memory(load_warps=2))
    launcher.launch()


def test_inactive_runtime_core_launches_without_vm_instructions():
    launcher = Launcher(num_sms=1)
    launcher.set_core(0, CoreConfig.inactive())
    launcher.launch()


@pytest.mark.skipif(
    not bool(config.nvshmem_enabled), reason="NVSHMEM runtime required"
)
def test_nine_warp_communication_assembly_launches():
    launcher = _terminated_launcher()
    launcher.i(TerminateComm())
    launcher.launch()
