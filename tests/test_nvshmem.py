from __future__ import annotations

import torch

import dae.nvshmem as nvshmem


class FakeBackend:
    SIGNAL_SET = 9
    SIGNAL_ADD = 10
    CMP_EQ = 0
    CMP_NE = 1
    CMP_GT = 2
    CMP_LE = 3
    CMP_LT = 4
    CMP_GE = 5

    def __init__(self):
        self.initialized = False
        self.calls = []
        self.signals = torch.zeros(8, dtype=torch.uint64)

    def _info(self):
        return {
            "rank": 0,
            "world_size": 2,
            "local_rank": 0,
            "local_size": 1,
            "device": 0,
            "pe": 0,
            "num_pes": 2,
            "mpi_thread_level": 2,
            "owns_mpi": True,
            "owns_nvshmem": True,
            "nvshmem_name": "fake",
            "nvshmem_version": (3, 4, 5),
            "symmetric_size": "512M",
            "allocation_count": 0,
        }

    def initialize(self, symmetric_size, device):
        self.initialized = True
        self.calls.append(("initialize", symmetric_size, device))
        return self._info()

    def is_initialized(self):
        return self.initialized

    def info(self):
        return self._info()

    def allocate_tensor(self, shape, dtype, zeroed):
        self.calls.append(("allocate_tensor", tuple(shape), dtype, zeroed))
        factory = torch.zeros if zeroed else torch.empty
        return factory(tuple(shape), dtype=dtype)

    def init_signal_space(self, count):
        self.calls.append(("init_signal_space", count))
        return self.signals[:count]

    def get_signal_space(self):
        return self.signals

    def is_symmetric_tensor(self, tensor):
        return tensor.untyped_storage().data_ptr() == self.signals.untyped_storage().data_ptr()

    def signal_on_stream(self, index, value, operation, pe, stream):
        self.calls.append(("signal", index, value, operation, pe, stream))

    def wait_signal_on_stream(self, index, comparison, value, stream):
        self.calls.append(("wait", index, comparison, value, stream))

    def quiet_on_stream(self, stream):
        self.calls.append(("quiet", stream))

    def barrier_all(self):
        self.calls.append(("barrier",))

    def finalize(self):
        self.initialized = False
        self.calls.append(("finalize",))


def install_fake_backend(monkeypatch):
    backend = FakeBackend()
    monkeypatch.setattr(nvshmem, "_backend", backend)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    return backend


def test_init_and_runtime_info(monkeypatch):
    backend = install_fake_backend(monkeypatch)

    info = nvshmem.init(symmetric_size="1G", device="cuda:0")

    assert info.pe == 0
    assert info.num_pes == 2
    assert info.nvshmem_version == (3, 4, 5)
    assert backend.calls == [("initialize", "1G", 0)]


def test_symmetric_tensor_factories(monkeypatch):
    backend = install_fake_backend(monkeypatch)
    nvshmem.init()

    tensor = nvshmem.empty(2, 3, dtype=torch.bfloat16)
    zeros = nvshmem.zeros((4,), dtype=torch.int32)

    assert tensor.shape == (2, 3)
    assert zeros.tolist() == [0, 0, 0, 0]
    assert ("allocate_tensor", (2, 3), torch.bfloat16, False) in backend.calls
    assert ("allocate_tensor", (4,), torch.int32, True) in backend.calls


def test_signal_space_and_stream_operations(monkeypatch):
    backend = install_fake_backend(monkeypatch)
    nvshmem.init()

    signals = nvshmem.init_signal_space(4)
    nvshmem.signal(1, 7, 1, op="add", stream=123)
    nvshmem.wait_signal(1, 7, comparison="ge", stream=123)
    nvshmem.quiet(stream=123)
    nvshmem.barrier()

    assert signals.dtype == torch.uint64
    assert nvshmem.is_symmetric_tensor(signals)
    assert ("signal", 1, 7, backend.SIGNAL_ADD, 1, 123) in backend.calls
    assert ("wait", 1, backend.CMP_GE, 7, 123) in backend.calls
    assert ("quiet", 123) in backend.calls
    assert ("barrier",) in backend.calls


def test_rejects_invalid_shapes_and_sizes(monkeypatch):
    install_fake_backend(monkeypatch)

    for shape in [(-1,), (2, 1.5)]:
        try:
            nvshmem.allocate_tensor(shape)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError(f"shape {shape} should be rejected")

    try:
        nvshmem.init(symmetric_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero symmetric_size should be rejected")
