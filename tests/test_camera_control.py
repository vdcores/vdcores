from pathlib import Path
import sys

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dae.camera_control import (camera_graph, generated_frames, input_factory, load_frames,
                                validate_frames, verify_camera)
from dae.graph_execution import TorchCudaTaskBackend, execute_graph


@pytest.fixture(autouse=True)
def threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("lanes,serial", [(1, False), (4, False), (2, True)])
def test_camera_to_control_feedback_matches_independent_oracle(lanes, serial):
    images = generated_frames(height=64, width=96)
    result = execute_graph(camera_graph(64, 96), input_factory(images), frames=30,
                           lanes=lanes, initial=torch.zeros(3), serial=serial)
    assert verify_camera(result, images, frames=30, lanes=lanes)["passed"]
    outputs = torch.stack([row["output"] for row in result["frames"]])
    assert outputs[:, 1].min() < outputs[:, 1].max()  # speed responds to scene.
    assert outputs[:, 2].max() > 0  # obstacles were actually observed.


def test_recorded_frames_round_trip_and_rejection(tmp_path):
    images = generated_frames(count=3, height=32, width=48)
    path = tmp_path / "recorded.npz"
    np.savez_compressed(path, frames=images)
    assert np.array_equal(load_frames(path), images)
    with pytest.raises(ValueError, match="uint8"):
        validate_frames(images.astype(float))
    with pytest.raises(ValueError, match="uint8"):
        validate_frames(images[0])


def test_oracle_detects_corruption():
    images = generated_frames(height=32, width=48)
    result = execute_graph(camera_graph(32, 48), input_factory(images), frames=2, initial=torch.zeros(3))
    result["frames"][0]["output"][0] += 1
    with pytest.raises(AssertionError, match="mismatch"):
        verify_camera(result, images, frames=2, lanes=1)


@pytest.mark.parametrize("lanes,serial", [(1, False), (4, False), (2, True)])
def test_real_cuda_camera_graph(lanes, serial):
    if not torch.cuda.is_available():
        pytest.skip("real CUDA is unavailable")
    images = generated_frames(height=64, width=96)
    result = execute_graph(camera_graph(64, 96), input_factory(images), frames=24,
                           lanes=lanes, initial=torch.zeros(3), serial=serial,
                           placement="gpu", gpu_backend=TorchCudaTaskBackend(slots=4))
    assert verify_camera(result, images, frames=24, lanes=lanes)["passed"]
    assert all("not_vdcores" in t["executor"] for t in result["tasks"] if t["device"] == "gpu")


def test_cuda_failure_drains_partial_submission():
    if not torch.cuda.is_available():
        pytest.skip("real CUDA is unavailable")
    from dae.graph_execution import Task
    from dae.heterogeneous import OperationSpec
    backend = TorchCudaTaskBackend(slots=1)
    def fail(x, **kwargs):
        x.square_()
        raise RuntimeError("injected CUDA submission failure")
    task = Task(OperationSpec("fail", 10), ("x",), fail)
    with pytest.raises(RuntimeError, match="injected"):
        backend.submit(task, (torch.ones(10),))
    assert backend.inflight
    backend.close()
    assert not backend.inflight
