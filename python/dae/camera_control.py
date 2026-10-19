"""Camera replay -> branching classical perception -> steering/speed command.

This is a non-learned image-processing workload, not validated robot autonomy.
Generated scenes are fixtures; recorded uint8 RGB NPZ input is supported without
network/model downloads. Python coordinates the graph and remains in timing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .graph_execution import Task, TaskGraph
from .heterogeneous import OperationSpec


def generated_frames(count=24, height=128, width=192, seed=7):
    if count < 1 or height < 16 or width < 16:
        raise ValueError("need positive frame count and dimensions >= 16")
    rng = np.random.default_rng(seed)
    images = rng.integers(20, 45, size=(count, height, width, 3), dtype=np.uint8)
    for frame, image in enumerate(images):
        for y in range(height // 2, height):
            center = width // 2 + int(width * 0.07 * np.sin(frame / 4))
            span = int(width * (0.10 + 0.22 * y / height))
            for x in (center - span, center + span):
                image[y, max(0, x - 2):min(width, x + 3)] = 235
        if frame % 8 >= 4:
            x = width // 2 + int(width * 0.12 * np.cos(frame))
            image[height * 2 // 3:height * 9 // 10,
                  max(0, x - width // 12):min(width, x + width // 12)] = (225, 35, 30)
    return images


def validate_frames(images):
    if (not isinstance(images, np.ndarray) or images.dtype != np.uint8 or images.ndim != 4
            or images.shape[-1] != 3 or images.shape[0] < 1 or min(images.shape[1:3]) < 16):
        raise ValueError("frames must be uint8 [frames, height>=16, width>=16, 3]")
    if images.nbytes > 512 * 1024 * 1024:
        raise ValueError("frame fixture exceeds 512 MiB")
    return np.ascontiguousarray(images)


def load_frames(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        return validate_frames(archive["frames"])


def frame_identity(images, *, source):
    return {"source": source, "shape": list(images.shape), "dtype": str(images.dtype),
            "sha256": hashlib.sha256(images.tobytes()).hexdigest()}


def preprocess(frame, previous, *, device):
    # Bottom-half ROI, normalized RGB, with a bounded previous-steering offset.
    # Never mutate replay storage or the previous output shared with control.
    shift = round(float(previous[0]) * frame.shape[1] * 0.02)
    roi = frame[frame.shape[0] // 2:].permute(2, 0, 1).to(torch.float32).div_(255)
    return torch.roll(roi, shifts=shift, dims=2).contiguous()


def lane_perception(rgb, *, device):
    gray = rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114
    smooth = F.avg_pool2d(gray[None, None], 3, stride=1, padding=1)[0, 0]
    padded = F.pad(smooth, (1, 1, 1, 1))
    gx = (-padded[:-2, :-2] + padded[:-2, 2:] - 2 * padded[1:-1, :-2]
          + 2 * padded[1:-1, 2:] - padded[2:, :-2] + padded[2:, 2:])
    weights = (smooth > 0.65).to(torch.float32)
    columns = weights.sum(dim=0)
    count = columns.sum()
    coordinates = torch.linspace(-1, 1, rgb.shape[2], device=device)
    offset = (columns * coordinates).sum() / count.clamp_min(1)
    confidence = (count / (rgb.shape[1] * rgb.shape[2]) * 12).clamp(0, 1)
    confidence = confidence * (gx.abs().mean() * 20).clamp(0, 1)
    return torch.stack((offset, confidence))


def obstacle_perception(rgb, *, device):
    red = (rgb[0] > 0.6) & (rgb[0] > rgb[1] * 1.5) & (rgb[0] > rgb[2] * 1.5)
    # Spatially smooth the mask to reject isolated red noise before occupancy.
    clean = F.avg_pool2d(red[None, None].to(torch.float32), 3, stride=1, padding=1)[0, 0] > 0.5
    columns = clean.to(torch.float32).sum(dim=0)
    count = columns.sum()
    coordinates = torch.linspace(-1, 1, rgb.shape[2], device=device)
    centroid = (columns * coordinates).sum() / count.clamp_min(1)
    occupancy = count / (rgb.shape[1] * rgb.shape[2])
    return torch.stack((occupancy, centroid))


def control(lane, obstacle, previous, *, device):
    steering = (0.6 * previous[0] - 0.4 * lane[0] - 0.15 * obstacle[1] *
                (obstacle[0] * 10).clamp(0, 1)).clamp(-1, 1)
    target_speed = lane[1] * (1 - obstacle[0] * 12).clamp(0, 1)
    speed = (0.5 * previous[1] + 0.5 * target_speed).clamp(0, 1)
    return torch.stack((steering, speed, obstacle[0]))


def camera_graph(height, width):
    n = (height - height // 2) * width
    return TaskGraph([
        Task(OperationSpec("preprocess", n, devices=("cpu",)), ("frame", "previous"), preprocess),
        Task(OperationSpec("lane", n, ("preprocess",)), ("preprocess",), lane_perception),
        Task(OperationSpec("obstacle", n, ("preprocess",)), ("preprocess",), obstacle_perception),
        Task(OperationSpec("control", 3, ("lane", "obstacle"), ("cpu",)),
             ("lane", "obstacle", "previous"), control),
    ])


def input_factory(images):
    frames = [torch.from_numpy(frame) for frame in images]
    def inputs(lane, frame, previous):
        return {"frame": frames[(frame + lane * 3) % len(frames)], "previous": previous}
    return inputs


def _mean3(image):
    padded = np.pad(image, 1)
    h, w = image.shape
    return sum(padded[y:y+h, x:x+w] for y in range(3) for x in range(3)) / 9


def reference_step(image, previous):
    """Independent NumPy oracle; no calls into the Torch stage functions."""
    shift = round(float(previous[0]) * image.shape[1] * 0.02)
    rgb = np.roll(image[image.shape[0] // 2:].astype(np.float64) / 255, shift, axis=1)
    smooth = _mean3(rgb[:, :, 0] * .299 + rgb[:, :, 1] * .587 + rgb[:, :, 2] * .114)
    weights = smooth > .65
    coordinates = np.linspace(-1, 1, rgb.shape[1])
    lane_offset = (weights.sum(axis=0) * coordinates).sum() / max(1, weights.sum())
    confidence = np.clip(weights.mean() * 12, 0, 1)
    padded = np.pad(smooth, 1)
    gx = (-padded[:-2, :-2] + padded[:-2, 2:] - 2 * padded[1:-1, :-2]
          + 2 * padded[1:-1, 2:] - padded[2:, :-2] + padded[2:, 2:])
    confidence *= np.clip(np.abs(gx).mean() * 20, 0, 1)
    red = (rgb[:, :, 0] > .6) & (rgb[:, :, 0] > 1.5 * rgb[:, :, 1]) & (rgb[:, :, 0] > 1.5 * rgb[:, :, 2])
    clean = _mean3(red.astype(np.float64)) > .5
    occupancy = clean.mean()
    centroid = (clean.sum(axis=0) * coordinates).sum() / max(1, clean.sum())
    steering = np.clip(.6 * previous[0] - .4 * lane_offset - .15 * centroid * np.clip(occupancy * 10, 0, 1), -1, 1)
    speed = np.clip(.5 * previous[1] + .5 * confidence * np.clip(1 - occupancy * 12, 0, 1), 0, 1)
    return np.array([steering, speed, occupancy])


def verify_camera(result, images, *, frames, lanes, tolerance=2e-5):
    if len(result["frames"]) != frames * lanes:
        raise AssertionError("missing camera output")
    records = {(row["lane"], row["frame"]): row for row in result["frames"]}
    if len(records) != frames * lanes:
        raise AssertionError("duplicate camera output")
    max_error = 0.
    for lane in range(lanes):
        previous = np.zeros(3)
        prior_end = 0
        for frame in range(frames):
            row = records[lane, frame]
            expected = reference_step(images[(frame + lane * 3) % len(images)], previous)
            actual = row["output"].tolist() if hasattr(row["output"], "tolist") else row["output"]
            error = float(np.max(np.abs(np.asarray(actual) - expected)))
            if not np.isfinite(error) or error > tolerance:
                raise AssertionError(f"camera output mismatch at lane={lane} frame={frame}: {actual} != {expected}")
            if row["admitted_ns"] < prior_end:
                raise AssertionError("feedback lane reused before previous frame completed")
            prior_end, previous = row["end_ns"], expected
            max_error = max(max_error, error)
    tasks = {(t["lane"], t["frame"], t["task"]): t for t in result["tasks"]}
    graph = camera_graph(*images.shape[1:3])
    if len(tasks) != frames * lanes * len(graph.tasks):
        raise AssertionError("missing/duplicate task trace")
    for (lane, frame, name), task in tasks.items():
        if not task["ready_ns"] <= task["submitted_ns"] <= task["started_ns"] <= task["finished_ns"] <= task["completed_ns"]:
            raise AssertionError("task timestamp order violation")
        if any(tasks[lane, frame, dep]["completed_ns"] > task["submitted_ns"]
               for dep in graph.tasks[name].operation.dependencies):
            raise AssertionError("task ran before a dependency completed")
    return {"passed": True, "checked_frames": frames * lanes, "max_abs_error": max_error,
            "absolute_tolerance": tolerance, "oracle": "independent_numpy_camera_feedback"}
