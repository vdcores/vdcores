"""Launch the split-KV example and compare it with its PyTorch reference."""

from __future__ import annotations

import math
import os
import runpy
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = REPO_ROOT / "app" / "python"
TARGET = APP_DIR / "attention_split_kv.py"


def main() -> None:
    sys.path.insert(0, str(APP_DIR))
    sys.argv = [str(TARGET), "--launch"]
    state = runpy.run_path(str(TARGET), run_name="__main__")

    _, expected = state["gqa_ref"]()
    actual = state["matO_attn_view"]
    expected = expected.reshape(actual.shape)

    mean_error = (actual.float() - expected.float()).abs().mean().item()
    reference_scale = expected.float().abs().mean().item()
    normalized_mean_error = mean_error / max(reference_scale, 1.0e-12)
    limit = float(os.environ.get("ATTENTION_MEAN_ERROR_LIMIT", "0.05"))

    print(f"Split-KV normalized mean error: {normalized_mean_error:.6f}")
    if not math.isfinite(normalized_mean_error) or normalized_mean_error > limit:
        raise AssertionError(
            f"split-KV normalized mean error {normalized_mean_error:.6f} "
            f"exceeds limit {limit:.6f}"
        )


if __name__ == "__main__":
    main()
