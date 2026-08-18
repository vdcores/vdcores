#!/usr/bin/env python3
"""A fake tunable schedule, used to test tools/autotune.py without a GPU.

The real targets cannot run on a host without CUDA and PyTorch, so this script
stands in for one. It implements the contract that makes a schedule tunable:

- declare knobs and notes through `dae.tune`
- accept `--dry-build`
- exit non-zero, with the reason on stdout, when a knob combination is illegal

The legality rules mirror `SchedGemv.validate()` in `python/dae/schedule.py`
for the Qwen3 1.7B geometry, including the documented result that `down_proj`
is legal on 96 SMs but not on 128, so a driver tested against this fixture is
exercised against realistic rejections rather than invented ones.
"""

import argparse
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNE_PATH = os.path.join(REPO_ROOT, "python", "dae", "tune.py")

spec = importlib.util.spec_from_file_location("dae_tune_for_fake_sched", TUNE_PATH)
dae_tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dae_tune)

# Qwen3 1.7B geometry.
HIDDEN = 2048
INTERMIDIATE = 6144
QW = 2048
KW = VW = 1024
FULL_SMS = 132
NUM_SMS = 128
RMS_SMS = 8

# Gemv_M64N8: TileM, TileN, TileK = 64, 8, 256 with n_batch = 4.
TILE_M = 64
TILE_K = 256
N_BATCH = 4
MIN_K_PER_FOLD = TILE_K * N_BATCH


def validate_gemv(stage, M, K, num_sms):
    """The subset of SchedGemv.validate() that knob values can violate."""
    m_tiles = M // TILE_M
    assert M % TILE_M == 0, f"{stage}: M={M} is not a multiple of TileM={TILE_M}"
    assert m_tiles > 0, f"{stage}: M={M} is smaller than TileM={TILE_M}"
    assert num_sms % m_tiles == 0, (
        f"{stage}: SMS must be multiple of M tiles when auto folding, "
        f"got SMS={num_sms}, M={M}, TileM={TILE_M}"
    )
    fold = num_sms // m_tiles
    assert K % fold == 0, f"{stage}: K={K} is not divisible by fold={fold}"
    k_per_fold = K // fold
    assert k_per_fold % TILE_K == 0, f"{stage}: Invalid fold for given K size"
    assert k_per_fold % MIN_K_PER_FOLD == 0, (
        f"{stage}: Invalid fold for Gemv_M64N8: k_per_fold={k_per_fold} must be a "
        f"multiple of TileK * n_batch = {TILE_K} * {N_BATCH} = {MIN_K_PER_FOLD}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-build", action="store_true")
    args = parser.parse_args()

    tune = dae_tune.load("fake_sched")
    tune.note("full_sms", FULL_SMS)
    tune.note("num_sms", NUM_SMS)
    tune.note("rms_sms", RMS_SMS)
    tune.note("hidden", HIDDEN)
    tune.note("intermediate", INTERMIDIATE)
    tune.note("gemv_tile_m", TILE_M)

    sms_choices = [16, 32, 48, 64, 80, 96, 112, 128]
    base_choices = [0, 32, 64, 96]

    q_sms = tune.sms("q_proj", 64, sms_choices)
    k_sms = tune.sms("k_proj", 32, sms_choices)
    v_sms = tune.sms("v_proj", 32, sms_choices)
    out_sms = tune.sms("out_proj", 64, sms_choices)
    gate_low_sms = tune.sms("gate_low", 64, sms_choices)
    gate_high_sms = tune.sms("gate_high", 64, sms_choices)
    up_low_sms = tune.sms("up_low", 64, sms_choices)
    up_high_sms = tune.sms("up_high", 64, sms_choices)
    down_sms = tune.sms("down_proj", 96, sms_choices)
    tune.sms("silu", 4, [1, 2, 4, 8])

    tune.base_sm("q_proj", 0, base_choices)
    tune.base_sm("k_proj", 64, base_choices)
    tune.base_sm("v_proj", 96, base_choices)
    tune.base_sm("out_proj", 0, base_choices)
    tune.base_sm("gate_low", 0, base_choices)
    tune.base_sm("gate_high", 0, base_choices)
    tune.base_sm("up_low", 64, base_choices)
    tune.base_sm("up_high", 64, base_choices)
    tune.base_sm("down_proj", 0, base_choices)
    tune.base_sm("silu", 128, [128, 129, 130])

    tune.int_knob("logits.split_m", 6, choices=[2, 3, 4, 6, 8, 12])
    tune.str_set_knob("no_prefetch", (), choices=["q_proj", "logits", "all"])
    mlp_low = tune.int_knob("mlp.low", 4096, choices=[2048, 3072, 4096, 5120])

    mlp_high = INTERMIDIATE - mlp_low
    if mlp_high <= 0:
        raise ValueError(f"Expected intermediate size larger than {mlp_low}, got {INTERMIDIATE}")

    validate_gemv("q_proj", QW, HIDDEN, q_sms)
    validate_gemv("k_proj", KW, HIDDEN, k_sms)
    validate_gemv("v_proj", VW, HIDDEN, v_sms)
    validate_gemv("out_proj", HIDDEN, HIDDEN, out_sms)
    validate_gemv("gate_low", mlp_low, HIDDEN, gate_low_sms)
    validate_gemv("gate_high", mlp_high, HIDDEN, gate_high_sms)
    validate_gemv("up_low", mlp_low, HIDDEN, up_low_sms)
    validate_gemv("up_high", mlp_high, HIDDEN, up_high_sms)
    validate_gemv("down_proj", HIDDEN, INTERMIDIATE, down_sms)

    print(tune.summary())
    if args.dry_build:
        print(f"[dry-build] built fake schedule with mlp_low={mlp_low}, mlp_high={mlp_high}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
