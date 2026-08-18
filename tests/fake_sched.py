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
import random
import sys
import time

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


# Idealized per-stage work, used only to give the fake a knob-dependent time.
STAGE_WORK_NS = {
    "q_proj": 24_000_000,
    "k_proj": 12_000_000,
    "v_proj": 12_000_000,
    "out_proj": 24_000_000,
    "gate_low": 24_000_000,
    "gate_high": 12_000_000,
    "up_low": 24_000_000,
    "up_high": 12_000_000,
    "down_proj": 36_000_000,
}
FIXED_NS = 900_000


def synthetic_ns(values, rng):
    """A knob-dependent execution time, plus optional host-like noise.

    More SMs on a stage means less time on that stage, with a small
    per-stage launch cost so the ideal is not simply "everything at 128".
    """
    total = FIXED_NS
    for stage, work in STAGE_WORK_NS.items():
        sms = values.get(stage, 1)
        total += work / max(sms, 1) + 2_000 * sms

    noise = float(os.environ.get("FAKE_SCHED_NOISE", "0"))
    if noise > 0:
        total *= 1.0 + rng.gauss(0.0, noise)
        # The real host is bimodal: a minority of fresh processes land in a
        # much slower mode. Reproduce that so the driver is tested against it.
        if rng.random() < float(os.environ.get("FAKE_SCHED_SLOW_PROB", "0.15")):
            total *= 1.7
    return max(total, 1.0)


def run_bench(iterations, values):
    if os.environ.get("FAKE_SCHED_HANG"):
        print(f"[bench] VDCores with {NUM_SMS} SMs...", flush=True)
        while True:  # simulate a barrier deadlock after launch
            time.sleep(1)

    seed = os.environ.get("FAKE_SCHED_SEED")
    rng = random.Random(int(seed) if seed is not None else None)

    print(f"[bench] VDCores with {NUM_SMS} SMs...")
    samples = sorted(synthetic_ns(values, rng) for _ in range(max(iterations, 1)))
    mid = len(samples) // 2
    median = samples[mid] if len(samples) % 2 else (samples[mid - 1] + samples[mid]) / 2
    print(f"Benchmark Results on {NUM_SMS} SMs and {iterations} iterations:")
    print(f"Min execution time (ns): {samples[0]:.2f}")
    print(f"Median execution time (ns): {median:.2f}")
    print(f"Average execution time (ns): {sum(samples) / len(samples):.2f}")
    print(f"Max execution time (ns): {samples[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-build", action="store_true")
    parser.add_argument("-b", "--bench", type=int, nargs="?", const=1, default=None)
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
    if args.bench is not None:
        run_bench(args.bench, {
            "q_proj": q_sms, "k_proj": k_sms, "v_proj": v_sms, "out_proj": out_sms,
            "gate_low": gate_low_sms, "gate_high": gate_high_sms,
            "up_low": up_low_sms, "up_high": up_high_sms, "down_proj": down_sms,
        })
    return 0


if __name__ == "__main__":
    sys.exit(main())
