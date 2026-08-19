#!/usr/bin/env python3
"""Tests for tools/autotune.py.

The end-to-end cases drive the real driver against `tests/fake_sched.py`, a
stand-in target that reproduces the Qwen3 1.7B fold rules. That keeps the whole
discover/enumerate/check pipeline testable on a host with no CUDA and no
PyTorch.

Run with:

    python tests/test_autotune.py
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTOTUNE_PATH = os.path.join(REPO_ROOT, "tools", "autotune.py")
FAKE_SCHED = os.path.join("tests", "fake_sched.py")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


autotune = load_module("autotune_under_test", AUTOTUNE_PATH)


SAMPLE = {
    "namespace": "sample",
    "notes": {"full_sms": 132, "intermediate": 6144},
    "knobs": {"q_proj.sms": 64, "q_proj.base_sm": 0, "mlp.low": 4096, "fixed.sms": 8},
    "specs": {
        "q_proj.sms": {"kind": "int", "default": 64, "value": 64, "origin": "default",
                       "choices": [32, 64, 96, 128]},
        "q_proj.base_sm": {"kind": "int", "default": 0, "value": 0, "origin": "default",
                           "choices": [0, 64]},
        "mlp.low": {"kind": "int", "default": 4096, "value": 4096, "origin": "default",
                    "choices": [2048, 4096]},
        "fixed.sms": {"kind": "int", "default": 8, "value": 8, "origin": "default"},
    },
}


def sample_registry():
    return autotune.KnobRegistry(json.loads(json.dumps(SAMPLE)))


def test_registry_reads_specs_notes_and_baseline():
    registry = sample_registry()
    assert registry.namespace == "sample"
    assert registry.notes["full_sms"] == 132
    assert registry.baseline["q_proj.sms"] == 64
    assert registry.choices("q_proj.sms") == [32, 64, 96, 128]
    assert registry.choices("missing") == []


def test_registry_rejects_a_dump_without_specs():
    try:
        autotune.KnobRegistry({"namespace": "sample", "knobs": {}})
    except ValueError:
        return
    raise AssertionError("expected ValueError on a registry with no specs")


def test_sweepable_skips_knobs_without_alternatives():
    registry = sample_registry()
    assert "fixed.sms" not in registry.sweepable()
    assert "q_proj.sms" in registry.sweepable()


def test_stages_need_both_sms_and_base_sm():
    registry = sample_registry()
    assert registry.stages() == ["q_proj"]


def test_sweep_varies_exactly_one_knob_per_candidate():
    registry = sample_registry()
    candidates = autotune.sweep_candidates(registry)

    assert candidates[0].label == "baseline"
    assert candidates[0].overrides == {}
    for candidate in candidates[1:]:
        assert len(candidate.overrides) == 1, candidate.label
        name, value = next(iter(candidate.overrides.items()))
        assert candidate.values[name] == value
        # every other knob stays on the baseline
        others = {k: v for k, v in candidate.values.items() if k != name}
        assert others == {k: v for k, v in registry.baseline.items() if k != name}

    # the baseline value itself is never re-emitted as a variation
    labels = [candidate.label for candidate in candidates]
    assert "q_proj.sms=64" not in labels


def test_sweep_can_be_restricted_to_one_knob():
    registry = sample_registry()
    candidates = autotune.sweep_candidates(registry, only={"mlp.low"})
    assert [c.label for c in candidates] == ["baseline", "mlp.low=2048"]


def test_static_rule_rejects_sm_range_past_the_device():
    registry = sample_registry()
    values = dict(registry.baseline)
    values["q_proj.base_sm"] = 64
    values["q_proj.sms"] = 128
    reason = autotune.static_reject_reason(registry, values)
    assert reason is not None and "exceeds full_sms=132" in reason, reason


def test_static_rule_accepts_a_range_that_fits():
    registry = sample_registry()
    values = dict(registry.baseline)
    values["q_proj.base_sm"] = 64
    values["q_proj.sms"] = 64
    assert autotune.static_reject_reason(registry, values) is None


def test_static_rule_rejects_an_mlp_split_outside_the_layer():
    registry = sample_registry()
    values = dict(registry.baseline)
    values["mlp.low"] = 6144
    reason = autotune.static_reject_reason(registry, values)
    assert reason is not None and "mlp.low" in reason, reason


def test_static_rules_are_skipped_when_notes_are_missing():
    payload = json.loads(json.dumps(SAMPLE))
    payload["notes"] = {}
    registry = autotune.KnobRegistry(payload)
    values = dict(registry.baseline)
    values["q_proj.base_sm"] = 64
    values["q_proj.sms"] = 128
    # Without full_sms the driver must not invent a device size.
    assert autotune.static_reject_reason(registry, values) is None


def test_classify_failure_prefers_the_assertion_line():
    output = "some log\nAssertionError: down_proj: Invalid fold\nTraceback noise\n"
    assert "Invalid fold" in autotune.classify_failure(output)
    assert autotune.classify_failure("") == "no output"
    assert autotune.classify_failure("just a line\n") == "just a line"


# ------------------------------------------------------------- end to end


def discover_fake(tmp):
    path = os.path.join(tmp, "fake.knobs.json")
    rc = autotune.main([
        "discover",
        "--script", FAKE_SCHED,
        "--namespace", "fake_sched",
        "--dry-build-arg=--dry-build",
        "-o", path,
    ])
    assert rc == 0
    return path


def test_discover_writes_a_usable_registry():
    with tempfile.TemporaryDirectory() as tmp:
        registry = autotune.KnobRegistry.from_file(discover_fake(tmp))
        assert registry.namespace == "fake_sched"
        assert len(registry.specs) == 23
        assert registry.notes["full_sms"] == 132
        assert registry.baseline["down_proj.sms"] == 96


def test_check_reproduces_the_documented_down_proj_fold_limit():
    with tempfile.TemporaryDirectory() as tmp:
        knobs = discover_fake(tmp)
        results_path = os.path.join(tmp, "results.json")
        rc = autotune.main([
            "check",
            "--knobs", knobs,
            "--script", FAKE_SCHED,
            "--dry-build-arg=--dry-build",
            "--knob", "down_proj.sms",
            "-o", results_path,
        ])
        assert rc == 0

        with open(results_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        by_label = {result["label"]: result for result in payload["results"]}
        assert by_label["baseline"]["ok"] is True

        # 96 SMs is the documented widest legal down projection; 128 is not.
        assert by_label["down_proj.sms=64"]["ok"] is True
        assert by_label["down_proj.sms=32"]["ok"] is True
        assert by_label["down_proj.sms=128"]["ok"] is False
        assert "k_per_fold=1536" in by_label["down_proj.sms=128"]["reason"]
        assert by_label["down_proj.sms=128"]["stage"] == "build-reject"

        # an SM count that is not a multiple of the M tiles is rejected too
        assert by_label["down_proj.sms=48"]["ok"] is False


def test_static_only_check_needs_no_target_process():
    with tempfile.TemporaryDirectory() as tmp:
        knobs = discover_fake(tmp)
        results_path = os.path.join(tmp, "static.json")
        rc = autotune.main([
            "check",
            "--knobs", knobs,
            "--script", "does/not/exist.py",
            "--static-only",
            "--knob", "v_proj.sms",
            "-o", results_path,
        ])
        assert rc == 0

        with open(results_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        by_label = {result["label"]: result for result in payload["results"]}
        # v_proj sits at base_sm 96, so wide placements run off the device and
        # are caught with no subprocess at all.
        assert by_label["v_proj.sms=48"]["stage"] == "static-reject"
        assert by_label["v_proj.sms=16"]["stage"] == "static-ok"
        assert all(result["seconds"] == 0.0 for result in payload["results"])


def test_check_flags_a_broken_baseline():
    with tempfile.TemporaryDirectory() as tmp:
        knobs_path = discover_fake(tmp)
        with open(knobs_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        # 48 SMs is not a multiple of q_proj's 32 M tiles.
        payload["knobs"]["q_proj.sms"] = 48
        with open(knobs_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

        rc = autotune.main([
            "check",
            "--knobs", knobs_path,
            "--script", FAKE_SCHED,
            "--dry-build-arg=--dry-build",
            "--knob", "logits.split_m",
        ])
        assert rc == 1, "a failing baseline must be reported as a failure"


# --------------------------------------------------------- timing helpers


BENCH_OUTPUT = """
[bench] VDCores with 128 SMs...
Benchmark Results on 128 SMs and 20 iterations:
Min execution time (ns): 2010000.00
Median execution time (ns): 2280000.00
Average execution time (ns): 2350000.50
Max execution time (ns): 3910000.00
"""


def stats_args(**overrides):
    import argparse
    defaults = {"bootstrap": 400, "confidence": 0.95}
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_parse_bench_output_reads_all_four_lines():
    timings = autotune.parse_bench_output(BENCH_OUTPUT)
    assert timings["median"] == 2280000.0
    assert timings["min"] == 2010000.0
    assert timings["max"] == 3910000.0


def test_parse_bench_output_returns_none_without_a_median():
    assert autotune.parse_bench_output("[bench] started\nthen nothing\n") is None


def test_percentile_handles_small_samples():
    assert autotune.percentile([], 0.5) is None
    assert autotune.percentile([7.0], 0.5) == 7.0
    assert autotune.percentile([0.0, 10.0], 0.5) == 5.0
    assert autotune.percentile([0.0, 1.0, 2.0, 3.0], 0.25) == 0.75


def test_summarize_samples_reports_spread():
    stats = autotune.summarize_samples([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["median"] == 2.5
    assert stats["iqr"] == stats["p75"] - stats["p25"]


def test_correction_spreads_the_error_budget():
    assert autotune.corrected_confidence(0.95, 1) == 0.95
    assert abs(autotune.corrected_confidence(0.95, 20) - 0.9975) < 1e-9
    # opting out leaves the level untouched
    assert autotune.corrected_confidence(0.95, 20, enabled=False) == 0.95


def test_decide_calls_a_clear_win_faster():
    import random as _random
    baseline = [5.00e6] * 8
    candidate = [4.50e6] * 8
    verdict, delta, (low, high) = autotune.decide(
        candidate, baseline, min_effect_ns=0.05e6,
        args=stats_args(), rng=_random.Random(0))
    assert verdict == "faster", (verdict, delta, low, high)
    assert delta == -0.5e6
    assert high < 0


def test_decide_calls_an_overlapping_difference_same():
    import random as _random
    # Same distribution, offset by far less than the spread.
    baseline = [5.0e6, 5.2e6, 4.8e6, 5.1e6, 4.9e6, 5.3e6]
    candidate = [5.1e6, 4.9e6, 5.2e6, 4.8e6, 5.0e6, 5.2e6]
    verdict, _, _ = autotune.decide(
        candidate, baseline, min_effect_ns=0.05e6,
        args=stats_args(), rng=_random.Random(0))
    assert verdict == "same", verdict


def test_decide_refuses_to_judge_a_single_sample():
    import random as _random
    verdict, _, interval = autotune.decide(
        [4.0e6], [5.0e6], min_effect_ns=0.0,
        args=stats_args(), rng=_random.Random(0))
    assert verdict == "insufficient"
    assert interval == (None, None)


def test_decide_respects_the_minimum_effect_floor():
    import random as _random
    # A tiny but perfectly consistent win must still be called uninteresting.
    verdict, _, _ = autotune.decide(
        [4.999e6] * 8, [5.0e6] * 8, min_effect_ns=0.05e6,
        args=stats_args(), rng=_random.Random(0))
    assert verdict == "same", verdict


def test_drift_report_measures_first_half_against_second():
    drift = autotune.drift_report([1.0, 1.0, 2.0, 2.0])
    assert drift["first_half_median"] == 1.0
    assert drift["second_half_median"] == 2.0
    assert drift["shift"] == 1.0
    assert autotune.drift_report([1.0, 2.0]) is None


# --------------------------------------------------- timing, end to end


def measure_args(**overrides):
    import argparse
    defaults = {
        "workdir": REPO_ROOT, "script": FAKE_SCHED, "dry_build_arg": ["--dry-build"],
        "repeats": 3, "iterations": 3, "warmup": 0, "seed": 0, "kill_stale": False,
        "post_launch_timeout": 60.0, "idle_timeout": 20.0, "hard_timeout": 120.0,
        "knob": None, "no_prebuild": False, "build_timeout": 60.0, "max": None,
        "min_effect_pct": 1.0, "bootstrap": 400, "confidence": 0.95,
        "no_correction": False, "confirm_rounds": 2, "out": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_measure_finds_a_real_effect_on_a_quiet_host():
    with tempfile.TemporaryDirectory() as tmp:
        knobs = discover_fake(tmp)
        out = os.path.join(tmp, "timing.json")
        args = measure_args(knobs=knobs, knob=["gate_high.sms", "silu.sms"], out=out)
        assert autotune.cmd_measure(args) == 0

        with open(out, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        by_label = {result["label"]: result for result in payload["results"]}
        # A narrower gate_high contends with fewer stages, which the fixture's
        # overlap term charges for, so this is a real -3.9% effect.
        assert by_label["gate_high.sms=32"]["verdict"] == "faster", by_label
        # silu does not appear in that cost model at all, so it must not win
        assert by_label["silu.sms=1"]["verdict"] == "same"
        assert by_label["silu.sms=2"]["verdict"] == "same"
        assert payload["corrected_confidence"] > payload["confidence"]


def test_measure_does_not_crown_a_winner_out_of_pure_noise():
    previous = os.environ.get("FAKE_SCHED_NOISE")
    os.environ["FAKE_SCHED_NOISE"] = "0.05"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            knobs = discover_fake(tmp)
            out = os.path.join(tmp, "noise-timing.json")
            # silu has no effect, so every verdict here should be "same".
            args = measure_args(knobs=knobs, knob=["silu.sms"], repeats=4, out=out)
            assert autotune.cmd_measure(args) == 0

            with open(out, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            for result in payload["results"]:
                assert result["verdict"] in ("same", "unconfirmed"), result
    finally:
        if previous is None:
            os.environ.pop("FAKE_SCHED_NOISE", None)
        else:
            os.environ["FAKE_SCHED_NOISE"] = previous


def test_run_timed_reports_a_hang_rather_than_waiting_forever():
    previous = os.environ.get("FAKE_SCHED_HANG")
    os.environ["FAKE_SCHED_HANG"] = "1"
    try:
        target = {"namespace": "fake_sched", "script": FAKE_SCHED, "dry_build_args": []}
        args = measure_args(knobs=None, post_launch_timeout=4.0, idle_timeout=2.0,
                            hard_timeout=30.0)
        status, ns, reason = autotune.run_timed(target, {}, args, REPO_ROOT)
        assert status == "hang", (status, reason)
        assert ns is None
        assert "deadlock" in reason
    finally:
        if previous is None:
            os.environ.pop("FAKE_SCHED_HANG", None)
        else:
            os.environ["FAKE_SCHED_HANG"] = previous


# ------------------------------------------------- milestone 4: groups


def test_knob_groups_pair_sms_with_base_sm():
    groups = autotune.knob_groups(sample_registry())
    by_name = {group.name: group for group in groups}
    assert by_name["q_proj"].knobs == ["q_proj.sms", "q_proj.base_sm"]
    # a knob with no partner stays on its own
    assert by_name["mlp.low"].knobs == ["mlp.low"]
    # a knob with a single choice is not worth a group
    assert "fixed.sms" not in by_name


def test_group_combinations_are_the_cross_product():
    registry = sample_registry()
    group = next(g for g in autotune.knob_groups(registry) if g.name == "q_proj")
    combos = group.combinations(registry)
    assert len(combos) == 4 * 2 == group.size(registry)
    assert {"q_proj.sms": 128, "q_proj.base_sm": 64} in combos


def test_group_candidates_hold_every_other_knob_at_current():
    registry = sample_registry()
    group = next(g for g in autotune.knob_groups(registry) if g.name == "q_proj")
    current = dict(registry.baseline)
    current["mlp.low"] = 2048

    candidates = autotune.group_candidates(registry, group, current)
    # the cross product minus the combination already in `current`
    assert len(candidates) == 4 * 2 - 1
    for candidate in candidates:
        assert candidate.values["mlp.low"] == 2048
        assert set(candidate.overrides) == {"q_proj.sms", "q_proj.base_sm"}


def test_paired_move_reaches_a_placement_a_single_knob_sweep_cannot():
    """The whole reason milestone 4 exists.

    With `q_proj.base_sm` pinned at 64, a 128-SM q projection runs off the end
    of a 132-SM device. Moving both knobs together reaches it.
    """
    registry = sample_registry()
    registry.baseline["q_proj.base_sm"] = 64

    single = [
        candidate for candidate in autotune.sweep_candidates(registry)
        if candidate.overrides.get("q_proj.sms") == 128
    ]
    assert single, "expected a single-knob candidate to exist"
    assert all(
        autotune.static_reject_reason(registry, candidate.values) is not None
        for candidate in single
    ), "a single-knob sweep should not be able to reach sms=128 from base_sm=64"

    group = next(g for g in autotune.knob_groups(registry) if g.name == "q_proj")
    paired = [
        candidate
        for candidate in autotune.group_candidates(registry, group, dict(registry.baseline))
        if candidate.overrides == {"q_proj.sms": 128, "q_proj.base_sm": 0}
    ]
    assert len(paired) == 1
    assert autotune.static_reject_reason(registry, paired[0].values) is None


# ------------------------------------------------ milestone 4: presets


def test_preset_overrides_the_registry_baseline():
    registry = sample_registry()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preset.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"namespace": "sample", "knobs": {"q_proj.sms": 96}}, handle)
        assert registry.apply_preset(path) == 1
    assert registry.baseline["q_proj.sms"] == 96
    # knobs the preset did not mention keep their dumped values
    assert registry.baseline["mlp.low"] == 4096


def test_preset_rejects_a_foreign_namespace():
    registry = sample_registry()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preset.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"namespace": "someone_else", "knobs": {"q_proj.sms": 96}}, handle)
        try:
            registry.apply_preset(path)
        except SystemExit as exc:
            assert "someone_else" in str(exc)
        else:
            raise AssertionError("expected a foreign namespace to be rejected")


def test_preset_rejects_unknown_knobs():
    registry = sample_registry()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preset.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"knobs": {"not_a_knob": 1}}, handle)
        try:
            registry.apply_preset(path)
        except SystemExit as exc:
            assert "not_a_knob" in str(exc)
        else:
            raise AssertionError("expected an unknown knob to be rejected")


def test_saved_preset_is_accepted_as_a_config_file():
    """The preset a search writes has to be usable as DAE_TUNE_CONFIG."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "preset.json")
        autotune.save_preset(
            path,
            {"namespace": "fake_sched"},
            {"q_proj.sms": 32, "q_proj.base_sm": 0},
            meta={"adopted": []},
        )
        env = dict(os.environ, DAE_TUNE_CONFIG=path)
        env.pop("DAE_TUNE_SET", None)
        proc = subprocess.run(
            [sys.executable, FAKE_SCHED, "--dry-build"],
            cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "q_proj.sms = 32" in proc.stdout


# ---------------------------------------- milestone 4: correctness gate


def test_correctness_gate_passes_a_good_config():
    args = argparse.Namespace(workdir=REPO_ROOT, correctness_timeout=120.0)
    target = {
        "namespace": "fake_sched",
        "script": FAKE_SCHED,
        "correctness_args": ["--correctness"],
    }
    ok, reason = autotune.run_correctness(target, {"q_proj.sms": 64}, args)
    assert ok is True, reason


def test_correctness_gate_catches_a_wrong_config():
    args = argparse.Namespace(workdir=REPO_ROOT, correctness_timeout=120.0)
    target = {
        "namespace": "fake_sched",
        "script": FAKE_SCHED,
        "correctness_args": ["--correctness"],
    }
    previous = os.environ.get("FAKE_SCHED_WRONG_IF")
    os.environ["FAKE_SCHED_WRONG_IF"] = "q_proj.sms=32"
    try:
        ok, reason = autotune.run_correctness(target, {"q_proj.sms": 32}, args)
    finally:
        if previous is None:
            os.environ.pop("FAKE_SCHED_WRONG_IF", None)
        else:
            os.environ["FAKE_SCHED_WRONG_IF"] = previous
    assert ok is False
    assert "tolerance" in reason


def test_correctness_gate_reports_when_a_target_has_no_check():
    args = argparse.Namespace(workdir=REPO_ROOT, correctness_timeout=120.0)
    ok, reason = autotune.run_correctness(
        {"namespace": "fake_sched", "script": FAKE_SCHED}, {}, args)
    assert ok is None
    assert "no correctness check" in reason


# ----------------------------------------------- milestone 4: the search


def search_argv(knobs, extra):
    return [
        "search",
        "--knobs", knobs,
        "--script", FAKE_SCHED,
        "--dry-build-arg=--dry-build",
        "--correctness-arg=--correctness",
        "--min-effect-pct", "0.5",
        *extra,
    ]


def test_search_adopts_a_paired_move_and_confirms_it_end_to_end():
    previous = os.environ.get("FAKE_SCHED_SEED")
    os.environ["FAKE_SCHED_SEED"] = "1"  # quiet host, deterministic timings
    try:
        with tempfile.TemporaryDirectory() as tmp:
            knobs = discover_fake(tmp)
            out = os.path.join(tmp, "search.json")
            preset = os.path.join(tmp, "preset.json")
            rc = autotune.main(search_argv(knobs, [
                "--group", "gate_high",
                "--repeats", "3", "--confirm-rounds", "2", "--final-rounds", "3",
                "--max-passes", "2",
                "--preset-out", preset, "-o", out,
            ]))
            assert rc == 0
            with open(out, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            assert os.path.exists(preset)
    finally:
        if previous is None:
            os.environ.pop("FAKE_SCHED_SEED", None)
        else:
            os.environ["FAKE_SCHED_SEED"] = previous

    # The fixture charges for overlapping SM ranges, so moving gate_high off
    # the crowded low SMs is a real win that needs both knobs to move.
    assert payload["adopted"], "expected the search to adopt at least one step"
    assert "gate_high.base_sm" in payload["changed"]
    assert payload["head_to_head"]["verdict"] == "faster"
    assert payload["head_to_head"]["delta_pct"] < 0


def test_search_refuses_a_fast_but_wrong_configuration():
    """A schedule that is legal, builds, and is fast can still be wrong."""
    saved = {key: os.environ.get(key) for key in ("FAKE_SCHED_SEED", "FAKE_SCHED_WRONG_IF")}
    os.environ["FAKE_SCHED_SEED"] = "1"
    os.environ["FAKE_SCHED_WRONG_IF"] = "gate_high.base_sm=96"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            knobs = discover_fake(tmp)
            out = os.path.join(tmp, "search.json")
            rc = autotune.main(search_argv(knobs, [
                "--group", "gate_high",
                "--repeats", "3", "--confirm-rounds", "2", "--final-rounds", "0",
                "--max-passes", "1", "-o", out,
            ]))
            assert rc == 0
            with open(out, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert payload["adopted"] == [], "a wrong schedule must not be adopted"
    failures = [entry for entry in payload["trace"] if entry.get("correctness_failed")]
    assert failures, "expected the correctness gate to record the rejection"
    assert "tolerance" in failures[0]["correctness_failed"]


def test_search_with_the_gate_disabled_would_have_taken_it():
    """Shows the previous test is really testing the gate, not legality."""
    saved = {key: os.environ.get(key) for key in ("FAKE_SCHED_SEED", "FAKE_SCHED_WRONG_IF")}
    os.environ["FAKE_SCHED_SEED"] = "1"
    os.environ["FAKE_SCHED_WRONG_IF"] = "gate_high.base_sm=96"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            knobs = discover_fake(tmp)
            out = os.path.join(tmp, "search.json")
            rc = autotune.main(search_argv(knobs, [
                "--group", "gate_high", "--no-correctness-gate",
                "--repeats", "3", "--confirm-rounds", "2", "--final-rounds", "0",
                "--max-passes", "1", "-o", out,
            ]))
            assert rc == 0
            with open(out, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert payload["changed"].get("gate_high.base_sm") == 96


def test_search_converges_instead_of_running_every_pass():
    previous = os.environ.get("FAKE_SCHED_SEED")
    os.environ["FAKE_SCHED_SEED"] = "1"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            knobs = discover_fake(tmp)
            out = os.path.join(tmp, "search.json")
            rc = autotune.main(search_argv(knobs, [
                "--group", "v_proj",
                "--repeats", "3", "--confirm-rounds", "2", "--final-rounds", "0",
                "--max-passes", "5", "-o", out,
            ]))
            assert rc == 0
            with open(out, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
    finally:
        if previous is None:
            os.environ.pop("FAKE_SCHED_SEED", None)
        else:
            os.environ["FAKE_SCHED_SEED"] = previous

    # It should stop as soon as a pass adopts nothing, not burn all five.
    assert payload["passes"] < 5


def test_unknown_group_is_rejected():
    registry = sample_registry()
    try:
        autotune.knob_groups(registry, only={"not_a_stage"})
    except SystemExit as exc:
        assert "not_a_stage" in str(exc)
    else:
        raise AssertionError("expected an unknown group name to be rejected")


def test_classify_failure_ignores_cpp_stack_frames():
    """A c10 stack frame contains "Error:" and must not outrank the message.

    Observed for real: a whole sweep of runtime failures reported only
    "frame #0: c10::Error::Error(c10::SourceLocation, ...)", which says
    nothing except that an error happened.
    """
    output = (
        "[bench] VDCores with 132 SMs...\n"
        "RuntimeError: launch_dae failed: misaligned address\n"
        "Exception raised from py_launch_dae at src/torch_runtime.cu:162\n"
        "frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0xe8\n"
        "frame #1: something_else + 0x24\n"
    )
    assert autotune.classify_failure(output) == (
        "RuntimeError: launch_dae failed: misaligned address"
    )


def test_classify_failure_still_finds_a_python_assertion():
    output = (
        "  File \"schedule.py\", line 471, in _on_place\n"
        "AssertionError: SMS must be multiple of M tiles, got SMS=48\n"
    )
    assert "SMS must be multiple" in autotune.classify_failure(output)


def test_measure_rounds_stops_retrying_a_candidate_that_never_runs():
    """A schedule can dry-build and still die on launch every single time."""
    calls = []

    def fake_run_timed(target, values, args, workdir):
        label = values["label"]
        calls.append(label)
        if label == "broken":
            return "fail", None, "launch_dae failed: misaligned address"
        return "ok", 1_000_000.0, None

    candidates = [
        autotune.Candidate({"label": "current"}, {}, "current"),
        autotune.Candidate({"label": "broken"}, {"k": 1}, "broken"),
        autotune.Candidate({"label": "fine"}, {"k": 2}, "fine"),
    ]
    args = argparse.Namespace(repeats=8, seed=0, drop_after=2)

    original = autotune.run_timed
    autotune.run_timed = fake_run_timed
    try:
        samples, failures = autotune.measure_rounds(None, candidates, args, ".")
    finally:
        autotune.run_timed = original

    # tried twice, then dropped, rather than all eight rounds
    assert calls.count("broken") == 2, calls.count("broken")
    assert calls.count("fine") == 8
    assert calls.count("current") == 8
    assert samples["broken"] == []
    assert "broken" in failures


def test_measure_rounds_keeps_a_candidate_that_recovers():
    """One transient failure should not disqualify a working schedule."""
    state = {"n": 0}

    def fake_run_timed(target, values, args, workdir):
        if values["label"] == "flaky":
            state["n"] += 1
            if state["n"] == 1:
                return "fail", None, "transient"
        return "ok", 1_000_000.0, None

    candidates = [
        autotune.Candidate({"label": "current"}, {}, "current"),
        autotune.Candidate({"label": "flaky"}, {"k": 1}, "flaky"),
    ]
    args = argparse.Namespace(repeats=6, seed=0, drop_after=2)

    original = autotune.run_timed
    autotune.run_timed = fake_run_timed
    try:
        samples, _ = autotune.measure_rounds(None, candidates, args, ".")
    finally:
        autotune.run_timed = original

    assert len(samples["flaky"]) == 5, samples["flaky"]


def test_measure_rounds_never_drops_the_reference():
    """Dropping the reference would leave nothing to compare against."""
    def fake_run_timed(target, values, args, workdir):
        return "fail", None, "everything is broken"

    candidates = [autotune.Candidate({"label": "current"}, {}, "current")]
    args = argparse.Namespace(repeats=4, seed=0, drop_after=2)

    original = autotune.run_timed
    autotune.run_timed = fake_run_timed
    calls = []
    autotune.run_timed = lambda *a, **k: (calls.append(1), ("fail", None, "x"))[1]
    try:
        autotune.measure_rounds(None, candidates, args, ".")
    finally:
        autotune.run_timed = original
    assert len(calls) == 4


def count_fixture_processes():
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True).stdout
    return sum(1 for line in out.splitlines() if "fake_sched.py" in line and "grep" not in line)


def test_a_hung_run_does_not_leak_the_worker():
    """The failure that wrecked a real autotuning run.

    The wrapper used `selectors` on the pipe but `readline()` through a
    userspace buffer. A child that flushed several lines at once had the
    remainder sitting in that buffer while the pipe looked idle, so the launch
    pattern was never seen, no post-launch timeout applied, and the caller's
    hard timeout killed the wrapper and orphaned the worker. Five orphans
    holding ~19GB each filled a 94GB GPU, after which every candidate failed
    out of memory and the search reported "nothing beat the baseline".
    """
    before = count_fixture_processes()
    env = dict(os.environ, FAKE_SCHED_HANG="1")
    proc = subprocess.run(
        [sys.executable, os.path.join("tests", "script", "run_with_launch_timeout.py"),
         "--launch-pattern", "[bench]",
         "--post-launch-timeout", "3", "--post-launch-idle-timeout", "3",
         "--", sys.executable, FAKE_SCHED, "-b", "3"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    # the hang is detected, not waited out by an outer timeout
    assert proc.returncode == 124, proc.returncode
    assert "launch detected" in proc.stderr, proc.stderr
    assert "post-launch" in proc.stderr, proc.stderr

    time.sleep(2)
    assert count_fixture_processes() <= before, (
        "the hung worker outlived the wrapper; it would hold its device memory"
    )


def test_infrastructure_failures_are_not_legality_rejections():
    assert autotune.is_infrastructure_failure(
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB")
    assert autotune.is_infrastructure_failure("RuntimeError: no CUDA-capable device")
    # a real schedule rejection must not be swallowed as an infra problem
    assert not autotune.is_infrastructure_failure(
        "AssertionError: SMS must be multiple of M tiles, got SMS=48")
    assert not autotune.is_infrastructure_failure(
        "RuntimeError: launch_dae failed: misaligned address")
    assert not autotune.is_infrastructure_failure(None)


def test_filter_legal_aborts_rather_than_calling_an_oom_illegal():
    """An out-of-memory device makes every candidate look illegal."""
    registry = sample_registry()
    candidates = autotune.sweep_candidates(registry, only={"mlp.low"}, skip_baseline=True)

    original = autotune.run_dry_build
    autotune.run_dry_build = lambda *a, **k: (
        False, "torch.OutOfMemoryError: CUDA out of memory", 0.1)
    args = argparse.Namespace(workdir=".", build_timeout=1.0, no_prebuild=False)
    try:
        autotune.filter_legal(registry, {"namespace": "sample"}, candidates, args)
    except autotune.InfrastructureError as exc:
        assert "host reason" in str(exc)
    else:
        raise AssertionError("expected an OOM to abort rather than reject")
    finally:
        autotune.run_dry_build = original


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
