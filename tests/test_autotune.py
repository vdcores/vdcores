#!/usr/bin/env python3
"""Tests for tools/autotune.py.

The end-to-end cases drive the real driver against `tests/fake_sched.py`, a
stand-in target that reproduces the Qwen3 1.7B fold rules. That keeps the whole
discover/enumerate/check pipeline testable on a host with no CUDA and no
PyTorch.

Run with:

    python tests/test_autotune.py
"""

import importlib.util
import json
import os
import sys
import tempfile

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
