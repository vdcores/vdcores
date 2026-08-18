#!/usr/bin/env python3
"""Tests for the schedule tuning configuration layer.

`python/dae/tune.py` is stdlib-only on purpose, so this test loads it directly
from its file path instead of importing the `dae` package. That keeps the test
runnable on a host without CUDA, PyTorch, or the built `dae.runtime` extension.

Run with:

    python tests/test_tune.py
"""

import importlib.util
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNE_PATH = os.path.join(REPO_ROOT, "python", "dae", "tune.py")


def load_tune_module():
    spec = importlib.util.spec_from_file_location("dae_tune_under_test", TUNE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tune = load_tune_module()


def declare_sample(config):
    """A stand-in for the knob block of an app schedule."""
    return {
        "q_proj.sms": config.sms("q_proj", 64, [32, 64, 96, 128], legacy_env="QWEN1P7B_QPROJ_SMS"),
        "q_proj.base_sm": config.base_sm("q_proj", 0, [0, 64]),
        "mlp.low": config.int_knob("mlp.low", 4096, choices=[2048, 4096]),
        "no_prefetch": config.str_set_knob("no_prefetch", (), legacy_env="QWEN1P7B_NO_PREFETCH"),
    }


def test_defaults_when_environment_is_empty():
    config = tune.TuneConfig("qwen3_1p7b", environ={})
    values = declare_sample(config)
    assert values["q_proj.sms"] == 64, values
    assert values["mlp.low"] == 4096, values
    assert values["no_prefetch"] == frozenset(), values
    assert config.unused_keys() == []
    assert all(spec.origin == "default" for spec in config.specs().values())


def test_config_file_values_override_defaults():
    config = tune.TuneConfig(
        "qwen3_1p7b",
        file_values={"q_proj.sms": 96, "no_prefetch": "logits,down_proj"},
        environ={},
    )
    values = declare_sample(config)
    assert values["q_proj.sms"] == 96, values
    assert values["no_prefetch"] == frozenset({"logits", "down_proj"}), values
    assert config.specs()["q_proj.sms"].origin == "config"
    assert config.specs()["mlp.low"].origin == "default"


def test_legacy_env_beats_config_file():
    config = tune.TuneConfig(
        "qwen3_1p7b",
        file_values={"q_proj.sms": 96},
        environ={"QWEN1P7B_QPROJ_SMS": "32"},
    )
    values = declare_sample(config)
    assert values["q_proj.sms"] == 32, values
    assert config.specs()["q_proj.sms"].origin == "env:QWEN1P7B_QPROJ_SMS"


def test_inline_set_beats_legacy_env():
    config = tune.TuneConfig(
        "qwen3_1p7b",
        file_values={"q_proj.sms": 96},
        inline_values={"q_proj.sms": "128"},
        environ={"QWEN1P7B_QPROJ_SMS": "32"},
    )
    values = declare_sample(config)
    assert values["q_proj.sms"] == 128, values
    assert config.specs()["q_proj.sms"].origin == f"env:{tune.ENV_SET}"


def test_unknown_keys_are_reported_not_silently_dropped():
    config = tune.TuneConfig(
        "qwen3_1p7b",
        file_values={"q_proj.sms": 96, "qproj_sms": 96, "typo.sms": 1},
        environ={},
    )
    declare_sample(config)
    assert config.unused_keys() == ["qproj_sms", "typo.sms"], config.unused_keys()


def test_declaring_a_knob_twice_is_an_error():
    config = tune.TuneConfig("qwen3_1p7b", environ={})
    config.int_knob("mlp.low", 4096)
    try:
        config.int_knob("mlp.low", 2048)
    except tune.KnobError:
        return
    raise AssertionError("expected KnobError on duplicate knob declaration")


def test_bad_value_is_an_error():
    config = tune.TuneConfig("qwen3_1p7b", file_values={"q_proj.sms": "sixty-four"}, environ={})
    try:
        config.sms("q_proj", 64)
    except tune.KnobError:
        return
    raise AssertionError("expected KnobError on an unparsable int knob")


def test_dump_round_trips_through_config_file():
    config = tune.TuneConfig("qwen3_1p7b", file_values={"q_proj.sms": 96}, environ={})
    declare_sample(config)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dump.json")
        config.dump(path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        assert payload["namespace"] == "qwen3_1p7b"
        assert payload["knobs"]["q_proj.sms"] == 96
        # str_set knobs must serialize as a sorted list, not a set repr.
        assert payload["knobs"]["no_prefetch"] == []
        assert payload["specs"]["q_proj.sms"]["choices"] == [32, 64, 96, 128]
        assert payload["specs"]["q_proj.sms"]["origin"] == "config"

        # Feeding the dump straight back in must reproduce the same values.
        namespace, values = tune._load_config_file(path)
        assert namespace == "qwen3_1p7b"
        reloaded = tune.TuneConfig("qwen3_1p7b", file_values=values, environ={})
        assert declare_sample(reloaded) == declare_sample(
            tune.TuneConfig("qwen3_1p7b", file_values={"q_proj.sms": 96}, environ={})
        )


def test_namespace_mismatch_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "other.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"namespace": "llama3", "knobs": {"q_proj.sms": 96}}, handle)
        try:
            tune.load("qwen3_1p7b", environ={tune.ENV_CONFIG: path})
        except tune.KnobError:
            return
    raise AssertionError("expected KnobError when the config namespace does not match")


def test_flat_config_file_is_accepted():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "flat.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"q_proj.sms": 32}, handle)
        namespace, values = tune._load_config_file(path)
        assert namespace is None
        assert values == {"q_proj.sms": 32}


def test_inline_override_parsing():
    parsed = tune._parse_inline_overrides(" q_proj.sms=64 , down_proj.sms=96 ,")
    assert parsed == {"q_proj.sms": "64", "down_proj.sms": "96"}, parsed
    try:
        tune._parse_inline_overrides("q_proj.sms")
    except tune.KnobError:
        return
    raise AssertionError("expected KnobError on a malformed DAE_TUNE_SET entry")


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
