"""Schedule tuning configuration layer.

App schedules declare the knobs an autotuner is allowed to vary through a
`TuneConfig`. Declaring a knob does three things at once:

- resolves its value for this process,
- records the value and where it came from,
- publishes the legal search values so a driver can enumerate candidates
  without hardcoding a copy of the schedule's constraints.

Resolution order, highest priority first:

1. `DAE_TUNE_SET="q_proj.sms=64,down_proj.sms=96"` inline overrides
2. the knob's legacy environment variable, such as `QWEN1P7B_QPROJ_SMS`
3. the JSON config file named by `DAE_TUNE_CONFIG`
4. the default baked into the schedule

`DAE_TUNE_DUMP=path.json` writes the fully resolved configuration plus the
knob specs at process exit. That dump is the self-describing knob registry:
a driver runs a schedule once with `--dry-build` to learn what is tunable,
then feeds edited copies back in through `DAE_TUNE_CONFIG`.

This module is intentionally stdlib-only so it can be imported on a host
without CUDA, PyTorch, or the built `dae.runtime` extension.
"""

import atexit
import json
import os
import sys

ENV_CONFIG = "DAE_TUNE_CONFIG"
ENV_SET = "DAE_TUNE_SET"
ENV_DUMP = "DAE_TUNE_DUMP"

KIND_INT = "int"
KIND_BOOL = "bool"
KIND_STR_SET = "str_set"


class KnobError(ValueError):
    """Raised when a knob is declared inconsistently or given an unusable value."""


def _parse_int(name: str, raw):
    if isinstance(raw, bool):
        raise KnobError(f"knob {name}: expected an int, got bool {raw!r}")
    if isinstance(raw, int):
        return raw
    try:
        return int(str(raw).strip())
    except ValueError:
        raise KnobError(f"knob {name}: expected an int, got {raw!r}") from None


def _parse_bool(name: str, raw):
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, int):
        return bool(raw)
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", ""):
        return False
    raise KnobError(f"knob {name}: expected a bool, got {raw!r}")


def _parse_str_set(name: str, raw):
    if isinstance(raw, (list, tuple, set, frozenset)):
        tokens = [str(item).strip() for item in raw]
    else:
        tokens = [token.strip() for token in str(raw).split(",")]
    return frozenset(token for token in tokens if token)


_PARSERS = {
    KIND_INT: _parse_int,
    KIND_BOOL: _parse_bool,
    KIND_STR_SET: _parse_str_set,
}


class KnobSpec:
    """One tunable schedule parameter."""

    def __init__(self, name, kind, default, choices=None, doc=None, legacy_env=None):
        self.name = name
        self.kind = kind
        self.default = default
        self.choices = list(choices) if choices is not None else None
        self.doc = doc
        self.legacy_env = legacy_env
        self.value = default
        self.origin = "default"

    def as_json(self):
        payload = {
            "kind": self.kind,
            "default": _jsonable(self.default),
            "value": _jsonable(self.value),
            "origin": self.origin,
        }
        if self.choices is not None:
            payload["choices"] = [_jsonable(choice) for choice in self.choices]
        if self.doc:
            payload["doc"] = self.doc
        if self.legacy_env:
            payload["legacy_env"] = self.legacy_env
        return payload


def _jsonable(value):
    if isinstance(value, (frozenset, set)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _load_config_file(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise KnobError(f"{ENV_CONFIG}={path}: expected a JSON object at the top level")
    # Accept both a flat {"knob": value} mapping and a full dump that nests the
    # values under "knobs", so a dump can be edited and fed straight back in.
    knobs = payload.get("knobs", payload) if "knobs" in payload else payload
    if not isinstance(knobs, dict):
        raise KnobError(f"{ENV_CONFIG}={path}: 'knobs' must be a JSON object")
    namespace = payload.get("namespace") if isinstance(payload.get("namespace"), str) else None
    return namespace, {str(key): value for key, value in knobs.items() if key != "namespace"}


def _parse_inline_overrides(raw: str):
    overrides = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise KnobError(f"{ENV_SET}: expected name=value entries, got {chunk!r}")
        key, value = chunk.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


class TuneConfig:
    """Resolved knob values for one schedule namespace."""

    def __init__(self, namespace, file_values=None, inline_values=None,
                 source=None, environ=None):
        self.namespace = namespace
        self.source = source
        self._environ = os.environ if environ is None else environ
        self._file_values = dict(file_values or {})
        self._inline_values = dict(inline_values or {})
        self._specs = {}

    # -- declaration ------------------------------------------------------

    def knob(self, name, kind, default, choices=None, doc=None, legacy_env=None):
        """Declare a knob and return its resolved value."""
        if name in self._specs:
            raise KnobError(f"knob {name}: declared twice in namespace {self.namespace}")

        parse = _PARSERS[kind]
        spec = KnobSpec(name, kind, parse(name, default), choices, doc, legacy_env)

        raw, origin = self._lookup(name, legacy_env)
        if raw is not None:
            spec.value = parse(name, raw)
            spec.origin = origin

        self._specs[name] = spec
        return spec.value

    def _lookup(self, name, legacy_env):
        if name in self._inline_values:
            return self._inline_values[name], f"env:{ENV_SET}"
        if legacy_env is not None:
            raw = self._environ.get(legacy_env)
            if raw is not None:
                if name in self._file_values:
                    print(
                        f"[tune] warning: {legacy_env}={raw!r} overrides the config-file "
                        f"value for {name!r} ({self._file_values[name]!r})",
                        file=sys.stderr,
                    )
                return raw, f"env:{legacy_env}"
        if name in self._file_values:
            return self._file_values[name], "config"
        return None, "default"

    def int_knob(self, name, default, choices=None, doc=None, legacy_env=None):
        return self.knob(name, KIND_INT, default, choices, doc, legacy_env)

    def bool_knob(self, name, default, doc=None, legacy_env=None):
        return self.knob(name, KIND_BOOL, default, [False, True], doc, legacy_env)

    def str_set_knob(self, name, default=(), choices=None, doc=None, legacy_env=None):
        return self.knob(name, KIND_STR_SET, default, choices, doc, legacy_env)

    def sms(self, stage, default, choices=None, legacy_env=None, doc=None):
        """Declare the SM count for one placed schedule stage."""
        return self.int_knob(
            f"{stage}.sms",
            default,
            choices=choices,
            doc=doc or f"SM count for the {stage} stage",
            legacy_env=legacy_env,
        )

    def base_sm(self, stage, default, choices=None, legacy_env=None, doc=None):
        """Declare the first SM of the range for one placed schedule stage."""
        return self.int_knob(
            f"{stage}.base_sm",
            default,
            choices=choices,
            doc=doc or f"First SM of the {stage} stage placement range",
            legacy_env=legacy_env,
        )

    # -- introspection ----------------------------------------------------

    def specs(self):
        return dict(self._specs)

    def values(self):
        return {name: _jsonable(spec.value) for name, spec in self._specs.items()}

    def unused_keys(self):
        """Config-file and inline keys that no knob declaration ever read."""
        declared = set(self._specs)
        supplied = set(self._file_values) | set(self._inline_values)
        return sorted(supplied - declared)

    def as_json(self):
        return {
            "namespace": self.namespace,
            "source": self.source,
            "knobs": self.values(),
            "specs": {name: spec.as_json() for name, spec in self._specs.items()},
        }

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.as_json(), handle, indent=2, sort_keys=True)
            handle.write("\n")
        return path

    def summary(self):
        """One line per knob that is not sitting at its default."""
        changed = [spec for spec in self._specs.values() if spec.origin != "default"]
        if not changed:
            return f"[tune] {self.namespace}: all {len(self._specs)} knobs at defaults"
        lines = [f"[tune] {self.namespace}: {len(changed)} of {len(self._specs)} knobs overridden"]
        for spec in sorted(changed, key=lambda item: item.name):
            lines.append(f"[tune]   {spec.name} = {_jsonable(spec.value)} ({spec.origin})")
        return "\n".join(lines)

    def report(self):
        unused = self.unused_keys()
        if unused:
            print(
                f"[tune] warning: {self.namespace}: ignored unknown knob(s): {', '.join(unused)}",
                file=sys.stderr,
            )
        dump_path = self._environ.get(ENV_DUMP)
        if dump_path:
            self.dump(dump_path)
            print(f"[tune] wrote {len(self._specs)} knob specs to {dump_path}", file=sys.stderr)


def load(namespace, environ=None):
    """Build the `TuneConfig` for one schedule namespace from the environment."""
    environ = os.environ if environ is None else environ

    file_values = {}
    source = None
    config_path = environ.get(ENV_CONFIG)
    if config_path:
        file_namespace, file_values = _load_config_file(config_path)
        source = config_path
        if file_namespace is not None and file_namespace != namespace:
            raise KnobError(
                f"{ENV_CONFIG}={config_path}: config is for namespace "
                f"{file_namespace!r}, but this schedule is {namespace!r}"
            )

    inline_raw = environ.get(ENV_SET)
    inline_values = _parse_inline_overrides(inline_raw) if inline_raw else {}

    config = TuneConfig(
        namespace,
        file_values=file_values,
        inline_values=inline_values,
        source=source,
        environ=environ,
    )
    # Unknown keys and the dump can only be resolved once the schedule has
    # finished declaring its knobs, which is anywhere between here and exit.
    atexit.register(config.report)
    return config
