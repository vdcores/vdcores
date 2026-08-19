#!/usr/bin/env python3
"""Schedule autotuner driver.

This stage of the driver only answers "is this candidate schedule buildable",
not "is it fast". It has two filters:

- static: rules evaluated from the candidate values alone, no subprocess
- dry-build: run the target schedule with `--dry-build` and see if it builds

The static filter is free and runs anywhere. The dry-build filter needs the
machine that can construct the schedule, but it does not run the kernel, so it
costs seconds rather than the ~10s of a real profiled run. Everything the
driver knows about a target comes from the knob registry that target dumps, so
adding a tunable schedule does not mean editing this file.

Typical use:

    python tools/autotune.py discover --target qwen3_1p7b -o tuning/qwen3_1p7b.knobs.json
    python tools/autotune.py check --knobs tuning/qwen3_1p7b.knobs.json -o tuning/qwen3_1p7b.legality.json
    python tools/autotune.py report tuning/qwen3_1p7b.legality.json
"""

import argparse
import json
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT_WRAPPER = os.path.join("tests", "script", "run_with_launch_timeout.py")
HANG_RETURNCODE = 124
BENCH_MARKER = "[bench]"

TARGETS = {
    "qwen3_1p7b": {
        "namespace": "qwen3_1p7b",
        "script": "app/python/qwen3_1p7b/sched.py",
        "dry_build_args": ["--dry-build"],
        "correctness_args": ["--correctness"],
    },
    "llama3_8b": {
        "namespace": "llama3_8b",
        "script": "app/python/llama3/sched.py",
        "dry_build_args": ["--dry-build"],
        # Needs a Hugging Face credential and real weights; --dry-build does not.
        #
        # -N 8 on purpose. The default 128-step check compares the *final*
        # decode position against a greedy reference, and over that many steps
        # a sub-ulp difference eventually flips one token, after which the two
        # runs are decoding different sequences entirely and every tensor
        # differs by ~100%. That is divergence, not a wrong schedule. Eight
        # tokens still match exactly, so it is a real end-to-end check of every
        # projection, attention, the MLP, logits and argmax, and it is stable.
        "correctness_args": ["--correctness", "-N", "8"],
    },
}


# ---------------------------------------------------------------- registry


class KnobRegistry:
    """The knob surface of one target, as dumped by `dae.tune`."""

    def __init__(self, payload):
        self.namespace = payload.get("namespace")
        self.notes = payload.get("notes", {})
        self.specs = payload.get("specs", {})
        self.baseline = payload.get("knobs", {})
        if not self.specs:
            raise ValueError("knob registry has no specs; was DAE_TUNE_DUMP written by dae.tune?")

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def apply_preset(self, path):
        """Start from a saved configuration instead of the schedule's defaults.

        A search that stops early, or one run per knob group on separate days,
        is only useful if the next run can pick up where the last one left off.
        A preset is just a config file, so the same artifact also feeds
        `DAE_TUNE_CONFIG` directly.
        """
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        knobs = payload.get("knobs", payload) if "knobs" in payload else payload
        namespace = payload.get("namespace") if isinstance(payload, dict) else None
        if namespace and self.namespace and namespace != self.namespace:
            raise SystemExit(
                f"preset is for namespace {namespace!r}, but these knobs are "
                f"{self.namespace!r}"
            )
        knobs = {key: value for key, value in knobs.items() if key != "namespace"}
        unknown = sorted(name for name in knobs if name not in self.specs)
        if unknown:
            raise SystemExit(f"preset sets knobs this target does not have: {', '.join(unknown)}")
        self.baseline.update(knobs)
        return len(knobs)

    def choices(self, name):
        return self.specs.get(name, {}).get("choices") or []

    def sweepable(self):
        """Knobs with more than one candidate value, in declaration order."""
        return [name for name in self.specs if len(self.choices(name)) > 1]

    def stages(self):
        """Stage names that have both an SM count and a base SM knob."""
        found = []
        for name in self.specs:
            if not name.endswith(".sms"):
                continue
            stage = name[: -len(".sms")]
            if f"{stage}.base_sm" in self.specs:
                found.append(stage)
        return found


# ------------------------------------------------------------ static rules


def rule_sms_positive(registry, values):
    for name, value in values.items():
        if name.endswith(".sms") and isinstance(value, int) and value < 1:
            return f"{name}={value} must be at least 1"
    return None


def rule_sm_range_fits(registry, values):
    full_sms = registry.notes.get("full_sms")
    if full_sms is None:
        return None
    for stage in registry.stages():
        sms = values.get(f"{stage}.sms")
        base = values.get(f"{stage}.base_sm")
        if not isinstance(sms, int) or not isinstance(base, int):
            continue
        if base < 0:
            return f"{stage}.base_sm={base} must not be negative"
        if base + sms > full_sms:
            return (
                f"{stage} range [{base}, {base + sms}) exceeds full_sms={full_sms}"
            )
    return None


def rule_mlp_split_in_range(registry, values):
    low = values.get("mlp.low")
    intermediate = registry.notes.get("intermediate")
    if not isinstance(low, int) or not isinstance(intermediate, int):
        return None
    if low <= 0 or low >= intermediate:
        return f"mlp.low={low} must be inside (0, intermediate={intermediate})"
    return None


STATIC_RULES = (rule_sms_positive, rule_sm_range_fits, rule_mlp_split_in_range)


def static_reject_reason(registry, values):
    for rule in STATIC_RULES:
        reason = rule(registry, values)
        if reason is not None:
            return reason
    return None


# -------------------------------------------------------------- candidates


class Candidate:
    def __init__(self, values, overrides, label):
        self.values = values
        self.overrides = overrides
        self.label = label


def sweep_candidates(registry, only=None, skip_baseline=False):
    """One candidate per off-baseline choice, varying a single knob at a time.

    This is the coordinate sweep that produces a legality map. The actual
    search over combinations comes later; here each candidate isolates one
    knob so a rejection can be attributed to it.
    """
    candidates = []
    if not skip_baseline:
        candidates.append(Candidate(dict(registry.baseline), {}, "baseline"))

    for name in registry.sweepable():
        if only and name not in only:
            continue
        current = registry.baseline.get(name)
        for choice in registry.choices(name):
            if choice == current:
                continue
            values = dict(registry.baseline)
            values[name] = choice
            candidates.append(Candidate(values, {name: choice}, f"{name}={choice}"))
    return candidates


# ---------------------------------------------------------- knob groups


class KnobGroup:
    """Knobs that have to move together, plus their legal combinations.

    A stage's SM count and its base SM are one group. Moving them one at a
    time makes wide placements unreachable: `up_low.sms=128` only becomes
    legal once `up_low.base_sm` drops to 0, and a single-knob sweep never
    tries that pair, so it reports the value as illegal when it is merely
    unreachable from the current baseline.
    """

    def __init__(self, name, knobs):
        self.name = name
        self.knobs = knobs

    def combinations(self, registry):
        """Cross product of every member knob's choices."""
        combos = [{}]
        for knob in self.knobs:
            choices = registry.choices(knob) or [registry.baseline.get(knob)]
            combos = [
                dict(combo, **{knob: choice})
                for combo in combos
                for choice in choices
            ]
        return combos

    def size(self, registry):
        total = 1
        for knob in self.knobs:
            total *= max(len(registry.choices(knob)), 1)
        return total


def knob_groups(registry, only=None):
    """Placement pairs first, then every remaining knob on its own."""
    groups = []
    grouped = set()
    for stage in registry.stages():
        knobs = [f"{stage}.sms", f"{stage}.base_sm"]
        groups.append(KnobGroup(stage, knobs))
        grouped.update(knobs)
    for name in registry.specs:
        if name in grouped or len(registry.choices(name)) <= 1:
            continue
        groups.append(KnobGroup(name, [name]))
    if only:
        known = {group.name for group in groups}
        unknown = sorted(set(only) - known)
        if unknown:
            raise SystemExit(
                f"unknown group(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}"
            )
        groups = [group for group in groups if group.name in only]
    return groups


def group_candidates(registry, group, current):
    """Vary one group across its combinations, holding everything else at `current`."""
    candidates = []
    for combo in group.combinations(registry):
        if all(current.get(knob) == value for knob, value in combo.items()):
            continue
        values = dict(current)
        values.update(combo)
        label = " ".join(f"{knob}={value}" for knob, value in combo.items())
        candidates.append(Candidate(values, dict(combo), label))
    return candidates


def config_key(values):
    """Hashable identity for one configuration, for caching build results."""
    return json.dumps(values, sort_keys=True, default=str)


# ------------------------------------------------------------ dry-build run


def classify_failure(output):
    """Pull the most informative line out of a failed build."""
    interesting = ("AssertionError", "ValueError", "KnobError", "Error:", "error:")
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in reversed(lines):
        if any(marker in line for marker in interesting):
            return line[:400]
    return lines[-1][:400] if lines else "no output"


def run_dry_build(target, values, timeout, workdir):
    """Build one candidate in a subprocess. Returns (ok, reason, seconds)."""
    script = os.path.join(workdir, target["script"])
    command = [sys.executable, script, *target.get("dry_build_args", [])]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # A stale legacy export would silently outrank the config file we write.
    env.pop("DAE_TUNE_SET", None)
    env.pop("DAE_TUNE_DUMP", None)

    handle, config_path = tempfile.mkstemp(prefix="autotune-", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as config_file:
        json.dump({"namespace": target["namespace"], "knobs": values}, config_file)
    env["DAE_TUNE_CONFIG"] = config_path

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, f"dry-build timed out after {timeout}s", time.monotonic() - started
    finally:
        os.unlink(config_path)

    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        return False, classify_failure(proc.stdout or ""), elapsed
    return True, None, elapsed


# -------------------------------------------------------------- timed runs

BENCH_LINE = re.compile(r"^(Min|Median|Average|Max) execution time \(ns\):\s*([0-9.]+)", re.M)


def parse_bench_output(text):
    """Pull the timing block printed by `dae.bench` out of a run's output."""
    found = {match.group(1).lower(): float(match.group(2)) for match in BENCH_LINE.finditer(text)}
    return found if "median" in found else None


def write_config(target, values):
    handle, path = tempfile.mkstemp(prefix="autotune-", suffix=".json")
    with os.fdopen(handle, "w", encoding="utf-8") as config_file:
        json.dump({"namespace": target["namespace"], "knobs": values}, config_file)
    return path


def clean_env(extra=None):
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # A stale export would silently outrank the config file we write.
    env.pop("DAE_TUNE_SET", None)
    env.pop("DAE_TUNE_DUMP", None)
    env.pop("DAE_TUNE_CONFIG", None)
    env.update(extra or {})
    return env


def kill_stale(target, workdir):
    """Kill leftover runs of *this target only*.

    Deliberately not a blanket `killall python`: the driver is itself a python
    process, and so is anything else the user happens to be running.
    """
    subprocess.run(["pkill", "-f", target["script"]], cwd=workdir,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_timed(target, values, args, workdir):
    """Time one candidate in a fresh process. Returns (status, ns, reason)."""
    if args.kill_stale:
        kill_stale(target, workdir)

    config_path = write_config(target, values)
    env = clean_env({
        "DAE_TUNE_CONFIG": config_path,
        "DAE_BENCH_WARMUP": str(args.warmup),
    })

    command = [
        sys.executable, TIMEOUT_WRAPPER,
        "--launch-pattern", BENCH_MARKER,
        "--post-launch-timeout", str(args.post_launch_timeout),
        "--post-launch-idle-timeout", str(args.idle_timeout),
        "--", sys.executable, target["script"], "-b", str(args.iterations),
    ]

    try:
        proc = subprocess.run(
            command, cwd=workdir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.hard_timeout, text=True,
        )
        output = proc.stdout or ""
        returncode = proc.returncode
    except subprocess.TimeoutExpired as expired:
        output = expired.stdout or ""
        returncode = HANG_RETURNCODE
    finally:
        os.unlink(config_path)

    if returncode == HANG_RETURNCODE:
        return "hang", None, "no progress after launch; likely a barrier deadlock"
    if returncode != 0:
        return "fail", None, classify_failure(output)

    timings = parse_bench_output(output)
    if timings is None:
        return "fail", None, "run succeeded but printed no benchmark results"
    return "ok", timings["median"], None


def filter_legal(registry, target, candidates, args, cache=None):
    """Static rules first, then a dry build. Returns (kept, skipped).

    The static rules are free, so they run first and keep the subprocess count
    down. `cache` memoizes dry-build verdicts across passes of a search, where
    the same configuration is otherwise rebuilt every pass.
    """
    kept, skipped = [], []
    for candidate in candidates:
        reason = static_reject_reason(registry, candidate.values)
        if reason is None and not getattr(args, "no_prebuild", False):
            key = config_key(candidate.values)
            if cache is not None and key in cache:
                reason = cache[key]
            else:
                ok, build_reason, _ = run_dry_build(
                    target, candidate.values, args.build_timeout, args.workdir
                )
                reason = None if ok else build_reason
                if cache is not None:
                    cache[key] = reason
        if reason is None:
            kept.append(candidate)
        else:
            skipped.append({"label": candidate.label, "reason": reason})
    return kept, skipped


def run_correctness(target, values, args):
    """Run the target's own correctness check under one configuration.

    Returns (True, None), (False, reason), or (None, reason) when the target
    declares no correctness check. A schedule that is fast but wrong is not an
    improvement, and timing alone cannot tell the difference, so a search must
    not adopt a configuration it has not verified.
    """
    correctness_args = target.get("correctness_args")
    if not correctness_args:
        return None, "target declares no correctness check"

    config_path = write_config(target, values)
    env = clean_env({"DAE_TUNE_CONFIG": config_path})
    command = [sys.executable, target["script"], *correctness_args]
    try:
        proc = subprocess.run(
            command, cwd=args.workdir, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=args.correctness_timeout, text=True,
        )
        output, returncode = proc.stdout or "", proc.returncode
    except subprocess.TimeoutExpired as expired:
        output, returncode = expired.stdout or "", HANG_RETURNCODE
    finally:
        os.unlink(config_path)

    if returncode == HANG_RETURNCODE:
        return False, f"correctness run timed out after {args.correctness_timeout}s"
    if returncode != 0:
        return False, classify_failure(output)
    return True, None


def save_preset(path, target, values, meta=None):
    """Write a configuration in the format `DAE_TUNE_CONFIG` already accepts.

    `tune._load_config_file` ignores top-level keys other than `knobs` and
    `namespace`, so the provenance block rides along without breaking the
    round trip.
    """
    payload = {"namespace": target["namespace"], "knobs": values}
    if meta:
        payload["search"] = meta
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return path


# ------------------------------------------------------- noise-aware stats


def percentile(values, fraction):
    """Linear-interpolated percentile, safe for the small samples we collect."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_samples(samples):
    if not samples:
        return None
    p25 = percentile(samples, 0.25)
    p75 = percentile(samples, 0.75)
    return {
        "n": len(samples),
        "min": min(samples),
        "median": statistics.median(samples),
        "max": max(samples),
        "p25": p25,
        "p75": p75,
        "iqr": p75 - p25,
    }


def bootstrap_delta_ci(candidate_samples, baseline_samples, rounds, confidence, rng):
    """Confidence interval for the difference of medians, by resampling.

    A fixed threshold on the raw spread is not enough: with a handful of runs
    per configuration the medians themselves are uncertain, and a comparison
    that ignores that will keep promoting lucky runs. Resampling turns the
    sample count into the width of the interval, so too few repeats produces
    an interval that straddles zero rather than a confident wrong answer.

    Resampling also survives the bimodal timing this host shows, which a
    mean-and-standard-deviation test would not.
    """
    deltas = []
    for _ in range(rounds):
        candidate = statistics.median(
            [rng.choice(candidate_samples) for _ in candidate_samples])
        baseline = statistics.median(
            [rng.choice(baseline_samples) for _ in baseline_samples])
        deltas.append(candidate - baseline)
    tail = (1.0 - confidence) / 2.0
    return percentile(deltas, tail), percentile(deltas, 1.0 - tail)


def corrected_confidence(confidence, comparisons, enabled=True):
    """Bonferroni-style correction for testing many candidates at once.

    A 95% interval is wrong one time in twenty by construction, so a sweep of
    forty candidates should be expected to crown roughly two winners that are
    nothing but luck. Spreading the error budget across the comparisons keeps
    the sweep-level false positive rate at the level the user asked for.
    """
    if not enabled or comparisons <= 1:
        return confidence
    return 1.0 - (1.0 - confidence) / comparisons


def decide(candidate_samples, baseline_samples, min_effect_ns, args, rng, confidence=None):
    """Verdict, observed delta, and the confidence interval behind it."""
    delta = statistics.median(candidate_samples) - statistics.median(baseline_samples)

    if len(candidate_samples) < 2 or len(baseline_samples) < 2:
        return "insufficient", delta, (None, None)

    low, high = bootstrap_delta_ci(
        candidate_samples, baseline_samples,
        args.bootstrap, args.confidence if confidence is None else confidence, rng,
    )
    # Two conditions, both required: the interval must exclude zero, and the
    # effect must be big enough to be worth acting on.
    if high < 0 and delta < -min_effect_ns:
        verdict = "faster"
    elif low > 0 and delta > min_effect_ns:
        verdict = "slower"
    else:
        verdict = "same"
    return verdict, delta, (low, high)


def measure_rounds(target, candidates, args, workdir, on_sample=None):
    """Measure every candidate once per round, in shuffled order.

    Round-robin rather than all-repeats-of-A-then-all-of-B, so that slow drift
    in the host spreads across every candidate instead of penalizing whichever
    one happened to be measured last.
    """
    rng = random.Random(args.seed)
    samples = {candidate.label: [] for candidate in candidates}
    failures = {}

    for round_index in range(args.repeats):
        order = list(candidates)
        rng.shuffle(order)
        for candidate in order:
            status, ns, reason = run_timed(target, candidate.values, args, workdir)
            if status == "ok":
                samples[candidate.label].append(ns)
            else:
                failures.setdefault(candidate.label, (status, reason))
            if on_sample is not None:
                on_sample(round_index, candidate, status, ns, reason)
    return samples, failures


def drift_report(baseline_samples):
    """Compare the first half of baseline samples against the second half."""
    if len(baseline_samples) < 4:
        return None
    half = len(baseline_samples) // 2
    first = summarize_samples(baseline_samples[:half])
    second = summarize_samples(baseline_samples[half:])
    return {
        "first_half_median": first["median"],
        "second_half_median": second["median"],
        "shift": second["median"] - first["median"],
    }


# ------------------------------------------------------------- subcommands


def resolve_target(args):
    if args.script:
        if not args.namespace:
            raise SystemExit("--script also needs --namespace")
        return {
            "namespace": args.namespace,
            "script": args.script,
            "dry_build_args": args.dry_build_arg or [],
        }
    if args.target not in TARGETS:
        raise SystemExit(f"unknown target {args.target!r}; known: {', '.join(sorted(TARGETS))}")
    return TARGETS[args.target]


def cmd_discover(args):
    target = resolve_target(args)
    script = os.path.join(args.workdir, target["script"])
    command = [sys.executable, script, *target.get("dry_build_args", [])]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DAE_TUNE_DUMP"] = os.path.abspath(args.out)
    env.pop("DAE_TUNE_CONFIG", None)
    env.pop("DAE_TUNE_SET", None)

    proc = subprocess.run(
        command, cwd=args.workdir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=args.timeout, text=True,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout or "")
        raise SystemExit(f"discover failed: {classify_failure(proc.stdout or '')}")

    registry = KnobRegistry.from_file(args.out)
    print(f"[discover] namespace={registry.namespace} knobs={len(registry.specs)} -> {args.out}")
    print(f"[discover] sweepable knobs: {len(registry.sweepable())}")
    if registry.notes:
        notes = ", ".join(f"{key}={value}" for key, value in sorted(registry.notes.items()))
        print(f"[discover] notes: {notes}")
    return 0


def cmd_enumerate(args):
    registry = KnobRegistry.from_file(args.knobs)
    candidates = sweep_candidates(registry, only=set(args.knob or []))
    for candidate in candidates:
        reason = static_reject_reason(registry, candidate.values)
        status = "static-reject" if reason else "candidate"
        suffix = f"  # {reason}" if reason else ""
        print(f"{status:14} {candidate.label}{suffix}")
    print(f"\n{len(candidates)} candidates from {len(registry.sweepable())} sweepable knobs")
    return 0


def target_for_registry(registry, args):
    known = TARGETS.get(registry.namespace, {})
    target = {
        "namespace": registry.namespace,
        "script": getattr(args, "script", None) or known.get("script"),
        "dry_build_args": getattr(args, "dry_build_arg", None) or known.get("dry_build_args", []),
        "correctness_args": (
            getattr(args, "correctness_arg", None) or known.get("correctness_args", [])
        ),
    }
    if not target["script"]:
        raise SystemExit(f"no script known for namespace {registry.namespace!r}; pass --script")
    return target


def cmd_check(args):
    registry = KnobRegistry.from_file(args.knobs)
    if getattr(args, "preset", None):
        registry.apply_preset(args.preset)
        print(f"[check] baseline taken from preset {args.preset}")
    target = target_for_registry(registry, args)

    candidates = sweep_candidates(registry, only=set(args.knob or []))
    if args.max is not None:
        candidates = candidates[: args.max]

    results = []
    counts = {"ok": 0, "static-reject": 0, "build-reject": 0}
    for index, candidate in enumerate(candidates, start=1):
        reason = static_reject_reason(registry, candidate.values)
        if reason is not None:
            stage, ok, seconds = "static-reject", False, 0.0
        elif args.static_only:
            stage, ok, seconds = "static-ok", True, 0.0
        else:
            ok, reason, seconds = run_dry_build(
                target, candidate.values, args.timeout, args.workdir
            )
            stage = "ok" if ok else "build-reject"

        key = "ok" if ok else stage
        counts[key] = counts.get(key, 0) + 1
        results.append({
            "label": candidate.label,
            "overrides": candidate.overrides,
            "stage": stage,
            "ok": ok,
            "reason": reason,
            "seconds": round(seconds, 2),
        })
        mark = "ok  " if ok else "FAIL"
        detail = "" if reason is None else f"  {reason}"
        print(f"[{index}/{len(candidates)}] {mark} {candidate.label}{detail}", flush=True)

    payload = {
        "namespace": registry.namespace,
        "script": target["script"],
        "static_only": bool(args.static_only),
        "baseline": registry.baseline,
        "notes": registry.notes,
        "counts": counts,
        "results": results,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\n[check] wrote {len(results)} results to {args.out}")

    print_summary(payload)
    baseline_failed = any(
        result["label"] == "baseline" and not result["ok"] for result in results
    )
    if baseline_failed:
        print("\n[check] baseline itself failed; treat every other result as suspect")
        return 1
    return 0


def print_summary(payload):
    results = payload["results"]
    by_knob = {}
    for result in results:
        if not result["overrides"]:
            continue
        name = next(iter(result["overrides"]))
        value = result["overrides"][name]
        by_knob.setdefault(name, []).append((value, result))

    print("\nLegal values per knob (baseline value marked *):")
    for name in sorted(by_knob):
        baseline_value = payload["baseline"].get(name)
        legal = [f"{baseline_value}*"]
        illegal = []
        for value, result in by_knob[name]:
            (legal if result["ok"] else illegal).append(str(value))
        line = f"  {name:22} legal: {', '.join(legal)}"
        if illegal:
            line += f"   rejected: {', '.join(illegal)}"
        print(line)

    counts = payload["counts"]
    total = sum(counts.values())
    print(
        f"\n{counts.get('ok', 0)}/{total} buildable, "
        f"{counts.get('static-reject', 0)} static-rejected, "
        f"{counts.get('build-reject', 0)} build-rejected"
    )

    reasons = {}
    for result in results:
        if result["ok"] or not result["reason"]:
            continue
        reasons[result["reason"]] = reasons.get(result["reason"], 0) + 1
    if reasons:
        print("\nRejection reasons:")
        for reason, count in sorted(reasons.items(), key=lambda item: -item[1]):
            print(f"  {count:3}x {reason}")


def cmd_noise(args):
    """Measure the same configuration repeatedly and report the spread.

    Run this before trusting any tuning result. If the spread is wider than
    the effect being searched for, the search cannot distinguish a better
    schedule from a lucky run, and that has to be fixed first.
    """
    registry = KnobRegistry.from_file(args.knobs)
    if getattr(args, "preset", None):
        registry.apply_preset(args.preset)
        print(f"[noise] measuring preset {args.preset}")
    target = target_for_registry(registry, args)
    baseline = Candidate(dict(registry.baseline), {}, "baseline")

    def progress(round_index, candidate, status, ns, reason):
        detail = f"{ns / 1e6:.3f} ms" if status == "ok" else f"{status}: {reason}"
        print(f"[{round_index + 1}/{args.repeats}] {detail}", flush=True)

    samples, failures = measure_rounds(target, [baseline], args, args.workdir, progress)
    values = samples["baseline"]
    if not values:
        status, reason = failures.get("baseline", ("fail", "no samples"))
        raise SystemExit(f"baseline never produced a timing ({status}: {reason})")

    stats = summarize_samples(values)
    spread_pct = stats["iqr"] / stats["median"] * 100
    range_pct = (stats["max"] - stats["min"]) / stats["median"] * 100
    drift = drift_report(values)

    print(f"\nRepeatability over {stats['n']} fresh runs of the baseline:")
    print(f"  min    {stats['min'] / 1e6:.3f} ms")
    print(f"  p25    {stats['p25'] / 1e6:.3f} ms")
    print(f"  median {stats['median'] / 1e6:.3f} ms")
    print(f"  p75    {stats['p75'] / 1e6:.3f} ms")
    print(f"  max    {stats['max'] / 1e6:.3f} ms")
    print(f"  IQR    {stats['iqr'] / 1e6:.3f} ms  ({spread_pct:.1f}% of median)")
    print(f"  range  {range_pct:.1f}% of median")
    if failures:
        print(f"  failed runs: {failures}")
    if drift:
        print(f"  drift  first half {drift['first_half_median'] / 1e6:.3f} ms -> "
              f"second half {drift['second_half_median'] / 1e6:.3f} ms")

    print(
        f"\nSuggested --min-effect-pct for this host: {max(spread_pct, 1.0):.1f}\n"
        "Improvements smaller than that cannot be told apart from noise here."
    )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({
                "namespace": registry.namespace,
                "samples_ns": values,
                "stats": stats,
                "spread_pct": spread_pct,
                "drift": drift,
            }, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[noise] wrote {len(values)} samples to {args.out}")
    return 0


def cmd_measure(args):
    registry = KnobRegistry.from_file(args.knobs)
    if getattr(args, "preset", None):
        registry.apply_preset(args.preset)
        print(f"[measure] baseline taken from preset {args.preset}")
    target = target_for_registry(registry, args)

    candidates = sweep_candidates(registry, only=set(args.knob or []))
    if args.max is not None:
        candidates = candidates[: args.max]

    kept, skipped = [], []
    for candidate in candidates:
        reason = static_reject_reason(registry, candidate.values)
        if reason is None and not args.no_prebuild and candidate.label != "baseline":
            ok, build_reason, _ = run_dry_build(
                target, candidate.values, args.build_timeout, args.workdir
            )
            reason = None if ok else build_reason
        if reason is None:
            kept.append(candidate)
        else:
            skipped.append({"label": candidate.label, "reason": reason})

    print(f"[measure] {len(kept)} candidates to time, {len(skipped)} rejected before timing")
    print(f"[measure] {args.repeats} rounds x {len(kept)} candidates = "
          f"{args.repeats * len(kept)} runs\n")

    def progress(round_index, candidate, status, ns, reason):
        detail = f"{ns / 1e6:.3f} ms" if status == "ok" else f"{status}: {reason}"
        print(f"[round {round_index + 1}/{args.repeats}] {candidate.label:24} {detail}", flush=True)

    samples, failures = measure_rounds(target, kept, args, args.workdir, progress)

    baseline_samples = samples.get("baseline", [])
    if not baseline_samples:
        raise SystemExit("baseline never produced a timing; fix that before trusting anything else")

    baseline_stats = summarize_samples(baseline_samples)
    min_effect_ns = args.min_effect_pct / 100.0 * baseline_stats["median"]
    rng = random.Random(args.seed + 1)
    comparisons = max(len(kept) - 1, 1)
    confidence = corrected_confidence(args.confidence, comparisons, not args.no_correction)

    results = []
    for candidate in kept:
        if candidate.label == "baseline":
            continue
        candidate_samples = samples[candidate.label]
        if not candidate_samples:
            status, reason = failures.get(candidate.label, ("fail", "no samples"))
            results.append({
                "label": candidate.label, "overrides": candidate.overrides,
                "verdict": status, "reason": reason, "stats": None,
            })
            continue
        stats = summarize_samples(candidate_samples)
        verdict, delta, (low, high) = decide(
            candidate_samples, baseline_samples, min_effect_ns, args, rng, confidence)
        results.append({
            "label": candidate.label,
            "overrides": candidate.overrides,
            "verdict": verdict,
            "reason": None,
            "stats": stats,
            "delta_ns": delta,
            "delta_pct": delta / baseline_stats["median"] * 100,
            "ci_low_ns": low,
            "ci_high_ns": high,
        })

    confirmations = confirm_winners(
        target, kept, results, baseline_stats, min_effect_ns, args, rng)

    payload = {
        "comparisons": comparisons,
        "corrected_confidence": confidence,
        "confirmations": confirmations,
        "namespace": registry.namespace,
        "baseline": registry.baseline,
        "baseline_stats": baseline_stats,
        "min_effect_ns": min_effect_ns,
        "min_effect_pct": args.min_effect_pct,
        "confidence": args.confidence,
        "repeats": args.repeats,
        "iterations": args.iterations,
        "drift": drift_report(baseline_samples),
        "skipped": skipped,
        "results": results,
    }

    print_timing_summary(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"\n[measure] wrote {len(results)} results to {args.out}")
    return 0


def confirm_winners(target, kept, results, baseline_stats, min_effect_ns, args, rng):
    """Re-measure apparent winners in a second independent pass.

    Correcting the confidence level lowers the false positive rate but does not
    remove it, and a schedule that only looks fast once is not a result. A
    winner has to win twice, on separately collected samples, before it is
    reported as one.
    """
    if args.confirm_rounds <= 0:
        return None
    winners = [result["label"] for result in results if result["verdict"] == "faster"]
    if not winners:
        return None

    by_label = {candidate.label: candidate for candidate in kept}
    retest = [by_label["baseline"]] + [by_label[label] for label in winners]
    print(f"\n[confirm] re-measuring {len(winners)} apparent winner(s) "
          f"over {args.confirm_rounds} fresh rounds")

    confirm_args = argparse.Namespace(**vars(args))
    confirm_args.repeats = args.confirm_rounds
    confirm_args.seed = args.seed + 1000

    def progress(round_index, candidate, status, ns, reason):
        detail = f"{ns / 1e6:.3f} ms" if status == "ok" else f"{status}: {reason}"
        print(f"[confirm {round_index + 1}/{args.confirm_rounds}] "
              f"{candidate.label:24} {detail}", flush=True)

    samples, _ = measure_rounds(target, retest, confirm_args, args.workdir, progress)
    baseline_samples = samples.get("baseline", [])
    if len(baseline_samples) < 2:
        return {"error": "confirmation baseline produced too few samples"}

    confidence = corrected_confidence(
        args.confidence, len(winners), not args.no_correction)
    confirmations = {}
    for label in winners:
        candidate_samples = samples.get(label, [])
        if len(candidate_samples) < 2:
            confirmations[label] = {"verdict": "insufficient"}
            continue
        verdict, delta, (low, high) = decide(
            candidate_samples, baseline_samples, min_effect_ns,
            confirm_args, rng, confidence)
        confirmations[label] = {
            "verdict": verdict,
            "delta_pct": delta / statistics.median(baseline_samples) * 100,
            "ci_low_ns": low,
            "ci_high_ns": high,
        }

    for result in results:
        confirmation = confirmations.get(result["label"])
        if confirmation is None:
            continue
        result["confirmed"] = confirmation["verdict"] == "faster"
        if not result["confirmed"]:
            result["verdict"] = "unconfirmed"
    return confirmations


def print_timing_summary(payload):
    baseline_stats = payload["baseline_stats"]
    print(
        f"\nBaseline: {baseline_stats['median'] / 1e6:.3f} ms median over "
        f"{baseline_stats['n']} runs, IQR {baseline_stats['iqr'] / 1e6:.3f} ms"
    )
    print(
        f"A candidate wins only if its bootstrap interval excludes zero and the effect "
        f"clears --min-effect-pct {payload['min_effect_pct']} "
        f"({payload['min_effect_ns'] / 1e6:.3f} ms)"
    )
    if payload.get("corrected_confidence"):
        print(
            f"Confidence {payload['confidence']:.0%} spread across "
            f"{payload['comparisons']} comparisons -> {payload['corrected_confidence']:.4%} "
            "per candidate"
        )

    drift = payload.get("drift")
    if drift and abs(drift["shift"]) > payload["min_effect_ns"]:
        print(
            f"\nWARNING: the baseline drifted by {drift['shift'] / 1e6:+.3f} ms between the "
            "first and second half of this session, which is more than the threshold. "
            "Treat these verdicts as unreliable and rerun on a quieter host."
        )

    timed = [result for result in payload["results"] if result["stats"]]
    ranked = sorted(timed, key=lambda result: result["stats"]["median"])
    print(f"\n{'candidate':26} {'median':>10} {'delta':>9} {'interval':>20}  verdict")
    for result in ranked:
        low, high = result.get("ci_low_ns"), result.get("ci_high_ns")
        interval = (
            f"[{low / 1e6:+.3f}, {high / 1e6:+.3f}]" if low is not None else "n/a"
        )
        print(
            f"  {result['label']:24} {result['stats']['median'] / 1e6:>8.3f} ms "
            f"{result['delta_pct']:>+7.1f}% {interval:>20}  {result['verdict']}"
        )

    broken = [result for result in payload["results"] if not result["stats"]]
    if broken:
        print("\nDid not produce a timing:")
        for result in broken:
            print(f"  {result['label']:24} {result['verdict']}: {result['reason']}")

    unconfirmed = [result for result in timed if result["verdict"] == "unconfirmed"]
    if unconfirmed:
        print("\nWon the first pass but not the confirmation pass, so not a result:")
        for result in unconfirmed:
            print(f"  {result['label']}")

    faster = [result for result in timed if result["verdict"] == "faster"]
    print(f"\n{len(faster)} candidate(s) beat the baseline")
    if not faster and timed:
        print("Nothing cleared the noise floor; the hand-tuned baseline stands for now.")


def _timing_progress(prefix, total_rounds):
    def progress(round_index, candidate, status, ns, reason):
        detail = f"{ns / 1e6:.3f} ms" if status == "ok" else f"{status}: {reason}"
        print(f"  [{prefix} {round_index + 1}/{total_rounds}] "
              f"{candidate.label:32} {detail}", flush=True)
    return progress


def head_to_head(target, origin_values, current_values, args, rng):
    """Time the original baseline against the searched config, directly.

    Greedy descent adopts each step against the configuration that preceded
    it, so a chain of individually-justified steps can still add up to less
    than it looks. The only honest number for the whole search is a fresh
    measurement of where it started against where it ended.
    """
    arms = [
        Candidate(dict(origin_values), {}, "origin"),
        Candidate(dict(current_values), {}, "searched"),
    ]
    final_args = argparse.Namespace(**vars(args))
    final_args.repeats = args.final_rounds
    final_args.seed = args.seed + 5000

    samples, _ = measure_rounds(
        target, arms, final_args, args.workdir,
        _timing_progress("final", args.final_rounds),
    )
    origin_samples = samples.get("origin", [])
    searched_samples = samples.get("searched", [])
    if len(origin_samples) < 2 or len(searched_samples) < 2:
        return {"verdict": "insufficient"}

    origin_median = statistics.median(origin_samples)
    min_effect_ns = args.min_effect_pct / 100.0 * origin_median
    verdict, delta, (low, high) = decide(
        searched_samples, origin_samples, min_effect_ns, final_args, rng)
    return {
        "verdict": verdict,
        "origin_median_ns": origin_median,
        "searched_median_ns": statistics.median(searched_samples),
        "delta_ns": delta,
        "delta_pct": delta / origin_median * 100,
        "ci_low_ns": low,
        "ci_high_ns": high,
        "rounds": args.final_rounds,
    }


def cmd_search(args):
    """Coordinate descent over knob groups.

    One group is optimized at a time, against the configuration reached so far
    rather than against the original baseline, and a step is only taken when
    it wins the same noise-aware test `measure` uses, wins it twice, and
    survives the correctness gate.
    """
    registry = KnobRegistry.from_file(args.knobs)
    if args.preset:
        applied = registry.apply_preset(args.preset)
        print(f"[search] starting from preset {args.preset} ({applied} knob(s))")
    target = target_for_registry(registry, args)

    origin = dict(registry.baseline)
    current = dict(origin)
    groups = knob_groups(registry, only=set(args.group or []))
    if not groups:
        raise SystemExit("no knob groups to search")

    total_space = 1
    for group in groups:
        total_space *= group.size(registry)
    print(f"[search] {len(groups)} group(s): {', '.join(g.name for g in groups)}")
    print(f"[search] product of group sizes is {total_space:,} configurations; "
          f"coordinate descent visits a small fraction of that")
    print(f"[search] up to {args.max_passes} pass(es), {args.repeats} rounds per group\n")

    rng = random.Random(args.seed + 7)
    build_cache = {}
    trace = []
    adopted = []
    runs = 0

    for pass_index in range(args.max_passes):
        improved = False
        print(f"===== pass {pass_index + 1}/{args.max_passes} =====")
        for group in groups:
            candidates = group_candidates(registry, group, current)
            kept, skipped = filter_legal(registry, target, candidates, args, build_cache)
            entry = {
                "pass": pass_index + 1,
                "group": group.name,
                "considered": len(candidates),
                "legal": len(kept),
                "rejected": skipped,
                "adopted": None,
            }
            if not kept:
                print(f"[{group.name}] no legal alternative to the current value, skipping")
                trace.append(entry)
                continue

            reference = Candidate(dict(current), {}, "current")
            arms = [reference] + kept
            print(f"[{group.name}] timing {len(kept)} legal alternative(s), "
                  f"{len(skipped)} rejected")
            samples, failures = measure_rounds(
                target, arms, args, args.workdir,
                _timing_progress(group.name, args.repeats),
            )
            runs += args.repeats * len(arms)

            ref_samples = samples.get("current", [])
            if len(ref_samples) < 2:
                print(f"[{group.name}] the current config produced too few timings, skipping")
                entry["error"] = "reference produced too few samples"
                trace.append(entry)
                continue

            ref_median = statistics.median(ref_samples)
            min_effect_ns = args.min_effect_pct / 100.0 * ref_median
            confidence = corrected_confidence(
                args.confidence, len(kept), not args.no_correction)

            scored = []
            for candidate in kept:
                candidate_samples = samples.get(candidate.label, [])
                if len(candidate_samples) < 2:
                    status, reason = failures.get(candidate.label, ("fail", "no samples"))
                    scored.append({"label": candidate.label, "verdict": status,
                                   "reason": reason, "delta_pct": None})
                    continue
                verdict, delta, (low, high) = decide(
                    candidate_samples, ref_samples, min_effect_ns, args, rng, confidence)
                scored.append({
                    "label": candidate.label,
                    "overrides": candidate.overrides,
                    "verdict": verdict,
                    "median_ns": statistics.median(candidate_samples),
                    "delta_ns": delta,
                    "delta_pct": delta / ref_median * 100,
                    "ci_low_ns": low,
                    "ci_high_ns": high,
                })
            entry["results"] = scored
            entry["reference_median_ns"] = ref_median

            winners = sorted(
                (item for item in scored if item["verdict"] == "faster"),
                key=lambda item: item["delta_ns"],
            )
            if not winners:
                print(f"[{group.name}] nothing beat the current config")
                trace.append(entry)
                continue

            best_label = winners[0]["label"]
            best = next(c for c in kept if c.label == best_label)

            if args.confirm_rounds > 0:
                print(f"  [confirm] re-measuring {best.label} over "
                      f"{args.confirm_rounds} fresh round(s)")
                confirm_args = argparse.Namespace(**vars(args))
                confirm_args.repeats = args.confirm_rounds
                confirm_args.seed = args.seed + 1000 + pass_index
                confirm_samples, _ = measure_rounds(
                    target, [reference, best], confirm_args, args.workdir,
                    _timing_progress("confirm", args.confirm_rounds),
                )
                runs += args.confirm_rounds * 2
                confirm_ref = confirm_samples.get("current", [])
                confirm_best = confirm_samples.get(best.label, [])
                if len(confirm_ref) < 2 or len(confirm_best) < 2:
                    confirm_verdict = "insufficient"
                else:
                    confirm_verdict, _, _ = decide(
                        confirm_best, confirm_ref, min_effect_ns, confirm_args, rng)
                if confirm_verdict != "faster":
                    print(f"  [confirm] {best.label} did not win twice "
                          f"({confirm_verdict}); not adopting")
                    entry["unconfirmed"] = best.label
                    trace.append(entry)
                    continue

            if not args.no_correctness_gate:
                ok, reason = run_correctness(target, best.values, args)
                if ok is None:
                    print(f"  [correctness] skipped: {reason}")
                elif not ok:
                    print(f"  [correctness] {best.label} FAILED: {reason}")
                    print(f"  [correctness] not adopting a schedule that computes "
                          f"the wrong answer")
                    entry["correctness_failed"] = reason
                    trace.append(entry)
                    continue
                else:
                    print(f"  [correctness] {best.label} ok")

            current = dict(best.values)
            improved = True
            entry["adopted"] = best.overrides
            adopted.append({
                "pass": pass_index + 1,
                "group": group.name,
                "overrides": best.overrides,
                "delta_pct": winners[0]["delta_pct"],
            })
            print(f"[{group.name}] ADOPTED {best.label} "
                  f"({winners[0]['delta_pct']:+.1f}% against the config before it)")
            trace.append(entry)

        if not improved:
            print(f"\n[search] pass {pass_index + 1} adopted nothing; converged")
            break
    else:
        print(f"\n[search] hit the {args.max_passes}-pass limit; "
              f"there may be more to find")

    final = None
    if adopted and args.final_rounds > 0:
        print(f"\n[search] final head-to-head: original baseline vs searched config, "
              f"{args.final_rounds} rounds each")
        final = head_to_head(target, origin, current, args, rng)

    changed = {name: value for name, value in current.items() if origin.get(name) != value}
    payload = {
        "namespace": registry.namespace,
        "origin": origin,
        "final_config": current,
        "changed": changed,
        "adopted": adopted,
        "head_to_head": final,
        "passes": len({entry["pass"] for entry in trace}),
        "timed_runs": runs,
        "min_effect_pct": args.min_effect_pct,
        "confidence": args.confidence,
        "repeats": args.repeats,
        "trace": trace,
    }

    print_search_summary(payload)

    if args.preset_out and changed:
        meta = {
            "adopted": adopted,
            "head_to_head": final,
            "min_effect_pct": args.min_effect_pct,
        }
        save_preset(args.preset_out, target, current, meta)
        print(f"\n[search] wrote preset to {args.preset_out}")
        print(f"  reuse it with: DAE_TUNE_CONFIG={args.preset_out} "
              f"python {target['script']}")
    elif args.preset_out:
        print(f"\n[search] nothing changed, so no preset written to {args.preset_out}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[search] wrote the full trace to {args.out}")
    return 0


def print_search_summary(payload):
    changed = payload["changed"]
    print("\n" + "=" * 62)
    if not changed:
        print("Search finished without changing anything.")
        print("Every group was already at the best value the objective could "
              "distinguish, so the starting schedule stands.")
        return

    print(f"Search changed {len(changed)} knob(s) over {payload['passes']} pass(es), "
          f"{payload['timed_runs']} timed runs:")
    for step in payload["adopted"]:
        overrides = ", ".join(f"{k}={v}" for k, v in step["overrides"].items())
        print(f"  pass {step['pass']}  {step['group']:14} {overrides:36} "
              f"{step['delta_pct']:+.1f}%")

    final = payload.get("head_to_head")
    if not final:
        print("\nNo head-to-head was run, so the combined effect is unverified.")
        return
    if final.get("verdict") == "insufficient":
        print("\nThe head-to-head did not collect enough samples to judge the "
              "combined effect.")
        return

    print(f"\nHead-to-head over {final['rounds']} fresh rounds each:")
    print(f"  original baseline  {final['origin_median_ns'] / 1e6:.3f} ms")
    print(f"  searched config    {final['searched_median_ns'] / 1e6:.3f} ms")
    print(f"  difference         {final['delta_pct']:+.1f}%  "
          f"[{final['ci_low_ns'] / 1e6:+.3f}, {final['ci_high_ns'] / 1e6:+.3f}] ms  "
          f"-> {final['verdict']}")
    if final["verdict"] == "faster":
        print("\nThe searched schedule beats the original end to end.")
    else:
        print("\nWARNING: the individual steps won against the config that "
              "preceded them, but the combined result does not beat the "
              "original baseline on a fresh measurement. Treat the steps as "
              "noise that survived, not as a result.")


def cmd_report(args):
    with open(args.results, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "baseline_stats" in payload:
        print_timing_summary(payload)
    else:
        print_summary(payload)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="VDCores schedule autotuner driver")
    parser.add_argument("--workdir", default=REPO_ROOT, help="Repo root to run targets from")
    sub = parser.add_subparsers(dest="command", required=True)

    discover = sub.add_parser("discover", help="Dump a target's knob registry")
    discover.add_argument("--target", default="qwen3_1p7b")
    discover.add_argument("--script", help="Ad-hoc target script instead of --target")
    discover.add_argument("--namespace", help="Namespace for an ad-hoc --script target")
    discover.add_argument("--dry-build-arg", action="append", help="Extra target argument")
    discover.add_argument("-o", "--out", required=True, help="Where to write the registry")
    discover.add_argument("--timeout", type=float, default=600.0)
    discover.set_defaults(func=cmd_discover)

    enumerate_cmd = sub.add_parser("enumerate", help="List candidates without running them")
    enumerate_cmd.add_argument("--knobs", required=True)
    enumerate_cmd.add_argument("--knob", action="append", help="Restrict to these knobs")
    enumerate_cmd.set_defaults(func=cmd_enumerate)

    check = sub.add_parser("check", help="Filter candidates by static rules and dry-build")
    check.add_argument("--knobs", required=True)
    check.add_argument("--knob", action="append", help="Restrict to these knobs")
    check.add_argument("--script", help="Override the target script")
    check.add_argument("--dry-build-arg", action="append")
    check.add_argument("--static-only", action="store_true",
                       help="Skip the dry-build subprocess; needs no GPU")
    check.add_argument("--max", type=int, help="Stop after this many candidates")
    check.add_argument("--timeout", type=float, default=600.0)
    check.add_argument("-o", "--out")
    check.add_argument("--preset",
                       help="Filter against this saved config instead of the defaults")
    check.set_defaults(func=cmd_check)

    def add_timing_args(parser_):
        parser_.add_argument("--knobs", required=True)
        parser_.add_argument("--script", help="Override the target script")
        parser_.add_argument("--repeats", type=int, default=5,
                             help="Fresh processes per candidate")
        parser_.add_argument("--iterations", type=int, default=20,
                             help="In-process bench iterations, passed as -b")
        parser_.add_argument("--warmup", type=int, default=1,
                             help="DAE_BENCH_WARMUP launches before timing")
        parser_.add_argument("--seed", type=int, default=0,
                             help="Seed for the per-round shuffle")
        parser_.add_argument("--kill-stale", action="store_true",
                             help="pkill leftover runs of this target between measurements")
        parser_.add_argument("--post-launch-timeout", type=float, default=120.0)
        parser_.add_argument("--idle-timeout", type=float, default=30.0)
        parser_.add_argument("--hard-timeout", type=float, default=900.0)
        parser_.add_argument("-o", "--out")

    def add_objective_args(parser_):
        parser_.add_argument("--min-effect-pct", type=float, default=1.0,
                             help="Smallest improvement worth acting on, as a percent")
        parser_.add_argument("--bootstrap", type=int, default=2000,
                             help="Resamples used for the confidence interval")
        parser_.add_argument("--confidence", type=float, default=0.95,
                             help="Sweep-level confidence, 0-1")
        parser_.add_argument("--no-correction", action="store_true",
                             help="Do not spread the error budget across candidates")
        parser_.add_argument("--confirm-rounds", type=int, default=10,
                             help="Fresh rounds used to re-test winners; 0 disables")
        parser_.add_argument("--build-timeout", type=float, default=600.0)
        parser_.add_argument("--no-prebuild", action="store_true",
                             help="Skip the dry-build prefilter before timing")
        parser_.add_argument("--dry-build-arg", action="append")
        parser_.add_argument("--preset",
                             help="Start from this saved config instead of the defaults")

    noise = sub.add_parser(
        "noise", help="Measure the baseline repeatedly and report the spread")
    add_timing_args(noise)
    noise.add_argument("--preset",
                       help="Measure this saved config instead of the defaults")
    noise.set_defaults(func=cmd_noise, dry_build_arg=None)

    measure = sub.add_parser("measure", help="Time candidates against the baseline")
    add_timing_args(measure)
    add_objective_args(measure)
    measure.add_argument("--knob", action="append", help="Restrict to these knobs")
    measure.add_argument("--max", type=int, help="Stop after this many candidates")
    measure.set_defaults(func=cmd_measure)

    search = sub.add_parser(
        "search",
        help="Coordinate descent over knob groups, moving SM count and base SM together",
    )
    add_timing_args(search)
    add_objective_args(search)
    search.add_argument("--group", action="append",
                        help="Restrict the search to these groups (stage name or knob name)")
    search.add_argument("--max-passes", type=int, default=3,
                        help="Stop after this many sweeps over the groups")
    search.add_argument("--final-rounds", type=int, default=10,
                        help="Rounds for the original-vs-searched head-to-head; 0 disables")
    search.add_argument("--preset-out",
                        help="Write the winning configuration here as a reusable config")
    search.add_argument("--no-correctness-gate", action="store_true",
                        help="Adopt steps without verifying the schedule still computes "
                             "the right answer")
    search.add_argument("--correctness-arg", action="append",
                        help="Override the target's correctness invocation")
    search.add_argument("--correctness-timeout", type=float, default=1800.0)
    search.set_defaults(func=cmd_search)

    report = sub.add_parser("report", help="Print a saved results file")
    report.add_argument("results")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
