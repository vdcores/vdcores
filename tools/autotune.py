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
import subprocess
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TARGETS = {
    "qwen3_1p7b": {
        "namespace": "qwen3_1p7b",
        "script": "app/python/qwen3_1p7b/sched.py",
        "dry_build_args": ["--dry-build"],
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


def cmd_check(args):
    registry = KnobRegistry.from_file(args.knobs)
    target = {
        "namespace": registry.namespace,
        "script": args.script or TARGETS.get(registry.namespace, {}).get("script"),
        "dry_build_args": args.dry_build_arg or
                          TARGETS.get(registry.namespace, {}).get("dry_build_args", []),
    }
    if not target["script"]:
        raise SystemExit(
            f"no script known for namespace {registry.namespace!r}; pass --script"
        )

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


def cmd_report(args):
    with open(args.results, "r", encoding="utf-8") as handle:
        print_summary(json.load(handle))
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
    check.set_defaults(func=cmd_check)

    report = sub.add_parser("report", help="Print a saved results file")
    report.add_argument("results")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
