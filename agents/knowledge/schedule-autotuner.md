# Schedule Autotuner

Work toward a driver that searches VDCores schedule parameters (SM placement,
fold, operator grouping, overlap boundary) instead of tuning them by hand.

## Milestones

1. Knob configuration layer, and port `app/python/qwen3_1p7b/sched.py` to it. **done**
2. `tools/autotune.py` with static + dry-build legality filtering only. **done**
3. Timed runs plus a noise-aware objective. **done**
4. Coordinate-descent search, presets, correctness gate.
5. Extend to `app/python/llama3/sched.py`, which first needs a `--dry-build` mode.

## Knob Configuration Layer

- Lives in [python/dae/tune.py](python/dae/tune.py).
- Stdlib-only on purpose. It imports without CUDA, PyTorch, or the built
  `dae.runtime` extension, so a driver or test host can read knob metadata.
- An app schedule declares knobs through `tune.load("<namespace>")` and then
  `config.sms(...)`, `config.base_sm(...)`, `config.int_knob(...)`,
  `config.bool_knob(...)`, `config.str_set_knob(...)`.
- Declaring a knob resolves its value, records where the value came from, and
  publishes the candidate search values for that knob.

### Resolution Order

Highest priority first:

1. `DAE_TUNE_SET="q_proj.sms=64,down_proj.sms=96"` inline overrides
2. the knob's legacy environment variable, such as `QWEN1P7B_QPROJ_SMS`
3. the JSON config file named by `DAE_TUNE_CONFIG`
4. the default declared in the schedule

A legacy environment variable that shadows a config-file value prints a warning
to stderr, because that combination is usually a stale shell export rather than
an intended override.

### Self-Describing Dump

`DAE_TUNE_DUMP=path.json` writes the resolved configuration plus the knob specs
at process exit, through an `atexit` hook registered by `tune.load(...)`. The
hook is needed because knobs are declared lazily while the schedule builds, so
the full registry only exists once the schedule has finished constructing.

That dump is the knob registry for the driver. The intended loop is:

```bash
DAE_TUNE_DUMP=/tmp/qwen_knobs.json python app/python/qwen3_1p7b/sched.py --dry-build
```

then edit copies of the `knobs` object and feed them back through
`DAE_TUNE_CONFIG`. The dump round-trips: a full dump and a flat
`{"knob": value}` mapping are both accepted as config input.

### Deliberate Non-Goals

- The layer does not validate that a knob value is legal. Legality lives in
  `Schedule.place()` -> `_on_place()` -> `validate()` in
  [python/dae/schedule.py](python/dae/schedule.py),
  and the driver is meant to discover legal values by trial construction. The
  `choices` on each knob are search candidates, not a legality claim.
- Config-file keys that no knob reads are reported as a warning, not an error.
  The driver generates configs from a dump, so typos should not normally occur.

## Qwen3 1.7B Knob Surface

`app/python/qwen3_1p7b/sched.py` declares 23 knobs under namespace `qwen3_1p7b`.

- `<stage>.sms` for `q_proj`, `k_proj`, `v_proj`, `out_proj`, `gate_low`,
  `gate_high`, `up_low`, `up_high`, `silu`, `down_proj`
- `<stage>.base_sm` for the same stages except `silu`, which has its own range
- `logits.split_m`
- `mlp.low`, the store-backed/reduce-backed MLP split point
- `no_prefetch`, the set of stages built without weight prefetch

All defaults reproduce the previous hand-tuned values, including the `base_sm`
arguments that used to be hardcoded inside the `place(...)` calls. The former
`QWEN1P7B_*` environment variables all still work as legacy aliases, so the
sweeps recorded in
[agents/knowledge/qwen3-1p7b-scheduling.md](agents/knowledge/qwen3-1p7b-scheduling.md)
remain reproducible.

### Stages Left Untunable On Purpose

- `Gqa` asserts an exact SM count in `SchedAttentionDecoding`, so its placement
  is derived, not tuned.
- `argmax` and `logits_proj` are placed on `full_sms` because the selected
  argmax atom `ARGMAX_PARTIAL_bf16_1152_50688_132` bakes in the 132-SM shape.
- `embed_rms`, `pre_attn_rms`, `post_attn_rms` follow `rms_sms`, and the
  `clear_*` and `restore_bars_*` helpers sit on fixed spare SMs.

## Testing

[tests/test_tune.py](tests/test_tune.py)
covers resolution order, dump round-trip, and the error cases. It loads
`tune.py` directly by file path rather than importing the `dae` package, so it
runs on a host with no CUDA and no PyTorch:

```bash
python tests/test_tune.py
```

## Driver

[tools/autotune.py](tools/autotune.py) answers "is this candidate buildable",
not yet "is it fast". Everything it knows about a target comes from that
target's knob dump, so adding a tunable schedule does not mean editing the
driver.

```bash
python tools/autotune.py discover --target qwen3_1p7b -o tuning/qwen3_1p7b.knobs.json
python tools/autotune.py check --knobs tuning/qwen3_1p7b.knobs.json -o tuning/qwen3_1p7b.legality.json
python tools/autotune.py report tuning/qwen3_1p7b.legality.json
```

Two filters, cheapest first:

- **static**: rules evaluated from candidate values alone, no subprocess.
  Currently SM count positive, `base_sm + sms <= full_sms`, and `mlp.low`
  inside the layer. Each rule is skipped when the note it needs is absent, so
  the driver never invents a device geometry it was not told about.
- **dry-build**: run the target with `--dry-build` under `DAE_TUNE_CONFIG` and
  read the exit status. The rejection reason is lifted out of the target's own
  `AssertionError`, so the report explains *why* in the schedule's own words.

`check --static-only` skips the subprocess entirely and needs no GPU.
`check` exits non-zero if the baseline itself fails to build, because every
other result in that run is then measured against a broken reference.

Candidates come from a coordinate sweep: one candidate per off-baseline choice,
varying a single knob at a time, so a rejection can be attributed to one knob.

## Legality Map Is Conditional On The Baseline

A one-knob-at-a-time sweep reports what is legal *given the other knobs sit at
baseline*, which is not the same as what is reachable overall. Concretely, in
the fixture run, `up_low.sms=128` is rejected only because the baseline puts
`up_low.base_sm=64`, so the range runs off the device. With
`up_low.base_sm=0` it is legal.

So the milestone 4 search cannot move one knob at a time for placement. SM
count and base SM of a stage have to move as a pair, or wide placements are
unreachable from the current baseline.

## Fixture

[tests/fake_sched.py](tests/fake_sched.py) is a stand-in target that declares
the same 23-knob surface and reimplements the subset of `SchedGemv.validate()`
that knob values can violate, for the Qwen3 1.7B geometry. It reproduces the
documented result that `down_proj` is legal on 96 SMs and rejected on 128 with
`k_per_fold=1536`, so the driver is exercised against real rejections on a host
with no GPU.

Sweeping all 23 knobs against the fixture produced 107 candidates, of which 40
build: 29 static-rejected and 38 build-rejected. Most knobs turn out to have
only one or two legal alternatives to their current value, which is a useful
early signal that the search space is far smaller than the raw knob count
suggests.

```bash
python tests/test_autotune.py
```

## Measurement

```bash
python tools/autotune.py noise   --knobs tuning/qwen3_1p7b.knobs.json --repeats 30
python tools/autotune.py measure --knobs tuning/qwen3_1p7b.knobs.json -o tuning/qwen3_1p7b.timing.json
```

Run `noise` first. It times the baseline repeatedly and reports the spread and
a suggested `--min-effect-pct`. If the spread is wider than the effects being
searched for, no amount of searching will find a real winner, and that has to
be fixed before tuning means anything.

Each measurement is a fresh process, run through
[tests/script/run_with_launch_timeout.py](tests/script/run_with_launch_timeout.py)
so a schedule that builds and then deadlocks is recorded as `hang` instead of
stalling the sweep. `--kill-stale` is available for the documented stale-worker
problem, but it `pkill`s only the target script path. A blanket `killall python`
would take the driver down with it.

Candidates are measured round-robin, one sample each per round, in an order
reshuffled every round. Measuring all repeats of one candidate before starting
the next would charge any slow drift in the host to whichever candidate
happened to run during it.

### Why The Objective Is Not A Threshold

The first implementation compared medians against a threshold derived from the
baseline spread. Against the fixture it promoted `silu.sms=1`, a knob the
fixture's cost model does not use at all, on a -1.3% difference. A direct
60-run-per-side check put the true difference at +0.4%, so that verdict was
pure luck.

Three things replaced it:

- **Bootstrap intervals.** The interval for the difference of medians is built
  by resampling, so a small number of runs produces a wide interval rather than
  a confident wrong answer. Resampling also survives the bimodal timing this
  host shows, which a mean-and-standard-deviation test would not.
- **Multiple-comparison correction.** A 95% interval is wrong one time in
  twenty by construction, so a 40-candidate sweep should be expected to crown
  about two winners that are noise. The error budget is spread across the
  comparisons. `--no-correction` opts out.
- **A confirmation pass.** Apparent winners are re-measured on separately
  collected samples and reported only if they win twice. Anything that does not
  survive is reported as `unconfirmed`.

A candidate is called `faster` only when the interval excludes zero *and* the
effect clears `--min-effect-pct`. Statistical significance on a 0.1% effect is
not a reason to change a schedule.

The cost of this is power: on a noisy host, a real but small effect will be
reported as `same` until enough rounds are collected. That is the intended
trade. The fixture's true -1.2% `gate_low` effect is detected immediately with
noise off, and correctly withheld at 15 rounds with 3% per-run noise.

### Fixed: Dropped Tail Output In The Timeout Wrapper

`run_with_launch_timeout.py` stopped reading as soon as the child exited, which
left buffered output unread. A fast-exiting run lost its final lines, and those
are the lines carrying the benchmark block and `[perf]`. It now drains the pipe
after the loop. This affected any use of the wrapper, not just the autotuner.
