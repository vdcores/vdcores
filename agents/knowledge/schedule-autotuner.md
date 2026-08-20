# Schedule Autotuner

A driver that searches VDCores schedule parameters (SM placement, fold) instead
of tuning them by hand. Targets `qwen3_1p7b` and `llama3_8b`.

**Result:** on Llama-3.1-8B the search beat the hand-tuned schedule by 1.6%
(`silu.sms` 4 -> 2), independently verified at -1.74% with non-overlapping
distributions. See [Result](#result).

All five milestones are done: knob layer, legality filtering, noise-aware
timing, coordinate-descent search with a correctness gate, and the Llama3 port.

## Knob Configuration Layer

[python/dae/tune.py](../../python/dae/tune.py). Stdlib-only on purpose: it imports
without CUDA, PyTorch, or the built `dae.runtime`, so a driver or test host can
read knob metadata. Schedules declare knobs through `tune.load("<namespace>")`
then `sms(...)`, `base_sm(...)`, `int_knob(...)`, `bool_knob(...)`,
`str_set_knob(...)`.

Resolution order, highest first:

1. `DAE_TUNE_SET="q_proj.sms=64,down_proj.sms=96"` inline overrides
2. the knob's legacy env var, e.g. `QWEN1P7B_QPROJ_SMS`
3. the JSON file named by `DAE_TUNE_CONFIG`
4. the default declared in the schedule

A legacy env var shadowing a config-file value warns on stderr, because that is
usually a stale shell export.

`DAE_TUNE_DUMP=path.json` writes the resolved config plus knob specs at process
exit via an `atexit` hook -- needed because knobs are declared lazily while the
schedule builds. That dump is the registry the driver reads, so adding a
tunable schedule never means editing the driver. It round-trips: both a full
dump and a flat `{"knob": value}` mapping are accepted as config input.

**Non-goals.** The layer does not validate legality; that lives in
`Schedule.place()` -> `_on_place()` -> `validate()` in
[python/dae/schedule.py](../../python/dae/schedule.py). A knob's `choices` are search
candidates, not a legality claim. Unknown config keys warn rather than error.

## Knob Surfaces

**Qwen3 1.7B** -- 23 knobs, namespace `qwen3_1p7b`: `<stage>.sms` and
`<stage>.base_sm` for the ten GEMV/SiLU stages, plus `logits.split_m`,
`mlp.low`, and `no_prefetch`.

**Llama3 8B** -- 33 knobs, namespace `llama3_8b`: `<stage>.sms` and
`<stage>.base_sm` for 16 stages, plus `logits.split_m`.

All defaults reproduce the previously hardcoded `place(...)` arguments exactly,
and the `QWEN1P7B_*` env vars still work as legacy aliases, so the sweeps in
[qwen3-1p7b-scheduling.md](qwen3-1p7b-scheduling.md) remain reproducible.

### Left Untunable On Purpose

- `Gqa` derives placement from `N * NUM_KV_HEAD`; `argmax` is fixed at the SM
  count its atom encodes (`ARGMAX_PARTIAL_bf16_1152_50688_132` on Qwen3,
  `..._1024_65536_128` on Llama3). The `*_rms` stages follow `rms_sms`, and
  `clear_*` / `restore_bars_*` sit on fixed spare SMs.
- Llama3's MLP split constants (4096, 2048, 6144, 8192) are coupled across nine
  stages. Unlike Qwen3's single `mlp.low` there is no one value to turn, so the
  split is left alone.
- Llama3's `logits_slice` stays at `8192 * logits_fold` because its argmax atom
  bakes in 65536. Only the `split_M` fold is exposed.
- `no_prefetch` is Qwen3-only; Llama3 has no `maybe_no_prefetch` helper.

## Driver

[tools/autotune.py](../../tools/autotune.py). Subcommands: `discover`, `enumerate`,
`check`, `noise`, `measure`, `search`, `report`.

```bash
python tools/autotune.py discover --target llama3_8b -o tuning/llama3_8b.knobs.json
python tools/autotune.py check    --knobs tuning/llama3_8b.knobs.json -o tuning/llama3_8b.legality.json
python tools/autotune.py search   --knobs tuning/llama3_8b.knobs.json \
    --repeats 8 --min-effect-pct 1.0 --preset-out tuning/llama3_8b.preset.json
```

Two legality filters, cheapest first:

- **static** -- arithmetic on candidate values, no subprocess: SM count
  positive, `base_sm + sms <= full_sms`, `mlp.low` inside the layer. Each rule
  is skipped when the note it needs is absent, so the driver never invents a
  device geometry. `check --static-only` needs no GPU.
- **dry-build** -- run the target with `--dry-build` under `DAE_TUNE_CONFIG`
  (~2s, against ~14s for a profiled run). The rejection reason is lifted from
  the target's own `AssertionError`, so the report explains why in the
  schedule's own words.

`check` exits non-zero if the baseline itself fails to build.

### Legality Is The Binding Constraint

| Target | Buildable | Static-rejected | Build-rejected |
| --- | --- | --- | --- |
| Qwen3 1.7B | 41 / 115 | 29 | 45 |
| Llama3 8B | 45 / 159 | 50 | 64 |

Freedom is very unevenly distributed. On Llama3 the non-GEMV stages
(`k_rope`, `q_rope`, `silu_fused`) accept 19 of 31 combinations; the GEMV
stages are boxed in by the fold rules, and `gate_fused`, `up_fused` and
`logits.split_m` have **no** legal alternative at all.

### The Legality Map Is Conditional On The Baseline

A one-knob sweep reports what is legal *given the other knobs sit at baseline*.
`up_low.sms=128` is rejected only because the baseline pins
`up_low.base_sm=64`; with `base_sm=0` it is legal. On Llama3, ten knobs have
`base_sm` legal only at 0, and all ten belong to stages at 128 SMs where any
shift runs off a 132-SM device.

Measured: moving `sms` and `base_sm` together reaches **113 legal
configurations against 44** for single-knob sweeps (2.6x), expanding 12 of 16
stages. This is why `search` moves knobs in groups.

## Search

Coordinate descent over knob **groups**, not knobs. A stage's `.sms` and
`.base_sm` are one group; every other knob is a group of one. Each group is
optimized against the configuration reached so far, not the original baseline.
`--group` restricts the search to named groups.

A step is adopted only if it clears three bars:

1. wins the noise-aware test `measure` uses, with the error budget spread
   across that group's candidates;
2. wins **again** on separately collected samples (`--confirm-rounds`);
3. passes the **correctness gate**.

**Correctness gate.** Timing cannot tell a fast schedule from a fast *wrong*
one. Before adopting, the driver runs the target's own correctness check under
the candidate and refuses the step if it fails. The invocation comes from the
target entry (`correctness_args`) or `--correctness-arg`; a target declaring no
check reports that rather than silently passing. `--no-correctness-gate` opts
out.

**Head-to-head.** Greedy descent adopts each step against its predecessor, so a
chain of justified steps can add up to less than the sum of its parts. After
convergence the original baseline is timed directly against the final config on
fresh samples. If that does not come out `faster`, the summary says so and tells
the reader to treat the steps as noise that survived.

**Presets.** `--preset-out` writes the winning config in the format
`DAE_TUNE_CONFIG` already accepts, with a `search` provenance block that
`tune._load_config_file` ignores. The same artifact re-runs the tuned schedule
and seeds the next search via `--preset`, which `check`, `noise` and `measure`
also accept.

## Measurement

Run `noise` before trusting anything: it times the baseline repeatedly and
suggests a `--min-effect-pct`. If the spread is wider than the effect being
searched for, no amount of searching will find a real winner.

| Target | Run-to-run IQR | Per profiled run |
| --- | --- | --- |
| Llama3 8B (128-token decode) | **0.4%** | ~14s |
| Qwen3 1.7B (single token) | **1.8%** | ~20s |

Llama3 is the better target: each run averages 128 decode steps, so per-process
variance is proportionally smaller. **Qwen3 cannot be improved this way** --
its bench path only works at `-N 1`; longer decodes hit
`assert len(self.cinsts) <= ctensor.shape[0]` in `launcher.py`. Since the effect
found on Llama3 was 1.6%, below Qwen3's 1.8% floor, an equivalent win there
would be invisible without many more repeats.

Each measurement is a fresh process run through
[run_with_launch_timeout.py](../../tests/script/run_with_launch_timeout.py), so
a schedule that builds then deadlocks is recorded as `hang`. Candidates are
measured round-robin in a reshuffled order each round, so host drift spreads
across candidates instead of penalising whichever ran last.

### Why The Objective Is Not A Threshold

The first implementation compared medians against a threshold from the baseline
spread. It promoted a knob the fixture's cost model does not use at all, on a
-1.3% difference that a 60-run check put at +0.4%. Three things replaced it:

- **Bootstrap intervals** on the difference of medians. Few runs produce a wide
  interval rather than a confident wrong answer, and resampling survives the
  bimodal timing this host shows.
- **Multiple-comparison correction.** A 95% interval is wrong 1 time in 20, so
  a 40-candidate sweep expects ~2 winners that are noise. `--no-correction`
  opts out.
- **A confirmation pass.** Winners are re-measured on separate samples and
  reported only if they win twice.

A candidate is `faster` only when the interval excludes zero **and** the effect
clears `--min-effect-pct`. Statistical significance on a 0.1% effect is not a
reason to change a schedule. The cost is power: a real but small effect reads
`same` until enough rounds accumulate. That is the intended trade.

## Result

Llama3 8B on a GH200, 2 passes, **2056 timed runs**, ~8.5 hours.

```
Search changed 1 knob(s) over 2 pass(es):
  pass 1  silu   silu.sms=2, silu.base_sm=128   -1.5%

Head-to-head over 10 fresh rounds each:
  original baseline  598.388 ms
  searched config    588.848 ms
  difference         -1.6%  [-10.969, -7.165] ms  -> faster
```

Clean run: 0 out-of-memory rejections, 0 groups skipped for host reasons, 0
aborts, 226 legal candidates timed. Preset at `tuning/llama3_8b.preset.json`.

**Independently verified** outside the driver -- 6 interleaved rounds, fresh
process each, sequential on an idle GPU:

```
baseline  median 599.83 ms  (n=6, 598.0-603.2)
tuned     median 589.37 ms  (n=6, 588.6-592.3)
delta     -1.74%            tuned faster in 6/6 paired rounds
```

The distributions do **not overlap**: the slowest tuned run (592.3ms) beats the
fastest baseline run (598.0ms). The driver's own -1.59% is the conservative
number. Verify on an idle GPU one process at a time; two concurrent benchmarks
once produced a 23,437ms reading.

### Read It Honestly

- **Small but solid.** 1.6% against a 1.0% effect floor and 0.4% noise floor.
- **The win did not need paired knobs.** `silu.base_sm` never moved. Pairing
  made 69 extra configurations reachable and none of them won. The mechanism is
  validated; the premise is not yet paid off.
- **The hand-tuned schedule was mostly right.** 14 of 17 groups in pass 1, and
  every group in pass 2, found nothing better.
- **Not an optimum.** Coordinate descent over a hand-chosen grid at 1%
  resolution. The defensible claim is "no single-group move detectably improves
  on this schedule".
- The win came from `silu`, the smallest stage, not a large projection. On
  Qwen3 the same knob moves the *other* way (`silu.sms=1/2` were +40%/+56%
  slower), so intuition does not port between schedules.

### The Correctness Gate Caught A Faster, Wrong Schedule

`gate_high.sms=128` cleared the effect floor, won the statistical test, and won
the confirmation pass. It then failed the gate. Confirmed by three standalone
runs with a baseline control on an idle GPU:

```
gate_high.sms=128  run 1: FAIL logits_high 169.887% <= 10.000%
gate_high.sms=128  run 2: FAIL logits_high 160.618%
gate_high.sms=128  run 3: FAIL logits_high 169.844%
baseline (control): PASS
```

Deterministic, ~165% error, baseline passes under the identical invocation.
Every timing-based defence -- bootstrap intervals, correction, confirmation --
was fully convinced. Only running the model caught it. Without the gate this
run would have reported it as a second win.

## Building A Runnable Runtime

**A default `make pyext` build cannot launch these schedules.** It builds them
fine and then dies:

```
ValueError: Missing runtime opcode for op-family instruction
OP_GEMV_WGMMA__M_64__N_8__K_256__BLOAD_4__RESIDUAL_0
```

The default op selection is static-only (26 ops, 0 dynamic). The GEMV op family
is generated on demand, so the required set must be dumped and compiled in:

```bash
python app/python/llama3/sched.py --dry-build -w ops.txt   # 11 operators
DAE_COMPUTE_OPS_FILE=ops.txt make runtime.o
python setup.py build_ext --inplace
python -c "from dae.runtime import opcode; print([n for n in vars(opcode) if 'GEMV' in n])"
```

Qwen3 needs 9 ops, Llama3 needs 11, sharing 6 (they differ in hidden size, so a
different RMS norm, and in vocabulary, so a different argmax pair; Llama3 also
needs RoPE and a second SiLU). **Build with the union of 14** for one runtime
serving both. For the current knob surfaces this costs nothing at search time --
the op set is identical across all legal candidates -- but **re-check whenever a
knob surface or choice list changes**, or `measure` will report spurious
failures that are really missing opcodes.

`pip install -e .` **does not rebuild on generated-header changes.** setuptools
tracks `src/torch_runtime.cu`, not `build/generated/dae/*.inc`; it reports
"Successfully installed" while reusing the stale `.so`. Use
`python setup.py build_ext --inplace`, or clear `build/temp.*`, `build/lib.*`
and `python/dae/*.so` first.

`setup.sh` is not reproducible on a non-conda host. Two durable facts: CUTLASS
must be **4.x or newer** (`include/task/attention.cuh` includes
`cute/algorithm/tensor_reduce.hpp`, absent in 3.x), and `argmax.cuh`,
`attention.cuh` and `runtime.cuh` rely on `<cfloat>` and `<array>` arriving
transitively from CUTLASS 3.x, so on 4.x they need
`NVCC_PREPEND_FLAGS="-include cfloat -include array"` until those includes are
added properly.

## Llama3 Builds Without Weights

Llama-3.1-8B is gated and a legality sweep runs the target ~160 times, so
downloading 16GB per attempt was never an option.
[dry_build.py](../../app/python/llama3/dry_build.py) supplies synthetic
stand-ins with the same *attribute shape* the real `transformers` model exposes
(`model.model.layers[i].self_attn.q_proj.weight`, `model.model.rotary_emb`,
`model.lm_head.weight`). The rest of `sched.py` runs unchanged, so TMA
descriptors, barrier wiring and `place()` -> `validate()` give the *real*
legality answer; only the values are fake.

Deliberately not a restructure into a `runtime_context.py` the way Qwen3 is
organised: matching the attribute shape touches four places in the 1,170-line
script, extracting a context object would touch most of it for no extra
capability.

- `--dry-build` needs no token, no download, no network, and defaults to a
  1-step decode schedule (placement legality does not depend on decode count).
- It refuses `--prompt`/`--message`; there is no tokenizer.
- The compute-op dump (`-w`) works from a dry build, so a runnable runtime can
  be built before any weights exist.
- Credentials resolve from `HF_TOKEN` or a stored `hf auth login`. Without
  either, the script explains itself instead of raising `KeyError`.

[tests/test_llama3_dry_build.py](../../tests/test_llama3_dry_build.py) pins the
attribute-shape contract on CPU with a tiny geometry, and checks the stub
geometry against the published `config.json` when a credential is available --
the whole legality map was computed against those numbers.

## Traps

- **`--dry-build` cannot see launch-time failures.** A schedule can build and
  then die with `launch_dae failed: misaligned address`. Roughly **17% of
  schedules that pass both filters still deadlock or crash at launch**, so the
  build check overstates the runnable space. Not a correctness risk -- a
  candidate with no timing cannot be adopted -- but budget for it.
- **Llama3's correctness gate needs `-N 8`.** The default 128-step check
  compares the final decode position against a greedy reference and **fails on
  the unmodified baseline**: a sub-ulp difference eventually flips one greedy
  token and everything after is incomparable, giving ~100% errors. At `-N 8` all
  eight tokens match exactly. A gate that fails on the baseline makes `search`
  refuse everything and looks like "found nothing" rather than "gate is broken".
- **Do not set `HF_HOME` for gated targets.** `hf auth login` stores its token
  under `~/.cache/huggingface/`, and `HF_HOME` relocates that lookup, so the
  credential silently stops resolving. Schedules take their weight cache from
  `--hf-cache-dir`, which is independent.
- **Bare asserts hide reasons.** Four Llama3 build-rejections report only
  `AssertionError` because `SchedGemv.validate()` has asserts with no message
  (e.g. `assert K % self.fold == 0`). Adding messages would make the driver's
  report self-explaining.

### Infrastructure Failures Must Be Loud

A search whose headline output can be "we found nothing" must never let a broken
host produce that sentence. One run reported exactly that while the GPU was
full and nothing had executed. Four fixes, all with regression tests:

- The timeout wrapper reads on a **thread**. It previously mixed `selectors` on
  the OS pipe with buffered `readline()`, so a child flushing several lines at
  once left the remainder unread while the pipe looked idle -- stranding the
  wrapper before the launch marker was ever seen.
- Children spawn with `start_new_session=True` and are killed by **process
  group**. `subprocess.run(timeout=)` signals only the direct child, orphaning
  the real worker; five orphans at ~19GB filled a 94GB device.
- `filter_legal` raises `InfrastructureError` on host-reason failures
  (`INFRA_MARKERS`) instead of recording "illegal".
- `search` **aborts and exits 2** when the reference config cannot be timed,
  rather than continuing to a null result.

Two throughput fixes came from the same investigation: the driver now passes
`--startup-timeout` (default 120s, against a healthy run's ~9s to reach
`[bench]`) because a pre-launch deadlock otherwise burned the full 900s
`--hard-timeout` -- 45 hangs once consumed 11 of 12 hours -- and a hang
disqualifies a candidate on the **first** attempt (`--drop-after-hang`), since
across 172 runs every hanging candidate hung both times and none recovered.
Cost of a deadlocking candidate: 1800s -> 120s.

## Fixture And Testing

[tests/fake_sched.py](../../tests/fake_sched.py) is a GPU-free stand-in target
declaring the same knob surface and reimplementing the subset of
`SchedGemv.validate()` that knob values can violate. It reproduces the
documented result that `down_proj` is legal on 96 SMs and rejected on 128 with
`k_per_fold=1536`.

It also charges for contention: stages whose SM ranges overlap pay
`OVERLAP_NS_PER_SM` per shared SM, ~39% of its baseline time. Without that term
the fixture has no opinion about `base_sm` and a paired search would have
nothing to find. The constant is tuned so both effects are real, and so the best
paired `gate_high` move (-5.3%) beats the best single-knob one (-3.9%) -- which
is what lets the fixture tell a paired search apart from a coordinate one.
`FAKE_SCHED_WRONG_IF` makes a chosen config report wrong answers, so the
correctness gate is testable without a GPU.

```bash
python tests/test_tune.py              # 11 passed
python tests/test_autotune.py          # 54 passed
python tests/test_llama3_dry_build.py  # 10 passed
```

All three run on a host with no CUDA and no PyTorch extension built.

## Figures

[docs/autotune/](../../docs/autotune/) holds the result, funnel, reachability
and noise figures. They were generated from a `search` run trace, which is a
run artifact and not kept in the repo; `--out` writes a new one.
