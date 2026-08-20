# Schedule Autotuner

Work toward a driver that searches VDCores schedule parameters (SM placement,
fold, operator grouping, overlap boundary) instead of tuning them by hand.

## Milestones

1. Knob configuration layer, and port `app/python/qwen3_1p7b/sched.py` to it. **done**
2. `tools/autotune.py` with static + dry-build legality filtering only. **done**
3. Timed runs plus a noise-aware objective. **done**
4. Coordinate-descent search, presets, correctness gate. **done**
5. Extend to `app/python/llama3/sched.py`, which first needs a `--dry-build` mode. **done**

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

The real-hardware sweep confirmed this is the dominant limitation, not a corner
case. Of the 23 knobs, five have **no** legal alternative to their current value
when the others sit at baseline: `gate_low.sms`, `up_low.sms`, `logits.split_m`,
`mlp.low`, and `silu.base_sm`. Most of the rest have one or two. A single-knob
sweep therefore explores only the immediate neighbourhood of a schedule a human
already hand-tuned, which is close to a tautological test.

## Fixture

[tests/fake_sched.py](tests/fake_sched.py) is a stand-in target that declares
the same 23-knob surface and reimplements the subset of `SchedGemv.validate()`
that knob values can violate, for the Qwen3 1.7B geometry. It reproduces the
documented result that `down_proj` is legal on 96 SMs and rejected on 128 with
`k_per_fold=1536`, so the driver is exercised against real rejections on a host
with no GPU.

The fixture also charges for contention: stages whose SM ranges overlap pay
`OVERLAP_NS_PER_SM` per shared SM, which is about 39% of its baseline time.
Without that term the fake has no opinion about `base_sm` at all, and a search
over `(sms, base_sm)` pairs would have nothing to find. The constant is tuned so
that both effects are real: more SMs on a stage still helps, and de-overlapping
also helps. Critically, the best paired move for `gate_high` (`sms=32,
base_sm=96`, -5.3%) beats the best single-knob move (`sms=32`, -3.9%), so the
fixture can tell a paired search apart from a coordinate one.

Sweeping all 23 knobs against the fixture produced 107 candidates, of which 40
build: 29 static-rejected and 38 build-rejected. Most knobs turn out to have
only one or two legal alternatives to their current value, which is a useful
early signal that the search space is far smaller than the raw knob count
suggests.

```bash
python tests/test_autotune.py
```

## Llama3 8B Target

`app/python/llama3/sched.py` declares 33 knobs under namespace `llama3_8b` and
is registered in `tools/autotune.py` as target `llama3_8b`.

- `<stage>.sms` and `<stage>.base_sm` for `q_proj`, `q_rope`, `k_proj`,
  `k_rope`, `v_proj`, `out_proj`, `gate_low`, `gate_high`, `up_low`, `up_high`,
  `silu`, `gate_fused`, `up_fused`, `silu_fused`, `down_low`, `down_high`
- `logits.split_m`

All defaults reproduce the previously hardcoded `place(...)` arguments exactly.

### Building Without Weights

Unlike Qwen3, Llama-3.1-8B is **gated**: it needs `HF_TOKEN` set on an account
that has been granted access. A legality sweep runs the target ~160 times, so
downloading and loading 16GB per attempt was never an option.

[app/python/llama3/dry_build.py](app/python/llama3/dry_build.py) supplies
synthetic stand-ins with the same *attribute shape* the real `transformers`
model exposes -- `model.model.layers[i].self_attn.q_proj.weight`,
`model.model.rotary_emb`, `model.lm_head.weight`, and so on. The rest of
`sched.py` runs against them unchanged, so the TMA descriptors, barrier wiring,
and `place()` -> `validate()` produce the *real* legality answer while only the
values are fake.

This was deliberately not a restructure of the 1,170-line script into a
`runtime_context.py` the way Qwen3 is organized. Matching the attribute shape
touches four places in `sched.py`; extracting a context object would touch
most of it, for no additional capability.

Consequences worth knowing:

- `--dry-build` needs no token, no download, and no network.
- It refuses `--prompt`/`--message`, because there is no tokenizer to turn text
  into the token count the schedule is built around.
- It defaults to a **1-step** decode schedule. Placement legality does not
  depend on the decode step count, and the normal 128-step schedule takes far
  longer to construct.
- Without `--dry-build` and without `HF_TOKEN`, the script now exits with an
  explanation instead of a bare `KeyError` on `os.environ['HF_TOKEN']`.
- The compute-op dump works from a dry build too:
  `python app/python/llama3/sched.py --dry-build -w ops.txt` writes 11
  operators, so a runnable runtime can be built before any weights exist.

[tests/test_llama3_dry_build.py](tests/test_llama3_dry_build.py) pins the
attribute-shape contract on CPU with a tiny geometry, so it needs neither CUDA
nor a download:

```bash
python tests/test_llama3_dry_build.py
```

### Legality Map

45 of 159 candidates build: 50 static-rejected, 64 build-rejected.

Two things stand out against Qwen3:

- **The paired-move problem is worse here.** Ten knobs have `base_sm` legal
  *only* at 0, and all ten belong to stages sitting at 128 SMs, where any shift
  runs off a 132-SM device. Their `base_sm` is not really pinned; it is pinned
  *given* `sms=128`. Drop the stage to 64 SMs and 32 or 64 become reachable.
  A single-knob sweep cannot see that, which is exactly what `search` is for.
- **The non-GEMV stages are far freer.** `q_rope.sms` and `silu_fused.sms`
  accept every one of their eight candidate values, because they are not bound
  by the GEMV fold rules that reject most SM counts elsewhere.

### Left Untunable On Purpose

- The MLP split constants (`4096`, `2048`, `6144`, `8192`, `mlp_split`) are
  structurally coupled across `gate_proj_low/high`, `up_proj_low/high`,
  `gate_proj_fused`, `up_proj_fused`, `silu_fused`, and `down_proj_low/high`.
  Unlike Qwen3's single `mlp.low`, there is no one value to turn here, so the
  split is left alone rather than exposed as a knob that cannot be moved safely.
- `logits_slice` stays at `8192 * logits_fold`, because the selected argmax atom
  `ARGMAX_PARTIAL_bf16_1024_65536_128` bakes in that 65536. Only the `split_M`
  fold is exposed, as `logits.split_m`.
- `Gqa` derives its placement from `N * NUM_KV_HEAD`, `Argmax` is fixed at the
  128 its atom encodes, the `*_rms` stages follow `rms_sms`, and `copy_hidden`,
  `clear_*`, and `restore_bars_*` sit on fixed spare SMs.
- `no_prefetch` is a Qwen3 knob only. Llama3 has no `maybe_no_prefetch` helper,
  and adding one is a separate change.

### Verified On Real Hardware

Run on the GH200 with a Hugging Face credential in place.

| Check | Result |
| --- | --- |
| stub geometry vs published `config.json` | all 8 fields match |
| one timed run, `-N 1 -b 10` | 4.577 ms median, 218 tok/s |
| one timed run, default 128-step decode | 600 ms median, TBT 4.72 ms, 212 tok/s, **12s wall per run** |
| `noise --repeats 12 --iterations 10` | IQR **0.4%** of median, range 2.0%, drift negligible |
| correctness gate through the driver | passes |
| qwen3 after the shared rebuild | 1.524 ms, unchanged within noise |

**Llama3 is the better tuning target of the two.** Its noise floor is 0.4%
against Qwen3's 1.8%, because a run averages 128 decode steps instead of
timing a single token. Effects around 1% are detectable here that are invisible
on Qwen3. The default workload also lands at 12s per profiled run, which is the
budget the original proposal asked for, so no workload shaping was needed.

### The Correctness Gate Needs `-N 8`, Not The Default

The default `--correctness` run compares the **final** decode position against a
greedy reference over 128 steps, and it **fails on the unmodified baseline**:

```
[correctness] FAIL v_proj: 141.261% <= 5.000%
[correctness] FAIL final_hidden: 98.839% <= 5.000%
[correctness] FAIL logits_low: 102.385% <= 10.000%
```

Errors near 100% are not numerical drift, they are two runs decoding different
sequences. A sub-ulp difference eventually flips one greedy token, and
everything after that is incomparable. At `-N 8` all eight generated tokens
match exactly, so the schedule is fine and the check is simply not meaningful at
128 steps.

The target entry therefore pins `["--correctness", "-N", "8"]`. This matters
more than it sounds: a gate that fails on the baseline would make `search`
refuse to adopt *anything*, and it would look like "the search found nothing"
rather than "the gate is broken".

### The Dry-Build Blind Spot Is Real, Not Theoretical

A schedule can pass `--dry-build` and still die on launch. Observed on the
first Llama3 search:

```
DAE_TUNE_SET="down_high.sms=64,down_high.base_sm=32" python app/python/llama3/sched.py -b 3
RuntimeError: launch_dae failed: misaligned address
```

`--dry-build` constructs the schedule but never launches it, so an alignment
constraint that only bites the real launch is invisible to `check`. Two of the
three legal `down_high` placements fail this way.

This is not a correctness risk for the search -- a candidate that never
produces a timing cannot be adopted -- but it has two consequences that were
worth fixing before spending a night on it:

1. **`classify_failure` reported nothing useful.** A c10 stack frame like
   `c10::Error::Error(c10::SourceLocation, ...)` contains the substring
   `Error:`, so it outranked the real message and every runtime failure in the
   sweep read as `frame #0: c10::Error::Error(...)`. Stack frames are now
   dropped before the search for an informative line.
2. **Failing candidates were retried every round.** Eight rounds against a
   schedule that crashes every time buys nothing. `measure_rounds` now stops
   retrying after `--drop-after` failures (default 2, so a transient failure
   still gets a second chance) when the candidate has produced no timing at
   all. The reference arm is never dropped. On the observed failure rate this
   is roughly an hour saved per pass.

### How A Whole Run Was Lost, And What Changed

The first Llama3 search ran to completion, converged on pass 1, and reported:

```
Search finished without changing anything.
Every group was already at the best value the objective could distinguish.
```

That was false. The host had run out of device memory. Five orphaned workers,
each holding ~19GB, filled the 94GB GPU; every later run failed
`torch.OutOfMemoryError`, every later dry build failed the same way, and the
driver dutifully reported each group as "no legal alternative to the current
value". **A broken run is indistinguishable from a real null result unless the
driver goes out of its way to say so.**

The chain, from root cause outward:

1. **The timeout wrapper stalled on buffered output.**
   `run_with_launch_timeout.py` used `selectors` on the child's pipe but
   `readline()` through a userspace buffer. A child that flushes several lines
   at once delivers them as one chunk: the first `readline()` consumes the
   chunk, the rest sit in the buffer, and the pipe then looks idle to the
   selector. The wrapper waited forever on a child that had already spoken,
   before the launch pattern was seen -- so no post-launch timeout applied.
   Replaced with a reader thread, which has no such interaction.
2. **Killing the wrapper orphaned the worker.** The driver's `hard_timeout`
   (900s, matching the observed 900s spacing between orphans) killed the
   wrapper via `subprocess.run(timeout=...)`, which signals only the direct
   child. Children are now spawned with `start_new_session=True` and the whole
   process group is signalled, in both the wrapper and `run_child`.
3. **An out-of-memory host was reported as a legality result.** `filter_legal`
   now raises `InfrastructureError` when a dry build fails for a host reason
   (`INFRA_MARKERS`), instead of recording "illegal".
4. **A dead reference was treated as a skip.** If the current configuration
   cannot be timed, there is nothing to measure against, so `search` now aborts
   rather than continuing and reporting a null result. It exits 2 and prints
   what to check.

The lesson worth keeping: for a search whose headline output is "we found
nothing", every infrastructure failure must be loud, because the quiet version
of that failure looks exactly like the answer.

### Hangs Dominated The Budget Until The Startup Timeout Was Set

The second Llama3 search ran cleanly -- no orphans, no out-of-memory, the
infrastructure fixes held -- and still got through only **6 of 17 groups in 12
hours**. The accounting:

| | |
| --- | --- |
| timed runs attempted | 172 |
| produced a timing | 126, at ~14s each = 0.5h |
| hung | 45 |
| a healthy run reaches `[bench]` in | ~9s |

`run_with_launch_timeout.py` takes `--startup-timeout`, but it defaults to
`None` and the driver never passed it. So the wrapper's timeouts only applied
*after* the launch marker appeared. A schedule that deadlocks **before** that --
during construction or launch setup -- was not caught by the wrapper at all,
and ran until the driver's `--hard-timeout` of 900s. 45 x 900s is 11.25 hours,
which is essentially the whole run.

Two changes:

- The driver now passes `--startup-timeout` (default 120s). A healthy Llama3
  run reaches the marker in ~9s, so that is more than an order of magnitude of
  slack, and it turns a 900s hang into a 120s one.
- A hang now disqualifies a candidate on the **first** attempt
  (`--drop-after-hang`, default 1), while other failures still get the second
  chance `--drop-after` allows. The evidence: across those 172 runs every
  hanging candidate hung on *both* attempts and not one recovered. Deadlock is
  a property of the schedule, not luck, so the second attempt only pays the
  timeout again.

Together these cut the cost of a deadlocking candidate from 1800s to 120s, a
15x reduction on the dominant term.

Also worth knowing: the hangs are concentrated in `k_proj` and `k_rope`, and
never hit the reference configuration. They are real deadlocking schedules, not
flakiness, and the search handles them correctly -- it was only ever a question
of how long it waited to find out.

### One Environment Trap

Do not set `HF_HOME` when running a gated target. `hf auth login` stores its
token under `~/.cache/huggingface/`, and `HF_HOME` relocates that lookup, so the
credential silently stops resolving and the run fails as though the account had
no access. The schedules take their weight cache from `--hf-cache-dir`, which is
independent, so `HF_HOME` is never needed here.

## Building A Runnable Runtime

Structural, and easy to lose a day to. A default `make pyext` build produces a
runtime that **builds** every schedule but **cannot launch** the qwen3_1p7b one:

```
ValueError: Missing runtime opcode for op-family instruction
OP_GEMV_WGMMA__M_64__N_8__K_256__BLOAD_4__RESIDUAL_0
```

The default op selection is static-only (26 ops, 0 dynamic). The op family the
GEMV stages need is generated on demand, so the required set has to be dumped
from the schedule first and compiled in:

```bash
python app/python/qwen3_1p7b/sched.py --dry-build -w ops.txt   # 9 operators
DAE_COMPUTE_OPS_FILE=ops.txt make runtime.o
python setup.py build_ext --inplace
```

Two traps worth knowing:

- **`--dry-build` does not catch this.** It constructs the schedule but never
  encodes instructions, so the missing opcode only surfaces at launch. The
  driver's legality filter is therefore blind to it by construction: a
  candidate can pass `check` and still fail under `measure`.
- **`pip install -e .` does not rebuild on generated-header changes.** setuptools
  tracks `src/torch_runtime.cu`, not `build/generated/dae/*.inc`. After
  regenerating the op set it reports "Successfully installed" while reusing the
  stale object and `.so`. Use `python setup.py build_ext --inplace`, or clear
  `build/temp.*`, `build/lib.*`, and `python/dae/*.so` first. Confirm with:

  ```bash
  python -c "from dae.runtime import opcode; print([n for n in vars(opcode) if 'GEMV' in n])"
  ```

Qwen3 needs 9 operators and Llama3 needs 11, sharing only 6: the two models
differ in hidden size (so a different RMS norm) and vocabulary (so a different
argmax pair), and Llama3 additionally needs RoPE and a second SiLU. Building
with the **union of 14** produces one runtime that runs both, which is what is
installed here.

For the current knob surface this costs nothing at search time: the union of
required ops across all 41 legal Qwen3 candidates is identical to the
baseline's 9.
The knob that would have changed it, `mlp.low`, drives
`OP_SILU_MUL_SHARED_BF16_K_4096_INTER`, but every non-default `mlp.low` value is
rejected as illegal anyway. **Re-check this whenever the knob surface or a
choice list changes**, or `measure` will start reporting spurious failures that
are really missing opcodes.

The environment itself is not reproducible from `setup.sh` on a non-conda host.
The two durable facts: CUTLASS must be **4.x or newer**, because
`include/task/attention.cuh` includes `cute/algorithm/tensor_reduce.hpp`, which
does not exist in 3.x; and `include/task/argmax.cuh`, `include/task/attention.cuh`,
and `include/dae/runtime.cuh` rely on `<cfloat>` and `<array>` arriving as
transitive includes from CUTLASS 3.x, so on 4.x they need
`NVCC_PREPEND_FLAGS="-include cfloat -include array"` until those includes are
added properly.

## Search

[tools/autotune.py](tools/autotune.py) `search` is coordinate descent over knob
**groups**, not knobs.

```bash
python tools/autotune.py search --knobs tuning/qwen3_1p7b.knobs.json \
    --repeats 8 --min-effect-pct 1.8 \
    --preset-out tuning/qwen3_1p7b.preset.json -o tuning/qwen3_1p7b.search.json
```

### Groups, Because Single Knobs Cannot Reach The Interesting Placements

A group is a set of knobs that must move together. A stage's `.sms` and
`.base_sm` are one group; every other knob is a group of one. This is the whole
reason milestone 4 exists: `up_low.sms=128` is illegal only because the baseline
pins `up_low.base_sm=64`, so a single-knob sweep reports it as illegal when it
is merely *unreachable*. Groups enumerate the cross product, so the pair is
reachable.

`--group` restricts the search to named groups, which is how to spend a limited
timing budget on the stages that matter.

### What It Takes To Adopt A Step

Each group is optimized against the configuration reached so far, not against
the original baseline. A step is adopted only if it clears three bars:

1. it wins the same noise-aware test `measure` uses, with the error budget
   spread across that group's candidates;
2. it wins **again** on separately collected samples (`--confirm-rounds`);
3. it passes the **correctness gate**.

### The Correctness Gate

Timing cannot tell a fast schedule from a fast *wrong* schedule. Before adopting
a step, the driver runs the target's own correctness check under the candidate
configuration and refuses the step if it fails. The invocation comes from the
target entry (`correctness_args`, `--correctness` for qwen3_1p7b) or from
`--correctness-arg`. A target that declares no check reports that rather than
silently passing. `--no-correctness-gate` opts out.

### The Head-To-Head Is The Only Honest Number

Greedy descent adopts each step against the config that preceded it, so a chain
of individually-justified steps can add up to less than the sum of its parts.
After the search converges, the original baseline is timed directly against the
final configuration on fresh samples. If that comparison does not come out
`faster`, the summary says so in as many words and tells the reader to treat the
steps as noise that survived, not as a result.

### Presets

`--preset-out` writes the winning configuration in the format
`DAE_TUNE_CONFIG` already accepts, with a `search` provenance block that
`tune._load_config_file` ignores. So the same artifact both re-runs the tuned
schedule and seeds the next search through `--preset`, which `check`, `noise`,
and `measure` also accept. A search that is interrupted, or run one group per
day, can pick up where it left off.

### Verified Against The Fixture

A full-surface search on `tests/fake_sched.py` (seeded, noise off) adopts three
paired moves, converges on the second pass, and the head-to-head confirms
-10.5% end to end. With `FAKE_SCHED_WRONG_IF` set to the winning placement, the
same search adopts nothing and records the correctness rejection; with
`--no-correctness-gate` it takes the wrong config, which is what makes that a
test of the gate rather than of legality.

## Verified On Real Hardware

Run on a GH200 480GB (132 SMs, CUDA 12.8, PyTorch 2.7.0+cu128) on 2026-08-19.
Milestones 1-3 all work end to end against the real target, not just the fixture.

| Check | Result |
| --- | --- |
| `python tests/test_tune.py` | 11/11 pass |
| `python tests/test_autotune.py` | 28/28 pass |
| `sched.py --dry-build` | ok, 23 knobs at defaults |
| `autotune.py discover` | 23 knobs, notes `full_sms=132` |
| `autotune.py check` | 41/115 buildable, 29 static-reject, 45 build-reject, ~9 min |
| one timed run through the wrapper | ok, median 1.515 ms |
| `autotune.py noise --repeats 12` | IQR 1.8% of median, range 3.7%, drift negligible |

The fixture is a fair proxy but not exact: it predicted 107 candidates / 40
legal, the real target gives 115 / 41.

Budget: roughly 20s per timed run including model load, so a full 41-candidate
sweep at 8 rounds is about two hours.

### The Objective Behaves Correctly On Real Data

`measure` over `q_proj.sms`, `down_proj.sms`, `v_proj.base_sm`, `silu.sms`,
8 rounds, `--min-effect-pct 1.8`:

```
Baseline: 1.537 ms median over 8 runs, IQR 0.029 ms
Confidence 95% spread across 8 comparisons -> 99.3750% per candidate

  down_proj.sms=64   1.542 ms    +0.4%   [-0.018, +0.055]  same
  v_proj.base_sm=32  1.549 ms    +0.8%   [-0.006, +0.068]  same
  q_proj.sms=32      1.552 ms    +1.0%   [-0.007, +0.042]  same
  v_proj.base_sm=64  1.563 ms    +1.7%   [+0.004, +0.070]  same
  v_proj.base_sm=0   1.568 ms    +2.0%   [-0.001, +0.060]  same
  down_proj.sms=32   1.753 ms   +14.1%   [+0.194, +0.243]  slower
  silu.sms=2         2.152 ms   +40.0%   [+0.488, +0.661]  slower
  silu.sms=1         2.397 ms   +56.0%   [+0.817, +0.936]  slower

0 candidate(s) beat the baseline
```

Two things this establishes beyond what the fixture tests could:

- `v_proj.base_sm=64` has an interval that excludes zero yet is still reported
  `same`, because +1.7% does not clear the 1.8% floor. The "significant but not
  worth acting on" path fires on real data, not only in unit tests.
- Nothing was crowned. On this knob subset the hand-tuned baseline stands.

That last result should not be read as "search does not help". Every candidate
here moved a single knob away from a schedule a human had already tuned, so the
region tested is the one most likely to be already optimal. The paired
`(sms, base_sm)` moves that milestone 4 exists to explore were unreachable in
this sweep by construction.

### Rejections That Explain Nothing

Four of the 45 build-rejections (`q_proj.sms=96`, `k_proj.sms=48`,
`out_proj.sms=96`, `gate_high.sms=96`) report a bare `AssertionError`.
`classify_failure` is behaving correctly; the cause is that
`SchedGemv.validate()` in [python/dae/schedule.py](python/dae/schedule.py) has
asserts with no message string, such as `assert K % self.fold == 0`. Adding
messages there would make the driver's report self-explaining.

## Result: The Llama3 Search

The first complete search on `llama3_8b`, on a GH200, 2 passes, **2056 timed
runs**, ~8.5 hours.

```
Search changed 1 knob(s) over 2 pass(es), 2056 timed runs:
  pass 1  silu           silu.sms=2, silu.base_sm=128         -1.5%

Head-to-head over 10 fresh rounds each:
  original baseline  598.388 ms
  searched config    588.848 ms
  difference         -1.6%  [-10.969, -7.165] ms  -> faster
```

The interval excludes zero, the run was clean (0 out-of-memory rejections, 0
groups skipped for host reasons, 0 aborts), and 226 legal candidates were timed
across the two passes. **Search beat a hand-tuned schedule, end to end, by
1.6%.**

### Read It Honestly

- **The margin is small.** 1.6% against a 1.0% effect floor and a 0.4% noise
  floor. Real and reproducible, not dramatic.
- **The winning move did not need paired knobs.** `silu.base_sm` stayed at 128;
  only `silu.sms` moved, 4 -> 2. Pairing made 113 configurations reachable
  against 44 for single-knob sweeps (2.6x), and **none of those extra 69
  configurations won**. Milestone 4's mechanism is validated; its premise is
  not yet paid off.
- **Almost everything else was already optimal.** 14 of 17 groups in pass 1 and
  every group in pass 2 returned "nothing beat the current config". The
  hand-tuned schedule was good, and the search mostly confirmed that.
- **Not an optimum.** Coordinate descent finds a local optimum along group
  coordinates, over a hand-chosen grid of candidate values, at a 1% resolution.
  The defensible claim is "no single-group move detectably improves on this",
  nothing stronger.

### The Correctness Gate Caught A Real Bug

`gate_high.sms=128` cleared the timing test **and** the confirmation pass, then
failed the correctness gate. Settled with three standalone runs plus a baseline
control:

```
gate_high.sms=128  run 1: FAIL logits_high: 169.887% <= 10.000%
gate_high.sms=128  run 2: FAIL logits_high: 160.618% <= 10.000%
gate_high.sms=128  run 3: FAIL logits_high: 169.844% <= 10.000%
baseline (control): PASS
```

Deterministic, ~165% error in the high half of the logits projection, and the
baseline passes under the identical invocation. This is a genuinely wrong
schedule that is measurably faster, and **only running the model caught it** --
the bootstrap intervals and the confirmation pass were both fully convinced.

That is the concrete argument for the gate: without it this search would have
adopted a wrong schedule and reported it as a win.

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
trade. The fixture's true `gate_high` effect is detected immediately with noise off,
and correctly withheld at 15 rounds with 3% per-run noise.

### Fixed: Dropped Tail Output In The Timeout Wrapper

`run_with_launch_timeout.py` stopped reading as soon as the child exited, which
left buffered output unread. A fast-exiting run lost its final lines, and those
are the lines carrying the benchmark block and `[perf]`. It now drains the pipe
after the loop. This affected any use of the wrapper, not just the autotuner.
