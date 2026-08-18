# Schedule Autotuner

Work toward a driver that searches VDCores schedule parameters (SM placement,
fold, operator grouping, overlap boundary) instead of tuning them by hand.

## Milestones

1. Knob configuration layer, and port `app/python/qwen3_1p7b/sched.py` to it. **done**
2. `tools/autotune.py` with static + dry-build legality filtering only. No GPU needed.
3. Timed runs plus a noise-aware objective.
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
