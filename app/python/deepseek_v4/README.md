# DeepSeek-V4-Flash Live Decode Demo

[`sched.py`](sched.py) is the user-facing, batch-one DeepSeek-V4-Flash demo.
It accepts ordinary user text, formats and tokenizes it with the released
checkpoint, performs an offline PyTorch prefill, and then generates tokens
with reusable VDCores instruction images.

The demo and the production runtime are intentionally separate:

- [`sched.py`](sched.py) owns prompt formatting, tokenization, offline
  reference prefill, text streaming, and presentation of measurements.
- [`deepseek_v4_inference.py`](../../../python/dae/deepseek_v4_inference.py)
  owns reusable-flow planning and production token execution. It has no
  tokenizer, reference model, or terminal UI.
- [`deepseek_v4_live.py`](../../../python/dae/deepseek_v4_live.py) owns the
  persistent BF16 KV caches and FP32 compressor state shared by prepared
  instruction images.

Run all commands below from the repository root.

## Hardware and software requirements

The measured configuration uses one NVIDIA GB300 GPU, CUDA 13, Python 3.12,
and the released `DeepSeek-V4-Flash-NVFP4` checkpoint. The VDCores decode path
also needs its offline MXFP4 FFN representation.

Install the general project dependencies first, then install the prefill-only
packages:

```bash
pip install -r app/python/deepseek_v4/requirements-prefill.txt
pip install --no-build-isolation \
  'git+https://github.com/Dao-AILab/fast-hadamard-transform.git@v1.1.0'
```

The prefill dependencies are not used in the timed VDCores decode path.

## Prepare the checkpoint once

Set the checkpoint location:

```bash
export DSV4_CHECKPOINT=/path/to/DeepSeek-V4-Flash-NVFP4
```

Create the VDCores MXFP4 FFN files. Weight format and layout conversion are
offline operations and are not inserted into inference:

```bash
python tools/convert_deepseek_v4_ffn_mxfp.py "$DSV4_CHECKPOINT"
```

Create the one-file model-parallel-one image used by the PyTorch prefill
loader:

```bash
python "$DSV4_CHECKPOINT/inference/convert.py" \
  --hf-ckpt-path "$DSV4_CHECKPOINT" \
  --save-path "$DSV4_CHECKPOINT/vdcores-pytorch-mp1" \
  --n-experts 256 \
  --model-parallel 1 \
  --expert-dtype fp4
```

The resulting paths used by the demo are:

```text
$DSV4_CHECKPOINT/
├── encoding/
├── inference/
├── vdcores-mxfp4-ffn-v1/
└── vdcores-pytorch-mp1/
    └── model0-mp1.safetensors
```

The released routed experts are NVFP4 in the reference checkpoint. Offline
prefill explicitly dequantizes selected experts in PyTorch. Timed VDCores FFN
execution consumes the separately prepared MXFP4 representation directly.

## Build the compact production image

Build the runtime with only the operators selected by the production image:

```bash
DAE_COMPUTE_OPS_FILE=benchmarks/deepseek_v4_live.ops \
  make -B -j2 num_insts=512 mxfp_direct_tma=1 pyext
```

`num_insts=512` and `mxfp_direct_tma=1` are compile-time runtime settings.
Rebuild the extension if either setting or the CUDA runtime sources change.

## Run an arbitrary user prompt

Pass user text with `--user-prompt`:

```bash
python app/python/deepseek_v4/sched.py \
  --checkpoint "$DSV4_CHECKPOINT" \
  --mxfp-ffn-root "$DSV4_CHECKPOINT/vdcores-mxfp4-ffn-v1" \
  --prefill-checkpoint "$DSV4_CHECKPOINT/vdcores-pytorch-mp1" \
  --device-span-tokens 3 \
  --max-new-tokens 256 \
  --user-prompt "Explain how asynchronous GPU pipelines overlap memory transfers with matrix computation, then provide a concise Python example and discuss synchronization, correctness, scheduling, resource allocation, and performance tradeoffs in practical inference systems. Compare persistent kernels, CUDA graphs, tensor memory accelerators, and conventional launches for autoregressive decoding. Thanks."
```

The positional prompt argument remains as a legacy alias, but do not provide
it together with `--user-prompt`.

The checkpoint formatter determines the actual token count. A prompt with
`P` formatted tokens is executed as follows:

1. PyTorch prefills tokens `0 .. P-2` and exports its persistent state.
2. The final prompt token is the VDCores input at position `P-1`.
3. VDCores produces `N` new tokens from positions `P-1 .. P+N-2`.

Consequently, the benchmark below used a 65-token formatted prompt: PyTorch
prefilled exactly 64 tokens, and VDCores consumed the 65th prompt token before
generating 256 outputs.

## Prompt and generation limits

User text and formatted prompt length are not fixed. The current production
controller accepts:

- any nonempty prompt that the checkpoint tokenizer can encode;
- first-token, short-context, full-window, and indexed long-context variants;
- a live-cache extent satisfying
  `(formatted_prompt_tokens - 1) + max_new_tokens <= 65,536`; and
- between 1 and 256 newly generated tokens per controller instance.

The 256-token limit is a host-controller policy, not a requirement that the
prompt have a particular length. Offline PyTorch prefill time and memory grow
with the prompt and can become the practical limit before the live-cache
capacity is reached.

## Multi-token persistent launches

`--device-span-tokens 3` enables the selected multi-token policy:

- Positions before 128 remain singleton launches because short-CSA history
  still needs host-side packing between tokens.
- At and after position 128, an ordinary run can execute three autoregressive
  tokens in one persistent kernel launch.
- Ratio-4 CSA, ratio-128 HCA, and long-context index-selection boundaries are
  never crossed by one image. A prepared singleton image handles the boundary,
  after which the reusable ordinary image resumes.

Inside a multi-token launch, GPR2 contains the absolute position and GPR3 the
exclusive terminal position. The memory virtual core derives token-history,
embedding, routing-hash, RoPE/APE, KV, and output addresses from those states.
Argmax stores token `position + 1`, and the next iteration reads it directly.
The dependency uses the ordinary memory-command/barrier path; no issue barrier
or Python round trip appears inside the span.

For the exact 64-prefill plus 256-decode run, the planner prepared five image
shapes and executed 160 launches:

```text
normal singleton + short CSA + HCA + normal triplet + CSA singleton
```

Set `--device-span-tokens 1` to force the compatibility path with one
persistent launch per output token.

## Stop behavior

By default, the tokenizer EOS token stops generation. Additional token IDs can
be supplied by repeating `--stop-token-id`. `--ignore-eos` disables only the
tokenizer EOS stop, while `--max-decode-seconds` checks a host wall-time budget
before starting another device span.

Stops are observed after a completed span. A three-token span can therefore
compute as many as two tokens beyond a stop token. Those speculative tokens are
discarded from the returned generation. Structural state has advanced through
the span, which is harmless because generation ends at that point.

## Output and timing fields

The demo reports distinct phases so offline work is not mistaken for decode
latency:

- `[prompt]` reports the formatted prompt token count.
- `[prefill]` reports offline PyTorch model loading and prefill time.
- `[prepare]` reports reusable-image construction and the planned launch count.
- `[decode]` reports each logical output token and amortized timing.
- `[perf]` reports medians over logical output tokens.

For a multi-token launch, its elapsed time is divided by the number of tokens
in the span. The timing columns mean:

| Field | Meaning |
|---|---|
| `device_ms` | Internal device-counter grid frontier, amortized per token. |
| `cuda_ms` | CUDA-event launch duration, amortized per token. |
| `wall_ms` | Host launch, synchronization, and token readback, amortized per token. |
| `tokens_per_s` | `1000 / median_wall_ms`; a median-derived rate, not total-job wall throughput. |

Use `--quiet-stream` for performance runs. It suppresses repeated text decoding
but retains token, completion, and timing output. Avoid `--verbose-prepare`
when measuring preparation overhead because it prints the full internal
schedule report.

## Measured 64-prefill plus 256-decode result

Commit `1d9c6d1` was measured on one GB300 with the command above and
`--ignore-eos --quiet-stream` so the complete 256-token budget ran:

| Measurement | Result |
|---|---:|
| Formatted prompt tokens | 65 |
| Offline-prefilled tokens | 64 |
| Generated tokens | 256 |
| PyTorch prefill | 71.295 s |
| Reusable-image preparation | 53.701 s |
| Prepared image shapes | 5 |
| Persistent launches | 160 |
| Median device time | 5.443808 ms/token |
| Median CUDA-event time | 5.537654 ms/token |
| Median Python wall time | 5.680572 ms/token |
| Median-derived throughput | 176.04 token/s |

The historical `--device-span-tokens 1` measurement was 5.411 ms device,
6.406 ms Python wall, and 156.1 token/s. Multi-token control therefore keeps
device work essentially flat while reducing the host-visible per-token gap.
Offline prefill and reusable-image preparation are intentionally excluded from
the decode rate.

This is a production-path performance run across CSA and HCA boundaries. It is
not a new 256-token, token-for-token reference-parity measurement.

## Fast schedule smoke test

To skip the tokenizer and reference prefill while testing image preparation
and decode scheduling:

```bash
python app/python/deepseek_v4/sched.py \
  --checkpoint "$DSV4_CHECKPOINT" \
  --mxfp-ffn-root "$DSV4_CHECKPOINT/vdcores-mxfp4-ffn-v1" \
  --input-token-id 1234 \
  --decode-start-position 128 \
  --device-span-tokens 3 \
  --max-new-tokens 16 \
  --ignore-eos \
  --quiet-stream
```

This mode allocates zero-initialized persistent history. It is useful for
schedule and performance smoke tests but is not a text-correctness test.

## Main command-line options

| Option | Purpose |
|---|---|
| `--checkpoint PATH` | Released DeepSeek-V4-Flash checkpoint; required. |
| `--mxfp-ffn-root PATH` | Offline VDCores MXFP4 FFN directory. |
| `--prefill-checkpoint PATH` | MP1 PyTorch prefill image; defaults below the checkpoint. |
| `--user-prompt TEXT` | Arbitrary user message to format and prefill. |
| `--thinking-mode {chat,thinking}` | Checkpoint conversation-format mode. |
| `-N`, `--max-new-tokens N` | Output-token budget in `[1,256]`. |
| `--device-span-tokens N` | Maximum device-controlled span; production default is 3. |
| `--stop-token-id ID` | Additional stop token; repeatable. |
| `--ignore-eos` | Run past the tokenizer EOS token. |
| `--max-decode-seconds S` | Stop before a new span after the wall-time budget. |
| `--quiet-stream` | Disable cumulative per-token text rendering. |
| `--verbose-prepare` | Print internal schedule construction details. |
| `--input-token-id ID` | Skip real prompt and prefill for a synthetic smoke test. |
| `--decode-start-position P` | Synthetic starting position; requires `--input-token-id`. |

Use `python app/python/deepseek_v4/sched.py --help` for the authoritative
parser defaults.

## Troubleshooting

- If `model0-mp1.safetensors` is missing, run the checkpoint's MP1 conversion
  command shown above and pass the resulting directory with
  `--prefill-checkpoint`.
- If the instruction image exceeds capacity or runtime opcodes disagree,
  rebuild `pyext` with the compact operator list and compile-time settings
  shown above.
- If prompt plus decode exceeds the cache limit, reduce `--max-new-tokens` or
  shorten the formatted prompt.
- If a benchmark stops early, use `--ignore-eos` and remove explicit stop
  tokens. Do this only for measurement; normal user generation should retain
  EOS handling.
- If a textual result is needed, do not use the synthetic
  `--input-token-id` path because it has no real prefetched history.
