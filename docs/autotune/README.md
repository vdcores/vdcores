# Schedule Autotuner Results

Figures generated from the Llama-3.1-8B search on a GH200 (132 SMs).
Generated from the search trace (`tuning/llama3_8b.search.json`, 2056 timed
runs), which is a run artifact and is not kept in the repo. Re-running
`autotune.py search` produces a new one.

| Figure | What it shows |
| --- | --- |
| `result.png` | The headline. Autotuned vs hand-tuned decode latency, re-measured independently of the driver: 6 interleaved rounds, fresh process each, sequential on an idle GPU. −1.74%, tuned winning 6/6 paired rounds, and the two distributions do not overlap. |
| `funnel.png` | Why legality dominates. 479 enumerated combinations reduce to 113 that build, ~90 that run without deadlocking, and 1 that is both faster and correct. |
| `reachability.png` | Why placement knobs move in pairs. Varying SM count and base SM together reaches 113 legal configurations against 44 for one-knob-at-a-time sweeps. Note that the single win did **not** require pairing. |
| `noise.png` | Why Llama3 was the tuning target. Its run-to-run spread is 0.4% because each run averages 128 decode steps; Qwen3 measures a single token and spreads 1.8%, which is wider than the effect that was found. |

Regenerate with the script recorded in
[agents/knowledge/schedule-autotuner.md](../../agents/knowledge/schedule-autotuner.md).
