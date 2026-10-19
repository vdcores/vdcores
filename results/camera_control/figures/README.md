# Camera-control task graph figures

Generated-scene A100 experiments; ordinary Torch execution, not the original VDCores GPU VM.

## Camera Pipeline

The implemented path and generated scene fixture; this is software-in-the-loop, not real-world autonomy.

![The implemented path and generated scene fixture; this is software-in-the-loop, not real-world autonomy.](camera_pipeline.png)

[Editable SVG](camera_pipeline.svg)

## Method Comparison

All four idle-host configurations. Bars are medians of three trial summaries; whiskers span trials.

![All four idle-host configurations. Bars are medians of three trial summaries; whiskers span trials.](method_comparison.png)

[Editable SVG](method_comparison.svg)

## Background Load

Matched 480×640/four-loop comparisons with explicit CPU or GPU competing work. p99 from 200-frame trials is coarse.

![Matched 480×640/four-loop comparisons with explicit CPU or GPU competing work. p99 from 200-frame trials is coarse.](background_load.png)

[Editable SVG](background_load.svg)

## Periodic Deadlines

Scheduled arrival latency includes backlog; active processing latency is separate. Dots show individual trial medians; miss fractions pool all measured frames. Every frame is retained.

![Scheduled arrival latency includes backlog; active processing latency is separate. Dots show individual trial medians; miss fractions pool all measured frames. Every frame is retained.](periodic_deadlines.png)

[Editable SVG](periodic_deadlines.svg)

## Tail Latency

12,000 measured frames per method across three trials. Pooled curves and individual per-loop/trial p99 preserve slow observations.

![12,000 measured frames per method across three trials. Pooled curves and individual per-loop/trial p99 preserve slow observations.](tail_latency.png)

[Editable SVG](tail_latency.svg)

## Feedback Commands

Stateful commands from actual recorded experiment outputs, not an illustrative trajectory or physical robot trace.

![Stateful commands from actual recorded experiment outputs, not an illustrative trajectory or physical robot trace.](feedback_commands.png)

[Editable SVG](feedback_commands.svg)

## Adaptive Decisions

Actual CPU/GPU task decisions. A high GPU selection rate does not by itself establish a performance benefit.

![Actual CPU/GPU task decisions. A high GPU selection rate does not by itself establish a performance benefit.](adaptive_decisions.png)

[Editable SVG](adaptive_decisions.svg)

[Trial summary data](summary.json). Regenerate with `python tools/plot_camera_control.py`.
