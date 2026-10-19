# Bandwidth timeline plot

On branch `bw_timeline`, `plot_mem_trace_compare.py` plots the Llama3-8B memory-bandwidth timeline used for the paper's Figure 4 comparison.

Inputs: `mirage-llama8b.npz` (401 transfer records), `vdc_llama3_effective_bw.json` (15 aggregate bandwidth points), and `event.py` (stage annotations). The script defaults to 1-us baseline bins and 5-us VDE smoothing. It plots estimated effective bandwidth, not measured HBM counter traffic.

Run from repo root:

```sh
python plot_mem_trace_compare.py mirage-llama8b.npz vdc_llama3_effective_bw.json --events event.py --output /tmp/bandwidth-comparison.png
```

Requires NumPy and Matplotlib; `/opt/miniconda3/bin/python` worked on the local Mac. Existing rendered variants include `eval_bw_over_time.pdf` and `eval_bw_over_time_anno.pdf`. Exact final-paper annotations may differ from script output.

`trace.json` is not needed for plotting these saved inputs and currently fails standard JSON parsing at line 24.

## Compact heatmap

`plot_mem_trace_heatmap.py` renders two thin strips (Mirage above VDE) on a shared time axis and shared color scale, with a bandwidth colorbar at right and no stage annotations or duration bars. Darker blue means higher effective bandwidth. It reuses the original baseline binning and defaults to the comparison plot's 5-us VDE smoothing. Each strip stops at its own completion time. The default 0.5-inch gap reserves space for manual annotations; `--gap-in` adjusts it without changing strip thickness.

```sh
python plot_mem_trace_heatmap.py mirage-llama8b.npz vdc_llama3_effective_bw.json
```

Default outputs: `output/pdf/effective_bw_heatmap.pdf` and a matching PNG. `--vmax` fixes the shared scale; `--vdc-smooth-window-ns 0` disables VDE smoothing. Verified by execution and visual inspection of the rendered PDF.
