# Camera-control task graph experiments

## Scope

Python remains the coordinator. `camera_control.py` describes preprocessing,
parallel lane and obstacle perception, and a joined stateful controller using
`graph_execution.py`. Previous steering shifts the next frame's bottom-half ROI;
speed and steering are recursively filtered. This is classical image processing,
not a trained detector or physical robot controller. It emits commands but does
not drive hardware or simulate camera motion from those commands.

The Torch CUDA adapter uses ordinary operators and explicit per-task copies;
these measurements do not execute the original VDCores VM. The distinct
`VDCoresTaskBackend` wraps prepared Launchers but still needs workload binding
and compatible-Hopper hardware validation. Do not report its unit tests as an
end-to-end VDCores result.

## Local and recorded-input checks

```bash
python -m pytest -q tests/test_graph_execution.py tests/test_camera_control.py
python tools/run_camera_control.py --methods cpu --iterations 200 --lanes 4 \
  --output results/camera_control/local-validation.json.gz
python tools/run_camera_control.py --input /absolute/path/camera.npz \
  --methods cpu --output results/camera_control/recorded-validation.json.gz
```

NPZ schema: `frames`, uint8 `[N, H, W, 3]` RGB, N>=4 and H/W>=16. No pickle.
The current heuristic expects bright lane markings and red obstacles; arbitrary
real scenes are not guaranteed to produce useful detections. An application
owner must choose meaningful recordings, labels, and accuracy/safety criteria.
Without a file, seed-7 generated scenes are used and labeled as generated.
First third of frames calibrates task costs, remaining frames evaluate behavior.
The independent NumPy reference advances its own feedback state and verifies
every output, including warmup, and all dependency/feedback timestamp ordering.

## GPU comparison and bounded matrix

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/run_camera_control.py \
  --height 480 --width 640 --lanes 4 --iterations 200 --repeats 3 \
  --output results/camera_control/comparison.json.gz
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python tools/run_camera_sweep.py \
  --output-dir results/camera_control/new-a100-suite
```

Run sequentially on an otherwise idle GPU, never alongside correctness tests.
Each method executes identical perception and control formulas, input frames,
CPU-only preprocessing/control, worker count, and explicit transfer boundaries:

- `cpu`: dependency-ready CPU-only perception with two worker threads by default.
- `serial-gpu`: one outstanding task globally; GPU perception, CPU preprocess
  and control. This is a blocking-service baseline, not the best possible fused
  or GPU-resident implementation.
- `dag-gpu`: same operations/devices as serial, but independent ready tasks and
  lanes may overlap on bounded streams.
- `adaptive`: independently calibrated CPU/GPU service costs per perception
  task, 5% switch margin, and conservative pending-work queue estimates.

Calibration includes dispatch, pinning/allocation, H2D, kernels, D2H, and observed
completion. It is performed afresh on this host at the exact image resolution;
the runner cannot reuse old energy-reduction/GH200 costs. Fixed CPU stage queue
costs are not calibrated by this CLI yet, and GPU queue estimates assume full
serialized service despite possible overlap. Those are explicit model limits.
Initial library warmup is separate, per-run warmup frames are retained but
excluded from reported latency. Trials rotate/reverse method order; three trials
do not fully counterbalance four methods. Each result records order and raw data.

Sweep: 128x192 and 480x640 at 1/4 lanes; separate two-worker CPU load and GPU
matrix-multiply load; 10 ms and 2 ms periodic releases; and a 1,000-frame/loop
longer run, all with three trials. GPU load is 2048-square float32 matmul pulses
with a 1 ms pause; CPU load is 512-square matmul per background thread. They
represent explicit competing work, not guaranteed fixed utilization percentages.

## Interpretation and remaining controls

- Closed-loop saturation admits the next frame after previous control. Periodic
  runs use scheduled timestamps, retain all frames, and include accumulated
  arrival backlog in response latency/deadline misses. No drops are hidden.
- Raw frame and task traces include ready, submit, start, finish, and observed
  completion times. CUDA H2D/kernel/D2H intervals use a separate device clock.
- Compare DAG GPU with serial GPU to isolate the scheduling change. Compare
  adaptive with the better of fixed CPU/GPU, not only a deliberately slow option.
- Median/p95/p99, per-lane tails, throughput, and 10 ms deadline misses are
  reported. A 200-frame trial has coarse p99; preserve longer runs and outliers.
- CPU core-equivalent use covers the process run, including warmup and background
  load threads. It is not whole-system utilization or energy consumption.
- Per-task allocations and instrumentation stay inside the measured path. Add
  reusable buffers, an optimized fused GPU baseline, and camera-stage CUDA Graph
  replay as separate controls before claiming an optimized-runtime advantage.
- Pending gates: real recorded-scene accuracy, a learned model if the application
  needs one, physical sensor/actuator integration, and actual VDCores execution.
