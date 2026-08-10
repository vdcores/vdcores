# DeepSeek-V4 resident loop assembly

- Group layers by identical task shape: SWA/hash, CSA/hash, and one HCA/CSA
  score-routed pair.
- Render each group body once. Use two dependency-barrier banks, an inner
  two-iteration `LOOP`/`LOOPC` with `bar_shift`, a memory-only reload/wait in
  the repeated body, and an outer loop for the remaining repetitions.
- Keep checkpoint weights resident. Replace representative direct 1D loads
  with `LayeredSchedule` pointer columns selected by one allocator-owned linear
  layer index. Reset it once per family and advance it once per repeated body.
- Keep one fixed HBM route-result buffer. Routed LDU descriptors pair it with
  the current layer's expert pointer table; do not add an indirect store path.
- Place persistent KV/index state in contiguous layer-major buffers and use
  counter-derived ordinary store offsets.
- Start with `DeepSeekV4ShapePolicy` assignments. Profile the complete resident
  token before changing task tiles or SM counts.
- Build with `track_profile=1` for layer review. `--profile-layers` records one
  LDU-local counter range for completed layers and another for dependency
  reloads; it does not unroll the loop body or add a thread fence. The benchmark
  prepares the instruction/TMA image and primes the runtime outside measured
  token samples.
- Functional command (worker-local checkpoint host):
  `mpi-run -n 1 --host 10.0.16.24 -- .../deepseek_v4_resident_one_launch.py
  --checkpoint /mnt/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4 --layers 43
  --vocab-size 129280 --expected-token-id 14`.
- Layer-profile command: append `--profile-layers --iterations 5`; use the
  median sample's counter row rather than the minimum sample.
