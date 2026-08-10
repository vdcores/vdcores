# DeepSeek-V4 resident loop assembly

- Group layers by identical task shape: SWA/hash, CSA/hash, and one HCA/CSA
  score-routed pair.
- Render each group body once. Use two dependency-barrier banks, an inner
  two-iteration `LOOP`/`LOOPC` with `bar_shift`, one memory-only bank reload,
  and an outer loop for the remaining repetitions.
- Keep checkpoint weights resident. Replace representative direct 1D loads
  with `LayeredSchedule` pointer columns selected by memory-loop counters.
- Keep one fixed HBM route-result buffer. Routed LDU descriptors pair it with
  the current layer's expert pointer table; do not add an indirect store path.
- Place persistent KV/index state in contiguous layer-major buffers and use
  counter-derived ordinary store offsets.
- Start with `DeepSeekV4ShapePolicy` assignments. Profile the complete resident
  token before changing task tiles or SM counts.
