# GEMM Scheduler Notes

- `SchedGemm` lives in `python/dae/schedule.py`.
- Use `SchedGemm` instead of `SchedGemv` when the runtime `N` extent spans multiple atom tiles.

## Why It Exists

`SchedGemv` assumes:

- the runtime `N` extent is exactly one atom-wide tile
- SM mapping only needs to cover `M` tiles, then fold across `K`

`SchedGemm` changes that to:

- tile across both `M` and `N`
- map one SM lane to one `(m_tile, n_tile)` output tile
- fold only across `K`
- when `num_sms` is smaller than the number of output tiles, assign multiple output tiles to each SM lane in a strided loop

This matches GEMM-shaped atoms such as:

- `Gemm_M64N64`
- `Gemm_M64N128K64`

## Output Convention

Unlike the GEMV helper path, `SchedGemm` uses the natural GEMM output layout:

- logical output tensor shape is `[M, N]`
- store coordinates are addressed as `(m, n)`
