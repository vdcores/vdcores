# Attention Performance Notes

## Non-split Decode Regression Lesson

- `include/task/attention.cuh` is performance-sensitive beyond the executed math because its templates inline into the monolithic `dae2` interpreter kernel in `src/runtime.cu`.
- For the Llama decode path, cooperatively rewriting the swizzled shared-memory `Q` tile with `thr_mma_qk.partition_A(sQ)` caused a large codegen blow-up even though the runtime path stayed on the non-`splitK` opcode:
  - `b94f48b5fef60e50551d18648549565fa9fa4d4c`: `167` registers and about `17007` SASS lines
  - unpatched `60e6becfb0a945bae3dd2e8e81f459adf421ad12`: `191` registers and about `75391` SASS lines
- The practical symptom was `app/python/llama3/sched.py -b 1` slowing from about `79.76 ms` to about `651.01 ms`.
- A safer way to keep the `exp2`-domain softmax math is to scale the post-QK accumulator fragment by `M_LOG2E / sqrt(head_dim)` instead of rewriting the shared-memory `Q` tile. That restored the benchmark to about `79.76 ms`.

## Useful Checks

- Rebuild just the runtime object first:
  - `source "$(conda info --base)/etc/profile.d/conda.sh" && conda deactivate && conda activate && make clean runtime.o`
- Record ptxas resource usage from the build log.
- Compare static code size quickly with:
  - `cuobjdump --dump-sass runtime.o | wc -l`
- For an end-to-end timing sanity check, use:
  - `python tests/script/run_with_launch_timeout.py --post-launch-timeout 180 --post-launch-idle-timeout 30 -- python app/python/llama3/sched.py -b 1`

## SM100 swapped GQA decode

- For Llama-style GQA with four query heads per KV head, the retained Blackwell
  layout follows CUTLASS example 93: `K[128,128] * Q[8,128]` for QK and
  `V[128,128] * P[8,128]` for PV. This reduces the Q TMA tile from 16 KiB to
  2 KiB and maps one sequence/output row to each 32-DP TMEM-load thread.
- Split-KV should publish unnormalized partial output plus local `(max, sum)`.
  Keeping the final-output pointer in the same barriered metadata record removes
  a pointer-only memory instruction and lets the reducer store directly.
- For the B8/S512 two-split case, a fixed reducer with one warp per live query
  row is slightly faster than the generic split loop. Have lanes 0 and 1 form
  the two normalized weights, broadcast them within the warp, and have every
  lane combine two adjacent BF16x2 values using aligned 64-bit loads/stores.
  A same-image 500-epoch A/B measured 6.464 us versus 6.560 us; 16-thread row
  mappings and direct-L2 partial reads were slower.
- CUTE partitions the full eight-column UMMA output even when only four GQA
  columns are live. Match CUTLASS example 93's raw BMM2 correction instead:
  select a 128-bit repeated 32-DP `tcgen05.ld/st`, use four F32 registers per
  thread, and address the TMEM fragment directly. This retained 96 registers
  for the task image but improved B8/S128 from 3.360 to 3.328 us and B8/S512
  from 6.464 to 6.400 us.
- Apply the same raw four-column load to QK, then transpose the 128x4 F32 score
  tile through the unused half of the 8 KiB scratch slot. Assign one warp to
  each query and four sequence positions to each lane. The local four-value
  reduction plus one warp reduction replaces the CTA-wide max and sum exchange,
  drops the selected image from 96 to 64 registers, and moves B8/S512 from
  6.400 to 6.176 us. A two-SM reducer duplicated TMA/task setup and regressed
  to 6.208 us, so retain one four-row reducer SM.
- A second tcgen05 completion barrier that issued the next QK before the current
  CUDA-core softmax increased registers from 96 to 128 and regressed B8/S256;
  retain the simpler sequential QK/softmax/PV loop unless a future tile changes
  that balance.
- Selecting only the swapped opcode in the Llama-8B 12-op image lowers the
  persistent runtime from about 202 registers to 128 with zero spills. The
  128-step schedule measured 377.529 ms (2.949 ms TBT) on a 152-SM GB200.
