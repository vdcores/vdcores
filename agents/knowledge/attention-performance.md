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
