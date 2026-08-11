# DeepSeek-V4-Flash runtime contract

## Scope

The target is single-GPU decode of `DeepSeek-V4-Flash-NVFP4`: routed expert
weights use ModelOpt NVFP4, while the remaining quantized linear weights use
E4M3 plus UE8M0 block-128 scales.  Functional model coverage comes before
kernel tuning; performance work should be driven by an end-to-end profile.

## Model shape

- 43 transformer layers, hidden size 4096, 4 hyper-connections.
- 256 routed experts plus one shared expert; top-6 routed experts are active.
- Routed expert intermediate size 2048.
- 64 query heads, one KV head, head dimension 512.
- Alternating compressed-attention ratios 4 and 128, with a 128-token sliding
  window, plus three hash layers.

## Quantized tensor contracts

- NVFP4 weight: packed `uint8 [M,K/2]`, E4M3 scale `[M,K/16]`, and scalar
  FP32 `weight_scale_2`.  Activations use the same packed/per-16 form and a
  scalar input scale.  The GEMV output multiplier is the product of the two
  scalar scales.
- FP8 weight/activation: E4M3 values.  Weight scales are UE8M0 per 128x128
  tile; activation scales are UE8M0 per contiguous K128 block.
- Decode outputs are BF16 with FP32 accumulation in the current tasks.

## Implemented task foundation

- A raw-checkpoint CUDA-core NVFP4 GEMV used as the current fast functional
  path and correctness oracle.
- A native SM100 block-scaled UMMA NVFP4 path with verified operand swizzles,
  E4M3 scale placement, UTCCP-to-TMEM transfer, and all eight output columns.
- A raw-checkpoint E4M3/UE8M0 block-128 FP8 GEMV.
- A BF16-weight/input/output GEMV with FP32 accumulation for checkpoint
  linears that are intentionally unquantized: routing, compression, index
  weighting, embeddings, and the vocabulary head.
- Python checkpoint-contract quantize/dequantize helpers and standalone
  correctness/latency benchmarks.

The native UMMA path is intentionally not the production performance claim
yet: it synchronously reformats every K256 slice and only has one natural SM
per M128 tile.  Defer its prepacking/TMA/split-K redesign until the broad
model path can be profiled.

## Broad functional task coverage

The correctness-first single-token path now covers every DeepSeek-specific
operation needed to assemble decode without substituting another framework's
model math:

- BF16-to-E4M3/UE8M0 block-128 activation quantization and BF16-to-ModelOpt
  packed NVFP4/per-16-E4M3 activation quantization.  Both schedules shard only
  at complete scale-block boundaries and can use all available SMs.
- Partial interleaved RoPE over the final 64 dimensions of both the 512-wide
  attention heads and 128-wide indexer heads, including the inverse attention
  output rotation.
- Sparse 64-head, 512-dimensional attention over supplied window/compressed
  indices with the learned denominator-only attention sink.
- Ratio-4/ratio-128 gated compressed-KV pooling, normalized Hadamard rotation,
  learned 64x128 index scoring, exact streaming top-512 selection, and decode
  index helpers.  The selector retains 512 candidates while merging 512-row
  chunks, so it remains exact beyond the initial 1024 rows without a large
  thread-local array.
- Sqrt-softplus top-6 routing, hash routing, bounded 2048-wide SwiGLU, routed
  plus shared expert reduction, and the existing quantized projection tasks.
- 512/1024 RMSNorm, FP32-weight/BF16-input small projection, mHC pre/Sinkhorn,
  mHC post, and final/MTP mHC-head reduction.

DeepSeek compute tasks have a strict shared-memory contract: queue operands
are allocator slot masks, never global pointers or `MInst` metadata. LDU
resolves static, routed, and indexed addresses and fills shared slots; STU
owns every global write. Bulk payloads use TMA, while small or misaligned
metadata uses an LDU/STU copy opcode. Large linears, top-k inputs, compression
rows, and indexed KV reads stream through bounded shared tiles. No task uses
`__threadfence`; compute-group synchronization stays inside compute and
memory dependencies/barriers stay in the memory VM.

The breadth-first layer assembly uses `SequentialProgram` to render every
placed task into a single resident launch. Each stage boundary is released by
the producer's final STU completion on every active SM, and the next stage
waits on its first allocating LDU instruction on each load port. It does not
insert `IssueBarrier`, host synchronization, or a compute-side fence. Placed
schedules remain owned by the program because placement can create HBM routing
or indexed-load tables referenced by LDU instructions. Single-SM stages rotate
across resident SMs to keep the per-SM instruction image bounded.

The synthetic one-layer launch now exercises all three layer templates:
SWA/hash routing, CSA/index routing, and HCA/ratio-128 compression. Producers
write directly into their next consumer's cache/state destination; the two
residual sublayers ping-pong between resident buffers, so there are no explicit
inter-stage copy tasks. Expert gate/up/down projections use
`SchedRoutedNvfp4Gemv`, so the route result is consumed by LDU rather than by
compute.

The synthetic 43-layer transformer plus output head now also executes as one
launch. The host assembler emits one SWA/hash body repeated twice, one CSA/hash
body, one HCA/CSA score-routed pair repeated twenty times, and the head. At
position zero this represents 3,524 logical stages with 330 queued stage
bodies, 341 compute instructions, 1,470 memory instructions, and 571 dependency
counters. The functional build uses `global_insts=1`, leaving the 4,096-entry
instruction queues in HBM rather than consuming shared memory for a 512-entry
image.

Loop dependency reuse stays entirely in the memory VM. Repeated bodies use two
barrier banks: the inner `LOOP` shifts all body dependencies to the second bank.
The memory-only reload command is part of that repeated body, waits for the
current bank's final STU completion, restores both banks, and remains FIFO ahead
of the next iteration's LDU commands. For the twenty HCA/CSA pairs this is a
two-way inner loop nested in a ten-way outer loop. The two LDU handlers
rendezvous on memory-only `cuda::barrier` objects; no compute thread joins.
There is no `IssueBarrier`, compute-side synchronization, `__threadfence`, or
model-data copy in this path.

Real layer weights are not unrolled into 43 instruction bodies.
`LayeredSchedule` replaces a representative schedule's direct 1D weight loads
with compact HBM pointer columns. One allocator-owned linear layer index is
reset before a repeated family and advanced once per body. Every layer-indexed
dynamic load uses that index directly, avoiding per-load `RepeatM` address
arithmetic and schedule-specific `loop1`/`loop2` opcodes. LDU resolves the
checkpoint address. Routed loads use a two-word descriptor containing one
fixed route-result address plus the current layer's expert pointer table.
Router output therefore stays in one HBM buffer; no indirect store path exists.
Persistent cache outputs use ordinary counter-strided STU addresses.

`DeepSeekV4ShapePolicy` supplies the initial functional tile/SM assignment from
operator shape: FP8 linears expose row tiles, NVFP4 linears expose aligned M8
groups, activation quantizers expose complete scale blocks, and sparse
attention uses one SM per head. These are breadth-first assignments and remain
subject to end-to-end profiling after the resident flow passes.

## Checkpoint-resident one-launch gate

`benchmarks/deepseek_v4_resident_one_launch.py` loads the durable worker-local
checkpoint into one B300, builds routed pointer tables without copying weight
payloads, and assembles position zero from four bodies: layers 0-1, layer 2,
the odd HCA family, and the even CSA family. The odd/even bodies share a 2x10
loop index while retaining distinct checkpoint pointer columns. The router
writes one fixed eight-word HBM result, and all six experts resolve weight,
weight-scale, input-scale, and alpha fields in LDU.

The 2026-08-10 full gate used 153.379 GiB of resident checkpoint storage and
left 29.933 GiB free. It represented 3,781 logical stages with 354 queued stage
bodies, 612 dependency counters, 1,197 compute instructions, and 4,052 memory
instructions in the unchanged 4,096-entry images. One VDCores launch over all
43 layers and the 129,280-row head emitted reference token 14 for input token
791 at position zero.

The original 13,364.171-ms report was not TBT. The CUDA start event was queued
before the first call materialized and uploaded the 152-SM instruction/TMA
image, so the stream sat idle between the event and the resident kernel. The
benchmark now calls `prepare_launch()` before timing and performs one untimed
prime launch. Queue construction/upload remains visible as setup time but is
not charged to a token.

## Per-layer internal-counter profile

The diagnostic path adds four static memory instructions, raising the full
image from 4,052 to 4,056 memory instructions. At each logical layer tail, an
LDU control command waits on the existing final-STU dependency and records
`globaltimer[layer_base + internal_layer_counter]`; it then increments that
LDU-local counter. The 43 layers are still looped rather than unrolled. The
existing barrier-reload command likewise records a second internal-counter
range after rearming dependencies. Profile slots 2-63 are layer frontiers,
64-95 are reload frontiers, and 96-127 remain aggregate runtime counters. This
uses neither `IssueBarrier`, `__threadfence`, nor a compute/memory joint
barrier.

Job `20260810T213255Z-1100716` ran five profiled, checkpoint-resident samples on
one B300 and preserved reference token 14. CUDA samples were 80.794, 79.530,
72.035, 81.054, and 75.447 ms; the median was 79.530 ms. The corresponding
median-sample internal span was 79.275 ms:

| Component | Time (ms) | Internal span |
| --- | ---: | ---: |
| 43 transformer layers | 74.797 | 94.35% |
| 23 loop dependency reloads | 2.555 | 3.22% |
| 129,280-row output head | 1.923 | 2.43% |

Layer-family summaries from that same sample were:

| Family | Layers | Median layer (ms) | Sum (ms) |
| --- | ---: | ---: | ---: |
| SWA/hash | 2 | 0.583 | 1.165 |
| CSA/hash | 1 | 0.735 | 0.735 |
| HCA/score | 20 | 2.482 | 41.849 |
| CSA/score | 20 | 1.573 | 31.048 |

Pair reloads median 0.138 ms and total 2.424 ms, so reload is measurable but
not the primary gap. The stronger structural signal is iteration-dependent
layer time: the first HCA/CSA pair was 0.762/0.640 ms, while later HCA layers
clustered as high as 2.49 ms and CSA layers as high as 2.10 ms despite common
shapes. The next review should therefore correlate these layer frontiers with
allocator slot stalls, LDU queue/dependency wait, and STU service counters
before changing task tiles.

## Single-layer placement and overlap review

Job `20260810T233130Z-1687171` profiled the marker-free layer-0 SWA/hash
program plus a 4,096-row head on one B300. Five CUDA samples were 0.651,
0.615, 0.669, 0.662, and 0.671 ms; the median sample was 0.662 ms and its
internal compute frontier was 0.527 ms. A separate layer-marker run attributed
about 0.487 ms to the transformer layer and 0.055 ms to the small head.

The transformer layer contains 83 logical stages and 82 internal stage edges.
Attention has 34 stages with an equal-stage placement coverage of 47.8%; FFN
has 49 stages with 68.6%; the complete layer has 60.1%. This is only a static
placement proxy, not achieved utilization: 40 stages use all 152 SMs while 18
stages use one SM. The one-SM stages rotate their base SM, but the strict stage
chain means that rotation does not create concurrency.

`SequentialProgram` currently makes every edge a global phase dependency. The
producer's final STU writeback on every active SM decrements a device counter,
and the first allocating consumer load waits for zero in LDU. The marker-free
run measured exactly 7,584 producer arrivals and 7,687 contended consumer
checks out of 7,712 possible stage gates (99.68%). Aggregate per-SM counters
showed LDU dependency waiting over 56.3% of the grid envelope, allocator slot
stalling over 65.2%, a compute non-wait upper bound of 32.7%, and STU service
of 8.0%. These warp-role percentages overlap and must not be added.

All 54,609 payload load commands used LDU0; LDU1 processed no payload command.
The allocator reserves a shared slot before LDU evaluates the dependency, so
a blocked load retains its slot and later commands can fill the 24-slot arena.
No `IssueBarrier` or `__threadfence` was present or executed. The address
resolution model remains correct: routed and indexed addresses are resolved
inside LDU from HBM/local routing state, not in compute.

The missing broad overlap is a DAG rather than a kernel-launch issue. Q and KV
branches can run concurrently after hidden quantization; eight output groups
can fan out after inverse RoPE; router/routed experts and the shared expert can
overlap; six routed experts are independent after route selection; and W1/W3
are independent inside each expert. These must stay in the same resident
VDCores launch and queues. Only true joins need memory-role dependencies.

The first overlap pass should therefore combine ready-task-aware SM placement,
operand-specific LDU dependencies, both LDU ports, and non-slot-consuming
dependency admission. Later adaptive fusion may retain compatible producer
tiles in allocator-owned shared slots through `LoadReg`/`StoreReg`-style task
handoffs instead of forcing an STU-to-HBM-to-LDU round trip.

## Repeated-token variance diagnosis

The full-token path has a second structural problem that is large enough to
mask task-kernel A/B results. In one no-track baseline run, five identical
samples were 47.752, 86.907, 47.233, 47.077, and 84.758 ms. A longer tracked
run produced 69.296--75.781 ms samples, while a 120-sample run again separated
into a fast mode near 45.5 ms and a common slow mode near 81--86 ms. Output
token 14 remained stable.

`--profile-all-samples` extends the resident benchmark to retain and report
all per-layer frontier rows and aggregate counters. The tracked samples did
the same amount of work: 2,455,899 allocator instructions, 2,465,931 LDU0
commands, 10,032 LDU1 commands, and 670,291 stores per token. Reloads stayed
near 0.14 ms per repeated pair. The slowdown instead appears inside nominally
identical layer bodies, frequently in almost exact 0.524288 ms increments;
later HCA/CSA layers commonly land near 1.96/2.10 ms after the first pair is
near 0.75/0.78 ms.

This evidence rules out growing instruction counts and an indirect-layer
counter leak. The leading runtime hypothesis is delayed cross-SM visibility
in the LDU dependency poll: STU releases a global barrier with `atomicSub`, but
LDU currently polls the same word through an ordinary volatile load. Test an
L2-coherent LDU load or atomic observation before attributing the effect to
kernel math. The assigned GPU was otherwise process-isolated, although the
other three GPUs on its GB200 module were occupied and cumulative hardware
power-brake time was present, so clock/power telemetry remains a secondary
control.

The next discriminator records the physical SM ID and `clock64` at resident
kernel start/end. A 20-sample one-layer run showed no fixed straggler and no
frequency mode: the frontier moved across physical SMs and the median inferred
SM clock stayed at 2.047--2.055 GHz in both fast and slow samples. Most layer
frontier spreads were below 1 us. The reproducible mode change is allocator
backpressure instead: fast samples had about 7.6k slot-stall events, while slow
samples had about 10.4k, despite the same 54,609 allocator instructions. All
54,761 payload/control commands went to LDU0 except the 152 duplicated profile
controls on LDU1. This elevates load-port balancing and dependency-aware slot
admission above clock tuning or physical-SM exclusion.

## Performance target and optimization phases

The performance target is 16 ms TBT for the complete 43-layer network plus
head on one GPU, corresponding to roughly 0.35-0.4 ms per layer after allowing
for the head and resident-loop overhead. The measured 79.53 ms baseline is a
4.97x gap, so optimization decisions must be driven by end-to-end attribution
rather than isolated best-case task numbers.

Work proceeds in three measured phases:

1. Kernel and VDCores task optimization. Compare matched FP4, FP8, attention,
   quantization, normalization, routing, and reduction tasks with FlashInfer,
   FA4 where semantics match, and Triton. Consider Blackwell-native layouts
   and instructions, preprocessing immutable checkpoint layout, loading scale
   and data in one TMA transaction, and using fixed/raw addresses or
   register-preloaded values for short frequent state such as routing results.
   Use adaptive fusion and shared-memory register handoff only when it removes
   measured traffic or latency without exposing global pointers to compute.
2. Overlap optimization. Replace the false linear stage chain with true DAG
   edges, jointly assign SMs across ready branches, use both LDU ports, prefetch
   independent weights/metadata, and release or hand off producer tiles at the
   narrowest safe granularity. This remains one launch, one compute queue, and
   one memory queue per resident SM.
3. Progressive-layer slowdown. Correlate layer and loop iteration with
   allocator stalls, dependency waits, LDU/STU service, instruction/cache
   behavior, barrier reloads, and frequency/thermal state. Separate runtime
   state growth or counter drift from architecture/system effects, then fix
   only reproduced causes.

Commit a milestone only after the applicable correctness gate passes and a
representative median improvement exceeds run-to-run noise. Record rejected
experiments in `.agentlog/`; do not preserve tuning that merely moves time
between counters or regresses complete-token latency.

## Verified GB200 baselines (2026-08-10)

- NVFP4 CUDA, M2048 K4096, 128 SMs: bit-exact BF16 reference; 8.256 us median
  task time in a five-iteration smoke.
- NVFP4 native UMMA, M2048 K4096, 16 SMs: all eight columns bit-exact; 79.712
  us median task time.  This is a correctness baseline, not a tuned result.
- FP8 block-128, M4096 K4096, 152 SMs: max absolute error 1.5e-5 versus the
  quantized reference; 16.320 us median kernel span.
- BF16 checkpoint GEMV, M256 K4096, 152 SMs: bit-exact BF16 reference and
  9.920 us in a one-iteration router-shape smoke.
- Broad shared-slot functional sweep, one GB200: all 25 checks passed, including exact
  activation-quantized bit patterns, 64x512 sparse attention with top-k 512,
  both compression ratios, learned indexing, MoE routing/reduction, bounded
  SwiGLU, and all mHC stages.  After parallel quantization/index selection, the
  selective image uses 64 registers, nine barriers, a 112-byte stack frame, no
  spills, and 14,720 bytes static shared memory.
- One-launch synthetic layer templates, one GB200: SWA used 85 stages and a
  71/329 max compute/memory instruction image; CSA used 108 stages and 94/402;
  HCA used 93 stages and 79/353. All three completed in one launch with finite,
  repeatable outputs. The 24-op selective image used 70 registers, nine
  barriers, a 112-byte stack frame, no spills, and 14,720 bytes static shared
  memory.
- FP8 LM-head shape M129280 K4096, 152 SMs: max absolute error 0.003906 and
  656.800 us for the one-iteration projection-shape stress check.  M4096 K4096
  remained at 15.840 us median after pointer-offset sharding.

## Matched task comparison

`benchmarks/deepseek_v4_triton_tasks.py` supplies shape- and math-matched
Triton references for the task classes where Triton is the applicable
baseline.  It uses CUDA-graph replay to remove Python launch overhead and
performs bit-exact checks for both activation quantizers.  The VDCores sweep
uses repeated internal profile spans.  On the same single GB200, representative
medians in microseconds were:

| Task | Shape | VDCores | Triton |
| --- | --- | ---: | ---: |
| FP8 activation quantization | K4096 | 2.656 | 2.032 |
| NVFP4 activation quantization | K4096 | 3.680 | 1.984 |
| Sparse attention | H64, D512, K512 | 197.856 | 101.779 |
| Index score | rows640, H64, D128 | 19.008 | 2.646 |
| Top-512 | rows640 | 39.648 | 20.912 |
| RMSNorm | K512 | 2.144 | 1.338 |
| RMSNorm | K1024 | 2.112 | 1.626 |
| Bounded SwiGLU | K2048 | 2.176 | 1.517 |

At 4096 index rows, exact VDCores index score/top-512 measured 63.840/280.896
us versus Triton/PyTorch 7.152/33.187 us.  These remaining attention and
indexing gaps are explicit end-to-end profile targets, not parity claims.
The structural pass reduced K4096 NVFP4 quantization from 361.216 to 3.680 us
and rows640 top-512 from 47.500 ms to 39.648 us.

FA4 is not a semantic reference for the supplied-index, shared-D512 sparse
attention task: its standard dense/paged decode interface does not implement
the model's selected-index plus denominator-only-sink contract.  FlashInfer is
the applicable external reference for native NVFP4 GEMV and paged attention;
run those comparisons only from a worker that has its matching CUDA-13
environment and a free GPU.  Do not silently substitute an unavailable
package or disturb an occupied device.

These timings remain exploration data.  Assemble and profile the whole model
before doing finer task tuning.

## Synthetic single-GPU decode flow

`python/dae/deepseek_v4_flow.py` describes the official per-token layer order
and cache/index cardinalities.  `benchmarks/deepseek_v4_synthetic_decode.py`
connects that plan to VDCores schedules for all attention families, both
compressors, CSA indexing, mHC residual paths, NVFP4 routed experts, the FP8
shared expert and other quantized projections, and the final head.  It keeps
checkpoint-sized tensor dimensions while reusing one deterministic tensor per
weight shape; this makes it a topology/dataflow test rather than a checkpoint,
quality, memory-footprint, or TBT claim.

On one GB200, the following breadth checks completed with finite residuals and
logits:

- 43 layers at positions 0 and 3: window-only startup followed by the first
  ratio-4 compressor/index-cache boundary.
- 43 layers at position 127: 2 SWA, 21 CSA, 20 HCA, and the full 129,280-token
  vocabulary head.
- 43 layers at position 4095: the same layer mix with 1,024 compressed CSA
  rows, exact top-512 selection, and 640 attention candidates.

The synthetic graph deliberately remains untuned and currently executes many
individual task launches. It is a reference harness, not the VDCores assembly
target. Production assembly follows Llama/Qwen: schedules flatten task and
memory instructions into one compute queue and one memory queue, then issue
one `Launcher.launch()` per decode token.

## Single-launch routed expert foundation

Routed expert selection no longer needs a host readback to choose checkpoint
addresses. `RoutedAddressTable` keeps eight padded route-id words (six valid
ids), routing metadata, and an expert-by-field pointer table in HBM. After the
router output dependency becomes ready, `OP_ALLOC_ROUTED_TMA_LOAD_1D` reads
the selected expert and pointer through L2, copies the selected tensor slice
into normal allocator-owned shared slots, and publishes only the slot mask to
compute. There is no queued-address compatibility object and no alloc-warp
issue barrier. The first routed load carries the route dependency; same-port
LDU ordering covers later expert fields.

Sparse attention uses the same boundary through
`OP_ALLOC_INDEXED_TMA_LOAD_1D`: LDU reads runtime KV row ids and streams those
rows into shared memory. `RepeatM` advances compact 24-byte HBM lookup records,
so top-512 attention does not require hundreds of materialized memory
instructions. Compute sees shared Q, indices, sink, KV row tiles, and output
slots only.

The first composed proof used one `Launcher.launch()` for hash routing and a
four-SM sharded NVFP4 GEMV. Expert 37 remained on device from routing through
the LDU tensor load, and the output was bit-exact to the quantized reference.
This is the required routing/addressing foundation for a whole-layer queued
schedule, not a TBT result.

## Real-checkpoint preflight

`python/dae/deepseek_v4_checkpoint.py` generates and validates the exact raw
non-MTP checkpoint contract, parses safetensors headers without materializing
payloads, and lazily loads only explicitly requested tensors while preserving
their FP8/NVFP4 layouts.  `benchmarks/deepseek_v4_checkpoint_audit.py` can audit
either a local download or HTTPS range-read headers from the official NVIDIA
repository.

At NVIDIA revision `7fc18be2b215ae48260383d4a228ec8a033046f7`, all 46 remote
shard headers passed: 135,235 total tensors, including 133,660 base-inference
tensors and 1,575 MTP tensors.  Total tensor payload is 168,266,793,544 bytes;
the base model without MTP is 164,673,005,788 bytes (153.36 GiB).  A 189,471
MiB GB200 therefore has about 31.67 GiB left before caches, activations, task
images, and allocator overhead.  This is a viable but tight single-GPU fit.

The pinned checkpoint is downloaded on worker `10.0.16.24` at
`/mnt/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4`; its Hugging Face cache is
also worker-local at `/mnt/checkpoints/huggingface-cache`.  The model directory
occupies 157 GiB on local EXT4 storage.  Its local header audit passed the same
135,235-tensor, 46-shard contract as the remote audit.  Checkpoint-backed jobs
must be pinned to this host because `/mnt` is worker-local; checkpoints must
never be copied into the NFS source tree or a worker home directory.

`DeepSeekV4Checkpoint.load_fp8_linear()` and `load_nvfp4_linear()` bind a
named checkpoint prefix to the raw schedule-ready tensors without
dequantizing or rewriting it.  The real-checkpoint task smoke on one GB200
passed for `layers.2.attn.wq_a` (E4M3/UE8M0, exact against the quantized
reference) and `layers.2.ffn.experts.0.w1` (packed NVFP4, 0.048096 maximum
absolute BF16 error).  This verifies checkpoint-to-VDCores dtype, layout, and
scalar-scale routing for both quantization families.

`DeepSeekV4ResidentCheckpoint` packs the non-MTP tensors into aligned,
per-shard device buffers while preserving their raw dtypes and exposes
read-only views without per-layer device copies. On `10.0.16.24:1`, all 45
base-model shards loaded from the durable `/mnt` checkpoint in 69.459 seconds:
153.364 GiB of tensor payload occupied 153.379 GiB of storage and left 29.933
GiB free. A resident layer-0 plus 4,096-row head smoke then passed. This
validates model residency and memory headroom; that smoke still used the
streaming task launcher and is not a one-launch or TBT result.

`benchmarks/deepseek_v4_checkpoint_decode.py` is the first complete
position-zero real-weight VDCores flow.  It streams one input token through
every transformer layer and the vocabulary head without materializing the full
checkpoint in host or device memory.  Embedding and head rows are sliced
directly from their safetensors shards, while each layer loads only its active
routed experts and releases the layer weights before advancing.  On one
allocator-managed GPU (`10.0.16.24:1`), token 791 passed all 43 layers (2 SWA,
21 CSA, and 20 HCA), both hash and score routing, NVFP4 routed experts, the FP8
shared expert and attention projections, BF16 routers/compressors, mHC, and
the full 129,280-row head.

The first run exposed a model-semantic error that isolated task tests had
mirrored: the non-symmetric mHC Sinkhorn matrix must update residual streams as
`comb.T @ residual`, matching both the official Transformers model and vLLM's
TileLang kernel.  After correcting the CUDA task and reference, all 43 layers
selected token 14 with finite logits in [-31.875, 18.625].  A matched vLLM
0.23.0 greedy run from the same one-token prompt emitted `[14, 223]`, so the
first VDCores token now agrees with the framework reference.  The corrected
streaming run took 86.522 seconds, which is I/O- and launch-dominated and is
not a TBT measurement.

This position-zero run is a breadth gate, not a quality or TBT result.  RoPE is
identity there, and historical compressed/index cache state is empty, so the
compressor/index projections execute for coverage but their outputs are not
consumed.  The harness also streams weights from local storage and launches
tasks individually.

The harness now also supports the first four autoregressive positions.  Its
generated main and Yarn-scaled compressor RoPE tables match the official
Transformers implementation exactly, and it keeps a 128-row circular window
KV cache plus FP32 compressor/index partial state for every layer.  A two-step
run from token 791 recomputed position zero, fed token 14 back at position one,
attended both live KV rows in all 43 layers, and emitted `[14, 223]`, exactly
matching vLLM's greedy IDs.  The streaming steps took 87.241 and 88.270 seconds
(175.519 seconds total); these timings include checkpoint I/O, allocation, and
individual Python launches and remain non-performance data.

The position-three gate exposed that ordinary Q/K and inverse-output RoPE is
layer-dependent: SWA layers use the main theta, while CSA/HCA layers use the
same Yarn-scaled compressor theta as their compressed keys.  After fixing that
selection, a four-step, 43-layer run emitted `[14, 223, 18, 90]`, exactly
matching vLLM.  Position three pooled and normalized the first ratio-4 entry,
ran the 128-wide Hadamard indexer and top-k selection, and consumed the selected
compressed row in all 21 CSA layers.  The four streamed steps took 354.070
seconds total and remain functional rather than performance data.  The next
cache-state breadth gate is the first ratio-128 HCA entry; it should be reached
without treating the streaming harness's checkpoint I/O as model TBT.

## Profiling and optimization gate

Detailed tuning starts only after the broad cache-state and resident-model
functional gates.  The position-three CSA gate now passes.  Next make the full
non-MTP checkpoint resident on one GPU, retain enough memory for the matched
cache lengths, and measure launch-inclusive decode without checkpoint I/O or
per-layer allocation/free time.  Validate the ratio-128 HCA transition as part
of that resident flow before optimizing individual tasks.

Use batch 1 and identical greedy two-token requests for VDCores, vLLM, and
SGLang at context lengths 128, 512, and 4096.  Record warmups, at least 30 timed
steps, min/median/p90 TBT, cache dtype/layout, allocated memory, task-image
revision, and framework versions.  Profile one representative context before
tuning, attributing total time and launch count to attention, routed/shared
experts, quantization, mHC/norm, and the output head.  Prioritize only stages
that materially contribute to model TBT.  Expected first candidates are
persistent schedule composition, quantize-to-GEMV fusion, adjacent mHC
post/pre fusion, overlap of independent attention projections, and removal of
Python launcher/allocation work; task microbenchmarks remain diagnostic rather
than the optimization objective.

The CUDA-13/Blackwell vLLM 0.23.0 environment on that worker completed a real
TP=1, one-GPU, two-token inference at context 128.  vLLM selected FP4 experts
through FlashInfer TRT-LLM, FP8 DeepGEMM linears, FP8 MLA KV cache, and FP8
Lightning Indexer cache.  It loaded 153.97 GiB of model state, retained 25.57
GiB for KV cache (63,071 tokens), and exited cleanly.  The environment's PyPI
CUDA compiler had drifted to 13.3 while its runtime headers remained 13.0;
pinning both `CUDA_HOME` and `CUDA_PATH` to `/usr/local/cuda` selected the
coherent system CUDA 13.0 toolchain and fixed TileLang mHC compilation without
changing installed packages.

The first cold start spent 1,177 seconds profiling, compiling cached SM100
operators, warming DeepGEMM, and running FlashInfer's built-in 21-profile MoE
selection.  The one-sample eager harness reported a 0.736501 ms first-to-second
token interval.  Treat this only as an E2E functional smoke result: it had no
warmup or statistical sample set and is not a framework TBT baseline or parity
claim.

All GPU checks must run through the cluster MPI launcher with one rank and the
target checkout on `PYTHONPATH`.  Runs using this checkpoint must also select
worker `10.0.16.24` and pass the explicit `/mnt` checkpoint path.
