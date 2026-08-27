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

### Dual-LDU operand overlap

The first accepted overlap milestone greedily assigns each stage's default
allocating loads to the LDU port with fewer encoded bytes while preserving
explicit port-1 assignments. Both ports remain children of the same allocator
instruction stream and resident launch, and the first load on every active
port receives the same true stage dependency.

On one layer, job `20260811T005301Z-2094018` split 54,913 load commands into
28,752 on LDU0 and 26,161 on LDU1. Thirty tracked samples tightened to
0.5585--0.5716 ms with a 0.5643 ms median, versus a controlled 0.7019 ms
single-port median; the transformer frontier fell to about 0.431 ms and
allocator slot-stall events stabilized near 7.0k.

The 43-layer gate, job `20260811T005339Z-2097329`, preserved reference token
14 and measured 34.569--36.113 ms over ten tracked samples, median 35.275 ms.
The previous six-sample tracked median was 72.831 ms. The median-sample
internal breakdown is 30.827 ms in layers, 2.833 ms in reloads, and 1.688 ms
in the full head. Work counts did not change; LDU0/LDU1 handled
1,308,392/1,167,571 commands. This removes the large token-level timing modes,
but HCA layers still grow from roughly 0.58 ms to occasional 1.24 ms while CSA
layers are near 0.53--0.56 ms, so dependency-aware admission and the broader
task DAG remain necessary for the 16 ms target.

The matching non-instrumented production build, job
`20260811T005835Z-2121571`, ran 30 post-warmup tokens with reference token 14.
Samples were 33.360--34.500 ms with a 34.074 ms median. Use this stable value,
not the tracked diagnostic timing, as the starting TBT for the next loop.

### NVFP4 activation-quantization placement

The accepted Blackwell quantizer maps one scale block to each eight-thread
subgroup. Sixteen subgroups reduce and pack 16 blocks concurrently with a
compact block loop, warp-local synchronization, and one final compute-group
barrier. The shape policy therefore assigns 16 blocks per SM: K4096 uses 16
SMs and K2048 uses eight instead of spreading one or two tiny blocks over most
of the device.

The task stayed bit-exact and improved from 3.904 to 3.712 us. The one-layer
checkpoint gate measured 0.557 ms median. Full checkpoint job
`20260811T011035Z-2181920` preserved reference token 14 over 30 samples and
measured 33.007--34.276 ms, median 33.731 ms. This is a 0.342 ms complete-token
improvement over the 34.074 ms dual-LDU production boundary.

### Same-placement dependency elision

The first explicit-DAG milestone omits a predecessor edge only when adjacent
stages use the identical SM interval. Per-SM compute, LDU, and STU queue order
then makes the later stage tail dominate the earlier tail; the next true edge
is their join. The assembler rejects an elided edge across different
placements. This is queue ordering inside the same resident launch, not an
additional schedule or launch.

This removes false W1-to-W3 edges for the six routed experts and shared expert,
and same-shape compressor value-to-gate edges. W3 still dynamically resolves
the selected expert's addresses in LDU; its identical-placement W1 predecessor
has already passed the route edge, so no redundant route wait is needed.
There is no IssueBarrier, compute/memory joint barrier, or thread fence.

The one-layer checkpoint median improved from 0.557 to 0.517 ms. Full job
`20260811T013151Z-2288944` preserved token 14 over 30 samples and measured
31.319--34.329 ms, median 32.130 ms. The queued compute/memory images remained
1,197/4,052 instructions while dependency counters fell from 612 to 555. This
is a 1.601 ms (4.7%) complete-token improvement over the 33.731 ms boundary.

The post-DAG repeated-layer profile, job `20260811T013645Z-2313767`, ran six
tracked tokens in 31.230--31.831 ms. Layer, reload, and head spans were
27.054--27.621, 2.546--2.625, and 1.436--1.494 ms. CSA stayed near
0.49--0.51 ms throughout. HCA moved from about 0.50--0.55 ms on its first body
to a mostly stable 0.81--0.83 ms mode, with occasional 0.64--0.66 ms bodies;
it no longer grew monotonically with loop position.

All samples executed exactly 2,191,707 allocator instructions,
1,110,248/1,101,523 LDU commands, and 538,195 stores. Inferred SM clocks stayed
near 2.05 GHz and layer-frontier spreads remained below one microsecond. This
confirms that the original progressive slowdown was queue/allocator
backpressure rather than host, thermal, clock, or fixed-physical-SM overhead.
The remaining system-level targets are the repeatable HCA mode and roughly
2.6 ms of loop dependency reloads.

### Active-bank barrier reload

Each compact loop iteration consumes one shifted dependency-barrier bank, but
the original reload restored both banks every time. LDU now derives the active
bank's first barrier from the shifted completion barrier and restores only that
bank. The assembler supplies one bank's barrier count. The reload still drains
both LDU ports, waits for the current STU tail, and provides the memory-only
loop-carried dependency on every iteration; no task unrolling or new launch is
introduced.

The two-layer tracked gate cut each reload from roughly 0.059--0.060 ms to
0.029--0.032 ms. Full tracked job `20260811T014612Z-2360742` measured
26.627--27.297 ms over six tokens, median 26.973 ms. Reload time fell from
about 2.59 to 1.34 ms, and reduced reset/backpressure also lowered layer time
from about 27.3 to 23.9 ms. Production job `20260811T014956Z-2378731`
preserved token 14 over 30 samples and measured 26.713--27.797 ms, median
27.206 ms. This is a 4.924 ms (15.3%) complete-token improvement over the
32.130 ms boundary.

### Zero-copy routed W1/W3 adaptive fusion

Routed W1 and W3 consume the same packed NVFP4 activation and per-16 scale on
every SM. A dedicated LDU instruction now TMA-loads each operand into ordinary
allocator-owned shared slots and records that slot mask in the issuing LDU's
four-entry local register file. W1 omits those masks from its release; W3 uses
fixed-port `RegLoad` commands to consume the same slots and releases them after
its compute task. The two operands are pinned to distinct LDU ports because
the register files are handler-local. This is a slot alias, not an
inter-stage copy, and does not expose a global address to compute.

The handoff retains three shared slots per active SM only across each adjacent
W1/W3 pair. It removes one 2,048-byte activation load and one 256-byte scale
load per SM, route rank, and layer: about 86.2 MiB of repeated HBM traffic for
the 43-layer token. It adds no compute or memory queue entries, dependency
counter, launch, or fence. Fixed-port annotations are preserved by the
dual-LDU balancer so a later schedule rewrite cannot silently move one half of
the handoff to the wrong handler.

Focused job `20260811T020928Z-2477023` ran route, retained W1, and reused W3 in
one launch and was bit-exact (`max_abs=0`). The one-layer checkpoint gate
measured 0.5169 ms median. Production job `20260811T020955Z-2479547`
preserved token 14 over 30 samples and measured 26.234--27.088 ms, median
26.571 ms. This improves the 27.206 ms boundary by 0.635 ms (2.3%) and leaves
10.571 ms to the 16 ms target.

### Stage-group overlap attribution

The post-fusion full-model profile, job `20260811T021559Z-2510693`, preserved
token 14 over six samples and measured 26.805--27.440 ms, median 27.213 ms in
the tracked build. Layer, reload, and head spans were 23.77--24.25,
1.346--1.356, and 1.394--1.455 ms. Every token executed 2,191,707 allocator
instructions, 1,188,680/1,023,091 LDU commands, and 538,195 stores. Both LDUs
still spent about 68--70% of the grid envelope waiting on dependencies and the
allocator about 69--70% stalled for slots, while clocks and frontier spreads
were stable. Remaining slowdown is therefore structural serialization and
slot pressure, not progressive work, clock, or host overhead.

The one-layer diagnostic can mark 55 existing dependency frontiers without
changing the task graph. Its control descriptors use reserved special slots
7/8; slots 0/1 remain dedicated to the adaptive W1/W3 register handoff. This
separation is required because the allocator can publish a later control
descriptor before an earlier RegLoad has drained from an LDU FIFO.

Job `20260811T022342Z-2548920` preserved layer-0 token 2835 over five tracked
samples. The median sample's 0.443 ms layer span divided into 0.195 ms for
attention and 0.249 ms for FFN. The largest serial branch groups were:

| Group | Serial time (us) | Independent branches |
| --- | ---: | ---: |
| Six routed experts | 173.9 | 6 |
| Eight attention output groups | 69.9 | 8 |
| Q path | 45.2 | 1 of Q/KV fan-out |
| Shared expert | 30.7 | 1 alongside routed experts |
| KV path | 9.2 | 1 of Q/KV fan-out |

The idealized branch maxima are about 35.1 us for routed experts and 8.7 us
for attention output groups. Partitioning will make the real values larger,
but the measurement puts roughly 0.20--0.24 ms/layer of structural overlap
ahead of smaller task tuning. The next overlap loop should therefore add a
real fan-out/join representation and shape-derived SM partitions for the six
routed experts and eight output groups, retaining the single resident launch.

### Shape-partitioned attention output fan-out

The resident assembler now supports named wait/release groups. Multiple
producers can decrement one join counter and multiple consumers can observe
one ready counter; stages without an explicit group retain the existing strict
edge. The mechanism is entirely inside the per-SM instruction queues and
shifted barrier banks of the same resident launch.

After inverse RoPE, one 64-arrival ready counter releases eight output
quantize/GEMV branches. The shape policy assigns each branch a contiguous
19-SM partition. Each quantizer releases a private branch counter to its GEMV,
and all eight GEMVs contribute 19 arrivals to one 152-arrival join before
`o_rank` quantization. This replaces serial whole-grid assignments without
changing output layout or adding an inter-stage copy.

The one-layer tracked gate, job `20260811T023413Z-2601579`, preserved token
2835 and measured 0.499 ms median. The full tracked job
`20260811T023449Z-2604883` preserved token 14 and measured 25.289--25.748 ms,
median 25.643 ms. Its median-sample layer/reload/head spans were
22.848/1.217/1.408 ms. Compared with the post-fusion profile, dependency
counters fell from 555 to 506, allocator instructions from 2,191,707 to
2,047,571, and stores from 538,195 to 503,107.

Production job `20260811T023842Z-2623799` preserved token 14 over 30 samples
and measured 24.682--25.206 ms, median 24.952 ms. This improves the 26.571 ms
boundary by 1.619 ms (6.1%) and leaves 8.952 ms to the 16 ms target.

### Shape-partitioned routed-expert fan-out

The six routed experts now execute as independent subgraphs in six uniform
25-SM partitions (150 active SMs, with the two remainder SMs intentionally
idle). Each rank uses 16 SMs for input quantization, 25 for W1/W3, one for
SwiGLU, eight for middle quantization, and 25 for W2. Private ready counters
connect those phases, W1 and W3 jointly release the gate/up join, and all six
W2 tails contribute to one 150-arrival expert join. The shared expert waits on
that join. This is still one resident VDCores launch with only per-SM compute
and memory instruction queues.

A 25-SM row shard is larger than the routed TMA uint16 transaction limit, so
the shape policy bounds K=4096 projections to 24-row tiles and K=2048
projections to 56-row tiles. The checkpoint routing table contains one direct
LDU-resolved pointer field per SM and row tile. No compute task performs
pointer arithmetic, and no HBM inter-stage copy is added. Within each
partition, the first W1 tile retains the packed activation and scale in the
two LDU-local slot registers; later W1 tiles and every W3 tile reuse those
same shared allocations, with the last W3 tile releasing them.

The multi-tile route/W1/W3 smoke, job `20260811T025223Z-2691911`, was
bit-exact. The layer-0 checkpoint gate, job `20260811T025254Z-2694432`,
preserved token 2835 and measured 0.398--0.417 ms over ten samples, median
0.404 ms. Full exploration job `20260811T025318Z-2696481` preserved token 14
at a 21.310 ms median. The 30-sample production acceptance, job
`20260811T025524Z-2706785`, measured 21.525--22.131 ms, median 21.752 ms. The
resident image now uses 471 barriers, 1,089 compute instructions, and 3,572
memory instructions. This improves the 24.952 ms boundary by 3.200 ms (12.8%)
and leaves 5.752 ms to the 16 ms target.

The post-fan-out tracked job `20260811T030016Z-2731223` measured
22.244--22.912 ms over six correct tokens, median 22.436 ms. A representative
sample split into 19.753 ms of layers, 1.132 ms of active-bank reloads, and
1.369 ms for the vocabulary head. Every sample executed exactly 1,767,383
allocator instructions, 1,058,562/728,885 LDU commands, and 456,409 stores;
SM clocks stayed near 2.05 GHz. CSA layers remained near 0.39--0.42 ms while
HCA varied around a stable mode rather than increasing with loop position.
The progressive slowdown remains eliminated; the variation is queue and slot
backpressure on fixed work, not accumulated state or a system-frequency loss.

Stage profiling now marks only the aggregate routed-expert join. Inserting a
marker within an earlier rank would place it in every SM queue before later
sibling branches and destroy the overlap being measured. DAG-safe job
`20260811T030329Z-2749377` attributed a 0.306 ms layer span to 0.148 ms of
attention and 0.158 ms of FFN. Routed experts take 78.7 us after fan-out, while
the still-serial shared expert takes 33.1 us. The Q and KV paths take 45.5 and
9.2 us, and the eight-way attention-output group now takes 27.5 us. These are
the next broad overlap/adaptive-fusion candidates.

### Shape-weighted Q-prefix/KV fan-out

The independent query-prefix and KV projection chains now fan out immediately
after the hidden-state FP8 quantizer and join only before `q_b`, the first
query consumer that also requires completed KV state. Their contiguous SM
partitions are derived from output shape: the 1,024-row Q branch receives 101
SMs and the 512-row KV branch receives 51 SMs. Q projection, normalization,
and quantization remain on the first partition; KV projection, normalization,
and RoPE remain on the second. This is queue-level placement inside the same
resident launch, with no intermediate copy or global pointer in a compute
task.

Matched production task probes measured 8.160 us for the Q projection at
M1024/K4096 on 101 SMs and 8.064 us for KV at M512/K4096 on 51 SMs. The
one-layer checkpoint gate, job `20260811T033437Z-2901475`, preserved token
2835 over 20 samples at a 0.401 ms median. Full exploration job
`20260811T033501Z-2902408` preserved token 14 at an 18.338 ms median.

The 30-sample production acceptance, job `20260811T033838Z-2906723`,
preserved token 14 and measured 18.343--19.036 ms, median 18.614 ms. The
resident image uses 464 barriers, 1,089 compute instructions, and 3,564 memory
instructions. This improves the 21.752 ms boundary by 3.138 ms (14.4%) and
leaves 2.614 ms to the 16 ms target. Samples do not grow with token iteration,
so the overlap did not reintroduce the earlier progressive queue-pressure
failure.

### Parallel active-bank reload

Loop barrier restoration is now distributed across all 152 LDU0 handlers.
Each handler restores a disjoint strided slice of the active shifted bank,
then a single device-scope release/acquire counter forms the memory-only
completion barrier before either local LDU port advances. There is no compute
participant, IssueBarrier, or thread fence. The first attempted publication
scheme let a designated final arrival publish a separate phase word; it
passed the two-layer probe but stalled on a repeated full-model launch and was
rejected. The retained count-threshold form makes every LDU0 handler observe
the completed phase directly.

Tracked full-checkpoint job `20260811T040008Z-2932683` preserved token 14 and
measured 17.941 ms. Active-bank reload total fell from about 1.11 ms to 0.065
ms; layer bodies were 16.332 ms and the head was 1.337 ms. The shorter global
pause also reduced downstream queue pressure, so the gain is larger than the
reload interval alone.

Production job `20260811T040350Z-2936582` preserved token 14 over 30 samples
and measured 16.284--16.478 ms, median 16.397 ms. This improves the 18.614 ms
boundary by 2.217 ms (11.9%) and leaves 0.397 ms to the 16 ms target. The
sample sequence remains flat with iteration, confirming that the earlier
progressive system/queue overhead has not returned.

### FP8 vocabulary head and queue-native argmax

The immutable BF16 vocabulary weight is now preprocessed once, after checkpoint
residency, into the same E4M3/UE8M0 block-128 format used by the other FP8
linears. The 129,280-by-4,096 converted weight occupies 0.493 GiB and took
0.697 seconds in the production acceptance run; neither cost is part of TBT.
Each token quantizes the normalized hidden state and uses the existing
shape-sharded FP8 GEMV rather than 850 scalar BF16 row groups.

Argmax remains inside the resident queues. The 152 GEMV shards feed 152
allocator-shared BF16 ranges to generic partial reducers; ordinary STU writes
one 16-byte value/index record per shard, and one final shared-memory reducer
emits the int64 token. Ties choose the lowest vocabulary index. Compute tasks
see only allocator slots: there is no compute-side global pointer, indirect
store, host readback, extra launch, IssueBarrier, or thread fence.

The one-layer full-vocabulary checkpoint gate, job
`20260811T051003Z-3011199`, emitted token 78571 exactly matching an independent
BF16 vocabulary GEMV and measured 0.817 ms for the layer plus head. Full-model
exploration job `20260811T051031Z-3011830` emitted token 14 at a 15.407 ms
median. Production job `20260811T051221Z-3014131` then preserved token 14 over
30 samples and measured 15.103--15.290 ms, median 15.222 ms. This improves the
16.397 ms boundary by 1.175 ms (7.2%) and clears the 16 ms target by 0.778 ms.
The first and last samples were 15.220 and 15.237 ms, so the prior progressive
slowdown remains absent. The selective SM100a image uses 75 registers, nine
barriers, no spills, and a compact repeated program of 467 barriers, 297
compute instructions, and 1,301 memory instructions.

Tracked one-layer job `20260811T051822Z-3024096` attributes 0.302--0.322 ms to
the layer body and 0.328--0.330 ms to the complete FP8 head and argmax. The
routed-expert join remains the largest layer group at 79--81 us; the query/KV
and eight attention-output branches are already overlapped. A full tracked
prime completed, but the instrumented image stalled on its second 43-layer
launch. The exact benchmark orphan was identified and removed without touching
other worker jobs. This is isolated from the zero-instrumentation production
image, which remains repeatable over 30 launches.

Two-layer tracked control job `20260811T052418Z-3031738` remained repeatable:
the two SWA/hash layers measured 0.306--0.326 ms each, active-bank reloads were
2.6--2.8 us each, the head was 0.327--0.330 ms, and clocks stayed at
2.03--2.05 GHz. This confirms that neither layer placement nor the distributed
reload introduces progressive work at small repeat count; full-model profile
instrumentation itself needs a separate bounded repair before another all-layer
counter trace.

An adaptive-fusion experiment retained eight per-SM FP8-head maxima in one
shared slot with `RegStore`/`RegLoad`, eliminating the full-logit round trip and
one argmax stage. It was token-correct and a 30-sample full run measured a
15.153 ms median, but a controlled revision A/B rejected it: committed baseline
`b9f5357` measured 0.670 ms median for one layer plus head with a tight
0.650--0.686 ms range, while the fused candidate repeated at 0.773 ms median
and was bimodal over 0.631--0.806 ms. The apparent full-model gain was queue
phase noise, so all fused source was reverted rather than retained.

## Performance target and optimization phases

The performance target is 16 ms TBT for the complete 43-layer network plus
head on one GPU, corresponding to roughly 0.35-0.4 ms per layer after allowing
for the head and resident-loop overhead. The first measured baseline was
79.53 ms; the accepted production boundary is now 15.222 ms. Further
optimization decisions remain driven by end-to-end attribution rather than
isolated best-case task numbers.

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

## Fixed-context framework comparison (2026-08-11)

`benchmarks/blackwell_fixed_context_decode.py` now runs statistical one-GPU,
batch-one, context-128 decode baselines with three warmups and 30 timed second
tokens. The strict SGLang path accepts an explicit MoE runner so the NVFP4
checkpoint uses `flashinfer_mxfp4`; its default Triton runner is not compatible
with this checkpoint's routed-expert hidden-size/layout contract. SGLang also
requires the upstream DeepSeek FlashMLA package used by its release image; the
dedicated SGLang environment has revision
`15f13e5030374295491c5ce31b02d7e63a7772c6` installed without modifying the
system environment or checkpoint.

The current results on the same checkpoint worker are:

| Runtime | Job | Min (ms) | Median (ms) | P90 (ms) | Max (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| VDCores production | `20260811T051221Z-3014131` | 15.103 | 15.222 | - | 15.290 |
| vLLM 0.23.0 | `20260811T055307Z-3067569` | 5.661 | 6.016 | 6.085 | 6.148 |
| SGLang 0.5.12.post1 | `20260811T062825Z-3124544` | 6.855 | 7.322 | 7.345 | 7.355 |

VDCores is currently 2.53x the vLLM median and 2.08x the SGLang median. It
would need another 9.206 ms versus vLLM or 7.900 ms versus SGLang to reach
those observed medians. This is not yet a semantic parity comparison: the
framework harness decodes after a real 128-token prefill, whereas the current
resident VDCores performance program is the position-zero/cache-empty flow.
Framework token IDs therefore are not expected to match token 14, and the
ratios are a structural prioritization signal rather than a parity claim.

The per-layer VDCores profile still makes the next loop unambiguous: a typical
layer body is about 0.30--0.32 ms and the routed-expert join is its largest
measured group at 79--81 us. Establish a matched M/K/top-6 NVFP4 task
comparison against FlashInfer before changing production, then prioritize a
Blackwell-native or better-prepacked expert path only if the task gap explains
a material fraction of complete-token time. Dense FP8 and real-context
attention follow; small barrier or instruction-count tuning stays behind
these board-level gates.

The first matched NVFP4 board exploration completed that gate. The VDCores
static raw-checkpoint task measured 5.504 us for M2048/K4096 on 128 SMs and
5.664 us for M4096/K2048 on 128 SMs. FlashInfer 0.6.12 CUTLASS CUDA-graph
amortized medians were 4.579 us and 3.656 us respectively in jobs
`20260811T065145Z-3187993` and `20260811T065325Z-3192804`. The graph result is
the relevant comparison because VDCores is already inside one persistent
launch; FlashInfer's independently launched event medians include much larger
host/kernel-launch effects.

`deepseek_v4_nvfp4_gemv.py --routed-tiles` now exercises the actual dynamic
LDU pointer resolution, uint16-bounded row tiling, and retained activation used
by a production expert partition. On the production 25-SM share, job
`20260811T065825Z-3241097` measured 19.008 us for M2048/K4096 and job
`20260811T065849Z-3244549` measured 17.408 us for M4096/K2048; both were
bit-exact to the raw-checkpoint quantized reference. The respective 50-SM
medians were 11.856 and 10.656 us, showing ordinary SM scaling rather than a
dominant routed-address penalty.

These values explain the 79--81 us top-6 join: each 25-SM branch executes two
up/gate projections, SwiGLU and requantization, then one down projection. A
small route-address or barrier change cannot close the framework gap. The next
kernel proof should instead stream prepacked K tiles through LDU into native
SM100 block-scaled UMMA, overlap tile loads with tensor-core issue, and avoid
the current native oracle's synchronous in-task checkpoint reformat. Weight
data and scales should share a load transaction when the preprocessed layout
demonstrates a task and end-to-end win without duplicating the resident model.

That native layout proof now passes. Setup preprocessing emits one combined
18,432-byte M128/K256 weight tile (16,384 bytes FP4 data plus 2,048 bytes native
scales) and one 3,072-byte broadcast N8/K256 activation tile. The token-time
task consumes one fixed-port LDU load per operand, moves native scales to TMEM,
and issues block-scaled SM100 UMMA from a compact runtime K-tile loop. It has no
global compute pointer, indirect store, issue barrier, thread fence, or
inter-stage HBM copy. The narrow image uses 44 registers, nine barriers, an
80-byte stack frame, and no spills; 81 focused tests pass.

Random-scale outputs were exact at M128/K256 and M128/K4096. On production
projection shapes, combined-layout jobs `20260811T073746Z-3577928` and
`20260811T073804Z-3580496` measured 15.936 us for M2048/K4096 on 16 SMs and
8.800 us for M4096/K2048 on 32 SMs. These improve on the current routed
25-SM medians of 19.008 and 17.408 us by 16.2% and 49.4%, respectively. They
remain slower than FlashInfer's whole-GPU graph-amortized kernels and are a
kernel milestone, not a network-TBT result.

The weight conversion is immutable setup work and the combined representation
does not exceed the raw FP4-data-plus-scale byte count, so production should
replace rather than duplicate resident expert storage. Activation layout is
token-dependent: the task benchmark preprocesses it outside the measured GEMV
span, so production must make the NVFP4 quantizer emit the native activation
tile directly or hand it off through shared memory. Six-expert placement also
needs a shape-derived wave assignment: W1/W3 require 96 M128 tiles and fit
concurrently, while W2 has 192 tiles and cannot assign one tile to every SM at
once on a 152-SM GPU. Do those structural integrations and measure the complete
routed-expert group before further UMMA instruction tuning.

The token-dependent activation gap is now closed at the task boundary. A new
quantizer consumes BF16 plus the selected expert's scalar input scale and emits
the final 3,072-byte N8/K256 native tile directly. Quantized FP4 bytes follow
Blackwell's 128-byte TMA XOR swizzle and scale bytes are written in the native
SFB layout; there is no raw FP4/scale output and no layout-copy stage. Precise
round-to-nearest division makes its result byte-exact to the independent
ModelOpt reference, including scale-boundary cases that fast reciprocal
multiplication rounded differently.

Job `20260811T081132Z-3810639` measured the K4096 quantizer at 2.880 us on 16
shape-derived K256 SMs, versus the previous raw-layout 3.680 us task baseline,
and its following M2048/K4096 GEMV measured 15.728 us. Job
`20260811T081156Z-3813722` measured K2048 quantization at 2.752 us on eight SMs
and M4096/K2048 GEMV at 8.768 us. Both native layouts and final GEMV outputs
were exact. The selective image uses 48 registers, nine barriers, an 80-byte
stack frame, and no spills; 82 focused tests pass. Routed scale lookup,
top-six wave placement, resident weight replacement, and complete-group TBT
remain the next acceptance boundary.

The adaptive-fusion and routed-base proof now removes the next structural
costs without changing that acceptance boundary. A bulk activation mode gives
each compute task one contiguous shared allocation for all K256 native tiles;
the runtime K loop derives each tile address by a fixed shift. M2048/K4096
dropped from 15.728 to 9.536 us in job `20260811T081644Z-3848595`, and
M4096/K2048 dropped from 8.768 to 5.984 us in job
`20260811T081707Z-3852210`. A W1/W3-style pair retains that allocation with
`TmaLoadReg1D` and consumes it with `RegLoad`, so no data is copied: job
`20260811T081812Z-3861409` measured both projections together at 18.016 us,
1.056 us below two independent bulk loads.

The first routed-native version repeated the HBM route and pointer-table read
for every K tile and regressed W1 to 14.880 us in job
`20260811T082144Z-3882817`; it was rejected. The accepted LDU path resolves
the route once into one port-local scalar address register while loading the
first combined weight tile, then issues fixed compile-time offsets for the
remaining K tiles. Compute still sees only shared-slot masks. The first routed
load on both LDU ports waits for route completion, with no `IssueBarrier`,
thread fence, indirect store, global compute pointer, or inter-stage copy.

Routed W1 then measured 9.728 us in job `20260811T082826Z-3939218`, only
0.192 us over static bulk. Shape-derived placement assigns contiguous M128
tiles to any `1..M/128` SMs; routed W2 therefore maps 32 tiles over its 25-SM
production share, with seven SMs processing a second tile through local
activation reuse. Job `20260811T082847Z-3941836` measured 10.784 us, 38.1%
below the prior routed CUDA-core 17.408-us task. The selective image remains at
48 registers, nine barriers, an 80-byte stack, and no spills; 85 focused tests
pass. This accepts the adaptive-fusion/routed-address primitives. Six-route
concurrency, resident replacement, and the complete routed-expert span remain
the next milestone.

The six-route structural proof is now complete in
`deepseek_v4_nvfp4_top6.py`. All six expert branches occupy disjoint
shape-derived 25-SM partitions in one VDCores launch. Each branch has its own
route-to-input-quant-to-W1/W3-to-SwiGLU-to-middle-quant-to-W2 dependency
chain, so no top-six-wide barrier separates parallel phases; only the final
weighted reduction joins all branches. W1 and W3 return their M128 shards to
port-local shared registers, and a same-SM bounded SwiGLU consumes them. Thus
gate and up vectors never enter HBM. Only the 4-KiB fused middle vector is
stored for the required 16-producer to 8-quantizer/25-W2-SM repartition.

The first correct flow, job `20260811T084739Z-4104921`, measured 80.736 us.
Internal counters in job `20260811T085058Z-4132334` showed both LDU ports
spending about 54% of the grid envelope on dependency waits and STU waiting
82.8%. Shared shard fusion reduced the non-instrumented median to 73.632 us in
job `20260811T085921Z-9836`. Profiling then exposed a VDCores runtime bug:
`RegStore` marked retained slots as no-writeback, but the completion queue
still sent 192 useless commands through STU. Honoring that marker reduced
store commands from 627 to 435; the later `RegLoad` consumer remains solely
responsible for freeing the shared slots.

The remaining critical-path outlier was routing. A serial thread-zero top-six
scan took 33.664 us, while hash routing isolated score transformation and I/O
at 4.064 us in job `20260811T091843Z-173370`. The accepted score route keeps
two candidates per compute thread and performs six bounded block-wide maximum
reductions with deterministic expert-id tie breaking. It uses compute-group
synchronization only, no memory-thread barrier, thread fence, or full expert
unroll. The route phase fell to 7.680 us in job
`20260811T092200Z-198349`.

The final non-instrumented dynamic-routing run, job
`20260811T092223Z-204171`, was exact for native activation layouts, all six
selected expert projections, fused middle values, W2 outputs, route weights,
and weighted reduction. It measured 45.056/45.536/46.368 us
min/median/max, with a 46.816-us whole-kernel median and a 9/75 maximum
compute/memory instruction image. This is 43.6% below the first 80.736-us
flow. The selective image uses 62 registers, nine barriers, an 80-byte stack,
and no spills; 88 focused tests pass. If the 35.2-us delta carries through all
43 checkpoint layers, the prior 15.222-ms network median projects to about
13.71 ms, but that remains an estimate until native resident weights and the
layered route opcodes are integrated and re-profiled.

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

## Native resident top-six milestone (2026-08-11)

The resident loader can now replace each routed expert's raw packed FP4 data
and E4M3 scales with the combined SM100 UMMA tile layout while loading a shard.
The native layout is exactly the same size as the two raw tensors. Conversion
uses a small temporary linear and overwrites their contiguous resident span;
it never duplicates the 153-GiB checkpoint. The durable source remains
`/mnt/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4`. A CUDA converter was
byte-exact against the queued VDCores prepack oracle for multiple M128 and
K256 tiles, including the resident overwrite path.

Repeated layer bodies now have indirect/layer routed-base LDU opcodes. The
first native weight tile resolves the layer and expert in LDU and caches its
base on that load port; later K tiles use fixed offsets. Compute continues to
see allocator-owned shared slots only. There is no compute global pointer,
address queue, indirect store, issue barrier, thread fence, or token-time
prepack/copy stage.

The production FFN assigns six disjoint 25-SM branches from the model shape.
Each branch quantizes its native input on 16 SMs, retains W1/W3 M128 outputs in
separate port-local shared registers, fuses them on those 16 SMs, materializes
only the required 2,048-value middle repartition, quantizes it on eight SMs,
and executes W2 over 25 SMs. Seven W2 SMs process a second M128 output tile.
Only the weighted expert reduction joins all six branches.

One-layer job `20260811T094703Z-432626` preserved the independent full-vocab
token `78571` and measured 0.645344--0.670272 ms, median 0.653664 ms. Full
job `20260811T094747Z-441094` loaded 153.379 GiB in 72.227 seconds, retained
29.912 GiB free, preserved token `14`, and measured
13.019616/13.040736/13.052544 ms min/median/max over ten samples. This is
2.181376 ms (14.3%) below the accepted 15.222112-ms baseline and 2.959264 ms
under the 16-ms target. Samples remain flat, so the prior progressive slowdown
has not returned.

The one-launch repeated program shrank from 467 barriers and 297/1,301 maximum
compute/memory instructions to 425 barriers and 269/1,269 instructions. The
29-op SM100a image uses 75 registers, nine barriers, a 96-byte stack, 2,448
bytes static shared memory, and no spills. The next loop is overlap profiling,
especially shared-expert execution versus the routed branches, followed by a
fresh per-layer slowdown/counter audit before further kernel tuning.

## Routed/shared FFN overlap milestone (2026-08-11)

Internal stage frontiers showed that the native top-six routed branches took
about 45--54 us after routing, while the shared expert was still serialized
behind their join for another 31--38 us.  The shared path was therefore an
independent critical-path tail, not an NVFP4 task-kernel deficit.  The FFN now
derives a 96/56-SM split directly from the model shape: each of the six routed
experts receives 16 SMs (one M128 W2 tile per SM for each wave), and the FP8
shared expert executes concurrently on SMs 96--151.  Both paths join only
before the weighted expert reduction.

The dependency graph publishes the normalized FFN input once.  Routing and
the hidden FP8 quantizer wait on that boundary independently; only the shared
expert waits for hidden quantization.  Routed native activation quantization
continues directly from routing.  This removes the old routed-to-shared hard
join without adding a launch, global pointer, copy, issue barrier, thread
fence, or cross-role barrier.

Tracked one-layer job `20260811T095819Z-543150` preserved reference token
`78571` and measured a roughly 229--234 us layer span despite marker overhead,
down from about 264 us before the split.  The joined expert-output frontier was
51--53 us after routing/dispatch, showing that the shared path fits beneath the
routed path instead of adding its former serial tail.  A non-stage-profile
one-layer run measured 0.579808/0.586160/0.597408 ms min/median/max.

Production full-network job `20260811T100727Z-618515` loaded 153.379 GiB from
the durable `/mnt` checkpoint in 70.884 seconds, retained 29.912 GiB free, and
emitted exact token `14`.  With three warmups, 30 samples measured
11.591968/11.652480/11.737472 ms min/median/max.  This is 1.388256 ms (10.6%)
faster than the 13.040736-ms native-resident milestone, 3.569632 ms (23.5%)
faster than the original 15.222112-ms accepted baseline, and 4.347520 ms under
the 16-ms target.  The first- and last-15 medians were 11.650720 and
11.662976 ms, while the fitted sample slope was -0.000317 ms/iteration; there
is no progressive slowdown.

The repeated 43-layer image now has 418 barriers and 253/1,197 maximum
compute/memory instructions, down from 425 and 269/1,269.  The production
29-op SM100a binary remains spill-free at 75 registers, nine barriers, a
96-byte stack, and 2,448 bytes static shared memory; all 88 focused tests pass.
This is an accepted overlap milestone; future tuning should profile attention
and dense FP8 work against matched FA4/FlashInfer/Triton primitives rather
than fine-tune the now hidden shared-expert path.

## Full-layer counter audit and native FP8 board (2026-08-11)

Tracked full-network job `20260811T101318Z-667822` emitted token 14 in
11.647648 ms. Its internal span was 11.559520 ms: 11.206240 ms across the 43
layers, 0.062720 ms in layer reloads, and 0.290560 ms in the head. HCA score
layers stayed around 0.236--0.258 ms and CSA score layers around
0.279--0.285 ms. Reloads remained 2.624--2.880 us, all SM-frontier spreads
were at most 0.448 us, and physical-SM clocks stayed 2.018--2.039 GHz. The
first/last sample audit and these counters independently reject cumulative
architectural or system slowdown. The remaining 58--59% compute/allocator
wait and 57--58% LDU dependency wait are per-layer backpressure, not growing
state.

Representative position-zero layer job `20260811T101516Z-684733` measured a
234.688-us body. Attention consumed 137.408 us and FFN 97.280 us. Sparse SWA
itself was only 5.408 us; the material attention costs were q_b, o_a, and o_b
FP8 projections. Matched isolated VDCores scalar FP8 medians were 7.680 us for
M1024/K4096, 23.424 us for M32768/K1024, 21.184 us for M1024/K4096 under the
production 19-SM placement, and 21.248 us for M4096/K8192. Graph-amortized
DeepGEMM medians were respectively 5.9816, 6.5608, 5.9952, and 8.9768 us.

The native SM100 MXF8 proof preprocesses each immutable weight into combined
M128/K128 data-plus-scale records and directly quantizes BF16 into combined
N8/K128 activation records. Token-time compute receives only allocator-owned
shared addresses and streams both operands in bounded K loops. There is no
raw/global compute pointer, intermediate layout copy, indirect store, issue
barrier, thread fence, or full K unroll. FP8 activation records require a
2,048-byte stride: the natural 1,536-byte payload put every odd UTCCP scale
descriptor on only a 512-byte boundary and corrupted K tile 1. Padding to the
next 1,024-byte scale boundary made weight data, weight scales, direct
activation layout, and BF16 output exact for K128 through K8192. Zero blocks
also bypass floating division so fast-math FTZ cannot create NaN FP8 data.

The first integration retained one 16-KiB activation allocation while eight
16,896-byte weight records streamed. That is not a valid VDCores slot plan:
the activation consumes two 8-KiB slots and each weight consumes three, so the
load window can demand 26 slots from the 24-slot arena. It intermittently
deadlocked after 3--11 repeated full-depth launches. Neither removing
`LoadReg` reuse nor changing the LDU dependency poll fixed it. The accepted
schedule instead loads four activation records as one 8-KiB, one-slot chunk,
streams at most four three-slot weights, then releases the chunk. Its peak
operand footprint is 13 slots and it completed the repeated-depth and
production liveness gates.

The native task is selectively profitable. Exact production-shape medians are
18.848 us for M1024/K4096, 11.296 us for M32768/K1024, and 39.600 us for
M4096/K8192; direct activation quantization is about 1.8--2.0 us. Thus q_b is
2.07x faster than the existing 23.424-us VDCores task, while q_a/o_a and o_b
must retain their scalar paths. Barrier batching was rejected after regressing
those three shapes to 36.896, 13.056, and 73.184 us. Alternating weight loads
over both LDU ports was neutral, and 128 versus 152 q_b SMs was also tied;
these variants are not part of the accepted policy. The spill-free selective
image uses 44 registers, nine barriers, a 112-byte stack, and 2,192 bytes of
static shared memory. Integrate only q_b, then require an exact full-network
measurement before accepting an end-to-end milestone.

## Resident native FP8 q_b milestone (2026-08-11)

The resident loader now converts only main-attention and indexer q_b checkpoint
linears into native M128/K128 records while each shard is loaded. Raw E4M3 data
and UE8M0 scales are staged in their final resident span, converted with one
shape-reused temporary, and overwritten in place. The complete checkpoint is
never duplicated. Native q_b storage raises the resident image only from
153.379 to 153.426 GiB and leaves 29.795 GiB free on the 184-GiB B300.

The production graph directly quantizes the normalized 1,024-wide q rank into
eight 2,048-byte native activation records. Main q_b uses 152 SMs; the
8,192-row index q_b uses 64 SMs. Both use LDU1 for bounded one-slot activation
chunks and LDU0 for combined data-plus-scale weight records. The final bounded
task is exact at M32768/K1024 and measures 12.032 us in its task benchmark,
versus 23.424 us for the scalar VDCores q_b task. q_a, o_a, o_b, shared
experts, and the vocabulary head remain on their measured faster scalar FP8
paths.

Production job `20260811T123330Z-2057880` used one GPU, one persistent launch,
the durable `/mnt` checkpoint, three warmups, and 30 timed tokens. It preserved
reference token 14 and measured 11.459520/11.593600/11.739296 ms
min/median/max. This improves the prior accepted 11.652480-ms median by
0.058880 ms (0.51%) and is 4.406400 ms under the 16-ms target. First/last-15
medians were 11.594176/11.593024 ms; no progressive slowdown or liveness stall
remained. The one-launch image has 418 barriers and 245/1,213 maximum
compute/memory instructions. Its 32-op SM100a binary uses 70 registers, nine
barriers, a 96-byte stack, 2,448 bytes static shared memory, and no spills.

## Shape-sharded mHC post milestone (2026-08-11)

The refreshed position-zero stage profile in job
`20260811T124820Z-2243628` measured a 227.040-us layer body. Both mHC-post
updates were still one-SM elementwise tasks and cost 11.872 and 11.648 us.
The task has no cross-dimension dependency: each output dimension reads the
same four post coefficients and 4x4 combination matrix, then independently
updates four residual streams.

The accepted schedule derives 32 SMs from 4,096 / 128 dimensions. Each SM
LDU-loads one 128-value branch shard, four matching residual shards, and the
small fixed coefficients into normal shared slots. Compute updates the four
local output shards with ordinary loops, and STU writes those shards. The
working set is 11 of 24 slots. There is no compute global pointer, issue
barrier, thread fence, inter-stage copy, or per-layer unroll. The same task
opcode now carries only its local width.

Isolated exact medians were 10.480 us for the original one-SM task in job
`20260811T125149Z-2282925`, then 7.200/7.008/7.104 us for 16/32/64 SMs in
jobs `20260811T125712Z-2345603`, `20260811T125650Z-2341476`, and
`20260811T125740Z-2350648`. The natural 128-dimension/32-SM shape won the
coarse screen. In one-layer job `20260811T125814Z-2356950`, the two mHC-post
boundaries fell to 6.688 and 6.848 us and the layer body fell to 215.872 us,
saving 11.168 us from the matched tracked baseline.

The complete tracked job `20260811T125853Z-2365516` preserved token 14 and
measured 11.324192--11.374656 ms across five samples. HCA/CSA family bands,
2.2--2.8-us reloads, per-SM frontiers, and roughly 2.025-GHz clocks remained
flat across all 43 layers; the previously reported progressive slowdown is
still absent.

Production job `20260811T130421Z-2428119` used one B300, one persistent
launch, the durable `/mnt` checkpoint, three warmups, and 30 timed tokens. It
preserved token 14 and measured 11.087136/11.178160/11.298816 ms
min/median/max. First/last-15 medians were 11.176416/11.184384 ms and the
linear slope was -0.000497 ms per iteration. This saves 0.415440 ms (3.58%)
from the native-q_b milestone and is 4.821840 ms under the 16-ms target. The
queue image has 418 barriers and 253/1,301 maximum compute/memory
instructions. The 32-op production binary remains spill-free at 70 registers,
nine barriers, a 96-byte stack, and 2,448 bytes static shared memory.

## Resident context-128 shape gate (2026-08-11)

`deepseek_v4_resident_one_launch.py --context-length 128` now executes the
position-127 decode shape in the existing single persistent launch. SWA, CSA,
and HCA consume 128, 160, and 129 attention candidates respectively. Every
compressed layer runs the current-token compressor boundary; CSA also runs its
index compressor, Hadamard transform, 32-row score, and top-k selection. The
current window KV and compressed rows are produced by the layer inside the
launch. The immutable prefix rows are deterministic resident test data, so
this is a workload/correctness gate and not prompt-semantic parity with the
framework baselines.

The compressor pool accepts a separately loaded current row and fuses the
position-dependent APE into that final row. This avoids an add/copy stage and
keeps every operand in the ordinary LDU/shared-slot protocol. The segmented
ratio-4 path is exact against the PyTorch reference. Job
`20260811T155858Z-294886` measured 9.696 us for its eight-row/512-wide form and
115.424 us for the serial 128-row form, immediately identifying ratio-128
pooling as a structural kernel target.

Full-network job `20260811T160651Z-391620` used one GB200, one launch, three
warmups, and five measured samples. It emitted repeatable token 5, and the FP8
head agreed with the independent BF16 head. Samples were
29.872768--30.415264 ms with a 30.149376-ms median. The context-one regression
job `20260811T160854Z-417208` still emitted reference token 14 in 11.026720 ms.
The context-128 image has 451 barriers and 255/1,313 maximum compute/memory
instructions. `benchmarks/deepseek_v4_resident.ops` selects its 30 operators;
the SM100 binary uses 84 registers, nine barriers, a 96-byte stack, 2,448 bytes
of static shared memory, and no spills.

The strict target is 5.414155 ms, ten percent below vLLM's 6.015728-ms
context-128 median; this also beats the corresponding 6.589777-ms SGLang
target. The 30.149376-ms shape-gate baseline must first be decomposed by
representative SWA/CSA/HCA stage and memory/compute idle counters. In
particular, replace the serial ratio-128 pool and row-at-a-time sparse
attention with board-competitive mechanisms before changing narrow runtime
parameters.

## Context-128 contiguous-attention traffic milestone (2026-08-11)

At positions through 127, CSA's compressed cache has at most 32 rows while the
selection cap is 512. SWA, HCA, and CSA therefore all attend the complete
contiguous candidate cache; top-k may permute CSA rows, but softmax attention
is invariant to that permutation. The resident schedule now proves
`compressed_selected == compressed_rows` before selecting its contiguous
path. Longer shapes retain the indexed task.

A coarse 1/4/8/16-row screen selected 16 rows as the automatic crossover:
scalar wins at one and four rows, the two paths tie at eight, and grouped
traffic is clearly ahead by 16. This keeps the position-zero path on scalar
attention without tuning a narrow threshold.
Full context-one regression job `20260811T164157Z-807590` emitted reference
token 14 and measured an 11.045088-ms median across five samples.

The accepted task uses four compute warps for four row scores, evaluates four
online-softmax updates together, and transfers the four adjacent BF16 KV rows
with one 4-KiB fixed-address TMA. A memory `RepeatM` advances by four rows, so
the layer family keeps a compact queue instead of unrolling row loads. Compute
receives only allocator-owned slot masks; it has no index or global pointer,
copy stage, issue barrier, joint memory/compute barrier, or thread fence.

On one allocator-managed `.24` GB200, task job
`20260811T163112Z-683642` measured scalar/contiguous medians of
100.032/64.224 us at 128 rows, 106.624/65.344 us at 129 rows, and
124.000/78.656 us at 160 rows. All outputs passed the independent BF16
reference. The corresponding matched Triton medians are 26.263, 29.354, and
32.582 us, so a 2.2--2.4x task gap remains.

The representative CSA stage gate (`20260811T163146Z-690318`) passed and put
the 160-row attention boundary at roughly 78.5 us. The complete tracked job
`20260811T163220Z-696275` used one GPU, one launch, three warmups, and five
samples; it emitted stable token 5 and measured
21.463552/21.530111/22.030048 ms min/median/max. The same-build scalar control
`20260811T163416Z-718238` measured a 27.485537-ms median, so grouped contiguous
traffic saves 5.955426 ms (21.7%). Relative to the original 30.149376-ms shape
gate, the cumulative reduction is 28.6%.

The selected 31-op tracked image uses 80 registers, nine barriers, a 208-byte
stack, 2,448 bytes of static shared memory, and no spills. Representative full
token counters still show 74--75% compute m2c wait, 75--77% LDU dependency
wait, and 76--77% allocator slot stall after normalization to the SM-grid
envelope. Those large independent idle fractions prioritize dependency and
placement mechanisms over narrow softmax tuning. The strict 5.414155-ms target
is not yet met.

## Exhaustive CSA-selection elimination (2026-08-11)

`DeepSeekV4LayerDecodePlan.requires_index_selection` is true only when CSA has
more compressed rows than its 512-row cap. If the compressed cache fits under
the cap, every row is selected and the index query, query RoPE/Hadamard,
head-weight projection, score, and top-k cannot change attention. The schedule
now removes those six stages while retaining the index compressor that creates
future cache state. A forced mode preserves the old path for A/B checks.

Representative layer-2 job `20260811T164730Z-878540` emitted the same token
and logit range in both modes. Auto/forced medians were 0.532480/0.613056 ms;
forced query preparation, score, and top-k boundaries were 11.104, 4.928, and
32.608 us. The auto layer contains 98 rather than 104 logical stages.

Full tracked job `20260811T164840Z-892133` used one `.24` GB200, one launch,
three warmups, and five samples. It emitted stable token 5; samples were
20.070593/19.114304/19.982817/20.051071/20.516159 ms, with a 20.051071-ms
median. This saves 1.479040 ms (6.87%) from contiguous attention and 10.098305
ms (33.49%) from the initial context-128 gate. The looped program drops from
4,030 to 3,904 logical stages, 448 to 430 barriers, and 223/1,105 to 213/1,059
maximum compute/memory instructions.

Tracked samples still report 71--73% compute m2c wait, 72--76% LDU dependency
wait, and 74--75% allocator slot stall. The semantic elimination is accepted,
but the 20.051071-ms result remains 3.70x the strict 5.414155-ms target.

## Packed/sharded ratio-128 pooling (2026-08-11)

Ratio-128 compressor pooling is independent by output dimension. Its history
is now preprocessed once into contiguous
`[4 shards, 16 blocks, 8 rows, value/score, 128 dims]` storage. Each 8-KiB TMA
carries both FP32 operands for eight rows, and four SMs own disjoint 128-wide
output shards. The dynamic value, score, and APE tail remains in its producer
layout and is loaded directly, so there is no runtime repack or inter-stage
copy. A normal memory repeat advances blocks; compute contains ordinary loops
and only compute-group synchronization.

Functional job `20260811T165754Z-1004507` passed an independent BF16
reference. Scalar/packed ratio-128 medians were 119.264/11.456 us, a 10.4x
task speedup. The ratio-4 control stayed on its existing 9.888-us segmented
path. The shape policy derives four SMs from 512 / 128 rather than using a
tuned occupancy constant.

Representative HCA job `20260811T165901Z-1017040` emitted identical token and
logit range. Packed/scalar medians were 0.476160/0.924352 ms, and the pool
boundary fell from 136.000 to 13.408 us. Full tracked job
`20260811T165935Z-1023821` used one `.24` GB200, one launch, three warmups, and
five samples. It emitted stable token 5 and measured
14.873792/14.880672/14.990528 ms min/median/max. This saves 5.170399 ms (25.8%)
from exhaustive-selection elimination and 15.268704 ms (50.6%) from the
initial context-128 gate.

The 32-op tracked build is spill-free at 78 registers, nine barriers, a
208-byte stack, and 2,448 bytes static shared memory. Grid-normalized compute
m2c wait is now about 63%, allocator slot stall 66%, and LDU dependency wait
64--66%. The accepted 14.880672-ms result beats the earlier 16-ms objective
but remains 9.466517 ms above the strict 5.414155-ms framework target.

## Barrier-free queued-step profiling (2026-08-11)

The context-128 optimization boundary is revision `cb27544`. A fresh tracked
full-network run measured a 15.154208-ms CUDA median and a 15.055872-ms
internal span: 14.664736 ms in 43 transformer layers, 0.063520 ms in reloads,
and 0.327616 ms in the output head. HCA-score and CSA-score layers had stable
0.320496/0.364704-ms medians, so this boundary has no iteration-dependent
layer slowdown.

Diagnostic builds can wrap each queued stage's compute sequence in a paired
`ProfileStep`. The begin marker stores low global-timer bits plus the current
cumulative M2C-pop wait; the end marker replaces that slot with elapsed time
and wait delta. It records only on SMs that execute compute for the stage.
There is no new producer/consumer dependency, memory participant, issue
barrier, compute/memory barrier, or thread fence. The existing profile row
supports 62 stages per window; multiple one-layer launches cover longer
families. Production operator images do not select `OP_PROFILE_EVENT`.

The representative SWA/HCA/CSA natural layer frontiers were
304.640/335.840/399.616 us, split into 208.768/232.416/294.656 us of
attention and 95.872/103.424/104.960 us of FFN. Queued-step measurements show
the following structural costs:

- Sparse attention performs 60.832, 60.576, and 74.528 us of active compute
  in SWA, HCA, and CSA. The associated M2C readiness wait is only 3.104 us in
  SWA and 16.672--23.904 us in compressed families. The matched Triton task
  is 26--33 us, so attention has a kernel gap independent of scheduler wait.
- Each of the eight disjoint o_a branches performs about 18.4--21.0 us of
  active projection work; o_b performs about 19.5--20.0 us. Later branches
  can first wait 72--97 us in their local queue. Because branches overlap,
  those waits must not be summed; natural join frontiers remain the source of
  critical-path attribution.
- Routed expert projections perform about 7.4--8.7 us each, while shared
  expert projections perform about 13.4--14.6 us. Router compute is short but
  begins after a 25--28-us readiness wait. CSA's main and index compressor
  chains also contain several small compute steps separated by M2C waits.

This separates the next board exploration into two mechanisms. First reduce
the attention and dense FP8 active task times against Triton/FA4/FlashInfer
and DeepGEMM references. Then shorten concrete readiness edges or revise
shape-derived placement where the per-step counters show idle queues. Keep
the accepted one-SM-per-head attention placement until a bounded, liveness-
safe alternative passes repeated launches; do not reuse the abandoned
grouped multi-SM schedule.

## Matched context-128 per-step gap profile (2026-08-12)

The accepted routed-expert source boundary remains `648fe02`; profiling did
not rerun or replace its accepted performance gate. Commit `cf579d1` adds only
the missing SGLang profiler bracket to the fixed-context harness. All three
runtimes use one GPU. The framework runs execute a real 127-token prefill and
measure the next decode at context 128 with the complete 129,280-row
vocabulary. The VDCores resident gate uses its deterministic position-127
cache and a 4,096-row output gate; applying the separately measured
full-vocabulary increment is therefore a projection, not a measured
pointer-era full-vocabulary run.

| Runtime | Accepted median (ms) | Full-vocabulary comparison (ms) | Gap from VDCores (ms) |
| --- | ---: | ---: | ---: |
| VDCores, one launch | 14.569616 | 15.005072 projected | - |
| vLLM 0.23.0 | 6.015728 | 6.015728 measured | 8.989344 |
| SGLang 0.5.12.post1 | 7.321974 | 7.321974 measured | 7.683098 |

The projected complete VDCores token is 2.494x the vLLM median and 2.049x the
SGLang median. The strict 10%-below-vLLM target is 5.414155 ms, leaving a
9.590917-ms projected gap.

For framework attribution, a Torch/Kineto trace brackets prefill plus decode.
The target decode is isolated by its GPU annotation, and layer boundaries are
the alternating attention/FFN mHC-pre kernels. Phase time is the union of GPU
kernels, copies, and memsets between boundaries, so concurrent streams are not
double counted. The vLLM profile job `20260812T173220Z-627132` measured a
6.119955-ms three-sample median and stored
`/tmp/dsv4-vllm-c128-torch/fixed_context_128_dp0_pp0_tp0_dcp0_ep0_rank0.1786556926469320683.pt.trace.json`.
The SGLang profile job `20260812T175034Z-820778` measured a 7.282197-ms
three-sample median and stored
`/tmp/dsv4-sglang-c128-torch/1786557318.1659656-TP-0.trace.json.gz`. Those
short profiler-run medians are sanity checks only; the 30-sample accepted
medians above remain authoritative.

The family-level phase split is:

| Family | Layers | VDCores attention / FFN / layer (us) | vLLM GPU-busy (us) | SGLang GPU-busy (us) |
| --- | ---: | ---: | ---: | ---: |
| SWA | 2 | 208.768 / 95.872 / 304.640 | 101.410 / 54.978 / 156.388 | 129.813 / 58.305 / 188.118 |
| HCA | 20 | 232.416 / 103.424 / 335.840 | 79.062 / 55.987 / 135.049 | 94.387 / 59.533 / 153.920 |
| CSA | 21 | 294.656 / 104.960 / 399.616 | 103.962 / 56.587 / 160.548 | 110.256 / 59.293 / 169.549 |

Across the 43 layers, the VDCores representative-family frontiers sum to
11.253632 ms of attention and 4.464384 ms of FFN. vLLM's decode trace has
3.967244 ms and 2.418024 ms respectively; the diagnostic gaps are therefore
7.286388 ms in attention and 2.046360 ms in FFN. SGLang has 4.462737 ms and
2.552417 ms, for gaps of 6.790895 ms and 1.911967 ms. Attention accounts for
78.1% of the layer-body gap to vLLM and 78.0% of the gap to SGLang. The
framework full-vocabulary head occupies another 0.203814 ms of vLLM GPU-busy
time and 0.233575 ms of SGLang GPU-busy time.

The VDCores representative per-step natural frontiers are:

| Stage (us) | SWA | HCA | CSA |
| --- | ---: | ---: | ---: |
| Attention mHC/hidden/KV/Q preparation | 50.304 | 50.944 | 51.584 |
| Compressor and indexer | - | 31.872 | 58.752 |
| Sparse attention | 63.456 | 65.408 | 78.112 |
| Inverse RoPE | 2.592 | 2.464 | 2.528 |
| o_a join frontier | 58.624 | 47.072 | 63.456 |
| o_b | 27.744 | 27.584 | 33.056 |
| Attention mHC post | 6.048 | 7.072 | 7.168 |
| FFN mHC pre | 13.408 | 14.304 | 14.688 |
| Route frontier | 12.768 | 18.336 | 18.752 |
| Shared+routed expert join | 57.312 | 57.632 | 58.784 |
| Expert reduction | 6.112 | 6.752 | 5.920 |
| FFN mHC post | 6.272 | 6.400 | 6.816 |

These VDCores values are representative natural producer frontiers from the
internal-counter profile, while the framework values are unioned GPU-busy
time under profiler instrumentation. The 15.718016-ms sum of representative
VDCores layers and the 6.385268/7.015153-ms framework layer sums must not be
treated as an exact subtraction of the accepted end-to-end medians: the
production VDCores loop overlaps adjacent work, and Torch profiling changes
framework timing. They are a critical-path attribution, not a new TBT result.

The mechanism-level comparison identifies two broad gaps:

- VDCores sparse attention is 60--75 us of active task work and 63--78 us at
  the natural frontier. The framework FlashMLA split-KV plus combine kernels
  are about 16.7 us per layer in both traces. Even the matched Triton task is
  only 26--33 us, leaving a kernel-level gap before queue tuning.
- VDCores o_a and o_b frontiers are 47--63 us and 28--33 us per layer. The
  eight o_a branches overlap, so their local waits are not additive, but the
  final join remains much later than the framework's DeepGEMM/NVJET dense
  projection kernels, whose individual decode shapes are generally about
  6--12 us.
- The complete framework FFN phase is 55--60 us per layer. VDCores spends
  57--59 us at the shared+routed expert join alone, then pays mHC, routing,
  reduction, and post work for a 96--105-us FFN. Router arithmetic itself is
  only 2.6--2.9 us; its 25--29-us local readiness/admission wait confirms that
  routing arithmetic is not the first target.

The next optimization order is therefore attention mechanism first
(FlashMLA/FA4-style Blackwell tiling, TMA/data-scale overlap, and fewer dense
projection queue frontiers), then FFN admission and shared/routed overlap.
Small router arithmetic, barrier-count, or instruction tuning cannot explain
the measured gap and stays behind those board-level experiments.

## Native FP8 split-K projection mechanism (2026-08-12)

The output-row-only MXF8 schedule underutilizes the grid at the attention
projection shapes. The selected split-K policy assigns one SM to each
`(M128,K-shard)` tile: split-2 for M1024/K4096 and split-4 for M4096/K8192.
Both policies keep K2048 on each SM. Eight simultaneous `o_a` groups therefore
occupy 128 SMs, and `o_b` independently occupies 128 SMs.

UMMA accumulates each shard in FP32 TMEM. The epilogue drains only logical N
column zero to one contiguous FP32 M128 shared tile, and STU performs native
2-D TMA reduce-add into a fixed `[1,M]` FP32 accumulator. This cuts the
reduction traffic and accumulator storage by eight while retaining the same
single TMA transaction and avoiding a compute reduction task. No global
pointer reaches compute, and the implementation adds no inter-stage copy,
thread fence, issue barrier, or full K unroll.

The isolated task, including TMA reduction but excluding accumulator reset,
measures 10.144 us for M1024/K4096 split-2 and 10.464 us for M4096/K8192
split-4. Both are exact against the FP32 dequantized reference. Recorded
unsplit native medians were 18.848 and 39.600 us respectively. Matched vLLM
DeepGEMM kernels were 9.589 and 11.628 us, so the split task is within 5.8% at
`o_a` and 10.0% faster at `o_b` before integration.

These are not resident-stage numbers. A repeated one-launch flow must reset
the accumulator every layer. First extend the same shape-derived split-K/TMA
reduction mechanism to all attention GEMMs and integrate it in the resident
graph without fusion. Require layer and early full-model gates before treating
the task gain as TBT progress.

### Projection-wide split policy screen

The split schedule now balances an arbitrary number of `(M128,K-shard)` work
tiles over at most 152 resident SMs. One SM may issue several bounded tasks;
only its last reduce store releases the stage barrier, so barrier arrival count
remains the physical SM count rather than the logical work count. The same
mechanism can therefore cover both narrow underfilled projections and wide
projections without creating another launch or instruction stream.

| Projection shape | Split / SMs | Isolated median (us) | Reference |
|---|---:|---:|---:|
| Q_a, M1024/K4096 | 8 / 64 | 3.552 | vLLM 5.982 |
| KV, M512/K4096 | 8 / 32 | 3.488 | resident gate pending |
| Q_b, M32768/K1024 | 2 / 152 | 11.616 | native unsplit 12.032; vLLM 6.561 |
| Index Q_b, M8192/K1024 | 2 / 128 | 3.632 | resident gate pending |
| Eight O_a groups, aggregate M8192/K4096 | 2 / 128 | 10.528 | vLLM 9.589 |
| O_b, M4096/K8192 | 4 / 128 | 10.464 | vLLM 11.628 |

All rows are exact against the dequantized FP32 oracle within the benchmark's
tolerance. They include TMA reduce-add but exclude accumulator reset and the
model-dtype handoff. The screen also rejected over-splitting: aggregate
O_a split-4/split-8 measured 10.912/11.680 us, and O_b split-8/split-16 measured
10.944/11.680 us. Q_b remains the unresolved kernel gap because its large M
grid was already saturated before splitting.

Blackwell TMA reduce-add also accepts a BF16 tensor map. The split epilogue can
therefore convert each FP32 TMEM partial to BF16 in shared memory and reduce it
directly into the normal model output buffer. This is a GEMM output mode, not a
consumer fusion, and removes the otherwise-required FP32-to-BF16 copy stage.
Across the selected shapes, 100-sample medians are 3.552 us (Q_a), 3.488 us
(KV), 11.744 us (Q_b), 3.648 us (index Q_b), 10.368 us (aggregate O_a), and
10.368 us (O_b). Maximum absolute deviation from the BF16 full-K reference is
at most 0.007812. Accumulator reset is still outside these spans and remains an
explicit resident integration requirement.

### Projection-wide resident integration

Partial BF16 rounding showed model sensitivity in an early single-layer
diagnostic, while the exact fallback of FP32 TMA reduction plus a queued
FP32-to-BF16 finalizer is bitwise equal to the full-K Q_b oracle. The accepted
policy permits this small drift, so `--fp8-splitk-reduction=bf16` is the default
split-K endpoint and `fp32` remains an exact diagnostic fallback. All normal
BF16 projection outputs are disjoint views of one contiguous arena. One
in-queue zero-fill resets the arena, then Q_a/KV/Q_b/index-Q_b/O_a/O_b reduce
directly into their consumer-visible model buffers. No projection is fused
with a producer or consumer, and the BF16 path has no conversion/copy stage.

The complete graph is larger than the shared-memory instruction image. Build
the selective runtime with `global_insts=1` and
`DAE_COMPUTE_OPS_FILE=benchmarks/deepseek_v4_resident.ops`; this selects
`DAE_LOAD_INSTRUCTIONS=0`, keeps the compute and memory programs in HBM, and
raises the per-SM instruction capacity from 512 to 4096. It remains one
persistent GPU kernel launch and one unified VDCores instruction queue; HBM is
only the backing store for instruction fetch.

The exact FP32 fallback job `20260812T204651Z-2823771` queued 218 compute and
1,101 memory instructions, preserved full-network token 5, and measured a
15.536512-ms median. BF16-direct one-layer job
`20260812T205453Z-2920570` exercised all six projection families at context
128, used 61 compute and 301 memory instructions, and preserved token 2759.
Its ten-sample 0.407904-ms median is 12.7% below the three-sample FP32/finalizer
gate. Q_b had maximum absolute error 0.003906, mean absolute error 0.000042,
and cosine similarity 0.99999577 against the full-K BF16 oracle.

Full BF16-direct job `20260812T205520Z-2926038` loaded 153.364 GiB from the
durable checkpoint at `/mnt/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4`, used
one B300 and one persistent launch, and queued 202 compute plus 1,069 memory
instructions. It preserved token 5; five samples measured
14.250208/14.713952/15.006816 ms min/median/max. BF16 direct improves the exact
FP32 endpoint by 0.822560 ms (5.29%) and the accepted 14.747968-ms fully affine
VDCores boundary by 0.034016 ms (0.23%). It remains 2.45x the 6.015728-ms vLLM
median and 2.01x the 7.321974-ms SGLang median. Profile the integrated
projection frontiers under the same HBM-instruction build before attempting
fusion.

## Split-32 UMMA attention and native O_a handoff (2026-08-13)

The clean-room sparse-attention producer splits cache rows into K32 shards.
Each CTA keeps all 64 query heads in UMMA's M dimension, issues four D128 QK
products, performs a masked local softmax, and issues four D128 PV products.
It writes one `[64,512]` BF16 partial plus FP32 `(max,mass)` metadata per
shard. A stable reducer merges those nonlinear softmax states, includes the
attention sink as a zero-output candidate, applies inverse RoPE to the last 64
logical dimensions in registers, and directly emits the native MXF8 operand
for `O_a`.

The layout is head-major despite the grouped consumer view. The same bytes are
viewed as `[64,4,2048]` by the reducer and `[8,32,2048]` by the eight `O_a`
groups. For each head, the four records correspond to consecutive logical
K128 blocks; inverse RoPE touches offsets 64--127 only in the fourth record.
For each interleaved pair, the inverse is
`even'=even*cos+odd*sin`, `odd'=odd*cos-even*sin`. This matches vLLM's
`fused_inv_rope_fp8_quant` order and avoids an intermediate inverse-RoPE BF16
tensor. It also removes the previous eight independently materialized and
quantized group inputs. The only cross-SM materialization is the single native
MXF8 consumer tensor.

At context 128, SWA/HCA/CSA contain 128/129/160 attention rows. Focused B300
medians are:

| Rows | Split-32 producer (us) | + merge/inv-RoPE/quant (us) | + grouped O_a (us) |
| ---: | ---: | ---: | ---: |
| 128 | 15.184 | 20.080 | 29.920 |
| 129 | 15.152 | 20.528 | 30.528 |
| 160 | 15.312 | 20.224 | 30.432 |

The isolated native inverse-RoPE/quant task is 3.664 us versus vLLM's
6.912-us event median. The producer is 8.3--9.3% below the approximately
16.7-us FlashMLA split-KV-plus-combine mechanism seen in the framework traces.
The complete fused attention/epilogue is 20.080/20.528/20.224 us versus
vLLM's attention-plus-inverse-quant 53.952/55.008/28.416 us for SWA/HCA/CSA.
After adding `O_a`, the corresponding vLLM event spans are
70.976/72.128/44.800 us, so the native handoff is 57.8%, 57.7%, and 32.1%
lower. The isolated BF16-reduction `O_a` kernel remains 10.368 us versus a
9.589-us raw DeepGEMM kernel; the target is won at the requested fused task
boundary, not by claiming that isolated subkernel is faster.

The 128/129/160-row output checks have maximum absolute error
0.003906/0.003906/0.001953, mean absolute error
0.000199/0.000826/0.000187, and cosine similarity
0.999308/0.994604/0.999232. TMA reduce-add remains enabled for the BF16 split-K
`O_a` projections under the accepted rounding policy. The softmax merge is an
explicit max/mass reduction because it is nonlinear and cannot be represented
by TMA add without changing the attention result.

The resident schedule first releases the common split partials, then places
eight 8-head reducers on the first half of eight disjoint 16-SM partitions.
Each reducer releases only its own native group, allowing that partition's
16-SM split-K `O_a` stage to start without waiting for the other seven groups.
A bounded one-layer CSA gate measured a 0.342944-ms median from its first three
valid timed iterations versus 0.390592 ms for the matched legacy contiguous
path's first valid iteration. Later replay still hits the pre-existing
resident barrier-reuse/repeatability failure, so this is not an accepted
steady-state gate.

The full 43-layer, context-128, 129,280-vocabulary run loaded 153.364 GiB,
passed its full-head reference, and produced one valid timed sample at
10.425344 ms (95.92 token/s). That is 2.647040 ms or 20.25% below the prior
13.072384-ms VDCores endpoint. The next replay hung at the known resident
barrier-reuse boundary, so no median is reported. The current matched vLLM
reference is 7.607872 ms (131.44 token/s): this bounded VDCores sample remains
2.817472 ms, 37.0%, or 1.370x slower end to end even though the targeted
attention steps are faster.

## Compact FP8 weight-scale experiment (2026-08-13)

The native MXF8 weight record is 16,896 bytes: 16 KiB of swizzled FP8 data and
a 512-byte expanded UE8M0 SFA image. It therefore occupies three 8-KiB slots.
The opt-in raw-scale schedule instead TMA-loads only the 16-KiB data record
into two slots and publishes one raw pointer for the task's contiguous scale
row. Compute warp 0 loads each one-byte checkpoint scale, expands the uniform
512-byte SFA image with vector stores into unused padding of the corresponding
2-KiB activation record, and issues UTCCP to TMEM. No scale payload passes
through LDU and no extra shared slot is allocated. The raw pointer is consumed
before normal records, allowing scale production to overlap both LDU ports.

A register-to-TMEM prototype was also tested. TMEM ownership requires all four
compute warps to store their own 32-row quadrant, and the necessary cross-warp
rendezvous made that form slower. Reusing activation padding reduced the
focused image to 56 registers with zero spills, but the packed path still won
matched task A/B:

| Shape / schedule | Packed SFA (us) | Compact raw SFA (us) |
| --- | ---: | ---: |
| Q_a M1024/K4096, split 8 / 64 SMs | 3.888 | 4.192 |
| Q_a M1024/K4096, split 16 / 128 SMs | 2.912 | 3.232 |
| O_a aggregate M8192/K4096, split 2 / 128 SMs | 12.160 | 12.352 |

The compact result matches the dequantized oracle (`max_abs <= 0.007812`); a
diagnostic run measured cosine similarity 0.99999690. It saves one slot and
3.0% of the weight-record bytes but is not selected by the resident path.

Partial K chunks were retained in the generic split-K scheduler because they
enable two-tile shards. They are not the production Q_a policy: in a matched
one-layer/context-1 resident A/B with exact FP32 reduction, split 16 measured
0.325344 ms versus 0.318128 ms for split 8. Direct BF16 split 16 also failed
the repeatable-token gate. Split 8 remains the cross-step choice. On this
narrow probe FP32/fused-finalizer measured 0.318128 ms versus 0.345632 ms for
BF16 direct reduction; the global default remains BF16 because the existing
43-layer measurement favors it.

## Rejected native-FP8 scale-tail FIFO (2026-08-13)

A dedicated 8-KiB, byte-granular FIFO was prototyped after the 24 ordinary
slots.  It assigned arbitrary 16-byte-aligned spans with monotonic tickets and
strict FIFO retirement, reducing each native FP8 weight from three ordinary
slots to two plus a 512-byte FIFO allocation.  This makes an eight-weight
window fit in 18 ordinary slots, whereas the old combined-record form would
need 26.  It did not improve any accepted projection schedule.

The tracked bounded Q_b schedule already has zero allocator stalls at its
13-slot peak.  In a matched prototype image, unsplit M32768/K1024 Q_b measured
11.584 us with ordinary combined records.  FIFO bulk loads with a shared
weight/scale barrier measured 26.304 us at an eight-tile window.  Separate
barriers and the opposite LDU were best at 16.864 us, still 45.6% slower.
Production split-2 Q_b regressed 12.416 to 18.800 us; aggregate split-2 O_a
regressed 11.040 to 16.416 us; split-4 O_b regressed 11.072 to 16.416 us.  All
paths were numerically exact.

Classic `cp.async` was also rejected.  The current LDU handler is single-lane,
so one 512-byte scale requires 32 separate 16-byte issues; every comparable
separate-barrier `cp.async` variant was slower than one bulk scale copy.
Sharing a barrier on one LDU serialized the two discontiguous transfers, while
separate barriers/LDUs still paid an extra command and readiness token per
weight.  Increasing the activation window from four to eight did not offset
that cost.

Keep the contiguous 16,896-byte record and bounded four-weight stream.  Revisit
a special scale arena only after measuring real ordinary-slot stalls, or when
one transaction/readiness event can target discontiguous data and scale
destinations.  The experimental runtime code was removed.

The standalone benchmark now performs immutable weight layout conversion from
Python setup with `runtime.prepack_fp8_checkpoint`, matching the resident
checkpoint loader, and omits the setup-only prepack opcode from its selective
operator image.  Token-time activation quantization remains a VDCores task.
The Python-owned converter is byte-exact against the prior native layout; the
final image measured three identical 10.848-us Q_b medians with zero error.

## Staged packed-scale UMMA and static dispatch (2026-08-13)

The accepted packed path keeps the contiguous 16,896-byte checkpoint record,
but stores two adjacent K128 scale layouts in the first record tail and loads
the second weight as data-only. Token-time activation quantization emits the
same pack-2 layout. LDU0 streams weights while LDU1 loads an eight-K128
activation chunk. A four-stage full/empty barrier ring lets warp 0 issue UMMA
while warp 1 observes completion and returns exact slot masks; the phase mask
persists across sequential tasks. Two M128 outputs share each activation
stream and remain in disjoint TMEM columns until their delayed epilogues.

Scale pack, output-group count, split reduction width, and quant scale pack are
generated compute-family fields, following the WGMMA operator-family model.
The production manifest selects only pack-2 handlers with the one/two-row
shapes needed by its schedules. `CInst` carries only the K128 tile count; the
inner task has no runtime variant switch. The focused image compiled at 54
registers, nine barriers, an 80-byte stack, and zero spills, versus 64
registers for the runtime-dispatched multi-variant image. The broader resident
image remains spill-free at 70 registers because other selected operators set
its register bound.

Q_b now uses unsplit K1024 on 128 SMs, exactly two M128 rows per static task.
The exact standalone job `20260813T172958Z-1391236` measured 9.216/9.376/9.984
us min/median/max, down 13.6% from the accepted 10.848-us unsplit baseline and
20.2% from the prior 11.744-us BF16 split endpoint. It remains 42.9% slower
than the matched 6.5608-us vLLM/DeepGEMM result, so this is a material kernel
gain but not the requested framework-performance crossover.

The pack-2 static regression sweep was numerically correct. Q_a remained
3.552 us; KV measured 3.424 us; index Q_b measured 3.680 us; the actual
per-group O_a shape measured 9.248 us versus 10.368 us; and O_b measured
9.584 us versus 10.368 us. Maximum BF16-reference error was 0.007812. All 72
schedule/runtime tests pass.

The public 168,305,308,121-byte checkpoint snapshot was subsequently installed
in coordinator tmpfs at
`/dev/shm/checkpoints/nvidia/DeepSeek-V4-Flash-NVFP4`. The 478.4-GiB tmpfs had
294 GiB free before the install and 137 GiB free afterward; the host retained
about 1.1 TiB of available RAM. The first resident attempt
`20260813T174744Z-1579015` loaded the model but caught a graph-construction
invariant: pack-2 Q-rank quantization has four scale work groups, while its
explicit placement still requested the unpacked eight SMs. The resident caller
now caps only that native packed placement at the number of scale groups;
scalar placement and UMMA GEMV policies are unchanged.

Full-vocabulary job `20260813T175101Z-1613009` then used one coordinator B300,
one persistent launch, all 43 layers, context 128, and the complete 129,280-row
head. It loaded 153.364 GiB of tensor payload in 123.172 seconds, retained
24.467 GiB of GPU memory, queued 223 compute and 1,245 memory instructions, and
emitted exact token 5. After three warmups, ten samples measured
15.618624/15.822816/16.658720 ms min/median/max. Reject this as a performance
milestone: the median is 0.639392 ms (4.21%) slower than the prior
15.183424-ms full-vocabulary boundary and 2.63x the 6.015728-ms vLLM reference.
The isolated packed-UMMA gains therefore do not survive the present integrated
schedule.

The earlier 10.830944-ms report was a context-one, five-sample warp-parallel
Sinkhorn experiment, not the context-128 end-to-end gate. It was already
rejected because its coefficient reduction order could alter near-tie routes
and its matched layer profile regressed; the accepted context-one production
median remained 11.178160 ms.

## Clean-room packed-scale replay (2026-08-13)

The packed UMMA work was replayed on top of the clean-room pre-attention
implementation rather than merging the older resident graph. The clean-room
`HC_PRE_RMS`, split-32 attention reducer, and split-K KV finalizer remain the
owners of their fused boundaries. Their native FP8 producers now emit pack-2
scale records directly, and Q-rank placement is capped at the number of packed
scale groups. The resident path has one compile-time pack value and one
compile-time output grouping. Split-K retains its unavoidable one-output tail
handler, so the production manifest contains exactly three generated family
handlers: pack-2 quant, pack-2/two-output stream, and pack-2/one-output BF16
split-K.

This last manifest trim was important. Keeping the unused stream-one and
split-K-two specializations raised the persistent kernel to 234 registers.
Removing them reduced the selective image to 52 operators, three generated
handlers, 96 registers, nine barriers, a 224-byte stack, and zero spills. That
matches the clean-room kernel's 96-register bound. The actual one-layer and
43-layer programs enumerate only those three handlers.

A context-128 CSA step profile first compared the same layer, checkpoint, and
GPU. Q_b active compute fell from 7.264 to 6.016 us; the eight O_a groups fell
from roughly 7.1 to 5.3 us each; and O_b fell from 7.232 to 5.248 us. The
untrimmed packed image nevertheless measured 0.408064 ms versus 0.405056 ms
because its inflated register image increased downstream waits. After the
static-handler trim, the normal resident image measured 0.362816 ms versus
0.378272 ms for clean-room, a 15.456-us or 4.09% one-layer improvement.

Matched full-model job `20260813T191201Z-2447717` ran both normal images in one
GPU allocation from the same job-local tmpfs checkpoint. Both used one GB200,
one persistent launch, all 43 layers, context 128, split-K BF16 projections,
UMMA attention, and the full 129,280-row head. Clean-room measured 10.430368
ms and clean-room-plus-packed measured 10.341024 ms. The replay therefore
saves 0.089344 ms, or 0.86%, at the full boundary. It remains 2.733152 ms,
35.9%, or 1.359x slower than the matched 7.607872-ms vLLM reference; the
requested framework crossover is not achieved.

Treat those full-model numbers as bounded performance samples, not an
exact-token acceptance gate. The clean-room baseline reproduced its known
resident replay/state failure: its prime launch passed the head oracle with
token 6005, while the next timed launch emitted 1531. The packed replay likewise
passed its prime head oracle with token 294, then emitted 1 on the timed
launch. Thus the older pre-clean-room expected token 5 is not a valid gate for
this graph, and neither branch is repeatable yet. The packed replay does not
claim to fix that inherited issue.

The final native-Miniconda correctness job `20260813T191600Z-2485970` passed
all 72 schedule/runtime tests and the clean-room pre-attention smoke. Fused
attention plus O_a passed at 128/129/160 rows with maximum absolute error
0.003906/0.003906/0.001953, cosine similarity
0.999308/0.994604/0.999232, and 28.448/29.280/29.056-us medians.

## NVFP4 FFN K pipeline and grouped scheduling (2026-08-14)

The NVFP4 UMMA family now has a real four-stage K256 pipeline. A task-local
TMEM allocator reserves independent SFA/SFB columns per in-flight stage and
independent FP32 accumulator columns per output group. Warp 0 issues UTCCP and
UMMA without waiting for the preceding stage; warp 1 waits completion, retires
the matching shared slots, and signals when a ring stage is reusable. Only the
first K block uses zero accumulation, and the result drains once after the
last K tile. The same compute task therefore serves split and nonsplit work;
the following memory instruction determines whether the FP32 result is stored
or TMA reduce-added. Activation loads carry a configurable number of adjacent
K256 records, while weights remain tile-streamed.

The paired specialization shares each activation load and SFB stage between
gate and up while retaining separate routed alphas, SFA stages, and FP32
accumulators. A grouped scheduler interleaves all six route ranks across each
work wave and can run gate then up or paired gate/up. After split-K TMA
reduction, one task applies bounded SwiGLU to the FP32 gate/up sums and emits
the final native 3072-byte K256 NVFP4 W2 operand directly. It does not create a
BF16 intermediate or run a repack. This boundary cannot legally move before
split-K reduction because SwiGLU is nonlinear.

Matched 200-sample top-6 medians were 44.960 us for the retained per-expert
local schedule, 48.272 us for the best separate grouped schedule (gate/up
split 4, W2 split 2), and 48.384 us for paired gate/up after balancing the
shared activation load onto LDU0. Split 2/4/8 separate gate/up measured
50.272/48.272/57.632 us; W2 split 4 regressed to 52.416 us. The grouped path
improves W2/join by about 0.9 us but loses more at the FP32 gate/up reduction
and fused activation boundary, so it remains an experimental benchmark path
and is not selected for production.

Both LDU lanes cache the complete six-expert routing decision in scalar
registers, selected by a switch to avoid dynamically indexed local memory.
The route-address key is invalidated at the barrier-reload boundary after the
queues drain. The focused cache build moves from 54 to 56 registers without
spills; the full resident image uses 99 registers, nine barriers, a 224-byte
stack, and zero spills. Production's shared expert already begins from hidden
readiness on disjoint SMs 96--151 while routing and routed experts use SMs
0--95; shared W2 is split-K2, so Python list reordering does not expose an
earlier dependency frontier.

Framework-first baselines remain the acceptance reference. The vLLM/
FlashInfer TRTLLM routed-expert graph measured 16.996 us without route packing
(17.815 us including it); standalone FlashInfer W1/W2 kernels measured
4.6016/3.6816 us, and DeepGEMM's closest FP8 cases measured 5.4416/4.0048 us.
The isolated VDCores W1 split-K task reaches 3.808 us, but the 44.960-us full
top-6 flow does not cross the framework result. Final worker verification
passed 123 tests, and resident job `20260814T010022Z-2183840` preserved token
2835 at a 0.330096-ms one-layer median over 20 samples.

An exact two-schedule wave isolation keeps all 152 SMs active, 2,048 logical
K-tile works, and the same worker-load histogram
`{K8:32, K10:16, K12:8, K16:96}`. The one-wave form has 152 queue entries.
The two-wave form has 152 entries carrying 2,000 tiles at queue index zero and
eight K6 entries carrying 48 tiles at queue index one. Wave index is local
persistent-SM queue depth, not a global barrier or another kernel launch; the
second entries begin as their eight workers become free. Matched 20-warmup,
200-sample jobs `20260814T032101Z-4004556` and
`20260814T032116Z-4008300` both measured a 14.816-us frontier median. Thus the
similar time is expected: K6+K6 and K12 give those eight workers the same
logical load, while the split form differs mainly by one extra task epilogue,
FP32 TMA-reduce boundary, and task prologue.

This K6+K6 equality is worker-load accounting, not local accumulation: the two
queued K6 tasks target different shared-expert M tiles. Exact operator counting
shows 3,008 scheduled operators and 42,679,296 bytes for one wave versus 3,024
operators and 42,683,392 bytes for two waves. All activation and weight loads
are identical. The only delta is eight additional FP8 K6 compute-task entries
and eight 512-byte FP32 TMA reduce-adds. Diagnostic compute markers from jobs
`20260814T033711Z-29827` and `20260814T033733Z-34839` show the one-wave shared
K12 class completing at 7.584 us and the two-wave K6 tail completing at 8.160
us, while the unchanged routed K16 class completes at 13.904/13.920 us. The
split tail therefore really costs about 0.576 us, but still has roughly 5.76
us of slack before the routed critical frontier. The diagnostic final
frontiers were 14.720 and 14.752 us; task markers perturb those values, so use
the uninstrumented 14.816-us matched medians for the endpoint comparison.

## Linear-1 4,096-tile worker schedule (2026-08-14)

The gate-plus-up-only Linear-1 graph has 4,096 native UMMA tiles: 1,024 shared
FP8 K128 tiles (`2 projections * 16 M tiles * 32 K tiles`) and 3,072 routed
NVFP4 K256 tiles (`2 * 6 experts * 16 M * 16 K`). The schedule-only benchmark
is `benchmarks/deepseek_v4_linear1_queues.py`; it uses the existing group-1
FP32 UMMA and TMA-reduce handlers without rebuilding the operator image.
Routing, packing, allocation, and output zeroing are outside its timed
frontier. Queue validation proves exact K coverage for all 224 output rows.

The best measured graph gives every SM one shared head, retains 152 routed K16
anchors, and splits exactly 40 routed rows as K6+K5+K5. Thus every queue owns
one K16 routed task; 40 queues also own K6, 80 also own K5, and 32 have no
routed tail. The 152 shared heads are the feasible complement: 24 shared rows
use K12+K6+K6+K4+K4, and eight use K14+K6+K6+K6. Longest routed tails are
paired with shortest shared heads. Exact queue classes are 40
`shared K4 + routed K16+K6`, eight `K4 + K16+K5`, 72 `K6 + K16+K5`, 24
`K12 + K16`, and eight `K14 + K16`. There is no global barrier between the
shared head and routed tail.

The decisive split-count sweep used round-robin launches on one GPU to remove
clock-ramp bias. Forty extra routed reductions (K8+K8 shards) measured 25.184
us, the selected 80-extra plan measured 23.488 us, and 88 extra reductions
measured 25.824 us. Fewer splits leave K24 routed tails; more splits create
four-task queues and pay another task/reduction boundary. A mixed-size
measured-cost search at the same task count was also slower (23.712 versus
23.424 us in its matched run), so uniform K6+K5+K5 remains selected.

Final 20,000-round warmup and 10,000-sample matched job
`20260814T045121Z-1067772` measured 29.408 us for unsplit routed K16 and
22.880 us for the selected schedule, a 6.528-us, 22.2%, or 1.285x improvement.
Their full persistent-kernel medians were 29.792 and 23.264 us. These numbers
cover gate and up only: no SiLU, W2, routing operator, or checkpoint conversion
is included.

The matching framework comparison must distinguish single-step latency from
graph-amortized throughput. On the same GB200, vLLM 0.27.1 with FlashInfer
0.6.16.post3 measured 16.9344 us for the full routed-MoE graph at `inner=20`,
but 22.064 us at `inner=1` in job `20260814T050752Z-1300427`. A 20,000-warmup
Nsight capture, job `20260814T050533Z-1270278`, measured the fused six-expert
routed gate+up+SwiGLU kernel itself at 9.536 us. The corresponding shared
FP8 gate+up DeepGEMM, shape M4096/K4096, measured 9.856 us at `inner=1` in job
`20260814T050837Z-1311476`. Their serialized component sum is therefore
19.392 us, 3.488 us or 18.0% below the current 22.880-us combined schedule.
This is a component comparison rather than an observed fused vLLM frontier:
FlashInfer excludes the shared expert and includes SwiGLU, while the current
VDCores graph includes shared plus routed gate/up, stops at FP32 accumulators,
and excludes SwiGLU.

## Shape-specialized NVFP4 K512 mainloop (2026-08-14)

The routed M128/K4096 experiment now uses eight K512 stages, each containing
two independently described K256 SM100 operands. Weight data remains streamed
through allocator-owned M2C slots, while a preloaded 128-byte metadata record
supplies the FP32 alpha and the compact weight/activation-scale addresses.
Checkpoint preprocessing stores each K512 weight as 32 KiB of data plus a
separate 4-KiB native SFA record; each K512 activation is 2 KiB of data plus
32 compact SFB scale bytes. The production raw-address instruction form is
still deferred.

Warp 2 bulk-copies weight scales through a two-stage shared-memory ring, warp
3 reads and expands only the live compact SFB bytes, warp 0 performs UTCCP and
submits consecutive UMMAs without waiting for the preceding completion, and
warp 1 retires allocator records after their completion barriers. One task-
local TMEM allocator reserves the accumulator and scale pipeline. The FP32
accumulator remains resident across all eight K512 stages, uses zero only for
the first UMMA, and drains once after the full K reduction. Scratch reuse is
generation-safe: it is released after dependent UTCCP/UMMA submission but not
before the submitted UMMA has captured the scale record.

On one allocated GB200 GPU in Conda `base`, job
`20260814T132634Z-671385` measured 6.816 us for one M128/K4096 result with zero
FP32 reference error, down from 14.368 us for the prior resident K256 task.
The image uses 80 registers, 13 barriers, 14,624 bytes static plus 210 KiB
dynamic shared memory, and has no spills. FlashInfer 0.6.16 measured 4.462 us
in job `20260814T132918Z-718546`, but used a two-SM CTA pair; normalized by SM
budget this is 8.924 SM-us/result versus 6.816 for VDCores. The next acceptance
gate is the complete 4,096-tile Linear-1 worker schedule, not the isolated
single-SM result.

## K512 Linear-1 worker integration (2026-08-14)

The 4,096-tile gate-plus-up benchmark now selects the K512 routed task while
preserving the original accounting: 1,024 shared K128 tiles plus 3,072 routed
K256-equivalent tiles. The routed work becomes 1,536 K512 stages. Python
checkpoint setup creates separate 32-KiB data and 4-KiB SFA records, compact
2-KiB activation data and 32-byte scale records, and routed 128-byte metadata
containing alpha plus the selected scale bases. Random packed activations and
random per-block scales are checked against Python dequantization outside the
timed frontier.

The selected worker plan does not split routed K. It assigns 192 static K8
tasks over 152 workers, so 40 queues receive two routed tasks and 112 receive
one. The 152 strict-priority shared heads have histogram
`{K2:40,K6:8,K8:72,K10:32}`: every double routed tail receives a K2 head.
Those 40 tails pair gate then up for the same route rank and M tile. The first
task retains the common 16-KiB activation allocation in an LDU register; the
second republishes the same slot mask and finally releases it. A packed retain
bit in the K512 instruction controls slot lifetime without changing its UMMA
mainloop. Legal even-boundary split-K sweeps were all slower because they lose
the static eight-stage specialization and add task/reduction boundaries.

On one allocated GB200 GPU in Conda `base`, matched activation-reuse job
`20260814T144518Z-1973483` measured 16.864 us without retention and 16.128 us
with retention; both had zero maximum FP32 error. The final 20,000-warmup,
10,000-sample retained run `20260814T144543Z-1980577` measured 16.608 us for
the Linear-1 frontier and 17.056 us for the complete persistent kernel. The
fresh selected K256 control `20260814T143232Z-1759294` measured 22.432 us, so
K512 plus the new queue plan saves 5.824 us (26.0%) and is 1.351x faster.

The same-scope external component reference remains FlashInfer routed
gate/up/SwiGLU at 9.536 us plus DeepGEMM shared gate/up at 9.856 us, or 19.392
us serialized. The 16.608-us VDCores result is 2.784 us (14.4%) lower, but it
still stops at FP32 gate/up accumulators and excludes SwiGLU, while the
FlashInfer component includes routed SwiGLU and excludes the shared expert.

## Native MXFP4-weight / MXFP8-activation UMMA (2026-08-14)

The SM100 mixed `MXF8F6F4` family consumes MXFP4 weights and MXFP8
activations directly; there is no FP8 conversion in the kernel. The weight
type must be `cutlass::detail::float_e2m1_unpacksmem_t`, not the ordinary
`float_e2m1_t`: the former selects mixed-family E2M1 format code 5 after the
TMA unpack transform, while the latter selects standalone format code 1,
which this instruction family interprets as E5M2. The fixed atom is
M128/N8/K128 with UE8M0 block scales and an FP32 accumulator.

Token-time HBM is already native. Weight data is packed uint8
`[M/128,8 K512,4 K128,128,64]`, so one K512 record is 32 KiB. A custom 5-D
tensor map uses `CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B` plus SW128 to expand
that transaction into four 16-KiB K128 SMEM descriptors (64 KiB destination).
One activation K512 record is 4 KiB. Weight and activation scale images are
native uint8 `[M/128,8,2048]` and `[8,2048]`; all duplication and swizzling is
checkpoint/token preparation outside the measured path.

Both requested scale-delivery strategies are implemented. The direct-TMA
strategy caches the complete SFA and SFB base addresses once in LDU0 and LDU1,
respectively. It then loads one 2-KiB SFA half and one 2-KiB SFB half per K512
tile directly into a dedicated task-tail ring; these records never consume or
free normal allocator slots. The winning selective image uses three 4-KiB
stages. Both LDUs observe one shared empty-phase barrier for each physical
stage, and compute warp 1 releases that stage only after the dependent UMMA
completes. Each operand's first scale-TMA command also seeds its LDU-local
base, so there are no standalone setup commands. The two base-address
mailboxes are distinct; each LDU acknowledges after copying its mailbox, and
the allocator rendezvous is deferred until a later output task is about to
reuse them. A single-tile task therefore pays no publication wait. This
machinery is compiled only with `mxfp_direct_tma=1`, so it adds no LDU state,
branches, or barriers to the metadata-only production image.

The faster metadata strategy encodes the complete 48-bit metadata-record
pointer in the compute instruction. Compute warps dereference record offsets
16 and 24 directly, so the metadata record consumes no allocator slot and no
memory instruction. Warp 2 and warp 3 then issue coalesced 16-byte `cp.async`
copies into a four-stage shared scale ring; warp 0 performs dependent UTCCP
copies to its task-local TMEM scale columns and queues consecutive UMMAs, and
warp 1 retires each expanded weight allocation after its completion barrier.
The FP32 accumulator remains resident for all 32 K128 UMMAs and drains once,
directly to FP32.

The mixed task now follows the WGMMA compute-family contract instead of
encoding activation packing in runtime arguments. Its generated families have
independent `K` and `BLOAD` fields. `K=512` means every weight command carries
one packed K512 tile (32 KiB from HBM, 64 KiB after TMA expansion), while
`BLOAD` is the number of consecutive 4-KiB activation K512 tiles in one
allocator load. `BLOAD=1/2/4` are tiled streaming schedules and `BLOAD=8` is
the full K4096 activation allocation; all cases still stream one weight tile
at a time. The schedule emits one activation allocation followed by `BLOAD`
independent weight allocations per chunk. Legal BLOAD values are 1, 2, 4,
and 8 for both TMA-scale and metadata-scale families.

GB200 sweep job `20260814T190142Z-2059079` verified exact 4096.0 FP32 output
for all eight scale-family/BLOAD combinations. Compiling all eight variants
into one persistent image distorted the short-task latency, so that manifest
is retained only as `deepseek_v4_mxfp4_mxfp8_sweep.ops`. The latency-critical
default manifest selects only metadata K512/BLOAD2 plus profiling and
termination. Its 25-slot, 64-instruction, 220-KiB image uses 115 registers,
16 barriers, an 80-byte stack, and zero spills. Job
`20260814T190952Z-2193068` measured 9.216 us task median and 9.696 us full
resident-kernel envelope over 500 samples with exact output. This improves the
pre-family 9.504/9.984-us milestone while keeping activation packing a true
compile-time family parameter. FlashInfer 0.6.16.post3 / vLLM 0.27.1 job
`20260814T184205Z-1730615`, using the correct
`group_gemm_mxfp8_mxfp4_nt_groupwise` backend at rows=4, N=128, K=4096,
tile-N=64, tile-K=128, measured 9.6208 us per CUDA-graph inner call with
quantized sanity `mean_relative=0.12491284`, `cosine=0.99230981`. Thus the
VDCores task frontier is 4.2% lower; its full envelope is 0.8% higher. The
comparison is intentionally explicit: FlashInfer's legal API minimum is four
activation rows and tile-N=64 launches two output CTAs, while VDCores owns one
SM and receives an N8 activation tile already laid out in HBM. That historical
single-row check used eight equal setup rows; the UMMA kernel itself did not
create the input rows.

Rejected variants are useful boundaries. A blocking global-to-shared scale
copy was about 0.32 us slower than `cp.async`; preloading both complete 16-KiB
scale images serialized the prologue and regressed metadata to 14.624 us; a
future `cp.async` group (`wait_group 1`) regressed to 9.728 us; a 32-byte LDU
metadata copy regressed to 9.952 us; and a 26-slot shaped allocator regressed
to 10.496 us. Splitting weight movement into K256 commands removed measured
allocator stalls but increased command overhead and reached only 12.06 us in
the instrumented build. Keep the K512 expanded transaction and one completed
async scale stage at a time.

## Direct-TMA versus raw-address phase trace (2026-08-14)

Fine-grained timestamps are optional and separate from the normal aggregate
profiler. Build with `mxfp_timeline=1` to record task entry, activation TMA
issue/visibility, both scale-producer intervals, scale visibility, weight TMA
issue/visibility, UMMA issue/completion, output readiness, and task exit for
each of the eight K512 tiles. A normal `track_profile=1` build does not execute
these event stores.

The final matched timestamp runs were direct-TMA BLOAD8 job
`20260814T220444Z-1093699` and raw-address BLOAD2 job
`20260814T213244Z-489815`. All times below are microseconds from compute-task
entry. `Scale section` is the interval exposed on the issuer's critical path,
not the producer's total lifetime.

| K512 tile | raw scale section | TMA scale section | raw UMMA issue | TMA UMMA issue |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.064 | 0.640 | 2.656 | 3.328 |
| 1 | 0.032 | 0.192 | 3.424 | 4.384 |
| 2 | 0.032 | 0.192 | 4.864 | 5.696 |
| 3 | 0.032 | 0.192 | 5.696 | 6.752 |
| 4 | 0.064 | 0.224 | 7.040 | 8.128 |
| 5 | 0.096 | 0.192 | 7.872 | 9.472 |
| 6 | 0.064 | 0.192 | 9.280 | 10.784 |
| 7 | 0.064 | 0.192 | 10.144 | 11.872 |

The direct path now issues activation TMA at 0.640 us and observes it at 1.056
us, ahead of raw address at 0.960 and 1.280 us. Nevertheless, its first UMMA
is 0.672 us later because tile-0 scale and weight visibility consume 0.640 and
1.024 us after activation readiness. Raw scale producers are already
1.760--3.552 us ahead when the issuer consumes their records. Direct TMA
instead places two scale completion tokens before every streamed weight token,
so later tiles expose 0.192--0.224 us in their scale section versus
0.032--0.096 us for raw address. Direct output becomes ready at 12.032 us
versus 10.304 us for raw address. UMMA itself is not the exposed blocker:
measured issue-to-completion deltas are only 0--0.064 us in both traces.

Clean timestamp-disabled controls on the same worker state measured 12.544 us
task / 13.024 us kernel for three-stage direct TMA in job
`20260814T220219Z-1048033`, versus 9.952 / 10.592 us for raw address in final
control job `20260814T221041Z-1205454`. The direct path is therefore 2.592 us
(26.0%) slower at the task frontier. Immediate paired publication with two
stages measured
14.176 us; adding the third stage reached 13.056 us, and deferred publication
plus first-scale fusion reached 12.544 us. The pre-change direct BLOAD8 path
was 14.624 us. The historical
9.216-us raw result was rechecked by compiling exact revision `3d0b9af`; that
binary also measured 9.984 us in the current worker state, so the older number
is retained as a historical clock/runtime result rather than used for this
phase comparison.

The clean aggregate counters point to the same cause. Direct TMA spends 43.0%
of its internal span in compute-side M2C waits and 0% in allocator-slot stalls;
raw address spends 31.0% in M2C waits despite 16.3% allocator-slot stalls.
Removing ordinary slot pressure therefore does not compensate for putting two
additional scale-completion dependencies on every K512 consumer frontier.

## Fused mixed-expert MXFP4 gate/up/SiLU (2026-08-15)

Linear-1 now has an unsplit task that owns one expert's M128 intermediate
slice for the complete K4096 reduction. Gate and up accumulate independently
in FP32 TMEM; neither accumulator is written after each K tile. The selected
mixed graph places the 16 shared-expert slices first, followed by 16 slices
for each of six routed experts. Thus one eight-row activation tile produces
112 independent tasks on 112 workers, with routing and native operand
preparation outside the timed frontier.

The selected mainloop uses two task-local K512 stages. Warp 1 streams packed
gate then up weights through direct TMA, warps 2 and 3 copy native SFA/SFB
records, and warp 0 issues raw `tcgen05.mma` instructions. Every K512 bundle
contains four K128 UMMAs. Each K128 member has disjoint task-local TMEM scale
columns, so its UTCCP scale copy can execute while the preceding UMMA still
reads its own columns. The issuer calls `umma_arrive` once per bundle and does
not wait for the preceding bundle before submitting the next; warp 2 retires
the bounded in-flight window and releases each full/empty stage. The first
K128 UMMA of each projection uses accumulator-zero and all later UMMAs use
accumulator-one.

Gate SiLU is distributed across the four compute warps at the gate-to-up
transition. The scale warps drain their gate rows while the weight producer
prefetches up stages; warp 1 drains its rows after filling that ring; warp 0
drains its rows after committing the final up bundle, overlapping those TMEM
loads and SFU operations with the last in-flight UMMA. After the one final
compute-group join, each thread retains its eight independent SwiGLU values in
registers through block-32 power-of-two scale reduction and native E4M3
packing. It then publishes one contiguous 1,536-byte
`[1024-byte data | 512-byte UE8M0]` record. The production path sends that
record directly from task-local shared memory to HBM. Compiling
`DAE_MXFP_GATE_UP_DIRECT_OUTPUT=0` retains the allocator-queued store.
Only active data rows and native scale locations are initialized. Scale
padding, and inactive rows when the compile-time row count is below eight,
are intentionally unspecified; clearing them added stores and a
compute-group barrier without changing any consumed value.

The task metadata is one 128-byte record per M128 slice. Offsets 16, 24, and
32 contain the gate-weight, activation, and up-weight scale base addresses.
Offset 40 packs the gate/up tensor-map indices plus output-tile index; offset
48 contains the direct output-record address. Weight HBM is task-major
`[task,8 K512,4 K128,128,64 packed bytes]`; gate/up scale HBM is
`[task,8,2048]`; the shared activation data/scales are `[8,4096]` and
`[8,2048]`. These are already scale-expanded and swizzled native images, so
the timed kernel performs no format conversion, repack, or row replication.

The optimization controls establish the retained boundaries. Raw UMMA issue
is the default and the CUTE issue form remains build-selectable. Per-lane
asynchronous scale copies beat a warp-2 bulk-copy variant at full occupancy.
Independent TMEM scale columns removed the UTCCP/previous-UMMA write-after-read
dependency; the image remains one CTA per SM because 223 KiB dynamic shared
memory, rather than its 246 registers, is the occupancy limit. K128 is exact
but measured 28.704 us at the task frontier and 29.248 us for the kernel in
job `20260815T050714Z-864015`, so the two-stage K512 family remains selected.

The profiling-free production image has five allocator slots, 64 instruction
entries, 223 KiB dynamic shared memory, 246 registers, nine barriers, an
80-byte stack, and no spills. The correctness fixture supplies eight distinct
physical activation rows `[0x60,...,0x67]`; the kernel must return the eight
Python-derived output rows `[0x78,0x7a,0x7c,0x77,0x79,0x7b,0x7c,0x76]` plus
32 row-specific scale entries at native offsets `row * 16 + [0,1,2,3]`.
This makes a column-zero replication fail byte-for-byte while leaving padding
outside the contract. Single-task job `20260815T054352Z-1632466` measured
12.224 us. Final selected-only job `20260815T060507Z-2082799` passed the full
check over 2,000 timed iterations; the 112-task mixed wave measured 12.448 us
at the task frontier and 13.088 us for the persistent kernel.

Output-row count remains a useful compile parameter. The selected
`DAE_MXFP_GATE_UP_FIXED_OUTPUT_ROWS=8` preserves all eight UMMA columns.
Setting it below eight emits only the requested real columns; it never
replicates column zero and does not clear inactive record bytes. The final
zero-free row sweep was:

| output rows | epilogue | task median (us) | kernel median (us) | job |
| ---: | --- | ---: | ---: | --- |
| 1 | BF16 scalar tail | 11.904 | 12.544 | `20260815T053454Z-1436127` |
| 2 | BF162 | 12.032 | 12.704 | `20260815T053726Z-1501078` |
| 4 | BF162 | 12.000 | 12.608 | `20260815T053239Z-1382827` |
| 8 | BF162 | 12.480 | 13.152 | `20260815T054000Z-1556257` |
| 8 | FP32 | **12.448** | **13.088** | `20260815T060507Z-2082799` |

Removing inactive-data and scale-padding clears improved the matched four-row
kernel from 12.704 to 12.608 us. The row-count switch is therefore retained,
while ROWS=8 remains the default contract.

The optional BF16 epilogue is genuinely vectorized: disassembly contains
`F2FP.BF16.F32.PACK_AB`, `HMUL2.BF16_V2`, and `HFMA2.BF16_V2`. SiLU's
exponential remains scalar/SFU, while paired conversion and multiply use
BF162. It is not the default. A Python-only sweep of 765,952 finite
representative cases found 21,158 serialized MXFP8 data-byte changes and
7,568 scale-byte changes versus FP32, with 0.601% maximum relative pre-quant
error. Full-row BF162 was also 0.128 us slower than FP32. The deterministic
performance fixture happens to serialize identically and has zero pre-quant
error, so that fixture alone is not used to justify BF16 accuracy.

Matched kernel-level framework results used vLLM 0.27.1 and FlashInfer
0.6.16.post3 in the worker Conda environment:

| implementation | legal decode layout | GEMM (us) | activation/quant (us) | chained Linear-1 (us) |
| --- | --- | ---: | ---: | ---: |
| VDCores fused | 7 experts x 8 useful rows, no row replication | fused | fused | **13.088** |
| vLLM/DeepGEMM | 56 logical rows padded to 288 | 10.603 | 2.410 | 13.865 |
| FlashInfer + vLLM | 7 groups x 8 useful rows | 31.818 | 1.814 + 1.508 | 36.530 |

The DeepGEMM result is job `20260815T050239Z-775889`; the FlashInfer result is
job `20260815T050618Z-847378`. VDCores is 0.777 us or 5.6% lower latency than
the production vLLM/DeepGEMM chain and 2.79x faster than the legal FlashInfer
chain. Its task frontier alone is 10.2% below the DeepGEMM chained result. The
padding difference is part of each framework's legal shape contract and is
stated explicitly rather than normalized away. Only the fixed-ring fused
family is retained: all 112 unsplit slices fit in one wave and the fused path
meets the framework acceptance target without another HBM or kernel boundary.

## Native MXFP4/MXFP8 full FFN (2026-08-15)

The selected full-FFN frontier is two stream-ordered focused kernels for one
shared and six routed experts, eight useful rows, hidden size 4096, and expert
intermediate size 2048. Linear-1 launches 112 CTAs: 16 M128 slices for the
shared expert first, followed by 16 slices for each routed expert. Every CTA
reduces the complete K4096 gate and up projections in FP32 TMEM, applies SiLU
and multiplication, and stores one native 1,536-byte MXFP8 record:
`[1024-byte E4M3 8x128 data | 512-byte UE8M0 scale image]`. Linear-2 consumes
these records in place. There is no conversion, repack, row replication, or
FP32 Linear-1 write/read boundary in the timed graph.

Linear-1 weight HBM uses K-tile-major
`[8 K512,112 tasks,4 K128,128,64 packed bytes]`, so a CTA wave reads adjacent
weight records for each K tile. Its scale planes use the matching K-tile-major
task stride. Activation data and scales are already native, duplicated, and
swizzled in HBM. The retained family parameter independently selects one
streamed K512 activation tile or all eight batched tiles; the full-FFN image
selects eight. TMA descriptors use 256-byte L2 promotion by default.

Linear-2 launches all 224 expert/M128 output tasks as independent CTAs. Each
CTA computes one M128/N8/K2048 projection with eight K256 bundles and a
two-stage shared-memory ring. Warp 1 streams packed weights with TMA, warp 2
copies weight scales and retires completed stages, warp 3 copies native MXFP8
data and scales, and warp 0 performs UTCCP plus raw UMMA issue. The issuer
commits the next bundle without waiting at the preceding issue point; the
retire warp bounds the two-bundle window. Only the first K128 UMMA uses
accumulator-zero. All later UMMAs accumulate into the same FP32 TMEM fragment,
which drains once after the full K dimension.

The focused down kernel allocates 64 TMEM columns and relinquishes the TMEM
allocation permit immediately after allocation. The permit protects
allocation, not subsequent use of disjoint columns, so two down CTAs can
reside on one SM. Its direct task-local scratch starts at offset zero and fits
in an 80-KiB dynamic-shared-memory launch; generic allocator scratch remains
independent. This changes the 224-task launch from two serialized CTA waves to
all task starts within one scheduling wave. The final compile reports 111
registers, one barrier, and zero spills for down; focused Linear-1 reports 220
registers, nine barriers, and zero spills.

The shared-expert CTA for each M128 output block zeros the final FP32 tile and
publishes a release edge. All seven expert CTAs apply their route scale in the
FP32 shared epilogue, wait for that edge, and issue direct FP32 TMA reduce-add.
The two kernels are ordered on one CUDA stream, so selected Linear-2 does not
scan 16 Linear-1 readiness flags. The task still supports a block-wise
readiness edge for alternative schedules, but the measured sequential path
avoids that polling and its dependency stalls.

The final matched runs used the same GB300 GPU, vLLM 0.27.1 with its vendored
DeepGEMM backend, 20 FFNs per CUDA graph, 200 warmups, and 2,000 timed samples.
Route packing and input/checkpoint preparation are outside both timed
frontiers.

| implementation | FC1 | activation/quant | FC2 | reduce | full FFN |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM/DeepGEMM | 10.6176 us | 2.4256 us | 6.9312 us | 3.8608 us | 30.4800 us |
| VDCores focused | fused | fused | fused down + TMA reduce | fused | **24.8720 us** |

The vLLM result is job `20260815T113724Z-1449985`; the immediately following
VDCores result is job `20260815T113749Z-1459095`. VDCores is 1.2255x faster,
or 18.399% lower latency, exceeding the 10% acceptance target. The VDCores
minimum/median/maximum were 24.6688/24.8720/33.6944 us. Linear-1 native data
and active scale bytes pass exact equality checks, and the final FP32 output
passes `torch.testing.assert_close`; maximum relative error is 1.4e-7. The
reported 64.0 maximum absolute difference occurs on the deliberately large
uniform-value fixture and is covered by that relative tolerance.

Rejected schedules were removed from the implementation surface. A
dependency-driven resident/concurrent launch stretched the down span to about
21.8 us and full FFN to 30.24 us. Paired-M down took 24.096 us by itself;
split-K2, N16, and cooperative two-CTA variants added work or unused output
lanes. Establishing the final output with a shared copy instead of zero plus
seven TMA reduce-adds regressed full FFN to 31.088 us. TMA L2 promotion 128
also lost to 256. The retained source therefore contains only the one-CTA N8
task used by the K512 generic schedule and K256 focused kernel.

The selected build uses
`benchmarks/deepseek_v4_mxfp4_mxfp8_full_ffn_two_kernel.ops`, one generic
allocator slot, 64 instruction entries, 223 KiB configured shared memory, and
`ffn_specialized=1`. Build and benchmark inside the worker Conda environment;
GPU execution must go through the cluster allocator. The host schedule suite
passes 86 tests after evaluating it with the normal 24-slot schedule capacity.

## Resident fused-FFN profiling checkpoint (2026-08-16)

The first resident integration checkpoint is diagnostic, not correctness
qualified. The cleanroom release baseline emitted token 14 in 10.793312 ms;
the fused integration measured 14.770304 ms but a fresh uninstrumented run
(`20260816T022900Z-3275374`) emitted token 0. The integrated schedule currently
lets `ffn.route` wait on activation quantization without a router-completion
edge, so Top-6 may consume incomplete router logits. Restore that dependency
before accepting either correctness or performance from this path.

| release flow | CUDA TBT (ms) | first token | qualification |
| --- | ---: | ---: | --- |
| cleanroom baseline | **10.793312** | 14 | correct |
| fused resident integration | 14.770304 | 0 | performance-only, incorrect |
| delta | **+3.976992 (+36.85%)** | - | +92.488 us/layer |

Lightweight per-SM aggregation preserves the release register and stack
footprints. Identically instrumented 43-layer samples measured 13.407872 ms
for the baseline and 17.235617 ms for the fused integration, a 3.827745-ms
delta versus the 3.976992-ms release delta. Passive phase windows are not
additive: the current FFN tail spills into the next attention window on SMs
that do not execute HC-post.

| matched lightweight profile | baseline (ms) | fused integration (ms) | delta (ms) |
| --- | ---: | ---: | ---: |
| CUDA sample | 13.407872 | 17.235617 | **+3.827745** |
| internal resident span | 13.195168 | 16.982592 | +3.787424 |

The identical-marker CUDA delta accounts for 96.25% of the release delta.
Absolute marker-enabled times are not release TBT. The phase entries below
are passive queue windows on different slowest SMs; they overlap and must not
be added.

| passive queue window | baseline total (ms) | baseline mean (us/layer) | fused total (ms) | fused mean (us/layer) | mean delta (us/layer) |
| --- | ---: | ---: | ---: | ---: | ---: |
| attention / prior-tail wait | 8.528640 | 198.340 | 12.097824 | 281.345 | **+83.005** |
| FFN-local | 9.702368 | 225.636 | 5.299232 | 123.238 | -102.398 |

The attention-window increase is not an attention-math regression. On SMs
without the final HC-post task, the passive FFN-end marker can advance before
the true expert join. Their following attention window then includes the
previous FFN's outstanding tail.

The FFN table reports the mean duration on the slowest participating SM over
43 layers. The old grouped entries are sums along the corresponding local
expert queue. Expert branches and independent stages can overlap, so the
rows are diagnostic components rather than an additive TBT decomposition.

| FFN component | baseline (us) | fused integration (us) | delta (us) |
| --- | ---: | ---: | ---: |
| HC project | 6.148 | 5.629 | -0.519 |
| HC pre/RMS | 14.650 | 15.472 | +0.822 |
| hidden activation quantization | 25.746 | 17.852 | -7.894 |
| router compute-queue interval | 169.708 | 198.181 | +28.473 |
| Top-6 queue interval | 36.824 | 8.376 | -28.448 |
| routed input quant + W1 + W3 + SiLU | 70.654 | fused Linear-1: 54.203 | **-16.451** |
| routed middle quant + W2/down | 23.903 | fused down: 71.365 | **+47.462** |
| shared W1 + W3 + SiLU + middle quant | 57.961 | fused Linear-1: 52.467 | -5.494 |
| shared W2/down | 23.783 | 17.656 | -6.127 |
| HC post | 12.374 | 11.083 | -1.291 |

The router entry is not GEMV arithmetic time. Its begin marker is ahead of
the input-barrier-gated loads; its end marker follows the local compute task's
C2M output push but not STU completion or a grid-wide router barrier. A
representative layer-3 per-step trace separates it as follows:

| router local trace | baseline (us) | fused integration (us) |
| --- | ---: | ---: |
| compute-queue interval | 141.760 | 131.328 |
| M2C / input and data wait | 138.592 | 127.520 |
| compute-active portion | **3.168** | **3.808** |

The structural bottleneck is fused routed down. Its 192 tasks are placed over
152 SM queues, leaving 40 queues with two tasks and 112 with one. The grid
mean is 31.944 us while the slowest-SM mean is 71.365 us (maximum occurrence
74.272 us). Fused routed Linear-1 is balanced at 52.536-us grid mean and
54.203-us slowest-SM mean. Relative to the old routed path, fused Linear-1
saves about 16.451 us per layer, while fused down costs about 47.462 us,
leaving a 31.011-us local regression that is amplified by the all-SM join and
cross-layer barriers. Fix the route dependency and repeat this same breakdown
before changing the down schedule.

| fused routed stage | grid mean (us) | slowest-SM mean (us) | maximum occurrence (us) | slowest/grid |
| --- | ---: | ---: | ---: | ---: |
| Linear-1 | 52.536 | 54.203 | 55.360 | 1.03x |
| down | 31.944 | **71.365** | 74.272 | **2.23x** |

## Retained-ring one-launch phase isolation (2026-08-17)

Isolated Linear-1 and Linear-2 must be measured with the same selected-op
image as the one-launch FFN. Doing so keeps the 204-register, nine-barrier
compute-dispatch footprint identical and separates command-stream integration
from kernel-size effects. Protocol matching also matters: isolated Linear-1
must publish its ready records, and isolated Linear-2 must execute already
satisfied per-record, per-K256 readiness checks.

Matched release job `20260817T222707Z-2955717` used 112 workers on one GB300.
The isolated local medians were 12.384 us for Linear-1 and 12.416 us for
Linear-2; their kernel spans were 13.920 and 13.120 us. Two prebuilt-input
one-launch controls measured 15.008--15.136 us local Linear-1,
12.608--12.752 us local Linear-2, and 29.792--30.464 us for the kernel. Thus
Linear-1 is about 22% slower inside the combined command stream even when
Down reads an independent, already-ready activation tensor. Linear-2's
prebuilt median is close to isolated, but its tail is wider.

With the real Linear-1-to-Down readiness edge, two samples measured
15.680--15.856 us local Linear-1, 13.600--16.752 us local Linear-2, and
32.128--34.144 us kernel spans. The variable Down time is a producer-tail and
placement effect; a global phase span must not be interpreted as one SM's
compute duration. Use the per-worker event-2-to-4 and event-4-to-5 deltas.

Track-profile job `20260817T222940Z-2956490` found zero slot stalls in either
isolated phase but exactly one stalled allocation on every worker in both
full controls. Linear-1 owns a 16-slot retained weight ring plus its four-slot
activation allocation; the allocator runs ahead to the following eight-slot
Down ring before the old ring reaches UMMA last use. The resulting roughly
37% grid-envelope slot-stall counter is overlapping allocator-warp wait, not
a 37% latency attribution. LDU0 queue wait stayed near 9%, compute M2C wait
stayed in the 4--7% range, and LDU dependency wait was zero. In particular,
LDU queue-wait counters measure time waiting for another command, not slow
memory service.

The existing lease-handoff control removed all 112 stalls and reduced
allocator instructions from 336 to 224 in job
`20260817T223150Z-2957240`, but recovered only 0.352--0.384 us of
instrumented Linear-1 time and did not consistently improve the full kernel.
Allocator lookahead is therefore a measured secondary interference source,
not the entire integration gap. Whole-kernel Nsight Compute profiles also
showed full HBM read traffic and executed instructions close to the isolated
sum; do not attribute the remaining gap to extra memory bytes or to a
different compute-handler image without a more local trace.

The Down-prefix sweep and address-alias control refine that conclusion. In
release job `20260817T224459Z-2960877`, adding 0/112/224 prebuilt-input Down
tasks changed Linear-1 local median from 12.240 to 13.216 to 15.568 us,
standard deviation from 0.312 to 0.492 to 0.690 us, and P95 from 12.925 to
14.173 to 16.526 us. A single exposed worker was about 1.18 us slower than
the 111 unexposed workers in the same launch, proving a local memory-command
lookahead effect.

Temporary global-timer trace `20260817T224929Z-2962217` showed the first Down
weight TMA starting about 0.9--1.1 us before the same worker's Linear-1 end,
after its Linear-1 ring reached last use. At full load the earliest Down TMA
can overlap the latest worker's Linear-1 weight stream by less than 1 us.
However, the second Down TMA starts at least 3--5.8 us after every Linear-1
worker finishes, so it cannot directly explain the large 112-to-224 change.

The large steady-state difference is cache-history/working-set interference
across repeated FFN invocations. Effective clocks remain equal near 1.96 GHz;
cold first-shot Linear-1 does not show a consistent 224-task penalty. In
`20260817T225431Z-2963692`, 224 commands with unchanged TMA byte count but
second-wave coordinates aliased onto the first 112 uniform weight records
reduced steady-state Linear-1 from 16.096/16.224 us to 13.440/13.408 us.
Thus an isolated repeated Linear-1 keeps its own weights resident, whereas a
repeated full FFN lets the unique Down weight working set displace them before
the next replay. Treat isolated-versus-full phase timing as a cache-state
comparison unless both paths receive an equivalent preceding weight stream.
The temporary alias and timestamp hooks were removed after this proof.

## Full-FFN cold-data comparison (2026-08-17)

Cold here means cold operand data, not first-process JIT. Code and one-FFN
CUDA graphs are warm. Before every timed replay, the same stream performs a
read/modify/write traversal of a 260-MiB buffer and records the start event
after that traversal. GB200 L2 measured 135,528,448 bytes (129.25 MiB), so the
scrub exceeds twice L2 while remaining outside the timed interval. All paths
use one shared plus six routed experts, eight rows, H4096/I2048, 20 warmups,
and 100 measured one-FFN replays; input/checkpoint quantization and routing
preparation remain setup-only.

| implementation | min | median | P90 | stddev | max (us) |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM/DeepGEMM | 43.264 | **44.016** | 45.536 | 0.768 | 47.872 |
| VDCores task-direct | 43.712 | **44.512** | 45.792 | 0.903 | 47.872 |
| VDCores retained ring | 64.256 | **66.976** | 69.376 | 1.605 | 76.608 |
| FlashInfer grouped composed | 82.624 | **85.696** | 85.760 | 0.933 | 86.816 |

The jobs are `20260817T233055Z-2974569`,
`20260817T233608Z-2976354`, `20260817T233143Z-2974827`, and
`20260817T233118Z-2974712`, respectively, all on worker GPU ordinal 3.
Task-direct is only 0.496 us (1.13%) above DeepGEMM and should be treated as
tied at this run-to-run precision. Retained ring is 22.464 us (50.47%) above
task-direct, while remaining 18.720 us (21.84%) below the FlashInfer
composition.

The FlashInfer number is the legal seven-group composition—grouped FC1,
vLLM SwiGLU, per-expert MXFP8 quantization/packing, grouped FC2, and routed
reduction—not its fused TRTLLM MoE entrypoint. Correctness reruns bracketed the
paired result at 83.680--89.824 us, with mean-relative error 0.2051 and cosine
0.97886 against a BF16 reference. Bounded 300-s/top-7 and 120-s/top-6 attempts
at the fused TRTLLM entrypoint did not finish first tactic setup, so no fused
FlashInfer latency is claimed.

After the task-direct measurement, the installed release image was restored
to the retained-ring full-FFN build. Its compile again reports 204 registers,
nine barriers, an 80-byte stack frame, zero spills, and 3,936 bytes static
shared memory.

## Retained-ring issue-gate rejection and weight prefetch (2026-08-18)

A temporary memory-side issue gate made each worker decrement a device atomic
only after its final Linear-1 gate/up weight TMA issue, then delayed the first
Down ring until 32, 16, 8, or zero issuers remained. It did not improve cold
latency: medians stayed at 66.704--66.944 us versus 66.784--66.928 us for
bracketed ungated baselines. Hot medians were 34.450, 35.586, 35.587, and
36.778 us respectively, versus a 34.625--35.362-us baseline range. The strict
gate loses useful overlap and cannot repair cross-replay cache displacement.
The experiment, including its build option, command encoding, scheduler state,
and device atomics, was removed from the milestone.

The retained implementation instead gives weight TMAs an L2 evict-first
policy and issues `cp.async.bulk.prefetch.tensor` for tile `k+1` after tile
`k`. This is the default; `DAE_MXFP_WEIGHT_PREFETCH=0` remains a constexpr A/B
control. The cleaned release job `20260818T002921Z-2993479` measured a
32.470-us hot median and a 51.824-us cold median (67.296-us P90). Earlier
prefetch runs measured 32.334/32.394 us hot and 49.920/51.984 us cold, versus
34.625--35.362 us hot and 66.784--66.928 us cold without prefetch. Thus the
repeatable hot improvement is 6.2--8.2%, and the cleaned cold median is about
22.5% lower. The cold tail remains bimodal, so the hint has not proven a
stable P90 improvement.

The same 100-warmup, 500-sample, 20-FFN graph hot protocol and 100-sample
one-FFN cold protocol produced this final comparison on worker GPU ordinal 3:

| implementation | hot median | cold median | cold P90 (us) |
| --- | ---: | ---: | ---: |
| VDCores task-direct, two kernels | **26.293** | **45.808** | 46.819 |
| vLLM/DeepGEMM | 30.776 | **44.256** | **45.043** |
| VDCores retained-ring prefetch, one persistent kernel | 32.470 | 51.824 | 67.296 |
| FlashInfer grouped composition | not measured | 84.928 | 85.763 |

The jobs are `20260818T003115Z-2994064`,
`20260818T003158Z-2994351`, `20260818T002921Z-2993479`, and
`20260818T003223Z-2994509`, respectively. Retained prefetch is 6.178 us
(23.5%) slower than task-direct hot and 6.016 us (13.1%) slower cold. It is
1.694 us (5.5%) slower than DeepGEMM hot and 7.568 us (17.1%) slower cold,
while its cold median is 33.104 us (39.0%) below the legal FlashInfer grouped
composition. Every result passed its correctness check.

The selected retained image still compiles at 204 registers, nine barriers,
an 80-byte stack frame, zero spills, and 3,936 bytes static shared memory.
Exact exploratory jobs are recorded in
`.agentlog/2026-08-18-retained-ring-issue-prefetch.md`.

## Coupled weight-scale TMA versus DeepGEMM (2026-08-18)

Moving retained-ring weight scales from the task-owned `cp.async` warp into
the LDU weight loop is not inherently incompatible with a fast kernel.
DeepGEMM's SM100 FP8-by-FP4 1D1D kernel does exactly this: one TMA producer
issues A, B, SFA, and SFB into the same stage, then accounts every transaction
on one `full_barrier`. It prefetches all five tensor-map descriptors at kernel
entry. A separate warp waits for that barrier, transposes the scale-factor
shared-memory image for UTCCP, and publishes a secondary scale-ready barrier.

The exact vLLM/DeepGEMM JIT specializations used by the H4096/I2048 benchmark
explain why that coupling is cheap there. Both FC1 and FC2 select K128,
M32/N128, two-CTA multicast, swap-AB, and **11 TMA stages**. With FP8 A and
unpack-in-SMEM FP4 B, each CTA stage reserves 2 KiB A, 16 KiB B, 512 B SFA,
and 512 B SFB. DeepGEMM's heuristic explicitly spends the remaining shared
memory on as many producer stages as fit. FC2 has sixteen K128 iterations, so
the producer can make eleven resident before recycling a stage. The two-CTA
UMMA covers up to 256 output channels by 32 decode rows per K128 iteration.

VDCores Down instead has two K256 stages, M128/N8 UMMA, and eight K256
iterations. Only two bundles (the equivalent of four K128 blocks) can be
resident before the LDU waits for consumer retirement. Its coupled path issues
the 16-KiB packed-weight tensor TMA, then a descriptorless 1-KiB scale bulk
copy, and makes the one weight-full barrier expect both. The original
task-owned scale producer executes two warp-wide 16-byte `cp.async` issue
iterations for that 1 KiB and can run independently of the LDU weight TMA.
Consequently, moving the small copy into the two-stage LDU pipeline exposes
TMA request admission/completion latency that DeepGEMM hides behind a much
deeper producer window and substantially more UMMA work per stage.

The measured isolated Down control supports this distinction: task-owned
scales measured 12.608 us, coupled LDU TMA scales 13.472 us (+0.864 us,
+6.9%), a separate scale barrier 13.760 us, and a second TMA LDU 13.952 us.
Splitting the barrier or issuer did not create more lookahead or more work
between stage waits, so neither addressed the exposed fixed latency.

DeepGEMM also retires a producer stage directly with
`umma_arrive(empty_barrier)`. VDCores commits UMMA to `umma_full`; a completion
warp waits on that barrier and then arrives `stage_empty`. That relay adds a
small stage-reuse bubble precisely where a two-stage pipeline is sensitive.
DeepGEMM's descriptor-backed, host-prepacked scale layout is a secondary
difference, not the main explanation: its kernel still performs a 32-thread
SMEM transpose and UTCCP before UMMA, while the VDCores scale image is already
in its native UTCCP layout.

The focused next controls are therefore (1) increase retained Down from two
to four K256 stages while keeping the coupled barrier, and then (2) make UMMA
arrive directly on the resident `stage_empty` barrier. Only after those should
a descriptor-backed scale tensor TMA be compared with the current 1-D bulk
copy. The coupled-barrier design itself should not be rejected based on the
current two-stage result.

## DeepGEMM structure transplant controls (2026-08-18)

Follow-up controls rejected the two most literal pipeline transplants. Four
K256 Down stages increased isolated latency from 13.280 to 14.592 us and full
hot latency from 33.437 to 34.965 us; the ring-handoff form also failed
correctness. A constexpr direct-retirement image then made UMMA completion
arrive directly on each resident `stage_empty`, with only a final task-local
completion token. It preserved the 162-register/9-barrier resource footprint
but measured 13.600/14.752 us Linear-1 task/kernel, 14.112 us Down kernel, and
33.457/66.304 us full hot/cold. The relay control was 13.616/14.752,
14.080, and 33.437/66.304 us respectively. Direct retirement is therefore
off the critical path here, and the experiment was removed.

The more fundamental source difference is geometry and ownership. The exact
DeepGEMM specialization dedicates about 217 KiB shared memory to one GEMM,
uses a two-CTA M256/N32 UMMA with one cluster leader, runs one common scheduler
across TMA/MMA/scale-transpose/epilogue roles, and pipelines two output stages.
VDCores uses independent M128/N8 issuers and reserves most dynamic shared
memory for its general allocator. Moving only DeepGEMM's TMA block to an LDU
would omit the clustered instruction footprint and shared scheduler while
adding command publication around it.

A temporary 152-worker placement, intended to copy DeepGEMM's all-SM
scheduling instead of the existing 112-worker/112-Linear-1-task placement,
failed correctness with one missing/NaN Down region in job
`20260818T044740Z-3082907`. Extra Down-only workers need an explicit phase and
reduction-ownership protocol; the placement relaxation was removed.

The architectural direction is now a specialized resident FFN operator, not a
literal DeepGEMM fork split into generic commands. Keep the task-direct fused
math and epilogues—which already measure 26.293 us hot versus fresh DeepGEMM
at 29.768 us—and, if LDU residency is mandatory, make one long-running LDU
producer share a compile-time stage machine with that compute path. Detailed
commands and job IDs are in
`.agentlog/2026-08-18-deepgemm-architecture-compare.md`.

## Specialized DAE2 resident FFN result (2026-08-18)

The specialized resident implementation now stays fully inside VDCores: one
Python enqueue, one persistent `dae2` launch, one queued compute task, and one
queued resident memory task per worker. The fixed two-instruction memory
dispatcher preserves allocator-warp -> LDU queue publication but omits generic
decoder/allocation/M2C work that cannot occur in this image. LDU0 owns the
Linear-1 and two Down weight/scale pipelines; it does not wait on an issue
gate or activation barrier.

LDU1 receives the same queued memory command and prepares the paired expert-0
BF16 reduction destinations. A local pause prevents its readiness traffic from
interfering with Linear-1: LDU1 performs the required clears, then waits on a
CTA-local token; LDU0 releases that token after its Linear-1 stream retires and
immediately continues into Down loading. LDU1 resolves both global
zero-readiness dependencies and publishes two one-shot local tokens consumed
by the two Down epilogues. This keeps the weight stream independent while
moving the actual output-data dependency to the memory side.

The final image uses 246 registers, nine barriers, a 48-byte stack, zero
spills, and 2,208 bytes static shared memory. Hot medians are 28.1936 and
28.1872 us; the 100-sample cold median/P90 are 42.144/43.456 us. All output
checks passed. Against the fresh vLLM/DeepGEMM 29.768-us hot and 44.816-us
cold baselines, this is 5.29-5.31% faster hot and 5.96% faster cold. Jobs are
`20260818T091618Z-3239526`, `20260818T091636Z-3239758`, and
`20260818T091654Z-3239920`.

Nsight Compute confirms that the change removes rather than relocates the
tail dependency. Before local publication, 188 samples were attributed to the
Down epilogue's device-scope readiness load. In the paused version that
hotspot is absent, remaining Down epilogue synchronization accounts for ten
samples, and final-role-join samples drop from 708 to 439. An unpaused control
increased Linear-1 local standard deviation from roughly 0.17 to 0.88 us,
which is direct evidence that even a small second-LDU polling workload can
interfere when all resident warps become runnable at kernel entry.

Full implementation, rejected placement controls, commands, and job IDs are
recorded in `.agentlog/2026-08-18-dae2-resident-ffn.md`.

## Normal-runtime-only resident FFN execution (2026-08-18)

The retained-ring FFN now has one memory-runtime implementation. `dae2`
always enters the generic allocator, store, and single-thread LDU executors;
the fixed fast allocator/LDU dispatchers, runtime LDU selector, terminate-only
store executor, and their build/Python configuration surface were removed.
The resident Linear-1 and auxiliary Down commands remain ordinary memory
instructions: Python emits one immutable command for each LDU FIFO, the
allocator publishes both through M2LD, and each LDU decodes its resident
operator inside the normal loop.

The cleanup preserves the 248-register kernel cap and the software-interleaved
Down metadata fetch. The selected normal image builds at 244 registers, nine
barriers, a 112-byte stack, zero spills, and 2,368 bytes static shared memory.

## Unified coupled-stream normal operator (2026-08-18)

The normal runtime now represents all resident MXFP loads with one
parameterized memory opcode. Linear-1 uses a homogeneous 16-operation
gate/up-weight-plus-SFA stream on LDU0; Down weight-plus-SFA follows on the
same LDU and fixed allocation area; Down activation-plus-SFB uses the same
opcode on LDU1. Python encodes the fixed-area slot count and stage depth in
each command and rewrites the adjacent same-LDU/same-area pair into a local
chain. The allocator still publishes both immutable commands, but LDU0
dequeues the second command inline and reuses the existing slots and barriers
without a release/reallocation round trip.

The four-instruction selected image builds at 243 registers, nine barriers, a
64-byte stack, zero spills, and 2,400 bytes static shared memory. A correctness
smoke passed, and the single bounded performance job
`20260818T225318Z-4092865` measured 31.056 us hot end-to-end and 48.224/48.256
us cold median/P90; device-counter medians were 26.112 us hot and 36.576 us
cold. Output passed the existing BF16 cross-split-K tolerance. This is 4.3%
hot and 7.6% cold slower than the existing 29.768/44.816-us vLLM/DeepGEMM
reference, and 10.2% hot and 14.4% cold slower than the prior specialized
resident image. The result establishes functional genericity and local lease
reuse, not a performance win over the specialized dispatcher.

## VDCores full-FFN device-counter comparison (2026-08-18)

Matched GPU-globaltimer distributions are now recorded for all VDCores FFN
ownership paths. The protocol was 100 warmups, 500 hot samples with 20 FFNs
per graph, and 100 one-FFN cold samples after a 260-MiB L2 scrub. Resident
counters span post-queue-initialization through compute termination. The
task-direct combined counter spans the first Linear-1 kernel entry through the
last Down kernel exit, including the stream-ordered kernel gap.

| path | hot counter median/P90 | cold counter median/P90 | hot/cold CUDA-event median (us) |
| --- | ---: | ---: | ---: |
| task-direct, task-owned loads | 23.840 / 24.128 | 37.568 / 38.253 | 25.492 / 45.184 |
| specialized resident executor | 24.000 / 24.672 | **31.968 / 32.419** | 28.708 / 44.176 |
| unified normal-runtime LDU | 26.112 / 26.560 | 36.576 / 37.408 | 31.056 / 48.224 |

The direct component medians are 14.176 us Linear-1, a 0.960-us inter-kernel
gap, and 8.704 us Down when hot, and 21.952, 1.632, and 14.032 us when cold.
The per-sample combined medians are 23.840 and 37.568 us; medians of correlated
components do not necessarily sum to the combined median. Thus task ownership
remains best hot, by 0.160 us versus the specialized resident executor and
2.272 us versus unified normal LDU. The specialized resident executor is best
cold; unified normal LDU is 4.608 us higher and task-direct is 5.600 us higher.

Jobs are `20260818T225318Z-4092865` (unified),
`20260818T232543Z-207900` (specialized commit `4ae375e`), and
`20260818T234108Z-350610` (task-direct). All passed output checks. The direct
benchmark permanently reports min/median/P90/stddev/max for Linear-1, the
inter-kernel gap, Down, and their combined device envelope. The installed
worker runtime was restored to the four-command unified resident image after
the controls.

## Production FFN conclusion (2026-08-19)

The unified coupled-stream operator is the sole resident FFN load path in the
normal DAE2 runtime. It uses two fixed stages, coupled weight/scale completion,
K+1 streaming-weight prefetch, an LDU0-local Linear-1-to-Down chain, an
independent LDU1 activation/scale stream, Python-initialized BF16 reduction
output, and direct compute-side TMA reduction stores.

The allocator-ring, continuation/handoff, standalone resident commands,
separate scale barriers, pair/LDU output clearing, STU reduction, split-LDU
selection, and FFN tile-timeline controls were removed along with their build
and Python configuration surface. For future archaeology only, the complete
pre-cleanup implementation is retained in commit `79022cc`; production code
does not carry compatibility branches for those experiments.

The cleaned selected image remains at 243 registers, nine barriers, a 64-byte
stack, zero spills, and 2,400 bytes static shared memory. A bounded validation
run (`20260819T004918Z-960160`, 100 hot samples and 20 cold samples) passed all
output checks. It measured 30.838 us hot and 48.288 us cold by CUDA events,
with 25.856 us hot and 35.712 us cold device-counter medians. Relative to the
pre-cleanup unified result, hot latency improved by 0.218 us while cold latency
was unchanged within the short-run variance.

## Full-image FFN/attention unroll limit (2026-08-23)

The production routed MXFP image now leaves the Linear-1 projection loop, the
Linear-1/Linear-2 outer K-bundle loops, and the split64 attention QK-tile loop
rolled. Inner CuTe fragment loops and the ring/UMMA schedule remain unchanged.
In the exact 27-handler full image this reduces `dae2` text from 675,712 to
576,256 bytes (-14.72%), and ptxas allocation from 246 to 242 registers, with
no spills.

The matched 43-layer/context-128 raw device frontier is 16.471360 ms before
and 16.399104 ms after (-0.44%, within launch variation). The isolated resident
FFN changes from 61.664 to 61.696 us, and split64 attention changes by only
+0.2% to +0.4%. All checks pass and the full path emits token 7249. This rules
out handler code size as the dominant source of the approximately 16-ms
single-token time while retaining the substantially smaller image. Full
commands and job IDs are in `.agentlog/2026-08-23-dsv4-full-image-unroll.md`.

## Allocator-lane-owned LDU publication (2026-08-23)

The generic allocator now treats its control mailboxes according to their
actual ownership: allocator lane zero publishes them and one lane in each LDU
consumes them.  The publication-lifetime barrier therefore has three
participants instead of all 32 allocator lanes plus both LDUs.  The remaining
allocator lanes do not access the mailbox; the next allocator iteration's
full-mask shuffle reconverges the warp before lane-derived state is consumed.
This is a normal-runtime change, not a specialized executor or IssueBarrier.

Detail traces showed the old 34-participant barrier had already completed on
both LDUs while allocator lane zero remained unscheduled for about 55 us.  The
compute handler had already begun the following layer, and the allocator took
only 0.064--0.224 us to fetch its next instruction once it resumed.  Explicit
M2C polling sleeps did not alter this delay and regressed repeated-layer time;
warp-leader polling and the legacy all-consumer M2C contract are invalid for
the divergent full image.  Those rejected variants and their build switches
were removed.

Matched profiling-free measurements used the same 29-handler full image, 10
warmups, and 101 samples.  One layer improved from 0.367488 to 0.353312 ms
(-3.9%); two repeated layers improved from 0.892864 to 0.679456 ms (-23.9%).
The 43-layer raw device-frontier minimum improved from 14.815616 to 14.068768
ms (-5.0%).  Its repeated-launch median also improved from 83.301312 to
16.817696 ms, although cross-token reset drift remains tracked separately
from the requested raw single-token estimate.  All cases passed and produced
the same output tokens as their controls.  Ptxas resources are unchanged at
242 registers, ten barriers, a 224-byte stack, and zero spills.

Jobs are `20260823T211950Z-1654299` and
`20260823T212013Z-1657271` (candidate small controls),
`20260823T212248Z-1680329` and `20260823T212310Z-1685772`
(baseline small controls), `20260823T212351Z-1692424` (baseline full), and
`20260823T212721Z-1726488` (candidate full).

## Direct-BF16 routed MXFP Down handoff (2026-08-24)

The routed MXFP4 x MXFP8 Down specialization now reduces directly into the
BF16 `[8,4096]` model handoff.  Expert zero establishes each destination tile
with a TMA copy; routed experts and the selected K1024 halves use BF16 TMA
reduce-add.  The separate FP32 `[8,4096]` accumulator and resident
`FP32_TO_BF16` stage are gone.  Writing converted values directly into the
N-major shared fragment was important: retaining a second BF16 register
fragment measured 184.384 us for one layer, while direct shared stores won.
This intentionally accepts BF16 cross-expert/split-K accumulation error.

The production image builds at 246 registers, ten barriers, a 256-byte stack,
zero spills, and 15,104 bytes static shared memory.  A hot one-layer run
(`20260824T130828Z-1378564`) measured a 172.448-us device-frontier median,
down from the preceding 177.600-us result.  The repeated-launch 43-layer
validation (`20260824T124400Z-1035208`) kept token 201 and improved the device
median from 5.405120 to 5.345760 ms (-59.360 us, -1.10%).  The final hot run
(`20260824T133139Z-1709174`) measured 5.347968 ms device and 5.460128 ms CUDA
median, with token 201, a 2.0 top-logit margin, and 0.1875 maximum repeated
logit delta.

Three adjacent controls were rejected.  Removing the overlapped 16-CTA
native-record split and gathering activation data with one strided 3-D TMA
slowed the matched hot median from 172.448 to 173.920 us: placing that gather
inside LDU delays its first weight transaction, whereas the original split is
hidden under routing.  Refining four Down tiles to K512 quarters reduced the
predicted SM work spread but slowed the median to 173.760 us because eight
extra BF16 reduction transactions cost more than the smaller tail.  Finally,
dropping the unexecuted general FP32 finalizer from the selected switch slowed
the hot median to 178.912 us; moving the existing profile opcode into its
slot recovered only to 176.896 us.  The valid fallback handler therefore
remains selected to preserve the measured 29-case dispatch layout.

Timing and repeated-state validation must remain separate for one-layer
comparisons.  `--validate-each-launch` clones all intermediates between
samples, disturbs cache residency, and moved otherwise comparable medians
into the 184-us range.  Candidates are timed without inter-sample cloning and
then validated separately; full-model validation remains the final accuracy
gate.

## Generation-safe asynchronous barrier-bank reload (2026-08-24)

The repeated HCA/CSA body now owns two shifted dependency-counter banks.  At
each body tail, 16 selected LDUs partition the just-consumed bank into
contiguous slices while the alternate bank begins.  A three-phase worker
counter first gathers every clear command, then proves every worker observed
the old completion counter at zero before any slice can restore it, and
finally publishes the stores.  The last worker increments an adjacent
monotonic ready generation.  This prevents both restoration of the completion
counter in front of a late worker and zero-valued ABA across bank reuse.

Every LDU port receives one compact wait command at the repeated-body entry.
The allocator substitutes the current outer-loop counter as its expected
generation; the command normally observes an already-complete clear and costs
only decode/publication.  It also invalidates handler-local route IDs.  This
boundary is required because SMs absent from the first stage can otherwise
look ahead to later counters, and ping-pong route-record addresses do not
guarantee that every LDU port consumed the alternate address.  It is a direct
bank-generation dependency, not an `IssueBarrier`.  B's first pass remains
free to overlap A's tail because its independent bank begins at generation
zero.

Two more aggressive controls were rejected.  Joining clear completion only
to the nominal root dependency allowed independent per-SM paths to consume
stale counters.  Moving the check into `LOOPM` covered allocator backedges but
did not invalidate LDU-local routing state.  Both produced unstable top-token
results in the ten-pair diagnostic.  The selected LDU-entry boundary made all
21 timed ten-pair launches stable; its cold prime chose the other member of a
near-tied BF16 top-2 pair, consistent with the intentionally order-dependent
BF16 split-K reduction.

CUDA built-in atomics were retained for the hot reload counters.  A direct
microtrack measured 2.880 us for `atomicAdd` versus 3.328 us for
`cuda::atomic_ref` (-13.5%).  A corrected repeated service test measured
3.367 versus 3.545 us (-5.0%); complete pair time did not improve reliably,
so the primitive is not the dominant endpoint cost.

Full job `20260824T201042Z-3720755` used one persistent launch, 43 layers,
context one, 10 warmups, and 101 timed samples.  All launches emitted token
201.  CUDA time was 5.342240/5.358112/5.423136 ms min/median/max and the raw
device frontier was 5.257728/5.269696/5.329376 ms.  The preceding synchronous
5.313216-ms device and 5.406016-ms CUDA medians therefore improve by 43.520 us
(0.82%) and 47.904 us (0.89%).  This is statistically tied with the earlier
coalesced asynchronous sample at 5.267520/5.362368 ms; the generation-safe
form is selected because it closes the reproduced reuse races.  The kernel
remains at 246 registers, ten barriers, a 256-byte stack, zero spills, and
15,104 bytes static shared memory.

An invalid combined-generation entry control was also rejected.  It changed
the expected value from the bank-local outer generation to
`2 * outer + inner`.  Because the grouped entry command already selects a
different ready counter for A and B, B's first pass then waited for generation
one on B's own zero-valued counter.  That counter can be incremented only by
B's tail clear, creating a direct self-cycle.  The matched ten-pair baseline
(`20260824T202303Z-3919480`) measured 2.461536 ms at the device frontier, but
the combined-generation candidate (`20260824T202706Z-3991184`) never completed
its prime launch and was stopped after a bounded wait.  The existing grouped,
bank-local outer generation is already the exact reuse dependency: A0 and B0
both observe generation zero on separate counters, then A1/B1 independently
wait for A0/B0 clear completion.  Keep B's first independent load free so it
can overlap A's tail.  If later profiling supports a B weight fence, its source
must be A's memory-side last-issue publication, not task completion,
bank-clear completion, or an `IssueBarrier`.

## Full-image Linear-1 tile-unroll rejection (2026-08-24)

Operator-first profiling isolated a real Linear-1 scheduling opportunity, but
the full-model promotion gate rejected it.  In the focused routed-FFN image,
leaving the two-projection loop rolled while changing the two eight-tile
compute loops from `#pragma unroll 1` to `#pragma unroll 2` reduced the
instrumented device median from 45.664 to 40.064 us.  Fully cloning both the
projection and tile loops reached 34.112 us.  The gain came from a shorter
ring-empty/UMMA wait envelope, not from faster memory transactions.

The same controls behaved differently in the 29-handler production image.
The fully rolled, mixed, and fully cloned instrumented medians were
48.224/47.712/47.456 us.  The mixed image retained 246 registers, a 304-byte
profile stack, and zero spills, proving that the focused gain largely vanished
through full-image instruction placement/scheduling rather than register
spilling.  The uninstrumented mixed image (`20260824T211525Z-613663`) improved
the isolated hot FFN device median from 45.824 to 45.344 us, but regressed its
cold median from 57.120 to 58.656 us.

Most importantly, the exact 43-layer/context-one promotion
(`20260824T211646Z-637582`) passed token-201 correctness but measured
5.325248 ms at the device frontier and 5.407136 ms by CUDA event.  The
immediately adjacent rebuilt rolled control (`20260824T212223Z-721711`) passed
all 101 timed launches and measured 5.312896/5.399264 ms, making the mixed
unroll 12.352/7.872 us slower in the matched run.  The earlier selected clear
sample remains faster at 5.269696/5.358112 ms; the difference between that run
and the fresh rolled control is launch/clock variation and is not attributed
to source.  Both Linear-1 outer loops therefore remain rolled in production.
Do not promote a focused FFN unroll solely from its isolated result; require
the full-handler-image operator measurement and the 43-layer gate because
cloned control bodies perturb scheduling elsewhere in the persistent kernel.

## Device-call-isolated Linear-1 expansion (2026-08-24)

The rejected inline expansion above identified the right local schedule but
put it in the wrong code-layout scope.  Moving the complete Linear-1 compute
handler behind a `__noinline__` device-call boundary makes the two-projection
and two eight-tile loops safe to fully unroll without cloning their bodies into
the persistent dispatch handler.  A rolled noinline control measured 45.888 us
versus 45.824 us inline, so the call boundary itself is neutral.  Expanding the
body behind that boundary reduced the isolated full-image routed FFN to
39.552 us and its Linear-1 local interval to 24.736 us.  The parent kernel
remains at 246 registers, ten barriers, a 256-byte stack, zero spills, and
15,104 bytes static shared memory; the new child function also has zero stack
and zero spills.

Matched promotion gates confirmed that this is not an isolated-kernel effect.
Two repeated uses of layer zero improved from 342.112 to 326.720 us device
time (-4.50%).  A heterogeneous HCA/CSA pair using layers three and four
improved from 359.776 us (`20260824T214427Z-1103082`) to 339.584 us
(`20260824T214106Z-1048862`), a 20.192-us or 5.61% gain, with the same output
token.

The compile-flag candidate (`20260824T214813Z-1166533`) passed all 101
43-layer/context-one launches with token 201 and measured 5.040288 ms at the
device frontier and 5.119616 ms by CUDA event.  After hard-wiring the selected
layout and removing the exploratory build switches, the clean default rebuild
(`20260824T215257Z-1255189`) reproduced 5.045184/5.128960 ms.  Relative to the
adjacent rolled production control at 5.312896/5.399264 ms, the final default
is 267.712 us (5.04%) faster on-device and 270.304 us (5.01%) faster by CUDA
event.  The source change is deliberately limited to the Linear-1 function
boundary and its three fixed unroll directives.

A second clean production rebuild and full-model verification
(`20260824T222851Z-1855313`) used the 29-op inference image, one persistent
launch, 43 layers, context one, 10 warmups, and 101 timed samples.  Every
sample again emitted token 201.  CUDA-event time was
5.111264/5.122304/5.146080 ms min/median/max, while the raw device frontier was
5.028096/5.038624/5.050880 ms.  The medians are 0.13% faster than the first
clean default run and therefore reproduce the milestone within run-to-run
variation.  The rebuilt kernel retained 246 registers, ten barriers, a
256-byte stack, zero spills, and 15,104 bytes static shared memory.  No Down
scheduling, pipeline-depth, profiling, or other exploratory switch was
enabled in this verification.

## Participant-preserving C2M polling default (2026-08-25)

A narrow production-shaped trace isolated a bimodal FFN output tail in the
legacy C2M `arrive_and_wait` path.  Across eleven one-layer samples, the
compute-end-to-STU-publication tail had a 17.792-us median and reached
22.272 us, while actual STU service remained approximately 0.5 us.  Reusing
the existing exact-token participant-polling queue collapsed every sample to
0.384--0.512 us and improved the one-layer device median from 201.728 to
181.120 us.  The diagnostic timestamp path was removed after attribution.

The same compile-time policy improved one HCA/CSA pair from the recorded
373.088-us direct-wait control to 306.464 us and ten pairs from 2.759008 to
2.385536 ms.  The adjacent 43-layer direct-wait rebuild
(`20260825T035619Z-3124526`) measured 5.838176 ms device and 5.931904 ms CUDA;
the polling build (`20260825T035245Z-3071696`) measured 5.059360 and
5.148768 ms.  This is a 778.816-us device reduction.  The fact that the
previous 5.038624-ms artifact cannot be reproduced by a clean legacy-wait
build is consistent with flag-sensitive build residue having preserved the
polling specialization, although that old binary is no longer available for
direct inspection.  The production source now selects participant polling as
a constexpr and removes the optional Makefile switch.

The final clean default (`20260825T040112Z-3198082`) retained 246 registers,
ten barriers, a 256-byte stack, zero spills, and 15,104 bytes shared memory.
All 101 launches emitted token 201 and measured 5.085408 ms device and
5.176832 ms CUDA median.  This is still 46.784 us slower than the best
historical device median, so it is a reproducibility/runtime correction rather
than a new best-time claim.  Its one-layer control
(`20260825T040045Z-3192349`) measured 178.752 us device and 258.112 us CUDA.

## Context-one attention scheduling audit (2026-08-25)

The context-one critical path was decomposed before changing its schedule.
Across layer-three samples, the eight O_a groups each spend approximately
7.5--8.2 us in their coupled projection and finish within roughly 1 us of one
another.  The whole-vector O-rank native-FP8 quantization then adds about
1.7--2.0 us before O_b.  O_b itself is not dependency limited after release:
its M2C wait is 0.064--0.128 us.  The detailed producer trace is
`20260825T045621Z-4010550`.

Two bounded scheduling changes were rejected.  First, O_b split-K4 increased
the directly measured ready-to-completion span from 4.864--5.184 us to
7.776 us median.  Its one-layer device median was 188.160 us versus the prior
187.584-us split-K8 control, so production remains split-K8.  Jobs are
`20260825T045441Z-3981494` and the adjacent split-K8 bracket
`20260825T045526Z-3992125`.

Second, eight per-group O_a completion barriers allowed each four-SM O-rank
quant slice to overlap its own producer.  With matched two-event profiling it
advanced O_b readiness from 50.080 to 48.704 us, HC-post completion from
59.040 to 57.312 us, and the one-layer device median from 189.024 to
185.440 us (`20260825T045908Z-4052001` versus
`20260825T045959Z-4064231`).  It nevertheless failed the next integration
gate: a heterogeneous HCA/CSA pair regressed from 311.456 to 316.672 us device
and from 398.272 to 402.560 us CUDA (`20260825T050217Z-4095425` versus
`20260825T050128Z-4082242`).  Per-layer profiling localized 8.320 us of the
loss to the first layer and 1.056 us to the second, while allocator, LDU, and
STU command counts were exactly unchanged.  The candidate expanded the graph
from 28 to 35 dependency barriers per layer; duplicating that form in the
unrolled two-body diagnostic also failed to reach prime.  It was removed.
Do not trade the existing whole O_a join for additional fine-grained barriers
without first solving this multi-body liveness and placement cost.

The native-FP8 K8192 quant operator's placement was independently swept to
rule out over-parallelization.  Exact device medians were 3.776, 4.128,
4.640, and 5.600 us for 32, 16, 8, and 4 SMs respectively (jobs
`20260825T051207Z-53169`, `20260825T051217Z-54986`,
`20260825T051227Z-57412`, and `20260825T051236Z-60866`).  The production
32-SM placement is selected and the diagnostic option was removed.

A restored ten-pair profile (`20260825T051401Z-84624`) completed stably with
token 1115 at 2.441056 ms device median.  After the cold first pair, its HCA
positions commonly measured 113--116 us and CSA positions 120--123 us.
Repeating the identical HCA family in the ordinary one-bank loop did not
reproduce a stable odd/even split; reload service was 3.0--3.6 us per layer
(`20260825T051443Z-93500`).  A synthetic two-family tensor-alias control was
invalid because the family counter-stride contract assumes distinct streams
and it did not reach prime.  Therefore the observed HCA/CSA difference is not
yet proof of barrier-bank asymmetry and must not motivate a runtime change
without a valid matched control.

A temporary track-build timestamp was then placed around the actual
`LDU_WAIT_BARRIER` bank-entry gate used by asynchronous pair reload.  In a
ten-pair run, the ten exposed waits were only 0.352--0.608 us each and summed
to 4.928 us (`20260825T052247Z-212056`).  Thus the asynchronous clear is
already almost entirely hidden behind the preceding pair; its worker service
time is not an exposed multi-microsecond critical path.  The timestamp was
removed after attribution.  An attempted isolated layer-four control did not
reach prime under the one-layer reset construction, although layer four is
stable inside the production pair, so no standalone layer-four performance
claim is made from that invalid diagnostic.

## Repeated-layer frontier and FFN attribution (2026-08-25)

Matched step-profile sweeps compared one HCA/CSA pair with the final pair of a
ten-pair run across all 36 stages.  The final context-one attention tail
(O-rank quantization, O-b, and HC-post) retained the same or slightly shorter
local spans in the repeated run.  The apparent roughly 3-us FFN increase in a
coarse stage timestamp was inherited from an earlier producer frontier rather
than generated inside the FFN.  The paired sweeps are
`20260825T053617Z-423483`/`20260825T053640Z-430132`,
`20260825T053741Z-443490`/`20260825T053803Z-451962`,
`20260825T053825Z-456065`/`20260825T053845Z-459977`, and
`20260825T053912Z-468161`/`20260825T054002Z-479782`.

An MXFP FFN-detail build then timestamped dependency readiness, Linear-1, and
Down directly.  In a one-pair run (`20260825T054447Z-545799`), readiness to
Linear-1 completion was 27.424 us and readiness to final compute completion
was 43.712 us.  At the final pair of a ten-pair run
(`20260825T054509Z-551181`), the corresponding spans were 27.200 and
43.296 us.  The complete local CSA layer spans were 125.056 and 125.088 us.
Therefore neither Linear-1 nor Down slows under repetition; do not attribute
the full-run tail to recurrent FFN resource interference without a new direct
counterexample.

## Rejected HC-projection queue and placement changes (2026-08-25)

Fixed LDU port assignments improved the isolated HC projection from 5.824 to
5.440 us with weight on LDU0 (`20260825T054907Z-613247` and
`20260825T055015Z-631654`) and to 5.472 us with weight on LDU1
(`20260825T055329Z-680888`).  Neither integrated.  Prefetching the weight
before activation readiness made a one-layer run 206.240 us versus the
198.304-us control, and fixed LDU0 without prefetch measured 207.264 us.  The
reverse mapping appeared faster in one layer at 195.968 us but regressed a
heterogeneous pair from 318.912 to 326.816 us
(`20260825T055435Z-701022` versus `20260825T055457Z-705345`).  All fixed-port
and prefetch code was removed.  A K16384 construction was also rejected before
launch because its 65,536-byte weight command exceeds the generic load
command's 16-bit byte-count field; it provided no kernel performance result.

Finally, placing the projection reset on SMs 16--151 while hidden quantization
used SMs 0--15 improved a one-layer device median from 203.712 to 196.544 us.
It failed the production-like pair gate: the unprofiled median regressed from
318.912 to 324.928 us.  A matched profile did show small local improvements
(HCA 140.864 to 139.552 us, CSA 122.688 to 122.592 us, and total device
frontier 328.544 to 324.960 us), proving that the reset/quant join can advance
but also that the effect is not robust enough to explain the unprofiled
integration result.  The candidate job is `20260825T060301Z-816856`; its
diagnostic placement option was removed rather than promoted.

## HC pre-RMS metadata-prefetch rejection (2026-08-25)

The HC pre-RMS packed metadata address can be published before its residual
input is ready, which lets LDU0 begin the residual transfer while LDU1 waits
on the projection dependency.  In a tracked one-layer run this reduced the
HC pre-RMS critical median from 6.080 to 4.800 us and the complete device
frontier from 199.744 to 193.920 us (`20260825T061934Z-1080490` and
`20260825T062226Z-1111712`).  It did not generalize to the FFN placement, and
moving the task off the projection SM only converted the same dependency into
an exposed wait.

The ten-pair gate rejected the change.  Its matched control measured
2.445728 ms device and 2.531648 ms CUDA (`20260825T062946Z-1217093`), while
the metadata-prefetch schedule measured 2.466112/2.551168 ms
(`20260825T063012Z-1223825`).  A final-layer trace showed that the small local
HC pre-RMS benefit was offset by longer following projection and reset waits:
early residual traffic was contending with later LDU/allocator work.  The
prefetch placement and all diagnostic options were removed.

## Packed HC projection record direct-output path (2026-08-25)

Operator profiling found a persistent row-zero straggler in the 24-SM HC
projection.  Row zero produces the ordinary projection scalar, residual
square sum, and packed pre-RMS metadata, while peer rows produce only one
scalar.  The tracked control measured a 4.640-us critical projection median
against approximately 2.400 us for peer tasks; 2.112 us of the critical span
was M2C wait.  The following HC pre-RMS measured 6.080 us
(`20260825T061934Z-1080490`).

The selected implementation treats the adjacent square sum and row-zero
projection scalar as one raw output record.  Compute writes those two FP32
values directly to their final addresses, avoiding two tiny compute-to-shared
to-STU data roundtrips.  It still follows the VDcores publication contract:
compute returns the raw writeback token through C2M after its stores, STU
first drains the queued metadata-tail store, and a no-copy raw STU command
releases the stage output barrier.  The producer uses special mailbox 27;
mailbox 24 cannot be reused because the immediately following pre-RMS raw
input also occupies 24 and special mailboxes bypass allocator occupancy.  A
dump showed the conflicting commands at pc 0 and pc 5
(`20260825T071808Z-1957567`).  This solution adds neither a fence nor an
IssueBarrier.

The initial diagnostic that relied on the unrelated metadata-tail STU
publication was faster but was not selected: it lacked a compute-store to STU
release edge, and one full-model repetition showed a 7.1875 maximum logit
variation.  Adding the correct raw writeback first reused mailbox 24 and hung
at prime; assigning mailbox 27 fixed the descriptor lifetime.  The ordered
one-layer run passed 101 samples at 180.960 us device
(`20260825T071926Z-1974203`).  Its HCA/CSA pair measured 296.736 us versus the
adjacent 297.408-us control, and ten pairs measured 2.344064 ms versus
2.371552 ms (`20260825T071949Z-1980707`,
`20260825T070026Z-1693847`, `20260825T072014Z-1985609`, and
`20260825T070051Z-1699076`).

Two clean 43-layer/context-one runs emitted token 201 in every timed launch.
They measured 4.993088/5.090784 and 4.993728/5.084576 ms at the device/CUDA
medians (`20260825T072041Z-1988767` and
`20260825T072149Z-2002059`).  Relative to the accepted
5.085408/5.176832-ms production control (`20260825T040112Z-3198082`), the
reproduced result saves 91.680 us (1.80%) on-device and 92.256 us (1.78%) by
CUDA event.  Repeated-logit maximum variation was 1.125 and 1.03125 in the two
ordered runs.  The kernel remains at 246 registers, ten barriers, a 256-byte
stack, zero spills, and 15,104 bytes static shared memory.

## Post-milestone one-layer critical-frontier baseline (2026-08-25)

The next optimization cycle began from the selected packed-record path and
profiled all queued layer steps in the same 29-handler image.  Two bounded
windows avoid event aliasing; jobs are `20260825T072859Z-2114608` and
`20260825T072930Z-2124951`.  The large raw elapsed values for projection
reset, FFN quant/split/route, and the resident FFN are not serialized costs:
those tasks intentionally begin early and spend most of the interval waiting
for their producer.  Ready-to-end frontiers are the relevant measure.

The directly exposed attention spans were 6.129 us for Q-b, approximately
7.44--7.96 us for each parallel O-a group, 1.391 us for O-rank
quantization, 4.861 us for O-b, and 1.908 us for HC-post.  In the FFN half,
the spans were 2.880 us for HC projection, 3.986 us for HC pre-RMS,
3.314 us for MXFP8 quantization, 1.461 us for router GEMV completion,
0.811 us for activation-layout split, 2.944 us for prepared top-6 routing,
45.830 us for the resident FFN, and 1.775 us for HC-post.  The resident FFN
span agrees with its existing isolated full-image measurement, so it is not
evidence of new attention/FFN interference.  Prepared routing finishes about
0.94 us after the quant/split branch and is therefore the producer that gates
resident-FFN readiness in this layer.  Any next orchestration change should
target that bounded route-to-resident path, or provide new evidence that an
earlier independent load delays it; do not optimize the intentionally hidden
wait envelopes themselves.

## Prepared-route publication and placement rejection (2026-08-25)

A temporary raw-output timestamp split the prepared top-6 handoff into its
compute-to-C2M dequeue, no-copy STU service, and publication wake-up pieces.
Across eleven control samples, the corresponding means were 0.250, 0.233,
and 0.244 us, or 0.727 us total (`20260825T074555Z-2368718`).  An adjacent
control reproduced 0.215/0.227/0.227 us, or 0.669 us total
(`20260825T074942Z-2421912`).  There is therefore no long C2M queue tail,
slow STU command, or barrier wake-up to remove; the route branch's remaining
lag is its top-6 computation rather than its VDcores publication path.

Per-worker timestamps also tested whether route placement on vcore 128 makes
that worker the resident-FFN straggler.  Vcores 129--131 finished at the same
frontier, 132--135 only 0.6--0.8 us earlier, and the short-work vcores
144--151 approximately 3.7 us earlier.  Moving route to vcore 144 shared its
prior quant/split STU traffic: publication plus wake-up increased from about
0.73 to 1.30 us and the tracked one-layer device median regressed from
188.544 to 191.584 us (`20260825T074707Z-2383134`).  Vcore 132 produced
contradictory 185.280- and 195.680-us samples and then lost to the adjacent
183.936-us restored control (`20260825T074819Z-2396992`,
`20260825T074900Z-2407720`, and `20260825T074942Z-2421912`).  Production
therefore retains vcore 128.  All route-publication and placement diagnostics
were removed; do not replace the existing C2M/STU release with a direct
compute atomic or add an IssueBarrier based on this profile.

## Coupled-FP8 issue and dispatch-layout rejections (2026-08-25)

The next operator-first cycle used the unchanged 29-handler full inference
image.  Its eleven-sample control (`20260825T075756Z-2558450`) measured
5.993 us for Q-b, 7.484 us averaged across the eight O-a groups, 1.591 us for
O-rank quantization, and 4.893 us for O-b.  The one-layer device frontier was
198.144 us in that tracked run.

Moving the large coupled-FP8 handler behind a device-call boundary was a
direct regression (`20260825T080112Z-2596197`): Q-b, mean O-a, and O-b rose
to 7.121, 8.087, and 5.536 us, while the layer rose to 206.016 us.  Ptxas also
moved from 246 registers/304-byte profile stack to 248 registers/432 bytes.
The handler remains force-inlined.

Temporary LDU and compute-phase counters then established why generic
load-side changes are unlikely to help.  For Q-a, both coupled LDU services
finish ahead of the task; source loads have 0--0.032 us wait, the first two
stage-empty waits are only 0.032--0.096 us, and the post-issue compute tail is
about 2.8--3.2 us (`20260825T080801Z-2709222` and
`20260825T081511Z-2816550`).  The latter diagnostic showed the scale-copy/UMMA
issue interval dominating its instrumented coupled task; ring release, output
allocation, epilogue, and publication together added only about 0.35 us.
All temporary timestamps and reporter accommodations were removed.

Partially unrolling the dynamic K-pair loop by two did not reduce the coupled
operator spans (`20260825T082054Z-2905094` and
`20260825T082129Z-2913413`).  The adjacent rolled control
(`20260825T082439Z-2958205`) was also faster at the layer frontier, so the
loop remains rolled.

Context-one O-a split-K4 provided a useful scheduling counterexample.  It
filled all sixteen SMs assigned to each group and reduced mean O-a from
7.441 to 5.448 us.  Its O-a join and O-b completion advanced by 4.535 and
4.055 us (`20260825T082647Z-2988959`,
`20260825T082726Z-2998150`, and control
`20260825T082802Z-3007926`).  Nevertheless it failed the next integration
gate: an HCA/CSA pair measured 313.120 us versus bracketed split-K2 controls
at 309.504 and 309.696 us (`20260825T083034Z-3043025`,
`20260825T083007Z-3036937`, and `20260825T083105Z-3051516`).  Doubling the
partial-result contributors creates more recurring allocator/STU/reduction
pressure than its local compute saving, so production remains split-K2.

Two switch-layout experiments were also rejected.  Adding an early coupled
opcode branch perturbed surrounding handlers and raised the tracked layer to
196.064 us (`20260825T083621Z-3124162`).  Conversely, removing the two early
split-attention tests reduced the parent from 246 to 244 registers, but its
unprofiled one-layer median was effectively tied with the restored control:
193.312 versus 193.472 us (`20260825T084226Z-3218960` and
`20260825T084535Z-3265422`), with mixed operator deltas.  The dense normal
switch and its existing split-attention placement are retained unchanged.

## HC pre-RMS coefficient-split rejection (2026-08-25)

The next operator-first cycle profiled the fused HC pre-RMS task before
changing its schedule.  The unchanged full-image operator measured 5.376 us
hot (`20260825T090315Z-3525277`).  A timestamped build measured 5.760 us and
showed that the three hidden/RMS worker warps completed their BF16 output at
4.352 us, while warp zero did not finish the Sinkhorn coefficient path until
5.376 us (`20260825T091114Z-3653086`).  This was direct evidence for a
roughly one-microsecond local warp-zero tail, not evidence for an LDU or slot
stall.

Two smaller changes failed the isolated gate.  Rolling the 20-iteration
Sinkhorn loop increased the instrumented median from 5.760 to 5.856 us
(`20260825T091437Z-3702544`).  Replacing two warp-zero M2C pops with a shared
worker token tied the 5.760-us envelope but extended task publication from
5.888 to 6.144 us (`20260825T092213Z-3796346`).  Divergently publishing the
hidden output before warp zero completed was also invalid: normal C2M FIFO
phases require all compute participants to advance together, and both tested
orderings stalled before prime.  No divergent queue path was retained.

A separate normal compute task then tested whether the coefficient work could
overlap the following attention/FFN body.  Hidden/RMS alone was correct at
4.672 us, 1.088 us faster than the instrumented fused task
(`20260825T100653Z-240861`).  The coefficient task was also numerically exact,
but its isolated envelope was 13.696 us.  Direct timestamps attributed 2.048
us to its two M2C operands and 10.496 us to the dependent 4x4 Sinkhorn
shuffle/divide chain; the 64-byte direct output issue was below the 32-ns
timestamp resolution (`20260825T102908Z-588670`).  Staging the 224-byte input
through LDU and redundantly executing the chain in all four compute warps did
not materially shorten that serial dependency, so neither memory delivery nor
an STU output copy was the missing optimization.

The first integrated image exposed a separate descriptor-lifetime bug.  Both
the attention and FFN coefficient commands used raw special mailbox 26 on
vcore 130.  Because special mailboxes bypass allocator occupancy, the later
descriptor could overwrite the earlier one before STU drained its token; the
attention join then never received its release.  The command dump in
`20260825T102212Z-477347` showed the two uses at memory PCs 2 and 6.  Assigning
distinct mailboxes removed the deadlock and preserved the established
layer-zero token 2835.

The repaired one-layer schedule still failed the performance gate.  Its
device/CUDA medians were 200.896/295.616 us
(`20260825T102503Z-518641`), versus the matched tracked control's 191.264-us
device frontier (`20260825T085440Z-3405948`).  The coefficient task occupied
vcore 130 after hidden readiness, so the following all-152-SM projection reset
could not complete until that vcore finished the 10.5-us serial chain.  The
local hidden-output advance therefore became a larger placement delay.  The
separate opcode, schedules, compile flags, timestamps, and extra dependency
groups were removed; production keeps the fused warp-zero coefficient path.
The restored 29-handler image rebuilt at 246 registers, ten barriers, a
256-byte stack, and zero spills; its isolated fused task passed at a 4.960-us
hot median (`20260825T103811Z-738661`).

## Prepared-route and premature-LDU-wait rejection (2026-08-25)

The next operator-first cycle decomposed the prepared score-routing task in
the unchanged 29-handler image.  Its refined isolated result was 3.584 us hot
and exact (`20260825T105806Z-870201`).  Entry-to-score readiness consumed
1.536 us; after the scores arrived, candidate loads, six selections,
normalization/output, and publication occupied 1.280 us.  The raw output
descriptor was already available when the score operand arrived.

A score-routed layer-3 trace then separated the two M2C operands and the
surrounding queued steps (`20260825T105638Z-869164`).  The route task entered
at 14.304 us, received its prepared scores at 70.304 us, received its ungated
raw output descriptor 0.096 us later, and published at 72.960 us.  The
128-worker router GEMV ended at 68.832 us, so its 2-KiB output TMA contributed
a directly measured 1.472-us producer-to-consumer tail.  The complete route
task spent 56.032 of 59.264 us in M2C waits; these intentionally early queue
times are not compute costs.

MXFP-FFN detail counters proved that early resident commands overlapped the
route interval (`20260825T110319Z-873772`).  On route vcore 128, the LDU1
Down-activation command entered at 5.120 us and did not pass its internal poll
until 99.872 us.  The already data-barriered LDU0 Linear-1 command entered at
72.800 us, passed its dependency at 75.328 us, and issued no weight traffic
before that point.  Therefore the overlap was waiting-warps rather than an
early weight TMA.  CUTLASS's transaction-barrier wait uses a timed
`mbarrier.try_wait` with a 10,000,000-tick timeout; it is not a tight software
poll loop.

Putting every Down-activation command on the existing route/split data
dependency looked positive only in the timestamp-heavy diagnostic image:
the layer frontier moved from 192.704 to 186.432 us
(`20260825T110319Z-873772` and `20260825T110506Z-874559`).  The real
246-register production image rejected it.  A clean control measured
176.416/264.992 us device/CUDA (`20260825T111010Z-876887`), while the
all-worker fence measured 179.840/266.912 us
(`20260825T111051Z-877171`).  Delaying useful Down setup cost more than the
diagnostic waiting-warp saving.

A fence restricted to route vcore 128 measured 176.352/264.032 us
(`20260825T111130Z-877495`), but bracketed production controls were 176.416
and 179.936 us (`20260825T111010Z-876887` and
`20260825T111202Z-877758`).  Its 0.064-us apparent device improvement over
the first control is below run-to-run variation and was also removed.  No
IssueBarrier was introduced.  Production retains the original ungated LDU1
Down-activation command and the necessary direct data barrier on dynamic
Linear-1 weights.

## Generation-safe asynchronous barrier reload default (2026-08-25)

The repeated HCA/CSA gate directly compared the synchronous bank reload with
the existing generation-safe asynchronous clear.  Both runs used ten repeats
of the same two-layer body, ten warmups, and 31 measured launches.  The
synchronous control measured 2.350592 ms device and 2.435456 ms CUDA
(`20260825T143207Z-982763`); the 16-worker asynchronous path measured
2.306208/2.393280 ms (`20260825T143442Z-984151`).  The saving is 44.384 us
device and 42.176 us CUDA over twenty layers, or about 2.22/2.11 us per layer.
The clear remains an ordinary LDU operation on disjoint workers and joins the
useful HC-post completion through its normal data dependency.  It introduces
neither an IssueBarrier nor a thread fence.

The promoted 43-layer run used one persistent kernel launch, ten warmups, and
101 measured launches.  Every sample emitted token 201.  Device time was
4.898688--4.965760 ms, with a 4.920960-ms median; CUDA-event time was
5.010944 ms median (`20260825T143514Z-984349`).  Relative to the prior
4.993088/5.090784-ms milestone, this saves 72.128 us device and 79.840 us CUDA.
It is 38.49% below the historical 8-ms boundary and 18.20% below the
6.0157-ms vLLM reference on device time.  Repeated-layer numerical sensitivity
was also reproduced in the synchronous control, so it is not evidence of an
asynchronous-clear race.  The asynchronous path is now the compile-time
default; an explicit `async_barrier_reload=0` still produces the synchronous
diagnostic image.

Two alternative clear mechanisms were rejected before promotion.  A
register-load/store template copy on 16 workers measured 2.308704/2.396576 ms
over ten pairs (`20260825T144759Z-988484`), slightly behind the intrinsic
asynchronous clear.  An eight-worker TMA template copy produced overlapping
A/B ranges: 2.294528/2.389824 and 2.307232/2.393120 ms for the candidates,
versus 2.352640/2.428800 and 2.302016/2.399008 ms for adjacent controls
(`20260825T144927Z-989000`, `20260825T145007Z-989274`,
`20260825T145045Z-989451`, and `20260825T145138Z-989729`).  Copying a pristine
barrier template therefore adds allocator, compute, and STU/TMA traffic without
removing an exposed critical path; all template-copy code was removed.

The asynchronous worker width was finally bracketed at the repeated-pair gate.
Eight workers exposed clear latency at 2.371296 ms device
(`20260825T152018Z-999660`).  Thirty-two workers measured 2.301408 ms with ten
warmups and 2.331712 ms with 100 warmups
(`20260825T152243Z-1000422` and `20260825T152911Z-1002357`).  However the
unchanged 16-worker image itself ranged from 2.301440 to 2.381184 ms across
adjacent rebuilds/runs, including 2.371360 and 2.301440 ms with 100 warmups
(`20260825T152556Z-1001407`, `20260825T152645Z-1001708`, and
`20260825T153150Z-1003080`).  The apparent 16-versus-32 delta reverses inside
the measured run-state variation, so production retains 16 workers.  The
restored production image compiles at 246 registers, ten barriers, a 256-byte
stack, zero spills, and 15,104 bytes static shared memory.

## Down second-wave placement rejection (2026-08-25)

MXFP FFN-detail profiling tested whether consistently late first-wave workers
should be exempted from the split-K Down second wave.  The candidate retained
the exact 152 full plus 144 half-task decomposition and changed only the eight
workers that received no second task.  It failed the one-layer gate: an
adjacent control measured 0.187072 ms device with a 43.840-us
ready-to-compute-end FFN frontier (`20260825T151408Z-997255`), while the
closing candidate measured 0.202304 ms and 54.912 us
(`20260825T151449Z-997600`).  A first candidate sample was also slower at
0.191328 ms (`20260825T151315Z-997000`).  Aggregate first-wave stragglers were
not stable across runs, and remapping the entire subsequent split record stream
perturbed reduction/cache placement more than it helped the selected workers.
The remap was removed; production retains the contiguous split-record mapping.

## Reproducible fixed-context framework baselines (2026-08-27)

The production comparison now has a checked-in, one-process-per-context
framework harness and complete environment bootstraps under
`benchmarks/framework_baselines/`.  Both references use the full
`DeepSeek-V4-Flash-NVFP4` checkpoint, batch one, BF16 model/head activations,
FP8 KV, two generated tokens, three warmups, and 21 samples.  The metric is
the first-to-second engine-token interval, so prefill, graph capture, tuning,
and JIT are excluded while the measured attention length is exact.  Context
one remains unavailable for framework comparison because it has no
prompt-backed first-to-second-token equivalent.

At contexts 128, 256, 512, and 1024, accepted VDcores launch-inclusive
medians are 5.326880, 5.406688, 5.475424, and 5.477216 ms.  vLLM 0.27.1
medians are 6.874858, 7.523229, 6.870377, and 6.906794 ms; SGLang
0.5.12.post1 medians are 7.289749, 7.319158, 7.324534, and 7.439609 ms.
The full matrix, raw min/p90/max distributions, job IDs, exact dependency
versions, setup commands, and the benchmark-only vLLM TMA-compatible KV-pool
stride repair are recorded in `benchmarks/framework_baselines/README.md`.
