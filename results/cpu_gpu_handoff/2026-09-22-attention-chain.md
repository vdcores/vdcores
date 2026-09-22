# VDCores Attention CPU/GPU Handoff Chain

Date: 2026-09-22  
Instance: `anant_testing`  
Tested implementation: `986d27b80bf358eab9458a278cf7b848eaa43b5d`  
CUTLASS dependency: `098de2a652cf8f00fd70b2df54051c7eccbb855a`

## What This Adds

This is the first complete fixed heterogeneous dependency chain:

```text
VDCores split-KV attention
  -> GPU system-scope atomic: counter 0 -> 1
  -> CPU observes 1 and computes a 1024-element dot product
  -> CPU atomic acknowledgement: counter 1 -> 2
  -> prequeued GPU consumer observes 2 and computes a 1024-element dot product
  -> GPU system-scope atomic: counter 2 -> 4
  -> CPU observes complete chain
```

The GPU consumer and its completion signal are one CUDA kernel. This avoids a
host-side CUDA API call behind a GPU polling kernel, which can synchronize and
prevent the CPU from issuing its acknowledgement.

The matched blocking baseline runs the same attention, CPU dot product, and GPU
consumer. Its difference is orchestration: it synchronizes after attention,
runs the CPU operation, then launches and synchronizes the GPU consumer. The
handoff path prequeues both GPU stages and connects them with the coherent
counter.

## Environment

- NVIDIA GH200 480GB, driver 580.105.08
- CUDA toolkit 12.8.93
- PyTorch 2.7.0 with CUDA 12.8
- Python 3.10.12
- Linux 6.8.0-1046-nvidia-64k, ARM64
- Hardware host-page-table coherence and host-native atomics enabled

The selective VDCores build contained:

- `OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split`
- `OP_ATTN_SPLIT_POST_REDUCE`
- `OP_TERMINATEC`

## Repeated Results

| Trial | Iterations | Blocking median | Handoff median | Median saved | Reduction |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 3.373 ms | 2.838 ms | 0.535 ms | 15.87% |
| 2 | 100 | 3.389 ms | 2.833 ms | 0.556 ms | 16.40% |
| 3 | 100 | 3.392 ms | 2.839 ms | 0.553 ms | 16.31% |
| Final | 1,000 | 3.389 ms | 2.847 ms | 0.541 ms | 15.97% |

Final 1,000-iteration distribution:

| Measurement | Blocking median | Handoff median | Blocking p95 | Handoff p95 |
|---|---:|---:|---:|---:|
| Attention completion visible to CPU | 2.677 ms | 2.828 ms | 2.843 ms | 3.016 ms |
| CPU operation | 11.904 us | 13.057 us | 12.929 us | 14.592 us |
| Post-CPU GPU consumer | 694.726 us | 4.192 us | 841.991 us | 5.152 us |
| Full chain | 3.389 ms | 2.847 ms | 3.579 ms | 3.034 ms |

The full-chain p95 improved by 0.545 ms, or 15.22%. Split-attention output
matched the PyTorch reference with normalized mean error `0.002002`, well below
the configured `0.05` limit. CPU and GPU dot products were checked exactly.

## Meaning

The important result is not that the GPU dot product became faster. The
handoff path moves its launch before the CPU work and leaves it waiting on the
coherent counter. Once the CPU acknowledges completion, the already-resident
GPU consumer finishes and reports completion in 4.192 us median. The blocking
path spends 694.726 us after the CPU step on host-side launch, pointer setup,
and device synchronization.

The handoff is not free: making attention completion visible through the
counter costs about 151 us relative to the blocking attention boundary. The
prequeued consumer avoids roughly 691 us later in the chain, producing the net
541 us end-to-end improvement. This supports the core VDCores-Embedded idea:
explicit cross-device dependencies can outperform stop-the-world synchronization
and relaunch when the next operation is known early enough to prequeue.

This result also explains the earlier standalone round-trip measurement. Fresh
PyBind/CUDA launches measured around 540 us, while a prequeued GPU waiter reacts
to the CPU acknowledgement and completes useful work in about 4 us. Most of the
standalone number was launch/orchestration overhead, not coherent-atomic latency.

## Limits And Next Question

This remains a fixed prototype, not a CPU virtual-core scheduler or dynamic
placement policy. The GPU consumer is a purpose-built wait-and-dot kernel, the
CPU and GPU operations are small, and results are from one GH200. The next
question is where the crossover lies as CPU work, GPU work, and contention vary;
that workload sweep should precede a dynamic CPU-versus-GPU placement policy.

Complete raw results, including three repeat trials, are in
`2026-09-22-attention-chain.json`.
