# Grace Host Data Plane for PoolInst

The host backend is a compile-time `PoolInst` specialization, not a second EP
protocol. Its only substitutions are the two remote payload publication sites
in `include/dae/pool_slice.cuh`:

- dispatch activation bytes followed by the normal data-ready generation;
- weighted-combine bytes followed by the normal return-ready generation.

Route metadata, per-source ordered queues, destination row reservation,
dynamic gather, reduction, scatter, dependency checks, and retirement all run
through the same `pool_slice_exchange_streaming` implementation as the device
backend. The final zero-row host-ring entry is only the ordered data-plane epoch
notification; it does not replace PoolInst `END` queue instructions.

## Transport

`PoolSliceHostWeightedExchange` publishes compact descriptors to coherent
Grace/GPU rings allocated with ordinary aligned `malloc`. A descriptor names
HBM row indices, a contiguous remote interval, the remote ready word, and its
monotonic generation. It never stages activation bytes in host memory.

The Grace worker drives one mlx5 DCI per PE and one local DCT target. Each peer
has an address handle, but does not consume a separate send QP. One RDMA write
concatenates up to eight noncontiguous HBM rows into the destination interval;
the eight-SGE cap keeps the DCI address-vector WQE within the provider-supported
encoding. An inline eight-byte RDMA write publishes readiness after the data on
the same ordered DCI. The 128-WR queue window permits several peer batches in
flight while retaining one transport queue. A compile-time 512-row request
bound covers the supported inference token capacity even when a low PoolInst
group limit produces groups larger than 32 rows.

RC remains available as a comparison/fallback under `--host-transport rc`.
It uses one QP per peer and was measurably more intrusive to NVSHMEM IBGDA at
eight PEs, even when idle.

## Ordering

- GPU producers and Grace consumers use release/acquire operations on
  per-slot monotonic generations. There is no `threadfence_system`, logical
  timeout, reset handshake, or shared producer atomic.
- Data and its ready write share one ordered transport queue. Native GH200
  GPUDirect owner ordering lets PoolInst consume HBM after observing readiness.
- Metadata and payload are independent planes and remain concurrently issued.
  No host-specific metadata fence, completion phase, dependency, or retirement
  branch is present.
- Ring entries execute in order. The zero-row entry retires one host transport
  epoch only after all earlier data and ready writes complete.

## Entry Points

- Common ABI: `include/dae/pool_host_abi.h`
- GPU coherent-ring publisher: `include/dae/pool_host.cuh`
- PoolInst ABI/operator: `include/dae/pool_slice_abi.cuh` and
  `include/dae/pool_slice.cuh`
- Grace verbs engine: `benchmarks/host_sgl/host_sgl_verbs.cc`
- End-to-end EP harness: `benchmarks/pool_slice_host_e2e.py`
- Standalone transport harness: `benchmarks/host_sgl_benchmark.py`

Build the helper with `make -C benchmarks/host_sgl` and the VDCores extension
with `make nvshmem-pyext`.

## Vista Verification (2026-07-28)

Exact BF16 hidden-7168, top-8 weighted dispatch/combine correctness passed at
2, 4, and 8 PEs and at 32, 128, and 256 tokens/PE. The final two-PE 32-token
run measured 0.143 ms end to end; the final eight-PE 32-token run measured
0.477 ms. Earlier eight-sample DC sweeps measured 0.181/0.417/0.431 ms at
2 PEs and 32/128/256 tokens, 0.335/0.630/1.080 ms at 4 PEs, and
0.440/0.973 ms at 8 PEs for 32/128 tokens. Eight-PE 256-token measurements
were sensitive to simultaneous host-payload/IBGDA metadata HCA contention;
increasing the single DCI window from 32 to 128 WRs improved one matched run
from 2.706 to 1.973 ms, but repeated runs remained variable.

The retained result is therefore the simpler one-DCI protocol, not DCI streams
or a metadata-wide quiet. DCI streams added completion bookkeeping without a
stable gain. A metadata quiet occasionally helped one short run but regressed
longer samples and small messages, so it was removed. Future 8-PE tuning should
target HCA arbitration/pacing without changing the PoolInst control protocol.
