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
has an address handle, but does not consume a separate send QP. One RDMA WQE
concatenates up to eight noncontiguous HBM source runs into the destination
interval; adjacent indexed rows first collapse into one run. The eight-SGE cap
keeps the DCI address-vector WQE within the provider-supported encoding. An
inline eight-byte RDMA write publishes readiness after the data on the same
ordered DCI. The default 128-WR queue window permits several peer batches in
flight while retaining one transport queue. A compile-time 512-row request
bound covers the supported inference token capacity even when a low PoolInst
group limit produces groups larger than 32 rows.

RC remains available under `--host-transport rc`. It uses one QP per peer and
currently outperforms the single serialized DCI at four and eight PEs, at the
cost of more QP state. DCI remains the low-resource/default experiment.

## Ordering

- GPU producers and Grace consumers use release/acquire operations on
  per-slot monotonic generations. There is no `threadfence_system`, logical
  timeout, reset handshake, or shared producer atomic.
- Data and its ready write share one ordered transport queue. Native GH200
  GPUDirect owner ordering lets PoolInst consume HBM after observing readiness.
- Metadata and payload are independent planes and remain concurrently issued.
  A local GPU generation opens host descriptor publication after the metadata
  publisher has submitted its packet. It adds no remote acknowledgement,
  fence, completion phase, or retirement branch, so NIC delivery still
  overlaps.
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

### Profile/optimize update (2026-07-28)

The Grace verbs engine now merges consecutive indexed source rows into one
SGE. Arbitrary noncontiguous rows keep their original indexed semantics. At
two PEs this reduced complete dispatch+return data WQEs from 16 to 12 for 128
tokens and from 32 to 20 for 256 tokens; a clean 128-token end-to-end sample
was 0.243 ms. `--requested-send-wr` exposes SQ capacity. `--paired-device`
alternates ordinary device and host epochs, reversing their order every
iteration, in the same process and with the same verbs resources.

Same-process comparisons showed that host overhead is shape dependent and
that process/QP placement can move metadata closure by hundreds of
microseconds for both backends. Device remains preferable for the smallest
shape; host SGL amortizes descriptor overhead at larger shapes. A CTA-scoped
NVSHMEM quiet between metadata and host data was explicitly rejected after a
256-token regression to 1.542 ms. Do not use that quiet as NIC arbitration.

At four PEs, a single DC initiator serialized three destinations: one
128-token run measured 1.034 ms and used 24 final data WRs. One RC QP per peer
reduced this to 0.502 ms and 18 WRs, while the paired device path moved with
the same metadata-closure noise. Thus host transport selection is
scale-specific: retain one DCI as the low-resource/default experiment, but
use per-peer RC for the current four-PE performance path. Alternating paired
epochs are required before attributing an absolute delta to host submission.

### Eight-PE consolidation (2026-07-28)

RC and DC now coalesce adjacent source rows within each request. They also
merge consecutive requests when their remote intervals are adjacent, while
retaining one ordered readiness write per queue instruction. Arbitrary
noncontiguous source lists keep true SGL behavior. At 8 PEs/128 tokens the RC
path commonly finishes with two to five data WQEs per peer rather than six.

The compact 32-bit PoolInst route format benefits host and device equally.
One 8-PE/128-token paired run measured 0.620 ms for host RC and 0.629 ms for
the device path in the same verbs process. At 256 tokens, representative
paired runs were 1.100/1.050 ms and later 1.777/1.671 ms; nearly all movement
was the shared metadata-closure boundary, not host descriptor or return time.

Keeping one merged RC batch in flight per peer measured 0.832 ms versus
0.835 ms for the unrestricted peer pipeline and was removed. Pinning the
Grace progress thread likewise had no stable gain because every MPI rank had
all 72 CPUs available. No timeout, delayed batching window, metadata quiet, or
new device-side poll was added.

DMA-BUF and legacy peer-memory registration both work on Vista GH200. Their
sequential results drift with the same paired-device metadata phase, so
`--registration auto` remains the portable default and explicit modes are for
allocation-local A/Bs. Merely keeping the host MR/QPs alive can make standalone
NVSHMEM metadata much slower; always report the alternating paired device
control. Reusing NVSHMEM's already-registered MR keys would be the clean next
system optimization if a supported API becomes available.
