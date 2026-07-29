"""End-to-end PoolInst EP with Grace verbs payload delivery.

Metadata queues, dynamic-read dependency resolution, expert gather, weighted
reduction/scatter, and retirement are the ordinary VDCores PoolInst protocol.
Only remote data plus its existing HBM ready generation use the host SGL ring.
"""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import statistics
import threading

from mpi4py import MPI
import torch

import dae.nvshmem as nvshmem
from dae.pool_slice import (
    PoolSliceStatus,
    allocate_pool_slice,
    build_pool_slice_copy_program,
)
from host_sgl_benchmark import (
    HostSglDcEpochRingGroup,
    HostSglDcQueue,
    HostSglEndpoint,
    HostSglEpochRingGroup,
    HostSglQueue,
    _collective_call,
    _gpudirect_write_ordering,
    _register_storage,
)
from nccl_ep_reference import Timing, print_result, rank_max
from pool_slice_nccl_compare import _balanced_expert_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-pe", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument("--experts-per-pe", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--route-placement",
        choices=("clustered", "source-local", "remote-clustered", "spread"),
        default="clustered",
    )
    parser.add_argument("--pool-blocks", type=int, default=32)
    parser.add_argument("--data-groups", dest="group_limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--symmetric-size", default="1G")
    parser.add_argument("--port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=0)
    parser.add_argument("--requested-sge", type=int, default=64)
    parser.add_argument(
        "--requested-send-wr",
        type=int,
        default=128,
        help="host verbs SQ depth; transport capacity only, not protocol state",
    )
    parser.add_argument(
        "--paired-device",
        action="store_true",
        help=(
            "alternate device and host PoolInst epochs with the same verbs "
            "resources to suppress process- and time-local fabric drift"
        ),
    )
    parser.add_argument("--host-transport", choices=("dc", "rc"), default="dc")
    parser.add_argument("--ib-device")
    parser.add_argument(
        "--registration",
        choices=("auto", "dmabuf", "legacy"),
        default="auto",
        help=(
            "GPU HBM MR mechanism; auto tries DMA-BUF and then legacy "
            "peer-memory, while explicit modes support allocation-local A/Bs"
        ),
    )
    parser.add_argument("--pcie-bar1", action="store_true")
    parser.add_argument(
        "--library",
        type=Path,
        default=Path(__file__).with_name("host_sgl") / "libhost_sgl_verbs.so",
    )
    return parser.parse_args()


def _endpoint_bytes(endpoint: HostSglEndpoint) -> bytes:
    return ctypes.string_at(ctypes.byref(endpoint), ctypes.sizeof(endpoint))


def run_host_pool(args: argparse.Namespace, runtime, comm: MPI.Comm) -> tuple[Timing, dict]:
    signals = nvshmem.init_signal_space(runtime.num_pes)
    expert_capacity_rows = runtime.num_pes * args.tokens_per_pe
    local_routes = args.tokens_per_pe * args.top_k
    buffers = allocate_pool_slice(
        signals,
        num_pes=runtime.num_pes,
        my_pe=runtime.pe,
        local_readers=args.experts_per_pe,
        token_capacity=args.tokens_per_pe,
        route_capacity=local_routes,
        expert_capacity_rows=expert_capacity_rows,
        hidden_size=args.hidden_size,
        dtype=torch.bfloat16,
        group_limit=args.group_limit,
        pool_blocks=args.pool_blocks,
        in_place_expert_output=True,
        weighted_return=True,
        host_data_plane=True,
    )
    if buffers.data_plane_arena is None:
        raise RuntimeError("host data-plane arena was not allocated")

    token_values = torch.arange(
        runtime.pe * args.tokens_per_pe * args.hidden_size,
        (runtime.pe + 1) * args.tokens_per_pe * args.hidden_size,
        dtype=torch.float32,
        device=buffers.token_pool.device,
    ).view_as(buffers.token_pool)
    buffers.token_pool.copy_((token_values.remainder(97) - 48).to(torch.bfloat16))
    returned = nvshmem.zeros(
        (args.tokens_per_pe, args.hidden_size), dtype=torch.bfloat16
    )
    global_ids = runtime.pe * args.tokens_per_pe + torch.arange(
        args.tokens_per_pe, dtype=torch.int64
    )
    expert_ids = _balanced_expert_ids(
        global_ids,
        top_k=args.top_k,
        num_pes=runtime.num_pes,
        experts_per_pe=args.experts_per_pe,
        placement=args.route_placement,
    ).reshape(-1)
    source_rows = torch.arange(
        args.tokens_per_pe, dtype=torch.int64
    ).repeat_interleave(args.top_k)
    buffers.write_routes(
        expert_ids,
        source_rows=source_rows,
        origin_rows=source_rows,
        route_weights=torch.full((local_routes,), 1.0 / args.top_k),
    )
    buffers.prepare(buffers.token_pool, returned)

    exported = None
    registration = None
    queues: list[HostSglQueue] = []
    dc_queue: HostSglDcQueue | None = None
    rings = []
    worker = None
    try:
        native_ordering = _gpudirect_write_ordering()
        if native_ordering < 200:
            raise RuntimeError(
                "PoolInst host delivery requires native GPUDirect owner ordering"
            )
        exported, registration, export_error = _collective_call(
            comm,
            "pool data-plane HBM registration",
            lambda: _register_storage(buffers.data_plane_arena, args),
        )
        mr = registration.mr.contents
        peers = [pe for pe in range(runtime.num_pes) if pe != runtime.pe]
        if peers and args.host_transport == "dc":
            dc_queue = _collective_call(
                comm,
                "host PoolInst DC creation",
                lambda: HostSglDcQueue(
                    args.library.resolve(),
                    context=registration.context,
                    pd=registration.pd,
                    port=args.port,
                    gid_index=args.gid_index,
                    requested_send_wr=args.requested_send_wr,
                    requested_send_sge=args.requested_sge,
                ),
            )
            for _ in peers:
                rings.append(
                    _collective_call(
                        comm,
                        "host PoolInst ring creation",
                        dc_queue.create_coherent_ring,
                    )
                )
            preliminary_endpoint = _endpoint_bytes(
                _collective_call(
                    comm, "host PoolInst DCT endpoint query", dc_queue.endpoint
                )
            )
            preliminary_by_rank = comm.allgather(preliminary_endpoint)
            first_remote = HostSglEndpoint.from_buffer_copy(
                preliminary_by_rank[peers[0]]
            )
            _collective_call(
                comm,
                "host PoolInst DCT activation",
                lambda: dc_queue.activate_target(first_remote),
            )
            local_endpoint = _endpoint_bytes(
                _collective_call(
                    comm, "host PoolInst active DCT query", dc_queue.endpoint
                )
            )
            endpoints_by_rank = comm.allgather(local_endpoint)
            remote_endpoints = [
                HostSglEndpoint.from_buffer_copy(endpoints_by_rank[peer])
                for peer in peers
            ]
            _collective_call(
                comm,
                "host PoolInst DC connection",
                lambda: dc_queue.connect(remote_endpoints),
            )
        else:
            for _ in peers:
                queue = _collective_call(
                    comm,
                    "host PoolInst QP creation",
                    lambda: HostSglQueue(
                        args.library.resolve(),
                        context=registration.context,
                        pd=registration.pd,
                        port=args.port,
                        gid_index=args.gid_index,
                        requested_send_wr=args.requested_send_wr,
                        requested_send_sge=args.requested_sge,
                    ),
                )
                queues.append(queue)
                rings.append(
                    _collective_call(
                        comm,
                        "host PoolInst ring creation",
                        queue.create_coherent_ring,
                    )
                )

            local_endpoints: list[bytes | None] = [None] * runtime.num_pes
            for peer, queue in zip(peers, queues, strict=True):
                local_endpoints[peer] = _endpoint_bytes(
                    _collective_call(
                        comm, "host PoolInst endpoint query", queue.endpoint
                    )
                )
            endpoints_by_rank = comm.allgather(local_endpoints)
            for peer, queue in zip(peers, queues, strict=True):
                remote_bytes = endpoints_by_rank[peer][runtime.pe]
                if remote_bytes is None:
                    raise RuntimeError(f"PE {peer} did not publish its paired QP")
                endpoint = HostSglEndpoint.from_buffer_copy(remote_bytes)
                _collective_call(
                    comm,
                    "host PoolInst QP connection",
                    lambda q=queue, e=endpoint: q.connect(e),
                )

        local_memory = (
            buffers.delivery_pool.data_ptr(),
            buffers.return_inbox.data_ptr(),
            buffers.control.data_ptr(),
            int(mr.rkey),
        )
        memory_by_rank = comm.allgather(local_memory)
        peer_records = [(0, 0, 0, 0, 0) for _ in range(runtime.num_pes)]
        for peer, ring in zip(peers, rings, strict=True):
            remote_delivery, remote_return, remote_control, remote_rkey = (
                memory_by_rank[peer]
            )
            peer_records[peer] = (
                int(ring._memory),
                remote_delivery,
                remote_return,
                remote_control,
                remote_rkey,
            )
        peer_routes = torch.tensor(
            peer_records, dtype=torch.uint64, device=buffers.token_pool.device
        )
        buffers.prepare_host_data_plane(peer_routes, local_lkey=int(mr.lkey))
        program = build_pool_slice_copy_program(
            buffers,
            benchmark_barrier=nvshmem.benchmark_barrier,
            in_place_identity=True,
            source_preloaded=True,
            host_data_plane=True,
        )
        device_program = None
        if args.paired_device:
            device_program = build_pool_slice_copy_program(
                buffers,
                benchmark_barrier=nvshmem.benchmark_barrier,
                in_place_identity=True,
                source_preloaded=True,
                host_data_plane=False,
            )

        if dc_queue is not None:
            ring_group = HostSglDcEpochRingGroup(dc_queue, rings)
        else:
            ring_group = HostSglEpochRingGroup(queues, rings)
        rounds = args.warmup + args.iterations
        worker_error: list[BaseException] = []
        paired_device: dict[str, float] = {}

        device_gather: list[float] = []
        device_tail: list[float] = []
        device_total: list[float] = []
        device_phases: dict[str, list[float]] = {
            "first_data_published": [],
            "metadata_closed": [],
            "payload_done": [],
            "first_gather": [],
            "compute_ready": [],
            "first_return_put": [],
            "return_payload_done": [],
            "scatter_done": [],
        }

        def progress() -> None:
            try:
                for _ in range(rounds):
                    ring_group.consume()
            except BaseException as error:
                worker_error.append(error)
                print(
                    f"PE {runtime.pe} host progress failed: {error}",
                    flush=True,
                )

        worker = threading.Thread(target=progress, name="pool-host-progress")
        worker.start()

        gather_samples: list[float] = []
        tail_samples: list[float] = []
        total_samples: list[float] = []
        phase_samples: dict[str, list[float]] = {
            "first_data_published": [],
            "metadata_closed": [],
            "payload_done": [],
            "first_gather": [],
            "compute_ready": [],
            "first_return_put": [],
            "return_payload_done": [],
            "scatter_done": [],
        }

        def run_device_epoch(iteration: int, sequence: int) -> None:
            assert device_program is not None
            buffers.set_sequence(sequence)
            torch.cuda.synchronize(runtime.device)
            comm.Barrier()
            device_program.launch()
            torch.cuda.synchronize(runtime.device)
            gather_ns, tail_ns, total_ns = device_program.timing_ns()
            overlap = device_program.overlap_timing_ns()
            overlap.update(device_program.weighted_return_timing_ns())
            if iteration < args.warmup:
                return
            device_gather.append(rank_max(comm, gather_ns / 1.0e6))
            device_tail.append(rank_max(comm, tail_ns / 1.0e6))
            device_total.append(rank_max(comm, total_ns / 1.0e6))
            for name in device_phases:
                value = overlap.get(name)
                if value is not None:
                    device_phases[name].append(
                        rank_max(comm, float(value) / 1.0e6)
                    )

        def run_host_epoch(iteration: int, sequence: int) -> None:
            buffers.set_sequence(sequence)
            torch.cuda.synchronize(runtime.device)
            comm.Barrier()
            program.launch()
            torch.cuda.synchronize(runtime.device)
            if worker_error:
                raise RuntimeError(f"host progress failed: {worker_error[0]}")
            gather_ns, tail_ns, total_ns = program.timing_ns()
            overlap = program.overlap_timing_ns()
            overlap.update(program.weighted_return_timing_ns())
            if iteration < args.warmup:
                return
            gather_samples.append(rank_max(comm, gather_ns / 1.0e6))
            tail_samples.append(rank_max(comm, tail_ns / 1.0e6))
            total_samples.append(rank_max(comm, total_ns / 1.0e6))
            for name in phase_samples:
                value = overlap.get(name)
                if value is not None:
                    phase_samples[name].append(
                        rank_max(comm, float(value) / 1.0e6)
                    )

        # Alternate which path runs first so neither measurement inherits a
        # fixed warm-cache or transport-order advantage. The host progress
        # worker sees exactly one host epoch per iteration and remains idle
        # during the ordinary device PoolInst epoch.
        sequence = 0
        for iteration in range(rounds):
            if device_program is not None and iteration % 2 == 0:
                sequence += 1
                run_device_epoch(iteration, sequence)
            sequence += 1
            run_host_epoch(iteration, sequence)
            if device_program is not None and iteration % 2 != 0:
                sequence += 1
                run_device_epoch(iteration, sequence)

        worker.join()
        worker = None
        if worker_error:
            raise RuntimeError(f"host progress failed: {worker_error[0]}")

        if device_program is not None:
            paired_device = {
                "gather_ms": statistics.median(device_gather),
                "tail_ms": statistics.median(device_tail),
                "total_ms": statistics.median(device_total),
            }
            for name, samples in device_phases.items():
                if samples:
                    paired_device[f"{name}_ms"] = statistics.median(samples)

        torch.testing.assert_close(returned, buffers.token_pool, rtol=0, atol=0)
        status, senders, _, returned_slices, dispatch_ready = buffers.control_state()
        if status != PoolSliceStatus.OK:
            raise RuntimeError(f"PoolInst status is {status.name}")
        if senders != runtime.num_pes or returned_slices != runtime.num_pes:
            raise RuntimeError("PoolInst sender/return retirement is incomplete")
        if dispatch_ready != sequence:
            raise RuntimeError("PoolInst final generation is incomplete")

        model = {
            "protocol": "pool-gather-streaming-host-data",
            "pool_blocks": buffers.pool_count,
            "data_group_limit": buffers.group_limit,
            "host_transport": args.host_transport,
            "host_peer_qps": 1 if dc_queue is not None else len(queues),
            "host_max_sge": (
                [dc_queue.max_sge]
                if dc_queue is not None
                else [queue.max_sge for queue in queues]
            ),
            "host_final_data_wrs": (
                ring_group.posted_data_wrs if ring_group is not None else []
            ),
            "registration": registration.registration_mode,
            "registration_export_error": export_error,
            "paired_device": paired_device,
            "paired_schedule": (
                "alternating" if device_program is not None else None
            ),
        }
        for name, samples in phase_samples.items():
            if samples:
                model[f"median_{name}_ms"] = statistics.median(samples)
        return Timing(gather_samples, tail_samples, total_samples), model
    finally:
        if worker is not None:
            worker.join()
        for ring in rings:
            ring.close()
        for queue in queues:
            queue.close()
        if dc_queue is not None:
            dc_queue.close()
        if registration is not None:
            registration.close()
        if exported is not None:
            os.close(exported.fd)


def main() -> None:
    args = parse_args()
    if min(
        args.tokens_per_pe,
        args.hidden_size,
        args.experts_per_pe,
        args.top_k,
        args.pool_blocks,
        args.iterations,
    ) <= 0 or args.warmup < 0:
        raise ValueError("sizes must be positive and warmup nonnegative")
    comm = MPI.COMM_WORLD
    runtime = nvshmem.init(symmetric_size=args.symmetric_size)
    try:
        if runtime.num_pes != comm.Get_size() or runtime.pe != comm.Get_rank():
            raise RuntimeError("NVSHMEM and MPI rank topology differs")
        if runtime.num_pes < 2 or runtime.num_pes > 8:
            raise RuntimeError("the host PoolInst experiment supports 2--8 PEs")
        result = run_host_pool(args, runtime, comm)
        comm.Barrier()
        if runtime.rank == 0:
            print(
                "host-pool configuration: "
                f"pes={runtime.num_pes} tokens/pe={args.tokens_per_pe} "
                f"hidden={args.hidden_size} experts/pe={args.experts_per_pe} "
                f"top_k={args.top_k} placement={args.route_placement} "
                f"pool_blocks={args.pool_blocks} warmup={args.warmup} "
                f"iterations={args.iterations}"
            )
            print_result("host-pool-slice", *result)
    finally:
        nvshmem.finalize()


if __name__ == "__main__":
    main()
