import statistics
import time
import traceback

import torch
import torch.nn.functional as F

import dae.nvshmem as nvshmem
from dae.instructions import NvshmemPut, NvshmemWait
from dae.launcher import *
from dae.schedule import Schedule
from dae.util import dae_app, tensor_diff


class ManualGemm(Schedule):
    def __init__(self, Atom, MNK, tmas, MNK_base, num_k_folds):
        super().__init__()
        self.Atom = Atom  # Tiled GEMM instruction type/factory.
        self.MNK = MNK  # Dimensions of the GEMM subproblem handled by this schedule.
        self.tmas = tmas  # Accessors for loading A/B tiles and reducing/storing C tiles.
        self.MNK_base = MNK_base  # Global M/N/K offset where this subproblem begins.
        self.num_k_folds = num_k_folds  # Number of TileK-wide contributions accumulated per output tile.

    def schedule(self, sm: int):
        if sm < 0:
            return []

        BaseM, BaseN, BaseK = self.MNK_base # Obtain starting coords
        TileM, TileN, TileK = self.Atom.MNK # Obtain tile sizes
        loadA, loadB, reduceC = self.tmas  # Unpack the A-load, B-load, and C-output accessors.
        M, N, K = self.MNK # Obtain this subproblem's M and N extents.

        instructions = [] # Allocate the instruction list

        m_tiles = M // TileM # Compute number of M tiles
        n_tiles = N // TileN # Compute number of N tiles

        for m_tile_id in range(sm, m_tiles, self.num_sms): # Distribute output-row tiles across SMs in a strided pattern.
            m = BaseM + m_tile_id * TileM # compute M coord

            for n_tile_id in range(n_tiles): # Generate every output-column tile for this SM-owned M tile.
                n = BaseN + n_tile_id * TileN # compute N coord

                instructions.append(self.Atom(self.num_k_folds)) # Creates and appends one GEMM compute instruction for the current output tile (m, n)

                for fold in range(self.num_k_folds): # Loop that walks through every K tile that contributes to the current output tile
                    k = BaseK + fold * TileK # compute K coord

                    instructions.append(loadB.cord(n, k)) # Load the proper tile from matrix B

                    instructions.append(loadA.cord(m, k)) # Load the proper tile from matrix A

                instructions.append(reduceC.cord(m, n)) # Reduce/store the completed accumulator into the current C output tile.

        return instructions # Return the completed list of instructions


def seeded_rand(shape, seed, dtype, device):
    # Create a reproducible random tensor without modifying PyTorch's
    # global random-number generator. Values are in approximately [-0.5, 0.5).

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    return (torch.rand(shape, dtype=dtype, device=device, generator=generator) - 0.5)


def device_timed(fxn):
    # Record CUDA events immediately before and after fxn submits its
    # stream work. Launcher/TmaTensor/schedule construction occurs
    # outside this helper and is counted separately in build_ns.
    #
    # If fxn performs host-side work before it finishes submitting the
    # GPU operation, the GPU can sit between the two event records while
    # waiting for that submission. The resulting interval is therefore
    # an upper bound on pure device execution time.
    #
    # Synchronizing the end event after every call also serializes the
    # measured launches/chunks: one measured operation completes before
    # the next begins, so this benchmark does not measure launch overlap.
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    fxn()
    end_evt.record()
    end_evt.synchronize()

    return start_evt.elapsed_time(end_evt)  # Result is in milliseconds (stored as a float)


# ============================================================================
# Iteration configuration (identical to the single-GPU script)
# ============================================================================

WARMUP_ITERS = 2 # Number of warmup iterations
PIPELINE_ITERS = 10 # Number of pipeline iterations
TOTAL_ITERS = WARMUP_ITERS + PIPELINE_ITERS # Total number of iterations

# Device-time phases are reported per iteration.
PHASES = ("gemm1", "silu", "gemm2", "comm", "add")


def main():
    runtime = nvshmem.init(symmetric_size="512M")  # Initialize NVSHMEM with a 512 MB symmetric memory heap for each PE.

    tp_rank = runtime.pe # Store this process's NVSHMEM PE number as its tensor-parallel rank.
    tp_size = runtime.num_pes # Store the total number of PEs participating in tensor-parallel execution.

    torch.cuda.set_device(runtime.device) # Set the active CUDA device to the GPU assigned to this PE.

    gpu = torch.device("cuda", runtime.device) # Create a reusable PyTorch device object for the GPU assigned to this PE.

    try:
        if tp_size != 2: # Stop immediately unless the program was launched with exactly 2 PEs.
            raise RuntimeError(
                "This test requires exactly 2 PEs, "
                f"got {tp_size}"
            )

        dtype = torch.bfloat16 # Use BF16 tensors for the GEMMs and intermediate activations.


        Atom = Gemm_M64N64K64 # Select the VDCores GEMM operation.
                              # Each operation computes a 64 x 64 output tile using K chunks of size 64.

        TileM, TileN, TileK = Atom.MNK # Extract the tile dimensions supported by this GEMM operation.

        # Full model dimensions.
        M = TileM * 128
        N = TileN * 8
        K = TileK * 16
        N_out = TileN * 8

        num_sms = 128 # Number of streaming multiprocessors
        n_chunk_tiles = 4 # Process four output-column tiles in each GEMM launch.

        assert K % TileK == 0 # Make sure that the dimensions work
        assert M % TileM == 0
        assert N % TileN == 0
        assert N % TileK == 0
        assert N % tp_size == 0
        assert N_out % TileN == 0

        # Each PE owns half of the hidden dimension. The hidden dimension is the width of the intermediate activation between the two linear layers of the MLP.
        N_local = N // tp_size

        # ====================
        # matA: M × K
        # matB: N × K
        # matC = matA @ matB.T
        # So:
        # matC: M × N
        # ====================

        assert N_local % TileN == 0
        assert N_local % TileK == 0

        MAT_A_SEED = 0 # These are just fixed random seeds used to generate reproducible tensors (for repeatable testing and fair comparisons)
        MAT_B_SEED_BASE = 100
        MAT_D_SEED_BASE = 200

        # ================================================================
        # Tensor allocation (allocations occur before timing)
        # ================================================================

        # Replicate the full M x K input activation on both PEs so each rank
        # can compute its own column-parallel shard of the first GEMM.
        matA = seeded_rand((M, K), seed = MAT_A_SEED, dtype=dtype, device=gpu)

        # Store this PE's N_local x K row shard of the first-layer weight matrix.
        # PE 0 and PE 1 use different seeds so their shards concatenate into the
        # full N x K matrix used by the correctness reference.
        matB_local = seeded_rand((N_local, K), seed = MAT_B_SEED_BASE + tp_rank, dtype=dtype, device=gpu)

        # Allocate the M x N_local hidden-activation shard produced by this PE's
        # column-parallel first GEMM before applying SiLU.
        matC_local = torch.zeros(M, N_local, dtype=dtype, device=gpu)

        # Store this PE's N_out x N_local column shard of the second-layer weight
        # matrix. It multiplies this rank's hidden shard and produces a full
        # M x N_out partial output for the row-parallel second GEMM.
        matD_local = seeded_rand((N_out, N_local), seed = MAT_D_SEED_BASE + tp_rank, dtype=dtype, device=gpu)

        # ================================================================
        # NVSHMEM allocation
        # ================================================================

        # init_signal_space(n) collectively allocates a zero-initialized symmetric
        # uint64 tensor of exactly n signal entries and barriers. We allocate one
        # entry per iteration (warmup + measured), so no signal is ever reused and
        # an unused entry is guaranteed to be zero (a WAIT on it genuinely blocks).
        
        # A signal that was set in iteration i must never be reused in iteration
        # i+1: PE 1's WAIT would fall through immediately on the already-set
        # signal, corrupting both correctness and timing.
        signals = nvshmem.init_signal_space(TOTAL_ITERS)

        # Same symmetric address on both PEs.
        # PE 0 computes its partial here. It then transfers that partial
        # into PE 1's matching symmetric allocation.
        received_partial_0 = nvshmem.zeros(M, N_out, dtype=dtype)

        # PE 1's own second-GEMM partial must remain separate from the
        # incoming PE 0 partial.
        local_partial_1 = torch.zeros(M, N_out, dtype=dtype, device=gpu)

        # Final output buffer. Allocate before timing so allocation cost
        # is excluded from the measured pipeline.
        res = torch.empty(M, N_out, dtype=dtype, device=gpu)

        # Number of TileK-wide reduction chunks required by each first-GEMM
        # output tile across the full input dimension K.
        num_k_folds_first = K // TileK

        # Number of TileN-wide hidden-output tiles owned by this PE in the
        # column-parallel first GEMM.
        n_tiles_local = N_local // TileN

        # Number of TileK-wide reduction chunks required by each second-GEMM
        # output tile across this PE's local hidden shard.
        num_k_folds_second = (N_local // TileK)

        # Number of TileN-wide output tiles needed to produce this PE's full
        # M x N_out partial result in the row-parallel second GEMM.
        n_tiles_second = (N_out // TileN)

        def run_pipeline(signal_id: int, measure: bool):
            # ============================================================
            # Per-iteration reset (outside the timed region)
            # ============================================================

            matC_local.zero_()
            received_partial_0.zero_()
            local_partial_1.zero_()
            res.zero_()

            torch.cuda.synchronize(runtime.device)

            # Both PEs must have finished resetting before either starts
            # the timed pipeline.
            nvshmem.barrier()

            # Approach-2 instrumentation accumulators. build_ns counts
            # host-side launcher/TmaTensor/schedule construction;
            # device_ms counts CUDA-event-timed stream work per phase.
            build_ns = 0
            device_ms = {phase: 0.0 for phase in PHASES}

            # Wall-clock timing of the whole pipeline (PE 1 only), same
            # scope as the uninstrumented version so results stay
            # comparable across runs.
            pipeline_start_ns = None
            if measure and tp_rank == 1:
                pipeline_start_ns = time.perf_counter_ns()

            # ============================================================
            # First GEMM: column parallel
            #
            # Each PE computes:
            #
            # matC_local = matA @ matB_local.t()
            #
            # matC_local shape: M x N_local
            # ============================================================

            # Process this PE's first-GEMM output columns in groups of
            # n_chunk_tiles TileN-wide tiles. With the current dimensions, this PE
            # owns four N tiles, so the full local hidden shard fits in one launch.
            for n_base_tile in range(0, n_tiles_local, n_chunk_tiles):
                cur_n_tiles = min(n_chunk_tiles, n_tiles_local - n_base_tile) # Use fewer tiles for the final launch if the remaining local output width is smaller than n_chunk_tiles.

                N_chunk = cur_n_tiles * TileN # Compute what size chunk needs to be dealt with
                BaseN = n_base_tile * TileN # Compute starting location for N dimension

                build_start_ns = time.perf_counter_ns() # Start timing the build portion

                dae = Launcher(num_sms, device=gpu) # Create the Launcher

                loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K) # Define loadA

                loadB = TmaTensor(dae, matB_local).wgmma_load(TileN, TileK * Atom.n_batch, Major.K) # Define loadB

                reduceC = TmaTensor(dae, matC_local,).wgmma("reduce", TileM, TileN, Major.K) # Define reduceC

                gemm = ManualGemm(Atom, MNK=(M, N_chunk, K), tmas=(loadA, loadB, reduceC), MNK_base=(0, BaseN, 0), num_k_folds=num_k_folds_first).place(num_sms, base_sm=0) # define the GEMM

                dae.i(gemm, TerminateC(), TerminateM()) # Add the schedule and termination instructions to the launcher

                build_ns += time.perf_counter_ns() - build_start_ns # Obtain the total build time

                device_ms["gemm1"] += device_timed(lambda: dae_app(dae)) # Launch the chunk and add its serialized CUDA-event interval to the first-GEMM device-time total.

            torch.cuda.synchronize(runtime.device) # Make sure all first GEMM launches have been completed before continuing

            # Local SiLU activation. (This remains inside the timed pipeline.)
            silu_result = {}

            device_ms["silu"] += device_timed(lambda: silu_result.__setitem__("t", F.silu(matC_local).contiguous())) # Launch the SiLU and store the result with timing data collected

            matSilu_local = silu_result["t"] # Retrieve the SiLU output tensor that was stored in the temporary dictionary here

            torch.cuda.synchronize(runtime.device) # Ensure this PE's SiLU operation is complete before GEMM 2 reads matSilu_local.

            # ============================================================
            # Second GEMM: row parallel
            #
            # Each PE computes:
            #
            # matC_partial =
            #     matSilu_local @ matD_local.t()
            #
            # Each partial has full shape M x N_out.
            # ============================================================

            if tp_rank == 0:
                # PE 0 writes directly into the symmetric source buffer.
                matC_partial = received_partial_0
            else:
                # PE 1 keeps its own partial separate from the receive buffer.
                matC_partial = local_partial_1

            for n_base_tile in range(0, n_tiles_second, n_chunk_tiles):
                cur_n_tiles = min(n_chunk_tiles, n_tiles_second - n_base_tile)

                N_chunk = cur_n_tiles * TileN
                BaseN = n_base_tile * TileN

                build_start_ns = time.perf_counter_ns()

                dae = Launcher(num_sms, device=gpu) # Create the launcher

                loadA = TmaTensor(dae, matSilu_local,).wgmma_load(TileM, TileK, Major.K) # Define loadA

                loadB = TmaTensor(dae, matD_local).wgmma_load(TileN, TileK * Atom.n_batch, Major.K) # Define loadB

                reduceC = TmaTensor(dae, matC_partial,).wgmma("reduce", TileM, TileN, Major.K) # Define reduceC

                gemm = ManualGemm(Atom, MNK=(M, N_chunk, N_local), tmas=(loadA, loadB, reduceC), MNK_base=(0, BaseN, 0), num_k_folds=num_k_folds_second).place(num_sms, base_sm=0) # Define the GEMM

                dae.i(gemm, TerminateC(), TerminateM()) # Add the schedule and termination instructions to the launcher

                build_ns += time.perf_counter_ns() - build_start_ns # Isolate the build timing

                device_ms["gemm2"] += device_timed(lambda: dae_app(dae)) # Launch the GEMM and collect the timing data

            torch.cuda.synchronize(runtime.device) # Ensure this PE's second-GEMM launches are complete before the following distributed barrier.

            # Finish both local second-GEMM phases before communication begins.
            # PE 0's source buffer must be complete before the PUT, and aligning
            # the PEs prevents unfinished GEMM work from entering the WAIT interval.
            nvshmem.barrier()

            # ============================================================
            # Cross-device handoff
            #
            # PE 0 launches a PUT that copies its M x N_out partial to the
            # matching symmetric allocation on PE 1 and sets this iteration's
            # signal. PE 1 launches a WAIT for that signal before reading the
            # received buffer.
            #
            # On PE 1, "comm" measures the CUDA-event interval until WAIT
            # completes. It is not a pure transfer measurement: depending on
            # the relative progress of the PEs, it can include launcher and
            # submission overhead, PE skew, some or all of the 8 MiB transfer,
            # and signal delivery. This timer cannot separate those components.
            # ============================================================

            build_start_ns = time.perf_counter_ns() # Begin timing host-side construction of the communication launcher.

            communication = Launcher(num_sms=1, device=gpu, signal_array=signals) # Build a one-SM launcher with access to the shared NVSHMEM signal array.

            if tp_rank == 0:
                # PE 0 copies its full second-GEMM partial to PE 1's matching
                # symmetric buffer and sets this iteration's completion signal.
                communication.i(NvshmemPut(address=(received_partial_0.data_ptr()), nbytes=(received_partial_0.nbytes), target_pe=1, signal_id=signal_id), TerminateC(), TerminateM())

            else:
                # PE 1 waits for the signal associated with this iteration before
                # reading the received partial during the final add.
                communication.i(NvshmemWait(signal_id=signal_id), TerminateC(), TerminateM())

            build_ns += time.perf_counter_ns() - build_start_ns # Accumulate the host time spent constructing this communication launch.

            device_ms["comm"] += device_timed(lambda: dae_app(communication)) # Launch the PUT or WAIT and accumulate its serialized CUDA-event interval.

            torch.cuda.synchronize(runtime.device) # Ensure this PE's PUT or WAIT launcher has completed before PE 1 reads the received partial during the final add.

            # ============================================================
            # Final partial-output reduction on PE 1
            # ============================================================

            metrics = None

            if tp_rank == 1:
                assert received_partial_0.is_contiguous()
                assert local_partial_1.is_contiguous()
                assert res.is_contiguous()

                device_ms["add"] += device_timed(lambda: torch.add(received_partial_0, local_partial_1, out=res)) # Add the two row-parallel partial outputs and accumulate the CUDA-event interval for the resulting GPU operation.

                torch.cuda.synchronize(runtime.device) # Ensure synchronization

                if measure:
                    pipeline_end_ns = time.perf_counter_ns()

                    metrics = {"wall_ns": pipeline_end_ns - pipeline_start_ns, "build_ns": build_ns, "device_ms": device_ms}

            # Keep the PEs in lockstep between iterations. Without this,
            # PE 0 could race ahead into the next iteration's PUT while
            # PE 1 is still reading received_partial_0 for the final add.
            nvshmem.barrier()

            return metrics

        # ================================================================
        # Warmup iterations (untimed)
        # ================================================================

        if tp_rank == 1:
            print(f"Running {WARMUP_ITERS} warmup iteration(s)...", flush=True)

        for iteration in range(WARMUP_ITERS):
            run_pipeline(signal_id=iteration, measure=False)

        # ================================================================
        # Timed iterations
        # ================================================================

        if tp_rank == 1:
            print(f"Running {PIPELINE_ITERS} measured iteration(s)...", flush=True)

        iteration_metrics = []

        for iteration in range(PIPELINE_ITERS):
            if tp_rank == 1:
                print(f"Starting measured iteration {iteration + 1}/{PIPELINE_ITERS}", flush=True)

            metrics = run_pipeline(signal_id=WARMUP_ITERS + iteration, measure=True)

            if tp_rank == 1: # Print and deal with metrics
                iteration_metrics.append(metrics)

                wall_us = metrics["wall_ns"] / 1_000
                build_us = metrics["build_ns"] / 1_000
                device_us = sum(metrics["device_ms"].values()) * 1_000
                other_us = wall_us - build_us - device_us

                print(
                    f"Pipeline iteration {iteration + 1}: "
                    f"wall {wall_us:.3f} us | "
                    f"build {build_us:.3f} us | "
                    f"device(<=) {device_us:.3f} us | "
                    f"other-host {other_us:.3f} us",
                    flush=True,
                )

        # ================================================================
        # Results and full validation on PE 1
        # ================================================================

        if tp_rank == 1: # Print stats
            wall_us_list = [m["wall_ns"] / 1_000 for m in iteration_metrics]
            build_us_list = [m["build_ns"] / 1_000 for m in iteration_metrics]
            device_us_list = [sum(m["device_ms"].values()) * 1_000 for m in iteration_metrics]
            other_us_list = [
                wall - build - device
                for wall, build, device in zip(wall_us_list, build_us_list, device_us_list)
            ]

            def print_stats(label, values_us): # Fxn to print the timing stats
                print(
                    f"  {label:12s} min {min(values_us):14.3f} us | "
                    f"median {statistics.median(values_us):14.3f} us | "
                    f"mean {statistics.mean(values_us):14.3f} us | "
                    f"max {max(values_us):14.3f} us",
                    flush=True,
                )

            print("\nMulti-GPU TP pipeline latency (PE 1):", flush=True)
            print(f"  Iterations: {PIPELINE_ITERS}", flush=True)
            print_stats("wall", wall_us_list)
            print_stats("build", build_us_list)
            print_stats("device(<=)", device_us_list)
            print_stats("other-host", other_us_list)

            # Per-phase device-time breakdown (median across iterations).
            # "comm" is PE 1's receiver-observed WAIT interval, bundling submission
            # overhead, PE skew, transfer progress, and signal delivery.
            print("\nDevice-time phase breakdown (median across iterations, PE 1):", flush=True)

            for phase in PHASES:
                phase_us = [m["device_ms"][phase] * 1_000 for m in iteration_metrics]
                print(
                    f"  {phase:6s} median {statistics.median(phase_us):12.3f} us "
                    f"(min {min(phase_us):12.3f}, max {max(phase_us):12.3f})",
                    flush=True,
                )

            print(
                "\nNOTE: device(<=) times are CUDA-event intervals around each "
                "measured phase operation; they include submission latency inside "
                "the timed call and are therefore upper bounds on pure device "
                "execution time.",
                flush=True,
            )

            # ============================================================
            # Correctness check outside the timed region
            # (validates the result of the final measured iteration)
            # ============================================================

            matB_rank_0 = seeded_rand(
                (N_local, K),
                seed=MAT_B_SEED_BASE,
                dtype=dtype,
                device=gpu,
            )

            matB_rank_1 = seeded_rand(
                (N_local, K),
                seed=MAT_B_SEED_BASE + 1,
                dtype=dtype,
                device=gpu,
            )

            matB_full = torch.cat((matB_rank_0, matB_rank_1), dim=0)

            matD_rank_0 = seeded_rand((N_out, N_local), seed=MAT_D_SEED_BASE, dtype=dtype, device=gpu)

            matD_rank_1 = seeded_rand((N_out, N_local), seed=MAT_D_SEED_BASE + 1, dtype=dtype, device=gpu)

            matD_full = torch.cat((matD_rank_0, matD_rank_1), dim=1)

            reference_hidden = F.silu(matA @ matB_full.T)
            reference_output = reference_hidden @ matD_full.T

            torch.cuda.synchronize(runtime.device)

            print("\n[PE 1] Full MLP difference:", flush=True)
            tensor_diff("full_mlp", reference_output, res)

        nvshmem.barrier()

    except Exception:
        traceback.print_exc()
        raise

    finally:
        nvshmem.finalize()

if __name__ == "__main__":
    main()