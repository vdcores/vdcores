import statistics
import time

import torch
import torch.nn.functional as F

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


def device_timed(fxn):
    # Measure the device-side duration of fxn's stream work using CUDA
    # events. NOTE: the interval also contains the host-side latency of
    # submitting the work to the stream (everything between the two
    # record() calls), so treat this as an UPPER BOUND on true device
    # execution time. Even so, it excludes the dominant host cost we are
    # trying to isolate: launcher/schedule construction.
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    fxn()
    end_evt.record()
    end_evt.synchronize()

    return start_evt.elapsed_time(end_evt)  # milliseconds (float)


# ============================================================================
# Configuration (kept identical to the multi-GPU script)
# ============================================================================

WARMUP_ITERS = 2 # Number of warmup iterations
PIPELINE_ITERS = 10 # Number of pipeline iterations

# Device-time phases reported per iteration. (No "comm" phase on a
# single device; kept as separate names so output lines up with the
# multi-GPU script's breakdown.)
PHASES = ("gemm1", "silu", "gemm2", "add")

gpu = torch.device("cuda")

torch.manual_seed(0) # Sets PyTorch’s global random seed so torch.rand(...) generates the same matrices on repeated runs.

dtype = torch.bfloat16 # Use BF16 for the input matrices, weights, intermediate activations, partial outputs, and final result.

Atom = Gemm_M64N64K64 # Select the GEMM wanted
TileM, TileN, TileK = Atom.MNK # Obtain tile sizes

# Full MLP dimensions.
M = TileM * 128
N = TileN * 8
K = TileK * 16
N_out = TileN * 8

num_k_folds = K // TileK # Number of k folds
num_sms = 128 # Num SMs

tp_size = 2 # Simulates two tensor-parallel ranks sequentially on the same GPU.
n_chunk_tiles = 4 # Num chunk tiles

assert K % TileK == 0 # Assert dimensions are correct
assert M % TileM == 0
assert N % TileN == 0
assert N % TileK == 0
assert N_out % TileN == 0


# ============================================================================
# Tensor-parallel partitioning
# ============================================================================

# Simulate two tensor-parallel ranks sequentially on one GPU.

# GEMM 1 is column-parallel: divide the N output tiles into one
# contiguous hidden-output shard per virtual rank.
n_tiles = N // TileN

assert n_tiles % tp_size == 0
n_tiles_per_rank = (n_tiles // tp_size)
num_k_folds_second = (N // TileK)

# GEMM 2 is row-parallel: divide the hidden reduction dimension N
# into TileK-aligned shards. Each virtual rank reduces across only
# K_shard hidden elements and produces a full M x N_out partial.

assert (num_k_folds_second % tp_size == 0)
k_folds_per_rank = (num_k_folds_second // tp_size) # Num k folds per rank
K_shard = (k_folds_per_rank * TileK) # Size of k-shards

# Number of output-column tiles in each full second-GEMM partial.

n_tiles_second = (N_out // TileN)


# ============================================================================
# Tensor allocation (allocations occur before timing)
# ============================================================================

matA = (torch.rand(M, K, dtype=dtype, device=gpu,) - 0.5) # create random matrices containing both positive and negative values

matB = (torch.rand(N, K, dtype=dtype, device=gpu) - 0.5)

matC = torch.zeros(M, N, dtype=dtype, device=gpu)

matD = (torch.rand(N_out, N, dtype=dtype, device=gpu) - 0.5)

# Allocate one full M x N_out second-GEMM partial for each virtual rank.
# Adding the two partials reconstructs the complete row-parallel output.

matC_partials = [torch.zeros(M,N_out,dtype=dtype,device=gpu) for tp in range(tp_size)] # Python list of zero-filled output matrices, one for each tensor-parallel rank

res = torch.zeros(M, N_out, dtype=dtype, device=gpu)


def run_pipeline(measure: bool):
    # Reset output tensors before timing.
    matC.zero_()

    for partial in matC_partials:
        partial.zero_()

    res.zero_()

    torch.cuda.synchronize()

    # Approach-2 instrumentation accumulators. build_ns counts host-side
    # launcher/TmaTensor/schedule construction; device_ms counts
    # CUDA-event-timed stream work per phase.
    build_ns = 0
    device_ms = {phase: 0.0 for phase in PHASES}

    if measure:
        pipeline_start_ns = (
            time.perf_counter_ns()
        )

    # ========================================================================
    # First GEMM
    #
    # The two virtual ranks execute sequentially on one GPU.
    #
    # matC = matA @ matB.t()
    # ========================================================================

    # Execute the virtual ranks sequentially. Each rank computes a
    # contiguous range of first-GEMM output columns in the shared matC.
    for tp_rank in range(tp_size):
        # Tile range owned by this virtual rank in the hidden/output dimension.
        rank_n_start_tile = (tp_rank * n_tiles_per_rank)

        rank_n_end_tile = (rank_n_start_tile + n_tiles_per_rank)

        for n_base_tile in range(rank_n_start_tile, rank_n_end_tile, n_chunk_tiles):
            cur_n_tiles = min(n_chunk_tiles, (rank_n_end_tile - n_base_tile))
            N_chunk = (cur_n_tiles * TileN)
            BaseN = (n_base_tile * TileN)

            build_start_ns = time.perf_counter_ns()

            dae = Launcher(num_sms, device=gpu)

            loadA = TmaTensor(dae, matA).wgmma_load(TileM, TileK, Major.K)
            loadB = TmaTensor(dae, matB).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
            reduceC = TmaTensor(dae, matC).wgmma("reduce", TileM, TileN, Major.K)

            gemm = ManualGemm(Atom, MNK=(M, N_chunk, K), tmas=(loadA, loadB, reduceC), MNK_base=(0, BaseN, 0), num_k_folds=num_k_folds).place(num_sms, base_sm=0)

            dae.i(gemm, TerminateC(), TerminateM())

            build_ns += time.perf_counter_ns() - build_start_ns

            device_ms["gemm1"] += device_timed(lambda: dae_app(dae))

    torch.cuda.synchronize()

    # ========================================================================
    # SiLU activation
    # ========================================================================

    silu_result = {}

    device_ms["silu"] += device_timed(
        lambda: silu_result.__setitem__("t", F.silu(matC).contiguous())
    )

    matSilu = silu_result["t"]

    torch.cuda.synchronize()

    # ========================================================================
    # Second GEMM
    #
    # The two virtual ranks execute sequentially on one GPU.
    # Each rank produces one complete M x N_out partial.
    # ========================================================================

    # Split GEMM 2's reduction dimension into contiguous hidden shards.
    # BaseK selects where this virtual rank begins reading in both matSilu
    # and matD, while K_shard limits computation to that shard. Each rank
    # writes a separate full M x N_out partial.
    for tp_rank in range(tp_size):
        rank_k_start_fold = tp_rank * k_folds_per_rank
        BaseK = rank_k_start_fold * TileK
        matC_partial = matC_partials[tp_rank]

        for n_base_tile in range(0, n_tiles_second, n_chunk_tiles):
            cur_n_tiles = min(n_chunk_tiles, n_tiles_second - n_base_tile)
            N_chunk = cur_n_tiles * TileN
            BaseN = n_base_tile * TileN

            build_start_ns = time.perf_counter_ns()

            dae = Launcher(num_sms, device=gpu)

            loadA = TmaTensor(dae, matSilu).wgmma_load(TileM, TileK, Major.K)
            loadB = TmaTensor(dae, matD).wgmma_load(TileN, TileK * Atom.n_batch, Major.K)
            reduceC = TmaTensor(dae, matC_partial).wgmma("reduce", TileM, TileN, Major.K)

            # Schedule this virtual rank's M x N_chunk partial. BaseK offsets both
            # operands into the rank's hidden shard, while K_shard and
            # k_folds_per_rank prevent reduction across the other rank's shard.
            gemm = ManualGemm(Atom, MNK=(M, N_chunk, K_shard), tmas=(loadA, loadB, reduceC), MNK_base=(0, BaseN, BaseK), num_k_folds=k_folds_per_rank).place(num_sms, base_sm=0)

            dae.i(gemm, TerminateC(), TerminateM())

            build_ns += time.perf_counter_ns() - build_start_ns

            device_ms["gemm2"] += device_timed(lambda: dae_app(dae))

    torch.cuda.synchronize()

    # ========================================================================
    # Final partial-output reduction
    #
    # This matches the torch.add reduction currently used by the
    # multi-GPU implementation.
    # ========================================================================

    device_ms["add"] += device_timed(
        lambda: torch.add(matC_partials[0], matC_partials[1], out=res)
    )

    torch.cuda.synchronize()

    if not measure:
        return None

    pipeline_end_ns = time.perf_counter_ns()

    return {
        "wall_ns": pipeline_end_ns - pipeline_start_ns,
        "build_ns": build_ns,
        "device_ms": device_ms,
    }


# ============================================================================
# Warmup iterations (untimed)
# ============================================================================

print(f"Running {WARMUP_ITERS} warmup iteration(s)...", flush=True)

for iteration in range(WARMUP_ITERS):
    run_pipeline(measure=False)


# ============================================================================
# Timed iterations
# ============================================================================

print(f"Running {PIPELINE_ITERS} measured iteration(s)...", flush=True)

iteration_metrics = []

for iteration in range(PIPELINE_ITERS):
    print(f"Starting measured iteration {iteration + 1}/{PIPELINE_ITERS}", flush=True)

    metrics = run_pipeline(measure=True)

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


# ============================================================================
# Results
# ============================================================================

wall_us_list = [m["wall_ns"] / 1_000 for m in iteration_metrics]
build_us_list = [m["build_ns"] / 1_000 for m in iteration_metrics]
device_us_list = [sum(m["device_ms"].values()) * 1_000 for m in iteration_metrics]
other_us_list = [
    wall - build - device
    for wall, build, device in zip(wall_us_list, build_us_list, device_us_list)
]


def print_stats(label, values_us):
    print(
        f"  {label:12s} min {min(values_us):14.3f} us | "
        f"median {statistics.median(values_us):14.3f} us | "
        f"mean {statistics.mean(values_us):14.3f} us | "
        f"max {max(values_us):14.3f} us",
        flush=True,
    )


print("\nSingle-GPU sequential TP pipeline latency:", flush=True)
print(f"  Iterations: {PIPELINE_ITERS}", flush=True)
print_stats("wall", wall_us_list)
print_stats("build", build_us_list)
print_stats("device(<=)", device_us_list)
print_stats("other-host", other_us_list)

# Per-phase device-time breakdown (median across iterations).
print("\nDevice-time phase breakdown (median across iterations):", flush=True)

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

# ============================================================================
# Correctness check
# (validates the result of the final measured iteration)
# ============================================================================

reference_hidden = F.silu(matA @ matB.T)
reference_output = reference_hidden @ matD.T

torch.cuda.synchronize()

print("Full MLP difference:")
tensor_diff("full_mlp", reference_output, res)