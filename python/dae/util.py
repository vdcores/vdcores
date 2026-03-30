import torch
import argparse
import numpy as np
import time

EMPTY_MEM_TRACE_DTYPE = [
    ("sm_id", np.int32),
    ("start", np.uint64),
    ("end", np.uint64),
    ("size", np.uint32),
    ("opcode", np.uint16),
]

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from .launcher import extract_compute_operator_names
from .runtime import config


DEFAULT_COMPUTE_OPS_FILE = "dae_compute_ops.vdcore.build"
MEM_TRACE_DTYPE = np.dtype(
    {
        "names": ["start", "end", "size", "opcode"],
        "formats": ["<u8", "<u8", "<u4", "<u2"],
        "offsets": [0, 8, 16, 20],
        "itemsize": config.mem_trace_record_size,
    }
)


def tensor_diff(name : str,
                t1 : torch.Tensor, t2 : torch.Tensor, ref : torch.Tensor | None = None, threshold : float = 1.5):
    if ref is None:
        ref = t1
    diff = (t1 - t2).abs().float().mean().item() / ref.abs().float().mean().item() * 100
    print(f"Ave Diff {name}: {diff} %. ")
    # print both mat if diff is large
    if diff > threshold:
        print(f"{name} t1:", t1)
        print(f"{name} t2:", t2)
    # calculate checksum of both to verify if it's layout diff
    checksum1 = (t1.float().sum().item(), t1.shape)
    checksum2 = (t2.float().sum().item(), t2.shape)
    print(f"{name} checksum t1: {checksum1}, t2: {checksum2}")

def dump_insts(dae, smid : int):
    dae.build_instructions()

    sm0 = dae.builder[smid]
    print(f"[sm={smid}] Compute Instructions:")
    for i, inst in enumerate(sm0.built_cinsts):
        print(f"{i:02}: {inst}")
    print(f"[sm={smid}] Memory Instructions:")
    for i, inst in enumerate(sm0.built_minsts):
        print(f"{i:02}: {inst}")


def write_compute_operator_file(dae, path: str = DEFAULT_COMPUTE_OPS_FILE):
    operator_names = extract_compute_operator_names(dae)
    with open(path, "w", encoding="utf-8") as f:
        for name in operator_names:
            f.write(f"{name}\n")
    print(f"[compute-ops] wrote {len(operator_names)} operators to {path}")
    return path

class ProfileParser:
    def __init__(self, dae):
        self.dae = dae
        self.count = 2 # skip the start and end time
        self.profile_data = dae.profile.cpu().numpy()
        self.history = None

        self.profile_data = self.profile_data[64:128,:] # only profile first 128 SMs
        self.opt_raw = False

    def parse(self, prof: str):
        if prof.startswith('@'):
            if prof[1:] == 'raw':
                self.opt_raw = True
            else:
                raise ValueError(f"Unknown profile option: {prof}")
            return

        if prof.startswith("="):
            idx = int(prof[1:])
            self.count = idx
            self.history = None
            return

        # multi-parse
        if ':' in prof:
            name, repeat = prof.split(':')
            repeat = int(repeat)
            for i in range(repeat):
                self.parse(f'{name}r{i}')
            return

        data = self.profile_data[:, self.count]
        if prof.startswith('+'):
            data = data - self.profile_data[:, 0]

        if self.opt_raw:
            print(f"[profile] {prof}: {data}")

        data = np.mean(data) / 1e3
        print_data = data
        if '^' in prof and self.history is not None:
            print_data = data - self.history

        print(f"[profile] {prof}: {print_data:.3f} us")
        
        self.history = data
        self.count += 1


def read_mem_trace(dae):
    if not config.enable_mem_trace or config.max_mem_trace_records == 0:
        return np.recarray(0, dtype=EMPTY_MEM_TRACE_DTYPE)

    counts = dae.mem_trace_count.cpu().numpy().astype(np.int64, copy=False)
    trace = dae.mem_trace.cpu().numpy().view(MEM_TRACE_DTYPE).reshape(
        dae.num_sms, config.max_mem_trace_records
    )

    records = []
    for sm_id, count in enumerate(counts):
        if count <= 0:
            continue
        sm_records = trace[sm_id, :count].copy()
        sm_records = sm_records[sm_records["end"] > sm_records["start"]]
        if sm_records.size == 0:
            continue
        sm_ids = np.full(sm_records.shape[0], sm_id, dtype=np.int32)
        records.append(
            np.rec.fromarrays(
                [sm_ids, sm_records["start"], sm_records["end"], sm_records["size"], sm_records["opcode"]],
                names=["sm_id", "start", "end", "size", "opcode"],
            )
        )

    if not records:
        return np.recarray(0, dtype=EMPTY_MEM_TRACE_DTYPE)

    return np.concatenate(records)


def _build_mem_trace_records(sm_id, start, end, size, opcode):
    return np.rec.fromarrays(
        [sm_id, start, end, size, opcode],
        names=["sm_id", "start", "end", "size", "opcode"],
    )


def load_mem_trace_npz(path: str):
    with np.load(path) as data:
        required = {"sm_id", "start", "end", "size", "opcode"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing required trace arrays in {path}: {sorted(missing)}")
        return _build_mem_trace_records(
            data["sm_id"].astype(np.int32, copy=False),
            data["start"].astype(np.uint64, copy=False),
            data["end"].astype(np.uint64, copy=False),
            data["size"].astype(np.uint32, copy=False),
            data["opcode"].astype(np.uint16, copy=False),
        )


def load_mem_trace_runs_npz(path: str):
    with np.load(path) as data:
        required = {"iteration", "sm_id", "start", "end", "size", "opcode"}
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"Missing required benchmark trace arrays in {path}: {sorted(missing)}")

        iteration = data["iteration"].astype(np.int32, copy=False)
        sm_id = data["sm_id"].astype(np.int32, copy=False)
        start = data["start"].astype(np.uint64, copy=False)
        end = data["end"].astype(np.uint64, copy=False)
        size = data["size"].astype(np.uint32, copy=False)
        opcode = data["opcode"].astype(np.uint16, copy=False)

        if iteration.size == 0:
            num_runs = int(data["num_runs"][0]) if "num_runs" in data.files else 0
            return [np.recarray(0, dtype=EMPTY_MEM_TRACE_DTYPE) for _ in range(num_runs)]

        num_runs = int(max(iteration.max() + 1, int(data["num_runs"][0]) if "num_runs" in data.files else 0))
        runs = []
        for run_idx in range(num_runs):
            mask = iteration == run_idx
            if not np.any(mask):
                runs.append(np.recarray(0, dtype=EMPTY_MEM_TRACE_DTYPE))
                continue
            runs.append(_build_mem_trace_records(sm_id[mask], start[mask], end[mask], size[mask], opcode[mask]))
        return runs


def compute_effective_bw_series(trace_records, bin_us: float = 1.0):
    if trace_records.size == 0:
        return np.array([]), np.array([])

    bin_ns = max(bin_us * 1e3, 1.0)
    start_ns = float(trace_records["start"].min())
    end_ns = float(trace_records["end"].max())
    num_bins = max(1, int(np.ceil((end_ns - start_ns) / bin_ns)))
    bytes_per_bin = np.zeros(num_bins, dtype=np.float64)

    for record in trace_records:
        rec_start = float(record["start"])
        rec_end = float(record["end"])
        if rec_end <= rec_start:
            continue

        total_bytes = float(record["size"])
        left = int((rec_start - start_ns) // bin_ns)
        right = int(np.ceil((rec_end - start_ns) / bin_ns))
        for bin_idx in range(max(0, left), min(num_bins, right)):
            bin_start = start_ns + bin_idx * bin_ns
            bin_end = bin_start + bin_ns
            overlap = max(0.0, min(rec_end, bin_end) - max(rec_start, bin_start))
            if overlap <= 0:
                continue
            bytes_per_bin[bin_idx] += total_bytes * overlap / (rec_end - rec_start)

    bw_gbps = bytes_per_bin / (bin_ns * 1e-9) / 1e9
    times_us = (start_ns + (np.arange(num_bins) + 0.5) * bin_ns - start_ns) / 1e3
    return times_us, bw_gbps


def compute_effective_bw_heatmap(trace_records, num_sms: int, bin_us: float = 1.0):
    if trace_records.size == 0:
        return np.array([]), np.zeros((num_sms, 0), dtype=np.float64)

    bin_ns = max(bin_us * 1e3, 1.0)
    start_ns = float(trace_records["start"].min())
    end_ns = float(trace_records["end"].max())
    num_bins = max(1, int(np.ceil((end_ns - start_ns) / bin_ns)))
    bytes_per_bin = np.zeros((num_sms, num_bins), dtype=np.float64)

    for record in trace_records:
        rec_start = float(record["start"])
        rec_end = float(record["end"])
        if rec_end <= rec_start:
            continue

        sm_id = int(record["sm_id"])
        if sm_id < 0 or sm_id >= num_sms:
            continue

        total_bytes = float(record["size"])
        left = int((rec_start - start_ns) // bin_ns)
        right = int(np.ceil((rec_end - start_ns) / bin_ns))
        for bin_idx in range(max(0, left), min(num_bins, right)):
            bin_start = start_ns + bin_idx * bin_ns
            bin_end = bin_start + bin_ns
            overlap = max(0.0, min(rec_end, bin_end) - max(rec_start, bin_start))
            if overlap <= 0:
                continue
            bytes_per_bin[sm_id, bin_idx] += total_bytes * overlap / (rec_end - rec_start)

    bw_gbps = bytes_per_bin / (bin_ns * 1e-9) / 1e9
    times_us = (start_ns + (np.arange(num_bins) + 0.5) * bin_ns - start_ns) / 1e3
    return times_us, bw_gbps


def average_effective_bw_series(trace_runs, bin_us: float = 1.0):
    if not trace_runs:
        return np.array([]), np.array([])

    bw_runs = []
    max_bins = 0
    for trace in trace_runs:
        _, bw = compute_effective_bw_series(trace, bin_us=bin_us)
        bw_runs.append(bw)
        max_bins = max(max_bins, bw.shape[0])

    if max_bins == 0:
        return np.array([]), np.array([])

    bw_sum = np.zeros(max_bins, dtype=np.float64)
    bw_count = np.zeros(max_bins, dtype=np.int32)
    for bw in bw_runs:
        if bw.size == 0:
            continue
        bw_sum[:bw.shape[0]] += bw
        bw_count[:bw.shape[0]] += 1

    valid = bw_count > 0
    avg_bw = np.zeros(max_bins, dtype=np.float64)
    avg_bw[valid] = bw_sum[valid] / bw_count[valid]
    times_us = (np.arange(max_bins) + 0.5) * bin_us
    return times_us, avg_bw


def average_effective_bw_heatmap(trace_runs, num_sms: int, bin_us: float = 1.0):
    if not trace_runs:
        return np.array([]), np.zeros((num_sms, 0), dtype=np.float64)

    heatmaps = []
    max_bins = 0
    for trace in trace_runs:
        _, heatmap = compute_effective_bw_heatmap(trace, num_sms=num_sms, bin_us=bin_us)
        heatmaps.append(heatmap)
        max_bins = max(max_bins, heatmap.shape[1])

    if max_bins == 0:
        return np.array([]), np.zeros((num_sms, 0), dtype=np.float64)

    bw_sum = np.zeros((num_sms, max_bins), dtype=np.float64)
    bw_count = np.zeros((num_sms, max_bins), dtype=np.int32)
    for heatmap in heatmaps:
        if heatmap.size == 0:
            continue
        bw_sum[:, :heatmap.shape[1]] += heatmap
        bw_count[:, :heatmap.shape[1]] += 1

    valid = bw_count > 0
    avg_heatmap = np.zeros((num_sms, max_bins), dtype=np.float64)
    avg_heatmap[valid] = bw_sum[valid] / bw_count[valid]
    times_us = (np.arange(max_bins) + 0.5) * bin_us
    return times_us, avg_heatmap


def save_effective_bw_plot(dae, path: str | None = None, bin_us: float = 1.0):
    if plt is None:
        raise RuntimeError("matplotlib is required to save the effective bandwidth plot")

    trace_records = read_mem_trace(dae)
    if trace_records.size == 0:
        print("[mem-trace] no memory trace records captured")
        return None

    times_us, bw_gbps = compute_effective_bw_series(trace_records, bin_us=bin_us)
    if path is None:
        path = f"effective_bw_{int(time.time())}.png"

    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
    ax.plot(times_us, bw_gbps, linewidth=1.5)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Effective BW (GB/s)")
    ax.set_title("Effective Memory Bandwidth Over Time")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote effective bandwidth plot to {path}")
    print(f"[mem-trace] average effective bandwidth: {bw_gbps.mean():.3f} GB/s")
    print(f"[mem-trace] peak effective bandwidth: {bw_gbps.max():.3f} GB/s")
    return path


def save_effective_bw_plot_from_trace(trace_records, path: str | None = None, bin_us: float = 1.0, title: str = "Effective Memory Bandwidth Over Time"):
    if plt is None:
        raise RuntimeError("matplotlib is required to save the effective bandwidth plot")
    if trace_records.size == 0:
        print("[mem-trace] no memory trace records captured")
        return None

    times_us, bw_gbps = compute_effective_bw_series(trace_records, bin_us=bin_us)
    if path is None:
        path = f"effective_bw_{int(time.time())}.png"

    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
    ax.plot(times_us, bw_gbps, linewidth=1.5)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Effective BW (GB/s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote effective bandwidth plot to {path}")
    print(f"[mem-trace] average effective bandwidth: {bw_gbps.mean():.3f} GB/s")
    print(f"[mem-trace] peak effective bandwidth: {bw_gbps.max():.3f} GB/s")
    return path


def save_effective_bw_heatmap(dae, path: str | None = None, bin_us: float = 1.0):
    if plt is None:
        raise RuntimeError("matplotlib is required to save the effective bandwidth heatmap")

    trace_records = read_mem_trace(dae)
    if trace_records.size == 0:
        print("[mem-trace] no memory trace records captured")
        return None

    times_us, bw_heatmap = compute_effective_bw_heatmap(trace_records, num_sms=dae.num_sms, bin_us=bin_us)
    if bw_heatmap.size == 0:
        print("[mem-trace] no valid memory trace samples captured")
        return None
    if path is None:
        path = f"effective_bw_heatmap_{int(time.time())}.png"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    time_extent = times_us[-1] + 0.5 * bin_us if times_us.size > 0 else bin_us
    im = ax.imshow(
        bw_heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0.0, time_extent, -0.5, dae.num_sms - 0.5],
        cmap="magma",
    )
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("SM ID")
    ax.set_title("Per-SM Effective Memory Bandwidth Heatmap")
    fig.colorbar(im, ax=ax, label="Effective BW (GB/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote effective bandwidth heatmap to {path}")
    return path


def save_effective_bw_heatmap_from_trace(trace_records, num_sms: int, path: str | None = None, bin_us: float = 1.0, title: str = "Per-SM Effective Memory Bandwidth Heatmap"):
    if plt is None:
        raise RuntimeError("matplotlib is required to save the effective bandwidth heatmap")
    if trace_records.size == 0:
        print("[mem-trace] no memory trace records captured")
        return None

    times_us, bw_heatmap = compute_effective_bw_heatmap(trace_records, num_sms=num_sms, bin_us=bin_us)
    if bw_heatmap.size == 0:
        print("[mem-trace] no valid memory trace samples captured")
        return None
    if path is None:
        path = f"effective_bw_heatmap_{int(time.time())}.png"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    time_extent = times_us[-1] + 0.5 * bin_us if times_us.size > 0 else bin_us
    im = ax.imshow(
        bw_heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0.0, time_extent, -0.5, num_sms - 0.5],
        cmap="magma",
    )
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("SM ID")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Effective BW (GB/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote effective bandwidth heatmap to {path}")
    return path


def save_effective_bw_plot_from_runs(trace_runs, path: str | None = None, bin_us: float = 1.0):
    if plt is None:
        raise RuntimeError("matplotlib is required to save the effective bandwidth plot")
    if not trace_runs:
        print("[mem-trace] no memory trace records captured")
        return None

    times_us, bw_gbps = average_effective_bw_series(trace_runs, bin_us=bin_us)
    if bw_gbps.size == 0:
        print("[mem-trace] no valid memory trace samples captured")
        return None
    if path is None:
        path = f"effective_bw_avg_{int(time.time())}.png"

    fig, ax = plt.subplots(figsize=(12, 4), dpi=200)
    ax.plot(times_us, bw_gbps, linewidth=1.5)
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("Effective BW (GB/s)")
    ax.set_title(f"Average Effective Memory Bandwidth Over Time ({len(trace_runs)} runs)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote averaged effective bandwidth plot to {path}")
    print(f"[mem-trace] average effective bandwidth: {bw_gbps.mean():.3f} GB/s")
    print(f"[mem-trace] peak effective bandwidth: {bw_gbps.max():.3f} GB/s")
    return path


def save_effective_bw_heatmap_from_runs(trace_runs, num_sms: int, path: str | None = None, bin_us: float = 1.0):
    if plt is None:
        raise RuntimeError("matplotlib is required to save the effective bandwidth heatmap")
    if not trace_runs:
        print("[mem-trace] no memory trace records captured")
        return None

    times_us, bw_heatmap = average_effective_bw_heatmap(trace_runs, num_sms=num_sms, bin_us=bin_us)
    if bw_heatmap.size == 0:
        print("[mem-trace] no valid memory trace samples captured")
        return None
    if path is None:
        path = f"effective_bw_heatmap_avg_{int(time.time())}.png"

    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    time_extent = times_us[-1] + 0.5 * bin_us if times_us.size > 0 else bin_us
    im = ax.imshow(
        bw_heatmap,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[0.0, time_extent, -0.5, num_sms - 0.5],
        cmap="magma",
    )
    ax.set_xlabel("Time (us)")
    ax.set_ylabel("SM ID")
    ax.set_title(f"Average Per-SM Effective Memory Bandwidth Heatmap ({len(trace_runs)} runs)")
    fig.colorbar(im, ax=ax, label="Effective BW (GB/s)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[mem-trace] wrote averaged effective bandwidth heatmap to {path}")
    return path


def save_mem_trace(dae, path: str | None = None):
    trace_records = read_mem_trace(dae)
    if trace_records.size == 0:
        print("[mem-trace] no memory trace records captured")
        return None

    if path is None:
        path = f"mem_trace_{int(time.time())}.npz"

    np.savez(
        path,
        sm_id=trace_records["sm_id"],
        start=trace_records["start"],
        end=trace_records["end"],
        size=trace_records["size"],
        opcode=trace_records["opcode"],
    )
    print(f"[mem-trace] wrote raw trace to {path}")
    return path


def save_mem_trace_runs(trace_runs, path: str | None = None):
    nonempty_runs = [trace for trace in trace_runs if trace.size > 0]
    if not nonempty_runs:
        print("[mem-trace] no memory trace records captured")
        return None

    if path is None:
        path = f"mem_trace_runs_{int(time.time())}.npz"

    iteration = np.concatenate([
        np.full(trace.shape[0], idx, dtype=np.int32)
        for idx, trace in enumerate(trace_runs)
        if trace.size > 0
    ])
    concatenated = np.concatenate(nonempty_runs)
    np.savez(
        path,
        iteration=iteration,
        sm_id=concatenated["sm_id"],
        start=concatenated["start"],
        end=concatenated["end"],
        size=concatenated["size"],
        opcode=concatenated["opcode"],
        num_runs=np.array([len(trace_runs)], dtype=np.int32),
    )
    print(f"[mem-trace] wrote {len(trace_runs)} benchmark traces to {path}")
    return path

def dae_app(dae, total_bytes = None):
    parser = argparse.ArgumentParser(description="VDCores frontend")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-b", "--bench", type=int, nargs="?", const=1, default=None,
                        help="Run benchmark N times (default: 1)")
    group.add_argument("-l", "--launch", action="store_true",
                        help="Launch configuration")
    parser.add_argument("-i", "--instdump", type=int, nargs="?", const=0, default=None,
                        help="Dump instructions for SM ID (default: 0)")
    parser.add_argument("-p", "--profile", type=str, nargs="+", default=None,
                        help="Profile with VDCores profiling counters")
    parser.add_argument("--mem-bw-plot", type=str, nargs="?", const="", default=None,
                        help="Save a memory-trace effective bandwidth plot (default filename if omitted)")
    parser.add_argument("--mem-bw-heatmap", type=str, nargs="?", const="", default=None,
                        help="Save a per-SM memory-trace effective bandwidth heatmap (default filename if omitted)")
    parser.add_argument("--mem-bw-bin-us", type=float, default=1.0,
                        help="Time bin width in microseconds for memory-trace bandwidth plots")
    parser.add_argument("--mem-trace-save", type=str, nargs="?", const="", default=None,
                        help="Save raw memory-trace data as .npz (default filename if omitted)")
    parser.add_argument("-w", "--write-compute-ops", type=str, nargs="?", const=DEFAULT_COMPUTE_OPS_FILE, default=None,
                        help=f"Write the launcher compute-operator list to a file (default: {DEFAULT_COMPUTE_OPS_FILE})")
    
    parsed = parser.parse_args()
    did_work = False
    
    if parsed.instdump is not None:
        dump_insts(dae, parsed.instdump)
        print()
        did_work = True

    if parsed.write_compute_ops is not None:
        write_compute_operator_file(dae, parsed.write_compute_ops)
        did_work = True

    executed = False
    trace_runs = None
    if parsed.launch:
        print(f"[launch] VDCores with {dae.num_sms} SMs...")
        dae.launch()
        executed = True
    elif parsed.bench is not None:
        # Prewarm
        # for _ in range(1):
        #     dae.launch()
        torch.cuda.synchronize()

        print(f"[bench] VDCores with {dae.num_sms} SMs...")
        collect_traces = (
            parsed.mem_trace_save is not None
            or parsed.mem_bw_plot is not None
            or parsed.mem_bw_heatmap is not None
        )
        bench_result = dae.bench(
            parsed.bench,
            total_bytes=total_bytes,
            trace_collector=read_mem_trace if collect_traces else None,
        )
        trace_runs = bench_result["trace_runs"]
        torch.cuda.synchronize()
        executed = True
    elif not did_work:
        print(f"DAE NO EXECUTION MODE.")

    if executed and parsed.profile is not None:
        pp = ProfileParser(dae)
        for prof in parsed.profile:
            pp.parse(prof)

    if executed and parsed.mem_trace_save is not None:
        output_path = parsed.mem_trace_save or None
        if parsed.bench is not None:
            save_mem_trace_runs(trace_runs or [], path=output_path)
        else:
            save_mem_trace(dae, path=output_path)

    if executed and parsed.mem_bw_plot is not None:
        output_path = parsed.mem_bw_plot or None
        if parsed.bench is not None:
            save_effective_bw_plot_from_runs(trace_runs or [], path=output_path, bin_us=parsed.mem_bw_bin_us)
        else:
            save_effective_bw_plot(dae, path=output_path, bin_us=parsed.mem_bw_bin_us)

    if executed and parsed.mem_bw_heatmap is not None:
        output_path = parsed.mem_bw_heatmap or None
        if parsed.bench is not None:
            save_effective_bw_heatmap_from_runs(
                trace_runs or [],
                num_sms=dae.num_sms,
                path=output_path,
                bin_us=parsed.mem_bw_bin_us,
            )
        else:
            save_effective_bw_heatmap(dae, path=output_path, bin_us=parsed.mem_bw_bin_us)
