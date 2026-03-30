import torch
import argparse
import numpy as np
import time

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
        return np.rec.array([], dtype=[("sm_id", np.int32), ("start", np.uint64), ("end", np.uint64), ("size", np.uint32), ("opcode", np.uint16)])

    return np.concatenate(records)


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
    parser.add_argument("--mem-bw-bin-us", type=float, default=1.0,
                        help="Time bin width in microseconds for memory-trace bandwidth plots")
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
        dae.bench(parsed.bench, total_bytes=total_bytes)
        torch.cuda.synchronize()
        executed = True
    elif not did_work:
        print(f"DAE NO EXECUTION MODE.")

    if executed and parsed.profile is not None:
        pp = ProfileParser(dae)
        for prof in parsed.profile:
            pp.parse(prof)

    if executed and parsed.mem_bw_plot is not None:
        output_path = parsed.mem_bw_plot or None
        save_effective_bw_plot(dae, path=output_path, bin_us=parsed.mem_bw_bin_us)
