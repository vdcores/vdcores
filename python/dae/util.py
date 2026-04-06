import torch
import argparse
import numpy as np

from .compiled_mode import DEFAULT_COMPILED_SPEC_FILE
from .launcher import extract_compute_operator_names


DEFAULT_COMPUTE_OPS_FILE = "dae_compute_ops.vdcore.build"


def tensor_diff(name : str,
                t1 : torch.Tensor, t2 : torch.Tensor, ref : torch.Tensor | None = None):
    if ref is None:
        ref = t1
    diff = (t1 - t2).abs().float().mean().item() / ref.abs().float().mean().item() * 100
    print(f"Ave Diff {name}: {diff} %. ")
    # print both mat if diff is large
    if diff > 1.5:
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


def write_compiled_spec_file(dae, path: str = DEFAULT_COMPILED_SPEC_FILE):
    written = dae.write_compiled_spec(path)
    print(f"[compiled-spec] wrote {written}")
    return written


def _format_ns_us(value_ns: float) -> str:
    return f"{value_ns / 1e3:.3f} us"


def _print_sm_time_group(name: str, sm_ids: np.ndarray, starts: np.ndarray, ends: np.ndarray, base_start: int):
    if len(sm_ids) == 0:
        return

    group_starts = starts[sm_ids].astype(np.float64)
    group_ends = ends[sm_ids].astype(np.float64)
    durations = group_ends - group_starts
    start_offsets = group_starts - float(base_start)
    end_offsets = group_ends - float(base_start)

    print(f"[sm-times] {name}: {len(sm_ids)} sms")
    print(
        "[sm-times]   duration: "
        f"mean={_format_ns_us(durations.mean())} "
        f"median={_format_ns_us(np.median(durations))} "
        f"min={_format_ns_us(durations.min())} "
        f"max={_format_ns_us(durations.max())}"
    )
    print(
        "[sm-times]   start-offset: "
        f"mean={_format_ns_us(start_offsets.mean())} "
        f"median={_format_ns_us(np.median(start_offsets))} "
        f"min={_format_ns_us(start_offsets.min())} "
        f"max={_format_ns_us(start_offsets.max())}"
    )
    print(
        "[sm-times]   end-offset: "
        f"mean={_format_ns_us(end_offsets.mean())} "
        f"median={_format_ns_us(np.median(end_offsets))} "
        f"min={_format_ns_us(end_offsets.min())} "
        f"max={_format_ns_us(end_offsets.max())}"
    )


def print_sm_time_summary(dae, limit: int = 12):
    profile_data = dae.profile.detach().cpu().numpy()
    starts = profile_data[:, 0]
    ends = profile_data[:, 1]
    valid_sms = np.nonzero((starts != 0) & (ends != 0))[0]
    if len(valid_sms) == 0:
        print("[sm-times] no valid per-SM start/end timestamps were recorded")
        return

    base_start = int(starts[valid_sms].min())
    print("[sm-times] per-SM start/end summary")
    _print_sm_time_group("all", valid_sms, starts, ends, base_start)

    sm_program_ids = None
    try:
        compiled_spec = dae.compiled_spec()
        raw_program_ids = compiled_spec.get("sm_program_ids")
        if raw_program_ids is not None and len(raw_program_ids) == dae.num_sms:
            sm_program_ids = np.asarray(raw_program_ids, dtype=np.int32)
    except Exception:
        sm_program_ids = None

    if sm_program_ids is not None:
        for program_id in sorted(np.unique(sm_program_ids)):
            group_sms = valid_sms[sm_program_ids[valid_sms] == program_id]
            _print_sm_time_group(f"program {int(program_id)}", group_sms, starts, ends, base_start)

    durations = (ends[valid_sms] - starts[valid_sms]).astype(np.float64)
    start_offsets = (starts[valid_sms] - base_start).astype(np.float64)
    slow_order = np.argsort(durations)[::-1]
    head_count = min(limit, len(valid_sms))
    print(f"[sm-times] slowest {head_count} sms by duration")
    for idx in slow_order[:head_count]:
        sm_id = int(valid_sms[idx])
        program_id = int(sm_program_ids[sm_id]) if sm_program_ids is not None else -1
        label = f"sm={sm_id}"
        if program_id >= 0:
            label += f" program={program_id}"
        print(
            f"[sm-times]   {label}: "
            f"start={_format_ns_us(start_offsets[idx])} "
            f"duration={_format_ns_us(durations[idx])} "
            f"end={_format_ns_us(start_offsets[idx] + durations[idx])}"
        )

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
    parser.add_argument("--profile-sm-times", action="store_true",
                        help="Print per-SM start/end timing summary after execution")
    parser.add_argument("--profile-sm-times-limit", type=int, default=12,
                        help="Number of slowest SMs to print for --profile-sm-times")
    parser.add_argument("-w", "--write-compute-ops", type=str, nargs="?", const=DEFAULT_COMPUTE_OPS_FILE, default=None,
                        help=f"Write the launcher compute-operator list to a file (default: {DEFAULT_COMPUTE_OPS_FILE})")
    parser.add_argument("--write-compiled-spec", type=str, nargs="?", const=DEFAULT_COMPILED_SPEC_FILE, default=None,
                        help=f"Write the launcher compiled-program spec to a file (default: {DEFAULT_COMPILED_SPEC_FILE})")
    parser.add_argument("--mode", choices=["interpreted", "compiled"], default="interpreted",
                        help="Select launcher execution mode")
    
    parsed = parser.parse_args()
    did_work = False
    
    if parsed.instdump is not None:
        dump_insts(dae, parsed.instdump)
        print()
        did_work = True

    if parsed.write_compute_ops is not None:
        write_compute_operator_file(dae, parsed.write_compute_ops)
        did_work = True

    if parsed.write_compiled_spec is not None:
        write_compiled_spec_file(dae, parsed.write_compiled_spec)
        did_work = True

    dae.set_mode(parsed.mode)
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
    if executed and parsed.profile_sm_times:
        print_sm_time_summary(dae, limit=max(parsed.profile_sm_times_limit, 1))

    return executed
