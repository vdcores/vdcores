import torch
import argparse
import numpy as np

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

def dump_insts(dae, smid : int, mode: str = "interp"):
    dae.build_instructions(mode=mode)

    if mode == "compile_cuda":
        artifacts = dae.last_compile_artifacts
        if artifacts is None:
            raise RuntimeError("compile_cuda dump requested before compile artifacts were generated")
        print(f"[sm={smid}][mode={mode}] Compute Instructions:")
        for i, inst in enumerate(artifacts.emitted_compute[smid]):
            print(f"{i:02}: {inst}")
        print(f"[sm={smid}][mode={mode}] Memory Instructions:")
        for i, inst in enumerate(artifacts.emitted_memory[smid]):
            print(f"{i:02}: {inst}")
        if artifacts.generated_cuda is not None:
            print(f"[cuda-source] {artifacts.generated_cuda.path}")
        return

    sm0 = dae.builder[smid]
    print(f"[sm={smid}][mode={mode}] Compute Instructions:")
    for i, inst in enumerate(sm0.built_cinsts):
        print(f"{i:02}: {inst}")
    print(f"[sm={smid}][mode={mode}] Memory Instructions:")
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
    parser.add_argument("-w", "--write-compute-ops", type=str, nargs="?", const=DEFAULT_COMPUTE_OPS_FILE, default=None,
                        help=f"Write the launcher compute-operator list to a file (default: {DEFAULT_COMPUTE_OPS_FILE})")
    parser.add_argument("--mode", choices=dae.COMPILE_MODES, default="interp",
                        help="Execution/compiler mode")
    parser.add_argument("--emit-cuda", type=str, nargs="?", const="build/generated/dae_compiled_program.cu", default=None,
                        help="Emit split-loop CUDA source for compile_cuda mode without launching")
    
    parsed = parser.parse_args()
    did_work = False
    
    if parsed.instdump is not None:
        dump_insts(dae, parsed.instdump, mode=parsed.mode)
        print()
        did_work = True

    if parsed.write_compute_ops is not None:
        write_compute_operator_file(dae, parsed.write_compute_ops)
        did_work = True

    if parsed.emit_cuda is not None:
        generated = dae.emit_cuda_source(parsed.emit_cuda)
        print(f"[emit-cuda] wrote {generated.path}")
        did_work = True

    executed = False
    if parsed.launch:
        print(f"[launch] VDCores with {dae.num_sms} SMs in mode={parsed.mode}...")
        dae.launch(mode=parsed.mode)
        executed = True
    elif parsed.bench is not None:
        # Prewarm
        # for _ in range(1):
        #     dae.launch()
        torch.cuda.synchronize()

        print(f"[bench] VDCores with {dae.num_sms} SMs in mode={parsed.mode}...")
        dae.bench(parsed.bench, total_bytes=total_bytes, mode=parsed.mode)
        torch.cuda.synchronize()
        executed = True
    elif not did_work:
        print(f"DAE NO EXECUTION MODE.")

    if executed and parsed.profile is not None:
        pp = ProfileParser(dae)
        for prof in parsed.profile:
            pp.parse(prof)
