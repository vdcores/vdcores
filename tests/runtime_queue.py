"""Exercise queue wraparound, slot spans, retained slots and global dependencies.

Run after make pyext with OP_COPY and OP_TERMINATEC selected:
    na run python tests/runtime_queue.py
"""

import torch

from dae.instructions import Copy, RegLoad, RegStore, RepeatM, TmaLoad1D, TmaStore1D
from dae.launcher import Launcher


def check_copy(sms, copies, port, retained, tile_bytes=8192):
    source = torch.randint(0, 2**30, (sms, copies, tile_bytes // 4),
                           dtype=torch.int32, device="cuda")
    output = torch.empty_like(source)
    launcher = Launcher(sms)
    completion = launcher.new_bar(sms * copies)

    def instructions(sm):
        steps = [(TmaLoad1D(source[sm, 0]).port(port), tile_bytes)]
        if retained:
            steps += [(RegStore(0, size=tile_bytes).port(port), 0),
                      (RegLoad(0).port(port), 0)]
        steps += [(TmaStore1D(output[sm, 0]).port(1 - port).bar(completion), tile_bytes)]
        return [Copy(copies * (2 if retained else 1), tile_bytes),
                RepeatM.on(copies, *steps)]

    launcher.i(instructions)
    launcher.s()
    for _ in range(3):
        source.random_(0, 2**30)
        output.fill_(-1)
        launcher.launch()
        torch.testing.assert_close(output, source, rtol=0, atol=0)
        assert launcher.bars.view(torch.uint32)[completion].item() == 0


def check_barrier_updates():
    launcher = Launcher(1)
    marker = launcher.new_bar(3)
    launcher.s()
    for value in (3, 3, 5):
        launcher.set_bar(marker, value)
        launcher.launch()
        assert launcher.bars.view(torch.uint32)[marker].item() == value
        assert launcher.bars_src.view(torch.uint32)[marker].item() == value
        launcher.bars.view(torch.uint32)[marker] = 0
        launcher.bars_src.view(torch.uint32)[marker] = 0
    # Public count updates and new barriers must invalidate the reset image.
    launcher.bar_values[marker] = 7
    added = launcher.new_bar(11)
    launcher.launch()
    counts = launcher.bars.view(torch.uint32).cpu()
    assert counts[marker].item() == 7 and counts[added].item() == 11
    launcher.new_bar(None)
    try:
        launcher.launch()
    except ValueError as error:
        assert "unbound barrier" in str(error)
    else:
        raise AssertionError("Unbound barrier must be rejected before launch")
    print("PASS: repeated barrier reset, count updates and unbound validation", flush=True)


def check_dependency_chain(sms):
    """A store must publish its dependency before waiting for the next token."""
    tile_bytes = 8192
    source = torch.empty((sms, tile_bytes // 4), dtype=torch.int32, device="cuda")
    intermediate = torch.empty_like(source)
    output = torch.empty_like(source)
    launcher = Launcher(sms)
    ready = launcher.new_bar(sms)
    done = launcher.new_bar(sms)

    def instructions(sm):
        return [Copy(2, tile_bytes),
                TmaLoad1D(source[sm]).port(0),
                TmaStore1D(intermediate[sm]).port(1).bar(ready),
                TmaLoad1D(intermediate[(sm + 1) % sms]).port(0).bar(ready),
                TmaStore1D(output[sm]).port(1).bar(done)]

    launcher.i(instructions)
    launcher.s()
    for _ in range(3):
        source.random_(0, 2**30)
        intermediate.fill_(-1)
        output.fill_(-1)
        launcher.launch()
        torch.testing.assert_close(output, source.roll(-1, 0), rtol=0, atol=0)
        counts = launcher.bars.view(torch.uint32).cpu()
        assert counts[ready].item() == 0 and counts[done].item() == 0
    print(f"PASS: {sms} SMs, store-to-load dependency chain", flush=True)


if __name__ == "__main__":
    for sms in (1, torch.cuda.get_device_properties(0).multi_processor_count):
        for copies in (1, 31, 32, 33, 65, 129):
            for port in (0, 1):
                for retained in (False, True):
                    check_copy(sms, copies, port, retained)
        print(f"PASS: {sms} SMs, both ports, ring wraparound and retained slots", flush=True)
        for slots in (2, 3, 4, 5, 7):
            for port in (0, 1):
                check_copy(sms, 33, port, False, tile_bytes=slots * 8192)
        print(f"PASS: {sms} SMs, contiguous allocations of 2, 3, 4, 5 and 7 slots", flush=True)
        check_dependency_chain(sms)
    check_barrier_updates()
