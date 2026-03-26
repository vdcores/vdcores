import torch

from dae.instructions import LoadRegisterM
from dae.launcher import *
from dae.util import dump_insts


gpu = torch.device("cuda")

num_sms = 1
block_bytes = 64
block_elems = block_bytes // 2
blocks_per_repeat = 4

src = torch.arange(
    8 * block_elems,
    dtype=torch.float32,
    device=gpu,
).to(torch.bfloat16)
dst_a = torch.zeros(blocks_per_repeat * block_elems, dtype=torch.bfloat16, device=gpu)
dst_b = torch.zeros(blocks_per_repeat * block_elems, dtype=torch.bfloat16, device=gpu)

dae = Launcher(num_sms, device=gpu)


def program(sm: int):
    del sm
    return [
        Copy(2 * blocks_per_repeat, size=block_bytes),

        # Seed both repeat lanes once, then keep reusing them across repeat regions.
        LoadRegisterM(reg_id=0, value=block_bytes, reg=0, reg_end=2),
        LoadRegisterM(reg_id=1, value=0, reg=0, reg_end=2),

        RepeatM(
            blocks_per_repeat,
            reg=0,
            reg_end=0,
            base_reg=0,
        ),
        TmaLoad1D(src[:block_elems], bytes=block_bytes),
        TmaStore1D(dst_a[:block_elems], bytes=block_bytes).jump(),

        # Keep the source accumulator hot, but reset the store lane so the second
        # repeat writes back to a fresh destination buffer.
        LoadRegisterM(reg_id=1, value=0, reg=1, reg_end=2),
        RepeatM(
            blocks_per_repeat,
            reg=0,
            reg_end=0,
            base_reg=0,
        ),
        TmaLoad1D(src[:block_elems], bytes=block_bytes),
        TmaStore1D(dst_b[:block_elems], bytes=block_bytes).jump(),
    ]


dae.i(
    program,
    TerminateM(),
    TerminateC(),
)

dump_insts(dae, 0)
dae.launch()

expected_a = src[: blocks_per_repeat * block_elems]
expected_b = src[blocks_per_repeat * block_elems : 2 * blocks_per_repeat * block_elems]

torch.cuda.synchronize()
torch.testing.assert_close(dst_a, expected_a)
torch.testing.assert_close(dst_b, expected_b)

print("allocwarp load-register repeat test passed")
