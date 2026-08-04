import os

import torch

from dae.instructions import Gemv_M64N8, Gemv_M128N8
from dae.launcher import Launcher
from dae.model import GemvLayer


def main() -> None:
    device = torch.device("cuda")
    tile_m = int(os.environ.get("DAE_GEMV_SMOKE_M", "64"))
    atoms = {64: Gemv_M64N8, 128: Gemv_M128N8}
    if tile_m not in atoms:
        raise ValueError("DAE_GEMV_SMOKE_M must be 64 or 128")
    atom = atoms[tile_m]
    tile_m, tile_n, tile_k = atom.MNK
    k = int(os.environ.get("DAE_GEMV_SMOKE_K", str(tile_k * atom.n_batch)))
    if k % (tile_k * atom.n_batch) != 0:
        raise ValueError("DAE_GEMV_SMOKE_K must be divisible by TileK * n_batch")

    generator = torch.Generator(device=device).manual_seed(0)
    matrix = torch.rand((tile_m, k), generator=generator, dtype=torch.bfloat16, device=device) - 0.5
    vector_batch = (
        torch.rand((tile_n, k), generator=generator, dtype=torch.bfloat16, device=device) - 0.5
    )
    output = torch.zeros((tile_n, tile_m), dtype=torch.bfloat16, device=device)

    launcher = Launcher(1, device=device)
    layer = GemvLayer(launcher, atom, "blackwell_smoke", (matrix, vector_batch, output))
    launcher.s(layer.schedule().place(1))
    launcher.launch()

    expected = matrix @ vector_batch.t()
    torch.testing.assert_close(output.t(), expected, rtol=2e-2, atol=5e-2)
    print(
        "blackwell SM100 GEMV smoke passed:",
        f"shape=({tile_m}, {tile_n}, {k})",
        f"max_abs_error={(output.t() - expected).abs().max().item():.6f}",
    )


if __name__ == "__main__":
    main()
