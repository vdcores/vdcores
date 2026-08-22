import pytest
import torch

from dae.deepseek_v4_mxfp_checkpoint import (
    pack_mxfp4_data,
    pack_mxfp4_scales,
)


@pytest.mark.parametrize("tile_k", (128, 256, 512))
def test_pack_mxfp4_data_round_trip(tile_k):
    rows, k = 256, 1024
    packed = torch.arange(rows * (k // 2), dtype=torch.int64).to(
        torch.uint8
    ).reshape(rows, k // 2)
    native = pack_mxfp4_data(packed, tile_k=tile_k)
    recovered = native.permute(0, 3, 1, 2, 4).reshape_as(packed)
    torch.testing.assert_close(recovered, packed, rtol=0, atol=0)


@pytest.mark.parametrize("tile_k", (128, 256, 512))
def test_pack_mxfp4_scales_preserves_native_k128_order(tile_k):
    rows, k = 256, 1024
    native_128x4 = torch.arange(
        rows * (k // 32), dtype=torch.int64
    ).to(torch.uint8)
    grouped = pack_mxfp4_scales(
        native_128x4,
        rows=rows,
        k=k,
        tile_k=tile_k,
    )
    torch.testing.assert_close(
        grouped.reshape(-1), native_128x4, rtol=0, atol=0
    )
