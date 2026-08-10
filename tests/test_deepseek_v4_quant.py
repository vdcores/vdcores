import torch

from dae.deepseek_v4_quant import (
    dequantize_fp8_block128,
    dequantize_nvfp4,
    quantize_fp8_block128,
    quantize_nvfp4,
)


def test_nvfp4_checkpoint_contract_round_trip():
    codebook = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
    )
    source = torch.stack((codebook, codebook.flip(0)))

    packed, block_scale, scale2 = quantize_nvfp4(source, 1.0)

    assert packed.dtype == torch.uint8
    assert packed.shape == (2, 8)
    assert block_scale.dtype == torch.float8_e4m3fn
    assert block_scale.shape == (2, 1)
    torch.testing.assert_close(
        dequantize_nvfp4(packed, block_scale, scale2), source
    )


def test_fp8_block128_checkpoint_contract_round_trip():
    source = torch.zeros((128, 128), dtype=torch.float32)
    source[0, 0] = 448.0
    source[1, 1] = -448.0
    source[2, 2] = 1.0

    quantized, scale = quantize_fp8_block128(source)

    assert quantized.dtype == torch.float8_e4m3fn
    assert quantized.shape == source.shape
    assert scale.dtype == torch.float8_e8m0fnu
    assert scale.shape == (1, 1)
    torch.testing.assert_close(
        dequantize_fp8_block128(quantized, scale), source
    )
