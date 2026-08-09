import pytest

from dae.attention_config import select_blackwell_attention_config


@pytest.mark.parametrize(
    ("batch", "seq_len", "expected"),
    [
        (1, 64, (128, 1, 8)),
        (1, 128, (128, 1, 8)),
        (1, 129, (128, 2, 16)),
        (1, 512, (128, 4, 32)),
        (1, 2048, (128, 16, 128)),
        (2, 512, (128, 4, 64)),
        (2, 2048, (128, 8, 128)),
        (4, 512, (128, 4, 128)),
        (4, 2048, (128, 4, 128)),
        (8, 512, (128, 2, 128)),
        (8, 2048, (128, 2, 128)),
        # Eighteen blocks cannot use the 16-way cap equally, so select the
        # largest legal divisor instead of silently dropping a block.
        (1, 2304, (128, 9, 72)),
    ],
)
def test_measured_blackwell_attention_regimes(batch, seq_len, expected):
    config = select_blackwell_attention_config(batch, seq_len)
    assert (config.kv_tile, config.split_kv, config.num_sms) == expected


def test_attention_selector_rejects_more_base_ctas_than_sms():
    with pytest.raises(ValueError, match="needs 160 SMs"):
        select_blackwell_attention_config(20, 64)


def test_attention_selector_rejects_unencodable_split_count():
    with pytest.raises(ValueError, match=r"\[1, 16\]"):
        select_blackwell_attention_config(1, 512, max_split=17)
