import pytest

from dae.attention_config import select_blackwell_attention_config


@pytest.mark.parametrize(
    ("batch", "seq_len", "expected"),
    [
        (1, 64, (64, 1, 8)),
        (1, 128, (128, 1, 8)),
        (1, 512, (64, 8, 64)),
        (1, 2048, (128, 16, 128)),
        (2, 512, (64, 8, 128)),
        (4, 512, (128, 4, 128)),
        (8, 512, (128, 2, 128)),
    ],
)
def test_measured_blackwell_attention_regimes(batch, seq_len, expected):
    config = select_blackwell_attention_config(batch, seq_len)
    assert (config.kv_tile, config.split_kv, config.num_sms) == expected


def test_attention_selector_rejects_more_base_ctas_than_sms():
    with pytest.raises(ValueError, match="needs 160 SMs"):
        select_blackwell_attention_config(20, 64)
