import pytest

from dae.deepseek_v4_inference import reusable_flow_plan


def test_reusable_flow_plan_collapses_256_token_demo_to_four_images():
    plans = reusable_flow_plan(62, 256)

    assert [plan.variant for plan in plans] == [
        "normal",
        "csa_short",
        "hca",
        "csa",
    ]
    assert [(plan.first_position, plan.last_position) for plan in plans] == [
        (62, 317),
        (63, 123),
        (127, 255),
        (131, 315),
    ]


def test_reusable_flow_plan_crosses_hca_and_index_boundaries():
    plans = reusable_flow_plan(2047, 6)

    assert [plan.variant for plan in plans] == [
        "hca",
        "normal",
        "indexed_csa",
        "indexed_normal",
    ]
    assert [(plan.first_position, plan.last_position) for plan in plans] == [
        (2047, 2047),
        (2048, 2050),
        (2051, 2051),
        (2052, 2052),
    ]


@pytest.mark.parametrize(
    ("first_position", "max_new_tokens"),
    [(-1, 1), (0, 0), (0, 257), (65535, 2)],
)
def test_reusable_flow_plan_rejects_invalid_ranges(
    first_position, max_new_tokens
):
    with pytest.raises(ValueError):
        reusable_flow_plan(first_position, max_new_tokens)
