import pytest

from dae.deepseek_v4_inference import (
    device_token_span_plan,
    reusable_flow_plan,
)


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


def test_device_token_spans_batch_only_full_window_normal_runs():
    spans = device_token_span_plan(126, 10, max_span_tokens=3)

    assert [span.token_count for span in spans] == [1, 1, 3, 1, 3, 1]
    assert [span.first_position for span in spans] == [
        126,
        127,
        128,
        131,
        132,
        135,
    ]
    assert [span.variant for span in spans] == [
        "normal",
        "hca",
        "normal",
        "csa",
        "normal",
        "csa",
    ]


def test_device_token_spans_do_not_cross_index_selection_boundary():
    spans = device_token_span_plan(2048, 7, max_span_tokens=8)

    assert [
        (span.first_position, span.token_count, span.variant)
        for span in spans
    ] == [
        (2048, 3, "normal"),
        (2051, 1, "indexed_csa"),
        (2052, 3, "indexed_normal"),
    ]


def test_device_token_span_default_preserves_per_token_launches():
    spans = device_token_span_plan(128, 6)
    assert len(spans) == 6
    assert all(span.token_count == 1 for span in spans)


@pytest.mark.parametrize("max_span_tokens", [0, 257])
def test_device_token_span_rejects_invalid_launch_width(max_span_tokens):
    with pytest.raises(ValueError, match="max_span_tokens"):
        device_token_span_plan(
            128,
            1,
            max_span_tokens=max_span_tokens,
        )


@pytest.mark.parametrize(
    ("first_position", "max_new_tokens"),
    [(-1, 1), (0, 0), (0, 257), (65535, 2)],
)
def test_reusable_flow_plan_rejects_invalid_ranges(
    first_position, max_new_tokens
):
    with pytest.raises(ValueError):
        reusable_flow_plan(first_position, max_new_tokens)
