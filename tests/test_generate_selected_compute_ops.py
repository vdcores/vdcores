import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_selected_compute_ops",
    ROOT / "tools" / "generate_selected_compute_ops.py",
)
assert SPEC is not None and SPEC.loader is not None
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_full_ffn_selection_keeps_linear1_near_dispatch_loop() -> None:
    selected = [
        GENERATOR.MXFP_GATE_UP_FIXED_RING_OP,
        GENERATOR.MXFP_DOWN_FIXED_RING_OP,
        "OP_PROFILE_EVENT",
        "OP_TERMINATEC",
    ]

    ordered = GENERATOR.order_selected_entries(selected)

    assert ordered == [
        GENERATOR.MXFP_DOWN_FIXED_RING_OP,
        GENERATOR.MXFP_GATE_UP_FIXED_RING_OP,
        "OP_PROFILE_EVENT",
        "OP_TERMINATEC",
    ]
    assert selected[0] == GENERATOR.MXFP_GATE_UP_FIXED_RING_OP


def test_dispatch_layout_hint_requires_both_ffn_handlers() -> None:
    selected = [GENERATOR.MXFP_GATE_UP_FIXED_RING_OP, "OP_TERMINATEC"]

    assert GENERATOR.order_selected_entries(selected) == selected
