import torch

from .op_families import ComputeOpFamilyRef, family_name
from .runtime import opcode

UNRESOLVED_COMPUTE_OPCODE_PLACEHOLDER = 0xFFFF


def decode_opcode(op: int) -> str:
    for name, value in vars(opcode).items():
        if value == op:
            return name

    # Some memory instructions toggle the writeback bit after the base opcode
    # has already been chosen, so try decoding with it set as well.
    op_with_writeback = op | 2
    for name, value in vars(opcode).items():
        if value == op_with_writeback:
            return name

    return f"UNKNOWN_OPCODE[0x{op:04x}]"


def encode_bfloat16_u16(value: float) -> int:
    return torch.tensor(value, dtype=torch.bfloat16).view(torch.uint16).item()


def normalize_compute_opcode_reference(op_ref: int | str | ComputeOpFamilyRef) -> tuple[int | None, str | None]:
    if isinstance(op_ref, int):
        return op_ref, None
    return None, family_name(op_ref)


def resolve_compute_opcode_value(
    opcode_value: int | None,
    op_family_name: str | None,
    *,
    allow_unresolved: bool = False,
) -> int:
    if opcode_value is not None:
        return opcode_value
    assert op_family_name is not None
    try:
        return int(getattr(opcode, op_family_name))
    except AttributeError as exc:
        if allow_unresolved:
            return UNRESOLVED_COMPUTE_OPCODE_PLACEHOLDER
        raise ValueError(
            f"Missing runtime opcode for op-family instruction {op_family_name}. "
            "Rebuild dae.runtime with this generated compute op."
        ) from exc


def compute_operator_name(opcode_value: int | None, op_family_name: str | None) -> str:
    if op_family_name is not None:
        return op_family_name
    assert opcode_value is not None
    return decode_opcode(opcode_value)


def encode_compute_instruction_tensor(
    opcode_value: int | None,
    op_family_name: str | None,
    args: list[int],
    tensor: torch.Tensor | None = None,
) -> torch.Tensor:
    if tensor is None:
        tensor = torch.empty((4,), dtype=torch.uint16)
    else:
        tensor = tensor.view(torch.uint16)
        assert tensor.numel() == 4

    tensor.zero_()
    tensor[0] = resolve_compute_opcode_value(
        opcode_value,
        op_family_name,
        allow_unresolved=True,
    )
    assert len(args) <= 3
    for index, arg in enumerate(args):
        assert 0 <= arg < 2**16, "args must be uint16"
        tensor[index + 1] = arg
    return tensor.view(torch.uint8)


# Keep the historic typo as an alias for compatibility with older code.
dedcode_opcode = decode_opcode


__all__ = [
    "compute_operator_name",
    "decode_opcode",
    "dedcode_opcode",
    "encode_compute_instruction_tensor",
    "encode_bfloat16_u16",
    "normalize_compute_opcode_reference",
    "resolve_compute_opcode_value",
    "UNRESOLVED_COMPUTE_OPCODE_PLACEHOLDER",
]
