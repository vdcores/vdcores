import torch

from .runtime import opcode


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


# Keep the historic typo as an alias for compatibility with older code.
dedcode_opcode = decode_opcode


__all__ = [
    "decode_opcode",
    "dedcode_opcode",
    "encode_bfloat16_u16",
]
