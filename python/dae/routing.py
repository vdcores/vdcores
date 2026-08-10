"""On-device routing state for queue-backed dynamic address resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class RoutedAddressTable:
    """HBM routing ids plus an expert-by-field device-pointer table.

    The first 32 bytes are consumed directly by the LDU handler: six int32
    route ids, the pointer-field stride, and the expert count.  The remaining
    storage is a row-major uint64 pointer table.  Target tensors are retained
    so every queued pointer remains valid for the table's lifetime.
    """

    ROUTE_COUNT = 6
    HEADER_BYTES = 32

    def __init__(self, columns: Mapping[str, Sequence[torch.Tensor]]):
        if not columns:
            raise ValueError("routed address table requires at least one field")
        self._field_ids = {name: index for index, name in enumerate(columns)}
        if any(not name for name in self._field_ids):
            raise ValueError("routed address field names must be non-empty")

        normalized = tuple(tuple(column) for column in columns.values())
        expert_count = len(normalized[0])
        if expert_count <= 0:
            raise ValueError("routed address table requires at least one expert")
        if expert_count > 0x7FFFFFFF:
            raise ValueError("routed address expert count must fit in int32")
        if len(normalized) > 0xFFFF:
            raise ValueError("routed address field count must fit in uint16")
        if any(len(column) != expert_count for column in normalized):
            raise ValueError("every routed address field must cover every expert")

        first = normalized[0][0]
        if not isinstance(first, torch.Tensor) or first.device.type != "cuda":
            raise ValueError("routed address targets must be CUDA tensors")
        device = first.device
        targets = []
        pointer_rows = []
        for expert in range(expert_count):
            pointer_row = []
            for column in normalized:
                target = column[expert]
                if not isinstance(target, torch.Tensor):
                    raise ValueError("routed address targets must be tensors")
                if target.device != device:
                    raise ValueError("routed address targets must share one CUDA device")
                if not target.is_contiguous():
                    raise ValueError("routed address targets must be contiguous")
                address = target.data_ptr()
                if not 0 < address < (1 << 63):
                    raise ValueError("routed address pointer must fit in signed int64")
                targets.append(target)
                pointer_row.append(address)
            pointer_rows.append(pointer_row)

        self.expert_count = expert_count
        self.field_count = len(normalized)
        self.device = device
        self._targets = tuple(targets)
        self.state = torch.empty(
            (4 + expert_count * self.field_count,),
            dtype=torch.int64,
            device=device,
        )
        header = self.state[:4].view(torch.int32)
        header.zero_()
        header[6] = self.field_count
        header[7] = expert_count
        pointers = torch.tensor(
            pointer_rows,
            dtype=torch.int64,
            device=device,
        )
        self.state[4:].copy_(pointers.reshape(-1))
        self.route_indices = header[: self.ROUTE_COUNT]
        self.pointer_table = self.state[4:].reshape(expert_count, self.field_count)

    def field(self, name: str) -> int:
        try:
            return self._field_ids[name]
        except KeyError:
            raise KeyError(f"unknown routed address field {name!r}") from None


__all__ = ["RoutedAddressTable"]
