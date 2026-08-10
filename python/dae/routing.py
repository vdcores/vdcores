"""On-device routing state for queue-backed dynamic address resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


class RoutedAddressTable:
    """HBM routing ids plus an expert-by-field device-pointer table.

    The first 48 bytes are consumed directly by the LDU handler: eight int32
    route-id slots, the pointer-field stride, the expert count, and two padding
    words.  The remaining storage is a row-major uint64 pointer table.  Target
    tensors are retained so every routed load target remains valid.
    """

    ROUTE_COUNT = 6
    HEADER_BYTES = 48
    MAX_FIELDS = 1 << 13

    @classmethod
    def from_pointer_columns(
        cls,
        columns: Mapping[str, Sequence[int]],
        *,
        device,
        owners: Sequence[object] = (),
    ) -> "RoutedAddressTable":
        """Build a routed table from retained-storage device addresses."""
        if not columns:
            raise ValueError("routed address table requires at least one field")
        field_ids = {name: index for index, name in enumerate(columns)}
        if any(not name for name in field_ids):
            raise ValueError("routed address field names must be non-empty")
        normalized = tuple(
            tuple(int(address) for address in column)
            for column in columns.values()
        )
        expert_count = len(normalized[0])
        if expert_count <= 0:
            raise ValueError("routed address table requires at least one expert")
        if expert_count > 0x7FFFFFFF:
            raise ValueError("routed address expert count must fit in int32")
        if len(normalized) > cls.MAX_FIELDS:
            raise ValueError("routed address field count must fit in 13 bits")
        if any(len(column) != expert_count for column in normalized):
            raise ValueError("every routed address field must cover every expert")
        if any(
            not 0 < address < (1 << 63)
            for column in normalized
            for address in column
        ):
            raise ValueError("routed address pointers must fit in signed int64")

        target_device = torch.device(device)
        if target_device.type != "cuda":
            raise ValueError("routed address table must reside on CUDA")
        pointer_columns = torch.tensor(normalized, dtype=torch.int64)
        pointer_rows = pointer_columns.t().contiguous()

        self = cls.__new__(cls)
        self._field_ids = field_ids
        self.expert_count = expert_count
        self.field_count = len(normalized)
        self.device = target_device
        self._targets = tuple(owners)
        self.state = torch.empty(
            (6 + expert_count * self.field_count,),
            dtype=torch.int64,
            device=target_device,
        )
        header = self.state[:6].view(torch.int32)
        header.zero_()
        header[8] = self.field_count
        header[9] = expert_count
        self.state[6:].copy_(pointer_rows.to(target_device).reshape(-1))
        self.route_indices = header[: self.ROUTE_COUNT]
        self.route_indices_storage = header[:8]
        self.pointer_table = self.state[6:].reshape(expert_count, self.field_count)
        return self

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
        if len(normalized) > self.MAX_FIELDS:
            raise ValueError("routed address field count must fit in 13 bits")
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
            (6 + expert_count * self.field_count,),
            dtype=torch.int64,
            device=device,
        )
        header = self.state[:6].view(torch.int32)
        header.zero_()
        header[8] = self.field_count
        header[9] = expert_count
        pointers = torch.tensor(
            pointer_rows,
            dtype=torch.int64,
            device=device,
        )
        self.state[6:].copy_(pointers.reshape(-1))
        self.route_indices = header[: self.ROUTE_COUNT]
        self.route_indices_storage = header[:8]
        self.pointer_table = self.state[6:].reshape(expert_count, self.field_count)

    def field(self, name: str) -> int:
        try:
            return self._field_ids[name]
        except KeyError:
            raise KeyError(f"unknown routed address field {name!r}") from None


class IndexedLoadTable:
    """Stable HBM pointers used by LDU for runtime indexed row loads."""

    def __init__(self, rows: torch.Tensor, indices: torch.Tensor):
        if rows.device.type != "cuda" or indices.device != rows.device:
            raise ValueError("indexed rows and indices must share one CUDA device")
        if not rows.is_contiguous() or rows.ndim < 2:
            raise ValueError("indexed rows must be a contiguous row-major tensor")
        if indices.dtype != torch.int32 or not indices.is_contiguous():
            raise ValueError("indexed load indices must be contiguous int32")
        row_count = rows.shape[0]
        row_bytes = rows[0].numel() * rows.element_size()
        if row_count <= 0 or row_count > 0x7FFFFFFF:
            raise ValueError("indexed row count must fit in int32")
        if row_bytes <= 0 or row_bytes > 0x7FFFFFFF:
            raise ValueError("indexed row stride must fit in int32")
        self.rows = rows
        self.indices = indices
        records = [
            [
                rows.data_ptr(),
                indices.data_ptr() + rank * indices.element_size(),
                (row_bytes << 32) | row_count,
            ]
            for rank in range(indices.numel())
        ]
        self.state = torch.tensor(
            records,
            dtype=torch.int64,
            device=rows.device,
        )


__all__ = ["IndexedLoadTable", "RoutedAddressTable"]
