"""Shape-derived first-pass tile and SM assignments for DeepSeek-V4 decode."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeAssignment:
    task: str
    rows: int
    k: int
    num_sms: int
    row_alignment: int = 1
    tile_rows: int = 1
    tile_k: int = 0

    def shard(self, sm: int) -> tuple[int, int]:
        """Return an aligned ``(row_start, row_count)`` for one assigned SM."""
        if not 0 <= sm < self.num_sms:
            return 0, 0
        groups = self.rows // self.row_alignment
        groups_per_sm, extra = divmod(groups, self.num_sms)
        group_start = sm * groups_per_sm + min(sm, extra)
        group_count = groups_per_sm + int(sm < extra)
        return (
            group_start * self.row_alignment,
            group_count * self.row_alignment,
        )


class DeepSeekV4ShapePolicy:
    """Correctness-first assignments derived only from operator shape.

    These choices expose all naturally independent row/scale/head tiles up to
    the resident SM count. They are intentionally an initial schedule, not a
    tuned performance policy.
    """

    def __init__(self, resident_sms: int):
        if resident_sms <= 0:
            raise ValueError("resident_sms must be positive")
        self.resident_sms = resident_sms

    def _rows(
        self,
        task: str,
        rows: int,
        k: int,
        *,
        alignment: int = 1,
        tile_rows: int = 1,
        tile_k: int = 0,
    ) -> ShapeAssignment:
        if rows <= 0 or k <= 0 or rows % alignment:
            raise ValueError(f"invalid {task} shape M={rows} K={k}")
        return ShapeAssignment(
            task,
            rows,
            k,
            min(self.resident_sms, rows // alignment),
            row_alignment=alignment,
            tile_rows=tile_rows,
            tile_k=tile_k,
        )

    def fp8_gemv(self, rows: int, k: int) -> ShapeAssignment:
        if k % 128:
            raise ValueError("FP8 GEMV K must be divisible by 128")
        return self._rows(
            "fp8_gemv",
            rows,
            k,
            tile_rows=max(1, 65520 // k),
            tile_k=128,
        )

    def fp8_umma_gemv(self, rows: int, k: int) -> ShapeAssignment:
        if rows % 128 or k % 128:
            raise ValueError("native FP8 GEMV requires M128/K128 alignment")
        return self._rows(
            "fp8_umma_gemv",
            rows,
            k,
            alignment=128,
            tile_rows=128,
            tile_k=128,
        )

    def nvfp4_gemv(self, rows: int, k: int) -> ShapeAssignment:
        if k % 256:
            raise ValueError("NVFP4 GEMV K must be divisible by 256")
        packed_row_bytes = k // 2
        tile_rows = (65520 // packed_row_bytes // 8) * 8
        if tile_rows <= 0:
            raise ValueError("NVFP4 K is too large for one aligned routed tile")
        return self._rows(
            "nvfp4_gemv",
            rows,
            k,
            alignment=8,
            tile_rows=tile_rows,
            tile_k=256,
        )

    def bf16_gemv(self, rows: int, k: int) -> ShapeAssignment:
        return self._rows(
            "bf16_gemv", rows, k, tile_rows=1, tile_k=min(k, 16384)
        )

    def fp32_bf16_gemv(self, rows: int, k: int) -> ShapeAssignment:
        return self._rows(
            "fp32_bf16_gemv", rows, k, tile_rows=1, tile_k=min(k, 16384)
        )

    def quantize(self, k: int, block: int) -> ShapeAssignment:
        if k <= 0 or block <= 0 or k % block:
            raise ValueError("quantization K must contain complete scale blocks")
        blocks = k // block
        blocks_per_sm = 16 if block == 16 else 1
        return ShapeAssignment(
            f"quantize_{block}",
            blocks,
            k,
            min(
                self.resident_sms,
                (blocks + blocks_per_sm - 1) // blocks_per_sm,
            ),
            row_alignment=1,
            tile_rows=blocks_per_sm,
            tile_k=block,
        )

    def attention(self, heads: int, head_dim: int) -> ShapeAssignment:
        return self._rows(
            "sparse_attention",
            heads,
            head_dim,
            tile_rows=1,
            tile_k=head_dim,
        )

    def parallel_partition(self, branch: int, branches: int) -> tuple[int, int]:
        """Return a shape-independent contiguous SM share for one ready branch."""
        if branches <= 0 or branches > self.resident_sms:
            raise ValueError("parallel branch count must fit the resident SM grid")
        if not 0 <= branch < branches:
            raise ValueError("parallel branch index is out of range")
        base_width, extra = divmod(self.resident_sms, branches)
        width = base_width + int(branch < extra)
        base = branch * base_width + min(branch, extra)
        return base, width

    def uniform_parallel_partition(
        self, branch: int, branches: int
    ) -> tuple[int, int]:
        """Return equal contiguous shares, leaving any remainder SMs idle."""
        if branches <= 0 or branches > self.resident_sms:
            raise ValueError("parallel branch count must fit the resident SM grid")
        if not 0 <= branch < branches:
            raise ValueError("parallel branch index is out of range")
        width = self.resident_sms // branches
        return branch * width, width

    def weighted_parallel_partition(
        self, branch: int, weights: tuple[int, ...]
    ) -> tuple[int, int]:
        """Partition the grid by positive shape weights with one SM minimum."""
        if not weights or len(weights) > self.resident_sms:
            raise ValueError("weighted branches must fit the resident SM grid")
        if any(weight <= 0 for weight in weights):
            raise ValueError("parallel branch weights must be positive")
        if not 0 <= branch < len(weights):
            raise ValueError("parallel branch index is out of range")
        remaining = self.resident_sms - len(weights)
        total_weight = sum(weights)
        extras = [remaining * weight // total_weight for weight in weights]
        unassigned = remaining - sum(extras)
        remainders = [remaining * weight % total_weight for weight in weights]
        order = sorted(
            range(len(weights)), key=lambda index: (-remainders[index], index)
        )
        for index in order[:unassigned]:
            extras[index] += 1
        widths = [extra + 1 for extra in extras]
        return sum(widths[:branch]), widths[branch]


__all__ = ["DeepSeekV4ShapePolicy", "ShapeAssignment"]
