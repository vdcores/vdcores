"""Persistent state shared by reusable DeepSeek-V4 decode flows."""

from __future__ import annotations

import torch

from .deepseek_v4 import DeepSeekV4FlashConfig


class DeepSeekV4LiveDecodeState:
    """Layer-private KV and compressor state for batch-one decoding.

    Storage is family-major and has a constant byte stride between layers so a
    compact VDCores layer loop can address both reads and writebacks from its
    existing loop counters.  Scratch tensors remain owned by each prepared
    structural flow; only state that survives a token launch lives here.
    """

    def __init__(
        self,
        max_seq_len: int,
        *,
        device: torch.device | str = "cuda",
        config: DeepSeekV4FlashConfig | None = None,
    ):
        self.config = config or DeepSeekV4FlashConfig()
        self.device = torch.device(device)
        self.max_seq_len = int(max_seq_len)
        if not 1 <= self.max_seq_len <= 65536:
            raise ValueError("live decode max_seq_len must be in [1,65536]")

        self.layers_by_kind = {
            kind: tuple(
                layer_id
                for layer_id in range(self.config.num_layers)
                if self.config.attention_kind(layer_id) == kind
            )
            for kind in ("swa", "csa", "hca")
        }
        self.layer_offsets = {
            layer_id: (kind, offset)
            for kind, layer_ids in self.layers_by_kind.items()
            for offset, layer_id in enumerate(layer_ids)
        }

        self.attention_cache_storage = {}
        for kind, layer_ids in self.layers_by_kind.items():
            representative = layer_ids[0]
            ratio = self.config.compress_ratios[representative]
            compressed_capacity = (
                self.max_seq_len // ratio if ratio else 0
            )
            self.attention_cache_storage[kind] = torch.zeros(
                (
                    len(layer_ids),
                    self.config.sliding_window + compressed_capacity,
                    self.config.head_dim,
                ),
                dtype=torch.bfloat16,
                device=self.device,
            )

        # Before the sliding window fills, exhaustive CSA attention is packed
        # as [valid window, compressed rows].  Those regions overlap the
        # not-yet-written tail of the 128-row window, so retain the at-most-31
        # completed compressed rows separately until position 127.  This is a
        # small, explicit state buffer rather than an in-place overlapping
        # cache move.
        self.short_csa_compressed = torch.zeros(
            (
                len(self.layers_by_kind["csa"]),
                self.config.sliding_window // 4,
                self.config.head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._prepared_position: int | None = None
        self._pending_short_csa: tuple[int, int] | None = None

        csa_layers = self.layers_by_kind["csa"]
        csa_capacity = self.max_seq_len // 4
        self.index_cache_storage = torch.zeros(
            (
                len(csa_layers),
                csa_capacity,
                self.config.index_head_dim,
            ),
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Ratio-4 compression alternates two eight-row banks.  Rows 0..3 hold
        # the preceding group's overlap half; rows 4..7 hold the current
        # group's ordinary half.  The first group uses rows 0..3 directly.
        self.csa_pool_values = torch.zeros(
            (len(csa_layers), 2, 8, self.config.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.csa_pool_scores = torch.zeros_like(self.csa_pool_values)
        self.index_pool_values = torch.zeros(
            (len(csa_layers), 2, 8, self.config.index_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.index_pool_scores = torch.zeros_like(self.index_pool_values)

        hca_layers = self.layers_by_kind["hca"]
        self.hca_pool_values = torch.zeros(
            (len(hca_layers), 128, self.config.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.hca_pool_scores = torch.zeros_like(self.hca_pool_values)

    def _offset(self, layer_id: int, expected_kind: str | None = None):
        try:
            kind, offset = self.layer_offsets[int(layer_id)]
        except KeyError as exc:
            raise ValueError(f"invalid transformer layer {layer_id}") from exc
        if expected_kind is not None and kind != expected_kind:
            raise ValueError(
                f"layer {layer_id} is {kind}, expected {expected_kind}"
            )
        return kind, offset

    def attention_cache(self, layer_id: int, compressed_rows: int):
        kind, offset = self._offset(layer_id)
        rows = self.config.sliding_window + int(compressed_rows)
        storage = self.attention_cache_storage[kind]
        if not self.config.sliding_window <= rows <= storage.shape[1]:
            raise ValueError("requested attention cache view exceeds capacity")
        return storage[offset, :rows]

    def index_cache(self, layer_id: int, compressed_rows: int):
        _, offset = self._offset(layer_id, "csa")
        rows = int(compressed_rows)
        if not 0 <= rows <= self.index_cache_storage.shape[1]:
            raise ValueError("requested index cache view exceeds capacity")
        return self.index_cache_storage[offset, :rows]

    def csa_pool_history(self, layer_id: int, position: int, *, index=False):
        _, offset = self._offset(layer_id, "csa")
        group = int(position) // 4
        bank = group & 1
        rows = 3 if group == 0 else 7
        values = self.index_pool_values if index else self.csa_pool_values
        scores = self.index_pool_scores if index else self.csa_pool_scores
        return values[offset, bank, :rows], scores[offset, bank, :rows]

    def csa_pool_destinations(self, layer_id: int, position: int, *, index=False):
        _, offset = self._offset(layer_id, "csa")
        group, within = divmod(int(position), 4)
        bank = group & 1
        next_bank = (group + 1) & 1
        normal_row = within if group == 0 else 4 + within
        values = self.index_pool_values if index else self.csa_pool_values
        scores = self.index_pool_scores if index else self.csa_pool_scores
        return (
            values[offset, next_bank, within],
            scores[offset, next_bank, within],
            values[offset, bank, normal_row],
            scores[offset, bank, normal_row],
        )

    def csa_pool_storage(self, layer_id: int, *, index=False):
        _, offset = self._offset(layer_id, "csa")
        values = self.index_pool_values if index else self.csa_pool_values
        scores = self.index_pool_scores if index else self.csa_pool_scores
        return values[offset], scores[offset]

    def hca_pool_history(self, layer_id: int, position: int):
        _, offset = self._offset(layer_id, "hca")
        within = int(position) % 128
        return (
            self.hca_pool_values[offset, :within],
            self.hca_pool_scores[offset, :within],
        )

    def hca_pool_destination(self, layer_id: int, position: int):
        _, offset = self._offset(layer_id, "hca")
        within = int(position) % 128
        return (
            self.hca_pool_values[offset, within],
            self.hca_pool_scores[offset, within],
        )

    def hca_pool_storage(self, layer_id: int):
        _, offset = self._offset(layer_id, "hca")
        return self.hca_pool_values[offset], self.hca_pool_scores[offset]

    def persistent_tensors(self) -> tuple[torch.Tensor, ...]:
        """Return every tensor that a completed token may update in place."""

        return (
            *self.attention_cache_storage.values(),
            self.short_csa_compressed,
            self.index_cache_storage,
            self.csa_pool_values,
            self.csa_pool_scores,
            self.index_pool_values,
            self.index_pool_scores,
            self.hca_pool_values,
            self.hca_pool_scores,
        )

    @torch.inference_mode()
    def prepare_decode_position(self, position: int) -> None:
        """Pack short-context CSA rows before one sequential decode token."""

        position = int(position)
        if not 0 <= position < self.max_seq_len:
            raise ValueError("decode position is outside live-state capacity")
        if self._prepared_position is not None:
            if position == self._prepared_position:
                return
            if position != self._prepared_position + 1:
                raise ValueError(
                    "reusable live decode positions must advance sequentially"
                )
        elif 0 < position < 4:
            # The PyTorch importer mirrors the checkpoint's compact first
            # group in rows 0..3.  Reusable VDCores state uses the ordinary
            # half (rows 4..7) from token zero onward so the same address
            # transform applies before and after the first compression.
            for storage in (
                self.csa_pool_values,
                self.csa_pool_scores,
                self.index_pool_values,
                self.index_pool_scores,
            ):
                storage[:, 0, 4 : 4 + position].copy_(
                    storage[:, 0, :position]
                )

        if self._pending_short_csa is not None:
            packed_row, compressed_row = self._pending_short_csa
            self.short_csa_compressed[:, compressed_row].copy_(
                self.attention_cache_storage["csa"][:, packed_row]
            )
            self._pending_short_csa = None

        window = self.config.sliding_window
        if position < window:
            completed = position // 4
            if completed:
                self.attention_cache_storage["csa"][
                    :, position + 1 : position + 1 + completed
                ].copy_(self.short_csa_compressed[:, :completed])
            if (position + 1) % 4 == 0 and position + 1 < window:
                self._pending_short_csa = (
                    position + 1 + completed,
                    completed,
                )

        self._prepared_position = position

    @torch.inference_mode()
    def import_pytorch_prefill(self, model, prefix_length: int) -> None:
        """Import the official batch-one PyTorch prefill state.

        ``prefix_length`` tokens must already have been processed together by
        ``model(..., start_pos=0)``.  The next token is intentionally excluded:
        VDCores consumes that final prompt token at decode position
        ``prefix_length``.  No layout or numeric conversion is performed here;
        the official implementation and VDCores both retain BF16 KV rows and
        FP32 incremental compressor state.
        """

        prefix_length = int(prefix_length)
        if not 0 <= prefix_length < self.max_seq_len:
            raise ValueError(
                "prefill prefix length must be smaller than live capacity"
            )
        layers = tuple(model.layers)
        if len(layers) != self.config.num_layers:
            raise ValueError(
                "PyTorch prefill model must contain all 43 transformer layers"
            )

        window = self.config.sliding_window
        for layer_id, layer in enumerate(layers):
            kind, _ = self._offset(layer_id)
            ratio = self.config.compress_ratios[layer_id]
            compressed_rows = prefix_length // ratio if ratio else 0
            destination = self.attention_cache(layer_id, compressed_rows)
            source = layer.attn.kv_cache[0]
            if source.dtype != torch.bfloat16 or source.shape[-1] != (
                self.config.head_dim
            ):
                raise ValueError(
                    f"layer {layer_id} has an incompatible PyTorch KV cache"
                )
            window_rows = min(prefix_length, window)
            if window_rows:
                destination[:window_rows].copy_(source[:window_rows])
            if compressed_rows:
                destination[
                    window : window + compressed_rows
                ].copy_(source[window : window + compressed_rows])
                if kind == "csa" and prefix_length < window:
                    self.short_csa_compressed[
                        self.layer_offsets[layer_id][1], :compressed_rows
                    ].copy_(
                        source[window : window + compressed_rows]
                    )

            if kind == "csa":
                index_source = layer.attn.indexer.kv_cache[0]
                self.index_cache(layer_id, compressed_rows).copy_(
                    index_source[:compressed_rows]
                )
                self._import_ratio4_pool(
                    layer_id,
                    prefix_length,
                    layer.attn.compressor,
                    index=False,
                )
                self._import_ratio4_pool(
                    layer_id,
                    prefix_length,
                    layer.attn.indexer.compressor,
                    index=True,
                )
            elif kind == "hca":
                remainder = prefix_length % 128
                if remainder:
                    _, offset = self._offset(layer_id, "hca")
                    compressor = layer.attn.compressor
                    self.hca_pool_values[offset, :remainder].copy_(
                        compressor.kv_state[0, :remainder, : self.config.head_dim]
                    )
                    self.hca_pool_scores[offset, :remainder].copy_(
                        compressor.score_state[
                            0, :remainder, : self.config.head_dim
                        ]
                    )

    def _import_ratio4_pool(
        self,
        layer_id: int,
        prefix_length: int,
        compressor,
        *,
        index: bool,
    ) -> None:
        """Map the official 2x4 overlap state into VDCores ping-pong banks."""

        _, offset = self._offset(layer_id, "csa")
        width = (
            self.config.index_head_dim if index else self.config.head_dim
        )
        values = self.index_pool_values if index else self.csa_pool_values
        scores = self.index_pool_scores if index else self.csa_pool_scores
        group, remainder = divmod(int(prefix_length), 4)
        bank = group & 1
        next_bank = (group + 1) & 1
        source_values = compressor.kv_state[0]
        source_scores = compressor.score_state[0]

        # A completed preceding group becomes the overlap half of the group
        # that VDCores will continue filling.
        if group:
            values[offset, bank, :4].copy_(source_values[:4, :width])
            scores[offset, bank, :4].copy_(source_scores[:4, :width])

        if remainder:
            current = source_values[4 : 4 + remainder]
            current_scores = source_scores[4 : 4 + remainder]
            ordinary_start = 0 if group == 0 else 4
            values[
                offset,
                bank,
                ordinary_start : ordinary_start + remainder,
            ].copy_(current[:, width : 2 * width])
            scores[
                offset,
                bank,
                ordinary_start : ordinary_start + remainder,
            ].copy_(current_scores[:, width : 2 * width])
            # The first half is already the overlap prefix of the subsequent
            # group, so retain it in the opposite bank as decode advances.
            values[offset, next_bank, :remainder].copy_(current[:, :width])
            scores[offset, next_bank, :remainder].copy_(
                current_scores[:, :width]
            )
