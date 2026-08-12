"""Affine resident storage for routed DeepSeek expert operands."""

from __future__ import annotations


class AffineExpertArena:
    """One fixed-base, expert-major arena with no device pointer records.

    Route IDs and route weights occupy fixed addresses at the start of the
    allocation.  Every expert field is then addressed as::

        base + layer_offset + expert_id * expert_stride + field_offset

    The native DeepSeek-V4 w1/w3/w2 payloads have equal byte sizes, which keeps
    both the expert and layer strides constant.
    """

    ALIGNMENT = 256
    HEADER_BYTES = 256
    ROUTE_SLOTS = 8
    ROUTE_INDICES_OFFSET = 0
    ROUTE_WEIGHTS_OFFSET = 32
    META_FIELD_BYTES = 16
    TAGS = ("w1", "w3", "w2")
    META_FIELDS = ("weight_scale_2", "input_scale", "alpha")
    NATIVE_TILE_BYTES = 18432

    def __init__(self, storage, config, layer_ids):
        import torch

        if storage.device.type != "cuda" or storage.dtype != torch.uint8:
            raise ValueError("affine expert arena must be CUDA uint8 storage")
        if not storage.is_contiguous():
            raise ValueError("affine expert arena storage must be contiguous")
        self.storage = storage
        self.config = config
        self.layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        if not self.layer_ids or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("affine expert layers must be unique and non-empty")
        self._layer_slots = {
            layer_id: slot for slot, layer_id in enumerate(self.layer_ids)
        }

        self.weight_bytes = self._weight_bytes(config)
        metadata_bytes = len(self.TAGS) * len(self.META_FIELDS) * self.META_FIELD_BYTES
        self.expert_stride = self._align(
            len(self.TAGS) * self.weight_bytes + metadata_bytes
        )
        self.layer_stride = config.num_experts * self.expert_stride
        expected_bytes = self.HEADER_BYTES + len(self.layer_ids) * self.layer_stride
        if storage.numel() != expected_bytes:
            raise ValueError(
                f"affine expert arena expected {expected_bytes} bytes, got {storage.numel()}"
            )
        if self.expert_stride % self.ALIGNMENT:
            raise AssertionError("affine expert stride must be 256-byte aligned")
        if self.expert_stride // self.ALIGNMENT > 0xFFFF:
            raise ValueError("affine expert stride cannot be encoded by the LDU preload")

        self.route_indices = storage[
            self.ROUTE_INDICES_OFFSET : self.ROUTE_INDICES_OFFSET + 32
        ].view(torch.int32)
        self.route_weights = storage[
            self.ROUTE_WEIGHTS_OFFSET : self.ROUTE_WEIGHTS_OFFSET + 32
        ].view(torch.float32)

    @classmethod
    def allocate(cls, config, layer_ids, *, device):
        import torch

        layer_ids = tuple(int(layer_id) for layer_id in layer_ids)
        storage = torch.empty(
            (cls.storage_bytes(config, len(layer_ids)),),
            dtype=torch.uint8,
            device=device,
        )
        arena = cls(storage, config, layer_ids)
        storage[: cls.HEADER_BYTES].zero_()
        return arena

    @classmethod
    def storage_bytes(cls, config, layer_count: int) -> int:
        if layer_count <= 0:
            raise ValueError("affine expert arena requires at least one layer")
        weight_bytes = cls._weight_bytes(config)
        metadata_bytes = len(cls.TAGS) * len(cls.META_FIELDS) * cls.META_FIELD_BYTES
        expert_stride = cls._align(len(cls.TAGS) * weight_bytes + metadata_bytes)
        return cls.HEADER_BYTES + layer_count * config.num_experts * expert_stride

    @classmethod
    def _weight_bytes(cls, config) -> int:
        shapes = (
            (config.expert_intermediate_size, config.hidden_size),
            (config.expert_intermediate_size, config.hidden_size),
            (config.hidden_size, config.expert_intermediate_size),
        )
        sizes = {
            (rows // 128) * (k // 256) * cls.NATIVE_TILE_BYTES
            for rows, k in shapes
        }
        if len(sizes) != 1:
            raise ValueError("affine expert arena requires equal w1/w3/w2 native sizes")
        return sizes.pop()

    @classmethod
    def _align(cls, value: int) -> int:
        return (int(value) + cls.ALIGNMENT - 1) & -cls.ALIGNMENT

    def layer_offset(self, layer_id: int) -> int:
        try:
            slot = self._layer_slots[int(layer_id)]
        except KeyError:
            raise KeyError(f"layer {layer_id} is not resident in the affine expert arena") from None
        return self.HEADER_BYTES + slot * self.layer_stride

    def layer_storage(self, layer_id: int):
        start = self.layer_offset(layer_id)
        return self.storage[start : start + self.layer_stride]

    def expert_offset(self, layer_id: int, expert_id: int) -> int:
        if not 0 <= expert_id < self.config.num_experts:
            raise ValueError("expert ID is outside the affine arena")
        return self.layer_offset(layer_id) + expert_id * self.expert_stride

    def weight_offset(self, layer_id: int, expert_id: int, tag: str) -> int:
        try:
            tag_index = self.TAGS.index(tag)
        except ValueError:
            raise KeyError(f"unknown affine expert weight {tag!r}") from None
        return self.expert_offset(layer_id, expert_id) + tag_index * self.weight_bytes

    def metadata_offset(
        self, layer_id: int, expert_id: int, tag: str, field: str
    ) -> int:
        try:
            tag_index = self.TAGS.index(tag)
            field_index = self.META_FIELDS.index(field)
        except ValueError:
            raise KeyError(f"unknown affine expert metadata {tag}.{field}") from None
        metadata_base = len(self.TAGS) * self.weight_bytes
        metadata_index = tag_index * len(self.META_FIELDS) + field_index
        return (
            self.expert_offset(layer_id, expert_id)
            + metadata_base
            + metadata_index * self.META_FIELD_BYTES
        )

    def field_offset(self, layer_id: int, name: str, *, tile: int = 0) -> int:
        """Return a layer-zero-relative field offset before dynamic terms."""

        tag, separator, field = name.partition(".")
        if tag not in self.TAGS or not separator:
            raise KeyError(f"unknown affine expert field {name!r}")
        self.layer_offset(layer_id)
        layer_zero = self.HEADER_BYTES
        if field == "weight":
            if tile < 0:
                raise ValueError("affine expert weight tile must be non-negative")
            offset = self.weight_offset(self.layer_ids[0], 0, tag)
            offset += tile * self.NATIVE_TILE_BYTES * (
                self.config.hidden_size // 256
                if tag in ("w1", "w3")
                else self.config.expert_intermediate_size // 256
            )
            if offset + self.NATIVE_TILE_BYTES > layer_zero + self.expert_stride:
                raise ValueError("affine expert weight tile exceeds its record")
            return offset
        if tile:
            raise ValueError("affine metadata fields do not have tiles")
        return self.metadata_offset(self.layer_ids[0], 0, tag, field)


__all__ = ["AffineExpertArena"]
