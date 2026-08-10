"""Raw-checkpoint contract and lazy loader for DeepSeek-V4-Flash-NVFP4."""

from __future__ import annotations

import json
import math
import re
import struct
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .deepseek_v4 import DeepSeekV4FlashConfig


INDEX_FILENAME = "model.safetensors.index.json"
_MAX_HEADER_BYTES = 128 * 1024 * 1024
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class ExpectedTensorSpec:
    dtype: str
    shape: tuple[int, ...]

    @property
    def nbytes(self) -> int:
        return math.prod(self.shape) * _DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class SafeTensorSpec:
    name: str
    filename: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def nbytes(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


@dataclass(frozen=True)
class DeepSeekV4CheckpointAudit:
    tensor_count: int
    inference_tensor_count: int
    mtp_tensor_count: int
    shard_count: int
    tensor_bytes: int
    files_checked: bool


@dataclass(frozen=True)
class Fp8LinearCheckpointTensors:
    """Raw E4M3/UE8M0 tensors for one checkpoint linear."""

    prefix: str
    weight: object
    scale: object


@dataclass(frozen=True)
class Nvfp4LinearCheckpointTensors:
    """Raw packed-E2M1 tensors and scalar scales for one routed linear."""

    prefix: str
    weight: object
    weight_scale: object
    weight_scale_2: object
    input_scale: object

    @property
    def alpha(self):
        """Return the scalar multiplier consumed by ``SchedNvfp4Gemv``."""
        return (self.weight_scale_2 * self.input_scale).reshape(1)


def _put(
    specs: dict[str, ExpectedTensorSpec],
    name: str,
    dtype: str,
    shape: tuple[int, ...] | list[int],
) -> None:
    if name in specs:
        raise AssertionError(f"duplicate expected checkpoint tensor {name}")
    specs[name] = ExpectedTensorSpec(dtype, tuple(shape))


def _put_fp8_linear(
    specs: dict[str, ExpectedTensorSpec],
    prefix: str,
    m: int,
    k: int,
) -> None:
    if m % 128 or k % 128:
        raise ValueError(f"FP8 checkpoint matrix {prefix} is not block-128 aligned")
    _put(specs, f"{prefix}.weight", "F8_E4M3", (m, k))
    _put(specs, f"{prefix}.scale", "F8_E8M0", (m // 128, k // 128))


def _put_nvfp4_linear(
    specs: dict[str, ExpectedTensorSpec],
    prefix: str,
    m: int,
    k: int,
) -> None:
    if k % 16:
        raise ValueError(f"NVFP4 checkpoint matrix {prefix} is not group-16 aligned")
    _put(specs, f"{prefix}.weight", "U8", (m, k // 2))
    _put(specs, f"{prefix}.weight_scale", "F8_E4M3", (m, k // 16))
    _put(specs, f"{prefix}.weight_scale_2", "F32", ())
    _put(specs, f"{prefix}.input_scale", "F32", ())


def expected_inference_tensor_specs(
    config: DeepSeekV4FlashConfig | None = None,
) -> dict[str, ExpectedTensorSpec]:
    """Return the exact non-MTP tensor contract for the NVIDIA checkpoint."""
    cfg = config or DeepSeekV4FlashConfig()
    specs: dict[str, ExpectedTensorSpec] = {}
    hidden = cfg.hidden_size
    hc_width = cfg.hc_mult * hidden
    hc_mixes = cfg.hc_mult * (cfg.hc_mult + 2)

    _put(specs, "embed.weight", "BF16", (cfg.vocab_size, hidden))
    _put(specs, "norm.weight", "BF16", (hidden,))
    _put(specs, "head.weight", "BF16", (cfg.vocab_size, hidden))
    _put(specs, "hc_head_base", "F32", (cfg.hc_mult,))
    _put(specs, "hc_head_fn", "F32", (cfg.hc_mult, hc_width))
    _put(specs, "hc_head_scale", "F32", (1,))

    for layer_id in range(cfg.num_layers):
        layer = f"layers.{layer_id}"
        for branch in ("attn", "ffn"):
            _put(specs, f"{layer}.hc_{branch}_base", "F32", (hc_mixes,))
            _put(specs, f"{layer}.hc_{branch}_fn", "F32", (hc_mixes, hc_width))
            _put(specs, f"{layer}.hc_{branch}_scale", "F32", (3,))

        attn = f"{layer}.attn"
        _put(specs, f"{layer}.attn_norm.weight", "BF16", (hidden,))
        _put(specs, f"{attn}.attn_sink", "F32", (cfg.num_heads,))
        _put(specs, f"{attn}.q_norm.weight", "BF16", (cfg.q_lora_rank,))
        _put(specs, f"{attn}.kv_norm.weight", "BF16", (cfg.head_dim,))
        _put_fp8_linear(specs, f"{attn}.wq_a", cfg.q_lora_rank, hidden)
        _put_fp8_linear(
            specs,
            f"{attn}.wq_b",
            cfg.num_heads * cfg.head_dim,
            cfg.q_lora_rank,
        )
        _put_fp8_linear(specs, f"{attn}.wkv", cfg.head_dim, hidden)
        group_width = cfg.num_heads * cfg.head_dim // cfg.o_groups
        _put_fp8_linear(
            specs,
            f"{attn}.wo_a",
            cfg.o_groups * cfg.o_lora_rank,
            group_width,
        )
        _put_fp8_linear(
            specs,
            f"{attn}.wo_b",
            hidden,
            cfg.o_groups * cfg.o_lora_rank,
        )

        ratio = cfg.compress_ratios[layer_id]
        if ratio:
            compressor_width = cfg.head_dim * (2 if ratio == 4 else 1)
            compressor = f"{attn}.compressor"
            _put(specs, f"{compressor}.ape", "F32", (ratio, compressor_width))
            _put(specs, f"{compressor}.norm.weight", "BF16", (cfg.head_dim,))
            _put(specs, f"{compressor}.wgate.weight", "BF16", (compressor_width, hidden))
            _put(specs, f"{compressor}.wkv.weight", "BF16", (compressor_width, hidden))
        if ratio == 4:
            indexer = f"{attn}.indexer"
            _put_fp8_linear(
                specs,
                f"{indexer}.wq_b",
                cfg.index_heads * cfg.index_head_dim,
                cfg.q_lora_rank,
            )
            _put(specs, f"{indexer}.weights_proj.weight", "BF16", (cfg.index_heads, hidden))
            index_width = 2 * cfg.index_head_dim
            compressor = f"{indexer}.compressor"
            _put(specs, f"{compressor}.ape", "F32", (ratio, index_width))
            _put(specs, f"{compressor}.norm.weight", "BF16", (cfg.index_head_dim,))
            _put(specs, f"{compressor}.wgate.weight", "BF16", (index_width, hidden))
            _put(specs, f"{compressor}.wkv.weight", "BF16", (index_width, hidden))

        ffn = f"{layer}.ffn"
        _put(specs, f"{layer}.ffn_norm.weight", "BF16", (hidden,))
        _put(specs, f"{ffn}.gate.weight", "BF16", (cfg.num_experts, hidden))
        if layer_id < cfg.num_hash_layers:
            _put(
                specs,
                f"{ffn}.gate.tid2eid",
                "I64",
                (cfg.vocab_size, cfg.experts_per_token),
            )
        else:
            _put(specs, f"{ffn}.gate.bias", "F32", (cfg.num_experts,))

        shared = f"{ffn}.shared_experts"
        _put_fp8_linear(specs, f"{shared}.w1", cfg.expert_intermediate_size, hidden)
        _put_fp8_linear(specs, f"{shared}.w3", cfg.expert_intermediate_size, hidden)
        _put_fp8_linear(specs, f"{shared}.w2", hidden, cfg.expert_intermediate_size)

        for expert_id in range(cfg.num_experts):
            expert = f"{ffn}.experts.{expert_id}"
            _put_nvfp4_linear(
                specs,
                f"{expert}.w1",
                cfg.expert_intermediate_size,
                hidden,
            )
            _put_nvfp4_linear(
                specs,
                f"{expert}.w3",
                cfg.expert_intermediate_size,
                hidden,
            )
            _put_nvfp4_linear(
                specs,
                f"{expert}.w2",
                hidden,
                cfg.expert_intermediate_size,
            )

    return specs


def _validate_shard_name(filename: str) -> None:
    path = Path(filename)
    if path.name != filename or path.is_absolute() or path.suffix != ".safetensors":
        raise ValueError(f"unsafe checkpoint shard name {filename!r}")


def _read_exact(file, size: int) -> bytes:
    payload = file.read(size)
    if len(payload) != size:
        raise ValueError("truncated safetensors file")
    return payload


def _parse_safetensors_header(
    filename: str,
    payload: bytes,
    data_size: int,
) -> dict[str, SafeTensorSpec]:
    try:
        header = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid safetensors JSON header") from error

    specs: dict[str, SafeTensorSpec] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ValueError("invalid safetensors tensor entry")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"unsupported safetensors dtype {dtype!r} for {name}")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or dim < 0 for dim in shape
        ):
            raise ValueError(f"invalid safetensors shape for {name}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(offset, int) for offset in offsets)
        ):
            raise ValueError(f"invalid safetensors offsets for {name}")
        start, end = offsets
        if not 0 <= start <= end <= data_size:
            raise ValueError(f"out-of-range safetensors offsets for {name}")
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ValueError(f"payload size does not match dtype/shape for {name}")
        specs[name] = SafeTensorSpec(
            name=name,
            filename=filename,
            dtype=dtype,
            shape=tuple(shape),
            data_offsets=(start, end),
        )
    return specs


def read_safetensors_header(path: str | Path) -> dict[str, SafeTensorSpec]:
    """Read and validate a safetensors header without materializing payloads."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as file:
        header_size = struct.unpack("<Q", _read_exact(file, 8))[0]
        if header_size == 0 or header_size > _MAX_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header size {header_size}")
        if 8 + header_size > file_size:
            raise ValueError("safetensors header exceeds file size")
        payload = _read_exact(file, header_size)
    return _parse_safetensors_header(
        path.name,
        payload,
        file_size - 8 - header_size,
    )


def read_safetensors_header_url(
    url: str,
    *,
    filename: str | None = None,
    timeout: float = 60.0,
) -> dict[str, SafeTensorSpec]:
    """Range-read a remote safetensors header without downloading its payload."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise ValueError("remote safetensors audit requires an HTTPS URL")
    filename = filename or Path(parsed.path).name
    _validate_shard_name(filename)

    first_request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(first_request, timeout=timeout) as response:
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes 0-7/(\d+)", content_range)
        if response.status != 206 or match is None:
            raise ValueError("remote checkpoint server did not honor the header range")
        file_size = int(match.group(1))
        header_size = struct.unpack("<Q", _read_exact(response, 8))[0]
    if header_size == 0 or header_size > _MAX_HEADER_BYTES:
        raise ValueError(f"invalid safetensors header size {header_size}")
    if 8 + header_size > file_size:
        raise ValueError("safetensors header exceeds remote file size")

    last_byte = 7 + header_size
    header_request = urllib.request.Request(
        url,
        headers={"Range": f"bytes=8-{last_byte}"},
    )
    with urllib.request.urlopen(header_request, timeout=timeout) as response:
        expected_range = f"bytes 8-{last_byte}/{file_size}"
        if (
            response.status != 206
            or response.headers.get("Content-Range") != expected_range
        ):
            raise ValueError("remote checkpoint server returned the wrong header range")
        payload = _read_exact(response, header_size)
    return _parse_safetensors_header(
        filename,
        payload,
        file_size - 8 - header_size,
    )


class DeepSeekV4Checkpoint:
    """Index, audit, and lazily load a local raw NVIDIA checkpoint."""

    def __init__(
        self,
        root: str | Path,
        config: DeepSeekV4FlashConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config or DeepSeekV4FlashConfig()
        index_path = self.root / INDEX_FILENAME
        try:
            payload = json.loads(index_path.read_text())
        except FileNotFoundError:
            raise FileNotFoundError(f"missing checkpoint index: {index_path}") from None
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid checkpoint index: {index_path}") from error
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("checkpoint index has no weight_map")
        self.weight_map: dict[str, str] = {}
        for name, filename in weight_map.items():
            if not isinstance(name, str) or not isinstance(filename, str):
                raise ValueError("checkpoint weight_map entries must be strings")
            _validate_shard_name(filename)
            self.weight_map[name] = filename
        metadata = payload.get("metadata", {})
        total_size = metadata.get("total_size", 0) if isinstance(metadata, dict) else 0
        if not isinstance(total_size, int) or total_size < 0:
            raise ValueError("checkpoint metadata.total_size must be non-negative")
        self.total_size = total_size
        self._headers: dict[str, dict[str, SafeTensorSpec]] = {}

    @property
    def shard_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    def shard_for(self, name: str) -> Path:
        try:
            filename = self.weight_map[name]
        except KeyError:
            raise KeyError(f"checkpoint tensor {name!r} is not indexed") from None
        return self.root / filename

    def layer_tensor_names(self, layer_id: int) -> tuple[str, ...]:
        if not 0 <= layer_id < self.config.num_layers:
            raise IndexError("layer_id is outside the transformer")
        prefix = f"layers.{layer_id}."
        return tuple(name for name in self.weight_map if name.startswith(prefix))

    def _header(self, filename: str) -> dict[str, SafeTensorSpec]:
        if filename not in self._headers:
            self._headers[filename] = read_safetensors_header(self.root / filename)
        return self._headers[filename]

    def inspect(self, names: Iterable[str]) -> dict[str, SafeTensorSpec]:
        """Inspect indexed local tensors without loading their data."""
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in names:
            try:
                grouped[self.weight_map[name]].append(name)
            except KeyError:
                raise KeyError(f"checkpoint tensor {name!r} is not indexed") from None
        result: dict[str, SafeTensorSpec] = {}
        for filename, shard_names in grouped.items():
            header = self._header(filename)
            for name in shard_names:
                try:
                    result[name] = header[name]
                except KeyError:
                    raise ValueError(
                        f"index assigns {name!r} to {filename}, but its header does not"
                    ) from None
        return result

    def audit(self, *, require_files: bool = True) -> DeepSeekV4CheckpointAudit:
        """Validate inference names and, optionally, every local shard header."""
        expected = expected_inference_tensor_specs(self.config)
        actual_names = set(self.weight_map)
        missing = sorted(set(expected) - actual_names)
        unexpected = sorted(
            name for name in actual_names - set(expected) if not name.startswith("mtp.")
        )
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={missing[:8]}")
            if unexpected:
                details.append(f"unexpected={unexpected[:8]}")
            raise ValueError("checkpoint name contract mismatch: " + "; ".join(details))

        tensor_bytes = self.total_size
        if require_files:
            inspected = self.inspect(self.weight_map)
            for name, expected_spec in expected.items():
                actual = inspected[name]
                if (actual.dtype, actual.shape) != (
                    expected_spec.dtype,
                    expected_spec.shape,
                ):
                    raise ValueError(
                        f"checkpoint tensor {name} expected "
                        f"{expected_spec.dtype}{expected_spec.shape}, got "
                        f"{actual.dtype}{actual.shape}"
                    )
            tensor_bytes = sum(spec.nbytes for spec in inspected.values())
            if self.total_size and tensor_bytes != self.total_size:
                raise ValueError(
                    "checkpoint tensor bytes do not match metadata.total_size: "
                    f"{tensor_bytes} != {self.total_size}"
                )

        mtp_count = sum(name.startswith("mtp.") for name in actual_names)
        return DeepSeekV4CheckpointAudit(
            tensor_count=len(actual_names),
            inference_tensor_count=len(expected),
            mtp_tensor_count=mtp_count,
            shard_count=len(self.shard_names),
            tensor_bytes=tensor_bytes,
            files_checked=require_files,
        )

    def load_tensors(
        self,
        names: Iterable[str],
        *,
        device: str = "cpu",
    ) -> dict[str, object]:
        """Load named raw tensors, opening each owning shard only once."""
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in names:
            try:
                grouped[self.weight_map[name]].append(name)
            except KeyError:
                raise KeyError(f"checkpoint tensor {name!r} is not indexed") from None

        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("loading checkpoint data requires safetensors") from error

        result: dict[str, object] = {}
        for filename, shard_names in grouped.items():
            with safe_open(
                str(self.root / filename), framework="pt", device="cpu"
            ) as shard:
                for name in shard_names:
                    tensor = shard.get_tensor(name)
                    result[name] = tensor.clone() if device == "cpu" else tensor.to(device)
        return result

    def load_tensor_slice(
        self,
        name: str,
        index,
        *,
        device: str = "cpu",
    ):
        """Load a basic-indexed tensor slice without materializing the full tensor."""
        filename = self.weight_map.get(name)
        if filename is None:
            raise KeyError(f"checkpoint tensor {name!r} is not indexed")
        try:
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError("loading checkpoint data requires safetensors") from error
        with safe_open(
            str(self.root / filename), framework="pt", device="cpu"
        ) as shard:
            tensor = shard.get_slice(name)[index]
            tensor = tensor.clone()
        return tensor if device == "cpu" else tensor.to(device)

    def load_fp8_linear(
        self,
        prefix: str,
        *,
        device: str = "cpu",
    ) -> Fp8LinearCheckpointTensors:
        """Load one raw E4M3/UE8M0 checkpoint linear by prefix."""
        names = (f"{prefix}.weight", f"{prefix}.scale")
        specs = self.inspect(names)
        weight_spec, scale_spec = (specs[name] for name in names)
        if weight_spec.dtype != "F8_E4M3" or len(weight_spec.shape) != 2:
            raise ValueError(f"{prefix} is not a rank-2 checkpoint FP8 linear")
        rows, k = weight_spec.shape
        expected_scale = ((rows + 127) // 128, k // 128)
        if k % 128 or scale_spec.dtype != "F8_E8M0" or scale_spec.shape != expected_scale:
            raise ValueError(
                f"{prefix} FP8 scale must be F8_E8M0{expected_scale}, got "
                f"{scale_spec.dtype}{scale_spec.shape}"
            )
        tensors = self.load_tensors(names, device=device)
        return Fp8LinearCheckpointTensors(
            prefix=prefix,
            weight=tensors[names[0]],
            scale=tensors[names[1]],
        )

    def load_nvfp4_linear(
        self,
        prefix: str,
        *,
        device: str = "cpu",
    ) -> Nvfp4LinearCheckpointTensors:
        """Load one raw ModelOpt NVFP4 checkpoint linear by prefix."""
        names = (
            f"{prefix}.weight",
            f"{prefix}.weight_scale",
            f"{prefix}.weight_scale_2",
            f"{prefix}.input_scale",
        )
        specs = self.inspect(names)
        weight_spec, scale_spec, scale2_spec, input_scale_spec = (
            specs[name] for name in names
        )
        if weight_spec.dtype != "U8" or len(weight_spec.shape) != 2:
            raise ValueError(f"{prefix} is not a rank-2 checkpoint NVFP4 linear")
        rows, packed_k = weight_spec.shape
        expected_scale = (rows, packed_k // 8)
        if (
            packed_k % 16
            or scale_spec.dtype != "F8_E4M3"
            or scale_spec.shape != expected_scale
        ):
            raise ValueError(
                f"{prefix} NVFP4 scale must be F8_E4M3{expected_scale}, got "
                f"{scale_spec.dtype}{scale_spec.shape}"
            )
        for name, spec in zip(names[2:], (scale2_spec, input_scale_spec)):
            if spec.dtype != "F32" or spec.shape != ():
                raise ValueError(f"{name} must be a scalar F32 tensor")
        tensors = self.load_tensors(names, device=device)
        return Nvfp4LinearCheckpointTensors(
            prefix=prefix,
            weight=tensors[names[0]],
            weight_scale=tensors[names[1]],
            weight_scale_2=tensors[names[2]],
            input_scale=tensors[names[3]],
        )


class DeepSeekV4ResidentCheckpoint:
    """Read-only shard-packed device views of a raw inference checkpoint."""

    def __init__(
        self,
        checkpoint: DeepSeekV4Checkpoint,
        tensors: dict[str, object],
        storage_buffers: list[object],
        *,
        device,
        tensor_bytes: int,
        storage_bytes: int,
    ) -> None:
        self.source = checkpoint
        self.config = checkpoint.config
        self.root = checkpoint.root
        self.weight_map = checkpoint.weight_map
        self._tensors = tensors
        self._storage_buffers = storage_buffers
        self.device = device
        self.tensor_bytes = tensor_bytes
        self.storage_bytes = storage_bytes

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: DeepSeekV4Checkpoint,
        *,
        device,
        names: Iterable[str] | None = None,
        alignment: int = 256,
        reserve_bytes: int = 0,
        progress: Callable[[int, int, str, int, int], None] | None = None,
    ) -> "DeepSeekV4ResidentCheckpoint":
        """Pack selected tensors into aligned per-shard device storage.

        When ``names`` is omitted, all non-MTP inference tensors are loaded and
        validated against the model contract.  Returned tensor views are
        read-only by convention so repeated decode steps share one resident
        copy of every checkpoint weight.
        """
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError("resident checkpoint alignment must be a power of two")
        if reserve_bytes < 0:
            raise ValueError("resident checkpoint reserve must be non-negative")

        try:
            import torch
            from safetensors import safe_open
        except ImportError as error:
            raise RuntimeError(
                "loading a resident checkpoint requires torch and safetensors"
            ) from error

        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())

        expected = None
        if names is None:
            expected = expected_inference_tensor_specs(checkpoint.config)
            selected_names = tuple(expected)
        else:
            selected_names = tuple(names)
        if not selected_names:
            raise ValueError("resident checkpoint requires at least one tensor")
        if len(set(selected_names)) != len(selected_names):
            raise ValueError("resident checkpoint tensor names must be unique")

        specs = checkpoint.inspect(selected_names)
        if expected is not None:
            for name, expected_spec in expected.items():
                actual = specs[name]
                if (actual.dtype, actual.shape) != (
                    expected_spec.dtype,
                    expected_spec.shape,
                ):
                    raise ValueError(
                        f"checkpoint tensor {name} expected "
                        f"{expected_spec.dtype}{expected_spec.shape}, got "
                        f"{actual.dtype}{actual.shape}"
                    )

        grouped: dict[str, list[str]] = defaultdict(list)
        for name in selected_names:
            grouped[checkpoint.weight_map[name]].append(name)

        shard_layouts = []
        total_tensor_bytes = 0
        total_storage_bytes = 0
        for filename in sorted(grouped):
            offset = 0
            entries = []
            for name in grouped[filename]:
                offset = (offset + alignment - 1) & -alignment
                spec = specs[name]
                entries.append((name, offset, spec))
                offset += spec.nbytes
                total_tensor_bytes += spec.nbytes
            shard_layouts.append((filename, offset, entries))
            total_storage_bytes += offset

        if target_device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(target_device)
            if total_storage_bytes + reserve_bytes > free_bytes:
                raise MemoryError(
                    "resident checkpoint plus requested reserve exceeds free GPU "
                    f"memory: storage={total_storage_bytes} reserve={reserve_bytes} "
                    f"free={free_bytes}"
                )

        resident_tensors: dict[str, object] = {}
        storage_buffers: list[object] = []
        loaded_storage_bytes = 0
        for shard_index, (filename, shard_bytes, entries) in enumerate(
            shard_layouts, start=1
        ):
            packed = torch.empty((shard_bytes,), dtype=torch.uint8)
            dtypes = {}
            previous_end = 0
            with safe_open(
                str(checkpoint.root / filename), framework="pt", device="cpu"
            ) as shard:
                for name, offset, spec in entries:
                    if offset > previous_end:
                        packed[previous_end:offset].zero_()
                    source = shard.get_tensor(name)
                    source_bytes = (
                        source.contiguous().reshape(-1).view(torch.uint8)
                    )
                    if source_bytes.numel() != spec.nbytes:
                        raise ValueError(
                            f"checkpoint tensor {name} payload changed while loading"
                        )
                    packed[offset : offset + spec.nbytes].copy_(source_bytes)
                    dtypes[name] = source.dtype
                    previous_end = offset + spec.nbytes

            storage = packed.to(target_device)
            storage_buffers.append(storage)
            for name, offset, spec in entries:
                raw = storage[offset : offset + spec.nbytes]
                resident_tensors[name] = raw.view(dtypes[name]).reshape(spec.shape)
            loaded_storage_bytes += shard_bytes
            if progress is not None:
                progress(
                    shard_index,
                    len(shard_layouts),
                    filename,
                    shard_bytes,
                    loaded_storage_bytes,
                )

        return cls(
            checkpoint,
            resident_tensors,
            storage_buffers,
            device=target_device,
            tensor_bytes=total_tensor_bytes,
            storage_bytes=total_storage_bytes,
        )

    def _check_device(self, device) -> None:
        import torch

        requested = torch.device(device)
        if requested.type != self.device.type:
            raise ValueError(
                f"resident checkpoint is on {self.device}, not {requested}"
            )
        if requested.index is not None and requested.index != self.device.index:
            raise ValueError(
                f"resident checkpoint is on {self.device}, not {requested}"
            )

    def load_tensors(
        self,
        names: Iterable[str],
        *,
        device="cpu",
    ) -> dict[str, object]:
        """Return read-only resident tensor views without copying."""
        self._check_device(device)
        result = {}
        for name in names:
            try:
                result[name] = self._tensors[name]
            except KeyError:
                raise KeyError(
                    f"checkpoint tensor {name!r} is not resident"
                ) from None
        return result

    def load_tensor_slice(self, name: str, index, *, device="cpu"):
        """Return a basic-indexed view of one resident tensor."""
        return self.load_tensors([name], device=device)[name][index]

    def load_fp8_linear(
        self,
        prefix: str,
        *,
        device="cpu",
    ) -> Fp8LinearCheckpointTensors:
        names = (f"{prefix}.weight", f"{prefix}.scale")
        tensors = self.load_tensors(names, device=device)
        return Fp8LinearCheckpointTensors(
            prefix=prefix,
            weight=tensors[names[0]],
            scale=tensors[names[1]],
        )

    def load_nvfp4_linear(
        self,
        prefix: str,
        *,
        device="cpu",
    ) -> Nvfp4LinearCheckpointTensors:
        names = (
            f"{prefix}.weight",
            f"{prefix}.weight_scale",
            f"{prefix}.weight_scale_2",
            f"{prefix}.input_scale",
        )
        tensors = self.load_tensors(names, device=device)
        return Nvfp4LinearCheckpointTensors(
            prefix=prefix,
            weight=tensors[names[0]],
            weight_scale=tensors[names[1]],
            weight_scale_2=tensors[names[2]],
            input_scale=tensors[names[3]],
        )


__all__ = [
    "INDEX_FILENAME",
    "ExpectedTensorSpec",
    "SafeTensorSpec",
    "DeepSeekV4CheckpointAudit",
    "Fp8LinearCheckpointTensors",
    "Nvfp4LinearCheckpointTensors",
    "DeepSeekV4Checkpoint",
    "DeepSeekV4ResidentCheckpoint",
    "expected_inference_tensor_specs",
    "read_safetensors_header",
    "read_safetensors_header_url",
]
