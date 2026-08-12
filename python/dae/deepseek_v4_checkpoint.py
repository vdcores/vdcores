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
from .expert_affine import AffineExpertArena


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


@dataclass(frozen=True)
class NativeNvfp4LinearCheckpointTensors:
    """Combined SM100 data/scale tiles for one resident routed linear."""

    prefix: str
    weight_tiles: object
    weight_scale_2: object
    input_scale: object

    @property
    def alpha(self):
        return (self.weight_scale_2 * self.input_scale).reshape(1)


@dataclass(frozen=True)
class NativeFp8LinearCheckpointTensors:
    """Combined SM100 MXF8 weight tiles for one resident FP8 linear."""

    prefix: str
    weight_tiles: object


def _native_nvfp4_name(prefix: str) -> str:
    return f"{prefix}.__vdcores_native_weight"


def _native_fp8_name(prefix: str) -> str:
    return f"{prefix}.__vdcores_native_fp8_weight"


_AFFINE_EXPERT_PREFIX = re.compile(
    r"^layers\.(?P<layer>[0-9]+)\.ffn\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<tag>w1|w3|w2)$"
)


def _affine_expert_key(prefix: str):
    match = _AFFINE_EXPERT_PREFIX.fullmatch(prefix)
    if match is None:
        return None
    return (
        int(match.group("layer")),
        int(match.group("expert")),
        match.group("tag"),
    )


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
        native_nvfp4_prefixes: Iterable[str] = (),
        native_fp8_prefixes: Iterable[str] = (),
        affine_expert_arena: AffineExpertArena | None = None,
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
        self.native_nvfp4_prefixes = frozenset(native_nvfp4_prefixes)
        self.native_fp8_prefixes = frozenset(native_fp8_prefixes)
        self.affine_expert_arena = affine_expert_arena

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: DeepSeekV4Checkpoint,
        *,
        device,
        names: Iterable[str] | None = None,
        alignment: int = 256,
        reserve_bytes: int = 0,
        native_nvfp4: bool = False,
        affine_nvfp4_experts: bool = False,
        native_fp8_prefixes: Iterable[str] = (),
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
        requested_native_fp8 = frozenset(native_fp8_prefixes)
        if affine_nvfp4_experts and not native_nvfp4:
            raise ValueError("affine NVFP4 experts require native NVFP4 packing")
        if (native_nvfp4 or requested_native_fp8) and target_device.type != "cuda":
            raise ValueError("native resident packing requires a CUDA device")

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

        selected_name_set = set(selected_names)
        native_pairs: dict[str, tuple[str, str, tuple[int, int, int]]] = {}
        if native_nvfp4:
            for weight_name in selected_names:
                if not weight_name.endswith(".weight"):
                    continue
                prefix = weight_name[: -len(".weight")]
                scale_name = f"{prefix}.weight_scale"
                if scale_name not in selected_name_set:
                    continue
                weight_spec = specs[weight_name]
                scale_spec = specs[scale_name]
                if weight_spec.dtype != "U8" or len(weight_spec.shape) != 2:
                    continue
                rows, packed_k = weight_spec.shape
                if (
                    rows % 128
                    or packed_k % 128
                    or scale_spec.dtype != "F8_E4M3"
                    or scale_spec.shape != (rows, packed_k // 8)
                ):
                    raise ValueError(
                        f"{prefix} cannot be converted to native M128/K256 tiles"
                    )
                if checkpoint.weight_map[weight_name] != checkpoint.weight_map[scale_name]:
                    raise ValueError(
                        f"{prefix} weight and scale must share one checkpoint shard"
                    )
                native_pairs[prefix] = (
                    weight_name,
                    scale_name,
                    (rows // 128, packed_k // 128, 18432),
                )

        native_fp8_pairs: dict[
            str, tuple[str, str, tuple[int, int, int]]
        ] = {}
        for prefix in requested_native_fp8:
            weight_name = f"{prefix}.weight"
            scale_name = f"{prefix}.scale"
            if weight_name not in selected_name_set or scale_name not in selected_name_set:
                raise ValueError(
                    f"native FP8 prefix {prefix} is not fully resident"
                )
            weight_spec = specs[weight_name]
            scale_spec = specs[scale_name]
            if weight_spec.dtype != "F8_E4M3" or len(weight_spec.shape) != 2:
                raise ValueError(f"{prefix} is not a rank-2 E4M3 linear")
            rows, k = weight_spec.shape
            if (
                rows % 128
                or k % 128
                or scale_spec.dtype != "F8_E8M0"
                or scale_spec.shape != (rows // 128, k // 128)
            ):
                raise ValueError(
                    f"{prefix} cannot be converted to native M128/K128 tiles"
                )
            if checkpoint.weight_map[weight_name] != checkpoint.weight_map[scale_name]:
                raise ValueError(
                    f"{prefix} weight and scale must share one checkpoint shard"
                )
            native_fp8_pairs[prefix] = (
                weight_name,
                scale_name,
                (rows // 128, k // 128, 16896),
            )

        affine_pairs = {}
        affine_source_names = set()
        affine_layer_filenames: dict[int, str] = {}
        affine_layer_ids = ()
        if affine_nvfp4_experts:
            for prefix, pair in native_pairs.items():
                key = _affine_expert_key(prefix)
                if key is None:
                    continue
                layer_id, expert_id, tag = key
                if not 0 <= expert_id < checkpoint.config.num_experts:
                    raise ValueError(f"affine expert ID is out of range in {prefix}")
                scale2_name = f"{prefix}.weight_scale_2"
                input_name = f"{prefix}.input_scale"
                if scale2_name not in selected_name_set or input_name not in selected_name_set:
                    raise ValueError(f"affine expert {prefix} is missing scalar metadata")
                source_names = (pair[0], pair[1], scale2_name, input_name)
                filenames = {checkpoint.weight_map[name] for name in source_names}
                if len(filenames) != 1:
                    raise ValueError(
                        f"affine expert {prefix} must occupy one checkpoint shard"
                    )
                filename = filenames.pop()
                prior_filename = affine_layer_filenames.setdefault(layer_id, filename)
                if prior_filename != filename:
                    raise ValueError(
                        f"affine layer {layer_id} spans multiple checkpoint shards"
                    )
                affine_pairs[(layer_id, expert_id, tag)] = (prefix, *pair)
                affine_source_names.update(source_names)

            affine_layer_ids = tuple(sorted(affine_layer_filenames))
            expected_keys = {
                (layer_id, expert_id, tag)
                for layer_id in affine_layer_ids
                for expert_id in range(checkpoint.config.num_experts)
                for tag in AffineExpertArena.TAGS
            }
            if set(affine_pairs) != expected_keys:
                missing = expected_keys - set(affine_pairs)
                extra = set(affine_pairs) - expected_keys
                raise ValueError(
                    "affine expert arena requires complete layer/expert/tag sets: "
                    f"missing={len(missing)} extra={len(extra)}"
                )
            if not affine_layer_ids:
                raise ValueError("affine expert packing found no routed experts")

        ordinary_native_pairs = {
            prefix: pair
            for prefix, pair in native_pairs.items()
            if _affine_expert_key(prefix) is None or not affine_nvfp4_experts
        }
        grouped: dict[str, list[str]] = defaultdict(list)
        for name in selected_names:
            if name not in affine_source_names:
                grouped[checkpoint.weight_map[name]].append(name)

        shard_layouts = {}
        total_tensor_bytes = sum(spec.nbytes for spec in specs.values())
        total_storage_bytes = (
            AffineExpertArena.storage_bytes(
                checkpoint.config, len(affine_layer_ids)
            )
            if affine_layer_ids
            else 0
        )
        for filename in sorted(grouped):
            offset = 0
            entries = []
            weight_to_pair = {
                weight_name: (scale_name, shape)
                for weight_name, scale_name, shape in ordinary_native_pairs.values()
                if checkpoint.weight_map[weight_name] == filename
            }
            fp8_weight_to_pair = {
                weight_name: (scale_name, shape)
                for weight_name, scale_name, shape in native_fp8_pairs.values()
                if checkpoint.weight_map[weight_name] == filename
            }
            paired_scales = {
                scale_name
                for scale_name, _ in (
                    *weight_to_pair.values(),
                    *fp8_weight_to_pair.values(),
                )
            }
            ordered_names = []
            emitted = set()
            for name in grouped[filename]:
                if name in emitted or name in paired_scales:
                    continue
                ordered_names.append(name)
                emitted.add(name)
                if name in weight_to_pair:
                    scale_name, _ = weight_to_pair[name]
                    ordered_names.append(scale_name)
                    emitted.add(scale_name)
                elif name in fp8_weight_to_pair:
                    scale_name, _ = fp8_weight_to_pair[name]
                    ordered_names.append(scale_name)
                    emitted.add(scale_name)
            index = 0
            while index < len(ordered_names):
                name = ordered_names[index]
                offset = (offset + alignment - 1) & -alignment
                spec = specs[name]
                entries.append((name, offset, spec))
                if name in fp8_weight_to_pair:
                    scale_name, native_shape = fp8_weight_to_pair[name]
                    scale_spec = specs[scale_name]
                    scale_offset = offset + spec.nbytes
                    entries.append((scale_name, scale_offset, scale_spec))
                    native_bytes = math.prod(native_shape)
                    if spec.nbytes + scale_spec.nbytes > native_bytes:
                        raise ValueError(
                            f"{name} raw FP8 span exceeds native storage"
                        )
                    offset += native_bytes
                    index += 2
                    continue
                offset += spec.nbytes
                index += 1
            shard_layouts[filename] = (offset, entries)
            total_storage_bytes += offset

        if target_device.type == "cuda":
            free_bytes, _ = torch.cuda.mem_get_info(target_device)
            if total_storage_bytes + reserve_bytes > free_bytes:
                raise MemoryError(
                    "resident checkpoint plus requested reserve exceeds free GPU "
                    f"memory: storage={total_storage_bytes} reserve={reserve_bytes} "
                    f"free={free_bytes}"
                )

        affine_layers_by_filename: dict[str, list[int]] = defaultdict(list)
        for layer_id, filename in affine_layer_filenames.items():
            affine_layers_by_filename[filename].append(layer_id)
        all_filenames = tuple(
            sorted(set(shard_layouts) | set(affine_layers_by_filename))
        )

        resident_tensors: dict[str, object] = {}
        affine_arena = (
            AffineExpertArena.allocate(
                checkpoint.config, affine_layer_ids, device=target_device
            )
            if affine_layer_ids
            else None
        )
        storage_buffers: list[object] = (
            [affine_arena.storage] if affine_arena is not None else []
        )
        loaded_storage_bytes = (
            AffineExpertArena.HEADER_BYTES if affine_arena is not None else 0
        )
        for shard_index, filename in enumerate(all_filenames, start=1):
            shard_bytes, entries = shard_layouts.get(filename, (0, ()))
            packed = (
                torch.empty((shard_bytes,), dtype=torch.uint8)
                if shard_bytes
                else None
            )
            affine_shard_layers = tuple(sorted(affine_layers_by_filename[filename]))
            if len(affine_shard_layers) > 1:
                raise ValueError(
                    f"checkpoint shard {filename} contains multiple affine expert layers"
                )
            affine_layer_id = (
                affine_shard_layers[0] if affine_shard_layers else None
            )
            affine_packed = (
                torch.empty((affine_arena.layer_stride,), dtype=torch.uint8)
                if affine_layer_id is not None
                else None
            )
            dtypes = {}
            previous_end = 0
            with safe_open(
                str(checkpoint.root / filename), framework="pt", device="cpu"
            ) as shard:
                for name, offset, spec in entries:
                    if offset > previous_end:
                        packed[previous_end:offset].zero_()
                    source = shard.get_tensor(name)
                    source_bytes = source.contiguous().reshape(-1).view(torch.uint8)
                    if source_bytes.numel() != spec.nbytes:
                        raise ValueError(
                            f"checkpoint tensor {name} payload changed while loading"
                        )
                    packed[offset : offset + spec.nbytes].copy_(source_bytes)
                    dtypes[name] = source.dtype
                    previous_end = offset + spec.nbytes

                if affine_layer_id is not None:
                    layer_base = affine_arena.layer_offset(affine_layer_id)
                    for expert_id in range(checkpoint.config.num_experts):
                        up_input = None
                        for tag in AffineExpertArena.TAGS:
                            prefix, weight_name, scale_name, native_shape = affine_pairs[
                                (affine_layer_id, expert_id, tag)
                            ]
                            weight_spec = specs[weight_name]
                            scale_spec = specs[scale_name]
                            weight = shard.get_tensor(weight_name).contiguous()
                            scale = shard.get_tensor(scale_name).contiguous()
                            weight_bytes = weight.reshape(-1).view(torch.uint8)
                            scale_bytes = scale.reshape(-1).view(torch.uint8)
                            if (
                                weight_bytes.numel() != weight_spec.nbytes
                                or scale_bytes.numel() != scale_spec.nbytes
                                or weight_spec.nbytes + scale_spec.nbytes
                                != affine_arena.weight_bytes
                            ):
                                raise ValueError(
                                    f"{prefix} does not fill one affine native field"
                                )
                            field_offset = (
                                affine_arena.weight_offset(
                                    affine_layer_id, expert_id, tag
                                )
                                - layer_base
                            )
                            affine_packed[
                                field_offset : field_offset + weight_spec.nbytes
                            ].copy_(weight_bytes)
                            affine_packed[
                                field_offset
                                + weight_spec.nbytes : field_offset
                                + affine_arena.weight_bytes
                            ].copy_(scale_bytes)

                            scale2_name = f"{prefix}.weight_scale_2"
                            input_name = f"{prefix}.input_scale"
                            scale2 = shard.get_tensor(scale2_name).reshape(())
                            input_scale = shard.get_tensor(input_name).reshape(())
                            if tag == "w1":
                                up_input = input_scale
                            elif tag == "w3" and not torch.equal(up_input, input_scale):
                                raise ValueError(
                                    f"layer {affine_layer_id} expert {expert_id} "
                                    "w1/w3 input scales differ"
                                )
                            metadata_values = {
                                "weight_scale_2": scale2,
                                "input_scale": input_scale,
                                "alpha": scale2 * input_scale,
                            }
                            for field, value in metadata_values.items():
                                metadata_offset = (
                                    affine_arena.metadata_offset(
                                        affine_layer_id, expert_id, tag, field
                                    )
                                    - layer_base
                                )
                                metadata = affine_packed[
                                    metadata_offset : metadata_offset
                                    + AffineExpertArena.META_FIELD_BYTES
                                ]
                                metadata.zero_()
                                metadata[:4].copy_(
                                    value.to(torch.float32)
                                    .contiguous()
                                    .reshape(-1)
                                    .view(torch.uint8)
                                )

            storage = packed.to(target_device) if packed is not None else None
            if storage is not None:
                storage_buffers.append(storage)
            entry_by_name = {
                name: (offset, spec) for name, offset, spec in entries
            }
            native_source_names = {
                name
                for weight_name, scale_name, _ in (
                    *ordinary_native_pairs.values(),
                    *native_fp8_pairs.values(),
                )
                for name in (weight_name, scale_name)
            }
            for name, offset, spec in entries:
                if name in native_source_names:
                    continue
                raw = storage[offset : offset + spec.nbytes]
                resident_tensors[name] = raw.view(dtypes[name]).reshape(spec.shape)

            if ordinary_native_pairs or native_fp8_pairs or affine_layer_id is not None:
                from . import runtime

            temporaries = {}
            for prefix, (
                weight_name,
                scale_name,
                native_shape,
            ) in ordinary_native_pairs.items():
                if checkpoint.weight_map[weight_name] != filename:
                    continue
                weight_offset, weight_spec = entry_by_name[weight_name]
                scale_offset, scale_spec = entry_by_name[scale_name]
                if scale_offset != weight_offset + weight_spec.nbytes:
                    raise ValueError(
                        f"{prefix} resident weight/scale span is not contiguous"
                    )
                weight = storage[
                    weight_offset : weight_offset + weight_spec.nbytes
                ].view(dtypes[weight_name]).reshape(weight_spec.shape)
                checkpoint_scale = storage[
                    scale_offset : scale_offset + scale_spec.nbytes
                ].view(dtypes[scale_name]).reshape(scale_spec.shape)
                temporary = temporaries.get(native_shape)
                if temporary is None:
                    temporary = torch.empty(
                        native_shape, dtype=torch.uint8, device=target_device
                    )
                    temporaries[native_shape] = temporary
                runtime.prepack_nvfp4_checkpoint(
                    weight, checkpoint_scale, temporary
                )
                native_bytes = weight_spec.nbytes + scale_spec.nbytes
                native = storage[
                    weight_offset : weight_offset + native_bytes
                ].reshape(native_shape)
                native.copy_(temporary)
                resident_tensors[_native_nvfp4_name(prefix)] = native

            fp8_temporaries = {}
            for prefix, (
                weight_name,
                scale_name,
                native_shape,
            ) in native_fp8_pairs.items():
                if checkpoint.weight_map[weight_name] != filename:
                    continue
                weight_offset, weight_spec = entry_by_name[weight_name]
                scale_offset, scale_spec = entry_by_name[scale_name]
                if scale_offset != weight_offset + weight_spec.nbytes:
                    raise ValueError(
                        f"{prefix} resident FP8 weight/scale span is not contiguous"
                    )
                weight = storage[
                    weight_offset : weight_offset + weight_spec.nbytes
                ].view(dtypes[weight_name]).reshape(weight_spec.shape)
                checkpoint_scale = storage[
                    scale_offset : scale_offset + scale_spec.nbytes
                ].view(dtypes[scale_name]).reshape(scale_spec.shape)
                temporary = fp8_temporaries.get(native_shape)
                if temporary is None:
                    temporary = torch.empty(
                        native_shape, dtype=torch.uint8, device=target_device
                    )
                    fp8_temporaries[native_shape] = temporary
                runtime.prepack_fp8_checkpoint(
                    weight, checkpoint_scale, temporary
                )
                native_bytes = math.prod(native_shape)
                native = storage[
                    weight_offset : weight_offset + native_bytes
                ].reshape(native_shape)
                native.copy_(temporary)
                resident_tensors[_native_fp8_name(prefix)] = native

            if affine_layer_id is not None:
                affine_arena.layer_storage(affine_layer_id).copy_(affine_packed)
                affine_temporaries = {}
                for expert_id in range(checkpoint.config.num_experts):
                    for tag in AffineExpertArena.TAGS:
                        prefix, weight_name, scale_name, native_shape = affine_pairs[
                            (affine_layer_id, expert_id, tag)
                        ]
                        weight_spec = specs[weight_name]
                        scale_spec = specs[scale_name]
                        field_offset = affine_arena.weight_offset(
                            affine_layer_id, expert_id, tag
                        )
                        field = affine_arena.storage[
                            field_offset : field_offset + affine_arena.weight_bytes
                        ]
                        weight = field[: weight_spec.nbytes].reshape(weight_spec.shape)
                        checkpoint_scale = field[
                            weight_spec.nbytes : weight_spec.nbytes + scale_spec.nbytes
                        ].view(torch.float8_e4m3fn).reshape(scale_spec.shape)
                        temporary = affine_temporaries.get(native_shape)
                        if temporary is None:
                            temporary = torch.empty(
                                native_shape, dtype=torch.uint8, device=target_device
                            )
                            affine_temporaries[native_shape] = temporary
                        runtime.prepack_nvfp4_checkpoint(
                            weight, checkpoint_scale, temporary
                        )
                        native = field.reshape(native_shape)
                        native.copy_(temporary)
                        resident_tensors[_native_nvfp4_name(prefix)] = native
                        for field_name, tensor_name in (
                            ("weight_scale_2", f"{prefix}.weight_scale_2"),
                            ("input_scale", f"{prefix}.input_scale"),
                        ):
                            metadata_offset = affine_arena.metadata_offset(
                                affine_layer_id, expert_id, tag, field_name
                            )
                            resident_tensors[tensor_name] = affine_arena.storage[
                                metadata_offset : metadata_offset + 4
                            ].view(torch.float32).reshape(())

            effective_shard_bytes = shard_bytes + (
                affine_arena.layer_stride if affine_layer_id is not None else 0
            )
            loaded_storage_bytes += effective_shard_bytes
            if progress is not None:
                progress(
                    shard_index,
                    len(all_filenames),
                    filename,
                    effective_shard_bytes,
                    loaded_storage_bytes,
                )

        return cls(
            checkpoint,
            resident_tensors,
            storage_buffers,
            device=target_device,
            tensor_bytes=total_tensor_bytes,
            storage_bytes=total_storage_bytes,
            native_nvfp4_prefixes=native_pairs,
            native_fp8_prefixes=native_fp8_pairs,
            affine_expert_arena=affine_arena,
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

    def load_native_nvfp4_linear(
        self,
        prefix: str,
        *,
        device="cpu",
    ) -> NativeNvfp4LinearCheckpointTensors:
        names = (
            _native_nvfp4_name(prefix),
            f"{prefix}.weight_scale_2",
            f"{prefix}.input_scale",
        )
        tensors = self.load_tensors(names, device=device)
        return NativeNvfp4LinearCheckpointTensors(
            prefix=prefix,
            weight_tiles=tensors[names[0]],
            weight_scale_2=tensors[names[1]],
            input_scale=tensors[names[2]],
        )

    def load_native_fp8_linear(
        self,
        prefix: str,
        *,
        device="cpu",
    ) -> NativeFp8LinearCheckpointTensors:
        name = _native_fp8_name(prefix)
        tensor = self.load_tensors((name,), device=device)[name]
        return NativeFp8LinearCheckpointTensors(
            prefix=prefix,
            weight_tiles=tensor,
        )


__all__ = [
    "INDEX_FILENAME",
    "ExpectedTensorSpec",
    "SafeTensorSpec",
    "DeepSeekV4CheckpointAudit",
    "Fp8LinearCheckpointTensors",
    "Nvfp4LinearCheckpointTensors",
    "NativeNvfp4LinearCheckpointTensors",
    "NativeFp8LinearCheckpointTensors",
    "DeepSeekV4Checkpoint",
    "DeepSeekV4ResidentCheckpoint",
    "expected_inference_tensor_specs",
    "read_safetensors_header",
    "read_safetensors_header_url",
]
