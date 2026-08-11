import copy as pycopy
import warnings

from .runtime import *
from .launcher import *
from .routing import IndexedLoadTable

class Schedule:
    def __init__(self):
        self.num_sms = None
        self.base_sm = 0
        self._bars = {}

    def _clone(self):
        clone = pycopy.copy(self)
        clone._bars = self._bars.copy()
        return clone

    def _on_place(self):
        pass

    def place(self, num_sms: int, base_sm: int = 0):
        clone = self._clone()
        clone.num_sms = num_sms
        clone.base_sm = base_sm
        clone._on_place()
        return clone

    def bar(self, role: str, bar_id: int):
        self._bars[role] = bar_id
        return self

    def _bar(self, role: str):
        return self._bars.get(role)

    def _require_placed(self):
        if self.num_sms is None:
            raise ValueError(f"{self.__class__.__name__} must be placed before querying barrier release counts")

    def _bar_release_if_present(self, role: str, count: int):
        if self._bar(role) is None:
            return 0
        self._require_placed()
        return count

    def bar_release_count(self, role: str):
        return 0

    def collect_barrier_release_counts(self):
        counts = {}
        for role, bar_id in self._bars.items():
            count = self.bar_release_count(role)
            if count <= 0:
                continue
            counts[bar_id] = counts.get(bar_id, 0) + count
        return counts

    def map_sm(self, sm: int):
        # This function decides how to map SM to SM ID. this can create scheudle
        # that is not strictly round-robin, e.g., for hierarchical scheduling,
        # we may want to map SMs in the same fold together.
        if self.num_sms is None:
            raise ValueError(f"{self.__class__.__name__} must be placed before scheduling")
        sm -= self.base_sm
        if sm < 0 or sm >= self.num_sms:
            return -1
        return sm
    def schedule(self, sm: int):
        raise NotImplementedError("Schedule.schedule() must be implemented by subclass")
    def __call__(self, sm: int):
        mapped_sm = self.map_sm(sm)
        return self.schedule(mapped_sm)


class LayeredSchedule(Schedule):
    """Select layer-specific 1D operands from HBM without unrolling a task.

    The wrapped schedule is rendered once from the first tensor in every
    mapping. Direct LDU addresses that fall inside those representative
    tensors are replaced by one device pointer column. Compact indirect-load
    opcodes let the allocator select that column entry with its linear layer
    index; the selected source is resolved only by LDU.
    """

    _ADDRESS_ONLY_OPS = {
        opcode.OP_ALLOC_ROUTED_TMA_LOAD_1D & ~((1 << 6) - 1),
        opcode.OP_ALLOC_ROUTED_TMA_LOAD_BASE_1D & ~((1 << 6) - 1),
        opcode.OP_ALLOC_INDEXED_TMA_LOAD_1D & ~((1 << 6) - 1),
    }
    _SUPPORTED_OPS = {
        opcode.OP_ALLOC_TMA_LOAD_1D & ~((1 << 6) - 1),
        opcode.OP_ALLOC_LDU_LOAD_1D & ~((1 << 6) - 1),
        opcode.OP_ALLOC_ROUTED_TMA_LOAD_1D & ~((1 << 6) - 1),
        opcode.OP_ALLOC_ROUTED_TMA_LOAD_BASE_1D & ~((1 << 6) - 1),
        opcode.OP_ALLOC_INDEXED_TMA_LOAD_1D & ~((1 << 6) - 1),
    }

    def __init__(
        self,
        schedule,
        tensor_groups,
        *,
        counter_strides=(),
        route_indices=None,
    ):
        super().__init__()
        self.inner = schedule
        self.tensor_groups = tuple(
            (representative, tuple(alternatives))
            for representative, alternatives in tensor_groups
        )
        self.counter_strides = tuple(
            (int(counter), int(stride))
            for counter, stride in counter_strides
        )
        self.route_indices = route_indices

    def _on_place(self):
        if not self.tensor_groups:
            raise ValueError("layered schedule requires at least one tensor group")
        group_lengths = {len(alternatives) for _, alternatives in self.tensor_groups}
        if len(group_lengths) != 1 or next(iter(group_lengths)) <= 0:
            raise ValueError("layered tensor groups must have one common nonzero length")
        self.layer_count = next(iter(group_lengths))
        for counter, stride in self.counter_strides:
            if not 0 <= counter < 32 or stride <= 0:
                raise ValueError("layered counter strides require reg [0,31] and positive stride")
        for representative, alternatives in self.tensor_groups:
            if not isinstance(representative, torch.Tensor):
                raise ValueError("layered representatives must be tensors")
            if representative.device.type != "cuda" or not representative.is_contiguous():
                raise ValueError("layered tensor groups must be contiguous CUDA tensors")
            for alternative in alternatives:
                if not isinstance(alternative, torch.Tensor):
                    raise ValueError("layered alternatives must be tensors")
                if (
                    alternative.device != representative.device
                    or alternative.dtype != representative.dtype
                    or alternative.shape != representative.shape
                    or not alternative.is_contiguous()
                ):
                    raise ValueError(
                        "layered alternatives must match representative device/dtype/shape"
                    )
        if self.route_indices is not None and (
            not isinstance(self.route_indices, torch.Tensor)
            or self.route_indices.device.type != "cuda"
            or self.route_indices.dtype != torch.int32
            or self.route_indices.numel() < 6
            or not self.route_indices.is_contiguous()
        ):
            raise ValueError("layered route indices must be contiguous CUDA int32")
        inner = self.inner._clone()
        inner._bars.update(self._bars)
        self.placed_inner = inner.place(self.num_sms)
        self._pointer_tables = []
        self._pointer_cache = {}

    def _match_group(self, inst):
        address = cords2addr(inst.cords)
        base_opcode = inst.opcode & ~((1 << 6) - 1)
        for group_id, (representative, alternatives) in enumerate(self.tensor_groups):
            start = representative.data_ptr()
            nbytes = representative.numel() * representative.element_size()
            offset = address - start
            if offset < 0 or offset >= nbytes:
                continue
            if base_opcode not in self._ADDRESS_ONLY_OPS and offset + inst.size > nbytes:
                continue
            return group_id, offset, alternatives
        return None

    def _transform_memory(self, inst):
        base_opcode = inst.opcode & ~((1 << 6) - 1)
        match = self._match_group(inst)
        if match is None:
            return inst
        if base_opcode not in self._SUPPORTED_OPS:
            raise ValueError(
                f"layered tensor address is used by unsupported opcode {base_opcode:#x}"
            )
        group_id, offset, alternatives = match
        cache_key = (group_id, offset, base_opcode)
        pointer_table = self._pointer_cache.get(cache_key)
        if pointer_table is None:
            if base_opcode in (
                opcode.OP_ALLOC_ROUTED_TMA_LOAD_1D & ~((1 << 6) - 1),
                opcode.OP_ALLOC_ROUTED_TMA_LOAD_BASE_1D & ~((1 << 6) - 1),
            ):
                if self.route_indices is None:
                    raise ValueError(
                        "layered routed loads require a fixed route-index buffer"
                    )
                pointer_table = torch.tensor(
                    [
                        [self.route_indices.data_ptr(), tensor.data_ptr() + offset]
                        for tensor in alternatives
                    ],
                    dtype=torch.int64,
                    device=alternatives[0].device,
                )
            else:
                pointer_table = torch.tensor(
                    [tensor.data_ptr() + offset for tensor in alternatives],
                    dtype=torch.int64,
                    device=alternatives[0].device,
                )
            self._pointer_cache[cache_key] = pointer_table
            self._pointer_tables.append(pointer_table)
        return indirect_1d_from(
            inst,
            pointer_table.reshape(-1)[:2],
            layer_indexed=self.layer_count > 1,
        )

    def _transform(self, item):
        if item is None or isinstance(item, ComputeInstruction):
            return item
        if isinstance(item, MemoryInstruction):
            return self._transform_memory(item)
        if isinstance(item, (list, tuple)):
            transformed = [self._transform(child) for child in item]
            return type(item)(transformed) if isinstance(item, tuple) else transformed
        if hasattr(item, "expand_instructions"):
            return self._transform(item.expand_instructions())
        if callable(item):
            return lambda sm: self._transform(item(sm))
        raise TypeError(f"unsupported layered schedule item {type(item).__name__}")

    def schedule(self, sm: int):
        return self._transform(self.placed_inner.schedule(sm))

    def bar_release_count(self, role: str):
        return self.placed_inner.bar_release_count(role)


def _aligned_row_shard(rows: int, num_sms: int, sm: int, alignment: int = 8):
    if rows % alignment:
        raise ValueError(f"row count must be divisible by {alignment}")
    groups = rows // alignment
    active_sms = min(num_sms, groups)
    if sm < 0 or sm >= active_sms:
        return active_sms, 0, 0
    groups_per_sm, extra = divmod(groups, active_sms)
    group_start = sm * groups_per_sm + min(sm, extra)
    group_count = groups_per_sm + (1 if sm < extra else 0)
    return active_sms, group_start * alignment, group_count * alignment


def _shared_load_1d(tensor, bytes: int | None = None):
    """Use bulk TMA when legal, otherwise the LDU metadata-copy path."""
    size = tensor.numel() * tensor.element_size() if bytes is None else bytes
    if size % 16 == 0 and tensor.data_ptr() % 16 == 0:
        return TmaLoad1D(tensor, bytes=size)
    return LduLoad1D(tensor, bytes=size)


def _shared_store_1d(tensor, bytes: int | None = None):
    """Use bulk TMA when legal, otherwise the STU metadata-copy path."""
    size = tensor.numel() * tensor.element_size() if bytes is None else bytes
    if size % 16 == 0 and tensor.data_ptr() % 16 == 0:
        return TmaStore1D(tensor, bytes=size)
    return StuStore1D(tensor, bytes=size)


class SchedNvfp4Gemv(Schedule):
    """Load one static NVFP4 shard through LDU and compute from shared slots."""

    def __init__(self, weight, weight_scale, activation, activation_scale,
                 alpha, output):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.activation = activation
        self.activation_scale = activation_scale
        self.alpha = alpha
        self.output = output

    def _expected_output_elements(self, rows):
        return rows

    def _output_view(self, row_start, row_count):
        return self.output[row_start:row_start + row_count]

    def _compute_instruction(self, row_count):
        return Nvfp4GemvSm100(row_count, self.k)

    def _on_place(self):
        if self.weight.dtype != torch.uint8 or self.weight.ndim != 2:
            raise ValueError("NVFP4 weight must be a rank-2 packed uint8 tensor")
        self.rows, packed_k = self.weight.shape
        self.k = packed_k * 2
        if self.k % 256:
            raise ValueError("shared NVFP4 GEMV requires K divisible by 256")
        if tuple(self.weight_scale.shape) != (self.rows, self.k // 16):
            raise ValueError("weight_scale must have shape [M, K/16]")
        if self.weight_scale.dtype != torch.float8_e4m3fn:
            raise ValueError("weight_scale must use torch.float8_e4m3fn")
        if self.activation.dtype != torch.uint8 or self.activation.numel() != self.k // 2:
            raise ValueError("activation must contain K/2 packed uint8 values")
        if (self.activation_scale.dtype != torch.float8_e4m3fn or
                self.activation_scale.numel() != self.k // 16):
            raise ValueError("activation_scale must contain K/16 E4M3 values")
        if self.alpha.dtype != torch.float32 or self.alpha.numel() != 1:
            raise ValueError("alpha must be a scalar float32 tensor")
        if (self.output.dtype != torch.bfloat16 or
                self.output.numel() != self._expected_output_elements(self.rows)):
            raise ValueError("output has the wrong BF16 element count")
        if self.num_sms <= 0:
            raise ValueError("NVFP4 GEMV requires at least one SM")
        self.active_sms, _, _ = _aligned_row_shard(
            self.rows, self.num_sms, 0
        )
        self.alpha_storage = torch.zeros(
            (4,), dtype=torch.float32, device=self.alpha.device
        )
        self.alpha_storage[0].copy_(self.alpha.reshape(-1)[0])

    def schedule(self, sm):
        active_sms, row_start, row_count = _aligned_row_shard(
            self.rows, self.num_sms, sm
        )
        if row_count == 0:
            return []
        weight = self.weight[row_start:row_start + row_count]
        weight_scale = self.weight_scale[row_start:row_start + row_count]
        output = self._output_view(row_start, row_count)
        return [
            self._compute_instruction(row_count),
            TmaLoad1D(weight),
            TmaLoad1D(weight_scale),
            TmaLoad1D(self.activation.reshape(-1)),
            TmaLoad1D(self.activation_scale.reshape(-1)),
            TmaLoad1D(self.alpha_storage),
            TmaStore1D(output).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.active_sms)


class SchedRoutedNvfp4Gemv(Schedule):
    """Resolve expert shards in LDU, then compute solely from shared slots."""

    def __init__(
        self,
        routing_state,
        route_rank,
        weight_fields,
        weight_scale_fields,
        alpha_field,
        rows,
        k,
        activation,
        activation_scale,
        output,
        *,
        route_ready=False,
        activation_mode="load",
    ):
        super().__init__()
        self.routing_state = routing_state
        self.route_rank = route_rank
        self.weight_fields = tuple(
            tuple(fields) if isinstance(fields, (list, tuple)) else (fields,)
            for fields in weight_fields
        )
        self.weight_scale_fields = tuple(
            tuple(fields) if isinstance(fields, (list, tuple)) else (fields,)
            for fields in weight_scale_fields
        )
        self.alpha_field = alpha_field
        self.rows = rows
        self.k = k
        self.activation = activation
        self.activation_scale = activation_scale
        self.output = output
        self.route_ready = bool(route_ready)
        if activation_mode not in ("load", "retain", "reuse"):
            raise ValueError(
                "activation_mode must be 'load', 'retain', or 'reuse'"
            )
        self.activation_mode = activation_mode

    def _on_place(self):
        if self.routing_state.device.type != "cuda":
            raise ValueError("routing state must be a CUDA tensor")
        if not self.routing_state.is_contiguous():
            raise ValueError("routing state must be contiguous")
        if not 0 <= self.route_rank < RoutedTmaLoad1D.ROUTE_COUNT:
            raise ValueError("route_rank must be in [0, 6)")
        if self.rows <= 0 or self.rows > 0xFFFF or self.rows % 8:
            raise ValueError("routed NVFP4 rows must be a positive multiple of 8")
        if self.k <= 0 or self.k > 0xFFFF or self.k % 256:
            raise ValueError("routed NVFP4 K must be a uint16 multiple of 256")
        if (self.activation.dtype != torch.uint8 or
                self.activation.numel() != self.k // 2):
            raise ValueError("activation must contain K/2 packed uint8 values")
        if (self.activation_scale.dtype != torch.float8_e4m3fn or
                self.activation_scale.numel() != self.k // 16):
            raise ValueError("activation_scale must contain K/16 E4M3 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != self.rows:
            raise ValueError("output must contain M BF16 values")
        self.active_sms, _, _ = _aligned_row_shard(
            self.rows, self.num_sms, 0
        )
        if (len(self.weight_fields) != self.active_sms or
                len(self.weight_scale_fields) != self.active_sms):
            raise ValueError("routed weight fields must contain one field per active SM")
        self.tile_rows = (65520 // (self.k // 2) // 8) * 8
        if self.tile_rows <= 0:
            raise ValueError("routed NVFP4 K is too large for an aligned tile")
        for sm in range(self.active_sms):
            _, _, row_count = _aligned_row_shard(
                self.rows, self.num_sms, sm
            )
            expected_tiles = (row_count + self.tile_rows - 1) // self.tile_rows
            if (
                len(self.weight_fields[sm]) != expected_tiles
                or len(self.weight_scale_fields[sm]) != expected_tiles
            ):
                raise ValueError(
                    "routed weight fields must contain one field per row tile"
                )
        flat_fields = (
            *(field for fields in self.weight_fields for field in fields),
            *(field for fields in self.weight_scale_fields for field in fields),
            self.alpha_field,
        )
        for field in flat_fields:
            if not 0 <= field <= RoutedTmaLoad1D.MAX_POINTER_FIELD:
                raise ValueError("routed pointer fields must fit in 13 bits")

    def schedule(self, sm):
        _, row_start, row_count = _aligned_row_shard(
            self.rows, self.num_sms, sm
        )
        if row_count == 0:
            return []
        route_bar = self._bar("route")
        if route_bar is None and not self.route_ready:
            raise ValueError("routed NVFP4 GEMV requires a route barrier")
        instructions = []
        tile_count = len(self.weight_fields[sm])
        for tile_index, tile_offset in enumerate(
            range(0, row_count, self.tile_rows)
        ):
            tile_count_rows = min(self.tile_rows, row_count - tile_offset)
            weight_bytes = tile_count_rows * (self.k // 2)
            scale_bytes = tile_count_rows * (self.k // 16)
            weight_load = RoutedTmaLoad1D(
                self.routing_state,
                self.route_rank,
                self.weight_fields[sm][tile_index],
                weight_bytes,
            )
            if route_bar is not None:
                weight_load.bar(route_bar)

            first_tile = tile_index == 0
            final_tile = tile_index + 1 == tile_count
            if first_tile and self.activation_mode in ("load", "retain"):
                if self.activation_mode == "retain" or not final_tile:
                    activation_load = TmaLoadReg1D(
                        self.activation.reshape(-1), 0, 0
                    )
                    activation_scale_load = TmaLoadReg1D(
                        self.activation_scale.reshape(-1), 1, 1
                    )
                else:
                    activation_load = TmaLoad1D(self.activation.reshape(-1))
                    activation_scale_load = TmaLoad1D(
                        self.activation_scale.reshape(-1)
                    )
            else:
                activation_load = RegLoad(0, slot_id=0).fixed_port(0)
                activation_scale_load = RegLoad(1, slot_id=1).fixed_port(1)

            retain_activation = (
                self.activation_mode == "retain" or not final_tile
            )
            tile_start = row_start + tile_offset
            store = TmaStore1D(
                self.output[tile_start:tile_start + tile_count_rows]
            )
            if final_tile:
                store.bar(self._bar("output"))
            instructions.extend(
                (
                    Nvfp4GemvSm100(
                        tile_count_rows,
                        self.k,
                        retain_activation=retain_activation,
                    ),
                    weight_load,
                    RoutedTmaLoad1D(
                        self.routing_state,
                        self.route_rank,
                        self.weight_scale_fields[sm][tile_index],
                        scale_bytes,
                    ),
                    activation_load,
                    activation_scale_load,
                    RoutedTmaLoad1D(
                        self.routing_state,
                        self.route_rank,
                        self.alpha_field,
                        16,
                    ),
                    store,
                )
            )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.active_sms)


class SchedNvfp4GemvUmma(SchedNvfp4Gemv):
    """Map one native block-scaled UMMA tile to each resident SM."""

    def __init__(self, weight, weight_scale, activation, activation_scale,
                 alpha, output, output_columns=1):
        if output_columns not in (1, 8):
            raise ValueError("NVFP4 UMMA output_columns must be 1 or 8")
        self.output_columns = output_columns
        super().__init__(
            weight, weight_scale, activation, activation_scale, alpha, output
        )

    def _expected_output_elements(self, rows):
        return rows * self.output_columns

    def _output_view(self, row_start, row_count):
        start = row_start * self.output_columns
        stop = (row_start + row_count) * self.output_columns
        return self.output.reshape(-1)[start:stop]

    def _compute_instruction(self, row_count):
        return Nvfp4GemvUmmaSm100(row_count, self.k, self.output_columns)


class SchedNvfp4UmmaPrepack(Schedule):
    """Setup-only conversion to combined native data/scale K256 tiles."""

    WEIGHT = Nvfp4UmmaPrepackSm100.WEIGHT
    ACTIVATION = Nvfp4UmmaPrepackSm100.ACTIVATION
    TILE_M = 128
    TILE_K = 256
    WEIGHT_TILE_BYTES = 18432
    ACTIVATION_TILE_BYTES = 3072

    def __init__(self, kind, data, scale_tiles, output, data_tma):
        super().__init__()
        self.kind = kind
        self.data = data
        self.scale_tiles = scale_tiles
        self.output = output
        self.data_tma = data_tma

    def _on_place(self):
        if self.kind not in (self.WEIGHT, self.ACTIVATION):
            raise ValueError("NVFP4 prepack kind must be weight or activation")
        if self.data.dtype != torch.uint8 or self.data.ndim != 2:
            raise ValueError("NVFP4 prepack data must be rank-2 packed uint8")
        if self.data.shape[1] % (self.TILE_K // 2):
            raise ValueError("NVFP4 prepack K must be K256 aligned")
        self.k_tiles = self.data.shape[1] // (self.TILE_K // 2)
        if self.kind == self.WEIGHT:
            if self.data.shape[0] % self.TILE_M:
                raise ValueError("NVFP4 prepack weight rows must be M128 aligned")
            self.m_tiles = self.data.shape[0] // self.TILE_M
            if self.num_sms != self.m_tiles:
                raise ValueError("weight prepack requires one SM per M128 tile")
            expected_scales = (self.m_tiles, self.k_tiles, self.TILE_M, 16)
            expected_output = (
                self.m_tiles,
                self.k_tiles,
                self.WEIGHT_TILE_BYTES,
            )
            expected_tma_bytes = self.TILE_M * (self.TILE_K // 2)
        else:
            self.m_tiles = 1
            if tuple(self.data.shape) != (8, self.k_tiles * (self.TILE_K // 2)):
                raise ValueError("activation prepack data must be [8,K/2]")
            if self.num_sms != 1:
                raise ValueError("activation prepack requires one SM")
            expected_scales = (self.k_tiles, 16)
            expected_output = (self.k_tiles, self.ACTIVATION_TILE_BYTES)
            expected_tma_bytes = 8 * (self.TILE_K // 2)
        if (
            self.scale_tiles.dtype != torch.float8_e4m3fn
            or tuple(self.scale_tiles.shape) != expected_scales
            or not self.scale_tiles.is_contiguous()
        ):
            raise ValueError(f"NVFP4 prepack scales must be {expected_scales} E4M3")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != expected_output
            or not self.output.is_contiguous()
        ):
            raise ValueError(f"NVFP4 prepack output must be uint8 {expected_output}")
        if self.data_tma.size != expected_tma_bytes:
            raise ValueError("NVFP4 prepack data TMA has the wrong tile size")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = [Nvfp4UmmaPrepackSm100(self.kind, self.k_tiles)]
        row_start = sm * self.TILE_M if self.kind == self.WEIGHT else 0
        for tile in range(self.k_tiles):
            packed_k_start = tile * (self.TILE_K // 2)
            scale = (
                self.scale_tiles[sm, tile]
                if self.kind == self.WEIGHT
                else self.scale_tiles[tile]
            )
            output = (
                self.output[sm, tile]
                if self.kind == self.WEIGHT
                else self.output[tile]
            )
            instructions.extend(
                (
                    self.data_tma.cord(row_start, packed_k_start),
                    TmaLoad1D(scale.reshape(-1)),
                    TmaStore1D(output.reshape(-1)),
                )
            )
        return instructions


class SchedDsv4Nvfp4QuantUmmaB(Schedule):
    """Emit native N8/K256 FP4 data and scales without an intermediate copy."""

    TILE_K = SchedNvfp4UmmaPrepack.TILE_K
    TILE_BYTES = SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(self, input, global_scale, output):
        super().__init__()
        self.input = input
        self.global_scale = global_scale
        self.output = output

    def _on_place(self):
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("native NVFP4 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % self.TILE_K:
            raise ValueError("native NVFP4 quant K must be K256 aligned")
        self.k_tiles = self.k // self.TILE_K
        if self.num_sms != self.k_tiles:
            raise ValueError("native NVFP4 quant requires one SM per K256 tile")
        if self.global_scale.dtype != torch.float32 or self.global_scale.numel() != 1:
            raise ValueError("native NVFP4 global scale must be scalar FP32")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != (self.k_tiles, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                f"native NVFP4 output must be uint8 [{self.k_tiles},{self.TILE_BYTES}]"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        start = sm * self.TILE_K
        return [
            Dsv4Nvfp4QuantUmmaBSm100(1),
            _shared_load_1d(self.input[start : start + self.TILE_K]),
            _shared_load_1d(self.global_scale.reshape(-1)),
            _shared_store_1d(self.output[sm].reshape(-1)).bar(
                self._bar("output")
            ),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedRoutedDsv4Nvfp4QuantUmmaB(Schedule):
    """Resolve one expert scale in LDU and emit final native activation tiles."""

    TILE_K = SchedDsv4Nvfp4QuantUmmaB.TILE_K
    TILE_BYTES = SchedDsv4Nvfp4QuantUmmaB.TILE_BYTES

    def __init__(self, routing_state, route_rank, scale_field, input, output):
        super().__init__()
        self.routing_state = routing_state
        self.route_rank = route_rank
        self.scale_field = scale_field
        self.input = input
        self.output = output

    def _on_place(self):
        if self.routing_state.device.type != "cuda":
            raise ValueError("routing state must be a CUDA tensor")
        if not 0 <= self.route_rank < RoutedTmaLoad1D.ROUTE_COUNT:
            raise ValueError("route_rank must be in [0, 6)")
        if not 0 <= self.scale_field <= RoutedTmaLoad1D.MAX_POINTER_FIELD:
            raise ValueError("routed native scale field must fit in 13 bits")
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("routed native NVFP4 input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % self.TILE_K:
            raise ValueError("routed native NVFP4 K must be K256 aligned")
        self.k_tiles = self.k // self.TILE_K
        if self.num_sms != self.k_tiles:
            raise ValueError("routed native quant requires one SM per K256 tile")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != (self.k_tiles, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                f"routed native output must be uint8 [{self.k_tiles},{self.TILE_BYTES}]"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        route_bar = self._bar("route")
        if route_bar is None:
            raise ValueError("routed native NVFP4 quant requires a route barrier")
        start = sm * self.TILE_K
        return [
            Dsv4Nvfp4QuantUmmaBSm100(1),
            _shared_load_1d(self.input[start : start + self.TILE_K]),
            RoutedTmaLoad1D(
                self.routing_state,
                self.route_rank,
                self.scale_field,
                16,
            ).bar(route_bar),
            _shared_store_1d(self.output[sm].reshape(-1)).bar(
                self._bar("output")
            ),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedNvfp4GemvUmmaStream(Schedule):
    """One M128 output tile with combined data/scale K256 operand loads."""

    TILE_M = SchedNvfp4UmmaPrepack.TILE_M
    WEIGHT_TILE_BYTES = SchedNvfp4UmmaPrepack.WEIGHT_TILE_BYTES
    ACTIVATION_TILE_BYTES = SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(
        self,
        weight_tiles,
        activation_tiles,
        alpha,
        output,
        *,
        activation_mode="stream",
    ):
        super().__init__()
        self.weight_tiles = weight_tiles
        self.activation_tiles = activation_tiles
        self.alpha = alpha
        self.output = output
        if activation_mode not in ("stream", "load", "retain", "reuse"):
            raise ValueError(
                "native activation_mode must be stream, load, retain, or reuse"
            )
        self.activation_mode = activation_mode

    def _on_place(self):
        if (
            self.weight_tiles.dtype != torch.uint8
            or self.weight_tiles.ndim != 3
            or self.weight_tiles.shape[2] != self.WEIGHT_TILE_BYTES
            or not self.weight_tiles.is_contiguous()
        ):
            raise ValueError("streaming weight tiles must be [M/128,K/256,18432] uint8")
        self.m_tiles, self.k_tiles, _ = self.weight_tiles.shape
        if self.num_sms != self.m_tiles:
            raise ValueError("streaming NVFP4 requires one M128 tile per SM")
        if (
            self.activation_tiles.dtype != torch.uint8
            or tuple(self.activation_tiles.shape)
            != (self.k_tiles, self.ACTIVATION_TILE_BYTES)
            or not self.activation_tiles.is_contiguous()
        ):
            raise ValueError("streaming activation tiles must be [K/256,3072] uint8")
        self.rows = self.m_tiles * self.TILE_M
        if self.alpha.dtype != torch.float32 or self.alpha.numel() != 4:
            raise ValueError("streaming NVFP4 alpha storage must contain four FP32 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != self.rows:
            raise ValueError("streaming NVFP4 output must contain M BF16 values")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = [
            Nvfp4GemvUmmaStreamSm100(
                self.k_tiles,
                retain_activation=self.activation_mode == "retain",
                bulk_activation=self.activation_mode != "stream",
            ),
            TmaLoad1D(self.alpha).fixed_port(1),
        ]
        if self.activation_mode == "load":
            instructions.append(
                TmaLoad1D(self.activation_tiles.reshape(-1)).fixed_port(1)
            )
        elif self.activation_mode == "retain":
            instructions.append(
                TmaLoadReg1D(
                    self.activation_tiles.reshape(-1), 0, 1
                )
            )
        elif self.activation_mode == "reuse":
            instructions.append(RegLoad(0, slot_id=0).fixed_port(1))
        for tile in range(self.k_tiles):
            instructions.append(
                TmaLoad1D(
                    self.weight_tiles[sm, tile].reshape(-1)
                ).fixed_port(0)
            )
            if self.activation_mode == "stream":
                instructions.append(
                    TmaLoad1D(
                        self.activation_tiles[tile].reshape(-1)
                    ).fixed_port(1)
                )
        row_start = sm * self.TILE_M
        instructions.append(
            TmaStore1D(
                self.output[row_start : row_start + self.TILE_M]
            ).bar(self._bar("output"))
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedRoutedNvfp4GemvUmmaStream(Schedule):
    """Resolve combined native weight tiles and run one or more M128 tasks/SM."""

    TILE_M = SchedNvfp4UmmaPrepack.TILE_M
    WEIGHT_TILE_BYTES = SchedNvfp4UmmaPrepack.WEIGHT_TILE_BYTES
    ACTIVATION_TILE_BYTES = SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(
        self,
        routing_state,
        route_rank,
        weight_fields,
        alpha_field,
        activation_tiles,
        output,
        *,
        route_ready=False,
        activation_mode="load",
        output_mode="store",
        output_register=0,
        output_port=0,
    ):
        super().__init__()
        self.routing_state = routing_state
        self.route_rank = route_rank
        self.weight_fields = tuple(weight_fields)
        self.alpha_field = alpha_field
        self.activation_tiles = activation_tiles
        self.output = output
        self.route_ready = bool(route_ready)
        if activation_mode not in ("stream", "load", "retain", "reuse"):
            raise ValueError(
                "routed native activation_mode must be stream, load, retain, or reuse"
            )
        self.activation_mode = activation_mode
        if output_mode not in ("store", "retain"):
            raise ValueError("routed native output_mode must be store or retain")
        if not 0 <= output_register < 4:
            raise ValueError("routed native output register must be in [0, 4)")
        if output_port not in (0, 1):
            raise ValueError("routed native output port must be 0 or 1")
        self.output_mode = output_mode
        self.output_register = output_register
        self.output_port = output_port

    def _on_place(self):
        if self.routing_state.device.type != "cuda":
            raise ValueError("routing state must be a CUDA tensor")
        if not 0 <= self.route_rank < RoutedTmaLoad1D.ROUTE_COUNT:
            raise ValueError("route_rank must be in [0, 6)")
        if not self.weight_fields:
            raise ValueError("routed native weights require at least one M128 tile")
        self.m_tiles = len(self.weight_fields)
        if not 0 < self.num_sms <= self.m_tiles:
            raise ValueError("routed native GEMV needs 1..M/128 SMs")
        if (
            self.activation_tiles.dtype != torch.uint8
            or self.activation_tiles.ndim != 2
            or self.activation_tiles.shape[1] != self.ACTIVATION_TILE_BYTES
            or not self.activation_tiles.is_contiguous()
        ):
            raise ValueError("routed activation tiles must be [K/256,3072] uint8")
        self.k_tiles = self.activation_tiles.shape[0]
        for field in (
            *self.weight_fields,
            self.alpha_field,
        ):
            if not 0 <= field <= RoutedTmaLoad1D.MAX_POINTER_FIELD:
                raise ValueError("routed native pointer fields must fit in 13 bits")
        self.rows = self.m_tiles * self.TILE_M
        if self.output_mode == "store":
            if (
                self.output is None
                or self.output.dtype != torch.bfloat16
                or self.output.numel() != self.rows
            ):
                raise ValueError("routed native output must contain M BF16 values")
        elif self.output is not None:
            raise ValueError("retained routed native output must not name HBM storage")
        if self.output_mode == "retain" and self.num_sms != self.m_tiles:
            raise ValueError(
                "retained routed output requires one M128 tile per SM"
            )

    def _tile_shard(self, sm):
        tiles_per_sm, extra = divmod(self.m_tiles, self.num_sms)
        tile_start = sm * tiles_per_sm + min(sm, extra)
        tile_count = tiles_per_sm + int(sm < extra)
        return tile_start, tile_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        route_bar = self._bar("route")
        if route_bar is None and not self.route_ready:
            raise ValueError("routed native NVFP4 GEMV requires a route barrier")
        tile_start, tile_count = self._tile_shard(sm)
        instructions = []
        for local_index, output_tile in enumerate(
            range(tile_start, tile_start + tile_count)
        ):
            first_output = local_index == 0
            final_output = local_index + 1 == tile_count
            retain = self.activation_mode == "retain" or not final_output
            bulk = self.activation_mode != "stream"
            instructions.append(
                Nvfp4GemvUmmaStreamSm100(
                    self.k_tiles,
                    retain_activation=retain,
                    bulk_activation=bulk,
                )
            )
            alpha_load = RoutedTmaLoad1D(
                self.routing_state,
                self.route_rank,
                self.alpha_field,
                16,
            ).fixed_port(1)
            if first_output and route_bar is not None:
                alpha_load.bar(route_bar)
            instructions.append(alpha_load)

            if self.activation_mode == "stream":
                activation_kind = "stream"
            elif self.activation_mode == "retain":
                activation_kind = "retain" if first_output else "reuse"
            elif self.activation_mode == "reuse":
                activation_kind = "reuse"
            elif first_output:
                activation_kind = "retain" if not final_output else "load"
            else:
                activation_kind = "reuse"
            if activation_kind == "load":
                instructions.append(
                    TmaLoad1D(self.activation_tiles.reshape(-1)).fixed_port(1)
                )
            elif activation_kind == "retain":
                instructions.append(
                    TmaLoadReg1D(
                        self.activation_tiles.reshape(-1), 0, 1
                    )
                )
            elif activation_kind == "reuse":
                instructions.append(RegLoad(0, slot_id=0).fixed_port(1))

            for k_tile in range(self.k_tiles):
                if k_tile == 0:
                    weight_load = RoutedTmaLoadBase1D(
                        self.routing_state,
                        self.route_rank,
                        self.weight_fields[output_tile],
                        self.WEIGHT_TILE_BYTES,
                    )
                else:
                    weight_load = TmaLoadAddressReg1D(
                        RoutedTmaLoadBase1D.ADDRESS_REGISTER,
                        k_tile * self.WEIGHT_TILE_BYTES,
                        self.WEIGHT_TILE_BYTES,
                    )
                if first_output and k_tile == 0 and route_bar is not None:
                    weight_load.bar(route_bar)
                instructions.append(weight_load.fixed_port(0))
                if activation_kind == "stream":
                    instructions.append(
                        TmaLoad1D(
                            self.activation_tiles[k_tile].reshape(-1)
                        ).fixed_port(1)
                    )
            if self.output_mode == "store":
                row_start = output_tile * self.TILE_M
                store = TmaStore1D(
                    self.output[row_start : row_start + self.TILE_M]
                )
                if final_output:
                    store.bar(self._bar("output"))
            else:
                store = RegStore(
                    self.output_register,
                    size=self.TILE_M * 2,
                ).fixed_port(self.output_port)
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output" or self.output_mode != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4SwiGluShard128(Schedule):
    """Consume retained gate/up M128 shards and store only their fused result."""

    TILE_K = 128

    def __init__(
        self,
        gate_register,
        gate_port,
        up_register,
        up_port,
        output,
        *,
        swiglu_limit=10.0,
    ):
        super().__init__()
        self.gate_register = gate_register
        self.gate_port = gate_port
        self.up_register = up_register
        self.up_port = up_port
        self.output = output
        self.swiglu_limit = swiglu_limit

    def _on_place(self):
        if self.gate_port == self.up_port:
            raise ValueError("retained gate and up shards require separate LDU ports")
        if self.gate_port not in (0, 1) or self.up_port not in (0, 1):
            raise ValueError("retained shard ports must be 0 or 1")
        if not 0 <= self.gate_register < 4 or not 0 <= self.up_register < 4:
            raise ValueError("retained shard registers must be in [0, 4)")
        if (
            self.output.dtype != torch.bfloat16
            or self.output.numel() != self.num_sms * self.TILE_K
            or not self.output.is_contiguous()
        ):
            raise ValueError("sharded SwiGLU output must be contiguous BF16 [SM,128]")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        start = sm * self.TILE_K
        return [
            Dsv4SiluClampMul128(1, self.swiglu_limit),
            TmaStore1D(self.output[start : start + self.TILE_K]).bar(
                self._bar("output")
            ),
            RegLoad(self.gate_register).fixed_port(self.gate_port),
            RegLoad(self.up_register).fixed_port(self.up_port),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedFp8UmmaPrepack(Schedule):
    """Setup-only conversion to combined native MXF8 K128 tiles."""

    WEIGHT = Fp8UmmaPrepackSm100.WEIGHT
    ACTIVATION = Fp8UmmaPrepackSm100.ACTIVATION
    TILE_M = 128
    TILE_N = 8
    TILE_K = 128
    WEIGHT_TILE_BYTES = 16896
    ACTIVATION_TILE_BYTES = 2048

    def __init__(self, kind, data, scale_tiles, output, data_tma):
        super().__init__()
        self.kind = kind
        self.data = data
        self.scale_tiles = scale_tiles
        self.output = output
        self.data_tma = data_tma

    def _on_place(self):
        if self.kind not in (self.WEIGHT, self.ACTIVATION):
            raise ValueError("FP8 prepack kind must be weight or activation")
        if self.data.dtype != torch.float8_e4m3fn or self.data.ndim != 2:
            raise ValueError("FP8 prepack data must be rank-2 E4M3")
        if self.data.shape[1] % self.TILE_K:
            raise ValueError("FP8 prepack K must be K128 aligned")
        self.k_tiles = self.data.shape[1] // self.TILE_K
        if self.kind == self.WEIGHT:
            if self.data.shape[0] % self.TILE_M:
                raise ValueError("FP8 prepack weight rows must be M128 aligned")
            self.m_tiles = self.data.shape[0] // self.TILE_M
            if not 0 < self.num_sms <= self.m_tiles:
                raise ValueError("weight prepack needs 1..M/128 SMs")
            expected_scales = (self.m_tiles, self.k_tiles)
            expected_output = (
                self.m_tiles,
                self.k_tiles,
                self.WEIGHT_TILE_BYTES,
            )
            expected_tma_bytes = self.TILE_M * self.TILE_K
        else:
            self.m_tiles = 1
            if tuple(self.data.shape) != (
                self.TILE_N,
                self.k_tiles * self.TILE_K,
            ):
                raise ValueError("FP8 activation prepack data must be [8,K]")
            if self.num_sms != 1:
                raise ValueError("FP8 activation prepack requires one SM")
            expected_scales = (self.k_tiles,)
            expected_output = (
                self.k_tiles,
                self.ACTIVATION_TILE_BYTES,
            )
            expected_tma_bytes = self.TILE_N * self.TILE_K
        if (
            self.scale_tiles.dtype != torch.float8_e8m0fnu
            or tuple(self.scale_tiles.shape) != expected_scales
            or not self.scale_tiles.is_contiguous()
        ):
            raise ValueError(
                f"FP8 prepack scales must be UE8M0 {expected_scales}"
            )
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != expected_output
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                f"FP8 prepack output must be uint8 {expected_output}"
            )
        if self.data_tma.size != expected_tma_bytes:
            raise ValueError("FP8 prepack data TMA has the wrong tile size")

    def _tile_shard(self, sm):
        tiles_per_sm, extra = divmod(self.m_tiles, self.num_sms)
        tile_start = sm * tiles_per_sm + min(sm, extra)
        tile_count = tiles_per_sm + int(sm < extra)
        return tile_start, tile_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        if self.kind == self.ACTIVATION:
            instructions = [Fp8UmmaPrepackSm100(self.kind, self.k_tiles)]
            output_tile = 0
            tile_count = 1
        else:
            output_tile, tile_count = self._tile_shard(sm)
            instructions = []
        for m_tile in range(output_tile, output_tile + tile_count):
            if self.kind == self.WEIGHT:
                instructions.append(
                    Fp8UmmaPrepackSm100(self.kind, self.k_tiles)
                )
            row_start = m_tile * self.TILE_M if self.kind == self.WEIGHT else 0
            for k_tile in range(self.k_tiles):
                scale = (
                    self.scale_tiles[m_tile, k_tile]
                    if self.kind == self.WEIGHT
                    else self.scale_tiles[k_tile]
                )
                output = (
                    self.output[m_tile, k_tile]
                    if self.kind == self.WEIGHT
                    else self.output[k_tile]
                )
                instructions.extend(
                    (
                        self.data_tma.cord(
                            row_start, k_tile * self.TILE_K
                        ).fixed_port(0),
                        _shared_load_1d(scale.reshape(-1)).fixed_port(1),
                        TmaStore1D(output.reshape(-1)),
                    )
                )
        return instructions


class SchedDsv4Fp8QuantUmmaB(Schedule):
    """Quantize BF16 directly into combined native N8/K128 MXF8 tiles."""

    TILE_K = SchedFp8UmmaPrepack.TILE_K
    TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output

    def _on_place(self):
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("native FP8 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % self.TILE_K:
            raise ValueError("native FP8 quant K must be K128 aligned")
        self.k_tiles = self.k // self.TILE_K
        if self.num_sms != self.k_tiles:
            raise ValueError("native FP8 quant requires one SM per K128 tile")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != (self.k_tiles, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                f"native FP8 output must be uint8 "
                f"[{self.k_tiles},{self.TILE_BYTES}]"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        start = sm * self.TILE_K
        return [
            Dsv4Fp8QuantUmmaBSm100(1),
            _shared_load_1d(
                self.input[start : start + self.TILE_K]
            ).fixed_port(1),
            TmaStore1D(self.output[sm].reshape(-1)).bar(
                self._bar("output")
            ),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedFp8GemvUmmaStream(Schedule):
    """Shape-sharded M128/K128 native MXF8 projection."""

    TILE_M = SchedFp8UmmaPrepack.TILE_M
    WEIGHT_TILE_BYTES = SchedFp8UmmaPrepack.WEIGHT_TILE_BYTES
    ACTIVATION_TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES
    ACTIVATION_TILES_PER_CHUNK = 4

    def __init__(self, weight_tiles, activation_tiles, output):
        super().__init__()
        self.weight_tiles = weight_tiles
        self.activation_tiles = activation_tiles
        self.output = output

    def _on_place(self):
        if (
            self.weight_tiles.dtype != torch.uint8
            or self.weight_tiles.ndim != 3
            or self.weight_tiles.shape[2] != self.WEIGHT_TILE_BYTES
            or not self.weight_tiles.is_contiguous()
        ):
            raise ValueError(
                "native FP8 weights must be [M/128,K/128,16896] uint8"
            )
        self.m_tiles, self.k_tiles, _ = self.weight_tiles.shape
        if not 0 < self.num_sms <= self.m_tiles:
            raise ValueError("native FP8 GEMV needs 1..M/128 SMs")
        if not 0 < self.k_tiles <= 64:
            raise ValueError("native FP8 GEMV supports 1..64 K128 tiles")
        if (
            self.activation_tiles.dtype != torch.uint8
            or tuple(self.activation_tiles.shape)
            != (self.k_tiles, self.ACTIVATION_TILE_BYTES)
            or not self.activation_tiles.is_contiguous()
        ):
            raise ValueError(
                "native FP8 activations must be [K/128,2048] uint8"
            )
        self.rows = self.m_tiles * self.TILE_M
        if (
            self.output.dtype != torch.bfloat16
            or self.output.numel() != self.rows
            or not self.output.is_contiguous()
        ):
            raise ValueError("native FP8 output must contain M BF16 values")

    def _tile_shard(self, sm):
        tiles_per_sm, extra = divmod(self.m_tiles, self.num_sms)
        tile_start = sm * tiles_per_sm + min(sm, extra)
        tile_count = tiles_per_sm + int(sm < extra)
        return tile_start, tile_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        tile_start, tile_count = self._tile_shard(sm)
        instructions = []
        for output_tile in range(tile_start, tile_start + tile_count):
            final_output = output_tile + 1 == tile_start + tile_count
            instructions.append(
                Fp8GemvUmmaStreamSm100(self.k_tiles)
            )
            for chunk_start in range(
                0, self.k_tiles, self.ACTIVATION_TILES_PER_CHUNK
            ):
                chunk_stop = min(
                    chunk_start + self.ACTIVATION_TILES_PER_CHUNK,
                    self.k_tiles,
                )
                instructions.append(
                    TmaLoad1D(
                        self.activation_tiles[
                            chunk_start:chunk_stop
                        ].reshape(-1)
                    ).fixed_port(1)
                )
                for k_tile in range(chunk_start, chunk_stop):
                    instructions.append(
                        TmaLoad1D(
                            self.weight_tiles[
                                output_tile, k_tile
                            ].reshape(-1)
                        ).fixed_port(0)
                    )
            row_start = output_tile * self.TILE_M
            store = TmaStore1D(
                self.output[row_start : row_start + self.TILE_M]
            )
            if final_output:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedFp8Block128Gemv(Schedule):
    """Shard an E4M3/UE8M0 checkpoint GEMV across resident SMs."""

    def __init__(self, weight, weight_scale, activation, activation_scale,
                 output):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.activation = activation
        self.activation_scale = activation_scale
        self.output = output

    def _on_place(self):
        if self.weight.dtype != torch.float8_e4m3fn or self.weight.ndim != 2:
            raise ValueError("FP8 weight must be a rank-2 E4M3 tensor")
        self.rows, self.k = self.weight.shape
        if self.k % 128:
            raise ValueError("FP8 GEMV K must be divisible by 128")
        expected_weight_sf = ((self.rows + 127) // 128, self.k // 128)
        if (self.weight_scale.dtype != torch.float8_e8m0fnu or
                tuple(self.weight_scale.shape) != expected_weight_sf):
            raise ValueError(
                f"weight_scale must be UE8M0 with shape {expected_weight_sf}"
            )
        if (self.activation.dtype != torch.float8_e4m3fn or
                self.activation.numel() != self.k):
            raise ValueError("activation must contain K E4M3 values")
        if (self.activation_scale.dtype != torch.float8_e8m0fnu or
                self.activation_scale.numel() != self.k // 128):
            raise ValueError("activation_scale must contain K/128 UE8M0 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != self.rows:
            raise ValueError("output must contain M BF16 values")
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("FP8 GEMV requires 1 <= num_sms <= M")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        max_tile_rows = max(1, 65520 // self.k)
        instructions = []
        row_end = row_start + row_count
        for tile_start in range(row_start, row_end, max_tile_rows):
            tile_end = min(tile_start + max_tile_rows, row_end)
            tile_rows = tile_end - tile_start
            scale_start = tile_start // 128
            scale_end = (tile_end + 127) // 128
            instructions += [
                Fp8Block128GemvSm100(
                    tile_rows, self.k, tile_start % 128
                ),
                _shared_load_1d(self.weight[tile_start:tile_end]),
                _shared_load_1d(self.weight_scale[scale_start:scale_end]),
                _shared_load_1d(self.activation.reshape(-1)),
                _shared_load_1d(self.activation_scale.reshape(-1)),
            ]
            store = _shared_store_1d(self.output[tile_start:tile_end])
            if tile_end == row_end:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Rope512_64(Schedule):
    def __init__(self, input, table, output, inverse=False):
        super().__init__()
        self.input = input
        self.table = table
        self.output = output
        self.inverse = inverse

    def _on_place(self):
        if (self.input.dtype != torch.bfloat16 or self.input.ndim != 2 or
                self.input.shape[1] != 512):
            raise ValueError("DeepSeek RoPE input must be BF16 [rows,512]")
        if self.output.dtype != torch.bfloat16 or self.output.shape != self.input.shape:
            raise ValueError("DeepSeek RoPE output must match the input")
        if (self.table.dtype != torch.float32 or
                tuple(self.table.shape) != (32, 2)):
            raise ValueError("DeepSeek RoPE table must be FP32 [32,2]")
        self.rows = self.input.shape[0]
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek partial RoPE requires 1 <= num_sms <= rows")

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions += [
                Dsv4Rope512_64(1, self.inverse),
                TmaLoad1D(self.input[row]),
                TmaLoad1D(self.table),
                TmaStore1D(self.output[row]).bar(self._bar("output")),
            ]
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.rows)


class SchedDsv4Rope128_64(SchedDsv4Rope512_64):
    def _on_place(self):
        if (self.input.dtype != torch.bfloat16 or self.input.ndim != 2 or
                self.input.shape[1] != 128):
            raise ValueError("DeepSeek index RoPE input must be BF16 [rows,128]")
        if self.output.dtype != torch.bfloat16 or self.output.shape != self.input.shape:
            raise ValueError("DeepSeek index RoPE output must match the input")
        if (self.table.dtype != torch.float32 or
                tuple(self.table.shape) != (32, 2)):
            raise ValueError("DeepSeek index RoPE table must be FP32 [32,2]")
        self.rows = self.input.shape[0]
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek index RoPE requires 1 <= num_sms <= rows")

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions += [
                Dsv4Rope128_64(1, self.inverse),
                TmaLoad1D(self.input[row]),
                TmaLoad1D(self.table),
                TmaStore1D(self.output[row]).bar(self._bar("output")),
            ]
        return instructions


class SchedDsv4SparseAttention512(Schedule):
    def __init__(self, q, kv, indices, sink, output):
        super().__init__()
        self.q = q
        self.kv = kv
        self.indices = indices
        self.sink = sink
        self.output = output

    def _on_place(self):
        if (self.q.dtype != torch.bfloat16 or self.q.ndim != 2 or
                self.q.shape[1] != 512):
            raise ValueError("DeepSeek sparse Q must be BF16 [heads,512]")
        self.heads = self.q.shape[0]
        if self.num_sms != self.heads:
            raise ValueError("DeepSeek sparse attention uses one SM per head")
        if (self.kv.dtype != torch.bfloat16 or self.kv.ndim != 2 or
                self.kv.shape[1] != 512):
            raise ValueError("DeepSeek sparse KV must be BF16 [rows,512]")
        if self.indices.dtype != torch.int32 or self.indices.ndim != 1:
            raise ValueError("DeepSeek sparse indices must be int32 [topk]")
        if self.indices.numel() <= 0:
            raise ValueError("DeepSeek sparse attention needs at least one index")
        if self.sink.dtype != torch.float32 or self.sink.numel() != self.heads:
            raise ValueError("DeepSeek attention sink must be FP32 [heads]")
        if self.output.dtype != torch.bfloat16 or self.output.shape != self.q.shape:
            raise ValueError("DeepSeek sparse output must match Q")
        self.indexed_table = IndexedLoadTable(self.kv, self.indices)

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = [
            Dsv4SparseAttention512(self.indices.numel()),
            TmaLoad1D(self.q[sm]),
            _shared_load_1d(self.indices),
            _shared_load_1d(self.sink[sm:sm + 1]),
        ]
        load = IndexedTmaLoad1D(
            self.indexed_table.state[0], 512 * 2
        )
        step = (load, IndexedTmaLoad1D.RECORD_BYTES)
        if self._bar("indices") is None:
            instructions += RepeatM.on(self.indices.numel(), step)
        else:
            instructions += RepeatM.onSync(
                0, self._bar("indices"), self.indices.numel(), step
            )
        instructions.append(
            TmaStore1D(self.output[sm]).bar(self._bar("output"))
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4RouteTop6(Schedule):
    def __init__(self, logits, bias, hash_indices, output_indices,
                 output_weights, hash_routing=False, route_scale=1.5):
        super().__init__()
        self.logits = logits
        self.bias = bias
        self.hash_indices = hash_indices
        self.output_indices = output_indices
        self.output_weights = output_weights
        self.hash_routing = hash_routing
        self.route_scale = route_scale

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek routing currently uses exactly one SM")
        if self.logits.dtype != torch.bfloat16 or self.logits.numel() != 256:
            raise ValueError("DeepSeek routing logits must contain 256 BF16 values")
        if self.bias.dtype != torch.float32 or self.bias.numel() != 256:
            raise ValueError("DeepSeek routing bias must contain 256 FP32 values")
        if self.hash_indices.dtype != torch.int32 or self.hash_indices.numel() != 8:
            raise ValueError("DeepSeek hash routing storage must contain eight int32 values")
        if self.output_indices.dtype != torch.int32 or self.output_indices.numel() != 8:
            raise ValueError("DeepSeek route-id storage must contain eight int32 values")
        if self.output_weights.dtype != torch.float32 or self.output_weights.numel() != 8:
            raise ValueError("DeepSeek route-weight storage must contain eight FP32 values")

    def schedule(self, sm):
        if sm != 0:
            return []
        return [
            Dsv4RouteTop6(self.hash_routing, self.route_scale),
            TmaLoad1D(self.logits),
            TmaLoad1D(self.bias),
            TmaLoad1D(self.hash_indices),
            TmaStore1D(self.output_indices).bar(self._bar("output")),
            TmaStore1D(self.output_weights),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4ExpertReduce(Schedule):
    def __init__(self, routed, weights, shared, output):
        super().__init__()
        self.routed = routed
        self.weights = weights
        self.shared = shared
        self.output = output

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek expert reduction currently uses exactly one SM")
        if self.routed.dtype != torch.bfloat16 or tuple(self.routed.shape) != (6, 4096):
            raise ValueError("routed expert outputs must be BF16 [6,4096]")
        if self.weights.dtype != torch.float32 or self.weights.numel() != 6:
            raise ValueError("expert weights must contain six FP32 values")
        if self.shared.dtype != torch.bfloat16 or self.shared.numel() != 4096:
            raise ValueError("shared expert output must contain 4096 BF16 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != 4096:
            raise ValueError("expert reduction output must contain 4096 BF16 values")

    def schedule(self, sm):
        if sm != 0:
            return []
        return [
            Dsv4ExpertReduce(),
            TmaLoad1D(self.routed),
            _shared_load_1d(self.weights),
            TmaLoad1D(self.shared),
            TmaStore1D(self.output).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4Fp32Bf16Gemv(Schedule):
    TILE_K = 8192

    def __init__(self, weight, input, output):
        super().__init__()
        self.weight = weight
        self.input = input
        self.output = output

    def _on_place(self):
        if self.weight.dtype != torch.float32 or self.weight.ndim != 2:
            raise ValueError("DeepSeek mHC weight must be rank-2 FP32")
        self.rows, self.k = self.weight.shape
        if self.input.dtype != torch.bfloat16 or self.input.numel() != self.k:
            raise ValueError("DeepSeek mHC input must be BF16 [K]")
        if self.output.dtype != torch.float32 or self.output.numel() != self.rows:
            raise ValueError("DeepSeek mHC GEMV output must be FP32 [rows]")
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek mHC GEMV requires 1 <= num_sms <= rows")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        instructions = []
        for local_row, row in enumerate(range(row_start, row_start + row_count)):
            instructions.append(Dsv4Fp32Bf16Gemv(self.k, self.TILE_K))
            for column in range(0, self.k, self.TILE_K):
                end = min(column + self.TILE_K, self.k)
                instructions += [
                    _shared_load_1d(self.weight[row, column:end]),
                    _shared_load_1d(self.input[column:end]),
                ]
            store = _shared_store_1d(self.output[row:row + 1])
            if local_row + 1 == row_count:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Bf16Gemv(Schedule):
    """Shard an unquantized checkpoint BF16 GEMV across resident SMs."""

    TILE_K = 16384

    def __init__(self, weight, input, output):
        super().__init__()
        self.weight = weight
        self.input = input
        self.output = output

    def _on_place(self):
        if self.weight.dtype != torch.bfloat16 or self.weight.ndim != 2:
            raise ValueError("DeepSeek BF16 GEMV weight must be rank-2 BF16")
        self.rows, self.k = self.weight.shape
        if self.input.dtype != torch.bfloat16 or self.input.numel() != self.k:
            raise ValueError("DeepSeek BF16 GEMV input must contain K BF16 values")
        if self.output.dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("DeepSeek BF16 GEMV output must be BF16 or FP32")
        if self.output.numel() != self.rows:
            raise ValueError("DeepSeek BF16 GEMV output must contain M values")
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek BF16 GEMV requires 1 <= num_sms <= M")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        instructions = []
        for local_row, row in enumerate(range(row_start, row_start + row_count)):
            instructions.append(Dsv4Bf16Gemv(
                self.k, self.TILE_K,
                output_fp32=self.output.dtype == torch.float32,
            ))
            for column in range(0, self.k, self.TILE_K):
                end = min(column + self.TILE_K, self.k)
                instructions += [
                    _shared_load_1d(self.weight[row, column:end]),
                    _shared_load_1d(self.input.reshape(-1)[column:end]),
                ]
            store = _shared_store_1d(self.output[row:row + 1])
            if local_row + 1 == row_count:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4HcPre(Schedule):
    def __init__(self, residual, mixes, scale, base, output, post, comb,
                 sinkhorn_iters=20, epsilon=1.0e-6):
        super().__init__()
        self.residual = residual
        self.mixes = mixes
        self.scale = scale
        self.base = base
        self.output = output
        self.post = post
        self.comb = comb
        self.sinkhorn_iters = sinkhorn_iters
        self.epsilon = epsilon

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek mHC pre currently uses exactly one SM")
        if self.residual.dtype != torch.bfloat16 or tuple(self.residual.shape) != (4, 4096):
            raise ValueError("mHC residual must be BF16 [4,4096]")
        if self.mixes.dtype != torch.float32 or self.mixes.numel() != 24:
            raise ValueError("mHC mixes must contain 24 FP32 values")
        if self.scale.dtype != torch.float32 or self.scale.numel() != 3:
            raise ValueError("mHC scale must contain three FP32 values")
        if self.base.dtype != torch.float32 or self.base.numel() != 24:
            raise ValueError("mHC base must contain 24 FP32 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != 4096:
            raise ValueError("mHC pre output must contain 4096 BF16 values")
        if self.post.dtype != torch.float32 or self.post.numel() != 4:
            raise ValueError("mHC post coefficients must contain four FP32 values")
        if self.comb.dtype != torch.float32 or tuple(self.comb.shape) != (4, 4):
            raise ValueError("mHC combination matrix must be FP32 [4,4]")

    def schedule(self, sm):
        if sm != 0:
            return []
        return [
            Dsv4HcPre(self.sinkhorn_iters, self.epsilon),
            TmaLoad1D(self.residual),
            TmaLoad1D(self.mixes),
            _shared_load_1d(self.scale),
            TmaLoad1D(self.base),
            TmaStore1D(self.output).bar(self._bar("output")),
            TmaStore1D(self.post),
            TmaStore1D(self.comb),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4HcPost(Schedule):
    def __init__(self, branch, residual, post, comb, output):
        super().__init__()
        self.branch = branch
        self.residual = residual
        self.post = post
        self.comb = comb
        self.output = output

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek mHC post currently uses exactly one SM")
        if self.branch.dtype != torch.bfloat16 or self.branch.numel() != 4096:
            raise ValueError("mHC branch must contain 4096 BF16 values")
        if self.residual.dtype != torch.bfloat16 or tuple(self.residual.shape) != (4, 4096):
            raise ValueError("mHC residual must be BF16 [4,4096]")
        if self.post.dtype != torch.float32 or self.post.numel() != 4:
            raise ValueError("mHC post coefficients must contain four FP32 values")
        if self.comb.dtype != torch.float32 or tuple(self.comb.shape) != (4, 4):
            raise ValueError("mHC combination matrix must be FP32 [4,4]")
        if self.output.dtype != torch.bfloat16 or tuple(self.output.shape) != (4, 4096):
            raise ValueError("mHC post output must be BF16 [4,4096]")

    def schedule(self, sm):
        if sm != 0:
            return []
        return [
            Dsv4HcPost(),
            TmaLoad1D(self.branch),
            TmaLoad1D(self.residual),
            TmaLoad1D(self.post),
            TmaLoad1D(self.comb),
            TmaStore1D(self.output).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4Hadamard(Schedule):
    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output

    def _on_place(self):
        if (self.input.dtype != torch.bfloat16 or self.input.ndim != 2 or
                self.input.shape[1] not in (128, 512)):
            raise ValueError("DeepSeek Hadamard input must be BF16 [rows,128|512]")
        if self.output.dtype != torch.bfloat16 or self.output.shape != self.input.shape:
            raise ValueError("DeepSeek Hadamard output must match the input")
        self.rows, self.width = self.input.shape
        if self.num_sms != self.rows:
            raise ValueError("DeepSeek Hadamard uses one SM per row")

    def schedule(self, sm):
        if sm < 0:
            return []
        return [
            Dsv4Hadamard(self.width),
            TmaLoad1D(self.input[sm]),
            TmaStore1D(self.output[sm]).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4GatedPool(Schedule):
    def __init__(self, values, scores, output):
        super().__init__()
        self.values = values
        self.scores = scores
        self.output = output

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek gated pooling currently uses exactly one SM")
        if (self.values.dtype != torch.float32 or self.values.ndim != 2 or
                self.values.shape[1] not in (128, 512)):
            raise ValueError("DeepSeek gated-pool values must be FP32 [rows,128|512]")
        if self.scores.dtype != torch.float32 or self.scores.shape != self.values.shape:
            raise ValueError("DeepSeek gated-pool scores must match the values")
        if (self.output.dtype != torch.bfloat16 or self.output.ndim != 1 or
                self.output.shape[0] != self.values.shape[1]):
            raise ValueError("DeepSeek gated-pool output must be BF16 [width]")
        self.pool_rows, self.width = self.values.shape

    def schedule(self, sm):
        if sm != 0:
            return []
        row_bytes = self.width * self.values.element_size()
        instructions = [Dsv4GatedPool(self.pool_rows, self.width)]
        instructions += RepeatM.on(
            self.pool_rows,
            (TmaLoad1D(self.values[0]), row_bytes),
            (TmaLoad1D(self.scores[0]), row_bytes),
        )
        instructions.append(
            TmaStore1D(self.output).bar(self._bar("output"))
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4IndexScore(Schedule):
    TILE_ROWS = 240

    def __init__(self, q, kv, head_weights, output):
        super().__init__()
        self.q = q
        self.kv = kv
        self.head_weights = head_weights
        self.output = output

    def _on_place(self):
        if self.q.dtype != torch.bfloat16 or tuple(self.q.shape) != (64, 128):
            raise ValueError("DeepSeek index Q must be BF16 [64,128]")
        if (self.kv.dtype != torch.bfloat16 or self.kv.ndim != 2 or
                self.kv.shape[1] != 128):
            raise ValueError("DeepSeek index KV must be BF16 [rows,128]")
        self.rows = self.kv.shape[0]
        if self.head_weights.dtype != torch.float32 or self.head_weights.numel() != 64:
            raise ValueError("DeepSeek index head weights must be FP32 [64]")
        if self.output.dtype != torch.float32 or self.output.numel() != self.rows:
            raise ValueError("DeepSeek index scores must be FP32 [rows]")
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek index score requires 1 <= num_sms <= rows")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        instructions = []
        row_end = row_start + row_count
        for tile_start in range(row_start, row_end, self.TILE_ROWS):
            tile_end = min(tile_start + self.TILE_ROWS, row_end)
            instructions += [
                Dsv4IndexScore(tile_end - tile_start),
                TmaLoad1D(self.q),
                TmaLoad1D(self.kv[tile_start:tile_end]),
                TmaLoad1D(self.head_weights),
            ]
            store = _shared_store_1d(self.output[tile_start:tile_end])
            if tile_end == row_end:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4TopK512(Schedule):
    def __init__(self, scores, output, index_offset=0):
        super().__init__()
        self.scores = scores
        self.output = output
        self.index_offset = index_offset

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek index top-k currently uses exactly one SM")
        if self.scores.dtype != torch.float32 or self.scores.ndim != 1:
            raise ValueError("DeepSeek index scores must be FP32 [rows]")
        if (self.scores.numel() <= 0 or self.scores.numel() > 0xFFFF or
                self.output.numel() <= 0 or
                self.output.numel() > min(self.scores.numel(), 512)):
            raise ValueError("DeepSeek index top-k dimensions are invalid")
        if self.output.dtype != torch.int32 or self.output.ndim != 1:
            raise ValueError("DeepSeek index output must be int32 [topk]")

    def schedule(self, sm):
        if sm != 0:
            return []
        instructions = [
            Dsv4TopK512(
                self.scores.numel(), self.output.numel(), self.index_offset
            ),
        ]
        first = min(self.scores.numel(), 1024)
        instructions.append(_shared_load_1d(self.scores[:first]))
        for start in range(first, self.scores.numel(), 512):
            instructions.append(
                _shared_load_1d(self.scores[start:min(start + 512, self.scores.numel())])
            )
        instructions.append(
            _shared_store_1d(self.output).bar(self._bar("output"))
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4HcHead(Schedule):
    def __init__(self, residual, mixes, scale, base, output,
                 epsilon=1.0e-6):
        super().__init__()
        self.residual = residual
        self.mixes = mixes
        self.scale = scale
        self.base = base
        self.output = output
        self.epsilon = epsilon

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek mHC head currently uses exactly one SM")
        if self.residual.dtype != torch.bfloat16 or tuple(self.residual.shape) != (4, 4096):
            raise ValueError("mHC head residual must be BF16 [4,4096]")
        if self.mixes.dtype != torch.float32 or self.mixes.numel() != 4:
            raise ValueError("mHC head mixes must contain four FP32 values")
        if self.scale.dtype != torch.float32 or self.scale.numel() != 1:
            raise ValueError("mHC head scale must contain one FP32 value")
        if self.base.dtype != torch.float32 or self.base.numel() != 4:
            raise ValueError("mHC head base must contain four FP32 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != 4096:
            raise ValueError("mHC head output must contain 4096 BF16 values")

    def schedule(self, sm):
        if sm != 0:
            return []
        return [
            Dsv4HcHead(self.epsilon),
            TmaLoad1D(self.residual),
            TmaLoad1D(self.mixes),
            _shared_load_1d(self.scale),
            TmaLoad1D(self.base),
            TmaStore1D(self.output).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4Fp8Quant128(Schedule):
    def __init__(self, input, output, scale):
        super().__init__()
        self.input = input
        self.output = output
        self.scale = scale

    def _on_place(self):
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("DeepSeek FP8 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % 128:
            raise ValueError("DeepSeek FP8 quant K must be divisible by 128")
        self.blocks = self.k // 128
        if self.num_sms <= 0 or self.num_sms > self.blocks:
            raise ValueError(
                "DeepSeek FP8 quant requires 1 <= num_sms <= K/128"
            )
        if (self.output.dtype != torch.float8_e4m3fn or
                self.output.shape != self.input.shape):
            raise ValueError("DeepSeek FP8 quant output must be E4M3 [K]")
        if (self.scale.dtype != torch.float8_e8m0fnu or
                self.scale.numel() != self.k // 128):
            raise ValueError("DeepSeek FP8 quant scale must be UE8M0 [K/128]")

    def schedule(self, sm):
        if sm < 0:
            return []
        blocks_per_sm, extra = divmod(self.blocks, self.num_sms)
        block_start = sm * blocks_per_sm + min(sm, extra)
        block_count = blocks_per_sm + (1 if sm < extra else 0)
        source = self.input[block_start * 128:(block_start + block_count) * 128]
        quantized = self.output[block_start * 128:(block_start + block_count) * 128]
        scales = self.scale[block_start:block_start + block_count]
        return [
            Dsv4Fp8Quant128(block_count * 128),
            _shared_load_1d(source),
            _shared_store_1d(quantized),
            _shared_store_1d(scales).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Nvfp4Quant16(Schedule):
    def __init__(self, input, global_scale, output, scale):
        super().__init__()
        self.input = input
        self.global_scale = global_scale
        self.output = output
        self.scale = scale

    def _on_place(self):
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("DeepSeek NVFP4 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % 16:
            raise ValueError("DeepSeek NVFP4 quant K must be divisible by 16")
        self.blocks = self.k // 16
        if self.num_sms <= 0 or self.num_sms > self.blocks:
            raise ValueError(
                "DeepSeek NVFP4 quant requires 1 <= num_sms <= K/16"
            )
        if self.global_scale.dtype != torch.float32 or self.global_scale.numel() != 1:
            raise ValueError("DeepSeek NVFP4 global scale must be scalar FP32")
        if self.output.dtype != torch.uint8 or self.output.numel() != self.k // 2:
            raise ValueError("DeepSeek NVFP4 quant output must be packed uint8 [K/2]")
        if (self.scale.dtype != torch.float8_e4m3fn or
                self.scale.numel() != self.k // 16):
            raise ValueError("DeepSeek NVFP4 quant scale must be E4M3 [K/16]")

    def schedule(self, sm):
        if sm < 0:
            return []
        blocks_per_sm, extra = divmod(self.blocks, self.num_sms)
        block_start = sm * blocks_per_sm + min(sm, extra)
        block_count = blocks_per_sm + (1 if sm < extra else 0)
        source = self.input[block_start * 16:(block_start + block_count) * 16]
        quantized = self.output[block_start * 8:(block_start + block_count) * 8]
        scales = self.scale[block_start:block_start + block_count]
        return [
            Dsv4Nvfp4Quant16(block_count * 16),
            _shared_load_1d(source),
            _shared_load_1d(self.global_scale.reshape(-1)),
            _shared_store_1d(quantized),
            _shared_store_1d(scales).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedRoutedDsv4Nvfp4Quant16(Schedule):
    """Quantize with the selected expert's input scale resolved by LDU."""

    def __init__(
        self,
        routing_state,
        route_rank,
        scale_field,
        input,
        output,
        scale,
    ):
        super().__init__()
        self.routing_state = routing_state
        self.route_rank = route_rank
        self.scale_field = scale_field
        self.input = input
        self.output = output
        self.scale = scale

    def _on_place(self):
        if self.routing_state.device.type != "cuda":
            raise ValueError("routing state must be a CUDA tensor")
        if not 0 <= self.route_rank < RoutedTmaLoad1D.ROUTE_COUNT:
            raise ValueError("route_rank must be in [0, 6)")
        if not 0 <= self.scale_field <= RoutedTmaLoad1D.MAX_POINTER_FIELD:
            raise ValueError("routed scale field must fit in 13 bits")
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("DeepSeek NVFP4 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % 16:
            raise ValueError("DeepSeek NVFP4 quant K must be divisible by 16")
        self.blocks = self.k // 16
        if self.num_sms <= 0 or self.num_sms > self.blocks:
            raise ValueError("DeepSeek NVFP4 quant requires 1 <= num_sms <= K/16")
        if self.output.dtype != torch.uint8 or self.output.numel() != self.k // 2:
            raise ValueError("DeepSeek NVFP4 quant output must be packed uint8 [K/2]")
        if (
            self.scale.dtype != torch.float8_e4m3fn
            or self.scale.numel() != self.k // 16
        ):
            raise ValueError("DeepSeek NVFP4 quant scale must be E4M3 [K/16]")

    def schedule(self, sm):
        if sm < 0:
            return []
        route_bar = self._bar("route")
        if route_bar is None:
            raise ValueError("routed NVFP4 quant requires a route barrier")
        blocks_per_sm, extra = divmod(self.blocks, self.num_sms)
        block_start = sm * blocks_per_sm + min(sm, extra)
        block_count = blocks_per_sm + (1 if sm < extra else 0)
        source = self.input[block_start * 16:(block_start + block_count) * 16]
        quantized = self.output[block_start * 8:(block_start + block_count) * 8]
        scales = self.scale[block_start:block_start + block_count]
        return [
            Dsv4Nvfp4Quant16(block_count * 16),
            _shared_load_1d(source),
            RoutedTmaLoad1D(
                self.routing_state,
                self.route_rank,
                self.scale_field,
                16,
            ).bar(route_bar),
            _shared_store_1d(quantized),
            _shared_store_1d(scales).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SubgridSchedule(Schedule):
    """Place one schedule on a fixed subgrid of a larger queued stage."""

    def __init__(self, schedule, num_sms: int, base_sm: int = 0):
        super().__init__()
        self.inner = schedule
        self.subgrid_sms = int(num_sms)
        self.subgrid_base = int(base_sm)

    def _on_place(self):
        if self.subgrid_sms <= 0:
            raise ValueError("subgrid schedule requires at least one SM")
        if (
            self.subgrid_base < 0
            or self.subgrid_base + self.subgrid_sms > self.num_sms
        ):
            raise ValueError("subgrid placement exceeds its queued stage")
        inner = self.inner._clone()
        inner._bars.update(self._bars)
        self.placed_inner = inner.place(
            self.subgrid_sms, self.subgrid_base
        )

    def schedule(self, sm: int):
        return self.placed_inner(sm)

    def bar_release_count(self, role: str):
        self._require_placed()
        return self.placed_inner.bar_release_count(role)


class ListSchedule(Schedule):
    def __init__(self, items, lead_bars=None, tail_bars=None, warn_boundary_bars=False):
        super().__init__()
        self.items = list(items)
        self.lead_bars = set() if lead_bars is None else set(lead_bars)
        self.tail_bars = set() if tail_bars is None else set(tail_bars)
        self.warn_boundary_bars = warn_boundary_bars
        self._warned_boundary_roles = set()

    def _clone(self):
        clone = super()._clone()
        clone.items = [
            item._clone() if isinstance(item, Schedule) else item
            for item in self.items
        ]
        clone.lead_bars = self.lead_bars.copy()
        clone.tail_bars = self.tail_bars.copy()
        clone.warn_boundary_bars = self.warn_boundary_bars
        clone._warned_boundary_roles = self._warned_boundary_roles.copy()
        return clone

    def _schedule_items(self):
        return [item for item in self.items if isinstance(item, Schedule)]

    def warn_on_boundary_bars(self, enable=True):
        self.warn_boundary_bars = enable
        return self

    def _maybe_warn_boundary_bar(self, role: str, num_schedules: int):
        if not self.warn_boundary_bars or num_schedules <= 1:
            return
        if role in self._warned_boundary_roles:
            return
        if role in self.lead_bars or role in self.tail_bars:
            warnings.warn(
                f"ListSchedule bar('{role}', ...) only applies to boundary schedule(s); "
                "this may be insufficient for interior dependencies",
                stacklevel=3,
            )
            self._warned_boundary_roles.add(role)

    def _apply_boundary_bars(self):
        schedules = self._schedule_items()
        if not schedules:
            return

        first = schedules[0]
        last = schedules[-1]
        for role, bar_id in self._bars.items():
            self._maybe_warn_boundary_bar(role, len(schedules))
            applied = False
            if role in self.lead_bars:
                first.bar(role, bar_id)
                applied = True
            if role in self.tail_bars:
                last.bar(role, bar_id)
                applied = True
            if not applied:
                if first is last:
                    first.bar(role, bar_id)
                else:
                    raise ValueError(f"ListSchedule cannot route bar role '{role}'")

    def place(self, num_sms: int, base_sm: int = 0):
        clone = self._clone()
        clone.num_sms = num_sms
        clone.base_sm = base_sm
        clone.items = [
            item.place(num_sms, base_sm) if isinstance(item, Schedule) else item
            for item in clone.items
        ]
        clone._apply_boundary_bars()
        return clone

    def bar(self, role: str, bar_id: int):
        super().bar(role, bar_id)
        self._apply_boundary_bars()
        return self

    def __call__(self, sm: int):
        insts = []
        for item in self.items:
            if callable(item):
                insts.append(item(sm))
            else:
                insts.append(item)
        return insts

    def __getitem__(self, idx):
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def bar_release_count(self, role: str):
        return sum(
            item.bar_release_count(role)
            for item in self.items
            if isinstance(item, Schedule)
        )

    def collect_barrier_release_counts(self):
        counts = {}
        for item in self.items:
            if not isinstance(item, Schedule):
                continue
            for bar_id, count in item.collect_barrier_release_counts().items():
                counts[bar_id] = counts.get(bar_id, 0) + count
        return counts

class SchedCopy(Schedule):
    def __init__(self,
                 tmas,
                 size = None,
                 before_copy = None,
                 count = 1):
        super().__init__()
        self.tmas = tmas
        self.count = count
        self.before_copy = before_copy

        if size is None:
            assert tmas[0].size == tmas[1].size, "Size must be specified when load and store TMA sizes do not match"
            size = tmas[0].size
        self.size = size

    def schedule(self, sm: int):
        if sm < 0:
            return []

        load, store = self.tmas
        load = load.cord(sm)
        store = store.cord(sm)
        if self.before_copy is not None:
            load.jump()

        return [
            Copy(1, size = self.size),

            self.before_copy,
            load.bar(self._bar("load")),
            store.bar(self._bar("store")),
        ]

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedRope(Schedule):
    def __init__(self, Atom, tmas):
        super().__init__()
        self.Atom = Atom
        self.tmas = tmas

    def schedule(self, sm: int):
        if sm < 0:
            return []
        table, load, store = [tma.cord(sm) for tma in self.tmas]

        return [
            self.Atom(),

            table,
            load,
            store.bar(self._bar("store")).group(),
        ]

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedAttentionDecoding(Schedule):
    # for decoding, the Qtile len will always be 1
    def __init__(self, reqs: int, seq_len: int,
                 KV_BLOCK_SIZE : int, NUM_KV_HEADS : int,
                 matO : torch.Tensor,
                 tmas,
                 side_input=None,
                 k_store=None,
                 token_pos=None,
                 num_active_q=64,
                 seq_len_counter_reg: int | None = None,
                 num_kv_block_counter_reg: int | None = None,
                 max_loop_count: int = 1,
                 outer_seq_len_counter_reg: int | None = None,
                 outer_seq_len_counter_stride: int = 0,
                 swapped_qk_pv: bool = False,
                 q_head_bars=None,
                 kv_head_bars=None,
                 o_head_bars=None,
                 head_major: bool = False):
        super().__init__()
        self.reqs = reqs
        self.seq_len = seq_len
        self.num_heads = NUM_KV_HEADS
        self.matO = matO
        self.tmas = tmas
        self.side_input = side_input
        self.k_store = k_store
        self.token_pos = token_pos
        self.num_active_q = num_active_q
        self.seq_len_counter_reg = seq_len_counter_reg
        self.num_kv_block_counter_reg = num_kv_block_counter_reg
        self.max_loop_count = max_loop_count
        self.outer_seq_len_counter_reg = outer_seq_len_counter_reg
        self.outer_seq_len_counter_stride = outer_seq_len_counter_stride
        self.swapped_qk_pv = swapped_qk_pv
        self.q_head_bars = q_head_bars
        self.kv_head_bars = kv_head_bars
        self.o_head_bars = o_head_bars
        self.head_major = head_major
        if (q_head_bars is None) != (kv_head_bars is None):
            raise ValueError("q_head_bars and kv_head_bars must be provided together")
        if q_head_bars is not None:
            if len(q_head_bars) != NUM_KV_HEADS or len(kv_head_bars) != NUM_KV_HEADS:
                raise ValueError("head-barrier arrays must match NUM_KV_HEADS")
        if o_head_bars is not None and len(o_head_bars) != NUM_KV_HEADS:
            raise ValueError("output head-barrier array must match NUM_KV_HEADS")
        self.required_sms = reqs * NUM_KV_HEADS
        self.block_size = KV_BLOCK_SIZE
        self.use_qwen_fused_qk = side_input is not None
        self.direct_output = matO.shape[-1] == 128 and not self.use_qwen_fused_qk
        if swapped_qk_pv:
            if not self.direct_output or self.block_size != 128:
                raise ValueError(
                    "swapped SM100 decode requires direct HDIM128 output and KV128"
                )
            if self.num_active_q > 4:
                raise ValueError(
                    "swapped SM100 decode supports at most four active GQA heads"
                )
            self.AttentionInst = ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT
        else:
            self.AttentionInst = select_attention_decode_instruction(
                matO.shape[-1], direct_output=self.direct_output
            )
        if self.use_qwen_fused_qk and not all(side is not None for side in (side_input, k_store, token_pos)):
            raise ValueError("SchedAttentionDecoding requires side_input, k_store, and token_pos together for the fused Qwen path")

    def _on_place(self):
        assert self.num_sms == self.required_sms, f"SchedAttentionDecoding requires {self.required_sms} SMs, got {self.num_sms}"

    def _map_req_head(self, sm: int):
        if self.head_major:
            return sm % self.reqs, sm // self.reqs
        return sm // self.num_heads, sm % self.num_heads

    def schedule(self, sm: int):
        if sm < 0:
            return []

        req, head = self._map_req_head(sm)
        head_dim = self.matO.shape[-1]
        q_bar = self.q_head_bars[head] if self.q_head_bars is not None else self._bar("q")
        kv_bar = self.kv_head_bars[head] if self.kv_head_bars is not None else self._bar("k")
        o_bar = self.o_head_bars[head] if self.o_head_bars is not None else self._bar("o")

        tQ, tK, tV = self.tmas

        num_kv_blocks = (self.seq_len + self.block_size - 1) // self.block_size
        seq_len_last_block = self.seq_len % self.block_size
        if seq_len_last_block == 0:
            seq_len_last_block = self.block_size
        if self.seq_len_counter_reg is not None:
            max_last_block_len = seq_len_last_block + self.max_loop_count - 1
            if max_last_block_len > self.block_size:
                raise ValueError(
                    "Dynamic decode attention sequence length would cross the current KV block: "
                    f"base={seq_len_last_block}, max_loop_count={self.max_loop_count}, block={self.block_size}"
                )

        # we only handle a single Q token here
        insts = [
            self.AttentionInst(
                num_kv_blocks,
                self.num_active_q,
                seq_len_last_block,
                need_norm=self.use_qwen_fused_qk,
                need_rope=self.use_qwen_fused_qk,
                seq_len_counter_reg=self.seq_len_counter_reg,
                num_kv_block_counter_reg=self.num_kv_block_counter_reg,
                kv_block_size=self.block_size,
                outer_seq_len_counter_reg=self.outer_seq_len_counter_reg,
                outer_seq_len_counter_stride=self.outer_seq_len_counter_stride,
            ),
        ]
        if self.use_qwen_fused_qk:
            insts += [
                self.side_input.cord(self.token_pos * 3 * head_dim).group(),
                self.k_store[req].cord((self.token_pos * self.num_heads + head) * head_dim).group(),
            ]
        insts += [
            tQ.cord(req, head).bar(q_bar).group(),
            RepeatM.on(num_kv_blocks - 1,
                # this k-barrier will also barrier following V load
                [tK.cord(req, 0, head, 0).port(1).group(), tK.cord2tma(0, self.block_size, 0, 0)],
                [tV.cord(req, 0, head, 0).port(1).group(), tV.cord2tma(0, self.block_size, 0, 0)],
                count_counter_reg=self.num_kv_block_counter_reg,
            ),
            # TODO(zhiyuang): reuse the accumulator register
            # only the last block has new generated KV cache
        ]
        last_k = tK.cord(req, self.block_size * (num_kv_blocks - 1), head, 0).bar(kv_bar).group().port(1)
        last_v = tV.cord(req, self.block_size * (num_kv_blocks - 1), head, 0).group().port(1)
        if self.num_kv_block_counter_reg is not None:
            last_k = RepeatM.offsetByCounter(self.num_kv_block_counter_reg, last_k, tK.cord2tma(0, self.block_size, 0, 0))
            last_v = RepeatM.offsetByCounter(self.num_kv_block_counter_reg, last_v, tV.cord2tma(0, self.block_size, 0, 0))
        insts += [last_k, last_v]
        if self.direct_output:
            output = (
                RawAddress(self.matO[req, head, ...], config.num_slots)
                .bar(o_bar)
                .writeback()
                .group()
            )
        else:
            output = (
                TmaStore1D(self.matO[req, head, ...], numSlots=2)
                .bar(o_bar)
                .group()
            )
        insts.append(output)
        return insts

    def bar_release_count(self, role: str):
        if role != "o":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedAttention(Schedule):
    def __init__(self,
                 reqs : int,
                 active_new_len: int,
                 cached_seq_len: int,
                 QKVHdim: tuple[int, int, int],
                 QKVTile: tuple[int, int],
                 QKVSeqlen: tuple[int, int],
                 tmas: tuple[TmaTensor],
                 need_norm: bool,
                 need_rope: bool,
                 rope_table: RawAddress):
        super().__init__()
        self.tmas = tmas
        self.reqs = reqs
        self.QKVHdim = QKVHdim
        self.QKVSeqlen = QKVSeqlen
        self.QKVTile = QKVTile
        self.active_new_len = active_new_len
        self.cached_seq_len = cached_seq_len
        self.need_norm = need_norm
        self.need_rope = need_rope
        self.rope_table = rope_table
        self.AttentionInst = select_attention_decode_instruction(QKVHdim[2])

        self.required_sms = reqs * QKVHdim[1]

    def _on_place(self):
        assert self.num_sms == self.required_sms, f"SchedAttention requires {self.required_sms} SMs, got {self.num_sms}"

    def describe(self):
        print(f"SchedAttention: reqs={self.reqs}, active_new_len={self.active_new_len}, QKVHdim={self.QKVHdim}, QKVSeqlen={self.QKVSeqlen}, QKVTile={self.QKVTile}, sms={self.num_sms}")
    
    def schedule(self, sm: int):
        if sm < 0:
            return []

        tQ, tK, tV, tO = self.tmas
        
        NUM_Q_HEAD, NUM_KV_HEAD, HEAD_DIM = self.QKVHdim
        Q_SEQ_LEN, KV_SEQ_LEN = self.QKVSeqlen
        assert self.active_new_len <= Q_SEQ_LEN, "Active new length cannot exceed maximum Q sequence length"
        # KV_SEQ_LEN assume to be new KV
        assert self.cached_seq_len <= KV_SEQ_LEN, "Cached sequence length cannot exceed maximum KV sequence length"
        QTile, KVTile = self.QKVTile

        HEAD_GROUP_SIZE = NUM_Q_HEAD // NUM_KV_HEAD

        # TODO(zhiyuang): why this mapping?
        head = sm % NUM_KV_HEAD
        req = sm // NUM_KV_HEAD

        insts = []
        for q in range(0, self.active_new_len, QTile):
            insts += [
                self.AttentionInst(min(self.active_new_len-q, QTile), hist_len=q+self.cached_seq_len, need_norm=self.need_norm, need_rope=self.need_rope),
                tQ.cord(req, q * HEAD_GROUP_SIZE, head, 0).bar(self._bar("q")).group(),
                self.rope_table if self.need_rope else [],
                # FIXME (zijian): this calculation should separate cached kv and new kv
                RepeatM.on((self.cached_seq_len + self.active_new_len + KVTile - 1) // KVTile,
                    # this k-barrier will also barrier following V load
                    [tK.cord(req, 0, head, 0).bar(self._bar("k")).group(), tK.cord2tma(0, KVTile, 0, 0)],
                    [tV.cord(req, 0, head, 0).group(), tV.cord2tma(0, KVTile, 0, 0)],
                ),
                tO.cord(req, q * HEAD_GROUP_SIZE, head, 0).port(1).bar(self._bar("o")).group(),
            ]
        return insts

    def bar_release_count(self, role: str):
        if role != "o":
            return 0
        q_tile = self.QKVTile[0]
        q_iters = (self.active_new_len + q_tile - 1) // q_tile
        return self._bar_release_if_present(role, self.num_sms * q_iters)


class SchedGemv(Schedule):
    def __init__(self, Atom,
                 MNK: tuple[int, int, int],
                 tmas: tuple[TmaTensor],
                 fold : int | None = None,
                 exec = True,
                 prefetch = True,
                 group = True):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas

        TileM, TileN, TileK = Atom.MNK
        # process MNK
        MNK_base = []
        MNK_size = []
        assert len(MNK) == 3, "MNK must be a tuple of 3 dimensions"
        for dim in MNK:
            if isinstance(dim, int):
                MNK_base.append(0)
                MNK_size.append(dim)
            elif isinstance(dim, tuple) and len(dim) == 2:
                base, size = dim
                MNK_base.append(base)
                MNK_size.append(size)
            else:
                raise ValueError(f"Invalid MNK dimension: {dim}")
        
        self.MNK = MNK_size
        self.MNK_base = MNK_base

        self.fold = fold
        self.exec = exec
        self.prefetch = prefetch
        self.group = group
        self.sm_per_fold = None
        self.k_per_fold = None

    def _on_place(self):
        TileM, _, _ = self.Atom.MNK
        M, _, K = self.MNK

        if self.fold is None:
            assert self.num_sms % (M // TileM) == 0, f"SMS must be multiple of M tiles when auto folding, got SMS={self.num_sms}, M={M}, TileM={TileM}"
            self.fold = self.num_sms // (M // TileM)

        self.sm_per_fold = self.num_sms // self.fold
        self.k_per_fold = K // self.fold
        self.validate()
    
    def validate(self):
        # TODO(zhiyuang): more validation on fold?
        TileM, TileN, TileK = self.Atom.MNK
        M, N, K = self.MNK
        min_k_per_fold = TileK * self.Atom.n_batch

        assert K % TileK == 0
        assert M % TileM == 0
        assert N % TileN == 0

        assert self.MNK_base[1] == 0, "N dimension must start from 0 for current schedule design"
        assert self.MNK[1] == N, "N dimension must cover the whole range for current schedule design"

        # verify fold
        assert self.num_sms % self.fold == 0
        assert K % self.fold == 0
        assert self.sm_per_fold == (M // TileM), "Invalid fold for given SMS and M size"
        assert self.k_per_fold % TileK == 0, "Invalid fold for given K size"
        assert self.k_per_fold % min_k_per_fold == 0, (
            f"Invalid fold for {self.Atom.__name__}: k_per_fold={self.k_per_fold} must be a multiple of "
            f"TileK * n_batch = {TileK} * {self.Atom.n_batch} = {min_k_per_fold}"
        )
        assert self.k_per_fold >= min_k_per_fold, (
            f"Invalid fold for {self.Atom.__name__}: k_per_fold={self.k_per_fold} must be at least "
            f"TileK * n_batch = {TileK} * {self.Atom.n_batch} = {min_k_per_fold}"
        )

        # verify storeC. if fold > 1, storeC must be reduction
        assert len(self.tmas) == 3, "Expect at 3 TMA tensors: loadA, loadB, storeC"
        if self.fold > 1:
            assert self.tmas[-1].mode == "reduce", f"storeC must be reduction mode when fold > 1, got mode {self.tmas[-1].mode}"
    
    def schedule(self, sm: int):
        # TODO(zhiyuang): different SM mode?
        if sm < 0:
            return []

        TileM, TileN, TileK = self.Atom.MNK
        baseM, _, baseK = self.MNK_base
        n_batch = self.Atom.n_batch

        loadA, loadB, storeC = self.tmas

        m = baseM + (sm % self.sm_per_fold) * TileM
        k = baseK + (sm // self.sm_per_fold) * self.k_per_fold

        n_repeat = self.k_per_fold // (TileK * n_batch)

        # TODO(zhiyuang): more detailed group control
        load_group = self.group and (self._bar("load") is not None)
        store_group = self.group and (self._bar("store") is not None)

        storeC_cord = storeC.cord(0, m)

        insts = [
            self.Atom(self.k_per_fold // TileK),

            RepeatM.onSync(0, self._bar("load"), n_repeat,
                (loadB.cord(0, k).group(load_group), loadB.cord2tma(0, TileK * n_batch)),
                *[
                    (loadA.cord(m, k + TileK * i).group(load_group), loadA.cord2tma(0, TileK * n_batch))
                    for i in range(n_batch)
                ],
                asyncPort=self.prefetch,
            ),

            storeC_cord.bar(self._bar("store")).group(store_group),
        ]
        return insts
    
    # combinators
    def split(self, dim: int, div: int):
        # create N new schedules that split the given dim by div
        assert dim in (0, 1, 2), "dim must be 0 (M), 1 (N), or 2 (K)"
        assert self.MNK[dim] % div == 0, "Cannot split dimension that is not divisible by div"

        new_schedules = []
        for i in range(div):
            new_MNK = list(self.MNK)
            new_base = list(self.MNK_base)
            size = new_MNK[dim] // div
            base = new_base[dim] + i * size
            new_MNK[dim] = size
            new_base[dim] = base
            new_schedule = SchedGemv(
                self.Atom,
                ((new_base[0], new_MNK[0]),
                 (new_base[1], new_MNK[1]),
                 (new_base[2], new_MNK[2])),
                self.tmas,
                fold=self.fold,
                prefetch=self.prefetch,
                group=self.group,
            )
            new_schedules.append(new_schedule)
        split_schedule = ListSchedule(new_schedules, lead_bars={"load"}, tail_bars={"store"})
        split_schedule._bars = self._bars.copy()
        if self.num_sms is not None:
            split_schedule = split_schedule.place(self.num_sms, self.base_sm)
        else:
            split_schedule._apply_boundary_bars()
        return split_schedule

    def split_M(self, div: int):
        return self.split(0, div)
    def split_K(self, div: int):
        return self.split(2, div)

    def no_prefetch(self):
        self.prefetch = False
        return self

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedGemvPhasedActivation(SchedGemv):
    """One GEMV task whose activation repeats observe staged producer bars.

    Consecutive repeats with the same barrier stay in one RepeatM program, so
    the compute opcode, tile shape, and normal weight/activation pipeline are
    unchanged. Only genuine producer boundaries add another memory segment.
    """

    def __init__(self, Atom, MNK, tmas, activation_bars,
                 fold: int | None = None, prefetch=True, group=True):
        super().__init__(
            Atom, MNK, tmas, fold=fold, prefetch=prefetch, group=group
        )
        self.activation_bars = list(activation_bars)

    def validate(self):
        super().validate()
        TileK = self.Atom.MNK[2]
        repeat_k = TileK * self.Atom.n_batch
        base_k = self.MNK_base[2]
        total_k = base_k + self.MNK[2]
        assert base_k % repeat_k == 0
        assert total_k % repeat_k == 0
        assert len(self.activation_bars) >= total_k // repeat_k
        assert self._bar("load") is None, (
            "phased activation GEMV takes its load barriers explicitly"
        )

    def schedule(self, sm: int):
        if sm < 0:
            return []

        TileM, _, TileK = self.Atom.MNK
        baseM, _, baseK = self.MNK_base
        n_batch = self.Atom.n_batch
        loadA, loadB, storeC = self.tmas

        m = baseM + (sm % self.sm_per_fold) * TileM
        k = baseK + (sm // self.sm_per_fold) * self.k_per_fold
        repeat_k = TileK * n_batch
        n_repeat = self.k_per_fold // repeat_k
        store_group = self.group and self._bar("store") is not None

        phased_loads = []
        local_repeat = 0
        while local_repeat < n_repeat:
            global_repeat = k // repeat_k + local_repeat
            bar_id = self.activation_bars[global_repeat]
            run_length = 1
            while (
                local_repeat + run_length < n_repeat
                and self.activation_bars[global_repeat + run_length] == bar_id
            ):
                run_length += 1

            segment_k = k + local_repeat * repeat_k
            load_steps = [
                (loadB.cord(0, segment_k).group(),
                 loadB.cord2tma(0, repeat_k)),
                *[
                    (loadA.cord(m, segment_k + TileK * i).group(),
                     loadA.cord2tma(0, repeat_k))
                    for i in range(n_batch)
                ],
            ]
            phased_loads.extend(
                RepeatM.onSync(
                    0, bar_id, run_length, *load_steps,
                    asyncPort=self.prefetch,
                )
            )
            local_repeat += run_length

        return [
            self.Atom(self.k_per_fold // TileK),
            phased_loads,
            storeC.cord(0, m).bar(self._bar("store")).group(store_group),
        ]


class SchedGemvUpSiLU(SchedGemv):
    """Up projection whose final UMMA group overlaps gate SiLU work."""

    def __init__(self, MNK, tmas, gate_reg: int):
        super().__init__(Gemv_M64N8UpSiLU, MNK, tmas)
        self.gate_reg = gate_reg

    def schedule(self, sm: int):
        insts = super().schedule(sm)
        if not insts:
            return insts
        insts.insert(len(insts) - 1, RegLoad(self.gate_reg))
        return insts


class SchedGemvMGroup(Schedule):
    def __init__(self, Atom, MNK, tmas, direct_output,
                 direct_output_slot: int = 24, group: bool = True):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas
        self.direct_output = direct_output
        self.direct_output_slot = direct_output_slot
        self.group = group

    def _on_place(self):
        TileM, TileN, TileK = self.Atom.MNK
        M, N, K = self.MNK
        assert M == self.num_sms * TileM * self.Atom.output_groups
        assert N == TileN
        assert K % (TileK * self.Atom.n_batch) == 0
        assert tuple(self.direct_output.shape) == (N, M)
        assert len(self.tmas) == 2

    def schedule(self, sm: int):
        if sm < 0:
            return []

        TileM, _, TileK = self.Atom.MNK
        _, _, K = self.MNK
        loadA, loadB = self.tmas
        m = sm * TileM
        output_group_stride = self.num_sms * TileM
        n_repeat = K // (TileK * self.Atom.n_batch)
        load_group = self.group and self._bar("load") is not None
        store_group = self.group and self._bar("store") is not None

        load_steps = [
            (loadB.cord(0, 0).group(load_group),
             loadB.cord2tma(0, TileK * self.Atom.n_batch)),
        ]
        for k_tile in range(self.Atom.n_batch):
            for output_group in range(self.Atom.output_groups):
                group_m = m + output_group * output_group_stride
                load_a = loadA.cord(
                    group_m, k_tile * TileK
                ).group(load_group)
                load_steps.append((
                    load_a,
                    loadA.cord2tma(0, TileK * self.Atom.n_batch),
                ))

        output = (
            RawAddress(self.direct_output, self.direct_output_slot)
            .delta(m * self.direct_output.element_size())
            .bar(self._bar("store"))
            .writeback()
            .group(store_group)
        )
        return [
            self.Atom(K // TileK, self.direct_output.shape[-1],
                      output_group_stride),
            RepeatM.onSync(0, self._bar("load"), n_repeat, *load_steps),
            output,
        ]

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedGemvMGroupArgmax(Schedule):
    """Grouped LM-head GEMV that emits one argmax record per task/token."""

    def __init__(self, Atom, MNK, tmas, mat_out_partial,
                 vocabulary_base: int, partial_base: int,
                 partial_slot: int = 30, group: bool = False):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas
        self.mat_out_partial = mat_out_partial
        self.vocabulary_base = vocabulary_base
        self.partial_base = partial_base
        self.partial_slot = partial_slot
        self.group = group

    def _on_place(self):
        tile_m, tile_n, tile_k = self.Atom.MNK
        m, n, k = self.MNK
        assert m == self.num_sms * tile_m * self.Atom.output_groups
        assert n == tile_n
        assert k % (tile_k * self.Atom.n_batch) == 0
        assert self.mat_out_partial.shape[0] == n
        assert self.mat_out_partial.shape[2] == 16
        assert self.partial_base + self.num_sms <= self.mat_out_partial.shape[1]
        assert self.vocabulary_base % tile_m == 0
        assert len(self.tmas) == 2

    def schedule(self, sm: int):
        if sm < 0:
            return []

        tile_m, _, tile_k = self.Atom.MNK
        _, _, k = self.MNK
        load_a, load_b = self.tmas
        m = sm * tile_m
        output_group_stride = self.num_sms * tile_m
        n_repeat = k // (tile_k * self.Atom.n_batch)
        load_group = self.group and self._bar("load") is not None

        load_steps = [
            (load_b.cord(0, 0).group(load_group),
             load_b.cord2tma(0, tile_k * self.Atom.n_batch)),
        ]
        for k_tile in range(self.Atom.n_batch):
            for output_group in range(self.Atom.output_groups):
                group_m = m + output_group * output_group_stride
                load_steps.append((
                    load_a.cord(group_m, k_tile * tile_k).group(load_group),
                    load_a.cord2tma(0, tile_k * self.Atom.n_batch),
                ))

        partial = self.partial_base + sm
        partial_out = (
            RawAddress(self.mat_out_partial[0, partial], self.partial_slot)
            .bar(self._bar("partial"))
            .writeback()
        )
        return [
            self.Atom(
                k // tile_k,
                output_group_stride,
                self.vocabulary_base + m,
            ),
            RepeatM.onSync(0, self._bar("load"), n_repeat, *load_steps),
            partial_out,
        ]

    def bar_release_count(self, role: str):
        if role != "partial":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedGemvMGroupReduce(Schedule):
    def __init__(self, Atom, MNK, tmas, group: bool = True):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas
        self.group = group
        self.m_groups = None
        self.fold = None
        self.k_per_fold = None

    def _on_place(self):
        TileM, TileN, TileK = self.Atom.MNK
        M, N, K = self.MNK
        assert M % (TileM * self.Atom.output_groups) == 0
        assert N == TileN
        self.m_groups = M // (TileM * self.Atom.output_groups)
        assert self.num_sms % self.m_groups == 0
        self.fold = self.num_sms // self.m_groups
        assert K % self.fold == 0
        self.k_per_fold = K // self.fold
        assert self.k_per_fold % (TileK * self.Atom.n_batch) == 0
        assert len(self.tmas) == 3
        assert self.tmas[-1].mode == "reduce"

    def schedule(self, sm: int):
        if sm < 0:
            return []

        TileM, _, TileK = self.Atom.MNK
        loadA, loadB, storeC = self.tmas
        m = (sm % self.m_groups) * TileM
        k = (sm // self.m_groups) * self.k_per_fold
        output_group_stride = self.m_groups * TileM
        n_repeat = self.k_per_fold // (TileK * self.Atom.n_batch)
        load_group = self.group and self._bar("load") is not None
        store_group = self.group and self._bar("store") is not None

        load_steps = [
            (loadB.cord(0, k).group(load_group),
             loadB.cord2tma(0, TileK * self.Atom.n_batch)),
        ]
        for k_tile in range(self.Atom.n_batch):
            for output_group in range(self.Atom.output_groups):
                group_m = m + output_group * output_group_stride
                load_steps.append((
                    loadA.cord(group_m, k + k_tile * TileK).group(load_group),
                    loadA.cord2tma(0, TileK * self.Atom.n_batch),
                ))

        output = (
            storeC.cord(0, m)
            .bar(self._bar("store"))
            .group(store_group)
        )

        return [
            self.Atom(self.k_per_fold // TileK),
            RepeatM.onSync(0, self._bar("load"), n_repeat, *load_steps),
            output,
        ]

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedGemm(Schedule):
    def __init__(self, Atom,
                 MNK: tuple[int, int, int],
                 tmas: tuple[TmaTensor],
                 fold: int | None = None,
                 prefetch=True,
                 group=True):
        super().__init__()
        self.Atom = Atom
        self.tmas = tmas

        mnk_base = []
        mnk_size = []
        assert len(MNK) == 3, "MNK must be a tuple of 3 dimensions"
        for dim in MNK:
            if isinstance(dim, int):
                mnk_base.append(0)
                mnk_size.append(dim)
            elif isinstance(dim, tuple) and len(dim) == 2:
                base, size = dim
                mnk_base.append(base)
                mnk_size.append(size)
            else:
                raise ValueError(f"Invalid MNK dimension: {dim}")

        self.MNK = mnk_size
        self.MNK_base = mnk_base

        self.fold = fold
        self.prefetch = prefetch
        self.group = group
        self.m_tiles = None
        self.n_tiles = None
        self.tiles_per_fold = None
        self.total_workers = None
        self.k_per_fold = None

    def _on_place(self):
        tile_m, tile_n, _ = self.Atom.MNK
        m, n, k = self.MNK

        self.m_tiles = m // tile_m
        self.n_tiles = n // tile_n
        self.tiles_per_fold = self.m_tiles * self.n_tiles

        if self.fold is None:
            self.fold = max(1, self.num_sms // self.tiles_per_fold)

        self.total_workers = self.tiles_per_fold * self.fold
        self.k_per_fold = k // self.fold
        self.validate()

    def validate(self):
        tile_m, tile_n, tile_k = self.Atom.MNK
        m, n, k = self.MNK
        min_k_per_fold = tile_k * self.Atom.n_batch

        assert m % tile_m == 0
        assert n % tile_n == 0
        assert k % tile_k == 0

        assert self.fold >= 1
        assert self.num_sms % self.fold == 0
        assert k % self.fold == 0
        assert self.k_per_fold % tile_k == 0
        assert self.k_per_fold % min_k_per_fold == 0, (
            f"Invalid fold for {self.Atom.__name__}: k_per_fold={self.k_per_fold} must be a multiple of "
            f"TileK * n_batch = {tile_k} * {self.Atom.n_batch} = {min_k_per_fold}"
        )
        assert self.k_per_fold >= min_k_per_fold, (
            f"Invalid fold for {self.Atom.__name__}: k_per_fold={self.k_per_fold} must be at least "
            f"TileK * n_batch = {tile_k} * {self.Atom.n_batch} = {min_k_per_fold}"
        )

        assert len(self.tmas) == 3, "Expect at 3 TMA tensors: loadA, loadB, storeC"
        if self.fold > 1:
            assert self.tmas[-1].mode == "reduce", (
                f"storeC must be reduction mode when fold > 1, got mode {self.tmas[-1].mode}"
            )

    def schedule(self, sm: int):
        if sm < 0:
            return []

        tile_m, tile_n, tile_k = self.Atom.MNK
        base_m, base_n, base_k = self.MNK_base
        n_batch = self.Atom.n_batch
        loadA, loadB, storeC = self.tmas

        load_group = self.group and (self._bar("load") is not None)
        store_group = self.group and (self._bar("store") is not None)

        insts = []
        for worker in range(sm, self.total_workers, self.num_sms):
            tile_idx = worker % self.tiles_per_fold
            fold_idx = worker // self.tiles_per_fold

            m = base_m + (tile_idx % self.m_tiles) * tile_m
            n = base_n + (tile_idx // self.m_tiles) * tile_n
            k = base_k + fold_idx * self.k_per_fold

            n_repeat = self.k_per_fold // (tile_k * n_batch)

            insts.extend([
                self.Atom(self.k_per_fold // tile_k),
                RepeatM.onSync(0, self._bar("load"), n_repeat,
                    (loadB.cord(n, k).group(load_group), loadB.cord2tma(0, tile_k * n_batch)),
                    *[
                        (loadA.cord(m, k + tile_k * i).group(load_group), loadA.cord2tma(0, tile_k * n_batch))
                        for i in range(n_batch)
                    ],
                    asyncPort=self.prefetch,
                ),
                storeC.cord(m, n).bar(self._bar("store")).group(store_group),
            ])
        return insts

    def split(self, dim: int, div: int):
        assert dim in (0, 1, 2), "dim must be 0 (M), 1 (N), or 2 (K)"
        assert self.MNK[dim] % div == 0, "Cannot split dimension that is not divisible by div"

        new_schedules = []
        for i in range(div):
            new_mnk = list(self.MNK)
            new_base = list(self.MNK_base)
            size = new_mnk[dim] // div
            base = new_base[dim] + i * size
            new_mnk[dim] = size
            new_base[dim] = base
            new_schedule = SchedGemm(
                self.Atom,
                ((new_base[0], new_mnk[0]),
                 (new_base[1], new_mnk[1]),
                 (new_base[2], new_mnk[2])),
                self.tmas,
                fold=self.fold,
                prefetch=self.prefetch,
                group=self.group,
            )
            new_schedules.append(new_schedule)

        split_schedule = ListSchedule(new_schedules, lead_bars={"load"}, tail_bars={"store"})
        split_schedule._bars = self._bars.copy()
        if self.num_sms is not None:
            split_schedule = split_schedule.place(self.num_sms, self.base_sm)
        else:
            split_schedule._apply_boundary_bars()
        return split_schedule

    def split_M(self, div: int):
        return self.split(0, div)

    def split_N(self, div: int):
        return self.split(1, div)

    def split_K(self, div: int):
        return self.split(2, div)

    def no_prefetch(self):
        self.prefetch = False
        return self

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.total_workers)

class SchedGemvRope(Schedule):
    def __init__(self,
                 MNK: tuple[int, int, int],
                 tmas: tuple[TmaTensor],
                 rope_table: RawAddress,
                 hist_seq_len: int,
                 rope_counter_offsets=None,
                 Atom=Gemv_M64N8_ROPE_128,
                 ):
        super().__init__()
        self.Atom = Atom
        self.MNK = MNK
        self.tmas = tmas

        MNK_base = []
        MNK_size = []
        assert len(MNK) == 3, "MNK must be a tuple of 3 dimensions"
        for dim in MNK:
            if isinstance(dim, int):
                MNK_base.append(0)
                MNK_size.append(dim)
            elif isinstance(dim, tuple) and len(dim) == 2:
                base, size = dim
                MNK_base.append(base)
                MNK_size.append(size)
            else:
                raise ValueError(f"Invalid MNK dimension: {dim}")
        
        self.MNK = MNK_size
        self.MNK_base = MNK_base
        self.rope_table = rope_table
        self.hist_seq_len = hist_seq_len
        self.rope_counter_offsets = rope_counter_offsets or []

        self.fold = None
        self.prefetch = True
        self.sm_per_fold = None
        self.k_per_fold = None

    def _on_place(self):
        self.fold = self.num_sms // (self.MNK[0] // self.Atom.MNK[0])
        self.sm_per_fold = self.num_sms // self.fold
        self.k_per_fold = self.MNK[2] // self.fold
        self.validate()
    
    def validate(self):
        TileM, TileN, TileK = self.Atom.MNK
        M, N, K = self.MNK
        assert 128 % TileM == 0, "TileM must divide 128 for rope fusion"

        # verify fold
        assert self.num_sms % (M // TileM) == 0, f"SMS must be multiple of M tiles, got SMS={self.num_sms}, M={M}, TileM={TileM}"
        assert self.num_sms % self.fold == 0
        assert K % self.fold == 0
        assert self.sm_per_fold == (M // TileM), "Invalid fold for given SMS and M size"
        assert self.k_per_fold % TileK == 0, "Invalid fold for given K size"
    
    def schedule(self, sm: int):
        # TODO(zhiyuang): different SM mode?
        if sm < 0:
            return []

        TileM, TileN, TileK = self.Atom.MNK
        baseM, _, baseK = self.MNK_base
        n_batch = self.Atom.n_batch
        loadA, loadB, storeC = self.tmas

        m = baseM + (sm % self.sm_per_fold) * TileM
        k = baseK + (sm // self.sm_per_fold) * self.k_per_fold

        n_repeat = self.k_per_fold // (TileK * n_batch)

        rope_table = self.rope_table.copy().delta(self.hist_seq_len * 128 * 2)
        for counter_reg, delta in self.rope_counter_offsets:
            rope_table = CounterOffsetMemoryInstruction(
                counter_reg, rope_table, delta)

        insts = [
            self.Atom(self.k_per_fold // TileK, self.hist_seq_len, m % 128),
            rope_table,
            RepeatM.onSync(0, self._bar("load"), n_repeat,
                (loadB.cord(0, k).group(), loadB.cord2tma(0, TileK * n_batch)),
                *[
                    (loadA.cord(m, k + TileK * i).group(), loadA.cord2tma(0, TileK * n_batch))
                    for i in range(n_batch)
                ],
                asyncPort=self.prefetch,
            ),
            storeC.cord(0, m).bar(self._bar("store")).group(),
        ]
        return insts

    def bar_release_count(self, role: str):
        if role != "store":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedRMSShared(Schedule):
    def __init__(self,
                 num_token: int,
                 epsilon: float,
                 tmas,
                 hidden_size: int | None = None,
                 group: bool = True,
                 embedding = None):
        super().__init__()
        self.num_token = num_token
        self.epsilon = epsilon
        self.tmas = tmas
        self.hidden_size = hidden_size
        self.group = group
        self.embedding = embedding

    def _on_place(self):
        assert self.num_token % self.num_sms == 0, "Number of tokens must be divisible by number of SMs"
        self.workload_per_sm = self.num_token // self.num_sms

    def _resolve_hidden_size(self):
        if self.hidden_size is not None:
            return self.hidden_size

        weight = self.tmas[0]
        if hasattr(weight, "size") and weight.size % 2 == 0:
            return weight.size // 2
        raise ValueError("SchedRMSShared requires hidden_size or a byte-sized weight TMA")

    def schedule(self, sm):
        if sm < 0:
            return []

        hidden_size = self._resolve_hidden_size()
        per_token_size = hidden_size * 2
        start_token_id = sm * self.workload_per_sm
        weight, load, store = self.tmas

        load = load \
            .cord(per_token_size * start_token_id) \
            .bar(self._bar("input")).group(self.group)
        store = store \
            .cord(per_token_size * start_token_id) \
            .bar(self._bar("output")).group(self.group)
        if self.embedding is not None:
            load.jump()
        
        return [
            select_rms_smem_instruction(hidden_size)(self.workload_per_sm, self.epsilon),
            weight.group(),
            self.embedding,
            load,
            store,
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)

class SchedRMS(Schedule):
    def __init__(self,
                 num_token: int,
                 epsilon: float,
                 input_glob: torch.Tensor,
                 output_glob: torch.Tensor,
                 weights_glob: torch.Tensor | None = None,
                 hidden_size: int | None = None,
                 use_glob: bool = False,
                 group: bool = True,
                 embedding = None):
        super().__init__()
        self.num_token = num_token
        self.epsilon = epsilon
        self.input_glob = input_glob
        self.output_glob = output_glob
        if weights_glob is None:
            weights_glob = torch.ones(
                self.input_glob.shape[-1],
                dtype=self.input_glob.dtype,
                device=self.input_glob.device,
            )
        self.weights_glob = weights_glob
        self.hidden_size = hidden_size if hidden_size is not None else input_glob.shape[-1]
        self.use_glob = use_glob
        self.group = group
        self.embedding = embedding

    def _on_place(self):
        assert self.num_token % self.num_sms == 0, "Number of tokens must be divisible by number of SMs"
        self.workload_per_sm = self.num_token // self.num_sms
        # TODO (zijian): residual store in case when rms starts from SM128, we should consider fuse in the kernel

    def schedule(self, sm):
        if sm < 0:
            return []

        if sm < 128:
            # regular rms path
            start_token_id = sm * self.workload_per_sm
            if self.use_glob:
                weight = RawAddress(self.weights_glob, 26)
                load = RawAddress(self.input_glob[start_token_id:start_token_id+self.workload_per_sm], 24)
                store = RawAddress(self.output_glob[start_token_id:start_token_id+self.workload_per_sm], 25)
                kernel = select_rms_glob_instruction(self.hidden_size)
            else:
                loadTensors = self.input_glob[start_token_id:start_token_id+self.workload_per_sm]
                if len(loadTensors) == 1:
                    loadTensors = loadTensors[0]
                storeTensors = self.output_glob[start_token_id:start_token_id+self.workload_per_sm]
                if len(storeTensors) == 1:
                    storeTensors = storeTensors[0]
                
                weight = TmaLoad1D(self.weights_glob)
                load = TmaLoad1D(loadTensors)
                store = TmaStore1D(storeTensors)
                kernel = select_rms_smem_instruction(self.hidden_size)
                # TODO(zhiyuang): recheck this when refector on repeat is done.
                if self.embedding is not None:
                    load.jump()

            load = load.bar(self._bar("input")).group(self.group)
            store = store.bar(self._bar("output")).group(self.group)
            
            insts = [
                kernel(self.workload_per_sm, self.epsilon),
                weight,
                self.embedding,
                load,
                store,
            ]
        
        return insts

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedSiLU(Schedule):
    def __init__(self,
                 base_raw_slot: int,
                 num_token: int,
                 output_size: int,
                 gate_glob: torch.Tensor,
                 up_glob: torch.Tensor,
                 out_glob: torch.Tensor):
        super().__init__()
        self.base_raw_slot = base_raw_slot
        self.num_token = num_token
        self.output_size = output_size
        # NOTE[zijian]: pass in first row only to bypass contiguous check
        self.gate_glob = gate_glob
        self.up_glob = up_glob
        self.out_glob = out_glob

    def _on_place(self):
        assert self.output_size % self.num_sms == 0, "Output size must be divisible by number of SMs"
        self.workload = self.output_size // self.num_sms
    
    def schedule(self, sm):
        if sm < 0:
            return []
        k_offset = sm * self.workload
        gate_addr = RawAddress(self.gate_glob[k_offset:], self.base_raw_slot).bar(self._bar("gate"))
        up_addr = RawAddress(self.up_glob[k_offset:], self.base_raw_slot+1).bar(self._bar("up"))
        out_addr = RawAddress(self.out_glob[k_offset:], self.base_raw_slot+2).bar(self._bar("out")).writeback()
        insts = [
            SILU_MUL_F16_K_12288(self.num_token, self.workload),
            gate_addr,
            up_addr,
            out_addr,
        ]
        return insts

    def bar_release_count(self, role: str):
        if role not in ("gate", "up", "out"):
            return 0
        return self._bar_release_if_present(role, self.num_sms)

    def split(self, div: int, bars: list[tuple[int]]):
        assert self.output_size % div == 0, "Cannot split K dimension evenly"
        new_schedules = []
        new_output_size = self.output_size // div
        for i in range(div):
            gate_bar, up_bar, out_bar = bars[i]
            new_schedule = SchedSiLU(
                self.base_raw_slot + i * 3,
                self.num_token,
                new_output_size,
                self.gate_glob[i * new_output_size:(i + 1) * new_output_size],
                self.up_glob[i * new_output_size:(i + 1) * new_output_size],
                self.out_glob[i * new_output_size:(i + 1) * new_output_size],
            )
            new_schedule.bar("gate", gate_bar).bar("up", up_bar).bar("out", out_bar)
            new_schedules.append(new_schedule)
        split_schedule = ListSchedule(new_schedules, lead_bars={"gate", "up"}, tail_bars={"out"})
        split_schedule._bars = self._bars.copy()
        if self.num_sms is not None:
            split_schedule = split_schedule.place(self.num_sms, self.base_sm)
        else:
            split_schedule._apply_boundary_bars()
        return split_schedule

class SchedSmemSiLUInterleaved(Schedule):
    def __init__(self,
                 num_token: int,
                 gate_glob: torch.Tensor,
                 up_glob: torch.Tensor,
                 out_glob: torch.Tensor,
                 shards_per_token: int = 1,
                 fixed_shard_id: int | None = None,
                 swiglu_limit: float = 0.0):
        super().__init__()
        self.num_token = num_token
        self.gate_glob = gate_glob
        self.up_glob = up_glob
        self.out_glob = out_glob
        self.shards_per_token = shards_per_token
        self.fixed_shard_id = fixed_shard_id
        self.swiglu_limit = swiglu_limit

    def _on_place(self):
        if self.shards_per_token == 3:
            assert self.gate_glob.shape == self.up_glob.shape == self.out_glob.shape
            assert self.gate_glob.shape[-1] == 6144, (
                "Three-way SwiGLU sharding requires a 6144-element prefix"
            )
            if self.fixed_shard_id is None:
                assert self.num_sms == self.num_token * self.shards_per_token, (
                    "Three-way SwiGLU sharding requires one SM per token shard"
                )
            else:
                assert 0 <= self.fixed_shard_id < self.shards_per_token
                assert self.num_sms == self.num_token, (
                    "A fixed SwiGLU shard requires one SM per token"
                )
            self.tokens_per_sm = 1
            return
        assert self.shards_per_token == 1, "Supported SwiGLU shard counts are 1 and 3"
        if self.gate_glob.shape != self.up_glob.shape or self.gate_glob.shape != self.out_glob.shape:
            raise ValueError("SwiGLU gate, up, and output tensors must match")
        if self.gate_glob.shape[-1] not in (2048, 4096):
            raise ValueError("SwiGLU supports 2048- or 4096-wide rows")
        assert self.num_token % self.num_sms == 0, "Number of tokens must be divisible by number of SMs"
        self.tokens_per_sm = self.num_token // self.num_sms

    def schedule(self, sm):
        if sm < 0:
            return []

        if self.shards_per_token == 3:
            if self.fixed_shard_id is None:
                token_id = sm // self.shards_per_token
                shard_id = sm % self.shards_per_token
            else:
                token_id = sm
                shard_id = self.fixed_shard_id
            shard_width = self.gate_glob.shape[-1] // self.shards_per_token
            shard_start = shard_id * shard_width
            shard_end = shard_start + shard_width
            fine_input = self._bar(f"input{shard_id}")
            fine_output = self._bar(f"output{shard_id}")
            input_bar = fine_input if fine_input is not None else self._bar("input")
            output_bar = fine_output if fine_output is not None else self._bar("output")
            return [
                (Dsv4SiluClampMul2048(1, self.swiglu_limit)
                 if self.swiglu_limit > 0
                 else SILU_MUL_SHARED_BF16_K_2048_INTER(1)),
                TmaStore1D(self.out_glob[token_id, shard_start:shard_end])
                    .bar(output_bar).group(),
                TmaLoad1D(self.gate_glob[token_id, shard_start:shard_end])
                    .bar(input_bar).group(),
                TmaLoad1D(self.up_glob[token_id, shard_start:shard_end]),
            ]

        start_token_id = sm * self.tokens_per_sm
        end_token_id = (sm + 1) * self.tokens_per_sm
        insts = []
        for i in range(start_token_id, end_token_id):
            gate = TmaLoad1D(self.gate_glob[i])
            if i == start_token_id:
                gate = gate.bar(self._bar("input")).group()

            width = self.gate_glob.shape[-1]
            if width == 2048:
                silu = (
                    Dsv4SiluClampMul2048(1, self.swiglu_limit)
                    if self.swiglu_limit > 0
                    else SILU_MUL_SHARED_BF16_K_2048_INTER(1)
                )
            else:
                if self.swiglu_limit > 0:
                    raise ValueError(
                        "bounded SwiGLU is currently implemented for K=2048"
                    )
                silu = SILU_MUL_SHARED_BF16_K_4096_INTER(1)

            insts.extend([
                silu,
                TmaStore1D(self.out_glob[i]).bar(self._bar("output")).group(),
                gate,
                TmaLoad1D(self.up_glob[i]),
            ])
        return insts

    def bar_release_count(self, role: str):
        if role == "output":
            return self._bar_release_if_present(
                role, self.num_token * self.shards_per_token
            )
        if role.startswith("output") and role[6:].isdigit():
            shard_id = int(role[6:])
            if 0 <= shard_id < self.shards_per_token:
                return self._bar_release_if_present(role, self.num_token)
        return 0



class SchedRegSiLUFused(Schedule):
    def __init__(self,
                 num_token: int,
                 store_tma: TmaTensor,
                 reg_gate: int,
                 reg_up: int,
                 base_offset: int,
                 stride: int):
        super().__init__()
        self.num_token = num_token
        self.store_tma = store_tma
        self.reg_gate = reg_gate
        self.reg_up = reg_up
        self.base_offset = base_offset
        self.stride = stride

    def schedule(self, sm):
        if sm < 0:
            return []

        return [
            SILU_MUL_SHARED_BF16_K_64_SW128(self.num_token),
            self.store_tma.cord(0, self.base_offset + sm * self.stride).bar(self._bar("output")).group(),
            RegLoad(self.reg_gate),
            RegLoad(self.reg_up),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedSmemSiLU_K_4096_N_1(Schedule):
    def __init__(self,
                 gate_tma: TmaLoad1D,
                 up_tma: TmaLoad1D,
                 out_tma: TmaStore1D,
                 base_sm: int,
                 ):
        super().__init__()
        self.base_sm = base_sm

        self.gate_tma = gate_tma
        self.up_tma = up_tma
        self.out_tma = out_tma

    def __call__(self, sm: int):
        return self.schedule(sm)
    
    def schedule(self, sm):
        if sm != self.base_sm:
            # only 1 SM is needed
            return []
        insts = [
            SILU_MUL_SHARED_BF16_K_4096(),
            self.out_tma,
            self.gate_tma,
            self.up_tma,
        ]
        return insts

class SchedArgmaxSmemPartial(Schedule):
    """Reduce disjoint BF16 logit ranges to compact shared/STU records."""

    RECORD_BYTES = 16

    def __init__(self, logits: torch.Tensor, partials: torch.Tensor):
        super().__init__()
        self.logits = logits
        self.partials = partials

    def _on_place(self):
        if (self.logits.dtype != torch.bfloat16 or self.logits.ndim != 1 or
                not self.logits.is_contiguous()):
            raise ValueError("shared argmax logits must be contiguous rank-1 BF16")
        if self.logits.numel() % 8:
            raise ValueError("shared argmax logits must contain a multiple of 8 values")
        if self.num_sms > self.logits.numel() // 8:
            raise ValueError("shared argmax requires at least eight logits per SM")
        if (self.partials.dtype != torch.uint8 or
                tuple(self.partials.shape) != (self.num_sms, self.RECORD_BYTES) or
                not self.partials.is_contiguous()):
            raise ValueError(
                "shared argmax partials must be contiguous uint8 [num_sms,16]"
            )

    def schedule(self, sm: int):
        if sm < 0:
            return []
        _, row_start, row_count = _aligned_row_shard(
            self.logits.numel(), self.num_sms, sm, alignment=8
        )
        return [
            ArgmaxSmemPartialBf16(row_count, row_start),
            _shared_load_1d(self.logits[row_start:row_start + row_count]),
            _shared_store_1d(self.partials[sm]).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedArgmaxSmemReduce(Schedule):
    """Reduce compact absolute-index records to one int64 token."""

    def __init__(self, partials: torch.Tensor, output: torch.Tensor):
        super().__init__()
        self.partials = partials
        self.output = output

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("shared argmax reduction uses exactly one SM")
        if (self.partials.dtype != torch.uint8 or self.partials.ndim != 2 or
                self.partials.shape[1] != SchedArgmaxSmemPartial.RECORD_BYTES or
                not self.partials.is_contiguous()):
            raise ValueError("shared argmax partials must be contiguous uint8 [N,16]")
        if not 1 <= self.partials.shape[0] <= 0xFFFF:
            raise ValueError("shared argmax partial count must fit in uint16")
        if (self.output.dtype != torch.int64 or self.output.numel() != 1 or
                not self.output.is_contiguous()):
            raise ValueError("shared argmax output must be one contiguous int64")

    def schedule(self, sm: int):
        if sm < 0:
            return []
        return [
            ArgmaxSmemReduceBf16(self.partials.shape[0]),
            _shared_load_1d(self.partials),
            _shared_store_1d(self.output).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedArgmax(Schedule):
    def __init__(self,
                 num_token: int,
                 logits_slice: int,
                 num_slice: int,
                 AtomPartial,
                 AtomReduce,
                 matLogits: list[torch.Tensor],
                 matOutVal: torch.Tensor,
                 matOutIdx: torch.Tensor,
                 matFinalOut: torch.Tensor,
                 final_counter_reg: int | None = None,
                 final_counter_stride: int = 0,
                 final_counter_offsets: list[tuple[int, int]] | None = None):
        super().__init__()
        self.num_token = num_token
        self.logits_slice = logits_slice
        self.num_slice = num_slice
        self.matLogits = matLogits
        self.matOutVal = matOutVal
        self.matOutIdx = matOutIdx
        self.matFinalOut = matFinalOut
        self.AtomPartial = AtomPartial
        self.AtomReduce = AtomReduce
        self.final_counter_reg = final_counter_reg
        self.final_counter_stride = final_counter_stride
        self.final_counter_offsets = final_counter_offsets
    
    def _on_place(self):
        self.validate()

    def validate(self):
        assert len(self.matLogits) == self.num_slice, "Number of logits slices must match vocab size and slice size"
        assert self.matOutVal.shape == (self.num_token, self.num_sms)
        assert self.matOutIdx.shape == (self.num_token, self.num_sms)
        assert self.matFinalOut.shape == (self.num_token,)

        sm_per_slice = self.num_sms // self.num_slice
        assert self.num_sms % self.num_slice == 0, "Number of SMs must be divisible by number of slices for current schedule design"
        c_per_sm = self.logits_slice // sm_per_slice 
        assert self.logits_slice % c_per_sm == 0, "Logits slice size must be divisible by chunk size per SM for current schedule design"
        assert self.AtomPartial.CHUNK_SIZE == c_per_sm, f"AtomPartial chunk size missmatch, expected {c_per_sm}, got {self.AtomPartial.CHUNK_SIZE}"
        assert self.AtomPartial.I_STRIDE == self.logits_slice, f"AtomPartial i_stride missmatch, expected {self.logits_slice}, got {self.AtomPartial.I_STRIDE}"
        assert self.AtomPartial.SMS == self.num_sms, f"AtomPartial SMS missmatch, expected {self.num_sms}, got {self.AtomPartial.SMS}"
        assert self.AtomReduce.CHUNK_SIZE == c_per_sm, f"AtomReduce chunk size missmatch, expected {c_per_sm}, got {self.AtomReduce.CHUNK_SIZE}"
        assert self.AtomReduce.SMS == self.num_sms, f"AtomReduce SMS missmatch, expected {self.num_sms}, got {self.AtomReduce.SMS}"

    def schedule(self, sm):
        if sm < 0:
            return []
        # decide which slice
        sm_per_slice = self.num_sms // self.num_slice
        slice_idx = sm // sm_per_slice
        c_per_sm = self.logits_slice // sm_per_slice
        slice_ofst = (sm % sm_per_slice) * c_per_sm
        insts = [
            self.AtomPartial(self.num_token),
            # FIXME(zhiyuang): the index 0 for batched mode?
            RawAddress(self.matLogits[slice_idx][0,slice_ofst], 24).bar(self._bar("load")),
            RawAddress(self.matOutVal[0,sm], 25).bar(self._bar("val")).writeback(),
            RawAddress(self.matOutIdx[0,sm], 26).bar(self._bar("idx")).writeback(),
        ]
        if sm >= self.num_token:
            return insts

        final_out = RawAddress(self.matFinalOut[sm], 29).bar(self._bar("final")).writeback()
        final_counter_offsets = self.final_counter_offsets
        if final_counter_offsets is None and self.final_counter_reg is not None:
            final_counter_offsets = [(self.final_counter_reg, self.final_counter_stride)]
        if final_counter_offsets:
            final_out = RepeatM.offsetByCounters(final_counter_offsets, final_out)

        insts += [
            self.AtomReduce(1),

            RawAddress(self.matOutVal[sm], 27).bar(self._bar("val")),
            RawAddress(self.matOutIdx[sm], 28).bar(self._bar("idx")),
            final_out,
        ]
        return insts

    def bar_release_count(self, role: str):
        if role == "val" or role == "idx":
            return self._bar_release_if_present(role, self.num_sms)
        if role == "final":
            return self._bar_release_if_present(role, min(self.num_token, self.num_sms))
        return 0


class SchedArgmaxReduceGlobal(Schedule):
    """Reduce absolute-index argmax records produced by fused LM-head tasks."""

    def __init__(self, num_token: int, AtomReduce,
                 mat_out_partial: torch.Tensor,
                 mat_final_out: torch.Tensor,
                 final_counter_offsets: list[tuple[int, int]] | None = None):
        super().__init__()
        self.num_token = num_token
        self.AtomReduce = AtomReduce
        self.mat_out_partial = mat_out_partial
        self.mat_final_out = mat_final_out
        self.final_counter_offsets = final_counter_offsets

    def _on_place(self):
        assert self.num_sms == self.num_token
        assert self.mat_out_partial.shape == (
            self.num_token, self.AtomReduce.PARTIAL_TASKS, 16)
        assert self.mat_final_out.shape == (self.num_token,)

    def schedule(self, sm: int):
        if sm < 0:
            return []
        final_out = (
            RawAddress(self.mat_final_out[sm], 29)
            .bar(self._bar("final"))
            .writeback()
        )
        if self.final_counter_offsets:
            final_out = RepeatM.offsetByCounters(
                self.final_counter_offsets, final_out)
        return [
            self.AtomReduce(1),
            RawAddress(self.mat_out_partial[sm], 27)
            .bar(self._bar("partial")),
            final_out,
        ]

    def bar_release_count(self, role: str):
        if role != "final":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


def interleave(*schedules):
    final = []
    for scheds in zip(*schedules):
        for sched in scheds:
            final.append(sched)
    return final
