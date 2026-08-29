import copy as pycopy
import warnings

from .runtime import *
from .launcher import *
from .instructions import Dsv4AttentionContext1Fp8Sm100
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
        for group_id, (representative, alternatives) in enumerate(
            self.tensor_groups
        ):
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


class LayerStateSchedule(LayeredSchedule):
    """Select persistent per-layer state in one compact loop body.

    Ordinary state reads use the same LDU-resolved pointer columns as
    :class:`LayeredSchedule`.  State writeback cannot use an indirect store
    opcode, so live state is allocated as evenly-strided layer-major storage
    and its destination address is stepped from the active loop counters.
    This keeps both directions in the normal allocator/LDU/STU protocol.
    """

    _STORE_OPS = {
        opcode.OP_ALLOC_WB_TMA_STORE_1D & ~((1 << 6) - 1),
        opcode.OP_ALLOC_WB_STU_STORE_1D & ~((1 << 6) - 1),
    }

    def _on_place(self):
        super()._on_place()
        self._group_strides = []
        for representative, alternatives in self.tensor_groups:
            if len(alternatives) == 1:
                self._group_strides.append(0)
                continue
            stride = alternatives[1].data_ptr() - alternatives[0].data_ptr()
            if stride <= 0 or any(
                tensor.data_ptr()
                != alternatives[0].data_ptr() + layer * stride
                for layer, tensor in enumerate(alternatives)
            ):
                stride = None
            self._group_strides.append(stride)

    def _transform_memory(self, inst):
        base_opcode = inst.opcode & ~((1 << 6) - 1)
        match = self._match_group(inst)
        if match is None:
            return inst
        group_id, _, _ = match
        if base_opcode in self._STORE_OPS:
            stride = self._group_strides[group_id]
            if stride is None:
                raise ValueError(
                    "layer-state writeback matched a tensor group without a "
                    f"constant byte stride; group={group_id}"
                )
            if stride == 0:
                return inst
            return RepeatM.offsetByCounters(
                tuple(
                    (counter, layer_stride * stride)
                    for counter, layer_stride in self.counter_strides
                ),
                inst,
            )
        return super()._transform_memory(inst)


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
        activation_mode="load",
        activation_tiles_per_load=None,
        pipeline=False,
    ):
        super().__init__()
        self.weight_tiles = weight_tiles
        self.activation_tiles = activation_tiles
        self.alpha = alpha
        self.output = output
        if activation_mode not in ("load", "retain", "reuse"):
            raise ValueError(
                "native activation_mode must be load, retain, or reuse"
            )
        self.activation_mode = activation_mode
        self.activation_tiles_per_load = (
            None
            if activation_tiles_per_load is None
            else int(activation_tiles_per_load)
        )
        self.pipeline = bool(pipeline)

    def _on_place(self):
        if (
            self.weight_tiles.dtype != torch.uint8
            or self.weight_tiles.ndim != 3
            or self.weight_tiles.shape[-1] != self.WEIGHT_TILE_BYTES
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
        if self.activation_mode == "load":
            if self.activation_tiles_per_load is None:
                self.activation_tiles_per_load = (
                    min(4, self.k_tiles) if self.pipeline else 1
                )
            if not 0 < self.activation_tiles_per_load <= self.k_tiles:
                raise ValueError(
                    "NVFP4 activation tiles per load must be in [1,K tiles]"
                )
            if (
                not self.pipeline
                and self.activation_tiles_per_load not in (1, self.k_tiles)
            ):
                raise ValueError(
                    "legacy NVFP4 UMMA supports one or all activation tiles "
                    "per load"
                )
        else:
            if self.activation_tiles_per_load not in (None, self.k_tiles):
                raise ValueError(
                    "retained/reused NVFP4 activation must cover all K tiles"
                )
            self.activation_tiles_per_load = self.k_tiles

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        if self.pipeline:
            compute = Nvfp4GemvUmmaPipelineSm100(
                self.k_tiles,
                retain_activation=self.activation_mode == "retain",
                activation_tiles_per_load=self.activation_tiles_per_load,
            )
        else:
            compute = Nvfp4GemvUmmaStreamSm100(
                self.k_tiles,
                retain_activation=self.activation_mode == "retain",
                bulk_activation=self.activation_tiles_per_load == self.k_tiles,
            )
        instructions = [compute, TmaLoad1D(self.alpha).fixed_port(1)]
        if (
            self.activation_mode == "load"
            and not self.pipeline
            and self.activation_tiles_per_load == self.k_tiles
        ):
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

        def append_weight(tile):
            instructions.append(
                TmaLoad1D(self.weight_tiles[sm, tile].reshape(-1)).fixed_port(0)
            )

        if self.pipeline and self.activation_mode == "load":
            for chunk_start in range(
                0, self.k_tiles, self.activation_tiles_per_load
            ):
                chunk_stop = min(
                    chunk_start + self.activation_tiles_per_load,
                    self.k_tiles,
                )
                instructions.append(
                    TmaLoad1D(
                        self.activation_tiles[chunk_start:chunk_stop].reshape(-1)
                    ).fixed_port(1)
                )
                for tile in range(chunk_start, chunk_stop):
                    append_weight(tile)
        else:
            for tile in range(self.k_tiles):
                append_weight(tile)
                if (
                    self.activation_mode == "load"
                    and self.activation_tiles_per_load == 1
                ):
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
        pipeline=False,
        activation_tiles_per_load=None,
        output_scale=None,
        swiglu_limit=None,
    ):
        super().__init__()
        self.routing_state = routing_state
        self.route_rank = route_rank
        self.weight_fields = tuple(weight_fields)
        self.alpha_field = alpha_field
        self.activation_tiles = activation_tiles
        self.output = output
        self.route_ready = bool(route_ready)
        if activation_mode not in ("load", "retain", "reuse"):
            raise ValueError(
                "routed native activation_mode must be load, retain, or reuse"
            )
        self.activation_mode = activation_mode
        if output_mode not in (
            "store", "retain", "silu_store", "fp32_store", "reduce"
        ):
            raise ValueError(
                "routed native output_mode must be store, retain, "
                "silu_store, fp32_store, or reduce"
            )
        if not 0 <= output_register < 4:
            raise ValueError("routed native output register must be in [0, 4)")
        if output_port not in (0, 1):
            raise ValueError("routed native output port must be 0 or 1")
        self.output_mode = output_mode
        self.output_register = output_register
        self.output_port = output_port
        self.pipeline = bool(pipeline)
        self.output_scale = output_scale
        self.swiglu_limit = (
            None if swiglu_limit is None else float(swiglu_limit)
        )
        self.activation_tiles_per_load = (
            None
            if activation_tiles_per_load is None
            else int(activation_tiles_per_load)
        )

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
        if self.activation_mode == "load":
            if self.activation_tiles_per_load is None:
                self.activation_tiles_per_load = (
                    min(4, self.k_tiles) if self.pipeline else self.k_tiles
                )
            if not 0 < self.activation_tiles_per_load <= self.k_tiles:
                raise ValueError(
                    "routed NVFP4 activation tiles per load must be in "
                    "[1,K tiles]"
                )
            if (
                not self.pipeline
                and self.activation_tiles_per_load not in (1, self.k_tiles)
            ):
                raise ValueError(
                    "legacy routed NVFP4 UMMA supports one or all activation "
                    "tiles per load"
                )
        else:
            if self.activation_tiles_per_load not in (None, self.k_tiles):
                raise ValueError(
                    "retained/reused routed activation must cover all K tiles"
                )
            self.activation_tiles_per_load = self.k_tiles
        for field in (
            *self.weight_fields,
            self.alpha_field,
        ):
            if not 0 <= field <= RoutedTmaLoad1D.MAX_POINTER_FIELD:
                raise ValueError("routed native pointer fields must fit in 13 bits")
        self.rows = self.m_tiles * self.TILE_M
        if self.output_mode in ("store", "silu_store"):
            if (
                self.output is None
                or self.output.dtype != torch.bfloat16
                or self.output.numel() != self.rows
            ):
                raise ValueError("routed native output must contain M BF16 values")
            if self.output_scale is not None:
                raise ValueError("stored routed output does not take an output scale")
        elif self.output_mode == "retain":
            if self.output is not None:
                raise ValueError(
                    "retained routed native output must not name HBM storage"
                )
            if self.output_scale is not None:
                raise ValueError("retained routed output does not take an output scale")
        elif self.output_mode == "fp32_store":
            if (
                self.output is None
                or self.output.dtype != torch.float32
                or self.output.numel() != self.rows
                or not self.output.is_contiguous()
            ):
                raise ValueError(
                    "routed FP32 output must contain one contiguous M vector"
                )
            if (
                self.output_scale is None
                or self.output_scale.dtype != torch.float32
                or self.output_scale.numel() < 1
                or not self.output_scale.is_contiguous()
            ):
                raise ValueError(
                    "routed FP32 output requires a contiguous FP32 scale"
                )
        else:
            reduce_output = getattr(self.output, "mat", None)
            if (
                getattr(self.output, "mode", None) != "reduce"
                or reduce_output is None
                or reduce_output.dtype != torch.float32
                or tuple(reduce_output.shape) != (1, self.rows)
                or not reduce_output.is_contiguous()
            ):
                raise ValueError(
                    "routed reduced output must be row-major FP32 [1,M]"
                )
            if (
                self.output_scale is None
                or self.output_scale.dtype != torch.float32
                or self.output_scale.numel() < 1
                or not self.output_scale.is_contiguous()
            ):
                raise ValueError(
                    "routed reduced output requires a contiguous FP32 scale"
                )
        if self.output_mode == "retain" and self.num_sms != self.m_tiles:
            raise ValueError(
                "retained routed output requires one M128 tile per SM"
            )
        if self.output_mode == "silu_store":
            if self.swiglu_limit is None or self.swiglu_limit <= 0:
                raise ValueError("fused routed SwiGLU requires a positive limit")
            if self.pipeline:
                raise ValueError(
                    "fused routed SwiGLU currently uses the normal stream task"
                )
            if self.num_sms != self.m_tiles:
                raise ValueError(
                    "fused routed SwiGLU requires one M128 tile per SM"
                )
        elif self.swiglu_limit is not None:
            raise ValueError("SwiGLU limit requires silu_store output mode")

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
        for local_tile, output_tile in enumerate(
            range(tile_start, tile_start + tile_count)
        ):
            first_output = local_tile == 0
            final_output = local_tile + 1 == tile_count
            if self.activation_mode == "retain":
                activation_kind = "retain" if first_output else "reuse"
            elif self.activation_mode == "reuse":
                activation_kind = "reuse"
            elif (
                self.pipeline
                and self.activation_tiles_per_load != self.k_tiles
            ):
                # A partial activation allocation cannot outlive this task.
                # Stream the requested number of K tiles for every output tile
                # instead of silently promoting the parameter to a full-K
                # retained allocation when an SM owns multiple output tiles.
                activation_kind = "load"
            elif first_output:
                activation_kind = "retain" if not final_output else "load"
            else:
                activation_kind = "reuse"
            retain = activation_kind == "retain" or (
                activation_kind == "reuse" and not final_output
            )
            task_activation_tiles_per_load = (
                self.activation_tiles_per_load
                if activation_kind == "load"
                else self.k_tiles
            )
            if (
                self.output_mode in ("fp32_store", "reduce")
                and self.pipeline
            ):
                compute = Nvfp4GemvUmmaPipelineFp32Sm100(
                    self.k_tiles,
                    retain_activation=retain,
                    activation_tiles_per_load=(
                        task_activation_tiles_per_load
                    ),
                )
            elif self.output_mode in ("fp32_store", "reduce"):
                compute = Nvfp4GemvUmmaFp32Sm100(
                    self.k_tiles,
                    retain_activation=retain,
                    bulk_activation=(
                        task_activation_tiles_per_load == self.k_tiles
                    ),
                )
            elif self.pipeline:
                compute = Nvfp4GemvUmmaPipelineSm100(
                    self.k_tiles,
                    retain_activation=retain,
                    activation_tiles_per_load=(
                        task_activation_tiles_per_load
                    ),
                )
            else:
                compute = Nvfp4GemvUmmaStreamSm100(
                    self.k_tiles,
                    retain_activation=retain,
                    bulk_activation=(
                        task_activation_tiles_per_load == self.k_tiles
                    ),
                    swiglu_limit=(
                        self.swiglu_limit
                        if self.output_mode == "silu_store"
                        else None
                    ),
                )
            instructions.append(compute)
            alpha_load = RoutedTmaLoad1D(
                self.routing_state,
                self.route_rank,
                self.alpha_field,
                16,
            ).fixed_port(1)
            if first_output and route_bar is not None:
                alpha_load.bar(route_bar)
            instructions.append(alpha_load)
            if self.output_mode in ("fp32_store", "reduce"):
                instructions.append(
                    _shared_load_1d(
                        self.output_scale, bytes=4
                    ).fixed_port(1)
                )

            if (
                activation_kind == "load"
                and not self.pipeline
                and task_activation_tiles_per_load == self.k_tiles
            ):
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

            def append_weight(k_tile):
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

            if self.pipeline and activation_kind == "load":
                for chunk_start in range(
                    0, self.k_tiles, task_activation_tiles_per_load
                ):
                    chunk_stop = min(
                        chunk_start + task_activation_tiles_per_load,
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
                        append_weight(k_tile)
            else:
                for k_tile in range(self.k_tiles):
                    append_weight(k_tile)
                    if (
                        activation_kind == "load"
                        and task_activation_tiles_per_load == 1
                    ):
                        instructions.append(
                            TmaLoad1D(
                                self.activation_tiles[k_tile].reshape(-1)
                            ).fixed_port(1)
                        )
            if self.output_mode == "silu_store":
                instructions.append(
                    RegLoad(self.output_register).fixed_port(self.output_port)
                )
            if self.output_mode in ("store", "silu_store"):
                row_start = output_tile * self.TILE_M
                store = TmaStore1D(
                    self.output[row_start : row_start + self.TILE_M]
                )
            elif self.output_mode == "fp32_store":
                row_start = output_tile * self.TILE_M
                store = TmaStore1D(
                    self.output[row_start : row_start + self.TILE_M]
                )
            elif self.output_mode == "reduce":
                row_start = output_tile * self.TILE_M
                store = self.output.cord(0, row_start)
            else:
                store = RegStore(
                    self.output_register,
                    size=self.TILE_M * 2,
                ).fixed_port(self.output_port)
            if final_output:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output" or self.output_mode == "retain":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedNvfp4GemvUmmaSplitK(Schedule):
    """Shard native NVFP4 K and TMA-reduce route-scaled FP32 partials."""

    TILE_M = SchedNvfp4UmmaPrepack.TILE_M
    WEIGHT_TILE_BYTES = SchedNvfp4UmmaPrepack.WEIGHT_TILE_BYTES
    ACTIVATION_TILE_BYTES = SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(
        self,
        weight_tiles,
        activation_tiles,
        alpha,
        output_scale,
        output_reduce,
        split_k: int,
    ):
        super().__init__()
        self.weight_tiles = weight_tiles
        self.activation_tiles = activation_tiles
        self.alpha = alpha
        self.output_scale = output_scale
        self.output_reduce = output_reduce
        self.split_k = int(split_k)

    def _on_place(self):
        if (
            self.weight_tiles.dtype != torch.uint8
            or self.weight_tiles.ndim != 3
            or self.weight_tiles.shape[2] != self.WEIGHT_TILE_BYTES
            or not self.weight_tiles.is_contiguous()
        ):
            raise ValueError(
                "split-K NVFP4 weights must be [M/128,K/256,18432] uint8"
            )
        self.m_tiles, self.k_tiles, _ = self.weight_tiles.shape
        if self.split_k <= 1 or self.k_tiles % self.split_k:
            raise ValueError(
                "split_k must divide NVFP4 K tiles and be greater than one"
            )
        self.k_tiles_per_split = self.k_tiles // self.split_k
        if (
            self.activation_tiles.dtype != torch.uint8
            or tuple(self.activation_tiles.shape)
            != (self.k_tiles, self.ACTIVATION_TILE_BYTES)
            or not self.activation_tiles.is_contiguous()
        ):
            raise ValueError(
                "split-K NVFP4 activations must be [K/256,3072] uint8"
            )
        if (
            self.alpha.dtype != torch.float32
            or self.alpha.numel() != 4
            or not self.alpha.is_contiguous()
        ):
            raise ValueError("split-K NVFP4 alpha storage must contain four FP32")
        if (
            self.output_scale.dtype != torch.float32
            or self.output_scale.numel() < 1
            or not self.output_scale.is_contiguous()
        ):
            raise ValueError("split-K NVFP4 output scale must contain FP32 data")
        self.rows = self.m_tiles * self.TILE_M
        output = getattr(self.output_reduce, "mat", None)
        if (
            getattr(self.output_reduce, "mode", None) != "reduce"
            or output is None
            or output.dtype != torch.float32
            or tuple(output.shape) != (1, self.rows)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "split-K NVFP4 output must be row-major FP32 reduce [1,M]"
            )
        self.work_tiles = self.m_tiles * self.split_k
        if not 0 < self.num_sms <= self.work_tiles:
            raise ValueError(
                f"split-K NVFP4 GEMV requires 1..{self.work_tiles} SMs"
            )

    def _work_shard(self, sm):
        work_per_sm, extra = divmod(self.work_tiles, self.num_sms)
        work_start = sm * work_per_sm + min(sm, extra)
        work_count = work_per_sm + int(sm < extra)
        return work_start, work_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        work_start, work_count = self._work_shard(sm)
        work_stop = work_start + work_count
        instructions = []
        for work in range(work_start, work_stop):
            split = work // self.m_tiles
            output_tile = work % self.m_tiles
            k_start = split * self.k_tiles_per_split
            k_stop = k_start + self.k_tiles_per_split
            instructions.extend(
                (
                    Nvfp4GemvUmmaFp32Sm100(self.k_tiles_per_split),
                    TmaLoad1D(self.alpha).fixed_port(1),
                    _shared_load_1d(self.output_scale, bytes=4).fixed_port(1),
                    TmaLoad1D(
                        self.activation_tiles[k_start:k_stop].reshape(-1)
                    ).fixed_port(1),
                )
            )
            for k_tile in range(k_start, k_stop):
                instructions.append(
                    TmaLoad1D(
                        self.weight_tiles[output_tile, k_tile].reshape(-1)
                    ).fixed_port(0)
                )
            store = self.output_reduce.cord(
                0, output_tile * self.TILE_M
            )
            if work + 1 == work_stop:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedRoutedNvfp4ExpertGroupSplitK(Schedule):
    """Wave-shard all routed experts for one projection or a gate/up pair."""

    TILE_M = SchedNvfp4UmmaPrepack.TILE_M
    WEIGHT_TILE_BYTES = SchedNvfp4UmmaPrepack.WEIGHT_TILE_BYTES
    ACTIVATION_TILE_BYTES = SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(
        self,
        routing_state,
        weight_field_groups,
        alpha_fields,
        activation_tiles,
        output_reduces,
        output_scale,
        split_k: int,
        *,
        route_ready=False,
        pipeline=False,
        activation_tiles_per_load=None,
        aggregate_routes=False,
    ):
        super().__init__()
        self.routing_state = routing_state
        self.weight_field_groups = tuple(
            tuple(tuple(split_fields) for split_fields in output_fields)
            for output_fields in weight_field_groups
        )
        self.alpha_fields = tuple(alpha_fields)
        self.activation_tiles = activation_tiles
        self.output_reduces = tuple(output_reduces)
        self.output_scale = output_scale
        self.split_k = int(split_k)
        self.route_ready = bool(route_ready)
        self.pipeline = bool(pipeline)
        self.activation_tiles_per_load = (
            None
            if activation_tiles_per_load is None
            else int(activation_tiles_per_load)
        )
        self.aggregate_routes = bool(aggregate_routes)

    def _on_place(self):
        if self.routing_state.device.type != "cuda":
            raise ValueError("grouped routed state must be on CUDA")
        self.output_groups = len(self.weight_field_groups)
        if self.output_groups not in (1, 2):
            raise ValueError("grouped routed split-K supports one or two outputs")
        if len(self.alpha_fields) != self.output_groups:
            raise ValueError("each grouped output requires one routed alpha field")
        if len(self.output_reduces) != self.output_groups:
            raise ValueError("each grouped output requires one TMA reduction tensor")
        if self.output_groups == 2 and not self.pipeline:
            raise ValueError("paired gate/up split-K requires the pipelined task")
        if (
            self.activation_tiles.dtype != torch.uint8
            or self.activation_tiles.ndim != 3
            or self.activation_tiles.shape[2] != self.ACTIVATION_TILE_BYTES
            or not self.activation_tiles.is_contiguous()
        ):
            raise ValueError(
                "grouped activations must be contiguous [routes,K/256,3072] uint8"
            )
        self.route_count, self.k_tiles, _ = self.activation_tiles.shape
        if not 0 < self.route_count <= RoutedTmaLoad1D.ROUTE_COUNT:
            raise ValueError("grouped routed split-K requires 1..6 routes")
        if self.split_k <= 1 or self.k_tiles % self.split_k:
            raise ValueError("grouped split_k must divide K tiles and exceed one")
        self.k_tiles_per_split = self.k_tiles // self.split_k
        if self.activation_tiles_per_load is None:
            self.activation_tiles_per_load = min(4, self.k_tiles_per_split)
        if not 0 < self.activation_tiles_per_load <= self.k_tiles_per_split:
            raise ValueError(
                "grouped activation tiles per load must be in [1,K/split_k]"
            )
        if not self.pipeline:
            self.activation_tiles_per_load = self.k_tiles_per_split
        m_tile_counts = {len(group) for group in self.weight_field_groups}
        if len(m_tile_counts) != 1 or next(iter(m_tile_counts)) <= 0:
            raise ValueError("grouped outputs must share one nonempty M tiling")
        self.m_tiles = next(iter(m_tile_counts))
        for group in self.weight_field_groups:
            for split_fields in group:
                if len(split_fields) != self.split_k:
                    raise ValueError(
                        "each grouped M tile requires one base field per K split"
                    )
                if any(
                    field < 0 or field > RoutedTmaLoad1D.MAX_POINTER_FIELD
                    for field in split_fields
                ):
                    raise ValueError("grouped routed fields must fit in 13 bits")
        if any(
            field < 0 or field > RoutedTmaLoad1D.MAX_POINTER_FIELD
            for field in self.alpha_fields
        ):
            raise ValueError("grouped routed alpha fields must fit in 13 bits")
        if (
            self.output_scale.dtype != torch.float32
            or self.output_scale.numel() not in (1, self.route_count)
            or not self.output_scale.is_contiguous()
        ):
            raise ValueError("grouped output scale must be scalar or one FP32/route")
        self.rows = self.m_tiles * self.TILE_M
        expected_rows = 1 if self.aggregate_routes else self.route_count
        for output_reduce in self.output_reduces:
            output = getattr(output_reduce, "mat", None)
            if (
                getattr(output_reduce, "mode", None) != "reduce"
                or output is None
                or output.dtype != torch.float32
                or tuple(output.shape) != (expected_rows, self.rows)
                or not output.is_contiguous()
            ):
                raise ValueError(
                    "grouped split-K output must be FP32 TMA reduce "
                    f"[{expected_rows},{self.rows}]"
                )
        self.work_tiles = self.route_count * self.m_tiles * self.split_k
        if not 0 < self.num_sms <= self.work_tiles:
            raise ValueError(
                f"grouped routed split-K requires 1..{self.work_tiles} SMs"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        route_bar = self._bar("route")
        if route_bar is None and not self.route_ready:
            raise ValueError("grouped routed split-K requires a route barrier")
        instructions = []
        first_work = True
        for work in range(sm, self.work_tiles, self.num_sms):
            inner_work, route_rank = divmod(work, self.route_count)
            split, output_tile = divmod(inner_work, self.m_tiles)
            k_start = split * self.k_tiles_per_split
            k_stop = k_start + self.k_tiles_per_split
            if self.output_groups == 2:
                compute = Nvfp4GemvUmmaPipelineFp32Group2Sm100(
                    self.k_tiles_per_split,
                    activation_tiles_per_load=self.activation_tiles_per_load,
                )
            elif self.pipeline:
                compute = Nvfp4GemvUmmaPipelineFp32Sm100(
                    self.k_tiles_per_split,
                    activation_tiles_per_load=self.activation_tiles_per_load,
                )
            else:
                compute = Nvfp4GemvUmmaFp32Sm100(
                    self.k_tiles_per_split,
                    bulk_activation=True,
                )
            instructions.append(compute)
            for alpha_field in self.alpha_fields:
                alpha_load = RoutedTmaLoad1D(
                    self.routing_state,
                    route_rank,
                    alpha_field,
                    16,
                ).fixed_port(1)
                if first_work and route_bar is not None:
                    alpha_load.bar(route_bar)
                instructions.append(alpha_load)
            scale = (
                self.output_scale.reshape(-1)[:1]
                if self.output_scale.numel() == 1
                else self.output_scale[route_rank : route_rank + 1]
            )
            instructions.append(
                _shared_load_1d(scale, bytes=4).fixed_port(1)
            )

            def append_weight(group, local_k):
                if local_k == 0:
                    load = RoutedTmaLoadBase1D(
                        self.routing_state,
                        route_rank,
                        self.weight_field_groups[group][output_tile][split],
                        self.WEIGHT_TILE_BYTES,
                    )
                else:
                    load = TmaLoadAddressReg1D(
                        RoutedTmaLoadBase1D.ADDRESS_REGISTER,
                        local_k * self.WEIGHT_TILE_BYTES,
                        self.WEIGHT_TILE_BYTES,
                    )
                if first_work and local_k == 0 and route_bar is not None:
                    load.bar(route_bar)
                instructions.append(load.fixed_port(group))

            if self.pipeline:
                for chunk_start in range(
                    0,
                    self.k_tiles_per_split,
                    self.activation_tiles_per_load,
                ):
                    chunk_stop = min(
                        chunk_start + self.activation_tiles_per_load,
                        self.k_tiles_per_split,
                    )
                    instructions.append(
                        TmaLoad1D(
                            self.activation_tiles[
                                route_rank,
                                k_start + chunk_start : k_start + chunk_stop,
                            ].reshape(-1)
                        ).fixed_port(0 if self.output_groups == 2 else 1)
                    )
                    for local_k in range(chunk_start, chunk_stop):
                        for group in range(self.output_groups):
                            append_weight(group, local_k)
            else:
                instructions.append(
                    TmaLoad1D(
                        self.activation_tiles[
                            route_rank, k_start:k_stop
                        ].reshape(-1)
                    ).fixed_port(1)
                )
                for local_k in range(self.k_tiles_per_split):
                    append_weight(0, local_k)

            output_row = 0 if self.aggregate_routes else route_rank
            final_work = work + self.num_sms >= self.work_tiles
            for group, output_reduce in enumerate(self.output_reduces):
                store = output_reduce.cord(
                    output_row, output_tile * self.TILE_M
                )
                if final_work and group + 1 == self.output_groups:
                    store.bar(self._bar("output"))
                instructions.append(store)
            first_work = False
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedRoutedDsv4Fp32SwiGluNvfp4QuantUmmaB(Schedule):
    """Fuse FP32 gate/up activation directly into native W2 input tiles."""

    TILE_K = 256
    TILE_BYTES = SchedNvfp4UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(
        self,
        routing_state,
        scale_field,
        gate,
        up,
        output,
        *,
        route_ready=False,
        swiglu_limit=10.0,
    ):
        super().__init__()
        self.routing_state = routing_state
        self.scale_field = scale_field
        self.gate = gate
        self.up = up
        self.output = output
        self.route_ready = bool(route_ready)
        self.swiglu_limit = swiglu_limit

    def _on_place(self):
        if self.routing_state.device.type != "cuda":
            raise ValueError("fused routed activation state must be on CUDA")
        if not 0 <= self.scale_field <= RoutedTmaLoad1D.MAX_POINTER_FIELD:
            raise ValueError("fused routed activation scale field must fit 13 bits")
        if (
            self.gate.dtype != torch.float32
            or self.up.dtype != torch.float32
            or self.gate.shape != self.up.shape
            or self.gate.ndim != 2
            or self.gate.shape[1] % self.TILE_K
            or not self.gate.is_contiguous()
            or not self.up.is_contiguous()
        ):
            raise ValueError(
                "fused FP32 SwiGLU inputs must be contiguous [routes,K256]"
            )
        self.route_count, self.rows = self.gate.shape
        if not 0 < self.route_count <= RoutedTmaLoad1D.ROUTE_COUNT:
            raise ValueError("fused FP32 SwiGLU requires 1..6 routes")
        self.k_tiles = self.rows // self.TILE_K
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape)
            != (self.route_count, self.k_tiles, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                "fused FP32 SwiGLU output must be native uint8 "
                "[routes,K/256,3072]"
            )
        self.work_tiles = self.route_count * self.k_tiles
        if not 0 < self.num_sms <= self.work_tiles:
            raise ValueError(
                f"fused FP32 SwiGLU requires 1..{self.work_tiles} SMs"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        route_bar = self._bar("route")
        if route_bar is None and not self.route_ready:
            raise ValueError("fused FP32 SwiGLU requires a route barrier")
        instructions = []
        first_work = True
        for work in range(sm, self.work_tiles, self.num_sms):
            tile, route_rank = divmod(work, self.route_count)
            start = tile * self.TILE_K
            stop = start + self.TILE_K
            scale = RoutedTmaLoad1D(
                self.routing_state,
                route_rank,
                self.scale_field,
                16,
            ).fixed_port(1)
            if first_work and route_bar is not None:
                scale.bar(route_bar)
            store = TmaStore1D(self.output[route_rank, tile].reshape(-1))
            if work + self.num_sms >= self.work_tiles:
                store.bar(self._bar("output"))
            instructions.extend(
                (
                    Dsv4Fp32SwiGluNvfp4QuantUmmaBSm100(
                        1, self.swiglu_limit
                    ),
                    TmaLoad1D(
                        self.gate[route_rank, start:stop]
                    ).fixed_port(0),
                    TmaLoad1D(
                        self.up[route_rank, start:stop]
                    ).fixed_port(1),
                    scale,
                    store,
                )
            )
            first_work = False
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
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
            Dsv4SiluClampMul(1, self.TILE_K, self.swiglu_limit),
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

    def __init__(
        self,
        input,
        output,
        scale_pack: int = 1,
        *,
        profile_store_event: int | None = None,
    ):
        super().__init__()
        self.input = input
        self.output = output
        self.scale_pack = int(scale_pack)
        self.profile_store_event = profile_store_event

    def _on_place(self):
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("native FP8 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % self.TILE_K:
            raise ValueError("native FP8 quant K must be K128 aligned")
        self.k_tiles = self.k // self.TILE_K
        if self.scale_pack not in (1, 2, 4) or self.k_tiles % self.scale_pack:
            raise ValueError(
                "native FP8 quant scale pack must be 1, 2, or 4 and divide K tiles"
            )
        self.scale_groups = self.k_tiles // self.scale_pack
        if not 0 < self.num_sms <= self.scale_groups:
            raise ValueError(
                "native FP8 quant requires 1..K/(128*scale_pack) SMs"
            )
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
        groups_per_sm, extra = divmod(self.scale_groups, self.num_sms)
        group_start = sm * groups_per_sm + min(sm, extra)
        group_count = groups_per_sm + int(sm < extra)
        tile_start = group_start * self.scale_pack
        tile_count = group_count * self.scale_pack
        tile_stop = tile_start + tile_count
        start = tile_start * self.TILE_K
        stop = tile_stop * self.TILE_K
        store = TmaStore1D(
            self.output[tile_start:tile_stop].reshape(-1)
        ).bar(self._bar("output"))
        if self.profile_store_event is not None:
            if not 0 <= self.profile_store_event < (1 << 15):
                raise ValueError("profile store event must fit 15 bits")
            store.arg = (1 << 15) | self.profile_store_event
        return [
            Dsv4Fp8QuantUmmaBSm100(tile_count, self.scale_pack),
            _shared_load_1d(
                self.input[start:stop]
            ).fixed_port(1),
            store,
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Mxfp8QuantFfnInput(Schedule):
    """Quantize BF16 into native K512 ``[data|SFB]`` records.

    A separately scheduled copy handoff converts the records into the two
    contiguous planes used by resident Linear-1; that copy overlaps routing.
    Only the 128 active SFB bytes are initialized; native-layout padding is
    deliberately unspecified.
    """

    TILE_K = 512
    DATA_BYTES = 8 * TILE_K
    SCALE_BYTES = 4 * 512
    RECORD_BYTES = DATA_BYTES + SCALE_BYTES

    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output

    def _on_place(self):
        if (
            self.input.dtype != torch.bfloat16
            or self.input.ndim != 1
            or not self.input.is_contiguous()
        ):
            raise ValueError("FFN MXFP8 quant input must be contiguous BF16 [K]")
        self.k = self.input.numel()
        if self.k % self.TILE_K:
            raise ValueError("FFN MXFP8 quant K must be K512 aligned")
        self.k_tiles = self.k // self.TILE_K
        if self.num_sms != self.k_tiles:
            raise ValueError("FFN MXFP8 quant requires one resident task per K512")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != (self.k_tiles, self.RECORD_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                "FFN MXFP8 output must be contiguous uint8 [K/512,6144]"
            )
        if self.input.device != self.output.device:
            raise ValueError("FFN MXFP8 input and output must share one device")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        start = sm * self.TILE_K
        return [
            Dsv4Mxfp8QuantFfnInputSm100(1),
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


class SchedDsv4SplitMxfp8FfnInputRecords(Schedule):
    """Split native K512 ``[data|SFB]`` records into resident FFN planes.

    This is a pure layout handoff through the existing copy task.  Eight data
    copies and eight scale copies are independent, so the production schedule
    uses exactly sixteen resident workers and can overlap unrelated router
    work on a disjoint CTA placement.
    """

    RECORDS = 8
    DATA_BYTES = SchedDsv4Mxfp8QuantFfnInput.DATA_BYTES
    SCALE_BYTES = SchedDsv4Mxfp8QuantFfnInput.SCALE_BYTES
    RECORD_BYTES = SchedDsv4Mxfp8QuantFfnInput.RECORD_BYTES

    def __init__(self, records, data, scales):
        super().__init__()
        self.records = records
        self.data = data
        self.scales = scales

    def _on_place(self):
        if self.num_sms != 2 * self.RECORDS:
            raise ValueError("FFN MXFP8 record split requires exactly 16 workers")
        expected = (
            ("records", self.records, (self.RECORDS, self.RECORD_BYTES)),
            ("data", self.data, (self.RECORDS, self.DATA_BYTES)),
            ("scales", self.scales, (self.RECORDS, self.SCALE_BYTES)),
        )
        for name, tensor, shape in expected:
            if (
                tensor.dtype != torch.uint8
                or tuple(tensor.shape) != shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"FFN MXFP8 split {name} must be contiguous uint8{shape}"
                )
        if any(
            tensor.device != self.records.device
            for tensor in (self.data, self.scales)
        ):
            raise ValueError("FFN MXFP8 split tensors must share one device")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        if sm < self.RECORDS:
            row = sm
            source = self.records[row, : self.DATA_BYTES]
            destination = self.data[row]
            size = self.DATA_BYTES
        else:
            row = sm - self.RECORDS
            source = self.records[row, self.DATA_BYTES :]
            destination = self.scales[row]
            size = self.SCALE_BYTES
        return [
            Copy(1, size),
            TmaLoad1D(source).bar(self._bar("input")).fixed_port(sm & 1),
            TmaStore1D(destination).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4InverseRopeFp8QuantUmmaB(Schedule):
    """Fuse inverse final-64 RoPE with native O_a activation packing."""

    HEAD_DIM = 512
    K_TILES = 4
    SCALE_PACK = 2
    TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(self, input, table, output):
        super().__init__()
        self.input = input
        self.table = table
        self.output = output

    def _on_place(self):
        if (
            self.input.dtype != torch.bfloat16
            or self.input.ndim != 2
            or self.input.shape[1] != self.HEAD_DIM
            or not self.input.is_contiguous()
        ):
            raise ValueError("inverse-RoPE/native-FP8 input must be BF16 [H,512]")
        self.rows = self.input.shape[0]
        if not 0 < self.num_sms <= self.rows:
            raise ValueError("inverse-RoPE/native-FP8 requires 1..H SMs")
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != (32, 2):
            raise ValueError("inverse-RoPE table must be FP32 [32,2]")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape)
            != (self.rows, self.K_TILES, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError("native O_a input must be uint8 [H,4,2048]")

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions.extend(
                (
                    Dsv4InverseRopeFp8QuantUmmaBSm100(),
                    TmaLoad1D(self.input[row]),
                    TmaLoad1D(self.table),
                    TmaStore1D(self.output[row].reshape(-1)).bar(
                        self._bar("output")
                    ),
                )
            )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.rows)


class SchedDsv4RmsFp8QuantUmmaB(Schedule):
    """Fuse one weighted RMS row with sharded native MXF8 packing."""

    TILE_K = SchedFp8UmmaPrepack.TILE_K
    TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES
    SCALE_PACK = 2

    def __init__(
        self, input, weight, output, epsilon: float, scale_pack: int = 2
    ):
        super().__init__()
        self.input = input
        self.weight = weight
        self.output = output
        self.epsilon = epsilon
        self.scale_pack = int(scale_pack)

    def _on_place(self):
        if (
            self.input.dtype != torch.bfloat16
            or self.input.ndim != 1
            or not self.input.is_contiguous()
        ):
            raise ValueError("fused RMS/native-FP8 input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % self.TILE_K:
            raise ValueError("fused RMS/native-FP8 K must be K128 aligned")
        if (
            self.weight.dtype != torch.bfloat16
            or tuple(self.weight.shape) != (self.k,)
            or not self.weight.is_contiguous()
        ):
            raise ValueError("fused RMS/native-FP8 weight must be BF16 [K]")
        if self.epsilon <= 0:
            raise ValueError("fused RMS/native-FP8 epsilon must be positive")
        if self.scale_pack != self.SCALE_PACK:
            raise ValueError("fused RMS/native-FP8 currently selects static pack-2")
        self.k_tiles = self.k // self.TILE_K
        if self.k_tiles % self.scale_pack:
            raise ValueError("fused RMS/native-FP8 K tiles must fit scale packs")
        self.scale_groups = self.k_tiles // self.scale_pack
        if not 0 < self.num_sms <= self.scale_groups:
            raise ValueError("fused RMS/native-FP8 requires 1..scale-groups SMs")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != (self.k_tiles, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                "fused RMS/native-FP8 output must be uint8 [K/128,2048]"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        groups_per_sm, extra = divmod(self.scale_groups, self.num_sms)
        group_start = sm * groups_per_sm + min(sm, extra)
        group_count = groups_per_sm + int(sm < extra)
        tile_start = group_start * self.scale_pack
        tile_count = group_count * self.scale_pack
        tile_stop = tile_start + tile_count
        return [
            Dsv4RmsFp8QuantUmmaBSm100(
                self.k_tiles, tile_start, tile_count, self.epsilon
            ),
            TmaLoad1D(self.input).fixed_port(1),
            TmaLoad1D(self.weight).fixed_port(0),
            TmaStore1D(
                self.output[tile_start:tile_stop].reshape(-1)
            ).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Fp32RmsFp8QuantUmmaB(SchedDsv4RmsFp8QuantUmmaB):
    """Consume the FP32 split-K accumulator directly before native packing."""

    def _on_place(self):
        if (
            self.input.dtype != torch.float32
            or self.input.ndim != 1
            or not self.input.is_contiguous()
        ):
            raise ValueError(
                "FP32 RMS/native-FP8 input must be a contiguous vector"
            )
        self.k = self.input.numel()
        if self.k % self.TILE_K:
            raise ValueError("FP32 RMS/native-FP8 K must be K128 aligned")
        if (
            self.weight.dtype != torch.bfloat16
            or tuple(self.weight.shape) != (self.k,)
            or not self.weight.is_contiguous()
        ):
            raise ValueError("FP32 RMS/native-FP8 weight must be BF16 [K]")
        if self.epsilon <= 0:
            raise ValueError("FP32 RMS/native-FP8 epsilon must be positive")
        if self.scale_pack != self.SCALE_PACK:
            raise ValueError("FP32 RMS/native-FP8 currently selects static pack-2")
        self.k_tiles = self.k // self.TILE_K
        if self.k_tiles % self.scale_pack:
            raise ValueError("FP32 RMS/native-FP8 K tiles must fit scale packs")
        self.scale_groups = self.k_tiles // self.scale_pack
        if not 0 < self.num_sms <= self.scale_groups:
            raise ValueError("FP32 RMS/native-FP8 requires 1..scale-groups SMs")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape) != (self.k_tiles, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError(
                "FP32 RMS/native-FP8 output must be uint8 [K/128,2048]"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        groups_per_sm, extra = divmod(self.scale_groups, self.num_sms)
        group_start = sm * groups_per_sm + min(sm, extra)
        group_count = groups_per_sm + int(sm < extra)
        tile_start = group_start * self.scale_pack
        tile_count = group_count * self.scale_pack
        tile_stop = tile_start + tile_count
        return [
            Dsv4Fp32RmsFp8QuantUmmaBSm100(
                self.k_tiles, tile_start, tile_count, self.epsilon
            ),
            TmaLoad1D(self.input).fixed_port(1),
            TmaLoad1D(self.weight).fixed_port(0),
            TmaStore1D(
                self.output[tile_start:tile_stop].reshape(-1)
            ).bar(self._bar("output")),
        ]


class SchedDsv4Bf16GemvGroup4SplitK(Schedule):
    """Shard grouped M512 BF16 projection work across K and TMA-reduce FP32."""

    TILE_M = 128
    TILE_K = 128
    OUTPUT_GROUPS = 4
    ACTIVATION_TILES_PER_CHUNK = 4

    def __init__(
        self,
        weight,
        weight_tma,
        activation,
        output_reduce,
        split_k: int,
        *,
        layer_indexed_weight=False,
        layer_indexed_output=False,
        direct_output=False,
    ):
        super().__init__()
        self.weight = weight
        self.weight_tma = weight_tma
        self.activation = activation
        self.output_reduce = output_reduce
        self.split_k = int(split_k)
        self.layer_indexed_weight = bool(layer_indexed_weight)
        self.layer_indexed_output = bool(layer_indexed_output)
        self.direct_output = bool(direct_output)

    def _on_place(self):
        if (
            self.weight.dtype != torch.bfloat16
            or self.weight.ndim not in (2, 3)
            or not self.weight.is_contiguous()
        ):
            raise ValueError(
                "grouped BF16 weight must be contiguous [M,K] or [L,M,K]"
            )
        self.rows, self.k = self.weight.shape[-2:]
        if self.weight.ndim == 2 and self.layer_indexed_weight:
            raise ValueError(
                "rank-2 grouped BF16 weight cannot be layer indexed"
            )
        if self.weight.ndim == 3 and not self.layer_indexed_weight:
            raise ValueError(
                "rank-3 grouped BF16 weight requires layer indexing"
            )
        group_rows = self.TILE_M * self.OUTPUT_GROUPS
        if self.rows % group_rows or self.k % self.TILE_K:
            raise ValueError("grouped BF16 projection requires M512/K128 alignment")
        if (
            self.activation.dtype != torch.bfloat16
            or self.activation.numel() != self.k
            or not self.activation.is_contiguous()
        ):
            raise ValueError("grouped BF16 activation must be contiguous BF16 [K]")
        self.k_tiles = self.k // self.TILE_K
        if self.split_k <= 0 or self.k_tiles % self.split_k:
            raise ValueError(
                "grouped BF16 split-K must divide the K128 tile count"
            )
        if self.direct_output and self.split_k != 1:
            raise ValueError("direct grouped BF16 output requires split-K one")
        self.k_tiles_per_split = self.k_tiles // self.split_k
        if (
            self.k_tiles_per_split != 2
            and self.k_tiles_per_split % self.ACTIVATION_TILES_PER_CHUNK
        ):
            raise ValueError(
                "grouped BF16 split-K requires two or a multiple of four "
                "K128 tiles per shard"
            )
        self.activation_tiles_per_chunk = min(
            self.ACTIVATION_TILES_PER_CHUNK,
            self.k_tiles_per_split,
        )
        self.m_groups = self.rows // group_rows
        self.work_items = self.m_groups * self.split_k
        if not 0 < self.num_sms <= self.work_items:
            raise ValueError(
                f"grouped BF16 split-K requires 1..{self.work_items} SMs"
            )
        if (
            getattr(self.weight_tma, "mode", None) != "load"
            or getattr(self.weight_tma, "mat", None) is not self.weight
        ):
            raise ValueError("grouped BF16 weight TMA must load the schedule matrix")
        output = getattr(self.output_reduce, "mat", None)
        expected_output_shape = (
            (self.weight.shape[0], self.rows // self.TILE_M, self.TILE_M)
            if self.layer_indexed_output
            else (self.rows // self.TILE_M, self.TILE_M)
        )
        if (
            getattr(self.output_reduce, "mode", None)
            != ("store" if self.direct_output else "reduce")
            or output is None
            or output.dtype != torch.float32
            or tuple(output.shape) != expected_output_shape
            or not output.is_contiguous()
        ):
            raise ValueError(
                "grouped BF16 output has the wrong reduction layout: "
                f"expected={expected_output_shape} "
                f"actual={None if output is None else tuple(output.shape)} "
                f"weight={tuple(self.weight.shape)} "
                f"layer_indexed={self.layer_indexed_output}"
            )

    def _work_shard(self, sm):
        work_per_sm, extra = divmod(self.work_items, self.num_sms)
        work_start = sm * work_per_sm + min(sm, extra)
        work_count = work_per_sm + int(sm < extra)
        return work_start, work_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        work_start, work_count = self._work_shard(sm)
        work_stop = work_start + work_count
        group_rows = self.TILE_M * self.OUTPUT_GROUPS
        instructions = []
        for work in range(work_start, work_stop):
            split = work // self.m_groups
            m_group = work % self.m_groups
            m_start = m_group * group_rows
            k_start = split * self.k_tiles_per_split
            k_stop = k_start + self.k_tiles_per_split
            instructions.append(
                Dsv4Bf16GemvGroup4SplitKSm100(self.k_tiles_per_split)
            )
            for chunk_start in range(
                k_start, k_stop, self.activation_tiles_per_chunk
            ):
                chunk_stop = chunk_start + self.activation_tiles_per_chunk
                instructions.append(
                    TmaLoad1D(
                        self.activation[
                            chunk_start * self.TILE_K :
                            chunk_stop * self.TILE_K
                        ]
                    ).fixed_port(1)
                )
                for k_tile in range(chunk_start, chunk_stop):
                    if self.weight.ndim == 2:
                        weight_load = self.weight_tma.cord(
                            m_start,
                            k_tile * self.TILE_K,
                        ).fixed_port(0)
                        weight_delta = [0, self.TILE_M, 0]
                    else:
                        weight_load = self.weight_tma.cord(
                            0,
                            m_start,
                            k_tile * self.TILE_K,
                        ).fixed_port(0)
                        flags = weight_load.opcode & ((1 << 6) - 1)
                        weight_load.opcode = (
                            opcode.OP_ALLOC_LAYER_TMA_LOAD_4D | flags
                        )
                        weight_delta = [0, self.TILE_M, 0, 0]
                    instructions.extend(
                        RepeatM.on(
                            self.OUTPUT_GROUPS,
                            (weight_load, weight_delta),
                        )
                    )
            if self.layer_indexed_output:
                store = self.output_reduce.cord(
                    0, m_start // self.TILE_M, 0
                )
                flags = store.opcode & ((1 << 6) - 1)
                store.opcode = (
                    (
                        opcode.OP_ALLOC_WB_LAYER_TMA_STORE_3D
                        if self.direct_output
                        else opcode.OP_ALLOC_WB_LAYER_TMA_REDUCE_ADD_3D
                    )
                    | flags
                )
            else:
                store = self.output_reduce.cord(
                    m_start // self.TILE_M, 0
                )
            if work + 1 == work_stop:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4ZeroFill(Schedule):
    """Shard an in-queue zero fill over a contiguous tensor."""

    def __init__(self, gate, output, *, profile_store_event=None):
        super().__init__()
        self.gate = gate
        self.output = output
        self.profile_store_event = profile_store_event

    def _on_place(self):
        if not self.output.is_contiguous() or self.output.numel() <= 0:
            raise ValueError("zero-fill output must be nonempty and contiguous")
        if self.gate is not None and (
            self.gate.dtype != torch.uint32
            or self.gate.numel() != 1
            or not self.gate.is_contiguous()
        ):
            raise ValueError("zero-fill gate must be one contiguous uint32")
        element_bytes = self.output.element_size()
        total_bytes = self.output.numel() * element_bytes
        if total_bytes % 4 or 4 % element_bytes:
            raise ValueError("zero-fill output must contain complete uint32 words")
        self.elements_per_word = 4 // element_bytes
        self.total_words = total_bytes // 4
        if self.total_words % 4:
            raise ValueError("zero-fill output must contain complete 16-byte blocks")
        self.total_blocks = self.total_words // 4
        if not 0 < self.num_sms <= self.total_blocks:
            raise ValueError("zero-fill requires 1..output-block-count SMs")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        blocks_per_sm, extra = divmod(self.total_blocks, self.num_sms)
        block_start = sm * blocks_per_sm + min(sm, extra)
        block_count = blocks_per_sm + int(sm < extra)
        word_start = block_start * 4
        word_count = block_count * 4
        element_start = word_start * self.elements_per_word
        element_stop = (word_start + word_count) * self.elements_per_word
        output = self.output.reshape(-1)[element_start:element_stop]
        instructions = [
            Dsv4ZeroFill(word_count * 4, has_gate=self.gate is not None),
        ]
        if self.gate is not None:
            instructions.append(
                LduLoad1D(self.gate).bar(self._bar("gate"))
            )
        store = TmaStore1D(output).bar(self._bar("output"))
        if self.profile_store_event is not None:
            if not 0 <= self.profile_store_event < (1 << 14):
                raise ValueError("profile store event must fit 14 bits")
            store.arg = (
                (1 << 15) | (1 << 14) | self.profile_store_event
            )
        instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Fp32ToBf16(Schedule):
    """Finalize an FP32 split-K accumulator in model dtype."""

    TILE = 128

    def __init__(self, input, output):
        super().__init__()
        self.input = input
        self.output = output

    def _on_place(self):
        if (
            self.input.dtype != torch.float32
            or self.output.dtype != torch.bfloat16
            or self.input.ndim != 1
            or self.output.ndim != 1
            or self.input.shape != self.output.shape
            or not self.input.is_contiguous()
            or not self.output.is_contiguous()
            or self.input.numel() % self.TILE
        ):
            raise ValueError(
                "projection finalizer requires matching contiguous FP32/BF16 M128 vectors"
            )
        self.tiles = self.input.numel() // self.TILE
        if not 0 < self.num_sms <= self.tiles:
            raise ValueError("projection finalizer requires 1..M/128 SMs")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        tiles_per_sm, extra = divmod(self.tiles, self.num_sms)
        tile_start = sm * tiles_per_sm + min(sm, extra)
        tile_count = tiles_per_sm + int(sm < extra)
        start = tile_start * self.TILE
        stop = (tile_start + tile_count) * self.TILE
        return [
            Dsv4Fp32ToBf16(stop - start),
            TmaLoad1D(self.input[start:stop]).fixed_port(0),
            TmaStore1D(self.output[start:stop]).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedMxfp4Mxfp8GemvUmmaK512(Schedule):
    """M128/N8/K4096 native MXFP4 x MXFP8 projection."""

    TILE_M = 128
    K512_TILES = 8
    WEIGHT_DATA_BYTES = 32768
    WEIGHT_K128_TILES = 4
    WEIGHT_PACKED_K128_BYTES = 64
    WEIGHT_SCALE_BYTES = 2048
    ACTIVATION_DATA_BYTES = 4096
    ACTIVATION_SCALE_BYTES = 2048
    METADATA_BYTES = 128

    def __init__(
        self,
        weight_data,
        weight_scale,
        activation_data,
        activation_scale,
        output,
        weight_tma,
        *,
        scale_mode: str,
        metadata=None,
        activation_tiles_per_load: int = 4,
        tma_scale_ports: tuple[int, int] = (0, 1),
    ):
        super().__init__()
        self.weight_data = weight_data
        self.weight_scale = weight_scale
        self.activation_data = activation_data
        self.activation_scale = activation_scale
        self.output = output
        self.weight_tma = weight_tma
        self.scale_mode = scale_mode
        self.metadata = metadata
        self.activation_tiles_per_load = int(activation_tiles_per_load)
        self.tma_scale_ports = tuple(int(port) for port in tma_scale_ports)

    def _on_place(self):
        if self.scale_mode not in ("tma", "metadata"):
            raise ValueError("MXFP4/MXFP8 scale mode must be tma or metadata")
        if self.activation_tiles_per_load not in (1, 2, 4, 8):
            raise ValueError(
                "MXFP4/MXFP8 activation tiles per load must be 1, 2, 4, or 8"
            )
        if len(self.tma_scale_ports) != 2 or any(
            port not in (0, 1) for port in self.tma_scale_ports
        ):
            raise ValueError("MXFP4/MXFP8 TMA scale ports must be a pair of LDU IDs")
        if self.scale_mode == "tma" and self.tma_scale_ports != (0, 1):
            raise ValueError(
                "direct MXFP4/MXFP8 TMA requires SFA on LDU0 and SFB on LDU1"
            )
        if (
            self.weight_data.dtype != torch.uint8
            or self.weight_data.ndim != 5
            or self.weight_data.shape[1:] != (
                self.K512_TILES,
                self.WEIGHT_K128_TILES,
                self.TILE_M,
                self.WEIGHT_PACKED_K128_BYTES,
            )
            or not self.weight_data.is_contiguous()
        ):
            raise ValueError(
                "MXFP4 weight data must be packed contiguous uint8 "
                "[M/128,8,4,128,64]"
            )
        self.m_tiles = self.weight_data.shape[0]
        if not 0 < self.num_sms <= self.m_tiles:
            raise ValueError("MXFP4/MXFP8 projection needs 1..M/128 SMs")
        if (
            self.weight_tma is None
            or getattr(self.weight_tma, "rank", None) != 5
            or getattr(self.weight_tma, "size", None) != self.WEIGHT_DATA_BYTES
            or getattr(self.weight_tma, "num_slots", None) != 8
        ):
            raise ValueError(
                "MXFP4 weight TMA must be a packed K512 5D load"
            )
        if (
            self.weight_scale.dtype != torch.uint8
            or tuple(self.weight_scale.shape) != (
                self.m_tiles, self.K512_TILES, self.WEIGHT_SCALE_BYTES
            )
            or not self.weight_scale.is_contiguous()
        ):
            raise ValueError(
                "MXFP4 scales must be native uint8 [M/128,8,2048]"
            )
        if (
            self.activation_data.dtype != torch.uint8
            or tuple(self.activation_data.shape) != (
                self.K512_TILES, self.ACTIVATION_DATA_BYTES
            )
            or not self.activation_data.is_contiguous()
        ):
            raise ValueError(
                "MXFP8 activation data must be native uint8 [8,4096]"
            )
        if (
            self.activation_scale.dtype != torch.uint8
            or tuple(self.activation_scale.shape) != (
                self.K512_TILES, self.ACTIVATION_SCALE_BYTES
            )
            or not self.activation_scale.is_contiguous()
        ):
            raise ValueError(
                "MXFP8 scales must be native uint8 [8,2048]"
            )
        if (
            self.output.dtype != torch.float32
            or self.output.numel() != self.m_tiles * self.TILE_M
            or not self.output.is_contiguous()
        ):
            raise ValueError("MXFP4/MXFP8 output must be contiguous FP32 [M]")
        tensors = (
            self.weight_data,
            self.weight_scale,
            self.activation_data,
            self.activation_scale,
            self.output,
        )
        if any(tensor.device != self.output.device for tensor in tensors):
            raise ValueError("MXFP4/MXFP8 tensors must share one CUDA device")
        if self.scale_mode == "metadata":
            if (
                self.metadata is None
                or self.metadata.dtype != torch.uint8
                or tuple(self.metadata.shape) != (
                    self.m_tiles, self.METADATA_BYTES
                )
                or not self.metadata.is_contiguous()
                or self.metadata.device != self.output.device
            ):
                raise ValueError(
                    "metadata scale mode requires uint8 [M/128,128] records"
                )

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
        tile_stop = tile_start + tile_count
        for output_tile in range(tile_start, tile_stop):
            if self.scale_mode == "tma":
                instructions.append(
                    Mxfp4Mxfp8GemvUmmaK512TmaScaleFp32Sm100(
                        self.activation_tiles_per_load
                    )
                )
            else:
                instructions.extend((
                    Mxfp4Mxfp8GemvUmmaK512MetaScaleFp32Sm100(
                        self.metadata[output_tile].data_ptr(),
                        self.activation_tiles_per_load
                    ),
                ))
            for chunk_start in range(
                0, self.K512_TILES, self.activation_tiles_per_load
            ):
                chunk_stop = chunk_start + self.activation_tiles_per_load
                instructions.append(
                    TmaLoad1D(
                        self.activation_data[chunk_start:chunk_stop].reshape(-1)
                    ).fixed_port(1)
                )
                for k_tile in range(chunk_start, chunk_stop):
                    if self.scale_mode == "tma":
                        scale_stage = k_tile % TmaLoadMxfpScale1D.STAGES
                        if k_tile == 0:
                            instructions.extend((
                                TmaLoadMxfpScaleBase1D(
                                    self.weight_scale[output_tile].reshape(-1),
                                    operand=TmaLoadMxfpScale1D.WEIGHT,
                                ).fixed_port(self.tma_scale_ports[0]),
                                TmaLoadMxfpScaleBase1D(
                                    self.activation_scale.reshape(-1),
                                    operand=TmaLoadMxfpScale1D.ACTIVATION,
                                ).fixed_port(self.tma_scale_ports[1]),
                            ))
                        else:
                            instructions.extend((
                                TmaLoadMxfpScale1D(
                                    stage=scale_stage,
                                    operand=TmaLoadMxfpScale1D.WEIGHT,
                                ).fixed_port(self.tma_scale_ports[0]),
                                TmaLoadMxfpScale1D(
                                    stage=scale_stage,
                                    operand=TmaLoadMxfpScale1D.ACTIVATION,
                                ).fixed_port(self.tma_scale_ports[1]),
                            ))
                    instructions.append(
                        self.weight_tma.cord(output_tile, k_tile).fixed_port(0)
                    )
            row_start = output_tile * self.TILE_M
            store = TmaStore1D(
                self.output[row_start : row_start + self.TILE_M]
            )
            if output_tile + 1 == tile_stop:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedMxfp4Mxfp8GateUpSiluFixedRing(Schedule):
    """Fused Linear-1 with an optional allocator-owned retained LDU ring."""

    TILE_M = SchedMxfp4Mxfp8GemvUmmaK512.TILE_M
    K512_TILES = SchedMxfp4Mxfp8GemvUmmaK512.K512_TILES
    WEIGHT_K128_TILES = SchedMxfp4Mxfp8GemvUmmaK512.WEIGHT_K128_TILES
    WEIGHT_PACKED_K128_BYTES = (
        SchedMxfp4Mxfp8GemvUmmaK512.WEIGHT_PACKED_K128_BYTES
    )
    WEIGHT_DATA_BYTES = SchedMxfp4Mxfp8GemvUmmaK512.WEIGHT_DATA_BYTES
    WEIGHT_SCALE_BYTES = SchedMxfp4Mxfp8GemvUmmaK512.WEIGHT_SCALE_BYTES
    ACTIVATION_DATA_BYTES = (
        SchedMxfp4Mxfp8GemvUmmaK512.ACTIVATION_DATA_BYTES
    )
    ACTIVATION_SCALE_BYTES = (
        SchedMxfp4Mxfp8GemvUmmaK512.ACTIVATION_SCALE_BYTES
    )
    OUTPUT_DATA_BYTES = 8 * TILE_M
    OUTPUT_SCALE_BYTES = 512
    METADATA_BYTES = 128

    def __init__(
        self,
        gate_weight_data,
        gate_weight_scale,
        up_weight_data,
        up_weight_scale,
        activation_data,
        activation_scale,
        output_data,
        output_scale,
        gate_weight_tma,
        up_weight_tma,
        metadata,
        *,
        tile_k: int = 512,
    ):
        super().__init__()
        self.tile_k = int(tile_k)
        if self.tile_k not in (128, 512):
            raise ValueError("fixed-ring fused gate/up supports K128 or K512")
        self.ring_stages = 10 if self.tile_k == 128 else 2
        self.k_tiles = 4096 // self.tile_k
        self.weight_k128_tiles = self.tile_k // 128
        self.weight_data_bytes = self.TILE_M * self.tile_k // 2
        self.weight_scale_bytes = self.weight_k128_tiles * 512
        self.activation_data_bytes = self.weight_k128_tiles * 8 * 128
        self.activation_scale_bytes = self.weight_k128_tiles * 512
        self.weight_slots = self.TILE_M * self.tile_k // (8 * 1024)
        self.gate_weight_data = gate_weight_data
        self.gate_weight_scale = gate_weight_scale
        self.up_weight_data = up_weight_data
        self.up_weight_scale = up_weight_scale
        self.activation_data = activation_data
        self.activation_scale = activation_scale
        self.output_data = output_data
        self.output_scale = output_scale
        self.gate_weight_tma = gate_weight_tma
        self.up_weight_tma = up_weight_tma
        self.metadata = metadata

    def _validate_tensors(self):
        expected_weight_tail = (
            self.k_tiles,
            self.weight_k128_tiles,
            self.TILE_M,
            self.WEIGHT_PACKED_K128_BYTES,
        )
        for name, tensor in (
            ("gate", self.gate_weight_data),
            ("up", self.up_weight_data),
        ):
            if (
                tensor.dtype != torch.uint8
                or tensor.ndim != 5
                or tuple(tensor.shape[1:]) != expected_weight_tail
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} MXFP4 data must be packed contiguous uint8 "
                    "[M/128,K/tile_k,tile_k/128,128,64]"
                )
        if self.gate_weight_data.shape != self.up_weight_data.shape:
            raise ValueError("fused gate/up MXFP4 data shapes must match")
        self.m_tiles = self.gate_weight_data.shape[0]
        if not 0 < self.num_sms <= self.m_tiles:
            raise ValueError("fused gate/up projection needs 1..M-slices SMs")
        for name, descriptor in (
            ("gate", self.gate_weight_tma),
            ("up", self.up_weight_tma),
        ):
            if (
                descriptor is None
                or getattr(descriptor, "rank", None) != 5
                or getattr(descriptor, "size", None) != self.weight_data_bytes
                or getattr(descriptor, "num_slots", None) != self.weight_slots
            ):
                raise ValueError(
                    f"{name} MXFP4 TMA must match packed K{self.tile_k}"
                )
        expected_scale_shape = (
            self.m_tiles,
            self.k_tiles,
            self.weight_scale_bytes,
        )
        for name, tensor in (
            ("gate", self.gate_weight_scale),
            ("up", self.up_weight_scale),
        ):
            if (
                tensor.dtype != torch.uint8
                or tuple(tensor.shape) != expected_scale_shape
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} MXFP4 scales must be native uint8 "
                    "[M/128,K/tile_k,tile_k/128*512]"
                )
        if (
            self.activation_data.dtype != torch.uint8
            or tuple(self.activation_data.shape)
            != (self.k_tiles, self.activation_data_bytes)
            or not self.activation_data.is_contiguous()
        ):
            raise ValueError("MXFP8 activation data shape does not match tile K")
        if (
            self.activation_scale.dtype != torch.uint8
            or tuple(self.activation_scale.shape)
            != (self.k_tiles, self.activation_scale_bytes)
            or not self.activation_scale.is_contiguous()
        ):
            raise ValueError("MXFP8 activation scale shape does not match tile K")
        if (
            self.output_data.dtype != torch.uint8
            or tuple(self.output_data.shape)
            != (self.m_tiles, self.OUTPUT_DATA_BYTES)
            or self.output_data.stride(-1) != 1
        ):
            raise ValueError(
                "fused MXFP8 data output must have contiguous uint8 rows "
                "[M/128,1024]"
            )
        if (
            self.output_scale.dtype != torch.uint8
            or tuple(self.output_scale.shape)
            != (self.m_tiles, self.OUTPUT_SCALE_BYTES)
            or self.output_scale.stride(-1) != 1
        ):
            raise ValueError(
                "fused MXFP8 scale output must have contiguous uint8 rows "
                "[M/128,512]"
            )
        if (
            self.metadata.dtype != torch.uint8
            or tuple(self.metadata.shape) != (self.m_tiles, self.METADATA_BYTES)
            or not self.metadata.is_contiguous()
        ):
            raise ValueError("fused raw scales require uint8 [M/128,128] metadata")
        tensors = (
            self.gate_weight_data,
            self.gate_weight_scale,
            self.up_weight_data,
            self.up_weight_scale,
            self.activation_data,
            self.activation_scale,
            self.output_data,
            self.output_scale,
            self.metadata,
        )
        if any(tensor.device != self.output_data.device for tensor in tensors):
            raise ValueError("fused MXFP4/MXFP8 tensors must share one CUDA device")

    def _tile_shard(self, sm):
        tiles_per_sm, extra = divmod(self.m_tiles, self.num_sms)
        tile_start = sm * tiles_per_sm + min(sm, extra)
        tile_count = tiles_per_sm + int(sm < extra)
        return tile_start, tile_count

    def _on_place(self):
        self._validate_tensors()
        record_bytes = self.OUTPUT_DATA_BYTES + self.OUTPUT_SCALE_BYTES
        if (
            self.output_data.stride(0) != record_bytes
            or self.output_scale.stride(0) != record_bytes
            or self.output_scale.data_ptr()
            != self.output_data.data_ptr() + self.OUTPUT_DATA_BYTES
        ):
            raise ValueError(
                "fixed-ring output must be one [data|scale] HBM record per tile"
            )
        self.output_record = self.output_data.as_strided(
            (self.output_data.shape[0], record_bytes),
            (record_bytes, 1),
        )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        tile_start, tile_count = self._tile_shard(sm)
        tile_stop = tile_start + tile_count
        instructions = []
        for output_tile in range(tile_start, tile_stop):
            instructions.extend(self._task_instructions(output_tile))
        return instructions

    def _task_instructions(self, output_tile):
        return [
            Mxfp4Mxfp8GateUpSiluFixedRingSm100(
                self.metadata[output_tile].data_ptr(),
                tile_k=self.tile_k,
                stages=self.ring_stages,
            )
        ]

    def bar_release_count(self, role: str):
        return 0


class SchedMxfp4Mxfp8DownFixedRing(Schedule):
    """K2048 Linear-2 over native Linear-1 MXFP8 records."""

    TILE_M = 128
    TILE_N = 8
    # Match the accepted full-FFN task: two K256 stages keep the fixed ring at
    # 80 KiB while preserving the allocator arena in front of the scratchpad.
    TILE_K = 256
    K_TILES = 8
    K128_PER_TILE = 2
    INTERMEDIATE = 2048
    HIDDEN = 4096
    DOWN_TILES_PER_EXPERT = HIDDEN // TILE_M
    ACTIVATION_TILES_PER_EXPERT = INTERMEDIATE // 128
    ACTIVATION_RECORD_BYTES = 1536
    WEIGHT_PACKED_K128_BYTES = 64
    WEIGHT_DATA_BYTES = TILE_M * TILE_K // 2
    WEIGHT_SCALE_BYTES = K128_PER_TILE * 512
    METADATA_BYTES = 128

    def __init__(
        self,
        weight_data,
        weight_scale,
        activation_records,
        final_output,
        weight_tma,
        metadata,
        *,
        output_n_major: bool = False,
        fp32_output: bool = False,
    ):
        super().__init__()
        self.weight_data = weight_data
        self.weight_scale = weight_scale
        self.activation_records = activation_records
        self.final_output = final_output
        self.weight_tma = weight_tma
        self.metadata = metadata
        self.output_n_major = bool(output_n_major)
        self.fp32_output = bool(fp32_output)

    def _on_place(self):
        if (
            self.weight_data.dtype != torch.uint8
            or self.weight_data.ndim != 5
            or tuple(self.weight_data.shape[1:])
            != (
                self.K_TILES,
                self.K128_PER_TILE,
                self.TILE_M,
                self.WEIGHT_PACKED_K128_BYTES,
            )
            or not self.weight_data.is_contiguous()
        ):
            raise ValueError(
                "down MXFP4 data must be packed contiguous uint8 "
                "[tasks,8,2,128,64]"
            )
        self.tasks = self.weight_data.shape[0]
        if self.tasks % self.DOWN_TILES_PER_EXPERT:
            raise ValueError("down task count must contain complete experts")
        self.experts = self.tasks // self.DOWN_TILES_PER_EXPERT
        if not 0 < self.num_sms <= self.tasks:
            raise ValueError("down projection needs 1..task-count SMs")
        if (
            self.weight_scale.dtype != torch.uint8
            or tuple(self.weight_scale.shape)
            != (self.tasks, self.K_TILES, self.WEIGHT_SCALE_BYTES)
            or not self.weight_scale.is_contiguous()
        ):
            raise ValueError(
                "down MXFP4 scales must be native uint8 [tasks,8,1024]"
            )
        if (
            self.activation_records.dtype != torch.uint8
            or tuple(self.activation_records.shape)
            != (
                self.experts,
                self.ACTIVATION_TILES_PER_EXPERT,
                self.ACTIVATION_RECORD_BYTES,
            )
            or not self.activation_records.is_contiguous()
        ):
            raise ValueError(
                "down activation must be native contiguous uint8 "
                "[experts,16,1536]"
            )
        expected_output_shape = (
            (self.TILE_N, self.HIDDEN)
            if self.output_n_major
            else (self.DOWN_TILES_PER_EXPERT, self.TILE_M, self.TILE_N)
        )
        expected_output_dtype = (
            torch.float32 if self.fp32_output else torch.bfloat16
        )
        output_layout_valid = (
            self.final_output.ndim == 2
            and self.final_output.stride(1) == 1
            and self.final_output.stride(0) >= self.HIDDEN
        ) if self.output_n_major else self.final_output.is_contiguous()
        if (
            self.final_output.dtype != expected_output_dtype
            or tuple(self.final_output.shape) != expected_output_shape
            or not output_layout_valid
        ):
            layout = "[8,4096]" if self.output_n_major else "[32,128,8]"
            raise ValueError(
                "down final output must be "
                f"{str(expected_output_dtype).removeprefix('torch.').upper()} "
                f"M-contiguous {layout}"
            )
        if (
            self.metadata.dtype != torch.uint8
            or tuple(self.metadata.shape) not in (
                (self.tasks, self.METADATA_BYTES),
                (3 * self.tasks, self.METADATA_BYTES),
            )
            or not self.metadata.is_contiguous()
        ):
            raise ValueError(
                "down metadata must be contiguous uint8 [tasks,128] or "
                "[3*tasks,128]"
            )
        self.metadata_variants = self.metadata.shape[0] // self.tasks
        if (
            self.weight_tma is None
            or getattr(self.weight_tma, "rank", None) != 5
            or getattr(self.weight_tma, "size", None) != self.WEIGHT_DATA_BYTES
        ):
            raise ValueError("down MXFP4 TMA must match packed M128/K256")
        tensors = (
            self.weight_data,
            self.weight_scale,
            self.activation_records,
            self.final_output,
            self.metadata,
        )
        if any(tensor.device != self.weight_data.device for tensor in tensors):
            raise ValueError("down MXFP4/MXFP8 tensors must share one device")

        if self.num_sms < self.DOWN_TILES_PER_EXPERT:
            raise ValueError(
                "shared-first down scheduling requires at least 32 workers"
            )
        self.task_queues = [[] for _ in range(self.num_sms)]
        for task in range(self.DOWN_TILES_PER_EXPERT):
            self.task_queues[task].append(task)
        for task in range(self.DOWN_TILES_PER_EXPERT, self.tasks):
            worker = min(
                range(self.num_sms),
                key=lambda index: (len(self.task_queues[index]), index),
            )
            self.task_queues[worker].append(task)

    def _tile_shard(self, sm):
        tiles_per_sm, extra = divmod(self.tasks, self.num_sms)
        tile_start = sm * tiles_per_sm + min(sm, extra)
        tile_count = tiles_per_sm + int(sm < extra)
        return tile_start, tile_count

    def _task_instructions(self, task):
        return [
            Mxfp4Mxfp8DownFixedRingSm100(
                self.metadata[task].data_ptr()
            )
        ]

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = []
        task_queue = self.task_queues[sm]
        for task in task_queue:
            instructions.extend(self._task_instructions(task))
        return instructions


class SchedMxfp4Mxfp8ResidentFfn(Schedule):
    """One compute command and three coupled streams per FFN worker."""

    def __init__(self, linear1, down):
        super().__init__()
        self.linear1 = linear1
        self.down = down

    def _on_place(self):
        self.placed_linear1 = self.linear1.place(self.num_sms)
        self.placed_down = self.down.place(self.num_sms)
        if self.placed_linear1.m_tiles != self.num_sms:
            raise ValueError("resident FFN requires one Linear-1 task per worker")
        if max(map(len, self.placed_down.task_queues)) > 2:
            raise ValueError("resident FFN supports at most two Down tasks per worker")
        # The full DSV4 shape has 16 Linear-1 slices and 32 Down tiles per
        # expert. Keep both Down tasks on a worker whose Linear-1 record belongs
        # to that expert. This removes cross-expert readiness skew without
        # changing the two-task load or the shared/routed reduction protocol.
        if (
            self.num_sms == self.placed_down.experts * 16
            and self.placed_down.DOWN_TILES_PER_EXPERT == 32
        ):
            self.placed_down.task_queues = []
            for worker in range(self.num_sms):
                expert, local_slice = divmod(worker, 16)
                expert_base = expert * self.placed_down.DOWN_TILES_PER_EXPERT
                self.placed_down.task_queues.append(
                    [expert_base + local_slice, expert_base + 16 + local_slice]
                )

        # Gate and up are one homogeneous operation stream for the memory
        # virtual core. One descriptor addresses 16 task-major K512 records,
        # and the matching SFA records have the same order. SFB remains one
        # shared 16-KiB tensor instead of being replicated into every worker's
        # prepacked stream, preserving its cold-cache reuse.
        self.linear1_stream_weights = torch.cat(
            (
                self.placed_linear1.gate_weight_data,
                self.placed_linear1.up_weight_data,
            ),
            dim=1,
        ).contiguous()
        self.linear1_stream_scales = torch.cat(
            (
                self.placed_linear1.gate_weight_scale,
                self.placed_linear1.up_weight_scale,
            ),
            dim=1,
        ).contiguous()
        launcher = getattr(self.placed_linear1.gate_weight_tma, "launcher", None)
        if launcher is None:
            raise ValueError("resident FFN stream prepack requires one launcher")
        self.linear1_stream_tma = TmaTensor(
            launcher, self.linear1_stream_weights
        ).mxfp4_load(512)

        plans = torch.zeros((self.num_sms, 8), dtype=torch.int64, device="cpu")
        for worker, task_queue in enumerate(self.placed_down.task_queues):
            plans[worker, 0] = self.placed_linear1.metadata[worker].data_ptr()
            for index, task in enumerate(task_queue):
                plans[worker, 1 + index] = (
                    self.placed_down.metadata[task].data_ptr()
                )
            plans[worker, 3] = len(task_queue)
            plans[worker, 4] = self.linear1_stream_scales[
                worker, 0
            ].data_ptr()
            plans[worker, 5] = (
                self.linear1_stream_tma.arg | (worker << 32)
            )
            plans[worker, 6] = (
                self.placed_linear1.activation_data.data_ptr()
            )
            plans[worker, 7] = (
                self.placed_linear1.activation_scale.data_ptr()
            )
        self.resident_plans = plans.to(self.placed_linear1.metadata.device)
        self.task_queues = self.placed_down.task_queues

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        plan_address = self.resident_plans[sm].data_ptr()
        instructions = [
            Mxfp4Mxfp8ResidentFfnSm100(plan_address),
            TmaLoadMxfpCoupledStream(
                plan_address,
                kind=TmaLoadMxfpCoupledStream.LINEAR1,
                stages=2,
                area_slots=(168 * 1024) // config.slot_size,
                area_id=0,
                mailbox=8,
                port=0,
            ),
            TmaLoadMxfpCoupledStream(
                plan_address,
                kind=TmaLoadMxfpCoupledStream.DOWN_WEIGHT,
                stages=2,
                area_slots=(76 * 1024 + config.slot_size - 1)
                    // config.slot_size,
                area_id=0,
                mailbox=6,
                port=0,
            ),
        ]
        # The second LDU receives the same operator with activation/SFB
        # geometry. Its mailbox is immutable while LDU0 locally chains the two
        # commands above through their shared area-zero ownership.
        instructions.append(
            TmaLoadMxfpCoupledStream(
                plan_address,
                kind=TmaLoadMxfpCoupledStream.DOWN_ACTIVATION,
                stages=2,
                area_slots=(76 * 1024 + config.slot_size - 1)
                    // config.slot_size,
                area_id=0,
                mailbox=7,
                port=1,
            )
        )
        return instructions


class SchedMxfp4Mxfp8RoutedResidentFfn(Schedule):
    """Route-selected all-MX resident FFN with one homogeneous weight image.

    The physical worker image remains seven experts: shared plus six routed
    ranks.  LDU0 maps each routed rank to its offline-packed expert block from
    the 128-byte route record; compute continues to operate on fixed physical
    activation records. The output is the N-major BF16 ``[8,4096]`` model
    handoff; routed and split-K tasks reduce directly into that tensor.
    """

    ROUTE_RECORD_BYTES = 128
    CHECKPOINT_EXPERTS = 256
    LINEAR1_SLICES = 16
    DOWN_SLICES = 32

    def __init__(
        self,
        linear1,
        down,
        route_record,
        linear1_stream_weights,
        linear1_stream_scales,
        down_stream_weights,
        down_stream_scales,
        *,
        profile_output_event=None,
    ):
        super().__init__()
        self.linear1 = linear1
        self.down = down
        self.route_record = route_record
        self.linear1_stream_weights = linear1_stream_weights
        self.linear1_stream_scales = linear1_stream_scales
        self.down_stream_weights = down_stream_weights
        self.down_stream_scales = down_stream_scales
        self.profile_output_event = profile_output_event

    def _validate_streams(self):
        stream_experts = self.CHECKPOINT_EXPERTS + 1
        linear1_tasks = stream_experts * self.LINEAR1_SLICES
        down_tasks = stream_experts * self.DOWN_SLICES
        if (
            self.route_record.dtype != torch.uint8
            or self.route_record.device.type != "cuda"
            or self.route_record.numel() != self.ROUTE_RECORD_BYTES
            or not self.route_record.is_contiguous()
        ):
            raise ValueError(
                "routed resident FFN requires one contiguous CUDA uint8[128] "
                "route record"
            )
        if (
            self.linear1_stream_weights.dtype != torch.uint8
            or tuple(self.linear1_stream_weights.shape)
            != (linear1_tasks, 16, 4, 128, 64)
            or not self.linear1_stream_weights.is_contiguous()
        ):
            raise ValueError(
                "offline Linear-1 MX stream must be uint8 "
                "[257*16,16,4,128,64]"
            )
        if (
            self.linear1_stream_scales.dtype != torch.uint8
            or tuple(self.linear1_stream_scales.shape)
            != (linear1_tasks, 16, 2048)
            or not self.linear1_stream_scales.is_contiguous()
        ):
            raise ValueError(
                "offline Linear-1 MX scales must be uint8 [257*16,16,2048]"
            )
        if (
            self.down_stream_weights.dtype != torch.uint8
            or tuple(self.down_stream_weights.shape)
            != (down_tasks, 8, 2, 128, 64)
            or not self.down_stream_weights.is_contiguous()
        ):
            raise ValueError(
                "offline Down MX stream must be uint8 [257*32,8,2,128,64]"
            )
        if (
            self.down_stream_scales.dtype != torch.uint8
            or tuple(self.down_stream_scales.shape)
            != (down_tasks, 8, 1024)
            or not self.down_stream_scales.is_contiguous()
        ):
            raise ValueError(
                "offline Down MX scales must be uint8 [257*32,8,1024]"
            )
        tensors = (
            self.linear1_stream_weights,
            self.linear1_stream_scales,
            self.down_stream_weights,
            self.down_stream_scales,
        )
        if any(tensor.device != self.route_record.device for tensor in tensors):
            raise ValueError("route metadata and offline MX streams must share a GPU")

    def _on_place(self):
        self._validate_streams()
        if self.num_sms not in (7 * self.LINEAR1_SLICES, 152):
            raise ValueError(
                "routed resident FFN requires 112 or 152 physical workers"
            )
        self.linear1_worker_base = self.num_sms - 7 * self.LINEAR1_SLICES
        self.placed_linear1 = self.linear1.place(7 * self.LINEAR1_SLICES)
        self.placed_down = self.down.place(self.num_sms)
        if self.placed_linear1.m_tiles != 7 * self.LINEAR1_SLICES:
            raise ValueError("routed resident FFN requires one Linear-1 task per worker")
        if self.placed_down.fp32_output or not self.placed_down.output_n_major:
            raise ValueError(
                "routed resident FFN output must use BF16 [8,4096] layout"
            )
        if max(map(len, self.placed_down.task_queues)) > 2:
            raise ValueError("routed resident FFN supports two Down tasks per worker")
        if (
            self.placed_down.experts != 7
            or self.placed_down.DOWN_TILES_PER_EXPERT != self.DOWN_SLICES
        ):
            raise ValueError("routed resident FFN needs seven physical expert slots")

        self.down_metadata_variants = self.placed_down.metadata_variants
        self.placed_down.task_queues = [[] for _ in range(self.num_sms)]
        if self.num_sms == 112:
            for worker in range(self.num_sms):
                physical_expert, local_slice = divmod(
                    worker, self.LINEAR1_SLICES
                )
                expert_base = physical_expert * self.DOWN_SLICES
                self.placed_down.task_queues[worker] = [
                    expert_base + local_slice,
                    expert_base + 16 + local_slice,
                ]
        else:
            if self.down_metadata_variants != 3:
                raise ValueError(
                    "152-worker Down balancing requires full/half metadata"
                )
            # One full K2048 tile per worker forms the first wave. Replace the
            # remaining 72 full tiles with 144 K1024 halves, assigning at most
            # one half to each worker. Every SM therefore carries either 1.0
            # or 1.5 equal-work tiles instead of the old 1-or-2 tile tail.
            for worker in range(self.num_sms):
                self.placed_down.task_queues[worker].append(3 * worker)
            split_record = 0
            for output_task in range(self.num_sms, self.placed_down.tasks):
                for split_variant in (1, 2):
                    self.placed_down.task_queues[split_record].append(
                        3 * output_task + split_variant
                    )
                    split_record += 1
        if max(map(len, self.placed_down.task_queues)) > 2:
            raise ValueError("routed resident FFN supports two Down tasks per worker")

        launcher = getattr(self.placed_linear1.gate_weight_tma, "launcher", None)
        if launcher is None:
            raise ValueError("routed resident streams require one launcher")
        self.linear1_stream_tma = TmaTensor(
            launcher, self.linear1_stream_weights
        ).mxfp4_load(512)
        self.down_stream_tma = TmaTensor(
            launcher, self.down_stream_weights
        ).mxfp4_load(256)

        plans = torch.zeros((self.num_sms, 13), dtype=torch.int64, device="cpu")
        linear1_scale_base = self.linear1_stream_scales.data_ptr()
        for worker, task_queue in enumerate(self.placed_down.task_queues):
            has_linear1 = worker >= self.linear1_worker_base
            linear1_worker = worker - self.linear1_worker_base
            physical_expert = (
                linear1_worker // self.LINEAR1_SLICES
                if has_linear1 else 0
            )
            local_slice = (
                linear1_worker % self.LINEAR1_SLICES
                if has_linear1 else 0
            )
            route_rank = physical_expert - 1
            if has_linear1:
                plans[worker, 0] = self.placed_linear1.metadata[
                    linear1_worker
                ].data_ptr()
            plans[worker, 1] = self.placed_down.metadata[task_queue[0]].data_ptr()
            if len(task_queue) > 1:
                plans[worker, 2] = self.placed_down.metadata[
                    task_queue[1]
                ].data_ptr()
            plans[worker, 3] = len(task_queue)
            if has_linear1 and physical_expert == 0:
                plans[worker, 4] = self.linear1_stream_scales[
                    linear1_worker, 0
                ].data_ptr()
                linear1_coordinate = linear1_worker
            elif has_linear1:
                plans[worker, 4] = linear1_scale_base
                linear1_coordinate = local_slice
            if has_linear1:
                plans[worker, 5] = (
                    self.linear1_stream_tma.arg | (linear1_coordinate << 32)
                )
                plans[worker, 6] = self.placed_linear1.activation_data.data_ptr()
                plans[worker, 7] = self.placed_linear1.activation_scale.data_ptr()
            plans[worker, 8] = self.route_record.data_ptr()
            plans[worker, 9] = (
                (route_rank & 0xFFFFFFFF) | (local_slice << 32)
            )
            plans[worker, 10] = self.down_stream_scales.data_ptr()
            plans[worker, 11] = self.down_stream_tma.arg
            plans[worker, 12] = int(has_linear1)
        self.resident_plans = plans.to(self.route_record.device)
        self.task_queues = self.placed_down.task_queues

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        plan_address = self.resident_plans[sm].data_ptr()
        has_linear1 = sm >= self.linear1_worker_base
        linear1_worker = sm - self.linear1_worker_base
        linear1_dynamic_expert = (
            has_linear1 and linear1_worker >= self.LINEAR1_SLICES
        )
        down_dynamic_expert = self.down_metadata_variants == 3 or (
            self.task_queues[sm][0] // self.DOWN_SLICES > 0
        )
        instructions = [Mxfp4Mxfp8RoutedResidentFfnSm100(plan_address)]
        if has_linear1:
            instructions.append(
                TmaLoadMxfpCoupledStream(
                    plan_address,
                    kind=TmaLoadMxfpCoupledStream.LINEAR1,
                    stages=2,
                    area_slots=(168 * 1024) // config.slot_size,
                    area_id=0,
                    # Ordinary router and prepared-top-k raw operands occupy
                    # special mailboxes 1, 3, 4, and 5. Use the final mailbox
                    # so look-ahead cannot overwrite a live raw operand.
                    mailbox=8,
                    port=0,
                    dynamic_expert=linear1_dynamic_expert,
                ).bar(self._bar("input"))
            )
        down_weight = TmaLoadMxfpCoupledStream(
            plan_address,
            kind=TmaLoadMxfpCoupledStream.DOWN_WEIGHT,
            stages=2,
            area_slots=(76 * 1024 + config.slot_size - 1)
                // config.slot_size,
            area_id=0,
            mailbox=6,
            port=0,
            dynamic_expert=down_dynamic_expert,
            down_only=not has_linear1,
        )
        if not has_linear1:
            down_weight.bar(self._bar("input"))
        instructions.extend(
            [
                down_weight,
            TmaLoadMxfpCoupledStream(
                plan_address,
                kind=TmaLoadMxfpCoupledStream.DOWN_ACTIVATION,
                stages=2,
                area_slots=(76 * 1024 + config.slot_size - 1)
                    // config.slot_size,
                area_id=0,
                mailbox=7,
                port=1,
                down_only=not has_linear1,
            ),
            RawAddress(
                self.placed_down.final_output,
                config.num_slots + 5,
                profile_event=self.profile_output_event,
            ).writeback().bar(self._bar("output")).fixed_port(1),
            ]
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedLayeredMxfp4Mxfp8RoutedResidentFfn(Schedule):
    """Counter-select per-layer MX streams without expanding the FFN task.

    Compute keeps one layer-invariant resident plan.  Only the adjacent
    Linear-1 and Down-weight LDU commands receive a counter-derived plan
    address; keeping those two commands adjacent preserves the allocator-less
    retained-ring handoff in the normal launcher rewrite.
    """

    def __init__(
        self,
        resident,
        linear1_metadata_layers,
        down_metadata_layers,
        linear1_weight_layers,
        linear1_scale_layers,
        down_weight_layers,
        down_scale_layers,
        *,
        counter_strides=(),
        linear1_tmas=None,
        down_tmas=None,
    ):
        super().__init__()
        self.resident = resident
        self.linear1_metadata_layers = tuple(linear1_metadata_layers)
        self.down_metadata_layers = tuple(down_metadata_layers)
        self.linear1_weight_layers = tuple(linear1_weight_layers)
        self.linear1_scale_layers = tuple(linear1_scale_layers)
        self.down_weight_layers = tuple(down_weight_layers)
        self.down_scale_layers = tuple(down_scale_layers)
        self.counter_strides = tuple(
            (int(counter), int(stride))
            for counter, stride in counter_strides
        )
        self.supplied_linear1_tmas = (
            None if linear1_tmas is None else tuple(linear1_tmas)
        )
        self.supplied_down_tmas = (
            None if down_tmas is None else tuple(down_tmas)
        )

    def _on_place(self):
        layer_groups = (
            self.linear1_metadata_layers,
            self.down_metadata_layers,
            self.linear1_weight_layers,
            self.linear1_scale_layers,
            self.down_weight_layers,
            self.down_scale_layers,
        )
        layer_counts = {len(group) for group in layer_groups}
        if len(layer_counts) != 1 or next(iter(layer_counts)) <= 0:
            raise ValueError("layered resident FFN inputs need one common layer count")
        self.layer_count = next(iter(layer_counts))
        if self.layer_count > 1 and not self.counter_strides:
            raise ValueError("multi-layer resident FFN requires loop-counter strides")
        if any(
            counter < 0 or counter >= 32 or stride <= 0
            for counter, stride in self.counter_strides
        ):
            raise ValueError(
                "resident FFN counter strides require reg [0,31] and positive stride"
            )

        resident = self.resident._clone()
        resident._bars.update(self._bars)
        self.placed_resident = resident.place(self.num_sms)
        if self.num_sms not in (112, 152):
            raise ValueError(
                "layered routed resident FFN requires 112 or 152 workers"
            )
        device = self.placed_resident.route_record.device
        representatives = (
            self.placed_resident.linear1_stream_weights,
            self.placed_resident.linear1_stream_scales,
            self.placed_resident.down_stream_weights,
            self.placed_resident.down_stream_scales,
        )
        stream_groups = (
            self.linear1_weight_layers,
            self.linear1_scale_layers,
            self.down_weight_layers,
            self.down_scale_layers,
        )
        for representative, layers in zip(representatives, stream_groups):
            if layers[0].data_ptr() != representative.data_ptr():
                raise ValueError(
                    "layer zero MXFP stream must match the resident representative"
                )
            for tensor in layers:
                if (
                    tensor.device != device
                    or tensor.dtype != representative.dtype
                    or tensor.shape != representative.shape
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "layered MXFP streams must match representative storage"
                    )
        expected_linear1_metadata = (112, 128)
        expected_down_metadata = (
            7 * 32 * self.placed_resident.down_metadata_variants,
            128,
        )
        for layer in range(self.layer_count):
            linear1_metadata = self.linear1_metadata_layers[layer]
            down_metadata = self.down_metadata_layers[layer]
            if (
                linear1_metadata.device != device
                or linear1_metadata.dtype != torch.uint8
                or tuple(linear1_metadata.shape) != expected_linear1_metadata
                or not linear1_metadata.is_contiguous()
            ):
                raise ValueError(
                    "layered Linear-1 metadata must be CUDA uint8[112,128]"
                )
            if (
                down_metadata.device != device
                or down_metadata.dtype != torch.uint8
                or tuple(down_metadata.shape) != expected_down_metadata
                or not down_metadata.is_contiguous()
            ):
                raise ValueError(
                    "layered Down metadata must be CUDA uint8[224,128]"
                )
        if (
            self.linear1_metadata_layers[0].data_ptr()
            != self.placed_resident.placed_linear1.metadata.data_ptr()
            or self.down_metadata_layers[0].data_ptr()
            != self.placed_resident.placed_down.metadata.data_ptr()
        ):
            raise ValueError(
                "layer zero metadata must match the resident representative"
            )

        # The representative schedule already validates layer zero and owns
        # its two descriptors.  Materialize only the additional per-layer TMA
        # maps; no data or scale payload is repacked here.
        if self.supplied_linear1_tmas is None:
            linear1_tmas = [self.placed_resident.linear1_stream_tma]
            for layer in range(1, self.layer_count):
                linear1_tmas.append(
                    TmaTensor(
                        linear1_tmas[0].launcher,
                        self.linear1_weight_layers[layer],
                    ).mxfp4_load(512)
                )
        else:
            linear1_tmas = list(self.supplied_linear1_tmas)
        if self.supplied_down_tmas is None:
            down_tmas = [self.placed_resident.down_stream_tma]
            for layer in range(1, self.layer_count):
                down_tmas.append(
                    TmaTensor(
                        down_tmas[0].launcher,
                        self.down_weight_layers[layer],
                    ).mxfp4_load(256)
                )
        else:
            down_tmas = list(self.supplied_down_tmas)
        if len(linear1_tmas) != self.layer_count or len(down_tmas) != self.layer_count:
            raise ValueError("supplied layered MXFP TMA lists must cover every layer")
        if any(
            getattr(tma, "launcher", None)
            is not getattr(self.placed_resident.linear1_stream_tma, "launcher", None)
            for tma in (*linear1_tmas, *down_tmas)
        ):
            raise ValueError("layered MXFP TMA descriptors must share one launcher")
        self.linear1_tmas = tuple(linear1_tmas)
        self.down_tmas = tuple(down_tmas)

        plans = torch.zeros(
            (self.layer_count, self.num_sms, 13),
            dtype=torch.int64,
            device="cpu",
        )
        route_record = self.placed_resident.route_record
        activation_data = self.placed_resident.placed_linear1.activation_data
        activation_scale = self.placed_resident.placed_linear1.activation_scale
        task_queues = self.placed_resident.task_queues
        for layer in range(self.layer_count):
            linear1_scales = self.linear1_scale_layers[layer]
            down_scales = self.down_scale_layers[layer]
            if (
                linear1_scales.device != device
                or down_scales.device != device
                or not linear1_scales.is_contiguous()
                or not down_scales.is_contiguous()
            ):
                raise ValueError("layered MXFP scale streams must be contiguous CUDA")
            for worker, task_queue in enumerate(task_queues):
                has_linear1 = worker >= self.placed_resident.linear1_worker_base
                linear1_worker = (
                    worker - self.placed_resident.linear1_worker_base
                )
                physical_expert = (
                    linear1_worker // 16 if has_linear1 else 0
                )
                local_slice = linear1_worker % 16 if has_linear1 else 0
                route_rank = physical_expert - 1
                if has_linear1:
                    plans[layer, worker, 0] = self.linear1_metadata_layers[
                        layer
                    ][linear1_worker].data_ptr()
                plans[layer, worker, 1] = (
                    self.down_metadata_layers[layer][task_queue[0]].data_ptr()
                )
                if len(task_queue) > 1:
                    plans[layer, worker, 2] = self.down_metadata_layers[
                        layer
                    ][task_queue[1]].data_ptr()
                plans[layer, worker, 3] = len(task_queue)
                if has_linear1 and physical_expert == 0:
                    plans[layer, worker, 4] = linear1_scales[
                        linear1_worker, 0
                    ].data_ptr()
                    linear1_coordinate = linear1_worker
                elif has_linear1:
                    plans[layer, worker, 4] = linear1_scales.data_ptr()
                    linear1_coordinate = local_slice
                if has_linear1:
                    plans[layer, worker, 5] = (
                        self.linear1_tmas[layer].arg
                        | (linear1_coordinate << 32)
                    )
                    plans[layer, worker, 6] = activation_data.data_ptr()
                    plans[layer, worker, 7] = activation_scale.data_ptr()
                plans[layer, worker, 8] = route_record.data_ptr()
                plans[layer, worker, 9] = (
                    (route_rank & 0xFFFFFFFF) | (local_slice << 32)
                )
                plans[layer, worker, 10] = down_scales.data_ptr()
                plans[layer, worker, 11] = self.down_tmas[layer].arg
                plans[layer, worker, 12] = int(has_linear1)
        self.layered_plans = plans.to(device)
        self.plan_layer_bytes = (
            self.layered_plans.stride(0)
            * self.layered_plans.element_size()
        )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        base = self.placed_resident.schedule(sm)
        if len(base) not in (4, 5):
            raise ValueError("routed resident FFN command contract changed")
        compute = base[0]
        output = base[-1]
        down_activation = base[-2]
        down_weight = base[-3]
        linear1 = base[1] if len(base) == 5 else None
        plan_address = self.layered_plans[0, sm].data_ptr()
        down_weight = down_weight.copy()
        down_weight.set_cords(addr2cords(plan_address))
        down_activation = down_activation.copy()
        down_activation.set_cords(addr2cords(plan_address))
        offsets = tuple(
            (counter, stride * self.plan_layer_bytes)
            for counter, stride in self.counter_strides
        )
        if linear1 is not None:
            linear1 = linear1.copy()
            linear1.set_cords(addr2cords(plan_address))
            weight_window = RepeatM.offsetWindowByCounters(
                offsets, linear1, down_weight
            )
        else:
            weight_window = RepeatM.offsetWindowByCounters(
                offsets, down_weight
            )
        return [compute, weight_window, down_activation, output]

    def bar_release_count(self, role: str):
        self._require_placed()
        return self.placed_resident.bar_release_count(role)


class SchedFp8GemvUmmaStream(Schedule):
    """Shape-sharded M128/K128 native MXF8 projection."""

    TILE_M = SchedFp8UmmaPrepack.TILE_M
    WEIGHT_TILE_BYTES = SchedFp8UmmaPrepack.WEIGHT_TILE_BYTES
    WEIGHT_DATA_BYTES = SchedFp8UmmaPrepack.TILE_M * SchedFp8UmmaPrepack.TILE_K
    ACTIVATION_TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES
    ACTIVATION_TILES_PER_CHUNK = 4
    PACKED_ACTIVATION_TILES_PER_CHUNK = 8

    def __init__(
        self,
        weight_tiles,
        activation_tiles,
        output,
        scale_pack: int = 1,
        output_group_size: int = 1,
    ):
        super().__init__()
        self.weight_tiles = weight_tiles
        self.activation_tiles = activation_tiles
        self.output = output
        self.scale_pack = int(scale_pack)
        self.output_group_size = int(output_group_size)

    def _on_place(self):
        if (
            self.weight_tiles.dtype != torch.uint8
            or self.weight_tiles.ndim != 3
            or self.weight_tiles.shape[2] != self.WEIGHT_TILE_BYTES
            or not self.weight_tiles.is_contiguous()
        ):
            raise ValueError(
                "native FP8 weights must use combined 16896-byte M128/K128 tiles"
            )
        self.m_tiles, self.k_tiles, _ = self.weight_tiles.shape
        if not 0 < self.num_sms <= self.m_tiles:
            raise ValueError("native FP8 GEMV needs 1..M/128 SMs")
        if not 0 < self.k_tiles <= 64:
            raise ValueError("native FP8 GEMV supports 1..64 K128 tiles")
        if self.scale_pack not in (1, 2, 4) or self.k_tiles % self.scale_pack:
            raise ValueError(
                "native FP8 GEMV scale pack must be 1, 2, or 4 and divide K tiles"
            )
        if self.ACTIVATION_TILES_PER_CHUNK % self.scale_pack:
            raise ValueError("native FP8 scale pack must divide the activation chunk")
        self.activation_tiles_per_chunk = (
            self.ACTIVATION_TILES_PER_CHUNK
            if self.scale_pack == 1
            else self.PACKED_ACTIVATION_TILES_PER_CHUNK
        )
        if self.output_group_size not in (1, 2):
            raise ValueError("native FP8 output group size must be 1 or 2")
        if self.output_group_size > 1 and self.scale_pack == 1:
            raise ValueError("grouped native FP8 GEMV requires packed scales")
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
        tile_stop = tile_start + tile_count
        for group_start in range(
            tile_start, tile_stop, self.output_group_size
        ):
            group_stop = min(group_start + self.output_group_size, tile_stop)
            output_tiles = range(group_start, group_stop)
            instructions.append(
                Fp8GemvUmmaStreamSm100(
                    self.k_tiles, self.scale_pack, group_stop - group_start
                )
            )
            for chunk_start in range(
                0, self.k_tiles, self.activation_tiles_per_chunk
            ):
                chunk_stop = min(
                    chunk_start + self.activation_tiles_per_chunk,
                    self.k_tiles,
                )
                instructions.append(
                    TmaLoad1D(
                        self.activation_tiles[
                            chunk_start:chunk_stop
                        ].reshape(-1)
                    ).fixed_port(1)
                )
                for scale_start in range(
                    chunk_start, chunk_stop, self.scale_pack
                ):
                    for output_tile in output_tiles:
                        for k_tile in range(
                            scale_start, scale_start + self.scale_pack
                        ):
                            weight = self.weight_tiles[
                                output_tile, k_tile
                            ].reshape(-1)
                            if k_tile % self.scale_pack:
                                weight = weight[: self.WEIGHT_DATA_BYTES]
                            instructions.append(
                                TmaLoad1D(weight).fixed_port(0)
                            )
            for output_tile in output_tiles:
                row_start = output_tile * self.TILE_M
                store = TmaStore1D(
                    self.output[row_start : row_start + self.TILE_M]
                )
                if output_tile + 1 == tile_stop:
                    store.bar(self._bar("output"))
                instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedFp8GemvUmmaSplitK(Schedule):
    """Split native MXF8 K over SMs and reduce FP32 partials in STU."""

    TILE_M = SchedFp8UmmaPrepack.TILE_M
    WEIGHT_TILE_BYTES = SchedFp8UmmaPrepack.WEIGHT_TILE_BYTES
    WEIGHT_DATA_BYTES = SchedFp8UmmaPrepack.TILE_M * SchedFp8UmmaPrepack.TILE_K
    ACTIVATION_TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES
    ACTIVATION_TILES_PER_CHUNK = 4
    PACKED_ACTIVATION_TILES_PER_CHUNK = 8
    OUTPUT_ROWS = 1

    def __init__(
        self,
        weight_tiles,
        activation_tiles,
        output_reduce,
        split_k: int,
        scale_pack: int = 1,
        output_group_size: int = 1,
    ):
        super().__init__()
        self.weight_tiles = weight_tiles
        self.activation_tiles = activation_tiles
        self.output_reduce = output_reduce
        self.split_k = int(split_k)
        self.scale_pack = int(scale_pack)
        self.output_group_size = int(output_group_size)

    def _on_place(self):
        if (
            self.weight_tiles.dtype != torch.uint8
            or self.weight_tiles.ndim != 3
            or self.weight_tiles.shape[2] != self.WEIGHT_TILE_BYTES
            or not self.weight_tiles.is_contiguous()
        ):
            raise ValueError(
                "split-K native FP8 weights must be combined "
                "16896-byte M128/K128 tiles"
            )
        self.m_tiles, self.k_tiles, _ = self.weight_tiles.shape
        if self.split_k <= 1 or self.k_tiles % self.split_k:
            raise ValueError("split_k must divide K tiles and be greater than one")
        self.k_tiles_per_split = self.k_tiles // self.split_k
        if (
            self.scale_pack not in (1, 2, 4)
            or self.k_tiles_per_split % self.scale_pack
        ):
            raise ValueError(
                "split-K FP8 scale pack must be 1, 2, or 4 and divide each shard"
            )
        if self.ACTIVATION_TILES_PER_CHUNK % self.scale_pack:
            raise ValueError("split-K FP8 scale pack must divide the activation chunk")
        self.activation_tiles_per_chunk = (
            self.ACTIVATION_TILES_PER_CHUNK
            if self.scale_pack == 1
            else self.PACKED_ACTIVATION_TILES_PER_CHUNK
        )
        if self.output_group_size not in (1, 2):
            raise ValueError("split-K FP8 output group size must be 1 or 2")
        if self.output_group_size > 1 and self.scale_pack == 1:
            raise ValueError("grouped split-K FP8 GEMV requires packed scales")
        if (
            self.activation_tiles.dtype != torch.uint8
            or tuple(self.activation_tiles.shape)
            != (self.k_tiles, self.ACTIVATION_TILE_BYTES)
            or not self.activation_tiles.is_contiguous()
        ):
            raise ValueError(
                "split-K native FP8 activations must be [K/128,2048] uint8"
            )
        self.rows = self.m_tiles * self.TILE_M
        output = getattr(self.output_reduce, "mat", None)
        if (
            getattr(self.output_reduce, "mode", None) != "reduce"
            or output is None
            or output.dtype not in (torch.bfloat16, torch.float32)
            or tuple(output.shape) != (self.OUTPUT_ROWS, self.rows)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "split-K output must be a row-major TMA reduce tensor "
                "over contiguous BF16 or FP32 [1,M]"
            )
        self.reduction_bytes = output.element_size()
        self.work_tiles = self.m_tiles * self.split_k
        if not 0 < self.num_sms <= self.work_tiles:
            raise ValueError(
                f"split-K FP8 GEMV requires 1..{self.work_tiles} SMs"
            )

    def _work_shard(self, sm):
        work_per_sm, extra = divmod(self.work_tiles, self.num_sms)
        work_start = sm * work_per_sm + min(sm, extra)
        work_count = work_per_sm + int(sm < extra)
        return work_start, work_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        work_start, work_count = self._work_shard(sm)
        work_stop = work_start + work_count
        instructions = []
        work = work_start
        while work < work_stop:
            split = work // self.m_tiles
            output_tile = work % self.m_tiles
            group_count = min(
                self.output_group_size,
                work_stop - work,
                self.m_tiles - output_tile,
            )
            output_tiles = range(output_tile, output_tile + group_count)
            k_start = split * self.k_tiles_per_split
            k_stop = k_start + self.k_tiles_per_split
            instructions.append(
                Fp8GemvUmmaSplitKSm100(
                    self.k_tiles_per_split,
                    self.reduction_bytes,
                    self.scale_pack,
                    group_count,
                )
            )
            for chunk_start in range(
                k_start, k_stop, self.activation_tiles_per_chunk
            ):
                chunk_stop = min(
                    k_stop,
                    chunk_start + self.activation_tiles_per_chunk,
                )
                instructions.append(
                    TmaLoad1D(
                        self.activation_tiles[
                            chunk_start:chunk_stop
                        ].reshape(-1)
                    ).fixed_port(1)
                )
                for scale_start in range(
                    chunk_start, chunk_stop, self.scale_pack
                ):
                    for grouped_output_tile in output_tiles:
                        for k_tile in range(
                            scale_start, scale_start + self.scale_pack
                        ):
                            weight = self.weight_tiles[
                                grouped_output_tile, k_tile
                            ].reshape(-1)
                            if k_tile % self.scale_pack:
                                weight = weight[: self.WEIGHT_DATA_BYTES]
                            instructions.append(
                                TmaLoad1D(weight).fixed_port(0)
                            )
            for grouped_output_tile in output_tiles:
                store = self.output_reduce.cord(
                    0, grouped_output_tile * self.TILE_M
                )
                if work + group_count == work_stop and (
                    grouped_output_tile + 1 == output_tile + group_count
                ):
                    store.bar(self._bar("output"))
                instructions.append(store)
            work += group_count
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedFp8GemvUmmaCoupled(Schedule):
    """Common M256/K256 retained-ring MXFP8 projection schedule.

    The input tensors use the existing native K128 prepack.  Placement performs
    setup-only compaction into homogeneous K-pair streams; the device operator
    receives no projection identity or shape-specific implementation choice.
    An optional leading batch dimension represents independent projections
    with distinct weights and activations, such as the eight O-a projections.
    """

    TILE_M = 128
    OUTPUT_TILES = 2
    TILE_K = 128
    SCALE_PACK = 2
    WEIGHT_TILE_BYTES = 16896
    WEIGHT_DATA_BYTES = TILE_M * TILE_K
    WEIGHT_SCALE_BYTES = 512
    ACTIVATION_TILE_BYTES = 2048
    ACTIVATION_DATA_BYTES = 8 * TILE_K
    WEIGHT_STREAM_BYTES = (
        OUTPUT_TILES * SCALE_PACK * WEIGHT_DATA_BYTES
        + OUTPUT_TILES * WEIGHT_SCALE_BYTES
    )
    AREA_SLOTS = TmaLoadMxfpCoupledStream.FP8_AREA_SLOTS

    def __init__(
        self,
        weight_tiles,
        activation_tiles,
        output,
        *,
        split_k: int = 1,
        balanced_k: bool = False,
        weight_layers=None,
    ):
        super().__init__()
        self.weight_tiles = weight_tiles
        self.weight_layers = (
            (weight_tiles,)
            if weight_layers is None
            else tuple(weight_layers)
        )
        self.activation_tiles = activation_tiles
        self.output = output
        self.split_k = int(split_k)
        self.balanced_k = bool(balanced_k)

    def _on_place(self):
        if not self.weight_layers:
            raise ValueError("coupled FP8 requires at least one weight layer")
        for layer_weight in self.weight_layers:
            if (
                not isinstance(layer_weight, torch.Tensor)
                or layer_weight.dtype != torch.uint8
                or layer_weight.ndim not in (3, 4)
                or layer_weight.shape[-1] != self.WEIGHT_TILE_BYTES
                or not layer_weight.is_contiguous()
            ):
                raise ValueError(
                    "coupled FP8 weights must be contiguous uint8 "
                    "[M/128,K/128,16896] or [B,M/128,K/128,16896]"
                )
            if (
                layer_weight.device != self.weight_tiles.device
                or layer_weight.shape != self.weight_tiles.shape
            ):
                raise ValueError(
                    "coupled FP8 weight layers must share device and shape"
                )
        if self.weight_layers[0].data_ptr() != self.weight_tiles.data_ptr():
            raise ValueError(
                "the first coupled FP8 weight layer must be the representative"
            )
        self.weight_batches_by_layer = tuple(
            weight.unsqueeze(0) if weight.ndim == 3 else weight
            for weight in self.weight_layers
        )
        self.weight_batches = self.weight_batches_by_layer[0]
        self.layer_count = len(self.weight_batches_by_layer)
        self.batch_size, self.m_tiles, self.k_tiles, _ = (
            self.weight_batches.shape
        )
        if self.m_tiles % self.OUTPUT_TILES or self.k_tiles % self.SCALE_PACK:
            raise ValueError("coupled FP8 requires M128 and K256 alignment")
        if (
            self.activation_tiles.dtype != torch.uint8
            or self.activation_tiles.ndim not in (2, 3)
            or not self.activation_tiles.is_contiguous()
        ):
            raise ValueError(
                "coupled FP8 activations must be contiguous uint8 "
                "[K/128,2048] or [B,K/128,2048]"
            )
        self.activation_batches = (
            self.activation_tiles.unsqueeze(0)
            if self.activation_tiles.ndim == 2
            else self.activation_tiles
        )
        if tuple(self.activation_batches.shape) != (
            self.batch_size,
            self.k_tiles,
            self.ACTIVATION_TILE_BYTES,
        ):
            raise ValueError(
                "coupled FP8 activation batch/shape does not match weights"
            )
        self.m_pairs = self.m_tiles // self.OUTPUT_TILES
        self.k_pairs = self.k_tiles // self.SCALE_PACK
        if self.balanced_k and self.split_k != 1:
            raise ValueError(
                "balanced coupled FP8 placement cannot also use uniform split_k"
            )
        if self.split_k <= 0 or self.k_pairs % self.split_k:
            raise ValueError("coupled FP8 split_k must divide K/256")
        self.k_pairs_per_split = self.k_pairs // self.split_k
        self.work_per_batch = self.m_pairs * self.split_k
        self.work_tiles = self.batch_size * self.work_per_batch
        max_useful_sms = (
            self.batch_size * self.m_pairs * self.k_pairs
            if self.balanced_k
            else self.work_tiles
        )
        if not 0 < self.num_sms <= max_useful_sms:
            raise ValueError(
                f"coupled FP8 projection requires 1..{max_useful_sms} SMs"
            )
        if not self.balanced_k:
            max_work_per_sm = (
                self.work_tiles + self.num_sms - 1
            ) // self.num_sms
            if (
                max_work_per_sm * self.k_pairs_per_split
                > TmaLoadMxfpCoupledStream.MAX_PHASE_BASE + 1
            ):
                raise ValueError(
                    "coupled FP8 placement exceeds its per-SM phase window"
                )
        self.rows = self.m_tiles * self.TILE_M
        self.total_rows = self.batch_size * self.rows
        self.uses_reduction = self.split_k > 1 or self.balanced_k
        if not self.uses_reduction:
            if (
                not isinstance(self.output, torch.Tensor)
                or self.output.dtype != torch.bfloat16
                or self.output.numel() != self.total_rows
                or not self.output.is_contiguous()
            ):
                raise ValueError(
                    "unsplit coupled FP8 output must be contiguous BF16 [B,M]"
                )
            self.direct_output = self.output.reshape(-1)
            self.reduction_bytes = Fp8GemvUmmaCoupledSm100.BF16_BYTES
        else:
            output_mat = getattr(self.output, "mat", None)
            flat_reduce_shape = (1, self.total_rows)
            tiled_reduce_shape = (
                self.total_rows // self.TILE_M,
                self.TILE_M,
            )
            if (
                getattr(self.output, "mode", None) != "reduce"
                or output_mat is None
                or output_mat.dtype not in (torch.bfloat16, torch.float32)
                or tuple(output_mat.shape)
                not in (flat_reduce_shape, tiled_reduce_shape)
                or not output_mat.is_contiguous()
            ):
                raise ValueError(
                    "split coupled FP8 output must be contiguous BF16/FP32 "
                    "flat or M128-tiled row-major TMA reduction storage"
                )
            self.tiled_reduce_output = (
                tuple(output_mat.shape) == tiled_reduce_shape
            )
            self.reduction_bytes = output_mat.element_size()
            # The common task writes both adjacent M128 accumulators into one
            # allocator lease, so its store descriptor must cover the paired
            # M256 result as one transaction.
            if self.tiled_reduce_output:
                self.paired_reduce_output = TmaTensor(
                    self.output.launcher, output_mat
                ).rowmajor_2d("reduce", self.OUTPUT_TILES, self.TILE_M)
            else:
                self.paired_reduce_output = TmaTensor(
                    self.output.launcher, output_mat
                ).rowmajor_2d(
                    "reduce", 1, self.OUTPUT_TILES * self.TILE_M
                )

        # Offline layout conversion: data for the four M128/K128 products is
        # contiguous, followed by one packed SFA image per output.  The
        # immutable weights are compacted during setup.  Activations remain in
        # their producer-written two-tile layout; LDU1 gathers the two data
        # images and the packed SFB image directly so placement never snapshots
        # a dynamic activation before its producer runs.
        def compact_weight(weight_batches):
            weight_data = (
                weight_batches[..., : self.WEIGHT_DATA_BYTES]
                .contiguous()
                .reshape(
                    self.batch_size,
                    self.m_pairs,
                    self.OUTPUT_TILES,
                    self.k_pairs,
                    self.SCALE_PACK,
                    self.WEIGHT_DATA_BYTES,
                )
                .permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .reshape(self.batch_size, self.m_pairs, self.k_pairs, -1)
            )
            weight_scale = (
                weight_batches[
                    ..., :: self.SCALE_PACK, self.WEIGHT_DATA_BYTES :
                ]
                .contiguous()
                .reshape(
                    self.batch_size,
                    self.m_pairs,
                    self.OUTPUT_TILES,
                    self.k_pairs,
                    self.WEIGHT_SCALE_BYTES,
                )
                .permute(0, 1, 3, 2, 4)
                .contiguous()
                .reshape(self.batch_size, self.m_pairs, self.k_pairs, -1)
            )
            stream = torch.cat((weight_data, weight_scale), dim=3).contiguous()
            if stream.shape[3] != self.WEIGHT_STREAM_BYTES:
                raise ValueError("internal coupled FP8 weight layout mismatch")
            return stream

        self.weight_stream_layers = tuple(
            compact_weight(weight_batches)
            for weight_batches in self.weight_batches_by_layer
        )
        self.weight_stream = self.weight_stream_layers[0]
        if self.balanced_k:
            self._build_balanced_tasks()
        else:
            plans = torch.empty(
                (self.work_tiles, self.layer_count, 2),
                dtype=torch.int64,
                device="cpu",
            )
            for work in range(self.work_tiles):
                batch, batch_work = divmod(work, self.work_per_batch)
                split, m_pair = divmod(batch_work, self.m_pairs)
                pair_start = split * self.k_pairs_per_split
                for layer, weight_stream in enumerate(
                    self.weight_stream_layers
                ):
                    plans[work, layer, 0] = weight_stream[
                        batch, m_pair, pair_start
                    ].data_ptr()
                    plans[work, layer, 1] = self.activation_batches[
                        batch, pair_start * self.SCALE_PACK
                    ].data_ptr()
            self.stream_plans = plans.to(self.weight_tiles.device)

    def _build_balanced_tasks(self):
        """Split only the output tiles needed to equalize K-pair work."""
        output_pairs = self.batch_size * self.m_pairs
        whole_per_sm, remaining_outputs = divmod(
            output_pairs, self.num_sms
        )
        tasks = [[] for _ in range(self.num_sms)]
        pair_loads = [0 for _ in range(self.num_sms)]
        next_output = 0
        for _ in range(whole_per_sm):
            for sm in range(self.num_sms):
                batch, m_pair = divmod(next_output, self.m_pairs)
                tasks[sm].append((batch, m_pair, 0, self.k_pairs))
                pair_loads[sm] += self.k_pairs
                next_output += 1

        target_pair_load = (
            output_pairs * self.k_pairs + self.num_sms - 1
        ) // self.num_sms
        shard_limit = target_pair_load - whole_per_sm * self.k_pairs
        if remaining_outputs and shard_limit <= 0:
            raise ValueError("internal balanced coupled FP8 shard limit is zero")
        shards = []
        for output_index in range(
            next_output, next_output + remaining_outputs
        ):
            batch, m_pair = divmod(output_index, self.m_pairs)
            pair_start = 0
            while pair_start < self.k_pairs:
                pair_count = min(
                    shard_limit, self.k_pairs - pair_start
                )
                shards.append((batch, m_pair, pair_start, pair_count))
                pair_start += pair_count

        for task in shards:
            sm = min(
                range(self.num_sms),
                key=lambda index: (
                    pair_loads[index], len(tasks[index]), index
                ),
            )
            tasks[sm].append(task)
            pair_loads[sm] += task[3]

        if max(pair_loads, default=0) > (
            TmaLoadMxfpCoupledStream.MAX_PHASE_BASE + 1
        ):
            raise ValueError(
                "balanced coupled FP8 placement exceeds its per-SM phase window"
            )

        plan_specs = []
        self.balanced_tasks = [[] for _ in range(self.num_sms)]
        for sm, sm_tasks in enumerate(tasks):
            for batch, m_pair, pair_start, pair_count in sm_tasks:
                plan_index = len(plan_specs)
                plan_specs.append((batch, m_pair, pair_start))
                self.balanced_tasks[sm].append(
                    (batch, m_pair, pair_start, pair_count, plan_index)
                )
        self.work_tiles = len(plan_specs)
        self.active_sms = sum(bool(sm_tasks) for sm_tasks in tasks)
        plans = torch.empty(
            (self.work_tiles, self.layer_count, 2),
            dtype=torch.int64,
            device="cpu",
        )
        for plan_index, (batch, m_pair, pair_start) in enumerate(plan_specs):
            for layer, weight_stream in enumerate(self.weight_stream_layers):
                plans[plan_index, layer, 0] = weight_stream[
                    batch, m_pair, pair_start
                ].data_ptr()
                plans[plan_index, layer, 1] = self.activation_batches[
                    batch, pair_start * self.SCALE_PACK
                ].data_ptr()
        self.stream_plans = plans.to(self.weight_tiles.device)

    def _work_shard(self, sm):
        work_per_sm, extra = divmod(self.work_tiles, self.num_sms)
        work_start = sm * work_per_sm + min(sm, extra)
        work_count = work_per_sm + int(sm < extra)
        return work_start, work_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        if self.balanced_k:
            work_items = self.balanced_tasks[sm]
        else:
            work_start, work_count = self._work_shard(sm)
            work_items = []
            for work in range(work_start, work_start + work_count):
                batch, batch_work = divmod(work, self.work_per_batch)
                _, m_pair = divmod(batch_work, self.m_pairs)
                work_items.append(
                    (
                        batch,
                        m_pair,
                        0,
                        self.k_pairs_per_split,
                        work,
                    )
                )
        instructions = []
        phase_base = 0
        work_count = len(work_items)
        for local_work, (
            batch,
            m_pair,
            _,
            pair_count,
            plan_index,
        ) in enumerate(work_items):
            instructions.append(
                Fp8GemvUmmaCoupledSm100(
                    pair_count,
                    self.reduction_bytes,
                    phase_base,
                )
            )
            instructions.append(
                TmaLoadMxfpCoupledStream(
                    self.stream_plans[plan_index, 0].data_ptr(),
                    kind=TmaLoadMxfpCoupledStream.FP8_GEMV,
                    stages=TmaLoadMxfpCoupledStream.FP8_STAGES,
                    area_slots=self.AREA_SLOTS,
                    area_id=local_work,
                    stream_length=pair_count,
                    phase_base=phase_base,
                    layer_indexed=self.layer_count > 1,
                )
            )
            output_tile = (
                (batch * self.m_pairs + m_pair) * self.OUTPUT_TILES
            )
            if not self.uses_reduction:
                row_start = output_tile * self.TILE_M
                store = TmaStore1D(
                    self.direct_output[
                        row_start : row_start
                        + self.OUTPUT_TILES * self.TILE_M
                    ]
                )
            elif self.tiled_reduce_output:
                store = self.paired_reduce_output.cord(output_tile, 0)
            else:
                store = self.paired_reduce_output.cord(
                    0, output_tile * self.TILE_M
                )
            if local_work + 1 == work_count:
                store.bar(self._bar("output"))
            instructions.append(store)
            phase_base += pair_count
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        release_sms = (
            self.active_sms if self.balanced_k else self.num_sms
        )
        return self._bar_release_if_present(role, release_sms)


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


class SchedFp8Block128GateUpSwiGlu(Schedule):
    """Interleave gate/up row shards and apply SwiGLU in the up epilogue."""

    def __init__(
        self,
        gate_weight,
        gate_scale,
        up_weight,
        up_scale,
        activation,
        activation_scale,
        output,
        *,
        swiglu_limit=10.0,
    ):
        super().__init__()
        self.gate_weight = gate_weight
        self.gate_scale = gate_scale
        self.up_weight = up_weight
        self.up_scale = up_scale
        self.activation = activation
        self.activation_scale = activation_scale
        self.output = output
        self.swiglu_limit = float(swiglu_limit)

    def _on_place(self):
        if (
            self.gate_weight.dtype != torch.float8_e4m3fn
            or self.gate_weight.ndim != 2
            or self.up_weight.dtype != self.gate_weight.dtype
            or self.up_weight.shape != self.gate_weight.shape
        ):
            raise ValueError("gate/up FP8 weights must have one matching shape")
        self.rows, self.k = self.gate_weight.shape
        if self.k % 128:
            raise ValueError("gate/up FP8 GEMV K must be divisible by 128")
        expected_scale_shape = ((self.rows + 127) // 128, self.k // 128)
        for scale in (self.gate_scale, self.up_scale):
            if (
                scale.dtype != torch.float8_e8m0fnu
                or tuple(scale.shape) != expected_scale_shape
            ):
                raise ValueError(
                    "gate/up weight scales must match block-128 weight shape"
                )
        if (
            self.activation.dtype != torch.float8_e4m3fn
            or self.activation.numel() != self.k
            or self.activation_scale.dtype != torch.float8_e8m0fnu
            or self.activation_scale.numel() != self.k // 128
        ):
            raise ValueError("gate/up activation must use one scale per K128")
        if (
            self.output.dtype != torch.bfloat16
            or self.output.numel() != self.rows
            or not self.output.is_contiguous()
        ):
            raise ValueError("gate/up SwiGLU output must be contiguous M BF16")
        if not 0 < self.num_sms <= self.rows:
            raise ValueError("gate/up FP8 GEMV requires 1..M SMs")
        if self.swiglu_limit <= 0:
            raise ValueError("bounded SwiGLU limit must be positive")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + int(sm < extra)
        row_end = row_start + row_count
        max_tile_rows = max(1, 65520 // self.k)
        instructions = []
        for tile_start in range(row_start, row_end, max_tile_rows):
            tile_end = min(tile_start + max_tile_rows, row_end)
            tile_rows = tile_end - tile_start
            scale_start = tile_start // 128
            scale_end = (tile_end + 127) // 128
            row_in_scale_block = tile_start % 128
            instructions.extend(
                (
                    Fp8Block128GemvSm100(
                        tile_rows, self.k, row_in_scale_block
                    ),
                    _shared_load_1d(self.gate_weight[tile_start:tile_end]),
                    _shared_load_1d(self.gate_scale[scale_start:scale_end]),
                    _shared_load_1d(self.activation.reshape(-1)),
                    _shared_load_1d(self.activation_scale.reshape(-1)),
                    RegStore(1, size=tile_rows * 2).fixed_port(0),
                    Fp8Block128GemvSm100(
                        tile_rows,
                        self.k,
                        row_in_scale_block,
                        swiglu_limit=self.swiglu_limit,
                    ),
                    _shared_load_1d(self.up_weight[tile_start:tile_end]),
                    _shared_load_1d(self.up_scale[scale_start:scale_end]),
                    _shared_load_1d(self.activation.reshape(-1)),
                    _shared_load_1d(self.activation_scale.reshape(-1)),
                    RegLoad(1).fixed_port(0),
                )
            )
            store = _shared_store_1d(self.output[tile_start:tile_end])
            if tile_end == row_end:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedFp8Block128GemvBf16(Schedule):
    """Fuse BF16 block-128 quantization into each row-sharded FP8 GEMV."""

    def __init__(self, weight, weight_scale, activation, output):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.activation = activation
        self.output = output

    def _on_place(self):
        if self.weight.dtype != torch.float8_e4m3fn or self.weight.ndim != 2:
            raise ValueError("fused FP8 weight must be a rank-2 E4M3 tensor")
        self.rows, self.k = self.weight.shape
        if self.k % 128:
            raise ValueError("fused FP8 GEMV K must be divisible by 128")
        scratch_bytes = (
            config.dynamic_smem_size - config.num_slots * config.slot_size
        )
        if self.k + self.k // 128 > scratch_bytes:
            raise ValueError("fused FP8 activation does not fit special shared scratch")
        expected_weight_sf = ((self.rows + 127) // 128, self.k // 128)
        if (
            self.weight_scale.dtype != torch.float8_e8m0fnu
            or tuple(self.weight_scale.shape) != expected_weight_sf
        ):
            raise ValueError(
                f"fused weight_scale must be UE8M0 with shape {expected_weight_sf}"
            )
        if (
            self.activation.dtype != torch.bfloat16
            or self.activation.numel() != self.k
            or not self.activation.is_contiguous()
        ):
            raise ValueError("fused FP8 activation must contain contiguous K BF16 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != self.rows:
            raise ValueError("fused FP8 output must contain M BF16 values")
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("fused FP8 GEMV requires 1 <= num_sms <= M")

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
                Fp8Block128GemvBf16Sm100(
                    tile_rows, self.k, tile_start % 128
                ),
                _shared_load_1d(self.weight[tile_start:tile_end]),
                _shared_load_1d(self.weight_scale[scale_start:scale_end]),
                _shared_load_1d(self.activation.reshape(-1)).bar(
                    self._bar("activation")
                ),
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


class SchedDsv4PreloadRopeTables(Schedule):
    """Copy one packed RoPE metadata record into fixed per-SM scratch."""

    def __init__(self, tables):
        super().__init__()
        self.tables = tuple(tables)

    def _on_place(self):
        if not 1 <= len(self.tables) <= 4:
            raise ValueError("DeepSeek resident RoPE preload requires 1-4 tables")
        for table in self.tables:
            if (
                table.dtype != torch.float32
                or tuple(table.shape) != (32, 2)
                or table.device.type != "cuda"
                or not table.is_contiguous()
            ):
                raise ValueError("DeepSeek RoPE table must be FP32 [32,2]")
        devices = {table.device for table in self.tables}
        if len(devices) != 1:
            raise ValueError("DeepSeek RoPE tables must share one CUDA device")
        # This setup-time pack is the persistent HBM layout consumed by every
        # SM.  A raw metadata record avoids fragmenting the allocator directly
        # in front of the 16-slot attention ring and its contiguous 8-slot Q.
        self.packed_tables = torch.stack(self.tables).contiguous()

    def schedule(self, sm):
        if sm < 0:
            return []
        return [
            Dsv4PreloadRopeTables(len(self.tables)),
            RawAddress(self.packed_tables, config.num_slots),
        ]


class SchedDsv4Rope512_64(Schedule):
    def __init__(
        self,
        input,
        table,
        output,
        inverse=False,
        fixed_table_id=None,
    ):
        super().__init__()
        self.input = input
        self.table = table
        self.output = output
        self.inverse = inverse
        self.fixed_table_id = fixed_table_id

    def _on_place(self):
        if (self.input.dtype != torch.bfloat16 or self.input.ndim != 2 or
                self.input.shape[1] != 512):
            raise ValueError("DeepSeek RoPE input must be BF16 [rows,512]")
        if self.output.dtype != torch.bfloat16 or self.output.shape != self.input.shape:
            raise ValueError("DeepSeek RoPE output must match the input")
        if (self.table.dtype != torch.float32 or
                tuple(self.table.shape) != (32, 2)):
            raise ValueError("DeepSeek RoPE table must be FP32 [32,2]")
        if self.fixed_table_id is not None and not 0 <= self.fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        self.rows = self.input.shape[0]
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek partial RoPE requires 1 <= num_sms <= rows")

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions.append(
                Dsv4Rope64(
                    1,
                    512,
                    self.inverse,
                    fixed_table_id=self.fixed_table_id,
                )
            )
            instructions.append(TmaLoad1D(self.input[row]))
            if self.fixed_table_id is None:
                instructions.append(TmaLoad1D(self.table))
            instructions.append(
                TmaStore1D(self.output[row]).bar(self._bar("output"))
            )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.rows)


class SchedDsv4RmsRope512_64(Schedule):
    """Fuse per-row 512-wide RMSNorm and the final-64 rotary transform."""

    def __init__(
        self,
        input,
        table,
        output,
        *,
        epsilon: float,
        weight=None,
        fixed_table_id=None,
        profile_store_event=None,
    ):
        super().__init__()
        self.input = input
        self.table = table
        self.output = output
        self.epsilon = epsilon
        self.weight = weight
        self.fixed_table_id = fixed_table_id
        self.profile_store_event = profile_store_event

    def _on_place(self):
        if (
            self.input.dtype != torch.bfloat16
            or self.input.ndim != 2
            or self.input.shape[1] != 512
            or not self.input.is_contiguous()
        ):
            raise ValueError("fused RMS/RoPE input must be contiguous BF16 [rows,512]")
        if (
            self.output.dtype != torch.bfloat16
            or self.output.shape != self.input.shape
            or not self.output.is_contiguous()
        ):
            raise ValueError("fused RMS/RoPE output must match the input")
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != (32, 2):
            raise ValueError("fused RMS/RoPE table must be FP32 [32,2]")
        if self.weight is not None and (
            self.weight.dtype != torch.bfloat16
            or tuple(self.weight.shape) != (512,)
            or not self.weight.is_contiguous()
        ):
            raise ValueError("fused RMS/RoPE weight must be BF16 [512]")
        if self.epsilon <= 0:
            raise ValueError("fused RMS/RoPE epsilon must be positive")
        if self.fixed_table_id is not None and not 0 <= self.fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        if self.profile_store_event is not None and not (
            0 <= self.profile_store_event < (1 << 15)
        ):
            raise ValueError("fused RMS/RoPE profile event must fit 15 bits")
        self.rows = self.input.shape[0]
        if not 0 < self.num_sms <= self.rows:
            raise ValueError("fused RMS/RoPE requires 1..rows SMs")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions.extend(
                (
                    Dsv4RmsRope512_64(
                        weighted=self.weight is not None,
                        epsilon=self.epsilon,
                        fixed_table_id=self.fixed_table_id,
                    ),
                    TmaLoad1D(self.input[row]),
                )
            )
            if self.weight is not None:
                instructions.append(TmaLoad1D(self.weight))
            if self.fixed_table_id is None:
                instructions.append(TmaLoad1D(self.table))
            store = TmaStore1D(self.output[row]).bar(self._bar("output"))
            if self.profile_store_event is not None:
                store.arg = (1 << 15) | self.profile_store_event
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.rows)


class SchedDsv4Fp32RmsRope512_64(Schedule):
    """Finalize FP32 split-K rows directly into attention-ready BF16."""

    def __init__(
        self,
        input,
        table,
        output,
        *,
        epsilon: float,
        weight=None,
        fixed_table_id=None,
    ):
        super().__init__()
        self.input = input
        self.table = table
        self.output = output
        self.epsilon = epsilon
        self.weight = weight
        self.fixed_table_id = fixed_table_id

    def _on_place(self):
        if (
            self.input.dtype != torch.float32
            or self.input.ndim != 2
            or self.input.shape[1] != 512
            or not self.input.is_contiguous()
        ):
            raise ValueError(
                "FP32 fused RMS/RoPE input must be contiguous [rows,512]"
            )
        if (
            self.output.dtype != torch.bfloat16
            or self.output.shape != self.input.shape
            or not self.output.is_contiguous()
        ):
            raise ValueError("FP32 fused RMS/RoPE output must match the input")
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != (32, 2):
            raise ValueError("FP32 fused RMS/RoPE table must be FP32 [32,2]")
        if self.weight is not None and (
            self.weight.dtype != torch.bfloat16
            or tuple(self.weight.shape) != (512,)
            or not self.weight.is_contiguous()
        ):
            raise ValueError("FP32 fused RMS/RoPE weight must be BF16 [512]")
        if self.epsilon <= 0:
            raise ValueError("FP32 fused RMS/RoPE epsilon must be positive")
        if self.fixed_table_id is not None and not 0 <= self.fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        self.rows = self.input.shape[0]
        if not 0 < self.num_sms <= self.rows:
            raise ValueError("FP32 fused RMS/RoPE requires 1..rows SMs")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions.extend(
                (
                    Dsv4Fp32RmsRope512_64(
                        weighted=self.weight is not None,
                        epsilon=self.epsilon,
                        fixed_table_id=self.fixed_table_id,
                    ),
                    TmaLoad1D(self.input[row]),
                )
            )
            if self.weight is not None:
                instructions.append(TmaLoad1D(self.weight))
            if self.fixed_table_id is None:
                instructions.append(TmaLoad1D(self.table))
            instructions.append(
                TmaStore1D(self.output[row]).bar(self._bar("output"))
            )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.rows)


class SchedDsv4Fp32RopeHadamard128(Schedule):
    """Finalize FP32 index-Q rows into their cache-scoring representation."""

    def __init__(self, input, table, output, *, fixed_table_id=None):
        super().__init__()
        self.input = input
        self.table = table
        self.output = output
        self.fixed_table_id = fixed_table_id

    def _on_place(self):
        if (
            self.input.dtype != torch.float32
            or self.input.ndim != 2
            or self.input.shape[1] != 128
            or not self.input.is_contiguous()
        ):
            raise ValueError(
                "FP32 RoPE/Hadamard input must be contiguous [rows,128]"
            )
        if (
            self.output.dtype != torch.bfloat16
            or self.output.shape != self.input.shape
            or not self.output.is_contiguous()
        ):
            raise ValueError("FP32 RoPE/Hadamard output must match the input")
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != (32, 2):
            raise ValueError("FP32 RoPE/Hadamard table must be FP32 [32,2]")
        if self.fixed_table_id is not None and not 0 <= self.fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        self.rows = self.input.shape[0]
        if not 0 < self.num_sms <= self.rows:
            raise ValueError("FP32 RoPE/Hadamard requires 1..rows SMs")

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions.extend(
                (
                    Dsv4Fp32RopeHadamard128(self.fixed_table_id),
                    TmaLoad1D(self.input[row]),
                )
            )
            if self.fixed_table_id is None:
                instructions.append(TmaLoad1D(self.table))
            instructions.append(
                TmaStore1D(self.output[row]).bar(self._bar("output"))
            )
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
        if self.fixed_table_id is not None and not 0 <= self.fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        self.rows = self.input.shape[0]
        if self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek index RoPE requires 1 <= num_sms <= rows")

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = []
        for row in range(sm, self.rows, self.num_sms):
            instructions.append(
                Dsv4Rope64(
                    1,
                    128,
                    self.inverse,
                    fixed_table_id=self.fixed_table_id,
                )
            )
            instructions.append(TmaLoad1D(self.input[row]))
            if self.fixed_table_id is None:
                instructions.append(TmaLoad1D(self.table))
            instructions.append(
                TmaStore1D(self.output[row]).bar(self._bar("output"))
            )
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


class SchedDsv4ContiguousAttention512Block4(Schedule):
    """Load four adjacent KV rows per TMA for an all-rows attention set."""

    ROWS_PER_BATCH = 4
    ROW_BYTES = 512 * 2

    def __init__(self, q, kv, rows, sink, output):
        super().__init__()
        self.q = q
        self.kv = kv
        self.rows = int(rows)
        self.sink = sink
        self.output = output

    def _on_place(self):
        if (self.q.dtype != torch.bfloat16 or self.q.ndim != 2 or
                self.q.shape[1] != 512):
            raise ValueError("DeepSeek contiguous Q must be BF16 [heads,512]")
        self.heads = self.q.shape[0]
        if self.num_sms != self.heads:
            raise ValueError("DeepSeek contiguous attention uses one SM per head")
        if (self.kv.dtype != torch.bfloat16 or self.kv.ndim != 2 or
                self.kv.shape[1] != 512 or not self.kv.is_contiguous()):
            raise ValueError("DeepSeek contiguous KV must be contiguous BF16 [rows,512]")
        if self.rows <= 0 or self.rows > self.kv.shape[0]:
            raise ValueError("DeepSeek contiguous attention rows exceed the KV cache")
        if self.sink.dtype != torch.float32 or self.sink.numel() != self.heads:
            raise ValueError("DeepSeek attention sink must be FP32 [heads]")
        if self.output.dtype != torch.bfloat16 or self.output.shape != self.q.shape:
            raise ValueError("DeepSeek contiguous output must match Q")

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = [
            Dsv4ContiguousAttention512Block4(self.rows),
            TmaLoad1D(self.q[sm]),
            _shared_load_1d(self.sink[sm:sm + 1]),
        ]
        full_batches, tail_rows = divmod(self.rows, self.ROWS_PER_BATCH)
        if full_batches:
            batch_bytes = self.ROWS_PER_BATCH * self.ROW_BYTES
            instructions += RepeatM.on(
                full_batches,
                (TmaLoad1D(self.kv[:self.ROWS_PER_BATCH].reshape(-1)), batch_bytes),
            )
        if tail_rows:
            tail_start = full_batches * self.ROWS_PER_BATCH
            instructions.append(TmaLoad1D(self.kv[tail_start:self.rows].reshape(-1)))
        instructions.append(
            TmaStore1D(self.output[sm]).bar(self._bar("output"))
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4ContiguousAttention512UmmaSm100(Schedule):
    """All-head K128 UMMA attention, optionally sharded by D128 output."""

    TILE = 128
    TILES_PER_VECTOR = 4

    def __init__(self, q, kv, rows, sink, output, *, q_tma, k_tma, v_tma,
                 output_tma):
        super().__init__()
        self.q = q
        self.kv = kv
        self.rows = int(rows)
        self.sink = sink
        self.output = output
        self.q_tma = q_tma
        self.k_tma = k_tma
        self.v_tma = v_tma
        self.output_tma = output_tma

    def _on_place(self):
        if self.num_sms not in (1, self.TILES_PER_VECTOR):
            raise ValueError("DeepSeek UMMA attention uses one or four SMs")
        if (self.q.dtype != torch.bfloat16 or self.q.shape != (64, 512) or
                not self.q.is_contiguous()):
            raise ValueError("DeepSeek UMMA Q must be contiguous BF16 [64,512]")
        if (self.kv.dtype != torch.bfloat16 or self.kv.ndim != 2 or
                self.kv.shape[1] != 512 or not self.kv.is_contiguous()):
            raise ValueError("DeepSeek UMMA KV must be contiguous BF16 [rows,512]")
        if self.rows <= 0 or self.rows > min(128, self.kv.shape[0]):
            raise ValueError("DeepSeek UMMA attention rows must be in [1,128]")
        if self.sink.dtype != torch.float32 or self.sink.shape != (64,):
            raise ValueError("DeepSeek UMMA attention sink must be FP32 [64]")
        if (self.output.dtype != torch.bfloat16 or
                self.output.shape != self.q.shape or
                not self.output.is_contiguous()):
            raise ValueError("DeepSeek UMMA output must match Q")

    def schedule(self, sm):
        if sm < 0:
            return []
        output_tile = sm if self.num_sms == self.TILES_PER_VECTOR else None
        output_tiles = (
            (output_tile,)
            if output_tile is not None
            else range(self.TILES_PER_VECTOR)
        )
        instructions = [
            Dsv4ContiguousAttention512UmmaSm100(self.rows, output_tile)
        ]
        num_blocks = (self.rows + self.TILE - 1) // self.TILE
        for block in range(num_blocks):
            row = block * self.TILE
            for wave in range(2):
                for tile in range(2):
                    column = (wave * 2 + tile) * self.TILE
                    instructions.append(self.q_tma.cord(0, column))
                for tile in range(2):
                    column = (wave * 2 + tile) * self.TILE
                    instructions.append(self.k_tma.cord(row, column))
            for tile in output_tiles:
                instructions.append(
                    self.v_tma.cord(row, tile * self.TILE)
                )
            if block == 0:
                instructions.append(_shared_load_1d(self.sink))

        for output_index, tile in enumerate(output_tiles):
            store = self.output_tma.cord(0, tile * self.TILE)
            if output_index + 1 == len(output_tiles):
                store = store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4ContiguousAttention512UmmaTail32Sm100(Schedule):
    """Native K128+K32 attention, optionally sharded by D128 output."""

    TILE = 128
    TILES_PER_VECTOR = 4

    def __init__(self, q, kv, rows, sink, output, *, q_tma,
                 prefix_k_tma, tail_k_tma, prefix_v_tma, tail_v_tma,
                 output_tma):
        super().__init__()
        self.q = q
        self.kv = kv
        self.rows = int(rows)
        self.sink = sink
        self.output = output
        self.q_tma = q_tma
        self.prefix_k_tma = prefix_k_tma
        self.tail_k_tma = tail_k_tma
        self.prefix_v_tma = prefix_v_tma
        self.tail_v_tma = tail_v_tma
        self.output_tma = output_tma

    def _on_place(self):
        if self.num_sms not in (1, self.TILES_PER_VECTOR):
            raise ValueError("DeepSeek UMMA tail attention uses one or four SMs")
        if (self.q.dtype != torch.bfloat16 or self.q.shape != (64, 512) or
                not self.q.is_contiguous()):
            raise ValueError("DeepSeek UMMA Q must be contiguous BF16 [64,512]")
        if (self.kv.dtype != torch.bfloat16 or self.kv.ndim != 2 or
                self.kv.shape[1] != 512 or not self.kv.is_contiguous()):
            raise ValueError("DeepSeek UMMA KV must be contiguous BF16 [rows,512]")
        if self.rows <= 128 or self.rows > min(160, self.kv.shape[0]):
            raise ValueError("DeepSeek UMMA tail rows must be in [129,160]")
        if self.sink.dtype != torch.float32 or self.sink.shape != (64,):
            raise ValueError("DeepSeek UMMA attention sink must be FP32 [64]")
        if (self.output.dtype != torch.bfloat16 or
                self.output.shape != self.q.shape or
                not self.output.is_contiguous()):
            raise ValueError("DeepSeek UMMA output must match Q")

    def schedule(self, sm):
        if sm < 0:
            return []
        output_tile = sm if self.num_sms == self.TILES_PER_VECTOR else None
        output_tiles = (
            (output_tile,)
            if output_tile is not None
            else range(self.TILES_PER_VECTOR)
        )
        instructions = [
            Dsv4ContiguousAttention512UmmaTail32Sm100(
                self.rows, output_tile
            )
        ]
        for wave in range(2):
            for tile in range(2):
                column = (wave * 2 + tile) * self.TILE
                instructions.append(self.q_tma.cord(0, column))
            for tile in range(2):
                column = (wave * 2 + tile) * self.TILE
                instructions.append(self.prefix_k_tma.cord(0, column))
        for tile in range(self.TILES_PER_VECTOR):
            instructions.append(
                self.tail_k_tma.cord(self.TILE, tile * self.TILE)
            )
        instructions.append(_shared_load_1d(self.sink))
        for tile in output_tiles:
            column = tile * self.TILE
            instructions.append(self.prefix_v_tma.cord(0, column))
            instructions.append(self.tail_v_tma.cord(self.TILE, column))
        for output_index, tile in enumerate(output_tiles):
            store = self.output_tma.cord(0, tile * self.TILE)
            if output_index + 1 == len(output_tiles):
                store = store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4AttentionSplit32UmmaSm100(Schedule):
    """K32 split-KV UMMA producers for all 64 DeepSeek attention heads."""

    TILE = 128
    KV_TILE = 32
    HEADS = 64

    def __init__(
        self,
        q,
        kv,
        rows,
        partials,
        metadata,
        *,
        q_tma,
        k_tma,
        v_tma,
        partial_tma,
    ):
        super().__init__()
        self.q = q
        self.kv = kv
        self.rows = int(rows)
        self.partials = partials
        self.metadata = metadata
        self.q_tma = q_tma
        self.k_tma = k_tma
        self.v_tma = v_tma
        self.partial_tma = partial_tma

    def _on_place(self):
        if (
            self.q.dtype != torch.bfloat16
            or tuple(self.q.shape) != (self.HEADS, 512)
            or not self.q.is_contiguous()
        ):
            raise ValueError("split-KV attention Q must be BF16 [64,512]")
        if (
            self.kv.dtype != torch.bfloat16
            or self.kv.ndim != 2
            or self.kv.shape[1] != 512
            or not self.kv.is_contiguous()
        ):
            raise ValueError("split-KV attention KV must be BF16 [rows,512]")
        if self.rows <= 0 or self.rows > self.kv.shape[0]:
            raise ValueError("split-KV attention row count exceeds its cache")
        self.num_splits = (self.rows + self.KV_TILE - 1) // self.KV_TILE
        if self.num_sms != self.num_splits:
            raise ValueError("split-KV producer requires one SM per K32 split")
        if (
            self.partials.dtype != torch.bfloat16
            or tuple(self.partials.shape)
            != (self.num_splits, self.HEADS, 512)
            or not self.partials.is_contiguous()
        ):
            raise ValueError("attention partials must be BF16 [splits,64,512]")
        if (
            self.metadata.dtype != torch.float32
            or tuple(self.metadata.shape)
            != (self.num_splits, self.HEADS, 2)
            or not self.metadata.is_contiguous()
        ):
            raise ValueError("attention metadata must be FP32 [splits,64,2]")

    def schedule(self, sm):
        if sm < 0:
            return []
        row = sm * self.KV_TILE
        active_tokens = min(self.KV_TILE, self.rows - row)
        instructions = [Dsv4AttentionSplit32UmmaSm100(active_tokens)]
        for wave in range(2):
            for tile in range(2):
                column = (wave * 2 + tile) * self.TILE
                instructions.append(self.q_tma.cord(0, column))
            for tile in range(2):
                column = (wave * 2 + tile) * self.TILE
                instructions.append(self.k_tma.cord(row, column))
        for tile in range(4):
            column = tile * self.TILE
            instructions.append(self.v_tma.cord(row, column))
            instructions.append(
                self.partial_tma.cord(sm * self.HEADS, column)
            )
        instructions.append(
            TmaStore1D(self.metadata[sm].reshape(-1)).bar(
                self._bar("output")
            )
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_splits)


class SchedDsv4AttentionSplit64UmmaSm100(Schedule):
    """B64 BF16 producers with one allocator-leased internal KV ring."""

    KV_TILE = 64
    HEADS = 64
    DIM = 512
    KV_LAYOUT_BYTES = KV_TILE * DIM * 2
    STAGE_BYTES = 2 * KV_LAYOUT_BYTES
    AREA_SLOTS = STAGE_BYTES // config.slot_size

    def __init__(
        self,
        q,
        kv,
        rows,
        partials,
        metadata,
        *,
        q_tma,
        kv_tma,
        kv_v_tma,
        partial_tma,
        gate_kv_last_split_only=False,
    ):
        super().__init__()
        self.q = q
        self.kv = kv
        self.rows = int(rows)
        self.partials = partials
        self.metadata = metadata
        self.q_tma = q_tma
        self.kv_tma = kv_tma
        self.kv_v_tma = kv_v_tma
        self.partial_tma = partial_tma
        self.gate_kv_last_split_only = bool(gate_kv_last_split_only)
        self.ring_plans = []
        self.split_rows = []

    def _on_place(self):
        if (
            self.q.dtype != torch.bfloat16
            or tuple(self.q.shape) != (self.HEADS, self.DIM)
            or not self.q.is_contiguous()
        ):
            raise ValueError("B64 attention Q must be BF16 [64,512]")
        if (
            self.kv.dtype != torch.bfloat16
            or self.kv.ndim != 2
            or self.kv.shape[1] != self.DIM
            or not self.kv.is_contiguous()
        ):
            raise ValueError("B64 attention KV must be BF16 [rows,512]")
        if self.rows <= 0 or self.rows > self.kv.shape[0]:
            raise ValueError("B64 attention row count exceeds its cache")
        self.num_splits = (self.rows + self.KV_TILE - 1) // self.KV_TILE
        if self.num_sms != self.num_splits:
            raise ValueError("B64 attention requires one SM per KV split")
        if (
            self.partials.dtype != torch.bfloat16
            or tuple(self.partials.shape)
            != (self.num_splits, self.HEADS, self.DIM)
            or not self.partials.is_contiguous()
        ):
            raise ValueError("B64 partials must be BF16 [splits,64,512]")
        if (
            self.metadata.dtype != torch.float32
            or tuple(self.metadata.shape)
            != (self.num_splits, self.HEADS, 2)
            or not self.metadata.is_contiguous()
        ):
            raise ValueError("B64 metadata must be FP32 [splits,64,2]")
        if self.q_tma.rank != 3 or self.kv_tma.rank != 3:
            raise ValueError("B64 Q/K require rank-3 K-major descriptors")
        if self.kv_v_tma.rank != 4:
            raise ValueError("B64 V requires a rank-4 MN-major descriptor")
        if self.q_tma.size != 0 or self.q_tma.num_slots != 8:
            raise ValueError("B64 Q descriptor must use encoded 64-KiB form")
        if (
            self.partial_tma.size != self.KV_LAYOUT_BYTES // 4
            or self.partial_tma.num_slots != 2
        ):
            raise ValueError("B64 partial descriptor must store one D128 tile")

        # Keep every non-final TMA origin aligned to the MN-major descriptor's
        # eight-row block while distributing work evenly across split CTAs.
        # This avoids a 64+1 tail and reduces producer stragglers without
        # changing the number of partials consumed by the reducer.
        full_row_groups, residual_rows = divmod(self.rows, 8)
        base_groups, extra_groups = divmod(full_row_groups, self.num_splits)
        split_lengths = [
            8 * (base_groups + (split < extra_groups))
            for split in range(self.num_splits)
        ]
        split_lengths[-1] += residual_rows
        row_start = 0
        self.split_rows = []
        for split_length in split_lengths:
            if not 1 <= split_length <= self.KV_TILE:
                raise ValueError("balanced B64 split is outside [1,64]")
            self.split_rows.append((row_start, split_length))
            row_start += split_length
        if row_start != self.rows:
            raise ValueError("balanced B64 split coverage is incomplete")

        self.ring_plans = []
        for row, _ in self.split_rows:
            v_coordinates = self.kv_v_tma.cord2tma(row, 0)
            next_v_coordinates = self.kv_v_tma.cord2tma(row, 128)
            self.ring_plans.append(
                build_internal_ring_tma_plan(
                    device=self.kv.device,
                    stage_bytes=self.STAGE_BYTES,
                    lanes={
                    0: {
                        "descriptor_index": self.kv_tma.arg,
                        "rank": self.kv_tma.rank,
                        "transaction_bytes": self.KV_LAYOUT_BYTES,
                        "coordinates": self.kv_tma.cord2tma(row, 0),
                    },
                        1: {
                            "descriptor_index": self.kv_v_tma.arg,
                            "rank": self.kv_v_tma.rank,
                            "issue_count": 4,
                            "transaction_bytes": self.KV_LAYOUT_BYTES // 4,
                            "destination_offset": self.KV_LAYOUT_BYTES,
                            "destination_issue_stride":
                                self.KV_LAYOUT_BYTES // 4,
                            "coordinates": v_coordinates,
                            "issue_delta": tuple(
                                next_value - value
                                for value, next_value in zip(
                                    v_coordinates, next_v_coordinates
                                )
                            ),
                        }
                    },
                )
            )

    def schedule(self, sm):
        if sm < 0:
            return []
        row, active_tokens = self.split_rows[sm]
        ring_load = TmaLoadInternalRingStream(
            self.ring_plans[sm],
            stages=1,
            stage_bytes=self.STAGE_BYTES,
            area_slots=self.AREA_SLOTS,
            area_id=0,
            stream_length=1,
            port_mask=3,
        )
        kv_bar = self._bar("kv")
        if kv_bar is not None and (
            not self.gate_kv_last_split_only
            or row + active_tokens == self.rows
        ):
            ring_load.bar(kv_bar)
        return [
            Dsv4AttentionSplit64UmmaSm100(
                active_tokens, ring_port_mask=3
            ),
            ring_load,
            self.q_tma.cord(0, 0).bar(self._bar("q")),
            RawAddress(
                self.metadata[sm], config.num_slots + 8
            ),
            self.partial_tma.cord(sm * self.HEADS, 256),
            self.partial_tma.cord(sm * self.HEADS, 384).bar(
                self._bar("output1")
                if self._bar("output1") is not None
                else self._bar("output")
            ),
            self.partial_tma.cord(sm * self.HEADS, 0),
            self.partial_tma.cord(sm * self.HEADS, 128).bar(
                self._bar("output0")
            ),
        ]

    def bar_release_count(self, role: str):
        if role not in ("output", "output0", "output1"):
            return 0
        return self._bar_release_if_present(role, self.num_splits)


class SchedLayeredDsv4AttentionSplit64UmmaSm100(Schedule):
    """Use one retained-ring attention body with a plan bank per layer.

    Each plan contains that layer's K- and V-TMA descriptors.  The allocator
    derives the plan-row address from the existing family loop counters before
    sending the unchanged retained-ring command to LDU.  Q, partials, metadata,
    allocator ownership, and the compute task remain identical to the ordinary
    split-64 schedule.
    """

    def __init__(self, inner, kv_layers, *, counter_strides):
        super().__init__()
        self.inner = inner
        self.kv_layers = tuple(kv_layers)
        self.counter_strides = tuple(
            (int(counter), int(stride))
            for counter, stride in counter_strides
        )

    def _on_place(self):
        if len(self.kv_layers) <= 1:
            raise ValueError("layered split attention requires multiple KV layers")
        if not self.counter_strides:
            raise ValueError("layered split attention requires counter strides")
        if any(
            counter < 0 or counter >= 32 or stride <= 0
            for counter, stride in self.counter_strides
        ):
            raise ValueError(
                "layered split-attention counters require reg [0,31] and "
                "positive strides"
            )
        representative = self.kv_layers[0]
        for kv in self.kv_layers:
            if (
                kv.device != representative.device
                or kv.dtype != torch.bfloat16
                or kv.shape != representative.shape
                or kv.ndim != 2
                or kv.shape[1] != SchedDsv4AttentionSplit64UmmaSm100.DIM
                or not kv.is_contiguous()
            ):
                raise ValueError(
                    "layered split-attention KV tensors must be matching "
                    "contiguous BF16 matrices"
                )

        inner = self.inner._clone()
        inner._bars.update(self._bars)
        self.placed_inner = inner.place(self.num_sms)
        if self.placed_inner.kv.data_ptr() != representative.data_ptr():
            raise ValueError(
                "layer zero KV must match the split-attention representative"
            )

        kv_tmas = [self.placed_inner.kv_tma]
        kv_v_tmas = [self.placed_inner.kv_v_tma]
        launcher = self.placed_inner.kv_tma.launcher
        for kv in self.kv_layers[1:]:
            kv_tmas.append(
                TmaTensor(launcher, kv).wgmma_load(64, 512, Major.K)
            )
            kv_v_tmas.append(
                TmaTensor(launcher, kv).wgmma_load(64, 128, Major.MN)
            )
        self.kv_tmas = tuple(kv_tmas)
        self.kv_v_tmas = tuple(kv_v_tmas)

        layer_plans = []
        for kv_tma, kv_v_tma in zip(self.kv_tmas, self.kv_v_tmas):
            split_plans = []
            for row, _ in self.placed_inner.split_rows:
                v_coordinates = kv_v_tma.cord2tma(row, 0)
                next_v_coordinates = kv_v_tma.cord2tma(row, 128)
                split_plans.append(
                    build_internal_ring_tma_plan(
                        device=representative.device,
                        stage_bytes=self.placed_inner.STAGE_BYTES,
                        lanes={
                            0: {
                                "descriptor_index": kv_tma.arg,
                                "rank": kv_tma.rank,
                                "transaction_bytes":
                                    self.placed_inner.KV_LAYOUT_BYTES,
                                "coordinates": kv_tma.cord2tma(row, 0),
                            },
                            1: {
                                "descriptor_index": kv_v_tma.arg,
                                "rank": kv_v_tma.rank,
                                "issue_count": 4,
                                "transaction_bytes":
                                    self.placed_inner.KV_LAYOUT_BYTES // 4,
                                "destination_offset":
                                    self.placed_inner.KV_LAYOUT_BYTES,
                                "destination_issue_stride":
                                    self.placed_inner.KV_LAYOUT_BYTES // 4,
                                "coordinates": v_coordinates,
                                "issue_delta": tuple(
                                    next_value - value
                                    for value, next_value in zip(
                                        v_coordinates, next_v_coordinates
                                    )
                                ),
                            },
                        },
                    )
                )
            layer_plans.append(torch.stack(split_plans))
        self.layered_plans = torch.stack(layer_plans).contiguous()
        self.plan_layer_bytes = (
            self.layered_plans.stride(0)
            * self.layered_plans.element_size()
        )

    def schedule(self, sm):
        instructions = list(self.placed_inner.schedule(sm))
        if sm < 0:
            return instructions
        if len(instructions) != 8 or not isinstance(
            instructions[1], TmaLoadInternalRingStream
        ):
            raise ValueError("split-64 retained-ring command contract changed")
        ring = instructions[1].copy()
        ring.set_cords(addr2cords(self.layered_plans[0, sm].data_ptr()))
        instructions[1] = RepeatM.offsetByCounters(
            tuple(
                (counter, stride * self.plan_layer_bytes)
                for counter, stride in self.counter_strides
            ),
            ring,
        )
        return instructions

    def bar_release_count(self, role: str):
        return self.placed_inner.bar_release_count(role)


class SchedDsv4AttentionContext1Fp8Sm100(Schedule):
    """Compute one-row sink attention and publish native O_a records."""

    HEADS = 64
    OUTPUT_GROUPS = 2
    TILES = 4
    TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(
        self,
        q,
        kv,
        sink,
        table,
        output,
        *,
        head_start=0,
        head_count=64,
        normalize_q=False,
    ):
        super().__init__()
        self.q = q
        self.kv = kv
        self.sink = sink
        self.table = table
        self.output = output
        self.head_start = int(head_start)
        self.head_count = int(head_count)
        self.normalize_q = bool(normalize_q)

    def _on_place(self):
        if (
            self.q.dtype != torch.bfloat16
            or tuple(self.q.shape) != (self.HEADS, 512)
            or not self.q.is_contiguous()
        ):
            raise ValueError("context-1 Q must be contiguous BF16 [64,512]")
        if (
            self.kv.dtype != torch.bfloat16
            or tuple(self.kv.shape) != (1, 512)
            or not self.kv.is_contiguous()
        ):
            raise ValueError("context-1 KV must be contiguous BF16 [1,512]")
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != (32, 2):
            raise ValueError("inverse-RoPE table must be FP32 [32,2]")
        if (
            self.sink.dtype != torch.float32
            or tuple(self.sink.shape) != (64,)
        ):
            raise ValueError("context-1 sink must be FP32 [64]")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape)
            != (self.HEADS, self.TILES, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError("native O_a output must be uint8 [64,4,2048]")
        if (
            self.head_start < 0
            or self.head_count <= 0
            or self.head_start + self.head_count > self.HEADS
        ):
            raise ValueError("context-1 attention head shard exceeds [0,64)")
        work_count = self.head_count
        if not 0 < self.num_sms <= work_count:
            raise ValueError("context-1 attention SMs exceed its head count")
        self.raw_record = torch.tensor(
            (
                self.q.data_ptr(),
                self.kv.data_ptr(),
                self.table.data_ptr(),
            ),
            dtype=torch.uint64,
            device=self.q.device,
        )

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = []
        work_count = self.head_count
        for work in range(sm, work_count, self.num_sms):
            head = self.head_start + work
            sink_start = head & ~3
            instructions.extend(
                (
                    Dsv4AttentionContext1Fp8Sm100(
                        head, normalize_q=self.normalize_q
                    ),
                    RawAddress(
                        self.raw_record, config.num_slots
                    ).bar(self._bar("input")).fixed_port(0),
                    LduLoad1D(
                        self.sink[sink_start : sink_start + 4], bytes=16
                    ).fixed_port(1),
                    RawAddress(
                        self.output, config.num_slots + 1
                    ).writeback().bar(self._bar("output")),
                )
            )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.head_count)


class SchedLayeredDsv4AttentionContext1Fp8Sm100(Schedule):
    """Select a layer-private context-one KV row in the allocator stream."""

    def __init__(self, inner, kv_layers, *, counter_strides):
        super().__init__()
        self.inner = inner
        self.kv_layers = tuple(kv_layers)
        self.counter_strides = tuple(
            (int(counter), int(stride))
            for counter, stride in counter_strides
        )

    def _on_place(self):
        if len(self.kv_layers) <= 1 or not self.counter_strides:
            raise ValueError("layered context-one attention requires a loop")
        representative = self.kv_layers[0]
        if any(
            kv.device != representative.device
            or kv.dtype != torch.bfloat16
            or tuple(kv.shape) != (1, 512)
            or not kv.is_contiguous()
            for kv in self.kv_layers
        ):
            raise ValueError(
                "layered context-one KV rows must be matching contiguous BF16"
            )
        inner = self.inner._clone()
        inner._bars.update(self._bars)
        self.placed_inner = inner.place(self.num_sms)
        if self.placed_inner.kv.data_ptr() != representative.data_ptr():
            raise ValueError("context-one representative KV row changed")
        self.raw_records = torch.tensor(
            [
                (
                    self.placed_inner.q.data_ptr(),
                    kv.data_ptr(),
                    self.placed_inner.table.data_ptr(),
                )
                for kv in self.kv_layers
            ],
            dtype=torch.uint64,
            device=representative.device,
        )
        self.record_bytes = (
            self.raw_records.stride(0) * self.raw_records.element_size()
        )

    def schedule(self, sm):
        instructions = list(self.placed_inner.schedule(sm))
        if sm < 0:
            return instructions
        for index, inst in enumerate(instructions):
            if (
                isinstance(inst, RawAddress)
                and (inst.num_slots & ((1 << 6) - 1)) == config.num_slots
                and not (inst.opcode & 0x2)
            ):
                selected = inst.copy()
                selected.set_cords(addr2cords(self.raw_records[0].data_ptr()))
                instructions[index] = RepeatM.offsetByCounters(
                    tuple(
                        (counter, stride * self.record_bytes)
                        for counter, stride in self.counter_strides
                    ),
                    selected,
                )
        return instructions

    def bar_release_count(self, role: str):
        return self.placed_inner.bar_release_count(role)


class SchedDsv4AttentionSplitReduceFp8Sm100(Schedule):
    """Merge split-KV partials and directly publish native O_a records."""

    HEADS = 64
    OUTPUT_GROUPS = 2
    TILES = 4
    TILE_BYTES = SchedFp8UmmaPrepack.ACTIVATION_TILE_BYTES

    def __init__(
        self,
        partials,
        metadata,
        sink,
        table,
        output,
        *,
        head_start=0,
        head_count=64,
    ):
        super().__init__()
        self.partials = partials
        self.metadata = metadata
        self.sink = sink
        self.table = table
        self.output = output
        self.head_start = int(head_start)
        self.head_count = int(head_count)

    def _on_place(self):
        if (
            self.partials.dtype != torch.bfloat16
            or self.partials.ndim != 3
            or tuple(self.partials.shape[1:]) != (self.HEADS, 512)
            or not self.partials.is_contiguous()
        ):
            raise ValueError("attention partials must be BF16 [splits,64,512]")
        self.num_splits = self.partials.shape[0]
        if not 1 <= self.num_splits <= 24:
            raise ValueError("attention reducer supports 1..24 splits")
        if (
            self.metadata.dtype != torch.float32
            or tuple(self.metadata.shape)
            != (self.num_splits, self.HEADS, 2)
            or not self.metadata.is_contiguous()
        ):
            raise ValueError("attention metadata must be FP32 [splits,64,2]")
        if self.sink.dtype != torch.float32 or tuple(self.sink.shape) != (64,):
            raise ValueError("attention sink must be FP32 [64]")
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != (32, 2):
            raise ValueError("inverse-RoPE table must be FP32 [32,2]")
        if (
            self.output.dtype != torch.uint8
            or tuple(self.output.shape)
            != (self.HEADS, self.TILES, self.TILE_BYTES)
            or not self.output.is_contiguous()
        ):
            raise ValueError("native O_a output must be uint8 [64,4,2048]")
        if (
            self.head_start < 0
            or self.head_count <= 0
            or self.head_start + self.head_count > self.HEADS
        ):
            raise ValueError("attention reducer head shard exceeds [0,64)")
        if not 0 < self.num_sms <= self.head_count * self.OUTPUT_GROUPS:
            raise ValueError("attention reducer SMs exceed its output groups")
        # One immutable device record replaces the old metadata/sink/table
        # loads and every per-split partial TMA.  The barrier is attached to
        # this record's LDU command, so compute cannot dereference any pointer
        # until producer writeback is globally complete.
        self.raw_record = torch.tensor(
            (
                self.partials.data_ptr(),
                self.metadata.data_ptr(),
                self.sink.data_ptr(),
                self.table.data_ptr(),
            ),
            dtype=torch.uint64,
            device=self.partials.device,
        )

    def schedule(self, sm):
        if sm < 0:
            return []
        instructions = []
        work_count = self.head_count * self.OUTPUT_GROUPS
        for work in range(sm, work_count, self.num_sms):
            head = self.head_start + work // self.OUTPUT_GROUPS
            output_group = work % self.OUTPUT_GROUPS
            instructions.extend(
                (
                    Dsv4AttentionSplitReduceFp8Sm100(
                        self.num_splits, head, output_group
                    ),
                    RawAddress(
                        self.raw_record, config.num_slots
                    ).bar(
                        self._bar(f"partials{output_group}")
                        if self._bar(f"partials{output_group}") is not None
                        else self._bar("partials")
                    ).fixed_port(0),
                    RawAddress(
                        self.output, config.num_slots + 1
                    ).writeback().bar(self._bar("output")),
                )
            )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(
            role, self.head_count * self.OUTPUT_GROUPS
        )


class SchedDsv4RouteTop6(Schedule):
    def __init__(self, logits, bias, hash_indices, output_indices,
                 output_weights, hash_routing=False, route_scale=1.5,
                 pretransformed=False, packed_output=None,
                 hash_counter_strides=()):
        super().__init__()
        self.logits = logits
        self.bias = bias
        self.hash_indices = hash_indices
        self.output_indices = output_indices
        self.output_weights = output_weights
        self.hash_routing = hash_routing
        self.route_scale = route_scale
        self.pretransformed = bool(pretransformed)
        self.packed_output = packed_output
        self.hash_counter_strides = tuple(
            (int(counter), int(stride))
            for counter, stride in hash_counter_strides
        )

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek routing currently uses exactly one SM")
        expected_shape = (256, 2) if self.pretransformed else (256,)
        if (
            self.logits.dtype != torch.float32
            or tuple(self.logits.shape) != expected_shape
            or not self.logits.is_contiguous()
        ):
            raise ValueError(
                "DeepSeek routing input must be contiguous FP32 "
                f"{list(expected_shape)}"
            )
        if self.pretransformed:
            if self.bias is not None:
                raise ValueError("pretransformed routing does not reload bias")
            if (
                self.packed_output is None
                or self.packed_output.dtype != torch.uint8
                or self.packed_output.numel() != 128
                or not self.packed_output.is_contiguous()
            ):
                raise ValueError(
                    "prepared routing requires one packed 128-byte route record"
                )
            if self.output_indices.data_ptr() != self.packed_output.data_ptr():
                raise ValueError("route indices must begin the packed output")
            if self.output_weights.data_ptr() != self.packed_output.data_ptr() + 32:
                raise ValueError("route weights must follow packed route indices")
        elif self.bias.dtype != torch.float32 or self.bias.numel() != 256:
            raise ValueError("DeepSeek routing bias must contain 256 FP32 values")
        elif self.packed_output is not None:
            raise ValueError("legacy routing does not use prepared stream metadata")
        if self.hash_counter_strides:
            if not self.pretransformed or not self.hash_routing:
                raise ValueError(
                    "layered hash selection requires prepared hash routing"
                )
            if (
                self.hash_indices.dtype != torch.int32
                or self.hash_indices.ndim != 2
                or self.hash_indices.shape[1] != 8
                or not self.hash_indices.is_contiguous()
            ):
                raise ValueError(
                    "layered hash routing storage must be contiguous int32[L,8]"
                )
            if any(
                counter < 0 or counter >= 32 or stride <= 0
                for counter, stride in self.hash_counter_strides
            ):
                raise ValueError(
                    "hash counter strides require reg [0,31] and positive stride"
                )
        elif self.hash_indices.dtype != torch.int32 or self.hash_indices.numel() != 8:
            raise ValueError("DeepSeek hash routing storage must contain eight int32 values")
        if self.output_indices.dtype != torch.int32 or self.output_indices.numel() != 8:
            raise ValueError("DeepSeek route-id storage must contain eight int32 values")
        if self.output_weights.dtype != torch.float32 or self.output_weights.numel() != 8:
            raise ValueError("DeepSeek route-weight storage must contain eight FP32 values")
    def schedule(self, sm):
        if sm != 0:
            return []
        if self.pretransformed:
            instructions = [
                Dsv4RouteTop6Prepared(
                    self.hash_routing,
                    self.route_scale,
                ),
            ]
            instructions.append(
                TmaLoad1D(self.logits).fixed_port(0).bar(self._bar("logits"))
            )
            if self.hash_routing:
                hash_indices = (
                    self.hash_indices[0]
                    if self.hash_indices.ndim == 2
                    else self.hash_indices
                )
                hash_address = RawAddress(
                    hash_indices, config.num_slots + 1
                ).fixed_port(1)
                if self.hash_counter_strides:
                    row_bytes = (
                        self.hash_indices.stride(0)
                        * self.hash_indices.element_size()
                    )
                    hash_address = RepeatM.offsetByCounters(
                        (
                            (counter, stride * row_bytes)
                            for counter, stride in self.hash_counter_strides
                        ),
                        hash_address,
                    )
                instructions.append(hash_address)
            instructions.append(
                RawAddress(
                    # C2M/STU publication is a 32-bit slot mask. The resident
                    # FFN image has 24 allocator slots, so special mailbox 8
                    # would be unrepresentable (absolute slot 32).
                    self.packed_output, config.num_slots + 3
                ).writeback().bar(self._bar("output"))
            )
            return instructions
        logits_load = TmaLoad1D(self.logits).fixed_port(0)
        logits_load.bar(self._bar("logits"))
        instructions = [
            Dsv4RouteTop6(self.hash_routing, self.route_scale),
            logits_load,
            TmaLoad1D(self.bias).fixed_port(1),
        ]
        if self.hash_routing:
            instructions.append(TmaLoad1D(self.hash_indices).fixed_port(1))
        instructions.extend((
            TmaStore1D(self.output_indices),
            TmaStore1D(self.output_weights).bar(self._bar("output")),
        ))
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(
            role, 1
        )


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


class SchedDsv4ExpertTmaReduceFp32(Schedule):
    """TMA reduce route-scaled FP32 expert tiles without cross-SM collisions."""

    TILE = 128

    def __init__(self, experts, output_reduce):
        super().__init__()
        self.experts = experts
        self.output_reduce = output_reduce

    def _on_place(self):
        if (
            self.experts.dtype != torch.float32
            or self.experts.ndim != 2
            or self.experts.shape[1] % self.TILE
            or not self.experts.is_contiguous()
        ):
            raise ValueError(
                "FP32 expert TMA reduction requires contiguous [E,M128-aligned]"
            )
        self.expert_count, self.rows = self.experts.shape
        if self.expert_count <= 0:
            raise ValueError("FP32 expert TMA reduction requires experts")
        output = getattr(self.output_reduce, "mat", None)
        if (
            getattr(self.output_reduce, "mode", None) != "reduce"
            or output is None
            or output.dtype != torch.float32
            or tuple(output.shape) != (1, self.rows)
            or not output.is_contiguous()
        ):
            raise ValueError(
                "FP32 expert output must be row-major TMA reduce [1,M]"
            )
        self.tiles = self.rows // self.TILE
        if self.num_sms != self.tiles:
            raise ValueError(
                "collision-free FP32 expert reduction uses one SM per M128 tile"
            )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        start = sm * self.TILE
        stop = start + self.TILE
        instructions = []
        for expert in range(self.expert_count):
            instructions.extend(
                (
                    Copy(1, self.TILE * 4),
                    TmaLoad1D(
                        self.experts[expert, start:stop]
                    ).fixed_port(expert & 1),
                    self.output_reduce.cord(0, start),
                )
            )
        instructions[-1].bar(self._bar("output"))
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Fp32Bf16Gemv(Schedule):
    TILE_K = 8192
    DIRECT_RECORD_SLOT = config.num_slots + 3
    FUSED_SPLITS = 16
    FUSED_HALVES_PER_TASK = 1
    FUSED_TILE_HIDDEN = 256
    FUSED_OUTPUTS_PER_TASK = 3
    FUSED_GROUPS = 8
    FUSED_TASK_SMS = FUSED_GROUPS * FUSED_SPLITS
    FUSED_RECORD_STRIDE = 32
    FUSED_TAIL_ITEMS = 28

    def __init__(
        self,
        weight,
        input,
        output,
        square_sum_output=None,
        metadata_scale=None,
        metadata_base=None,
        metadata_tail_output=None,
        fused_post_input_record=None,
        fused_post_output=None,
        fused_partial_metadata=None,
        packed_coefficients=None,
        launcher=None,
        prefetch_operands_before_resident_reset=False,
        profile_operands=False,
        captured_record=None,
        captured_weight=None,
        captured_coefficients=None,
    ):
        super().__init__()
        self.weight = weight
        self.input = input
        self.output = output
        self.square_sum_output = square_sum_output
        self.metadata_scale = metadata_scale
        self.metadata_base = metadata_base
        self.metadata_tail_output = metadata_tail_output
        self.fused_post_input_record = fused_post_input_record
        self.fused_post_output = fused_post_output
        self.fused_partial_metadata = fused_partial_metadata
        self.packed_coefficients = packed_coefficients
        self.launcher = launcher
        self.prefetch_operands_before_resident_reset = bool(
            prefetch_operands_before_resident_reset
        )
        self.profile_operands = bool(profile_operands)
        self.captured_record = captured_record
        self.captured_weight = captured_weight
        self.captured_coefficients = captured_coefficients

    def _on_place(self):
        self.fuse_hc_post = self.fused_post_input_record is not None
        if self.fuse_hc_post:
            expected_weight_shape = (
                self.FUSED_GROUPS,
                self.FUSED_SPLITS,
                self.FUSED_HALVES_PER_TASK,
                self.FUSED_OUTPUTS_PER_TASK,
                4,
                self.FUSED_TILE_HIDDEN,
            )
            if (
                self.weight.dtype != torch.float32
                or tuple(self.weight.shape) != expected_weight_shape
                or not self.weight.is_contiguous()
            ):
                raise ValueError(
                    "fused mHC weights must use packed FP32 "
                    f"{list(expected_weight_shape)} layout"
                )
            self.rows, self.k = 24, 4 * 4096
        else:
            if self.weight.dtype != torch.float32 or self.weight.ndim != 2:
                raise ValueError("DeepSeek mHC weight must be rank-2 FP32")
            self.rows, self.k = self.weight.shape
        if self.input.dtype != torch.bfloat16 or self.input.numel() != self.k:
            raise ValueError("DeepSeek mHC input must be BF16 [K]")
        if self.output.dtype != torch.float32 or self.output.numel() != self.rows:
            raise ValueError("DeepSeek mHC GEMV output must be FP32 [rows]")
        if self.fuse_hc_post != (self.fused_post_output is not None):
            raise ValueError(
                "fused mHC post input record and residual output are required together"
            )
        if self.fuse_hc_post:
            if self.rows != 24 or self.k != 4 * 4096:
                raise ValueError(
                    "fused mHC post projection requires FP32 [24,16384] weights"
                )
            if (
                self.fused_post_input_record.dtype != torch.bfloat16
                or tuple(self.fused_post_input_record.shape) != (5, 4096)
                or not self.fused_post_input_record.is_contiguous()
            ):
                raise ValueError(
                    "fused mHC post input record must be contiguous BF16 [5,4096]"
                )
            if (
                self.fused_post_output.dtype != torch.bfloat16
                or tuple(self.fused_post_output.shape) != (4, 4096)
                or not self.fused_post_output.is_contiguous()
            ):
                raise ValueError(
                    "fused mHC post output must be contiguous BF16 [4,4096]"
                )
            if self.input.data_ptr() != self.fused_post_output.data_ptr():
                raise ValueError(
                    "fused projection input must alias the materialized post output"
                )
            if (
                self.fused_partial_metadata is None
                or self.fused_partial_metadata.dtype != torch.float32
                or self.fused_partial_metadata.numel()
                != self.FUSED_SPLITS * self.FUSED_RECORD_STRIDE
                + self.FUSED_TAIL_ITEMS
                or not self.fused_partial_metadata.is_contiguous()
            ):
                raise ValueError(
                    "fused mHC partial metadata must be contiguous FP32 "
                    "split records plus a packed scale/base tail"
                )
            if (
                self.packed_coefficients is None
                or self.packed_coefficients.dtype != torch.float32
                or self.packed_coefficients.numel() != 20
                or not self.packed_coefficients.is_contiguous()
            ):
                raise ValueError(
                    "fused mHC post coefficients must be contiguous FP32 [20]"
                )
            if self.launcher is None:
                raise ValueError("fused mHC projection requires a TMA launcher")
            if self.captured_record is not None and (
                not self.profile_operands
                or self.captured_record.dtype != torch.bfloat16
                or tuple(self.captured_record.shape)
                != (
                    self.FUSED_TASK_SMS,
                    5,
                    self.FUSED_TILE_HIDDEN,
                )
                or not self.captured_record.is_contiguous()
                or self.captured_record.device != self.weight.device
            ):
                raise ValueError(
                    "fused mHC record capture must be a contiguous BF16 "
                    "[128,5,256] tensor in operand-profile mode"
                )
            if self.captured_weight is not None and (
                not self.profile_operands
                or self.captured_weight.dtype != torch.float32
                or tuple(self.captured_weight.shape)
                != (
                    self.FUSED_TASK_SMS,
                    self.FUSED_OUTPUTS_PER_TASK,
                    4,
                    self.FUSED_TILE_HIDDEN,
                )
                or not self.captured_weight.is_contiguous()
                or self.captured_weight.device != self.weight.device
            ):
                raise ValueError(
                    "fused mHC weight capture must be a contiguous FP32 "
                    "[128,3,4,256] tensor in operand-profile mode"
                )
            if self.captured_coefficients is not None and (
                not self.profile_operands
                or self.captured_coefficients.dtype != torch.float32
                or tuple(self.captured_coefficients.shape)
                != (self.FUSED_TASK_SMS, 20)
                or not self.captured_coefficients.is_contiguous()
                or self.captured_coefficients.device != self.weight.device
            ):
                raise ValueError(
                    "fused mHC coefficient capture must be a contiguous "
                    "FP32 [128,20] tensor in operand-profile mode"
                )
            self.profile_operand_descriptor = None
            if self.profile_operands:
                # TRACK_PROFILE interprets the normally direct coefficient
                # pointer as this diagnostic descriptor.  The
                # ordinary image never requests operand profiling and keeps
                # the original direct coefficient address/instruction path.
                self.profile_operand_descriptor = torch.tensor(
                    (
                        self.packed_coefficients.data_ptr(),
                        0
                        if self.captured_record is None
                        else self.captured_record.data_ptr(),
                        0
                        if self.captured_weight is None
                        else self.captured_weight.data_ptr(),
                        0
                        if self.captured_coefficients is None
                        else self.captured_coefficients.data_ptr(),
                    ),
                    dtype=torch.int64,
                    device=self.weight.device,
                )
            self.fused_record_tma = TmaTensor(
                self.launcher, self.fused_post_input_record
            ).rowmajor_2d(
                "load",
                5,
                self.FUSED_TILE_HIDDEN,
            )
            self.fused_output_tma = TmaTensor(
                self.launcher, self.fused_post_output
            ).rowmajor_2d(
                "store",
                4,
                self.FUSED_TILE_HIDDEN,
            )
        metadata_items = (
            self.square_sum_output,
            self.metadata_scale,
            self.metadata_base,
            self.metadata_tail_output,
        )
        if not self.fuse_hc_post and any(item is not None for item in metadata_items):
            if not all(item is not None for item in metadata_items):
                raise ValueError(
                    "DeepSeek mHC GEMV metadata outputs must be provided together"
                )
            if self.rows != 24:
                raise ValueError(
                    "DeepSeek mHC GEMV metadata producer requires 24 output rows"
                )
            if (
                self.square_sum_output.dtype != torch.float32
                or self.square_sum_output.numel() != 1
                or not self.square_sum_output.is_contiguous()
            ):
                raise ValueError(
                    "DeepSeek mHC GEMV square-sum output must be contiguous FP32 [1]"
                )
            if (
                self.metadata_scale.dtype != torch.float32
                or self.metadata_scale.numel() != 3
                or not self.metadata_scale.is_contiguous()
            ):
                raise ValueError(
                    "DeepSeek mHC GEMV metadata scale must be contiguous FP32 [3]"
                )
            if (
                self.metadata_base.dtype != torch.float32
                or self.metadata_base.numel() != 24
                or not self.metadata_base.is_contiguous()
            ):
                raise ValueError(
                    "DeepSeek mHC GEMV metadata base must be contiguous FP32 [24]"
                )
            if (
                self.metadata_tail_output.dtype != torch.float32
                or self.metadata_tail_output.numel() != 28
                or not self.metadata_tail_output.is_contiguous()
            ):
                raise ValueError(
                    "DeepSeek mHC GEMV metadata tail must be contiguous FP32 [28]"
                )
            record_address = self.square_sum_output.data_ptr()
            if self.output.data_ptr() != record_address + 4:
                raise ValueError(
                    "DeepSeek mHC GEMV output must follow the packed square sum"
                )
            if self.metadata_tail_output.data_ptr() != record_address + 28 * 4:
                raise ValueError(
                    "DeepSeek mHC GEMV metadata tail must start at packed offset 28"
                )
        if self.fuse_hc_post:
            expected_sms = self.FUSED_TASK_SMS
            if self.num_sms != expected_sms:
                raise ValueError(
                    f"fused mHC projection requires exactly {expected_sms} SMs"
                )
        elif self.num_sms <= 0 or self.num_sms > self.rows:
            raise ValueError("DeepSeek mHC GEMV requires 1 <= num_sms <= rows")

    def schedule(self, sm):
        if sm < 0:
            return []
        if self.fuse_hc_post:
            output_group, split = divmod(sm, self.FUSED_SPLITS)
            weight_load = _shared_load_1d(
                self.weight[output_group, split]
            ).fixed_port(0)
            if self.prefetch_operands_before_resident_reset:
                weight_load.annotation[
                    "prefetch_before_resident_reset"
                ] = True
            emit_residual = output_group == 0
            coefficients_bar = self._bar("coefficients")
            direct_coefficients = coefficients_bar is None
            task = [
                Dsv4Fp32Bf16Gemv(
                    self.FUSED_HALVES_PER_TASK * self.FUSED_TILE_HIDDEN,
                    output_group,
                    emit_square_sum=emit_residual,
                    fuse_hc_post=True,
                    profile_operands=self.profile_operands,
                    packed_coefficients_address=(
                        (
                            self.profile_operand_descriptor.data_ptr()
                            if self.profile_operand_descriptor is not None
                            else self.packed_coefficients.data_ptr()
                        )
                        if direct_coefficients
                        else None
                    ),
                ),
                weight_load,
            ]
            if not direct_coefficients:
                coefficients = RawAddress(
                    self.packed_coefficients, self.DIRECT_RECORD_SLOT
                ).fixed_port(1)
                if coefficients_bar is not None:
                    coefficients.bar(coefficients_bar)
                if self.prefetch_operands_before_resident_reset:
                    coefficients.annotation[
                        "prefetch_before_resident_reset"
                    ] = True
                task.append(coefficients)
            tile_start = (
                split * self.FUSED_HALVES_PER_TASK * self.FUSED_TILE_HIDDEN
            )
            if self.FUSED_HALVES_PER_TASK == 1:
                record_load = self.fused_record_tma.cord(0, tile_start)
            else:
                record_load = self.fused_record_tma.cord_pair_2d(
                    0,
                    tile_start,
                    delta_cols=self.FUSED_TILE_HIDDEN,
                )
            record_load = record_load.fixed_port(1)
            record_load.annotation["fused_hc_record"] = True
            if self.profile_operands:
                # coords[3] is unused by both the ordinary and paired 2-D TMA
                # encodings. The TRACK_PROFILE runtime recognizes this marker
                # and records the VDcores global-counter generation without
                # changing the production instruction stream.
                record_load.cords[3] = 0x4843
                record_load.annotation["profile_hc_global_bar"] = True
            record_bar = self._bar("record")
            if record_bar is not None:
                record_load.bar(record_bar)
            if self.prefetch_operands_before_resident_reset:
                record_load.annotation[
                    "prefetch_before_resident_reset"
                ] = True
            task.append(record_load)
            if emit_residual:
                if self.FUSED_HALVES_PER_TASK == 1:
                    residual_store = self.fused_output_tma.cord(0, tile_start)
                else:
                    residual_store = self.fused_output_tma.cord_pair_2d(
                        0,
                        tile_start,
                        delta_cols=self.FUSED_TILE_HIDDEN,
                    )
                if self._bar("residual") is not None:
                    residual_store.bar(self._bar("residual"))
                task.append(residual_store)
            metadata_bar = self._bar("metadata")
            if metadata_bar is None:
                metadata_bar = self._bar("output")
            # Give every producer a disjoint 16-byte destination.  Compute
            # writes these four FP32 words directly to HBM, then returns the
            # raw-address token through C2M.  STU performs no data copy; it
            # only publishes the metadata completion edge.  Group zero writes
            # [square_sum, mix0, mix1, mix2]; the remaining groups write three
            # mixes and intentionally leave their fourth word untouched.
            partial_start = (
                split * self.FUSED_RECORD_STRIDE + output_group * 4
            )
            partial_store = RawWritebackAddress(
                self.fused_partial_metadata[
                    partial_start:partial_start + 4
                ]
            ).bar(metadata_bar)
            task.append(partial_store)
            return task

        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        instructions = []
        for local_row, row in enumerate(range(row_start, row_start + row_count)):
            emit_square_sum = self.square_sum_output is not None and row == 0
            instructions.append(Dsv4Fp32Bf16Gemv(
                self.k,
                self.TILE_K,
                emit_square_sum=emit_square_sum,
            ))
            for column in range(0, self.k, self.TILE_K):
                end = min(column + self.TILE_K, self.k)
                instructions += [
                    _shared_load_1d(self.weight[row, column:end]),
                    _shared_load_1d(self.input[column:end]),
                ]
            is_last_row = local_row + 1 == row_count
            if emit_square_sum:
                # Square sum and row-zero output are adjacent in the packed
                # metadata record.  Compute writes both scalars directly and
                # returns this raw token through C2M after the stores.  STU's
                # no-copy writeback then supplies the projection completion
                # edge without a shared-memory data roundtrip.
                raw_output = RawAddress(
                    self.square_sum_output,
                    self.DIRECT_RECORD_SLOT,
                ).writeback()
                if is_last_row:
                    raw_output.bar(self._bar("output"))
                instructions.append(raw_output)
                instructions.append(_shared_load_1d(self.metadata_scale))
                instructions.append(_shared_load_1d(self.metadata_base))
                instructions.append(
                    _shared_store_1d(self.metadata_tail_output)
                )
            else:
                store = _shared_store_1d(self.output[row:row + 1])
                if is_last_row:
                    store.bar(self._bar("output"))
                instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role == "output":
            return self._bar_release_if_present(role, self.num_sms)
        if role == "metadata" and self.fuse_hc_post:
            return self._bar_release_if_present(
                role, self.FUSED_GROUPS * self.FUSED_SPLITS
            )
        if role == "residual" and self.fuse_hc_post:
            return self._bar_release_if_present(role, self.FUSED_SPLITS)
        return 0


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


class SchedDsv4RouterBf16Gemv(Schedule):
    """Vectorized BF16 router GEMV with fused route-score preparation."""

    def __init__(
        self,
        weight,
        input,
        bias,
        output,
        *,
        rows_per_task: int = 2,
    ):
        super().__init__()
        self.weight = weight
        self.input = input
        self.bias = bias
        self.output = output
        self.rows_per_task = int(rows_per_task)

    def _on_place(self):
        if self.rows_per_task not in (1, 2, 4):
            raise ValueError("router rows per task must be 1, 2, or 4")
        if (
            self.weight.dtype != torch.bfloat16
            or self.weight.ndim != 2
            or not self.weight.is_contiguous()
        ):
            raise ValueError("router weight must be contiguous BF16 [M,K]")
        self.rows, self.k = self.weight.shape
        if self.rows % self.rows_per_task:
            raise ValueError("router M must divide the constexpr row group")
        if self.k % 1024:
            raise ValueError("router K must be K1024 aligned")
        if (
            self.input.dtype != torch.bfloat16
            or self.input.numel() != self.k
            or not self.input.is_contiguous()
        ):
            raise ValueError("router input must be contiguous BF16 [K]")
        if (
            self.bias.dtype != torch.float32
            or self.bias.numel() != self.rows
            or not self.bias.is_contiguous()
        ):
            raise ValueError("router bias must be contiguous FP32 [M]")
        if (
            self.output.dtype != torch.float32
            or tuple(self.output.shape) != (self.rows, 2)
            or not self.output.is_contiguous()
        ):
            raise ValueError("router output must be contiguous FP32 [M,2]")
        if any(
            tensor.device != self.output.device
            for tensor in (self.weight, self.input, self.bias)
        ):
            raise ValueError("router tensors must share one device")
        self.tasks = self.rows // self.rows_per_task
        if not 0 < self.num_sms <= self.tasks:
            raise ValueError("router GEMV requires 1..M/rows_per_task SMs")
        self._paired_load_plans = None
        if self.rows_per_task == 2:
            self._paired_load_plans = torch.tensor(
                [
                    (
                        self.input.data_ptr(),
                        self.weight[task * self.rows_per_task].data_ptr(),
                    )
                    for task in range(self.tasks)
                ],
                dtype=torch.int64,
                device=self.output.device,
            )

    def _task_shard(self, sm):
        tasks_per_sm, extra = divmod(self.tasks, self.num_sms)
        task_start = sm * tasks_per_sm + min(sm, extra)
        task_count = tasks_per_sm + int(sm < extra)
        return task_start, task_count

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        task_start, task_count = self._task_shard(sm)
        instructions = []
        for local_task, task in enumerate(
            range(task_start, task_start + task_count)
        ):
            row_start = task * self.rows_per_task
            row_stop = row_start + self.rows_per_task
            instructions.append(
                Dsv4RouterBf16GemvSm100(self.k, self.rows_per_task)
            )
            if self.rows_per_task == 2:
                instructions.append(
                    TmaLoadPair1D(
                        self._paired_load_plans[task],
                        first_bytes=self.input.numel() * self.input.element_size(),
                        second_bytes=(
                            self.weight[row_start:row_stop].numel()
                            * self.weight.element_size()
                        ),
                    ).fixed_port(0)
                )
            else:
                instructions.append(_shared_load_1d(self.input).fixed_port(1))
                instructions.append(
                    _shared_load_1d(
                        self.weight[row_start:row_stop].reshape(-1)
                    ).fixed_port(0)
                )
            instructions.append(
                RawAddress(
                    self.bias[row_start:row_stop], config.num_slots + 4
                ).fixed_port(1)
            )
            store = RawAddress(
                self.output[row_start:row_stop], config.num_slots + 5
            ).writeback()
            if local_task + 1 == task_count:
                store.bar(self._bar("output"))
            instructions.append(store)
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedLayeredDsv4RouterBf16Gemv(Schedule):
    """Counter-selected BF16 router with the same vectorized compute task.

    Only compact pointer plans and biases are layered.  Checkpoint weights stay
    in their original resident allocations, and the allocator applies the
    active loop counters directly to the two memory-command addresses.
    """

    def __init__(
        self,
        weights,
        input,
        biases,
        output,
        *,
        counter_strides=(),
        rows_per_task: int = 2,
    ):
        super().__init__()
        self.weights = tuple(weights)
        self.input = input
        self.biases = tuple(biases)
        self.output = output
        self.counter_strides = tuple(
            (int(counter), int(stride))
            for counter, stride in counter_strides
        )
        self.rows_per_task = int(rows_per_task)

    def _on_place(self):
        if not self.weights or len(self.weights) != len(self.biases):
            raise ValueError("layered router requires matching weight/bias layers")
        representative = self.weights[0]
        if (
            representative.dtype != torch.bfloat16
            or representative.ndim != 2
            or not representative.is_contiguous()
        ):
            raise ValueError("layered router weight must be contiguous BF16 [M,K]")
        self.rows, self.k = representative.shape
        if self.rows_per_task not in (1, 2, 4):
            raise ValueError("router rows per task must be 1, 2, or 4")
        if self.rows % self.rows_per_task or self.k % 1024:
            raise ValueError("layered router shape must divide its constexpr task")
        if self.rows_per_task != 2:
            raise ValueError("layered router currently uses paired two-row loads")
        for weight in self.weights:
            if (
                weight.device != representative.device
                or weight.dtype != representative.dtype
                or weight.shape != representative.shape
                or not weight.is_contiguous()
            ):
                raise ValueError("layered router weights must have identical storage")
        for bias in self.biases:
            if (
                bias.device != representative.device
                or bias.dtype != torch.float32
                or bias.numel() != self.rows
                or not bias.is_contiguous()
            ):
                raise ValueError("layered router biases must be contiguous FP32 [M]")
        if (
            self.input.device != representative.device
            or self.input.dtype != torch.bfloat16
            or self.input.numel() != self.k
            or not self.input.is_contiguous()
        ):
            raise ValueError("layered router input must be contiguous BF16 [K]")
        if (
            self.output.device != representative.device
            or self.output.dtype != torch.float32
            or tuple(self.output.shape) != (self.rows, 2)
            or not self.output.is_contiguous()
        ):
            raise ValueError("layered router output must be contiguous FP32 [M,2]")
        if any(
            counter < 0 or counter >= 32 or stride <= 0
            for counter, stride in self.counter_strides
        ):
            raise ValueError(
                "layered router counter strides require reg [0,31] and positive stride"
            )
        if len(self.weights) > 1 and not self.counter_strides:
            raise ValueError("multi-layer router requires loop-counter strides")
        self.tasks = self.rows // self.rows_per_task
        if not 0 < self.num_sms <= self.tasks:
            raise ValueError("layered router requires 1..M/2 workers")

        plans = torch.empty(
            (len(self.weights), self.tasks, 2), dtype=torch.int64, device="cpu"
        )
        for layer, weight in enumerate(self.weights):
            plans[layer, :, 0] = self.input.data_ptr()
            plans[layer, :, 1] = torch.tensor(
                [
                    weight[task * self.rows_per_task].data_ptr()
                    for task in range(self.tasks)
                ],
                dtype=torch.int64,
            )
        self.paired_load_plans = plans.to(representative.device)
        self.bias_layers = torch.stack(self.biases).contiguous()
        self.plan_layer_bytes = (
            self.paired_load_plans.stride(0)
            * self.paired_load_plans.element_size()
        )
        self.bias_layer_bytes = (
            self.bias_layers.stride(0) * self.bias_layers.element_size()
        )

    def _task_shard(self, sm):
        tasks_per_sm, extra = divmod(self.tasks, self.num_sms)
        task_start = sm * tasks_per_sm + min(sm, extra)
        task_count = tasks_per_sm + int(sm < extra)
        return task_start, task_count

    def _offsets(self, layer_bytes):
        return tuple(
            (counter, stride * layer_bytes)
            for counter, stride in self.counter_strides
        )

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        task_start, task_count = self._task_shard(sm)
        instructions = []
        for local_task, task in enumerate(
            range(task_start, task_start + task_count)
        ):
            row_start = task * self.rows_per_task
            row_stop = row_start + self.rows_per_task
            instructions.append(
                Dsv4RouterBf16GemvSm100(self.k, self.rows_per_task)
            )
            pair = TmaLoadPair1D(
                self.paired_load_plans[0, task],
                first_bytes=self.input.numel() * self.input.element_size(),
                second_bytes=(
                    self.rows_per_task
                    * self.k
                    * self.weights[0].element_size()
                ),
            ).fixed_port(0)
            pair = RepeatM.offsetByCounters(
                self._offsets(self.plan_layer_bytes), pair
            )
            bias = RawAddress(
                self.bias_layers[0, row_start:row_stop],
                config.num_slots + 4,
            ).fixed_port(1)
            bias = RepeatM.offsetByCounters(
                self._offsets(self.bias_layer_bytes), bias
            )
            store = RawAddress(
                self.output[row_start:row_stop], config.num_slots + 5
            ).writeback()
            if local_task + 1 == task_count:
                store.bar(self._bar("output"))
            instructions.extend((pair, bias, store))
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


class SchedDsv4HcPreRms(Schedule):
    """Fuse mHC pre mixing with learned RMSNorm using packed raw metadata.

    The transport contract matches the single compile-time specialization in
    ``deepseek_v4.cuh``: one 56-float input record uses a fixed raw-address
    slot.  Its four padding floats are intentionally unspecified.  Post/comb
    follow the selected primary output, so its existing STU lease also carries
    their global address without another mailbox transaction.

    The primary output is mutually exclusive: either ``output`` publishes the
    canonical BF16 vector, or ``fp8_output`` plus ``fp8_scale`` publishes its
    block-128 E4M3/UE8M0 representation.  The FP8 path still performs the
    model-required internal BF16 rounding before quantization.
    """

    def __init__(
        self,
        residual,
        mixes,
        scale,
        base,
        norm_weight,
        output,
        post,
        comb,
        residual_square_sum=None,
        packed_metadata=None,
        packed_output=None,
        zero_fp32_output=None,
        fp8_output=None,
        fp8_scale=None,
        split_metadata_splits: int = 0,
    ):
        super().__init__()
        self.residual = residual
        self.mixes = mixes
        self.scale = scale
        self.base = base
        self.norm_weight = norm_weight
        self.output = output
        self.post = post
        self.comb = comb
        self.residual_square_sum = residual_square_sum
        self.packed_metadata = packed_metadata
        self.packed_output = packed_output
        self.zero_fp32_output = zero_fp32_output
        self.fp8_output = fp8_output
        self.fp8_scale = fp8_scale
        self.split_metadata_splits = split_metadata_splits

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("fused mHC/RMS uses exactly one SM")
        if (
            self.residual.dtype != torch.bfloat16
            or tuple(self.residual.shape) != (4, 4096)
        ):
            raise ValueError("fused mHC/RMS residual must be BF16 [4,4096]")
        if self.mixes.dtype != torch.float32 or self.mixes.numel() != 24:
            raise ValueError("fused mHC/RMS mixes must contain 24 FP32 values")
        if self.scale.dtype != torch.float32 or self.scale.numel() != 3:
            raise ValueError("fused mHC/RMS scale must contain three FP32 values")
        if self.base.dtype != torch.float32 or self.base.numel() != 24:
            raise ValueError("fused mHC/RMS base must contain 24 FP32 values")
        if (
            self.norm_weight.dtype != torch.bfloat16
            or self.norm_weight.numel() != 4096
            or not self.norm_weight.is_contiguous()
        ):
            raise ValueError("fused mHC/RMS weight must be BF16 [4096]")
        if self.post.dtype != torch.float32 or self.post.numel() != 4:
            raise ValueError("fused mHC/RMS post must contain four FP32 values")
        if (
            self.comb.dtype != torch.float32
            or tuple(self.comb.shape) != (4, 4)
        ):
            raise ValueError("fused mHC/RMS comb must be FP32 [4,4]")
        if (
            self.residual_square_sum is None
            or self.residual_square_sum.dtype != torch.float32
            or self.residual_square_sum.numel() != 1
        ):
            raise ValueError(
                "fused mHC/RMS residual square sum must contain one FP32 value"
            )
        expected_metadata_items = (
            self.split_metadata_splits * 32 + 28
            if self.split_metadata_splits
            else 56
        )
        if self.split_metadata_splits not in (0, 2, 4, 8, 16):
            raise ValueError(
                "fused mHC/RMS metadata splits must be 0, 2, 4, 8, or 16"
            )
        if (
            self.packed_metadata is None
            or self.packed_metadata.dtype != torch.float32
            or self.packed_metadata.numel() != expected_metadata_items
            or not self.packed_metadata.is_contiguous()
        ):
            raise ValueError(
                "packed mHC/RMS input metadata has the wrong split-record size"
            )
        if self.zero_fp32_output is not None and (
            self.zero_fp32_output.dtype != torch.float32
            or self.zero_fp32_output.numel() != 4096
            or not self.zero_fp32_output.is_contiguous()
        ):
            raise ValueError(
                "fused mHC/RMS zero output must be contiguous FP32 [4096]"
            )
        if (self.fp8_output is None) != (self.fp8_scale is None):
            raise ValueError(
                "fused mHC/RMS FP8 output and scale must be provided together"
            )
        if self.fp8_output is not None and (
            self.fp8_output.dtype != torch.float8_e4m3fn
            or self.fp8_output.numel() != 4096
            or not self.fp8_output.is_contiguous()
        ):
            raise ValueError(
                "fused mHC/RMS FP8 output must be contiguous E4M3 [4096]"
            )
        if self.fp8_scale is not None and (
            self.fp8_scale.dtype != torch.float8_e8m0fnu
            or self.fp8_scale.numel() != 32
            or not self.fp8_scale.is_contiguous()
        ):
            raise ValueError(
                "fused mHC/RMS FP8 scale must be contiguous UE8M0 [32]"
            )
        has_bf16_output = self.output is not None
        has_fp8_output = self.fp8_output is not None
        if has_bf16_output == has_fp8_output:
            raise ValueError(
                "fused mHC/RMS must output either BF16 or FP8, not both"
            )
        if has_bf16_output:
            if (
                self.output.dtype != torch.bfloat16
                or self.output.numel() != 4096
                or not self.output.is_contiguous()
            ):
                raise ValueError(
                    "fused mHC/RMS BF16 output must be contiguous BF16 [4096]"
                )
            if (
                self.packed_output is None
                or self.packed_output.dtype != torch.bfloat16
                or self.packed_output.numel() != 4136
                or not self.packed_output.is_contiguous()
            ):
                raise ValueError(
                    "packed BF16 mHC/RMS output must be contiguous storage [4136]"
                )
            if self.output.data_ptr() != self.packed_output.data_ptr():
                raise ValueError("mHC BF16 output must start the packed output")
            metadata_address = self.packed_output.data_ptr() + 4096 * 2
        else:
            if (
                self.packed_output is None
                or self.packed_output.dtype != torch.uint8
                or self.packed_output.numel() != 4176
                or not self.packed_output.is_contiguous()
            ):
                raise ValueError(
                    "packed FP8 mHC/RMS output must be contiguous byte storage [4176]"
                )
            if self.fp8_output.data_ptr() != self.packed_output.data_ptr():
                raise ValueError("mHC FP8 output must start the packed output")
            metadata_address = self.packed_output.data_ptr() + 4096
        if self.post.data_ptr() != metadata_address:
            raise ValueError("mHC post output must follow the primary output")
        if self.comb.data_ptr() != metadata_address + 4 * 4:
            raise ValueError("mHC comb output must follow post in the packed output")

    def schedule(self, sm):
        if sm != 0:
            return []
        if self.split_metadata_splits:
            metadata = TmaLoad1D(self.packed_metadata).fixed_port(1)
            if self._bar("metadata") is not None:
                metadata.bar(self._bar("metadata"))
        else:
            metadata = RawAddress(
                self.packed_metadata, config.num_slots
            ).fixed_port(1)
            if self._bar("metadata") is not None:
                metadata.bar(self._bar("metadata"))
        residual = TmaLoad1D(self.residual).fixed_port(0)
        if self._bar("residual") is not None:
            residual.bar(self._bar("residual"))
        instructions = [Dsv4HcPreRms(
            zero_fp32_output=self.zero_fp32_output is not None,
            output_fp8=self.fp8_output is not None,
            split_metadata_splits=self.split_metadata_splits,
        )]
        norm_weight = TmaLoad1D(self.norm_weight).fixed_port(1)
        if self._bar("reuse") is not None:
            norm_weight.bar(self._bar("reuse"))
        # Publish the allocator-owned output workspace before starting the
        # norm-weight transfer.  The task can mix directly into that lease
        # while the LDU fetches the norm weight, avoiding common scratch and
        # retaining the original overlap.
        instructions.extend((metadata, residual))
        if self.fp8_output is not None:
            instructions.extend(
                (
                    _shared_store_1d(self.fp8_output),
                    norm_weight,
                    _shared_store_1d(self.fp8_scale).bar(self._bar("output")),
                )
            )
        else:
            instructions.extend(
                (
                    TmaStore1D(self.output).bar(self._bar("output")),
                    norm_weight,
                )
            )
        if self.zero_fp32_output is not None:
            instructions.append(TmaStore1D(self.zero_fp32_output))
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4HcPost(Schedule):
    """Shard mHC post with an optional compact TMA/raw-address transport.

    The compact form keeps bulk branch/residual/output data on LDU/STU.  The
    contiguous 20-FP32 coefficient record is a direct, instruction-packed raw
    address and consumes no shared slot or memory command.  The legacy form
    remains available to diagnostic callers that have not constructed TMA
    descriptors; both forms execute the same compute opcode.
    """

    def __init__(
        self,
        branch,
        residual,
        post,
        comb,
        output,
        *,
        launcher=None,
        packed_coefficients=None,
        packed_input_record=None,
        packed_output_record=None,
    ):
        super().__init__()
        self.branch = branch
        self.residual = residual
        self.post = post
        self.comb = comb
        self.output = output
        self.launcher = launcher
        self.packed_coefficients = packed_coefficients
        self.packed_input_record = packed_input_record
        self.packed_output_record = packed_output_record
        self.compact_io = launcher is not None
        if self.compact_io != (packed_coefficients is not None):
            raise ValueError(
                "compact mHC post requires both launcher and packed coefficients"
            )
        if (packed_input_record is None) != (packed_output_record is None):
            raise ValueError("packed mHC post requires both input/output records")

    def _on_place(self):
        if (
            self.num_sms <= 0
            or 4096 % self.num_sms
            or (4096 // self.num_sms) % 8
        ):
            raise ValueError(
                "DeepSeek mHC post requires 16-byte-aligned equal shards"
            )
        if (
            self.branch.dtype not in (torch.bfloat16, torch.float32)
            or self.branch.numel() != 4096
            or not self.branch.is_contiguous()
        ):
            raise ValueError(
                "mHC branch must contain 4096 contiguous BF16 or FP32 values"
            )
        if self.residual.dtype != torch.bfloat16 or tuple(self.residual.shape) != (4, 4096):
            raise ValueError("mHC residual must be BF16 [4,4096]")
        if self.post.dtype != torch.float32 or self.post.numel() != 4:
            raise ValueError("mHC post coefficients must contain four FP32 values")
        if self.comb.dtype != torch.float32 or tuple(self.comb.shape) != (4, 4):
            raise ValueError("mHC combination matrix must be FP32 [4,4]")
        if self.output.dtype != torch.bfloat16 or tuple(self.output.shape) != (4, 4096):
            raise ValueError("mHC post output must be BF16 [4,4096]")
        if self.compact_io:
            if (
                self.packed_coefficients.dtype != torch.float32
                or self.packed_coefficients.numel() != 20
                or not self.packed_coefficients.is_contiguous()
            ):
                raise ValueError(
                    "compact mHC post coefficients must be contiguous FP32 [20]"
                )
            base_address = self.packed_coefficients.data_ptr()
            if self.post.data_ptr() != base_address:
                raise ValueError("mHC post coefficients must start the packed record")
            if self.comb.data_ptr() != base_address + 4 * 4:
                raise ValueError("mHC combination matrix must follow post coefficients")
            width = 4096 // self.num_sms
            self.packed_rw = self.packed_input_record is not None
            if self.packed_rw:
                if self.branch.dtype != torch.bfloat16:
                    raise ValueError(
                        "packed 2-D mHC post currently requires a BF16 branch"
                    )
                for name, record in (
                    ("input", self.packed_input_record),
                    ("output", self.packed_output_record),
                ):
                    if (
                        record.dtype != torch.bfloat16
                        or tuple(record.shape) != (6, 4096)
                        or not record.is_contiguous()
                    ):
                        raise ValueError(
                            f"packed mHC {name} record must be BF16 [6,4096]"
                        )
                if (
                    self.branch.data_ptr() != self.packed_input_record[0].data_ptr()
                    or self.residual.data_ptr()
                    != self.packed_input_record[1:5].data_ptr()
                    or self.output.data_ptr()
                    != self.packed_output_record[1:5].data_ptr()
                ):
                    raise ValueError("mHC tensors must be views of their packed records")
                self.input_record_tma = TmaTensor(
                    self.launcher, self.packed_input_record
                ).rowmajor_2d("load", 6, width)
                self.output_record_tma = TmaTensor(
                    self.launcher, self.packed_output_record
                ).rowmajor_2d("store", 6, width)
                if self.input_record_tma.size != self.output_record_tma.size:
                    raise ValueError("packed mHC input/output TMA sizes must match")
                self.packed_rw_slots = bytes2slots(self.input_record_tma.size)
            else:
                self.residual_tma = TmaTensor(
                    self.launcher, self.residual
                ).rowmajor_2d("load", 4, width)
                self.output_tma = TmaTensor(
                    self.launcher, self.output
                ).rowmajor_2d("store", 4, width)

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        width = 4096 // self.num_sms
        start = sm * width
        stop = start + width
        if self.compact_io:
            if self.packed_rw:
                coords = self.input_record_tma.cord2tma(0, start)
                output_coords = self.output_record_tma.cord2tma(0, start)
                if coords != output_coords:
                    raise ValueError("packed mHC input/output coordinates must match")
                return [
                    Dsv4HcPost(
                        width,
                        branch_fp32=self.branch.dtype == torch.float32,
                        packed_coefficients=self.packed_coefficients,
                        packed_rw=True,
                    ),
                    TmaReadWrite2D(
                        self.input_record_tma.arg,
                        self.output_record_tma.arg,
                        coords=(coords[0], coords[1]),
                        bytes=self.input_record_tma.size,
                        num_slots=self.packed_rw_slots,
                    ).bar(self._bar("output")),
                ]
            store = self.output_tma.cord(0, start).bar(self._bar("output"))
            return [
                Dsv4HcPost(
                    width,
                    branch_fp32=self.branch.dtype == torch.float32,
                    packed_coefficients=self.packed_coefficients,
                ),
                TmaLoad1D(self.branch[start:stop]),
                self.residual_tma.cord(0, start),
                store,
            ]
        instructions = [
            Dsv4HcPost(width, branch_fp32=self.branch.dtype == torch.float32),
            TmaLoad1D(self.branch[start:stop]),
            *(
                TmaLoad1D(self.residual[branch, start:stop])
                for branch in range(4)
            ),
            TmaLoad1D(self.post),
            TmaLoad1D(self.comb),
        ]
        stores = [
            TmaStore1D(self.output[branch, start:stop])
            for branch in range(4)
        ]
        stores[-1].bar(self._bar("output"))
        return instructions + stores

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedOverlapAsyncBarrierReload(Schedule):
    """Join useful work with a disjoint LDU barrier-bank clear.

    ``inner`` occupies the low SM range. Selected otherwise-idle SMs each
    restore a contiguous slice of an inactive barrier bank. Both paths publish
    the same ordinary output dependency, so the clear is latency-free whenever
    it finishes before the useful path.
    """

    def __init__(
        self,
        inner,
        inner_sms: int,
        bar_source: torch.Tensor,
        first_bar: int,
        count: int,
        worker_base: int,
        workers: int,
        *,
        special_slot: int = 2,
        clear_input_role: str = "input",
        skip_initial_loop: bool = False,
    ):
        super().__init__()
        self.inner = inner
        self.inner_sms = int(inner_sms)
        self.bar_source = bar_source
        self.first_bar = int(first_bar)
        self.count = int(count)
        self.worker_base = int(worker_base)
        self.workers = int(workers)
        self.special_slot = int(special_slot)
        self.clear_input_role = str(clear_input_role)
        self.skip_initial_loop = bool(skip_initial_loop)

    def _on_place(self):
        if (
            self.inner_sms <= 0
            or self.worker_base < self.inner_sms
            or self.workers <= 0
            or self.worker_base + self.workers > self.num_sms
            or self.count < self.workers
        ):
            raise ValueError("async barrier reload placement is invalid")
        if self.skip_initial_loop and self._bar("clear") is None:
            raise ValueError(
                "an initially skipped reload needs a clear-completion edge"
            )
        inner = self.inner._clone()
        inner._bars.update(self._bars)
        self.placed_inner = inner.place(self.inner_sms)

    def schedule(self, sm):
        if sm < 0 or sm >= self.num_sms:
            return []
        instructions = (
            list(self.placed_inner.schedule(sm))
            if sm < self.inner_sms
            else []
        )
        worker = sm - self.worker_base
        if 0 <= worker < self.workers:
            width, remainder = divmod(self.count, self.workers)
            local_count = width + int(worker < remainder)
            local_offset = worker * width + min(worker, remainder)
            clear = LduAsyncReloadBarriers(
                self.bar_source,
                self.first_bar + local_offset,
                local_count,
                self._bar(self.clear_input_role),
                self.special_slot,
                skip_initial_loop=self.skip_initial_loop,
            )
            if self._bar("clear") is not None:
                clear.bar(self._bar("clear"))
            instructions.append(clear)
        return instructions

    def bar_release_count(self, role: str):
        if role == "clear":
            return self._bar_release_if_present(role, self.workers)
        if role != "output":
            return self.placed_inner.bar_release_count(role)
        inner_count = self.placed_inner.bar_release_count(role)
        return self._bar_release_if_present(
            role, inner_count + self.workers
        )

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


class SchedDsv4CompressorStateStore(Schedule):
    """Publish overlap and ordinary compressor rows for a later token."""

    def __init__(
        self,
        values,
        scores,
        biases,
        output_values,
        output_scores,
    ):
        super().__init__()
        self.values = tuple(values)
        self.scores = tuple(scores)
        self.biases = tuple(biases)
        self.output_values = tuple(output_values)
        self.output_scores = tuple(output_scores)

    def _on_place(self):
        groups = (
            self.values,
            self.scores,
            self.biases,
            self.output_values,
            self.output_scores,
        )
        if (
            self.num_sms not in (1, 2)
            or any(len(group) != self.num_sms for group in groups)
        ):
            raise ValueError(
                "compressor-state publication requires one or two row tasks"
            )
        widths = {tensor.numel() for group in groups for tensor in group}
        if len(widths) != 1 or next(iter(widths)) not in (128, 512):
            raise ValueError(
                "compressor-state rows must share width 128 or 512"
            )
        self.width = next(iter(widths))
        for group in groups:
            for tensor in group:
                if (
                    tensor.dtype != torch.float32
                    or tensor.device.type != "cuda"
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        "compressor-state rows must be contiguous CUDA FP32"
                    )

    def schedule(self, sm):
        if sm < 0:
            return []
        return [
            Dsv4CompressorStateStore(self.width),
            TmaLoad1D(self.values[sm]),
            TmaLoad1D(self.scores[sm]),
            TmaLoad1D(self.biases[sm]),
            TmaStore1D(self.output_values[sm]),
            TmaStore1D(self.output_scores[sm]).bar(self._bar("output")),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4GatedPool(Schedule):
    def __init__(
        self,
        values,
        scores,
        output,
        *,
        tail_values=None,
        tail_scores=None,
        tail_bias=None,
    ):
        super().__init__()
        self.values = values
        self.scores = scores
        self.output = output
        self.tail_values = tail_values
        self.tail_scores = tail_scores
        self.tail_bias = tail_bias

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
        history_rows, self.width = self.values.shape
        if (self.tail_values is None) != (self.tail_scores is None):
            raise ValueError(
                "DeepSeek gated-pool tail values and scores must be supplied together"
            )
        if self.tail_values is not None:
            if (
                self.tail_values.dtype != torch.float32
                or self.tail_values.numel() != self.width
                or not self.tail_values.is_contiguous()
            ):
                raise ValueError(
                    "DeepSeek gated-pool tail values must be contiguous FP32 [width]"
                )
            if (
                self.tail_scores.dtype != torch.float32
                or self.tail_scores.shape != self.tail_values.shape
                or not self.tail_scores.is_contiguous()
            ):
                raise ValueError(
                    "DeepSeek gated-pool tail scores must match the tail values"
                )
        if self.tail_bias is not None:
            if self.tail_values is None:
                raise ValueError(
                    "DeepSeek gated-pool tail bias requires a tail row"
                )
            if (
                self.tail_bias.dtype != torch.float32
                or self.tail_bias.numel() != self.width
                or not self.tail_bias.is_contiguous()
            ):
                raise ValueError(
                    "DeepSeek gated-pool tail bias must be contiguous FP32 [width]"
                )
        self.pool_rows = history_rows + int(self.tail_values is not None)
        if self.pool_rows <= 0:
            raise ValueError("DeepSeek gated pooling needs at least one row")

    def schedule(self, sm):
        if sm != 0:
            return []
        row_bytes = self.width * self.values.element_size()
        instructions = [
            Dsv4GatedPool(
                self.pool_rows,
                self.width,
                tail_bias=self.tail_bias is not None,
            )
        ]
        if self.values.shape[0]:
            instructions += RepeatM.on(
                self.values.shape[0],
                (TmaLoad1D(self.values[0]), row_bytes),
                (TmaLoad1D(self.scores[0]), row_bytes),
            )
        if self.tail_values is not None:
            instructions += [
                TmaLoad1D(self.tail_values),
                TmaLoad1D(self.tail_scores),
            ]
            if self.tail_bias is not None:
                instructions.append(TmaLoad1D(self.tail_bias))
        instructions.append(
            TmaStore1D(self.output).bar(self._bar("output"))
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4GatedPoolRmsRope(Schedule):
    """Pool, weighted-normalize, rotate, and optionally Hadamard one row."""

    def __init__(
        self,
        values,
        scores,
        weight,
        table,
        output,
        *,
        epsilon: float,
        tail_values=None,
        tail_scores=None,
        tail_bias=None,
        hadamard=False,
        fixed_table_id=None,
        prefetch_static_inputs=False,
    ):
        super().__init__()
        self.values = values
        self.scores = scores
        self.weight = weight
        self.table = table
        self.output = output
        self.epsilon = epsilon
        self.tail_values = tail_values
        self.tail_scores = tail_scores
        self.tail_bias = tail_bias
        self.hadamard = bool(hadamard)
        self.fixed_table_id = fixed_table_id
        self.prefetch_static_inputs = bool(prefetch_static_inputs)

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("fused gated-pool epilogue uses exactly one SM")
        if (
            self.values.dtype != torch.float32
            or self.values.ndim != 2
            or self.values.shape[1] not in (128, 512)
        ):
            raise ValueError(
                "fused gated-pool values must be FP32 [rows,128|512]"
            )
        if self.scores.dtype != torch.float32 or self.scores.shape != self.values.shape:
            raise ValueError("fused gated-pool scores must match the values")
        history_rows, self.width = self.values.shape
        if (
            self.weight.dtype != torch.bfloat16
            or tuple(self.weight.shape) != (self.width,)
            or not self.weight.is_contiguous()
        ):
            raise ValueError("fused gated-pool weight must be BF16 [width]")
        if self.table.dtype != torch.float32 or tuple(self.table.shape) != (32, 2):
            raise ValueError("fused gated-pool RoPE table must be FP32 [32,2]")
        if (
            self.output.dtype != torch.bfloat16
            or self.output.numel() != self.width
            or not self.output.is_contiguous()
        ):
            raise ValueError("fused gated-pool output must be BF16 [width]")
        if self.hadamard and self.width != 128:
            raise ValueError("fused gated-pool Hadamard requires width 128")
        if self.epsilon <= 0:
            raise ValueError("fused gated-pool RMS epsilon must be positive")
        if self.fixed_table_id is not None and not 0 <= self.fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        if (self.tail_values is None) != (self.tail_scores is None):
            raise ValueError(
                "fused gated-pool tail values and scores must be paired"
            )
        if self.tail_values is not None:
            for name, tensor in (
                ("tail values", self.tail_values),
                ("tail scores", self.tail_scores),
            ):
                if (
                    tensor.dtype != torch.float32
                    or tensor.numel() != self.width
                    or not tensor.is_contiguous()
                ):
                    raise ValueError(
                        f"fused gated-pool {name} must be FP32 [width]"
                    )
        if self.prefetch_static_inputs and self.tail_values is None:
            raise ValueError(
                "static gated-pool prefetch requires a dynamic tail row"
            )
        if self.tail_bias is not None:
            if self.tail_values is None:
                raise ValueError("fused gated-pool bias requires a tail row")
            if (
                self.tail_bias.dtype != torch.float32
                or self.tail_bias.numel() != self.width
                or not self.tail_bias.is_contiguous()
            ):
                raise ValueError("fused gated-pool bias must be FP32 [width]")
        elif self.tail_values is not None:
            raise ValueError(
                "fused gated-pool dynamic tail requires an FP32 bias row"
            )
        self.pool_rows = history_rows + int(self.tail_values is not None)
        if self.pool_rows <= 0:
            raise ValueError("fused gated pooling needs at least one row")

    def schedule(self, sm):
        if sm != 0:
            return []
        row_bytes = self.width * self.values.element_size()
        instructions = [
            Dsv4GatedPoolRmsRope(
                self.pool_rows,
                self.width,
                tail_bias=self.tail_bias is not None,
                hadamard=self.hadamard,
                epsilon=self.epsilon,
                fixed_table_id=self.fixed_table_id,
            )
        ]
        if self.values.shape[0]:
            instructions += RepeatM.on(
                self.values.shape[0],
                (TmaLoad1D(self.values[0]), row_bytes),
                (TmaLoad1D(self.scores[0]), row_bytes),
            )
        if self.tail_values is not None:
            tail_values = TmaLoad1D(self.tail_values)
            tail_scores = TmaLoad1D(self.tail_scores)
            if self.prefetch_static_inputs:
                tail_bar = self._bar("tail")
                for tail_load in (tail_values, tail_scores):
                    if tail_bar is not None:
                        tail_load.bar(tail_bar)
                    tail_load.fixed_port(0)
            instructions.extend((tail_values, tail_scores))
            tail_bias = TmaLoad1D(self.tail_bias)
            if self.prefetch_static_inputs:
                tail_bias.fixed_port(1)
            instructions.append(tail_bias)
        weight = TmaLoad1D(self.weight)
        if self.prefetch_static_inputs:
            weight.fixed_port(1)
        instructions.append(weight)
        if self.fixed_table_id is None:
            instructions.append(TmaLoad1D(self.table))
        instructions.append(
            TmaStore1D(self.output.reshape(-1)).bar(self._bar("output"))
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4GatedPoolPacked8Shard128(Schedule):
    """Pool a prepacked history with one independent 128-wide shard per SM."""

    ROWS_PER_BLOCK = 8
    SHARD_WIDTH = 128

    def __init__(
        self,
        packed_history,
        history_rows,
        output,
        *,
        tail_values,
        tail_scores,
        tail_bias,
    ):
        super().__init__()
        self.packed_history = packed_history
        self.history_rows = int(history_rows)
        self.output = output
        self.tail_values = tail_values
        self.tail_scores = tail_scores
        self.tail_bias = tail_bias

    def _on_place(self):
        expected_tail = (
            self.packed_history.shape[0] * self.SHARD_WIDTH
            if self.packed_history.ndim == 5
            else -1
        )
        if (
            self.packed_history.dtype != torch.float32
            or self.packed_history.ndim != 5
            or tuple(self.packed_history.shape[2:])
            != (self.ROWS_PER_BLOCK, 2, self.SHARD_WIDTH)
            or not self.packed_history.is_contiguous()
        ):
            raise ValueError(
                "packed gated-pool history must be contiguous FP32 "
                "[shards,blocks,8,2,128]"
            )
        self.shards, self.blocks = self.packed_history.shape[:2]
        if self.num_sms != self.shards:
            raise ValueError("packed gated pooling uses one SM per width shard")
        if not 0 < self.history_rows <= self.blocks * self.ROWS_PER_BLOCK:
            raise ValueError("packed gated-pool row count exceeds its blocks")
        for name, tensor in (
            ("tail values", self.tail_values),
            ("tail scores", self.tail_scores),
            ("tail bias", self.tail_bias),
        ):
            if (
                tensor.dtype != torch.float32
                or tensor.numel() != expected_tail
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"packed gated-pool {name} must be contiguous FP32 [width]"
                )
        if (
            self.output.dtype != torch.bfloat16
            or self.output.ndim != 1
            or self.output.numel() != expected_tail
            or not self.output.is_contiguous()
        ):
            raise ValueError("packed gated-pool output must be contiguous BF16 [width]")

    def schedule(self, sm):
        if sm < 0:
            return []
        block_bytes = (
            self.ROWS_PER_BLOCK * 2 * self.SHARD_WIDTH
            * self.packed_history.element_size()
        )
        instructions = [Dsv4GatedPoolPacked8Shard128(self.history_rows)]
        instructions += RepeatM.on(
            self.blocks,
            (TmaLoad1D(self.packed_history[sm, 0].reshape(-1)), block_bytes),
        )
        shard_start = sm * self.SHARD_WIDTH
        shard_end = shard_start + self.SHARD_WIDTH
        instructions += [
            TmaLoad1D(self.tail_values[shard_start:shard_end]),
            TmaLoad1D(self.tail_scores[shard_start:shard_end]),
            TmaLoad1D(self.tail_bias[shard_start:shard_end]),
            TmaStore1D(self.output[shard_start:shard_end]).bar(
                self._bar("output")
            ),
        ]
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4GatedPoolPacked8RmsPartial(Schedule):
    """Pool four 128-wide HCA shards and publish FP32 RMS partials."""

    ROWS_PER_BLOCK = 8
    SHARD_WIDTH = 128

    def __init__(
        self,
        packed_history,
        history_rows,
        pooled_output,
        partial_output,
        *,
        tail_values,
        tail_scores,
        tail_bias,
    ):
        super().__init__()
        self.packed_history = packed_history
        self.history_rows = int(history_rows)
        self.pooled_output = pooled_output
        self.partial_output = partial_output
        self.tail_values = tail_values
        self.tail_scores = tail_scores
        self.tail_bias = tail_bias

    def _on_place(self):
        expected_width = (
            self.packed_history.shape[0] * self.SHARD_WIDTH
            if self.packed_history.ndim == 5
            else -1
        )
        if (
            self.packed_history.dtype != torch.float32
            or self.packed_history.ndim != 5
            or tuple(self.packed_history.shape[2:])
            != (self.ROWS_PER_BLOCK, 2, self.SHARD_WIDTH)
            or not self.packed_history.is_contiguous()
        ):
            raise ValueError(
                "packed pool/RMS history must be contiguous FP32 "
                "[shards,blocks,8,2,128]"
            )
        self.shards, self.blocks = self.packed_history.shape[:2]
        if self.num_sms != self.shards:
            raise ValueError("packed pool/RMS uses one SM per width shard")
        if not 0 < self.history_rows <= self.blocks * self.ROWS_PER_BLOCK:
            raise ValueError("packed pool/RMS row count exceeds its blocks")
        for name, tensor in (
            ("tail values", self.tail_values),
            ("tail scores", self.tail_scores),
            ("tail bias", self.tail_bias),
        ):
            if (
                tensor.dtype != torch.float32
                or tensor.numel() != expected_width
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"packed pool/RMS {name} must be FP32 [width]"
                )
        if (
            self.pooled_output.dtype != torch.float32
            or tuple(self.pooled_output.shape)
            != (self.shards, self.SHARD_WIDTH)
            or not self.pooled_output.is_contiguous()
        ):
            raise ValueError(
                "packed pool/RMS output must be FP32 [shards,128]"
            )
        if (
            self.partial_output.dtype != torch.float32
            or tuple(self.partial_output.shape) != (self.shards,)
            or not self.partial_output.is_contiguous()
        ):
            raise ValueError(
                "packed pool/RMS partials must be FP32 [shards]"
            )

    def schedule(self, sm):
        if sm < 0:
            return []
        block_bytes = (
            self.ROWS_PER_BLOCK * 2 * self.SHARD_WIDTH
            * self.packed_history.element_size()
        )
        instructions = [
            Dsv4GatedPoolPacked8RmsPartial(self.history_rows)
        ]
        instructions += RepeatM.on(
            self.blocks,
            (
                TmaLoad1D(self.packed_history[sm, 0].reshape(-1)),
                block_bytes,
            ),
        )
        shard_start = sm * self.SHARD_WIDTH
        shard_end = shard_start + self.SHARD_WIDTH
        instructions.extend(
            (
                TmaLoad1D(self.tail_values[shard_start:shard_end]),
                TmaLoad1D(self.tail_scores[shard_start:shard_end]),
                TmaLoad1D(self.tail_bias[shard_start:shard_end]),
                TmaStore1D(self.pooled_output[sm]),
                StuStore1D(self.partial_output[sm : sm + 1]).bar(
                    self._bar("output")
                ),
            )
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4GatedPoolPacked8HistoryState(Schedule):
    """Pool immutable HCA history into per-dimension FP32 softmax state."""

    ROWS_PER_BLOCK = 8
    SHARD_WIDTH = 128
    STATE_COMPONENTS = 3

    def __init__(self, packed_history, history_rows, state_output):
        super().__init__()
        self.packed_history = packed_history
        self.history_rows = int(history_rows)
        self.state_output = state_output

    def _on_place(self):
        if (
            self.packed_history.dtype != torch.float32
            or self.packed_history.ndim != 5
            or tuple(self.packed_history.shape[2:])
            != (self.ROWS_PER_BLOCK, 2, self.SHARD_WIDTH)
            or not self.packed_history.is_contiguous()
        ):
            raise ValueError(
                "packed history-state input must be contiguous FP32 "
                "[shards,blocks,8,2,128]"
            )
        self.shards, self.blocks = self.packed_history.shape[:2]
        if self.num_sms != self.shards:
            raise ValueError("packed history state uses one SM per width shard")
        if not 0 < self.history_rows <= self.blocks * self.ROWS_PER_BLOCK:
            raise ValueError("packed history-state row count exceeds its blocks")
        if (
            self.state_output.dtype != torch.float32
            or tuple(self.state_output.shape)
            != (self.shards, self.STATE_COMPONENTS, self.SHARD_WIDTH)
            or not self.state_output.is_contiguous()
        ):
            raise ValueError(
                "packed history state must be contiguous FP32 [shards,3,128]"
            )

    def schedule(self, sm):
        if sm < 0:
            return []
        block_bytes = (
            self.ROWS_PER_BLOCK * 2 * self.SHARD_WIDTH
            * self.packed_history.element_size()
        )
        instructions = [
            Dsv4GatedPoolPacked8HistoryState(self.history_rows)
        ]
        instructions += RepeatM.on(
            self.blocks,
            (
                TmaLoad1D(self.packed_history[sm, 0].reshape(-1)),
                block_bytes,
            ),
        )
        instructions.append(
            TmaStore1D(self.state_output[sm].reshape(-1)).bar(
                self._bar("output")
            )
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4GatedPoolTailRmsPartial(Schedule):
    """Merge a projected HCA tail and emit pooled values plus RMS partials."""

    SHARD_WIDTH = 128
    STATE_COMPONENTS = 3

    def __init__(
        self,
        history_state,
        pooled_output,
        partial_output,
        *,
        tail_values,
        tail_scores,
        tail_bias,
    ):
        super().__init__()
        self.history_state = history_state
        self.pooled_output = pooled_output
        self.partial_output = partial_output
        self.tail_values = tail_values
        self.tail_scores = tail_scores
        self.tail_bias = tail_bias

    def _on_place(self):
        if (
            self.history_state.dtype != torch.float32
            or self.history_state.ndim != 3
            or tuple(self.history_state.shape[1:])
            != (self.STATE_COMPONENTS, self.SHARD_WIDTH)
            or not self.history_state.is_contiguous()
        ):
            raise ValueError(
                "tail merge history state must be contiguous FP32 [shards,3,128]"
            )
        self.shards = self.history_state.shape[0]
        if self.num_sms != self.shards:
            raise ValueError("tail merge uses one SM per width shard")
        expected_width = self.shards * self.SHARD_WIDTH
        for name, tensor in (
            ("tail values", self.tail_values),
            ("tail scores", self.tail_scores),
            ("tail bias", self.tail_bias),
        ):
            if (
                tensor.dtype != torch.float32
                or tensor.numel() != expected_width
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"tail merge {name} must be contiguous FP32 [width]"
                )
        if (
            self.pooled_output.dtype != torch.float32
            or tuple(self.pooled_output.shape)
            != (self.shards, self.SHARD_WIDTH)
            or not self.pooled_output.is_contiguous()
        ):
            raise ValueError(
                "tail merge pooled output must be FP32 [shards,128]"
            )
        if (
            self.partial_output.dtype != torch.float32
            or tuple(self.partial_output.shape) != (self.shards,)
            or not self.partial_output.is_contiguous()
        ):
            raise ValueError("tail merge RMS partials must be FP32 [shards]")

    def schedule(self, sm):
        if sm < 0:
            return []
        shard_start = sm * self.SHARD_WIDTH
        shard_end = shard_start + self.SHARD_WIDTH
        return [
            Dsv4GatedPoolTailRmsPartial(),
            TmaLoad1D(self.history_state[sm].reshape(-1)),
            TmaLoad1D(self.tail_values[shard_start:shard_end]),
            TmaLoad1D(self.tail_scores[shard_start:shard_end]),
            TmaLoad1D(self.tail_bias[shard_start:shard_end]),
            TmaStore1D(self.pooled_output[sm]),
            StuStore1D(self.partial_output[sm : sm + 1]).bar(
                self._bar("output")
            ),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Fp32RmsRopeShard128(Schedule):
    """Normalize four FP32 pooled shards and rotate the final shard."""

    SHARDS = 4
    SHARD_WIDTH = 128

    def __init__(
        self,
        input,
        partials,
        weight,
        table,
        output,
        *,
        epsilon: float,
        fixed_table_id=None,
    ):
        super().__init__()
        self.input = input
        self.partials = partials
        self.weight = weight
        self.table = table
        self.output = output
        self.epsilon = epsilon
        self.fixed_table_id = fixed_table_id

    def _on_place(self):
        if self.num_sms != self.SHARDS:
            raise ValueError("FP32 pooled RMS/RoPE requires four SMs")
        if (
            self.input.dtype != torch.float32
            or tuple(self.input.shape) != (self.SHARDS, self.SHARD_WIDTH)
            or not self.input.is_contiguous()
        ):
            raise ValueError("FP32 pooled input must be [4,128]")
        if (
            self.partials.dtype != torch.float32
            or tuple(self.partials.shape) != (self.SHARDS,)
            or not self.partials.is_contiguous()
        ):
            raise ValueError("FP32 pooled RMS partials must be [4]")
        if (
            self.weight.dtype != torch.bfloat16
            or self.weight.numel() != self.SHARDS * self.SHARD_WIDTH
            or not self.weight.is_contiguous()
        ):
            raise ValueError("FP32 pooled RMS weight must be BF16 [512]")
        if (
            self.table.dtype != torch.float32
            or tuple(self.table.shape) != (32, 2)
        ):
            raise ValueError("FP32 pooled RMS/RoPE table must be FP32 [32,2]")
        if (
            self.output.dtype != torch.bfloat16
            or self.output.numel() != self.SHARDS * self.SHARD_WIDTH
            or not self.output.is_contiguous()
        ):
            raise ValueError("FP32 pooled RMS/RoPE output must be BF16 [512]")
        if self.epsilon <= 0:
            raise ValueError("FP32 pooled RMS/RoPE epsilon must be positive")
        if self.fixed_table_id is not None and not 0 <= self.fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")

    def schedule(self, sm):
        if sm < 0:
            return []
        start = sm * self.SHARD_WIDTH
        stop = start + self.SHARD_WIDTH
        instructions = [
            Dsv4Fp32RmsRopeShard128(
                sm,
                epsilon=self.epsilon,
                fixed_table_id=self.fixed_table_id,
            ),
            TmaLoad1D(self.input[sm]),
            TmaLoad1D(self.partials),
            TmaLoad1D(self.weight[start:stop]),
        ]
        if self.fixed_table_id is None:
            instructions.append(TmaLoad1D(self.table))
        instructions.append(
            TmaStore1D(self.output.reshape(-1)[start:stop]).bar(
                self._bar("output")
            )
        )
        return instructions

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


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


class SchedDsv4HcHeadRms(Schedule):
    def __init__(
        self,
        residual,
        mixes,
        scale,
        base,
        rms_weight,
        output,
        epsilon=1.0e-6,
        rms_epsilon=1.0e-6,
    ):
        super().__init__()
        self.residual = residual
        self.mixes = mixes
        self.scale = scale
        self.base = base
        self.rms_weight = rms_weight
        self.output = output
        self.epsilon = epsilon
        self.rms_epsilon = rms_epsilon

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("fused DeepSeek mHC head/RMS uses exactly one SM")
        if (
            self.residual.dtype != torch.bfloat16
            or tuple(self.residual.shape) != (4, 4096)
        ):
            raise ValueError("mHC head residual must be BF16 [4,4096]")
        if self.mixes.dtype != torch.float32 or self.mixes.numel() != 4:
            raise ValueError("mHC head mixes must contain four FP32 values")
        if self.scale.dtype != torch.float32 or self.scale.numel() != 1:
            raise ValueError("mHC head scale must contain one FP32 value")
        if self.base.dtype != torch.float32 or self.base.numel() != 4:
            raise ValueError("mHC head base must contain four FP32 values")
        if (
            self.rms_weight.dtype != torch.bfloat16
            or self.rms_weight.numel() != 4096
        ):
            raise ValueError("mHC head RMS weight must contain 4096 BF16 values")
        if self.output.dtype != torch.bfloat16 or self.output.numel() != 4096:
            raise ValueError("mHC head/RMS output must contain 4096 BF16 values")

    def schedule(self, sm):
        if sm != 0:
            return []
        return [
            Dsv4HcHeadRms(self.epsilon, self.rms_epsilon),
            TmaLoad1D(self.residual),
            TmaLoad1D(self.mixes),
            _shared_load_1d(self.scale),
            TmaLoad1D(self.base),
            TmaLoad1D(self.rms_weight),
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
                (Dsv4SiluClampMul(1, shard_width, self.swiglu_limit)
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
                    Dsv4SiluClampMul(1, width, self.swiglu_limit)
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

    def __init__(
        self,
        logits: torch.Tensor,
        partials: torch.Tensor,
        *,
        index_base: int = 0,
    ):
        super().__init__()
        self.logits = logits
        self.partials = partials
        self.index_base = int(index_base)

    def _on_place(self):
        if (self.logits.dtype != torch.bfloat16 or self.logits.ndim != 1 or
                not self.logits.is_contiguous()):
            raise ValueError("shared argmax logits must be contiguous rank-1 BF16")
        if self.logits.numel() % 8:
            raise ValueError("shared argmax logits must contain a multiple of 8 values")
        if self.num_sms > self.logits.numel() // 8:
            raise ValueError("shared argmax requires at least eight logits per SM")
        if not 0 <= self.index_base <= 0xFFFFFFFF - self.logits.numel():
            raise ValueError("shared argmax absolute index range must fit in uint32")
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
            ArgmaxSmemPartialBf16(row_count, self.index_base + row_start),
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
