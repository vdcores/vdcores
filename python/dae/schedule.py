import copy as pycopy
import warnings

from .runtime import *
from .launcher import *

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


class SchedNvfp4Gemv(Schedule):
    """Shard a ModelOpt NVFP4 matrix-vector multiply across resident SMs."""

    def __init__(self, weight, weight_scale, activation, activation_scale,
                 alpha, output, base_raw_slot=24):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.activation = activation
        self.activation_scale = activation_scale
        self.alpha = alpha
        self.output = output
        self.base_raw_slot = base_raw_slot

    def _expected_output_elements(self, rows):
        return rows

    def _on_place(self):
        if self.weight.dtype != torch.uint8 or self.weight.ndim != 2:
            raise ValueError("NVFP4 weight must be a rank-2 packed uint8 tensor")
        rows, packed_k = self.weight.shape
        self.rows = rows
        self.k = packed_k * 2
        if self.k % 32:
            raise ValueError("NVFP4 K must be divisible by 32")
        if tuple(self.weight_scale.shape) != (rows, self.k // 16):
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
        expected_output = self._expected_output_elements(rows)
        if (self.output.dtype != torch.bfloat16 or
                self.output.numel() != expected_output):
            raise ValueError(
                f"output must contain {expected_output} BF16 values"
            )
        if self.num_sms <= 0 or self.num_sms > rows:
            raise ValueError("NVFP4 GEMV requires 1 <= num_sms <= M")
        if self.base_raw_slot < config.num_slots:
            raise ValueError("base_raw_slot must name a special slot")
        if (self.base_raw_slot + 5 >= config.num_slots + config.num_special_slots or
                self.base_raw_slot + 5 >= 32):
            raise ValueError("NVFP4 GEMV needs six C2M-addressable consecutive special slots")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        slot = self.base_raw_slot
        return [
            Nvfp4GemvSm100(row_count, self.k),
            RawAddress(self.weight[row_start], slot),
            RawAddress(self.weight_scale[row_start], slot + 1),
            RawAddress(self.activation.reshape(-1), slot + 2),
            RawAddress(self.activation_scale.reshape(-1), slot + 3),
            RawAddress(self.alpha.reshape(-1), slot + 4),
            RawAddress(self.output[row_start], slot + 5)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedNvfp4GemvUmma(SchedNvfp4Gemv):
    """Map one native block-scaled UMMA tile to each resident SM."""

    def __init__(self, weight, weight_scale, activation, activation_scale,
                 alpha, output, base_raw_slot=24, output_columns=1):
        if output_columns not in (1, 8):
            raise ValueError("NVFP4 UMMA output_columns must be 1 or 8")
        self.output_columns = output_columns
        super().__init__(
            weight, weight_scale, activation, activation_scale,
            alpha, output, base_raw_slot
        )

    def _expected_output_elements(self, rows):
        return rows * self.output_columns

    def _on_place(self):
        super()._on_place()
        rows_per_sm = (self.rows + self.num_sms - 1) // self.num_sms
        if rows_per_sm > 128:
            raise ValueError("NVFP4 UMMA supports at most 128 output rows per SM")
        if self.k % 256:
            raise ValueError("NVFP4 UMMA K must be divisible by 256")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        slot = self.base_raw_slot
        return [
            Nvfp4GemvUmmaSm100(
                row_count, self.k, self.output_columns
            ),
            RawAddress(self.weight[row_start], slot),
            RawAddress(self.weight_scale[row_start], slot + 1),
            RawAddress(self.activation.reshape(-1), slot + 2),
            RawAddress(self.activation_scale.reshape(-1), slot + 3),
            RawAddress(self.alpha.reshape(-1), slot + 4),
            RawAddress(
                self.output[row_start * self.output_columns], slot + 5
            )
                .bar(self._bar("output")).writeback(),
        ]


class SchedFp8Block128Gemv(Schedule):
    """Shard an E4M3/UE8M0 checkpoint GEMV across resident SMs."""

    def __init__(self, weight, weight_scale, activation, activation_scale,
                 output, base_raw_slot=24):
        super().__init__()
        self.weight = weight
        self.weight_scale = weight_scale
        self.activation = activation
        self.activation_scale = activation_scale
        self.output = output
        self.base_raw_slot = base_raw_slot

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
        if self.base_raw_slot < config.num_slots or self.base_raw_slot + 4 >= 32:
            raise ValueError("FP8 GEMV needs five C2M-addressable special slots")

    def schedule(self, sm):
        if sm < 0:
            return []
        rows_per_sm, extra = divmod(self.rows, self.num_sms)
        row_start = sm * rows_per_sm + min(sm, extra)
        row_count = rows_per_sm + (1 if sm < extra else 0)
        slot = self.base_raw_slot
        return [
            Fp8Block128GemvSm100(row_count, self.k, row_start % 128),
            RawAddress(self.weight[row_start], slot),
            RawAddress(self.weight_scale[row_start // 128], slot + 1),
            RawAddress(self.activation.reshape(-1), slot + 2),
            RawAddress(self.activation_scale.reshape(-1), slot + 3),
            RawAddress(self.output[row_start], slot + 4)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4Rope512_64(Schedule):
    def __init__(self, input, table, output, inverse=False, base_raw_slot=24):
        super().__init__()
        self.input = input
        self.table = table
        self.output = output
        self.inverse = inverse
        self.base_raw_slot = base_raw_slot

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek partial RoPE currently uses exactly one SM")
        if (self.input.dtype != torch.bfloat16 or self.input.ndim != 2 or
                self.input.shape[1] != 512):
            raise ValueError("DeepSeek RoPE input must be BF16 [rows,512]")
        if self.output.dtype != torch.bfloat16 or self.output.shape != self.input.shape:
            raise ValueError("DeepSeek RoPE output must match the input")
        if (self.table.dtype != torch.float32 or
                tuple(self.table.shape) != (32, 2)):
            raise ValueError("DeepSeek RoPE table must be FP32 [32,2]")

    def schedule(self, sm):
        if sm != 0:
            return []
        slot = self.base_raw_slot
        return [
            Dsv4Rope512_64(self.input.shape[0], self.inverse),
            RawAddress(self.input, slot),
            RawAddress(self.table, slot + 1),
            RawAddress(self.output, slot + 2)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4SparseAttention512(Schedule):
    def __init__(self, q, kv, indices, sink, output, base_raw_slot=24):
        super().__init__()
        self.q = q
        self.kv = kv
        self.indices = indices
        self.sink = sink
        self.output = output
        self.base_raw_slot = base_raw_slot

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

    def schedule(self, sm):
        if sm < 0:
            return []
        slot = self.base_raw_slot
        return [
            Dsv4SparseAttention512(sm, self.indices.numel()),
            RawAddress(self.q, slot),
            RawAddress(self.kv, slot + 1),
            RawAddress(self.indices, slot + 2),
            RawAddress(self.sink, slot + 3),
            RawAddress(self.output, slot + 4)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4RouteTop6(Schedule):
    def __init__(self, logits, bias, hash_indices, output_indices,
                 output_weights, hash_routing=False, route_scale=1.5,
                 base_raw_slot=24):
        super().__init__()
        self.logits = logits
        self.bias = bias
        self.hash_indices = hash_indices
        self.output_indices = output_indices
        self.output_weights = output_weights
        self.hash_routing = hash_routing
        self.route_scale = route_scale
        self.base_raw_slot = base_raw_slot

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek routing currently uses exactly one SM")
        if self.logits.dtype != torch.bfloat16 or self.logits.numel() != 256:
            raise ValueError("DeepSeek routing logits must contain 256 BF16 values")
        if self.bias.dtype != torch.float32 or self.bias.numel() != 256:
            raise ValueError("DeepSeek routing bias must contain 256 FP32 values")
        if self.hash_indices.dtype != torch.int32 or self.hash_indices.numel() != 6:
            raise ValueError("DeepSeek hash routing must provide six int32 ids")
        if self.output_indices.dtype != torch.int32 or self.output_indices.numel() != 6:
            raise ValueError("DeepSeek route output must contain six int32 ids")
        if self.output_weights.dtype != torch.float32 or self.output_weights.numel() != 6:
            raise ValueError("DeepSeek route output must contain six FP32 weights")

    def schedule(self, sm):
        if sm != 0:
            return []
        slot = self.base_raw_slot
        return [
            Dsv4RouteTop6(self.hash_routing, self.route_scale),
            RawAddress(self.logits, slot),
            RawAddress(self.bias, slot + 1),
            RawAddress(self.hash_indices, slot + 2),
            RawAddress(self.output_indices, slot + 3).writeback(),
            RawAddress(self.output_weights, slot + 4)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4ExpertReduce(Schedule):
    def __init__(self, routed, weights, shared, output, base_raw_slot=24):
        super().__init__()
        self.routed = routed
        self.weights = weights
        self.shared = shared
        self.output = output
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4ExpertReduce(),
            RawAddress(self.routed, slot),
            RawAddress(self.weights, slot + 1),
            RawAddress(self.shared, slot + 2),
            RawAddress(self.output, slot + 3)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4Fp32Bf16Gemv(Schedule):
    def __init__(self, weight, input, output, base_raw_slot=24):
        super().__init__()
        self.weight = weight
        self.input = input
        self.output = output
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4Fp32Bf16Gemv(row_count, self.k),
            RawAddress(self.weight[row_start], slot),
            RawAddress(self.input, slot + 1),
            RawAddress(self.output[row_start], slot + 2)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4HcPre(Schedule):
    def __init__(self, residual, mixes, scale, base, output, post, comb,
                 sinkhorn_iters=20, epsilon=1.0e-6, base_raw_slot=24):
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
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4HcPre(self.sinkhorn_iters, self.epsilon),
            RawAddress(self.residual, slot),
            RawAddress(self.mixes, slot + 1),
            RawAddress(self.scale, slot + 2),
            RawAddress(self.base, slot + 3),
            RawAddress(self.output, slot + 4)
                .bar(self._bar("output")).writeback(),
            RawAddress(self.post, slot + 5).writeback(),
            RawAddress(self.comb, slot + 6).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4HcPost(Schedule):
    def __init__(self, branch, residual, post, comb, output, base_raw_slot=24):
        super().__init__()
        self.branch = branch
        self.residual = residual
        self.post = post
        self.comb = comb
        self.output = output
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4HcPost(),
            RawAddress(self.branch, slot),
            RawAddress(self.residual, slot + 1),
            RawAddress(self.post, slot + 2),
            RawAddress(self.comb, slot + 3),
            RawAddress(self.output, slot + 4)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4Hadamard(Schedule):
    def __init__(self, input, output, base_raw_slot=24):
        super().__init__()
        self.input = input
        self.output = output
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4Hadamard(sm, self.width),
            RawAddress(self.input, slot),
            RawAddress(self.output, slot + 1)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4GatedPool(Schedule):
    def __init__(self, values, scores, output, base_raw_slot=24):
        super().__init__()
        self.values = values
        self.scores = scores
        self.output = output
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4GatedPool(self.pool_rows, self.width),
            RawAddress(self.values, slot),
            RawAddress(self.scores, slot + 1),
            RawAddress(self.output, slot + 2)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4IndexScore(Schedule):
    def __init__(self, q, kv, head_weights, output, base_raw_slot=24):
        super().__init__()
        self.q = q
        self.kv = kv
        self.head_weights = head_weights
        self.output = output
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4IndexScore(row_count),
            RawAddress(self.q, slot),
            RawAddress(self.kv[row_start], slot + 1),
            RawAddress(self.head_weights, slot + 2),
            RawAddress(self.output[row_start], slot + 3)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, self.num_sms)


class SchedDsv4TopK512(Schedule):
    def __init__(self, scores, output, index_offset=0, base_raw_slot=24):
        super().__init__()
        self.scores = scores
        self.output = output
        self.index_offset = index_offset
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4TopK512(
                self.scores.numel(), self.output.numel(), self.index_offset
            ),
            RawAddress(self.scores, slot),
            RawAddress(self.output, slot + 1)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4HcHead(Schedule):
    def __init__(self, residual, mixes, scale, base, output,
                 epsilon=1.0e-6, base_raw_slot=24):
        super().__init__()
        self.residual = residual
        self.mixes = mixes
        self.scale = scale
        self.base = base
        self.output = output
        self.epsilon = epsilon
        self.base_raw_slot = base_raw_slot

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
        slot = self.base_raw_slot
        return [
            Dsv4HcHead(self.epsilon),
            RawAddress(self.residual, slot),
            RawAddress(self.mixes, slot + 1),
            RawAddress(self.scale, slot + 2),
            RawAddress(self.base, slot + 3),
            RawAddress(self.output, slot + 4)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4Fp8Quant128(Schedule):
    def __init__(self, input, output, scale, base_raw_slot=24):
        super().__init__()
        self.input = input
        self.output = output
        self.scale = scale
        self.base_raw_slot = base_raw_slot

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek FP8 activation quantization uses one SM")
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("DeepSeek FP8 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % 128:
            raise ValueError("DeepSeek FP8 quant K must be divisible by 128")
        if (self.output.dtype != torch.float8_e4m3fn or
                self.output.shape != self.input.shape):
            raise ValueError("DeepSeek FP8 quant output must be E4M3 [K]")
        if (self.scale.dtype != torch.float8_e8m0fnu or
                self.scale.numel() != self.k // 128):
            raise ValueError("DeepSeek FP8 quant scale must be UE8M0 [K/128]")

    def schedule(self, sm):
        if sm != 0:
            return []
        slot = self.base_raw_slot
        return [
            Dsv4Fp8Quant128(self.k),
            RawAddress(self.input, slot),
            RawAddress(self.output, slot + 1).writeback(),
            RawAddress(self.scale, slot + 2)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


class SchedDsv4Nvfp4Quant16(Schedule):
    def __init__(self, input, global_scale, output, scale, base_raw_slot=24):
        super().__init__()
        self.input = input
        self.global_scale = global_scale
        self.output = output
        self.scale = scale
        self.base_raw_slot = base_raw_slot

    def _on_place(self):
        if self.num_sms != 1:
            raise ValueError("DeepSeek NVFP4 activation quantization uses one SM")
        if self.input.dtype != torch.bfloat16 or self.input.ndim != 1:
            raise ValueError("DeepSeek NVFP4 quant input must be a BF16 vector")
        self.k = self.input.numel()
        if self.k % 16:
            raise ValueError("DeepSeek NVFP4 quant K must be divisible by 16")
        if self.global_scale.dtype != torch.float32 or self.global_scale.numel() != 1:
            raise ValueError("DeepSeek NVFP4 global scale must be scalar FP32")
        if self.output.dtype != torch.uint8 or self.output.numel() != self.k // 2:
            raise ValueError("DeepSeek NVFP4 quant output must be packed uint8 [K/2]")
        if (self.scale.dtype != torch.float8_e4m3fn or
                self.scale.numel() != self.k // 16):
            raise ValueError("DeepSeek NVFP4 quant scale must be E4M3 [K/16]")

    def schedule(self, sm):
        if sm != 0:
            return []
        slot = self.base_raw_slot
        return [
            Dsv4Nvfp4Quant16(self.k),
            RawAddress(self.input, slot),
            RawAddress(self.global_scale.reshape(-1), slot + 1),
            RawAddress(self.output, slot + 2).writeback(),
            RawAddress(self.scale, slot + 3)
                .bar(self._bar("output")).writeback(),
        ]

    def bar_release_count(self, role: str):
        if role != "output":
            return 0
        return self._bar_release_if_present(role, 1)


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
