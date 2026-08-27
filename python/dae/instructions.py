import copy

import torch

from .instruction_utils import (
    compute_operator_name as format_compute_operator_name,
    decode_opcode,
    dedcode_opcode,
    encode_bfloat16_u16,
    encode_compute_instruction_tensor,
    normalize_compute_opcode_reference,
    resolve_compute_opcode_value,
)
from .op_families import ComputeOpFamilyRef, family_ref
from .runtime import config, opcode
from .tma_utils import (
    Major,
    addr2cords,
    build_tma_1d,
    build_tma_rowmajor_2d,
    build_tma_mxfp4,
    build_tma_mxfp4_kmajor,
    build_tma_mxfp4_k512,
    build_tma_wgmma_kmajor,
    build_tma_wgmma_mnmajor_m128n8,
    build_tma_wgmma_mnmajor,
    build_tma_wgmma_tile_major,
    bytes2slots,
    cord_func_2d_kmajor,
    cord_func_m128n8_output,
    cord_func_2d_mnmajor,
    cord_func_2d_tile_major,
    cord_func_tma_1d,
    cord_func_rowmajor_2d,
    cord_func_mxfp4,
    cord_func_mxfp4_kmajor,
    cord_func_mxfp4_k512,
    cords2addr,
    get_tensor_address,
)


class Instruction:
    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError()


class ComputeInstruction(Instruction):
    def __init__(self, opcode: int | str | ComputeOpFamilyRef, args: list[int]):
        self.opcode, self.op_family_name = normalize_compute_opcode_reference(opcode)
        self.args = args

    def opcode_value(self) -> int:
        return resolve_compute_opcode_value(self.opcode, self.op_family_name)

    def compute_operator_name(self) -> str:
        return format_compute_operator_name(self.opcode, self.op_family_name)

    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        return encode_compute_instruction_tensor(self.opcode, self.op_family_name, self.args, tensor)

    def __repr__(self):
        return f"ComputeInstruction(opcode={self.compute_operator_name()}, args={self.args})"


class TerminateC(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_TERMINATEC, args=[])


class Gemv_M64N8(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=256, BLOAD=4, RESIDUAL=residual), args=[kTiles, nprefeth])


class Gemv_M64N8UpSiLU(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4

    def __init__(self, kTiles: int):
        super().__init__(
            opcode=opcode.OP_GEMV_SM100_M64N8_UP_SILU,
            args=[kTiles],
        )


class Gemv_M64N8IssuerOnly(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        if residual:
            raise ValueError("issuer-only M64N8 does not support residual input")
        super().__init__(
            opcode=opcode.OP_GEMV_SM100_M64N8_ISSUER_ONLY,
            args=[kTiles, nprefeth],
        )


class Nvfp4GemvSm100(ComputeInstruction):
    """ModelOpt-compatible W4A4 decode GEMV over one output-row shard."""

    def __init__(
        self,
        rows: int,
        k: int,
        retain_activation: bool = False,
    ):
        if rows <= 0 or rows > 0xFFFF:
            raise ValueError("NVFP4 GEMV rows must fit in a positive uint16")
        if k <= 0 or k > 0xFFFF or k % 32:
            raise ValueError("NVFP4 GEMV K must be a positive uint16 multiple of 32")
        super().__init__(
            opcode=opcode.OP_NVFP4_GEMV_SM100,
            args=[rows, k, int(retain_activation)],
        )


class Nvfp4GemvUmmaSm100(ComputeInstruction):
    """Native SM100 block-scaled UMMA over one <=128-row output tile."""

    def __init__(self, rows: int, k: int, output_columns: int = 1):
        if not 1 <= rows <= 128:
            raise ValueError("NVFP4 UMMA rows must be in [1, 128]")
        if k <= 0 or k > 0xFFFF or k % 256:
            raise ValueError("NVFP4 UMMA K must be a uint16 multiple of 256")
        if output_columns not in (1, 8):
            raise ValueError("NVFP4 UMMA output_columns must be 1 or 8")
        super().__init__(
            opcode=opcode.OP_NVFP4_GEMV_UMMA_SM100,
            args=[rows, k, output_columns],
        )


class Nvfp4GemvUmmaStreamSm100(ComputeInstruction):
    """Stream pre-swizzled K256 operands into native SM100 block-scale UMMA."""

    def __init__(
        self,
        k_tiles: int,
        *,
        retain_activation: bool = False,
        bulk_activation: bool = False,
        swiglu_limit: float | None = None,
    ):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("NVFP4 streaming UMMA K-tile count must fit uint16")
        if retain_activation and not bulk_activation:
            raise ValueError("retained native activation must use one bulk allocation")
        swiglu_limit_q4 = 0
        if swiglu_limit is not None:
            swiglu_limit_q4 = round(float(swiglu_limit) * 16.0)
            if not 0 < swiglu_limit_q4 <= 0x7FFF:
                raise ValueError(
                    "bounded SwiGLU limit must fit positive unsigned Q4"
                )
        super().__init__(
            opcode=opcode.OP_NVFP4_GEMV_UMMA_STREAM_SM100,
            args=[
                k_tiles,
                int(retain_activation) | (swiglu_limit_q4 << 1),
                int(bulk_activation),
            ],
        )


class Nvfp4GemvUmmaPipelineSm100(Nvfp4GemvUmmaStreamSm100):
    """Pipeline native K256 loads, scale staging, and UMMA issue."""

    def __init__(
        self,
        k_tiles: int,
        *,
        retain_activation: bool = False,
        activation_tiles_per_load: int = 4,
    ):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("NVFP4 pipelined UMMA K-tile count must fit uint16")
        if (
            activation_tiles_per_load <= 0
            or activation_tiles_per_load > k_tiles
        ):
            raise ValueError(
                "NVFP4 activation tiles per load must be in [1,K tiles]"
            )
        if retain_activation and activation_tiles_per_load != k_tiles:
            raise ValueError("retained activation must cover the full K operand")
        ComputeInstruction.__init__(
            self,
            opcode=opcode.OP_NVFP4_GEMV_UMMA_PIPELINE_SM100,
            args=[k_tiles, int(retain_activation), activation_tiles_per_load],
        )


class Nvfp4GemvUmmaFp32Sm100(ComputeInstruction):
    """Emit one route-scaled FP32 M128 result for the supplied K stream."""

    def __init__(
        self,
        k_tiles: int,
        *,
        retain_activation: bool = False,
        bulk_activation: bool = True,
    ):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("NVFP4 FP32 UMMA K-tile count must fit uint16")
        if retain_activation and not bulk_activation:
            raise ValueError("retained FP32 activation must use one bulk load")
        super().__init__(
            opcode=opcode.OP_NVFP4_GEMV_UMMA_FP32_SM100,
            args=[k_tiles, int(retain_activation), int(bulk_activation)],
        )


class Nvfp4GemvUmmaPipelineFp32Sm100(Nvfp4GemvUmmaPipelineSm100):
    """Pipeline native K256 operands and emit one FP32 M128 tile."""

    def __init__(
        self,
        k_tiles: int,
        *,
        retain_activation: bool = False,
        activation_tiles_per_load: int = 4,
    ):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("NVFP4 pipelined FP32 K-tile count must fit uint16")
        if (
            activation_tiles_per_load <= 0
            or activation_tiles_per_load > k_tiles
        ):
            raise ValueError(
                "NVFP4 activation tiles per load must be in [1,K tiles]"
            )
        if retain_activation and activation_tiles_per_load != k_tiles:
            raise ValueError("retained activation must cover the full K operand")
        ComputeInstruction.__init__(
            self,
            opcode=opcode.OP_NVFP4_GEMV_UMMA_PIPELINE_FP32_SM100,
            args=[
                k_tiles,
                int(retain_activation),
                activation_tiles_per_load,
            ],
        )


class Nvfp4GemvUmmaPipelineFp32Group2Sm100(
    Nvfp4GemvUmmaPipelineFp32Sm100
):
    """Share each activation K tile across two FP32 M128 outputs."""

    def __init__(
        self,
        k_tiles: int,
        *,
        activation_tiles_per_load: int = 4,
    ):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("paired NVFP4 K-tile count must fit uint16")
        if not 0 < activation_tiles_per_load <= k_tiles:
            raise ValueError(
                "paired NVFP4 activation tiles per load must be in [1,K tiles]"
            )
        ComputeInstruction.__init__(
            self,
            opcode=(
                opcode.OP_NVFP4_GEMV_UMMA_PIPELINE_FP32_GROUP2_SM100
            ),
            args=[
                k_tiles,
                0,
                activation_tiles_per_load,
            ],
        )


class Nvfp4GemvUmmaK512Fp32Sm100(ComputeInstruction):
    """K512 native UMMA with allocator-free, compute-warp scale staging."""

    def __init__(
        self,
        k_tiles: int,
        *,
        scale_stages: int = 2,
        weight_tiles_per_load: int = 1,
        retain_activation: bool = False,
    ):
        if k_tiles <= 0 or k_tiles > 8:
            raise ValueError("K512 NVFP4 tile count must be in [1,8]")
        if not 1 <= scale_stages <= config.nvfp4_scale_copy_stages:
            raise ValueError(
                "K512 NVFP4 scale stages must be in [1,"
                f"{config.nvfp4_scale_copy_stages}]"
            )
        if weight_tiles_per_load not in (1, 2):
            raise ValueError("K512 NVFP4 weight tiles per load must be one or two")
        if k_tiles % weight_tiles_per_load:
            raise ValueError("K512 NVFP4 weight load must divide the K tiles")
        super().__init__(
            opcode=opcode.OP_NVFP4_GEMV_UMMA_K512_FP32_SM100,
            args=[
                k_tiles,
                scale_stages,
                weight_tiles_per_load | (int(retain_activation) << 8),
            ],
        )


class Mxfp4Mxfp8GemvUmmaK512TmaScaleFp32Sm100(ComputeInstruction):
    """Native W4A8 task with K512 weights and compile-time activation BLOAD."""

    def __init__(self, activation_tiles_per_load: int = 4):
        if activation_tiles_per_load not in (1, 2, 4, 8):
            raise ValueError(
                "MXFP4/MXFP8 activation tiles per load must be 1, 2, 4, or 8"
            )
        super().__init__(
            opcode=family_ref(
                "MXFP4_MXFP8_GEMV_UMMA_TMA_SCALE_FP32_SM100",
                K=512,
                BLOAD=activation_tiles_per_load,
            ),
            args=[],
        )


class Mxfp4Mxfp8GemvUmmaK512MetaScaleFp32Sm100(ComputeInstruction):
    """Native W4A8 K4096 task with compute-side metadata scale loads."""

    def __init__(
        self,
        metadata_address: int,
        activation_tiles_per_load: int = 4,
    ):
        if activation_tiles_per_load not in (1, 2, 4, 8):
            raise ValueError(
                "MXFP4/MXFP8 activation tiles per load must be 1, 2, 4, or 8"
            )
        if metadata_address < 0 or metadata_address >= 1 << 48:
            raise ValueError("MXFP4/MXFP8 metadata pointer must fit in 48 bits")
        super().__init__(
            opcode=family_ref(
                "MXFP4_MXFP8_GEMV_UMMA_META_SCALE_FP32_SM100",
                K=512,
                BLOAD=activation_tiles_per_load,
            ),
            args=[
                metadata_address & 0xFFFF,
                (metadata_address >> 16) & 0xFFFF,
                (metadata_address >> 32) & 0xFFFF,
            ],
        )


class Mxfp4Mxfp8GateUpSiluFixedRingSm100(ComputeInstruction):
    """Task-local K128- or K512-member gate/up/SiLU weight pipeline."""

    def __init__(
        self,
        metadata_address: int,
        *,
        tile_k: int = 512,
        stages: int | None = None,
    ):
        if tile_k not in (128, 512):
            raise ValueError("fused MXFP4/MXFP8 fixed ring requires K128 or K512")
        if stages is None:
            stages = 10 if tile_k == 128 else 2
        if (tile_k, stages) not in (
            (128, 10), (128, 11), (512, 2), (512, 3)
        ):
            raise ValueError("invalid fused MXFP4/MXFP8 fixed-ring shape")
        if metadata_address < 0 or metadata_address >= 1 << 48:
            raise ValueError("fixed-ring MXFP4/MXFP8 metadata pointer must fit in 48 bits")
        super().__init__(
            opcode=family_ref(
                "MXFP4_MXFP8_GATE_UP_SILU_FIXED_RING_SM100",
                K=tile_k,
                STAGES=stages,
            ),
            args=[
                metadata_address & 0xFFFF,
                (metadata_address >> 16) & 0xFFFF,
                (metadata_address >> 32) & 0xFFFF,
            ],
        )


class Mxfp4Mxfp8DownFixedRingSm100(ComputeInstruction):
    """K2048 down projection consuming native fused-Linear1 records."""

    def __init__(self, metadata_address: int):
        if metadata_address < 0 or metadata_address >= 1 << 48:
            raise ValueError("fixed-ring MXFP4/MXFP8 down metadata pointer must fit in 48 bits")
        super().__init__(
            opcode=opcode.OP_MXFP4_MXFP8_DOWN_FIXED_RING_SM100,
            args=[
                metadata_address & 0xFFFF,
                (metadata_address >> 16) & 0xFFFF,
                (metadata_address >> 32) & 0xFFFF,
            ],
        )


class Mxfp4Mxfp8ResidentFfnSm100(ComputeInstruction):
    """One dedicated Linear-1 plus Down resident compute plan."""

    def __init__(self, plan_address: int):
        if plan_address < 0 or plan_address >= 1 << 48:
            raise ValueError("resident FFN plan pointer must fit in 48 bits")
        super().__init__(
            opcode=opcode.OP_MXFP4_MXFP8_RESIDENT_FFN_SM100,
            args=[
                plan_address & 0xFFFF,
                (plan_address >> 16) & 0xFFFF,
                (plan_address >> 32) & 0xFFFF,
            ],
        )


class Mxfp4Mxfp8RoutedResidentFfnSm100(ComputeInstruction):
    """Resident MX FFN with route-selected weights and N-major output."""

    def __init__(self, plan_address: int):
        if plan_address < 0 or plan_address >= 1 << 48:
            raise ValueError("routed resident FFN plan pointer must fit in 48 bits")
        super().__init__(
            opcode=opcode.OP_MXFP4_MXFP8_ROUTED_RESIDENT_FFN_SM100,
            args=[
                plan_address & 0xFFFF,
                (plan_address >> 16) & 0xFFFF,
                (plan_address >> 32) & 0xFFFF,
            ],
        )


class Nvfp4UmmaPrepackSm100(ComputeInstruction):
    """Prepack direct-TMA NVFP4 data and raw scales into one native tile."""

    WEIGHT = 0
    ACTIVATION = 1

    def __init__(self, kind: int, k_tiles: int):
        if kind not in (self.WEIGHT, self.ACTIVATION):
            raise ValueError("NVFP4 UMMA prepack kind must be weight or activation")
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("NVFP4 UMMA prepack K-tile count must fit uint16")
        super().__init__(
            opcode=opcode.OP_NVFP4_UMMA_PREPACK_SM100,
            args=[kind, k_tiles],
        )


class Fp8Block128GemvSm100(ComputeInstruction):
    """E4M3/UE8M0 block-128 decode GEMV over one output-row shard."""

    def __init__(
        self,
        rows: int,
        k: int,
        row_in_scale_block: int,
        *,
        swiglu_limit: float | None = None,
    ):
        if rows <= 0 or rows > 0xFFFF:
            raise ValueError("FP8 GEMV rows must fit in a positive uint16")
        if k <= 0 or k > 0xFFFF or k % 128:
            raise ValueError("FP8 GEMV K must be a positive uint16 multiple of 128")
        if not 0 <= row_in_scale_block < 128:
            raise ValueError("FP8 GEMV scale-block row offset must be in [0,128)")
        swiglu_limit_q4 = 0
        if swiglu_limit is not None:
            swiglu_limit_q4 = round(float(swiglu_limit) * 16.0)
            if not 0 < swiglu_limit_q4 < (1 << 9):
                raise ValueError(
                    "block-128 SwiGLU limit must fit positive unsigned Q4"
                )
        super().__init__(
            opcode=opcode.OP_FP8_BLOCK128_GEMV_SM100,
            args=[rows, k, row_in_scale_block | (swiglu_limit_q4 << 7)],
        )


class Fp8Block128GemvBf16Sm100(ComputeInstruction):
    """Quantize one BF16 activation in shared memory, then run FP8 GEMV."""

    def __init__(self, rows: int, k: int, row_in_scale_block: int):
        if rows <= 0 or rows > 0xFFFF:
            raise ValueError("fused FP8 GEMV rows must fit in a positive uint16")
        if k <= 0 or k > 0xFFFF or k % 128:
            raise ValueError(
                "fused FP8 GEMV K must be a positive uint16 multiple of 128"
            )
        if not 0 <= row_in_scale_block < 128:
            raise ValueError(
                "fused FP8 GEMV scale-block row offset must be in [0,128)"
            )
        super().__init__(
            opcode=opcode.OP_FP8_BLOCK128_GEMV_BF16_SM100,
            args=[rows, k, row_in_scale_block],
        )


class Fp8GemvUmmaStreamSm100(ComputeInstruction):
    """Stream combined native MXF8 operands through SM100 UMMA."""

    def __init__(
        self,
        k_tiles: int,
        scale_pack: int = 1,
        output_groups: int = 1,
    ):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("FP8 UMMA K-tile count must fit uint16")
        if scale_pack not in (1, 2, 4) or k_tiles % scale_pack:
            raise ValueError(
                "FP8 UMMA scale pack must be 1, 2, or 4 and divide K tiles"
            )
        if output_groups not in (1, 2):
            raise ValueError("FP8 UMMA output groups must be 1 or 2")
        if output_groups > 1 and scale_pack == 1:
            raise ValueError("grouped FP8 UMMA requires packed scales")
        super().__init__(
            opcode=family_ref(
                "FP8_GEMV_UMMA_STREAM_SM100",
                SCALE_PACK=scale_pack,
                OUTPUT_GROUPS=output_groups,
            ),
            args=[k_tiles],
        )


class Fp8GemvUmmaSplitKSm100(ComputeInstruction):
    """Emit one M128 partial for STU TMA reduce-add."""

    BF16_BYTES = 2
    FP32_BYTES = 4

    def __init__(
        self,
        k_tiles: int,
        reduction_bytes: int = FP32_BYTES,
        scale_pack: int = 1,
        output_groups: int = 1,
    ):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("split-K FP8 UMMA K-tile count must fit uint16")
        if reduction_bytes not in (self.BF16_BYTES, self.FP32_BYTES):
            raise ValueError("split-K reduction must use BF16 or FP32")
        if scale_pack not in (1, 2, 4) or k_tiles % scale_pack:
            raise ValueError(
                "split-K FP8 scale pack must be 1, 2, or 4 and divide K tiles"
            )
        if output_groups not in (1, 2):
            raise ValueError("split-K FP8 output groups must be 1 or 2")
        if output_groups > 1 and scale_pack == 1:
            raise ValueError("grouped split-K FP8 UMMA requires packed scales")
        super().__init__(
            opcode=family_ref(
                "FP8_GEMV_UMMA_SPLITK_SM100",
                SCALE_PACK=scale_pack,
                OUTPUT_GROUPS=output_groups,
                REDUCTION_BYTES=reduction_bytes,
            ),
            args=[k_tiles],
        )


class Fp8GemvUmmaCoupledSm100(ComputeInstruction):
    """Common retained-ring MXFP8 x MXFP8 M256/K256 task."""

    BF16_BYTES = 2
    FP32_BYTES = 4

    def __init__(
        self,
        k_pairs: int,
        reduction_bytes: int,
        phase_base: int,
    ):
        if not 1 <= int(k_pairs) <= 0xFFFF:
            raise ValueError("coupled FP8 K-pair count must fit uint16")
        if reduction_bytes not in (self.BF16_BYTES, self.FP32_BYTES):
            raise ValueError("coupled FP8 output must use BF16 or FP32")
        if not 0 <= int(phase_base) <= 0xFFFF:
            raise ValueError("coupled FP8 phase base must fit uint16")
        super().__init__(
            opcode=opcode.OP_FP8_GEMV_UMMA_COUPLED_SM100,
            args=[
                int(k_pairs),
                int(reduction_bytes),
                int(phase_base),
            ],
        )


class Fp8GemvUmmaStreamRawScaleSm100(ComputeInstruction):
    """Stream compact FP8 weights and populate SFA TMEM from a raw scale."""

    def __init__(self, k_tiles: int):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("FP8 UMMA K-tile count must fit uint16")
        super().__init__(
            opcode=opcode.OP_FP8_GEMV_UMMA_STREAM_RAW_SCALE_SM100,
            args=[k_tiles],
        )


class Fp8GemvUmmaSplitKRawScaleSm100(ComputeInstruction):
    """Emit an M128 partial while compute loads compact weight scales."""

    BF16_BYTES = 2
    FP32_BYTES = 4

    def __init__(self, k_tiles: int, reduction_bytes: int = FP32_BYTES):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("split-K FP8 UMMA K-tile count must fit uint16")
        if reduction_bytes not in (self.BF16_BYTES, self.FP32_BYTES):
            raise ValueError("split-K reduction must use BF16 or FP32")
        super().__init__(
            opcode=opcode.OP_FP8_GEMV_UMMA_SPLITK_RAW_SCALE_SM100,
            args=[k_tiles, reduction_bytes],
        )


class Fp8UmmaPrepackSm100(ComputeInstruction):
    """Pack checkpoint FP8 data and block-128 scales for native UMMA."""

    WEIGHT = 0
    ACTIVATION = 1

    def __init__(self, kind: int, k_tiles: int):
        if kind not in (self.WEIGHT, self.ACTIVATION):
            raise ValueError("FP8 UMMA prepack kind must be weight or activation")
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("FP8 UMMA prepack K-tile count must fit uint16")
        super().__init__(
            opcode=opcode.OP_FP8_UMMA_PREPACK_SM100,
            args=[kind, k_tiles],
        )


class Dsv4Fp8QuantUmmaBSm100(ComputeInstruction):
    """Quantize BF16 K128 tiles into the native N8 MXF8 B layout."""

    def __init__(self, k_tiles: int = 1, scale_pack: int = 1):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("native FP8 quant K-tile count must fit uint16")
        if scale_pack not in (1, 2, 4) or k_tiles % scale_pack:
            raise ValueError(
                "native FP8 quant scale pack must be 1, 2, or 4 and divide K tiles"
            )
        super().__init__(
            opcode=family_ref(
                "DSV4_FP8_QUANT_UMMA_B_SM100", SCALE_PACK=scale_pack
            ),
            args=[k_tiles],
        )


class Dsv4Mxfp8QuantFfnInputSm100(ComputeInstruction):
    """Pack BF16 K512 shards into native Linear-1 MXFP8 records."""

    def __init__(self, k512_tiles: int = 1):
        if k512_tiles <= 0 or k512_tiles > 0xFFFF:
            raise ValueError("FFN MXFP8 K512-tile count must fit uint16")
        super().__init__(
            opcode=opcode.OP_DSV4_MXFP8_QUANT_FFN_INPUT_SM100,
            args=[k512_tiles],
        )


class Dsv4InverseRopeFp8QuantUmmaBSm100(ComputeInstruction):
    """Inverse partial RoPE and native N8/K128 FP8 packing for one head."""

    def __init__(self):
        super().__init__(
            opcode=opcode.OP_DSV4_INV_ROPE_FP8_QUANT_UMMA_B_SM100,
            args=[],
        )


class Dsv4RmsFp8QuantUmmaBSm100(ComputeInstruction):
    """Fuse weighted RMSNorm with native N8/K128 MXF8 packing."""

    def __init__(
        self,
        k_tiles: int,
        output_tile_start: int,
        output_tile_count: int,
        epsilon: float,
    ):
        if not 1 <= k_tiles <= 0xFF:
            raise ValueError("fused RMS/native-FP8 K-tile count must fit uint8")
        if not 0 <= output_tile_start < k_tiles:
            raise ValueError("fused RMS/native-FP8 tile start is out of range")
        if not 1 <= output_tile_count <= 0xFF:
            raise ValueError("fused RMS/native-FP8 tile count must fit uint8")
        if output_tile_start + output_tile_count > k_tiles:
            raise ValueError("fused RMS/native-FP8 tile shard exceeds K")
        if epsilon <= 0:
            raise ValueError("fused RMS/native-FP8 epsilon must be positive")
        tile_shard = output_tile_start | (output_tile_count << 8)
        super().__init__(
            opcode=opcode.OP_DSV4_RMS_FP8_QUANT_UMMA_B_SM100,
            args=[k_tiles, tile_shard, encode_bfloat16_u16(epsilon)],
        )


class Dsv4Fp32RmsFp8QuantUmmaBSm100(ComputeInstruction):
    """Normalize an FP32 split-K accumulator and pack native MXF8."""

    def __init__(
        self,
        k_tiles: int,
        output_tile_start: int,
        output_tile_count: int,
        epsilon: float,
    ):
        if not 1 <= k_tiles <= 0xFF:
            raise ValueError("FP32 RMS/native-FP8 K-tile count must fit uint8")
        if not 0 <= output_tile_start < k_tiles:
            raise ValueError("FP32 RMS/native-FP8 tile start is out of range")
        if not 1 <= output_tile_count <= 0xFF:
            raise ValueError("FP32 RMS/native-FP8 tile count must fit uint8")
        if output_tile_start + output_tile_count > k_tiles:
            raise ValueError("FP32 RMS/native-FP8 tile shard exceeds K")
        if epsilon <= 0:
            raise ValueError("FP32 RMS/native-FP8 epsilon must be positive")
        tile_shard = output_tile_start | (output_tile_count << 8)
        super().__init__(
            opcode=opcode.OP_DSV4_FP32_RMS_FP8_QUANT_UMMA_B_SM100,
            args=[k_tiles, tile_shard, encode_bfloat16_u16(epsilon)],
        )


class Dsv4Bf16GemvGroup4SplitKSm100(ComputeInstruction):
    """Reuse one BF16 activation across four M128 split-K outputs."""

    def __init__(self, k_tiles: int):
        if (
            not 1 <= k_tiles <= 0xFFFF
            or (k_tiles != 2 and k_tiles % 4)
        ):
            raise ValueError(
                "grouped BF16 split-K tile count must be two or a positive "
                "multiple of four"
            )
        super().__init__(
            opcode=opcode.OP_DSV4_BF16_GEMV_GROUP4_SPLITK_SM100,
            args=[k_tiles],
        )


class Dsv4PreloadRopeTables(ComputeInstruction):
    def __init__(self, num_tables: int):
        if num_tables <= 0 or num_tables > 4:
            raise ValueError("DeepSeek resident RoPE table count must be in [1,4]")
        super().__init__(
            opcode=opcode.OP_DSV4_PRELOAD_ROPE_TABLES,
            args=[num_tables],
        )


class Dsv4Rope64(ComputeInstruction):
    def __init__(
        self,
        rows: int,
        head_dim: int,
        inverse: bool = False,
        fixed_table_id: int | None = None,
    ):
        if rows <= 0 or rows > 0xFFFF:
            raise ValueError("DeepSeek RoPE rows must fit in a positive uint16")
        if head_dim <= 64 or head_dim > 0xFFFF or head_dim % 2:
            raise ValueError(
                "DeepSeek RoPE head width must be an even uint16 above 64"
            )
        if fixed_table_id is not None and not 0 <= fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        table_selector = 0 if fixed_table_id is None else fixed_table_id + 1
        super().__init__(
            opcode=opcode.OP_DSV4_ROPE_64,
            args=[
                rows,
                head_dim,
                (table_selector << 1) | int(inverse),
            ],
        )


class Dsv4RmsRope512_64(ComputeInstruction):
    """Fuse one 512-wide RMSNorm row with its 64-wide rotary suffix."""

    def __init__(
        self,
        *,
        weighted: bool,
        epsilon: float,
        fixed_table_id: int | None = None,
    ):
        if epsilon <= 0:
            raise ValueError("fused RMS/RoPE epsilon must be positive")
        if fixed_table_id is not None and not 0 <= fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        super().__init__(
            opcode=opcode.OP_DSV4_RMS_ROPE_512_64,
            args=[
                int(weighted),
                0 if fixed_table_id is None else fixed_table_id + 1,
                encode_bfloat16_u16(epsilon),
            ],
        )


class Dsv4Fp32RmsRope512_64(ComputeInstruction):
    """Fuse an FP32 split-K finalizer, RMSNorm, and 512/64 RoPE."""

    def __init__(
        self,
        *,
        weighted: bool,
        epsilon: float,
        fixed_table_id: int | None = None,
    ):
        if epsilon <= 0:
            raise ValueError("FP32 fused RMS/RoPE epsilon must be positive")
        if fixed_table_id is not None and not 0 <= fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        super().__init__(
            opcode=opcode.OP_DSV4_FP32_RMS_ROPE_512_64,
            args=[
                int(weighted),
                0 if fixed_table_id is None else fixed_table_id + 1,
                encode_bfloat16_u16(epsilon),
            ],
        )


class Dsv4Fp32RopeHadamard128(ComputeInstruction):
    """Fuse an FP32 split-K finalizer with index RoPE and Hadamard."""

    def __init__(self, fixed_table_id: int | None = None):
        if fixed_table_id is not None and not 0 <= fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        super().__init__(
            opcode=opcode.OP_DSV4_FP32_ROPE_HADAMARD_128,
            args=[0 if fixed_table_id is None else fixed_table_id + 1],
        )


class Dsv4SparseAttention512(ComputeInstruction):
    def __init__(self, topk: int):
        if topk <= 0 or topk > 0xFFFF:
            raise ValueError("DeepSeek attention topk must fit in a positive uint16")
        super().__init__(
            opcode=opcode.OP_DSV4_SPARSE_ATTENTION_512,
            args=[topk],
        )


class Dsv4ContiguousAttention512Block4(ComputeInstruction):
    def __init__(self, rows: int):
        if rows <= 0 or rows > 0xFFFF:
            raise ValueError("DeepSeek attention rows must fit in a positive uint16")
        super().__init__(
            opcode=opcode.OP_DSV4_CONTIGUOUS_ATTENTION_512_BLOCK4,
            args=[rows],
        )


class Dsv4ContiguousAttention512UmmaSm100(ComputeInstruction):
    def __init__(self, rows: int, output_tile: int | None = None):
        if rows <= 0 or rows > 128:
            raise ValueError("DeepSeek UMMA attention rows must be in [1,128]")
        if output_tile is not None and not 0 <= output_tile < 4:
            raise ValueError("DeepSeek UMMA output tile must be in [0,3]")
        super().__init__(
            opcode=opcode.OP_DSV4_CONTIGUOUS_ATTENTION_512_UMMA_SM100,
            args=[rows] if output_tile is None else [rows, output_tile + 1],
        )


class Dsv4ContiguousAttention512UmmaTail32Sm100(ComputeInstruction):
    def __init__(self, rows: int, output_tile: int | None = None):
        if rows <= 128 or rows > 160:
            raise ValueError("DeepSeek UMMA tail rows must be in [129,160]")
        if output_tile is not None and not 0 <= output_tile < 4:
            raise ValueError("DeepSeek UMMA output tile must be in [0,3]")
        super().__init__(
            opcode=opcode.OP_DSV4_CONTIGUOUS_ATTENTION_512_UMMA_TAIL32_SM100,
            args=[rows] if output_tile is None else [rows, output_tile + 1],
        )


class Dsv4AttentionSplit32UmmaSm100(ComputeInstruction):
    """Produce one all-head K32 UMMA attention partial."""

    def __init__(self, active_tokens: int):
        if not 1 <= active_tokens <= 32:
            raise ValueError("K32 attention split must contain 1..32 tokens")
        super().__init__(
            opcode=opcode.OP_DSV4_ATTENTION_SPLIT32_UMMA_SM100,
            args=[active_tokens],
        )


class Dsv4AttentionSplit64UmmaSm100(ComputeInstruction):
    """Produce one B64 BF16 partial from a retained internal KV ring."""

    def __init__(self, active_tokens: int, *, ring_port_mask: int = 1):
        if not 1 <= active_tokens <= 64:
            raise ValueError("B64 attention split must contain 1..64 tokens")
        if ring_port_mask not in (1, 2, 3):
            raise ValueError("attention ring must select LDU0 and/or LDU1")
        super().__init__(
            opcode=opcode.OP_DSV4_ATTENTION_SPLIT64_UMMA_SM100,
            args=[active_tokens, ring_port_mask],
        )


class Dsv4AttentionSplitReduceFp8Sm100(ComputeInstruction):
    """Merge attention partials and emit inverse-RoPE native FP8."""

    def __init__(self, num_splits: int, head: int, output_group: int = 0):
        if not 1 <= num_splits <= 24:
            raise ValueError("attention reducer split count must be in [1,24]")
        if not 0 <= head < 64:
            raise ValueError("attention reducer head must be in [0,64)")
        if output_group not in (0, 1):
            raise ValueError("attention reducer output group must be 0 or 1")
        super().__init__(
            opcode=opcode.OP_DSV4_ATTENTION_SPLIT_REDUCE_FP8_SM100,
            args=[num_splits, head, output_group],
        )


class Dsv4AttentionContext1Fp8Sm100(ComputeInstruction):
    """Directly emit native FP8 attention for one Q/KV row."""

    def __init__(self, head: int, *, normalize_q: bool = False):
        if not 0 <= head < 64:
            raise ValueError("context-1 attention head must be in [0,64)")
        # num_splits == 0 selects the compact single-row path in the ordinary
        # split-attention handler. The output-group sentinel 2 means that the
        # one head task publishes all four native O_a tiles. Bit 2 fuses the
        # BF16 Q RMS/RoPE boundary into the task.
        super().__init__(
            opcode=opcode.OP_DSV4_ATTENTION_SPLIT_REDUCE_FP8_SM100,
            args=[0, head, 2 | (int(normalize_q) << 2)],
        )


class Dsv4RouteTop6(ComputeInstruction):
    def __init__(
        self,
        hash_routing: bool,
        route_scale: float = 1.5,
        *,
        pretransformed_raw: bool = False,
    ):
        if route_scale <= 0:
            raise ValueError("DeepSeek route_scale must be positive")
        super().__init__(
            opcode=opcode.OP_DSV4_ROUTE_TOP6,
            args=[
                int(hash_routing),
                encode_bfloat16_u16(route_scale),
                int(pretransformed_raw),
            ],
        )


class Dsv4RouteTop6Prepared(ComputeInstruction):
    """Select top-6 from projection-prepared original/biased score pairs."""

    def __init__(
        self,
        hash_routing: bool,
        route_scale: float = 1.5,
    ):
        if route_scale <= 0:
            raise ValueError("DeepSeek route_scale must be positive")
        super().__init__(
            opcode=opcode.OP_DSV4_ROUTE_TOP6_PREPARED,
            args=[
                int(hash_routing),
                encode_bfloat16_u16(route_scale),
            ],
        )


class Dsv4ExpertReduce(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_DSV4_EXPERT_REDUCE, args=[])


class Dsv4Fp32Bf16Gemv(ComputeInstruction):
    EMIT_SQUARE_SUM = 1 << 0
    FUSE_HC_POST = 1 << 1
    PROFILE_OPERANDS = 1 << 2
    DIRECT_COEFFICIENTS = 1 << 3
    _FUSED_POINTER_ALIGNMENT = 16
    _FUSED_POINTER_SHIFT = 4

    def __init__(
        self,
        k: int,
        tile_k: int,
        emit_square_sum: bool = False,
        fuse_hc_post: bool = False,
        profile_operands: bool = False,
        packed_coefficients_address: int | None = None,
    ):
        if k <= 0 or k > 0xFFFF:
            raise ValueError("DeepSeek FP32 GEMV K must fit in uint16")
        if tile_k < 0 or tile_k > 0xFFFF or (not fuse_hc_post and tile_k == 0):
            raise ValueError("DeepSeek FP32 GEMV tile K must fit in uint16")
        flags = (
            self.EMIT_SQUARE_SUM * int(bool(emit_square_sum))
            | self.FUSE_HC_POST * int(bool(fuse_hc_post))
            | self.PROFILE_OPERANDS * int(bool(profile_operands))
        )
        if fuse_hc_post and packed_coefficients_address is not None:
            if packed_coefficients_address % self._FUSED_POINTER_ALIGNMENT:
                raise ValueError(
                    "fused DeepSeek FP32 GEMV coefficients must be 16-byte aligned"
                )
            encoded_pointer = (
                packed_coefficients_address >> self._FUSED_POINTER_SHIFT
            )
            if encoded_pointer >= 1 << 44:
                raise ValueError(
                    "fused DeepSeek FP32 GEMV coefficient pointer must fit 48 bits"
                )
            # Fused mode does not consume k or tile_k.  Reuse those two fields
            # and the unused high twelve bits of flags for the aligned raw
            # pointer, eliminating a pointer-only LDU/M2C transaction.
            k = encoded_pointer & 0xFFFF
            tile_k = (encoded_pointer >> 16) & 0xFFFF
            flags |= self.DIRECT_COEFFICIENTS
            flags |= (encoded_pointer >> 32) << 4
        elif packed_coefficients_address is not None:
            raise ValueError(
                "packed coefficients are valid only for fused DeepSeek FP32 GEMV"
            )
        super().__init__(
            opcode=opcode.OP_DSV4_FP32_BF16_GEMV,
            args=[k, tile_k, flags],
        )


class Dsv4Bf16Gemv(ComputeInstruction):
    def __init__(self, k: int, tile_k: int, output_fp32: bool = False):
        if k <= 0 or k > 0xFFFF:
            raise ValueError("DeepSeek BF16 GEMV K must fit in uint16")
        if tile_k <= 0 or tile_k > 0xFFFF:
            raise ValueError("DeepSeek BF16 GEMV tile K must fit in uint16")
        super().__init__(
            opcode=opcode.OP_DSV4_BF16_GEMV,
            args=[k, tile_k, int(output_fp32)],
        )


class Dsv4RouterBf16GemvSm100(ComputeInstruction):
    """Vectorized router GEMV plus parallel score preparation."""

    def __init__(self, k: int, rows_per_task: int):
        if k <= 0 or k > 0xFFFF or k % 1024:
            raise ValueError("DSV4 router GEMV K must be a uint16 K1024 multiple")
        if rows_per_task not in (1, 2, 4):
            raise ValueError("DSV4 router rows per task must be 1, 2, or 4")
        super().__init__(
            opcode=family_ref(
                "DSV4_ROUTER_BF16_GEMV_SM100", ROWS=rows_per_task
            ),
            args=[k],
        )


class Dsv4HcPre(ComputeInstruction):
    def __init__(self, sinkhorn_iters: int = 20, epsilon: float = 1.0e-6):
        if sinkhorn_iters <= 0 or sinkhorn_iters > 0xFFFF:
            raise ValueError("DeepSeek Sinkhorn iterations must fit in uint16")
        if epsilon <= 0:
            raise ValueError("DeepSeek mHC epsilon must be positive")
        super().__init__(
            opcode=opcode.OP_DSV4_HC_PRE,
            args=[sinkhorn_iters, encode_bfloat16_u16(epsilon)],
        )


class Dsv4HcPreRms(ComputeInstruction):
    ZERO_FP32_OUTPUT = 1 << 0
    OUTPUT_FP8 = 1 << 1

    def __init__(
        self,
        zero_fp32_output: bool = False,
        output_fp8: bool = False,
        split_metadata_splits: int = 0,
    ):
        if split_metadata_splits not in (0, 2, 4, 8, 16):
            raise ValueError(
                "mHC pre-RMS metadata splits must be 0, 2, 4, 8, or 16"
            )
        flags = (
            self.ZERO_FP32_OUTPUT * int(bool(zero_fp32_output))
            | self.OUTPUT_FP8 * int(bool(output_fp8))
        )
        super().__init__(
            opcode=opcode.OP_DSV4_HC_PRE_RMS,
            args=[flags, split_metadata_splits],
        )


class Dsv4HcPost(ComputeInstruction):
    COMPACT_IO = 1 << 0
    BRANCH_FP32 = 1 << 1
    WIDTH_SHIFT = 2
    PACKED_RW = 1 << 6
    POINTER_ALIGNMENT = 128

    def __init__(
        self,
        width: int | None = None,
        branch_fp32: bool = False,
        packed_coefficients: torch.Tensor | None = None,
        packed_rw: bool = False,
    ):
        if packed_coefficients is not None:
            if width is None or width <= 0 or width > 4096:
                raise ValueError("compact mHC post requires a valid shard width")
            if width & (width - 1):
                raise ValueError("compact mHC post shard width must be a power of two")
            if packed_coefficients.device.type != "cuda":
                raise ValueError("packed mHC post coefficients must be on CUDA")
            address = packed_coefficients.data_ptr()
            if address & (self.POINTER_ALIGNMENT - 1):
                raise ValueError(
                    "packed mHC post coefficients must be 128-byte aligned"
                )
            width_log2 = width.bit_length() - 1
            encoded = (
                address
                | self.COMPACT_IO
                | self.BRANCH_FP32 * int(bool(branch_fp32))
                | (width_log2 << self.WIDTH_SHIFT)
                | self.PACKED_RW * int(bool(packed_rw))
            )
            super().__init__(
                opcode=opcode.OP_DSV4_HC_POST,
                args=[
                    encoded & 0xFFFF,
                    (encoded >> 16) & 0xFFFF,
                    (encoded >> 32) & 0xFFFF,
                ],
            )
            return
        if width is None:
            raise ValueError("legacy mHC post requires a shard width")
        if width <= 0 or width > 4096:
            raise ValueError("DeepSeek mHC post width must be in [1,4096]")
        super().__init__(
            opcode=opcode.OP_DSV4_HC_POST,
            args=[width, int(bool(branch_fp32))],
        )


class Dsv4SiluClampMul(ComputeInstruction):
    """Apply bounded SwiGLU to ordinary BF16 vectors."""

    def __init__(self, num_token: int, width: int, limit: float = 10.0):
        if num_token <= 0 or num_token > 0xFFFF:
            raise ValueError("DeepSeek SwiGLU token count must fit in uint16")
        if width <= 0 or width > 0xFFFF or width % 8:
            raise ValueError(
                "DeepSeek SwiGLU width must be an 8-aligned positive uint16"
            )
        if limit <= 0:
            raise ValueError("DeepSeek SwiGLU limit must be positive")
        super().__init__(
            opcode=opcode.OP_DSV4_SILU_CLAMP_MUL,
            args=[num_token, width, encode_bfloat16_u16(limit)],
        )


class Dsv4Fp32SwiGluNvfp4QuantUmmaBSm100(ComputeInstruction):
    """Fuse FP32 gate/up activation into the native NVFP4 W2 input."""

    def __init__(self, k_tiles: int, limit: float = 10.0):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("fused FP32 SwiGLU K-tile count must fit uint16")
        if limit <= 0:
            raise ValueError("FP32 SwiGLU limit must be positive")
        super().__init__(
            opcode=opcode.OP_DSV4_FP32_SWIGLU_NVFP4_QUANT_UMMA_B_SM100,
            args=[k_tiles, encode_bfloat16_u16(limit)],
        )


class Dsv4Hadamard(ComputeInstruction):
    def __init__(self, width: int):
        if width not in (128, 512):
            raise ValueError("DeepSeek Hadamard width must be 128 or 512")
        super().__init__(
            opcode=opcode.OP_DSV4_HADAMARD,
            args=[width],
        )


class Dsv4GatedPool(ComputeInstruction):
    def __init__(self, pool_rows: int, width: int, tail_bias: bool = False):
        if pool_rows <= 0 or pool_rows > 0xFFFF:
            raise ValueError("DeepSeek gated-pool rows must fit in uint16")
        if width not in (128, 512):
            raise ValueError("DeepSeek gated-pool width must be 128 or 512")
        super().__init__(
            opcode=opcode.OP_DSV4_GATED_POOL,
            args=[pool_rows, width, int(tail_bias)],
        )


class Dsv4GatedPoolRmsRope(ComputeInstruction):
    """Fuse scalar gated pooling, weighted RMSNorm, RoPE, and Hadamard."""

    def __init__(
        self,
        pool_rows: int,
        width: int,
        *,
        tail_bias: bool,
        hadamard: bool,
        epsilon: float,
        fixed_table_id: int | None = None,
    ):
        if pool_rows <= 0 or pool_rows > 0xFFFF:
            raise ValueError("fused gated-pool rows must fit uint16")
        if width not in (128, 512):
            raise ValueError("fused gated-pool width must be 128 or 512")
        if hadamard and width != 128:
            raise ValueError("fused gated-pool Hadamard requires width 128")
        if epsilon <= 0:
            raise ValueError("fused gated-pool RMS epsilon must be positive")
        if fixed_table_id is not None and not 0 <= fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        fixed_table_selector = (
            0 if fixed_table_id is None else fixed_table_id + 1
        )
        config = (
            int(width == 512)
            | (int(tail_bias) << 1)
            | (int(hadamard) << 2)
            | (fixed_table_selector << 3)
        )
        super().__init__(
            opcode=opcode.OP_DSV4_GATED_POOL_RMS_ROPE,
            args=[
                pool_rows,
                config,
                encode_bfloat16_u16(epsilon),
            ],
        )


class Dsv4GatedPoolPacked8Shard128(ComputeInstruction):
    def __init__(self, history_rows: int):
        if history_rows <= 0 or history_rows > 0xFFFF:
            raise ValueError("packed gated-pool rows must fit in uint16")
        super().__init__(
            opcode=opcode.OP_DSV4_GATED_POOL_PACKED8_SHARD128,
            args=[history_rows],
        )


class Dsv4GatedPoolPacked8RmsPartial(ComputeInstruction):
    def __init__(self, history_rows: int):
        if history_rows <= 0 or history_rows > 0xFFFF:
            raise ValueError(
                "packed pool/RMS history rows must fit uint16"
            )
        super().__init__(
            opcode=opcode.OP_DSV4_GATED_POOL_PACKED8_RMS_PARTIAL,
            args=[history_rows],
        )


class Dsv4GatedPoolPacked8HistoryState(ComputeInstruction):
    """Pool immutable packed history into stable FP32 softmax state."""

    def __init__(self, history_rows: int):
        if history_rows <= 0 or history_rows > 0xFFFF:
            raise ValueError(
                "packed history-state rows must fit in uint16"
            )
        super().__init__(
            opcode=opcode.OP_DSV4_GATED_POOL_PACKED8_HISTORY_STATE,
            args=[history_rows],
        )


class Dsv4GatedPoolTailRmsPartial(ComputeInstruction):
    """Merge the projected tail and publish pooled FP32 RMS partials."""

    def __init__(self):
        super().__init__(
            opcode=opcode.OP_DSV4_GATED_POOL_TAIL_RMS_PARTIAL,
            args=[],
        )


class Dsv4Fp32RmsRopeShard128(ComputeInstruction):
    def __init__(
        self,
        shard: int,
        *,
        epsilon: float,
        fixed_table_id: int | None = None,
    ):
        if not 0 <= shard < 4:
            raise ValueError("FP32 RMS/RoPE shard must be in [0,4)")
        if epsilon <= 0:
            raise ValueError("FP32 RMS/RoPE shard epsilon must be positive")
        if fixed_table_id is not None and not 0 <= fixed_table_id < 4:
            raise ValueError("DeepSeek fixed RoPE table ID must be in [0,4)")
        super().__init__(
            opcode=opcode.OP_DSV4_FP32_RMS_ROPE_SHARD128,
            args=[
                shard,
                0 if fixed_table_id is None else fixed_table_id + 1,
                encode_bfloat16_u16(epsilon),
            ],
        )


class Dsv4IndexScore(ComputeInstruction):
    def __init__(self, rows: int):
        if rows <= 0 or rows > 0xFFFF:
            raise ValueError("DeepSeek index-score rows must fit in uint16")
        super().__init__(opcode=opcode.OP_DSV4_INDEX_SCORE, args=[rows])


class Dsv4TopK512(ComputeInstruction):
    def __init__(self, rows: int, topk: int, index_offset: int = 0):
        if rows <= 0 or rows > 0xFFFF:
            raise ValueError("DeepSeek top-k rows must fit in uint16")
        if topk <= 0 or topk > min(rows, 512):
            raise ValueError("DeepSeek index top-k must be in [1,min(rows,512)]")
        if index_offset < 0 or index_offset > 0xFFFF:
            raise ValueError("DeepSeek index offset must fit in uint16")
        super().__init__(
            opcode=opcode.OP_DSV4_TOPK_512,
            args=[rows, topk, index_offset],
        )


class Dsv4HcHead(ComputeInstruction):
    def __init__(self, epsilon: float = 1.0e-6):
        if epsilon <= 0:
            raise ValueError("DeepSeek mHC head epsilon must be positive")
        super().__init__(
            opcode=opcode.OP_DSV4_HC_HEAD,
            args=[encode_bfloat16_u16(epsilon)],
        )


class Dsv4HcHeadRms(ComputeInstruction):
    def __init__(
        self, epsilon: float = 1.0e-6, rms_epsilon: float = 1.0e-6
    ):
        if epsilon <= 0 or rms_epsilon <= 0:
            raise ValueError("DeepSeek mHC-head/RMS epsilons must be positive")
        super().__init__(
            opcode=opcode.OP_DSV4_HC_HEAD_RMS,
            args=[
                encode_bfloat16_u16(epsilon),
                encode_bfloat16_u16(rms_epsilon),
            ],
        )


class Dsv4Fp8Quant128(ComputeInstruction):
    def __init__(self, k: int):
        if k <= 0 or k > 0xFFFF or k % 128:
            raise ValueError("DeepSeek FP8 quant K must be a uint16 multiple of 128")
        super().__init__(opcode=opcode.OP_DSV4_FP8_QUANT_128, args=[k])


class Dsv4Nvfp4Quant16(ComputeInstruction):
    def __init__(self, k: int):
        if k <= 0 or k > 0xFFFF or k % 16:
            raise ValueError("DeepSeek NVFP4 quant K must be a uint16 multiple of 16")
        super().__init__(opcode=opcode.OP_DSV4_NVFP4_QUANT_16, args=[k])


class Dsv4Nvfp4QuantUmmaBSm100(ComputeInstruction):
    """Quantize K256 tiles directly into the native N8 UMMA B layout."""

    def __init__(self, k_tiles: int):
        if k_tiles <= 0 or k_tiles > 0xFFFF:
            raise ValueError("DeepSeek native NVFP4 K-tile count must fit uint16")
        super().__init__(
            opcode=opcode.OP_DSV4_NVFP4_QUANT_UMMA_B_SM100,
            args=[k_tiles],
        )


class Gemv_M64N8K64(ComputeInstruction):
    MNK = (64, 8, 64)
    n_batch = 1

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=64, BLOAD=1, RESIDUAL=residual), args=[kTiles, nprefeth])

class Gemv_M64N8K128(ComputeInstruction):
    MNK = (64, 8, 128)
    n_batch = 1

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=128, BLOAD=1, RESIDUAL=residual), args=[kTiles, nprefeth])

class Gemv_M64N8B2(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 2

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=64, N=8, K=256, BLOAD=2, RESIDUAL=residual), args=[kTiles, nprefeth])


class Gemv_M128N8(ComputeInstruction):
    MNK = (128, 8, 128)
    n_batch = 4

    def __init__(self, kTiles: int, nprefeth=0, residual: bool = False):
        super().__init__(opcode=family_ref("GEMV_WGMMA", M=128, N=8, K=128, BLOAD=4, RESIDUAL=residual), args=[kTiles, nprefeth])


class Gemv_M128N8Direct4(ComputeInstruction):
    MNK = (128, 8, 128)
    n_batch = 4
    direct_output = True
    output_groups = 4

    def __init__(self, kTiles: int, output_stride: int, output_group_stride: int):
        if output_stride % self.MNK[0] or output_group_stride % self.MNK[0]:
            raise ValueError("direct M128 GEMV strides must be multiples of 128")
        super().__init__(
            opcode=opcode.OP_GEMV_SM100_M128N8_DIRECT4,
            args=[
                kTiles,
                output_stride // self.MNK[0],
                output_group_stride // self.MNK[0],
            ],
        )


class Gemv_M128N8Argmax4(ComputeInstruction):
    MNK = (128, 8, 128)
    n_batch = 4
    output_groups = 4

    def __init__(self, kTiles: int, output_group_stride: int,
                 vocabulary_base: int):
        if output_group_stride % self.MNK[0] or vocabulary_base % self.MNK[0]:
            raise ValueError("fused M128 GEMV offsets must be multiples of 128")
        super().__init__(
            opcode=opcode.OP_GEMV_SM100_M128N8_ARGMAX4,
            args=[
                kTiles,
                output_group_stride // self.MNK[0],
                vocabulary_base // self.MNK[0],
            ],
        )


class _GemvM128N8GroupedReduce(ComputeInstruction):
    MNK = (128, 8, 128)

    def __init__(self, kTiles: int):
        super().__init__(opcode=self.opcode, args=[kTiles])


class Gemv_M128N8Group4B2(_GemvM128N8GroupedReduce):
    n_batch = 2
    output_groups = 4
    opcode = opcode.OP_GEMV_SM100_M128N8_GROUP4_B2


class Gemv_M128N8Group4B3(_GemvM128N8GroupedReduce):
    n_batch = 3
    output_groups = 4
    opcode = opcode.OP_GEMV_SM100_M128N8_GROUP4_B3


class Gemv_M128N8Group4B4(_GemvM128N8GroupedReduce):
    n_batch = 4
    output_groups = 4
    opcode = opcode.OP_GEMV_SM100_M128N8_GROUP4_B4


class Gemv_M128N8Group4B7(_GemvM128N8GroupedReduce):
    n_batch = 7
    output_groups = 4
    opcode = opcode.OP_GEMV_SM100_M128N8_GROUP4_B7

class Gemm_M64N64(ComputeInstruction):
    MNK = (64, 64, 128)
    n_batch = 1

    def __init__(self, kTiles: int, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMM_M64N64, args=[kTiles])

class Gemm_M64N64K64(ComputeInstruction):
    MNK = (64, 64, 64)
    n_batch = 1

    def __init__(self, kTiles: int, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMM_M64N64K64, args=[kTiles])


class Gemm_M64N128K64(ComputeInstruction):
    MNK = (64, 128, 64)
    n_batch = 1

    def __init__(self, kTiles: int, residual: bool = False):
        super().__init__(opcode=opcode.OP_GEMM_M64N128K64, args=[kTiles])


class Gemv_M64N8_ROPE_128(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 4

    def __init__(self, kTiles: int, hist_len: int, head_dim_ofst: int):
        super().__init__(opcode=opcode.OP_GEMV_M64N8_ROPE_128, args=[kTiles, hist_len, head_dim_ofst])


class Gemv_M128N8_ROPE_128(ComputeInstruction):
    MNK = (128, 8, 128)
    n_batch = 4

    def __init__(self, kTiles: int, hist_len: int, head_dim_ofst: int):
        super().__init__(opcode=opcode.OP_GEMV_M128N8_ROPE_128, args=[kTiles, hist_len, head_dim_ofst])


class Gemv_M192N16(ComputeInstruction):
    MNK = (192, 8, 128)

    def __init__(self, kTiles: int):
        super().__init__(opcode=opcode.OP_GEMV_M192, args=[kTiles])

class Gemv_M64N8_MMA(ComputeInstruction):
    MNK = (64, 8, 256)
    n_batch = 1
    def __init__(self, kTiles: int):
        super().__init__(opcode=family_ref("GEMV_MMA", M=64, N=8, K=256), args=[kTiles])


class WGMMA_64x256x64_F16(ComputeInstruction):
    MNK = (64, 64, 256)

    def __init__(self, mTiles, kTiles, residual: bool = False):
        residual_flag = 1 if residual else 0
        super().__init__(opcode=opcode.OP_WGMMA_M64N256K16_F16, args=[mTiles, kTiles, residual_flag])


class WGMMA_64x256x64_BF16(ComputeInstruction):
    MNK = (64, 64, 128)

    def __init__(self, mTiles, kTiles, residual: bool = False):
        residual_flag = 1 if residual else 0
        super().__init__(opcode=opcode.OP_WGMMA_M64N256K16_BF16, args=[mTiles, kTiles, residual_flag])


class ROPE_INTERLEAVE_512(ComputeInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_ROPE_INTERLEAVE_512, args=[])


ATTENTION_DYNAMIC_LAST_KV_LEN_FLAG = 0x4
ATTENTION_DYNAMIC_NUM_KV_BLOCKS_FLAG = 0x8
ATTENTION_BLOCK_COUNTER_SHIFT = 4
ATTENTION_COUNTER_SHIFT = 8
ATTENTION_OUTER_COUNTER_SHIFT = 12


def _encode_attention_runtime_flags(
    need_norm: bool,
    need_rope: bool,
    seq_len_counter_reg: int | None = None,
    num_kv_block_counter_reg: int | None = None,
    outer_seq_len_counter_reg: int | None = None,
) -> int:
    flags = 0
    if need_norm:
        flags |= 1
    if need_rope:
        flags |= 2
    if seq_len_counter_reg is not None:
        assert 0 <= seq_len_counter_reg < config.num_loop_counters
        flags |= ATTENTION_DYNAMIC_LAST_KV_LEN_FLAG
        flags |= seq_len_counter_reg << ATTENTION_COUNTER_SHIFT
    if num_kv_block_counter_reg is not None:
        assert 0 <= num_kv_block_counter_reg < config.num_loop_counters
        flags |= ATTENTION_DYNAMIC_NUM_KV_BLOCKS_FLAG
        flags |= num_kv_block_counter_reg << ATTENTION_BLOCK_COUNTER_SHIFT
    if outer_seq_len_counter_reg is not None:
        assert 0 <= outer_seq_len_counter_reg < config.num_loop_counters
        flags |= outer_seq_len_counter_reg << ATTENTION_OUTER_COUNTER_SHIFT
    return flags

def _encode_attention_qkv_workload_flag(
    num_active_q: int,
    last_kv_active_token_len: int,
    kv_block_size: int = 64,
) -> int:
    assert 0 < num_active_q < 0x80, "num_active_q must fit below the KV-tile flag"
    assert kv_block_size in {64, 128}, "kv_block_size must be 64 or 128"
    kv_tile_flag = 0x80 if kv_block_size == 128 else 0
    return num_active_q | kv_tile_flag | (last_kv_active_token_len << 8)

class ATTENTION_M64N64K16_F16_F32_64_64_hdim(ComputeInstruction):
    HEAD_DIM = 128

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None, kv_block_size: int = 64, outer_seq_len_counter_reg: int | None = None, outer_seq_len_counter_stride: int = 0):
        if outer_seq_len_counter_reg is None:
            assert outer_seq_len_counter_stride == 0
        else:
            assert 0 < outer_seq_len_counter_stride < 256
        assert 0 < num_kv_block < 256
        encoded_num_kv_block = num_kv_block | (outer_seq_len_counter_stride << 8)
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim,
            args=[
                encoded_num_kv_block,
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len, kv_block_size),
                _encode_attention_runtime_flags(
                    need_norm,
                    need_rope,
                    seq_len_counter_reg,
                    num_kv_block_counter_reg,
                    outer_seq_len_counter_reg,
                ),
            ],
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA(ComputeInstruction):
    HEAD_DIM = 128

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None, kv_block_size: int = 64):
        assert kv_block_size == 64, "MMA attention only supports KV64"
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA,
            args=[
                num_kv_block,
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len),
                _encode_attention_runtime_flags(need_norm, need_rope, seq_len_counter_reg, num_kv_block_counter_reg),
            ],
        )


class ATTENTION_SM100_BF16_HDIM128_DIRECT(ComputeInstruction):
    HEAD_DIM = 128

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = False, need_rope: bool = False, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None, kv_block_size: int = 64, outer_seq_len_counter_reg: int | None = None, outer_seq_len_counter_stride: int = 0):
        assert not need_norm and not need_rope, (
            "direct SM100 attention expects Q/K normalization and RoPE to be "
            "scheduled separately"
        )
        if outer_seq_len_counter_reg is None:
            assert outer_seq_len_counter_stride == 0
        else:
            assert 0 < outer_seq_len_counter_stride < 256
        assert 0 < num_kv_block < 256
        encoded_num_kv_block = num_kv_block | (outer_seq_len_counter_stride << 8)
        super().__init__(
            opcode=opcode.OP_ATTENTION_SM100_BF16_HDIM128_DIRECT,
            args=[
                encoded_num_kv_block,
                _encode_attention_qkv_workload_flag(
                    num_active_q, last_kv_active_token_len, kv_block_size
                ),
                _encode_attention_runtime_flags(
                    need_norm,
                    need_rope,
                    seq_len_counter_reg,
                    num_kv_block_counter_reg,
                    outer_seq_len_counter_reg,
                ),
            ],
        )


class ATTENTION_SM100_BF16_HDIM128_SPLIT_DIRECT(ComputeInstruction):
    HEAD_DIM = 128

    def __init__(
        self,
        num_kv_block: int,
        split_idx: int,
        num_active_q: int,
        last_kv_active_token_len: int,
        kv_start_idx: int,
        *,
        kv_block_size: int = 64,
        **_unused,
    ):
        assert split_idx < 16
        super().__init__(
            opcode=opcode.OP_ATTENTION_SM100_BF16_HDIM128_SPLIT_DIRECT,
            args=[
                num_kv_block | (split_idx << 12),
                _encode_attention_qkv_workload_flag(
                    num_active_q, last_kv_active_token_len, kv_block_size
                ),
                kv_start_idx,
            ],
        )


class ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT(
    ATTENTION_SM100_BF16_HDIM128_DIRECT
):
    def __init__(self, *args, kv_block_size: int = 128, **kwargs):
        assert kv_block_size == 128, "swapped SM100 attention requires KV128"
        super().__init__(*args, kv_block_size=kv_block_size, **kwargs)
        self.opcode = opcode.OP_ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT


class ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT(
    ATTENTION_SM100_BF16_HDIM128_SPLIT_DIRECT
):
    def __init__(self, *args, kv_block_size: int = 128, **kwargs):
        assert kv_block_size == 128, "swapped SM100 attention requires KV128"
        super().__init__(*args, kv_block_size=kv_block_size, **kwargs)
        self.opcode = opcode.OP_ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT


class ATTENTION_M64N64K16_F16_F32_64_64_hdim64(ComputeInstruction):
    HEAD_DIM = 64

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None):
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64,
            args=[
                num_kv_block, 
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len), 
                _encode_attention_runtime_flags(need_norm, need_rope, seq_len_counter_reg, num_kv_block_counter_reg)
            ],
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA(ComputeInstruction):
    HEAD_DIM = 64

    def __init__(self, num_kv_block: int, num_active_q: int, last_kv_active_token_len: int, need_norm: bool = True, need_rope: bool = True, seq_len_counter_reg: int | None = None, num_kv_block_counter_reg: int | None = None):
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA,
            args=[
                num_kv_block,
                _encode_attention_qkv_workload_flag(num_active_q, last_kv_active_token_len),
                _encode_attention_runtime_flags(need_norm, need_rope, seq_len_counter_reg, num_kv_block_counter_reg),
            ],
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim_split(ComputeInstruction):
    HEAD_DIM = 128
    def __init__(self, num_kv_block: int, split_idx: int, num_active_q: int, last_kv_active_token_len: int, kv_start_idx: int, need_norm: bool = True, need_rope: bool = True, kv_block_size: int = 64):
        assert split_idx < 16, "split_idx must be less than 16 to fit in the instruction encoding"
        # pack need_norm and need_rope into a uint16 arg
        arg0 = num_kv_block | (split_idx << 12)
        arg1 = _encode_attention_qkv_workload_flag(
            num_active_q, last_kv_active_token_len, kv_block_size
        )
        arg2 = kv_start_idx # make this 16bit to support long seq
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split, 
            args=[arg0, arg1, arg2]
        )


class ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA(ComputeInstruction):
    HEAD_DIM = 128
    def __init__(self, num_kv_block: int, split_idx: int, num_active_q: int, last_kv_active_token_len: int, kv_start_idx: int, need_norm: bool = True, need_rope: bool = True):
        assert split_idx < 16, "split_idx must be less than 16 to fit in the instruction encoding"
        arg0 = num_kv_block | (split_idx << 12)
        arg1 = num_active_q | (last_kv_active_token_len << 8)
        arg2 = kv_start_idx
        super().__init__(
            opcode=opcode.OP_ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA,
            args=[arg0, arg1, arg2]
        )

class ATTN_SPLIT_POST_REDUCE(ComputeInstruction):
    HEAD_DIM = 128
    Q_TILE = 4
    def __init__(
        self,
        num_split: int,
        raw_partial: bool = False,
        direct_output: bool = False,
    ):
        assert not direct_output or raw_partial
        super().__init__(
            opcode=opcode.OP_ATTN_SPLIT_POST_REDUCE,
            args=[num_split, int(raw_partial) | (int(direct_output) << 1)],
        )


class SILU_MUL_SHARED_BF16_K_4096_INTER(ComputeInstruction):
    def __init__(self, num_token):
        super().__init__(opcode=opcode.OP_SILU_MUL_SHARED_BF16_K_4096_INTER, args=[num_token])


class SILU_MUL_SHARED_BF16_K_2048_INTER(ComputeInstruction):
    def __init__(self, num_token):
        super().__init__(opcode=opcode.OP_SILU_MUL_SHARED_BF16_K_2048_INTER, args=[num_token])


class SILU_MUL_SHARED_BF16_K_64_SW128(ComputeInstruction):
    def __init__(self, num_token):
        super().__init__(opcode=opcode.OP_SILU_MUL_SHARED_BF16_K_64_SW128, args=[num_token])


class RMS_NORM_F16_K_4096(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_4096, args=[num_token, encode_bfloat16_u16(epsilon)])


class RMS_NORM_F16_K_4096_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, epsilon: float):
        super().__init__(opcode=opcode.OP_RMS_NORM_F16_K_4096_SMEM, args=[num_token, encode_bfloat16_u16(epsilon)])


class RMS_NORM_F16_SMEM(ComputeInstruction):
    def __init__(self, num_token: int, hidden_size: int, epsilon: float):
        if num_token <= 0 or num_token > 0xFFFF:
            raise ValueError("RMS token count must fit in a positive uint16")
        if hidden_size <= 0 or hidden_size > 0xFFFF or hidden_size % 2:
            raise ValueError("RMS hidden size must be a positive even uint16")
        super().__init__(
            opcode=opcode.OP_RMS_NORM_F16_SMEM,
            args=[num_token, hidden_size, encode_bfloat16_u16(epsilon)],
        )


def select_attention_decode_instruction(head_dim: int, direct_output: bool = False):
    if head_dim == ATTENTION_M64N64K16_F16_F32_64_64_hdim.HEAD_DIM:
        if direct_output:
            return ATTENTION_SM100_BF16_HDIM128_DIRECT
        return ATTENTION_M64N64K16_F16_F32_64_64_hdim
    if head_dim == ATTENTION_M64N64K16_F16_F32_64_64_hdim64.HEAD_DIM:
        return ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA
    raise NotImplementedError(
        f"Missing attention decode kernel support for head_dim={head_dim}. "
        "Add a dedicated opcode/instruction path before launching this model."
    )


def select_rms_glob_instruction(hidden_size: int):
    if hidden_size == 4096:
        return RMS_NORM_F16_K_4096
    raise NotImplementedError(
        f"Missing global RMS kernel support for hidden_size={hidden_size}. "
        "Add a dedicated opcode/instruction path before launching this model."
    )


def select_rms_smem_instruction(hidden_size: int):
    if hidden_size == 4096:
        return RMS_NORM_F16_K_4096_SMEM
    if hidden_size in (128, 512, 1024, 2048, 5120):
        return lambda num_token, epsilon: RMS_NORM_F16_SMEM(
            num_token, hidden_size, epsilon
        )
    raise NotImplementedError(f"Missing shared-memory RMS kernel support for hidden_size={hidden_size}. Add a dedicated opcode/instruction path before launching this model.")


def ensure_cc0_supported_hidden_size(hidden_size: int):
    row_bytes = hidden_size * 2
    if row_bytes > 0:
        return
    raise NotImplementedError(f"Missing CC0 embedding-stride support for hidden_size={hidden_size}. Parameterize the memory op before launching this model.")


class ARGMAX_PARTIAL_bf16_1152_50688_132(ComputeInstruction):
    CHUNK_SIZE = 1152
    I_STRIDE = 50688
    SMS = 132

    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_PARTIAL_bf16_1152_50688_132, args=[num_active_token])


class ARGMAX_REDUCE_bf16_1152_132(ComputeInstruction):
    CHUNK_SIZE = 1152
    SMS = 132

    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_REDUCE_bf16_1152_132, args=[num_active_token])


class ARGMAX_PARTIAL_bf16_1024_65536_128(ComputeInstruction):
    CHUNK_SIZE = 1024
    I_STRIDE = 65536
    SMS = 128

    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_PARTIAL_bf16_1024_65536_128, args=[num_active_token])


class ARGMAX_REDUCE_bf16_1024_128(ComputeInstruction):
    CHUNK_SIZE = 1024
    SMS = 128

    def __init__(self, num_active_token: int):
        super().__init__(opcode=opcode.OP_ARGMAX_REDUCE_bf16_1024_128, args=[num_active_token])


class ARGMAX_REDUCE_GLOBAL_bf16_256(ComputeInstruction):
    PARTIAL_TASKS = 256

    def __init__(self, num_active_token: int):
        super().__init__(
            opcode=opcode.OP_ARGMAX_REDUCE_GLOBAL_bf16_256,
            args=[num_active_token],
        )


class ArgmaxSmemPartialBf16(ComputeInstruction):
    """Reduce one BF16 shared-memory row range to an absolute-index record."""

    def __init__(self, rows: int, row_start: int):
        if not 1 <= rows <= 0xFFFF:
            raise ValueError("shared argmax rows must fit in a positive uint16")
        if not 0 <= row_start <= 0xFFFFFFFF:
            raise ValueError("shared argmax row start must fit in uint32")
        super().__init__(
            opcode=opcode.OP_ARGMAX_SMEM_PARTIAL_BF16,
            args=[rows, row_start & 0xFFFF, row_start >> 16],
        )


class ArgmaxSmemReduceBf16(ComputeInstruction):
    """Reduce shared absolute-index records to one int64 token."""

    def __init__(self, records: int):
        if not 1 <= records <= 0xFFFF:
            raise ValueError("shared argmax record count must fit in uint16")
        super().__init__(
            opcode=opcode.OP_ARGMAX_SMEM_REDUCE_BF16,
            args=[records],
        )


class Dummy(ComputeInstruction):
    def __init__(self, iters: int):
        super().__init__(opcode=opcode.OP_DUMMY, args=[iters])


class Copy(ComputeInstruction):
    def __init__(self, iters: int, size: int):
        assert size % 4 == 0, "Copy size must be multiple of 4 bytes (size of uint32)"
        super().__init__(opcode=opcode.OP_COPY, args=[iters, size // 4])


class Dsv4ZeroFill(ComputeInstruction):
    """Fill one allocator-owned output span with zero words."""

    def __init__(self, size: int, *, has_gate: bool = True):
        if size <= 0 or size % 4 or size > 0xFFFF * 4:
            raise ValueError("zero-fill size must be a positive uint16 word count")
        super().__init__(
            opcode=opcode.OP_DSV4_ZERO_FILL,
            args=[size // 4, int(has_gate)],
        )


class Dsv4Fp32ToBf16(ComputeInstruction):
    """Convert one shared FP32 projection shard to BF16."""

    def __init__(self, elements: int):
        if elements <= 0 or elements > 0xFFFF:
            raise ValueError("FP32-to-BF16 element count must fit uint16")
        super().__init__(opcode=opcode.OP_DSV4_FP32_TO_BF16, args=[elements])


class ProfileEvent(ComputeInstruction):
    """Record a compute-warpgroup arrival timestamp in the per-SM profile."""

    def __init__(self, event_id: int, *, wait_for_memory: bool = False):
        if not 2 <= event_id < config.num_profile_events:
            raise ValueError(
                "profile event id must leave slots 0/1 for kernel start/end "
                f"and be below {config.num_profile_events}, got {event_id}"
            )
        super().__init__(
            opcode=opcode.OP_PROFILE_EVENT,
            args=[event_id, int(wait_for_memory)],
        )


class ProfileStep(ComputeInstruction):
    """Record one compute-side step duration and its M2C wait component.

    A begin/end pair reuses one profile slot.  The runtime packs the elapsed
    nanoseconds into the low 32 bits and the M2C-wait delta into the high 32
    bits when the end marker executes.  This is diagnostic-only and never
    joins a memory warp.
    """

    def __init__(self, event_id: int, *, begin: bool):
        if not (
            config.layer_profile_event_base
            <= event_id
            < config.reload_profile_event_base
        ):
            raise ValueError(
                "step profile event id must fit the layer-profile range "
                f"[{config.layer_profile_event_base}, "
                f"{config.reload_profile_event_base}), got {event_id}"
            )
        super().__init__(
            opcode=opcode.OP_PROFILE_EVENT,
            args=[event_id, 2 if begin else 3],
        )


class ProfileLayer(ComputeInstruction):
    """Record a loop-aware compute frontier for each logical layer."""

    def __init__(self, event_base: int, event_count: int):
        if event_base < config.layer_profile_event_base or event_count <= 0:
            raise ValueError("layer profile events must follow kernel slots 0/1")
        if event_base + event_count > config.reload_profile_event_base:
            raise ValueError("layer profile events overlap reload profile slots")
        super().__init__(
            opcode=opcode.OP_PROFILE_EVENT,
            args=[event_base, 6, event_count],
        )


class ProfileAggregate(ComputeInstruction):
    """Passively aggregate one repeated compute-stage duration per SM.

    The begin slot is temporary. The aggregate slot packs total nanoseconds,
    the maximum duration in 32-ns units, and the occurrence count. No memory
    dependency or task barrier is introduced.
    """

    def __init__(self, begin_event: int, aggregate_event: int, *, begin: bool):
        for event in (begin_event, aggregate_event):
            if not (
                config.layer_profile_event_base
                <= event
                < config.reload_profile_event_base
            ):
                raise ValueError(
                    "aggregate profile event must fit the layer-profile range"
                )
        if begin_event == aggregate_event:
            raise ValueError("aggregate profile begin/output events must differ")
        super().__init__(
            opcode=opcode.OP_PROFILE_EVENT,
            args=[begin_event, 4 if begin else 5, aggregate_event],
        )


class LoopC(ComputeInstruction):
    def __init__(self, count: int, pc: int, reg: int = 0):
        assert 0 <= reg < config.num_loop_counters, (
            "reg must select a runtime compute-loop counter"
        )
        super().__init__(opcode=opcode.OP_LOOPC, args=[count, pc, reg])

    @classmethod
    def toNext(cls, ptrs, count, reg: int = 0):
        def smfunc(sm_id: int):
            pc = ptrs[sm_id]
            return cls(count, pc, reg=reg)

        return smfunc


class MemoryInstruction(Instruction):
    def __init__(
        self,
        opcode: int,
        num_slots: int,
        arg: int,
        size: int,
        cords: list[int] = [],
        address: int | None = None,
    ):
        self.opcode = opcode
        self.num_slots = num_slots
        self.arg = arg
        self.size = size
        self.set_cords(cords)
        self.annotation = {}
        if address is not None:
            addr_bytes = address.to_bytes(8, byteorder="little")
            for i in range(4):
                self.cords[i] = int.from_bytes(addr_bytes[i * 2 : i * 2 + 2], byteorder="little")

    def set_cords(self, cords: list[int]):
        assert len(cords) <= 4, "Maximum 4 cords are supported"
        self.cords = cords + [0] * (4 - len(cords))
        for i in range(4):
            assert 0 <= self.cords[i] < 2**16, "cord values must be a uint16"

    def delta(self, delta):
        if isinstance(delta, int):
            addr = cords2addr(self.cords)
            self.cords = addr2cords(addr + delta)
        elif isinstance(delta, list):
            cords = delta
            assert len(cords) <= 4, "Maximum 4 cords are supported"
            cords = cords + [0] * (4 - len(cords))

            for i in range(4):
                self.cords[i] = self.cords[i] + cords[i]
        else:
            raise ValueError("delta must be int or list[int]")

        return self

    def group(self, enable=True):
        if enable:
            self.opcode = self.opcode | 4
        return self

    def jump(self):
        self.opcode = self.opcode | 8
        return self

    def bar(self, bar_id: int | None = None):
        if bar_id is not None:
            self.opcode = self.opcode | 16
            self.num_slots = self.num_slots | (bar_id << 6)
        return self

    def writeback(self):
        self.opcode = self.opcode | 2
        return self

    def port(self, port_id: int):
        if port_id == 0:
            return self
        if port_id == 1:
            self.opcode = self.opcode | 32
            return self
        raise ValueError("Only port 0 and 1 are supported")

    def fixed_port(self, port_id: int):
        """Pin an instruction to one LDU FIFO during load balancing."""
        if port_id not in (0, 1):
            raise ValueError("Only port 0 and 1 are supported")
        encoded_port = 1 if self.opcode & 32 else 0
        if encoded_port != port_id:
            if encoded_port == 1:
                raise ValueError("Cannot repin a port-1 memory instruction to port 0")
            self.port(1)
        self.annotation["fixed_port"] = port_id
        return self

    def copy(self):
        other = MemoryInstruction(
            opcode=self.opcode,
            num_slots=self.num_slots,
            arg=self.arg,
            size=self.size,
            cords=self.cords.copy(),
        )
        other.annotation = self.annotation.copy()
        return other

    def tensor(self, tensor: torch.Tensor | None = None) -> torch.Tensor:
        if tensor is None:
            tensor = torch.empty((8,), dtype=torch.uint16)
        else:
            tensor = tensor.view(torch.uint16)
            assert tensor.numel() == 8

        tensor[0] = self.opcode
        tensor[1] = self.size
        tensor[2] = self.num_slots
        tensor[3] = self.arg
        for i in range(4):
            tensor[4 + i] = self.cords[i]
        return tensor.view(torch.uint8)

    def __repr__(self):
        flags = []
        opcode_value = self.opcode
        num_slots = self.num_slots
        if opcode_value & 8:
            flags.append("JUMP")
            opcode_value = opcode_value & (~8)
        if opcode_value & 4:
            flags.append("GROUP")
            opcode_value = opcode_value & (~4)
        if opcode_value & 16:
            bar_id = num_slots >> 6
            num_slots = num_slots & 0x3F
            flags.append(f"BAR[{bar_id}]")
            opcode_value = opcode_value & (~16)
        if opcode_value & 2:
            flags.append("WB")
            opcode_value = opcode_value & (~2)
        if opcode_value & 32:
            flags.append("PORT1")
            opcode_value = opcode_value & (~32)

        return (
            "MemoryInstruction("
            f"opcode={decode_opcode(opcode_value)}, num_slots={num_slots}, "
            f"arg={self.arg}, size={self.size}, cords={self.cords}, "
            f"flags={flags}, anno={self.annotation})"
        )


class TerminateM(MemoryInstruction):
    def __init__(self):
        super().__init__(opcode=opcode.OP_TERMINATE, num_slots=0, arg=0, size=0, address=0)


class LoopM(MemoryInstruction):
    """
    This is a combond operation, will take care of loop registers and information to
    be updated alone the loop.
    current information include:
    - cords[1]:   accumulator registers (gpr[1]) to be cleared at the beginning of each loop iteration
    - cords[2:3]: resource group shift after each loop iteration
    """

    def __init__(
        self,
        count: int,
        pc: int,
        reg=0,
        bar_shift: int = 0,
        tma_shift: int = 0,
        advance_indirect_layer: bool = False,
        resource_group=None,
    ):
        if resource_group is not None:
            tma_shift, bar_shift = resource_group.get_shift()

        assert 0 <= reg < 32, "reg must be in [0,31]"
        assert tma_shift < 2**16, "tma_shift must be less than 65536"
        assert bar_shift < 2**10, "bar_shift must be less than 1024"
        bar_shift_mask = bar_shift << 6
        super().__init__(
            opcode=opcode.OP_LOOP,
            num_slots=reg,
            arg=0,
            size=count,
            cords=[pc, int(advance_indirect_layer), bar_shift_mask, tma_shift],
        )

    @classmethod
    def toNext(cls, ptrs, count: int, **kwargs):
        def smfunc(sm_id: int):
            pc = ptrs[sm_id]
            return cls(count, pc, **kwargs)

        return smfunc


class ResetIndirectLayer(MemoryInstruction):
    """Reset the allocator-owned linear index before one layer family."""

    def __init__(self):
        super().__init__(
            opcode=opcode.OP_RESET_INDIRECT_LAYER,
            num_slots=0,
            arg=0,
            size=0,
            address=0,
        )


class CounterOffsetMemoryInstruction:
    def __init__(self, counter_reg: int, inst: MemoryInstruction, delta):
        offsets = [(counter_reg, delta)]
        if isinstance(inst, CounterOffsetMemoryInstruction):
            self.inst = inst.inst
            self.offsets = inst.offsets + offsets
        else:
            self.inst = inst
            self.offsets = offsets

    def expand_instructions(self):
        return RepeatM.offsetByCounters(self.offsets, self.inst)

    def bar(self, *args, **kwargs):
        self.inst.bar(*args, **kwargs)
        return self

    def group(self, *args, **kwargs):
        self.inst.group(*args, **kwargs)
        return self

    def jump(self, *args, **kwargs):
        self.inst.jump(*args, **kwargs)
        return self

    def port(self, *args, **kwargs):
        self.inst.port(*args, **kwargs)
        return self

    def writeback(self, *args, **kwargs):
        self.inst.writeback(*args, **kwargs)
        return self

    def copy(self):
        new_inst = CounterOffsetMemoryInstruction(self.offsets[0][0], self.inst.copy(), self.offsets[0][1])
        new_inst.offsets = self.offsets.copy()
        return new_inst

    def __getattr__(self, name):
        return getattr(self.inst, name)

    def __repr__(self):
        return f"CounterOffsetMemoryInstruction(offsets={self.offsets}, inst={self.inst!r})"


class RepeatM(MemoryInstruction):
    COUNTER_MODE_FLAG = 0x8000
    COUNT_COUNTER_MODE_FLAG = 0x4000
    ACCUMULATE_MODE_FLAG = 0x2000
    SKIP_COUNT_SHIFT = 8
    SKIP_COUNT_MASK = 0x1F
    COUNTER_REG_MASK = 0x00FF

    def __init__(
        self,
        count: int,
        reg: int = 0,
        reg_end=None,
        delta_addr: int | None = None,
        delta_cords=[],
        counter_reg: int | None = None,
        count_counter_reg: int | None = None,
        accumulate: bool = False,
    ):
        if reg_end is None:
            reg_end = reg + 1
        assert 0 <= reg < 32, "reg must be in [0,31]"
        assert 0 <= reg_end <= 32, "reg_end must be in [0,32]"
        assert reg_end > reg, "reg_end must be greater than reg"
        arg = 0
        encoded_counter_reg = None
        if counter_reg is not None:
            assert 0 <= counter_reg <= self.COUNTER_REG_MASK, "counter_reg must fit in the REPEAT counter field"
            encoded_counter_reg = counter_reg
            arg |= self.COUNTER_MODE_FLAG
        if count_counter_reg is not None:
            assert 0 <= count_counter_reg <= self.COUNTER_REG_MASK, "count_counter_reg must fit in the REPEAT counter field"
            if encoded_counter_reg is not None and encoded_counter_reg != count_counter_reg:
                raise ValueError("REPEAT can only encode one counter register")
            encoded_counter_reg = count_counter_reg
            arg |= self.COUNT_COUNTER_MODE_FLAG
        if accumulate:
            arg |= self.ACCUMULATE_MODE_FLAG
        if encoded_counter_reg is not None:
            arg |= encoded_counter_reg
        super().__init__(
            opcode=opcode.OP_REPEAT,
            num_slots=(reg_end << 8) | reg,
            arg=arg,
            size=count,
            address=delta_addr,
            cords=delta_cords,
        )

    @classmethod
    def byCounter(cls, counter_reg: int, *steps):
        insts = []
        if len(steps) == 0:
            return insts

        regcords = []
        for i, (inst, delta) in enumerate(steps):
            if isinstance(delta, list):
                cords = delta
            elif isinstance(delta, int):
                cords = addr2cords(delta)
            else:
                raise ValueError("delta must be int or list[int]")

            if len(regcords) > 0 and regcords[-1][-1] == cords:
                regcords[-1][1] = i + 1
            else:
                regcords.append([i, i + 1, cords])

        insts.append(cls(1, reg=0, reg_end=32, delta_cords=[0], counter_reg=counter_reg))
        for reg_start, reg_end, delta_cords in regcords:
            insts += [cls(1, reg=reg_start, reg_end=reg_end, delta_cords=delta_cords, counter_reg=counter_reg)]
        for inst, _ in steps:
            insts.append(inst)
        insts[-1].jump()
        return insts

    @classmethod
    def offsetByCounter(cls, counter_reg: int, inst, delta):
        return cls.byCounter(counter_reg, (inst, delta))

    @classmethod
    def offsetByCounters(cls, counter_offsets, inst):
        offsets = [(counter_reg, delta) for counter_reg, delta in counter_offsets]
        if len(offsets) == 0:
            return inst
        target_reg = len(offsets)
        assert target_reg < 32, "offsetByCounters can combine at most 31 counter offsets"

        insts = [cls(1, reg=0, reg_end=32, delta_cords=[0])]
        for counter_reg, delta in offsets:
            if isinstance(delta, list):
                delta_cords = delta
                delta_addr = None
            elif isinstance(delta, int):
                delta_cords = []
                delta_addr = delta
            else:
                raise ValueError("delta must be int or list[int]")
            insts.append(cls(
                1,
                reg=target_reg,
                reg_end=target_reg + 1,
                delta_addr=delta_addr,
                delta_cords=delta_cords,
                counter_reg=counter_reg,
                accumulate=True,
            ))
        insts.append(inst)
        insts[-1].jump()
        return insts

    @classmethod
    def offsetWindowByCounters(cls, counter_offsets, *instructions):
        """Apply one counter-derived address offset to adjacent commands.

        The allocator's repeat registers correspond to the following memory
        instruction lanes.  Seeding one complete window therefore preserves
        adjacency between its final commands, which is required by LDU-local
        retained-ring handoff rewriting.
        """

        offsets = [
            (counter_reg, delta) for counter_reg, delta in counter_offsets
        ]
        instructions = tuple(instructions)
        if not instructions:
            return []
        if not offsets:
            return list(instructions)
        target_start = len(offsets)
        target_end = target_start + len(instructions)
        assert target_end <= 32, (
            "offsetWindowByCounters counter controls and target window must "
            "fit 32 allocator lanes"
        )

        expanded = [cls(1, reg=0, reg_end=32, delta_cords=[0])]
        for counter_reg, delta in offsets:
            if isinstance(delta, list):
                delta_cords = delta
                delta_addr = None
            elif isinstance(delta, int):
                delta_cords = []
                delta_addr = delta
            else:
                raise ValueError("delta must be int or list[int]")
            expanded.append(
                cls(
                    1,
                    reg=target_start,
                    reg_end=target_end,
                    delta_addr=delta_addr,
                    delta_cords=delta_cords,
                    counter_reg=counter_reg,
                    accumulate=True,
                )
            )
        expanded.extend(instructions)
        # The fixed resident commands below deliberately bypass allocation,
        # but they still form a normal allocator repeat window.  Mark its
        # final command so the runtime retires the window before decoding the
        # following instruction.
        expanded[-1].jump()
        return expanded

    @classmethod
    def onSync(cls, bar_inst_offset: int, bar_id: int | None, count: int, *steps, asyncPort: bool = True):
        if bar_id is None:
            return cls.on(count, *steps)

        port = 1 if asyncPort else 0

        insts = []
        for i, (inst, _) in enumerate(steps):
            if i == bar_inst_offset:
                inst.port(port)
            new_inst = inst.copy()
            if i == bar_inst_offset:
                new_inst.bar(bar_id)
            insts.append(new_inst)
        insts += cls.on1(count, *steps)
        return insts

    @classmethod
    def on1(cls, count: int, *steps):
        assert count > 0, "count must be greater than 0 to use on1"
        new_steps = []
        for inst, delta in steps:
            new_steps.append((inst.delta(delta), delta))
        return cls.on(count - 1, *new_steps)

    @classmethod
    def on(cls, count: int, *steps, count_counter_reg: int | None = None):
        insts = []
        if len(steps) == 0:
            return insts
        if count == 0 and count_counter_reg is None:
            return []

        regcords = []
        for i, (inst, delta) in enumerate(steps):
            if isinstance(delta, list):
                cords = delta
            elif isinstance(delta, int):
                cords = addr2cords(delta)
            else:
                raise ValueError("delta must be int or list[int]")

            if len(regcords) > 0 and regcords[-1][-1] == cords:
                regcords[-1][1] = i + 1
            else:
                regcords.append([i, i + 1, cords])

        if count > 1 or count_counter_reg is not None:
            for reg_start, reg_end, delta_cords in regcords:
                insts += [cls(0, reg=reg_start, reg_end=reg_end, delta_cords=delta_cords, count_counter_reg=count_counter_reg)]
            insts[-1].size = count
            assert len(steps) <= cls.SKIP_COUNT_MASK, "dynamic repeat window is too large to encode"
            insts[-1].arg |= len(steps) << cls.SKIP_COUNT_SHIFT

        for inst, _ in steps:
            insts.append(inst)

        if count > 1 or count_counter_reg is not None:
            insts[-1].jump()
        return insts


class RawAddress(MemoryInstruction):
    def __init__(
        self,
        tensor: torch.Tensor,
        slot_id: int,
        *,
        profile_event: int | None = None,
    ):
        assert tensor.device.type == "cuda"
        address = tensor.data_ptr()

        min_slot = config.num_slots
        max_slot = config.num_slots + config.num_special_slots - 1
        assert min_slot <= slot_id <= max_slot, (
            f"slot_id must be in the range of special slots [{min_slot}, {max_slot}]"
        )
        if profile_event is not None and not (
            0 <= profile_event < config.track_profile_event_base - 2
        ):
            raise ValueError("raw-address profile event exceeds diagnostic slots")
        super().__init__(
            opcode=opcode.OP_ALLOC_WB_RAW_ADDRESS,
            num_slots=slot_id,
            arg=slot_id,
            # RawAddress has no byte transfer. A nonzero size is therefore a
            # track-only event-base encoding consumed by allocator/STU.
            size=0 if profile_event is None else profile_event + 1,
            address=address,
        )


class RoutedTmaLoad1D(MemoryInstruction):
    """Resolve one routed field in LDU and copy it into shared memory."""

    ROUTE_COUNT = 6
    ROUTE_BITS = 3
    MAX_POINTER_FIELD = (1 << (16 - ROUTE_BITS)) - 1
    HEADER_BYTES = 48

    def __init__(
        self,
        routing_state: torch.Tensor,
        route_rank: int,
        pointer_field: int,
        bytes: int,
    ):
        assert routing_state.device.type == "cuda"
        if not routing_state.is_contiguous():
            raise ValueError("routing_state must be contiguous")
        if routing_state.numel() * routing_state.element_size() < self.HEADER_BYTES:
            raise ValueError("routing_state must contain the 32-byte routing header")
        if not 0 <= route_rank < self.ROUTE_COUNT:
            raise ValueError("route_rank must be in [0, 6)")
        if not 0 <= pointer_field <= self.MAX_POINTER_FIELD:
            raise ValueError("pointer_field must fit in 13 bits")
        if bytes <= 0 or bytes > 0xFFFF or bytes % 16:
            raise ValueError("routed TMA load size must be a 16-byte-aligned uint16")
        num_slots = bytes2slots(bytes)
        if num_slots > config.num_slots:
            raise ValueError("routed TMA load exceeds the shared-slot arena")
        super().__init__(
            opcode=opcode.OP_ALLOC_ROUTED_TMA_LOAD_1D,
            num_slots=num_slots,
            arg=(pointer_field << self.ROUTE_BITS) | route_rank,
            size=bytes,
            address=routing_state.data_ptr(),
        )


class RoutedTmaLoadBase1D(RoutedTmaLoad1D):
    """Resolve/load a routed base and cache its address in LDU register zero."""

    ADDRESS_REGISTER = 0

    def __init__(
        self,
        routing_state: torch.Tensor,
        route_rank: int,
        pointer_field: int,
        bytes: int,
    ):
        super().__init__(routing_state, route_rank, pointer_field, bytes)
        self.opcode = opcode.OP_ALLOC_ROUTED_TMA_LOAD_BASE_1D


class TmaLoadAddressReg1D(MemoryInstruction):
    """Load from one port-local routed base plus a compile-time byte offset."""

    def __init__(self, address_register: int, offset: int, bytes: int):
        if address_register != RoutedTmaLoadBase1D.ADDRESS_REGISTER:
            raise ValueError("routed LDU currently exposes address register zero")
        if offset < 0:
            raise ValueError("LDU address-register offset must be non-negative")
        if bytes <= 0 or bytes > 0xFFFF or bytes % 16:
            raise ValueError(
                "address-register TMA size must be a 16-byte-aligned uint16"
            )
        super().__init__(
            opcode=opcode.OP_ALLOC_TMA_LOAD_ADDRESS_REG_1D,
            num_slots=bytes2slots(bytes),
            arg=address_register,
            size=bytes,
            address=offset,
        )


class IndexedTmaLoad1D(MemoryInstruction):
    """Resolve one runtime row index in LDU and load the row into shared."""

    RECORD_BYTES = 24

    def __init__(self, indexed_record: torch.Tensor, bytes: int):
        if indexed_record.device.type != "cuda" or not indexed_record.is_contiguous():
            raise ValueError("indexed load record must be a contiguous CUDA tensor")
        if indexed_record.numel() * indexed_record.element_size() < self.RECORD_BYTES:
            raise ValueError("indexed load record must contain three uint64 values")
        if bytes <= 0 or bytes > 0xFFFF or bytes % 16:
            raise ValueError("indexed TMA load size must be a 16-byte-aligned uint16")
        super().__init__(
            opcode=opcode.OP_ALLOC_INDEXED_TMA_LOAD_1D,
            num_slots=bytes2slots(bytes),
            arg=0,
            size=bytes,
            address=indexed_record.data_ptr(),
        )


def _validate_indirect_pointer_entry(pointer_entry):
    if pointer_entry.device.type != "cuda" or not pointer_entry.is_contiguous():
        raise ValueError("indirect pointer entries must be contiguous CUDA tensors")
    if pointer_entry.dtype != torch.int64 or pointer_entry.numel() < 1:
        raise ValueError("indirect pointer entries must contain at least one int64")


def _indirect_layer_opcode(base_opcode: int, layer_indexed: bool) -> int:
    if not layer_indexed:
        return base_opcode
    variants = {
        opcode.OP_ALLOC_INDIRECT_TMA_LOAD_1D:
            opcode.OP_ALLOC_LAYER_TMA_LOAD_1D,
        opcode.OP_ALLOC_INDIRECT_LDU_LOAD_1D:
            opcode.OP_ALLOC_LAYER_LDU_LOAD_1D,
        opcode.OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_1D:
            opcode.OP_ALLOC_LAYER_ROUTED_TMA_LOAD_1D,
        opcode.OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_BASE_1D:
            opcode.OP_ALLOC_LAYER_ROUTED_TMA_LOAD_BASE_1D,
        opcode.OP_ALLOC_INDIRECT_INDEXED_TMA_LOAD_1D:
            opcode.OP_ALLOC_LAYER_INDEXED_TMA_LOAD_1D,
    }
    return variants[base_opcode]


class IndirectTmaLoad1D(MemoryInstruction):
    """Resolve one source pointer through HBM, then TMA-load shared slots."""

    def __init__(self, pointer_entry, bytes: int, *, layer_indexed=False):
        _validate_indirect_pointer_entry(pointer_entry)
        if bytes <= 0 or bytes > 0xFFFF or bytes % 16:
            raise ValueError("indirect TMA load size must be a 16-byte-aligned uint16")
        super().__init__(
            opcode=_indirect_layer_opcode(
                opcode.OP_ALLOC_INDIRECT_TMA_LOAD_1D, layer_indexed
            ),
            num_slots=bytes2slots(bytes),
            arg=0,
            size=bytes,
            address=pointer_entry.data_ptr(),
        )


class IndirectLduLoad1D(MemoryInstruction):
    """Resolve one source pointer through HBM for an arbitrary-size LDU copy."""

    def __init__(self, pointer_entry, bytes: int, *, layer_indexed=False):
        _validate_indirect_pointer_entry(pointer_entry)
        if bytes <= 0 or bytes > 0xFFFF:
            raise ValueError("indirect LDU load size must be a positive uint16")
        super().__init__(
            opcode=_indirect_layer_opcode(
                opcode.OP_ALLOC_INDIRECT_LDU_LOAD_1D, layer_indexed
            ),
            num_slots=bytes2slots(bytes),
            arg=0,
            size=bytes,
            address=pointer_entry.data_ptr(),
        )


class IndirectRoutedTmaLoad1D(MemoryInstruction):
    """Resolve fixed route IDs plus one layer pointer table in LDU."""

    def __init__(
        self,
        state_descriptor,
        route_rank: int,
        pointer_field: int,
        bytes: int,
        *,
        layer_indexed=False,
    ):
        _validate_indirect_pointer_entry(state_descriptor)
        if state_descriptor.numel() < 2:
            raise ValueError("indirect routed state descriptor needs two int64 words")
        if not 0 <= route_rank < RoutedTmaLoad1D.ROUTE_COUNT:
            raise ValueError("route_rank must be in [0, 6)")
        if not 0 <= pointer_field <= RoutedTmaLoad1D.MAX_POINTER_FIELD:
            raise ValueError("pointer_field must fit in 13 bits")
        if bytes <= 0 or bytes > 0xFFFF or bytes % 16:
            raise ValueError("indirect routed TMA load size must be aligned")
        super().__init__(
            opcode=_indirect_layer_opcode(
                opcode.OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_1D,
                layer_indexed,
            ),
            num_slots=bytes2slots(bytes),
            arg=(pointer_field << RoutedTmaLoad1D.ROUTE_BITS) | route_rank,
            size=bytes,
            address=state_descriptor.data_ptr(),
        )


class IndirectRoutedTmaLoadBase1D(IndirectRoutedTmaLoad1D):
    """Resolve a layer-routed tile and cache its base in the issuing LDU."""

    def __init__(
        self,
        state_descriptor,
        route_rank: int,
        pointer_field: int,
        bytes: int,
        *,
        layer_indexed=False,
    ):
        super().__init__(
            state_descriptor,
            route_rank,
            pointer_field,
            bytes,
            layer_indexed=False,
        )
        self.opcode = _indirect_layer_opcode(
            opcode.OP_ALLOC_INDIRECT_ROUTED_TMA_LOAD_BASE_1D,
            layer_indexed,
        )


class IndirectIndexedTmaLoad1D(MemoryInstruction):
    """Resolve an indexed-load record pointer before selecting its runtime row."""

    def __init__(self, record_pointer_entry, bytes: int, *, layer_indexed=False):
        _validate_indirect_pointer_entry(record_pointer_entry)
        if bytes <= 0 or bytes > 0xFFFF or bytes % 16:
            raise ValueError("indirect indexed TMA load size must be aligned")
        super().__init__(
            opcode=_indirect_layer_opcode(
                opcode.OP_ALLOC_INDIRECT_INDEXED_TMA_LOAD_1D,
                layer_indexed,
            ),
            num_slots=bytes2slots(bytes),
            arg=0,
            size=bytes,
            address=record_pointer_entry.data_ptr(),
        )


def indirect_1d_from(
    inst: MemoryInstruction, pointer_entry, *, layer_indexed=False
):
    """Replace one direct 1D memory instruction while preserving its flags."""
    base_opcode = inst.opcode & ~((1 << 6) - 1)
    builders = {
        opcode.OP_ALLOC_TMA_LOAD_1D & ~((1 << 6) - 1): IndirectTmaLoad1D,
        opcode.OP_ALLOC_LDU_LOAD_1D & ~((1 << 6) - 1): IndirectLduLoad1D,
        opcode.OP_ALLOC_ROUTED_TMA_LOAD_1D & ~((1 << 6) - 1): IndirectRoutedTmaLoad1D,
        opcode.OP_ALLOC_ROUTED_TMA_LOAD_BASE_1D & ~((1 << 6) - 1): IndirectRoutedTmaLoadBase1D,
        opcode.OP_ALLOC_INDEXED_TMA_LOAD_1D & ~((1 << 6) - 1): IndirectIndexedTmaLoad1D,
    }
    try:
        builder = builders[base_opcode]
    except KeyError:
        raise ValueError(
            f"memory opcode {base_opcode:#x} has no indirect 1D form"
        ) from None
    if builder in (IndirectRoutedTmaLoad1D, IndirectRoutedTmaLoadBase1D):
        replacement = builder(
            pointer_entry,
            inst.arg & 0x7,
            inst.arg >> 3,
            inst.size,
            layer_indexed=layer_indexed,
        )
    else:
        replacement = builder(
            pointer_entry, inst.size, layer_indexed=layer_indexed
        )
    replacement.opcode |= inst.opcode & ((1 << 6) - 1)
    replacement.num_slots = inst.num_slots
    replacement.arg = inst.arg
    replacement.annotation = inst.annotation.copy()
    return replacement


class LduReloadBarriers(MemoryInstruction):
    """Drain both LDU ports and restore one loop-local barrier range."""

    FIRST_BAR_MASK = (1 << 10) - 1
    LOOP_REG_SHIFT = 10
    SKIP_FINAL_LOOP = 1 << 14
    RESET_MXFP_RESIDENT = 1 << 15

    def __init__(
        self,
        bar_source: torch.Tensor,
        first_bar: int,
        count: int,
        special_slot: int,
        reset_mxfp_resident: bool = False,
        skip_final_loop_reg: int | None = None,
    ):
        if bar_source.device.type != "cuda" or not bar_source.is_contiguous():
            raise ValueError("barrier reload source must be a contiguous CUDA tensor")
        if first_bar < 0 or count <= 0:
            raise ValueError("barrier reload range must be non-empty")
        if first_bar > self.FIRST_BAR_MASK:
            raise ValueError("barrier reload start must fit its low byte")
        if count > 0xFFFF:
            raise ValueError("barrier reload count must fit uint16")
        if first_bar + count > config.max_bars - 2:
            raise ValueError("barrier reload range overlaps runtime handshake counters")
        if (
            bar_source.numel() * bar_source.element_size()
            < 4 * (first_bar + max(count, 1))
        ):
            raise ValueError("barrier reload source does not cover the requested range")
        if not 0 <= special_slot < config.num_special_slots:
            raise ValueError("barrier reload requires one special slot")
        if skip_final_loop_reg is None:
            loop_reg = 0
        else:
            loop_reg = skip_final_loop_reg
            if not 0 <= loop_reg < config.num_loop_counters:
                raise ValueError("terminal reload loop register is invalid")
        super().__init__(
            opcode=opcode.OP_LDU_RELOAD_BARRIERS,
            num_slots=config.num_slots + special_slot,
            arg=(
                first_bar
                | (loop_reg << self.LOOP_REG_SHIFT)
                | (self.SKIP_FINAL_LOOP if skip_final_loop_reg is not None else 0)
                | (self.RESET_MXFP_RESIDENT if reset_mxfp_resident else 0)
            ),
            size=count,
            address=bar_source.data_ptr(),
        )


class LduAsyncReloadBarriers(MemoryInstruction):
    """Restore one disjoint barrier-bank slice and publish its completion."""

    FIRST_BAR_MASK = (1 << 10) - 1
    SKIP_INITIAL_LOOP = 1 << 12
    SHIFT_TARGET = 1 << 13
    BANK_READY_COMPLETION = 1 << 14
    BANK_READY_LEADER = 1 << 15

    def __init__(
        self,
        bar_source: torch.Tensor,
        first_bar: int,
        count: int,
        input_bar: int,
        special_slot: int,
        *,
        skip_initial_loop: bool = False,
        shift_target: bool = False,
        bank_ready_completion: bool = False,
        bank_ready_leader: bool = False,
    ):
        if bar_source.device.type != "cuda" or not bar_source.is_contiguous():
            raise ValueError("async barrier reload source must be contiguous CUDA")
        if first_bar < 0 or count <= 0:
            raise ValueError("async barrier reload range must be non-empty")
        if first_bar > self.FIRST_BAR_MASK:
            raise ValueError("async barrier reload start must fit its low ten bits")
        if count >= (1 << 6):
            raise ValueError("one async barrier reload slice must fit six bits")
        if not 0 <= input_bar < config.max_bars - 2:
            raise ValueError("async barrier reload input dependency is invalid")
        if first_bar + count > config.max_bars - 2:
            raise ValueError(
                "async barrier reload range overlaps runtime handshake counters"
            )
        if bar_source.numel() * bar_source.element_size() < 4 * (first_bar + count):
            raise ValueError("async barrier reload source does not cover its range")
        if not 0 <= special_slot < config.num_special_slots:
            raise ValueError("async barrier reload requires one special slot")
        if bank_ready_leader and not bank_ready_completion:
            raise ValueError("async bank-ready leader requires bank completion")
        super().__init__(
            opcode=opcode.OP_LDU_ASYNC_RELOAD_BARRIERS,
            num_slots=config.num_slots + special_slot,
            arg=(
                first_bar
                | (self.SKIP_INITIAL_LOOP if skip_initial_loop else 0)
                | (self.SHIFT_TARGET if shift_target else 0)
                | (
                    self.BANK_READY_COMPLETION
                    if bank_ready_completion
                    else 0
                )
                | (self.BANK_READY_LEADER if bank_ready_leader else 0)
            ),
            size=count | (input_bar << 6),
            address=bar_source.data_ptr() + 4 * first_bar,
        )
        # The LDU performs the global counter publication itself after all
        # slice stores. Sequential composition attaches the join barrier.
        self.writeback()
        self.annotation["input_independent_writeback"] = True


class LduWaitBarrier(MemoryInstruction):
    """Wait for one monotonic bank-ready generation on both LDU ports."""

    def __init__(self, generation_reg: int, special_slot: int):
        if not 0 <= generation_reg < config.num_loop_counters:
            raise ValueError("LDU bank wait generation register is invalid")
        if not 0 <= special_slot < config.num_special_slots:
            raise ValueError("LDU bank wait requires one special slot")
        super().__init__(
            opcode=opcode.OP_LDU_WAIT_BARRIER,
            num_slots=config.num_slots + special_slot,
            arg=generation_reg,
            size=0,
            address=0,
        )


class LduProfileLayer(MemoryInstruction):
    """Record one completed-layer frontier without occupying a shared mailbox."""

    def __init__(
        self,
        event_base: int,
        event_count: int,
    ):
        if event_base < config.layer_profile_event_base or event_count <= 0:
            raise ValueError("layer profile events must follow kernel slots 0/1")
        if event_base + event_count > config.reload_profile_event_base:
            raise ValueError("layer profile events overlap reload profile slots")
        super().__init__(
            opcode=opcode.OP_LDU_PROFILE_LAYER,
            num_slots=0,
            arg=event_base,
            size=event_count,
            address=0,
        )


class IssueBarrier(MemoryInstruction):
    def __init__(self, bar: int):
        super().__init__(opcode=opcode.OP_ISSUE_BARRIER, num_slots=0, arg=0, size=0, address=0)
        self.bar(bar)


class CC0(MemoryInstruction):
    def __init__(self, tokens: torch.Tensor, idx: int, hidden_size: int = 4096, dtype_size: int = 2):
        addr = get_tensor_address(tokens[idx])
        row_bytes = hidden_size * dtype_size
        if row_bytes <= 0:
            raise ValueError(f"CC0 requires a positive embedding row size in bytes, got {row_bytes}")
        if (row_bytes & (row_bytes - 1)) == 0:
            shift = row_bytes.bit_length() - 1
            super().__init__(opcode=opcode.OP_CC0, num_slots=0, arg=shift, size=0, address=addr)
            return
        super().__init__(opcode=opcode.OP_CC0_ROW_BYTES, num_slots=0, arg=0, size=row_bytes, address=addr)


class RegStore(MemoryInstruction):
    def __init__(self, reg_id: int, shape: torch.Tensor = None, size=None):
        if size is None:
            assert shape is not None, "Either shape or size must be provided for RegStore"
            size = shape.numel() * shape.element_size()
        assert size is not None, "Size must be provided for RegStore"

        num_slots = bytes2slots(size)
        super().__init__(opcode=opcode.OP_ALLOC_WB_REG_STORE, num_slots=num_slots, arg=0, size=reg_id, address=0)

        self.mode = "reduce"

    def cord(self, *args):
        return self


class RegLoad(MemoryInstruction):
    def __init__(self, reg_id: int, slot_id=None):
        if slot_id is None:
            slot_id = reg_id
        assert slot_id < config.num_special_slots, (
            f"slot_id must be less than {config.num_special_slots} for RegLoad"
        )
        num_slots = config.num_slots + slot_id
        super().__init__(opcode=opcode.OP_ALLOC_REG_LOAD, num_slots=num_slots, arg=0, size=reg_id, address=0)

    def cord(self, *args):
        return self


class TmaLoad1D(MemoryInstruction):
    def __init__(self, src: torch.Tensor, bytes: int | None = None, numSlots: int | None = None):
        address = get_tensor_address(src)
        if bytes is None:
            bytes = src.numel() * src.element_size()
        if numSlots is None:
            numSlots = bytes2slots(bytes)
        super().__init__(
            opcode=opcode.OP_ALLOC_TMA_LOAD_1D,
            num_slots=numSlots,
            arg=0,
            size=bytes,
            address=address,
        )

    def cord(self, addr):
        new_inst = copy.copy(self)
        new_inst.delta(addr)
        return new_inst


class TmaLoadPair1D(MemoryInstruction):
    """Load two independent sources into one contiguous shared allocation."""

    def __init__(
        self,
        address_plan: torch.Tensor,
        *,
        first_bytes: int,
        second_bytes: int,
    ):
        if (
            address_plan.device.type != "cuda"
            or address_plan.dtype != torch.int64
            or address_plan.numel() != 2
            or not address_plan.is_contiguous()
        ):
            raise ValueError(
                "paired 1D TMA address plan must be contiguous CUDA int64[2]"
            )
        first_bytes = int(first_bytes)
        second_bytes = int(second_bytes)
        total_bytes = first_bytes + second_bytes
        if (
            first_bytes <= 0
            or second_bytes <= 0
            or first_bytes % 16
            or second_bytes % 16
            or total_bytes > 0xFFFF
        ):
            raise ValueError(
                "paired 1D TMA byte counts must be positive/aligned16 and fit uint16"
            )
        super().__init__(
            opcode=opcode.OP_ALLOC_TMA_LOAD_PAIR_1D,
            num_slots=bytes2slots(total_bytes),
            arg=first_bytes,
            size=total_bytes,
            address=get_tensor_address(address_plan),
        )
        self.address_plan = address_plan


class TmaLoadMxfpCoupledStream(MemoryInstruction):
    """Produce one common MX data/scale stream through the LDUs.

    Resident-FFN kinds retain their fixed-area special-mailbox contract.  The
    generic MXFP8 kind instead allocates ``area_slots`` from the normal arena,
    publishes one lease to compute, and dispatches the same plan to both LDUs.
    The Python builder may still chain adjacent fixed-area commands with the
    same ``area_id`` and port so their state is handed over locally.
    """

    LINEAR1 = 0
    DOWN_WEIGHT = 1
    DOWN_ACTIVATION = 2
    FP8_GEMV = 3
    TMA_RING = 4
    KIND_MASK = 0x000F
    STAGES_SHIFT = 4
    STAGES_MASK = 0x00F0
    LOCAL_CHAIN = 0x0100
    DYNAMIC_EXPERT = 0x0200
    DOWN_ONLY = 0x0400
    PHASE_BASE_SHIFT = 9
    MAX_PHASE_BASE = 0x7F
    FP8_STAGES = 2
    FP8_AREA_SLOTS = 17
    LAYER_INDEXED_SIZE = 0x8000
    STREAM_LENGTH_MASK = LAYER_INDEXED_SIZE - 1

    def __init__(
        self,
        plan_address: int,
        *,
        kind: int,
        stages: int,
        area_slots: int,
        area_id: int,
        mailbox: int | None = None,
        port: int | None = None,
        stream_length: int | None = None,
        phase_base: int = 0,
        layer_indexed: bool = False,
        dynamic_expert: bool = False,
        down_only: bool = False,
    ):
        if plan_address <= 0 or plan_address >= 1 << 64:
            raise ValueError("coupled-stream plan address must fit uint64")
        if kind not in (
            self.LINEAR1,
            self.DOWN_WEIGHT,
            self.DOWN_ACTIVATION,
            self.FP8_GEMV,
            self.TMA_RING,
        ):
            raise ValueError("unknown MXFP coupled-stream kind")
        if not 1 <= int(stages) <= 0xF:
            raise ValueError("coupled-stream pipeline depth must fit four bits")
        if not 1 <= int(area_slots) <= 0xFFFF:
            raise ValueError("coupled-stream area size must fit uint16")
        if not 0 <= int(area_id) <= 0xFFFF:
            raise ValueError("coupled-stream area id must fit uint16")
        if kind in (self.FP8_GEMV, self.TMA_RING):
            if dynamic_expert or down_only:
                raise ValueError(
                    "dynamic expert/down-only flags are reserved for resident FFN streams"
                )
            if kind == self.TMA_RING:
                raise ValueError(
                    "use TmaLoadInternalRingStream for descriptor-driven plans"
                )
            if int(stages) != self.FP8_STAGES:
                raise ValueError("coupled FP8 stream requires two stages")
            if int(area_slots) != self.FP8_AREA_SLOTS:
                raise ValueError("coupled FP8 stream requires a 17-slot ring")
            if mailbox is not None or port is not None:
                raise ValueError(
                    "allocator-owned coupled FP8 stream dispatches both LDUs"
                )
            if not 1 <= int(stream_length or 0) <= self.STREAM_LENGTH_MASK:
                raise ValueError(
                    "coupled FP8 stream length must fit the indexed uint15 field"
                )
            if not 0 <= int(phase_base) <= self.MAX_PHASE_BASE:
                raise ValueError(
                    "coupled FP8 memory phase base must fit seven bits"
                )
            super().__init__(
                opcode=opcode.OP_TMA_LOAD_MX_COUPLED_STREAM | 1,
                num_slots=int(area_slots),
                arg=(
                    int(kind)
                    | (int(stages) << self.STAGES_SHIFT)
                    | (int(phase_base) << self.PHASE_BASE_SHIFT)
                ),
                size=(
                    int(stream_length)
                    | (self.LAYER_INDEXED_SIZE if layer_indexed else 0)
                ),
                address=plan_address,
            )
            self.annotation["coupled_stream_area"] = int(area_id)
            self.annotation["coupled_stream_kind"] = int(kind)
            self.annotation["coupled_stream_allocator_lease"] = True
            self.annotation["coupled_stream_dual_port"] = True
            self.annotation["coupled_stream_layer_indexed"] = bool(
                layer_indexed
            )
            return
        if layer_indexed:
            raise ValueError(
                "only allocator-owned coupled FP8 plans may be layer indexed"
            )
        if mailbox is None or not 0 <= int(mailbox) < config.num_special_slots:
            raise ValueError("coupled-stream mailbox is outside special slots")
        if port not in (0, 1):
            raise ValueError("fixed coupled stream requires one LDU port")
        super().__init__(
            opcode=opcode.OP_TMA_LOAD_MX_COUPLED_STREAM,
            num_slots=config.num_slots + int(mailbox),
            arg=(
                int(kind)
                | (int(stages) << self.STAGES_SHIFT)
                | (self.DYNAMIC_EXPERT if dynamic_expert else 0)
                | (self.DOWN_ONLY if down_only else 0)
            ),
            size=int(area_slots),
            address=plan_address,
        )
        self.annotation["coupled_stream_area"] = int(area_id)
        self.annotation["coupled_stream_kind"] = int(kind)
        self.annotation["coupled_stream_mailbox"] = int(mailbox)
        self.annotation["coupled_stream_dynamic_expert"] = bool(
            dynamic_expert
        )
        self.annotation["coupled_stream_down_only"] = bool(down_only)
        self.fixed_port(int(port))

    def local_chain_source(self):
        inst = self.copy()
        inst.arg |= self.LOCAL_CHAIN
        inst.annotation["coupled_stream_local_chain"] = "source"
        return inst


def build_internal_ring_tma_plan(
    *,
    device,
    stage_bytes: int,
    lanes: dict[int, dict],
    cache_evict_first: bool = True,
) -> torch.Tensor:
    """Pack one two-port descriptor plan for the generic internal-ring LDU.

    Each lane dictionary contains ``descriptor_index``, ``rank``,
    ``transaction_bytes`` and optional issue/destination/coordinate fields.
    Coordinates and both deltas are four-element signed sequences; unused
    dimensions remain zero.  The resulting CUDA tensor is retained by the
    instruction object, so callers need no separate lifetime bookkeeping.
    """

    if not 16 <= int(stage_bytes) <= 0x7FFFFFFF:
        raise ValueError("internal-ring stage_bytes must fit positive int32")
    if any(port not in (0, 1) for port in lanes):
        raise ValueError("internal-ring plan supports only LDU ports 0 and 1")
    words = torch.zeros(36, dtype=torch.int32)
    words[0] = int(stage_bytes)
    words[1] = 1 if cache_evict_first else 0
    for port, lane in lanes.items():
        base = 4 + int(port) * 16
        descriptor_index = int(lane["descriptor_index"])
        rank = int(lane["rank"])
        issue_count = int(lane.get("issue_count", 1))
        transaction_bytes = int(lane["transaction_bytes"])
        if not 0 <= descriptor_index <= 0xFFFF:
            raise ValueError("internal-ring descriptor index must fit uint16")
        if not 1 <= rank <= 4:
            raise ValueError("internal-ring TMA rank must be in [1,4]")
        if not 1 <= issue_count <= 0xFF:
            raise ValueError("internal-ring issue_count must fit uint8")
        if transaction_bytes <= 0:
            raise ValueError("internal-ring transaction_bytes must be positive")
        words[base] = (
            descriptor_index | (rank << 16) | (issue_count << 24)
        )
        words[base + 1] = transaction_bytes
        words[base + 2] = int(lane.get("destination_offset", 0))
        words[base + 3] = int(lane.get("destination_issue_stride", 0))
        for field, offset in (
            ("coordinates", 4),
            ("iteration_delta", 8),
            ("issue_delta", 12),
        ):
            values = tuple(int(value) for value in lane.get(field, ()))
            if len(values) > 4:
                raise ValueError(f"internal-ring {field} has more than 4 values")
            if values:
                words[base + offset : base + offset + len(values)] = torch.tensor(
                    values, dtype=torch.int32
                )
    return words.to(device=device, non_blocking=False)


class TmaLoadInternalRingStream(MemoryInstruction):
    """Lease an allocator area once and fill its stages from up to two LDUs."""

    KIND = TmaLoadMxfpCoupledStream.TMA_RING
    STAGES_SHIFT = TmaLoadMxfpCoupledStream.STAGES_SHIFT
    PORT_MASK_SHIFT = 9

    def __init__(
        self,
        plan: torch.Tensor,
        *,
        stages: int,
        stage_bytes: int,
        area_slots: int,
        area_id: int,
        stream_length: int,
        port_mask: int,
    ):
        if (
            plan.dtype != torch.int32
            or plan.device.type != "cuda"
            or not plan.is_contiguous()
            or plan.numel() != 36
        ):
            raise ValueError("internal-ring plan must be contiguous CUDA int32[36]")
        if int(stages) not in (1, 2):
            raise ValueError("internal-ring stream supports one or two stages")
        if not 1 <= int(stream_length) <= 0xFFFF:
            raise ValueError("internal-ring stream length must fit uint16")
        if int(port_mask) not in (1, 2, 3):
            raise ValueError("internal-ring port mask must select LDU0 and/or LDU1")
        if int(stage_bytes) <= 0 or int(stage_bytes) % 16:
            raise ValueError("internal-ring stage size must be positive/aligned16")
        minimum_slots = bytes2slots(int(stages) * int(stage_bytes))
        if int(area_slots) != minimum_slots:
            raise ValueError(
                f"internal-ring area must use exactly {minimum_slots} slots"
            )
        if not 0 <= int(area_id) <= 0xFFFF:
            raise ValueError("internal-ring area id must fit uint16")
        super().__init__(
            opcode=opcode.OP_TMA_LOAD_MX_COUPLED_STREAM | 1,
            num_slots=int(area_slots),
            arg=(
                self.KIND
                | (int(stages) << self.STAGES_SHIFT)
                | (int(port_mask) << self.PORT_MASK_SHIFT)
            ),
            size=int(stream_length),
            address=get_tensor_address(plan),
        )
        self.plan = plan
        self.annotation["coupled_stream_area"] = int(area_id)
        self.annotation["coupled_stream_kind"] = self.KIND
        self.annotation["coupled_stream_allocator_lease"] = True
        self.annotation["coupled_stream_dual_port"] = True
        self.annotation["internal_ring_port_mask"] = int(port_mask)


class TmaLoadMxfpScale1D(MemoryInstruction):
    """Compact TMA of one native scale half from an LDU-cached base."""

    WEIGHT = 0
    ACTIVATION = 1
    STAGES = config.mxfp4_mxfp8_tma_scale_stages
    BYTES = 2048

    def __init__(self, *, stage: int, operand: int):
        if not 0 <= stage < self.STAGES:
            raise ValueError(
                f"direct MX scale stage must be in [0,{self.STAGES})"
            )
        if operand not in (self.WEIGHT, self.ACTIVATION):
            raise ValueError("direct MX scale operand must be weight or activation")
        super().__init__(
            opcode=opcode.OP_ALLOC_TMA_LOAD_MX_SCALE_1D,
            # Operand/stage are carried in LdCmd::slot. The corresponding
            # special MInst entry is not read by LDU and owns no payload bytes.
            num_slots=config.num_slots + operand * self.STAGES + stage,
            arg=0,
            size=0,
            address=0,
        )


class TmaLoadMxfpScaleBase1D(MemoryInstruction):
    """Seed one LDU-local scale base and issue its first direct TMA."""

    def __init__(self, src: torch.Tensor, *, operand: int):
        if operand not in (
            TmaLoadMxfpScale1D.WEIGHT,
            TmaLoadMxfpScale1D.ACTIVATION,
        ):
            raise ValueError("MX scale base operand must be weight or activation")
        super().__init__(
            opcode=opcode.OP_ALLOC_TMA_LOAD_MX_SCALE_BASE_1D,
            num_slots=config.num_slots + 6 + operand,
            arg=operand,
            size=0,
            address=get_tensor_address(src),
        )


class TmaLoad64K1D(TmaLoad1D):
    """Allocator TMA load for the one-byte-past-uint16 64 KiB case."""

    SIZE_MARKER = 0xFFFF

    def __init__(self, src: torch.Tensor):
        bytes = src.numel() * src.element_size()
        if bytes != 64 * 1024:
            raise ValueError(f"TmaLoad64K1D requires exactly 64 KiB, got {bytes}")
        MemoryInstruction.__init__(
            self,
            opcode=opcode.OP_ALLOC_TMA_LOAD_1D,
            num_slots=8,
            arg=self.SIZE_MARKER,
            size=0,
            address=get_tensor_address(src),
        )


class TmaLoadReg1D(TmaLoad1D):
    """TMA-load into shared slots retained for a later same-LDU RegLoad."""

    def __init__(
        self,
        src: torch.Tensor,
        reg_id: int,
        port_id: int,
        bytes: int | None = None,
        numSlots: int | None = None,
    ):
        if not 0 <= reg_id < 4:
            raise ValueError("TMA load register id must be in [0, 4)")
        super().__init__(src, bytes=bytes, numSlots=numSlots)
        self.opcode = opcode.OP_ALLOC_TMA_LOAD_REG_1D
        self.arg = reg_id
        self.fixed_port(port_id)


class LduLoad1D(MemoryInstruction):
    """Load arbitrary-sized metadata through LDU into normal shared slots."""

    def __init__(self, src: torch.Tensor, bytes: int | None = None):
        address = get_tensor_address(src)
        if bytes is None:
            bytes = src.numel() * src.element_size()
        if bytes <= 0 or bytes > 0xFFFF:
            raise ValueError("LDU load size must be a positive uint16")
        super().__init__(
            opcode=opcode.OP_ALLOC_LDU_LOAD_1D,
            num_slots=bytes2slots(bytes),
            arg=0,
            size=bytes,
            address=address,
        )


class TmaStore1D(MemoryInstruction):
    def __init__(self, dst: torch.Tensor, bytes: int | None = None, numSlots: int | None = None):
        address = get_tensor_address(dst)
        if bytes is None:
            bytes = dst.numel() * dst.element_size()
        if numSlots is None:
            numSlots = bytes2slots(bytes)
        super().__init__(
            opcode=opcode.OP_ALLOC_WB_TMA_STORE_1D,
            num_slots=numSlots,
            arg=0,
            size=bytes,
            address=address,
        )

    def cord(self, addr):
        new_inst = copy.copy(self)
        new_inst.delta(addr)
        return new_inst


class StuStore1D(MemoryInstruction):
    """Store arbitrary-sized metadata from normal shared slots through STU."""

    def __init__(self, dst: torch.Tensor, bytes: int | None = None):
        address = get_tensor_address(dst)
        if bytes is None:
            bytes = dst.numel() * dst.element_size()
        if bytes <= 0 or bytes > 0xFFFF:
            raise ValueError("STU store size must be a positive uint16")
        super().__init__(
            opcode=opcode.OP_ALLOC_WB_STU_STORE_1D,
            num_slots=bytes2slots(bytes),
            arg=0,
            size=bytes,
            address=address,
        )


class TmaReadWrite2D(MemoryInstruction):
    """One allocator lease loaded and stored by equal-shaped 2-D TMAs."""

    NO_INPUT_BARRIER = 0xFFFF

    def __init__(
        self,
        load_descriptor: int,
        store_descriptor: int,
        *,
        coords: tuple[int, int],
        bytes: int,
        num_slots: int,
    ):
        for descriptor in (load_descriptor, store_descriptor):
            if not 0 <= descriptor <= 0xFFFF:
                raise ValueError("TMA descriptor index must fit uint16")
        if bytes <= 0 or bytes > 0xFFFF:
            raise ValueError("2-D read/write size must fit uint16")
        if num_slots <= 0 or num_slots >= config.num_slots:
            raise ValueError("2-D read/write lease must use normal shared slots")
        super().__init__(
            opcode=opcode.OP_ALLOC_RW_TMA_2D,
            num_slots=num_slots,
            arg=load_descriptor,
            size=bytes,
            cords=[coords[0], coords[1], self.NO_INPUT_BARRIER, store_descriptor],
        )
        self.annotation["readwrite_load"] = True
        self.annotation["input_bar_coord"] = 2


class TmaTensor(MemoryInstruction):
    def __init__(self, launcher, mat: torch.Tensor):
        super().__init__(opcode=0, num_slots=0, arg=0, size=0, cords=[])
        self.launcher = launcher
        self.mat = mat
        self.cord_func = None

    def _rank2opcode(self, rank: int, action: str) -> int:
        opcode_map = {
            "reduce": {
                2: opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_2D,
                3: opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_3D,
                4: opcode.OP_ALLOC_WB_TMA_REDUCE_ADD_4D,
            },
            "load": {
                1: opcode.OP_ALLOC_TMA_LOAD_TENSOR_1D,
                2: opcode.OP_ALLOC_TMA_LOAD_2D,
                3: opcode.OP_ALLOC_TMA_LOAD_3D,
                4: opcode.OP_ALLOC_TMA_LOAD_4D,
                5: opcode.OP_ALLOC_TMA_LOAD_5D_FIX0,
            },
            "store": {
                1: opcode.OP_ALLOC_WB_TMA_STORE_1D,
                2: opcode.OP_ALLOC_WB_TMA_STORE_2D,
                3: opcode.OP_ALLOC_WB_TMA_STORE_3D,
                4: opcode.OP_ALLOC_WB_TMA_STORE_4D,
                5: opcode.OP_ALLOC_WB_TMA_STORE_5D_FIX0,
            },
        }
        try:
            return opcode_map[action][rank]
        except KeyError as exc:
            raise ValueError(f"Unsupported rank {rank} and action {action}") from exc

    def _build(self, action, tileM, tileN, tma_func, cord_func_builder):
        self.mode = action
        self.size = self.mat.element_size() * tileM * tileN
        self.num_slots = bytes2slots(self.size)
        rank, desc = tma_func(self.mat, tileM, tileN)
        self.rank = rank
        self.opcode = self._rank2opcode(rank, action)
        self.cord_func = cord_func_builder(self.mat, rank)
        self.desc = desc

        if not hasattr(self.launcher, "new_tma"):
            raise ValueError("launcher must expose new_tma()")
        self.arg = self.launcher.new_tma(desc)

        return self

    def cord2tma(self, *cords):
        if self.cord_func is None:
            raise ValueError("cord_func is not set, please call wgmma_load/wgmma_store first")
        return self.cord_func(*cords)

    def cord(self, *cords):
        inst = copy.copy(self)
        inst.set_cords(self.cord2tma(*cords))
        return inst

    def tensor1d(self, action: str, size: int):
        actions = ["load", "store"]
        assert action in actions, f"action must be one of {actions}, got {action}"
        return self._build(action, size, 1, build_tma_1d, cord_func_tma_1d)

    def rowmajor_2d(self, action: str, tile_rows: int, tile_cols: int):
        if action not in ("load", "store", "reduce"):
            raise ValueError(
                "row-major 2D TMA action must be load, store, or reduce"
            )
        return self._build(
            action,
            tile_rows,
            tile_cols,
            build_tma_rowmajor_2d,
            cord_func_rowmajor_2d,
        )

    def m128n8_output(self, action: str):
        """Rank-3 reducible TMA view of contiguous BF16/FP32 ``[8, M]``."""
        if action not in ("store", "reduce"):
            raise ValueError("M128N8 output action must be store or reduce")
        return self._build(
            action,
            128,
            8,
            build_tma_wgmma_mnmajor_m128n8,
            cord_func_m128n8_output,
        )

    def cord_pair_2d(
        self, row: int, col: int, *, delta_cols: int
    ) -> "MemoryInstruction":
        if self.rank != 2 or self.mode not in ("load", "store"):
            raise ValueError(
                "paired 2-D TMA requires a row-major rank-2 load or store"
            )
        if delta_cols <= 0 or delta_cols > 0xFFFF:
            raise ValueError("paired 2-D TMA column delta must fit uint16")
        coords = self.cord2tma(row, col)
        total_size = 2 * self.size
        if total_size > 0xFFFF:
            raise ValueError("paired 2-D TMA transfer exceeds uint16 bytes")
        pair_opcode = (
            opcode.OP_ALLOC_TMA_LOAD_PAIR_2D
            if self.mode == "load"
            else opcode.OP_ALLOC_WB_TMA_STORE_PAIR_2D
        )
        return MemoryInstruction(
            opcode=pair_opcode,
            num_slots=bytes2slots(total_size),
            arg=self.arg,
            size=total_size,
            cords=[coords[0], coords[1], delta_cols],
        )

    def wgmma(self, action: str, tileN: int, tileM: int, major: Major):
        actions = ["load", "store", "reduce"]
        assert action in actions, f"action must be one of {actions}, got {action}"
        if major == Major.K:
            return self._build(action, tileM, tileN, build_tma_wgmma_kmajor, cord_func_2d_kmajor)
        return self._build(action, tileM, tileN, build_tma_wgmma_mnmajor, cord_func_2d_mnmajor)

    def wgmma_load(self, tileN: int, tileM: int, major: Major):
        return self.wgmma("load", tileN, tileM, major)

    def wgmma_load_tiled(self, tileN: int, tileM: int):
        return self._build(
            "load",
            tileM,
            tileN,
            build_tma_wgmma_tile_major,
            cord_func_2d_tile_major,
        )

    def encode_64k(self):
        """Encode one descriptor-backed 64-KiB tile in uint16 MInst.size."""
        if self.size != 64 * 1024 or self.num_slots != 8:
            raise ValueError("64-KiB tensor encoding requires exactly eight slots")
        self.size = 0
        return self

    def mxfp4_k512_load(self):
        """Packed W4 HBM load with TMA expansion into a 64 KiB UMMA tile."""
        inst = self._build(
            "load",
            128,
            512,
            build_tma_mxfp4_k512,
            cord_func_mxfp4_k512,
        )
        # TMA moves 32 KiB of packed U4 values but writes a 64 KiB shared
        # image because ALIGN16B inserts one 8-byte gap per packed 8 bytes.
        inst.size = 32 * 1024
        inst.num_slots = 8
        return inst

    def mxfp4_load(self, tile_k: int):
        """Packed W4 HBM load expanded into a native M128/K-tile image."""
        if tile_k not in (128, 256, 512):
            raise ValueError("MXFP4 TMA tile K must be 128, 256, or 512")
        inst = self._build(
            "load",
            128,
            tile_k,
            build_tma_mxfp4,
            cord_func_mxfp4,
        )
        # HBM contributes half a byte/value; the 16U4 transform writes one
        # full byte/value into the UMMA shared-memory allocation.
        inst.size = 128 * tile_k // 2
        inst.num_slots = (128 * tile_k) // (8 * 1024)
        return inst

    def mxfp4_kmajor_load(self, tile_k: int):
        """Packed MXFP4 load from K-tile-major HBM."""
        if tile_k not in (128, 256, 512):
            raise ValueError("MXFP4 TMA tile K must be 128, 256, or 512")
        inst = self._build(
            "load",
            128,
            tile_k,
            build_tma_mxfp4_kmajor,
            cord_func_mxfp4_kmajor,
        )
        inst.size = 128 * tile_k // 2
        inst.num_slots = (128 * tile_k) // (8 * 1024)
        return inst

    def mxfp4_k128_load(self):
        return self.mxfp4_load(128)

    def wgmma_store(self, tileN: int, tileM: int, major: Major):
        return self.wgmma("store", tileN, tileM, major)


__all__ = [
    "decode_opcode",
    "dedcode_opcode",
    "Instruction",
    "ComputeInstruction",
    "TerminateC",
    "Gemv_M64N8",
    "Gemv_M64N8IssuerOnly",
    "Nvfp4GemvSm100",
    "Nvfp4GemvUmmaSm100",
    "Nvfp4GemvUmmaStreamSm100",
    "Nvfp4GemvUmmaPipelineSm100",
    "Nvfp4GemvUmmaFp32Sm100",
    "Nvfp4GemvUmmaPipelineFp32Sm100",
    "Nvfp4GemvUmmaPipelineFp32Group2Sm100",
    "Nvfp4GemvUmmaK512Fp32Sm100",
    "Mxfp4Mxfp8GemvUmmaK512TmaScaleFp32Sm100",
    "Mxfp4Mxfp8GemvUmmaK512MetaScaleFp32Sm100",
    "Mxfp4Mxfp8GateUpSiluFixedRingSm100",
    "Mxfp4Mxfp8DownFixedRingSm100",
    "Mxfp4Mxfp8ResidentFfnSm100",
    "Mxfp4Mxfp8RoutedResidentFfnSm100",
    "Nvfp4UmmaPrepackSm100",
    "Fp8Block128GemvSm100",
    "Fp8Block128GemvBf16Sm100",
    "Fp8GemvUmmaStreamSm100",
    "Fp8GemvUmmaSplitKSm100",
    "Fp8GemvUmmaCoupledSm100",
    "Fp8GemvUmmaStreamRawScaleSm100",
    "Fp8GemvUmmaSplitKRawScaleSm100",
    "Fp8UmmaPrepackSm100",
    "Dsv4RmsFp8QuantUmmaBSm100",
    "Dsv4Fp32RmsFp8QuantUmmaBSm100",
    "Dsv4Bf16GemvGroup4SplitKSm100",
    "Dsv4PreloadRopeTables",
    "Dsv4Rope64",
    "Dsv4RmsRope512_64",
    "Dsv4Fp32RmsRope512_64",
    "Dsv4Fp32RopeHadamard128",
    "Dsv4SparseAttention512",
    "Dsv4ContiguousAttention512Block4",
    "Dsv4ContiguousAttention512UmmaSm100",
    "Dsv4ContiguousAttention512UmmaTail32Sm100",
    "Dsv4AttentionSplit32UmmaSm100",
    "Dsv4AttentionSplit64UmmaSm100",
    "Dsv4AttentionSplitReduceFp8Sm100",
    "Dsv4AttentionContext1Fp8Sm100",
    "Dsv4RouteTop6",
    "Dsv4RouteTop6Prepared",
    "Dsv4ExpertReduce",
    "Dsv4Fp32Bf16Gemv",
    "Dsv4ZeroFill",
    "Dsv4Fp32ToBf16",
    "Dsv4Bf16Gemv",
    "Dsv4RouterBf16GemvSm100",
    "Dsv4HcPre",
    "Dsv4HcPreRms",
    "Dsv4HcPost",
    "Dsv4SiluClampMul",
    "Dsv4Fp32SwiGluNvfp4QuantUmmaBSm100",
    "Dsv4Hadamard",
    "Dsv4GatedPool",
    "Dsv4GatedPoolRmsRope",
    "Dsv4GatedPoolPacked8Shard128",
    "Dsv4GatedPoolPacked8RmsPartial",
    "Dsv4GatedPoolPacked8HistoryState",
    "Dsv4GatedPoolTailRmsPartial",
    "Dsv4Fp32RmsRopeShard128",
    "Dsv4IndexScore",
    "Dsv4TopK512",
    "Dsv4HcHead",
    "Dsv4HcHeadRms",
    "Dsv4Fp8Quant128",
    "Dsv4Nvfp4Quant16",
    "Dsv4Nvfp4QuantUmmaBSm100",
    "Dsv4Fp8QuantUmmaBSm100",
    "Dsv4Mxfp8QuantFfnInputSm100",
    "Dsv4InverseRopeFp8QuantUmmaBSm100",
    "Gemv_M64N8UpSiLU",
    "Gemv_M128N8",
    "Gemv_M128N8Direct4",
    "Gemv_M128N8Argmax4",
    "Gemv_M128N8Group4B2",
    "Gemv_M128N8Group4B3",
    "Gemv_M128N8Group4B4",
    "Gemv_M128N8Group4B7",
    "Gemv_M64N8K64",
    "Gemv_M64N8K128",
    "Gemv_M64N8B2",
    "Gemm_M64N64",
    "Gemm_M64N64K64",
    "Gemm_M64N128K64",
    "Gemv_M64N8_ROPE_128",
    "Gemv_M128N8_ROPE_128",
    "Gemv_M192N16",
    "Gemv_M64N8_MMA",
    "WGMMA_64x256x64_F16",
    "WGMMA_64x256x64_BF16",
    "ROPE_INTERLEAVE_512",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim_MMA",
    "ATTENTION_SM100_BF16_HDIM128_DIRECT",
    "ATTENTION_SM100_BF16_HDIM128_SPLIT_DIRECT",
    "ATTENTION_SM100_BF16_HDIM128_SWAP_DIRECT",
    "ATTENTION_SM100_BF16_HDIM128_SWAP_SPLIT_DIRECT",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim64",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim64_MMA",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim_split",
    "ATTENTION_M64N64K16_F16_F32_64_64_hdim_split_MMA",
    "ATTN_SPLIT_POST_REDUCE",
    "SILU_MUL_SHARED_BF16_K_4096_INTER",
    "SILU_MUL_SHARED_BF16_K_2048_INTER",
    "SILU_MUL_SHARED_BF16_K_64_SW128",
    "RMS_NORM_F16_K_4096",
    "RMS_NORM_F16_K_4096_SMEM",
    "RMS_NORM_F16_SMEM",
    "select_attention_decode_instruction",
    "select_rms_glob_instruction",
    "select_rms_smem_instruction",
    "ensure_cc0_supported_hidden_size",
    "ARGMAX_PARTIAL_bf16_1152_50688_132",
    "ARGMAX_REDUCE_bf16_1152_132",
    "ARGMAX_PARTIAL_bf16_1024_65536_128",
    "ARGMAX_REDUCE_bf16_1024_128",
    "ARGMAX_REDUCE_GLOBAL_bf16_256",
    "ArgmaxSmemPartialBf16",
    "ArgmaxSmemReduceBf16",
    "Dummy",
    "Copy",
    "ProfileEvent",
    "ProfileStep",
    "ProfileAggregate",
    "LoopC",
    "MemoryInstruction",
    "TerminateM",
    "LoopM",
    "ResetIndirectLayer",
    "CounterOffsetMemoryInstruction",
    "RepeatM",
    "RawAddress",
    "RoutedTmaLoad1D",
    "RoutedTmaLoadBase1D",
    "TmaLoadAddressReg1D",
    "IndexedTmaLoad1D",
    "IndirectTmaLoad1D",
    "IndirectLduLoad1D",
    "IndirectRoutedTmaLoad1D",
    "IndirectRoutedTmaLoadBase1D",
    "IndirectIndexedTmaLoad1D",
    "indirect_1d_from",
    "LduReloadBarriers",
    "LduAsyncReloadBarriers",
    "LduWaitBarrier",
    "LduLoad1D",
    "IssueBarrier",
    "CC0",
    "RegStore",
    "RegLoad",
    "TmaLoad1D",
    "TmaLoadPair1D",
    "TmaLoadMxfpCoupledStream",
    "build_internal_ring_tma_plan",
    "TmaLoadInternalRingStream",
    "TmaLoadMxfpScale1D",
    "TmaLoadMxfpScaleBase1D",
    "TmaLoad64K1D",
    "TmaLoadReg1D",
    "TmaStore1D",
    "StuStore1D",
    "TmaReadWrite2D",
    "TmaTensor",
]
