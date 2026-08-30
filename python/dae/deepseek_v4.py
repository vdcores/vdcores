"""DeepSeek-V4-Flash shapes and correctness references for vdcores tasks."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, log, pi, sqrt

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DeepSeekV4FlashConfig:
    vocab_size: int = 129280
    hidden_size: int = 4096
    num_layers: int = 43
    num_hash_layers: int = 3
    num_heads: int = 64
    num_kv_heads: int = 1
    head_dim: int = 512
    rope_dim: int = 64
    q_lora_rank: int = 1024
    o_lora_rank: int = 1024
    o_groups: int = 8
    num_experts: int = 256
    experts_per_token: int = 6
    shared_experts: int = 1
    expert_intermediate_size: int = 2048
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_epsilon: float = 1.0e-6
    sliding_window: int = 128
    index_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    route_scale: float = 1.5
    swiglu_limit: float = 10.0
    rms_epsilon: float = 1.0e-6
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling_factor: float = 16.0
    rope_original_max_position_embeddings: int = 65536
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    compress_ratios: tuple[int, ...] = (
        0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
        4, 128, 4, 128, 4, 128, 4, 0,
    )

    def __post_init__(self) -> None:
        if len(self.compress_ratios) != self.num_layers + 1:
            raise ValueError("compress_ratios must cover all layers plus MTP")

    def attention_kind(self, layer_id: int) -> str:
        if not 0 <= layer_id < self.num_layers:
            raise IndexError("layer_id is outside the transformer")
        return {0: "swa", 4: "csa", 128: "hca"}[
            self.compress_ratios[layer_id]
        ]


def _deepseek_v4_rope_inverse(
    *,
    compressed: bool,
    config: DeepSeekV4FlashConfig,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Return the checkpoint's exact FP32 inverse-frequency vector."""

    dim = config.rope_dim
    base = config.compress_rope_theta if compressed else config.rope_theta
    exponents = torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim
    frequencies = torch.pow(
        torch.tensor(base, dtype=torch.float32, device=device), exponents
    )
    inverse = frequencies.reciprocal()
    if compressed:
        interpolation = inverse / config.rope_scaling_factor

        def correction(rotations: float) -> float:
            return (
                dim
                * log(
                    config.rope_original_max_position_embeddings
                    / (rotations * 2.0 * pi)
                )
                / (2.0 * log(base))
            )

        low = max(floor(correction(config.rope_beta_fast)), 0)
        high = min(ceil(correction(config.rope_beta_slow)), dim - 1)
        ramp = torch.arange(dim // 2, dtype=torch.float32, device=device)
        ramp = ((ramp - low) / max(high - low, 0.001)).clamp(0.0, 1.0)
        extrapolation = 1.0 - ramp
        inverse = interpolation * (1.0 - extrapolation) + inverse * extrapolation
    return inverse


def deepseek_v4_rope_table(
    position: int,
    *,
    compressed: bool = False,
    config: DeepSeekV4FlashConfig | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the checkpoint's interleaved 64-wide main/compressor RoPE row."""
    if position < 0:
        raise ValueError("RoPE position must be non-negative")
    config = config or DeepSeekV4FlashConfig()
    inverse = _deepseek_v4_rope_inverse(
        compressed=compressed, config=config, device=device
    )
    angles = inverse * float(position)
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def deepseek_v4_rope_bank(
    num_positions: int,
    *,
    compressed: bool = False,
    config: DeepSeekV4FlashConfig | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build exact checkpoint RoPE rows for positions ``[0,num_positions)``."""

    if num_positions <= 0:
        raise ValueError("RoPE bank must contain at least one position")
    config = config or DeepSeekV4FlashConfig()
    inverse = _deepseek_v4_rope_inverse(
        compressed=compressed, config=config, device=device
    )
    positions = torch.arange(
        num_positions, dtype=torch.float32, device=device
    )[:, None]
    angles = positions * inverse[None, :]
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def _apply_partial_rope_64(
    tensor: torch.Tensor,
    table: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if tensor.shape[-1] not in (128, 512) or table.shape != (32, 2):
        raise ValueError("partial RoPE expects width 128/512 and table [32,2]")
    output = tensor.clone()
    rope = tensor[..., -64:].float().reshape(*tensor.shape[:-1], 32, 2)
    cosine = table[:, 0]
    sine = -table[:, 1] if inverse else table[:, 1]
    even = rope[..., 0]
    odd = rope[..., 1]
    output[..., -64:] = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2).to(tensor.dtype)
    return output


def apply_partial_rope_512_64(
    tensor: torch.Tensor,
    table: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply interleaved RoPE to the final 64 dimensions of 512-wide rows."""
    if tensor.shape[-1] != 512:
        raise ValueError("attention partial RoPE expects [...,512]")
    return _apply_partial_rope_64(tensor, table, inverse=inverse)


def apply_partial_rope_128_64(
    tensor: torch.Tensor,
    table: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply interleaved RoPE to the final 64 dimensions of 128-wide rows."""
    if tensor.shape[-1] != 128:
        raise ValueError("index partial RoPE expects [...,128]")
    return _apply_partial_rope_64(tensor, table, inverse=inverse)


def sparse_attention_512_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
) -> torch.Tensor:
    """Reference the shared-KV sparse decode attention including sink logits."""
    valid = indices[indices >= 0].long()
    selected = kv[valid].float()
    scores = q.float() @ selected.t() / sqrt(512.0)
    probabilities = torch.softmax(
        torch.cat((scores, sink.float().unsqueeze(1)), dim=1), dim=1
    )[:, : valid.numel()]
    return (probabilities @ selected).to(q.dtype)


def route_top6_reference(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    hash_indices: torch.Tensor | None = None,
    route_scale: float = 1.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference sqrt-softplus routing with bias-only expert selection."""
    scores = F.softplus(logits.float()).sqrt()
    if hash_indices is None:
        indices = (scores + bias.float()).topk(6).indices
    else:
        indices = hash_indices.to(torch.int64)
    weights = scores[indices]
    weights = weights / weights.sum() * route_scale
    return weights, indices.to(torch.int32)


def bounded_swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    limit: float = 10.0,
) -> torch.Tensor:
    output_dtype = gate.dtype
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    return (F.silu(gate) * up).to(output_dtype)


def hadamard_reference(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the normalized Sylvester Walsh-Hadamard transform."""
    width = tensor.shape[-1]
    if width not in (128, 512):
        raise ValueError("DeepSeek Hadamard width must be 128 or 512")
    output = tensor.float()
    stride = 1
    while stride < width:
        groups = output.reshape(*output.shape[:-1], -1, stride * 2)
        lhs = groups[..., :stride]
        rhs = groups[..., stride:]
        output = torch.cat((lhs + rhs, lhs - rhs), dim=-1).reshape_as(output)
        stride *= 2
    return (output / sqrt(width)).to(tensor.dtype)


def gated_pool_reference(
    values: torch.Tensor,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Pool contiguous compressor state rows independently per dimension."""
    if values.shape != scores.shape or values.ndim != 2:
        raise ValueError("gated pool expects matching [rows,width] tensors")
    return (values.float() * scores.float().softmax(dim=0)).sum(dim=0)


def pack_gated_pool_history(
    values: torch.Tensor,
    scores: torch.Tensor,
    *,
    shard_width: int = 128,
    rows_per_block: int = 8,
) -> torch.Tensor:
    """Prepack value/score rows into one slot-sized TMA block per shard."""
    if values.shape != scores.shape or values.ndim != 2:
        raise ValueError("gated-pool history expects matching [rows,width] tensors")
    if values.dtype != torch.float32 or scores.dtype != torch.float32:
        raise ValueError("gated-pool history must be FP32")
    rows, width = values.shape
    if rows <= 0 or width % shard_width:
        raise ValueError("gated-pool history needs rows and complete width shards")
    if shard_width <= 0 or rows_per_block <= 0:
        raise ValueError("gated-pool packing dimensions must be positive")
    shards = width // shard_width
    blocks = (rows + rows_per_block - 1) // rows_per_block
    padded_rows = blocks * rows_per_block
    paired = torch.zeros(
        (padded_rows, shards, 2, shard_width),
        dtype=torch.float32,
        device=values.device,
    )
    paired[:rows, :, 0].copy_(values.reshape(rows, shards, shard_width))
    paired[:rows, :, 1].copy_(scores.reshape(rows, shards, shard_width))
    return (
        paired.permute(1, 0, 2, 3)
        .contiguous()
        .reshape(shards, blocks, rows_per_block, 2, shard_width)
    )


def index_score_reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    head_weights: torch.Tensor,
) -> torch.Tensor:
    """Reference the learned ratio-4 compressed-KV index score."""
    if tuple(q.shape) != (64, 128) or kv.ndim != 2 or kv.shape[1] != 128:
        raise ValueError("index score expects Q [64,128] and KV [rows,128]")
    if head_weights.numel() != 64:
        raise ValueError("index score expects 64 head weights")
    dots = q.float() @ kv.float().t()
    return (dots.relu() * head_weights.float()[:, None]).sum(dim=0)


def decode_window_indices(start_pos: int, window_size: int = 128) -> torch.Tensor:
    """Build the official circular sliding-window indices for one decode token."""
    if start_pos < 0 or window_size <= 0:
        raise ValueError("decode position and window size must be valid")
    if start_pos >= window_size - 1:
        position = start_pos % window_size
        return torch.cat(
            (torch.arange(position + 1, window_size), torch.arange(position + 1))
        ).to(torch.int32)
    indices = torch.full((window_size,), -1, dtype=torch.int32)
    indices[: start_pos + 1] = torch.arange(start_pos + 1, dtype=torch.int32)
    return indices


def decode_compressed_indices(
    start_pos: int,
    ratio: int,
    *,
    index_offset: int = 128,
) -> torch.Tensor:
    """Build deterministic compressed-cache indices for non-indexer layers."""
    if start_pos < 0 or ratio <= 0 or index_offset < 0:
        raise ValueError("compressed index arguments must be valid")
    count = (start_pos + 1) // ratio
    return torch.arange(index_offset, index_offset + count, dtype=torch.int32)


def hc_coefficients_reference(
    residual: torch.Tensor,
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    sinkhorn_iters: int = 20,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Produce mHC pre/post/comb coefficients from the raw projection."""
    normalized_mixes = mixes.float() * torch.rsqrt(
        residual.float().square().mean() + 1.0e-6
    )
    pre = torch.sigmoid(normalized_mixes[:4] * scale[0] + base[:4]) + epsilon
    post = 2 * torch.sigmoid(
        normalized_mixes[4:8] * scale[1] + base[4:8]
    )
    comb = (
        normalized_mixes[8:].reshape(4, 4) * scale[2]
        + base[8:].reshape(4, 4)
    )
    comb = comb.softmax(dim=-1) + epsilon
    comb = comb / (comb.sum(dim=-2, keepdim=True) + epsilon)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + epsilon)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + epsilon)
    return pre, post, comb


def hc_pre_reference(
    residual: torch.Tensor,
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    sinkhorn_iters: int = 20,
    epsilon: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre, post, comb = hc_coefficients_reference(
        residual,
        mixes,
        scale,
        base,
        sinkhorn_iters=sinkhorn_iters,
        epsilon=epsilon,
    )
    hidden = (pre[:, None] * residual.float()).sum(dim=0).to(residual.dtype)
    return hidden, post, comb


def hc_post_reference(
    branch: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    output = post[:, None] * branch.float()[None, :]
    output += torch.einsum("ij,id->jd", comb.float(), residual.float())
    return output.to(branch.dtype)


def hc_head_reference(
    residual: torch.Tensor,
    mixes: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Reduce the four hyper-connection streams before the final/MTP head."""
    normalization = torch.rsqrt(residual.float().square().mean() + 1.0e-6)
    pre = torch.sigmoid(
        mixes.float() * normalization * scale.float().reshape(()) + base.float()
    ) + epsilon
    return (pre[:, None] * residual.float()).sum(dim=0).to(residual.dtype)


__all__ = [
    "DeepSeekV4FlashConfig",
    "deepseek_v4_rope_bank",
    "deepseek_v4_rope_table",
    "apply_partial_rope_128_64",
    "apply_partial_rope_512_64",
    "sparse_attention_512_reference",
    "route_top6_reference",
    "bounded_swiglu",
    "hadamard_reference",
    "gated_pool_reference",
    "pack_gated_pool_history",
    "index_score_reference",
    "decode_window_indices",
    "decode_compressed_indices",
    "hc_coefficients_reference",
    "hc_pre_reference",
    "hc_post_reference",
    "hc_head_reference",
]
