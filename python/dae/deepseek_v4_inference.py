"""Production host control for reusable DeepSeek-V4 VDCores decode flows.

This module intentionally contains no tokenizer, prompt formatting, reference
prefill, or terminal output.  Applications import the production controller
and provide an initialized :class:`DeepSeekV4LiveDecodeState`; the live demo in
``app/python/deepseek_v4/sched.py`` owns the user-facing pieces.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import torch

from benchmarks.deepseek_v4_resident_one_launch import (
    ResidentOneLaunchDecode,
    build_argument_parser,
)

from .deepseek_v4 import DeepSeekV4FlashConfig
from .deepseek_v4_live import DeepSeekV4LiveDecodeState


MAX_DECODE_TOKENS = 256
MAX_LIVE_SEQUENCE = 65536


@dataclass(frozen=True)
class DeepSeekV4FlowPlan:
    """One reusable instruction image and the positions assigned to it."""

    variant: str
    first_position: int
    last_position: int

    @property
    def max_position(self) -> int:
        return self.last_position


@dataclass(frozen=True)
class DeepSeekV4FlowPreparation:
    """Host-side preparation measurement for one reusable image."""

    plan: DeepSeekV4FlowPlan
    elapsed_s: float
    free_bytes: int


@dataclass(frozen=True)
class DeepSeekV4DecodeStep:
    """One production token launch and its timing boundaries."""

    step: int
    position: int
    variant: str
    input_token: int
    output_token: int
    cuda_ms: float
    device_ms: float
    wall_ms: float


@dataclass(frozen=True)
class DeepSeekV4Generation:
    """Completed greedy generation returned by the production controller."""

    steps: tuple[DeepSeekV4DecodeStep, ...]
    stop_reason: str

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(step.output_token for step in self.steps)


def reusable_flow_plan(
    first_position: int,
    max_new_tokens: int,
) -> tuple[DeepSeekV4FlowPlan, ...]:
    """Group a sequential decode range by structural instruction variant."""

    first_position = int(first_position)
    max_new_tokens = int(max_new_tokens)
    if first_position < 0:
        raise ValueError("first decode position must be non-negative")
    if not 1 <= max_new_tokens <= MAX_DECODE_TOKENS:
        raise ValueError(
            f"max_new_tokens must be in [1,{MAX_DECODE_TOKENS}]"
        )
    if first_position + max_new_tokens > MAX_LIVE_SEQUENCE:
        raise ValueError(
            f"decode range exceeds the {MAX_LIVE_SEQUENCE}-token live cache"
        )

    ranges: dict[str, list[int]] = {}
    for position in range(first_position, first_position + max_new_tokens):
        variant = ResidentOneLaunchDecode.reusable_variant_for_position(
            position
        )
        position_range = ranges.setdefault(variant, [position, position])
        position_range[1] = position
    return tuple(
        DeepSeekV4FlowPlan(variant, first, last)
        for variant, (first, last) in ranges.items()
    )


def _production_args(
    checkpoint: Path,
    mxfp_ffn_root: Path | None,
    *,
    context_length: int,
    token_id: int,
) -> argparse.Namespace:
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--layers",
        "43",
        "--context-length",
        str(context_length),
        "--token-id",
        str(token_id),
        "--vocab-size",
        str(DeepSeekV4FlashConfig().vocab_size),
        "--bf16-head",
        "--loopback-hc-fusion",
        "--allow-token-variation",
    ]
    if mxfp_ffn_root is not None:
        argv.extend(("--mxfp-ffn-root", str(mxfp_ffn_root)))
    return build_argument_parser().parse_args(argv)


class DeepSeekV4ProductionInference:
    """Prepare reusable images and execute sequential greedy decode tokens."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        live_state: DeepSeekV4LiveDecodeState,
        first_position: int,
        max_new_tokens: int,
        initial_token_id: int,
        mxfp_ffn_root: str | Path | None = None,
        device: torch.device | str = "cuda",
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.mxfp_ffn_root = (
            None if mxfp_ffn_root is None else Path(mxfp_ffn_root).resolve()
        )
        self.live_state = live_state
        self.device = torch.device(device)
        self.first_position = int(first_position)
        self.max_new_tokens = int(max_new_tokens)
        self.initial_token_id = int(initial_token_id)
        self.config = DeepSeekV4FlashConfig()
        if not 0 <= self.initial_token_id < self.config.vocab_size:
            raise ValueError("initial token is outside the vocabulary")

        self.flow_plans = reusable_flow_plan(
            self.first_position, self.max_new_tokens
        )
        required_capacity = self.first_position + self.max_new_tokens
        if self.live_state.max_seq_len < required_capacity:
            raise ValueError(
                "live state is smaller than the requested decode range"
            )
        if self.live_state.device != self.device:
            raise ValueError("live state and production inference devices differ")

        self._flows: dict[str, ResidentOneLaunchDecode] = {}
        self._prepared = False

    def prepare(
        self,
        *,
        verbose: bool = False,
        verify_state_unchanged: bool = False,
    ) -> tuple[DeepSeekV4FlowPreparation, ...]:
        """Build each structural image once without publishing a token."""

        if self._prepared:
            raise RuntimeError("production inference is already prepared")
        state_before = None
        if verify_state_unchanged:
            state_before = tuple(
                tensor.clone()
                for tensor in self.live_state.persistent_tensors()
            )

        weight_source = None
        started = time.perf_counter()
        preparations: list[DeepSeekV4FlowPreparation] = []
        for plan in self.flow_plans:
            args = _production_args(
                self.checkpoint,
                self.mxfp_ffn_root,
                context_length=plan.first_position + 1,
                token_id=self.initial_token_id,
            )
            output = (
                contextlib.nullcontext()
                if verbose
                else contextlib.redirect_stdout(io.StringIO())
            )
            with output:
                flow = ResidentOneLaunchDecode(
                    args,
                    self.device,
                    weight_source=weight_source,
                    live_state=self.live_state,
                    dynamic_max_position=plan.max_position,
                    dynamic_variant=plan.variant,
                )
            if weight_source is None:
                weight_source = flow
            self._flows[plan.variant] = flow
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
            preparations.append(
                DeepSeekV4FlowPreparation(
                    plan=plan,
                    elapsed_s=time.perf_counter() - started,
                    free_bytes=free_bytes,
                )
            )

        torch.cuda.synchronize(self.device)
        if state_before is not None and any(
            not torch.equal(before.view(torch.uint8), after.view(torch.uint8))
            for before, after in zip(
                state_before,
                self.live_state.persistent_tensors(),
                strict=True,
            )
        ):
            raise RuntimeError(
                "preparing reusable images modified live decode state"
            )
        self._prepared = True
        return tuple(preparations)

    def decode_step(
        self,
        *,
        step: int,
        input_token: int,
    ) -> DeepSeekV4DecodeStep:
        """Execute one already prepared persistent-kernel token launch."""

        if not self._prepared:
            raise RuntimeError("prepare must precede production decode")
        step = int(step)
        if not 0 <= step < self.max_new_tokens:
            raise ValueError("decode step is outside the prepared range")
        input_token = int(input_token)
        if not 0 <= input_token < self.config.vocab_size:
            raise ValueError("input token is outside the vocabulary")

        position = self.first_position + step
        variant = ResidentOneLaunchDecode.reusable_variant_for_position(
            position
        )
        flow = self._flows[variant]
        wall_started = time.perf_counter_ns()
        flow.set_decode_position(position)
        flow.set_input_token(input_token)
        output_token, cuda_ms, _ = flow.run_once()
        wall_ms = (time.perf_counter_ns() - wall_started) / 1.0e6
        return DeepSeekV4DecodeStep(
            step=step,
            position=position,
            variant=variant,
            input_token=input_token,
            output_token=output_token,
            cuda_ms=cuda_ms,
            device_ms=flow.device_frontier_ms(),
            wall_ms=wall_ms,
        )

    def generate(
        self,
        *,
        stop_token_ids: Iterable[int] = (),
        max_decode_seconds: float | None = None,
        on_step: Callable[[DeepSeekV4DecodeStep], None] | None = None,
    ) -> DeepSeekV4Generation:
        """Greedily decode the prepared range with optional host-side stops."""

        if max_decode_seconds is not None and max_decode_seconds <= 0:
            raise ValueError("max_decode_seconds must be positive")
        stops = {int(token_id) for token_id in stop_token_ids}
        token = self.initial_token_id
        steps: list[DeepSeekV4DecodeStep] = []
        started = time.perf_counter()
        stop_reason = "token_budget"
        for step_index in range(self.max_new_tokens):
            if (
                step_index
                and max_decode_seconds is not None
                and time.perf_counter() - started >= max_decode_seconds
            ):
                stop_reason = "time_budget"
                break
            step = self.decode_step(step=step_index, input_token=token)
            steps.append(step)
            if on_step is not None:
                on_step(step)
            token = step.output_token
            if token in stops:
                stop_reason = f"stop_token:{token}"
                break
        return DeepSeekV4Generation(tuple(steps), stop_reason)


__all__ = [
    "DeepSeekV4DecodeStep",
    "DeepSeekV4FlowPlan",
    "DeepSeekV4FlowPreparation",
    "DeepSeekV4Generation",
    "DeepSeekV4ProductionInference",
    "MAX_DECODE_TOKENS",
    "MAX_LIVE_SEQUENCE",
    "reusable_flow_plan",
]
