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
    tokens_per_launch: int = 1

    @property
    def max_position(self) -> int:
        return self.last_position

    @property
    def key(self) -> tuple[str, int]:
        return self.variant, self.tokens_per_launch


@dataclass(frozen=True)
class DeepSeekV4TokenSpan:
    """One contiguous decode span executed by a single persistent launch."""

    first_step: int
    first_position: int
    token_count: int
    variant: str

    @property
    def last_position(self) -> int:
        return self.first_position + self.token_count - 1

    @property
    def key(self) -> tuple[str, int]:
        return self.variant, self.token_count


@dataclass(frozen=True)
class DeepSeekV4FlowPreparation:
    """Host-side preparation measurement for one reusable image."""

    plan: DeepSeekV4FlowPlan
    elapsed_s: float
    free_bytes: int


@dataclass(frozen=True)
class DeepSeekV4DecodeStep:
    """One decoded token with launch time amortized across its device span."""

    step: int
    position: int
    variant: str
    input_token: int
    output_token: int
    launch_tokens: int
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


def device_token_span_plan(
    first_position: int,
    max_new_tokens: int,
    *,
    max_span_tokens: int = 1,
) -> tuple[DeepSeekV4TokenSpan, ...]:
    """Partition decode into safe same-structure device-controlled spans.

    Before the 128-row sliding window is full, host bookkeeping packs the
    short CSA history between tokens, so those positions remain single-token
    launches.  At and beyond position 128, ordinary tokens are grouped until
    the next ratio-4/ratio-128/index-selection structural boundary.
    """

    reusable_flow_plan(first_position, max_new_tokens)
    max_span_tokens = int(max_span_tokens)
    if not 1 <= max_span_tokens <= MAX_DECODE_TOKENS:
        raise ValueError(
            f"max_span_tokens must be in [1,{MAX_DECODE_TOKENS}]"
        )

    spans = []
    sliding_window = DeepSeekV4FlashConfig().sliding_window
    step = 0
    while step < max_new_tokens:
        position = first_position + step
        variant = ResidentOneLaunchDecode.reusable_variant_for_position(
            position
        )
        token_count = 1
        if position >= sliding_window and variant in {
            "normal",
            "indexed_normal",
        }:
            while token_count < min(max_span_tokens, max_new_tokens - step):
                candidate = position + token_count
                if (
                    ResidentOneLaunchDecode.reusable_variant_for_position(
                        candidate
                    )
                    != variant
                ):
                    break
                token_count += 1
        spans.append(
            DeepSeekV4TokenSpan(step, position, token_count, variant)
        )
        step += token_count
    return tuple(spans)


def _device_flow_plan(
    spans: tuple[DeepSeekV4TokenSpan, ...],
) -> tuple[DeepSeekV4FlowPlan, ...]:
    ranges: dict[tuple[str, int], list[int]] = {}
    for span in spans:
        position_range = ranges.setdefault(
            span.key, [span.first_position, span.last_position]
        )
        position_range[1] = span.last_position
    return tuple(
        DeepSeekV4FlowPlan(variant, first, last, token_count)
        for (variant, token_count), (first, last) in ranges.items()
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
        device_span_tokens: int = 1,
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
        self.device_span_tokens = int(device_span_tokens)
        self.config = DeepSeekV4FlashConfig()
        if not 0 <= self.initial_token_id < self.config.vocab_size:
            raise ValueError("initial token is outside the vocabulary")

        self.token_spans = device_token_span_plan(
            self.first_position,
            self.max_new_tokens,
            max_span_tokens=self.device_span_tokens,
        )
        self.flow_plans = _device_flow_plan(self.token_spans)
        required_capacity = self.first_position + self.max_new_tokens
        if self.live_state.max_seq_len < required_capacity:
            raise ValueError(
                "live state is smaller than the requested decode range"
            )
        if self.live_state.device != self.device:
            raise ValueError("live state and production inference devices differ")

        self._flows: dict[tuple[str, int], ResidentOneLaunchDecode] = {}
        self._token_history = (
            torch.empty(
                (self.live_state.max_seq_len + 1,),
                dtype=torch.int64,
                device=self.device,
            )
            if any(plan.tokens_per_launch > 1 for plan in self.flow_plans)
            else None
        )
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
                    multi_token_count=plan.tokens_per_launch,
                    token_history=self._token_history,
                )
            if weight_source is None:
                weight_source = flow
            self._flows[plan.key] = flow
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

        span = next(
            (item for item in self.token_spans if item.first_step == step),
            None,
        )
        if span is None or span.token_count != 1:
            raise ValueError(
                "decode_step must begin a prepared single-token span; use "
                "decode_span when device_span_tokens is greater than one"
            )
        return self.decode_span(step=step, input_token=input_token)[0]

    def decode_span(
        self,
        *,
        step: int,
        input_token: int,
    ) -> tuple[DeepSeekV4DecodeStep, ...]:
        """Execute the prepared span beginning at ``step``."""

        if not self._prepared:
            raise RuntimeError("prepare must precede production decode")
        step = int(step)
        input_token = int(input_token)
        if not 0 <= input_token < self.config.vocab_size:
            raise ValueError("input token is outside the vocabulary")
        span = next(
            (item for item in self.token_spans if item.first_step == step),
            None,
        )
        if span is None:
            raise ValueError("decode step is not the beginning of a prepared span")
        flow = self._flows[span.key]
        wall_started = time.perf_counter_ns()
        flow.set_decode_position(span.first_position)
        flow.set_input_token(input_token)
        if span.token_count == 1:
            output_token, cuda_ms, _ = flow.run_once()
            output_tokens = (output_token,)
        else:
            output_tokens, cuda_ms = flow.run_token_span()
            output_tokens = tuple(output_tokens)
        wall_ms = (time.perf_counter_ns() - wall_started) / 1.0e6
        device_ms = flow.device_frontier_ms()
        per_token_cuda_ms = cuda_ms / span.token_count
        per_token_device_ms = device_ms / span.token_count
        per_token_wall_ms = wall_ms / span.token_count
        inputs = (input_token, *output_tokens[:-1])
        return tuple(
            DeepSeekV4DecodeStep(
                step=step + offset,
                position=span.first_position + offset,
                variant=span.variant,
                input_token=inputs[offset],
                output_token=output_tokens[offset],
                launch_tokens=span.token_count,
                cuda_ms=per_token_cuda_ms,
                device_ms=per_token_device_ms,
                wall_ms=per_token_wall_ms,
            )
            for offset in range(span.token_count)
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
        for span in self.token_spans:
            if (
                span.first_step
                and max_decode_seconds is not None
                and time.perf_counter() - started >= max_decode_seconds
            ):
                stop_reason = "time_budget"
                break
            span_steps = self.decode_span(
                step=span.first_step,
                input_token=token,
            )
            for step in span_steps:
                steps.append(step)
                if on_step is not None:
                    on_step(step)
                token = step.output_token
                if token in stops:
                    stop_reason = f"stop_token:{token}"
                    break
            if stop_reason != "token_budget":
                break
        return DeepSeekV4Generation(tuple(steps), stop_reason)


__all__ = [
    "DeepSeekV4DecodeStep",
    "DeepSeekV4FlowPlan",
    "DeepSeekV4FlowPreparation",
    "DeepSeekV4Generation",
    "DeepSeekV4ProductionInference",
    "DeepSeekV4TokenSpan",
    "MAX_DECODE_TOKENS",
    "MAX_LIVE_SEQUENCE",
    "device_token_span_plan",
    "reusable_flow_plan",
]
