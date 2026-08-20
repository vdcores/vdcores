"""Synthetic stand-ins so the Llama3 schedule can be built without weights.

The autotuner asks "does this configuration build?" hundreds of times per
sweep. Answering that needs the tensor *shapes* the schedule places work
against, not the values in them, so downloading and loading 16GB of
Llama-3.1-8B for each attempt is pure cost.

Rather than restructure `sched.py`, this module hands it objects with the same
attribute shape the real `transformers` model exposes. Everything downstream --
the TMA descriptors, the barrier wiring, `place()` and its `validate()` -- runs
against these unchanged, so the legality answer is the real one.

Only the values are fake. Nothing here is suitable for execution, and
`--dry-build` never launches.
"""

from types import SimpleNamespace

import torch

# Llama-3.1-8B-Instruct geometry.
HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 14336
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
NUM_HIDDEN_LAYERS = 32
VOCAB_SIZE = 128256
RMS_NORM_EPS = 1.0e-5
ROPE_THETA = 500000.0
MAX_POSITION_EMBEDDINGS = 131072
EOS_TOKEN_IDS = [128001, 128008, 128009]

HEAD_DIM = HIDDEN_SIZE // NUM_ATTENTION_HEADS


def build_stub_config():
    """A config carrying the fields `sched.py` reads off the real one."""
    return SimpleNamespace(
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_key_value_heads=NUM_KEY_VALUE_HEADS,
        head_dim=HEAD_DIM,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        vocab_size=VOCAB_SIZE,
        rms_norm_eps=RMS_NORM_EPS,
        rope_parameters={"rope_theta": ROPE_THETA},
        rope_theta=ROPE_THETA,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        eos_token_id=list(EOS_TOKEN_IDS),
    )


def _rotary_embedding(device, dtype, head_dim=HEAD_DIM, config=None):
    """Stand in for `model.model.rotary_emb`.

    The real one returns (cos, sin) shaped [1, seq, head_dim]. These are the
    genuine RoPE values rather than noise: they cost almost nothing to compute
    and keep `matRope` a well-formed tensor, so a dry build cannot be derailed
    by a NaN sneaking into a shape calculation.
    """

    theta = getattr(config, "rope_theta", ROPE_THETA) if config else ROPE_THETA

    def rotary_emb(_hidden_states, position_ids):
        positions = position_ids.to(device=device, dtype=torch.float32)
        inv_freq = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        freqs = positions.unsqueeze(-1) * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    return rotary_emb


def build_stub_model(device, dtype=torch.bfloat16, num_layers=None, config=None):
    """A model-shaped object with correctly sized, uninitialized weights.

    Geometry comes from `config`, so a caller can stub a different Llama size
    (and a test can stub a tiny one) without touching this function.

    `torch.empty` rather than `zeros`: the contents are never read, and a 16GB
    memset per dry build would dominate the check it is meant to make cheap.
    """
    config = config or build_stub_config()
    if num_layers is None:
        num_layers = config.num_hidden_layers

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    head_dim = getattr(config, "head_dim", hidden // config.num_attention_heads)
    vocab = config.vocab_size

    def empty(*shape):
        return torch.empty(*shape, dtype=dtype, device=device)

    def linear(out_features, in_features):
        return SimpleNamespace(weight=empty(out_features, in_features))

    def norm(size=None):
        return SimpleNamespace(weight=empty(hidden if size is None else size))

    query_width = head_dim * config.num_attention_heads
    kv_width = head_dim * config.num_key_value_heads

    layers = [
        SimpleNamespace(
            input_layernorm=norm(),
            post_attention_layernorm=norm(),
            self_attn=SimpleNamespace(
                q_proj=linear(query_width, hidden),
                k_proj=linear(kv_width, hidden),
                v_proj=linear(kv_width, hidden),
                o_proj=linear(hidden, hidden),
            ),
            mlp=SimpleNamespace(
                up_proj=linear(intermediate, hidden),
                gate_proj=linear(intermediate, hidden),
                down_proj=linear(hidden, intermediate),
            ),
        )
        for _ in range(num_layers)
    ]

    inner = SimpleNamespace(
        layers=layers,
        embed_tokens=SimpleNamespace(weight=empty(vocab, hidden)),
        norm=norm(),
        rotary_emb=_rotary_embedding(device, dtype, head_dim, config),
    )
    return SimpleNamespace(
        model=inner,
        lm_head=SimpleNamespace(weight=empty(vocab, hidden)),
        dtype=dtype,
        config=config,
    )


def build_stub_tokenizer():
    """Enough tokenizer for the paths a dry build can still reach."""

    def _reject(*_args, **_kwargs):
        raise RuntimeError(
            "--dry-build has no tokenizer; it cannot take a prompt or decode text"
        )

    return SimpleNamespace(
        eos_token_id=EOS_TOKEN_IDS[0],
        chat_template=None,
        decode=_reject,
        apply_chat_template=_reject,
        __call__=_reject,
    )
