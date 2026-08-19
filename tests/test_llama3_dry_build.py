#!/usr/bin/env python3
"""Tests for app/python/llama3/dry_build.py.

The stand-ins exist so the Llama3 schedule can be built without downloading
16GB of weights. What matters is that they expose exactly the attributes
`sched.py` reads off the real transformers model, with the right shapes -- if
one is missing, the failure shows up as an AttributeError halfway through a
tuning sweep.

These run on CPU with a deliberately tiny geometry, so no CUDA and no model
download is needed:

    python tests/test_llama3_dry_build.py
"""

import importlib.util
import os
import sys
from types import SimpleNamespace

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRY_BUILD_PATH = os.path.join(REPO_ROOT, "app", "python", "llama3", "dry_build.py")

spec = importlib.util.spec_from_file_location("llama3_dry_build", DRY_BUILD_PATH)
dry_build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dry_build)

CPU = torch.device("cpu")


def tiny_config():
    """Same shape as the real config, small enough to allocate in a test."""
    return SimpleNamespace(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        vocab_size=256,
        rms_norm_eps=1.0e-5,
        rope_parameters={"rope_theta": 500000.0},
        rope_theta=500000.0,
        max_position_embeddings=1024,
        eos_token_id=[1],
    )


def test_stub_config_carries_llama31_8b_geometry():
    config = dry_build.build_stub_config()
    assert config.hidden_size == 4096
    assert config.intermediate_size == 14336
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.num_hidden_layers == 32
    assert config.vocab_size == 128256
    # sched.py reads the theta out of rope_parameters, not rope_theta
    assert config.rope_parameters["rope_theta"] == 500000.0


def test_head_dim_matches_what_sched_derives():
    """sched.py computes HEAD_DIM itself; the stub must agree with it."""
    config = dry_build.build_stub_config()
    assert config.hidden_size // config.num_attention_heads == dry_build.HEAD_DIM


def test_vocab_fits_the_padded_logits_buffer():
    """sched.py pads lm_head to logits_slice * logits_epoch and zeroes the tail.

    If the stub's vocab were larger than that padded size the resize would
    truncate real rows, and the dry build would validate a shape the real run
    never uses.
    """
    config = dry_build.build_stub_config()
    logits_slice = 8192 * 8
    logits_epoch = 2
    assert config.vocab_size <= logits_slice * logits_epoch


def test_stub_model_exposes_every_attribute_sched_reads():
    config = tiny_config()
    model = dry_build.build_stub_model(CPU, dtype=torch.float32, config=config)

    # the module-level reads in sched.py
    assert model.dtype == torch.float32
    assert model.model.embed_tokens.weight.shape == (256, 64)
    assert model.model.norm.weight.shape == (64,)
    assert model.lm_head.weight.shape == (256, 64)

    layers = model.model.layers
    assert len(layers) == 2
    for layer in layers:
        assert layer.input_layernorm.weight.shape == (64,)
        assert layer.post_attention_layernorm.weight.shape == (64,)
        # q is head_dim * num_attention_heads, k and v use the kv head count
        assert layer.self_attn.q_proj.weight.shape == (64, 64)
        assert layer.self_attn.k_proj.weight.shape == (32, 64)
        assert layer.self_attn.v_proj.weight.shape == (32, 64)
        assert layer.self_attn.o_proj.weight.shape == (64, 64)
        assert layer.mlp.up_proj.weight.shape == (128, 64)
        assert layer.mlp.gate_proj.weight.shape == (128, 64)
        assert layer.mlp.down_proj.weight.shape == (64, 128)


def test_lm_head_can_be_detached_and_resized():
    """sched.py does `model.lm_head.weight.detach()` then `.resize_()`."""
    config = tiny_config()
    model = dry_build.build_stub_model(CPU, dtype=torch.float32, config=config)
    weight = model.lm_head.weight.detach()
    weight.resize_(512, 64)
    assert weight.shape == (512, 64)


def test_rotary_embedding_returns_cos_and_sin_of_the_right_shape():
    config = tiny_config()
    model = dry_build.build_stub_model(CPU, dtype=torch.float32, config=config)
    positions = torch.arange(32).unsqueeze(0)
    cos, sin = model.model.rotary_emb(torch.zeros(1), positions)
    # sched.py indexes [0, :, :HEAD_DIM // 2] and assigns into a rope table
    assert cos.shape == (1, 32, 16)
    assert sin.shape == (1, 32, 16)
    assert cos.dtype == torch.float32


def test_rope_values_are_finite():
    """A NaN here would propagate into matRope and derail a shape calculation."""
    config = tiny_config()
    model = dry_build.build_stub_model(CPU, dtype=torch.float32, config=config)
    cos, sin = model.model.rotary_emb(torch.zeros(1), torch.arange(32).unsqueeze(0))
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()
    # position 0 is the identity rotation
    assert torch.allclose(cos[0, 0], torch.ones_like(cos[0, 0]))
    assert torch.allclose(sin[0, 0], torch.zeros_like(sin[0, 0]))


def real_config_or_none():
    """The published Llama-3.1-8B-Instruct config, if this host can reach it.

    Returns None when there is no Hugging Face credential or no network, so
    the suite still runs on a host that has neither.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    try:
        path = hf_hub_download(
            "meta-llama/Llama-3.1-8B-Instruct",
            "config.json",
            cache_dir=os.environ.get("HF_CACHE_DIR", "/tmp/huggingface_cache"),
        )
    except Exception:
        return None
    import json
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_stub_geometry_matches_the_published_config():
    """The legality sweep is only meaningful if the stub's shapes are real.

    Every candidate in the Llama3 legality map was judged against these
    numbers. If one drifts from the published config, that whole map becomes
    quietly wrong rather than loudly broken, so pin it here.

    Skipped when the model is unreachable, since it is gated.
    """
    real = real_config_or_none()
    if real is None:
        print("     (skipped: no Hugging Face credential or no network)")
        return

    stub = dry_build.build_stub_config()
    for field in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "vocab_size",
        "rms_norm_eps",
    ):
        assert getattr(stub, field) == real[field], (
            f"{field}: stub has {getattr(stub, field)}, published config has {real[field]}"
        )

    # rope_theta moved between config layouts across transformers versions
    real_theta = real.get("rope_theta")
    if real_theta is None:
        real_theta = (real.get("rope_parameters") or {}).get("rope_theta")
    assert stub.rope_parameters["rope_theta"] == real_theta


def test_stub_tokenizer_refuses_to_pretend_it_can_tokenize():
    """Better a clear error than silently scheduling the wrong token count."""
    tokenizer = dry_build.build_stub_tokenizer()
    assert tokenizer.chat_template is None
    assert tokenizer.eos_token_id in dry_build.EOS_TOKEN_IDS
    for call in (tokenizer.decode, tokenizer.apply_chat_template):
        try:
            call("anything")
        except RuntimeError as exc:
            assert "dry-build" in str(exc)
        else:
            raise AssertionError("expected the stub tokenizer to refuse")


def test_layer_count_can_be_overridden_without_touching_the_config():
    config = tiny_config()
    model = dry_build.build_stub_model(CPU, dtype=torch.float32, config=config, num_layers=1)
    assert len(model.model.layers) == 1


def main():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report and keep going
            failures += 1
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
