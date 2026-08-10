import json
import struct

import pytest
import torch

from dae.deepseek_v4 import DeepSeekV4FlashConfig
from dae.deepseek_v4_checkpoint import (
    DeepSeekV4Checkpoint,
    expected_inference_tensor_specs,
    read_safetensors_header,
)


def _write_index(root, weight_map, total_size=0):
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
        )
    )


def _small_config():
    return DeepSeekV4FlashConfig(
        num_layers=1,
        num_hash_layers=1,
        num_experts=2,
        compress_ratios=(0, 0),
    )


def test_expected_checkpoint_contract_has_raw_fp8_and_nvfp4_layouts():
    specs = expected_inference_tensor_specs()

    assert len(specs) == 133660
    assert sum(spec.nbytes for spec in specs.values()) == 164673005788
    assert specs["head.weight"].dtype == "BF16"
    assert specs["head.weight"].shape == (129280, 4096)
    assert specs["layers.2.attn.indexer.wq_b.scale"].dtype == "F8_E8M0"
    assert specs["layers.2.attn.indexer.wq_b.scale"].shape == (64, 8)
    expert = "layers.42.ffn.experts.255.w2"
    assert specs[f"{expert}.weight"].shape == (4096, 1024)
    assert specs[f"{expert}.weight_scale"].shape == (4096, 128)
    assert specs[f"{expert}.weight_scale_2"].shape == ()


def test_name_only_audit_accepts_mtp_but_rejects_missing_inference(tmp_path):
    config = _small_config()
    expected = expected_inference_tensor_specs(config)
    weight_map = {name: "model-00001-of-00001.safetensors" for name in expected}
    weight_map["mtp.0.placeholder"] = "model-00001-of-00001.safetensors"
    _write_index(tmp_path, weight_map, total_size=123)

    checkpoint = DeepSeekV4Checkpoint(tmp_path, config)
    audit = checkpoint.audit(require_files=False)

    assert audit.inference_tensor_count == len(expected)
    assert audit.mtp_tensor_count == 1
    assert audit.shard_count == 1
    assert audit.tensor_bytes == 123

    del weight_map[next(iter(expected))]
    _write_index(tmp_path, weight_map)
    with pytest.raises(ValueError, match="missing="):
        DeepSeekV4Checkpoint(tmp_path, config).audit(require_files=False)


def test_safetensors_header_inspection_validates_dtype_shape_and_offsets(tmp_path):
    header = {
        "value": {
            "dtype": "F32",
            "shape": [2],
            "data_offsets": [0, 8],
        }
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\x00" * 8)
    _write_index(tmp_path, {"value": shard.name}, total_size=8)

    specs = read_safetensors_header(shard)
    assert specs["value"].dtype == "F32"
    assert specs["value"].shape == (2,)
    assert specs["value"].nbytes == 8

    checkpoint = DeepSeekV4Checkpoint(tmp_path, _small_config())
    assert checkpoint.inspect(["value"])["value"].filename == shard.name
    loaded = checkpoint.load_tensors(["value"])
    torch.testing.assert_close(loaded["value"], torch.zeros(2))


def test_checkpoint_index_rejects_shard_path_traversal(tmp_path):
    _write_index(tmp_path, {"value": "../outside.safetensors"})
    with pytest.raises(ValueError, match="unsafe checkpoint shard"):
        DeepSeekV4Checkpoint(tmp_path, _small_config())
