#!/usr/bin/env python3
"""Audit local files or range-read official DeepSeek-V4 checkpoint headers."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from collections import defaultdict

from dae.deepseek_v4_checkpoint import (
    DeepSeekV4Checkpoint,
    expected_inference_tensor_specs,
    read_safetensors_header_url,
)


DEFAULT_REPO = "nvidia/DeepSeek-V4-Flash-NVFP4"
PINNED_REVISION = "7fc18be2b215ae48260383d4a228ec8a033046f7"


def _remote_url(repo: str, revision: str, filename: str) -> str:
    encoded_repo = "/".join(
        urllib.parse.quote(part, safe="") for part in repo.split("/")
    )
    encoded_revision = urllib.parse.quote(revision, safe="")
    encoded_filename = urllib.parse.quote(filename, safe="")
    return (
        f"https://huggingface.co/{encoded_repo}/resolve/{encoded_revision}/"
        f"{encoded_filename}?download=true"
    )


def audit_remote(repo: str, revision: str) -> None:
    index_url = _remote_url(repo, revision, "model.safetensors.index.json")
    with urllib.request.urlopen(index_url, timeout=60) as response:
        index = json.load(response)
    weight_map = index["weight_map"]
    expected = expected_inference_tensor_specs()
    actual_names = set(weight_map)
    missing = sorted(set(expected) - actual_names)
    unexpected = sorted(
        name for name in actual_names - set(expected) if not name.startswith("mtp.")
    )
    if missing or unexpected:
        raise ValueError(
            f"remote checkpoint name mismatch: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )

    by_shard: dict[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        by_shard[filename].append(name)
    inspected = {}
    for ordinal, filename in enumerate(sorted(by_shard), start=1):
        print(
            f"DSV4_CHECKPOINT_SHARD status=READING ordinal={ordinal} "
            f"total={len(by_shard)} file={filename}",
            flush=True,
        )
        header = read_safetensors_header_url(
            _remote_url(repo, revision, filename),
            filename=filename,
        )
        for name in by_shard[filename]:
            if name not in header:
                raise ValueError(f"{filename} header is missing indexed tensor {name}")
            inspected[name] = header[name]
        print(
            f"DSV4_CHECKPOINT_SHARD status=PASS ordinal={ordinal} "
            f"total={len(by_shard)} file={filename} tensors={len(header)}",
            flush=True,
        )

    for name, expected_spec in expected.items():
        actual = inspected[name]
        if (actual.dtype, actual.shape) != (expected_spec.dtype, expected_spec.shape):
            raise ValueError(
                f"{name} expected {expected_spec.dtype}{expected_spec.shape}, "
                f"got {actual.dtype}{actual.shape}"
            )
    tensor_bytes = sum(spec.nbytes for spec in inspected.values())
    inference_bytes = sum(inspected[name].nbytes for name in expected)
    mtp_bytes = tensor_bytes - inference_bytes
    metadata_bytes = index.get("metadata", {}).get("total_size", 0)
    if metadata_bytes and tensor_bytes != metadata_bytes:
        raise ValueError(
            f"header bytes {tensor_bytes} do not match index bytes {metadata_bytes}"
        )
    print(
        "DSV4_CHECKPOINT_AUDIT status=PASS mode=remote_headers "
        f"repo={repo} revision={revision} tensors={len(inspected)} "
        f"inference={len(expected)} mtp={len(inspected) - len(expected)} "
        f"shards={len(by_shard)} bytes={tensor_bytes} "
        f"inference_bytes={inference_bytes} mtp_bytes={mtp_bytes}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint")
    source.add_argument("--remote", action="store_true")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=PINNED_REVISION)
    args = parser.parse_args()

    if args.remote:
        audit_remote(args.repo, args.revision)
        return
    audit = DeepSeekV4Checkpoint(args.checkpoint).audit(require_files=True)
    print(
        "DSV4_CHECKPOINT_AUDIT status=PASS mode=local "
        f"tensors={audit.tensor_count} inference={audit.inference_tensor_count} "
        f"mtp={audit.mtp_tensor_count} shards={audit.shard_count} "
        f"bytes={audit.tensor_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
