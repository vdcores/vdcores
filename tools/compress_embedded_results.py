"""Losslessly archive this experiment's raw JSON, preserving policy usability.

Only generated embedded-loop data with a recognized experiment label is handled.
Verify a byte-for-byte decompression before replacing the original with .gz.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    files, before, after = 0, 0, 0
    for path in sorted(args.directory.rglob("*.json")):
        raw = path.read_bytes()
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("experiment") not in (
            "vdcores_embedded_fixed_feedback_loop", "vdcores_embedded_placement_calibration"
        ):
            continue
        target = path.with_suffix(".json.gz")
        if target.exists():
            if gzip.decompress(target.read_bytes()) != raw:
                raise RuntimeError(f"archive collision: {target}")
        else:
            target.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
        if gzip.decompress(target.read_bytes()) != raw:
            raise RuntimeError(f"archive verification failed: {target}")
        before += len(raw)
        after += target.stat().st_size
        path.unlink()
        files += 1
    print(f"Archived {files} generated result files: {before/1e6:.1f} MB -> {after/1e6:.1f} MB; losslessly recoverable with gunzip")


if __name__ == "__main__":
    main()
