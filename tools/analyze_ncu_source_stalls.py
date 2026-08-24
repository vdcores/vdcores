#!/usr/bin/env python3
"""Correlate an Nsight Compute source report with one inlined source file."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


STALL_COLUMNS = (
    "stall_barrier",
    "stall_long_sb",
    "stall_membar",
    "stall_no_inst",
    "stall_short_sb",
    "stall_wait",
)


def sass_sources(runtime_object: Path) -> dict[int, tuple[tuple[str, int], ...]]:
    with tempfile.TemporaryDirectory(prefix="vdcores-ncu-source-") as directory:
        subprocess.run(
            ["cuobjdump", "-xelf", "all", str(runtime_object.resolve())],
            cwd=directory,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        cubins = list(Path(directory).glob("*.cubin"))
        if len(cubins) != 1:
            raise RuntimeError(f"expected one cubin, found {len(cubins)}")
        disassembly = subprocess.run(
            ["nvdisasm", "-gi", str(cubins[0])],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout

    result: dict[int, tuple[tuple[str, int], ...]] = {}
    current: tuple[tuple[str, int], ...] = ()
    pending: list[tuple[str, int]] = []
    for text in disassembly.splitlines():
        if "//## File " in text:
            pending.extend(
                (path, int(line))
                for path, line in re.findall(r'"([^"]+)", line ([0-9]+)', text)
            )
            continue
        instruction = re.search(r"/\*([0-9a-fA-F]+)\*/", text)
        if instruction is None or ".byte" in text:
            continue
        if pending:
            current = tuple(dict.fromkeys(pending))
            pending.clear()
        result[int(instruction.group(1), 16)] = current
    return result


def source_rows(report: Path) -> tuple[list[str], list[list[str]]]:
    output = subprocess.run(
        [
            "ncu",
            "--import",
            str(report),
            "--page",
            "source",
            "--print-source",
            "sass",
            "--csv",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    rows = list(csv.reader(output.splitlines()))
    header = next(row for row in rows if row and row[0] == "Address")
    return header, [
        row
        for row in rows
        if len(row) == len(header) and row[0].startswith("0x")
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_object", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--source-suffix", required=True)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    mapping = sass_sources(args.runtime_object)
    header, rows = source_rows(args.report)
    column = {name: index for index, name in enumerate(header)}
    kernel_base = min(int(row[column["Address"]], 16) for row in rows)

    totals = Counter()
    by_line: dict[int, Counter] = defaultdict(Counter)
    locations = []
    for row in rows:
        offset = int(row[column["Address"]], 16) - kernel_base
        matching = [
            line
            for path, line in mapping.get(offset, ())
            if path.endswith(args.source_suffix)
        ]
        if not matching:
            continue
        line = matching[-1]
        values = {
            "samples": int(row[column["# Samples"]] or 0),
            "instructions": int(row[column["Instructions Executed"]] or 0),
            **{
                name: int(row[column[name]] or 0)
                for name in STALL_COLUMNS
            },
        }
        totals.update(values)
        by_line[line].update(values)
        if values["samples"]:
            locations.append(
                (
                    values["samples"],
                    line,
                    offset,
                    row[column["Source"]].strip(),
                    values,
                )
            )

    print(
        "NCU_SOURCE_STALL_SUMMARY "
        f"report={args.report} source={args.source_suffix} "
        f"samples={totals['samples']} instructions={totals['instructions']} "
        + " ".join(f"{name}={totals[name]}" for name in STALL_COLUMNS)
    )
    for samples, line, offset, sass, values in sorted(
        locations, reverse=True
    )[: args.top]:
        print(
            "NCU_SOURCE_STALL_LOCATION "
            f"line={line} offset=0x{offset:x} samples={samples} "
            + " ".join(f"{name}={values[name]}" for name in STALL_COLUMNS)
            + f" sass={sass}"
        )
    for line, values in sorted(
        by_line.items(), key=lambda item: item[1]["samples"], reverse=True
    )[: args.top]:
        if values["samples"]:
            print(
                "NCU_SOURCE_STALL_LINE "
                f"line={line} samples={values['samples']} "
                + " ".join(f"{name}={values[name]}" for name in STALL_COLUMNS)
            )
    for line, values in sorted(
        by_line.items(), key=lambda item: item[1]["instructions"], reverse=True
    )[: args.top]:
        if values["instructions"]:
            print(
                "NCU_SOURCE_INSTRUCTION_LINE "
                f"line={line} instructions={values['instructions']} "
                f"samples={values['samples']}"
            )


if __name__ == "__main__":
    main()
