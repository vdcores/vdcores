from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ComputeFamilyDefinition:
    family: str
    fields: tuple[str, ...]
    constraints: dict[str, int]


_COMP_FAMILY_PATTERN = re.compile(r"^\s*DAE_DEFINE_COMP_FAMILY\(\s*([A-Za-z0-9_]+)\s*,\s*(.+)\)\s*(?://.*)?$")


def parse_comp_family_definition(family: str, raw_definition: str) -> ComputeFamilyDefinition:
    values: dict[str, str] = {}
    for part in raw_definition.split(","):
        token = part.strip()
        if "=" not in token:
            raise ValueError(f"Malformed family definition token for {family}: {token}")
        key, value = (item.strip() for item in token.split("=", 1))
        values[key.upper()] = value

    if "FIELDS" not in values:
        raise ValueError(f"Malformed family definition for {family}: missing FIELDS")

    fields = tuple(field.strip().upper() for field in values.pop("FIELDS").split("|"))
    constraints = {key: int(value) for key, value in values.items()}
    return ComputeFamilyDefinition(
        family=family.upper(),
        fields=fields,
        constraints=constraints,
    )


def parse_comp_family_registry_lines(lines: list[str]) -> dict[str, ComputeFamilyDefinition]:
    definitions: dict[str, ComputeFamilyDefinition] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = _COMP_FAMILY_PATTERN.match(line)
        if match is None:
            continue

        family, raw_definition = match.groups()
        definition = parse_comp_family_definition(family, raw_definition)
        definitions[definition.family] = definition
    return definitions


def parse_comp_family_runtime_specs(specs: list[dict[str, str]]) -> dict[str, ComputeFamilyDefinition]:
    definitions: dict[str, ComputeFamilyDefinition] = {}
    for spec in specs:
        family = str(spec["family"])
        raw_definition = str(spec["definition"])
        definition = parse_comp_family_definition(family, raw_definition)
        definitions[definition.family] = definition
    return definitions


__all__ = [
    "ComputeFamilyDefinition",
    "parse_comp_family_definition",
    "parse_comp_family_registry_lines",
    "parse_comp_family_runtime_specs",
]
