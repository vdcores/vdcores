from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import runtime
from .op_family_specs import ComputeFamilyDefinition, parse_comp_family_runtime_specs


@dataclass(frozen=True)
class ComputeOpFamilyRef:
    canonical_name: str


@lru_cache(maxsize=1)
def _load_comp_family_definitions() -> dict[str, ComputeFamilyDefinition]:
    raw_specs = getattr(runtime, "compute_family_specs", None)
    if raw_specs is None:
        raise ValueError("dae.runtime does not export compute_family_specs")
    definitions = parse_comp_family_runtime_specs(list(raw_specs))
    if not definitions:
        raise ValueError("dae.runtime exported no compute family specs")
    return definitions


def _bool_to_int(value: bool) -> int:
    return 1 if value else 0


def _get_family_definition(family: str) -> ComputeFamilyDefinition:
    try:
        return _load_comp_family_definitions()[family.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown compute family definition for {family}") from exc


def _mangle_name(definition: ComputeFamilyDefinition, values: dict[str, int]) -> str:
    parts = [f"OP_{definition.family}"]
    parts.extend(f"{field}_{values[field]}" for field in definition.fields)
    return "__".join(parts)


def _validate_field_constraints(definition: ComputeFamilyDefinition, values: dict[str, int]) -> None:
    for field in definition.fields:
        value = values[field]
        fixed_key = f"{field}_FIXED"
        multiple_key = f"{field}_MULTIPLE"
        min_key = f"{field}_MIN"
        max_key = f"{field}_MAX"

        if fixed_key in definition.constraints and value != definition.constraints[fixed_key]:
            raise ValueError(f"Unsupported {definition.family} {field.lower()}={value}; expected {definition.constraints[fixed_key]}")
        if multiple_key in definition.constraints and (value <= 0 or value % definition.constraints[multiple_key] != 0):
            raise ValueError(
                f"Unsupported {definition.family} {field.lower()}={value}; "
                f"expected a positive multiple of {definition.constraints[multiple_key]}"
            )
        if min_key in definition.constraints and value < definition.constraints[min_key]:
            raise ValueError(f"Unsupported {definition.family} {field.lower()}={value}; expected >= {definition.constraints[min_key]}")
        if max_key in definition.constraints and value > definition.constraints[max_key]:
            raise ValueError(f"Unsupported {definition.family} {field.lower()}={value}; expected <= {definition.constraints[max_key]}")


def _build_family_ref(family: str, values: dict[str, int]) -> ComputeOpFamilyRef:
    definition = _get_family_definition(family)
    normalized = {field.upper(): int(value) for field, value in values.items()}
    missing = [field for field in definition.fields if field not in normalized]
    if missing:
        raise ValueError(f"Missing {family} family fields: {missing}")
    extras = [field for field in normalized if field not in definition.fields]
    if extras:
        raise ValueError(f"Unexpected {family} family fields: {extras}")
    _validate_field_constraints(definition, normalized)
    return ComputeOpFamilyRef(_mangle_name(definition, normalized))


class GemvFamily:
    FAMILY = "OP_GEMV"

    @classmethod
    def create_wgmma(
        cls,
        *,
        m: int,
        n: int,
        k: int,
        bload: int,
        residual: bool = False,
    ) -> ComputeOpFamilyRef:
        return _build_family_ref(
            "GEMV_WGMMA",
            {
                "M": m,
                "N": n,
                "K": k,
                "BLOAD": bload,
                "RESIDUAL": _bool_to_int(residual),
            },
        )

    @classmethod
    def create_mma(cls, *, m: int, n: int, k: int) -> ComputeOpFamilyRef:
        return _build_family_ref(
            "GEMV_MMA",
            {
                "M": m,
                "N": n,
                "K": k,
            },
        )


def gemv_family_spec_by_name(name: str) -> dict[str, int | str] | None:
    if not isinstance(name, str) or not name.startswith("OP_"):
        return None

    parts = name.split("__")
    if len(parts) < 2:
        return None

    family = parts[0][3:].upper()

    try:
        definition = _get_family_definition(family)
    except ValueError:
        return None

    if len(parts) != 1 + len(definition.fields):
        raise ValueError(
            f"Malformed {family} compute-op name: {name}. "
            f"Expected fields {definition.fields}."
        )

    values: dict[str, int] = {}
    for field, token in zip(definition.fields, parts[1:]):
        prefix = f"{field}_"
        if not token.startswith(prefix):
            raise ValueError(
                f"Malformed {family} compute-op name: {name}. "
                f"Expected token starting with {prefix}."
            )
        raw_value = token[len(prefix):]
        try:
            values[field] = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"Malformed {family} compute-op name: {name}. {field} must be an integer.") from exc

    _validate_field_constraints(definition, values)
    return {
        "name": _mangle_name(definition, values),
        "family": family.lower(),
        **{field.lower(): value for field, value in values.items()},
    }
    return None


def is_registered_family_name(name: str) -> bool:
    return gemv_family_spec_by_name(name) is not None


def validate_family_name(name: str) -> str:
    if not isinstance(name, str):
        raise TypeError(f"Expected family opcode name as str, got {type(name).__name__}")
    if not is_registered_family_name(name):
        raise ValueError(f"Unknown dynamic compute-op family name: {name}")
    return name


def family_name(ref: ComputeOpFamilyRef | str) -> str:
    if isinstance(ref, ComputeOpFamilyRef):
        return ref.canonical_name
    return validate_family_name(ref)


__all__ = [
    "ComputeFamilyDefinition",
    "ComputeOpFamilyRef",
    "GemvFamily",
    "family_name",
    "gemv_family_spec_by_name",
    "is_registered_family_name",
    "validate_family_name",
]
