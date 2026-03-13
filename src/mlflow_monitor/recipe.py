"""Recipe schema and parser for MLflow-Monitor v0-lite."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_REQUIRED_TOP_LEVEL_SECTIONS = {
    "identity",
    "input_binding",
    "contract_binding",
    "metrics_slices",
    "finding_policy",
    "output_binding",
}


@dataclass(frozen=True, slots=True)
class RecipeIdentity:
    """Identity and version metadata for a recipe."""

    recipe_id: str
    version: str


@dataclass(frozen=True, slots=True)
class RecipeInputBinding:
    """Input binding section for source run selection and required evidence."""

    run_selector: str
    source_experiment: str | None = None
    required_metrics: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    custom_reference_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeContractBinding:
    """Contract binding section for resolved contract selection."""

    contract_id: str


@dataclass(frozen=True, slots=True)
class RecipeMetricsSlices:
    """Metrics and slices section for analysis selection."""

    metrics: tuple[str, ...] = ()
    slices: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecipeFindingPolicy:
    """Finding policy section for interpreted output behavior."""

    profile: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeOutputBinding:
    """Output binding section for summary rendering options."""

    summary_mode: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeV0Lite:
    """Canonical v0-lite recipe runtime model."""

    identity: RecipeIdentity
    input_binding: RecipeInputBinding
    contract_binding: RecipeContractBinding
    metrics_slices: RecipeMetricsSlices
    finding_policy: RecipeFindingPolicy
    output_binding: RecipeOutputBinding


def parse_recipe_v0_lite(raw: Mapping[str, object]) -> RecipeV0Lite:
    """Parse a mapping into the canonical v0-lite recipe model."""
    missing_sections = sorted(_REQUIRED_TOP_LEVEL_SECTIONS - set(raw.keys()))
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Missing required recipe section(s): {missing}")

    unknown_sections = sorted(set(raw.keys()) - _REQUIRED_TOP_LEVEL_SECTIONS)
    if unknown_sections:
        unknown = ", ".join(unknown_sections)
        raise ValueError(f"Unknown/disallowed recipe section(s): {unknown}")

    identity = _parse_identity(_require_section(raw, "identity"))
    input_binding = _parse_input_binding(_require_section(raw, "input_binding"))
    contract_binding = _parse_contract_binding(_require_section(raw, "contract_binding"))
    metrics_slices = _parse_metrics_slices(_require_section(raw, "metrics_slices"))
    finding_policy = _parse_finding_policy(_require_section(raw, "finding_policy"))
    output_binding = _parse_output_binding(_require_section(raw, "output_binding"))

    return RecipeV0Lite(
        identity=identity,
        input_binding=input_binding,
        contract_binding=contract_binding,
        metrics_slices=metrics_slices,
        finding_policy=finding_policy,
        output_binding=output_binding,
    )


def _require_section(raw: Mapping[str, object], section_name: str) -> Mapping[str, object]:
    """Return a required section and ensure it is a mapping."""

    section = raw[section_name]
    if not isinstance(section, Mapping):
        raise ValueError(f"Section '{section_name}' must be a mapping.")
    return section


def _parse_identity(section: Mapping[str, object]) -> RecipeIdentity:
    """Parse identity section values."""

    return RecipeIdentity(
        recipe_id=_require_string(section, "recipe_id", "identity"),
        version=_require_string(section, "version", "identity"),
    )


def _parse_input_binding(section: Mapping[str, object]) -> RecipeInputBinding:
    """Parse input binding section values."""

    return RecipeInputBinding(
        run_selector=_require_string(section, "run_selector", "input_binding"),
        source_experiment=_optional_string(section, "source_experiment", "input_binding"),
        required_metrics=_optional_string_tuple(section, "required_metrics", "input_binding"),
        required_artifacts=_optional_string_tuple(section, "required_artifacts", "input_binding"),
        custom_reference_run_id=_optional_string(
            section,
            "custom_reference_run_id",
            "input_binding",
        ),
    )


def _parse_contract_binding(section: Mapping[str, object]) -> RecipeContractBinding:
    """Parse contract binding section values."""

    return RecipeContractBinding(
        contract_id=_require_string(section, "contract_id", "contract_binding"),
    )


def _parse_metrics_slices(section: Mapping[str, object]) -> RecipeMetricsSlices:
    """Parse metrics and slices section values."""

    return RecipeMetricsSlices(
        metrics=_optional_string_tuple(section, "metrics", "metrics_slices"),
        slices=_optional_string_tuple(section, "slices", "metrics_slices"),
    )


def _parse_finding_policy(section: Mapping[str, object]) -> RecipeFindingPolicy:
    """Parse finding policy section values."""

    return RecipeFindingPolicy(
        profile=_optional_string(section, "profile", "finding_policy"),
    )


def _parse_output_binding(section: Mapping[str, object]) -> RecipeOutputBinding:
    """Parse output binding section values."""

    return RecipeOutputBinding(
        summary_mode=_optional_string(section, "summary_mode", "output_binding"),
    )


def _require_string(section: Mapping[str, object], key: str, section_name: str) -> str:
    """Require one non-empty string field from a section."""

    if key not in section:
        raise ValueError(f"Section '{section_name}' missing required field '{key}'.")
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"Field '{section_name}.{key}' must be a string.")
    if not value:
        raise ValueError(f"Field '{section_name}.{key}' must be non-empty.")
    return value


def _optional_string(section: Mapping[str, object], key: str, section_name: str) -> str | None:
    """Return one optional non-empty string field from a section."""

    if key not in section or section[key] is None:
        return None
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"Field '{section_name}.{key}' must be a string when provided.")
    if not value:
        raise ValueError(f"Field '{section_name}.{key}' must be non-empty when provided.")
    return value


def _optional_string_tuple(
    section: Mapping[str, object],
    key: str,
    section_name: str,
) -> tuple[str, ...]:
    """Return an optional sequence of non-empty strings as an immutable tuple."""

    if key not in section or section[key] is None:
        return ()

    raw_value = section[key]
    if isinstance(raw_value, str) or not isinstance(raw_value, Sequence):
        raise ValueError(f"Field '{section_name}.{key}' must be a sequence of strings.")

    values: list[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            raise ValueError(f"Field '{section_name}.{key}' must contain only strings.")
        if not item:
            raise ValueError(f"Field '{section_name}.{key}' must not contain empty strings.")
        values.append(item)
    return tuple(values)
