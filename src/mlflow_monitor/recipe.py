"""Recipe schema and parser for MLflow-Monitor v0-lite.

This module defines the canonical in-memory recipe representation used by the
M1 recipe pipeline. It accepts only mapping inputs and intentionally excludes
file-format parsing (for example JSON/YAML text decoding), which is deferred.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mlflow_monitor.errors import RecipeValidationError, RecipeValidationIssue

_REQUIRED_TOP_LEVEL_SECTIONS = {
    "identity",
    "input_binding",
    "contract_binding",
    "metrics_slices",
    "finding_policy",
    "output_binding",
}

_ALLOWED_SECTION_FIELDS: dict[str, frozenset[str]] = {
    "identity": frozenset({"recipe_id", "version"}),
    "input_binding": frozenset(
        {
            "run_selector",
            "source_experiment",
            "required_metrics",
            "required_artifacts",
            "custom_reference_run_id",
        }
    ),
    "contract_binding": frozenset({"contract_id"}),
    "metrics_slices": frozenset({"metrics", "slices"}),
    "finding_policy": frozenset({"profile"}),
    "output_binding": frozenset({"summary_mode"}),
}


@dataclass(frozen=True, slots=True)
class RecipeReferenceCatalog:
    """Allowlisted external references used during recipe validation.

    Attributes:
        contract_ids: Valid contract IDs for recipe binding.
        finding_policy_profiles: Valid finding policy profile IDs.
        summary_modes: Valid output summary mode IDs.
    """

    contract_ids: frozenset[str]
    finding_policy_profiles: frozenset[str]
    summary_modes: frozenset[str]


@dataclass(frozen=True, slots=True)
class RecipeIdentity:
    """Identity and version metadata for a recipe.

    Attributes:
        recipe_id: Stable recipe identifier.
        version: Recipe version identifier.
    """

    recipe_id: str
    version: str


@dataclass(frozen=True, slots=True)
class RecipeInputBinding:
    """Input binding section for source run selection and required evidence.

    Attributes:
        run_selector: Selector used to resolve a source run.
        source_experiment: Optional source training experiment name.
        required_metrics: Optional metric names that must exist on the source run.
        required_artifacts: Optional artifact names that must exist on the source run.
        custom_reference_run_id: Optional additional same-timeline reference run id
            used for analysis diffs. This is not the baseline reference and does
            not replace default baseline/previous/LKG comparisons.
    """

    run_selector: str
    source_experiment: str | None = None
    required_metrics: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    custom_reference_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeContractBinding:
    """Contract binding section for resolved contract selection.

    Attributes:
        contract_id: Resolved contract identifier.
    """

    contract_id: str


@dataclass(frozen=True, slots=True)
class RecipeMetricsSlices:
    """Metrics and slices section for analysis selection.

    Attributes:
        metrics: Optional metric names selected by the recipe.
        slices: Optional slice names selected by the recipe.
    """

    metrics: tuple[str, ...] = ()
    slices: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecipeFindingPolicy:
    """Finding policy section for interpreted output behavior.

    Attributes:
        profile: Optional finding policy profile identifier.
    """

    profile: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeOutputBinding:
    """Output binding section for summary rendering options.

    Attributes:
        summary_mode: Optional summary mode identifier.
    """

    summary_mode: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeV0Lite:
    """Canonical v0-lite recipe runtime model.

    Attributes:
        identity: Recipe identity and version metadata.
        input_binding: Source run and evidence binding details.
        contract_binding: Resolved contract binding.
        metrics_slices: Metrics and slices selection.
        finding_policy: Finding policy configuration.
        output_binding: Output rendering and summary options.
    """

    identity: RecipeIdentity
    input_binding: RecipeInputBinding
    contract_binding: RecipeContractBinding
    metrics_slices: RecipeMetricsSlices
    finding_policy: RecipeFindingPolicy
    output_binding: RecipeOutputBinding


def validate_recipe_v0_lite(
    raw: Mapping[str, object],
    references: RecipeReferenceCatalog,
) -> RecipeV0Lite:
    """Validate and parse one v0-lite recipe payload.

    Args:
        raw: Raw recipe mapping containing v0-lite top-level sections.
        references: Allowlisted reference IDs for recipe bindings.

    Returns:
        Parsed recipe runtime model when validation succeeds.

    Raises:
        RecipeValidationError: If one or more validation checks fail.
    """
    try:
        parsed = parse_recipe_v0_lite(raw)
    except ValueError as exc:
        section, field = _extract_error_location(str(exc))
        raise RecipeValidationError(
            issues=(
                RecipeValidationIssue(
                    code="structural_error",
                    section=section,
                    field=field,
                    message=str(exc),
                ),
            )
        ) from exc

    issues: list[RecipeValidationIssue] = []
    issues.extend(_collect_unknown_nested_key_issues(raw))
    issues.extend(_collect_reference_issues(parsed, references))
    issues.extend(_collect_constraint_issues(parsed))

    if issues:
        raise RecipeValidationError(issues=tuple(issues))
    return parsed


def parse_recipe_v0_lite(raw: Mapping[str, object]) -> RecipeV0Lite:
    """Parse a mapping into the canonical v0-lite recipe model.

    Args:
        raw: Raw recipe mapping containing v0-lite top-level sections.

    Returns:
        Parsed recipe model with strongly typed section objects.

    Raises:
        ValueError: If required sections are missing, unknown sections are
            present, or any section value is malformed.
    """
    if not isinstance(raw, Mapping):
        raise ValueError("Recipe payload must be a mapping.")

    raw_keys = tuple(raw.keys())
    if any(not isinstance(key, str) for key in raw_keys):
        raise ValueError("Top-level recipe section names must be strings.")

    missing_sections = sorted(_REQUIRED_TOP_LEVEL_SECTIONS - set(raw_keys))
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Missing required recipe section(s): {missing}")

    unknown_sections = sorted(set(raw_keys) - _REQUIRED_TOP_LEVEL_SECTIONS)
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
    """Return a required section and ensure it is a mapping.

    Args:
        raw: Raw recipe mapping.
        section_name: Required top-level section key.

    Returns:
        Section mapping for the given key.

    Raises:
        ValueError: If the section is not a mapping.
    """
    section = raw[section_name]
    if not isinstance(section, Mapping):
        raise ValueError(f"Section '{section_name}' must be a mapping.")
    return section


def _collect_unknown_nested_key_issues(raw: Mapping[str, object]) -> list[RecipeValidationIssue]:
    """Collect issues for unknown keys nested inside known top-level sections."""
    issues: list[RecipeValidationIssue] = []

    for section_name in sorted(_REQUIRED_TOP_LEVEL_SECTIONS):
        section_raw = raw[section_name]
        if not isinstance(section_raw, Mapping):
            continue

        allowed_fields = _ALLOWED_SECTION_FIELDS[section_name]
        for field in sorted(section_raw.keys()):
            if not isinstance(field, str):
                continue
            if field in allowed_fields:
                continue
            issues.append(
                RecipeValidationIssue(
                    code="unknown_field",
                    section=section_name,
                    field=field,
                    message=f"Unknown/disallowed field '{section_name}.{field}'.",
                )
            )
    return issues


def _collect_reference_issues(
    recipe: RecipeV0Lite,
    references: RecipeReferenceCatalog,
) -> list[RecipeValidationIssue]:
    """Collect issues for unknown external references in a parsed recipe."""
    issues: list[RecipeValidationIssue] = []

    if recipe.contract_binding.contract_id not in references.contract_ids:
        issues.append(
            RecipeValidationIssue(
                code="unknown_reference",
                section="contract_binding",
                field="contract_id",
                message=(
                    "Unknown reference 'contract_binding.contract_id': "
                    f"{recipe.contract_binding.contract_id}."
                ),
            )
        )

    profile = recipe.finding_policy.profile
    if profile is not None and profile not in references.finding_policy_profiles:
        issues.append(
            RecipeValidationIssue(
                code="unknown_reference",
                section="finding_policy",
                field="profile",
                message=f"Unknown reference 'finding_policy.profile': {profile}.",
            )
        )

    summary_mode = recipe.output_binding.summary_mode
    if summary_mode is not None and summary_mode not in references.summary_modes:
        issues.append(
            RecipeValidationIssue(
                code="unknown_reference",
                section="output_binding",
                field="summary_mode",
                message=f"Unknown reference 'output_binding.summary_mode': {summary_mode}.",
            )
        )

    return issues


def _collect_constraint_issues(recipe: RecipeV0Lite) -> list[RecipeValidationIssue]:
    """Collect issues for v0 recipe constraints independent of external systems."""
    issues: list[RecipeValidationIssue] = []

    run_selector = recipe.input_binding.run_selector
    if not run_selector.strip():
        issues.append(
            RecipeValidationIssue(
                code="invalid_constraint",
                section="input_binding",
                field="run_selector",
                message=("Field 'input_binding.run_selector' must contain a raw non-empty run ID."),
            )
        )
    if run_selector == "latest" or ":" in run_selector:
        issues.append(
            RecipeValidationIssue(
                code="invalid_constraint",
                section="input_binding",
                field="run_selector",
                message=(
                    "Field 'input_binding.run_selector' must be a raw run ID; "
                    "selector modes like 'latest' and prefixed values are not allowed."
                ),
            )
        )

    _add_duplicate_issues(
        issues=issues,
        section="input_binding",
        field="required_metrics",
        values=recipe.input_binding.required_metrics,
    )
    _add_duplicate_issues(
        issues=issues,
        section="input_binding",
        field="required_artifacts",
        values=recipe.input_binding.required_artifacts,
    )
    _add_duplicate_issues(
        issues=issues,
        section="metrics_slices",
        field="metrics",
        values=recipe.metrics_slices.metrics,
    )
    _add_duplicate_issues(
        issues=issues,
        section="metrics_slices",
        field="slices",
        values=recipe.metrics_slices.slices,
    )

    return issues


def _add_duplicate_issues(
    issues: list[RecipeValidationIssue],
    section: str,
    field: str,
    values: tuple[str, ...],
) -> None:
    """Append one issue if a tuple of strings contains duplicates."""
    if len(values) == len(set(values)):
        return
    issues.append(
        RecipeValidationIssue(
            code="invalid_constraint",
            section=section,
            field=field,
            message=f"Field '{section}.{field}' must not contain duplicate entries.",
        )
    )


def _extract_error_location(message: str) -> tuple[str, str | None]:
    """Extract section/field location from parser error text when possible."""
    field_match = re.search(r"Field '([a-z_]+)\.([a-z_]+)'", message)
    if field_match:
        return field_match.group(1), field_match.group(2)

    missing_field_match = re.search(
        r"Section '([a-z_]+)' missing required field '([a-z_]+)'",
        message,
    )
    if missing_field_match:
        return missing_field_match.group(1), missing_field_match.group(2)

    section_match = re.search(r"Section '([a-z_]+)'", message)
    if section_match:
        return section_match.group(1), None

    return "recipe", None


def _parse_identity(section: Mapping[str, object]) -> RecipeIdentity:
    """Parse identity section values.

    Args:
        section: Identity section mapping.

    Returns:
        Parsed RecipeIdentity value.
    """
    return RecipeIdentity(
        recipe_id=_require_string(section, "recipe_id", "identity"),
        version=_require_string(section, "version", "identity"),
    )


def _parse_input_binding(section: Mapping[str, object]) -> RecipeInputBinding:
    """Parse input binding section values.

    Args:
        section: Input binding section mapping.

    Returns:
        Parsed RecipeInputBinding value.
    """
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
    """Parse contract binding section values.

    Args:
        section: Contract binding section mapping.

    Returns:
        Parsed RecipeContractBinding value.
    """
    return RecipeContractBinding(
        contract_id=_require_string(section, "contract_id", "contract_binding"),
    )


def _parse_metrics_slices(section: Mapping[str, object]) -> RecipeMetricsSlices:
    """Parse metrics and slices section values.

    Args:
        section: Metrics/slices section mapping.

    Returns:
        Parsed RecipeMetricsSlices value.
    """
    return RecipeMetricsSlices(
        metrics=_optional_string_tuple(section, "metrics", "metrics_slices"),
        slices=_optional_string_tuple(section, "slices", "metrics_slices"),
    )


def _parse_finding_policy(section: Mapping[str, object]) -> RecipeFindingPolicy:
    """Parse finding policy section values.

    Args:
        section: Finding policy section mapping.

    Returns:
        Parsed RecipeFindingPolicy value.
    """
    return RecipeFindingPolicy(
        profile=_optional_string(section, "profile", "finding_policy"),
    )


def _parse_output_binding(section: Mapping[str, object]) -> RecipeOutputBinding:
    """Parse output binding section values.

    Args:
        section: Output binding section mapping.

    Returns:
        Parsed RecipeOutputBinding value.
    """
    return RecipeOutputBinding(
        summary_mode=_optional_string(section, "summary_mode", "output_binding"),
    )


def _require_string(section: Mapping[str, object], key: str, section_name: str) -> str:
    """Require one non-empty string field from a section.

    Args:
        section: Section mapping containing the field.
        key: Required field key.
        section_name: Section name used in deterministic error messages.

    Returns:
        Non-empty string value for the required key.

    Raises:
        ValueError: If the field is missing, non-string, or empty.
    """
    if key not in section:
        raise ValueError(f"Section '{section_name}' missing required field '{key}'.")
    value = section[key]
    if not isinstance(value, str):
        raise ValueError(f"Field '{section_name}.{key}' must be a string.")
    if not value:
        raise ValueError(f"Field '{section_name}.{key}' must be non-empty.")
    return value


def _optional_string(section: Mapping[str, object], key: str, section_name: str) -> str | None:
    """Return one optional non-empty string field from a section.

    Args:
        section: Section mapping containing the field.
        key: Optional field key.
        section_name: Section name used in deterministic error messages.

    Returns:
        Optional non-empty string value, or ``None`` when omitted.

    Raises:
        ValueError: If the field is provided and is non-string or empty.
    """
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
    """Return an optional sequence of non-empty strings as an immutable tuple.

    Args:
        section: Section mapping containing the field.
        key: Optional field key.
        section_name: Section name used in deterministic error messages.

    Returns:
        Tuple of string values. Returns empty tuple when omitted.

    Raises:
        ValueError: If provided value is not a sequence of non-empty strings.
    """
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
