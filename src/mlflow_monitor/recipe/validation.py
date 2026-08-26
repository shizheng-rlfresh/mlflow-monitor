"""Recipe validation module for mlflow-monitor v0."""

from __future__ import annotations

import math
from collections.abc import Mapping

from mlflow_monitor.errors import RecipeValidationIssue

from .models import RECIPE_SCHEMA_VERSION

_TOP_LEVEL_FIELDS = frozenset(
    {
        "recipe_schema_version",
        "identity",
        "source_requirements",
        "contract",
        "analysis",
    }
)
_REQUIRED_TOP_LEVEL_FIELDS = (
    "recipe_schema_version",
    "identity",
    "contract",
)
_SECTION_FIELDS = {
    "identity": frozenset({"recipe_id", "recipe_version"}),
    "source_requirements": frozenset(
        {"source_experiment", "required_metric_names", "required_artifact_paths"}
    ),
    "contract": frozenset({"contract_id", "contract_version"}),
    "analysis": frozenset({"metric_names", "finding_policy_bindings"}),
}
_REQUIRED_SECTION_FIELDS = {
    "identity": ("recipe_id", "recipe_version"),
    "source_requirements": (),
    "contract": ("contract_id", "contract_version"),
    "analysis": (),
}
_POLICY_BINDING_FIELDS = frozenset({"finding_policy_id", "finding_policy_version", "parameters"})
_REQUIRED_POLICY_BINDING_FIELDS = ("finding_policy_id", "finding_policy_version")


def as_string(value: object) -> str:
    """Narrow a string value after validation succeeds."""
    assert isinstance(value, str)
    return value


def collect_recipe_issues(raw: object) -> list[RecipeValidationIssue]:
    """Collect structural issues in canonical schema order."""
    if not isinstance(raw, Mapping):
        return [_issue("invalid_type", "recipe", None, "Recipe must be a mapping.")]

    issues: list[RecipeValidationIssue] = []
    if any(not isinstance(key, str) for key in raw):
        issues.append(
            _issue(
                "invalid_json_key",
                "recipe",
                None,
                "Recipe field names must be strings.",
            )
        )

    string_keys = {key for key in raw if isinstance(key, str)}
    for field in _REQUIRED_TOP_LEVEL_FIELDS:
        if field not in string_keys:
            issues.append(
                _issue(
                    "missing_field",
                    "recipe",
                    field,
                    f"Missing required field 'recipe.{field}'.",
                )
            )
    for field in sorted(string_keys - _TOP_LEVEL_FIELDS):
        issues.append(
            _issue(
                "unknown_field",
                "recipe",
                field,
                f"Unknown field 'recipe.{field}'.",
            )
        )

    if "recipe_schema_version" in raw:
        value = raw["recipe_schema_version"]
        if (
            _validate_nonempty_string(
                value,
                section="recipe",
                field="recipe_schema_version",
                issues=issues,
            )
            and value != RECIPE_SCHEMA_VERSION
        ):
            issues.append(
                _issue(
                    "unsupported_version",
                    "recipe",
                    "recipe_schema_version",
                    f"Field 'recipe.recipe_schema_version' must be '{RECIPE_SCHEMA_VERSION}'.",
                )
            )

    _validate_section(raw, "identity", issues)
    _validate_section(raw, "source_requirements", issues)
    _validate_section(raw, "contract", issues)
    _validate_section(raw, "analysis", issues)
    return issues


def _validate_section(
    raw: Mapping[object, object],
    section: str,
    issues: list[RecipeValidationIssue],
) -> None:
    """Validate one present Recipe section."""
    if section not in raw:
        return
    value = raw[section]
    if not isinstance(value, Mapping):
        issues.append(
            _issue(
                "invalid_type",
                section,
                None,
                f"Section '{section}' must be a mapping.",
            )
        )
        return

    if any(not isinstance(key, str) for key in value):
        issues.append(
            _issue(
                "invalid_json_key",
                section,
                None,
                f"Field names in section '{section}' must be strings.",
            )
        )
    string_keys = {key for key in value if isinstance(key, str)}
    for field in _REQUIRED_SECTION_FIELDS[section]:
        if field not in string_keys:
            issues.append(
                _issue(
                    "missing_field",
                    section,
                    field,
                    f"Missing required field '{section}.{field}'.",
                )
            )
    for field in sorted(string_keys - _SECTION_FIELDS[section]):
        issues.append(
            _issue(
                "unknown_field",
                section,
                field,
                f"Unknown field '{section}.{field}'.",
            )
        )

    if section == "identity":
        _validate_named_string(value, section, "recipe_id", issues)
        _validate_named_string(value, section, "recipe_version", issues)
    elif section == "source_requirements":
        _validate_optional_named_string(value, section, "source_experiment", issues)
        _validate_string_list(value, section, "required_metric_names", issues)
        _validate_string_list(value, section, "required_artifact_paths", issues)
    elif section == "contract":
        _validate_named_string(value, section, "contract_id", issues)
        _validate_named_string(value, section, "contract_version", issues)
    elif section == "analysis":
        _validate_string_list(value, section, "metric_names", issues)
        _validate_policy_bindings(value, issues)


def _validate_named_string(
    mapping: Mapping[object, object],
    section: str,
    field: str,
    issues: list[RecipeValidationIssue],
) -> None:
    """Validate a present required string field."""
    if field in mapping:
        _validate_nonempty_string(mapping[field], section=section, field=field, issues=issues)


def _validate_optional_named_string(
    mapping: Mapping[object, object],
    section: str,
    field: str,
    issues: list[RecipeValidationIssue],
) -> None:
    """Validate an optional string whose explicit null is invalid."""
    if field in mapping:
        _validate_nonempty_string(mapping[field], section=section, field=field, issues=issues)


def _validate_nonempty_string(
    value: object,
    *,
    section: str,
    field: str,
    issues: list[RecipeValidationIssue],
) -> bool:
    """Validate a nonempty string and report one located issue."""
    location = f"{section}.{field}"
    if not isinstance(value, str):
        issues.append(
            _issue(
                "invalid_type",
                section,
                field,
                f"Field '{location}' must be a string.",
            )
        )
        return False
    if not value.strip():
        issues.append(
            _issue(
                "empty_value",
                section,
                field,
                f"Field '{location}' must be non-empty.",
            )
        )
        return False
    return True


def _validate_string_list(
    mapping: Mapping[object, object],
    section: str,
    field: str,
    issues: list[RecipeValidationIssue],
) -> None:
    """Validate one optional unique list of nonempty strings."""
    if field not in mapping:
        return
    value = mapping[field]
    location = f"{section}.{field}"
    if not isinstance(value, list):
        issues.append(
            _issue(
                "invalid_type",
                section,
                field,
                f"Field '{location}' must be a list of strings.",
            )
        )
        return

    seen: set[str] = set()
    duplicate_reported = False
    for index, item in enumerate(value):
        item_field = f"{field}[{index}]"
        if not isinstance(item, str):
            issues.append(
                _issue(
                    "invalid_type",
                    section,
                    item_field,
                    f"Field '{section}.{item_field}' must be a string.",
                )
            )
            continue
        if not item.strip():
            issues.append(
                _issue(
                    "empty_value",
                    section,
                    item_field,
                    f"Field '{section}.{item_field}' must be non-empty.",
                )
            )
        if item in seen and not duplicate_reported:
            issues.append(
                _issue(
                    "duplicate_value",
                    section,
                    field,
                    f"Field '{location}' must not contain duplicate values.",
                )
            )
            duplicate_reported = True
        seen.add(item)


def _validate_policy_bindings(
    analysis: Mapping[object, object],
    issues: list[RecipeValidationIssue],
) -> None:
    """Validate optional three-state Finding-policy authoring."""
    field = "finding_policy_bindings"
    if field not in analysis:
        return
    value = analysis[field]
    if not isinstance(value, list):
        issues.append(
            _issue(
                "invalid_type",
                "analysis",
                field,
                "Field 'analysis.finding_policy_bindings' must be a list.",
            )
        )
        return

    seen: set[tuple[str, str]] = set()
    for index, binding in enumerate(value):
        prefix = f"finding_policy_bindings[{index}]"
        if not isinstance(binding, Mapping):
            issues.append(
                _issue(
                    "invalid_type",
                    "analysis",
                    prefix,
                    f"Field 'analysis.{prefix}' must be a mapping.",
                )
            )
            continue
        if any(not isinstance(key, str) for key in binding):
            issues.append(
                _issue(
                    "invalid_json_key",
                    "analysis",
                    prefix,
                    f"Field names in 'analysis.{prefix}' must be strings.",
                )
            )
        string_keys = {key for key in binding if isinstance(key, str)}
        for required_field in _REQUIRED_POLICY_BINDING_FIELDS:
            if required_field not in string_keys:
                located = f"{prefix}.{required_field}"
                issues.append(
                    _issue(
                        "missing_field",
                        "analysis",
                        located,
                        f"Missing required field 'analysis.{located}'.",
                    )
                )
        for unknown_field in sorted(string_keys - _POLICY_BINDING_FIELDS):
            located = f"{prefix}.{unknown_field}"
            issues.append(
                _issue(
                    "unknown_field",
                    "analysis",
                    located,
                    f"Unknown field 'analysis.{located}'.",
                )
            )

        policy_id = binding.get("finding_policy_id")
        policy_version = binding.get("finding_policy_version")
        id_valid = False
        version_valid = False
        if "finding_policy_id" in binding:
            id_valid = _validate_nonempty_string(
                policy_id,
                section="analysis",
                field=f"{prefix}.finding_policy_id",
                issues=issues,
            )
        if "finding_policy_version" in binding:
            version_valid = _validate_nonempty_string(
                policy_version,
                section="analysis",
                field=f"{prefix}.finding_policy_version",
                issues=issues,
            )
        if id_valid and version_valid:
            identity = (as_string(policy_id), as_string(policy_version))
            if identity in seen:
                issues.append(
                    _issue(
                        "duplicate_value",
                        "analysis",
                        prefix,
                        "Finding-policy identity/version pairs must be unique.",
                    )
                )
            seen.add(identity)

        if "parameters" in binding:
            parameters = binding["parameters"]
            parameter_field = f"{prefix}.parameters"
            if not isinstance(parameters, Mapping):
                issues.append(
                    _issue(
                        "invalid_type",
                        "analysis",
                        parameter_field,
                        f"Field 'analysis.{parameter_field}' must be a mapping.",
                    )
                )
            else:
                _validate_json_mapping(parameters, "analysis", parameter_field, issues)


def _validate_json_mapping(
    value: Mapping[object, object],
    section: str,
    field: str,
    issues: list[RecipeValidationIssue],
) -> None:
    """Validate one recursively JSON-compatible parameter mapping."""
    if any(not isinstance(key, str) for key in value):
        issues.append(
            _issue(
                "invalid_json_key",
                section,
                field,
                f"Field '{section}.{field}' must contain only string object keys.",
            )
        )
    for key in sorted(key for key in value if isinstance(key, str)):
        _validate_json_value(value[key], section, f"{field}.{key}", issues)


def _validate_json_value(
    value: object,
    section: str,
    field: str,
    issues: list[RecipeValidationIssue],
) -> None:
    """Validate a recursively JSON-compatible value with finite numbers."""
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            issues.append(
                _issue(
                    "non_finite_number",
                    section,
                    field,
                    f"Field '{section}.{field}' must contain only finite numbers.",
                )
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, section, f"{field}[{index}]", issues)
        return
    if isinstance(value, Mapping):
        _validate_json_mapping(value, section, field, issues)
        return
    issues.append(
        _issue(
            "invalid_json_value",
            section,
            field,
            f"Field '{section}.{field}' must contain a JSON-compatible value.",
        )
    )


def _issue(
    code: str,
    section: str,
    field: str | None,
    message: str,
) -> RecipeValidationIssue:
    """Build one deterministic located Recipe issue."""
    return RecipeValidationIssue(code=code, section=section, field=field, message=message)
