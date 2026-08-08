"""Strict JSON-compatible Recipe schema and parsing."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from types import MappingProxyType

from mlflow_monitor.builtins import (
    SYSTEM_DEFAULT_CONTRACT_ID,
    SYSTEM_DEFAULT_RECIPE_ID,
)
from mlflow_monitor.errors import RecipeValidationError, RecipeValidationIssue

RECIPE_SCHEMA_VERSION = "v0"
SYSTEM_DEFAULT_RECIPE_VERSION = "v0"
SYSTEM_DEFAULT_CONTRACT_VERSION = "v0"

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

type RecipeJSONScalar = str | int | float | bool | None
type FrozenRecipeJSONValue = (
    RecipeJSONScalar | tuple[FrozenRecipeJSONValue, ...] | Mapping[str, FrozenRecipeJSONValue]
)
type FrozenRecipeParameters = Mapping[str, FrozenRecipeJSONValue]


@dataclass(frozen=True, slots=True)
class RecipeIdentity:
    """Stable user-authored Recipe identity.

    Attributes:
        recipe_id: Stable Recipe identifier.
        recipe_version: Exact user-authored Recipe version.
    """

    recipe_id: str
    recipe_version: str


@dataclass(frozen=True, slots=True)
class RecipeSourceRequirements:
    """Source Training Run requirements authored by a Recipe.

    Attributes:
        source_experiment: Optional owning experiment constraint.
        required_metric_names: Metrics that Prepare requires on the source.
        required_artifact_paths: Artifact paths that Prepare requires on the source.
    """

    source_experiment: str | None = None
    required_metric_names: tuple[str, ...] = ()
    required_artifact_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RecipeContractBinding:
    """Exact system Contract binding authored by a Recipe.

    Attributes:
        contract_id: Registered system Contract identifier.
        contract_version: Exact registered Contract version.
    """

    contract_id: str
    contract_version: str


@dataclass(frozen=True, slots=True)
class RecipeFindingPolicyBinding:
    """Exact Finding-policy binding and opaque JSON parameters.

    Attributes:
        finding_policy_id: Registered Finding-policy identifier.
        finding_policy_version: Exact registered policy version.
        parameters: Immutable structurally validated JSON parameters.
    """

    finding_policy_id: str
    finding_policy_version: str
    parameters: FrozenRecipeParameters


@dataclass(frozen=True, slots=True)
class RecipeAnalysis:
    """Three-state metric and Finding-policy analysis authoring.

    Attributes:
        metric_names: ``None`` for omitted, an empty tuple for none, or exact names.
        finding_policy_bindings: ``None`` for defaults, an empty tuple for none,
            or exact authored bindings.
    """

    metric_names: tuple[str, ...] | None = None
    finding_policy_bindings: tuple[RecipeFindingPolicyBinding, ...] | None = None


@dataclass(frozen=True, slots=True)
class Recipe:
    """Typed immutable representation of one structurally valid v0 Recipe.

    Attributes:
        recipe_schema_version: Exact Recipe schema version.
        identity: Stable Recipe identity.
        source_requirements: Source Training Run preconditions.
        contract: Exact system Contract binding.
        analysis: Three-state analysis authoring.
    """

    recipe_schema_version: str
    identity: RecipeIdentity
    source_requirements: RecipeSourceRequirements
    contract: RecipeContractBinding
    analysis: RecipeAnalysis


def build_system_default_recipe() -> dict[str, object]:
    """Return a fresh authoring mapping for the zero-configuration Recipe.

    Returns:
        Canonical minimal system-default Recipe authoring data.
    """
    return {
        "recipe_schema_version": RECIPE_SCHEMA_VERSION,
        "identity": {
            "recipe_id": SYSTEM_DEFAULT_RECIPE_ID,
            "recipe_version": SYSTEM_DEFAULT_RECIPE_VERSION,
        },
        "contract": {
            "contract_id": SYSTEM_DEFAULT_CONTRACT_ID,
            "contract_version": SYSTEM_DEFAULT_CONTRACT_VERSION,
        },
    }


def parse_recipe(raw: Mapping[str, object]) -> Recipe:
    """Parse a strict JSON-compatible Mapping into an immutable Recipe.

    Args:
        raw: Recipe authoring data using the canonical v0 Mapping shape.

    Returns:
        An immutable typed Recipe that preserves omitted, empty, and nonempty
        analysis selections.

    Raises:
        RecipeValidationError: If the Mapping is structurally invalid.
    """
    issues = _collect_recipe_issues(raw)
    if issues:
        raise RecipeValidationError(issues=tuple(issues))

    identity = _as_mapping(raw["identity"])
    contract = _as_mapping(raw["contract"])
    source_requirements = _optional_mapping(raw, "source_requirements")
    analysis = _optional_mapping(raw, "analysis")

    return Recipe(
        recipe_schema_version=_as_string(raw["recipe_schema_version"]),
        identity=RecipeIdentity(
            recipe_id=_as_string(identity["recipe_id"]),
            recipe_version=_as_string(identity["recipe_version"]),
        ),
        source_requirements=RecipeSourceRequirements(
            source_experiment=_optional_parsed_string(
                source_requirements,
                "source_experiment",
            ),
            required_metric_names=_parsed_string_tuple(
                source_requirements,
                "required_metric_names",
            ),
            required_artifact_paths=_parsed_string_tuple(
                source_requirements,
                "required_artifact_paths",
            ),
        ),
        contract=RecipeContractBinding(
            contract_id=_as_string(contract["contract_id"]),
            contract_version=_as_string(contract["contract_version"]),
        ),
        analysis=RecipeAnalysis(
            metric_names=_parsed_optional_string_tuple(analysis, "metric_names"),
            finding_policy_bindings=_parsed_policy_bindings(analysis),
        ),
    )


def load_recipe_json(path: str | PathLike[str]) -> Recipe:
    """Decode one JSON file and parse it through :func:`parse_recipe`.

    Args:
        path: Filesystem path to a UTF-8 JSON Recipe.

    Returns:
        The parsed immutable Recipe.

    Raises:
        OSError: If the file cannot be read.
        RecipeValidationError: If JSON decoding or Recipe validation fails.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RecipeValidationError(
            issues=(
                RecipeValidationIssue(
                    code="invalid_encoding",
                    section="recipe",
                    field=None,
                    message="Recipe file must be UTF-8 encoded.",
                ),
            )
        ) from exc
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecipeValidationError(
            issues=(
                RecipeValidationIssue(
                    code="invalid_json",
                    section="recipe",
                    field=None,
                    message="Recipe file must contain valid JSON.",
                ),
            )
        ) from exc
    return parse_recipe(decoded)


def _collect_recipe_issues(raw: object) -> list[RecipeValidationIssue]:
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
            identity = (_as_string(policy_id), _as_string(policy_version))
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


def _parsed_policy_bindings(
    analysis: Mapping[str, object],
) -> tuple[RecipeFindingPolicyBinding, ...] | None:
    """Build parsed policy bindings after validation succeeds."""
    if "finding_policy_bindings" not in analysis:
        return None
    raw_bindings = analysis["finding_policy_bindings"]
    assert isinstance(raw_bindings, list)
    bindings: list[RecipeFindingPolicyBinding] = []
    for raw_binding in raw_bindings:
        binding = _as_mapping(raw_binding)
        parameters = binding.get("parameters", {})
        assert isinstance(parameters, Mapping)
        bindings.append(
            RecipeFindingPolicyBinding(
                finding_policy_id=_as_string(binding["finding_policy_id"]),
                finding_policy_version=_as_string(binding["finding_policy_version"]),
                parameters=_freeze_json_mapping(parameters),
            )
        )
    return tuple(bindings)


def _freeze_json_mapping(value: Mapping[object, object]) -> FrozenRecipeParameters:
    """Defensively copy a valid JSON mapping into immutable values."""
    return MappingProxyType(
        {
            key: _freeze_json_value(value[key])
            for key in sorted(key for key in value if isinstance(key, str))
        }
    )


def _freeze_json_value(value: object) -> FrozenRecipeJSONValue:
    """Defensively copy one valid JSON value into immutable containers."""
    if isinstance(value, Mapping):
        return _freeze_json_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    assert value is None or isinstance(value, str | int | float | bool)
    return value


def _parsed_optional_string_tuple(
    mapping: Mapping[str, object],
    field: str,
) -> tuple[str, ...] | None:
    """Return None for omission and a tuple for an authored list."""
    if field not in mapping:
        return None
    return _parsed_string_tuple(mapping, field)


def _parsed_string_tuple(mapping: Mapping[str, object], field: str) -> tuple[str, ...]:
    """Build a tuple from a validated optional string list."""
    if field not in mapping:
        return ()
    value = mapping[field]
    assert isinstance(value, list)
    return tuple(_as_string(item) for item in value)


def _optional_parsed_string(mapping: Mapping[str, object], field: str) -> str | None:
    """Build an optional string after validation succeeds."""
    if field not in mapping:
        return None
    return _as_string(mapping[field])


def _optional_mapping(raw: Mapping[str, object], field: str) -> Mapping[str, object]:
    """Return a validated optional mapping or an empty mapping."""
    if field not in raw:
        return MappingProxyType({})
    return _as_mapping(raw[field])


def _as_mapping(value: object) -> Mapping[str, object]:
    """Narrow a mapping value after validation succeeds."""
    assert isinstance(value, Mapping)
    return value


def _as_string(value: object) -> str:
    """Narrow a string value after validation succeeds."""
    assert isinstance(value, str)
    return value


def _issue(
    code: str,
    section: str,
    field: str | None,
    message: str,
) -> RecipeValidationIssue:
    """Build one deterministic located Recipe issue."""
    return RecipeValidationIssue(code=code, section=section, field=field, message=message)
