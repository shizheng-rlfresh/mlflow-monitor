"""Recipe parser module for mlflow-monitor v0.

Strict JSON-compatible Recipe schema and parsing.

"""

from __future__ import annotations

import json
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType

from mlflow_monitor.builtins import SYSTEM_DEFAULT_CONTRACT_ID, SYSTEM_DEFAULT_RECIPE_ID
from mlflow_monitor.errors import RecipeValidationError, RecipeValidationIssue

from .models import (
    RECIPE_SCHEMA_VERSION,
    SYSTEM_DEFAULT_CONTRACT_VERSION,
    SYSTEM_DEFAULT_RECIPE_VERSION,
    FrozenRecipeJSONValue,
    FrozenRecipeParameters,
    Recipe,
    RecipeAnalysis,
    RecipeContractBinding,
    RecipeFindingPolicyBinding,
    RecipeIdentity,
    RecipeSourceRequirements,
)
from .validation import as_string, collect_recipe_issues


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
    issues = collect_recipe_issues(raw)
    if issues:
        raise RecipeValidationError(issues=tuple(issues))

    identity = _as_mapping(raw["identity"])
    contract = _as_mapping(raw["contract"])
    source_requirements = _optional_mapping(raw, "source_requirements")
    analysis = _optional_mapping(raw, "analysis")

    return Recipe(
        recipe_schema_version=as_string(raw["recipe_schema_version"]),
        identity=RecipeIdentity(
            recipe_id=as_string(identity["recipe_id"]),
            recipe_version=as_string(identity["recipe_version"]),
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
            contract_id=as_string(contract["contract_id"]),
            contract_version=as_string(contract["contract_version"]),
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
                finding_policy_id=as_string(binding["finding_policy_id"]),
                finding_policy_version=as_string(binding["finding_policy_version"]),
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
    return tuple(as_string(item) for item in value)


def _optional_parsed_string(mapping: Mapping[str, object], field: str) -> str | None:
    """Build an optional string after validation succeeds."""
    if field not in mapping:
        return None
    return as_string(mapping[field])


def _optional_mapping(raw: Mapping[str, object], field: str) -> Mapping[str, object]:
    """Return a validated optional mapping or an empty mapping."""
    if field not in raw:
        return MappingProxyType({})
    return _as_mapping(raw[field])


def _as_mapping(value: object) -> Mapping[str, object]:
    """Narrow a mapping value after validation succeeds."""
    assert isinstance(value, Mapping)
    return value
