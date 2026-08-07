"""Specifications for strict v0 Recipe compilation."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from mlflow_monitor.builtins import SYSTEM_DEFAULT_CONTRACT_ID
from mlflow_monitor.errors import RecipeValidationError
from mlflow_monitor.recipe import build_system_default_recipe, parse_recipe
from mlflow_monitor.recipe_compiler import (
    SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
    SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
    SYSTEM_DEFAULT_COMPILED_RECIPE,
    CompiledRecipe,
    ComponentRegistry,
    compile_recipe,
)


def test_component_registry_contains_only_immutable_system_components() -> None:
    registry = ComponentRegistry()

    assert tuple(registry.contracts) == ((SYSTEM_DEFAULT_CONTRACT_ID, "v0"),)
    assert tuple(registry.finding_policies) == (
        (
            SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
            SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
        ),
    )
    with pytest.raises(TypeError):
        registry.contracts[("custom", "1")] = registry.contracts[  # type: ignore[index]
            (SYSTEM_DEFAULT_CONTRACT_ID, "v0")
        ]
    with pytest.raises(TypeError):
        registry.finding_policies[("custom", "1")] = registry.finding_policies[  # type: ignore[index]
            (
                SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
                SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
            )
        ]


def test_compile_recipe_zero_configuration_expands_system_defaults() -> None:
    compiled = compile_recipe()

    assert isinstance(compiled, CompiledRecipe)
    assert compiled.effective_plan.to_dict() == {
        "recipe_schema_version": "v0",
        "identity": {
            "recipe_id": "system_default",
            "recipe_version": "v0",
        },
        "source_requirements": {
            "source_experiment": None,
            "required_metric_names": [],
            "required_artifact_paths": [],
        },
        "contract": {
            "contract_id": "default_permissive",
            "contract_version": "v0",
        },
        "analysis": {
            "metric_names": None,
            "finding_policy_bindings": [
                {
                    "finding_policy_id": "system-compatibility-findings",
                    "finding_policy_version": "v0",
                    "parameters": {},
                }
            ],
        },
    }
    assert compiled.contract.contract_id == SYSTEM_DEFAULT_CONTRACT_ID
    assert compiled.contract.version == "v0"
    assert len(compiled.finding_policy_bindings) == 1
    binding = compiled.finding_policy_bindings[0]
    assert binding.finding_policy_id == SYSTEM_COMPATIBILITY_FINDING_POLICY_ID
    assert binding.finding_policy_version == SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION
    assert binding.parameters == {}


def test_system_default_compiled_recipe_is_precomputed_and_reusable() -> None:
    assert SYSTEM_DEFAULT_COMPILED_RECIPE == compile_recipe()


def test_system_policy_evaluation_fails_closed_before_analyze_integration() -> None:
    binding = SYSTEM_DEFAULT_COMPILED_RECIPE.finding_policy_bindings[0]

    with pytest.raises(RuntimeError, match="evaluation is unavailable"):
        binding.policy.evaluate(
            parameters=binding.parameters,
            diffs=(),
            compatibility_evidence=(),
            reference_comparison_coverage=(),
        )


def test_compile_recipe_normalizes_system_only_recipe_canonically() -> None:
    raw = build_system_default_recipe()
    raw["source_requirements"] = {
        "required_metric_names": ["zeta", "alpha"],
        "required_artifact_paths": ["z/path", "a/path"],
    }
    raw["analysis"] = {
        "metric_names": ["zeta", "alpha"],
        "finding_policy_bindings": [
            {
                "finding_policy_id": SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
                "finding_policy_version": SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
            }
        ],
    }

    compiled = compile_recipe(parse_recipe(raw), component_registry=ComponentRegistry())
    serialized = compiled.effective_plan.to_dict()

    assert serialized["source_requirements"] == {
        "source_experiment": None,
        "required_metric_names": ["alpha", "zeta"],
        "required_artifact_paths": ["a/path", "z/path"],
    }
    assert serialized["analysis"] == {
        "metric_names": ["alpha", "zeta"],
        "finding_policy_bindings": [
            {
                "finding_policy_id": SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
                "finding_policy_version": SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
                "parameters": {},
            }
        ],
    }
    assert json.loads(json.dumps(serialized, sort_keys=True)) == serialized


def test_compile_recipe_preserves_explicit_empty_analysis_selections() -> None:
    raw = build_system_default_recipe()
    raw["analysis"] = {"metric_names": [], "finding_policy_bindings": []}

    compiled = compile_recipe(raw)

    assert compiled.effective_plan.analysis.metric_names == ()
    assert compiled.effective_plan.analysis.finding_policy_bindings == ()
    assert compiled.finding_policy_bindings == ()


def test_compile_recipe_accepts_mapping_through_the_strict_parser() -> None:
    assert compile_recipe(build_system_default_recipe()) == compile_recipe(
        parse_recipe(build_system_default_recipe())
    )


def test_compile_recipe_is_immutable() -> None:
    compiled = compile_recipe()

    with pytest.raises(FrozenInstanceError):
        compiled.contract = compiled.contract  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        compiled.effective_plan.identity.recipe_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        compiled.finding_policy_bindings[0].parameters["changed"] = True  # type: ignore[index]


def test_compile_recipe_rejects_unknown_policy_before_allocation() -> None:
    raw = build_system_default_recipe()
    raw["analysis"] = {
        "finding_policy_bindings": [
            {
                "finding_policy_id": "custom-policy",
                "finding_policy_version": "1",
            }
        ]
    }

    with pytest.raises(RecipeValidationError) as exc_info:
        compile_recipe(raw)

    issue = exc_info.value.issues[0]
    assert issue.code == "unknown_component"
    assert issue.section == "analysis"
    assert issue.field == "finding_policy_bindings[0]"


def test_compile_recipe_rejects_parameters_invalid_for_system_policy() -> None:
    raw = build_system_default_recipe()
    raw["analysis"] = {
        "finding_policy_bindings": [
            {
                "finding_policy_id": SYSTEM_COMPATIBILITY_FINDING_POLICY_ID,
                "finding_policy_version": SYSTEM_COMPATIBILITY_FINDING_POLICY_VERSION,
                "parameters": {"threshold": 0.5},
            }
        ]
    }

    with pytest.raises(RecipeValidationError) as exc_info:
        compile_recipe(raw)

    issue = exc_info.value.issues[0]
    assert issue.code == "invalid_policy_parameters"
    assert issue.section == "analysis"
    assert issue.field == "finding_policy_bindings[0].parameters"
