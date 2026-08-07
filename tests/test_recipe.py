"""Specifications for the strict v0 Recipe parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlflow_monitor.errors import RecipeValidationError, RecipeValidationIssue
from mlflow_monitor.recipe import (
    Recipe,
    build_system_default_recipe,
    load_recipe_json,
    parse_recipe,
)


def make_recipe() -> dict[str, object]:
    """Return a complete valid Recipe authoring mapping."""
    return {
        "recipe_schema_version": "v0",
        "identity": {
            "recipe_id": "churn-monitoring",
            "recipe_version": "3",
        },
        "source_requirements": {
            "source_experiment": "training/churn",
            "required_metric_names": ["accuracy"],
            "required_artifact_paths": ["model/MLmodel"],
        },
        "contract": {
            "contract_id": "default_permissive",
            "contract_version": "v0",
        },
        "analysis": {
            "metric_names": ["accuracy", "latency_p95"],
            "finding_policy_bindings": [
                {
                    "finding_policy_id": "relative-regression",
                    "finding_policy_version": "1",
                    "parameters": {
                        "maximum_decrease": 0.05,
                        "labels": ["quality", "regression"],
                    },
                }
            ],
        },
    }


def test_parse_recipe_builds_immutable_typed_recipe() -> None:
    raw = make_recipe()

    recipe = parse_recipe(raw)

    assert isinstance(recipe, Recipe)
    assert recipe.recipe_schema_version == "v0"
    assert recipe.identity.recipe_id == "churn-monitoring"
    assert recipe.identity.recipe_version == "3"
    assert recipe.source_requirements.source_experiment == "training/churn"
    assert recipe.source_requirements.required_metric_names == ("accuracy",)
    assert recipe.source_requirements.required_artifact_paths == ("model/MLmodel",)
    assert recipe.contract.contract_id == "default_permissive"
    assert recipe.contract.contract_version == "v0"
    assert recipe.analysis.metric_names == ("accuracy", "latency_p95")
    assert len(recipe.analysis.finding_policy_bindings or ()) == 1
    binding = (recipe.analysis.finding_policy_bindings or ())[0]
    assert binding.finding_policy_id == "relative-regression"
    assert binding.parameters == {
        "labels": ("quality", "regression"),
        "maximum_decrease": 0.05,
    }
    with pytest.raises(TypeError):
        binding.parameters["maximum_decrease"] = 0.1  # type: ignore[index]
    with pytest.raises(TypeError):
        binding.parameters["labels"][0] = "changed"  # type: ignore[index]

    raw_binding = raw["analysis"]
    assert isinstance(raw_binding, dict)
    raw_binding["metric_names"] = ["changed"]
    assert recipe.analysis.metric_names == ("accuracy", "latency_p95")


@pytest.mark.parametrize(
    ("analysis", "expected_metrics", "expected_policies"),
    [
        ({}, None, None),
        ({"metric_names": [], "finding_policy_bindings": []}, (), ()),
        (
            {
                "metric_names": ["accuracy"],
                "finding_policy_bindings": [
                    {
                        "finding_policy_id": "system-compatibility-findings",
                        "finding_policy_version": "v0",
                    }
                ],
            },
            ("accuracy",),
            ("system-compatibility-findings",),
        ),
    ],
)
def test_parse_recipe_preserves_three_state_analysis_authoring(
    analysis: dict[str, object],
    expected_metrics: tuple[str, ...] | None,
    expected_policies: tuple[str, ...] | None,
) -> None:
    raw = build_system_default_recipe()
    raw["analysis"] = analysis

    recipe = parse_recipe(raw)

    assert recipe.analysis.metric_names == expected_metrics
    bindings = recipe.analysis.finding_policy_bindings
    assert (
        None if bindings is None else tuple(binding.finding_policy_id for binding in bindings)
    ) == expected_policies


def test_parse_recipe_applies_optional_section_defaults() -> None:
    recipe = parse_recipe(build_system_default_recipe())

    assert recipe.source_requirements.source_experiment is None
    assert recipe.source_requirements.required_metric_names == ()
    assert recipe.source_requirements.required_artifact_paths == ()
    assert recipe.analysis.metric_names is None
    assert recipe.analysis.finding_policy_bindings is None


def test_load_recipe_json_decodes_through_the_mapping_parser(tmp_path: Path) -> None:
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(make_recipe()), encoding="utf-8")

    assert load_recipe_json(path) == parse_recipe(make_recipe())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_selector", "latest"),
        ("source_run_id", "source-123"),
        ("custom_reference_monitoring_run_id", "monitoring-123"),
        ("slices", ["country=us"]),
        ("output_binding", {"summary_mode": "compact"}),
        ("promotion", {"mode": "automatic"}),
    ],
)
def test_parse_recipe_rejects_out_of_scope_top_level_fields(
    field: str,
    value: object,
) -> None:
    raw = build_system_default_recipe()
    raw[field] = value

    with pytest.raises(RecipeValidationError) as exc_info:
        parse_recipe(raw)

    assert exc_info.value.issues == (
        RecipeValidationIssue(
            code="unknown_field",
            section="recipe",
            field=field,
            message=f"Unknown field 'recipe.{field}'.",
        ),
    )


@pytest.mark.parametrize(
    ("mutate", "section", "field", "code"),
    [
        (
            lambda raw: raw.__setitem__("analysis", {"metric_names": None}),
            "analysis",
            "metric_names",
            "invalid_type",
        ),
        (
            lambda raw: raw.__setitem__(
                "analysis",
                {
                    "finding_policy_bindings": [
                        {
                            "finding_policy_id": "policy",
                            "finding_policy_version": "1",
                            "parameters": None,
                        }
                    ]
                },
            ),
            "analysis",
            "finding_policy_bindings[0].parameters",
            "invalid_type",
        ),
        (
            lambda raw: raw.__setitem__(
                "source_requirements", {"required_metric_names": ["accuracy", "accuracy"]}
            ),
            "source_requirements",
            "required_metric_names",
            "duplicate_value",
        ),
        (
            lambda raw: raw.__setitem__("analysis", {"metric_names": ["accuracy", "accuracy"]}),
            "analysis",
            "metric_names",
            "duplicate_value",
        ),
        (
            lambda raw: raw.__setitem__("analysis", {"metric_names": [""]}),
            "analysis",
            "metric_names[0]",
            "empty_value",
        ),
        (
            lambda raw: raw.__setitem__(
                "analysis",
                {
                    "finding_policy_bindings": [
                        {
                            "finding_policy_id": "policy",
                            "finding_policy_version": "1",
                        },
                        {
                            "finding_policy_id": "policy",
                            "finding_policy_version": "1",
                        },
                    ]
                },
            ),
            "analysis",
            "finding_policy_bindings[1]",
            "duplicate_value",
        ),
        (
            lambda raw: raw.__setitem__(
                "analysis",
                {
                    "finding_policy_bindings": [
                        {
                            "finding_policy_id": "policy",
                            "finding_policy_version": "1",
                            "parameters": {"threshold": float("inf")},
                        }
                    ]
                },
            ),
            "analysis",
            "finding_policy_bindings[0].parameters.threshold",
            "non_finite_number",
        ),
        (
            lambda raw: raw.__setitem__(
                "analysis",
                {
                    "finding_policy_bindings": [
                        {
                            "finding_policy_id": "policy",
                            "finding_policy_version": "1",
                            "parameters": {"unsupported": ("tuple",)},
                        }
                    ]
                },
            ),
            "analysis",
            "finding_policy_bindings[0].parameters.unsupported",
            "invalid_json_value",
        ),
    ],
)
def test_parse_recipe_reports_deterministic_located_issues(
    mutate,
    section: str,
    field: str,
    code: str,
) -> None:
    raw = build_system_default_recipe()
    mutate(raw)

    with pytest.raises(RecipeValidationError) as first:
        parse_recipe(raw)
    with pytest.raises(RecipeValidationError) as second:
        parse_recipe(raw)

    assert first.value.issues == second.value.issues
    assert len(first.value.issues) == 1
    issue = first.value.issues[0]
    assert issue.code == code
    assert issue.section == section
    assert issue.field == field


def test_load_recipe_json_rejects_non_finite_numbers_with_parser_location(
    tmp_path: Path,
) -> None:
    raw = make_recipe()
    analysis = raw["analysis"]
    assert isinstance(analysis, dict)
    bindings = analysis["finding_policy_bindings"]
    assert isinstance(bindings, list)
    bindings[0]["parameters"]["maximum_decrease"] = float("nan")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(RecipeValidationError) as exc_info:
        load_recipe_json(path)

    issue = exc_info.value.issues[0]
    assert issue.code == "non_finite_number"
    assert issue.section == "analysis"
    assert issue.field == "finding_policy_bindings[0].parameters.maximum_decrease"
