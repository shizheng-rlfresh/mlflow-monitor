"""Unit tests for the v0-lite recipe parser and validator."""

import pytest

from mlflow_monitor.errors import RecipeValidationError
from mlflow_monitor.recipe import (
    RecipeReferenceCatalog,
    RecipeV0Lite,
    parse_recipe_v0_lite,
    validate_recipe_v0_lite,
)


def make_valid_recipe() -> dict[str, object]:
    """Create a valid v0-lite recipe payload for parser tests."""

    return {
        "identity": {"recipe_id": "default", "version": "v0"},
        "input_binding": {
            "run_selector": "train-run-123",
            "source_experiment": "training/churn",
            "required_metrics": ["f1", "auc"],
            "required_artifacts": ["metrics.json"],
            "custom_reference_run_id": "run-custom-1",
        },
        "contract_binding": {"contract_id": "contract-default"},
        "metrics_slices": {"metrics": ["f1", "auc"], "slices": ["region", "segment"]},
        "finding_policy": {"profile": "default_policy"},
        "output_binding": {"summary_mode": "standard"},
    }


def make_reference_catalog() -> RecipeReferenceCatalog:
    """Create a reference catalog with valid v0-lite IDs."""

    return RecipeReferenceCatalog(
        contract_ids=frozenset({"contract-default"}),
        finding_policy_profiles=frozenset({"default_policy"}),
        summary_modes=frozenset({"standard"}),
    )


def test_parse_recipe_v0_lite_parses_valid_recipe() -> None:
    raw = make_valid_recipe()

    parsed = parse_recipe_v0_lite(raw)

    assert isinstance(parsed, RecipeV0Lite)
    assert parsed.identity.recipe_id == "default"
    assert parsed.identity.version == "v0"
    assert parsed.input_binding.run_selector == "train-run-123"
    assert parsed.input_binding.source_experiment == "training/churn"
    assert parsed.input_binding.required_metrics == ("f1", "auc")
    assert parsed.input_binding.required_artifacts == ("metrics.json",)
    assert parsed.input_binding.custom_reference_run_id == "run-custom-1"
    assert parsed.contract_binding.contract_id == "contract-default"
    assert parsed.metrics_slices.metrics == ("f1", "auc")
    assert parsed.metrics_slices.slices == ("region", "segment")
    assert parsed.finding_policy.profile == "default_policy"
    assert parsed.output_binding.summary_mode == "standard"


def test_parse_recipe_v0_lite_rejects_unknown_top_level_section() -> None:
    raw = make_valid_recipe()
    raw["subject"] = {"subject_id": "churn_model"}

    with pytest.raises(ValueError, match="Unknown/disallowed recipe section\\(s\\): subject"):
        parse_recipe_v0_lite(raw)


def test_parse_recipe_v0_lite_rejects_missing_required_top_level_section() -> None:
    raw = make_valid_recipe()
    del raw["output_binding"]

    with pytest.raises(
        ValueError,
        match="Missing required recipe section\\(s\\): output_binding",
    ):
        parse_recipe_v0_lite(raw)


def test_parse_recipe_v0_lite_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="Recipe payload must be a mapping."):
        parse_recipe_v0_lite([])  # type: ignore[arg-type]


def test_parse_recipe_v0_lite_rejects_non_string_top_level_section_name() -> None:
    raw = make_valid_recipe()
    raw[1] = {"unexpected": "section"}  # type: ignore[index]

    with pytest.raises(
        ValueError,
        match="Top-level recipe section names must be strings.",
    ):
        parse_recipe_v0_lite(raw)


def test_parse_recipe_v0_lite_rejects_non_string_identity_field() -> None:
    raw = make_valid_recipe()
    raw["identity"] = {"recipe_id": 123, "version": "v0"}

    with pytest.raises(ValueError, match="Field 'identity.recipe_id' must be a string."):
        parse_recipe_v0_lite(raw)


def test_parse_recipe_v0_lite_rejects_non_sequence_required_metrics() -> None:
    raw = make_valid_recipe()
    raw["input_binding"] = {
        "run_selector": "latest",
        "required_metrics": "f1",
    }

    with pytest.raises(
        ValueError,
        match="Field 'input_binding.required_metrics' must be a sequence of strings.",
    ):
        parse_recipe_v0_lite(raw)


def test_parse_recipe_v0_lite_applies_defaults_for_omitted_optional_fields() -> None:
    raw = {
        "identity": {"recipe_id": "default", "version": "v0"},
        "input_binding": {"run_selector": "latest"},
        "contract_binding": {"contract_id": "contract-default"},
        "metrics_slices": {},
        "finding_policy": {},
        "output_binding": {},
    }

    parsed = parse_recipe_v0_lite(raw)

    assert parsed.input_binding.source_experiment is None
    assert parsed.input_binding.required_metrics == ()
    assert parsed.input_binding.required_artifacts == ()
    assert parsed.input_binding.custom_reference_run_id is None
    assert parsed.metrics_slices.metrics == ()
    assert parsed.metrics_slices.slices == ()
    assert parsed.finding_policy.profile is None
    assert parsed.output_binding.summary_mode is None


def test_parse_recipe_v0_lite_is_deterministic_for_same_input() -> None:
    raw = make_valid_recipe()

    parsed_once = parse_recipe_v0_lite(raw)
    parsed_twice = parse_recipe_v0_lite(raw)

    assert parsed_once == parsed_twice


@pytest.mark.parametrize(
    ("section_name", "bad_key"),
    [
        ("identity", "recipe_name"),
        ("input_binding", "required_metric"),
        ("contract_binding", "contract_name"),
        ("metrics_slices", "slice"),
        ("finding_policy", "severity_policy"),
        ("output_binding", "summary"),
    ],
)
def test_validate_recipe_v0_lite_rejects_unknown_nested_section_keys(
    section_name: str,
    bad_key: str,
) -> None:
    raw = make_valid_recipe()
    section = dict(raw[section_name])  # type: ignore[index]
    section[bad_key] = "unexpected"
    raw[section_name] = section

    with pytest.raises(RecipeValidationError, match=f"{section_name}\\.{bad_key}"):
        validate_recipe_v0_lite(raw, references=make_reference_catalog())


def test_validate_recipe_v0_lite_rejects_unknown_contract_reference() -> None:
    raw = make_valid_recipe()
    raw["contract_binding"] = {"contract_id": "contract-missing"}

    with pytest.raises(RecipeValidationError, match="contract_binding\\.contract_id"):
        validate_recipe_v0_lite(raw, references=make_reference_catalog())


def test_validate_recipe_v0_lite_rejects_unknown_finding_policy_profile() -> None:
    raw = make_valid_recipe()
    raw["finding_policy"] = {"profile": "profile-missing"}

    with pytest.raises(RecipeValidationError, match="finding_policy\\.profile"):
        validate_recipe_v0_lite(raw, references=make_reference_catalog())


def test_validate_recipe_v0_lite_rejects_unknown_output_summary_mode() -> None:
    raw = make_valid_recipe()
    raw["output_binding"] = {"summary_mode": "mode-missing"}

    with pytest.raises(RecipeValidationError, match="output_binding\\.summary_mode"):
        validate_recipe_v0_lite(raw, references=make_reference_catalog())


@pytest.mark.parametrize(
    "selector",
    [
        "",
        "   ",
        "latest",
        "run_id:abc",
    ],
)
def test_validate_recipe_v0_lite_rejects_invalid_run_selector(selector: str) -> None:
    raw = make_valid_recipe()
    raw["input_binding"] = {
        **raw["input_binding"],  # type: ignore[arg-type]
        "run_selector": selector,
    }

    with pytest.raises(RecipeValidationError, match="input_binding\\.run_selector"):
        validate_recipe_v0_lite(raw, references=make_reference_catalog())


@pytest.mark.parametrize(
    ("field_name", "values"),
    [
        ("required_metrics", ["f1", "f1"]),
        ("required_artifacts", ["metrics.json", "metrics.json"]),
        ("metrics", ["f1", "f1"]),
        ("slices", ["region", "region"]),
    ],
)
def test_validate_recipe_v0_lite_rejects_duplicate_list_entries(
    field_name: str,
    values: list[str],
) -> None:
    raw = make_valid_recipe()
    if field_name in {"required_metrics", "required_artifacts"}:
        raw["input_binding"] = {
            **raw["input_binding"],  # type: ignore[arg-type]
            field_name: values,
        }
    else:
        raw["metrics_slices"] = {
            **raw["metrics_slices"],  # type: ignore[arg-type]
            field_name: values,
        }

    with pytest.raises(RecipeValidationError, match=field_name):
        validate_recipe_v0_lite(raw, references=make_reference_catalog())


def test_validate_recipe_v0_lite_reports_structured_issue_for_unknown_field() -> None:
    raw = make_valid_recipe()
    raw["input_binding"] = {
        **raw["input_binding"],  # type: ignore[arg-type]
        "required_metric": ["f1"],
    }

    with pytest.raises(RecipeValidationError) as exc_info:
        validate_recipe_v0_lite(raw, references=make_reference_catalog())

    issue = exc_info.value.issues[0]
    assert issue.code == "unknown_field"
    assert issue.section == "input_binding"
    assert issue.field == "required_metric"


def test_validate_recipe_v0_lite_normalizes_parser_error_to_section_field() -> None:
    raw = make_valid_recipe()
    raw["input_binding"] = {
        **raw["input_binding"],  # type: ignore[arg-type]
        "run_selector": "",
    }

    with pytest.raises(RecipeValidationError) as exc_info:
        validate_recipe_v0_lite(raw, references=make_reference_catalog())

    issue = exc_info.value.issues[0]
    assert issue.code == "structural_error"
    assert issue.section == "input_binding"
    assert issue.field == "run_selector"
