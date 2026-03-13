"""Unit tests for the v0-lite recipe schema parser."""

import pytest

from mlflow_monitor.recipe import RecipeV0Lite, parse_recipe_v0_lite


def make_valid_recipe() -> dict[str, object]:
    """Create a valid v0-lite recipe payload for parser tests."""

    return {
        "identity": {"recipe_id": "default", "version": "v0"},
        "input_binding": {
            "run_selector": "latest",
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


def test_parse_recipe_v0_lite_parses_valid_recipe() -> None:
    raw = make_valid_recipe()

    parsed = parse_recipe_v0_lite(raw)

    assert isinstance(parsed, RecipeV0Lite)
    assert parsed.identity.recipe_id == "default"
    assert parsed.identity.version == "v0"
    assert parsed.input_binding.run_selector == "latest"
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
