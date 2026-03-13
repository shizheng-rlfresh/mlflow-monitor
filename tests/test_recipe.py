"""Unit tests for the v0-lite recipe schema parser."""

from mlflow_monitor.recipe import RecipeV0Lite, parse_recipe_v0_lite


def test_parse_recipe_v0_lite_parses_valid_recipe() -> None:
    raw = {
        "identity": {"recipe_id": "default", "version": "v0"},
        "input_binding": {"run_selector": "latest"},
        "contract_binding": {"contract_id": "contract-default"},
        "metrics_slices": {},
        "finding_policy": {},
        "output_binding": {},
    }

    parsed = parse_recipe_v0_lite(raw)

    assert isinstance(parsed, RecipeV0Lite)
    assert parsed.identity.recipe_id == "default"
    assert parsed.identity.version == "v0"
    assert parsed.input_binding.run_selector == "latest"
