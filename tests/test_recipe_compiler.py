"""Unit tests for the v0-lite recipe compilation pipeline."""

from mlflow_monitor.recipe import (
    SYSTEM_DEFAULT_CONTRACT_ID,
    SYSTEM_DEFAULT_RECIPE_ID,
    SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN,
    RecipeReferenceCatalog,
    resolve_recipe_v0_lite,
)
from mlflow_monitor.recipe_compiler import (
    CompiledRunPlan,
    compile_recipe_v0_lite,
)


def make_valid_recipe() -> dict[str, object]:
    """Create a valid v0-lite recipe payload for compiler tests."""

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


def make_reference_catalog_with_system_default() -> RecipeReferenceCatalog:
    """Create references that also include built-in system-default IDs."""

    return RecipeReferenceCatalog(
        contract_ids=frozenset({"contract-default", SYSTEM_DEFAULT_CONTRACT_ID}),
        finding_policy_profiles=frozenset({"default_policy"}),
        summary_modes=frozenset({"standard"}),
    )


def test_compile_recipe_v0_lite_compiles_user_recipe_into_run_plan() -> None:
    recipe = resolve_recipe_v0_lite(make_valid_recipe(), references=make_reference_catalog())

    compiled = compile_recipe_v0_lite(recipe)

    assert isinstance(compiled, CompiledRunPlan)
    assert compiled.identity.recipe_id == "default"
    assert compiled.identity.recipe_version == "v0"
    assert compiled.input.run_selector == "train-run-123"
    assert compiled.input.source_experiment == "training/churn"
    assert compiled.input.required_metrics == ("f1", "auc")
    assert compiled.input.required_artifacts == ("metrics.json",)
    assert compiled.input.custom_reference_run_id == "run-custom-1"
    assert compiled.contract.contract_id == "contract-default"
    assert compiled.analysis.metrics == ("f1", "auc")
    assert compiled.analysis.slices == ("region", "segment")
    assert compiled.analysis.finding_policy_profile == "default_policy"
    assert compiled.analysis.summary_mode == "standard"


def test_compile_recipe_v0_lite_compiles_system_default_recipe() -> None:
    recipe = resolve_recipe_v0_lite(
        None,
        references=make_reference_catalog_with_system_default(),
    )
    compiled = compile_recipe_v0_lite(recipe)

    assert compiled.identity.recipe_id == SYSTEM_DEFAULT_RECIPE_ID
    assert compiled.identity.recipe_version == "v0"
    assert compiled.input.run_selector == SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN
    assert compiled.input.source_experiment is None
    assert compiled.input.required_metrics == ()
    assert compiled.input.required_artifacts == ()
    assert compiled.input.custom_reference_run_id is None
    assert compiled.contract.contract_id == SYSTEM_DEFAULT_CONTRACT_ID
    assert compiled.analysis.metrics == ()
    assert compiled.analysis.slices == ()
    assert compiled.analysis.finding_policy_profile is None
    assert compiled.analysis.summary_mode is None


def test_compile_recipe_v0_lite_is_deterministic_for_same_resolved_recipe() -> None:
    recipe = resolve_recipe_v0_lite(make_valid_recipe(), references=make_reference_catalog())

    compiled_once = compile_recipe_v0_lite(recipe)
    compiled_twice = compile_recipe_v0_lite(recipe)

    assert compiled_once == compiled_twice


def test_resolve_and_compile_recipe_v0_lite_uses_user_recipe_when_provided() -> None:
    recipe = resolve_recipe_v0_lite(make_valid_recipe(), references=make_reference_catalog())

    compiled = compile_recipe_v0_lite(recipe)

    assert compiled.identity.recipe_id == "default"
    assert compiled.contract.contract_id == "contract-default"
    assert compiled.input.run_selector == "train-run-123"


def test_compile_recipe_v0_lite_preserves_omitted_optional_fields() -> None:
    raw = {
        "identity": {"recipe_id": "default", "version": "v0"},
        "input_binding": {"run_selector": "train-run-123"},
        "contract_binding": {"contract_id": "contract-default"},
        "metrics_slices": {},
        "finding_policy": {},
        "output_binding": {},
    }
    recipe = resolve_recipe_v0_lite(raw, references=make_reference_catalog())

    compiled = compile_recipe_v0_lite(recipe)

    assert compiled.input.source_experiment is None
    assert compiled.input.required_metrics == ()
    assert compiled.input.required_artifacts == ()
    assert compiled.input.custom_reference_run_id is None
    assert compiled.analysis.metrics == ()
    assert compiled.analysis.slices == ()
    assert compiled.analysis.finding_policy_profile is None
    assert compiled.analysis.summary_mode is None
