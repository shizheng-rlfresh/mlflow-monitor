"""Recipe compilation pipeline for MLflow-Monitor v0-lite."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow_monitor.recipe import (
    RecipeV0Lite,
)


@dataclass(frozen=True, slots=True)
class CompiledRunPlanIdentity:
    """Compiled recipe identity fields required by workflow."""

    recipe_id: str
    recipe_version: str


@dataclass(frozen=True, slots=True)
class CompiledRunPlanInput:
    """Compiled input-selection fields required by workflow."""

    run_selector: str
    source_experiment: str | None
    required_metrics: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    custom_reference_run_id: str | None


@dataclass(frozen=True, slots=True)
class CompiledRunPlanContract:
    """Compiled contract-selection fields required by workflow."""

    contract_id: str


@dataclass(frozen=True, slots=True)
class CompiledRunPlanAnalysis:
    """Compiled analysis-selection fields required by workflow."""

    metrics: tuple[str, ...]
    slices: tuple[str, ...]
    finding_policy_profile: str | None
    summary_mode: str | None


@dataclass(frozen=True, slots=True)
class CompiledRunPlan:
    """Deterministic workflow-facing recipe compilation output."""

    identity: CompiledRunPlanIdentity
    input: CompiledRunPlanInput
    contract: CompiledRunPlanContract
    analysis: CompiledRunPlanAnalysis


def compile_recipe_v0_lite(recipe: RecipeV0Lite) -> CompiledRunPlan:
    """Compile one resolved v0-lite recipe into a workflow-facing run plan.

    Args:
        recipe: Resolved and validated recipe runtime model.

    Returns:
        Deterministic compiled run plan for downstream workflow consumption.
    """
    return CompiledRunPlan(
        identity=CompiledRunPlanIdentity(
            recipe_id=recipe.identity.recipe_id,
            recipe_version=recipe.identity.version,
        ),
        input=CompiledRunPlanInput(
            run_selector=recipe.input_binding.run_selector,
            source_experiment=recipe.input_binding.source_experiment,
            required_metrics=recipe.input_binding.required_metrics,
            required_artifacts=recipe.input_binding.required_artifacts,
            custom_reference_run_id=recipe.input_binding.custom_reference_run_id,
        ),
        contract=CompiledRunPlanContract(
            contract_id=recipe.contract_binding.contract_id,
        ),
        analysis=CompiledRunPlanAnalysis(
            metrics=recipe.metrics_slices.metrics,
            slices=recipe.metrics_slices.slices,
            finding_policy_profile=recipe.finding_policy.profile,
            summary_mode=recipe.output_binding.summary_mode,
        ),
    )
