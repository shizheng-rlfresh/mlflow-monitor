"""Recipe compilation pipeline for the built-in recipe format."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow_monitor.recipe import (
    RecipeV0Lite,
)


@dataclass(frozen=True, slots=True)
class CompiledRunPlanIdentity:
    """Compiled recipe identity fields required by workflow.

    Attributes:
        recipe_id: Stable recipe identifier.
        recipe_version: Recipe version identifier.
    """

    recipe_id: str
    recipe_version: str


@dataclass(frozen=True, slots=True)
class CompiledRunPlanInput:
    """Compiled input-selection fields required by workflow.

    Attributes:
        run_selector: Selector for the run to monitor.
        source_experiment: Source experiment for the run to monitor.
        required_metrics: Metric names that must exist on the source run.
        required_artifacts: Artifact names that must exist on the source run.
        custom_reference_monitoring_run_id: Additional timeline reference monitoring run id used for analysis diffs.
    """  # noqa: E501

    run_selector: str
    source_experiment: str | None
    required_metrics: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    custom_reference_monitoring_run_id: str | None


@dataclass(frozen=True, slots=True)
class CompiledRunPlanContract:
    """Compiled contract-selection fields required by workflow.

    Attributes:
        contract_id: Stable contract identifier.
    """

    contract_id: str


@dataclass(frozen=True, slots=True)
class CompiledRunPlanAnalysis:
    """Compiled analysis-selection fields required by workflow.

    Attributes:
        metrics: Metric names selected by the recipe.
        slices: Slice names selected by the recipe.
        finding_policy_profile: Finding policy profile selected by the recipe.
        summary_mode: Summary mode selected by the recipe.
    """

    metrics: tuple[str, ...]
    slices: tuple[str, ...]
    finding_policy_profile: str | None
    summary_mode: str | None


@dataclass(frozen=True, slots=True)
class CompiledRunPlan:
    """Deterministic workflow-facing recipe compilation output.

    Attributes:
        identity: Compiled recipe identity fields required by workflow.
        input: Compiled input-selection fields required by workflow.
        contract: Compiled contract-selection fields required by workflow.
        analysis: Compiled analysis-selection fields required by workflow.
    """

    identity: CompiledRunPlanIdentity
    input: CompiledRunPlanInput
    contract: CompiledRunPlanContract
    analysis: CompiledRunPlanAnalysis


def compile_recipe_v0_lite(recipe: RecipeV0Lite) -> CompiledRunPlan:
    """Compile one resolved recipe into a workflow-facing run plan.

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
            custom_reference_monitoring_run_id=recipe.input_binding.custom_reference_monitoring_run_id,
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
