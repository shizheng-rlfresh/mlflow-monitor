"""Prepare stage module for mlflow-monitor v0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from mlflow_monitor.domain import DiffReferenceKind, LifecycleStatus, MonitoringRunReference
from mlflow_monitor.errors import (
    PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
    PrepareStageError,
)
from mlflow_monitor.gateway import MonitoringGateway
from mlflow_monitor.gateway.models import TimelineState
from mlflow_monitor.recipe.recipe_compiler import CompiledRecipe

from .prepared_context import PreparedContext, PreparedReferencePlanEntry


@dataclass(frozen=True, slots=True)
class BaselineResolutionResult:
    """Result of baseline source run resolution for prepare-stage context.

    Attributes:
        baseline_source_run_id: Resolved baseline source run id.
    """

    baseline_source_run_id: str


def prepare_run_context(
    *,
    monitoring_run_id: str,
    subject_id: str,
    compiled_recipe: CompiledRecipe,
    gateway: MonitoringGateway,
    source_run_id: str,
    sequence_index: int,
    baseline_source_run_id: str | None = None,
    custom_reference_monitoring_run_id: str | None = None,
) -> PreparedContext:
    """Resolve prepare-stage references and validate required source-run inputs.

    Args:
        monitoring_run_id: Monitoring run identifier being prepared.
        subject_id: Stable monitored subject identifier.
        compiled_recipe: Execution-ready compiled Recipe.
        gateway: Gateway used for timeline and source-run reads.
        source_run_id: Invocation-owned Source Training Run identifier.
        sequence_index: Stable allocation order within the Timeline.
        baseline_source_run_id: Optional baseline source run id used to initialize or
            explicitly confirm the immutable timeline baseline.
        custom_reference_monitoring_run_id: Optional invocation-owned Monitoring
            Run selected as a custom reference.

    Raises:
        PrepareStageError: If required Prepare inputs are missing or an invocation-owned
            custom Reference Monitoring Run is invalid.

    Returns:
        Success-only prepared context for later workflow stages.
    """
    timeline_state = gateway.get_timeline_state(subject_id)
    baseline_resolution_result = _resolve_baseline_for_prepare(
        subject_id=subject_id,
        compiled_recipe=compiled_recipe,
        gateway=gateway,
        timeline_state=timeline_state,
        baseline_source_run_id=baseline_source_run_id,
    )

    resolved_source_run_id = gateway.resolve_source_run_id(
        subject_id=subject_id,
        source_experiment=compiled_recipe.source_requirements.source_experiment,
        source_run_id=source_run_id,
    )
    if resolved_source_run_id is None:
        raise PrepareStageError(
            code="prepare_source_run_not_found",
            message=(
                "Source training run could not be resolved for "
                f"subject_id={subject_id} and source_run_id={source_run_id!r}."
            ),
            details=(("subject_id", subject_id),),
        )

    missing_metrics = gateway.get_missing_source_run_metrics(
        source_run_id=resolved_source_run_id,
        required_metrics=compiled_recipe.source_requirements.required_metric_names,
    )
    if missing_metrics:
        missing_metric = missing_metrics[0]
        raise PrepareStageError(
            code="prepare_missing_required_metric",
            message=(
                f"Source run {resolved_source_run_id} is missing required metric {missing_metric}."
            ),
            details=(("source_run_id", resolved_source_run_id), ("metric", missing_metric)),
        )

    missing_artifacts = gateway.get_missing_source_run_artifacts(
        source_run_id=resolved_source_run_id,
        required_artifacts=compiled_recipe.source_requirements.required_artifact_paths,
    )
    if missing_artifacts:
        missing_artifact = missing_artifacts[0]
        raise PrepareStageError(
            code="prepare_missing_required_artifact",
            message=(
                f"Source run {resolved_source_run_id} is missing required artifact "
                f"{missing_artifact}."
            ),
            details=(
                ("source_run_id", resolved_source_run_id),
                ("artifact", missing_artifact),
            ),
        )

    custom_reference: MonitoringRunReference | None = None
    if custom_reference_monitoring_run_id is not None:
        resolved_custom_monitoring_run_id = gateway.resolve_timeline_monitoring_run_id(
            subject_id,
            custom_reference_monitoring_run_id,
        )
        if resolved_custom_monitoring_run_id is None:
            raise PrepareStageError(
                code="prepare_custom_reference_not_found",
                message=(
                    "Custom reference monitoring run could not be resolved on the subject timeline."
                ),
                details=(("subject_id", subject_id),),
            )
        custom_record = gateway.get_monitoring_run(
            subject_id,
            resolved_custom_monitoring_run_id,
        )
        if custom_record is None:
            raise PrepareStageError(
                code="prepare_custom_reference_not_found",
                message=(
                    "Custom reference monitoring run could not be resolved on the subject timeline."
                ),
                details=(("subject_id", subject_id),),
            )
        if custom_record.lifecycle_status is not LifecycleStatus.CLOSED:
            raise PrepareStageError(
                code="prepare_custom_reference_not_closed",
                message="Custom reference Monitoring Run must be closed.",
                details=(
                    (
                        "custom_reference_monitoring_run_id",
                        resolved_custom_monitoring_run_id,
                    ),
                ),
            )
        custom_reference = MonitoringRunReference(
            kind=DiffReferenceKind.CUSTOM,
            monitoring_run_id=custom_record.monitoring_run_id,
            source_run_id=custom_record.source_run_id,
        )

    reference_plan = [
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.BASELINE,
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.BASELINE,
                monitoring_run_id=None,
                source_run_id=baseline_resolution_result.baseline_source_run_id,
            ),
            unavailable_reason=None,
        )
    ]

    previous = max(
        (
            record
            for record in gateway.list_timeline_monitoring_runs(subject_id)
            if record.lifecycle_status is LifecycleStatus.CLOSED
            and record.sequence_index < sequence_index
        ),
        key=lambda record: record.sequence_index,
        default=None,
    )
    if previous is not None:
        reference_plan.append(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.PREVIOUS,
                reference=MonitoringRunReference(
                    kind=DiffReferenceKind.PREVIOUS,
                    monitoring_run_id=previous.monitoring_run_id,
                    source_run_id=previous.source_run_id,
                ),
                unavailable_reason=None,
            )
        )
    else:
        reference_plan.append(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.PREVIOUS,
                reference=None,
                unavailable_reason="previous_reference_missing",
            )
        )

    reference_plan.append(
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.LKG,
            reference=None,
            unavailable_reason="lkg_not_selected",
        )
    )

    if custom_reference is not None:
        reference_plan.append(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.CUSTOM,
                reference=custom_reference,
                unavailable_reason=None,
            )
        )

    timeline_state = gateway.reconcile_timeline_baseline(
        subject_id,
        monitoring_run_id,
        baseline_resolution_result.baseline_source_run_id,
    )

    # reconciled_baseline_source_run_id is guaranteed to be a str
    # if .reconcile_timeline_baseline() returns without raising an exception.
    # Therefore, adding an error check here is unnecessary, and we can safely cast it to str.
    reconciled_baseline_source_run_id = cast(str, timeline_state.baseline_source_run_id)

    return PreparedContext(
        monitoring_run_id=monitoring_run_id,
        source_run_id=resolved_source_run_id,
        subject_id=subject_id,
        timeline_id=timeline_state.timeline_id,
        sequence_index=sequence_index,
        baseline_source_run_id=reconciled_baseline_source_run_id,
        effective_recipe=compiled_recipe.effective_plan,
        contract=compiled_recipe.contract,
        reference_plan=tuple(reference_plan),
    )


def _resolve_baseline_for_prepare(
    subject_id: str,
    compiled_recipe: CompiledRecipe,
    gateway: MonitoringGateway,
    timeline_state: TimelineState | None,
    baseline_source_run_id: str | None = None,
) -> BaselineResolutionResult:
    """Resolve the immutable baseline source run for prepare.

    Args:
        subject_id: Stable monitored subject identifier.
        compiled_recipe: Execution-ready compiled Recipe.
        gateway: Gateway used for timeline and source-run reads.
        timeline_state: Timeline state for the subject, if it exists.
        baseline_source_run_id: Caller-supplied baseline source run id to resolve.

    Raises:
        PrepareStageError: If an uninitialized timeline has no valid baseline, or if
            the provided baseline attempts to override an established baseline.

    Returns:
        Baseline resolution result containing the resolved baseline information.
    """
    # The timeline does not exist yet, so we cannot resolve a baseline without it.
    if timeline_state is None:
        raise PrepareStageError(
            code="prepare_missing_timeline",
            message=(
                f"No timeline exists for the subject_id={subject_id!r} "
                "and baseline resolution cannot proceed. "
                "Consider allocating a monitoring run first."
            ),
            details=(("subject_id", subject_id),),
        )

    established_baseline = timeline_state.baseline_source_run_id

    if established_baseline is None:
        if baseline_source_run_id is None or baseline_source_run_id == "":
            raise PrepareStageError(
                code="prepare_missing_baseline_for_uninitialized_timeline",
                message=(
                    f"The timeline for subject_id={subject_id!r} has no pinned baseline "
                    "and no baseline_source_run_id was provided. "
                    "A valid baseline_source_run_id is required to bootstrap the timeline."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

        resolved_baseline = gateway.resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=compiled_recipe.source_requirements.source_experiment,
            source_run_id=baseline_source_run_id,
        )

        if resolved_baseline is None:
            raise PrepareStageError(
                code="prepare_invalid_bootstrap_baseline",
                message=(
                    f"Baseline source run could not be resolved for subject_id={subject_id!r}, "
                    f"source_experiment={compiled_recipe.source_requirements.source_experiment!r}, "
                    f"and baseline_source_run_id={baseline_source_run_id!r}."
                ),
                details=(
                    ("subject_id", subject_id),
                    (
                        "compiled_recipe.source_requirements.source_experiment",
                        compiled_recipe.source_requirements.source_experiment,
                    ),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

        return BaselineResolutionResult(
            baseline_source_run_id=resolved_baseline,
        )

    if baseline_source_run_id is not None:
        resolved_baseline = gateway.resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=compiled_recipe.source_requirements.source_experiment,
            source_run_id=baseline_source_run_id,
        )

        if resolved_baseline is None:
            raise PrepareStageError(
                code="prepare_invalid_baseline",
                message=(
                    f"Baseline source run could not be resolved for subject_id={subject_id!r}, "
                    f"compiled_recipe.source_requirements.source_experiment="
                    f"{compiled_recipe.source_requirements.source_experiment!r}, "
                    f"and baseline_source_run_id={baseline_source_run_id!r}."
                ),
                details=(
                    ("subject_id", subject_id),
                    (
                        "compiled_recipe.source_requirements.source_experiment",
                        compiled_recipe.source_requirements.source_experiment,
                    ),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

        if resolved_baseline != established_baseline:
            raise PrepareStageError(
                code=PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE,
                message=(
                    f"Provided baseline_source_run_id={baseline_source_run_id!r} "
                    f"with resolved baseline_source_run_id={resolved_baseline!r} does not match "
                    f"existing timeline pinned baseline_source_run_id={established_baseline!r} "
                    f"for subject_id={subject_id!r}. "
                    "Overriding an existing timeline's baseline is not allowed."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

    return BaselineResolutionResult(
        baseline_source_run_id=established_baseline,
    )
