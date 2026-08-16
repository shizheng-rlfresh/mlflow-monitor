"""Workflow lifecycle helpers for MLflow-Monitor v0.

This module contains backend-agnostic workflow logic for two responsibilities:

1. Lifecycle transitions for monitoring runs.
2. Prepare-stage context resolution before contract checking begins.

Prepare-stage resolution combines caller inputs (run identity, compiled plan,
resolved contract, optional first-run baseline input) with gateway-resolved
state (timeline, source run, prior monitoring runs, and optional references).
The workflow layer decides what must be resolved for a run to proceed, while
the gateway owns all persistence-specific mechanics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from mlflow_monitor.contract_checker import (
    ContractChecker,
    make_contract_evaluation_context,
)
from mlflow_monitor.domain import (
    Contract,
    ContractCheckResult,
    LifecycleStatus,
    MonitoringRunReference,
    Run,
)
from mlflow_monitor.errors import (
    CheckStageError,
    InvalidRunTransition,
    InvariantViolation,
    PrepareStageError,
)
from mlflow_monitor.gateway import MonitoringGateway, TimelineState
from mlflow_monitor.invariant import validate_contract_check_result
from mlflow_monitor.recipe_compiler import CompiledRecipe

_ALLOWED_TRANSITIONS = {
    LifecycleStatus.CREATED: {
        LifecycleStatus.PREPARED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.PREPARED: {
        LifecycleStatus.CHECKED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.CHECKED: {
        LifecycleStatus.ANALYZED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.ANALYZED: {
        LifecycleStatus.CLOSED,
        LifecycleStatus.FAILED,
    },
    LifecycleStatus.CLOSED: set(),
    LifecycleStatus.FAILED: set(),
}


@dataclass(frozen=True, slots=True)
class BaselineResolutionResult:
    """Result of baseline source run resolution for prepare-stage context.

    Attributes:
        baseline_source_run_id: Resolved baseline source run id.
        requires_bootstrap: Whether the baseline source run must be bootstrapped.
    """

    baseline_source_run_id: str
    requires_bootstrap: bool


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """Resolved prepare-stage context required before contract checking.

    Attributes:
        artifact_schema_version: Version of the prepared context artifact schema.
        monitoring_run_id: Stable monitoring run identifier.
        source_run_id: Resolved source training run id.
        subject_id: Stable monitored subject identifier.
        timeline_id: Stable timeline identifier.
        sequence_index: Sequence index within the timeline.
        baseline_source_run_id: Resolved baseline source run id.
        effective_recipe: Resolved effective compiled Recipe.
        contract: Resolved contract.
        references: Tuple of resolved monitoring run references.
    """

    artifact_schema_version: str
    monitoring_run_id: str
    source_run_id: str
    subject_id: str
    timeline_id: str
    sequence_index: int
    baseline_source_run_id: str | None
    effective_recipe: CompiledRecipe
    contract: Contract
    references: tuple[MonitoringRunReference, ...]


def transition_run(run: Run, to_status: LifecycleStatus) -> Run:
    """Return a new run with an updated lifecycle status if the move is legal.

    Args:
        run: The run whose lifecycle should advance.
        to_status: The target lifecycle status.

    Raises:
        InvalidRunTransition: If the requested transition is not allowed in v0.

    Returns:
        A new run value with the updated lifecycle status.
    """
    from_status = run.lifecycle_status

    if to_status not in _ALLOWED_TRANSITIONS[from_status]:
        raise InvalidRunTransition(
            from_status=from_status,
            to_status=to_status,
            message=f"Cannot transition run from {from_status} to {to_status}.",
        )

    return replace(run, lifecycle_status=to_status)


def prepare_run_context(
    *,
    monitoring_run_id: str,
    subject_id: str,
    compiled_recipe: CompiledRecipe,
    gateway: MonitoringGateway,
    source_run_id: str,
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
        baseline_source_run_id: Optional baseline source run id used to bootstrap (pin)
            timeline baseline, or to explicitly confirm the baseline for an existing timeline.
        custom_reference_monitoring_run_id: Optional invocation-owned Monitoring
            Run selected as a custom reference.

    Raises:
        PrepareStageError: If required prepare-stage references or inputs are missing.

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

    if custom_reference_monitoring_run_id is not None:
        custom_reference_monitoring_run_id = gateway.resolve_timeline_monitoring_run_id(
            subject_id,
            custom_reference_monitoring_run_id,
        )
        if custom_reference_monitoring_run_id is None:
            raise PrepareStageError(
                code="prepare_custom_reference_not_found",
                message=(
                    "Custom reference monitoring run could not be resolved on the subject timeline."
                ),
                details=(("subject_id", subject_id),),
            )

    if baseline_resolution_result.requires_bootstrap:
        # race handling
        timeline_pin_baseline_result = gateway.pin_timeline_baseline(
            subject_id,
            baseline_resolution_result.baseline_source_run_id,
        )

        timeline_state = gateway.get_timeline_state(subject_id)
        if timeline_state is None:
            _id = subject_id
            raise PrepareStageError(
                code="prepare_timeline_initialization_failed",
                message=(
                    f"Timeline initialization did not materialize state for subject_id={_id}."
                ),
                details=(("subject_id", subject_id),),
            )
        if timeline_pin_baseline_result.baseline_pinned:
            if (
                timeline_state.baseline_source_run_id
                != baseline_resolution_result.baseline_source_run_id
            ):
                _id = subject_id
                raise PrepareStageError(
                    code="prepare_timeline_pin_failed",
                    message=(f"Timeline pinning did not materialize state for subject_id={_id}."),
                    details=(("subject_id", subject_id),),
                )
        elif (
            timeline_state.baseline_source_run_id
            != baseline_resolution_result.baseline_source_run_id
        ):
            _id = subject_id
            _provided_baseline = baseline_source_run_id
            _resolved_baseline = baseline_resolution_result.baseline_source_run_id
            _existing_baseline = timeline_state.baseline_source_run_id
            raise PrepareStageError(
                code="prepare_baseline_override_existing_timeline",
                message=(
                    f"Provided baseline_source_run_id={_provided_baseline!r} "
                    f"with resolved_baseline_source_run_id={_resolved_baseline!r} "
                    "does not match existing timeline "
                    f"baseline_source_run_id={_existing_baseline!r} for subject_id={_id}. "
                    "Overriding an existing timeline's baseline is not allowed."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", _provided_baseline),
                ),
            )
    else:
        timeline_state = gateway.get_timeline_state(subject_id)

    if timeline_state is None:
        _id = subject_id
        raise PrepareStageError(
            code="prepare_timeline_initialization_failed",
            message=(f"Timeline initialization did not materialize state for subject_id={_id}."),
            details=(("subject_id", subject_id),),
        )

    # This should be impossible to happen, adding this check as a safeguard.
    if not timeline_state.baseline_source_run_id:
        raise PrepareStageError(
            code="prepare_baseline_missing",
            message=(f"Timeline for subject_id={subject_id} does not have a pinned baseline."),
            details=(("subject_id", subject_id),),
        )

    timeline_runs = gateway.list_timeline_monitoring_runs(subject_id, exclude_failed=True)
    previous_monitoring_run_id = timeline_runs[-1].monitoring_run_id if timeline_runs else None

    return PreparedContext(
        artifact_schema_version="v0",
        monitoring_run_id=monitoring_run_id,
        source_run_id=resolved_source_run_id,
        subject_id=subject_id,
        timeline_id=timeline_state.timeline_id,
        sequence_index=0,
        baseline_source_run_id=timeline_state.baseline_source_run_id,
        effective_recipe=compiled_recipe,
        contract=compiled_recipe.contract,
        references=,
    )

    # return PreparedContext(
    #     monitoring_run_id=monitoring_run_id,
    #     subject_id=subject_id,
    #     recipe_id=compiled_recipe.identity.recipe_id,
    #     recipe_version=compiled_recipe.identity.recipe_version,
    #     contract_id=compiled_recipe.contract.contract_id,
    #     source_experiment=compiled_recipe.source_requirements.source_experiment,
    #     timeline_id=timeline_state.timeline_id,
    #     baseline_source_run_id=timeline_state.baseline_source_run_id,
    #     previous_monitoring_run_id=previous_monitoring_run_id,
    #     active_lkg_monitoring_run_id=gateway.resolve_active_lkg_monitoring_run_id(subject_id),
    #     custom_reference_monitoring_run_id=custom_reference_monitoring_run_id,
    #     source_run_id=resolved_source_run_id,
    #     contract=compiled_recipe.contract,
    #     required_metrics=compiled_recipe.source_requirements.required_metric_names,
    #     required_artifacts=compiled_recipe.source_requirements.required_artifact_paths,
    # )


def execute_contract_check(
    prepared_context: PreparedContext,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
) -> ContractCheckResult:
    """Evaluate the prepared contract context and return the check result.

    Args:
        prepared_context: Resolved prepare-stage context for one contract evaluation.
        gateway: Gateway used to read source-run evidence.
        contract_checker: Checker implementation.

    Raises:
        CheckStageError: If required evidence is missing or the checker result
            violates invariants.

    Returns:
        Validated contract check result for the prepared context.
    """
    baseline_evidence = gateway.get_source_run_contract_evidence(
        source_run_id=prepared_context.baseline_source_run_id,
    )
    if baseline_evidence is None:
        raise CheckStageError(
            code="check_missing_baseline_evidence",
            message="Baseline contract evidence could not be resolved.",
            details=(("baseline_source_run_id", prepared_context.baseline_source_run_id),),
        )

    current_evidence = gateway.get_source_run_contract_evidence(
        source_run_id=prepared_context.source_run_id,
    )
    if current_evidence is None:
        raise CheckStageError(
            code="check_missing_current_evidence",
            message="Current contract evidence could not be resolved.",
            details=(("source_run_id", prepared_context.source_run_id),),
        )

    evaluation_context = make_contract_evaluation_context(
        subject_id=prepared_context.subject_id,
        source_run_id=prepared_context.source_run_id,
        baseline_source_run_id=prepared_context.baseline_source_run_id,
        baseline_context=baseline_evidence,
        current_context=current_evidence,
    )

    result = contract_checker.check(prepared_context.contract, evaluation_context)

    try:
        validate_contract_check_result(result)
    except InvariantViolation as exc:
        raise CheckStageError(
            code="check_result_invalid",
            message="Contract checker produced an invalid contract check result.",
            details=(("monitoring_run_id", prepared_context.monitoring_run_id),),
        ) from exc

    return result


def _resolve_baseline_for_prepare(
    subject_id: str,
    compiled_recipe: CompiledRecipe,
    gateway: MonitoringGateway,
    timeline_state: TimelineState | None,
    baseline_source_run_id: str | None = None,
) -> BaselineResolutionResult:
    """Resolve baseline source run for prepare when no timeline exists.

    Handle races of timeline initialization and baseline bootstrapping.

    Args:
        subject_id: Stable monitored subject identifier.
        compiled_recipe: Execution-ready compiled Recipe.
        gateway: Gateway used for timeline and source-run reads.
        timeline_state: Timeline state for the subject, if it exists.
        baseline_source_run_id: Caller-supplied baseline source run id to resolve.

    Raises:
        PrepareStageError: If a new timeline must be bootstrapped but the provided baseline
                           is invalid or missing,
                           or if the provided baseline attempts to override an existing timeline.

    Returns:
        Baseline resolution result containing timeline and resolved baseline information.
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

    # The timeline exists, so we can check for a pinned baseline.
    pinned_baseline = timeline_state.baseline_source_run_id

    # If there is no pinned baseline, we will need to bootstrap the baseline
    if pinned_baseline is None:
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
            requires_bootstrap=True,
        )

    # If the timeline exists, we do not need to bootstrap the baseline.
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

        if resolved_baseline != pinned_baseline:
            raise PrepareStageError(
                code="prepare_baseline_override_existing_timeline",
                message=(
                    f"Provided baseline_source_run_id={baseline_source_run_id!r} "
                    f"with resolved baseline_source_run_id={resolved_baseline!r} does not match "
                    f"existing timeline pinned baseline_source_run_id={pinned_baseline!r} "
                    f"for subject_id={subject_id!r}. "
                    "Overriding an existing timeline's baseline is not allowed."
                ),
                details=(
                    ("subject_id", subject_id),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

    return BaselineResolutionResult(
        baseline_source_run_id=pinned_baseline,
        requires_bootstrap=False,
    )
