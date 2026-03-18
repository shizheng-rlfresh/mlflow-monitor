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

from mlflow_monitor.domain import Contract, LifecycleStatus, Run
from mlflow_monitor.errors import InvalidRunTransition, PrepareStageError
from mlflow_monitor.gateway import MonitoringGateway, TimelineState
from mlflow_monitor.recipe_compiler import CompiledRunPlan

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
    """Result of baseline source run resolution for prepare-stage context."""

    timeline_id: (
        str | None
    )  # might not be useful at all, but could be helpful for logging and error messages
    baseline_source_run_id: str
    requires_bootstrap: bool


@dataclass(frozen=True, slots=True)
class PreparedContext:
    """Resolved prepare-stage context required before contract checking."""

    run_id: str
    subject_id: str
    recipe_id: str
    recipe_version: str
    contract_id: str
    run_selector: str
    source_experiment: str | None
    timeline_id: str
    baseline_source_run_id: str
    previous_run_id: str | None
    active_lkg_run_id: str | None
    custom_reference_run_id: str | None
    source_run_id: str
    contract: Contract
    required_metrics: tuple[str, ...]
    required_artifacts: tuple[str, ...]


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
    run_id: str,
    subject_id: str,
    compiled_plan: CompiledRunPlan,
    resolved_contract: Contract,
    gateway: MonitoringGateway,
    runtime_source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> PreparedContext:
    """Resolve prepare-stage references and validate required source-run inputs.

    Args:
        run_id: Monitoring run identifier being prepared.
        subject_id: Stable monitored subject identifier.
        compiled_plan: Workflow-facing compiled recipe plan.
        resolved_contract: Effective resolved contract for the run.
        gateway: Gateway used for timeline and source-run reads.
        runtime_source_run_id: Caller-supplied source run id used only when the
            compiled selector is the reserved runtime token.
        baseline_source_run_id: Optional baseline source run id used to bootstrap
            a missing timeline baseline, or to explicitly confirm/pin the baseline
            for an existing timeline, regardless of the compiled run selector token.

    Raises:
        PrepareStageError: If required prepare-stage references or inputs are missing.

    Returns:
        Success-only prepared context for later workflow stages.
    """
    if resolved_contract.contract_id != compiled_plan.contract.contract_id:
        raise PrepareStageError(
            code="prepare_contract_mismatch",
            message=(
                "Resolved contract does not match compiled plan "
                f"({resolved_contract.contract_id!r} != {compiled_plan.contract.contract_id!r})."
            ),
            details=(
                ("resolved_contract_id", resolved_contract.contract_id),
                ("compiled_contract_id", compiled_plan.contract.contract_id),
            ),
        )

    timeline_state = gateway.get_timeline_state(subject_id)
    baseline_resolution_result = _resolve_baseline_for_prepare(
        subject_id=subject_id,
        compiled_plan=compiled_plan,
        gateway=gateway,
        timeline_state=timeline_state,
        baseline_source_run_id=baseline_source_run_id,
    )

    source_run_id = gateway.resolve_source_run_id(
        subject_id=subject_id,
        source_experiment=compiled_plan.input.source_experiment,
        run_selector=compiled_plan.input.run_selector,
        runtime_source_run_id=runtime_source_run_id,
    )
    if source_run_id is None:
        raise PrepareStageError(
            code="prepare_source_run_not_found",
            message=(
                "Source training run could not be resolved for "
                f"subject_id={subject_id} and run_selector={compiled_plan.input.run_selector!r}."
            ),
            details=(("subject_id", subject_id),),
        )

    missing_metrics = gateway.get_missing_source_run_metrics(
        run_id=source_run_id,
        required_metrics=compiled_plan.input.required_metrics,
    )
    if missing_metrics:
        missing_metric = missing_metrics[0]
        raise PrepareStageError(
            code="prepare_missing_required_metric",
            message=f"Source run {source_run_id} is missing required metric {missing_metric}.",
            details=(("source_run_id", source_run_id), ("metric", missing_metric)),
        )

    missing_artifacts = gateway.get_missing_source_run_artifacts(
        run_id=source_run_id,
        required_artifacts=compiled_plan.input.required_artifacts,
    )
    if missing_artifacts:
        missing_artifact = missing_artifacts[0]
        raise PrepareStageError(
            code="prepare_missing_required_artifact",
            message=(
                f"Source run {source_run_id} is missing required artifact {missing_artifact}."
            ),
            details=(("source_run_id", source_run_id), ("artifact", missing_artifact)),
        )

    custom_reference_run_id = compiled_plan.input.custom_reference_run_id
    if custom_reference_run_id is not None:
        custom_reference_run_id = gateway.resolve_timeline_run_id(
            subject_id,
            custom_reference_run_id,
        )
        if custom_reference_run_id is None:
            raise PrepareStageError(
                code="prepare_custom_reference_not_found",
                message=("Custom reference run could not be resolved on the subject timeline."),
                details=(("subject_id", subject_id),),
            )

    if baseline_resolution_result.requires_bootstrap:
        # race handling
        timeline_init_result = gateway.initialize_timeline(
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
        if timeline_init_result.created:
            if (
                timeline_state.baseline_source_run_id
                != baseline_resolution_result.baseline_source_run_id
            ):
                _id = subject_id
                raise PrepareStageError(
                    code="prepare_timeline_initialization_failed",
                    message=(
                        f"Timeline initialization did not materialize state for subject_id={_id}."
                    ),
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

    # for logging purpoose
    replace(
        baseline_resolution_result,
        timeline_id=timeline_state.timeline_id,
    )

    timeline_runs = gateway.list_timeline_runs(subject_id, exclude_failed=True)
    previous_run_id = timeline_runs[-1].run_id if timeline_runs else None

    return PreparedContext(
        run_id=run_id,
        subject_id=subject_id,
        recipe_id=compiled_plan.identity.recipe_id,
        recipe_version=compiled_plan.identity.recipe_version,
        contract_id=compiled_plan.contract.contract_id,
        run_selector=compiled_plan.input.run_selector,
        source_experiment=compiled_plan.input.source_experiment,
        timeline_id=timeline_state.timeline_id,
        baseline_source_run_id=timeline_state.baseline_source_run_id,
        previous_run_id=previous_run_id,
        active_lkg_run_id=gateway.resolve_active_lkg_run_id(subject_id),
        custom_reference_run_id=custom_reference_run_id,
        source_run_id=source_run_id,
        contract=resolved_contract,
        required_metrics=compiled_plan.input.required_metrics,
        required_artifacts=compiled_plan.input.required_artifacts,
    )


def _resolve_baseline_for_prepare(
    subject_id: str,
    compiled_plan: CompiledRunPlan,
    gateway: MonitoringGateway,
    timeline_state: TimelineState | None,
    baseline_source_run_id: str | None = None,
) -> BaselineResolutionResult:
    """Resolve baseline source run for prepare when no timeline exists.

    Handle races of timeline initialization and baseline bootstrapping.

    Args:
        subject_id: Stable monitored subject identifier.
        compiled_plan: Workflow-facing compiled recipe plan.
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
    if timeline_state is not None:
        if baseline_source_run_id is None:
            pass
        else:
            resolved_baseline_source_run_id = gateway.resolve_source_run_id(
                subject_id=subject_id,
                source_experiment=compiled_plan.input.source_experiment,
                run_selector=baseline_source_run_id,
            )
            if resolved_baseline_source_run_id == timeline_state.baseline_source_run_id:
                pass
            else:
                _id = subject_id
                _provided_baseline = baseline_source_run_id
                _existing_baseline = timeline_state.baseline_source_run_id
                raise PrepareStageError(
                    code="prepare_baseline_override_existing_timeline",
                    message=(
                        f"Provided baseline_source_run_id={_provided_baseline!r} "
                        f"with resolved_baseline_source_run_id={resolved_baseline_source_run_id!r} "
                        "does not match existing timeline "
                        f"baseline_source_run_id={_existing_baseline!r} for subject_id={_id}. "
                        "Overriding an existing timeline's baseline is not allowed."
                    ),
                    details=(
                        ("subject_id", subject_id),
                        ("baseline_source_run_id", _provided_baseline),
                    ),
                )

        return BaselineResolutionResult(
            timeline_id=timeline_state.timeline_id,
            baseline_source_run_id=timeline_state.baseline_source_run_id,
            requires_bootstrap=False,
        )
    elif baseline_source_run_id:
        resolved_baseline_source_run_id = gateway.resolve_source_run_id(
            subject_id=subject_id,
            source_experiment=compiled_plan.input.source_experiment,
            run_selector=baseline_source_run_id,
        )
        if resolved_baseline_source_run_id is None:
            raise PrepareStageError(
                code="prepare_invalid_bootstrap_baseline",
                message=(
                    f"Baseline source run could not be resolved for subject_id={subject_id}, "
                    f"compiled_plan.input.source_experiment={compiled_plan.input.source_experiment!r}, "  # noqa: E501
                    f"and baseline_source_run_id={baseline_source_run_id!r}."
                ),
                details=(
                    ("subject_id", subject_id),
                    (
                        "compiled_plan.input.source_experiment",
                        compiled_plan.input.source_experiment,
                    ),
                    ("baseline_source_run_id", baseline_source_run_id),
                ),
            )

        return BaselineResolutionResult(
            timeline_id=None,
            baseline_source_run_id=resolved_baseline_source_run_id,
            requires_bootstrap=True,
        )
    else:
        raise PrepareStageError(
            code="prepare_missing_baseline_no_timeline",
            message=(
                f"No timeline exists for subject_id={subject_id} "
                "and no baseline_source_run_id was provided. "
                "A valid baseline_source_run_id is required to bootstrap a new timeline."
            ),
            details=(
                ("subject_id", subject_id),
                ("baseline_source_run_id", baseline_source_run_id),
            ),
        )
