"""Internal orchestration for the current create/prepare/check workflow."""

from __future__ import annotations

from dataclasses import dataclass

from mlflow_monitor.contract_checker import ContractChecker
from mlflow_monitor.domain import (
    ContractCheckResult,
    DiffReferenceKind,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    CheckStageError,
    GatewayConsistencyViolation,
    PrepareStageError,
    TerminalRunRetryError,
)
from mlflow_monitor.gateway import (
    IdempotencyKey,
    MonitoringGateway,
    MonitoringRunRecord,
)
from mlflow_monitor.recipe_compiler import (
    SYSTEM_DEFAULT_COMPILED_RECIPE,
    CompiledRecipe,
)
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult
from mlflow_monitor.workflow import (
    PreparedContext,
    execute_contract_check,
    prepare_run_context,
)

_OWNED_FAILURES = (
    PrepareStageError,
    CheckStageError,
)


@dataclass(frozen=True, slots=True)
class OrchestrationState:
    """Resolved orchestration inputs and run state for one monitoring request.

    Attributes:
        subject_id: The ID of the monitored subject this run is associated with.
        source_run_id: The original run ID from the training system that produced this run.
        baseline_source_run_id: The source run ID of the baseline this run is compared against
        compiled_recipe: The execution-ready compiled Recipe for this run.
        custom_reference_monitoring_run_id: Optional invocation-owned Reference
            Monitoring Run identifier.
        timeline_id: The Timeline identity returned by monitoring-run allocation.
        monitoring_run_id: The unique ID of the monitoring run to be executed.
        existing_monitoring_run: The existing monitoring run record, if any.
        is_new_monitoring_run: Whether this is a new monitoring run.
        sequence_index: The sequential index of this run within its timeline,
                        starting at 0 for the first run.
    """

    subject_id: str
    source_run_id: str
    baseline_source_run_id: str | None
    compiled_recipe: CompiledRecipe
    custom_reference_monitoring_run_id: str | None
    timeline_id: str
    monitoring_run_id: str
    existing_monitoring_run: MonitoringRunRecord | None
    is_new_monitoring_run: bool
    sequence_index: int


def run_orchestration(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
    custom_reference_monitoring_run_id: str | None = None,
    recipe: CompiledRecipe | None = None,
) -> MonitorRunResult:
    """Execute the orchestration for one monitoring run, including prepare and check stages.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        source_run_id: The original run ID from the training system that produced this run.
        baseline_source_run_id: The source run ID of the baseline this run is compared against
        gateway: The monitoring gateway to use for persistence during orchestration.
        contract_checker: The contract checker to use for executing the contract check stage.
        custom_reference_monitoring_run_id: Optional invocation-owned Reference
            Monitoring Run identifier.
        recipe: Precompiled Recipe, or ``None`` for the system default.

    Returns:
        The result of the monitoring run execution, including comparability status and any findings.

    Raises:
        TypeError: If ``recipe`` is neither a ``CompiledRecipe`` nor ``None``.

    """  # noqa: E501
    compiled_recipe = _resolve_startup(recipe)
    state_or_result = _resolve_orchestration_state(
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        compiled_recipe=compiled_recipe,
        custom_reference_monitoring_run_id=custom_reference_monitoring_run_id,
        gateway=gateway,
    )
    if isinstance(state_or_result, MonitorRunResult):
        return state_or_result

    prepare_outcome = _run_prepare_monitoring_run_slice(state_or_result, gateway)
    if isinstance(prepare_outcome, MonitorRunResult):
        return prepare_outcome

    return _run_check_monitoring_run_slice(
        state=state_or_result,
        prepared_context=prepare_outcome,
        gateway=gateway,
        contract_checker=contract_checker,
    )


def _resolve_startup(recipe: CompiledRecipe | None) -> CompiledRecipe:
    """Validate and resolve the execution-ready Recipe before allocation."""
    if recipe is None:
        return SYSTEM_DEFAULT_COMPILED_RECIPE
    if not isinstance(recipe, CompiledRecipe):
        raise TypeError("recipe must be a CompiledRecipe or None.")
    return recipe


def _resolve_orchestration_state(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None,
    compiled_recipe: CompiledRecipe,
    custom_reference_monitoring_run_id: str | None,
    gateway: MonitoringGateway,
) -> OrchestrationState | MonitorRunResult:
    """Resolve idempotency state and apply rerun short-circuit policy."""
    idempotency_key = IdempotencyKey(
        subject_id=subject_id,
        source_run_id=source_run_id,
        recipe_id=compiled_recipe.identity.recipe_id,
        recipe_version=compiled_recipe.identity.recipe_version,
    )
    create_or_reuse_result = gateway.create_or_reuse_monitoring_run(idempotency_key)
    state = OrchestrationState(
        subject_id=subject_id,
        source_run_id=create_or_reuse_result.source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        compiled_recipe=compiled_recipe,
        custom_reference_monitoring_run_id=custom_reference_monitoring_run_id,
        timeline_id=create_or_reuse_result.timeline_id,
        monitoring_run_id=create_or_reuse_result.monitoring_run_id,
        existing_monitoring_run=create_or_reuse_result.existing_monitoring_run,
        is_new_monitoring_run=create_or_reuse_result.existing_monitoring_run is None,
        sequence_index=create_or_reuse_result.sequence_index,
    )
    return _short_circuit_existing_monitoring_run(state, gateway)


def _short_circuit_existing_monitoring_run(
    state: OrchestrationState,
    gateway: MonitoringGateway,
) -> OrchestrationState | MonitorRunResult:
    """Return an existing-run result early when idempotency policy requires it."""
    if state.existing_monitoring_run is None:
        return state

    if state.existing_monitoring_run.lifecycle_status is LifecycleStatus.FAILED:
        result = _build_failure_monitoring_run_result(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            timeline_id=state.timeline_id,
            stage="prepare",
            error=_build_terminal_failed_monitoring_run_rerun_error(
                subject_id=state.subject_id,
                monitoring_run_id=state.monitoring_run_id,
            ),
        )
        gateway.finalize_monitoring_run_result(
            monitoring_run_id=state.monitoring_run_id,
            result=result,
        )
        return result

    if state.existing_monitoring_run.contract_check_result is None:
        return state

    replay_error = _validate_checked_monitoring_run_rerun_inputs(
        subject_id=state.subject_id,
        baseline_source_run_id=state.baseline_source_run_id,
        source_experiment=state.compiled_recipe.source_requirements.source_experiment,
        gateway=gateway,
    )
    if replay_error is not None:
        return _build_failure_monitoring_run_result(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            timeline_id=state.timeline_id,
            stage="prepare",
            error=replay_error,
        )

    result = _build_existing_checked_monitoring_run_result(
        subject_id=state.subject_id,
        monitoring_run_id=state.monitoring_run_id,
        timeline_id=state.timeline_id,
        existing_monitoring_run=state.existing_monitoring_run,
    )
    gateway.finalize_monitoring_run_result(
        monitoring_run_id=state.monitoring_run_id,
        result=result,
    )
    return result


def _run_prepare_monitoring_run_slice(
    state: OrchestrationState,
    gateway: MonitoringGateway,
) -> PreparedContext | MonitorRunResult:
    """Run the prepare slice, including persistence and failure normalization."""
    if state.is_new_monitoring_run:
        gateway.upsert_monitoring_run(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            source_run_id=state.source_run_id,
            lifecycle_status=LifecycleStatus.CREATED,
            sequence_index=state.sequence_index,
        )

    try:
        prepared_context = prepare_run_context(
            monitoring_run_id=state.monitoring_run_id,
            subject_id=state.subject_id,
            compiled_recipe=state.compiled_recipe,
            gateway=gateway,
            source_run_id=state.source_run_id,
            baseline_source_run_id=state.baseline_source_run_id,
            custom_reference_monitoring_run_id=state.custom_reference_monitoring_run_id,
        )
    except _OWNED_FAILURES as exc:
        gateway.upsert_monitoring_run(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            source_run_id=state.source_run_id,
            lifecycle_status=LifecycleStatus.FAILED,
            sequence_index=state.sequence_index,
        )
        result = _build_failure_monitoring_run_result(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            timeline_id=state.timeline_id,
            stage="prepare",
            error=exc,
        )
        gateway.finalize_monitoring_run_result(
            monitoring_run_id=state.monitoring_run_id,
            result=result,
        )
        return result

    if (
        state.existing_monitoring_run is None
        or state.existing_monitoring_run.lifecycle_status is LifecycleStatus.CREATED
    ):
        gateway.upsert_monitoring_run(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            source_run_id=state.source_run_id,
            lifecycle_status=LifecycleStatus.PREPARED,
            sequence_index=state.sequence_index,
        )

    return prepared_context


def _run_check_monitoring_run_slice(
    *,
    state: OrchestrationState,
    prepared_context: PreparedContext,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
) -> MonitorRunResult:
    """Run the check slice, including persistence and success replay handling."""
    existing_run = gateway.get_monitoring_run(state.subject_id, state.monitoring_run_id)
    if existing_run is not None and existing_run.contract_check_result is not None:
        result = _build_existing_checked_monitoring_run_result(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            timeline_id=state.timeline_id,
            existing_monitoring_run=existing_run,
        )
        gateway.finalize_monitoring_run_result(
            monitoring_run_id=state.monitoring_run_id,
            result=result,
        )
        return result

    try:
        contract_check_result = execute_contract_check(
            prepared_context=prepared_context,
            gateway=gateway,
            contract_checker=contract_checker,
        )
    except _OWNED_FAILURES as exc:
        gateway.upsert_monitoring_run(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            source_run_id=state.source_run_id,
            lifecycle_status=LifecycleStatus.FAILED,
            sequence_index=state.sequence_index,
        )
        result = _build_failure_monitoring_run_result(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            timeline_id=state.timeline_id,
            stage="check",
            error=exc,
        )
        gateway.finalize_monitoring_run_result(
            monitoring_run_id=state.monitoring_run_id,
            result=result,
        )
        return result

    gateway.upsert_monitoring_run(
        subject_id=state.subject_id,
        monitoring_run_id=state.monitoring_run_id,
        source_run_id=state.source_run_id,
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=state.sequence_index,
        contract_check_result=contract_check_result,
        references=_build_monitoring_run_references(prepared_context, gateway),
    )
    result = _build_success_monitoring_run_result(
        subject_id=state.subject_id,
        monitoring_run_id=state.monitoring_run_id,
        timeline_id=state.timeline_id,
        prepared_context=prepared_context,
        contract_check_result=contract_check_result,
        gateway=gateway,
    )
    gateway.finalize_monitoring_run_result(
        monitoring_run_id=state.monitoring_run_id,
        result=result,
    )
    return result


def _build_success_monitoring_run_result(
    *,
    subject_id: str,
    monitoring_run_id: str,
    timeline_id: str,
    prepared_context,
    contract_check_result: ContractCheckResult,
    gateway: MonitoringGateway,
) -> MonitorRunResult:
    """Build the canonical success result for one checked monitoring run.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        monitoring_run_id: The ID of the monitoring run.
        timeline_id: The Timeline identity returned by monitoring-run allocation.
        prepared_context: The prepared context produced by the prepare stage for this run.
        contract_check_result: The result of the contract check stage for this run.
        gateway: The monitoring gateway to use for retrieving any additional information needed.

    Returns:
        The canonical success result for this run, including comparability status and any findings.
    """
    return MonitorRunResult(
        monitoring_run_id=monitoring_run_id,
        subject_id=subject_id,
        timeline_id=timeline_id,
        lifecycle_status=LifecycleStatus.CHECKED,
        comparability_status=contract_check_result.status,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        references=_build_monitoring_run_references(prepared_context, gateway),
        error=None,
    )


def _build_existing_checked_monitoring_run_result(
    *,
    subject_id: str,
    monitoring_run_id: str,
    timeline_id: str,
    existing_monitoring_run: MonitoringRunRecord,
) -> MonitorRunResult:
    """Build a success result for an already checked idempotent run.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        monitoring_run_id: The ID of the monitoring run.
        timeline_id: The Timeline identity returned by monitoring-run allocation.
        existing_monitoring_run: The previously persisted monitoring run record for this run.

    Returns:
        The canonical success result for an already checked run.
    """
    return MonitorRunResult(
        monitoring_run_id=monitoring_run_id,
        subject_id=subject_id,
        timeline_id=timeline_id,
        lifecycle_status=LifecycleStatus.CHECKED,
        comparability_status=existing_monitoring_run.contract_check_result.status
        if existing_monitoring_run.contract_check_result is not None
        else None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        references=existing_monitoring_run.references,
        error=None,
    )


def _build_failure_monitoring_run_result(
    *,
    subject_id: str,
    monitoring_run_id: str,
    timeline_id: str,
    stage: str,
    error: Exception,
) -> MonitorRunResult:
    """Build the canonical failed result for a prepare/check execution error.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        monitoring_run_id: The ID of the monitoring run.
        timeline_id: The Timeline identity returned by monitoring-run allocation.
        stage: The stage during which the error occurred (e.g., "prepare" or "check").
        error: The exception raised during execution.

    Returns:
        The canonical failure result for this run, including error details.
    """
    return MonitorRunResult(
        monitoring_run_id=monitoring_run_id,
        subject_id=subject_id,
        timeline_id=timeline_id,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        references=(),
        error=MonitorRunError(
            code=_error_code_for_stage(stage, error),
            message=str(error),
            stage=stage,
            details=_error_details(error),
        ),
    )


def _build_monitoring_run_references(
    prepared_context,
    gateway: MonitoringGateway,
) -> tuple[MonitoringRunReference, ...]:
    """Build ordered typed references for persistence.

    Args:
        prepared_context: The prepared context produced by the prepare stage for this run.
        gateway: Gateway used to hydrate source identity for monitoring-run references.

    Returns:
        Ordered typed references used during contract check.
    """
    references = [
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id=prepared_context.baseline_source_run_id,
        )
    ]
    if prepared_context.previous_monitoring_run_id is not None:
        references.append(
            _hydrate_monitoring_run_reference(
                kind=DiffReferenceKind.PREVIOUS,
                subject_id=prepared_context.subject_id,
                monitoring_run_id=prepared_context.previous_monitoring_run_id,
                gateway=gateway,
            )
        )
    if prepared_context.active_lkg_monitoring_run_id is not None:
        references.append(
            _hydrate_monitoring_run_reference(
                kind=DiffReferenceKind.LKG,
                subject_id=prepared_context.subject_id,
                monitoring_run_id=prepared_context.active_lkg_monitoring_run_id,
                gateway=gateway,
            )
        )
    if prepared_context.custom_reference_monitoring_run_id is not None:
        references.append(
            _hydrate_monitoring_run_reference(
                kind=DiffReferenceKind.CUSTOM,
                subject_id=prepared_context.subject_id,
                monitoring_run_id=prepared_context.custom_reference_monitoring_run_id,
                gateway=gateway,
            )
        )
    return tuple(references)


def _hydrate_monitoring_run_reference(
    *,
    kind: DiffReferenceKind,
    subject_id: str,
    monitoring_run_id: str,
    gateway: MonitoringGateway,
) -> MonitoringRunReference:
    """Hydrate a complete reference pair from a monitoring-run pointer."""
    record = gateway.get_monitoring_run(subject_id, monitoring_run_id)
    if record is None:
        raise GatewayConsistencyViolation(
            code="monitoring_reference_inconsistent",
            message=(
                "Monitoring reference could not be hydrated for "
                f"monitoring_run_id={monitoring_run_id!r}."
            ),
            details=(
                ("kind", kind.value),
                ("monitoring_run_id", monitoring_run_id),
            ),
        )
    return MonitoringRunReference(
        kind=kind,
        monitoring_run_id=record.monitoring_run_id,
        source_run_id=record.source_run_id,
    )


def _error_code_for_stage(stage: str, error: Exception) -> str:
    """Return the stable runtime error code for one failed stage.

    Args:
        stage: The stage during which the error occurred (e.g., "prepare" or "check").
        error: The exception raised during execution.

    Returns:
        A stable error code string that can be used for error categorization and handling.
    """
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    return f"{stage}_execution_error"


def _error_details(error: Exception) -> dict[str, str] | None:
    """Convert structured workflow error details into result-contract shape.

    Args:
        error: The exception raised during execution, which may have a
            ``details`` attribute containing structured information.

    Returns:
        A mapping of error detail keys to string values, or None if no details are available.
    """
    details = getattr(error, "details", ())
    if not details:
        return None
    normalized = {key: str(value) for key, value in details if value is not None}
    return normalized or None


def _build_terminal_failed_monitoring_run_rerun_error(
    *,
    subject_id: str,
    monitoring_run_id: str,
) -> TerminalRunRetryError:
    """Build a deterministic error for duplicate requests targeting failed runs.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        monitoring_run_id: The ID of the monitoring run.

    Returns:
        A deterministic error for duplicate requests targeting failed runs.
    """
    return TerminalRunRetryError(
        code="idempotent_run_retry_failed_terminal",
        message=(
            f"Cannot retry monitoring run {monitoring_run_id} for subject_id={subject_id}: "
            "the idempotent run is already in terminal FAILED state."
        ),
        details=(
            ("subject_id", subject_id),
            ("monitoring_run_id", monitoring_run_id),
        ),
    )


def _validate_checked_monitoring_run_rerun_inputs(
    *,
    subject_id: str,
    baseline_source_run_id: str | None,
    source_experiment: str | None,
    gateway: MonitoringGateway,
) -> PrepareStageError | None:
    """Validate caller-controlled checked-rerun inputs without re-running prepare."""
    timeline_state = gateway.get_timeline_state(subject_id)
    if timeline_state is None:
        return PrepareStageError(
            code="prepare_timeline_initialization_failed",
            message=(
                f"Timeline initialization did not materialize state for subject_id={subject_id}."
            ),
            details=(("subject_id", subject_id),),
        )

    if baseline_source_run_id is None:
        return None

    resolved_baseline_source_run_id = gateway.resolve_source_run_id(
        subject_id=subject_id,
        source_experiment=source_experiment,
        source_run_id=baseline_source_run_id,
    )
    if resolved_baseline_source_run_id == timeline_state.baseline_source_run_id:
        return None

    return PrepareStageError(
        code="prepare_baseline_override_existing_timeline",
        message=(
            f"Provided baseline_source_run_id={baseline_source_run_id!r} "
            f"with resolved_baseline_source_run_id={resolved_baseline_source_run_id!r} "
            "does not match existing timeline pinned "
            f"baseline_source_run_id={timeline_state.baseline_source_run_id!r} "
            f"for subject_id={subject_id}. Overriding an existing timeline's baseline "
            "is not allowed."
        ),
        details=(
            ("subject_id", subject_id),
            ("baseline_source_run_id", baseline_source_run_id),
        ),
    )
