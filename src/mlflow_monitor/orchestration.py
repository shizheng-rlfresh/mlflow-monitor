"""Internal execution orchestration for the M1 create/prepare/check slice."""

from __future__ import annotations

from collections.abc import Callable

from mlflow_monitor.builtins import SYSTEM_DEFAULT_CONTRACT_ID, SYSTEM_DEFAULT_RECIPE_ID
from mlflow_monitor.contract import resolve_contract_v0
from mlflow_monitor.contract_checker import ContractChecker
from mlflow_monitor.domain import ContractCheckResult, LifecycleStatus
from mlflow_monitor.errors import (
    CheckStageError,
    ContractResolutionError,
    PrepareStageError,
    RecipeValidationError,
    TerminalRunRetryError,
)
from mlflow_monitor.gateway import IdempotencyKey, MonitoringGateway, MonitoringRunRecord
from mlflow_monitor.recipe import (
    RecipeReferenceCatalog,
    resolve_recipe_v0_lite,
)
from mlflow_monitor.recipe_compiler import compile_recipe_v0_lite
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult
from mlflow_monitor.workflow import execute_contract_check, prepare_run_context

_OWNED_FAILURES = (
    PrepareStageError,
    CheckStageError,
    ContractResolutionError,
    RecipeValidationError,
)


def run_orchestration(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
    run_id_factory: Callable[[], str],
) -> MonitorRunResult:
    """Execute the orchestration for one monitoring run, including prepare and check stages.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        source_run_id: The original run ID from the training system that produced this run.
        baseline_source_run_id: The source run ID of the baseline this run is compared against
        gateway: The monitoring gateway to use for persistence during orchestration.
        contract_checker: The contract checker to use for executing the contract check stage.
        run_id_factory: A callable that produces new unique run IDs for monitoring runs.

    Returns:
        The result of the monitoring run execution, including comparability status and any findings.

    """
    references = RecipeReferenceCatalog(
        contract_ids=frozenset({SYSTEM_DEFAULT_CONTRACT_ID}),
        finding_policy_profiles=frozenset(),
        summary_modes=frozenset(),
    )
    recipe = resolve_recipe_v0_lite(None, references=references)
    compiled_plan = compile_recipe_v0_lite(recipe)
    resolved_contract = resolve_contract_v0(compiled_plan.contract.contract_id)
    idempotency_key = IdempotencyKey(
        subject_id=subject_id,
        source_run_id=source_run_id,
        recipe_id=SYSTEM_DEFAULT_RECIPE_ID,
        recipe_version=compiled_plan.identity.recipe_version,
    )
    run_id = gateway.get_or_create_idempotent_run_id(idempotency_key, run_id_factory)
    existing_run = gateway.get_monitoring_run(subject_id, run_id)
    is_new_run = existing_run is None
    sequence_index = (
        gateway.reserve_sequence_index(subject_id) if is_new_run else existing_run.sequence_index
    )

    if existing_run is not None and existing_run.lifecycle_status is LifecycleStatus.FAILED:
        return _build_failure_result(
            subject_id=subject_id,
            run_id=run_id,
            stage="prepare",
            error=_build_terminal_failed_rerun_error(subject_id=subject_id, run_id=run_id),
            gateway=gateway,
        )

    if existing_run is not None and existing_run.contract_check_result is not None:
        existing_check_result = existing_run.contract_check_result
        assert existing_check_result is not None
        replay_error = _validate_checked_rerun_inputs(
            subject_id=subject_id,
            baseline_source_run_id=baseline_source_run_id,
            source_experiment=compiled_plan.input.source_experiment,
            gateway=gateway,
        )
        if replay_error is not None:
            return _build_failure_result(
                subject_id=subject_id,
                run_id=run_id,
                stage="prepare",
                error=replay_error,
                gateway=gateway,
            )
        return _build_existing_checked_result(
            subject_id=subject_id,
            run_id=run_id,
            existing_run=existing_run,
            gateway=gateway,
        )

    if is_new_run:
        gateway.upsert_monitoring_run(
            subject_id=subject_id,
            run_id=run_id,
            lifecycle_status=LifecycleStatus.CREATED,
            sequence_index=sequence_index,
        )

    try:
        prepared_context = prepare_run_context(
            run_id=run_id,
            subject_id=subject_id,
            compiled_plan=compiled_plan,
            resolved_contract=resolved_contract,
            gateway=gateway,
            runtime_source_run_id=source_run_id,
            baseline_source_run_id=baseline_source_run_id,
        )
    except _OWNED_FAILURES as exc:
        gateway.upsert_monitoring_run(
            subject_id=subject_id,
            run_id=run_id,
            lifecycle_status=LifecycleStatus.FAILED,
            sequence_index=sequence_index,
        )
        return _build_failure_result(
            subject_id=subject_id,
            run_id=run_id,
            stage="prepare",
            error=exc,
            gateway=gateway,
        )

    if existing_run is None or existing_run.lifecycle_status is LifecycleStatus.CREATED:
        gateway.upsert_monitoring_run(
            subject_id=subject_id,
            run_id=run_id,
            lifecycle_status=LifecycleStatus.PREPARED,
            sequence_index=sequence_index,
        )

    existing_run = gateway.get_monitoring_run(subject_id, run_id)
    if existing_run is not None and existing_run.contract_check_result is not None:
        return _build_existing_checked_result(
            subject_id=subject_id,
            run_id=run_id,
            existing_run=existing_run,
            gateway=gateway,
        )

    try:
        contract_check_result = execute_contract_check(
            prepared_context=prepared_context,
            gateway=gateway,
            contract_checker=contract_checker,
        )
    except _OWNED_FAILURES as exc:
        gateway.upsert_monitoring_run(
            subject_id=subject_id,
            run_id=run_id,
            lifecycle_status=LifecycleStatus.FAILED,
            sequence_index=sequence_index,
        )
        return _build_failure_result(
            subject_id=subject_id,
            run_id=run_id,
            stage="check",
            error=exc,
            gateway=gateway,
        )

    gateway.upsert_monitoring_run(
        subject_id=subject_id,
        run_id=run_id,
        lifecycle_status=LifecycleStatus.CHECKED,
        sequence_index=sequence_index,
        contract_check_result=contract_check_result,
        reference_run_ids=_build_reference_run_ids(prepared_context),
    )
    return _build_success_result(
        subject_id=subject_id,
        run_id=run_id,
        prepared_context=prepared_context,
        contract_check_result=contract_check_result,
        gateway=gateway,
    )


def _build_success_result(
    *,
    subject_id: str,
    run_id: str,
    prepared_context,
    contract_check_result: ContractCheckResult,
    gateway: MonitoringGateway,
) -> MonitorRunResult:
    """Build the canonical success result for one checked monitoring run.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        run_id: The ID of the monitoring run.
        prepared_context: The prepared context produced by the prepare stage for this run.
        contract_check_result: The result of the contract check stage for this run.
        gateway: The monitoring gateway to use for retrieving any additional information needed.

    Returns:
        The canonical success result for this run, including comparability status and any findings.
    """
    timeline_state = gateway.get_timeline_state(subject_id)
    return MonitorRunResult(
        run_id=run_id,
        subject_id=subject_id,
        timeline_id=None if timeline_state is None else timeline_state.timeline_id,
        lifecycle_status=LifecycleStatus.CHECKED,
        comparability_status=contract_check_result.status,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        reference_run_ids=_build_reference_run_ids(prepared_context),
        error=None,
    )


def _build_existing_checked_result(
    *,
    subject_id: str,
    run_id: str,
    existing_run: MonitoringRunRecord,
    gateway: MonitoringGateway,
) -> MonitorRunResult:
    """Build a success result for an already checked idempotent run.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        run_id: The ID of the monitoring run.
        existing_run: The previously persisted monitoring run record for this run.
        gateway: The monitoring gateway to use for retrieving timeline information.

    Returns:
        The canonical success result for an already checked run.
    """
    timeline_state = gateway.get_timeline_state(subject_id)
    return MonitorRunResult(
        run_id=run_id,
        subject_id=subject_id,
        timeline_id=None if timeline_state is None else timeline_state.timeline_id,
        lifecycle_status=LifecycleStatus.CHECKED,
        comparability_status=existing_run.contract_check_result.status
        if existing_run.contract_check_result is not None
        else None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        reference_run_ids=existing_run.reference_run_ids,
        error=None,
    )


def _build_failure_result(
    *,
    subject_id: str,
    run_id: str,
    stage: str,
    error: Exception,
    gateway: MonitoringGateway,
) -> MonitorRunResult:
    """Build the canonical failed result for a prepare/check execution error.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        run_id: The ID of the monitoring run.
        stage: The stage during which the error occurred (e.g., "prepare" or "check").
        error: The exception raised during execution.
        gateway: The monitoring gateway to use for retrieving any additional information needed.

    Returns:
        The canonical failure result for this run, including error details.
    """
    timeline_state = gateway.get_timeline_state(subject_id)
    return MonitorRunResult(
        run_id=run_id,
        subject_id=subject_id,
        timeline_id=None if timeline_state is None else timeline_state.timeline_id,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        reference_run_ids={},
        error=MonitorRunError(
            code=_error_code_for_stage(stage, error),
            message=str(error),
            stage=stage,
            details=_error_details(error),
        ),
    )


def _build_reference_run_ids(prepared_context) -> dict[str, str]:
    """Build the minimal reference-run mapping for the synchronous result.

    Args:
        prepared_context: The prepared context produced by the prepare stage for this run.

    Returns:
        A mapping of reference types to source run IDs that were used as references during
            contract check.
    """
    reference_run_ids = {"baseline": prepared_context.baseline_source_run_id}
    if prepared_context.previous_run_id is not None:
        reference_run_ids["previous"] = prepared_context.previous_run_id
    if prepared_context.active_lkg_run_id is not None:
        reference_run_ids["lkg"] = prepared_context.active_lkg_run_id
    if prepared_context.custom_reference_run_id is not None:
        reference_run_ids["custom"] = prepared_context.custom_reference_run_id
    return reference_run_ids


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


def _build_terminal_failed_rerun_error(
    *,
    subject_id: str,
    run_id: str,
) -> TerminalRunRetryError:
    """Build a deterministic error for duplicate requests targeting failed runs."""
    return TerminalRunRetryError(
        code="idempotent_run_retry_failed_terminal",
        message=(
            f"Cannot retry monitoring run {run_id} for subject_id={subject_id}: "
            "the idempotent run is already in terminal FAILED state."
        ),
        details=(
            ("subject_id", subject_id),
            ("run_id", run_id),
        ),
    )


def _validate_checked_rerun_inputs(
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
        run_selector=baseline_source_run_id,
    )
    if resolved_baseline_source_run_id == timeline_state.baseline_source_run_id:
        return None

    return PrepareStageError(
        code="prepare_baseline_override_existing_timeline",
        message=(
            f"Provided baseline_source_run_id={baseline_source_run_id!r} "
            f"with resolved_baseline_source_run_id={resolved_baseline_source_run_id!r} "
            "does not match existing timeline "
            f"baseline_source_run_id={timeline_state.baseline_source_run_id!r} "
            f"for subject_id={subject_id}. Overriding an existing timeline's baseline "
            "is not allowed."
        ),
        details=(
            ("subject_id", subject_id),
            ("baseline_source_run_id", baseline_source_run_id),
        ),
    )
