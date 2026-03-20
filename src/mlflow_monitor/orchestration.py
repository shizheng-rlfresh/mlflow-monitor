"""Internal execution orchestration for the M1 create/prepare/check slice."""

from __future__ import annotations

from collections.abc import Callable

from mlflow_monitor.contract import SYSTEM_DEFAULT_CONTRACT_ID, resolve_contract_v0
from mlflow_monitor.contract_checker import ContractChecker
from mlflow_monitor.domain import ContractCheckResult, LifecycleStatus
from mlflow_monitor.gateway import IdempotencyKey, MonitoringGateway
from mlflow_monitor.recipe import (
    SYSTEM_DEFAULT_RECIPE_ID,
    RecipeReferenceCatalog,
    resolve_recipe_v0_lite,
)
from mlflow_monitor.recipe_compiler import compile_recipe_v0_lite
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult
from mlflow_monitor.workflow import execute_contract_check, prepare_run_context


def _run_monitoring(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
    run_id_factory: Callable[[], str],
) -> MonitorRunResult:
    """Execute the M1 create/prepare/check slice for one monitoring request."""
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
        gateway.reserve_sequence_index(subject_id)
        if is_new_run
        else existing_run.sequence_index
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
    except Exception as exc:
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
        return _build_success_result(
            subject_id=subject_id,
            run_id=run_id,
            prepared_context=prepared_context,
            contract_check_result=existing_run.contract_check_result,
            gateway=gateway,
        )

    try:
        contract_check_result = execute_contract_check(
            prepared_context=prepared_context,
            gateway=gateway,
            contract_checker=contract_checker,
        )
    except Exception as exc:
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
    """Build the canonical success result for one checked monitoring run."""
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


def _build_failure_result(
    *,
    subject_id: str,
    run_id: str,
    stage: str,
    error: Exception,
    gateway: MonitoringGateway,
) -> MonitorRunResult:
    """Build the canonical failed result for a prepare/check execution error."""
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
    """Build the minimal reference-run mapping for the synchronous result."""
    reference_run_ids = {"baseline": prepared_context.baseline_source_run_id}
    if prepared_context.previous_run_id is not None:
        reference_run_ids["previous"] = prepared_context.previous_run_id
    if prepared_context.active_lkg_run_id is not None:
        reference_run_ids["lkg"] = prepared_context.active_lkg_run_id
    if prepared_context.custom_reference_run_id is not None:
        reference_run_ids["custom"] = prepared_context.custom_reference_run_id
    return reference_run_ids


def _error_code_for_stage(stage: str, error: Exception) -> str:
    """Return the stable runtime error code for one failed stage."""
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    return f"{stage}_execution_error"


def _error_details(error: Exception) -> dict[str, str] | None:
    """Convert structured workflow error details into result-contract shape."""
    details = getattr(error, "details", ())
    if not details:
        return None
    normalized = {
        key: str(value)
        for key, value in details
        if value is not None
    }
    return normalized or None
