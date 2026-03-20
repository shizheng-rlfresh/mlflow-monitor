"""Monitor-layer orchestration helpers and public entrypoint stub."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from mlflow_monitor.contract_checker import ContractChecker
from mlflow_monitor.domain import ComparabilityStatus, Contract, LifecycleStatus, Run
from mlflow_monitor.errors import CheckStageError, InvalidRunTransition, PrepareStageError
from mlflow_monitor.gateway import MonitoringGateway
from mlflow_monitor.recipe_compiler import CompiledRunPlan
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult
from mlflow_monitor.workflow import (
    PreparedContext,
    execute_contract_check,
    prepare_run_context,
    transition_run,
)


def _build_created_run(
    *,
    run_id: str,
    timeline_id: str,
    sequence_index: int,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str,
    contract: Contract,
) -> Run:
    """Build the canonical created run value for monitor orchestration."""
    return Run(
        run_id=run_id,
        timeline_id=timeline_id,
        sequence_index=sequence_index,
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        contract=contract,
        lifecycle_status=LifecycleStatus.CREATED,
        comparability_status=ComparabilityStatus.PASS,
        contract_check_result=None,
        diff_ids=(),
        finding_ids=(),
    )


def _prepare_stage(
    *,
    run_id: str,
    subject_id: str,
    compiled_plan: CompiledRunPlan,
    resolved_contract: Contract,
    gateway: MonitoringGateway,
    runtime_source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> tuple[Run, PreparedContext]:
    """Resolve prepare context, transition to prepared, and persist the run."""
    prepared_context = prepare_run_context(
        run_id=run_id,
        subject_id=subject_id,
        compiled_plan=compiled_plan,
        resolved_contract=resolved_contract,
        gateway=gateway,
        runtime_source_run_id=runtime_source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    sequence_index = gateway.reserve_sequence_index(subject_id)
    created_run = _build_created_run(
        run_id=run_id,
        timeline_id=prepared_context.timeline_id,
        sequence_index=sequence_index,
        subject_id=subject_id,
        source_run_id=prepared_context.source_run_id,
        baseline_source_run_id=prepared_context.baseline_source_run_id,
        contract=prepared_context.contract,
    )
    prepared_run = transition_run(created_run, LifecycleStatus.PREPARED)
    gateway.upsert_monitoring_run(
        subject_id=prepared_run.subject_id,
        run_id=prepared_run.run_id,
        lifecycle_status=prepared_run.lifecycle_status,
        sequence_index=prepared_run.sequence_index,
    )
    return prepared_run, prepared_context


def _check_stage(
    *,
    prepared_run: Run,
    prepared_context: PreparedContext,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
) -> Run:
    """Execute check stage, transition to checked, and persist the run."""
    result = execute_contract_check(
        prepared_context=prepared_context,
        gateway=gateway,
        contract_checker=contract_checker,
    )
    run_with_result = replace(
        prepared_run,
        comparability_status=result.status,
        contract_check_result=result,
    )
    checked_run = transition_run(run_with_result, LifecycleStatus.CHECKED)
    gateway.upsert_monitoring_run(
        subject_id=checked_run.subject_id,
        run_id=checked_run.run_id,
        lifecycle_status=checked_run.lifecycle_status,
        sequence_index=checked_run.sequence_index,
        contract_check_result=checked_run.contract_check_result,
    )
    return checked_run


def _run_prepare_and_check(
    *,
    run_id: str,
    subject_id: str,
    compiled_plan: CompiledRunPlan,
    resolved_contract: Contract,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
    runtime_source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> Run:
    """Compose prepare and check orchestration through checked state."""
    prepared_run, prepared_context = _prepare_stage(
        run_id=run_id,
        subject_id=subject_id,
        compiled_plan=compiled_plan,
        resolved_contract=resolved_contract,
        gateway=gateway,
        runtime_source_run_id=runtime_source_run_id,
        baseline_source_run_id=baseline_source_run_id,
    )
    return _check_stage(
        prepared_run=prepared_run,
        prepared_context=prepared_context,
        gateway=gateway,
        contract_checker=contract_checker,
    )


def _checked_run_to_result(checked_run: Run) -> MonitorRunResult:
    """Convert a checked run into the public synchronous result contract."""
    return MonitorRunResult(
        run_id=checked_run.run_id,
        subject_id=checked_run.subject_id,
        timeline_id=checked_run.timeline_id,
        lifecycle_status=checked_run.lifecycle_status,
        comparability_status=checked_run.comparability_status,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        reference_run_ids={},
        error=None,
    )


def _error_details_to_mapping(
    details: tuple[tuple[str, str | int | None], ...],
) -> Mapping[str, str] | None:
    """Convert workflow error details into string-only result-contract details."""
    if not details:
        return None
    return {
        key: "" if value is None else str(value)
        for key, value in details
    }


def run(
    *,
    run_id: str,
    subject_id: str,
    compiled_plan: CompiledRunPlan,
    resolved_contract: Contract,
    gateway: MonitoringGateway,
    contract_checker: ContractChecker,
    runtime_source_run_id: str | None = None,
    baseline_source_run_id: str | None = None,
) -> MonitorRunResult:
    """Execute monitor orchestration through checked state and return a result."""
    try:
        checked_run = _run_prepare_and_check(
            run_id=run_id,
            subject_id=subject_id,
            compiled_plan=compiled_plan,
            resolved_contract=resolved_contract,
            gateway=gateway,
            contract_checker=contract_checker,
            runtime_source_run_id=runtime_source_run_id,
            baseline_source_run_id=baseline_source_run_id,
        )
    except PrepareStageError as exc:
        return MonitorRunResult(
            run_id=run_id,
            subject_id=subject_id,
            timeline_id=None,
            lifecycle_status=LifecycleStatus.FAILED,
            comparability_status=None,
            summary=None,
            finding_ids=(),
            diff_ids=(),
            reference_run_ids={},
            error=MonitorRunError(
                code=exc.code,
                message=exc.message,
                stage="prepare",
                details=_error_details_to_mapping(exc.details),
            ),
        )
    except CheckStageError as exc:
        return MonitorRunResult(
            run_id=run_id,
            subject_id=subject_id,
            timeline_id=None,
            lifecycle_status=LifecycleStatus.FAILED,
            comparability_status=None,
            summary=None,
            finding_ids=(),
            diff_ids=(),
            reference_run_ids={},
            error=MonitorRunError(
                code=exc.code,
                message=exc.message,
                stage="check",
                details=_error_details_to_mapping(exc.details),
            ),
        )
    except InvalidRunTransition as exc:
        return MonitorRunResult(
            run_id=run_id,
            subject_id=subject_id,
            timeline_id=None,
            lifecycle_status=LifecycleStatus.FAILED,
            comparability_status=None,
            summary=None,
            finding_ids=(),
            diff_ids=(),
            reference_run_ids={},
            error=MonitorRunError(
                code="invalid_run_transition",
                message=exc.message,
                stage="check",
                details={
                    "from_status": str(exc.from_status),
                    "to_status": str(exc.to_status),
                },
            ),
        )

    return _checked_run_to_result(checked_run)
