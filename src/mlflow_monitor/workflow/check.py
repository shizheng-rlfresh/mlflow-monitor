"""Check stage module for mlflow-monitor v0."""

from __future__ import annotations

from collections.abc import Mapping

from mlflow_monitor.contract_checker import ContractChecker, make_contract_evaluation_context
from mlflow_monitor.domain import ComparabilityStatus, ContractCheckReason, ContractCheckResult
from mlflow_monitor.errors import CheckStageError, GatewayConsistencyViolation, InvariantViolation
from mlflow_monitor.gateway import MonitoringGateway
from mlflow_monitor.invariant import validate_contract_check_result

from .prepared_context import PreparedContext

CONTRACT_CHECK_ARTIFACT_PATH = "outputs/contract_check.json"

_CONTRACT_CHECK_ARTIFACT_SCHEMA_VERSION = "v0"
_CONTRACT_CHECK_FIELDS = frozenset(
    {
        "artifact_schema_version",
        "monitoring_run_id",
        "source_run_id",
        "contract_id",
        "contract_version",
        "status",
        "reasons",
    }
)
_CONTRACT_CHECK_REASON_FIELDS = frozenset({"code", "message", "blocking"})


def contract_check_result_to_dict(
    context: PreparedContext,
    result: ContractCheckResult,
) -> dict[str, object]:
    """Serialize one complete Contract Check result for durable persistence.

    Args:
        context: Committed prepared context that owns the Check result.
        result: Validated Contract Check result to persist.

    Returns:
        JSON-compatible Contract Check artifact content.
    """
    return {
        "artifact_schema_version": _CONTRACT_CHECK_ARTIFACT_SCHEMA_VERSION,
        "monitoring_run_id": context.monitoring_run_id,
        "source_run_id": context.source_run_id,
        "contract_id": context.contract.contract_id,
        "contract_version": context.contract.contract_version,
        "status": result.status.value,
        "reasons": [
            {
                "code": reason.code,
                "message": reason.message,
                "blocking": reason.blocking,
            }
            for reason in result.reasons
        ],
    }


def hydrate_contract_check_result(
    raw: Mapping[str, object] | None,
    *,
    prepared_context: PreparedContext,
    projected_comparability_status: ComparabilityStatus | None,
) -> ContractCheckResult:
    """Hydrate and validate one persisted Contract Check result.

    Args:
        raw: Decoded Contract Check artifact, or ``None`` when missing.
        prepared_context: Committed prepared context that owns the artifact.
        projected_comparability_status: Optional metadata projection to validate.

    Returns:
        Complete validated Contract Check result with original reason ordering.

    Raises:
        GatewayConsistencyViolation: If persisted content or its projection is
            missing, malformed, or inconsistent with committed prepared state.
    """
    error = _contract_check_artifact_inconsistent(prepared_context.monitoring_run_id)
    if raw is None or set(raw) != _CONTRACT_CHECK_FIELDS:
        raise error
    if raw.get("artifact_schema_version") != _CONTRACT_CHECK_ARTIFACT_SCHEMA_VERSION:
        raise error

    expected_identity = {
        "monitoring_run_id": prepared_context.monitoring_run_id,
        "source_run_id": prepared_context.source_run_id,
        "contract_id": prepared_context.contract.contract_id,
        "contract_version": prepared_context.contract.contract_version,
    }
    if any(raw.get(field) != expected for field, expected in expected_identity.items()):
        raise error

    raw_status = raw.get("status")
    raw_reasons = raw.get("reasons")
    if not isinstance(raw_status, str) or not isinstance(raw_reasons, list):
        raise error

    try:
        status = ComparabilityStatus(raw_status)
    except ValueError as exc:
        raise error from exc

    reasons: list[ContractCheckReason] = []
    for item in raw_reasons:
        if not isinstance(item, Mapping) or set(item) != _CONTRACT_CHECK_REASON_FIELDS:
            raise error
        code = item.get("code")
        message = item.get("message")
        blocking = item.get("blocking")
        if (
            not isinstance(code, str)
            or not isinstance(message, str)
            or not isinstance(blocking, bool)
        ):
            raise error
        reasons.append(
            ContractCheckReason(
                code=code,
                message=message,
                blocking=blocking,
            )
        )

    result = ContractCheckResult(status=status, reasons=tuple(reasons))
    try:
        validate_contract_check_result(result)
    except InvariantViolation as exc:
        raise error from exc
    if (
        projected_comparability_status is not None
        and projected_comparability_status is not result.status
    ):
        raise error
    return result


def _contract_check_artifact_inconsistent(
    monitoring_run_id: str,
) -> GatewayConsistencyViolation:
    """Build the bounded consistency error for Contract Check persistence."""
    return GatewayConsistencyViolation.monitoring_run_json_artifact_inconsistent(
        monitoring_run_id=monitoring_run_id,
        path=CONTRACT_CHECK_ARTIFACT_PATH,
    )


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
