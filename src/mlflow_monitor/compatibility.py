"""Materialize and project Compatibility Evidence for MLflow-Monitor v0.

Use :func:`materialize_compatibility_evidence` with the committed prepared
context and its hydrated Contract Check result. Pass the returned records and
the same prepared context to :func:`compatibility_evidence_to_dict` to build the
canonical JSON payload. These helpers are pure: they do not read or write a
Gateway and do not advance the Monitoring Run lifecycle.

The ``artifact_schema_version`` and deterministic identity scheme are separate
version axes. The JSON projection uses artifact schema ``v0`` while evidence IDs
retain the ``v1`` identity scheme owned by :mod:`mlflow_monitor.identity`.
"""

from __future__ import annotations

from mlflow_monitor.domain import CompatibilityEvidence, ContractCheckResult
from mlflow_monitor.identity import make_compatibility_evidence_id
from mlflow_monitor.workflow.prepared_context import PreparedContext

# Artifact schemas and deterministic identities are versioned independently.
_COMPATIBILITY_EVIDENCE_ARTIFACT_SCHEMA_VERSION = "v0"


def materialize_compatibility_evidence(
    prepared_context: PreparedContext,
    contract_check_result: ContractCheckResult,
) -> tuple[CompatibilityEvidence, ...]:
    """Materialize identified evidence from a committed Contract Check result.

    Args:
        prepared_context: Committed prepared context that owns the Check result.
        contract_check_result: Committed Contract Check result already hydrated
            and validated against the same prepared context.

    Returns:
        Compatibility Evidence records in committed reason order.
    """
    return tuple(
        CompatibilityEvidence(
            compatibility_evidence_id=make_compatibility_evidence_id(
                monitoring_run_id=prepared_context.monitoring_run_id,
                source_run_id=prepared_context.source_run_id,
                baseline_source_run_id=prepared_context.baseline_source_run_id,
                contract_id=prepared_context.contract.contract_id,
                contract_version=prepared_context.contract.contract_version,
                reason_code=reason.code,
            ),
            monitoring_run_id=prepared_context.monitoring_run_id,
            source_run_id=prepared_context.source_run_id,
            baseline_source_run_id=prepared_context.baseline_source_run_id,
            contract_id=prepared_context.contract.contract_id,
            contract_version=prepared_context.contract.contract_version,
            reason=reason,
        )
        for reason in contract_check_result.reasons
    )


def compatibility_evidence_to_dict(
    prepared_context: PreparedContext,
    evidence_records: tuple[CompatibilityEvidence, ...],
) -> dict[str, object]:
    """Project Compatibility Evidence into its canonical JSON payload.

    The evidence records must have been materialized from the same prepared
    context supplied to this function.

    Args:
        prepared_context: Committed prepared context that owns the evidence.
        evidence_records: Compatibility Evidence records in committed reason
            order.

    Returns:
        JSON-compatible Compatibility Evidence artifact content.
    """
    return {
        "artifact_schema_version": _COMPATIBILITY_EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "monitoring_run_id": prepared_context.monitoring_run_id,
        "source_run_id": prepared_context.source_run_id,
        "baseline_source_run_id": prepared_context.baseline_source_run_id,
        "contract_id": prepared_context.contract.contract_id,
        "contract_version": prepared_context.contract.contract_version,
        "evidence": [
            {
                "compatibility_evidence_id": item.compatibility_evidence_id,
                "reason": {
                    "code": item.reason.code,
                    "message": item.reason.message,
                    "blocking": item.reason.blocking,
                },
            }
            for item in evidence_records
        ],
    }
