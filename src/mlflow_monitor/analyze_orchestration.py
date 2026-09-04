"""Internal Analyze stage commit; the public facade still stops after Check."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlflow_monitor.domain import LifecycleStatus
from mlflow_monitor.errors import GatewayConsistencyViolation
from mlflow_monitor.gateway.models import MonitoringRunRecord
from mlflow_monitor.gateway.protocol import MonitoringGateway
from mlflow_monitor.utils import canonical_json
from mlflow_monitor.workflow import (
    CONTRACT_CHECK_ARTIFACT_PATH,
    PREPARED_CONTEXT_ARTIFACT_PATH,
    hydrate_contract_check_result,
    hydrate_prepared_context,
)
from mlflow_monitor.workflow.analyze import execute_analyze
from mlflow_monitor.workflow.analyze_artifacts import (
    ANALYZE_ARTIFACT_PATHS,
    AnalyzeOutput,
    analyze_output_to_artifacts,
)
from mlflow_monitor.workflow.analyze_hydration import (
    hydrate_analyze_output,
    validate_partial_analyze_artifacts,
)

if TYPE_CHECKING:
    from mlflow_monitor.orchestration import OrchestrationState


def commit_analyze_stage(*, state: OrchestrationState, gateway: MonitoringGateway) -> AnalyzeOutput:
    """Commit or replay Analyze from an allocated, checked Monitoring Run.

    Args:
        state: Allocated identity and executable Recipe; saved Prepare and Check
            artifacts, not transient caller references, supply stage inputs.
        gateway: Monitoring-owned artifact and metadata persistence boundary.

    Returns:
        Complete validated Analyze output, computed once or hydrated on replay.

    Raises:
        GatewayConsistencyViolation: If lifecycle, owner, artifacts, or partial
            recovery disagree. Existing conflicting artifacts are never replaced.
        PreparedContextConsistencyViolation: If saved Prepare inputs disagree
            with the allocated identity or executable Recipe.
        AnalyzeStageError: If the current source is missing or policy execution
            fails. No terminal failure result or lifecycle marker is written.

    Notes:
        All computation precedes writes. Artifacts are written in dependency
        order and read back before the analyzed marker. This is write-last
        recovery, not a transactional compare-and-swap or multi-writer lock.
    """
    owner = _read_owner(state, gateway)
    context = hydrate_prepared_context(
        gateway.read_monitoring_run_json_artifact(
            monitoring_run_id=state.monitoring_run_id, path=PREPARED_CONTEXT_ARTIFACT_PATH
        ),
        compiled_recipe=state.compiled_recipe,
        monitoring_run_id=state.monitoring_run_id,
        source_run_id=state.source_run_id,
        subject_id=state.subject_id,
        timeline_id=state.timeline_id,
        sequence_index=state.sequence_index,
    )
    check = hydrate_contract_check_result(
        gateway.read_monitoring_run_json_artifact(
            monitoring_run_id=state.monitoring_run_id, path=CONTRACT_CHECK_ARTIFACT_PATH
        ),
        prepared_context=context,
        projected_comparability_status=owner.comparability_status,
    )
    if (
        owner.comparability_status is not check.status
        or owner.references != context.references
        or (owner.contract_check_result is not None and owner.contract_check_result != check)
    ):
        raise _artifact_error(state, CONTRACT_CHECK_ARTIFACT_PATH)
    existing = _read_artifacts(state, gateway)
    if owner.lifecycle_status is LifecycleStatus.ANALYZED:
        return hydrate_analyze_output(
            existing, prepared_context=context, contract_check_result=check
        )
    validate_partial_analyze_artifacts(
        existing, prepared_context=context, contract_check_result=check
    )
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=check,
        compiled_recipe=state.compiled_recipe,
        gateway=gateway,
    )
    proposed = analyze_output_to_artifacts(
        output, prepared_context=context, contract_check_result=check
    )
    # Compare the entire proposed set before filling even the first missing path.
    for path, raw in existing.items():
        if canonical_json(raw) != canonical_json(proposed[path]):
            raise _artifact_error(state, path)
    for path, payload in proposed.items():
        _require_unchanged_checked_owner(state, gateway, owner)
        if path not in existing:
            gateway.write_monitoring_run_json_artifact(
                monitoring_run_id=state.monitoring_run_id, data=payload, path=path
            )
    saved = _read_artifacts(state, gateway)
    hydrated = hydrate_analyze_output(saved, prepared_context=context, contract_check_result=check)
    for path in ANALYZE_ARTIFACT_PATHS:
        if canonical_json(saved[path]) != canonical_json(proposed[path]):
            raise _artifact_error(state, path)
    _require_unchanged_checked_owner(state, gateway, owner)
    gateway.upsert_monitoring_run(
        subject_id=state.subject_id,
        monitoring_run_id=state.monitoring_run_id,
        source_run_id=state.source_run_id,
        sequence_index=state.sequence_index,
        lifecycle_status=LifecycleStatus.ANALYZED,
    )
    return hydrated


def _read_artifacts(
    state: OrchestrationState, gateway: MonitoringGateway
) -> dict[str, dict[str, object]]:
    """Read only Analyze paths, preserving missing artifacts as absent entries."""
    present = {}
    for path in ANALYZE_ARTIFACT_PATHS:
        raw = gateway.read_monitoring_run_json_artifact(
            monitoring_run_id=state.monitoring_run_id, path=path
        )
        if raw is not None:
            present[path] = raw
    return present


def _read_owner(state: OrchestrationState, gateway: MonitoringGateway) -> MonitoringRunRecord:
    """Reject foreign allocation identities and non-Analyze lifecycle states."""
    owner = gateway.get_monitoring_run(state.subject_id, state.monitoring_run_id)
    if (
        owner is None
        or owner.monitoring_run_id != state.monitoring_run_id
        or owner.source_run_id != state.source_run_id
        or owner.sequence_index != state.sequence_index
        or owner.lifecycle_status not in {LifecycleStatus.CHECKED, LifecycleStatus.ANALYZED}
    ):
        raise GatewayConsistencyViolation.monitoring_run_upsert_field_override(
            fields=(("lifecycle_status", LifecycleStatus.ANALYZED.value),)
        )
    return owner


def _require_unchanged_checked_owner(
    state: OrchestrationState, gateway: MonitoringGateway, expected: MonitoringRunRecord
) -> None:
    """Fail closed if the stage or checked projections changed during execution."""
    owner = _read_owner(state, gateway)
    if owner.lifecycle_status is not LifecycleStatus.CHECKED or owner != expected:
        raise GatewayConsistencyViolation.monitoring_run_upsert_field_override(
            fields=(("lifecycle_status", LifecycleStatus.ANALYZED.value),)
        )


def _artifact_error(state: OrchestrationState, path: str) -> GatewayConsistencyViolation:
    """Build a bounded error without exposing saved or recomputed payloads."""
    return GatewayConsistencyViolation.monitoring_run_json_artifact_inconsistent(
        monitoring_run_id=state.monitoring_run_id, path=path
    )
