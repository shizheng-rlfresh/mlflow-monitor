"""Invariant checks for the MLflow-Monitor runtime."""

from collections.abc import Sequence

from mlflow_monitor.contract_checker import CONTRACT_CHECK_REASON_MESSAGE
from mlflow_monitor.domain import (
    CONTRACT_CHECK_REASON_CODE_BLOCKING,
    LKG,
    Baseline,
    ComparabilityStatus,
    CompatibilityEvidence,
    ContractCheckReason,
    ContractCheckReasonCode,
    ContractCheckResult,
    Diff,
    Finding,
    Run,
    Timeline,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.identity import make_compatibility_evidence_id, make_diff_id


def validate_timeline_ownership(
    timeline: Timeline,
    baseline: Baseline | None = None,
    lkg: LKG | None = None,
    runs: Sequence[Run] | None = None,
) -> None:
    """Validate that all provided records are owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        baseline: Optional baseline record to check ownership of.
        lkg: Optional last-known-good record to check ownership of.
        runs: Optional sequence of run records to check ownership of.

    Raises:
        InvariantViolation: If any record is found to violate timeline ownership.

    Returns:
        None if all records are valid.
    """
    if baseline is not None:
        _validate_baseline_ownership(timeline, baseline)
    if lkg is not None:
        _validate_lkg_ownership(timeline, lkg)
    if runs is not None:
        for run in runs:
            _validate_run_ownership(timeline, run)
    return None


def validate_baseline_immutability(
    baseline: Baseline,
    proposed_baseline: Baseline,
) -> None:
    """Validate that the baseline record has not changed since timeline creation.

    Args:
        baseline: The baseline record to validate.
        proposed_baseline: The new baseline record to compare against the existing one.

    Raises:
        InvariantViolation: If the baseline is found to have changed since timeline creation.

    Returns:
        None if the baseline is valid.
    """
    if baseline != proposed_baseline:
        fields = []
        if baseline.timeline_id != proposed_baseline.timeline_id:
            fields.append("timeline_id")
        if baseline.source_run_id != proposed_baseline.source_run_id:
            fields.append("source_run_id")
        if baseline.model_identity != proposed_baseline.model_identity:
            fields.append("model_identity")
        if baseline.parameter_fingerprint != proposed_baseline.parameter_fingerprint:
            fields.append("parameter_fingerprint")
        if baseline.data_snapshot_ref != proposed_baseline.data_snapshot_ref:
            fields.append("data_snapshot_ref")
        if baseline.run_config_ref != proposed_baseline.run_config_ref:
            fields.append("run_config_ref")
        if baseline.metric_snapshot != proposed_baseline.metric_snapshot:
            fields.append("metric_snapshot")
        if baseline.environment_context != proposed_baseline.environment_context:
            fields.append("environment_context")

        raise InvariantViolation(
            code="baseline_immutability_violation",
            message=f"Proposed Baseline {proposed_baseline} does not match existing Baseline {baseline}",  # noqa: E501
            entity="Baseline",
            field=", ".join(fields) if fields else None,
        )
    return None


def validate_lkg_membership(timeline: Timeline, lkg: LKG) -> None:
    """Validate that the LKG record belongs to the same timeline.

    Args:
        timeline: The timeline record to validate against.
        lkg: The LKG record to check membership of.

    Raises:
        InvariantViolation: If the LKG is found to not belong to the same timeline.

    Returns:
        None if the LKG is valid.
    """
    if lkg.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="lkg_membership_violation",
            message=f"LKG {lkg} does not belong to Timeline {timeline}",
            entity="LKG",
            field="timeline_id",
        )

    if lkg.monitoring_run_id not in timeline.monitoring_run_ids:
        raise InvariantViolation(
            code="lkg_membership_violation",
            message=f"LKG {lkg} does not belong to Timeline {timeline}",
            entity="LKG",
            field="monitoring_run_id",
        )

    if lkg.monitoring_run_id != timeline.active_lkg_monitoring_run_id:
        raise InvariantViolation(
            code="lkg_membership_violation",
            message=f"LKG {lkg} does not belong to Timeline {timeline}",
            entity="LKG",
            field="active_lkg_monitoring_run_id",
        )

    return None


def validate_finding_to_diff_evidence(finding: Finding, diff: Diff) -> None:
    """Validate that the provided diff record corresponds to one of the finding's evidence diff IDs.

    Args:
        finding: The finding record to validate.
        diff: The diff record to check for evidence membership.

    Raises:
        InvariantViolation: If any evidence diff ID does not correspond to a provided diff.

    Returns:
        None if all evidence diff IDs are valid.
    """
    evidence_diff_ids = [evidence_diff_id for evidence_diff_id in finding.evidence_diff_ids]

    if diff.diff_id not in evidence_diff_ids:
        raise InvariantViolation(
            code="finding_diff_evidence_violation",
            message=f"Diff diff_id {diff.diff_id} is not in Finding evidence {evidence_diff_ids}",
            entity="Diff",
            field="diff_id",
        )

    if diff.monitoring_run_id != finding.monitoring_run_id:
        raise InvariantViolation(
            code="finding_diff_evidence_violation",
            message=(
                f"Diff monitoring_run_id {diff.monitoring_run_id} does not "
                f"match Finding monitoring_run_id {finding.monitoring_run_id}"
            ),
            entity="Diff",
            field="monitoring_run_id",
        )

    return None


def validate_diff_identity(diff: Diff) -> None:
    """Validate that a Diff carries its derived deterministic identity.

    Args:
        diff: Diff whose identity should be validated.

    Raises:
        InvariantViolation: If the supplied Diff identity is not canonical.

    Returns:
        None if the Diff identity is canonical.
    """
    expected_diff_id = make_diff_id(
        monitoring_run_id=diff.monitoring_run_id,
        source_run_id=diff.source_run_id,
        reference=diff.reference,
        metric_name=diff.metric_name,
    )
    if diff.diff_id != expected_diff_id:
        raise InvariantViolation(
            code="diff_identity_mismatch",
            message=f"Diff identity {diff.diff_id!r} does not match its derived identity.",
            entity="Diff",
            field="diff_id",
        )


def validate_diff_identity_consistency(diffs: Sequence[Diff]) -> None:
    """Reject conflicting Diff content under one deterministic identity.

    Args:
        diffs: Diff records to validate together.

    Raises:
        InvariantViolation: If an identity is invalid or maps to different content.

    Returns:
        None if all Diff identities and content are consistent.
    """
    diffs_by_id: dict[str, Diff] = {}
    for diff in diffs:
        validate_diff_identity(diff)
        existing = diffs_by_id.get(diff.diff_id)
        if existing is not None and existing != diff:
            raise InvariantViolation(
                code="diff_identity_content_conflict",
                message=f"Diff identity {diff.diff_id!r} maps to conflicting content.",
                entity="Diff",
                field="diff_id",
            )
        diffs_by_id[diff.diff_id] = diff


def validate_compatibility_evidence_identity(evidence: CompatibilityEvidence) -> None:
    """Validate one Compatibility Evidence identity and reason.

    Args:
        evidence: Compatibility Evidence to validate.

    Raises:
        InvariantViolation: If its reason or supplied identity is invalid.

    Returns:
        None if the Compatibility Evidence identity and reason are valid.
    """
    _validate_compatibility_evidence_id(evidence)
    validate_contract_check_reason(evidence.reason)


def validate_compatibility_evidence_identity_consistency(
    evidence_records: Sequence[CompatibilityEvidence],
) -> None:
    """Reject conflicting Compatibility Evidence under one identity.

    Args:
        evidence_records: Compatibility Evidence records to validate together.

    Raises:
        InvariantViolation: If an identity is invalid or maps to different content.

    Returns:
        None if all Compatibility Evidence identities and content are consistent.
    """
    evidence_by_id: dict[str, CompatibilityEvidence] = {}
    for evidence in evidence_records:
        _validate_compatibility_evidence_id(evidence)
        existing = evidence_by_id.get(evidence.compatibility_evidence_id)
        if existing is not None and existing != evidence:
            raise InvariantViolation(
                code="compatibility_evidence_identity_content_conflict",
                message=(
                    "Compatibility Evidence identity "
                    f"{evidence.compatibility_evidence_id!r} maps to conflicting content."
                ),
                entity="CompatibilityEvidence",
                field="compatibility_evidence_id",
            )
        validate_contract_check_reason(evidence.reason)
        evidence_by_id[evidence.compatibility_evidence_id] = evidence


def _validate_compatibility_evidence_id(evidence: CompatibilityEvidence) -> None:
    """Validate only the deterministic portion of Compatibility Evidence."""
    expected_evidence_id = make_compatibility_evidence_id(
        monitoring_run_id=evidence.monitoring_run_id,
        source_run_id=evidence.source_run_id,
        baseline_source_run_id=evidence.baseline_source_run_id,
        contract_id=evidence.contract_id,
        contract_version=evidence.contract_version,
        reason_code=evidence.reason.code,
    )
    if evidence.compatibility_evidence_id != expected_evidence_id:
        raise InvariantViolation(
            code="compatibility_evidence_identity_mismatch",
            message=(
                "Compatibility Evidence identity "
                f"{evidence.compatibility_evidence_id!r} does not match its derived identity."
            ),
            entity="CompatibilityEvidence",
            field="compatibility_evidence_id",
        )


def validate_contract_check_result(result: ContractCheckResult) -> None:
    """Validate a contract-check result against the current reason taxonomy.

    Args:
        result: The contract-check result to validate.

    Raises:
        InvariantViolation: If the reason taxonomy or aggregate status is inconsistent.

    Returns:
        None if the contract-check result is valid.
    """
    reason_codes: set[str] = set(
        reason.code for reason in result.reasons if isinstance(reason.code, str)
    )
    if len(reason_codes) != len(result.reasons):
        raise InvariantViolation(
            code="contract_check_reason_code_duplicate",
            message="Contract check reason codes must be unique.",
            entity="ContractCheckResult",
            field="reasons",
        )
    for reason in result.reasons:
        validate_contract_check_reason(reason)

    _validate_contract_check_status(result)

    return None


def validate_contract_check_reason(reason: ContractCheckReason) -> None:
    """Validate a single contract-check reason against the current taxonomy.

    Args:
        reason: The contract-check reason to validate.

    Raises:
        InvariantViolation: If the reason code is unknown or its blocking flag is invalid.

    Returns:
        None if the reason is valid.
    """
    if reason.code not in ContractCheckReasonCode:
        raise InvariantViolation(
            code="contract_check_reason_code_unknown",
            message=f"Contract check reason code {reason.code!r} is not supported.",
            entity="ContractCheckReason",
            field="code",
        )

    reason_code = ContractCheckReasonCode(reason.code)
    expected_blocking = CONTRACT_CHECK_REASON_CODE_BLOCKING[reason_code]

    if reason.blocking != expected_blocking:
        raise InvariantViolation(
            code="contract_check_reason_blocking_mismatch",
            message=(
                f"Contract check reason {reason.code!r} must set blocking={expected_blocking}."
            ),
            entity="ContractCheckReason",
            field="blocking",
        )

    if reason.message != CONTRACT_CHECK_REASON_MESSAGE[reason_code]:
        raise InvariantViolation(
            code="contract_check_reason_message_mismatch",
            message=(
                f"Contract check reason {reason.code!r} must have message="
                f"{CONTRACT_CHECK_REASON_MESSAGE[reason_code]!r}."
            ),
            entity="ContractCheckReason",
            field="message",
        )

    return None


def _validate_contract_check_status(result: ContractCheckResult) -> None:
    """Validate that result status matches the blocking profile of its reasons.

    Args:
        result: The contract-check result to validate.

    Raises:
        InvariantViolation: If the status code and reasons do not align.

    Returns:
        None if the status code and reasons are aligned.
    """
    if result.status not in ComparabilityStatus:
        raise InvariantViolation(
            code="contract_check_status_unknown",
            message=f"Comparability status {result.status!r} is not supported.",
            entity="ContractCheckResult",
            field="status",
        )

    has_reasons = bool(result.reasons)
    has_blocking_reason = any(reason.blocking for reason in result.reasons)

    if result.status is ComparabilityStatus.PASS and has_reasons:
        raise InvariantViolation(
            code="contract_check_status_reason_mismatch",
            message="Comparability status 'pass' cannot include contract-check reasons.",
            entity="ContractCheckResult",
            field="status",
        )

    if result.status is ComparabilityStatus.WARN and has_blocking_reason:
        raise InvariantViolation(
            code="contract_check_status_reason_mismatch",
            message="Comparability status 'warn' cannot include blocking contract-check reasons.",
            entity="ContractCheckResult",
            field="status",
        )

    if result.status is ComparabilityStatus.WARN and not has_reasons:
        raise InvariantViolation(
            code="contract_check_status_reason_mismatch",
            message="Comparability status `warn` requires at least one non-blocking contract-check reason.",  # noqa: E501
            entity="ContractCheckResult",
            field="status",
        )

    if result.status is ComparabilityStatus.FAIL and not has_blocking_reason:
        raise InvariantViolation(
            code="contract_check_status_reason_mismatch",
            message=(
                "Comparability status 'fail' requires at least one blocking contract-check reason."
            ),
            entity="ContractCheckResult",
            field="status",
        )


def _validate_baseline_ownership(timeline: Timeline, baseline: Baseline) -> None:
    """Validate that the baseline record is owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        baseline: The baseline record to check ownership of.

    Raises:
        InvariantViolation: If the baseline is found to violate timeline ownership.

    Returns:
        None if the baseline is valid.
    """
    timeline_id = timeline.timeline_id

    if baseline.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="baseline_timeline_mismatch",
            message=f"Baseline {baseline.timeline_id} does not match Timeline {timeline_id}",
            entity="Baseline",
            field="timeline_id",
        )


def _validate_lkg_ownership(timeline: Timeline, lkg: LKG) -> None:
    """Validate that the LKG record is owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        lkg: The LKG record to check ownership of.

    Raises:
        InvariantViolation: If the LKG is found to violate timeline ownership.

    Returns:
        None if the LKG is valid.
    """
    if lkg.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="lkg_timeline_mismatch",
            message=f"LKG {lkg.timeline_id} does not match Timeline {timeline.timeline_id}",
            entity="LKG",
            field="timeline_id",
        )


def _validate_run_ownership(timeline: Timeline, run: Run) -> None:
    """Validate that the run record is owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        run: The run record to check ownership of.

    Raises:
        InvariantViolation: If the run is found to violate timeline ownership.

    Returns:
        None if the run is valid.
    """
    if run.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="run_timeline_mismatch",
            message=f"Run {run.timeline_id} does not match Timeline {timeline.timeline_id}",
            entity="Run",
            field="timeline_id",
        )
