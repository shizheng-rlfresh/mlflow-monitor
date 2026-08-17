"""Invariant checks for the MLflow-Monitor runtime."""

from collections.abc import Mapping, Sequence

from mlflow_monitor.contract_checker import CONTRACT_CHECK_REASON_MESSAGE
from mlflow_monitor.domain import (
    CONTRACT_CHECK_REASON_CODE_BLOCKING,
    ComparabilityStatus,
    CompatibilityEvidence,
    ContractCheckReason,
    ContractCheckReasonCode,
    ContractCheckResult,
    Diff,
    Finding,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.identity import (
    make_compatibility_evidence_id,
    make_diff_id,
    make_finding_id,
)


def validate_finding_evidence(
    finding: Finding,
    *,
    diffs: Sequence[Diff],
    compatibility_evidence: Sequence[CompatibilityEvidence],
) -> None:
    """Validate every evidence identity referenced by one Finding.

    Args:
        finding: Finding whose evidence references should be validated.
        diffs: Complete Diff records available to the Finding.
        compatibility_evidence: Complete Compatibility Evidence records available
            to the Finding.

    Raises:
        InvariantViolation: If supplied evidence is inconsistent, a referenced ID
            is missing, or referenced evidence belongs to another current pair.

    Returns:
        None if every evidence reference is valid.
    """
    validate_diff_identity_consistency(diffs)
    validate_compatibility_evidence_identity_consistency(compatibility_evidence)

    diffs_by_id = {diff.diff_id: diff for diff in diffs}
    compatibility_evidence_by_id = {
        evidence.compatibility_evidence_id: evidence for evidence in compatibility_evidence
    }

    validate_finding_evidence_references(
        finding,
        diffs_by_id=diffs_by_id,
        compatibility_evidence_by_id=compatibility_evidence_by_id,
    )


def validate_finding_evidence_references(
    finding: Finding,
    *,
    diffs_by_id: Mapping[str, Diff],
    compatibility_evidence_by_id: Mapping[str, CompatibilityEvidence],
) -> None:
    """Validate Finding references against prevalidated evidence indexes.

    Args:
        finding: Finding whose evidence references should be validated.
        diffs_by_id: Identity-consistent Diff records indexed by identity.
        compatibility_evidence_by_id: Identity-consistent Compatibility Evidence
            records indexed by identity.

    Raises:
        InvariantViolation: If a referenced identity is missing or its evidence
            belongs to another current pair.

    Returns:
        None if every evidence reference is valid.
    """
    for diff_id in finding.evidence_diff_ids:
        diff = diffs_by_id.get(diff_id)
        if diff is None:
            raise InvariantViolation(
                code="finding_diff_evidence_violation",
                message=f"Finding references unknown Diff identity {diff_id!r}.",
                entity="Finding",
                field="evidence_diff_ids",
            )
        _validate_finding_evidence_pair(
            finding=finding,
            evidence=diff,
            entity="Diff",
            code="finding_diff_evidence_violation",
        )

    for compatibility_evidence_id in finding.evidence_compatibility_ids:
        evidence = compatibility_evidence_by_id.get(compatibility_evidence_id)
        if evidence is None:
            raise InvariantViolation(
                code="finding_compatibility_evidence_violation",
                message=(
                    "Finding references unknown Compatibility Evidence identity "
                    f"{compatibility_evidence_id!r}."
                ),
                entity="Finding",
                field="evidence_compatibility_ids",
            )
        _validate_finding_evidence_pair(
            finding=finding,
            evidence=evidence,
            entity="CompatibilityEvidence",
            code="finding_compatibility_evidence_violation",
        )


def _validate_finding_evidence_pair(
    *,
    finding: Finding,
    evidence: Diff | CompatibilityEvidence,
    entity: str,
    code: str,
) -> None:
    """Validate one evidence record against its Finding's current pair."""
    for field_name in ("monitoring_run_id", "source_run_id"):
        finding_value = getattr(finding, field_name)
        evidence_value = getattr(evidence, field_name)
        if evidence_value != finding_value:
            raise InvariantViolation(
                code=code,
                message=(
                    f"{entity} {field_name} {evidence_value!r} does not match "
                    f"Finding {field_name} {finding_value!r}."
                ),
                entity=entity,
                field=field_name,
            )


def validate_finding_identity(finding: Finding) -> None:
    """Validate that a Finding carries its derived deterministic identity.

    Args:
        finding: Finding whose identity should be validated.

    Raises:
        InvariantViolation: If the supplied Finding identity is not canonical.

    Returns:
        None if the Finding identity is canonical.
    """
    expected_finding_id = make_finding_id(
        monitoring_run_id=finding.monitoring_run_id,
        source_run_id=finding.source_run_id,
        finding_policy_id=finding.finding_policy_id,
        finding_policy_version=finding.finding_policy_version,
        finding_rule_id=finding.finding_rule_id,
        evidence_diff_ids=finding.evidence_diff_ids,
        evidence_compatibility_ids=finding.evidence_compatibility_ids,
    )
    if finding.finding_id != expected_finding_id:
        raise InvariantViolation(
            code="finding_identity_mismatch",
            message=f"Finding identity {finding.finding_id!r} does not match its derived identity.",
            entity="Finding",
            field="finding_id",
        )


def validate_finding_identity_consistency(findings: Sequence[Finding]) -> None:
    """Reject conflicting Finding content under one deterministic identity.

    Args:
        findings: Finding records to validate together.

    Raises:
        InvariantViolation: If an identity is invalid or maps to different content.

    Returns:
        None if all Finding identities and content are consistent.
    """
    findings_by_id: dict[str, Finding] = {}
    for finding in findings:
        validate_finding_identity(finding)
        existing = findings_by_id.get(finding.finding_id)
        if existing is not None and existing != finding:
            raise InvariantViolation(
                code="finding_identity_content_conflict",
                message=f"Finding identity {finding.finding_id!r} maps to conflicting content.",
                entity="Finding",
                field="finding_id",
            )
        findings_by_id[finding.finding_id] = finding


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
