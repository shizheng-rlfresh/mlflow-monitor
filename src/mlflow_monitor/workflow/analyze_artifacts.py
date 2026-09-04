"""Typed output and canonical artifacts for the internal Analyze stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mlflow_monitor.compatibility import (
    compatibility_evidence_to_dict,
    materialize_compatibility_evidence,
)
from mlflow_monitor.domain import (
    ComparabilityStatus,
    CompatibilityEvidence,
    ContractCheckResult,
    Diff,
    DiffReference,
    Finding,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
from mlflow_monitor.errors import GatewayConsistencyViolation, InvariantViolation
from mlflow_monitor.invariant import (
    validate_diff_identity_consistency,
    validate_finding_evidence_references,
    validate_finding_identity_consistency,
)

if TYPE_CHECKING:
    from .prepared_context import PreparedContext

COMPATIBILITY_EVIDENCE_ARTIFACT_PATH = "outputs/compatibility_evidence.json"
DIFFS_ARTIFACT_PATH = "outputs/diffs.json"
FINDINGS_ARTIFACT_PATH = "outputs/findings.json"
ANALYZE_ARTIFACT_PATHS = (
    COMPATIBILITY_EVIDENCE_ARTIFACT_PATH,
    DIFFS_ARTIFACT_PATH,
    FINDINGS_ARTIFACT_PATH,
)


@dataclass(frozen=True, slots=True)
class AnalyzeOutput:
    """Immutable Analyze output without independent identity or persistence.

    Attributes:
        compatibility_evidence: Observations copied from committed Check reasons.
        diffs: Atomic metric comparisons in reference/metric order.
        reference_comparison_coverage: One group per planned reference.
        findings: Materialized conclusions in deterministic identity order.
    """

    compatibility_evidence: tuple[CompatibilityEvidence, ...]
    diffs: tuple[Diff, ...]
    reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...]
    findings: tuple[Finding, ...]

    def __post_init__(self) -> None:
        """Defensively freeze the output collections."""
        for field in (
            "compatibility_evidence",
            "diffs",
            "reference_comparison_coverage",
            "findings",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))


def validate_analyze_output(
    output: AnalyzeOutput,
    *,
    prepared_context: PreparedContext,
    contract_check_result: ContractCheckResult,
    selected_metric_names: tuple[str, ...] | None = None,
) -> None:
    """Validate Analyze lineage, metric accounting, and Finding citations.

    Args:
        output: Complete computed or hydrated Analyze output.
        prepared_context: Committed owner, Contract, Recipe, and references.
        contract_check_result: Authoritative Check status and ordered reasons.
        selected_metric_names: Resolved selection during execution. Hydration
            instead checks explicit Recipe selection or agreement between groups.

    Raises:
        GatewayConsistencyViolation: If any output contradicts its owning state
            or another output. Details identify only the owner and artifact path.
    """
    if output.compatibility_evidence != materialize_compatibility_evidence(
        prepared_context, contract_check_result
    ):
        raise _inconsistent(prepared_context, COMPATIBILITY_EVIDENCE_ARTIFACT_PATH)
    invalid = False
    try:
        _validate_diffs(output, prepared_context, contract_check_result, selected_metric_names)
    except (ValueError, TypeError, KeyError, AttributeError, InvariantViolation):
        invalid = True
    if invalid:
        raise _inconsistent(prepared_context, DIFFS_ARTIFACT_PATH)
    try:
        _validate_findings(output, prepared_context)
    except (ValueError, TypeError, KeyError, AttributeError, InvariantViolation):
        invalid = True
    if invalid:
        raise _inconsistent(prepared_context, FINDINGS_ARTIFACT_PATH)


def _require(condition: bool) -> None:
    """Reject inconsistent internal shapes without including payload values."""
    if not condition:
        raise ValueError("Inconsistent Analyze output.")


def _validate_diffs(
    output: AnalyzeOutput,
    context: PreparedContext,
    check: ContractCheckResult,
    selected_names: tuple[str, ...] | None,
) -> None:
    """Validate every reference group and its complete atomic metric accounting."""
    validate_diff_identity_consistency(output.diffs)
    diffs = {diff.diff_id: diff for diff in output.diffs}
    _require(len(diffs) == len(output.diffs))
    _require(len(output.reference_comparison_coverage) == len(context.reference_plan))
    accounted_ids: list[str] = []
    expected_names = context.effective_recipe.analysis.metric_names
    if selected_names is not None:
        _require(expected_names is None or expected_names == selected_names)
        expected_names = selected_names
    for entry, group in zip(
        context.reference_plan, output.reference_comparison_coverage, strict=True
    ):
        _require(group.reference_kind == entry.kind)
        expected_reference = (
            None
            if entry.reference is None
            else DiffReference(
                entry.kind, entry.reference.monitoring_run_id, entry.reference.source_run_id
            )
        )
        _require(group.reference == expected_reference)
        if entry.reference is None:
            _require(group.status is ReferenceComparisonStatus.UNAVAILABLE)
            _require(group.reason == entry.unavailable_reason)
        elif check.status is ComparabilityStatus.FAIL:
            _require(group.status is ReferenceComparisonStatus.SKIPPED)
            _require(group.reason == "current_not_comparable")
        elif group.status is not ReferenceComparisonStatus.COMPLETED:
            _require(group.status is ReferenceComparisonStatus.UNAVAILABLE)
            _require(group.reason == "reference_source_run_missing")
        if group.status is not ReferenceComparisonStatus.COMPLETED:
            _require(not group.diff_ids and not group.metric_unavailability)
            continue
        _require(group.reason is None)
        rows = tuple(diffs[identity] for identity in group.diff_ids)
        _require(all(diff.reference == expected_reference for diff in rows))
        names = tuple(diff.metric_name for diff in rows)
        unavailable_names = tuple(row.metric_name for row in group.metric_unavailability)
        _require(names == tuple(sorted(set(names))))
        _require(unavailable_names == tuple(sorted(set(unavailable_names))))
        combined_names = tuple(sorted((*names, *unavailable_names)))
        _require(len(set(combined_names)) == len(combined_names))
        if expected_names is None:
            expected_names = combined_names
        _require(combined_names == expected_names)
        accounted_ids.extend(group.diff_ids)
    _require(tuple(accounted_ids) == tuple(diffs))
    observations: dict[tuple[str | None, str], float] = {}
    for diff in output.diffs:
        _require(
            (diff.monitoring_run_id, diff.source_run_id)
            == (context.monitoring_run_id, context.source_run_id)
        )
        for source, value in (
            (diff.source_run_id, diff.current_value),
            (diff.reference.source_run_id, diff.reference_value),
        ):
            key = (source, diff.metric_name)
            _require(observations.setdefault(key, value) == value)


def _validate_findings(output: AnalyzeOutput, context: PreparedContext) -> None:
    """Validate canonical Findings against the exact compiled bindings and evidence."""
    validate_finding_identity_consistency(output.findings)
    ids = tuple(finding.finding_id for finding in output.findings)
    _require(ids == tuple(sorted(set(ids))))
    policies = {
        (binding.finding_policy_id, binding.finding_policy_version)
        for binding in context.effective_recipe.analysis.finding_policy_bindings
    }
    diffs = {diff.diff_id: diff for diff in output.diffs}
    evidence = {item.compatibility_evidence_id: item for item in output.compatibility_evidence}
    for finding in output.findings:
        _require(
            (finding.monitoring_run_id, finding.source_run_id)
            == (context.monitoring_run_id, context.source_run_id)
        )
        _require((finding.finding_policy_id, finding.finding_policy_version) in policies)
        for references in (finding.evidence_diff_ids, finding.evidence_compatibility_ids):
            _require(references == tuple(sorted(set(references))))
        validate_finding_evidence_references(
            finding, diffs_by_id=diffs, compatibility_evidence_by_id=evidence
        )


def analyze_output_to_artifacts(
    output: AnalyzeOutput,
    *,
    prepared_context: PreparedContext,
    contract_check_result: ContractCheckResult,
) -> dict[str, dict[str, object]]:
    """Validate and project the three artifacts in dependency order.

    Args:
        output: Complete Analyze output to persist.
        prepared_context: Committed owner and fixed reference plan.
        contract_check_result: Authoritative Check result.

    Returns:
        Canonical JSON payloads keyed by their exact artifact paths.

    Raises:
        GatewayConsistencyViolation: If output fails cross-artifact validation.
    """
    validate_analyze_output(
        output, prepared_context=prepared_context, contract_check_result=contract_check_result
    )
    return _project_artifacts(output, prepared_context)


def _project_artifacts(
    output: AnalyzeOutput, context: PreparedContext
) -> dict[str, dict[str, object]]:
    """Project previously validated values without adding duplicate row identities."""
    envelope: dict[str, object] = {
        "artifact_schema_version": "v0",
        "monitoring_run_id": context.monitoring_run_id,
        "source_run_id": context.source_run_id,
    }
    diffs = {diff.diff_id: diff for diff in output.diffs}
    return {
        COMPATIBILITY_EVIDENCE_ARTIFACT_PATH: compatibility_evidence_to_dict(
            context, output.compatibility_evidence
        ),
        DIFFS_ARTIFACT_PATH: {
            **envelope,
            "reference_groups": [
                {
                    "reference_kind": group.reference_kind.value,
                    "reference": None
                    if group.reference is None
                    else {
                        "kind": group.reference.kind.value,
                        "monitoring_run_id": group.reference.monitoring_run_id,
                        "source_run_id": group.reference.source_run_id,
                    },
                    "status": group.status.value,
                    "reason": group.reason,
                    "diffs": [_diff_row(diffs[identity]) for identity in group.diff_ids],
                    "metric_unavailability": [
                        {"metric_name": row.metric_name, "reason": row.reason}
                        for row in group.metric_unavailability
                    ],
                }
                for group in output.reference_comparison_coverage
            ],
        },
        FINDINGS_ARTIFACT_PATH: {
            **envelope,
            "findings": [_finding_row(finding) for finding in output.findings],
        },
    }


def _diff_row(diff: Diff) -> dict[str, object]:
    """Project one atomic Diff under inherited current and reference identities."""
    return {
        "diff_id": diff.diff_id,
        "metric_name": diff.metric_name,
        "current_value": diff.current_value,
        "reference_value": diff.reference_value,
        "delta": diff.delta,
    }


def _finding_row(finding: Finding) -> dict[str, object]:
    """Project one Finding under the artifact's current identity pair."""
    return {
        "finding_id": finding.finding_id,
        "finding_policy_id": finding.finding_policy_id,
        "finding_policy_version": finding.finding_policy_version,
        "finding_rule_id": finding.finding_rule_id,
        "severity": finding.severity.value,
        "category": finding.category,
        "summary": finding.summary,
        "recommendation": finding.recommendation,
        "evidence_diff_ids": list(finding.evidence_diff_ids),
        "evidence_compatibility_ids": list(finding.evidence_compatibility_ids),
    }


def _inconsistent(context: PreparedContext, path: str) -> GatewayConsistencyViolation:
    """Identify an invalid artifact without exposing its payload."""
    return GatewayConsistencyViolation.monitoring_run_json_artifact_inconsistent(
        monitoring_run_id=context.monitoring_run_id, path=path
    )
