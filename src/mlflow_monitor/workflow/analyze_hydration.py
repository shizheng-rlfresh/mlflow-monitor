"""Strict saved-output hydration for Analyze commit and replay."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from mlflow_monitor.compatibility import materialize_compatibility_evidence
from mlflow_monitor.domain import (
    ContractCheckResult,
    Diff,
    DiffReference,
    DiffReferenceKind,
    Finding,
    FindingSeverity,
    MetricComparisonUnavailable,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.utils import canonical_json

from .analyze_artifacts import (
    ANALYZE_ARTIFACT_PATHS,
    DIFFS_ARTIFACT_PATH,
    FINDINGS_ARTIFACT_PATH,
    AnalyzeOutput,
    _inconsistent,
    _project_artifacts,
    _require,
    _validate_diffs,
    _validate_findings,
)

if TYPE_CHECKING:
    from .prepared_context import PreparedContext


def hydrate_analyze_output(
    artifacts: Mapping[str, object],
    *,
    prepared_context: PreparedContext,
    contract_check_result: ContractCheckResult,
) -> AnalyzeOutput:
    """Reconstruct complete committed Analyze output without external reads.

    Args:
        artifacts: All three decoded Analyze artifacts keyed by canonical path.
        prepared_context: Validated committed owner and reference plan.
        contract_check_result: Validated committed Check result.

    Returns:
        Validated output whose rows inherit the saved owner and reference pairs.

    Raises:
        GatewayConsistencyViolation: If any artifact is missing, malformed, or
            inconsistent with its dependencies. Payload values are not exposed.
    """
    for path in ANALYZE_ARTIFACT_PATHS:
        if path not in artifacts:
            raise _inconsistent(prepared_context, path)
    return _hydrate_present(artifacts, prepared_context, contract_check_result)


def validate_partial_analyze_artifacts(
    artifacts: Mapping[str, object],
    *,
    prepared_context: PreparedContext,
    contract_check_result: ContractCheckResult,
) -> None:
    """Validate existing checked-stage artifacts before any recomputation.

    Args:
        artifacts: Present Analyze artifacts; missing dependencies are permitted.
        prepared_context: Validated committed owner and reference plan.
        contract_check_result: Validated committed Check result.

    Raises:
        GatewayConsistencyViolation: If any present artifact is inconsistent.
            Diff citations require complete validation after recomputation when
            the Diff artifact is absent.
    """
    _hydrate_present(artifacts, prepared_context, contract_check_result)


def _hydrate_present(
    artifacts: Mapping[str, object], context: PreparedContext, check: ContractCheckResult
) -> AnalyzeOutput:
    """Parse present artifacts, then validate each against its canonical projection."""
    diffs: tuple[Diff, ...] = ()
    groups: tuple[ReferenceComparisonCoverage, ...] = ()
    findings: tuple[Finding, ...] = ()
    evidence = materialize_compatibility_evidence(context, check)
    for path in ANALYZE_ARTIFACT_PATHS:
        if path not in artifacts:
            continue
        invalid = False
        try:
            raw = _object(artifacts[path])
            if path == DIFFS_ARTIFACT_PATH:
                diffs, groups = _read_groups(raw, context)
            if path == FINDINGS_ARTIFACT_PATH:
                findings = _read_findings(raw, context)
            output = AnalyzeOutput(evidence, diffs, groups, findings)
            if path == DIFFS_ARTIFACT_PATH:
                _validate_diffs(output, context, check, None)
            if path == FINDINGS_ARTIFACT_PATH:
                _validate_findings(
                    output, context, diffs_available=DIFFS_ARTIFACT_PATH in artifacts
                )
            expected = _project_artifacts(output, context)[path]
            _require(canonical_json(raw) == canonical_json(expected))
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, InvariantViolation):
            invalid = True
        if invalid:
            # Raise outside the handler so malformed payloads cannot leak through
            # exception chains or upstream constructor diagnostics.
            raise _inconsistent(context, path)
    return AnalyzeOutput(evidence, diffs, groups, findings)


def _object(raw: object) -> dict[str, Any]:
    """Require a JSON object; canonical projection checks its exact field set."""
    _require(isinstance(raw, dict))
    assert isinstance(raw, dict)
    return raw


def _rows(raw: object) -> list[Any]:
    """Require a JSON array before domain constructors can coerce collections."""
    _require(isinstance(raw, list))
    assert isinstance(raw, list)
    return raw


def _read_groups(
    raw: dict[str, Any], context: PreparedContext
) -> tuple[tuple[Diff, ...], tuple[ReferenceComparisonCoverage, ...]]:
    """Restore inherited identities without accepting row-level overrides."""
    diffs: list[Diff] = []
    groups: list[ReferenceComparisonCoverage] = []
    for item in _rows(raw["reference_groups"]):
        group = _object(item)
        reference_raw = group["reference"]
        reference = None
        if reference_raw is not None:
            reference_fields = _object(reference_raw)
            reference = DiffReference(
                kind=DiffReferenceKind(reference_fields["kind"]),
                monitoring_run_id=reference_fields["monitoring_run_id"],
                source_run_id=reference_fields["source_run_id"],
            )
        group_diffs: list[Diff] = []
        for row in _rows(group["diffs"]):
            _require(reference is not None)
            assert reference is not None
            fields = _object(row)
            group_diffs.append(
                Diff(
                    monitoring_run_id=context.monitoring_run_id,
                    source_run_id=context.source_run_id,
                    reference=reference,
                    diff_id=fields["diff_id"],
                    metric_name=fields["metric_name"],
                    current_value=fields["current_value"],
                    reference_value=fields["reference_value"],
                    delta=fields["delta"],
                )
            )
        unavailable = tuple(
            MetricComparisonUnavailable(
                metric_name=_object(row)["metric_name"], reason=_object(row)["reason"]
            )
            for row in _rows(group["metric_unavailability"])
        )
        groups.append(
            ReferenceComparisonCoverage(
                reference_kind=DiffReferenceKind(group["reference_kind"]),
                reference=reference,
                status=ReferenceComparisonStatus(group["status"]),
                reason=group["reason"],
                diff_ids=tuple(diff.diff_id for diff in group_diffs),
                metric_unavailability=unavailable,
            )
        )
        diffs.extend(group_diffs)
    return tuple(diffs), tuple(groups)


def _read_findings(raw: dict[str, Any], context: PreparedContext) -> tuple[Finding, ...]:
    """Restore canonical Findings without invoking their bound policies."""
    findings: list[Finding] = []
    for item in _rows(raw["findings"]):
        row = _object(item)
        findings.append(
            Finding(
                monitoring_run_id=context.monitoring_run_id,
                source_run_id=context.source_run_id,
                finding_id=row["finding_id"],
                finding_policy_id=row["finding_policy_id"],
                finding_policy_version=row["finding_policy_version"],
                finding_rule_id=row["finding_rule_id"],
                severity=FindingSeverity(row["severity"]),
                category=row["category"],
                summary=row["summary"],
                recommendation=row["recommendation"],
                evidence_diff_ids=tuple(_rows(row["evidence_diff_ids"])),
                evidence_compatibility_ids=tuple(_rows(row["evidence_compatibility_ids"])),
            )
        )
    return tuple(findings)
