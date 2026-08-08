"""Finding policy boundary and package-owned Finding materialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    Diff,
    Finding,
    FindingDraft,
    ReferenceComparisonCoverage,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.identity import make_finding_id
from mlflow_monitor.invariant import (
    validate_compatibility_evidence_identity_consistency,
    validate_diff_identity_consistency,
    validate_finding_evidence_references,
    validate_finding_identity,
)

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | Mapping[str, JSONValue]
type FrozenJSONValue = JSONScalar | tuple[FrozenJSONValue, ...] | Mapping[str, FrozenJSONValue]
type FrozenFindingPolicyParameters = Mapping[str, FrozenJSONValue]


class FindingPolicy(Protocol):
    """Registered extension point for versioned Finding interpretation."""

    @property
    def finding_policy_id(self) -> str:
        """Return the stable policy identifier."""
        ...

    @property
    def finding_policy_version(self) -> str:
        """Return the exact policy version."""
        ...

    def validate_parameters(
        self,
        parameters: Mapping[str, JSONValue],
    ) -> FrozenFindingPolicyParameters:
        """Validate and freeze one Recipe policy-parameter mapping.

        Args:
            parameters: JSON-compatible parameters from one Recipe binding.

        Returns:
            Validated parameters frozen for later policy evaluation.
        """
        ...

    def evaluate(
        self,
        *,
        parameters: FrozenFindingPolicyParameters,
        diffs: tuple[Diff, ...],
        compatibility_evidence: tuple[CompatibilityEvidence, ...],
        reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...],
    ) -> tuple[FindingDraft, ...]:
        """Propose transient Finding Drafts from immutable Analyze inputs.

        Args:
            parameters: Validated frozen parameters for this policy binding.
            diffs: Immutable atomic metric Diffs from the current Analyze output.
            compatibility_evidence: Immutable Compatibility Evidence from the
                current Analyze output.
            reference_comparison_coverage: Immutable coverage groups from the
                current Analyze output.

        Returns:
            Transient Finding Drafts for package-owned validation and
            materialization.
        """
        ...


def materialize_finding_drafts(
    *,
    monitoring_run_id: str,
    source_run_id: str,
    finding_policy_id: str,
    finding_policy_version: str,
    drafts: Sequence[FindingDraft],
    diffs: Sequence[Diff],
    compatibility_evidence: Sequence[CompatibilityEvidence],
) -> tuple[Finding, ...]:
    """Validate policy drafts and attach package-owned Finding identity.

    Args:
        monitoring_run_id: Monitoring Run that owns the Findings.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        finding_policy_id: Identifier of the policy that emitted the drafts.
        finding_policy_version: Version of the policy that emitted the drafts.
        drafts: Transient Finding conclusions emitted by the policy.
        diffs: Complete Diff records supplied to the policy.
        compatibility_evidence: Complete Compatibility Evidence records supplied
            to the policy.

    Returns:
        Canonically ordered materialized Findings with evidence identities sorted
        and deduplicated and identical retries removed.

    Raises:
        InvariantViolation: If a draft is invalid, references invalid evidence, or
            conflicts with another draft under one deterministic identity.
    """
    validate_diff_identity_consistency(diffs)
    validate_compatibility_evidence_identity_consistency(compatibility_evidence)
    diffs_by_id = {diff.diff_id: diff for diff in diffs}
    compatibility_evidence_by_id = {
        evidence.compatibility_evidence_id: evidence for evidence in compatibility_evidence
    }

    findings_by_id: dict[str, Finding] = {}
    for draft in drafts:
        if not isinstance(draft, FindingDraft):
            raise InvariantViolation(
                code="finding_draft_invalid",
                message="Finding policy output must contain only FindingDraft values.",
                entity="FindingDraft",
            )

        evidence_diff_ids = tuple(sorted(set(draft.evidence_diff_ids)))
        evidence_compatibility_ids = tuple(sorted(set(draft.evidence_compatibility_ids)))
        finding = Finding(
            finding_id=make_finding_id(
                monitoring_run_id=monitoring_run_id,
                source_run_id=source_run_id,
                finding_policy_id=finding_policy_id,
                finding_policy_version=finding_policy_version,
                finding_rule_id=draft.finding_rule_id,
                evidence_diff_ids=evidence_diff_ids,
                evidence_compatibility_ids=evidence_compatibility_ids,
            ),
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            finding_policy_id=finding_policy_id,
            finding_policy_version=finding_policy_version,
            finding_rule_id=draft.finding_rule_id,
            severity=draft.severity,
            category=draft.category,
            summary=draft.summary,
            recommendation=draft.recommendation,
            evidence_diff_ids=evidence_diff_ids,
            evidence_compatibility_ids=evidence_compatibility_ids,
        )
        validate_finding_evidence_references(
            finding,
            diffs_by_id=diffs_by_id,
            compatibility_evidence_by_id=compatibility_evidence_by_id,
        )
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

    return tuple(findings_by_id[finding_id] for finding_id in sorted(findings_by_id))
