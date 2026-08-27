"""Pure Analyze-stage Finding policy execution."""

from __future__ import annotations

from collections.abc import Sequence

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    Diff,
    Finding,
    ReferenceComparisonCoverage,
)
from mlflow_monitor.errors import (
    ANALYZE_FINDING_POLICY_EVALUATION_FAILED,
    ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT,
    ANALYZE_FINDING_POLICY_OUTPUT_INVALID,
    AnalyzeStageError,
    InvariantViolation,
)
from mlflow_monitor.finding_policy import materialize_finding_drafts
from mlflow_monitor.recipe_compiler import CompiledFindingPolicyBinding


def execute_finding_policies(
    *,
    monitoring_run_id: str,
    source_run_id: str,
    finding_policy_bindings: Sequence[CompiledFindingPolicyBinding],
    diffs: Sequence[Diff],
    compatibility_evidence: Sequence[CompatibilityEvidence],
    reference_comparison_coverage: Sequence[ReferenceComparisonCoverage],
) -> tuple[Finding, ...]:
    """Execute compiled Finding policies over immutable Analyze evidence.

    Args:
        monitoring_run_id: Monitoring Run that owns materialized Findings.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        finding_policy_bindings: Compiled policy implementations and frozen parameters.
        diffs: Complete Diff output from the current Analyze execution.
        compatibility_evidence: Complete Compatibility Evidence output from the
            current Analyze execution.
        reference_comparison_coverage: Complete reference coverage output from the
            current Analyze execution.

    Returns:
        Materialized Findings in deterministic identity order.

    Raises:
        AnalyzeStageError: If policy evaluation fails or a policy returns invalid
            or inconsistent output.
    """
    immutable_diffs = tuple(diffs)
    immutable_compatibility_evidence = tuple(compatibility_evidence)
    immutable_coverage = tuple(reference_comparison_coverage)
    findings_by_id: dict[str, Finding] = {}

    for binding in sorted(
        finding_policy_bindings,
        key=lambda value: (value.finding_policy_id, value.finding_policy_version),
    ):
        try:
            drafts = binding.policy.evaluate(
                parameters=binding.parameters,
                diffs=immutable_diffs,
                compatibility_evidence=immutable_compatibility_evidence,
                reference_comparison_coverage=immutable_coverage,
            )
        except Exception as exc:
            raise _policy_error(
                binding=binding,
                code=ANALYZE_FINDING_POLICY_EVALUATION_FAILED,
                message="Finding policy evaluation failed.",
            ) from exc

        if not isinstance(drafts, tuple):
            cause = TypeError("Finding policy output must be a tuple.")
            raise _policy_error(
                binding=binding,
                code=ANALYZE_FINDING_POLICY_OUTPUT_INVALID,
                message="Finding policy output is invalid.",
            ) from cause

        try:
            policy_findings = materialize_finding_drafts(
                monitoring_run_id=monitoring_run_id,
                source_run_id=source_run_id,
                finding_policy_id=binding.finding_policy_id,
                finding_policy_version=binding.finding_policy_version,
                drafts=drafts,
                diffs=immutable_diffs,
                compatibility_evidence=immutable_compatibility_evidence,
            )
        except InvariantViolation as exc:
            code = ANALYZE_FINDING_POLICY_OUTPUT_INVALID
            message = "Finding policy output is invalid."
            if exc.code == "finding_identity_content_conflict":
                code = ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT
                message = "Finding policy output is inconsistent."
            raise _policy_error(binding=binding, code=code, message=message) from exc

        for finding in policy_findings:
            existing = findings_by_id.get(finding.finding_id)
            if existing is not None and existing != finding:
                cause = InvariantViolation(
                    code="finding_identity_content_conflict",
                    message="Finding identity maps to conflicting content.",
                    entity="Finding",
                    field="finding_id",
                )
                raise _policy_error(
                    binding=binding,
                    code=ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT,
                    message="Finding policy output is inconsistent.",
                ) from cause
            findings_by_id[finding.finding_id] = finding

    return tuple(findings_by_id[finding_id] for finding_id in sorted(findings_by_id))


def _policy_error(
    *,
    binding: CompiledFindingPolicyBinding,
    code: str,
    message: str,
) -> AnalyzeStageError:
    """Build a bounded Analyze failure for one exact policy binding."""
    return AnalyzeStageError(
        code=code,
        message=message,
        details=(
            ("finding_policy_id", binding.finding_policy_id),
            ("finding_policy_version", binding.finding_policy_version),
        ),
    )
