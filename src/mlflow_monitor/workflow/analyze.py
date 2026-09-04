"""Backend-independent Analyze execution and Finding policy evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from mlflow_monitor.compatibility import materialize_compatibility_evidence
from mlflow_monitor.differ import ComputedDiffCoverage, compute_diffs_and_coverage
from mlflow_monitor.domain import (
    ComparabilityStatus,
    CompatibilityEvidence,
    ContractCheckResult,
    Diff,
    DiffReference,
    Finding,
    FindingDraft,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
from mlflow_monitor.errors import (
    ANALYZE_FINDING_POLICY_EVALUATION_FAILED,
    ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT,
    ANALYZE_FINDING_POLICY_OUTPUT_INVALID,
    ANALYZE_MISSING_CURRENT_SOURCE_RUN,
    AnalyzeStageError,
    InvariantViolation,
    PreparedContextConsistencyViolation,
)
from mlflow_monitor.finding_policy import materialize_finding_drafts
from mlflow_monitor.gateway.protocol import MonitoringGateway
from mlflow_monitor.invariant import validate_contract_check_result
from mlflow_monitor.recipe_compiler import CompiledFindingPolicyBinding, CompiledRecipe
from mlflow_monitor.utils import canonical_json

from .analyze_artifacts import AnalyzeOutput
from .prepared_context import PreparedContext


def execute_analyze(
    *,
    prepared_context: PreparedContext,
    contract_check_result: ContractCheckResult,
    compiled_recipe: CompiledRecipe,
    gateway: MonitoringGateway,
) -> AnalyzeOutput:
    """Compute Analyze output from committed Prepare and Check inputs.

    This internal stage reads metrics but never persists output or terminalizes a
    Monitoring Run. Each distinct source supplies one detached metric snapshot.

    Args:
        prepared_context: Hydrated committed identities, Contract, and reference plan.
        contract_check_result: Hydrated committed Check output.
        compiled_recipe: Executable Recipe matching the committed effective plan.
        gateway: Read-only source-metric boundary.

    Returns:
        Complete deterministic evidence, coverage, and Findings.

    Raises:
        AnalyzeStageError: If the current source is missing or a policy fails.
        PreparedContextConsistencyViolation: If the supplied Recipe disagrees
            with committed prepared state.
        InvariantViolation: If the supplied Check result is invalid.
    """
    if canonical_json(prepared_context.effective_recipe.to_dict()) != canonical_json(
        compiled_recipe.effective_plan.to_dict()
    ):
        raise PreparedContextConsistencyViolation.effective_recipe_mismatch(
            field="effective_recipe"
        )
    if prepared_context.contract != compiled_recipe.contract:
        raise PreparedContextConsistencyViolation.contract_mismatch(field="contract")
    validate_contract_check_result(contract_check_result)
    evidence = materialize_compatibility_evidence(prepared_context, contract_check_result)

    if contract_check_result.status is ComparabilityStatus.FAIL:
        computed = ComputedDiffCoverage(
            diffs=(),
            coverages=tuple(
                ReferenceComparisonCoverage(
                    reference_kind=entry.kind,
                    reference=None
                    if entry.reference is None
                    else DiffReference(
                        entry.kind, entry.reference.monitoring_run_id, entry.reference.source_run_id
                    ),
                    status=ReferenceComparisonStatus.UNAVAILABLE
                    if entry.reference is None
                    else ReferenceComparisonStatus.SKIPPED,
                    diff_ids=(),
                    metric_unavailability=(),
                    reason=entry.unavailable_reason
                    if entry.reference is None
                    else "current_not_comparable",
                )
                for entry in prepared_context.reference_plan
            ),
        )
    else:
        computed = _read_and_compare_metrics(prepared_context, gateway)

    findings = execute_finding_policies(
        monitoring_run_id=prepared_context.monitoring_run_id,
        source_run_id=prepared_context.source_run_id,
        finding_policy_bindings=compiled_recipe.finding_policy_bindings,
        diffs=computed.diffs,
        compatibility_evidence=evidence,
        reference_comparison_coverage=computed.coverages,
    )
    return AnalyzeOutput(evidence, computed.diffs, computed.coverages, findings)


def _read_and_compare_metrics(
    context: PreparedContext, gateway: MonitoringGateway
) -> ComputedDiffCoverage:
    """Observe each distinct source once, then compare the selected names."""
    selection = context.effective_recipe.analysis.metric_names
    current = gateway.get_source_run_metrics(context.source_run_id, selection)
    if current is None:
        raise AnalyzeStageError(
            code=ANALYZE_MISSING_CURRENT_SOURCE_RUN,
            message="Current Source Training Run could not be read for Analyze.",
            details=(("source_run_id", context.source_run_id),),
        )
    current = dict(current)
    names = tuple(sorted(current)) if selection is None else selection
    snapshots: dict[str, dict[str, float] | None] = {context.source_run_id: current}
    for entry in context.reference_plan:
        if entry.reference is None or entry.reference.source_run_id in snapshots:
            continue
        source = entry.reference.source_run_id
        metrics = gateway.get_source_run_metrics(source, names)
        snapshots[source] = None if metrics is None else dict(metrics)
    return compute_diffs_and_coverage(
        context.monitoring_run_id,
        context.source_run_id,
        names,
        current,
        context.reference_plan,
        snapshots,
    )


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
        drafts = _evaluate_policy(
            binding=binding,
            diffs=immutable_diffs,
            compatibility_evidence=immutable_compatibility_evidence,
            reference_comparison_coverage=immutable_coverage,
        )

        if not isinstance(drafts, tuple):
            raise _policy_error(
                binding=binding,
                code=ANALYZE_FINDING_POLICY_OUTPUT_INVALID,
                message="Finding policy output is invalid.",
            )

        policy_findings = _materialize_policy_drafts(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            binding=binding,
            drafts=drafts,
            diffs=immutable_diffs,
            compatibility_evidence=immutable_compatibility_evidence,
        )

        for finding in policy_findings:
            existing = findings_by_id.get(finding.finding_id)
            if existing is not None and existing != finding:
                raise _policy_error(
                    binding=binding,
                    code=ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT,
                    message="Finding policy output is inconsistent.",
                )
            findings_by_id[finding.finding_id] = finding

    return tuple(findings_by_id[finding_id] for finding_id in sorted(findings_by_id))


def _evaluate_policy(
    *,
    binding: CompiledFindingPolicyBinding,
    diffs: tuple[Diff, ...],
    compatibility_evidence: tuple[CompatibilityEvidence, ...],
    reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...],
) -> tuple[FindingDraft, ...]:
    """Evaluate one policy without retaining policy-controlled exceptions."""
    evaluation_error: AnalyzeStageError
    try:
        return binding.policy.evaluate(
            parameters=binding.parameters,
            diffs=diffs,
            compatibility_evidence=compatibility_evidence,
            reference_comparison_coverage=reference_comparison_coverage,
        )
    except Exception:
        evaluation_error = _policy_error(
            binding=binding,
            code=ANALYZE_FINDING_POLICY_EVALUATION_FAILED,
            message="Finding policy evaluation failed.",
        )
    raise evaluation_error


def _materialize_policy_drafts(
    *,
    monitoring_run_id: str,
    source_run_id: str,
    binding: CompiledFindingPolicyBinding,
    drafts: tuple[FindingDraft, ...],
    diffs: tuple[Diff, ...],
    compatibility_evidence: tuple[CompatibilityEvidence, ...],
) -> tuple[Finding, ...]:
    """Materialize one policy's drafts into bounded valid or inconsistent output."""
    output_error: AnalyzeStageError
    try:
        return materialize_finding_drafts(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            finding_policy_id=binding.finding_policy_id,
            finding_policy_version=binding.finding_policy_version,
            drafts=drafts,
            diffs=diffs,
            compatibility_evidence=compatibility_evidence,
        )
    except InvariantViolation as exc:
        code = ANALYZE_FINDING_POLICY_OUTPUT_INVALID
        message = "Finding policy output is invalid."
        if exc.code == "finding_identity_content_conflict":
            code = ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT
            message = "Finding policy output is inconsistent."
        output_error = _policy_error(binding=binding, code=code, message=message)
    except Exception:
        output_error = _policy_error(
            binding=binding,
            code=ANALYZE_FINDING_POLICY_OUTPUT_INVALID,
            message="Finding policy output is invalid.",
        )
    raise output_error


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
