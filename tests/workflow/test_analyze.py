"""Specifications for pure Analyze-stage Finding policy execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import pytest

from mlflow_monitor.domain import (
    CompatibilityEvidence,
    ContractCheckReason,
    Diff,
    DiffReference,
    DiffReferenceKind,
    FindingDraft,
    FindingSeverity,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
from mlflow_monitor.errors import (
    ANALYZE_FINDING_POLICY_EVALUATION_FAILED,
    ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT,
    ANALYZE_FINDING_POLICY_OUTPUT_INVALID,
    AnalyzeStageError,
)
from mlflow_monitor.finding_policy import (
    FrozenFindingPolicyParameters,
    JSONValue,
)
from mlflow_monitor.identity import (
    make_compatibility_evidence_id,
    make_diff_id,
)
from mlflow_monitor.recipe_compiler import CompiledFindingPolicyBinding
from mlflow_monitor.workflow import execute_finding_policies

MONITORING_RUN_ID = "monitoring-run-current"
SOURCE_RUN_ID = "train-run-current"


@dataclass(frozen=True, slots=True)
class _PolicyCall:
    """Inputs observed by one test Finding policy invocation."""

    finding_policy_id: str
    finding_policy_version: str
    parameters: FrozenFindingPolicyParameters
    diffs: tuple[Diff, ...]
    compatibility_evidence: tuple[CompatibilityEvidence, ...]
    reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...]


class _RecordingPolicy:
    """Finding policy double that records its immutable execution inputs."""

    def __init__(
        self,
        *,
        finding_policy_id: str,
        finding_policy_version: str,
        drafts: object,
        calls: list[_PolicyCall],
        failure: BaseException | None = None,
    ) -> None:
        self.finding_policy_id = finding_policy_id
        self.finding_policy_version = finding_policy_version
        self._drafts = drafts
        self._calls = calls
        self._failure = failure

    def validate_parameters(
        self,
        parameters: Mapping[str, JSONValue],
    ) -> FrozenFindingPolicyParameters:
        """Reject unexpected revalidation during Analyze execution."""
        del parameters
        raise AssertionError("Analyze must not revalidate compiled policy parameters.")

    def evaluate(
        self,
        *,
        parameters: FrozenFindingPolicyParameters,
        diffs: tuple[Diff, ...],
        compatibility_evidence: tuple[CompatibilityEvidence, ...],
        reference_comparison_coverage: tuple[ReferenceComparisonCoverage, ...],
    ) -> tuple[FindingDraft, ...]:
        """Record the exact policy inputs and return configured drafts."""
        self._calls.append(
            _PolicyCall(
                finding_policy_id=self.finding_policy_id,
                finding_policy_version=self.finding_policy_version,
                parameters=parameters,
                diffs=diffs,
                compatibility_evidence=compatibility_evidence,
                reference_comparison_coverage=reference_comparison_coverage,
            )
        )
        if self._failure is not None:
            raise self._failure
        return self._drafts  # type: ignore[return-value]


def _diff() -> Diff:
    reference = DiffReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id="train-run-baseline",
    )
    return Diff(
        diff_id=make_diff_id(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            reference=reference,
            metric_name="accuracy",
        ),
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        reference=reference,
        metric_name="accuracy",
        current_value=0.75,
        reference_value=0.5,
        delta=0.25,
    )


def _compatibility_evidence(
    *,
    monitoring_run_id: str = MONITORING_RUN_ID,
    source_run_id: str = SOURCE_RUN_ID,
) -> CompatibilityEvidence:
    reason = ContractCheckReason(
        code="environment_mismatch",
        message="Execution environment does not match the baseline.",
        blocking=False,
    )
    return CompatibilityEvidence(
        compatibility_evidence_id=make_compatibility_evidence_id(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            baseline_source_run_id="train-run-baseline",
            contract_id="default_permissive",
            contract_version="v0",
            reason_code=reason.code,
        ),
        monitoring_run_id=monitoring_run_id,
        source_run_id=source_run_id,
        baseline_source_run_id="train-run-baseline",
        contract_id="default_permissive",
        contract_version="v0",
        reason=reason,
    )


def _coverage(diff: Diff) -> ReferenceComparisonCoverage:
    return ReferenceComparisonCoverage(
        reference_kind=diff.reference.kind,
        reference=diff.reference,
        status=ReferenceComparisonStatus.COMPLETED,
        diff_ids=(diff.diff_id,),
        metric_unavailability=(),
        reason=None,
    )


def _draft(*, finding_rule_id: str, evidence_id: str) -> FindingDraft:
    return FindingDraft(
        finding_rule_id=finding_rule_id,
        severity=FindingSeverity.HIGH,
        category="quality",
        summary=f"Policy conclusion for {finding_rule_id}.",
        recommendation="Review the supporting evidence.",
        evidence_diff_ids=(),
        evidence_compatibility_ids=(evidence_id,),
    )


def _binding(
    policy: _RecordingPolicy,
    *,
    parameters: FrozenFindingPolicyParameters | None = None,
) -> CompiledFindingPolicyBinding:
    return CompiledFindingPolicyBinding(
        finding_policy_id=policy.finding_policy_id,
        finding_policy_version=policy.finding_policy_version,
        parameters=parameters or MappingProxyType({}),
        policy=policy,
    )


def _execute_one_policy(
    policy: _RecordingPolicy,
    *,
    compatibility_evidence: list[CompatibilityEvidence] | None = None,
    parameters: FrozenFindingPolicyParameters | None = None,
) -> tuple[object, ...]:
    evidence = compatibility_evidence
    if evidence is None:
        evidence = [_compatibility_evidence()]
    return execute_finding_policies(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_bindings=(_binding(policy, parameters=parameters),),
        diffs=(),
        compatibility_evidence=evidence,
        reference_comparison_coverage=(),
    )


def test_execute_finding_policies_uses_canonical_order_and_shared_immutable_inputs() -> None:
    """Compiled policies should execute independently over the same frozen evidence."""
    diff = _diff()
    evidence = _compatibility_evidence()
    coverage = _coverage(diff)
    calls: list[_PolicyCall] = []
    parameters_by_policy = {
        "a-policy": MappingProxyType({"threshold": 0.1}),
        "z-policy": MappingProxyType({"threshold": 0.2}),
    }
    bindings = tuple(
        CompiledFindingPolicyBinding(
            finding_policy_id=policy_id,
            finding_policy_version="1",
            parameters=parameters_by_policy[policy_id],
            policy=_RecordingPolicy(
                finding_policy_id=policy_id,
                finding_policy_version="1",
                drafts=(
                    _draft(
                        finding_rule_id=f"quality.{policy_id}",
                        evidence_id=evidence.compatibility_evidence_id,
                    ),
                ),
                calls=calls,
            ),
        )
        for policy_id in ("z-policy", "a-policy")
    )

    findings = execute_finding_policies(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        finding_policy_bindings=bindings,
        diffs=[diff],
        compatibility_evidence=[evidence],
        reference_comparison_coverage=[coverage],
    )

    assert tuple(call.finding_policy_id for call in calls) == ("a-policy", "z-policy")
    assert calls[0].parameters is parameters_by_policy["a-policy"]
    assert calls[1].parameters is parameters_by_policy["z-policy"]
    assert calls[0].diffs is calls[1].diffs
    assert calls[0].diffs == (diff,)
    assert calls[0].compatibility_evidence is calls[1].compatibility_evidence
    assert calls[0].compatibility_evidence == (evidence,)
    assert calls[0].reference_comparison_coverage is calls[1].reference_comparison_coverage
    assert calls[0].reference_comparison_coverage == (coverage,)
    assert tuple(finding.finding_id for finding in findings) == tuple(
        sorted(finding.finding_id for finding in findings)
    )
    assert {
        (
            finding.monitoring_run_id,
            finding.source_run_id,
            finding.finding_policy_id,
            finding.finding_policy_version,
        )
        for finding in findings
    } == {
        (MONITORING_RUN_ID, SOURCE_RUN_ID, "a-policy", "1"),
        (MONITORING_RUN_ID, SOURCE_RUN_ID, "z-policy", "1"),
    }


def test_execute_finding_policies_accepts_empty_bindings() -> None:
    assert (
        execute_finding_policies(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            finding_policy_bindings=(),
            diffs=(),
            compatibility_evidence=(),
            reference_comparison_coverage=(),
        )
        == ()
    )


def test_execute_finding_policies_accepts_empty_draft_output() -> None:
    calls: list[_PolicyCall] = []
    policy = _RecordingPolicy(
        finding_policy_id="empty-policy",
        finding_policy_version="1",
        drafts=(),
        calls=calls,
    )

    assert _execute_one_policy(policy, compatibility_evidence=[]) == ()
    assert len(calls) == 1


@pytest.mark.parametrize("invalid_output", ([], (object(),)))
def test_execute_finding_policies_rejects_malformed_policy_output(
    invalid_output: object,
) -> None:
    calls: list[_PolicyCall] = []
    policy = _RecordingPolicy(
        finding_policy_id="invalid-policy",
        finding_policy_version="2",
        drafts=invalid_output,
        calls=calls,
    )

    with pytest.raises(AnalyzeStageError) as exc_info:
        _execute_one_policy(policy)

    error = exc_info.value
    assert error.code == ANALYZE_FINDING_POLICY_OUTPUT_INVALID
    assert str(error) == "Finding policy output is invalid."
    assert error.details == (
        ("finding_policy_id", "invalid-policy"),
        ("finding_policy_version", "2"),
    )
    assert error.__cause__ is not None


def test_execute_finding_policies_rejects_unknown_evidence_as_invalid_output() -> None:
    calls: list[_PolicyCall] = []
    policy = _RecordingPolicy(
        finding_policy_id="unknown-evidence-policy",
        finding_policy_version="1",
        drafts=(
            _draft(
                finding_rule_id="quality.unknown-evidence",
                evidence_id="compatibility-evidence-v1-unknown",
            ),
        ),
        calls=calls,
    )

    with pytest.raises(AnalyzeStageError) as exc_info:
        _execute_one_policy(policy)

    assert exc_info.value.code == ANALYZE_FINDING_POLICY_OUTPUT_INVALID
    assert exc_info.value.__cause__ is not None


def test_execute_finding_policies_rejects_cross_pair_evidence_as_invalid_output() -> None:
    evidence = _compatibility_evidence(
        monitoring_run_id="monitoring-run-other",
        source_run_id="train-run-other",
    )
    calls: list[_PolicyCall] = []
    policy = _RecordingPolicy(
        finding_policy_id="cross-pair-policy",
        finding_policy_version="1",
        drafts=(
            _draft(
                finding_rule_id="quality.cross-pair",
                evidence_id=evidence.compatibility_evidence_id,
            ),
        ),
        calls=calls,
    )

    with pytest.raises(AnalyzeStageError) as exc_info:
        _execute_one_policy(policy, compatibility_evidence=[evidence])

    assert exc_info.value.code == ANALYZE_FINDING_POLICY_OUTPUT_INVALID
    assert exc_info.value.__cause__ is not None


def test_execute_finding_policies_classifies_conflicting_identity_as_inconsistent() -> None:
    evidence = _compatibility_evidence()
    original = _draft(
        finding_rule_id="quality.conflict",
        evidence_id=evidence.compatibility_evidence_id,
    )
    calls: list[_PolicyCall] = []
    policy = _RecordingPolicy(
        finding_policy_id="conflicting-policy",
        finding_policy_version="3",
        drafts=(original, replace(original, summary="Conflicting rendered content.")),
        calls=calls,
    )

    with pytest.raises(AnalyzeStageError) as exc_info:
        _execute_one_policy(policy, compatibility_evidence=[evidence])

    error = exc_info.value
    assert error.code == ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT
    assert str(error) == "Finding policy output is inconsistent."
    assert error.details == (
        ("finding_policy_id", "conflicting-policy"),
        ("finding_policy_version", "3"),
    )
    assert error.__cause__ is not None


def test_execute_finding_policies_bounds_policy_exception_and_stops_execution() -> None:
    calls: list[_PolicyCall] = []
    failure = RuntimeError("exception-secret must not escape")
    failing_policy = _RecordingPolicy(
        finding_policy_id="a-failing-policy",
        finding_policy_version="4",
        drafts=(),
        calls=calls,
        failure=failure,
    )
    later_policy = _RecordingPolicy(
        finding_policy_id="z-later-policy",
        finding_policy_version="1",
        drafts=(),
        calls=calls,
    )
    parameters = MappingProxyType({"secret": "parameter-secret must not escape"})

    with pytest.raises(AnalyzeStageError) as exc_info:
        execute_finding_policies(
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            finding_policy_bindings=(
                _binding(later_policy),
                _binding(failing_policy, parameters=parameters),
            ),
            diffs=(),
            compatibility_evidence=(),
            reference_comparison_coverage=(),
        )

    error = exc_info.value
    assert error.code == ANALYZE_FINDING_POLICY_EVALUATION_FAILED
    assert str(error) == "Finding policy evaluation failed."
    assert error.details == (
        ("finding_policy_id", "a-failing-policy"),
        ("finding_policy_version", "4"),
    )
    assert "exception-secret" not in str(error)
    assert "parameter-secret" not in str(error)
    assert error.__cause__ is failure
    assert tuple(call.finding_policy_id for call in calls) == ("a-failing-policy",)


def test_execute_finding_policies_does_not_convert_process_interruptions() -> None:
    calls: list[_PolicyCall] = []
    policy = _RecordingPolicy(
        finding_policy_id="interrupted-policy",
        finding_policy_version="1",
        drafts=(),
        calls=calls,
        failure=KeyboardInterrupt(),
    )

    with pytest.raises(KeyboardInterrupt):
        _execute_one_policy(policy)
