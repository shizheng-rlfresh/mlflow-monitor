"""Unit tests for invariants in mlflow_monitor."""

import pytest

from mlflow_monitor.domain import (
    LKG,
    Baseline,
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
    Diff,
    DiffReferenceKind,
    Finding,
    FindingSeverity,
    LifecycleStatus,
    Run,
    Timeline,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.invariant import (
    validate_baseline_immutability,
    validate_contract_check_result,
    validate_finding_to_diff_evidence,
    validate_lkg_membership,
    validate_timeline_ownership,
)

CONTRACT = Contract(
    contract_id="default",
    version="v0",
    schema_contract_ref=None,
    feature_contract_ref=None,
    metric_contract_ref=None,
    data_scope_contract_ref=None,
    execution_contract_ref=None,
)

BASELINE = Baseline(
    timeline_id="timeline-1",
    source_run_id="train-run-1",
    model_identity="model-a",
    parameter_fingerprint="params-v1",
    data_snapshot_ref="dataset-2026-03-01",
    run_config_ref="config-v1",
    metric_snapshot={"f1": 0.87},
    environment_context={"python": "3.12"},
)

LAST_KNOWN_GOOD = LKG(
    timeline_id="timeline-1",
    monitoring_run_id="monitoring-run-1",
)

TIMELINE = Timeline(
    timeline_id="timeline-1",
    subject_id="churn_model",
    monitoring_namespace="mlflow_monitor/churn_model",
    baseline=BASELINE,
    monitoring_run_ids=["monitoring-run-1", "monitoring-run-2"],
    active_lkg_monitoring_run_id="monitoring-run-1",
    active_contract=CONTRACT,
)

RUN = Run(
    monitoring_run_id="monitoring-run-1",
    timeline_id="timeline-1",
    sequence_index=0,
    subject_id="churn_model",
    source_run_id="train-run-1",
    baseline_source_run_id="train-run-1",
    contract=CONTRACT,
    lifecycle_status=LifecycleStatus.CLOSED,
    comparability_status=ComparabilityStatus.PASS,
    contract_check_result=None,
    diff_ids=("diff_1",),
    finding_ids=("finding_1",),
)

FINDING = Finding(
    finding_id="finding-1",
    monitoring_run_id="monitoring-run-2",
    severity=FindingSeverity.HIGH,
    category="data_drift",
    summary="Significant data drift detected in feature 'age'",
    evidence_diff_ids=(
        "diff-1",
        "diff-2",
    ),
    recommendation="Investigate recent changes",
)

DIFF = Diff(
    diff_id="diff-1",
    monitoring_run_id="monitoring-run-2",
    reference_monitoring_run_id="monitoring-run-1",
    reference_kind=DiffReferenceKind.BASELINE,
    metric_deltas={"kl": -0.05},
    metadata={"feature": "age"},
)


class TestInvariantTimelineOwnership:
    def test_valid_timeline_ownership(self) -> None:
        validate_timeline_ownership(TIMELINE, baseline=BASELINE, lkg=LAST_KNOWN_GOOD, runs=[RUN])

    def test_timeline_run_ownership(self) -> None:
        run = Run(
            monitoring_run_id="monitoring-run-2",  # different monitoring_run_id="run-2",
            timeline_id="timeline-2",  # different timeline_id to trigger violation
            sequence_index=1,
            subject_id="churn_model",
            source_run_id="train-run-2",
            baseline_source_run_id="train-run-1",
            contract=CONTRACT,
            lifecycle_status=LifecycleStatus.CLOSED,
            comparability_status=ComparabilityStatus.PASS,
            contract_check_result=None,
            diff_ids=(),
            finding_ids=(),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_timeline_ownership(TIMELINE, baseline=None, lkg=None, runs=[run])

        error = exc_info.value
        assert error.code == "run_timeline_mismatch"
        assert error.field == "timeline_id"
        assert error.entity == "Run"
        assert (
            error.message == f"Run {run.timeline_id} does not match Timeline {TIMELINE.timeline_id}"
        )

    def test_timeline_baseline_ownership(self) -> None:

        baseline = Baseline(
            timeline_id="timeline-2",  # different timeline_id to trigger violation
            source_run_id="train-run-1",
            model_identity="model-a",
            parameter_fingerprint="params-v1",
            data_snapshot_ref="dataset-2026-03-01",
            run_config_ref="config-v1",
            metric_snapshot={"f1": 0.87},
            environment_context={"python": "3.12"},
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_timeline_ownership(TIMELINE, baseline=baseline, lkg=None, runs=None)

        error = exc_info.value
        assert error.code == "baseline_timeline_mismatch"
        assert error.field == "timeline_id"
        assert error.entity == "Baseline"
        assert (
            error.message
            == f"Baseline {baseline.timeline_id} does not match Timeline {TIMELINE.timeline_id}"
        )

    def test_timeline_lkg_ownership(self) -> None:

        lkg = LKG(
            timeline_id="timeline-2",  # different timeline_id to trigger violation
            monitoring_run_id="monitoring-run-1",
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_timeline_ownership(TIMELINE, baseline=None, lkg=lkg, runs=None)

        error = exc_info.value
        assert error.code == "lkg_timeline_mismatch"
        assert error.field == "timeline_id"
        assert error.entity == "LKG"
        assert (
            error.message == f"LKG {lkg.timeline_id} does not match Timeline {TIMELINE.timeline_id}"
        )


class TestInvariantBaselineImmutability:
    def test_validate_baseline_immutability_accepts_identical_baseline(self) -> None:

        # Should not raise an exception since the proposed baseline is identical to the existing one
        validate_baseline_immutability(BASELINE, BASELINE)

    def test_validate_baseline_immutability_rejects_changed_baseline(self) -> None:

        modified_baseline = Baseline(
            timeline_id=BASELINE.timeline_id,
            source_run_id="train-run-2",
            model_identity=BASELINE.model_identity,
            parameter_fingerprint="params-v2",  # changed parameter fingerprint to trigger violation
            data_snapshot_ref=BASELINE.data_snapshot_ref,
            run_config_ref=BASELINE.run_config_ref,
            metric_snapshot=BASELINE.metric_snapshot,
            environment_context=BASELINE.environment_context,
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_baseline_immutability(BASELINE, modified_baseline)

        error = exc_info.value
        assert error.code == "baseline_immutability_violation"
        assert "source_run_id, parameter_fingerprint" == error.field


class TestInvariantLKGMembership:
    def test_lkg_membership_valid(self) -> None:
        """LKG with matching timeline_id and monitoring_run_id should pass validation."""

        validate_lkg_membership(TIMELINE, LAST_KNOWN_GOOD)

    def test_lkg_not_in_timeline_invalid(self) -> None:
        """LKG with non-matching timeline_id or monitoring_run_id should raise InvariantViolation."""  # noqa: E501

        different_timeline_lkg = LKG(timeline_id="timeline-2", monitoring_run_id="monitoring-run-1")

        with pytest.raises(InvariantViolation) as exc_info:
            validate_lkg_membership(TIMELINE, different_timeline_lkg)

        error = exc_info.value
        assert error.code == "lkg_membership_violation"
        assert error.field == "timeline_id"
        assert error.entity == "LKG"
        assert (
            error.message == f"LKG {different_timeline_lkg} does not belong to Timeline {TIMELINE}"
        )

    def test_lkg_in_timeline_but_not_in_runs_invalid(self) -> None:
        """LKG with matching timeline_id but in monitoring_run_ids should raise InvariantViolation."""  # noqa: E501

        nonmember_lkg = LKG(timeline_id="timeline-1", monitoring_run_id="monitoring-run-3")

        with pytest.raises(InvariantViolation) as exc_info:
            validate_lkg_membership(TIMELINE, nonmember_lkg)

        error = exc_info.value
        assert error.code == "lkg_membership_violation"
        assert error.field == "monitoring_run_id"
        assert error.entity == "LKG"
        assert error.message == f"LKG {nonmember_lkg} does not belong to Timeline {TIMELINE}"

    def test_lkg_in_timeline_but_not_active_lkg_invalid(self) -> None:
        """LKG matching timeline_id but monitoring_run_id not active lkg should raise InvariantViolation."""  # noqa: E501

        non_active_lkg = LKG(timeline_id="timeline-1", monitoring_run_id="monitoring-run-2")

        with pytest.raises(InvariantViolation) as exc_info:
            validate_lkg_membership(TIMELINE, non_active_lkg)

        error = exc_info.value
        assert error.code == "lkg_membership_violation"
        assert error.field == "active_lkg_monitoring_run_id"
        assert error.entity == "LKG"
        assert error.message == f"LKG {non_active_lkg} does not belong to Timeline {TIMELINE}"


class TestInvariantFindingToDiffEvidence:
    def test_finding_to_diff_evidence_valid(self) -> None:
        """Finding with all evidence diff_ids present in the timeline should pass validation."""

        validate_finding_to_diff_evidence(FINDING, DIFF)

    def test_finding_to_diff_evidence_different_monitoring_run_id_invalid(self) -> None:
        """Diff with monitoring_run_id different from Finding's monitoring_run_id should raise InvariantViolation."""  # noqa: E501

        mismatch_monitoring_run_id_diff = Diff(
            diff_id="diff-1",
            monitoring_run_id="monitoring-run-3",
            reference_monitoring_run_id="monitoring-run-3",
            reference_kind=DiffReferenceKind.BASELINE,
            metric_deltas={"kl": -0.05},
            metadata={"feature": "age"},
        )

        monitoring_run_id = FINDING.monitoring_run_id

        with pytest.raises(InvariantViolation) as exc_info:
            validate_finding_to_diff_evidence(FINDING, mismatch_monitoring_run_id_diff)

        error = exc_info.value
        assert error.code == "finding_diff_evidence_violation"
        assert error.field == "monitoring_run_id"
        assert error.entity == "Diff"
        assert (
            error.message
            == f"Diff monitoring_run_id {mismatch_monitoring_run_id_diff.monitoring_run_id} "
            f"does not match Finding monitoring_run_id {monitoring_run_id}"
        )

    def test_finding_to_diff_evidence_diff_id_not_in_evidence_invalid(self) -> None:
        """Diff with diff_id not in Finding's evidence_diff_ids should raise InvariantViolation."""

        non_evidence_diff = Diff(
            diff_id="diff-3",  # diff_id not in FINDING.evidence_diff_ids to trigger violation
            monitoring_run_id="monitoring-run-2",
            reference_monitoring_run_id="monitoring-run-1",
            reference_kind=DiffReferenceKind.BASELINE,
            metric_deltas={"kl": -0.05},
            metadata={"feature": "age"},
        )

        diff_ids = [diff_id for diff_id in FINDING.evidence_diff_ids]

        with pytest.raises(InvariantViolation) as exc_info:
            validate_finding_to_diff_evidence(FINDING, non_evidence_diff)

        error = exc_info.value
        assert error.code == "finding_diff_evidence_violation"
        assert error.field == "diff_id"
        assert error.entity == "Diff"
        assert (
            error.message
            == f"Diff diff_id {non_evidence_diff.diff_id} is not in Finding evidence {diff_ids}"
        )


class TestInvariantContractCheckResult:
    def test_contract_check_result_accepts_pass_without_reasons(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=(),
        )

        validate_contract_check_result(result)

    def test_contract_check_result_accepts_warn_with_environment_mismatch(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
            ),
        )

        validate_contract_check_result(result)

    def test_contract_check_result_accepts_fail_with_blocking_reason(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.FAIL,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Data schema does not match the baseline.",
                    blocking=True,
                ),
            ),
        )

        validate_contract_check_result(result)

    def test_contract_check_result_rejects_unknown_status(self) -> None:
        result = ContractCheckResult(
            status="unknown_status",  # type: ignore
            reasons=(),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_unknown"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_pass_with_reasons(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.PASS,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_warn_with_blocking_reason(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Data schema does not match the baseline.",
                    blocking=True,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_warn_with_no_non_blocking_reason(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_fail_with_only_non_blocking_reasons(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.FAIL,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_status_reason_mismatch"
        assert error.entity == "ContractCheckResult"
        assert error.field == "status"

    def test_contract_check_result_rejects_unknown_reason_code(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="metric_mismatch",
                    message="Metric definition differs.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_code_unknown"
        assert error.entity == "ContractCheckReason"
        assert error.field == "code"

    def test_contract_check_result_rejects_environment_mismatch_with_blocking_flag(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Python version differs.",
                    blocking=True,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_blocking_mismatch"
        assert error.entity == "ContractCheckReason"
        assert error.field == "blocking"

    def test_contract_check_result_rejects_schema_mismatch_without_blocking_flag(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.FAIL,
            reasons=(
                ContractCheckReason(
                    code="schema_mismatch",
                    message="Schema differs.",
                    blocking=False,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_blocking_mismatch"
        assert error.entity == "ContractCheckReason"
        assert error.field == "blocking"
