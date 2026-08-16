"""Unit tests for invariants in mlflow_monitor."""

import pytest

from mlflow_monitor.domain import (
    Baseline,
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
    LifecycleStatus,
    Run,
    Timeline,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.invariant import (
    validate_baseline_immutability,
    validate_contract_check_result,
    validate_timeline_ownership,
)

CONTRACT = Contract(
    contract_id="default",
    contract_version="v0",
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

TIMELINE = Timeline(
    timeline_id="timeline-1",
    subject_id="churn_model",
    baseline_source_run_id=BASELINE.source_run_id,
    entries=(),
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


class TestInvariantTimelineOwnership:
    def test_valid_timeline_ownership(self) -> None:
        validate_timeline_ownership(TIMELINE, baseline=BASELINE, runs=[RUN])

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
            validate_timeline_ownership(TIMELINE, baseline=None, runs=[run])

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
            validate_timeline_ownership(TIMELINE, baseline=baseline, runs=None)

        error = exc_info.value
        assert error.code == "baseline_timeline_mismatch"
        assert error.field == "timeline_id"
        assert error.entity == "Baseline"
        assert (
            error.message
            == f"Baseline {baseline.timeline_id} does not match Timeline {TIMELINE.timeline_id}"
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

    def test_contract_check_result_rejects_duplicate_reason_codes(self) -> None:
        reason = ContractCheckReason(
            code="environment_mismatch",
            message="Execution environment does not match the baseline.",
            blocking=False,
        )
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(reason, reason),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        error = exc_info.value
        assert error.code == "contract_check_reason_code_duplicate"
        assert error.entity == "ContractCheckResult"
        assert error.field == "reasons"

    def test_duplicate_reason_code_takes_precedence_over_changed_content(self) -> None:
        result = ContractCheckResult(
            status=ComparabilityStatus.WARN,
            reasons=(
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Execution environment does not match the baseline.",
                    blocking=False,
                ),
                ContractCheckReason(
                    code="environment_mismatch",
                    message="Changed duplicate content.",
                    blocking=True,
                ),
            ),
        )

        with pytest.raises(InvariantViolation) as exc_info:
            validate_contract_check_result(result)

        assert exc_info.value.code == "contract_check_reason_code_duplicate"

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
