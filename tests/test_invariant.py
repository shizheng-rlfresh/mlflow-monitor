# i probably will write a few tests
# 1. timeline ownership checks
# 2. baseline immutability checks
# 3. lkg in same timeline checks
# finding-to-diff evidence checks


import pytest

from mlflow_monitor.domain import (
    LKG,
    Baseline,
    ComparabilityStatus,
    Contract,
    LifecycleStatus,
    Run,
    Timeline,
)
from mlflow_monitor.errors import InvariantViolation
from mlflow_monitor.invariant import (
    validate_baseline_immutability,
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
    run_id="run-1",
)

TIMELINE = Timeline(
    timeline_id="timeline-1",
    subject_id="churn_model",
    monitoring_namespace="mlflow_monitor/churn_model",
    baseline=BASELINE,
    run_ids=["run-1", "run-2"],
    active_lkg_run_id="run-1",
    active_contract=CONTRACT,
)

RUN = Run(
    run_id="run-1",
    timeline_id="timeline-1",
    sequence_index=0,
    subject_id="churn_model",
    source_run_id="train-run-1",
    baseline_source_run_id="train-run-1",
    contract=CONTRACT,
    lifecycle_status=LifecycleStatus.CLOSED,
    comparability_status=ComparabilityStatus.PASS,
    contract_check_result=None,
    diff_ids=(),
    finding_ids=(),
)


class TestInvariantTimelineOwnership:
    def test_valid_timeline_ownership(self) -> None:
        validate_timeline_ownership(TIMELINE, baseline=BASELINE, lkg=LAST_KNOWN_GOOD, runs=[RUN])

    def test_timeline_run_ownership(self) -> None:
        run = Run(
            run_id="run-2",
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
            run_id="run-1",
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
        """LKG with matching timeline_id and run_id should pass validation."""

        validate_lkg_membership(TIMELINE, LAST_KNOWN_GOOD)

    def test_lkg_not_in_timeline_invalid(self) -> None:
        """LKG with non-matching timeline_id or run_id should raise InvariantViolation."""

        different_timeline_lkg = LKG(timeline_id="timeline-2", run_id="run-1")
        nonmember_lkg = LKG(timeline_id="timeline-1", run_id="run-2")

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
        """LKG with matching timeline_id but in run_ids should raise InvariantViolation."""

        nonmember_lkg = LKG(timeline_id="timeline-1", run_id="run-3")

        with pytest.raises(InvariantViolation) as exc_info:
            validate_lkg_membership(TIMELINE, nonmember_lkg)

        error = exc_info.value
        assert error.code == "lkg_membership_violation"
        assert error.field == "run_id"
        assert error.entity == "LKG"
        assert error.message == f"LKG {nonmember_lkg} does not belong to Timeline {TIMELINE}"

    def test_lkg_in_timeline_but_not_active_lkg_invalid(self) -> None:
        """LKG matching timeline_id but run_id not active lkg should raise InvariantViolation."""

        non_active_lkg = LKG(timeline_id="timeline-1", run_id="run-2")

        with pytest.raises(InvariantViolation) as exc_info:
            validate_lkg_membership(TIMELINE, non_active_lkg)

        error = exc_info.value
        assert error.code == "lkg_membership_violation"
        assert error.field == "active_lkg_run_id"
        assert error.entity == "LKG"
        assert error.message == f"LKG {non_active_lkg} does not belong to Timeline {TIMELINE}"
