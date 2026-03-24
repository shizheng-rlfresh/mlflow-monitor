"""Unit tests for the monitor run result contract."""

import pytest

from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult


def test_monitor_run_result_success_envelope_construction() -> None:
    """Success envelope should construct with required canonical fields."""
    result = MonitorRunResult(
        monitoring_run_id="monitoring-run-1",
        subject_id="churn_model",
        timeline_id="timeline-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        comparability_status=ComparabilityStatus.WARN,
        summary={"status": "ok"},
        finding_ids=("finding-1",),
        diff_ids=("diff-1",),
        reference_run_ids={"baseline": "train-run-1"},
    )

    assert result.monitoring_run_id == "monitoring-run-1"
    assert result.subject_id == "churn_model"
    assert result.timeline_id == "timeline-1"
    assert result.lifecycle_status is LifecycleStatus.CLOSED
    assert result.comparability_status is ComparabilityStatus.WARN
    assert result.summary == {"status": "ok"}
    assert result.finding_ids == ("finding-1",)
    assert result.diff_ids == ("diff-1",)
    assert result.reference_run_ids == {"baseline": "train-run-1"}
    assert result.error is None


def test_monitor_run_result_failure_envelope_construction() -> None:
    """Failure envelope should construct with structured error details."""
    error = MonitorRunError(
        code="prepare_missing_baseline",
        message="Missing baseline_source_run_id on first run.",
        stage="prepare",
        details={"subject_id": "churn_model"},
    )
    result = MonitorRunResult(
        monitoring_run_id="monitoring-run-2",
        subject_id="churn_model",
        timeline_id=None,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        reference_run_ids={},
        error=error,
    )

    assert result.lifecycle_status is LifecycleStatus.FAILED
    assert result.comparability_status is None
    assert result.error is error


def test_monitor_run_result_to_dict_serializes_enums_and_error() -> None:
    """to_dict should serialize enum statuses and nested error payload."""
    error = MonitorRunError(
        code="check_runtime_error",
        message="Unhandled checker exception.",
        stage="check",
        details={"checker": "default"},
    )
    result = MonitorRunResult(
        monitoring_run_id="monitoring-run-3",
        subject_id="fraud_model",
        timeline_id="timeline-3",
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=ComparabilityStatus.FAIL,
        summary={"outcome": "failed"},
        finding_ids=("finding-2",),
        diff_ids=("diff-2",),
        reference_run_ids={"baseline": "train-run-2"},
        error=error,
    )

    serialized = result.to_dict()

    assert serialized["lifecycle_status"] == "failed"
    assert serialized["comparability_status"] == "fail"
    assert serialized["error"] == {
        "code": "check_runtime_error",
        "message": "Unhandled checker exception.",
        "stage": "check",
        "details": {"checker": "default"},
    }


def test_monitor_run_result_to_dict_stable_keys_for_success_and_failure() -> None:
    """to_dict should emit the same top-level keys for success and failure."""
    success = MonitorRunResult(
        monitoring_run_id="monitoring-run-success",
        subject_id="churn_model",
        timeline_id="timeline-1",
        lifecycle_status=LifecycleStatus.CLOSED,
        comparability_status=ComparabilityStatus.PASS,
        summary={"status": "ok"},
        finding_ids=(),
        diff_ids=(),
        reference_run_ids={},
    )
    failure = MonitorRunResult(
        monitoring_run_id="monitoring-run-failure",
        subject_id="churn_model",
        timeline_id=None,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        reference_run_ids={},
        error=MonitorRunError(
            code="persist_error",
            message="Persistence write failed.",
            stage="close",
        ),
    )

    success_keys = set(success.to_dict().keys())
    failure_keys = set(failure.to_dict().keys())

    assert success_keys == failure_keys


def test_monitor_run_result_failed_requires_error() -> None:
    """Failed lifecycle status should require a structured error payload."""
    with pytest.raises(
        ValueError,
        match="lifecycle_status=failed requires a non-null error",
    ):
        MonitorRunResult(
            monitoring_run_id="monitoring-run-failed",
            subject_id="churn_model",
            timeline_id=None,
            lifecycle_status=LifecycleStatus.FAILED,
            comparability_status=None,
            summary=None,
            finding_ids=(),
            diff_ids=(),
            reference_run_ids={},
            error=None,
        )


def test_monitor_run_result_non_failed_forbids_error() -> None:
    """Non-failed lifecycle statuses should not include structured error payload."""
    with pytest.raises(
        ValueError,
        match="non-failed lifecycle_status must have error=None",
    ):
        MonitorRunResult(
            monitoring_run_id="monitoring-run-checked",
            subject_id="churn_model",
            timeline_id="timeline-1",
            lifecycle_status=LifecycleStatus.CHECKED,
            comparability_status=ComparabilityStatus.WARN,
            summary={"status": "warn"},
            finding_ids=(),
            diff_ids=(),
            reference_run_ids={},
            error=MonitorRunError(
                code="unexpected_error",
                message="Should not be set for non-failed lifecycle states.",
            ),
        )


def test_monitor_run_error_details_are_immutable_after_construction() -> None:
    """Error details should be copied defensively and immutable on the dataclass."""
    details = {"stage": "prepare"}
    error = MonitorRunError(
        code="prepare_error",
        message="Preparation failed.",
        details=details,
    )

    details["stage"] = "mutated"
    assert error.details == {"stage": "prepare"}

    with pytest.raises(TypeError):
        error.details["stage"] = "x"  # type: ignore[index]


def test_monitor_run_result_collections_are_immutable_after_construction() -> None:
    """Result collection fields should be copied defensively and immutable."""
    summary = {"status": "ok"}
    reference_run_ids = {"baseline": "train-run-1"}
    finding_ids = ["finding-1"]
    diff_ids = ["diff-1"]
    result = MonitorRunResult(
        monitoring_run_id="monitoring-run-immutability",
        subject_id="churn_model",
        timeline_id="timeline-1",
        lifecycle_status=LifecycleStatus.CHECKED,
        comparability_status=ComparabilityStatus.WARN,
        summary=summary,
        finding_ids=tuple(finding_ids),
        diff_ids=tuple(diff_ids),
        reference_run_ids=reference_run_ids,
    )

    summary["status"] = "mutated"
    reference_run_ids["baseline"] = "mutated"
    finding_ids.append("finding-2")
    diff_ids.append("diff-2")

    assert result.summary == {"status": "ok"}
    assert result.reference_run_ids == {"baseline": "train-run-1"}
    assert result.finding_ids == ("finding-1",)
    assert result.diff_ids == ("diff-1",)

    with pytest.raises(TypeError):
        result.summary["status"] = "x"  # type: ignore[index]
    with pytest.raises(TypeError):
        result.reference_run_ids["baseline"] = "x"  # type: ignore[index]
