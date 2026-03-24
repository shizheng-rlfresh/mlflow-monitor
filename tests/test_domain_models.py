"""Unit tests for domain models in mlflow_monitor."""

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


def test_canonical_entities_can_be_constructed() -> None:
    """Test that domain entities can be constructed with expected fields and types."""
    contract = Contract(
        contract_id="default",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )
    baseline = Baseline(
        timeline_id="timeline-1",
        source_run_id="train-run-1",
        model_identity="model-a",
        parameter_fingerprint="params-v1",
        data_snapshot_ref="dataset-2026-03-01",
        run_config_ref="config-v1",
        metric_snapshot={"f1": 0.87},
        environment_context={"python": "3.12"},
    )
    timeline = Timeline(
        timeline_id="timeline-1",
        subject_id="churn_model",
        monitoring_namespace="mlflow_monitor/churn_model",
        baseline=baseline,
        monitoring_run_ids=["monitoring-run-1"],
        active_lkg_monitoring_run_id="monitoring-run-1",
        active_contract=contract,
    )
    contract_check = ContractCheckResult(
        status=ComparabilityStatus.WARN,
        reasons=(
            ContractCheckReason(
                code="environment_mismatch",
                message="Python minor version differs",
                blocking=False,
            ),
        ),
    )
    diff = Diff(
        diff_id="diff-1",
        monitoring_run_id="monitoring-run-1",
        reference_monitoring_run_id="monitoring-run-0",
        reference_kind=DiffReferenceKind.BASELINE,
        metric_deltas={"f1": -0.02},
        metadata={"window": "full"},
    )
    finding = Finding(
        finding_id="finding-1",
        monitoring_run_id="monitoring-run-1",
        severity=FindingSeverity.HIGH,
        category="performance_regression",
        summary="F1 regressed against baseline",
        evidence_diff_ids=("diff-1",),
        recommendation="Investigate feature changes before promotion.",
    )
    run = Run(
        monitoring_run_id="monitoring-run-1",
        timeline_id="timeline-1",
        sequence_index=0,
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id="train-run-1",
        contract=contract,
        lifecycle_status=LifecycleStatus.CLOSED,
        comparability_status=ComparabilityStatus.WARN,
        contract_check_result=contract_check,
        diff_ids=("diff-1",),
        finding_ids=("finding-1",),
    )
    lkg = LKG(timeline_id="timeline-1", monitoring_run_id="monitoring-run-1")

    assert timeline.baseline.source_run_id == "train-run-1"
    assert run.contract_check_result is not None
    assert run.contract_check_result.status is ComparabilityStatus.WARN
    assert diff.reference_kind is DiffReferenceKind.BASELINE
    assert finding.evidence_diff_ids == ("diff-1",)
    assert lkg.monitoring_run_id == "monitoring-run-1"


def test_status_vocabularies_are_fixed() -> None:
    """Test that enum vocabularies have expected values."""
    assert {status.value for status in LifecycleStatus} == {
        "created",
        "prepared",
        "checked",
        "analyzed",
        "closed",
        "failed",
    }
    assert {status.value for status in ComparabilityStatus} == {"pass", "warn", "fail"}


def test_relationship_shapes_match_cast() -> None:
    """Test that related entities can be associated with correct field types."""
    contract = Contract(
        contract_id="default",
        version="v0",
        schema_contract_ref=None,
        feature_contract_ref=None,
        metric_contract_ref=None,
        data_scope_contract_ref=None,
        execution_contract_ref=None,
    )
    baseline = Baseline(
        timeline_id="timeline-1",
        source_run_id="train-run-1",
        model_identity="model-a",
        parameter_fingerprint="params-v1",
        data_snapshot_ref="dataset-2026-03-01",
        run_config_ref="config-v1",
        metric_snapshot={},
        environment_context={},
    )
    timeline = Timeline(
        timeline_id="timeline-1",
        subject_id="churn_model",
        monitoring_namespace="mlflow_monitor/churn_model",
        baseline=baseline,
        monitoring_run_ids=["monitoring-run-1", "monitoring-run-2"],
        active_lkg_monitoring_run_id=None,
        active_contract=contract,
    )
    run = Run(
        monitoring_run_id="monitoring-run-1",
        timeline_id=timeline.timeline_id,
        sequence_index=0,
        subject_id=timeline.subject_id,
        source_run_id="train-run-2",
        baseline_source_run_id=baseline.source_run_id,
        contract=contract,
        lifecycle_status=LifecycleStatus.CREATED,
        comparability_status=ComparabilityStatus.PASS,
        contract_check_result=None,
        diff_ids=(),
        finding_ids=(),
    )

    assert run.timeline_id == timeline.timeline_id
    assert timeline.active_contract.contract_id == "default"
    assert timeline.monitoring_run_ids == ["monitoring-run-1", "monitoring-run-2"]


def test_finding_references_one_or_more_diffs() -> None:
    """Test that a Finding can reference one or more Diff records."""
    finding = Finding(
        finding_id="finding-1",
        monitoring_run_id="monitoring-run-1",
        severity=FindingSeverity.MEDIUM,
        category="quality",
        summary="Regression detected",
        evidence_diff_ids=("diff-1", "diff-2"),
        recommendation="Review the latest run.",
    )

    assert finding.evidence_diff_ids == ("diff-1", "diff-2")


def test_baseline_carries_snapshot_context() -> None:
    """Test that Baseline can carry snapshot context for metrics and environment."""
    baseline = Baseline(
        timeline_id="timeline-1",
        source_run_id="train-run-1",
        model_identity="model-a",
        parameter_fingerprint="params-v1",
        data_snapshot_ref="dataset-2026-03-01",
        run_config_ref="config-v1",
        metric_snapshot={"precision": 0.91},
        environment_context={"python": "3.12", "sklearn": "1.7"},
    )

    assert baseline.metric_snapshot["precision"] == 0.91
    assert baseline.environment_context["sklearn"] == "1.7"


def test_baseline_snapshot_mappings_are_immutable() -> None:
    """Test that Baseline snapshot mappings are immutable after construction."""
    metric_snapshot = {"f1": 0.87}
    environment_context = {"python": "3.12"}
    baseline = Baseline(
        timeline_id="timeline-1",
        source_run_id="train-run-1",
        model_identity="model-a",
        parameter_fingerprint="params-v1",
        data_snapshot_ref="dataset-2026-03-01",
        run_config_ref="config-v1",
        metric_snapshot=metric_snapshot,
        environment_context=environment_context,
    )

    metric_snapshot["f1"] = 0.0
    environment_context["python"] = "3.11"

    assert baseline.metric_snapshot["f1"] == 0.87
    assert baseline.environment_context["python"] == "3.12"

    # Intentionally trigger a type error to verify immutability
    try:
        baseline.metric_snapshot["f1"] = 0.0  # pyright: ignore[reportIndexIssue]
    except TypeError:
        pass
    else:
        msg = "expected baseline metric snapshot to reject mutation"
        raise AssertionError(msg)
