"""Unit tests for domain models in mlflow_monitor."""

import pytest

from mlflow_monitor.domain import (
    Baseline,
    ComparabilityStatus,
    Contract,
    ContractCheckReason,
    ContractCheckResult,
    Diff,
    DiffReference,
    DiffReferenceKind,
    Finding,
    FindingSeverity,
    LifecycleStatus,
    LKGSelection,
    MonitoringRunReference,
    Run,
    Timeline,
    TimelineEntry,
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
        source_run_id="train-run-2",
        reference=DiffReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id="train-run-0",
        ),
        metric_name="f1",
        current_value=0.75,
        reference_value=0.5,
        delta=0.25,
    )
    finding = Finding(
        finding_id="finding-1",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-2",
        finding_policy_id="relative-regression",
        finding_policy_version="1",
        finding_rule_id="quality.f1_regression",
        severity=FindingSeverity.HIGH,
        category="performance_regression",
        summary="F1 regressed against baseline",
        recommendation="Investigate feature changes before promotion.",
        evidence_diff_ids=("diff-1",),
        evidence_compatibility_ids=(),
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
    timeline_entry = TimelineEntry(
        monitoring_run_id=run.monitoring_run_id,
        source_run_id=run.source_run_id,
        sequence_index=run.sequence_index,
        lifecycle_status=run.lifecycle_status,
        comparability_status=run.comparability_status,
    )
    timeline = Timeline(
        timeline_id="timeline-1",
        subject_id="churn_model",
        baseline_source_run_id=baseline.source_run_id,
        entries=(timeline_entry,),
    )
    lkg_selection = LKGSelection(
        lkg_selection_id="lkg-selection-1",
        timeline_id="timeline-1",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-2",
        supersedes_lkg_selection_ids=(),
    )

    assert timeline.baseline_source_run_id == "train-run-1"
    assert run.contract_check_result is not None
    assert run.contract_check_result.status is ComparabilityStatus.WARN
    assert diff.reference.kind is DiffReferenceKind.BASELINE
    assert finding.evidence_diff_ids == ("diff-1",)
    assert lkg_selection.monitoring_run_id == "monitoring-run-1"


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
    run = Run(
        monitoring_run_id="monitoring-run-1",
        timeline_id="timeline-1",
        sequence_index=0,
        subject_id="churn_model",
        source_run_id="train-run-2",
        baseline_source_run_id=baseline.source_run_id,
        contract=contract,
        lifecycle_status=LifecycleStatus.CLOSED,
        comparability_status=ComparabilityStatus.PASS,
        contract_check_result=None,
        diff_ids=(),
        finding_ids=(),
    )
    timeline_entry = TimelineEntry(
        monitoring_run_id=run.monitoring_run_id,
        source_run_id=run.source_run_id,
        sequence_index=run.sequence_index,
        lifecycle_status=run.lifecycle_status,
        comparability_status=run.comparability_status,
    )
    timeline = Timeline(
        timeline_id="timeline-1",
        subject_id="churn_model",
        baseline_source_run_id=baseline.source_run_id,
        entries=(timeline_entry,),
    )

    assert run.timeline_id == timeline.timeline_id
    assert timeline.entries == (timeline_entry,)


def test_finding_references_one_or_more_diffs() -> None:
    """Test that a Finding can reference one or more Diff records."""
    finding = Finding(
        finding_id="finding-1",
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-2",
        finding_policy_id="relative-regression",
        finding_policy_version="1",
        finding_rule_id="quality.regression",
        severity=FindingSeverity.MEDIUM,
        category="quality",
        summary="Regression detected",
        recommendation="Review the latest run.",
        evidence_diff_ids=("diff-1", "diff-2"),
        evidence_compatibility_ids=(),
    )

    assert finding.evidence_diff_ids == ("diff-1", "diff-2")


def test_monitoring_run_reference_serializes_paired_identity() -> None:
    """Monitoring-run references should expose explicit paired identity fields."""
    reference = MonitoringRunReference(
        kind=DiffReferenceKind.PREVIOUS,
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
    )

    assert reference.to_dict() == {
        "kind": "previous",
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-1",
    }


@pytest.mark.parametrize("reference_type", [MonitoringRunReference, DiffReference])
def test_baseline_reference_requires_source_only(reference_type: type) -> None:
    """Baseline references should carry the baseline source without a monitoring run."""
    reference = reference_type(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id="train-run-baseline",
    )

    assert reference.monitoring_run_id is None
    assert reference.source_run_id == "train-run-baseline"

    with pytest.raises(ValueError, match="must not set monitoring_run_id"):
        reference_type(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id="monitoring-run-baseline",
            source_run_id="train-run-baseline",
        )


@pytest.mark.parametrize("reference_type", [MonitoringRunReference, DiffReference])
@pytest.mark.parametrize(
    "kind",
    [DiffReferenceKind.PREVIOUS, DiffReferenceKind.LKG, DiffReferenceKind.CUSTOM],
)
def test_monitoring_reference_kinds_require_paired_identity(
    reference_type: type,
    kind: DiffReferenceKind,
) -> None:
    """Monitoring-run-backed references should require both immutable IDs."""
    with pytest.raises(ValueError, match="requires a non-empty monitoring_run_id"):
        reference_type(
            kind=kind,
            monitoring_run_id=None,
            source_run_id="train-run-1",
        )

    with pytest.raises(ValueError, match="requires a non-empty source_run_id"):
        reference_type(
            kind=kind,
            monitoring_run_id="monitoring-run-1",
            source_run_id="",
        )


def test_diff_requires_source_run_id_for_baseline_reference() -> None:
    """Baseline diff references should require a concrete source run id."""
    with pytest.raises(ValueError, match="requires a non-empty source_run_id"):
        DiffReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id="",
        )


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


def test_lkg_selection_supersession_should_not_contain_lkg_selection_id() -> None:
    """Test that an LKG selection cannot supersede itself."""
    with pytest.raises(ValueError, match="cannot supersede itself"):
        LKGSelection(
            lkg_selection_id="lkg-selection-1",
            timeline_id="timeline-1",
            monitoring_run_id="monitoring-run-1",
            source_run_id="train-run-2",
            supersedes_lkg_selection_ids=("lkg-selection-1",),
        )


def test_timeline_with_null_baseline_cannot_accept_closed_entries() -> None:
    """Test that a Timeline with a null baseline cannot accept closed entries."""

    closed_timeline_entry = TimelineEntry(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        sequence_index=0,
        lifecycle_status=LifecycleStatus.CLOSED,
        comparability_status=ComparabilityStatus.FAIL,
    )

    failed_timeline_entry = TimelineEntry(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-1",
        sequence_index=1,
        lifecycle_status=LifecycleStatus.FAILED,
        comparability_status=ComparabilityStatus.FAIL,
    )

    with pytest.raises(
        ValueError, match="cannot accept closed entries without a baseline_source_run_id"
    ):
        Timeline(
            timeline_id="timeline-1",
            subject_id="churn_model",
            baseline_source_run_id=None,
            entries=(closed_timeline_entry, failed_timeline_entry),
        )
