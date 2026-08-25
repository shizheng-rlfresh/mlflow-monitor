"""Specifications for pure atomic Diff and coverage computation."""

import math

import pytest

from mlflow_monitor.differ import ComputedDiffCoverage, compute_diffs_and_coverage
from mlflow_monitor.domain import (
    Diff,
    DiffReference,
    DiffReferenceKind,
    MetricComparisonUnavailable,
    MonitoringRunReference,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
from mlflow_monitor.identity import make_diff_id
from mlflow_monitor.workflow import PreparedReferencePlanEntry

MONITORING_RUN_ID = "monitoring-run-current"
SOURCE_RUN_ID = "train-run-current"
DIFF_ID_FIXTURE = "diff-v1-6b7105add110bd5993e8eb644e7231d49b10ac8e638a454dd9020befcec736b7"
METRIC_NAMES = ("accuracy", "precision")
CURRENT_METRICS = {"precision": 0.875, "accuracy": 0.75}
REFERENCE_METRICS_BY_SOURCE_RUN_ID = {
    "train-run-custom": {"precision": 0.375, "accuracy": 0.25},
    "train-run-lkg": {"precision": 0.8125, "accuracy": 0.8125},
    "train-run-previous": {"precision": 0.75, "accuracy": 0.625},
    "train-run-baseline": {"precision": 0.625, "accuracy": 0.5},
}


def _resolved_reference_plan() -> tuple[PreparedReferencePlanEntry, ...]:
    return (
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.BASELINE,
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.BASELINE,
                monitoring_run_id=None,
                source_run_id="train-run-baseline",
            ),
            unavailable_reason=None,
        ),
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.PREVIOUS,
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.PREVIOUS,
                monitoring_run_id="monitoring-run-previous",
                source_run_id="train-run-previous",
            ),
            unavailable_reason=None,
        ),
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.LKG,
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.LKG,
                monitoring_run_id="monitoring-run-lkg",
                source_run_id="train-run-lkg",
            ),
            unavailable_reason=None,
        ),
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.CUSTOM,
            reference=MonitoringRunReference(
                kind=DiffReferenceKind.CUSTOM,
                monitoring_run_id="monitoring-run-custom",
                source_run_id="train-run-custom",
            ),
            unavailable_reason=None,
        ),
    )


def _diff_reference(reference: MonitoringRunReference) -> DiffReference:
    return DiffReference(
        kind=reference.kind,
        monitoring_run_id=reference.monitoring_run_id,
        source_run_id=reference.source_run_id,
    )


def _expected_diffs(
    reference_plan: tuple[PreparedReferencePlanEntry, ...],
) -> tuple[Diff, ...]:
    expected: list[Diff] = []
    for plan_entry in reference_plan:
        assert plan_entry.reference is not None
        reference = _diff_reference(plan_entry.reference)
        assert reference.source_run_id is not None
        reference_metrics = REFERENCE_METRICS_BY_SOURCE_RUN_ID[reference.source_run_id]
        for metric_name in METRIC_NAMES:
            current_value = CURRENT_METRICS[metric_name]
            reference_value = reference_metrics[metric_name]
            expected.append(
                Diff(
                    diff_id=make_diff_id(
                        monitoring_run_id=MONITORING_RUN_ID,
                        source_run_id=SOURCE_RUN_ID,
                        reference=reference,
                        metric_name=metric_name,
                    ),
                    monitoring_run_id=MONITORING_RUN_ID,
                    source_run_id=SOURCE_RUN_ID,
                    reference=reference,
                    metric_name=metric_name,
                    current_value=current_value,
                    reference_value=reference_value,
                    delta=current_value - reference_value,
                )
            )
    return tuple(expected)


def _expected_completed_coverage(
    reference_plan: tuple[PreparedReferencePlanEntry, ...],
    diffs: tuple[Diff, ...],
) -> tuple[ReferenceComparisonCoverage, ...]:
    expected: list[ReferenceComparisonCoverage] = []
    for plan_entry in reference_plan:
        assert plan_entry.reference is not None
        reference = _diff_reference(plan_entry.reference)
        expected.append(
            ReferenceComparisonCoverage(
                reference_kind=plan_entry.kind,
                reference=reference,
                status=ReferenceComparisonStatus.COMPLETED,
                diff_ids=tuple(diff.diff_id for diff in diffs if diff.reference == reference),
                metric_unavailability=(),
                reason=None,
            )
        )
    return tuple(expected)


def test_compute_diffs_and_coverage_materializes_atomic_diffs_and_completed_groups() -> None:
    reference_plan = _resolved_reference_plan()

    result: ComputedDiffCoverage = compute_diffs_and_coverage(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        metric_names=METRIC_NAMES,
        current_metrics=CURRENT_METRICS,
        reference_plan=reference_plan,
        reference_metrics_by_source_run_id=REFERENCE_METRICS_BY_SOURCE_RUN_ID,
    )

    expected_diffs = _expected_diffs(reference_plan)
    assert result.diffs == expected_diffs
    assert result.diffs[0].diff_id == DIFF_ID_FIXTURE
    assert tuple((diff.reference.kind, diff.metric_name) for diff in result.diffs) == (
        (DiffReferenceKind.BASELINE, "accuracy"),
        (DiffReferenceKind.BASELINE, "precision"),
        (DiffReferenceKind.PREVIOUS, "accuracy"),
        (DiffReferenceKind.PREVIOUS, "precision"),
        (DiffReferenceKind.LKG, "accuracy"),
        (DiffReferenceKind.LKG, "precision"),
        (DiffReferenceKind.CUSTOM, "accuracy"),
        (DiffReferenceKind.CUSTOM, "precision"),
    )
    assert result.coverages == _expected_completed_coverage(reference_plan, expected_diffs)


def test_compute_diffs_and_coverage_completes_resolved_groups_for_empty_selection() -> None:
    reference_plan = _resolved_reference_plan()
    empty_reference_metrics = {
        plan_entry.reference.source_run_id: {}
        for plan_entry in reference_plan
        if plan_entry.reference is not None
    }

    result: ComputedDiffCoverage = compute_diffs_and_coverage(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        metric_names=(),
        current_metrics={},
        reference_plan=reference_plan,
        reference_metrics_by_source_run_id=empty_reference_metrics,
    )

    assert result.diffs == ()
    expected_coverage: list[ReferenceComparisonCoverage] = []
    for plan_entry in reference_plan:
        assert plan_entry.reference is not None
        expected_coverage.append(
            ReferenceComparisonCoverage(
                reference_kind=plan_entry.kind,
                reference=_diff_reference(plan_entry.reference),
                status=ReferenceComparisonStatus.COMPLETED,
                diff_ids=(),
                metric_unavailability=(),
                reason=None,
            )
        )
    assert result.coverages == tuple(expected_coverage)


@pytest.mark.parametrize(
    ("current_metrics", "reference_metrics", "expected_reason"),
    (
        pytest.param(
            {},
            {"accuracy": 0.5},
            "current_metric_missing",
            id="current_metric_missing",
        ),
        pytest.param(
            {"accuracy": 0.75},
            {},
            "reference_metric_missing",
            id="reference_metric_missing",
        ),
        pytest.param(
            {"accuracy": math.nan},
            {"accuracy": 0.5},
            "current_metric_not_finite",
            id="current_metric_not_finite",
        ),
        pytest.param(
            {"accuracy": 0.75},
            {"accuracy": math.inf},
            "reference_metric_not_finite",
            id="reference_metric_not_finite",
        ),
        pytest.param(
            {"accuracy": 1e308},
            {"accuracy": -1e308},
            "delta_not_finite",
            id="delta_not_finite",
        ),
    ),
)
def test_compute_diffs_and_coverage_records_metric_unavailability(
    current_metrics: dict[str, float],
    reference_metrics: dict[str, float],
    expected_reason: str,
) -> None:
    baseline_entry = _resolved_reference_plan()[0]
    assert baseline_entry.reference is not None
    reference = _diff_reference(baseline_entry.reference)

    result = compute_diffs_and_coverage(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        metric_names=("accuracy",),
        current_metrics=current_metrics,
        reference_plan=(baseline_entry,),
        reference_metrics_by_source_run_id={
            baseline_entry.reference.source_run_id: reference_metrics
        },
    )

    assert result == ComputedDiffCoverage(
        diffs=(),
        coverages=(
            ReferenceComparisonCoverage(
                reference_kind=DiffReferenceKind.BASELINE,
                reference=reference,
                status=ReferenceComparisonStatus.COMPLETED,
                diff_ids=(),
                metric_unavailability=(
                    MetricComparisonUnavailable(
                        metric_name="accuracy",
                        reason=expected_reason,
                    ),
                ),
                reason=None,
            ),
        ),
    )


def test_compute_diffs_and_coverage_uses_selected_names_without_intersecting_keys() -> None:
    baseline_entry = _resolved_reference_plan()[0]
    assert baseline_entry.reference is not None
    reference = _diff_reference(baseline_entry.reference)
    accuracy_diff_id = make_diff_id(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        reference=reference,
        metric_name="accuracy",
    )

    result = compute_diffs_and_coverage(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        metric_names=("accuracy", "precision", "recall"),
        current_metrics={
            "accuracy": 0.75,
            "precision": 0.875,
            "current-only-unselected": 1.0,
        },
        reference_plan=(baseline_entry,),
        reference_metrics_by_source_run_id={
            baseline_entry.reference.source_run_id: {
                "accuracy": 0.5,
                "recall": 0.625,
                "reference-only-unselected": 1.0,
            }
        },
    )

    assert result.diffs == (
        Diff(
            diff_id=accuracy_diff_id,
            monitoring_run_id=MONITORING_RUN_ID,
            source_run_id=SOURCE_RUN_ID,
            reference=reference,
            metric_name="accuracy",
            current_value=0.75,
            reference_value=0.5,
            delta=0.25,
        ),
    )
    assert result.coverages == (
        ReferenceComparisonCoverage(
            reference_kind=DiffReferenceKind.BASELINE,
            reference=reference,
            status=ReferenceComparisonStatus.COMPLETED,
            diff_ids=(accuracy_diff_id,),
            metric_unavailability=(
                MetricComparisonUnavailable(
                    metric_name="precision",
                    reason="reference_metric_missing",
                ),
                MetricComparisonUnavailable(
                    metric_name="recall",
                    reason="current_metric_missing",
                ),
            ),
            reason=None,
        ),
    )
    coverage = result.coverages[0]
    assert len(coverage.diff_ids) + len(coverage.metric_unavailability) == 3


def test_compute_diffs_and_coverage_prioritizes_current_missing_when_both_are_missing() -> None:
    baseline_entry = _resolved_reference_plan()[0]
    assert baseline_entry.reference is not None

    result = compute_diffs_and_coverage(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        metric_names=("accuracy",),
        current_metrics={},
        reference_plan=(baseline_entry,),
        reference_metrics_by_source_run_id={baseline_entry.reference.source_run_id: {}},
    )

    assert result.coverages[0].metric_unavailability == (
        MetricComparisonUnavailable(
            metric_name="accuracy",
            reason="current_metric_missing",
        ),
    )


@pytest.mark.parametrize(
    ("reference_kind", "unavailable_reason"),
    (
        pytest.param(
            DiffReferenceKind.PREVIOUS,
            "previous_reference_missing",
            id="previous_reference_missing",
        ),
        pytest.param(
            DiffReferenceKind.LKG,
            "lkg_not_selected",
            id="lkg_not_selected",
        ),
        pytest.param(
            DiffReferenceKind.LKG,
            "lkg_selection_inconsistent",
            id="lkg_selection_inconsistent",
        ),
    ),
)
def test_compute_diffs_and_coverage_materializes_prepare_time_unavailable_reference(
    reference_kind: DiffReferenceKind,
    unavailable_reason: str,
) -> None:
    reference_entry = PreparedReferencePlanEntry(
        kind=reference_kind,
        reference=None,
        unavailable_reason=unavailable_reason,
    )

    result = compute_diffs_and_coverage(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        metric_names=("accuracy",),
        current_metrics={"accuracy": 0.75},
        reference_plan=(reference_entry,),
        reference_metrics_by_source_run_id={},
    )

    assert result == ComputedDiffCoverage(
        diffs=(),
        coverages=(
            ReferenceComparisonCoverage(
                reference_kind=reference_kind,
                reference=None,
                status=ReferenceComparisonStatus.UNAVAILABLE,
                diff_ids=(),
                metric_unavailability=(),
                reason=unavailable_reason,
            ),
        ),
    )


def test_compute_diffs_and_coverage_retains_reference_when_source_run_is_missing() -> None:
    previous_entry = _resolved_reference_plan()[1]
    assert previous_entry.reference is not None
    reference = _diff_reference(previous_entry.reference)

    result = compute_diffs_and_coverage(
        monitoring_run_id=MONITORING_RUN_ID,
        source_run_id=SOURCE_RUN_ID,
        metric_names=METRIC_NAMES,
        current_metrics=CURRENT_METRICS,
        reference_plan=(previous_entry,),
        reference_metrics_by_source_run_id={previous_entry.reference.source_run_id: None},
    )

    assert result == ComputedDiffCoverage(
        diffs=(),
        coverages=(
            ReferenceComparisonCoverage(
                reference_kind=DiffReferenceKind.PREVIOUS,
                reference=reference,
                status=ReferenceComparisonStatus.UNAVAILABLE,
                diff_ids=(),
                metric_unavailability=(),
                reason="reference_source_run_missing",
            ),
        ),
    )
