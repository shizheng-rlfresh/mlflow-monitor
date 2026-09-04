"""Diffs and Coverages computation module for mlflow-monitor."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from mlflow_monitor.domain import (
    Diff,
    DiffReference,
    MetricComparisonUnavailable,
    MetricComparisonUnavailableReason,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
from mlflow_monitor.identity import make_diff_id

if TYPE_CHECKING:
    from mlflow_monitor.workflow.prepared_context import PreparedReferencePlanEntry

MetricValue = Mapping[str, float]
ReferenceMetricsBySourceRunId = Mapping[str, MetricValue | None]


@dataclass(frozen=True, slots=True)
class ComputedDiffCoverage:
    """Computed diffs and coverages for a given monitoring run and selected metrics."""

    diffs: tuple[Diff, ...]
    coverages: tuple[ReferenceComparisonCoverage, ...]


def compute_diffs_and_coverage(
    monitoring_run_id: str,
    source_run_id: str,
    metric_names: tuple[str, ...],
    current_metrics: dict[str, float],
    reference_plan: tuple[PreparedReferencePlanEntry, ...],
    reference_metrics_by_source_run_id: ReferenceMetricsBySourceRunId,
) -> ComputedDiffCoverage:
    """Compute diffs and coverages for the given metrics and reference plan.

    Args:
        monitoring_run_id: Identifier of the monitoring run.
        source_run_id: Identifier of the source run of the monitoring run.
        metric_names: Canonically ordered, unique selected metric names. An empty tuple selects
            no metrics
        current_metrics: Dictionary of current metrics with metric names as keys and their values
            as floats.
        reference_plan: Tuple of prepared reference plan entries to be used for comparison.
        reference_metrics_by_source_run_id: Dictionary mapping source run IDs to their corresponding
            metrics dictionaries.

    Returns:
        An instance of `ComputedDiffCoverage` containing the diffs and coverages for each metric.
    """
    diffs: list[Diff] = []
    coverages: list[ReferenceComparisonCoverage] = []

    for reference_entry in reference_plan:
        reference_diff_ids: list[str] = []
        reference_metric_unavailability: list[MetricComparisonUnavailable] = []

        reference_kind = reference_entry.kind

        # missing reference means the reference is not available for comparison.
        # so we mark the coverage as unavailable and continue to the next reference entry.
        if reference_entry.reference is None:
            coverages.append(
                ReferenceComparisonCoverage(
                    reference_kind=reference_kind,
                    reference=None,
                    status=ReferenceComparisonStatus.UNAVAILABLE,
                    diff_ids=(),
                    metric_unavailability=(),
                    reason=reference_entry.unavailable_reason,
                )
            )
            continue

        reference_monitoring_run_id = reference_entry.reference.monitoring_run_id
        reference_source_run_id = reference_entry.reference.source_run_id

        diff_reference = DiffReference(
            kind=reference_kind,
            monitoring_run_id=reference_monitoring_run_id,
            source_run_id=reference_source_run_id,
        )

        reference_metrics = reference_metrics_by_source_run_id.get(reference_source_run_id)

        # missing reference metrics means the reference is not available for comparison.
        # so we mark the coverage as unavailable and continue to the next reference entry.
        if reference_metrics is None:
            coverages.append(
                ReferenceComparisonCoverage(
                    reference_kind=reference_kind,
                    reference=diff_reference,
                    status=ReferenceComparisonStatus.UNAVAILABLE,
                    diff_ids=(),
                    metric_unavailability=(),
                    reason="reference_source_run_missing",
                )
            )
            continue

        # if reference is available for comparison, we continue to process diffs for each metric.
        for metric_name in metric_names:
            current_value = current_metrics.get(metric_name, None)
            reference_value = reference_metrics.get(metric_name, None)

            delta, reason = _validate_metric_values(current_value, reference_value)
            if delta is None:
                reference_metric_unavailability.append(
                    MetricComparisonUnavailable(
                        metric_name=metric_name,
                        reason=cast(str, reason),
                    )
                )
                # no diff can be computed if the metric values are invalid
                continue

            diff_id = make_diff_id(
                monitoring_run_id=monitoring_run_id,
                source_run_id=source_run_id,
                reference=diff_reference,
                metric_name=metric_name,
            )

            reference_diff_ids.append(diff_id)
            diffs.append(
                Diff(
                    diff_id=diff_id,
                    monitoring_run_id=monitoring_run_id,
                    source_run_id=source_run_id,
                    reference=diff_reference,
                    metric_name=metric_name,
                    current_value=cast(float, current_value),
                    reference_value=cast(float, reference_value),
                    delta=delta,
                )
            )

        coverages.append(
            ReferenceComparisonCoverage(
                reference_kind=reference_kind,
                reference=diff_reference,
                status=ReferenceComparisonStatus.COMPLETED,
                diff_ids=tuple(reference_diff_ids),
                metric_unavailability=tuple(reference_metric_unavailability),
                reason=None,
            )
        )

    return ComputedDiffCoverage(diffs=tuple(diffs), coverages=tuple(coverages))


def _validate_metric_values(
    current_value: float | None, reference_value: float | None
) -> tuple[float | None, MetricComparisonUnavailableReason | None]:
    """Validate the current and reference metric values.

    Args:
        current_value: The current metric value.
        reference_value: The reference metric value.

    Returns:
        A tuple containing the delta if the values are valid, and an optional reason code
            if it is not valid.
    """
    if current_value is None:
        return None, MetricComparisonUnavailableReason.CURRENT_METRIC_MISSING
    if not math.isfinite(current_value):
        return None, MetricComparisonUnavailableReason.CURRENT_METRIC_NOT_FINITE
    if reference_value is None:
        return None, MetricComparisonUnavailableReason.REFERENCE_METRIC_MISSING
    if not math.isfinite(reference_value):
        return None, MetricComparisonUnavailableReason.REFERENCE_METRIC_NOT_FINITE

    delta = current_value - reference_value
    if not math.isfinite(delta):
        return None, MetricComparisonUnavailableReason.DELTA_NOT_FINITE

    return delta, None
