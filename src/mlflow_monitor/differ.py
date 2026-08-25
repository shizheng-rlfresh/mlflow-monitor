"""Diffs and Coverages computation module for mlflow-monitor."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from mlflow_monitor.domain import (
    Diff,
    DiffReference,
    MetricComparisonUnavailable,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
from mlflow_monitor.identity import make_diff_id
from mlflow_monitor.workflow import PreparedReferencePlanEntry


@dataclass(frozen=True, slots=True)
class ComputedDiffCoverage:
    """Computed diffs and coverages for a given monitoring run and selected metrics."""

    diffs: tuple[Diff, ...]
    coverages: tuple[ReferenceComparisonCoverage, ...]


def compute_diffs_and_coverage(
    monitoring_run_id: str,
    source_run_id: str,
    metric_names: Sequence[str],
    current_metrics: dict[str, float],
    reference_plan: tuple[PreparedReferencePlanEntry, ...],
    reference_metrics_by_source_run_id: dict[str, dict[str, float] | None],
) -> ComputedDiffCoverage:
    """Compute diffs and coverages for the given metrics and reference plan.

    Args:
        monitoring_run_id: Identifier of the monitoring run.
        source_run_id: Identifier of the source run to compare against.
        metric_names: List of metric names to compute diffs and coverages for.
        current_metrics: Dictionary of current metrics with metric names as keys and their values as floats.
        reference_plan: Tuple of prepared reference plan entries.
        reference_metrics_by_source_run_id: Dictionary mapping source run IDs to their corresponding metrics dictionaries.

    Returns:
        An instance of `ComputedDiffCoverage` containing the diffs and coverages for each metric.
    """  # noqa: E501
    diffs = []
    coverages = []

    for reference_entry in reference_plan:
        _diff_ids = []
        _metric_unavailability: list[MetricComparisonUnavailable] = []
        _status = ReferenceComparisonStatus.COMPLETED

        reference_kind = reference_entry.kind

        if reference_entry.reference is None:
            _status = ReferenceComparisonStatus.UNAVAILABLE
            coverages.append(
                ReferenceComparisonCoverage(
                    reference_kind=reference_kind,
                    reference=None,
                    status=_status,
                    diff_ids=(),
                    metric_unavailability=(),
                    reason=reference_entry.unavailable_reason,
                )
            )
            continue

        reference_monitoring_run_id = reference_entry.reference.monitoring_run_id
        reference_source_run_id = reference_entry.reference.source_run_id

        _diff_reference = DiffReference(
            kind=reference_kind,
            monitoring_run_id=reference_monitoring_run_id,
            source_run_id=reference_source_run_id,
        )

        reference_metrics = reference_metrics_by_source_run_id.get(reference_source_run_id)
        if reference_metrics is None:
            _status = ReferenceComparisonStatus.UNAVAILABLE
            coverages.append(
                ReferenceComparisonCoverage(
                    reference_kind=reference_kind,
                    reference=_diff_reference,
                    status=_status,
                    diff_ids=(),
                    metric_unavailability=(),
                    reason="reference_source_run_missing",
                )
            )
            continue

        for metric_name in metric_names:
            current_value = current_metrics.get(metric_name, None)
            reference_value = reference_metrics.get(metric_name, None)

            delta, reason = _validate_metric_values(current_value, reference_value)
            if delta is None:
                _metric_unavailability.append(
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
                reference=_diff_reference,
                metric_name=metric_name,
            )

            _diff_ids.append(diff_id)
            diffs.append(
                Diff(
                    diff_id=diff_id,
                    monitoring_run_id=monitoring_run_id,
                    source_run_id=source_run_id,
                    reference=_diff_reference,
                    metric_name=metric_name,
                    current_value=cast(float, current_value),
                    reference_value=cast(float, reference_value),
                    delta=delta,
                )
            )

        coverages.append(
            ReferenceComparisonCoverage(
                reference_kind=reference_kind,
                reference=_diff_reference,
                status=_status,
                diff_ids=tuple(_diff_ids),
                metric_unavailability=tuple(_metric_unavailability),
                reason=None,
            )
        )

    return ComputedDiffCoverage(diffs=tuple(diffs), coverages=tuple(coverages))


def _validate_metric_values(
    current_value: float | None, reference_value: float | None
) -> tuple[float | None, str | None]:
    """Validate the current and reference metric values.

    Args:
        current_value: The current metric value.
        reference_value: The reference metric value.

    Returns:
        A tuple containing the delta if the values are valid, and an optional reason string
            if they are not valid.
    """
    if current_value is None:
        return None, "current_metric_missing"
    if not math.isfinite(current_value):
        return None, "current_metric_not_finite"
    if reference_value is None:
        return None, "reference_metric_missing"
    if not math.isfinite(reference_value):
        return None, "reference_metric_not_finite"

    delta = current_value - reference_value
    if not math.isfinite(delta):
        return None, "delta_not_finite"

    return delta, None
