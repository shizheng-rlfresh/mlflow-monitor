"""Diffs and Coverages computation module for mlflow-monitor."""

from collections.abc import Sequence
from dataclasses import dataclass

from mlflow_monitor.domain import (
    Diff,
    DiffReference,
    ReferenceComparisonCoverage,
    ReferenceComparisonStatus,
)
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
    reference_metrics_by_source_run_id: dict[str, dict[str, float]],
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
    _validate_metric_names(metric_names, current_metrics)

    diffs = []
    coverages = []

    for reference_entry in reference_plan:
        if reference_entry.reference is None:
            reference_comparison_status = ReferenceComparisonStatus.UNAVAILABLE
            reference_comparison_coverage = ReferenceComparisonCoverage(
                reference_kind=reference_entry.kind,
                reference=
            )
        diff_reference = DiffReference(
            kind=reference_entry.kind,
            monitoring_run_id=reference_entry.reference.monitoring_run_id,
            source_run_id=reference_entry.reference.source_run_id,
        )

    return ComputedDiffCoverage(diffs=(), coverages=())


def _validate_metric_names(metric_names: Sequence[str], current_metrics: dict[str, float]) -> None:
    """Validate that all metric names exist in the current metrics dictionary."""
    for metric_name in metric_names:
        if metric_name not in current_metrics:
            raise ValueError(f"Metric '{metric_name}' is not present in the current metrics.")
