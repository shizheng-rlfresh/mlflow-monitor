"""Diffs and Coverages computation module for mlflow-monitor."""

from collections.abc import Sequence
from dataclasses import dataclass

from mlflow_monitor.domain import (
    Diff,
    DiffReference,
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
    diffs = []
    coverages = []

    for reference_entry in reference_plan:
        reference_kind = reference_entry.kind
        assert reference_entry.reference is not None
        assert reference_entry.reference.source_run_id is not None
        reference_monitoring_run_id = reference_entry.reference.monitoring_run_id
        reference_source_run_id = reference_entry.reference.source_run_id

        _diff_ids = []
        _metric_unavailability = []
        _status = ReferenceComparisonStatus.COMPLETED

        _diff_reference = DiffReference(
            kind=reference_kind,
            monitoring_run_id=reference_monitoring_run_id,
            source_run_id=reference_source_run_id,
        )

        for metric_name in metric_names:
            diff_id = make_diff_id(
                monitoring_run_id=monitoring_run_id,
                source_run_id=source_run_id,
                reference=_diff_reference,
                metric_name=metric_name,
            )

            _diff_ids.append(diff_id)

            current_value = current_metrics[metric_name]
            reference_value = reference_metrics_by_source_run_id[reference_source_run_id][
                metric_name
            ]

            delta = current_value - reference_value
            diffs.append(
                Diff(
                    diff_id=diff_id,
                    monitoring_run_id=monitoring_run_id,
                    source_run_id=source_run_id,
                    reference=_diff_reference,
                    metric_name=metric_name,
                    current_value=current_value,
                    reference_value=reference_value,
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
