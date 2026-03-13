"""Invariant checks for the MLflow-Monitor v0 system."""

from collections.abc import Sequence

from mlflow_monitor.domain import LKG, Baseline, Run, Timeline
from mlflow_monitor.errors import InvariantViolation


def validate_timeline_ownership(
    timeline: Timeline,
    baseline: Baseline | None = None,
    lkg: LKG | None = None,
    runs: Sequence[Run] | None = None,
) -> None:
    """Validate that all provided records are owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        baseline: Optional baseline record to check ownership of.
        lkg: Optional last-known-good record to check ownership of.
        runs: Optional sequence of run records to check ownership of.

    Raises:
        InvariantViolation: If any record is found to violate timeline ownership.

    Returns:
        None if all records are valid.
    """
    if baseline is not None:
        _validate_baseline_ownership(timeline, baseline)
    if lkg is not None:
        _validate_lkg_ownership(timeline, lkg)
    if runs is not None:
        for run in runs:
            _validate_run_ownership(timeline, run)
    return None


def validate_baseline_immutability(
    baseline: Baseline,
    proposed_baseline: Baseline,
) -> None:
    """Validate that the baseline record has not changed since timeline creation.

    Args:
        baseline: The baseline record to validate.
        proposed_baseline: The new baseline record to compare against the existing one.

    Raises:
        InvariantViolation: If the baseline is found to have changed since timeline creation.

    Returns:
        None if the baseline is valid.
    """
    if baseline != proposed_baseline:
        fields = []
        if baseline.timeline_id != proposed_baseline.timeline_id:
            fields.append("timeline_id")
        if baseline.source_run_id != proposed_baseline.source_run_id:
            fields.append("source_run_id")
        if baseline.model_identity != proposed_baseline.model_identity:
            fields.append("model_identity")
        if baseline.parameter_fingerprint != proposed_baseline.parameter_fingerprint:
            fields.append("parameter_fingerprint")
        if baseline.data_snapshot_ref != proposed_baseline.data_snapshot_ref:
            fields.append("data_snapshot_ref")
        if baseline.run_config_ref != proposed_baseline.run_config_ref:
            fields.append("run_config_ref")
        if baseline.metric_snapshot != proposed_baseline.metric_snapshot:
            fields.append("metric_snapshot")
        if baseline.environment_context != proposed_baseline.environment_context:
            fields.append("environment_context")

        raise InvariantViolation(
            code="baseline_immutability_violation",
            message=f"Proposed Baseline {proposed_baseline} does not match existing Baseline {baseline}",
            entity="Baseline",
            field=", ".join(fields) if fields else None,
        )
    return None


def validate_lkg_membership(timeline: Timeline, lkg: LKG) -> None:
    """Validate that the LKG record belongs to the same timeline.

    Args:
        timeline: The timeline record to validate against.
        lkg: The LKG record to check membership of.

    Raises:
        InvariantViolation: If the LKG is found to not belong to the same timeline.

    Returns:
        None if the LKG is valid.
    """
    if lkg.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="lkg_membership_violation",
            message=f"LKG {lkg} does not belong to Timeline {timeline}",
            entity="LKG",
            field="timeline_id",
        )

    if lkg.run_id not in timeline.run_ids:
        raise InvariantViolation(
            code="lkg_membership_violation",
            message=f"LKG {lkg} does not belong to Timeline {timeline}",
            entity="LKG",
            field="run_id",
        )

    if lkg.run_id != timeline.active_lkg_run_id:
        raise InvariantViolation(
            code="lkg_membership_violation",
            message=f"LKG {lkg} does not belong to Timeline {timeline}",
            entity="LKG",
            field="active_lkg_run_id",
        )

    return None


def _validate_baseline_ownership(timeline: Timeline, baseline: Baseline) -> None:
    """Validate that the baseline record is owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        baseline: The baseline record to check ownership of.

    Raises:
        InvariantViolation: If the baseline is found to violate timeline ownership.

    Returns:
        None if the baseline is valid.
    """
    if baseline.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="baseline_timeline_mismatch",
            message=f"Baseline {baseline.timeline_id} does not match Timeline {timeline.timeline_id}",
            entity="Baseline",
            field="timeline_id",
        )


def _validate_lkg_ownership(timeline: Timeline, lkg: LKG) -> None:
    """Validate that the LKG record is owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        lkg: The LKG record to check ownership of.

    Raises:
        InvariantViolation: If the LKG is found to violate timeline ownership.

    Returns:
        None if the LKG is valid.
    """
    if lkg.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="lkg_timeline_mismatch",
            message=f"LKG {lkg.timeline_id} does not match Timeline {timeline.timeline_id}",
            entity="LKG",
            field="timeline_id",
        )


def _validate_run_ownership(timeline: Timeline, run: Run) -> None:
    """Validate that the run record is owned by the same timeline.

    Args:
        timeline: The timeline record to validate against.
        run: The run record to check ownership of.

    Raises:
        InvariantViolation: If the run is found to violate timeline ownership.

    Returns:
        None if the run is valid.
    """
    if run.timeline_id != timeline.timeline_id:
        raise InvariantViolation(
            code="run_timeline_mismatch",
            message=f"Run {run.timeline_id} does not match Timeline {timeline.timeline_id}",
            entity="Run",
            field="timeline_id",
        )
