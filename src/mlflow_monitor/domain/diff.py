"""Diff domain models for mlflow-monitor v0."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from .reference import DiffReference, DiffReferenceKind

RELATIVE_DELTA_TOLERANCE = 1e-6
ABSOLUTE_DELTA_TOLERANCE = 1e-6


class ReferenceComparisonStatus(StrEnum):
    """Status outcomes for metrics comparison in diff."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"


class ReferenceComparisonSkippedReason(StrEnum):
    """Reason codes for skipped reference comparison."""

    CURRENT_NOT_COMPARABLE = "current_not_comparable"


class ReferenceComparisonUnavailableReason(StrEnum):
    """Reason codes for unavailable reference comparison."""

    PREVIOUS_REFERENCE_MISSING = "previous_reference_missing"
    LKG_NOT_SELECTED = "lkg_not_selected"
    LKG_SELECTION_INCONSISTENT = "lkg_selection_inconsistent"
    REFERENCE_SOURCE_RUN_MISSING = "reference_source_run_missing"


class MetricComparisonUnavailableReason(StrEnum):
    """Reason codes for unavailable metric comparison."""

    CURRENT_METRIC_MISSING = "current_metric_missing"
    REFERENCE_METRIC_MISSING = "reference_metric_missing"
    CURRENT_METRIC_NOT_FINITE = "current_metric_not_finite"
    REFERENCE_METRIC_NOT_FINITE = "reference_metric_not_finite"
    DELTA_NOT_FINITE = "delta_not_finite"


REFERENCE_COMPARISON_STATUS_TO_REASON = MappingProxyType(
    {
        ReferenceComparisonStatus.COMPLETED: frozenset(),
        ReferenceComparisonStatus.SKIPPED: frozenset(
            (ReferenceComparisonSkippedReason.CURRENT_NOT_COMPARABLE.value,)
        ),
        ReferenceComparisonStatus.UNAVAILABLE: frozenset(
            (
                ReferenceComparisonUnavailableReason.PREVIOUS_REFERENCE_MISSING.value,
                ReferenceComparisonUnavailableReason.LKG_NOT_SELECTED.value,
                ReferenceComparisonUnavailableReason.LKG_SELECTION_INCONSISTENT.value,
                ReferenceComparisonUnavailableReason.REFERENCE_SOURCE_RUN_MISSING.value,
            )
        ),
    }
)


@dataclass(frozen=True, slots=True)
class MetricComparisonUnavailable:
    """Information about a metric that could not be compared in a diff.

    Attributes:
        metric_name: The name of the metric that is unavailable for comparison.
        reason: The reason why the metric comparison is unavailable.
    """

    metric_name: str
    reason: str

    def __post_init__(self) -> None:
        """Validate MetricComparisonUnavailable for atomic shape."""
        for field_name in ("metric_name", "reason"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"MetricComparisonUnavailable requires a non-empty string for "
                    f"field {field_name!r}."
                )
        metric_level_reason = self.reason
        if metric_level_reason not in MetricComparisonUnavailableReason:
            raise ValueError(
                f"MetricComparisonUnavailable 'reason' must be one of "
                f"{[reason.value for reason in MetricComparisonUnavailableReason]}, "
                f"got {metric_level_reason!r}."
            )


@dataclass(frozen=True, slots=True)
class Diff:
    """Objective change record between a run and one reference point.

    Attributes:
        diff_id: Unique identifier for the diff record.
        monitoring_run_id: The ID of the monitoring run this diff is associated with.
        source_run_id: The immutable source training run ID of the monitoring run.
        reference: Reference descriptor containing both reference kind and reference id.
        metric_name: The name of the metric being compared.
        current_value: The value of the metric for the current run.
        reference_value: The value of the metric for the reference run.
        delta: current_value - reference_value.
    """

    diff_id: str
    monitoring_run_id: str
    source_run_id: str
    reference: DiffReference
    metric_name: str
    current_value: float
    reference_value: float
    delta: float

    def __post_init__(self) -> None:
        """Validate Diff for atomic shape."""
        for field_name in (
            "diff_id",
            "monitoring_run_id",
            "source_run_id",
            "metric_name",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Diff requires a non-empty string for field {field_name!r}.")

        if not isinstance(self.reference, DiffReference):
            raise ValueError("Diff requires a valid DiffReference for the 'reference' field.")

        for field_name in ("current_value", "reference_value", "delta"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Diff requires a finite float for field {field_name!r}.")

            if isinstance(value, int):
                object.__setattr__(self, field_name, float(value))

        expected_delta = self.current_value - self.reference_value

        if not math.isfinite(expected_delta):
            raise ValueError("Diff computed delta must be a finite float.")

        if not math.isclose(
            self.delta,
            expected_delta,
            rel_tol=RELATIVE_DELTA_TOLERANCE,
            abs_tol=ABSOLUTE_DELTA_TOLERANCE,
        ):
            raise ValueError(
                "Diff delta must equal current_value - reference_value within "
                f"rel_tol={RELATIVE_DELTA_TOLERANCE}, abs_tol={ABSOLUTE_DELTA_TOLERANCE}."
            )


@dataclass(frozen=True, slots=True)
class ReferenceComparisonCoverage:
    """Coverage information for a specific reference kind in a diff comparison.

    Attributes:
        reference_kind: The kind of reference (e.g., baseline, previous, lkg, custom).
        reference: The specific reference instance being compared, if applicable.
        status: The status of the reference comparison (e.g., completed, skipped, unavailable).
        diff_ids: A tuple of diff IDs associated with this reference comparison.
        metric_unavailability: A tuple of MetricComparisonUnavailable instances indicating metrics that could not be compared.
        reason: An optional reason code constrained by the reference comparison status.
    """  # noqa: E501

    reference_kind: DiffReferenceKind
    reference: DiffReference | None
    status: ReferenceComparisonStatus
    diff_ids: tuple[str, ...]
    metric_unavailability: tuple[MetricComparisonUnavailable, ...]
    reason: str | None

    def __post_init__(self) -> None:
        """Validate ReferenceComparisonCoverage for atomic shape."""
        # defensive conversion to tuples for immutability
        diff_ids_tuple = tuple(self.diff_ids)
        metric_unavailability_tuple = tuple(self.metric_unavailability)

        object.__setattr__(self, "diff_ids", diff_ids_tuple)
        object.__setattr__(self, "metric_unavailability", metric_unavailability_tuple)

        if self.reference_kind not in DiffReferenceKind:
            raise ValueError(
                "ReferenceComparisonCoverage has an unrecognized "
                f"reference_kind: {self.reference_kind!r}."
            )

        if self.reference is not None:
            if not isinstance(self.reference, DiffReference):
                raise ValueError("Coverage reference must be a DiffReference.")
            if self.reference_kind != self.reference.kind:
                raise ValueError(
                    "ReferenceComparisonCoverage 'reference_kind' must match the kind of "
                    "the provided 'reference'."
                )

        if self.status == ReferenceComparisonStatus.COMPLETED:
            self._validate_completed_coverage()

        elif self.status == ReferenceComparisonStatus.SKIPPED:
            self._validate_skipped_coverage()

        elif self.status == ReferenceComparisonStatus.UNAVAILABLE:
            self._validate_unavailable_coverage()
        else:
            raise ValueError(
                f"ReferenceComparisonCoverage has an unrecognized status={self.status!r}."
            )

    def _validate_completed_coverage(self) -> None:
        if self.reference is None:
            raise ValueError(
                f"ReferenceComparisonCoverage with status={self.status!r} "
                "must have a valid reference."
            )
        if self.reason is not None:
            raise ValueError(
                f"ReferenceComparisonCoverage with status={self.status!r} "
                "must not have a reason code."
            )

    def _validate_skipped_coverage(self) -> None:
        if self.reference is None:
            raise ValueError(
                f"ReferenceComparisonCoverage with status={self.status!r} "
                "must have a valid reference."
            )
        if self.diff_ids:
            raise ValueError(
                f"ReferenceComparisonCoverage with status={self.status!r} "
                "must not have any diff IDs."
            )

        if self.metric_unavailability:
            raise ValueError(
                f"ReferenceComparisonCoverage with status={self.status!r} "
                "must not have any metric unavailability entries."
            )

        if (
            self.reason is None
            or self.reason
            not in REFERENCE_COMPARISON_STATUS_TO_REASON[ReferenceComparisonStatus.SKIPPED]
        ):
            raise ValueError(
                f"ReferenceComparisonCoverage with status={self.status!r} must have a reason code "
                f"from {REFERENCE_COMPARISON_STATUS_TO_REASON[ReferenceComparisonStatus.SKIPPED]}."
            )

    def _validate_unavailable_coverage(self) -> None:
        """Validate an unavailable reference-comparison group."""
        if self.diff_ids or self.metric_unavailability:
            raise ValueError("Unavailable coverage cannot contain metric results.")

        if (
            self.reason is None
            or self.reason
            not in REFERENCE_COMPARISON_STATUS_TO_REASON[ReferenceComparisonStatus.UNAVAILABLE]
        ):
            raise ValueError(
                f"ReferenceComparisonCoverage with status={self.status!r} must have a reason code "
                f"from {REFERENCE_COMPARISON_STATUS_TO_REASON[ReferenceComparisonStatus.UNAVAILABLE]}."  # noqa: E501
            )

        elif self.reason == ReferenceComparisonUnavailableReason.PREVIOUS_REFERENCE_MISSING:
            if self.reference_kind != DiffReferenceKind.PREVIOUS:
                raise ValueError(
                    f"ReferenceComparisonCoverage with status={self.status!r} "
                    f"and reason {self.reason!r} requires reference_kind='previous'."
                )
            if self.reference is not None:
                raise ValueError(
                    f"ReferenceComparisonCoverage with status={self.status!r} "
                    f"and reason {self.reason!r} requires reference=None."
                )

        elif self.reason == ReferenceComparisonUnavailableReason.LKG_NOT_SELECTED:
            if self.reference_kind != DiffReferenceKind.LKG:
                raise ValueError(
                    f"ReferenceComparisonCoverage with status={self.status!r} "
                    f"and reason {self.reason!r} requires reference_kind='lkg'."
                )
            if self.reference is not None:
                raise ValueError(
                    f"ReferenceComparisonCoverage with status={self.status!r} "
                    f"and reason {self.reason!r} requires reference=None."
                )

        elif self.reason == ReferenceComparisonUnavailableReason.LKG_SELECTION_INCONSISTENT:
            if self.reference_kind != DiffReferenceKind.LKG:
                raise ValueError(
                    f"ReferenceComparisonCoverage with status={self.status!r} "
                    f"and reason {self.reason!r} requires reference_kind='lkg'."
                )
            if self.reference is not None:
                raise ValueError(
                    f"ReferenceComparisonCoverage with status={self.status!r} "
                    f"and reason {self.reason!r} requires reference=None."
                )

        else:
            if self.reference is None:
                raise ValueError(
                    f"ReferenceComparisonCoverage with status={self.status!r} "
                    f"and reason {self.reason!r} requires a retained reference."
                )
