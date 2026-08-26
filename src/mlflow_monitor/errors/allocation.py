"""Custom exception types for allocation gateway inconsistencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .gateway import GatewayConsistencyCode, GatewayConsistencyViolation


class AllocationInconsistentReason(StrEnum):
    """Reasons for monitoring run allocation inconsistent reason code."""

    DUPLICATE_IDENTITY = "duplicate_identity"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    SEQUENCE_GAP = "sequence_gap"
    INVALID_ALLOCATION = "invalid_allocation"
    NEXT_SEQUENCE_AHEAD = "next_sequence_ahead"
    UNKNOWN_POINTER = "unknown_pointer"
    UNKNOWN_TAG = "unknown_tag"
    SOURCE_BINDING_CONFLICT = "source_binding_conflict"
    TIMELINE_CONFLICT = "timeline_conflict"


@dataclass(frozen=True, slots=True)
class AllocationConsistencyViolation(GatewayConsistencyViolation):
    """Raised when durable Monitoring Run allocation state is inconsistent."""

    @classmethod
    def _create(
        cls,
        *,
        reason: AllocationInconsistentReason,
        message: str,
        details: tuple[tuple[str, str | int | None], ...],
    ) -> AllocationConsistencyViolation:
        """Create a violation with its stable code and normalized reason."""
        return cls(
            code=GatewayConsistencyCode.MONITORING_ALLOCATION_INCONSISTENT.value,
            message=message,
            details=(("reason", reason.value), *details),
        )

    @classmethod
    def duplicate_identity(
        cls,
        *,
        first_monitoring_run_id: str,
        second_monitoring_run_id: str,
    ) -> AllocationConsistencyViolation:
        """Create a violation for two Monitoring Runs with the same identity."""
        return cls._create(
            reason=AllocationInconsistentReason.DUPLICATE_IDENTITY,
            message="Multiple Monitoring Runs claim the same allocation identity.",
            details=(
                ("first_monitoring_run_id", first_monitoring_run_id),
                ("second_monitoring_run_id", second_monitoring_run_id),
            ),
        )

    @classmethod
    def duplicate_sequence(
        cls,
        *,
        sequence_index: int,
        first_monitoring_run_id: str,
        second_monitoring_run_id: str,
    ) -> AllocationConsistencyViolation:
        """Create a violation for two Monitoring Runs with the same sequence index."""
        return cls._create(
            reason=AllocationInconsistentReason.DUPLICATE_SEQUENCE,
            message=f"Multiple Monitoring Runs claim sequence_index={sequence_index}.",
            details=(
                ("sequence_index", sequence_index),
                ("first_monitoring_run_id", first_monitoring_run_id),
                ("second_monitoring_run_id", second_monitoring_run_id),
            ),
        )

    @classmethod
    def sequence_gap(
        cls,
        *,
        expected_sequence_index: int,
        actual_sequence_index: int,
    ) -> AllocationConsistencyViolation:
        """Create a violation for a non-contiguous allocation sequence."""
        return cls._create(
            reason=AllocationInconsistentReason.SEQUENCE_GAP,
            message=(
                "Monitoring allocation sequence is not contiguous; "
                f"expected sequence_index={expected_sequence_index}, "
                f"got {actual_sequence_index}."
            ),
            details=(
                ("expected_sequence_index", expected_sequence_index),
                ("actual_sequence_index", actual_sequence_index),
            ),
        )

    @classmethod
    def missing_durable_tags(
        cls,
        *,
        monitoring_run_id: str | None,
        missing_tags: tuple[str, ...],
    ) -> AllocationConsistencyViolation:
        """Create a violation for an allocation missing required durable tags."""
        rendered_missing_tags = ", ".join(missing_tags)
        return cls._create(
            reason=AllocationInconsistentReason.INVALID_ALLOCATION,
            message=(
                f"Monitoring Run {monitoring_run_id!r} is missing durable "
                f"allocation tags: {rendered_missing_tags}."
            ),
            details=(
                ("monitoring_run_id", monitoring_run_id),
                ("missing_tags", rendered_missing_tags),
            ),
        )

    @classmethod
    def non_integer_sequence(
        cls,
        *,
        monitoring_run_id: str,
        raw_sequence_index: str,
    ) -> AllocationConsistencyViolation:
        """Create a violation for a non-integer allocation sequence index."""
        return cls._create(
            reason=AllocationInconsistentReason.INVALID_ALLOCATION,
            message=(
                f"Monitoring Run {monitoring_run_id!r} has a non-integer "
                f"sequence index: {raw_sequence_index!r}."
            ),
            details=(
                ("monitoring_run_id", monitoring_run_id),
                ("raw_sequence_index", raw_sequence_index),
            ),
        )

    @classmethod
    def negative_sequence(
        cls,
        *,
        monitoring_run_id: str,
        sequence_index: int,
    ) -> AllocationConsistencyViolation:
        """Create a violation for a negative allocation sequence index."""
        return cls._create(
            reason=AllocationInconsistentReason.INVALID_ALLOCATION,
            message=(
                f"Monitoring Run {monitoring_run_id!r} has a negative "
                f"sequence_index={sequence_index}."
            ),
            details=(
                ("monitoring_run_id", monitoring_run_id),
                ("sequence_index", sequence_index),
            ),
        )

    @classmethod
    def next_sequence_ahead(
        cls,
        *,
        persisted_next_sequence_index: int,
        durable_next_sequence_index: int,
    ) -> AllocationConsistencyViolation:
        """Create a violation for a persisted next sequence ahead of durable state."""
        return cls._create(
            reason=AllocationInconsistentReason.NEXT_SEQUENCE_AHEAD,
            message=(
                "Monitoring allocation next sequence index is ahead of durable state; "
                f"persisted sequence_index={persisted_next_sequence_index}, "
                f"durable sequence_index={durable_next_sequence_index}."
            ),
            details=(
                ("persisted_next_sequence_index", persisted_next_sequence_index),
                ("durable_next_sequence_index", durable_next_sequence_index),
            ),
        )

    @classmethod
    def unknown_pointer(cls, *, monitoring_run_id: str) -> AllocationConsistencyViolation:
        """Create a violation for a pointer to an unknown allocation."""
        return cls._create(
            reason=AllocationInconsistentReason.UNKNOWN_POINTER,
            message=(
                "Monitoring pointer references an unknown allocation for "
                f"monitoring_run_id={monitoring_run_id!r}."
            ),
            details=(("monitoring_run_id", monitoring_run_id),),
        )

    @classmethod
    def unknown_tag(
        cls,
        *,
        tag: str,
        monitoring_run_id: str,
    ) -> AllocationConsistencyViolation:
        """Create a violation for an experiment tag pointing to an unknown allocation."""
        return cls._create(
            reason=AllocationInconsistentReason.UNKNOWN_TAG,
            message=f"Experiment tag {tag!r} references an unknown allocation.",
            details=(
                ("tag", tag),
                ("monitoring_run_id", monitoring_run_id),
            ),
        )

    @classmethod
    def source_binding_conflict(
        cls,
        *,
        tag: str,
        monitoring_run_id: str,
        source_run_id: str,
        persisted_source_run_id: str,
    ) -> AllocationConsistencyViolation:
        """Create a violation for an allocation bound to a different source run."""
        return cls._create(
            reason=AllocationInconsistentReason.SOURCE_BINDING_CONFLICT,
            message=(
                f"Experiment tag {tag!r} points to monitoring_run_id={monitoring_run_id!r} "
                f"allocated for source_run_id={persisted_source_run_id!r}, "
                f"not source_run_id={source_run_id!r}."
            ),
            details=(
                ("tag", tag),
                ("monitoring_run_id", monitoring_run_id),
                ("source_run_id", source_run_id),
                ("persisted_source_run_id", persisted_source_run_id),
            ),
        )

    @classmethod
    def timeline_conflict(
        cls,
        *,
        sequence_index: int,
        indexed_monitoring_run_id: str,
        durable_monitoring_run_id: str | None,
    ) -> AllocationConsistencyViolation:
        """Create a violation for a timeline slot that conflicts with durable state."""
        return cls._create(
            reason=AllocationInconsistentReason.TIMELINE_CONFLICT,
            message=(
                f"Experiment timeline slot sequence_index={sequence_index} does not match "
                "its durable Monitoring Run allocation."
            ),
            details=(
                ("sequence_index", sequence_index),
                ("indexed_monitoring_run_id", indexed_monitoring_run_id),
                ("durable_monitoring_run_id", durable_monitoring_run_id),
            ),
        )
