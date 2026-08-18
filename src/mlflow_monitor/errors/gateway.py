"""Custom exceptions types and factory functions for Gateway errors.

Three main types of gateway errors:
1. `GatewayNamespaceViolation`: a gateway operation violates namespace constraints.
2. `TrainingRunMutationViolation`: an attempt to mutate source training run data.
3. `GatewayConsistencyViolation`: a gateway operation violates consistency constraints.

For GatewayConsistencyViolation,
there are several specific error codes (i.e., subtypes) defined in the `GatewayConsistencyCode`:
- "prepared_context_inconsistent": the prepared context is inconsistent with the expected state.
- "monitoring_allocation_inconsistent": the monitoring run allocation is inconsistent with the expected state.
- "monitoring_run_upsert_field_override": an upsert operation on a monitoring run violates field constraints.
- "timeline_state_not_found_for_subject_id": the timeline state for a given subject ID is not found.
- "monitoring_run_json_artifact_inconsistent": the JSON artifact of a monitoring run is inconsistent with the expected state.
- "monitoring_run_subject_inconsistent": the subject of a monitoring run is inconsistent with the expected state.
- "monitoring_reference_inconsistent": a monitoring reference is inconsistent with the expected state.

For "monitoring_allocation_inconsistent",
there are several specific reasons defined in the `MonitoringAllocationInconsistentReason`:
- "duplicate_identity": multiple monitoring runs claim the same allocation identity.
- "duplicate_sequence": multiple monitoring runs claim the sequence index.
- "sequence_gap": there is a gap in the sequence of monitoring runs.
- "invalid_allocation": the allocation of a monitoring run is invalid.
- "next_sequence_ahead": the next sequence index is ahead of the expected value.
- "unknown_pointer": the pointer to a monitoring run is unknown.
- "unknown_tag": the tag associated with a monitoring run is unknown.
- "source_binding_conflict": there is a conflict in the source binding of a monitoring run.
- "timeline_conflict": there is a conflict in the timeline of monitoring runs.

"""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from mlflow_monitor.domain import DiffReferenceKind


@dataclass(frozen=True, slots=True)
class GatewayNamespaceViolation(ValueError):
    """Raised when a gateway operation violates namespace constraints."""

    message: str

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


@dataclass(frozen=True, slots=True)
class TrainingRunMutationViolation(ValueError):
    """Raised when code attempts to mutate source training run data."""

    message: str

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


# GatewayConsistencyViolation error factories


class GatewayConsistencyCode(StrEnum):
    """Code for gateway consistency violations."""

    PREPARED_CONTEXT_INCONSISTENT = "prepared_context_inconsistent"
    MONITORING_ALLOCATION_INCONSISTENT = "monitoring_allocation_inconsistent"
    MONITORING_RUN_UPSERT_FIELD_OVERRIDE = "monitoring_run_upsert_field_override"
    TIMELINE_STATE_NOT_FOUND_FOR_SUBJECT_ID = "timeline_state_not_found_for_subject_id"
    MONITORING_RUN_JSON_ARTIFACT_INCONSISTENT = "monitoring_run_json_artifact_inconsistent"
    MONITORING_RUN_SUBJECT_INCONSISTENT = "monitoring_run_subject_inconsistent"
    MONITORING_REFERENCE_INCONSISTENT = "monitoring_reference_inconsistent"


class MonitoringAllocationInconsistentReason(StrEnum):
    """Reasons for monitoring run allocation inconsistent error code."""

    DUPLICATE_IDENTITY = "duplicate_identity"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    SEQUENCE_GAP = "sequence_gap"
    INVALID_ALLOCATION = "invalid_allocation"
    NEXT_SEQUENCE_AHEAD = "next_sequence_ahead"
    UNKNOWN_POINTER = "unknown_pointer"
    UNKNOWN_TAG = "unknown_tag"
    SOURCE_BINDING_CONFLICT = "source_binding_conflict"
    TIMELINE_CONFLICT = "timeline_conflict"


MONITORING_ALLOCATION_REASON_MESSAGE = MappingProxyType(
    {
        MonitoringAllocationInconsistentReason.DUPLICATE_IDENTITY: (
            "Multiple monitoring runs claim the same allocation identity.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.DUPLICATE_SEQUENCE: (
            "Multiple monitoring runs claim the sequence index.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.SEQUENCE_GAP: (
            "Monitoring allocation sequences must be contiguous from zero.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.INVALID_ALLOCATION: (
            "Monitoring run allocation is invalid.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.NEXT_SEQUENCE_AHEAD: (
            "Monitoring run allocation's next sequence index is "
            "ahead of the durable allocation state.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.UNKNOWN_POINTER: (
            "Monitoring run ID points to an unknown allocation.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.UNKNOWN_TAG: (
            "Experiment tag points to an unknown allocation.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.SOURCE_BINDING_CONFLICT: (
            "Source binding conflict detected.{context_message}"
        ),
        MonitoringAllocationInconsistentReason.TIMELINE_CONFLICT: (
            "Timeline conflict detected.{context_message}"
        ),
    }
)


@dataclass(frozen=True, slots=True)
class GatewayConsistencyViolation(ValueError):
    """Raised when a gateway operation violates consistency constraints."""

    code: str
    message: str
    details: tuple[tuple[str, str | int | None], ...] = ()

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message

    # monitoring allocation inconsistent error factory
    @classmethod
    def monitoring_allocation_inconsistent(
        cls,
        *,
        reason: MonitoringAllocationInconsistentReason | str,
        details: tuple[tuple[str, str | int | None], ...] = (),
        context_message: str | None = None,
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for inconsistent monitoring allocation."""
        normalized_reason = MonitoringAllocationInconsistentReason(reason)

        message = MONITORING_ALLOCATION_REASON_MESSAGE[normalized_reason].format(
            context_message=f" {context_message}" if context_message else ""
        )

        return cls(
            code=GatewayConsistencyCode.MONITORING_ALLOCATION_INCONSISTENT,
            message=message,
            details=(
                ("reason", normalized_reason.value),
                *details,
            ),
        )

    # prepared context inconsistent error factory
    @classmethod
    def prepared_context_inconsistent(
        cls, *, reason: str, field: str
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for inconsistent prepared context."""
        return cls(
            code=GatewayConsistencyCode.PREPARED_CONTEXT_INCONSISTENT,
            message="Persisted prepared context is missing, malformed, or inconsistent.",
            details=(
                ("reason", reason),
                ("field", field),
            ),
        )

    # monitoring run upsert field override error factory
    @classmethod
    def monitoring_run_upsert_field_override(
        cls, *, message: str, details: tuple[tuple[str, str | int | None], ...] = ()
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for monitoring run upsert field override."""
        return cls(
            code=GatewayConsistencyCode.MONITORING_RUN_UPSERT_FIELD_OVERRIDE,
            message=message,
            details=(*details,),
        )

    # timeline state not found for subject ID error factory
    @classmethod
    def timeline_state_not_found_for_subject_id(
        cls, *, subject_id: str
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for missing timeline state for a subject ID."""
        return cls(
            code=GatewayConsistencyCode.TIMELINE_STATE_NOT_FOUND_FOR_SUBJECT_ID,
            message=f"Timeline state not found for subject_id={subject_id!r}.",
            details=(("subject_id", subject_id),),
        )

    # monitoring run JSON artifact inconsistent error factory
    @classmethod
    def monitoring_run_json_artifact_inconsistent(
        cls, *, monitoring_run_id: str, path: str
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for inconsistent monitoring run JSON artifact."""
        return cls(
            code=GatewayConsistencyCode.MONITORING_RUN_JSON_ARTIFACT_INCONSISTENT,
            message=(
                f"Monitoring run JSON artifact is inconsistent for "
                f"monitoring_run_id={monitoring_run_id!r} "
                f"at path={path!r}."
            ),
            details=(
                ("monitoring_run_id", monitoring_run_id),
                ("path", path),
            ),
        )

    # monitoring run subject inconsistent error factory
    @classmethod
    def monitoring_run_subject_inconsistent(
        cls, *, subject_id: str, monitoring_run_id: str
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for a monitoring run not indexed on the subject ID."""  # noqa: E501
        return cls(
            code=GatewayConsistencyCode.MONITORING_RUN_SUBJECT_INCONSISTENT,
            message=(
                f"monitoring_run_id={monitoring_run_id!r} is not indexed "
                f"on subject_id={subject_id!r}."
            ),
            details=(
                ("subject_id", subject_id),
                ("monitoring_run_id", monitoring_run_id),
            ),
        )

    # monitoring reference inconsistent error factory
    @classmethod
    def monitoring_reference_inconsistent(
        cls, *, kind: DiffReferenceKind, monitoring_run_id: str
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for an inconsistent monitoring reference."""
        return cls(
            code=GatewayConsistencyCode.MONITORING_REFERENCE_INCONSISTENT,
            message=(
                f"Monitoring reference of kind={kind.value!r} is inconsistent for "
                f"monitoring_run_id={monitoring_run_id!r}."
            ),
            details=(
                ("kind", kind.value),
                ("monitoring_run_id", monitoring_run_id),
            ),
        )
