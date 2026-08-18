"""Gateway exception types and factory-owned consistency messages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from mlflow_monitor.domain import DiffReferenceKind


@dataclass(frozen=True, slots=True)
class GatewayNamespaceViolation(ValueError):
    """Raised when a gateway operation violates namespace constraints."""

    message: str

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


class TrainingRunMutationViolation(ValueError):
    """Raised when code attempts to mutate Source Training Run data."""

    code = "training_run_mutation_violation"
    reason = "attempted_mutation"

    def __init__(self, source_run_id: str, updates: Mapping[str, str]) -> None:
        """Initialize the violation with the source run ID and attempted updates."""
        update_fields = ", ".join(sorted(updates))
        self.message = (
            "Training runs are read-only in MLflow-Monitor; "
            f"Attempted to mutate Source Training Run {source_run_id!r}; "
            f"update fields: {update_fields}."
        )
        super().__init__(self.message)


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
class GatewayConsistencyViolation(ValueError):
    """Raised when a gateway operation violates consistency constraints."""

    code: str
    message: str
    details: tuple[tuple[str, str | int | None], ...] = ()

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message

    # monitoring run upsert field override error factory
    @classmethod
    def monitoring_run_upsert_field_override(
        cls,
        *,
        fields: tuple[tuple[str, str | int | None], ...] = (),
    ) -> GatewayConsistencyViolation:
        """Create a violation for one immutable Monitoring Run field override."""
        field = [f for f, _ in fields]
        return cls(
            code=GatewayConsistencyCode.MONITORING_RUN_UPSERT_FIELD_OVERRIDE.value,
            message=f"Attempted to override immutable Monitoring Run field {field!r}.",
            details=fields,
        )

    @classmethod
    def monitoring_run_upsert_source_binding_conflict(
        cls,
        *,
        monitoring_run_id: str,
        source_run_id: str | None,
        persisted_source_run_id: str | None,
    ) -> GatewayConsistencyViolation:
        """Create a violation for a Monitoring Run bound to another source run."""
        return cls(
            code=GatewayConsistencyCode.MONITORING_RUN_UPSERT_FIELD_OVERRIDE.value,
            message=(
                "Monitoring Run source binding conflicts with durable state for "
                f"monitoring_run_id={monitoring_run_id!r}; "
                f"source_run_id={source_run_id!r} does not match "
                f"persisted_source_run_id={persisted_source_run_id!r}."
            ),
            details=(
                ("monitoring_run_id", monitoring_run_id),
                ("source_run_id", source_run_id),
                ("persisted_source_run_id", persisted_source_run_id),
            ),
        )

    # timeline state not found for subject ID error factory
    @classmethod
    def timeline_state_not_found_for_subject_id(
        cls, *, subject_id: str
    ) -> GatewayConsistencyViolation:
        """Create a GatewayConsistencyViolation for missing timeline state for a subject ID."""
        return cls(
            code=GatewayConsistencyCode.TIMELINE_STATE_NOT_FOUND_FOR_SUBJECT_ID.value,
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
            code=GatewayConsistencyCode.MONITORING_RUN_JSON_ARTIFACT_INCONSISTENT.value,
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
            code=GatewayConsistencyCode.MONITORING_RUN_SUBJECT_INCONSISTENT.value,
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
            code=GatewayConsistencyCode.MONITORING_REFERENCE_INCONSISTENT.value,
            message=(
                f"Monitoring reference of kind={kind.value!r} is inconsistent for "
                f"monitoring_run_id={monitoring_run_id!r}."
            ),
            details=(
                ("kind", kind.value),
                ("monitoring_run_id", monitoring_run_id),
            ),
        )


@dataclass(frozen=True, slots=True)
class AllocationConsistencyViolation(GatewayConsistencyViolation):
    """Raised when durable Monitoring Run allocation state is inconsistent."""

    @classmethod
    def _create(
        cls,
        *,
        reason: MonitoringAllocationInconsistentReason,
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
            reason=MonitoringAllocationInconsistentReason.DUPLICATE_IDENTITY,
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
            reason=MonitoringAllocationInconsistentReason.DUPLICATE_SEQUENCE,
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
            reason=MonitoringAllocationInconsistentReason.SEQUENCE_GAP,
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
            reason=MonitoringAllocationInconsistentReason.INVALID_ALLOCATION,
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
            reason=MonitoringAllocationInconsistentReason.INVALID_ALLOCATION,
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
            reason=MonitoringAllocationInconsistentReason.INVALID_ALLOCATION,
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
            reason=MonitoringAllocationInconsistentReason.NEXT_SEQUENCE_AHEAD,
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
            reason=MonitoringAllocationInconsistentReason.UNKNOWN_POINTER,
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
            reason=MonitoringAllocationInconsistentReason.UNKNOWN_TAG,
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
            reason=MonitoringAllocationInconsistentReason.SOURCE_BINDING_CONFLICT,
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
            reason=MonitoringAllocationInconsistentReason.TIMELINE_CONFLICT,
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


class PreparedContextInconsistentReason(StrEnum):
    """Reasons for prepared context inconsistent reason code."""

    MISSING_ARTIFACT = "missing_artifact"
    UNSUPPORTED_ARTIFACT_SCHEMA_VERSION = "unsupported_artifact_schema_version"
    ALLOCATION_IDENTITY_MISMATCH = "allocation_identity_mismatch"
    INVALID_FIELD_TYPE = "invalid_field_type"
    INVALID_FIELDS = "invalid_fields"
    EFFECTIVE_RECIPE_MISMATCH = "effective_recipe_mismatch"
    CONTRACT_MISMATCH = "contract_mismatch"
    INVALID_REFERENCE = "invalid_reference"
    BASELINE_REFERENCE_MISMATCH = "baseline_reference_mismatch"
    NONCANONICAL_REFERENCES = "noncanonical_references"
    BROKEN_ARTIFACT = "broken_artifact"


@dataclass(frozen=True, slots=True)
class PreparedContextConsistencyViolation(GatewayConsistencyViolation):
    """Raised when durable Monitoring Run allocation state is inconsistent."""

    @classmethod
    def _create(
        cls,
        *,
        reason: PreparedContextInconsistentReason,
        details: tuple[tuple[str, str | int | None], ...],
    ) -> PreparedContextConsistencyViolation:
        """Create a violation with its stable code and normalized reason."""
        return cls(
            code=GatewayConsistencyCode.PREPARED_CONTEXT_INCONSISTENT.value,
            message="Persisted prepared context is missing, malformed, or inconsistent.",
            details=(("reason", reason.value), *details),
        )

    # prepared context inconsistent error factory
    @classmethod
    def missing_artifact(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for missing artifact."""
        return cls._create(
            reason=PreparedContextInconsistentReason.MISSING_ARTIFACT,
            details=(("field", field),),
        )

    @classmethod
    def broken_artifact(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for broken artifact."""
        return cls._create(
            reason=PreparedContextInconsistentReason.BROKEN_ARTIFACT,
            details=(("field", field),),
        )

    @classmethod
    def unsupported_artifact_schema_version(
        cls, *, field: str
    ) -> PreparedContextConsistencyViolation:
        """Create a violation for unsupported artifact schema version."""
        return cls._create(
            reason=PreparedContextInconsistentReason.UNSUPPORTED_ARTIFACT_SCHEMA_VERSION,
            details=(("field", field),),
        )

    @classmethod
    def allocation_identity_mismatch(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for allocation identity mismatch."""
        return cls._create(
            reason=PreparedContextInconsistentReason.ALLOCATION_IDENTITY_MISMATCH,
            details=(("field", field),),
        )

    @classmethod
    def invalid_field_type(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for invalid field type."""
        return cls._create(
            reason=PreparedContextInconsistentReason.INVALID_FIELD_TYPE,
            details=(("field", field),),
        )

    @classmethod
    def invalid_fields(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for invalid fields."""
        return cls._create(
            reason=PreparedContextInconsistentReason.INVALID_FIELDS,
            details=(("field", field),),
        )

    @classmethod
    def effective_recipe_mismatch(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for effective recipe mismatch."""
        return cls._create(
            reason=PreparedContextInconsistentReason.EFFECTIVE_RECIPE_MISMATCH,
            details=(("field", field),),
        )

    @classmethod
    def contract_mismatch(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for contract mismatch."""
        return cls._create(
            reason=PreparedContextInconsistentReason.CONTRACT_MISMATCH,
            details=(("field", field),),
        )

    @classmethod
    def invalid_reference(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for invalid reference."""
        return cls._create(
            reason=PreparedContextInconsistentReason.INVALID_REFERENCE,
            details=(("field", field),),
        )

    @classmethod
    def baseline_reference_mismatch(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for baseline reference mismatch."""
        return cls._create(
            reason=PreparedContextInconsistentReason.BASELINE_REFERENCE_MISMATCH,
            details=(("field", field),),
        )

    @classmethod
    def noncanonical_references(cls, *, field: str) -> PreparedContextConsistencyViolation:
        """Create a violation for noncanonical references."""
        return cls._create(
            reason=PreparedContextInconsistentReason.NONCANONICAL_REFERENCES,
            details=(("field", field),),
        )
