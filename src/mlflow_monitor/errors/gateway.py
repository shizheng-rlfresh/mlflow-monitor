"""Custom exception types for generic gateway inconsistencies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from mlflow_monitor.domain import DiffReferenceKind

from .utils import render_message_fields


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
        rendered_field = render_message_fields(updates)
        self.message = (
            "Training runs are read-only in MLflow-Monitor; "
            f"Attempted to mutate Source Training Run {source_run_id!r}; "
            f"update fields: {rendered_field}."
        )
        super().__init__(self.message)


# GatewayConsistencyViolation error factories


class GatewayConsistencyCode(StrEnum):
    """Code for gateway consistency violations."""

    PREPARED_CONTEXT_INCONSISTENT = "prepared_context_inconsistent"
    MONITORING_ALLOCATION_INCONSISTENT = "monitoring_allocation_inconsistent"
    MONITORING_TIMELINE_INCONSISTENT = "monitoring_timeline_inconsistent"
    MONITORING_RUN_UPSERT_FIELD_OVERRIDE = "monitoring_run_upsert_field_override"
    TIMELINE_STATE_NOT_FOUND_FOR_SUBJECT_ID = "timeline_state_not_found_for_subject_id"
    MONITORING_RUN_JSON_ARTIFACT_INCONSISTENT = "monitoring_run_json_artifact_inconsistent"
    MONITORING_RUN_SUBJECT_INCONSISTENT = "monitoring_run_subject_inconsistent"
    MONITORING_REFERENCE_INCONSISTENT = "monitoring_reference_inconsistent"


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
        if fields:
            rendered_field = render_message_fields(fields)
            message = f"Attempted to override immutable Monitoring Run field {rendered_field}."
        else:
            message = "Attempted to override immutable Monitoring Run field."

        return cls(
            code=GatewayConsistencyCode.MONITORING_RUN_UPSERT_FIELD_OVERRIDE.value,
            message=message,
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
