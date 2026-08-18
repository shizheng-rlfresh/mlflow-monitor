"""Custom exception types for MLflow-Monitor v0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mlflow_monitor.domain import DiffReferenceKind

PREPARED_BASELINE_OVERRIDE_EXISTING_BASELINE = "prepared_baseline_override_existing_baseline"


class GatewayConsistencyCode(StrEnum):
    """Code for gateway consistency violations."""

    PREPARED_CONTEXT_INCONSISTENT = "prepared_context_inconsistent"
    MONITORING_ALLOCATION_INCONSISTENT = "monitoring_allocation_inconsistent"
    MONITORING_RUN_UPSERT_FIELD_OVERRIDE = "monitoring_run_upsert_field_override"
    TIMELINE_STATE_NOT_FOUND_FOR_SUBJECT_ID = "timeline_state_not_found_for_subject_id"
    MONITORING_RUN_JSON_ARTIFACT_INCONSISTENT = "monitoring_run_json_artifact_inconsistent"
    MONITORING_RUN_SUBJECT_INCONSISTENT = "monitoring_run_subject_inconsistent"
    MONITORING_REFERENCE_INCONSISTENT = "monitoring_reference_inconsistent"


class MonitorAllocationReason(StrEnum):
    """Reasons for monitoring run allocation inconsistency."""

    DUPLICATE_IDENTITY = "duplicate_identity"
    DUPLICATE_SEQUENCE = "duplicate_sequence"
    SEQUENCE_GAP = "sequence_gap"
    INVALID_ALLOCATION = "invalid_allocation"
    NEXT_SEQUENCE_AHEAD = "next_sequence_ahead"
    UNKNOWN_POINTER = "unknown_pointer"
    SOURCE_BINDING_CONFLICT = "source_binding_conflict"
    TIMELINE_CONFLICT = "timeline_conflict"


@dataclass(frozen=True, slots=True)
class InvariantViolation(ValueError):
    """Raised when a domain invariant is violated."""

    code: str
    message: str
    entity: str
    field: str | None = None

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


@dataclass(frozen=True, slots=True)
class GatewayNamespaceViolation(ValueError):
    """Raised when a gateway operation violates namespace constraints."""

    message: str

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


@dataclass(frozen=True, slots=True)
class GatewayConsistencyViolation(ValueError):
    """Raised when a gateway operation violates consistency constraints."""

    code: str
    message: str
    details: tuple[tuple[str, str | int | None], ...] = ()

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


@dataclass(frozen=True, slots=True)
class PrepareStageError(ValueError):
    """Raised when prepare-stage workflow resolution fails deterministically."""

    code: str
    message: str
    details: tuple[tuple[str, str | None], ...] = ()

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


@dataclass(frozen=True, slots=True)
class CheckStageError(ValueError):
    """Raised when check-stage workflow evaluation fails deterministically."""

    code: str
    message: str
    details: tuple[tuple[str, str | None], ...] = ()

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


@dataclass(frozen=True, slots=True)
class ContractResolutionError(ValueError):
    """Raised when recipe-selected contract binding cannot be resolved."""

    code: str
    message: str
    details: tuple[tuple[str, str | None], ...] = ()

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


@dataclass(frozen=True, slots=True)
class TerminalRunRetryError(ValueError):
    """Raised when a duplicate request targets a terminal failed monitoring run."""

    code: str
    message: str
    details: tuple[tuple[str, str | int | None], ...] = ()

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message


@dataclass(frozen=True, slots=True)
class RecipeValidationIssue:
    """One machine-readable issue discovered during recipe validation."""

    code: str
    section: str
    message: str
    field: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeValidationError(ValueError):
    """Raised when one or more recipe validation checks fail."""

    issues: tuple[RecipeValidationIssue, ...]

    def __str__(self) -> str:
        """Return a deterministic joined message for all validation issues."""
        return "; ".join(issue.message for issue in self.issues)


# error factories


# GatewayConsistencyViolation error factories


def prepared_context_inconsistent(*, reason: str, field: str) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for inconsistent prepared context."""
    return GatewayConsistencyViolation(
        code=GatewayConsistencyCode.PREPARED_CONTEXT_INCONSISTENT,
        message="Persisted prepared context is missing, malformed, or inconsistent.",
        details=(
            ("reason", reason),
            ("field", field),
        ),
    )


def monitoring_allocation_inconsistent(
    *,
    reason: MonitorAllocationReason | str,
    message: str,
    details: tuple[tuple[str, str | int | None], ...] = (),
) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for inconsistent monitoring allocation."""
    normalized_reason = MonitorAllocationReason(reason)
    return GatewayConsistencyViolation(
        code=GatewayConsistencyCode.MONITORING_ALLOCATION_INCONSISTENT,
        message=message,
        details=(
            ("reason", normalized_reason.value),
            *details,
        ),
    )


def monitoring_run_upsert_field_override(
    message: str, details: tuple[tuple[str, str | int | None], ...] = ()
) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for monitoring run upsert field override."""
    return GatewayConsistencyViolation(
        code=GatewayConsistencyCode.MONITORING_RUN_UPSERT_FIELD_OVERRIDE,
        message=message,
        details=(*details,),
    )


def timeline_state_not_found_for_subject_id(*, subject_id: str) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for missing timeline state for a subject ID."""
    return GatewayConsistencyViolation(
        code=GatewayConsistencyCode.TIMELINE_STATE_NOT_FOUND_FOR_SUBJECT_ID,
        message=f"Timeline state not found for subject_id={subject_id!r}.",
        details=(("subject_id", subject_id),),
    )


def monitoring_run_json_artifact_inconsistent(
    *, monitoring_run_id: str, path: str
) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for inconsistent monitoring run JSON artifact."""
    return GatewayConsistencyViolation(
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


def monitoring_run_subject_inconsistent(
    *, subject_id: str, monitoring_run_id: str
) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for a monitoring run not indexed on the subject ID."""
    return GatewayConsistencyViolation(
        code=GatewayConsistencyCode.MONITORING_RUN_SUBJECT_INCONSISTENT,
        message=(
            f"monitoring_run_id={monitoring_run_id!r} is not indexed on subject_id={subject_id!r}."
        ),
        details=(
            ("subject_id", subject_id),
            ("monitoring_run_id", monitoring_run_id),
        ),
    )


def monitoring_reference_inconsistent(
    *, kind: DiffReferenceKind, monitoring_run_id: str
) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for an inconsistent monitoring reference."""
    return GatewayConsistencyViolation(
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
