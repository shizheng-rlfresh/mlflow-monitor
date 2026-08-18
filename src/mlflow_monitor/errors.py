"""Custom exception types for MLflow-Monitor v0."""

from __future__ import annotations

from dataclasses import dataclass

PREPARED_BASELINE_OVERRIDE_EXISTING_BASELINE = "prepare_baseline_override_existing_timeline"
_PREPARED_CONTEXT_INCONSISTENT = "prepared_context_inconsistent"
_MONITORING_ALLOCATION_INCONSISTENT = "monitoring_allocation_inconsistent"


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
        code=_PREPARED_CONTEXT_INCONSISTENT,
        message="Persisted prepared context is missing, malformed, or inconsistent.",
        details=(
            ("reason", reason),
            ("field", field),
        ),
    )


def monitoring_allocation_inconsistent(
    *, reason: str, message: str, details: tuple[tuple[str, str | int | None], ...] = ()
) -> GatewayConsistencyViolation:
    """Create a GatewayConsistencyViolation for inconsistent monitoring allocation."""
    return GatewayConsistencyViolation(
        code=_MONITORING_ALLOCATION_INCONSISTENT,
        message=message,
        details=(
            ("reason", reason),
            *details,
        ),
    )
