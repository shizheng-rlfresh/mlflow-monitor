"""Custom exception types for MLflow-Monitor v0."""

from __future__ import annotations

from dataclasses import dataclass


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
class InvalidRunTransition(ValueError):
    """Raised when workflow code requests an illegal lifecycle transition."""

    from_status: str
    to_status: str
    message: str

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
