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
