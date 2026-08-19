"""Custom exception types for workflow errors."""

from dataclasses import dataclass

PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE = "prepare_baseline_override_existing_baseline"


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
class TerminalRunRetryError(ValueError):
    """Raised when a duplicate request targets a terminal failed monitoring run."""

    code: str
    message: str
    details: tuple[tuple[str, str | int | None], ...] = ()

    def __str__(self) -> str:
        """Return the error message when the exception is converted to a string."""
        return self.message
