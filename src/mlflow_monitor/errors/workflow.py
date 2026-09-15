"""Custom exception types for workflow errors."""

from dataclasses import dataclass

PREPARE_BASELINE_OVERRIDE_EXISTING_BASELINE = "prepare_baseline_override_existing_baseline"
ANALYZE_FINDING_POLICY_EVALUATION_FAILED = "analyze_finding_policy_evaluation_failed"
ANALYZE_FINDING_POLICY_OUTPUT_INVALID = "analyze_finding_policy_output_invalid"
ANALYZE_FINDING_POLICY_OUTPUT_INCONSISTENT = "analyze_finding_policy_output_inconsistent"
ANALYZE_MISSING_CURRENT_SOURCE_RUN = "analyze_missing_current_source_run"


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
class AnalyzeStageError(ValueError):
    """Raised when Analyze lacks its current source or policy execution fails."""

    code: str
    message: str
    details: tuple[tuple[str, str], ...]

    def __str__(self) -> str:
        """Return the bounded error message for operator display."""
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
