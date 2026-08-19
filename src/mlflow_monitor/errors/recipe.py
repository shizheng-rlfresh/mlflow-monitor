"""Custom exception types for recipe validation errors."""

from dataclasses import dataclass


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
