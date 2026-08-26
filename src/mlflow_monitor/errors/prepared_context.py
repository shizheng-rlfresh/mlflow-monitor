"""Custom exception types for prepared context gateway inconsistencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .gateway import GatewayConsistencyCode, GatewayConsistencyViolation


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
    """Raised when persisted prepared context is missing or inconsistent."""

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
