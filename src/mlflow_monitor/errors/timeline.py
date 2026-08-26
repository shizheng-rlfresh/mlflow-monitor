"""Custom exception types for timeline gateway inconsistencies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from mlflow_monitor.gateway.models import TimelineClaim

from .gateway import GatewayConsistencyCode, GatewayConsistencyViolation


class TimelineInconsistentReason(StrEnum):
    """Reasons for monitoring Timeline inconsistency."""

    REQUEST_CONFLICT = "request_conflict"
    CLAIMS_CONFLICT = "claims_conflict"
    PROJECTION_CONFLICT = "projection_conflict"
    CLAIM_ADDRESS_MISMATCH = "claim_address_mismatch"


@dataclass(frozen=True, slots=True)
class TimelineConsistencyViolation(GatewayConsistencyViolation):
    """Raised when durable baseline claims or their projection conflict."""

    @classmethod
    def _create(
        cls,
        *,
        reason: TimelineInconsistentReason,
        message: str,
        details: tuple[tuple[str, str | int | None], ...],
    ) -> TimelineConsistencyViolation:
        """Create a violation with its stable code and normalized reason."""
        return cls(
            code=GatewayConsistencyCode.MONITORING_TIMELINE_INCONSISTENT.value,
            message=message,
            details=(("reason", reason.value), *details),
        )

    @classmethod
    def request_conflict(
        cls,
        *,
        requested_claim: TimelineClaim,
        existing_baseline_source_run_id: str | None,
    ) -> TimelineConsistencyViolation:
        """Create a violation for one Monitoring Run requesting a conflicting baseline."""
        return cls._create(
            reason=TimelineInconsistentReason.REQUEST_CONFLICT,
            message=(
                "Baseline request conflicts with durable Timeline state for "
                f"monitoring_run_id={requested_claim.monitoring_run_id!r}."
            ),
            details=(
                ("monitoring_run_id", requested_claim.monitoring_run_id),
                ("source_run_id", requested_claim.source_run_id),
                ("existing_baseline_source_run_id", existing_baseline_source_run_id),
                (
                    "requested_baseline_source_run_id",
                    requested_claim.claimed_baseline_source_run_id,
                ),
            ),
        )

    @classmethod
    def claims_conflict(
        cls,
        *,
        claims: tuple[TimelineClaim, ...],
        subject_id: str,
    ) -> TimelineConsistencyViolation:
        """Create a violation for Monitoring Runs claiming different baselines."""
        details = tuple(
            field
            for claim in claims
            for field in (
                ("monitoring_run_id", claim.monitoring_run_id),
                ("source_run_id", claim.source_run_id),
                ("baseline_source_run_id", claim.claimed_baseline_source_run_id),
            )
        )
        return cls._create(
            reason=TimelineInconsistentReason.CLAIMS_CONFLICT,
            message=(
                f"Monitoring Runs of subject_id={subject_id!r} "
                "contain conflicting immutable baseline claims."
            ),
            details=(("subject_id", subject_id), *details),
        )

    @classmethod
    def projection_conflict(
        cls,
        *,
        claims: tuple[TimelineClaim, ...],
        projected_baseline_source_run_id: str,
        subject_id: str,
    ) -> TimelineConsistencyViolation:
        """Create a violation for a projection contradicting durable claims."""
        claim_details = tuple(
            field
            for claim in claims
            for field in (
                ("monitoring_run_id", claim.monitoring_run_id),
                ("source_run_id", claim.source_run_id),
                ("baseline_source_run_id", claim.claimed_baseline_source_run_id),
            )
        )
        return cls._create(
            reason=TimelineInconsistentReason.PROJECTION_CONFLICT,
            message=f"subject_id={subject_id!r} baseline projection contradicts durable claims.",
            details=(
                ("subject_id", subject_id),
                ("projected_baseline_source_run_id", projected_baseline_source_run_id),
                *claim_details,
            ),
        )

    @classmethod
    def claim_address_mismatch(
        cls,
        *,
        monitoring_run_id: str,
        source_run_id: str,
        tag_key: str,
        claimed_baseline_source_run_id: str,
    ) -> TimelineConsistencyViolation:
        """Create a violation for a claim tag that does not match its expected address."""
        return cls._create(
            reason=TimelineInconsistentReason.CLAIM_ADDRESS_MISMATCH,
            message=(
                f"monitoring_run_id={monitoring_run_id!r} claimed baseline tag={tag_key!r} "
                "does not match its expected address for "
                f"claimed_baseline_source_run_id={claimed_baseline_source_run_id!r}."
            ),
            details=(
                ("monitoring_run_id", monitoring_run_id),
                ("source_run_id", source_run_id),
                ("tag_key", tag_key),
                ("claimed_baseline_source_run_id", claimed_baseline_source_run_id),
            ),
        )
