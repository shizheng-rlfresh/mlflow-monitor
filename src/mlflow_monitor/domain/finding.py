"""Findings domain models for mlflow-monitor v0."""

from dataclasses import dataclass
from enum import StrEnum


class FindingSeverity(StrEnum):
    """Priority levels for interpreted findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class FindingDraft:
    """Transient policy conclusion awaiting package-owned materialization.

    Attributes:
        finding_rule_id: Stable rule identity within the Finding policy.
        severity: Priority assigned by the Finding policy.
        category: Category assigned by the Finding policy.
        summary: Human-readable conclusion produced by the Finding policy.
        recommendation: Human-readable recommended response.
        evidence_diff_ids: Diff identities supporting the conclusion.
        evidence_compatibility_ids: Compatibility Evidence identities supporting
            the conclusion.
    """

    finding_rule_id: str
    severity: FindingSeverity
    category: str
    summary: str
    recommendation: str
    evidence_diff_ids: tuple[str, ...]
    evidence_compatibility_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the transient Finding conclusion."""
        _validate_finding_text_fields(
            entity="FindingDraft",
            value=self,
            field_names=(
                "finding_rule_id",
                "category",
                "summary",
                "recommendation",
            ),
        )
        _validate_finding_severity(entity="FindingDraft", severity=self.severity)
        _freeze_finding_evidence(self)


@dataclass(frozen=True, slots=True)
class Finding:
    """Immutable policy conclusion for one Monitoring Run.

    Attributes:
        finding_id: Deterministic identity for the Finding.
        monitoring_run_id: Monitoring Run that owns the Finding.
        source_run_id: Source Training Run evaluated by the Monitoring Run.
        finding_policy_id: Identifier of the Finding policy that emitted the draft.
        finding_policy_version: Version of the Finding policy that emitted the draft.
        finding_rule_id: Stable rule identity within the Finding policy.
        severity: Priority assigned by the Finding policy.
        category: Category assigned by the Finding policy.
        summary: Human-readable conclusion produced by the Finding policy.
        recommendation: Human-readable recommended response.
        evidence_diff_ids: Diff identities supporting the conclusion.
        evidence_compatibility_ids: Compatibility Evidence identities supporting
            the conclusion.
    """

    finding_id: str
    monitoring_run_id: str
    source_run_id: str
    finding_policy_id: str
    finding_policy_version: str
    finding_rule_id: str
    severity: FindingSeverity
    category: str
    summary: str
    recommendation: str
    evidence_diff_ids: tuple[str, ...]
    evidence_compatibility_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate and freeze the materialized Finding conclusion."""
        _validate_finding_text_fields(
            entity="Finding",
            value=self,
            field_names=(
                "finding_id",
                "monitoring_run_id",
                "source_run_id",
                "finding_policy_id",
                "finding_policy_version",
                "finding_rule_id",
                "category",
                "summary",
                "recommendation",
            ),
        )
        _validate_finding_severity(entity="Finding", severity=self.severity)
        _freeze_finding_evidence(self)


def _validate_finding_text_fields(
    *,
    entity: str,
    value: object,
    field_names: tuple[str, ...],
) -> None:
    """Validate required Finding text and identity fields."""
    for field_name in field_names:
        field_value = getattr(value, field_name)
        if not isinstance(field_value, str) or not field_value.strip():
            raise ValueError(f"{entity} requires a non-empty string for field {field_name!r}.")


def _validate_finding_severity(*, entity: str, severity: object) -> None:
    """Validate that Finding severity uses the canonical enum."""
    if not isinstance(severity, FindingSeverity):
        raise ValueError(f"{entity} requires a FindingSeverity for field 'severity'.")


def _freeze_finding_evidence(value: FindingDraft | Finding) -> None:
    """Defensively freeze and validate both Finding evidence collections."""
    entity = type(value).__name__
    evidence_collections: dict[str, tuple[str, ...]] = {}
    for field_name in ("evidence_diff_ids", "evidence_compatibility_ids"):
        supplied_ids = getattr(value, field_name)
        if isinstance(supplied_ids, str):
            raise ValueError(f"{entity} requires a collection for field {field_name!r}.")
        try:
            evidence_ids = tuple(supplied_ids)
        except TypeError as exc:
            raise ValueError(f"{entity} requires a collection for field {field_name!r}.") from exc
        if any(
            not isinstance(evidence_id, str) or not evidence_id.strip()
            for evidence_id in evidence_ids
        ):
            raise ValueError(
                f"{entity} requires non-empty string identities in field {field_name!r}."
            )
        evidence_collections[field_name] = evidence_ids
        object.__setattr__(value, field_name, evidence_ids)

    if not any(evidence_collections.values()):
        raise ValueError(f"{entity} requires at least one evidence identity.")
