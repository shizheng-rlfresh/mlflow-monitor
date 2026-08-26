"""Reference domain models for mlflow-monitor v0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DiffReferenceKind(StrEnum):
    """Reference kinds supported by diff records."""

    BASELINE = "baseline"
    PREVIOUS = "previous"
    LKG = "lkg"
    CUSTOM = "custom"


_MONITORING_RUN_REFERENCE_KINDS = frozenset(
    (
        DiffReferenceKind.BASELINE,
        DiffReferenceKind.PREVIOUS,
        DiffReferenceKind.LKG,
        DiffReferenceKind.CUSTOM,
    )
)


@dataclass(frozen=True, slots=True)
class DiffReference:
    """Reference descriptor for one diff comparison target.

    Attributes:
        kind: The reference kind for this diff (e.g., baseline, previous, lkg).
        monitoring_run_id: Referenced monitoring run identifier, or None for the
            source-only baseline.
        source_run_id: Immutable source run identifier for the reference. This is
            temporarily optional only for the legacy structural kind removed by
            V0-003.
    """

    kind: DiffReferenceKind
    monitoring_run_id: str | None
    source_run_id: str | None

    def __post_init__(self) -> None:
        """Validate that reference identity presence matches the reference kind."""
        _validate_reference_identity(
            entity="DiffReference",
            kind=self.kind,
            monitoring_run_id=self.monitoring_run_id,
            source_run_id=self.source_run_id,
        )


@dataclass(frozen=True, slots=True)
class MonitoringRunReference:
    """Typed run-level reference captured for one monitoring run.

    Attributes:
        kind: Reference kind for the monitoring run lineage.
        monitoring_run_id: Referenced monitoring run identifier, or None for the
            source-only baseline.
        source_run_id: Immutable source run identifier for the reference.
    """

    kind: DiffReferenceKind
    monitoring_run_id: str | None
    source_run_id: str

    def __post_init__(self) -> None:
        """Validate run-level reference shape."""
        try:
            kind = DiffReferenceKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"Unsupported monitoring run reference kind {self.kind!r}.") from exc
        if kind not in _MONITORING_RUN_REFERENCE_KINDS:
            raise ValueError(f"Unsupported monitoring run reference kind {kind.value!r}.")
        object.__setattr__(self, "kind", kind)
        _validate_reference_identity(
            entity="MonitoringRunReference",
            kind=kind,
            monitoring_run_id=self.monitoring_run_id,
            source_run_id=self.source_run_id,
        )

    def to_dict(self) -> dict[str, str | None]:
        """Serialize this run-level reference into a deterministic dictionary."""
        return {
            "kind": self.kind.value,
            "monitoring_run_id": self.monitoring_run_id,
            "source_run_id": self.source_run_id,
        }


def _validate_reference_identity(
    *,
    entity: str,
    kind: DiffReferenceKind,
    monitoring_run_id: str | None,
    source_run_id: str | None,
) -> None:
    """Validate source-only baseline and paired monitoring reference identity."""
    if source_run_id is None or not source_run_id.strip():
        raise ValueError(f"{entity} with kind={kind.value!r} requires a non-empty source_run_id.")
    if kind is DiffReferenceKind.BASELINE:
        if monitoring_run_id is not None:
            raise ValueError(f"{entity} with kind='baseline' must not set monitoring_run_id.")
        return
    if monitoring_run_id is None or not monitoring_run_id.strip():
        raise ValueError(
            f"{entity} with kind={kind.value!r} requires a non-empty monitoring_run_id."
        )
