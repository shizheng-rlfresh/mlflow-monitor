"""Canonical synchronous result contract for Python API and CLI outputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus, MonitoringRunReference


@dataclass(frozen=True, slots=True)
class MonitorRunError:
    """Minimal structured error payload for monitoring run failures.

    Attributes:
        code: Stable machine-readable error code.
        message: Human-readable error summary.
        stage: Optional stage name where the failure occurred.
        details: Optional additional key-value details for diagnostics.
    """

    code: str
    message: str
    stage: str | None = None
    details: Mapping[str, str] | None = None

    def __post_init__(self) -> None:
        """Freeze error details after a defensive copy."""
        if self.details is not None:
            object.__setattr__(
                self,
                "details",
                MappingProxyType(dict(self.details)),
            )

    def to_dict(self) -> dict[str, str | None | dict[str, str]]:
        """Serialize this error payload into a deterministic dictionary."""
        return {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "details": None if self.details is None else dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class MonitorRunResult:
    """Canonical Python API/CLI run result envelope.

    Attributes:
        monitoring_run_id: Unique monitoring run identifier.
        subject_id: Monitored subject identifier.
        timeline_id: Timeline identifier if known for this run.
        lifecycle_status: Current workflow lifecycle status.
        comparability_status: Optional comparability verdict if computed.
        summary: Optional structured summary payload.
        finding_ids: Finding identifiers associated with the run.
        diff_ids: Diff identifiers associated with the run.
        references: Ordered typed references used in analysis.
        error: Structured error payload for failed runs only.
    """

    monitoring_run_id: str
    subject_id: str
    timeline_id: str | None
    lifecycle_status: LifecycleStatus
    comparability_status: ComparabilityStatus | None
    summary: Mapping[str, str] | None
    finding_ids: tuple[str, ...]
    diff_ids: tuple[str, ...]
    references: tuple[MonitoringRunReference, ...]
    error: MonitorRunError | None = None

    def __post_init__(self) -> None:
        """Freeze mapping and sequence fields after defensive copies."""
        if self.lifecycle_status is LifecycleStatus.FAILED and self.error is None:
            raise ValueError(
                "MonitorRunResult with lifecycle_status=failed requires a non-null error."
            )
        if self.lifecycle_status is not LifecycleStatus.FAILED and self.error is not None:
            raise ValueError(
                "MonitorRunResult with non-failed lifecycle_status must have error=None."
            )
        if self.summary is not None:
            object.__setattr__(
                self,
                "summary",
                MappingProxyType(dict(self.summary)),
            )
        object.__setattr__(
            self,
            "finding_ids",
            tuple(self.finding_ids),
        )
        object.__setattr__(
            self,
            "diff_ids",
            tuple(self.diff_ids),
        )
        object.__setattr__(
            self,
            "references",
            tuple(self.references),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize this result envelope into a deterministic dictionary."""
        return {
            "monitoring_run_id": self.monitoring_run_id,
            "subject_id": self.subject_id,
            "timeline_id": self.timeline_id,
            "lifecycle_status": self.lifecycle_status.value,
            "comparability_status": (
                None if self.comparability_status is None else self.comparability_status.value
            ),
            "summary": None if self.summary is None else dict(self.summary),
            "finding_ids": list(self.finding_ids),
            "diff_ids": list(self.diff_ids),
            "references": [reference.to_dict() for reference in self.references],
            "error": None if self.error is None else self.error.to_dict(),
        }
