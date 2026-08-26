"""Gateway models for MLflow Monitor."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckResult,
    LifecycleStatus,
    MonitoringRunReference,
)


@dataclass(frozen=True, slots=True)
class GatewayConfig:
    """Configuration for persistence gateway implementations.

    Attributes:
        namespace_prefix: Prefix used for all monitoring namespace writes.
    """

    namespace_prefix: str = "mlflow_monitor"


@dataclass(frozen=True, slots=True)
class MonitoringRunRecord:
    """Minimal monitoring run record stored by the in-memory gateway.

    Attributes:
        monitoring_run_id: Monitoring run identifier.
        source_run_id: Immutable source training run identifier.
        sequence_index: Monotonic per-subject sequence index.
        lifecycle_status: Current lifecycle status.
        comparability_status: Optional comparability status for the run.
        contract_check_result: Optional contract check result for the run.
        references: Ordered typed references captured for the run.
    """

    monitoring_run_id: str
    source_run_id: str
    sequence_index: int
    lifecycle_status: LifecycleStatus
    comparability_status: ComparabilityStatus | None = None
    contract_check_result: ContractCheckResult | None = None
    references: tuple[MonitoringRunReference, ...] = ()

    def __post_init__(self) -> None:
        """Validate source identity and freeze references after a defensive copy."""
        if not self.source_run_id.strip():
            raise ValueError("MonitoringRunRecord.source_run_id must be non-empty.")
        object.__setattr__(
            self,
            "references",
            tuple(self.references),
        )


@dataclass(frozen=True, slots=True)
class TimelineState:
    """Reconciled timeline metadata exposed by a monitoring gateway.

    Attributes:
        timeline_id: Stable timeline identifier.
        baseline_source_run_id: Immutable baseline source run id, or `None`
            before any claim or legacy projection establishes it.

    Note:
        When the first Monitoring Run is allocated, `baseline_source_run_id`
        is `None`. Successful Prepare reconciles identical per-run claims into
        this Timeline projection.
    """

    timeline_id: str
    baseline_source_run_id: str | None


@dataclass(frozen=True, slots=True)
class TimelineClaim:
    """Baseline claim value for a Monitoring Run on a Timeline.

    Attributes:
        monitoring_run_id: Monitoring run identifier making the claim.
        source_run_id: Source training run identifier for the Monitoring Run
            making the claim.
        claimed_baseline_source_run_id: Baseline source run id being claimed.
    """

    monitoring_run_id: str
    source_run_id: str
    claimed_baseline_source_run_id: str


@dataclass(frozen=True, slots=True)
class IdempotencyKey:
    """Canonical identity for one monitoring intent.

    Attributes:
        subject_id: Monitored subject identifier.
        source_run_id: Source training run identifier.
        recipe_id: Recipe identifier used for this run.
        recipe_version: Recipe version used for this run.
    """

    subject_id: str
    source_run_id: str
    recipe_id: str
    recipe_version: str


@dataclass(frozen=True, slots=True)
class CreateOrReuseMonitoringRunResult:
    """Gateway-owned monitoring run allocation or replay result.

    Attributes:
        monitoring_run_id: Monitoring run identifier owned by the gateway.
        source_run_id: Immutable source training run identifier.
        sequence_index: Monotonic per-subject sequence index.
        timeline_id: Stable timeline identifier associated with this monitoring run.
        existing_monitoring_run: Existing stored monitoring run record, if any. This may be
            None even when a prior idempotency binding already exists.
        allocated: Whether this call created a new idempotency binding / monitoring-run
            allocation.
    """

    monitoring_run_id: str
    source_run_id: str
    timeline_id: str
    sequence_index: int
    existing_monitoring_run: MonitoringRunRecord | None
    allocated: bool

    def __post_init__(self) -> None:
        """Require the Timeline identity established by allocation."""
        if not isinstance(self.timeline_id, str) or not self.timeline_id.strip():
            raise ValueError(
                "CreateOrReuseMonitoringRunResult.timeline_id must be a non-empty string."
            )


@dataclass(frozen=True, slots=True)
class SourceRunRecord:
    """Minimal source training run record used by the in-memory gateway.

    Attributes:
        source_run_id: Source training run identifier.
        subject_id: Monitored subject identifier.
        source_experiment: Optional source experiment name for filtering.
        metrics: Mapping of metric names to values for the source run.
        artifacts: Sequence of artifact names logged in the source run.
        environment: Mapping of environment variable names to values for the source run.
        features: Sequence of feature names used in the source run.
        schema: Mapping of schema field names to types for the source run.
        data_scope: Optional string describing the data scope of the source run.
    """

    source_run_id: str
    subject_id: str
    source_experiment: str | None
    metrics: Mapping[str, float]
    artifacts: tuple[str, ...]
    environment: Mapping[str, str]
    features: tuple[str, ...]
    schema: Mapping[str, str]
    data_scope: str | None

    def __post_init__(self) -> None:
        """Freeze nested source-run collections after defensive copies."""
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(self, "features", tuple(self.features))
        object.__setattr__(self, "schema", MappingProxyType(dict(self.schema)))
