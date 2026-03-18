"""Persistence gateway abstractions for MLflow-Monitor v0.

This module separates three kinds of state used by workflow:

1. Source training runs
   Existing training-side runs that monitoring reads from during prepare.
   In the in-memory gateway, tests seed these with `add_source_run()`.

2. Timeline state
   Per-subject monitoring configuration anchored by a pinned
   `baseline_source_run_id`. This is absent before the first monitoring run
   and is created by `initialize_timeline()`.

3. Monitoring runs
   Runs owned by the monitoring timeline itself. In the in-memory gateway,
   tests seed or update these with `upsert_monitoring_run()`.

Lifecycle sketch:

- Before first monitoring run:
  - source training run exists
  - timeline state does not exist yet
  - monitoring runs do not exist yet

- First prepare:
  - workflow resolves the source run through the gateway
  - if timeline state is missing, workflow may initialize it with a caller-supplied baseline
  - prepare then continues using the pinned timeline baseline

- Later prepares:
  - workflow reads existing timeline state
  - baseline is resolved from timeline state, not from caller input
  - previous/custom monitoring references are resolved from monitoring runs
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from mlflow_monitor.domain import LifecycleStatus
from mlflow_monitor.errors import GatewayNamespaceViolation, TrainingRunMutationViolation
from mlflow_monitor.recipe import SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN


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
        run_id: Monitoring run identifier.
        sequence_index: Monotonic per-subject sequence index.
        lifecycle_status: Current lifecycle status.
    """

    run_id: str
    sequence_index: int
    lifecycle_status: LifecycleStatus


@dataclass(frozen=True, slots=True)
class TimelineState:
    """Persisted timeline metadata stored by the in-memory gateway.

    Attributes:
        timeline_id: Stable timeline identifier.
        baseline_source_run_id: Pinned baseline source run id.
    """

    timeline_id: str
    baseline_source_run_id: str


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
class SourceRunRecord:
    """Minimal source training run record used by the in-memory gateway."""

    run_id: str
    subject_id: str
    source_experiment: str | None
    metrics: Mapping[str, float]
    artifacts: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze nested source-run collections after defensive copies."""
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))


@dataclass(frozen=True, slots=True)
class TimelineInitializationResult:
    """Result of timeline initialization attempt."""

    timeline_id: str
    created: bool


class MonitoringGateway(Protocol):
    """Protocol for gateway-mediated monitoring persistence operations."""

    def get_or_create_idempotent_run_id(
        self,
        key: IdempotencyKey,
        factory: Callable[[], str],
    ) -> str:
        """Return existing run id for the key, or create and bind a new run id."""
        ...

    def initialize_timeline(
        self, subject_id: str, baseline_source_run_id: str
    ) -> TimelineInitializationResult:
        """Initialize timeline state once for a subject and return timeline initialiation status."""
        ...

    def reserve_sequence_index(self, subject_id: str) -> int:
        """Reserve and return the next sequence index for a subject."""
        ...

    def resolve_active_lkg_run_id(self, subject_id: str) -> str | None:
        """Resolve the active LKG run id for a subject, if any."""

    def set_active_lkg_run_id(self, subject_id: str, run_id: str | None) -> None:
        """Set or clear the active LKG run id for a subject."""

    def upsert_monitoring_run(
        self,
        subject_id: str,
        run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
    ) -> None:
        """Persist minimal monitoring run metadata for a subject."""

    def list_timeline_runs(
        self,
        subject_id: str,
        exclude_failed: bool = False,
    ) -> tuple[MonitoringRunRecord, ...]:
        """List timeline runs for a subject with visibility filtering."""
        ...

    def get_timeline_state(self, subject_id: str) -> TimelineState | None:
        """Return timeline state for a subject, if it exists."""
        ...

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        run_selector: str,
        runtime_source_run_id: str | None = None,
    ) -> str | None:
        """Resolve one concrete source training run id for prepare-stage use."""
        ...

    def get_missing_source_run_metrics(
        self,
        run_id: str,
        required_metrics: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required metrics that are absent from the source run."""
        ...

    def get_missing_source_run_artifacts(
        self,
        run_id: str,
        required_artifacts: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required artifacts that are absent from the source run."""
        ...

    def resolve_timeline_run_id(self, subject_id: str, run_id: str) -> str | None:
        """Resolve one monitoring run id on the subject timeline."""
        ...


class InMemoryMonitoringGateway:
    """Deterministic in-memory gateway implementation for testing and local use."""

    def __init__(self, config: GatewayConfig) -> None:
        """Initialize the in-memory stores for monitoring state.

        Args:
            config: Configuration for the gateway instance.

        Returns:
            None
        """
        self._config = config
        self._timeline_by_subject: dict[str, TimelineState] = {}
        self._next_sequence_by_subject: dict[str, int] = {}
        self._idempotency_bindings: dict[IdempotencyKey, str] = {}
        self._active_lkg_by_subject: dict[str, str] = {}
        self._runs_by_subject: dict[str, dict[str, MonitoringRunRecord]] = {}
        self._source_runs_by_id: dict[str, SourceRunRecord] = {}

    @property
    def config(self) -> GatewayConfig:
        """Return the gateway config."""
        return self._config

    def get_or_create_idempotent_run_id(
        self,
        key: IdempotencyKey,
        factory: Callable[[], str],
    ) -> str:
        """Return existing run id for the key, or create and bind a new run id.

        Args:
            key: Idempotency key representing the monitoring intent.
            factory: Factory function to generate a new run id if needed.

        Returns:
            The existing or newly created run id bound to the key.
        """
        self._validate_subject_id(key.subject_id)
        existing_run_id = self._idempotency_bindings.get(key)
        if existing_run_id is not None:
            return existing_run_id
        new_run_id = factory()
        self._idempotency_bindings[key] = new_run_id
        return new_run_id

    def initialize_timeline(
        self, subject_id: str, baseline_source_run_id: str
    ) -> TimelineInitializationResult:
        """Initialize timeline state once for a subject and return timeline id.

        Args:
            subject_id: Monitored subject identifier.
            baseline_source_run_id: Source training run id to pin as timeline baseline.

        Returns:
            Timeline initialization result containing the timeline id and status.
        """
        if not baseline_source_run_id:
            raise GatewayNamespaceViolation(message="baseline_source_run_id must be non-empty.")

        self._validate_subject_id(subject_id)
        if subject_id in self._timeline_by_subject:
            return TimelineInitializationResult(
                timeline_id=self._timeline_by_subject[subject_id].timeline_id,
                created=False,
            )
        timeline_state = TimelineState(
            timeline_id=f"timeline-{subject_id}",
            baseline_source_run_id=baseline_source_run_id,
        )
        self._timeline_by_subject[subject_id] = timeline_state
        return TimelineInitializationResult(
            timeline_id=timeline_state.timeline_id,
            created=True,
        )

    def get_timeline_state(self, subject_id: str) -> TimelineState | None:
        """Return timeline state for test and workflow read access.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            The timeline state for the subject, or None if not initialized.
        """
        self._validate_subject_id(subject_id)
        return self._timeline_by_subject.get(subject_id)

    def reserve_sequence_index(self, subject_id: str) -> int:
        """Reserve and return the next sequence index for a subject."""
        self._validate_subject_id(subject_id)
        next_index = self._next_sequence_by_subject.get(subject_id, 0)
        self._next_sequence_by_subject[subject_id] = next_index + 1
        return next_index

    def resolve_active_lkg_run_id(self, subject_id: str) -> str | None:
        """Resolve the active LKG run id for a subject, if any.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            The active LKG run id for the subject, or None if not set.
        """
        self._validate_subject_id(subject_id)
        return self._active_lkg_by_subject.get(subject_id)

    def set_active_lkg_run_id(self, subject_id: str, run_id: str | None) -> None:
        """Set or clear the active LKG run id for a subject.

        Args:
            subject_id: Monitored subject identifier.
            run_id: Run id to set as active LKG, or None to clear.

        Returns:
            None
        """
        self._validate_subject_id(subject_id)
        if run_id is None:
            self._active_lkg_by_subject.pop(subject_id, None)
            return
        self._active_lkg_by_subject[subject_id] = run_id

    def upsert_monitoring_run(
        self,
        subject_id: str,
        run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
    ) -> None:
        """Persist minimal monitoring run metadata for a subject.

        Args:
            subject_id: Monitored subject identifier.
            run_id: Monitoring run identifier.
            lifecycle_status: Current lifecycle status of the run.
            sequence_index: Monotonic per-subject sequence index for ordering.

        Returns:
            None
        """
        self._validate_subject_id(subject_id)
        self._validate_monitoring_namespace(subject_id)
        subject_runs = self._runs_by_subject.setdefault(subject_id, {})
        subject_runs[run_id] = MonitoringRunRecord(
            run_id=run_id,
            sequence_index=sequence_index,
            lifecycle_status=lifecycle_status,
        )

    def list_timeline_runs(
        self,
        subject_id: str,
        exclude_failed: bool = False,
    ) -> tuple[MonitoringRunRecord, ...]:
        """List timeline runs for a subject with visibility filtering.

        Trasient states are excluded and failed runs are included by default.

        Args:
            subject_id: Monitored subject identifier.
            exclude_failed: Whether exclude runs with FAILED lifecycle status.

        Returns:
            Tuple of monitoring run records for the subject, ordered by sequence index.
        """
        self._validate_subject_id(subject_id)
        runs = tuple(
            sorted(
                self._runs_by_subject.get(subject_id, {}).values(),
                key=lambda run: run.sequence_index,
            )
        )

        if exclude_failed:
            return tuple(run for run in runs if run.lifecycle_status is LifecycleStatus.CLOSED)
        return tuple(
            run
            for run in runs
            if run.lifecycle_status in {LifecycleStatus.FAILED, LifecycleStatus.CLOSED}
        )

    def add_source_run(
        self,
        *,
        subject_id: str,
        run_id: str,
        source_experiment: str | None,
        metrics: Mapping[str, float],
        artifacts: Sequence[str],
    ) -> None:
        """Register deterministic source-run state for tests and local use."""
        self._validate_subject_id(subject_id)
        self._source_runs_by_id[run_id] = SourceRunRecord(
            run_id=run_id,
            subject_id=subject_id,
            source_experiment=source_experiment,
            metrics=metrics,
            artifacts=tuple(artifacts),
        )

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        run_selector: str,
        runtime_source_run_id: str | None = None,
    ) -> str | None:
        """Resolve one concrete source training run id for prepare-stage use."""
        self._validate_subject_id(subject_id)
        candidate_run_id = (
            runtime_source_run_id
            if run_selector == SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN
            else run_selector
        )
        if not candidate_run_id:
            return None

        source_run = self._source_runs_by_id.get(candidate_run_id)
        if source_run is None:
            return None
        if source_run.subject_id != subject_id:
            return None
        if source_experiment is not None and source_experiment != source_run.source_experiment:
            return None
        return source_run.run_id

    def get_missing_source_run_metrics(
        self,
        run_id: str,
        required_metrics: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required metrics that are absent from the source run."""
        source_run = self._source_runs_by_id.get(run_id)
        if source_run is None:
            return tuple(dict.fromkeys(required_metrics))
        return tuple(
            metric_name
            for metric_name in dict.fromkeys(required_metrics)
            if metric_name not in source_run.metrics
        )

    def get_missing_source_run_artifacts(
        self,
        run_id: str,
        required_artifacts: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required artifacts that are absent from the source run."""
        source_run = self._source_runs_by_id.get(run_id)
        if source_run is None:
            return tuple(dict.fromkeys(required_artifacts))
        source_artifacts = set(source_run.artifacts)
        return tuple(
            artifact_name
            for artifact_name in dict.fromkeys(required_artifacts)
            if artifact_name not in source_artifacts
        )

    def resolve_timeline_run_id(self, subject_id: str, run_id: str) -> str | None:
        """Resolve one monitoring run id on the subject timeline."""
        self._validate_subject_id(subject_id)
        subject_runs = self._runs_by_subject.get(subject_id, {})
        if run_id not in subject_runs:
            return None
        return run_id

    def mutate_training_run(self, run_id: str, updates: Mapping[str, str]) -> None:
        """Reject any attempt to mutate training runs through the gateway.

        Args:
            run_id: Identifier of the training run being mutated.
            updates: Mapping of fields being updated with their new values.

        Returns:
            None
        """
        raise TrainingRunMutationViolation(
            message=(
                "Training runs are read-only in MLflow-Monitor; "
                f"attempted mutation for run_id={run_id} with updates={dict(updates)}"
            )
        )

    def build_monitoring_namespace(self, subject_id: str) -> str:
        """Build and return the monitoring namespace for a subject.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            The monitoring namespace for the subject.
        """
        self._validate_namespace_prefix(self._config.namespace_prefix)
        self._validate_subject_id(subject_id)
        return f"{self._config.namespace_prefix}/{subject_id}"

    def idempotency_bindings(self, subject_id: str) -> Mapping[str, str]:
        """Return immutable idempotency bindings for a subject.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            Mapping of idempotency keys to run ids for the subject.
        """
        self._validate_subject_id(subject_id)
        bindings = {
            (f"{key.source_run_id}|{key.recipe_id}|{key.recipe_version}"): run_id
            for key, run_id in self._idempotency_bindings.items()
            if key.subject_id == subject_id
        }
        return MappingProxyType(bindings)

    def _validate_namespace_prefix(self, prefix: str) -> None:
        """Validate namespace prefix can safely compose a monitoring namespace.

        Args:
            prefix: Namespace prefix to validate.

        Returns:
            None
        """
        if not prefix or "/" in prefix:
            raise GatewayNamespaceViolation(
                message=(f"namespace_prefix must be non-empty and must not contain '/': {prefix!r}")
            )

    def _validate_subject_id(self, subject_id: str) -> None:
        """Validate subject id can safely compose a monitoring namespace.

        Args:
            subject_id: Monitored subject identifier to validate.

        Returns:
            None
        """
        if not subject_id or "/" in subject_id:
            raise GatewayNamespaceViolation(
                message=(f"subject_id must be non-empty and must not contain '/': {subject_id!r}")
            )

    def _validate_monitoring_namespace(self, subject_id: str) -> None:
        """Validate monitoring namespace semantics for write operations.

        Args:
            subject_id: Monitored subject identifier to validate.

        Returns:
            None
        """
        namespace = self.build_monitoring_namespace(subject_id)
        expected_prefix = f"{self._config.namespace_prefix}/"
        if not namespace.startswith(expected_prefix):
            raise GatewayNamespaceViolation(
                message=(
                    "Monitoring writes must target namespace "
                    f"'{expected_prefix}*'; got {namespace!r}."
                )
            )
