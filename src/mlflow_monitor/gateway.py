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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

from mlflow_monitor.contract_checker import ContractEvidence
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckResult,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    GatewayConsistencyViolation,
    GatewayNamespaceViolation,
    TrainingRunMutationViolation,
)
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
        monitoring_run_id: Monitoring run identifier.
        sequence_index: Monotonic per-subject sequence index.
        lifecycle_status: Current lifecycle status.
        comparability_status: Optional comparability status for the run.
        contract_check_result: Optional contract check result for the run.
        references: Ordered typed references captured for the run.
    """

    monitoring_run_id: str
    sequence_index: int
    lifecycle_status: LifecycleStatus
    comparability_status: ComparabilityStatus | None = None
    contract_check_result: ContractCheckResult | None = None
    references: tuple[MonitoringRunReference, ...] = ()

    def __post_init__(self) -> None:
        """Freeze persisted references after a defensive copy."""
        object.__setattr__(
            self,
            "references",
            tuple(self.references),
        )


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
class CreateOrReuseMonitoringRunResult:
    """Gateway-owned monitoring run allocation or replay result.

    Attributes:
        monitoring_run_id: Monitoring run identifier owned by the gateway.
        sequence_index: Monotonic per-subject sequence index.
        existing_monitoring_run: Existing stored monitoring run record, if any.
        created: Whether this call created a new monitoring-run allocation.
    """

    monitoring_run_id: str
    sequence_index: int
    existing_monitoring_run: MonitoringRunRecord | None
    created: bool


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


@dataclass(frozen=True, slots=True)
class TimelineInitializationResult:
    """Result of timeline initialization attempt."""

    timeline_id: str
    created: bool


class MonitoringGateway(Protocol):
    """Protocol for gateway-mediated monitoring persistence operations."""

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        """Create a monitoring run allocation or return the existing idempotent one."""
        ...

    def initialize_timeline(
        self, subject_id: str, baseline_source_run_id: str
    ) -> TimelineInitializationResult:
        """Initialize timeline state once for a subject and return timeline initialiation status."""
        ...

    def resolve_active_lkg_monitoring_run_id(self, subject_id: str) -> str | None:
        """Resolve the active LKG monitoring run id for a subject, if any."""

    def set_active_lkg_monitoring_run_id(
        self, subject_id: str, monitoring_run_id: str | None
    ) -> None:
        """Set or clear the active LKG monitoring run id for a subject."""

    def upsert_monitoring_run(
        self,
        subject_id: str,
        monitoring_run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None = None,
        references: tuple[MonitoringRunReference, ...] | None = None,
    ) -> None:
        """Persist minimal monitoring run metadata for a subject."""

    def get_monitoring_run(
        self, subject_id: str, monitoring_run_id: str
    ) -> MonitoringRunRecord | None:
        """Return the monitoring run record for a given subject and monitoring run id if it exists."""  # noqa: E501

    def list_timeline_monitoring_runs(
        self,
        subject_id: str,
        exclude_failed: bool = False,
    ) -> tuple[MonitoringRunRecord, ...]:
        """List timeline monitoring runs for a subject with visibility filtering."""
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
        source_run_id: str,
        required_metrics: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required metrics that are absent from the source run."""
        ...

    def get_missing_source_run_artifacts(
        self,
        source_run_id: str,
        required_artifacts: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required artifacts that are absent from the source run."""
        ...

    def resolve_timeline_monitoring_run_id(
        self, subject_id: str, monitoring_run_id: str
    ) -> str | None:
        """Resolve one monitoring run id on the subject timeline."""
        ...

    def get_source_run_contract_evidence(self, source_run_id: str) -> ContractEvidence | None:
        """Return contract evidence for a source run, or None if the run is not found."""
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
        self._idempotency_bindings: dict[IdempotencyKey, tuple[str, int]] = {}
        self._active_lkg_by_subject: dict[str, str] = {}
        self._monitoring_runs_by_subject: dict[str, dict[str, MonitoringRunRecord]] = {}
        self._source_runs_by_id: dict[str, SourceRunRecord] = {}

    @property
    def config(self) -> GatewayConfig:
        """Return the gateway config."""
        return self._config

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        """Create a new monitoring run allocation or return the existing idempotent one."""
        self._validate_subject_id(key.subject_id)
        existing_binding = self._idempotency_bindings.get(key)
        if existing_binding is not None:
            existing_monitoring_run_id, sequence_index = existing_binding
            existing_monitoring_run = self.get_monitoring_run(
                key.subject_id, existing_monitoring_run_id
            )
            return CreateOrReuseMonitoringRunResult(
                monitoring_run_id=existing_monitoring_run_id,
                sequence_index=sequence_index,
                existing_monitoring_run=existing_monitoring_run,
                created=False,
            )

        sequence_index = self._next_sequence_by_subject.get(key.subject_id, 0)
        self._next_sequence_by_subject[key.subject_id] = sequence_index + 1
        new_monitoring_run_id = self._generate_monitoring_run_id()
        self._idempotency_bindings[key] = (new_monitoring_run_id, sequence_index)
        return CreateOrReuseMonitoringRunResult(
            monitoring_run_id=new_monitoring_run_id,
            sequence_index=sequence_index,
            existing_monitoring_run=None,
            created=True,
        )

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

    def resolve_active_lkg_monitoring_run_id(self, subject_id: str) -> str | None:
        """Resolve the active LKG monitoring run id for a subject, if any.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            The active LKG monitoring run id for the subject, or None if not set.
        """
        self._validate_subject_id(subject_id)
        return self._active_lkg_by_subject.get(subject_id)

    def set_active_lkg_monitoring_run_id(
        self, subject_id: str, monitoring_run_id: str | None
    ) -> None:
        """Set or clear the active LKG monitoring run id for a subject.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Monitoring run id to set as active LKG, or None to clear.

        Returns:
            None
        """
        self._validate_subject_id(subject_id)
        if monitoring_run_id is None:
            self._active_lkg_by_subject.pop(subject_id, None)
            return
        self._active_lkg_by_subject[subject_id] = monitoring_run_id

    def upsert_monitoring_run(
        self,
        subject_id: str,
        monitoring_run_id: str,
        lifecycle_status: LifecycleStatus,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None = None,
        references: tuple[MonitoringRunReference, ...] | None = None,
    ) -> None:
        """Persist minimal monitoring run metadata for a subject.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Monitoring run identifier.
            lifecycle_status: Current lifecycle status of the run.
            sequence_index: Monotonic per-subject sequence index for ordering.
            contract_check_result: Optional contract check result to persist for the run.
            references: Optional typed references to persist for the run.

        Raises:
            GatewayConsistencyViolation: If the upsert would override immutable and non empty fields
                                         of an existing monitoring run.

        Returns:
            None
        """
        self._validate_subject_id(subject_id)
        self._validate_monitoring_namespace(subject_id)

        comparability_status: ComparabilityStatus | None = (
            contract_check_result.status if contract_check_result else None
        )

        subject_runs = self._monitoring_runs_by_subject.setdefault(subject_id, {})
        monitoring_run = subject_runs.get(monitoring_run_id)

        if monitoring_run is None:
            subject_runs[monitoring_run_id] = MonitoringRunRecord(
                monitoring_run_id=monitoring_run_id,
                sequence_index=sequence_index,
                lifecycle_status=lifecycle_status,
                comparability_status=comparability_status,
                contract_check_result=contract_check_result,
                references=() if references is None else references,
            )
            return

        self._validate_upsert_existing_monitoring_run(
            monitoring_run,
            sequence_index,
            contract_check_result,
            references,
        )

        self._write_upsert_existing_monitoring_run(
            monitoring_run,
            subject_id,
            monitoring_run_id,
            lifecycle_status,
            comparability_status,
            contract_check_result,
            references,
        )

        return

    def get_monitoring_run(
        self, subject_id: str, monitoring_run_id: str
    ) -> MonitoringRunRecord | None:
        """Return the monitoring run record for a given subject and run id, if it exists.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Monitoring run identifier.

        Raises:
            GatewayNamespaceViolation: If the subject_id is invalid or does not match the expected
                                       monitoring namespace semantics.

        Returns:
            The monitoring run record for the given subject and run id, or None if not found.
        """
        self._validate_subject_id(subject_id)
        self._validate_monitoring_namespace(subject_id)
        subject_runs = self._monitoring_runs_by_subject.get(subject_id, {})
        return subject_runs.get(monitoring_run_id)

    def list_timeline_monitoring_runs(
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
                self._monitoring_runs_by_subject.get(subject_id, {}).values(),
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
        source_run_id: str,
        source_experiment: str | None,
        metrics: Mapping[str, float],
        artifacts: Sequence[str],
        environment: Mapping[str, str],
        features: Sequence[str],
        schema: Mapping[str, str],
        data_scope: str | None,
    ) -> None:
        """Register deterministic source-run state for tests and local use.

        Args:
            subject_id: Monitored subject identifier.
            source_run_id: Source training run identifier.
            source_experiment: Optional source experiment name for filtering.
            metrics: Mapping of metric names to values for the source run.
            artifacts: Sequence of artifact names logged in the source run.
            environment: Mapping of environment variable names to values for the source run.
            features: Sequence of feature names used in the source run.
            schema: Mapping of schema field names to types for the source run.
            data_scope: Optional string describing the data scope of the source run.

        Raises:
            GatewayNamespaceViolation: If the subject_id is invalid or does not match the expected
                                        monitoring namespace semantics.

        Returns:
            None
        """
        self._validate_subject_id(subject_id)
        self._source_runs_by_id[source_run_id] = SourceRunRecord(
            source_run_id=source_run_id,
            subject_id=subject_id,
            source_experiment=source_experiment,
            metrics=metrics,
            artifacts=tuple(artifacts),
            environment=environment,
            features=tuple(features),
            schema=schema,
            data_scope=data_scope,
        )

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        run_selector: str,
        runtime_source_run_id: str | None = None,
    ) -> str | None:
        """Resolve one concrete source training run id for prepare-stage use.

        Args:
            subject_id: Monitored subject identifier.
            source_experiment: Optional source experiment name to filter candidate runs.
            run_selector: Run selector string, either a concrete run id or a system token.
            runtime_source_run_id: Optional source run id from runtime context, used if the selector
                                    is the system default token.

        Raises:
            GatewayNamespaceViolation: If the subject_id is invalid or does not match the expected
                                        monitoring namespace semantics.

        Returns:
            The resolved source run id if a matching run is found, or None if not found.
        """
        self._validate_subject_id(subject_id)
        candidate_source_run_id = (
            runtime_source_run_id
            if run_selector == SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN
            else run_selector
        )
        if not candidate_source_run_id:
            return None

        source_run = self._source_runs_by_id.get(candidate_source_run_id)
        if source_run is None:
            return None
        if source_run.subject_id != subject_id:
            return None
        if source_experiment is not None and source_experiment != source_run.source_experiment:
            return None
        return source_run.source_run_id

    def get_missing_source_run_metrics(
        self,
        source_run_id: str,
        required_metrics: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required metrics that are absent from the source run.

        Args:
            source_run_id: Identifier of the source training run.
            required_metrics: Sequence of metric names required by the monitoring contract.

        Returns:
            Tuple of metric names that are required but not present in the source run.
        """
        source_run = self._source_runs_by_id.get(source_run_id)
        if source_run is None:
            return tuple(dict.fromkeys(required_metrics))
        return tuple(
            metric_name
            for metric_name in dict.fromkeys(required_metrics)
            if metric_name not in source_run.metrics
        )

    def get_missing_source_run_artifacts(
        self,
        source_run_id: str,
        required_artifacts: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required artifacts that are absent from the source run.

        Args:
            source_run_id: Identifier of the source training run.
            required_artifacts: Sequence of artifact names required by the monitoring contract.

        Returns:
            Tuple of artifact names that are required but not present in the source run.
        """
        source_run = self._source_runs_by_id.get(source_run_id)
        if source_run is None:
            return tuple(dict.fromkeys(required_artifacts))
        source_artifacts = set(source_run.artifacts)
        return tuple(
            artifact_name
            for artifact_name in dict.fromkeys(required_artifacts)
            if artifact_name not in source_artifacts
        )

    def resolve_timeline_monitoring_run_id(
        self, subject_id: str, monitoring_run_id: str
    ) -> str | None:
        """Resolve one monitoring run id on the subject timeline.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Candidate monitoring run identifier to resolve.

        Raises:
            GatewayNamespaceViolation: If the subject_id is invalid or does not match the expected
                                        monitoring namespace semantics.

        Returns:
            The resolved monitoring run id if it exists on the subject timeline,
            or None if not found.
        """
        self._validate_subject_id(subject_id)
        subject_runs = self._monitoring_runs_by_subject.get(subject_id, {})
        if monitoring_run_id not in subject_runs:
            return None
        return monitoring_run_id

    def mutate_training_run(self, source_run_id: str, updates: Mapping[str, str]) -> None:
        """Reject any attempt to mutate training runs through the gateway.

        Args:
            source_run_id: Identifier of the training run being mutated.
            updates: Mapping of fields being updated with their new values.

        Returns:
            None
        """
        raise TrainingRunMutationViolation(
            message=(
                "Training runs are read-only in MLflow-Monitor; "
                f"attempted mutation for source_run_id={source_run_id} with updates={dict(updates)}"
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
            (f"{key.source_run_id}|{key.recipe_id}|{key.recipe_version}"): monitoring_run_id
            for key, (monitoring_run_id, _sequence_index) in self._idempotency_bindings.items()
            if key.subject_id == subject_id
        }
        return MappingProxyType(bindings)

    def get_source_run_contract_evidence(self, source_run_id: str) -> ContractEvidence | None:
        """Return contract evidence for a source run, or None if the run is not found.

        Args:
            source_run_id: Identifier of the source training run.

        Returns:
            ContractEvidence containing the source run's metrics, environment, features, schema,
            and data scope; or None if the run is not found.
        """
        source_run = self._source_runs_by_id.get(source_run_id)
        if source_run is None:
            return None
        return ContractEvidence(
            metrics=source_run.metrics,
            environment=source_run.environment,
            features=source_run.features,
            schema=source_run.schema,
            data_scope=source_run.data_scope,
        )

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
        """Validate monitoring namespace semantics for write and read operations.

        Args:
            subject_id: Monitored subject identifier to validate.

        Raises:
            GatewayNamespaceViolation: If the subject_id is invalid or does not match the expected

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

    def _validate_upsert_existing_monitoring_run(
        self,
        monitoring_run: MonitoringRunRecord,
        sequence_index: int,
        contract_check_result: ContractCheckResult | None,
        references: tuple[MonitoringRunReference, ...] | None,
    ) -> None:
        """Validate that an existing monitoring run is updated consistently.

        Args:
            monitoring_run: The existing monitoring run record being updated.
            sequence_index: The new sequence index being written for the run.
            contract_check_result: The new contract check result being written for the run.
            references: The new typed references being written for the run.

        Raises:
            GatewayConsistencyViolation: If the upsert would override immutable and non empty fields
                                         of the existing monitoring run.

        Returns:
            None
        """
        details: tuple[tuple[str, str | int | None], ...] = ()

        if monitoring_run.sequence_index != sequence_index:
            details += (("sequence_index", sequence_index),)

        if (
            monitoring_run.contract_check_result is not None
            and contract_check_result is not None
            and monitoring_run.contract_check_result != contract_check_result
        ):
            details += (("contract_check_result", str(contract_check_result)),)

        if (
            monitoring_run.references
            and references is not None
            and monitoring_run.references != tuple(references)
        ):
            details += (("references", str(tuple(references))),)

        if bool(details):
            raise GatewayConsistencyViolation(
                code="monitoring_run_upsert_field_override",
                message=(
                    "Attempted to upsert monitoring run with immutable field value: "
                    + ", ".join(f"{field}={value!r}" for field, value in details)
                ),
                details=details,
            )

        return

    def _write_upsert_existing_monitoring_run(
        self,
        monitoring_run: MonitoringRunRecord,
        subject_id: str,
        monitoring_run_id: str,
        lifecycle_status: LifecycleStatus,
        comparability_status: ComparabilityStatus | None,
        contract_check_result: ContractCheckResult | None,
        references: tuple[MonitoringRunReference, ...] | None,
    ) -> None:
        """Persist an update to an existing monitoring run without mutating immutable fields.

        Args:
            monitoring_run: The existing monitoring run record being updated.
            subject_id: Monitored subject identifier.
            monitoring_run_id: Monitoring run identifier.
            lifecycle_status: Current lifecycle status of the run.
            comparability_status: Optional comparability status to persist for the run.
            contract_check_result: Optional contract check result to persist for the run.
            references: Optional typed references to persist for the run.

        Returns:
            None
        """
        effective_comparability_status = (
            monitoring_run.comparability_status
            if comparability_status is None
            else comparability_status
        )
        effective_contract_check_result = (
            monitoring_run.contract_check_result
            if contract_check_result is None
            else contract_check_result
        )
        effective_references = (
            monitoring_run.references if references is None else tuple(references)
        )
        self._monitoring_runs_by_subject[subject_id][monitoring_run_id] = MonitoringRunRecord(
            monitoring_run_id=monitoring_run_id,
            sequence_index=monitoring_run.sequence_index,
            lifecycle_status=lifecycle_status,
            comparability_status=effective_comparability_status,
            contract_check_result=effective_contract_check_result,
            references=effective_references,
        )

    def _generate_monitoring_run_id(self) -> str:
        """Return a new opaque monitoring run identifier."""
        return f"monitoring-run-{uuid4().hex}"
