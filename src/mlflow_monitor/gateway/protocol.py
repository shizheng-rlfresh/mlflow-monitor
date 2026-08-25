"""Persistence gateway abstractions for MLflow-Monitor v0.

This module separates three kinds of state used by workflow:

1. Source training runs
   Existing training-side runs that monitoring reads from during prepare.
   In the in-memory gateway, tests seed these with `add_source_run()`.

2. Timeline state
   Per-subject monitoring configuration created by the first Monitoring Run
   allocation. Its `baseline_source_run_id` remains `None` until a successful
   Prepare reconciles an immutable Monitoring Run baseline claim.

3. Monitoring runs
   Runs owned by the monitoring timeline itself. In the in-memory gateway,
   tests seed or update these with `upsert_monitoring_run()`.

Lifecycle sketch:

- Before first monitoring run:
  - source training run exists
  - timeline state does not exist yet
  - monitoring runs do not exist yet

- First Monitoring Run allocation and Prepare:
  - allocation creates Timeline identity with no established baseline
  - workflow resolves the source run through the gateway
  - Prepare validates all live inputs, then writes an immutable baseline claim
  - the Timeline baseline projection is reconciled from durable claims

- Later prepares:
  - workflow reads existing timeline state
  - baseline is resolved from timeline state, not from caller input
  - each successful Monitoring Run writes the same immutable baseline claim
  - previous/custom monitoring references are resolved from monitoring runs
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mlflow_monitor.contract_checker import ContractEvidence
from mlflow_monitor.domain import (
    ContractCheckResult,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.gateway.models import (
    CreateOrReuseMonitoringRunResult,
    IdempotencyKey,
    MonitoringRunRecord,
    TimelineState,
)
from mlflow_monitor.result_contract import MonitorRunResult


class MonitoringGateway(Protocol):
    """Protocol for gateway-mediated monitoring persistence operations."""

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        """Create or reuse a monitoring-run allocation and return replay context.

        The returned `existing_monitoring_run` reflects whether a persisted monitoring-run
        record already exists. It may be None even when `allocated` is False if an earlier
        allocation succeeded but the monitoring-run record has not been persisted yet.
        """
        ...

    def reconcile_timeline_baseline(
        self,
        subject_id: str,
        monitoring_run_id: str,
        baseline_source_run_id: str,
    ) -> TimelineState:
        """Persist one immutable baseline claim and reconcile Timeline state."""
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
        source_run_id: str,
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
        source_run_id: str,
    ) -> str | None:
        """Resolve one invocation-owned Source Training Run identifier."""
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

    def finalize_monitoring_run_result(
        self,
        *,
        monitoring_run_id: str,
        result: MonitorRunResult,
    ) -> None:
        """Ensure final result payloads and terminal monitoring-run state are persisted."""
        ...

    def read_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        path: str,
    ) -> dict[str, Any] | None:
        """Read one dictionary payload from a JSON artifact on a monitoring run."""
        ...

    def write_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        data: dict[str, Any],
        path: str,
    ) -> None:
        """Write one dictionary payload as a JSON artifact on a monitoring run."""
        ...

    def get_source_run_metrics(
        self,
        source_run_id: str,
        metric_names: Sequence[str] | None = None,
    ) -> dict[str, float] | None:
        """Return a dictionary of metric names to values for a source run."""
        ...
