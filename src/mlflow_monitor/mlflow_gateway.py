"""Real-MLflow monitoring gateway for the current create/prepare/check workflow.

This module is the runtime implementation of the `MonitoringGateway` protocol
for the real-MLflow path. Normal timeline reads are driven by:

1. One monitoring experiment per subject, named `{namespace_prefix}/{subject_id}`.
2. Experiment tags that act as the timeline index.
3. Direct `get_run()` calls for both training and monitoring runs.

Monitoring-run allocation additionally uses a narrow active-run search to
reconcile partial experiment-index writes from durable run identity tags.

Design constraints that matter here:

- Training runs are strictly read-only. The gateway may inspect metrics, params,
  tags, and artifact paths on training runs, but it must never mutate them.
- Monitoring-run allocation is gateway-owned because MLflow assigns run ids at
  `create_run()` time.
- Monitoring-run identity tags are the allocation source of truth, while the
  experiment-tag timeline index is a repairable projection used by normal reads.
- Monitoring-run baseline claims are the baseline source of truth, while the
  experiment baseline tag is a repairable projection of their unique value.
- Final run output is stored as `outputs/result.json` and the MLflow run is
  explicitly terminated by the gateway once orchestration reaches a terminal
  success or owned-failure outcome.

The implementation is intentionally conservative. It preserves the current
create -> prepare -> check semantics and detects contradictory durable state
without locks, winner selection, broader query support, or more expressive
persistence than the current runtime supports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any

from mlflow_monitor.contract_checker import ContractEvidence
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckResult,
    DiffReferenceKind,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import (
    AllocationConsistencyViolation,
    GatewayConsistencyViolation,
    GatewayNamespaceViolation,
    TimelineConsistencyViolation,
    TrainingRunMutationViolation,
)
from mlflow_monitor.gateway_models import (
    CreateOrReuseMonitoringRunResult,
    GatewayConfig,
    IdempotencyKey,
    MonitoringRunRecord,
    TimelineClaim,
    TimelineState,
)
from mlflow_monitor.mlflow_client import MonitoringRunTagSnapshot, MonitorMLflowClient
from mlflow_monitor.result_contract import MonitorRunResult
from mlflow_monitor.utils import canonical_json

_BASELINE_CLAIM_TAG_PREFIX = "monitoring.baseline_source_run_id."
_BASELINE_PROJECTION_TAG = "training.baseline_run_id"
_IDEMPOTENCY_TAG_SUFFIX = ".monitoring_run_id"
_LKG_TAG = "monitoring.lkg_run_id"
_LATEST_TAG = "monitoring.latest_run_id"
_NEXT_SEQUENCE_TAG = "monitoring.next_sequence_index"
_RUN_TAG_PREFIX = "monitoring.run."
_SOURCE_RUN_TAG = "training.source_run_id"
_SEQUENCE_INDEX_TAG = "monitoring.sequence_index"
_LIFECYCLE_STATUS_TAG = "monitoring.lifecycle_status"
_COMPARABILITY_STATUS_TAG = "monitoring.comparability_status"
_RECIPE_ID_TAG = "monitoring.recipe_id"
_RECIPE_VERSION_TAG = "monitoring.recipe_version"
_REFERENCE_TAG_PREFIX = "monitoring.reference."
_CONTRACT_CHECK_ARTIFACT_PATH = "outputs/contract_check.json"
_RESULT_ARTIFACT_PATH = "outputs/result.json"
_VISIBLE_NON_FAILED_STATUSES = frozenset({LifecycleStatus.CHECKED, LifecycleStatus.CLOSED})
_REFERENCE_KINDS = ("baseline", "previous", "lkg", "custom")


def _baseline_claim_tag_key(baseline_source_run_id: str) -> str:
    """Return the value-addressed tag key for one baseline claim."""
    digest = sha256(baseline_source_run_id.encode()).hexdigest()
    return f"{_BASELINE_CLAIM_TAG_PREFIX}{digest}"


class RequiredMonitoringRunTags(StrEnum):
    """Required monitoring run tag keys."""

    SOURCE_RUN_TAG = _SOURCE_RUN_TAG
    SEQUENCE_INDEX_TAG = _SEQUENCE_INDEX_TAG
    LIFECYCLE_STATUS_TAG = _LIFECYCLE_STATUS_TAG
    RECIPE_ID_TAG = _RECIPE_ID_TAG
    RECIPE_VERSION_TAG = _RECIPE_VERSION_TAG


@dataclass(frozen=True, slots=True)
class _MonitoringRunAllocation:
    """Durable monitoring-run allocation identity reconstructed from run tags."""

    key: IdempotencyKey
    monitoring_run_id: str
    sequence_index: int


class MLflowMonitoringGateway:
    """Monitoring gateway backed by real MLflow experiments and runs.

    The gateway translates protocol-level operations such as reconciling
    immutable baseline claims or creating or reusing the Monitoring Run for a
    Source Training Run into:

    - monitoring experiment creation/read
    - experiment-tag index maintenance
    - monitoring-run tag persistence
    - direct training-run evidence reads
    - final artifact logging and MLflow terminal status updates

    Args:
        config: Gateway namespace configuration.
        tracking_uri: Optional MLflow tracking URI override. Normal runtime
            usage relies on MLflow's default env/config resolution; this
            parameter mainly exists for tests and controlled internal wiring.
        artifact_location: Optional monitoring experiment artifact root passed
            only when the monitoring experiment is first created. Existing
            experiments keep their current artifact root.
    """

    def __init__(
        self,
        config: GatewayConfig,
        tracking_uri: str | None = None,
        artifact_location: str | None = None,
    ) -> None:
        """Initialize the gateway and its thin MLflow adapter."""
        self._config = config
        self._artifact_location = artifact_location
        self._mlflow = MonitorMLflowClient(tracking_uri=tracking_uri)

    @property
    def config(self) -> GatewayConfig:
        """Return the gateway configuration."""
        return self._config

    def create_or_reuse_monitoring_run(
        self, key: IdempotencyKey
    ) -> CreateOrReuseMonitoringRunResult:
        """Create or reuse a monitoring run after reconciling allocation state.

        Monitoring-run identity tags are the durable allocation record. Before
        allocating, the gateway validates those records and repairs their
        experiment-tag timeline projection. Ambiguous or contradictory state
        fails closed rather than risking a duplicate allocation.

        Args:
            key: Canonical monitoring intent identity.

        Returns:
            Allocation or replay information needed by orchestration.
        """
        self._validate_subject_id(key.subject_id)
        experiment_id = self._get_or_create_experiment_id(key.subject_id)
        experiment_tags = self._mlflow.get_monitoring_experiment_tags(experiment_id)
        allocations_by_key, sequence_index = self._reconcile_monitoring_run_allocations(
            subject_id=key.subject_id,
            experiment_id=experiment_id,
            experiment_tags=experiment_tags,
        )
        existing_allocation = allocations_by_key.get(key)
        if existing_allocation is not None:
            return CreateOrReuseMonitoringRunResult(
                monitoring_run_id=existing_allocation.monitoring_run_id,
                source_run_id=existing_allocation.key.source_run_id,
                timeline_id=experiment_id,
                sequence_index=existing_allocation.sequence_index,
                existing_monitoring_run=self.get_monitoring_run(
                    key.subject_id,
                    existing_allocation.monitoring_run_id,
                ),
                allocated=False,
            )

        source_run_name = self._mlflow.get_run_name(key.source_run_id)
        # MLflow assigns the monitoring run id, so the gateway has to create the
        # run before it can finish updating the experiment-level timeline index.
        monitoring_run_info = self._mlflow.create_monitoring_run(
            experiment_id,
            tags={
                RequiredMonitoringRunTags.SOURCE_RUN_TAG: key.source_run_id,
                RequiredMonitoringRunTags.SEQUENCE_INDEX_TAG: str(sequence_index),
                RequiredMonitoringRunTags.LIFECYCLE_STATUS_TAG: LifecycleStatus.CREATED.value,
                RequiredMonitoringRunTags.RECIPE_ID_TAG: key.recipe_id,
                RequiredMonitoringRunTags.RECIPE_VERSION_TAG: key.recipe_version,
            },
            source_run_name=source_run_name,
        )
        self._set_experiment_tags(
            experiment_id,
            {
                f"{_RUN_TAG_PREFIX}{sequence_index}": monitoring_run_info.run_id,
                _LATEST_TAG: monitoring_run_info.run_id,
                _NEXT_SEQUENCE_TAG: str(sequence_index + 1),
                self._idempotency_tag(key.source_run_id): monitoring_run_info.run_id,
            },
        )
        return CreateOrReuseMonitoringRunResult(
            monitoring_run_id=monitoring_run_info.run_id,
            source_run_id=key.source_run_id,
            timeline_id=experiment_id,
            sequence_index=sequence_index,
            existing_monitoring_run=None,
            allocated=True,
        )

    def reconcile_timeline_baseline(
        self,
        subject_id: str,
        monitoring_run_id: str,
        baseline_source_run_id: str,
    ) -> TimelineState:
        """Persist one immutable Monitoring Run claim and reconcile its projection.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Claiming Monitoring Run identifier.
            baseline_source_run_id: Immutable Baseline Source Run identifier.

        Returns:
            Reconciled Timeline state.

        Raises:
            GatewayConsistencyViolation: If no Timeline allocation exists or
                the Monitoring Run is not allocated to the subject.
            TimelineConsistencyViolation: If claims or their projection conflict.
        """
        self._validate_subject_id(subject_id)
        if not baseline_source_run_id:
            raise GatewayNamespaceViolation(message="baseline_source_run_id must be non-empty.")

        experiment_id = self._get_experiment_id(subject_id)
        if experiment_id is None:
            raise GatewayConsistencyViolation.timeline_state_not_found_for_subject_id(
                subject_id=subject_id
            )

        experiment_tags = self._mlflow.get_monitoring_experiment_tags(experiment_id)
        allocations_by_run_id = self._read_timeline_allocations(
            subject_id=subject_id,
            experiment_id=experiment_id,
        )
        if not allocations_by_run_id:
            raise GatewayConsistencyViolation.timeline_state_not_found_for_subject_id(
                subject_id=subject_id
            )
        if monitoring_run_id not in allocations_by_run_id:
            raise GatewayConsistencyViolation.monitoring_run_subject_inconsistent(
                subject_id=subject_id,
                monitoring_run_id=monitoring_run_id,
            )
        source_run_id = allocations_by_run_id[monitoring_run_id].key.source_run_id

        claims = self._read_baseline_claims(
            subject_id=subject_id,
            experiment_id=experiment_id,
        )
        established_baseline = self._resolve_baseline_projection(
            subject_id=subject_id,
            experiment_id=experiment_id,
            experiment_tags=experiment_tags,
            claims=claims,
            repair_missing_projection=False,
        )
        requested_claim = TimelineClaim(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            claimed_baseline_source_run_id=baseline_source_run_id,
        )
        if established_baseline is not None and established_baseline != baseline_source_run_id:
            raise TimelineConsistencyViolation.request_conflict(
                requested_claim=requested_claim,
                existing_baseline_source_run_id=established_baseline,
            )

        if requested_claim not in claims:
            self._mlflow.set_monitoring_run_tags(
                monitoring_run_id,
                {_baseline_claim_tag_key(baseline_source_run_id): baseline_source_run_id},
            )

        # Read claims again for concurrency safety.
        claims = self._read_baseline_claims(
            subject_id=subject_id,
            experiment_id=experiment_id,
        )
        unique_claim = self._get_unique_claimed_baseline(subject_id, claims)
        if requested_claim not in claims or unique_claim != baseline_source_run_id:
            raise TimelineConsistencyViolation.request_conflict(
                requested_claim=requested_claim,
                existing_baseline_source_run_id=unique_claim,
            )

        reconciled_baseline = self._resolve_baseline_projection(
            subject_id=subject_id,
            experiment_id=experiment_id,
            experiment_tags=self._mlflow.get_monitoring_experiment_tags(experiment_id),
            claims=claims,
            repair_missing_projection=True,
        )
        return TimelineState(
            timeline_id=experiment_id,
            baseline_source_run_id=reconciled_baseline,
        )

    def resolve_active_lkg_monitoring_run_id(self, subject_id: str) -> str | None:
        """Resolve the current active LKG monitoring run id, if any.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            The active LKG monitoring run id, or `None` when unset.
        """
        experiment_tags = self._get_experiment_tags_if_present(subject_id)
        if experiment_tags is None:
            return None
        lkg_monitoring_run_id = experiment_tags.get(_LKG_TAG)
        return lkg_monitoring_run_id or None

    def set_active_lkg_monitoring_run_id(
        self, subject_id: str, monitoring_run_id: str | None
    ) -> None:
        """Set or clear the current active LKG monitoring run id.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Monitoring run to mark as active LKG, or `None`
                to clear the tag.
        """
        self._validate_subject_id(subject_id)
        experiment_id = self._get_or_create_experiment_id(subject_id)
        self._mlflow.set_monitoring_experiment_tag(
            experiment_id,
            _LKG_TAG,
            "" if monitoring_run_id is None else monitoring_run_id,
        )

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
        """Persist monitoring-run tags on an existing MLflow run.

        This method is intentionally limited to monitoring-run state tags. It
        does not log the final result artifact and does not terminate the run;
        those terminal write concerns are handled by
        `finalize_monitoring_run_result()`.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Monitoring run id to update.
            source_run_id: Immutable source training run identifier.
            lifecycle_status: Lifecycle status to persist.
            sequence_index: Stable timeline sequence index for the run.
            contract_check_result: Optional check result used to derive the
                persisted comparability status.
            references: Optional ordered references used during check.
        """
        self._validate_subject_id(subject_id)
        if self.resolve_timeline_monitoring_run_id(subject_id, monitoring_run_id) is None:
            raise GatewayConsistencyViolation.monitoring_run_subject_inconsistent(
                subject_id=subject_id,
                monitoring_run_id=monitoring_run_id,
            )

        persisted_source_run_id = self._mlflow.get_run_tags(monitoring_run_id).get(_SOURCE_RUN_TAG)
        if persisted_source_run_id != source_run_id:
            raise self._source_identity_consistency_error(
                monitoring_run_id=monitoring_run_id,
                source_run_id=source_run_id,
                persisted_source_run_id=persisted_source_run_id,
            )
        monitoring_tags = {
            _SEQUENCE_INDEX_TAG: str(sequence_index),
            _LIFECYCLE_STATUS_TAG: lifecycle_status.value,
        }
        if contract_check_result is not None:
            monitoring_tags[_COMPARABILITY_STATUS_TAG] = contract_check_result.status.value
        if references is not None:
            for reference in references:
                if reference.kind is DiffReferenceKind.BASELINE:
                    reference_tag_value = reference.source_run_id
                else:
                    assert reference.monitoring_run_id is not None
                    persisted_reference_source_run_id = self._mlflow.get_run_tags(
                        reference.monitoring_run_id
                    ).get(_SOURCE_RUN_TAG)
                    if persisted_reference_source_run_id != reference.source_run_id:
                        raise self._source_identity_consistency_error(
                            monitoring_run_id=reference.monitoring_run_id,
                            source_run_id=reference.source_run_id,
                            persisted_source_run_id=persisted_reference_source_run_id,
                        )
                    reference_tag_value = reference.monitoring_run_id
                monitoring_tags[f"{_REFERENCE_TAG_PREFIX}{reference.kind.value}"] = (
                    reference_tag_value
                )
        self._mlflow.set_monitoring_run_tags(monitoring_run_id, monitoring_tags)

    def get_monitoring_run(
        self, subject_id: str, monitoring_run_id: str
    ) -> MonitoringRunRecord | None:
        """Return one monitoring run reconstructed from MLflow run tags.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Candidate monitoring run id.

        Returns:
            A reconstructed monitoring run record when the run is indexed on the
            subject timeline and contains the minimum required tags; otherwise
            `None`.

        Raises:
            GatewayConsistencyViolation: If a nonempty comparability projection
                is not a supported status.

        Notes:
            Comparability status is a metadata projection. The authoritative
            Check result, including reasons, is hydrated separately from
            ``outputs/contract_check.json`` by orchestration.
        """
        self._validate_subject_id(subject_id)
        if self.resolve_timeline_monitoring_run_id(subject_id, monitoring_run_id) is None:
            return None

        run_tags = self._mlflow.get_run_tags(monitoring_run_id)
        if not run_tags:
            return None
        lifecycle_status_value = run_tags.get(_LIFECYCLE_STATUS_TAG)
        sequence_index_value = run_tags.get(_SEQUENCE_INDEX_TAG)
        source_run_id = run_tags.get(_SOURCE_RUN_TAG)
        if lifecycle_status_value is None or sequence_index_value is None or not source_run_id:
            return None

        try:
            sequence_index = int(sequence_index_value)
            lifecycle_status = LifecycleStatus(lifecycle_status_value)
        except ValueError:
            return None

        comparability_status_value = run_tags.get(_COMPARABILITY_STATUS_TAG)
        try:
            comparability_status = (
                None
                if not comparability_status_value
                else ComparabilityStatus(comparability_status_value)
            )
        except ValueError as exc:
            raise GatewayConsistencyViolation.monitoring_run_json_artifact_inconsistent(
                monitoring_run_id=monitoring_run_id,
                path=_CONTRACT_CHECK_ARTIFACT_PATH,
            ) from exc

        return MonitoringRunRecord(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            sequence_index=sequence_index,
            lifecycle_status=lifecycle_status,
            comparability_status=comparability_status,
            contract_check_result=None,
            references=self._parse_references(run_tags),
        )

    def list_timeline_monitoring_runs(
        self,
        subject_id: str,
        exclude_failed: bool = False,
    ) -> tuple[MonitoringRunRecord, ...]:
        """List visible monitoring runs ordered by sequence index.

        Args:
            subject_id: Monitored subject identifier.
            exclude_failed: Whether failed runs should be excluded.

        Returns:
            Monitoring runs that are both indexed on the timeline and
            reconstructable from persisted run tags.
        """
        experiment_tags = self._get_experiment_tags_if_present(subject_id)
        if experiment_tags is None:
            return ()

        records: list[MonitoringRunRecord] = []
        next_sequence_index = self._read_next_sequence_index(experiment_tags)
        for sequence_index in range(next_sequence_index):
            monitoring_run_id = experiment_tags.get(f"{_RUN_TAG_PREFIX}{sequence_index}")
            if not monitoring_run_id:
                # A sparse index should not fail traversal for the whole
                # timeline; MVP listing tolerates missing slots.
                continue
            monitoring_run = self.get_monitoring_run(subject_id, monitoring_run_id)
            if monitoring_run is None:
                # The index may reference a run that can no longer be resolved.
                # Tolerate that and continue listing the remaining timeline.
                continue
            if exclude_failed:
                if monitoring_run.lifecycle_status in _VISIBLE_NON_FAILED_STATUSES:
                    records.append(monitoring_run)
                continue
            if monitoring_run.lifecycle_status in _VISIBLE_NON_FAILED_STATUSES | {
                LifecycleStatus.FAILED
            }:
                records.append(monitoring_run)
        return tuple(records)

    def get_timeline_state(self, subject_id: str) -> TimelineState | None:
        """Return the recorded Timeline state for an allocated subject.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            Timeline state containing its stable identity and optional pinned
            Baseline Source Run identity. Returns `None` when no durable
            Monitoring Run allocation exists.
        """
        experiment_id = self._get_experiment_id(subject_id)
        if experiment_id is None:
            return None

        allocations_by_run_id = self._read_timeline_allocations(
            subject_id=subject_id,
            experiment_id=experiment_id,
        )
        if not allocations_by_run_id:
            return None

        claims = self._read_baseline_claims(
            subject_id=subject_id,
            experiment_id=experiment_id,
        )
        baseline_source_run_id = self._resolve_baseline_projection(
            subject_id=subject_id,
            experiment_id=experiment_id,
            experiment_tags=self._mlflow.get_monitoring_experiment_tags(experiment_id),
            claims=claims,
            repair_missing_projection=True,
        )

        return TimelineState(
            timeline_id=experiment_id,
            baseline_source_run_id=baseline_source_run_id,
        )

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        source_run_id: str,
    ) -> str | None:
        """Resolve an invocation-owned Source Training Run via direct MLflow lookup.

        Args:
            subject_id: Monitored subject identifier.
            source_experiment: Optional source experiment filter from the
                compiled recipe.
            source_run_id: Caller-supplied Source Training Run identifier.

        Returns:
            The resolved source training run id when MLflow can fetch it;
            otherwise `None`.

        Notes:
            The current MVP path always provides an explicit source run id and
            does not rely on broader selector semantics. This implementation
            therefore resolves by direct run lookup, then applies the optional
            `source_experiment` filter against the owning MLflow experiment
            name of that run.
        """
        self._validate_subject_id(subject_id)
        if not source_run_id:
            return None
        if self._mlflow.get_run(source_run_id) is None:
            return None
        experiment_name = self._mlflow.get_run_experiment_name(source_run_id)
        if experiment_name is not None and experiment_name.startswith(
            f"{self._config.namespace_prefix}/"
        ):
            return None
        if source_experiment is None:
            return source_run_id
        return source_run_id if experiment_name == source_experiment else None

    def get_missing_source_run_metrics(
        self,
        source_run_id: str,
        required_metrics: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required metric names absent from the source training run.

        Args:
            source_run_id: Source training run id.
            required_metrics: Required metric names in caller order.

        Returns:
            Missing metric names, deduplicated while preserving first-seen
            order from the request.
        """
        source_run_metrics = self._mlflow.get_run_metrics(source_run_id)
        return tuple(
            metric_name
            for metric_name in dict.fromkeys(required_metrics)
            if metric_name not in source_run_metrics
        )

    def get_missing_source_run_artifacts(
        self,
        source_run_id: str,
        required_artifacts: Sequence[str],
    ) -> tuple[str, ...]:
        """Return required artifact paths absent from the source training run.

        Args:
            source_run_id: Source training run id.
            required_artifacts: Required artifact paths in caller order.

        Returns:
            Missing artifact paths, deduplicated while preserving first-seen
            order from the request.
        """
        source_artifact_paths = set(self._mlflow.list_artifact_paths(source_run_id))
        return tuple(
            artifact_name
            for artifact_name in dict.fromkeys(required_artifacts)
            if artifact_name not in source_artifact_paths
        )

    def resolve_timeline_monitoring_run_id(
        self, subject_id: str, monitoring_run_id: str
    ) -> str | None:
        """Resolve one monitoring run id if it is indexed on the subject timeline.

        Args:
            subject_id: Monitored subject identifier.
            monitoring_run_id: Candidate monitoring run id.

        Returns:
            The monitoring run id when it is present in the experiment-tag
            timeline index; otherwise `None`.
        """
        experiment_tags = self._get_experiment_tags_if_present(subject_id)
        if experiment_tags is None:
            return None
        return (
            monitoring_run_id
            if monitoring_run_id in self._indexed_monitoring_run_ids(experiment_tags)
            else None
        )

    def get_source_run_contract_evidence(self, source_run_id: str) -> ContractEvidence | None:
        """Return MVP contract evidence for one source training run.

        Args:
            source_run_id: Source training run id.

        Returns:
            Contract evidence derived from MLflow metrics, params, tags, and
            the MVP evidence conventions, or `None` when the run does not
            exist.
        """
        run = self._mlflow.get_run(source_run_id)
        if run is None:
            return None

        params = self._mlflow.get_run_params(source_run_id)
        tags = self._mlflow.get_run_tags(source_run_id)
        # For MVP, features are read from a single comma-separated field
        # rather than a richer schema artifact or structured payload.
        feature_columns = params.get("feature_columns", tags.get("feature_columns", ""))
        features = tuple(
            feature.strip() for feature in feature_columns.split(",") if feature.strip()
        )
        schema = {
            key.removeprefix("schema."): value
            for key, value in tags.items()
            if key.startswith("schema.")
        }
        environment = {
            key: value
            for key, value in tags.items()
            if not key.startswith("mlflow.")
            and not key.startswith("schema.")
            and key not in {"data_scope", "feature_columns"}
        }
        return ContractEvidence(
            metrics=self._mlflow.get_run_metrics(source_run_id),
            environment=environment,
            features=features,
            schema=schema,
            data_scope=tags.get("data_scope"),
        )

    def mutate_training_run(self, source_run_id: str, updates: Mapping[str, str]) -> None:
        """Reject all attempted source training-run mutations.

        Args:
            source_run_id: Source training run id being targeted.
            updates: Requested write payload.

        Raises:
            TrainingRunMutationViolation: Always. Training runs are read-only.
        """
        raise TrainingRunMutationViolation(
            source_run_id=source_run_id,
            updates=updates,
        )

    def finalize_monitoring_run_result(
        self,
        *,
        monitoring_run_id: str,
        result: MonitorRunResult,
    ) -> None:
        """Ensure the final result artifact exists and the MLflow run is terminal.

        Args:
            monitoring_run_id: Monitoring run id being finalized.
            result: Canonical terminal orchestration result for the run.

        Raises:
            ValueError: If called with a non-terminal lifecycle status.
        """
        if monitoring_run_id != result.monitoring_run_id:
            raise ValueError("monitoring_run_id must match result.monitoring_run_id.")
        if result.lifecycle_status not in {LifecycleStatus.CHECKED, LifecycleStatus.FAILED}:
            raise ValueError(
                "finalize_monitoring_run_result supports only CHECKED and FAILED terminal results."
            )

        expected_mlflow_status = (
            "FINISHED" if result.lifecycle_status is LifecycleStatus.CHECKED else "FAILED"
        )
        artifact_paths = self._mlflow.list_artifact_paths(monitoring_run_id)
        if _RESULT_ARTIFACT_PATH not in artifact_paths:
            self._mlflow.log_monitoring_run_json_artifact(
                monitoring_run_id,
                result.to_dict(),
                _RESULT_ARTIFACT_PATH,
            )

        run = self._mlflow.get_run(monitoring_run_id)
        current_status = None if run is None else run.info.status
        if current_status != expected_mlflow_status:
            self._mlflow.terminate_monitoring_run(monitoring_run_id, expected_mlflow_status)

    def build_monitoring_namespace(self, subject_id: str) -> str:
        """Build the monitoring experiment name for one subject.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            Monitoring experiment name in `{namespace_prefix}/{subject_id}`
            form.
        """
        self._validate_namespace_prefix(self._config.namespace_prefix)
        self._validate_subject_id(subject_id)
        return f"{self._config.namespace_prefix}/{subject_id}"

    def read_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        path: str,
    ) -> dict[str, Any] | None:
        """Read one exact-path JSON artifact from a Monitoring Run.

        Args:
            monitoring_run_id: Monitoring Run that owns the artifact.
            path: Exact artifact path to read.

        Returns:
            Decoded JSON object, or ``None`` when the artifact is missing.

        Raises:
            GatewayNamespaceViolation: If the target is not an allocated
                Monitoring Run in this Gateway namespace.
        """
        self._validate_monitoring_run_artifact_target(monitoring_run_id)
        return self._mlflow.read_monitoring_run_json_artifact(monitoring_run_id, path)

    def write_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        data: dict[str, Any],
        path: str,
    ) -> None:
        """Write one dictionary payload as a JSON artifact on a monitoring run.

        Args:
            monitoring_run_id: Monitoring run id being targeted.
            data: Dictionary payload to write as a JSON artifact.
            path: Artifact path within the monitoring run.

        Raises:
            GatewayConsistencyViolation: If an existing JSON artifact at the same path
                is inconsistent with the requested data.
            GatewayNamespaceViolation: If the monitoring run does not belong to
                a monitoring namespace.

        Returns:
            None.
        """
        self._validate_monitoring_run_artifact_target(monitoring_run_id)

        existing_artifact = self._mlflow.read_monitoring_run_json_artifact(
            monitoring_run_id,
            path,
        )

        if existing_artifact is None:
            self._mlflow.log_monitoring_run_json_artifact(
                monitoring_run_id,
                data,
                path,
            )
            return

        if canonical_json(existing_artifact) == canonical_json(data):
            return

        raise GatewayConsistencyViolation.monitoring_run_json_artifact_inconsistent(
            monitoring_run_id=monitoring_run_id,
            path=path,
        )

    def _get_or_create_experiment_id(self, subject_id: str) -> str:
        """Return the monitoring experiment id for a subject, creating it if needed."""
        return self._mlflow.get_or_create_monitoring_experiment(
            self.build_monitoring_namespace(subject_id),
            artifact_location=self._artifact_location,
        )

    def _get_experiment_id(self, subject_id: str) -> str | None:
        """Return the monitoring experiment id for a subject, if present."""
        self._validate_subject_id(subject_id)
        return self._mlflow.get_monitoring_experiment_id_by_name(
            self.build_monitoring_namespace(subject_id)
        )

    def _get_experiment_tags_if_present(self, subject_id: str) -> dict[str, str] | None:
        """Return monitoring experiment tags if the experiment exists."""
        experiment_id = self._get_experiment_id(subject_id)
        if experiment_id is None:
            return None
        return self._mlflow.get_monitoring_experiment_tags(experiment_id)

    def _read_timeline_allocations(
        self,
        *,
        subject_id: str,
        experiment_id: str,
    ) -> dict[str, _MonitoringRunAllocation]:
        """Return durable Timeline allocations keyed by Monitoring Run identifier."""
        snapshots = self._mlflow.list_monitoring_runs_with_tag(
            experiment_id=experiment_id,
            tag_key=_SOURCE_RUN_TAG,
        )
        allocations = (
            self._parse_monitoring_run_allocation(subject_id, snapshot) for snapshot in snapshots
        )
        return {allocation.monitoring_run_id: allocation for allocation in allocations}

    def _read_baseline_claims(
        self,
        *,
        subject_id: str,
        experiment_id: str,
    ) -> tuple[TimelineClaim, ...]:
        """Return deterministic durable baseline claims for allocated Monitoring Runs."""
        snapshots = self._mlflow.list_monitoring_runs_with_tag(
            experiment_id=experiment_id,
            tag_key=_SOURCE_RUN_TAG,
        )
        claims: list[TimelineClaim] = []
        for snapshot in snapshots:
            allocation = self._parse_monitoring_run_allocation(subject_id, snapshot)
            for tag_key, tag_value in snapshot.tags.items():
                if not tag_key.startswith(_BASELINE_CLAIM_TAG_PREFIX) or not tag_value:
                    continue

                claimed_baseline_source_run_id = tag_value
                if tag_key != _baseline_claim_tag_key(claimed_baseline_source_run_id):
                    raise TimelineConsistencyViolation.claim_address_mismatch(
                        monitoring_run_id=allocation.monitoring_run_id,
                        source_run_id=allocation.key.source_run_id,
                        tag_key=tag_key,
                        claimed_baseline_source_run_id=claimed_baseline_source_run_id,
                    )

                claims.append(
                    TimelineClaim(
                        monitoring_run_id=allocation.monitoring_run_id,
                        source_run_id=allocation.key.source_run_id,
                        claimed_baseline_source_run_id=claimed_baseline_source_run_id,
                    )
                )

        return tuple(
            sorted(
                claims,
                key=lambda claim: (
                    claim.monitoring_run_id,
                    claim.claimed_baseline_source_run_id,
                ),
            )
        )

    @staticmethod
    def _get_unique_claimed_baseline(
        subject_id: str, claims: tuple[TimelineClaim, ...]
    ) -> str | None:
        """Return the unique claimed baseline or fail closed on disagreement."""
        claimed_baselines = {claim.claimed_baseline_source_run_id for claim in claims}
        if len(claimed_baselines) > 1:
            raise TimelineConsistencyViolation.claims_conflict(claims=claims, subject_id=subject_id)
        return next(iter(claimed_baselines), None)

    def _resolve_baseline_projection(
        self,
        *,
        subject_id: str,
        experiment_id: str,
        experiment_tags: Mapping[str, str],
        claims: tuple[TimelineClaim, ...],
        repair_missing_projection: bool,
    ) -> str | None:
        """Validate claims and optionally repair their experiment projection."""
        projected_baseline = experiment_tags.get(_BASELINE_PROJECTION_TAG) or None
        claimed_baseline = self._get_unique_claimed_baseline(subject_id, claims)
        if claimed_baseline is None:
            return projected_baseline
        if projected_baseline is not None and projected_baseline != claimed_baseline:
            raise TimelineConsistencyViolation.projection_conflict(
                claims=claims,
                projected_baseline_source_run_id=projected_baseline,
                subject_id=subject_id,
            )
        if projected_baseline is None and repair_missing_projection:
            self._set_experiment_tags(
                experiment_id,
                {_BASELINE_PROJECTION_TAG: claimed_baseline},
            )
        return claimed_baseline

    def _reconcile_monitoring_run_allocations(
        self,
        *,
        subject_id: str,
        experiment_id: str,
        experiment_tags: Mapping[str, str],
    ) -> tuple[dict[IdempotencyKey, _MonitoringRunAllocation], int]:
        """Validate durable allocations and repair their experiment-tag projection."""
        snapshots = self._mlflow.list_monitoring_runs_with_tag(
            experiment_id,
            _SOURCE_RUN_TAG,
        )
        allocations = tuple(
            self._parse_monitoring_run_allocation(subject_id, snapshot) for snapshot in snapshots
        )
        allocations_by_key: dict[IdempotencyKey, _MonitoringRunAllocation] = {}
        allocations_by_sequence: dict[int, _MonitoringRunAllocation] = {}
        allocations_by_monitoring_run_id: dict[str, _MonitoringRunAllocation] = {}
        for allocation in allocations:
            existing_key_allocation = allocations_by_key.get(allocation.key)
            if existing_key_allocation is not None:
                raise AllocationConsistencyViolation.duplicate_identity(
                    first_monitoring_run_id=existing_key_allocation.monitoring_run_id,
                    second_monitoring_run_id=allocation.monitoring_run_id,
                )
            existing_sequence_allocation = allocations_by_sequence.get(allocation.sequence_index)
            if existing_sequence_allocation is not None:
                raise AllocationConsistencyViolation.duplicate_sequence(
                    sequence_index=allocation.sequence_index,
                    first_monitoring_run_id=existing_sequence_allocation.monitoring_run_id,
                    second_monitoring_run_id=allocation.monitoring_run_id,
                )
            allocations_by_key[allocation.key] = allocation
            allocations_by_sequence[allocation.sequence_index] = allocation
            allocations_by_monitoring_run_id[allocation.monitoring_run_id] = allocation

        ordered_allocations = tuple(
            sorted(allocations, key=lambda allocation: allocation.sequence_index)
        )
        for expected_sequence, allocation in enumerate(ordered_allocations):
            if allocation.sequence_index != expected_sequence:
                raise AllocationConsistencyViolation.sequence_gap(
                    expected_sequence_index=expected_sequence,
                    actual_sequence_index=allocation.sequence_index,
                )

        next_sequence_index = len(ordered_allocations)
        repairs = self._build_allocation_index_repairs(
            experiment_tags=experiment_tags,
            ordered_allocations=ordered_allocations,
            allocations_by_sequence=allocations_by_sequence,
            allocations_by_monitoring_run_id=allocations_by_monitoring_run_id,
            next_sequence_index=next_sequence_index,
        )
        if repairs:
            self._set_experiment_tags(experiment_id, repairs)
        return allocations_by_key, next_sequence_index

    def _parse_monitoring_run_allocation(
        self,
        subject_id: str,
        snapshot: MonitoringRunTagSnapshot,
    ) -> _MonitoringRunAllocation:
        """Parse one durable allocation record from a monitoring-run tag snapshot."""
        source_run_id = snapshot.tags.get(_SOURCE_RUN_TAG)
        recipe_id = snapshot.tags.get(_RECIPE_ID_TAG)
        recipe_version = snapshot.tags.get(_RECIPE_VERSION_TAG)
        raw_sequence_index = snapshot.tags.get(_SEQUENCE_INDEX_TAG)
        missing_tags = tuple(
            field
            for field, value in (
                (_SOURCE_RUN_TAG, source_run_id),
                (_RECIPE_ID_TAG, recipe_id),
                (_RECIPE_VERSION_TAG, recipe_version),
                (_SEQUENCE_INDEX_TAG, raw_sequence_index),
            )
            if not value
        )
        if not snapshot.run_id or missing_tags:
            raise AllocationConsistencyViolation.missing_durable_tags(
                monitoring_run_id=snapshot.run_id,
                missing_tags=missing_tags,
            )

        assert source_run_id is not None
        assert recipe_id is not None
        assert recipe_version is not None
        assert raw_sequence_index is not None

        try:
            sequence_index = int(raw_sequence_index)
        except (TypeError, ValueError) as exc:
            raise AllocationConsistencyViolation.non_integer_sequence(
                monitoring_run_id=snapshot.run_id,
                raw_sequence_index=raw_sequence_index,
            ) from exc

        if sequence_index < 0:
            raise AllocationConsistencyViolation.negative_sequence(
                monitoring_run_id=snapshot.run_id,
                sequence_index=sequence_index,
            )

        return _MonitoringRunAllocation(
            key=IdempotencyKey(
                subject_id=subject_id,
                source_run_id=source_run_id,
                recipe_id=recipe_id,
                recipe_version=recipe_version,
            ),
            monitoring_run_id=snapshot.run_id,
            sequence_index=sequence_index,
        )

    def _build_allocation_index_repairs(
        self,
        *,
        experiment_tags: Mapping[str, str],
        ordered_allocations: tuple[_MonitoringRunAllocation, ...],
        allocations_by_sequence: Mapping[int, _MonitoringRunAllocation],
        allocations_by_monitoring_run_id: Mapping[str, _MonitoringRunAllocation],
        next_sequence_index: int,
    ) -> dict[str, str]:
        """Return validated experiment-tag repairs in deterministic write order."""
        self._validate_timeline_index_tags(experiment_tags, allocations_by_sequence)
        self._validate_allocation_pointer_tags(experiment_tags, allocations_by_monitoring_run_id)

        persisted_next_sequence = self._read_next_sequence_index(experiment_tags)
        if persisted_next_sequence > next_sequence_index:
            raise AllocationConsistencyViolation.next_sequence_ahead(
                persisted_next_sequence_index=persisted_next_sequence,
                durable_next_sequence_index=next_sequence_index,
            )

        repairs: dict[str, str] = {}
        for allocation in ordered_allocations:
            timeline_tag = f"{_RUN_TAG_PREFIX}{allocation.sequence_index}"
            if timeline_tag not in experiment_tags:
                repairs[timeline_tag] = allocation.monitoring_run_id

        if ordered_allocations:
            latest_run_id = ordered_allocations[-1].monitoring_run_id
            if experiment_tags.get(_LATEST_TAG) != latest_run_id:
                repairs[_LATEST_TAG] = latest_run_id
            if persisted_next_sequence < next_sequence_index:
                repairs[_NEXT_SEQUENCE_TAG] = str(next_sequence_index)

        highest_allocation_by_source: dict[str, _MonitoringRunAllocation] = {}
        for allocation in ordered_allocations:
            highest_allocation_by_source[allocation.key.source_run_id] = allocation
        for source_run_id in sorted(highest_allocation_by_source):
            allocation = highest_allocation_by_source[source_run_id]
            idempotency_tag = self._idempotency_tag(source_run_id)
            if experiment_tags.get(idempotency_tag) != allocation.monitoring_run_id:
                repairs[idempotency_tag] = allocation.monitoring_run_id
        return repairs

    def _validate_timeline_index_tags(
        self,
        experiment_tags: Mapping[str, str],
        allocations_by_sequence: Mapping[int, _MonitoringRunAllocation],
    ) -> None:
        """Validate persisted timeline slots against durable allocations."""
        for tag_key, monitoring_run_id in sorted(experiment_tags.items()):
            if not tag_key.startswith(_RUN_TAG_PREFIX):
                continue
            raw_sequence_index = tag_key.removeprefix(_RUN_TAG_PREFIX)
            try:
                sequence_index = int(raw_sequence_index)
            except ValueError as exc:
                raise GatewayNamespaceViolation(
                    message=(
                        "Monitoring timeline index tag must end with a non-negative integer "
                        f"sequence; got key={tag_key!r} value={monitoring_run_id!r}."
                    )
                ) from exc
            if sequence_index < 0:
                raise GatewayNamespaceViolation(
                    message=(
                        "Monitoring timeline index tag must end with a non-negative integer "
                        f"sequence; got key={tag_key!r} value={monitoring_run_id!r}."
                    )
                )

            allocation = allocations_by_sequence.get(sequence_index)
            if allocation is None or allocation.monitoring_run_id != monitoring_run_id:
                raise AllocationConsistencyViolation.timeline_conflict(
                    sequence_index=sequence_index,
                    indexed_monitoring_run_id=monitoring_run_id,
                    durable_monitoring_run_id=None
                    if allocation is None
                    else allocation.monitoring_run_id,
                )

    def _validate_allocation_pointer_tags(
        self,
        experiment_tags: Mapping[str, str],
        allocations_by_run_id: Mapping[str, _MonitoringRunAllocation],
    ) -> None:
        """Validate latest and source pointers before any repair writes occur."""
        latest_run_id = experiment_tags.get(_LATEST_TAG)
        if latest_run_id and latest_run_id not in allocations_by_run_id:
            raise AllocationConsistencyViolation.unknown_pointer(
                monitoring_run_id=latest_run_id,
            )

        source_prefix = "training."
        for tag_key, monitoring_run_id in sorted(experiment_tags.items()):
            if not tag_key.startswith(source_prefix) or not tag_key.endswith(
                _IDEMPOTENCY_TAG_SUFFIX
            ):
                continue
            source_run_id = tag_key.removeprefix(source_prefix).removesuffix(
                _IDEMPOTENCY_TAG_SUFFIX
            )
            if not source_run_id:
                raise GatewayNamespaceViolation(
                    message=f"Monitoring idempotency tag has an empty source run id: {tag_key!r}."
                )
            allocation = allocations_by_run_id.get(monitoring_run_id)
            if allocation is None:
                raise AllocationConsistencyViolation.unknown_tag(
                    tag=tag_key,
                    monitoring_run_id=monitoring_run_id,
                )
            if allocation.key.source_run_id != source_run_id:
                raise AllocationConsistencyViolation.source_binding_conflict(
                    tag=tag_key,
                    monitoring_run_id=monitoring_run_id,
                    source_run_id=source_run_id,
                    persisted_source_run_id=allocation.key.source_run_id,
                )

    def _set_experiment_tags(self, experiment_id: str, tags: Mapping[str, str]) -> None:
        """Write a batch of experiment tags via the thin MLflow adapter."""
        for key, value in tags.items():
            self._mlflow.set_monitoring_experiment_tag(experiment_id, key, value)

    def _read_next_sequence_index(self, experiment_tags: Mapping[str, str]) -> int:
        """Return the next sequence index from experiment tags, defaulting to zero.

        The experiment tag stores an integer as a string because MLflow tags are
        string-valued.
        """
        raw_next_sequence_index = experiment_tags.get(_NEXT_SEQUENCE_TAG)
        if raw_next_sequence_index is None or raw_next_sequence_index == "":
            return 0
        try:
            next_sequence_index = int(raw_next_sequence_index)
        except ValueError as exc:
            raise GatewayNamespaceViolation(
                message=(
                    "monitoring.next_sequence_index must be a non-negative integer string; "
                    f"got {raw_next_sequence_index!r}."
                )
            ) from exc
        if next_sequence_index < 0:
            raise GatewayNamespaceViolation(
                message=(
                    "monitoring.next_sequence_index must be a non-negative integer string; "
                    f"got {raw_next_sequence_index!r}."
                )
            )
        return next_sequence_index

    def _indexed_monitoring_run_ids(self, experiment_tags: Mapping[str, str]) -> set[str]:
        """Return the set of monitoring run ids indexed on an experiment."""
        indexed_run_ids: set[str] = set()
        for key, value in experiment_tags.items():
            if not key.startswith(_RUN_TAG_PREFIX) or not value:
                continue
            try:
                int(key.removeprefix(_RUN_TAG_PREFIX))
            except ValueError:
                continue
            indexed_run_ids.add(value)
        return indexed_run_ids

    def _idempotency_tag(self, source_run_id: str) -> str:
        """Return the experiment-tag key for one source-run idempotency binding."""
        return f"training.{source_run_id}{_IDEMPOTENCY_TAG_SUFFIX}"

    def _parse_references(self, run_tags: Mapping[str, str]) -> tuple[MonitoringRunReference, ...]:
        """Return persisted monitoring references in canonical order.

        Reference order matters because result replay should preserve the same
        baseline/previous/LKG/custom ordering that orchestration emitted.
        """
        references: list[MonitoringRunReference] = []
        for kind in _REFERENCE_KINDS:
            reference_tag_value = run_tags.get(f"{_REFERENCE_TAG_PREFIX}{kind}")
            if not reference_tag_value:
                continue
            reference_kind = DiffReferenceKind(kind)
            if reference_kind is DiffReferenceKind.BASELINE:
                references.append(
                    MonitoringRunReference(
                        kind=reference_kind,
                        monitoring_run_id=None,
                        source_run_id=reference_tag_value,
                    )
                )
                continue
            reference_monitoring_run_id = reference_tag_value
            reference_source_run_id = self._mlflow.get_run_tags(reference_monitoring_run_id).get(
                _SOURCE_RUN_TAG
            )
            if not reference_source_run_id:
                raise self._source_identity_consistency_error(
                    monitoring_run_id=reference_monitoring_run_id,
                    source_run_id=None,
                    persisted_source_run_id=reference_source_run_id,
                )
            references.append(
                MonitoringRunReference(
                    kind=reference_kind,
                    monitoring_run_id=reference_monitoring_run_id,
                    source_run_id=reference_source_run_id,
                )
            )
        return tuple(references)

    def _source_identity_consistency_error(
        self,
        *,
        monitoring_run_id: str,
        source_run_id: str | None,
        persisted_source_run_id: str | None,
    ) -> GatewayConsistencyViolation:
        """Build a structured error for one contradictory monitoring/source pair."""
        return GatewayConsistencyViolation.monitoring_run_upsert_source_binding_conflict(
            monitoring_run_id=monitoring_run_id,
            source_run_id=source_run_id,
            persisted_source_run_id=persisted_source_run_id,
        )

    def _validate_namespace_prefix(self, prefix: str) -> None:
        """Validate namespace prefix can safely compose a monitoring namespace."""
        if not prefix or "/" in prefix:
            raise GatewayNamespaceViolation(
                message=(f"namespace_prefix must be non-empty and must not contain '/': {prefix!r}")
            )

    def _validate_subject_id(self, subject_id: str) -> None:
        """Validate subject id can safely compose a monitoring namespace."""
        if not subject_id or "/" in subject_id:
            raise GatewayNamespaceViolation(
                message=(f"subject_id must be non-empty and must not contain '/': {subject_id!r}")
            )

    def _validate_monitoring_run_artifact_target(self, monitoring_run_id: str) -> None:
        """Validate that an artifact target is a monitoring-owned allocation."""
        self._validate_namespace_prefix(self._config.namespace_prefix)
        experiment_name = self._mlflow.get_run_experiment_name(monitoring_run_id)
        monitoring_namespace_prefix = f"{self._config.namespace_prefix}/"
        if experiment_name is None or not experiment_name.startswith(monitoring_namespace_prefix):
            raise GatewayNamespaceViolation(
                message=(f"Run {monitoring_run_id!r} does not belong to a monitoring namespace.")
            )

        subject_id = experiment_name.removeprefix(monitoring_namespace_prefix)
        self._validate_subject_id(subject_id)
        run_tags = self._mlflow.get_run_tags(monitoring_run_id)
        missing_tags = tuple(
            tag.value for tag in RequiredMonitoringRunTags if not run_tags.get(tag.value)
        )
        if missing_tags:
            raise AllocationConsistencyViolation.missing_durable_tags(
                monitoring_run_id=monitoring_run_id,
                missing_tags=missing_tags,
            )
