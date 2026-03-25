"""Real-MLflow monitoring gateway for the MVP create/prepare/check slice.

This module is the runtime implementation of the `MonitoringGateway` protocol
for the MVP's real-MLflow path. It intentionally does not use `search_runs()`;
all timeline discovery is driven by:

1. One monitoring experiment per subject, named `{namespace_prefix}/{subject_id}`.
2. Experiment tags that act as the timeline index.
3. Direct `get_run()` calls for both training and monitoring runs.

Design constraints that matter here:

- Training runs are strictly read-only. The gateway may inspect metrics, params,
  tags, and artifact paths on training runs, but it must never mutate them.
- Monitoring-run allocation is gateway-owned because MLflow assigns run ids at
  `create_run()` time.
- The experiment-tag index is the source of truth for ordered monitoring-run
  traversal and idempotency.
- Final run output is stored as `outputs/result.json` and the MLflow run is
  explicitly terminated by the gateway once orchestration reaches a terminal
  success or owned-failure outcome.

The implementation is intentionally conservative and MVP-scoped. It preserves
the current M1 create -> prepare -> check semantics without attempting
concurrency hardening, broader query support, or more expressive persistence
than the ticket requires.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from mlflow_monitor.contract_checker import ContractEvidence
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckResult,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.errors import GatewayNamespaceViolation, TrainingRunMutationViolation
from mlflow_monitor.gateway import (
    CreateOrReuseMonitoringRunResult,
    GatewayConfig,
    IdempotencyKey,
    MonitoringRunRecord,
    TimelineInitializationResult,
    TimelineState,
)
from mlflow_monitor.mlflow_client import MonitorMLflowClient
from mlflow_monitor.recipe import SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN
from mlflow_monitor.result_contract import MonitorRunResult

_BASELINE_TAG = "training.baseline_run_id"
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
_RESULT_ARTIFACT_PATH = "outputs/result.json"
_VISIBLE_NON_FAILED_STATUSES = frozenset({LifecycleStatus.CHECKED, LifecycleStatus.CLOSED})
_REFERENCE_KINDS = ("baseline", "previous", "lkg", "custom")


class MLflowMonitoringGateway:
    """Monitoring gateway backed by real MLflow experiments and runs.

    The gateway translates protocol-level operations such as "initialize the
    subject timeline" or "create or reuse the monitoring run for this training
    run" into:

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
        """Create or reuse a monitoring run using experiment-tag indexes.

        The idempotency source of truth is the experiment tag
        `training.{source_run_id}.monitoring_run_id`. If it exists, this method
        returns the already-allocated monitoring run and attempts to reconstruct
        its persisted state. If it does not exist, the gateway allocates the
        next sequence index, creates the MLflow monitoring run, and updates the
        experiment-tag index.

        Args:
            key: Canonical monitoring intent identity.

        Returns:
            Allocation or replay information needed by orchestration.
        """
        self._validate_subject_id(key.subject_id)
        experiment_id = self._get_or_create_experiment_id(key.subject_id)
        experiment_tags = self._mlflow.get_monitoring_experiment_tags(experiment_id)
        idempotency_tag = self._idempotency_tag(key.source_run_id)
        existing_monitoring_run_id = experiment_tags.get(idempotency_tag)
        if existing_monitoring_run_id:
            # The index may outlive a fully reconstructed MonitoringRunRecord,
            # for example if allocation succeeded but later persistence did not.
            existing_monitoring_run = self.get_monitoring_run(
                key.subject_id, existing_monitoring_run_id
            )
            sequence_index = (
                existing_monitoring_run.sequence_index
                if existing_monitoring_run is not None
                else self._resolve_sequence_index(experiment_tags, existing_monitoring_run_id)
            )
            return CreateOrReuseMonitoringRunResult(
                monitoring_run_id=existing_monitoring_run_id,
                sequence_index=sequence_index,
                existing_monitoring_run=existing_monitoring_run,
                allocated=False,
            )

        sequence_index = self._read_next_sequence_index(experiment_tags)
        # MLflow assigns the monitoring run id, so the gateway has to create the
        # run before it can finish updating the experiment-level timeline index.
        monitoring_run_id = self._mlflow.create_monitoring_run(
            experiment_id,
            tags={
                _SOURCE_RUN_TAG: key.source_run_id,
                _SEQUENCE_INDEX_TAG: str(sequence_index),
                _LIFECYCLE_STATUS_TAG: LifecycleStatus.CREATED.value,
                _RECIPE_ID_TAG: key.recipe_id,
                _RECIPE_VERSION_TAG: key.recipe_version,
            },
        )
        self._set_experiment_tags(
            experiment_id,
            {
                idempotency_tag: monitoring_run_id,
                f"{_RUN_TAG_PREFIX}{sequence_index}": monitoring_run_id,
                _LATEST_TAG: monitoring_run_id,
                _NEXT_SEQUENCE_TAG: str(sequence_index + 1),
            },
        )
        return CreateOrReuseMonitoringRunResult(
            monitoring_run_id=monitoring_run_id,
            sequence_index=sequence_index,
            existing_monitoring_run=None,
            allocated=True,
        )

    def initialize_timeline(
        self, subject_id: str, baseline_source_run_id: str
    ) -> TimelineInitializationResult:
        """Initialize the experiment-backed timeline baseline once.

        The monitoring experiment doubles as the subject timeline. Timeline
        initialization therefore means "ensure the experiment exists and pin the
        baseline tag if it has not been pinned already."

        Args:
            subject_id: Monitored subject identifier.
            baseline_source_run_id: Source training run id to pin as baseline.

        Returns:
            Timeline initialization status with the stable timeline id.
        """
        self._validate_subject_id(subject_id)
        if not baseline_source_run_id:
            raise GatewayNamespaceViolation(message="baseline_source_run_id must be non-empty.")

        experiment_id = self._get_or_create_experiment_id(subject_id)
        experiment_tags = self._mlflow.get_monitoring_experiment_tags(experiment_id)
        existing_baseline_source_run_id = experiment_tags.get(_BASELINE_TAG)
        if existing_baseline_source_run_id:
            return TimelineInitializationResult(
                timeline_id=experiment_id,
                created=False,
            )

        self._set_experiment_tags(
            experiment_id,
            {
                _BASELINE_TAG: baseline_source_run_id,
                _NEXT_SEQUENCE_TAG: str(self._read_next_sequence_index(experiment_tags)),
            },
        )
        return TimelineInitializationResult(
            timeline_id=experiment_id,
            created=True,
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
            lifecycle_status: Lifecycle status to persist.
            sequence_index: Stable timeline sequence index for the run.
            contract_check_result: Optional check result used to derive the
                persisted comparability status.
            references: Optional ordered references used during check.
        """
        self._validate_subject_id(subject_id)
        monitoring_tags = {
            _SEQUENCE_INDEX_TAG: str(sequence_index),
            _LIFECYCLE_STATUS_TAG: lifecycle_status.value,
        }
        if contract_check_result is not None:
            monitoring_tags[_COMPARABILITY_STATUS_TAG] = contract_check_result.status.value
        if references is not None:
            for reference in references:
                monitoring_tags[f"{_REFERENCE_TAG_PREFIX}{reference.kind}"] = (
                    reference.reference_run_id
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

        Notes:
            MVP-04 persists comparability status but not the full structured
            `ContractCheckReason` payloads. When reconstructing a checked run
            from MLflow, this gateway therefore rebuilds only the status-facing
            portion of `ContractCheckResult` and leaves `reasons=()`. That is
            sufficient for the current replay semantics, which only consult the
            terminal lifecycle and overall comparability outcome.
        """
        self._validate_subject_id(subject_id)
        if self.resolve_timeline_monitoring_run_id(subject_id, monitoring_run_id) is None:
            return None

        run_tags = self._mlflow.get_run_tags(monitoring_run_id)
        if not run_tags:
            return None
        lifecycle_status_value = run_tags.get(_LIFECYCLE_STATUS_TAG)
        sequence_index_value = run_tags.get(_SEQUENCE_INDEX_TAG)
        if lifecycle_status_value is None or sequence_index_value is None:
            return None

        comparability_status_value = run_tags.get(_COMPARABILITY_STATUS_TAG)
        comparability_status = (
            None
            if not comparability_status_value
            else ComparabilityStatus(comparability_status_value)
        )
        # MVP persistence stores the comparability outcome, not the original
        # reason list. Replay paths only need the terminal status.
        contract_check_result = (
            None
            if comparability_status is None
            else ContractCheckResult(status=comparability_status, reasons=())
        )

        return MonitoringRunRecord(
            monitoring_run_id=monitoring_run_id,
            sequence_index=int(sequence_index_value),
            lifecycle_status=LifecycleStatus(lifecycle_status_value),
            comparability_status=comparability_status,
            contract_check_result=contract_check_result,
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
        """Return timeline state once the baseline has been pinned.

        Args:
            subject_id: Monitored subject identifier.

        Returns:
            Timeline state when the monitoring experiment exists and has a
            pinned baseline; otherwise `None`.
        """
        experiment_id = self._get_experiment_id(subject_id)
        if experiment_id is None:
            return None

        experiment_tags = self._mlflow.get_monitoring_experiment_tags(experiment_id)
        baseline_source_run_id = experiment_tags.get(_BASELINE_TAG)
        if not baseline_source_run_id:
            return None
        return TimelineState(
            timeline_id=experiment_id,
            baseline_source_run_id=baseline_source_run_id,
        )

    def resolve_source_run_id(
        self,
        subject_id: str,
        source_experiment: str | None,
        run_selector: str,
        runtime_source_run_id: str | None = None,
    ) -> str | None:
        """Resolve a concrete source training run id via direct MLflow lookup.

        Args:
            subject_id: Monitored subject identifier.
            source_experiment: Optional source experiment filter from the
                compiled recipe.
            run_selector: Either a concrete run id or the reserved runtime
                selector token.
            runtime_source_run_id: Caller-supplied training run id used only
                when the selector is the reserved runtime token.

        Returns:
            The resolved source training run id when MLflow can fetch it;
            otherwise `None`.

        Notes:
            The current MVP path always provides an explicit source run id and
            does not rely on broader selector semantics. This implementation
            therefore resolves by direct run lookup only. `source_experiment`
            is accepted to preserve protocol shape, but is not currently used
            as an additional filter in the real-MLflow path.
        """
        self._validate_subject_id(subject_id)
        _ = source_experiment
        candidate_source_run_id = (
            runtime_source_run_id
            if run_selector == SYSTEM_DEFAULT_RUN_SELECTOR_TOKEN
            else run_selector
        )
        if not candidate_source_run_id:
            return None
        return (
            candidate_source_run_id
            if self._mlflow.get_run(candidate_source_run_id) is not None
            else None
        )

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
            message=(
                "Training runs are read-only in MLflow-Monitor; "
                f"attempted mutation for source_run_id={source_run_id} with updates={dict(updates)}"
            )
        )

    def finalize_monitoring_run_result(
        self,
        *,
        monitoring_run_id: str,
        result: MonitorRunResult,
    ) -> None:
        """Write the final result artifact and terminate the MLflow run.

        Args:
            monitoring_run_id: Monitoring run id being finalized.
            result: Canonical terminal orchestration result for the run.

        Raises:
            ValueError: If called with a non-terminal lifecycle status.
        """
        self._mlflow.log_monitoring_run_json_artifact(
            monitoring_run_id,
            result.to_dict(),
            _RESULT_ARTIFACT_PATH,
        )
        if result.lifecycle_status is LifecycleStatus.CHECKED:
            self._mlflow.terminate_monitoring_run(monitoring_run_id, "FINISHED")
            return
        if result.lifecycle_status is LifecycleStatus.FAILED:
            self._mlflow.terminate_monitoring_run(monitoring_run_id, "FAILED")
            return
        raise ValueError(
            "finalize_monitoring_run_result supports only CHECKED and FAILED terminal results."
        )

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
        return 0 if raw_next_sequence_index in {None, ""} else int(raw_next_sequence_index)

    def _resolve_sequence_index(
        self,
        experiment_tags: Mapping[str, str],
        monitoring_run_id: str,
    ) -> int:
        """Resolve a monitoring run's sequence index from experiment tags.

        This is only used on idempotent replay paths where the run id is known
        from the idempotency tag but the persisted run record could not be
        reconstructed.
        """
        for key, value in experiment_tags.items():
            if not key.startswith(_RUN_TAG_PREFIX) or value != monitoring_run_id:
                continue
            return int(key.removeprefix(_RUN_TAG_PREFIX))
        raise GatewayNamespaceViolation(
            message=(
                f"Monitoring run {monitoring_run_id!r} is not indexed on the experiment timeline."
            )
        )

    def _indexed_monitoring_run_ids(self, experiment_tags: Mapping[str, str]) -> set[str]:
        """Return the set of monitoring run ids indexed on an experiment."""
        return {
            value
            for key, value in experiment_tags.items()
            if key.startswith(_RUN_TAG_PREFIX) and value
        }

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
            reference_run_id = run_tags.get(f"{_REFERENCE_TAG_PREFIX}{kind}")
            if not reference_run_id:
                continue
            references.append(
                MonitoringRunReference(kind=kind, reference_run_id=reference_run_id)
            )
        return tuple(references)

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
