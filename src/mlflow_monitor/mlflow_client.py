"""Thin MLflow client adapter for the monitoring runtime.

This module is the single runtime bridge between MLflow-Monitor and the
external MLflow tracking system for the MVP slice. All direct `MlflowClient`
usage is intentionally centralized here so later gateway code can depend on a
small, documented boundary instead of spreading MLflow-specific behavior across
the codebase.

What belongs here:
-------------------

- Thin wrappers around the concrete MLflow client operations required by MVP-03
- Deterministic normalization of a few MLflow-specific quirks
- Read helpers that expose plain Python values where that reduces MLflow
  leakage into later gateway code

What does not belong here:
--------------------------

- Workflow or orchestration logic
- Monitoring domain policy
- Broader query functionality such as run search
- Public SDK configuration policy beyond accepting an optional tracking URI
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mlflow import MlflowClient
from mlflow.entities import Experiment, Run
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST

_RESOURCE_ALREADY_EXISTS = "RESOURCE_ALREADY_EXISTS"
_RESOURCE_DOES_NOT_EXIST = "RESOURCE_DOES_NOT_EXIST"


def _normalize_error_code(error_code: object) -> str | None:
    """Return the canonical MLflow error-code name for known inputs."""
    if isinstance(error_code, str):
        return error_code
    if error_code == RESOURCE_ALREADY_EXISTS:
        return _RESOURCE_ALREADY_EXISTS
    if error_code == RESOURCE_DOES_NOT_EXIST:
        return _RESOURCE_DOES_NOT_EXIST
    return None


class MonitorMLflowClient:
    """Adapter around `MlflowClient` for the monitoring runtime.

    The adapter owns only MLflow-facing mechanics. Callers should rely on it
    for direct experiment/run operations, missing-run normalization, and
    deterministic artifact traversal. It is intentionally narrow so the future
    gateway can compose these operations without reimplementing MLflow quirks.

    Args:
        tracking_uri: Optional MLflow tracking URI. When omitted, MLflow will
            use its normal client-side resolution rules.
    """

    def __init__(self, tracking_uri: str | None = None) -> None:
        """Create one adapter instance bound to an optional tracking URI.

        Args:
            tracking_uri: Optional MLflow tracking URI forwarded directly to the
                underlying `MlflowClient`.
        """
        self._client = MlflowClient(tracking_uri=tracking_uri)

    def get_or_create_monitoring_experiment(
        self,
        name: str,
        artifact_location: str | None = None,
    ) -> str:
        """Return a monitoring experiment id, creating it if needed.

        This wrapper normalizes the common create race where another actor
        creates the experiment between the local existence check and the
        `create_experiment()` call. It also restores deleted monitoring
        experiments instead of treating them as disposable state: in this
        system the experiment is the authoritative bookkeeping timeline for a
        subject, so soft deletion is recovered back to active state.
        Artifact location only applies when the experiment is created for the
        first time; if the experiment already exists, MLflow preserves the
        existing artifact root and the provided `artifact_location` is ignored.

        Args:
            name: Monitoring experiment name to fetch or create.
            artifact_location: Optional MLflow artifact root to apply on first
                creation. This is mainly useful for callers that need local
                artifact placement to stay inside a dedicated temp directory,
                such as integration tests.

        Returns:
            The stable MLflow experiment id as a string.

        Raises:
            MlflowException: If MLflow reports an error other than the duplicate
                experiment case, or if the duplicate-race fallback still cannot
                resolve the experiment by name.
        """
        experiment = self._get_experiment_by_name(name)
        if experiment is not None:
            self._restore_monitoring_experiment_if_deleted(name, experiment)
            return experiment.experiment_id

        try:
            return self._client.create_experiment(
                name,
                artifact_location=artifact_location,
            )
        except MlflowException as exc:
            # Duplicate-create races are normal for get-or-create semantics.
            if _normalize_error_code(exc.error_code) != _RESOURCE_ALREADY_EXISTS:
                raise

        experiment = self._get_experiment_by_name(name)
        if experiment is None:
            raise MlflowException(
                f"Experiment {name!r} already exists but could not be resolved by name.",
                error_code=RESOURCE_ALREADY_EXISTS,
            )
        self._restore_monitoring_experiment_if_deleted(name, experiment)
        return experiment.experiment_id

    def get_monitoring_experiment_id_by_name(self, name: str) -> str | None:
        """Return an active monitoring experiment id for a name, or `None`.

        Args:
            name: Monitoring experiment name to resolve.

        Returns:
            The experiment id if MLflow resolves the name to an active
            experiment; otherwise `None`.
        """
        experiment = self._get_experiment_by_name(name)
        if experiment is None:
            return None
        if experiment.lifecycle_stage != "active":
            return None
        return experiment.experiment_id

    def get_monitoring_experiment_tags(self, experiment_id: str) -> dict[str, str]:
        """Return monitoring experiment tags as a detached plain mapping.

        Args:
            experiment_id: Monitoring experiment identifier to read.

        Returns:
            A copied dictionary of experiment tags. The caller may mutate the
            returned mapping without affecting MLflow state.
        """
        experiment = self._client.get_experiment(experiment_id)
        return dict(experiment.tags)

    def set_monitoring_experiment_tag(
        self,
        experiment_id: str,
        key: str,
        value: str,
    ) -> None:
        """Set one tag on a monitoring experiment.

        Args:
            experiment_id: Monitoring experiment identifier to update.
            key: Tag key.
            value: Tag value.
        """
        self._client.set_experiment_tag(experiment_id, key, value)

    def create_monitoring_run(self, experiment_id: str, tags: Mapping[str, str]) -> str:
        """Create a monitoring-owned run and return its MLflow run id.

        Args:
            experiment_id: Experiment that will own the run.
            tags: Initial run tags to apply at create time.

        Returns:
            The MLflow-assigned `monitoring_run_id` for the created run.
        """
        run = self._client.create_run(experiment_id, tags=dict(tags))
        return run.info.run_id

    def get_run(self, run_id: str) -> Run | None:
        """Return any MLflow run, or `None` when MLflow reports it as missing.

        This method is the main missing-run normalization point. Later gateway
        code can treat `None` as the deterministic absent-run signal without
        needing to understand MLflow's exception shape.

        Args:
            run_id: Run identifier to fetch.

        Returns:
            The raw MLflow `Run` when it exists, otherwise `None`. The supplied
            `run_id` may belong to either a source training run or a
            monitoring-owned run.

        Raises:
            MlflowException: If MLflow raises an error other than
                `RESOURCE_DOES_NOT_EXIST`.
        """
        try:
            return self._client.get_run(run_id)
        except MlflowException as exc:
            # Missing runs are normalized to None; other MLflow failures bubble up.
            if _normalize_error_code(exc.error_code) != _RESOURCE_DOES_NOT_EXIST:
                raise
            return None

    def terminate_monitoring_run(self, monitoring_run_id: str, status: str) -> None:
        """Terminate a monitoring-owned run with a final MVP status.

        Args:
            monitoring_run_id: Monitoring run identifier to terminate.
            status: Final MLflow run status. MVP supports only `FINISHED` and
                `FAILED`.

        Raises:
            ValueError: If `status` is not `FINISHED` or `FAILED`.
        """
        if status not in {"FINISHED", "FAILED"}:
            raise ValueError("status must be FINISHED or FAILED")
        self._client.set_terminated(monitoring_run_id, status=status)

    def set_monitoring_run_tags(
        self,
        monitoring_run_id: str,
        tags: Mapping[str, str],
    ) -> None:
        """Set multiple tags on an existing monitoring-owned run.

        Args:
            monitoring_run_id: Monitoring run identifier to update.
            tags: Tags to write.
        """
        for key, value in tags.items():
            self._client.set_tag(monitoring_run_id, key, value)

    def get_run_metrics(self, run_id: str) -> dict[str, float]:
        """Return run metrics as a plain dictionary.

        Args:
            run_id: Training or monitoring run identifier to inspect.

        Returns:
            A copied metric mapping. Missing runs normalize to `{}`.
        """
        run = self.get_run(run_id)
        if run is None:
            return {}
        return dict(run.data.metrics)

    def get_run_params(self, run_id: str) -> dict[str, str]:
        """Return run params as a plain dictionary.

        Args:
            run_id: Training or monitoring run identifier to inspect.

        Returns:
            A copied param mapping. Missing runs normalize to `{}`.
        """
        run = self.get_run(run_id)
        if run is None:
            return {}
        return dict(run.data.params)

    def get_run_tags(self, run_id: str) -> dict[str, str]:
        """Return run tags as a plain dictionary.

        Args:
            run_id: Training or monitoring run identifier to inspect.

        Returns:
            A copied tag mapping. Missing runs normalize to `{}`.
        """
        run = self.get_run(run_id)
        if run is None:
            return {}
        return dict(run.data.tags)

    def list_artifact_paths(self, run_id: str) -> list[str]:
        """Return all artifact file paths for a run in deterministic order.

        MLflow lists artifacts one directory at a time. This helper performs a
        recursive walk so later gateway code can ask for one flattened list of
        relative file paths.

        Args:
            run_id: Training or monitoring run identifier to inspect.

        Returns:
            Sorted relative artifact file paths. Missing runs normalize to `[]`.
        """
        if self.get_run(run_id) is None:
            return []
        return sorted(self._list_artifact_paths_recursive(run_id, path=None))

    def log_monitoring_run_json_artifact(
        self,
        monitoring_run_id: str,
        data: dict[str, Any],
        path: str,
    ) -> None:
        """Log one dictionary payload as a JSON artifact on a monitoring run.

        Args:
            monitoring_run_id: Monitoring run identifier that will own the
                artifact.
            data: JSON-serializable dictionary payload.
            path: Artifact path such as `outputs/result.json`.
        """
        self._client.log_dict(monitoring_run_id, data, path)

    def _list_artifact_paths_recursive(self, run_id: str, path: str | None) -> list[str]:
        """Collect file artifact paths under one optional artifact prefix."""
        paths: list[str] = []
        for artifact in self._client.list_artifacts(run_id, path):
            # MLflow surfaces directories and files uniformly; recurse only on directories.
            if artifact.is_dir:
                paths.extend(self._list_artifact_paths_recursive(run_id, artifact.path))
                continue
            paths.append(artifact.path)
        return paths

    def _restore_monitoring_experiment_if_deleted(
        self,
        name: str,
        experiment: Experiment,
    ) -> None:
        """Restore a deleted monitoring experiment, tolerating one restore race."""
        if experiment.lifecycle_stage != "deleted":
            return

        # Monitoring experiments are system-owned timeline state, not
        # disposable cache entries, so recover soft-deleted timelines.
        try:
            self._client.restore_experiment(experiment.experiment_id)
        except MlflowException as exc:
            if _normalize_error_code(exc.error_code) != _RESOURCE_DOES_NOT_EXIST:
                raise

            restored = self._get_experiment_by_name(name)
            if (
                restored is None
                or restored.experiment_id != experiment.experiment_id
                or restored.lifecycle_stage != "active"
            ):
                raise exc

    def _get_experiment_by_name(self, name: str) -> Experiment | None:
        """Return the raw MLflow experiment for internal lifecycle-aware flows."""
        return self._client.get_experiment_by_name(name)
