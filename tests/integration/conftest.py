"""Shared integration helpers for real-MLflow tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pytest
from mlflow import MlflowClient


def _create_training_run(
    *,
    raw: MlflowClient,
    experiment_name: str,
    artifact_root_uri: str,
    run_name: str,
    metrics: Mapping[str, float],
    params: Mapping[str, str],
    tags: Mapping[str, str],
    artifact_payload: Mapping[str, object] | None = None,
) -> str:
    """Create one terminal training run with MVP-shaped evidence."""
    experiment = raw.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = raw.create_experiment(
            experiment_name,
            artifact_location=artifact_root_uri,
        )
    else:
        experiment_id = experiment.experiment_id

    run = raw.create_run(experiment_id, tags={"mlflow.runName": run_name})
    run_id = run.info.run_id
    for key, value in metrics.items():
        raw.log_metric(run_id, key, value)
    for key, value in params.items():
        raw.log_param(run_id, key, value)
    for key, value in tags.items():
        raw.set_tag(run_id, key, value)
    if artifact_payload is not None:
        raw.log_dict(run_id, dict(artifact_payload), "outputs/training.json")
    raw.set_terminated(run_id, status="FINISHED")
    return run_id


def _snapshot_training_run(*, raw: MlflowClient, run_id: str) -> dict[str, Any]:
    """Return a stable snapshot of one training run for read-only assertions."""
    run = raw.get_run(run_id)
    return {
        "status": run.info.status,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
        "tags": dict(run.data.tags),
    }


def _assert_training_run_unchanged(
    *,
    raw: MlflowClient,
    run_id: str,
    snapshot: Mapping[str, Any],
) -> None:
    """Assert one training run still matches a previously captured snapshot."""
    current = _snapshot_training_run(raw=raw, run_id=run_id)
    assert current == dict(snapshot)


@pytest.fixture
def create_training_run() -> Callable[..., str]:
    """Expose the shared training-run factory to integration tests."""
    return _create_training_run


@pytest.fixture
def snapshot_training_run() -> Callable[..., dict[str, Any]]:
    """Expose the shared training-run snapshot helper to integration tests."""
    return _snapshot_training_run


@pytest.fixture
def assert_training_run_unchanged() -> Callable[..., None]:
    """Expose the shared read-only assertion helper to integration tests."""
    return _assert_training_run_unchanged
