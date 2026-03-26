"""Integration tests for the public monitor SDK facade."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from mlflow_monitor import monitor
from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus, MonitoringRunReference


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """Return a pytest-managed local MLflow SQLite tracking URI."""
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def artifact_root_uri(tmp_path: Path) -> str:
    """Return a pytest-managed artifact root for MLflow training experiments."""
    return (tmp_path / "artifacts").as_uri()


def _create_training_run(
    *,
    raw: MlflowClient,
    experiment_name: str,
    artifact_root_uri: str,
    run_name: str,
    metrics: dict[str, float],
    params: dict[str, str],
    tags: dict[str, str],
) -> str:
    """Create one source training run with MVP-shaped evidence."""
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
    raw.set_terminated(run_id, status="FINISHED")
    return run_id


def test_monitor_run_defaults_to_real_mlflow_gateway(
    tracking_uri: str,
    artifact_root_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    raw = MlflowClient(tracking_uri=tracking_uri)
    baseline_run_id = _create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="baseline",
        metrics={"f1": 0.87},
        params={"feature_columns": "age"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "data_scope": "validation:2026-03-01",
        },
    )
    current_run_id = _create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="current",
        metrics={"f1": 0.91},
        params={"feature_columns": "age"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "data_scope": "validation:2026-03-01",
        },
    )

    previous_tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        result = monitor.run(
            subject_id="churn_model",
            source_run_id=current_run_id,
            baseline_source_run_id=baseline_run_id,
        )
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    experiment = raw.get_experiment_by_name("mlflow_monitor/churn_model")
    assert experiment is not None
    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert result.comparability_status is ComparabilityStatus.PASS
    assert result.references == (
        MonitoringRunReference(kind="baseline", reference_run_id=baseline_run_id),
    )
    assert experiment.tags["monitoring.latest_run_id"] == result.monitoring_run_id

    monitoring_run = raw.get_run(result.monitoring_run_id)
    assert monitoring_run.info.status == "FINISHED"

    artifact_dir = Path(raw.download_artifacts(result.monitoring_run_id, "outputs"))
    payload = json.loads((artifact_dir / "result.json").read_text())
    assert payload["monitoring_run_id"] == result.monitoring_run_id
    assert payload["lifecycle_status"] == "checked"
