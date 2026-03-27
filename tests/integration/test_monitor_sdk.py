"""Integration tests for the public monitor SDK facade."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from mlflow_monitor import monitor
from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus, MonitoringRunReference


def test_monitor_run_defaults_to_real_mlflow_gateway(
    tracking_uri: str,
    artifact_root_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_training_run,
) -> None:
    monkeypatch.chdir(tmp_path)  # ensure we don't pollute the current directory with ./mlruns
    raw = MlflowClient(tracking_uri=tracking_uri)
    baseline_run_id = create_training_run(
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
    current_run_id = create_training_run(
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


def test_monitor_run_warn_outcome_via_environment_mismatch(
    tracking_uri: str,
    artifact_root_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_training_run,
) -> None:
    monkeypatch.chdir(tmp_path)  # ensure we don't pollute the current directory with ./mlruns
    raw = MlflowClient(tracking_uri=tracking_uri)
    baseline_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="baseline",
        metrics={"f1": 0.87},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "schema.income": "float",
            "data_scope": "validation:2026-03-01",
        },
    )
    current_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="current-warn",
        metrics={"f1": 0.85},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.11",
            "schema.age": "int",
            "schema.income": "float",
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

    monitoring_run = raw.get_run(result.monitoring_run_id)
    artifact_dir = Path(raw.download_artifacts(result.monitoring_run_id, "outputs"))
    payload = json.loads((artifact_dir / "result.json").read_text())

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert result.comparability_status is ComparabilityStatus.WARN
    assert monitoring_run.info.status == "FINISHED"
    assert payload["monitoring_run_id"] == result.monitoring_run_id
    assert payload["lifecycle_status"] == "checked"
    assert payload["comparability_status"] == "warn"
