"""Integration tests for the real-MLflow monitoring gateway."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlflow import MlflowClient

from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus, MonitoringRunReference
from mlflow_monitor.gateway import GatewayConfig, IdempotencyKey
from mlflow_monitor.mlflow_gateway import MLflowMonitoringGateway
from mlflow_monitor.orchestration import run_orchestration


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """Return a pytest-managed local MLflow SQLite tracking URI."""
    return f"sqlite:///{tmp_path / 'mlflow.db'}"


@pytest.fixture
def artifact_root_uri(tmp_path: Path) -> str:
    """Return a pytest-managed artifact root for MLflow experiments."""
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
    artifact_payload: dict[str, object] | None = None,
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
    if artifact_payload is not None:
        raw.log_dict(run_id, artifact_payload, "outputs/training.json")
    raw.set_terminated(run_id, status="FINISHED")
    return run_id


def test_mlflow_gateway_first_run_bootstraps_and_finalizes_result(
    tracking_uri: str,
    artifact_root_uri: str,
) -> None:
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
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id=current_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    experiment = raw.get_experiment_by_name("mlflow_monitor/churn_model")
    assert experiment is not None
    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert result.comparability_status is ComparabilityStatus.PASS
    assert result.references == (
        MonitoringRunReference(kind="baseline", reference_run_id=baseline_run_id),
    )
    assert experiment.tags["training.baseline_run_id"] == baseline_run_id
    assert experiment.tags["monitoring.latest_run_id"] == result.monitoring_run_id
    assert experiment.tags["monitoring.next_sequence_index"] == "1"
    assert experiment.tags["monitoring.run.0"] == result.monitoring_run_id
    assert (
        experiment.tags[f"training.{current_run_id}.monitoring_run_id"] == result.monitoring_run_id
    )

    monitoring_run = raw.get_run(result.monitoring_run_id)
    assert monitoring_run.info.status == "FINISHED"
    assert monitoring_run.data.tags["monitoring.lifecycle_status"] == "checked"
    assert monitoring_run.data.tags["monitoring.comparability_status"] == "pass"

    artifact_dir = Path(raw.download_artifacts(result.monitoring_run_id, "outputs"))
    payload = json.loads((artifact_dir / "result.json").read_text())
    assert payload["monitoring_run_id"] == result.monitoring_run_id
    assert payload["lifecycle_status"] == "checked"


def test_mlflow_gateway_reuses_baseline_resolves_previous_and_idempotent_rerun(
    tracking_uri: str,
    artifact_root_uri: str,
) -> None:
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
    first_current_run_id = _create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="current-1",
        metrics={"f1": 0.91},
        params={"feature_columns": "age"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "data_scope": "validation:2026-03-01",
        },
    )
    second_current_run_id = _create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="current-2",
        metrics={"f1": 0.92},
        params={"feature_columns": "age"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "data_scope": "validation:2026-03-02",
        },
    )
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    first = run_orchestration(
        subject_id="churn_model",
        source_run_id=first_current_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id=second_current_run_id,
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    replay = run_orchestration(
        subject_id="churn_model",
        source_run_id=second_current_run_id,
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    experiment = raw.get_experiment_by_name("mlflow_monitor/churn_model")
    assert experiment is not None
    assert second.references == (
        MonitoringRunReference(kind="baseline", reference_run_id=baseline_run_id),
        MonitoringRunReference(kind="previous", reference_run_id=first.monitoring_run_id),
    )
    assert second.comparability_status is ComparabilityStatus.FAIL
    assert replay.monitoring_run_id == second.monitoring_run_id
    assert replay.references == second.references
    assert experiment.tags["monitoring.latest_run_id"] == second.monitoring_run_id
    assert experiment.tags["monitoring.next_sequence_index"] == "2"
    assert experiment.tags["monitoring.run.0"] == first.monitoring_run_id
    assert experiment.tags["monitoring.run.1"] == second.monitoring_run_id


def test_mlflow_gateway_owned_failure_terminates_failed_and_leaves_training_runs_unchanged(
    tracking_uri: str,
    artifact_root_uri: str,
) -> None:
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
        artifact_payload={"kind": "training"},
    )
    baseline_before = raw.get_run(baseline_run_id)
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    result = run_orchestration(
        subject_id="churn_model",
        source_run_id="missing-training-run",
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    baseline_after = raw.get_run(baseline_run_id)
    monitoring_run = raw.get_run(result.monitoring_run_id)
    artifact_dir = Path(raw.download_artifacts(result.monitoring_run_id, "outputs"))
    payload = json.loads((artifact_dir / "result.json").read_text())

    assert result.lifecycle_status is LifecycleStatus.FAILED
    assert monitoring_run.info.status == "FAILED"
    assert payload["lifecycle_status"] == "failed"
    assert payload["error"]["code"] == "prepare_source_run_not_found"
    assert baseline_after.data.metrics == baseline_before.data.metrics
    assert baseline_after.data.params == baseline_before.data.params
    assert baseline_after.data.tags == baseline_before.data.tags


def test_mlflow_gateway_resolve_source_run_id_honors_source_experiment_filter(
    tracking_uri: str,
    artifact_root_uri: str,
) -> None:
    raw = MlflowClient(tracking_uri=tracking_uri)
    training_run_id = _create_training_run(
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
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    assert (
        gateway.resolve_source_run_id(
            subject_id="churn_model",
            source_experiment="training/churn",
            run_selector=training_run_id,
        )
        == training_run_id
    )
    assert (
        gateway.resolve_source_run_id(
            subject_id="churn_model",
            source_experiment="training/other",
            run_selector=training_run_id,
        )
        is None
    )


def test_mlflow_gateway_create_or_reuse_allocates_new_run_for_recipe_version_change(
    tracking_uri: str,
    artifact_root_uri: str,
) -> None:
    raw = MlflowClient(tracking_uri=tracking_uri)
    training_run_id = _create_training_run(
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
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    first = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id=training_run_id,
            recipe_id="system_default",
            recipe_version="v0",
        )
    )
    second = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id=training_run_id,
            recipe_id="system_default",
            recipe_version="v1",
        )
    )

    experiment = raw.get_experiment_by_name("mlflow_monitor/churn_model")
    assert experiment is not None
    assert first.allocated is True
    assert second.allocated is True
    assert second.monitoring_run_id != first.monitoring_run_id
    assert second.sequence_index == first.sequence_index + 1
    assert experiment.tags["monitoring.run.0"] == first.monitoring_run_id
    assert experiment.tags["monitoring.run.1"] == second.monitoring_run_id
