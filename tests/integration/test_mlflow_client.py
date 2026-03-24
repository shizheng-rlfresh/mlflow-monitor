"""Integration tests for the MLflow client adapter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlflow import MlflowClient
from mlflow.entities import RunStatus

from mlflow_monitor.mlflow_client import MonitorMLflowClient


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """Return a pytest-managed local MLflow file store path."""
    return str(tmp_path / "mlruns")


def test_get_experiment_id_by_name_returns_none_before_creation(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)

    assert client.get_experiment_id_by_name("churn-monitoring") is None


def test_get_or_create_experiment_creates_then_reuses(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)

    first = client.get_or_create_experiment("churn-monitoring")
    second = client.get_or_create_experiment("churn-monitoring")

    assert first == second
    assert client.get_experiment_id_by_name("churn-monitoring") == first


def test_experiment_tag_round_trip(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_or_create_experiment("fraud-monitoring")

    client.set_experiment_tag(experiment_id, "monitoring.latest_run_id", "run-123")

    assert client.get_experiment_tags(experiment_id) == {
        "monitoring.latest_run_id": "run-123"
    }


def test_create_run_and_read_run_data(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)
    raw = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_or_create_experiment("churn-monitoring")
    monitoring_run_id = client.create_run(
        experiment_id,
        tags={"training.source_run_id": "train-run-1"},
    )

    raw.log_metric(monitoring_run_id, "f1", 0.91)
    raw.log_param(monitoring_run_id, "feature_columns", "age,income")
    raw.set_tag(monitoring_run_id, "schema.age", "int")

    run = client.get_run(monitoring_run_id)

    assert run is not None
    assert run.info.run_id == monitoring_run_id
    assert client.get_run_metrics(monitoring_run_id) == {"f1": 0.91}
    assert client.get_run_params(monitoring_run_id) == {"feature_columns": "age,income"}
    tags = client.get_run_tags(monitoring_run_id)
    assert tags["training.source_run_id"] == "train-run-1"
    assert tags["schema.age"] == "int"


def test_get_run_returns_none_for_missing_run(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)

    assert client.get_run("missing-run-id") is None


def test_set_tags_updates_run_tags(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_or_create_experiment("churn-monitoring")
    monitoring_run_id = client.create_run(experiment_id, tags={})

    client.set_tags(
        monitoring_run_id,
        {
            "monitoring.lifecycle_status": "prepared",
            "monitoring.recipe_id": "default",
        },
    )

    tags = client.get_run_tags(monitoring_run_id)
    assert tags["monitoring.lifecycle_status"] == "prepared"
    assert tags["monitoring.recipe_id"] == "default"


@pytest.mark.parametrize("status", ["FINISHED", "FAILED"])
def test_terminate_run_sets_expected_status(tracking_uri: str, status: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_or_create_experiment("churn-monitoring")
    monitoring_run_id = client.create_run(experiment_id, tags={})

    client.terminate_run(monitoring_run_id, status)

    run = client.get_run(monitoring_run_id)
    assert run is not None
    assert run.info.status == status


def test_terminate_run_rejects_unknown_status(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_or_create_experiment("churn-monitoring")
    monitoring_run_id = client.create_run(experiment_id, tags={})

    with pytest.raises(ValueError, match="FINISHED or FAILED"):
        client.terminate_run(monitoring_run_id, RunStatus.to_string(RunStatus.RUNNING))


def test_list_artifact_paths_returns_recursive_sorted_paths(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_or_create_experiment("churn-monitoring")
    monitoring_run_id = client.create_run(experiment_id, tags={})

    client.log_json_artifact(monitoring_run_id, {"value": 1}, "outputs/result.json")
    client.log_json_artifact(monitoring_run_id, {"value": 2}, "nested/data.json")

    assert client.list_artifact_paths(monitoring_run_id) == [
        "nested/data.json",
        "outputs/result.json",
    ]


def test_log_json_artifact_writes_requested_json_path(tracking_uri: str) -> None:
    client = MonitorMLflowClient(tracking_uri=tracking_uri)
    raw = MlflowClient(tracking_uri=tracking_uri)
    experiment_id = client.get_or_create_experiment("churn-monitoring")
    monitoring_run_id = client.create_run(experiment_id, tags={})

    client.log_json_artifact(monitoring_run_id, {"status": "ok"}, "outputs/result.json")

    artifact_dir = Path(raw.download_artifacts(monitoring_run_id, "outputs"))
    payload = json.loads((artifact_dir / "result.json").read_text())

    assert payload == {"status": "ok"}
