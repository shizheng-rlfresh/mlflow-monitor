"""Focused unit tests for hard-to-trigger MLflow adapter branches."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

from mlflow_monitor.mlflow_client import MonitorMLflowClient


def test_get_or_create_experiment_recovers_from_duplicate_create_race() -> None:
    existing = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="active",
        tags={},
    )
    stub_client = MagicMock()
    stub_client.create_experiment.side_effect = MlflowException(
        "Experiment already exists.",
        error_code=RESOURCE_ALREADY_EXISTS,
    )
    stub_client.get_experiment_by_name.side_effect = [None, existing]

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_or_create_experiment("churn-monitoring") == "123"
    stub_client.create_experiment.assert_called_once_with(
        "churn-monitoring",
        artifact_location=None,
    )
    assert stub_client.get_experiment_by_name.call_args_list == [
        call("churn-monitoring"),
        call("churn-monitoring"),
    ]


def test_get_or_create_experiment_passes_artifact_location_on_first_create() -> None:
    stub_client = MagicMock()
    stub_client.get_experiment_by_name.return_value = None
    stub_client.create_experiment.return_value = "123"

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert (
        client.get_or_create_experiment(
            "churn-monitoring",
            artifact_location="file:///tmp/artifacts",
        )
        == "123"
    )
    stub_client.create_experiment.assert_called_once_with(
        "churn-monitoring",
        artifact_location="file:///tmp/artifacts",
    )


def test_get_or_create_experiment_restores_deleted_experiment() -> None:
    deleted = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="deleted",
        tags={},
    )
    stub_client = MagicMock()
    stub_client.get_experiment_by_name.return_value = deleted

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_or_create_experiment("churn-monitoring") == "123"
    stub_client.restore_experiment.assert_called_once_with("123")
    stub_client.create_experiment.assert_not_called()
