"""Focused unit tests for hard-to-trigger MLflow adapter branches."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest
from mlflow.entities import Experiment
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS, RESOURCE_DOES_NOT_EXIST

from mlflow_monitor.mlflow_client import MonitorMLflowClient


def test_get_or_create_monitoring_experiment_recovers_from_duplicate_create_race() -> None:
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

    assert client.get_or_create_monitoring_experiment("churn-monitoring") == "123"
    stub_client.create_experiment.assert_called_once_with(
        "churn-monitoring",
        artifact_location=None,
    )
    assert stub_client.get_experiment_by_name.call_args_list == [
        call("churn-monitoring"),
        call("churn-monitoring"),
    ]


def test_get_or_create_monitoring_experiment_passes_artifact_location_on_first_create() -> None:
    stub_client = MagicMock()
    stub_client.get_experiment_by_name.return_value = None
    stub_client.create_experiment.return_value = "123"

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert (
        client.get_or_create_monitoring_experiment(
            "churn-monitoring",
            artifact_location="file:///tmp/artifacts",
        )
        == "123"
    )
    stub_client.create_experiment.assert_called_once_with(
        "churn-monitoring",
        artifact_location="file:///tmp/artifacts",
    )


def test_get_or_create_monitoring_experiment_restores_deleted_experiment() -> None:
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

    assert client.get_or_create_monitoring_experiment("churn-monitoring") == "123"
    stub_client.restore_experiment.assert_called_once_with("123")
    stub_client.create_experiment.assert_not_called()


def test_get_or_create_monitoring_experiment_accepts_restore_race_for_deleted_experiment() -> None:
    deleted = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="deleted",
        tags={},
    )
    restored = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="active",
        tags={},
    )
    stub_client = MagicMock()
    stub_client.get_experiment_by_name.side_effect = [deleted, restored]
    stub_client.restore_experiment.side_effect = MlflowException(
        "No Experiment with id=123 exists",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_or_create_monitoring_experiment("churn-monitoring") == "123"
    assert stub_client.get_experiment_by_name.call_args_list == [
        call("churn-monitoring"),
        call("churn-monitoring"),
    ]
    stub_client.restore_experiment.assert_called_once_with("123")


def test_get_or_create_monitoring_experiment_reraises_failed_restore_when_not_restored() -> None:
    deleted = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="deleted",
        tags={},
    )
    restore_error = MlflowException(
        "No Experiment with id=123 exists",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    stub_client = MagicMock()
    stub_client.get_experiment_by_name.side_effect = [deleted, None]
    stub_client.restore_experiment.side_effect = restore_error

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    with pytest.raises(MlflowException) as exc_info:
        client.get_or_create_monitoring_experiment("churn-monitoring")

    assert exc_info.value is restore_error


def test_get_or_create_monitoring_experiment_restores_deleted_experiment_after_duplicate() -> None:
    deleted = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="deleted",
        tags={},
    )
    stub_client = MagicMock()
    stub_client.create_experiment.side_effect = MlflowException(
        "Experiment already exists.",
        error_code=RESOURCE_ALREADY_EXISTS,
    )
    stub_client.get_experiment_by_name.side_effect = [None, deleted]

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_or_create_monitoring_experiment("churn-monitoring") == "123"
    stub_client.restore_experiment.assert_called_once_with("123")
    assert stub_client.get_experiment_by_name.call_args_list == [
        call("churn-monitoring"),
        call("churn-monitoring"),
    ]


def test_get_or_create_monitoring_experiment_accepts_restore_race_after_duplicate() -> None:
    deleted = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="deleted",
        tags={},
    )
    restored = Experiment(
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
    stub_client.get_experiment_by_name.side_effect = [None, deleted, restored]
    stub_client.restore_experiment.side_effect = MlflowException(
        "No Experiment with id=123 exists",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_or_create_monitoring_experiment("churn-monitoring") == "123"
    assert stub_client.get_experiment_by_name.call_args_list == [
        call("churn-monitoring"),
        call("churn-monitoring"),
        call("churn-monitoring"),
    ]
    stub_client.restore_experiment.assert_called_once_with("123")


def test_get_or_create_monitoring_experiment_accepts_string_duplicate_error_code() -> None:
    existing = Experiment(
        experiment_id="123",
        name="churn-monitoring",
        artifact_location="/tmp/mlruns",
        lifecycle_stage="active",
        tags={},
    )
    exc = MlflowException(
        "Experiment already exists.",
        error_code=RESOURCE_ALREADY_EXISTS,
    )
    exc.error_code = "RESOURCE_ALREADY_EXISTS"
    stub_client = MagicMock()
    stub_client.create_experiment.side_effect = exc
    stub_client.get_experiment_by_name.side_effect = [None, existing]

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_or_create_monitoring_experiment("churn-monitoring") == "123"


def test_get_run_returns_none_for_proto_missing_run_error_code() -> None:
    stub_client = MagicMock()
    stub_client.get_run.side_effect = MlflowException(
        "Run missing.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_run("missing-run-id") is None


def test_get_run_returns_none_for_string_missing_run_error_code() -> None:
    exc = MlflowException(
        "Run missing.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )
    exc.error_code = "RESOURCE_DOES_NOT_EXIST"
    stub_client = MagicMock()
    stub_client.get_run.side_effect = exc

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_run("missing-run-id") is None
