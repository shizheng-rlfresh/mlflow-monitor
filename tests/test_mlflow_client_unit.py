"""Focused unit tests for hard-to-trigger MLflow adapter branches."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from mlflow.entities import Experiment, Run, ViewType
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)

from mlflow_monitor.mlflow_client import MonitorMLflowClient


class _PagedRuns(list[Run]):
    def __init__(self, runs: list[Run], token: str | None) -> None:
        super().__init__(runs)
        self.token = token


def test_list_monitoring_runs_with_tag_paginates_and_detaches_tags() -> None:
    first_tags = {"training.source_run_id": "source-1"}
    first_run = MagicMock()
    first_run.info.run_id = "run-1"
    first_run.data.tags = first_tags
    second_run = MagicMock()
    second_run.info.run_id = "run-2"
    second_run.data.tags = {"training.source_run_id": "source-2"}
    stub_client = MagicMock()
    stub_client.search_runs.side_effect = [
        _PagedRuns([first_run], "next-page"),
        _PagedRuns([second_run], None),
    ]

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    snapshots = client.list_monitoring_runs_with_tag(
        "experiment-1",
        "training.source_run_id",
    )

    assert tuple(snapshot.run_id for snapshot in snapshots) == ("run-1", "run-2")
    assert snapshots[0].tags == {"training.source_run_id": "source-1"}
    first_tags["training.source_run_id"] = "changed"
    assert snapshots[0].tags == {"training.source_run_id": "source-1"}
    with pytest.raises(TypeError):
        snapshots[0].tags["extra"] = "not-allowed"  # type: ignore[index]
    assert stub_client.search_runs.call_args_list == [
        call(
            experiment_ids=["experiment-1"],
            filter_string="tags.`training.source_run_id` IS NOT NULL",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000,
            page_token=None,
        ),
        call(
            experiment_ids=["experiment-1"],
            filter_string="tags.`training.source_run_id` IS NOT NULL",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=1000,
            page_token="next-page",
        ),
    ]


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


def test_get_run_experiment_name_returns_none_when_experiment_cannot_be_resolved() -> None:
    stub_run = MagicMock()
    stub_run.info.experiment_id = "missing-experiment"
    stub_client = MagicMock()
    stub_client.get_run.return_value = stub_run
    stub_client.get_experiment.side_effect = MlflowException(
        "Experiment missing.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert client.get_run_experiment_name("run-123") is None


def test_read_monitoring_run_json_artifact_downloads_the_exact_requested_path(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "prepared_context.json"
    artifact_path.write_text("{}", encoding="utf-8")
    stub_client = MagicMock()
    stub_client.download_artifacts.return_value = str(artifact_path)

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    client.read_monitoring_run_json_artifact(
        "monitoring-run-1",
        "state/prepared_context.json",
    )
    stub_client.download_artifacts.assert_called_once_with(
        "monitoring-run-1",
        "state/prepared_context.json",
    )


def test_read_monitoring_run_json_artifact_returns_none_when_artifact_is_missing() -> None:
    stub_client = MagicMock()
    stub_client.download_artifacts.side_effect = MlflowException(
        "Artifact is missing.",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    assert (
        client.read_monitoring_run_json_artifact(
            "monitoring-run-1",
            "state/prepared_context.json",
        )
        is None
    )


def test_read_monitoring_run_json_artifact_propagates_transport_failure() -> None:
    transport_error = MlflowException(
        "Artifact transport failed.",
        error_code=INTERNAL_ERROR,
    )
    stub_client = MagicMock()
    stub_client.download_artifacts.side_effect = transport_error

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    with pytest.raises(MlflowException) as exc_info:
        client.read_monitoring_run_json_artifact(
            "monitoring-run-1",
            "state/prepared_context.json",
        )

    assert exc_info.value is transport_error


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_log_monitoring_run_json_artifact_rejects_non_finite_numbers(
    value: float,
) -> None:
    stub_client = MagicMock()

    with patch("mlflow_monitor.mlflow_client.MlflowClient", return_value=stub_client):
        client = MonitorMLflowClient(tracking_uri="file:///ignored")

    with pytest.raises(ValueError):
        client.log_monitoring_run_json_artifact(
            "monitoring-run-1",
            {"nested": {"value": value}},
            "state/prepared_context.json",
        )

    assert not stub_client.method_calls
