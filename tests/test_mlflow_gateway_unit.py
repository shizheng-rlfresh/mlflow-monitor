"""Focused unit tests for MLflow gateway edge cases."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from mlflow_monitor.domain import LifecycleStatus
from mlflow_monitor.errors import GatewayNamespaceViolation
from mlflow_monitor.gateway import GatewayConfig, IdempotencyKey
from mlflow_monitor.mlflow_gateway import MLflowMonitoringGateway
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult


def _make_result(
    *,
    monitoring_run_id: str,
    lifecycle_status: LifecycleStatus,
) -> MonitorRunResult:
    return MonitorRunResult(
        monitoring_run_id=monitoring_run_id,
        subject_id="churn_model",
        timeline_id="timeline-1",
        lifecycle_status=lifecycle_status,
        comparability_status=None,
        summary=None,
        finding_ids=(),
        diff_ids=(),
        references=(),
        error=(
            None
            if lifecycle_status is not LifecycleStatus.FAILED
            else MonitorRunError(
                code="failed",
                message="failed",
                stage="check",
            )
        ),
    )


def test_create_or_reuse_monitoring_run_writes_idempotency_tag_last() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_tags.return_value = {}
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.create_monitoring_run.return_value = "monitoring-run-1"

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-1",
            recipe_id="system_default",
            recipe_version="v0",
        )
    )

    assert stub_client.set_monitoring_experiment_tag.call_args_list == [
        call("experiment-1", "monitoring.run.0", "monitoring-run-1"),
        call("experiment-1", "monitoring.latest_run_id", "monitoring-run-1"),
        call("experiment-1", "monitoring.next_sequence_index", "1"),
        call(
            "experiment-1",
            "training.train-run-1.monitoring_run_id",
            "monitoring-run-1",
        ),
    ]


def test_finalize_monitoring_run_result_rejects_mismatched_run_id_before_side_effects() -> None:
    stub_client = MagicMock()

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(ValueError, match="must match"):
        gateway.finalize_monitoring_run_result(
            monitoring_run_id="monitoring-run-1",
            result=_make_result(
                monitoring_run_id="monitoring-run-2",
                lifecycle_status=LifecycleStatus.CHECKED,
            ),
        )

    stub_client.log_monitoring_run_json_artifact.assert_not_called()
    stub_client.terminate_monitoring_run.assert_not_called()


def test_finalize_monitoring_run_result_rejects_non_terminal_status_before_side_effects() -> None:
    stub_client = MagicMock()

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(ValueError, match="supports only CHECKED and FAILED"):
        gateway.finalize_monitoring_run_result(
            monitoring_run_id="monitoring-run-1",
            result=_make_result(
                monitoring_run_id="monitoring-run-1",
                lifecycle_status=LifecycleStatus.CREATED,
            ),
        )

    stub_client.log_monitoring_run_json_artifact.assert_not_called()
    stub_client.terminate_monitoring_run.assert_not_called()


@pytest.mark.parametrize(
    ("lifecycle_status", "expected_mlflow_status"),
    [
        (LifecycleStatus.CHECKED, "FINISHED"),
        (LifecycleStatus.FAILED, "FAILED"),
    ],
)
def test_finalize_monitoring_run_result_persists_artifact_and_termination(
    lifecycle_status: LifecycleStatus,
    expected_mlflow_status: str,
) -> None:
    stub_client = MagicMock()

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    result = _make_result(
        monitoring_run_id="monitoring-run-1",
        lifecycle_status=lifecycle_status,
    )

    gateway.finalize_monitoring_run_result(
        monitoring_run_id="monitoring-run-1",
        result=result,
    )

    stub_client.log_monitoring_run_json_artifact.assert_called_once_with(
        "monitoring-run-1",
        result.to_dict(),
        "outputs/result.json",
    )
    stub_client.terminate_monitoring_run.assert_called_once_with(
        "monitoring-run-1",
        expected_mlflow_status,
    )


@pytest.mark.parametrize(
    ("tag_key", "tag_value"),
    [
        ("monitoring.comparability_status", "unknown"),
        ("monitoring.sequence_index", "not-an-int"),
        ("monitoring.lifecycle_status", "not-a-status"),
    ],
)
def test_get_monitoring_run_returns_none_for_malformed_persisted_tags(
    tag_key: str,
    tag_value: str,
) -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.0": "monitoring-run-1",
    }
    stub_client.get_run_tags.return_value = {
        "monitoring.sequence_index": "0",
        "monitoring.lifecycle_status": "checked",
        "monitoring.comparability_status": "pass",
        tag_key: tag_value,
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    assert gateway.get_monitoring_run("churn_model", "monitoring-run-1") is None


def test_resolve_timeline_monitoring_run_id_ignores_malformed_index_tag_keys() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.latest": "monitoring-run-1",
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    assert gateway.resolve_timeline_monitoring_run_id("churn_model", "monitoring-run-1") is None


def test_get_monitoring_run_returns_none_for_malformed_index_tag_key() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.latest": "monitoring-run-1",
    }
    stub_client.get_run_tags.return_value = {
        "monitoring.sequence_index": "0",
        "monitoring.lifecycle_status": "checked",
        "monitoring.comparability_status": "pass",
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    assert gateway.get_monitoring_run("churn_model", "monitoring-run-1") is None


def test_list_timeline_monitoring_runs_skips_malformed_reconstructed_run() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.next_sequence_index": "2",
        "monitoring.run.0": "monitoring-run-bad",
        "monitoring.run.1": "monitoring-run-good",
    }
    stub_client.get_run_tags.side_effect = [
        {
            "monitoring.sequence_index": "not-an-int",
            "monitoring.lifecycle_status": "checked",
        },
        {
            "monitoring.sequence_index": "1",
            "monitoring.lifecycle_status": "checked",
            "monitoring.comparability_status": "pass",
        },
    ]

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    runs = gateway.list_timeline_monitoring_runs("churn_model")

    assert tuple(run.monitoring_run_id for run in runs) == ("monitoring-run-good",)


def test_list_timeline_monitoring_runs_raises_for_malformed_next_sequence_index() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.next_sequence_index": "not-an-int",
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayNamespaceViolation, match="monitoring.next_sequence_index"):
        gateway.list_timeline_monitoring_runs("churn_model")


def test_create_or_reuse_monitoring_run_raises_for_malformed_next_sequence_index() -> None:
    stub_client = MagicMock()
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.next_sequence_index": "not-an-int",
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayNamespaceViolation, match="monitoring.next_sequence_index"):
        gateway.create_or_reuse_monitoring_run(
            IdempotencyKey(
                subject_id="churn_model",
                source_run_id="train-run-1",
                recipe_id="system_default",
                recipe_version="v0",
            )
        )


def test_create_or_reuse_monitoring_run_raises_for_malformed_index_tag_key() -> None:
    stub_client = MagicMock()
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "training.train-run-1.monitoring_run_id": "monitoring-run-1",
        "monitoring.run.latest": "monitoring-run-1",
    }
    stub_client.get_run_tags.return_value = {
        "monitoring.recipe_id": "system_default",
        "monitoring.recipe_version": "v0",
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayNamespaceViolation, match="monitoring.run.latest"):
        gateway.create_or_reuse_monitoring_run(
            IdempotencyKey(
                subject_id="churn_model",
                source_run_id="train-run-1",
                recipe_id="system_default",
                recipe_version="v0",
            )
        )
