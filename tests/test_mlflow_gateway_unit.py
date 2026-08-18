"""Focused unit tests for MLflow gateway edge cases."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from mlflow_monitor.domain import DiffReferenceKind, LifecycleStatus, MonitoringRunReference
from mlflow_monitor.errors import GatewayConsistencyViolation, GatewayNamespaceViolation
from mlflow_monitor.gateway import GatewayConfig, IdempotencyKey
from mlflow_monitor.mlflow_client import MonitoringRunInfo, MonitoringRunTagSnapshot
from mlflow_monitor.mlflow_gateway import MLflowMonitoringGateway
from mlflow_monitor.result_contract import MonitorRunError, MonitorRunResult

_MONITORING_ALLOCATION_INCONSISTENT_FOR_TESTS = "monitoring_allocation_inconsistent"


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


def _make_mlflow_run(status: str) -> SimpleNamespace:
    return SimpleNamespace(info=SimpleNamespace(status=status))


def _allocation_snapshot(
    *,
    run_id: str,
    source_run_id: str,
    sequence_index: int | str,
    recipe_version: str = "v0",
) -> MonitoringRunTagSnapshot:
    return MonitoringRunTagSnapshot(
        run_id=run_id,
        tags={
            "training.source_run_id": source_run_id,
            "monitoring.sequence_index": str(sequence_index),
            "monitoring.lifecycle_status": "created",
            "monitoring.recipe_id": "system_default",
            "monitoring.recipe_version": recipe_version,
        },
    )


@pytest.mark.parametrize("persisted_tag_count", range(5))
def test_create_or_reuse_monitoring_run_repairs_partial_allocation_index(
    persisted_tag_count: int,
) -> None:
    canonical_tags = [
        ("monitoring.run.0", "monitoring-run-1"),
        ("monitoring.latest_run_id", "monitoring-run-1"),
        ("monitoring.next_sequence_index", "1"),
        ("training.train-run-1.monitoring_run_id", "monitoring-run-1"),
    ]
    experiment_tags = dict(canonical_tags[:persisted_tag_count])
    stub_client = MagicMock()
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = experiment_tags
    stub_client.list_monitoring_runs_with_tag.return_value = (
        _allocation_snapshot(
            run_id="monitoring-run-1",
            source_run_id="train-run-1",
            sequence_index=0,
        ),
    )
    stub_client.get_run_tags.return_value = dict(
        stub_client.list_monitoring_runs_with_tag.return_value[0].tags
    )

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    result = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-1",
            recipe_id="system_default",
            recipe_version="v0",
        )
    )

    assert result.monitoring_run_id == "monitoring-run-1"
    assert result.source_run_id == "train-run-1"
    assert result.timeline_id == "experiment-1"
    assert result.sequence_index == 0
    assert result.allocated is False
    assert stub_client.set_monitoring_experiment_tag.call_args_list == [
        call("experiment-1", key, value) for key, value in canonical_tags[persisted_tag_count:]
    ]
    stub_client.create_monitoring_run.assert_not_called()


def test_create_or_reuse_monitoring_run_repairs_orphan_before_new_allocation() -> None:
    stub_client = MagicMock()
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {}
    stub_client.list_monitoring_runs_with_tag.return_value = (
        _allocation_snapshot(
            run_id="monitoring-run-1",
            source_run_id="train-run-1",
            sequence_index=0,
        ),
    )
    stub_client.create_monitoring_run.return_value = MonitoringRunInfo(
        run_id="monitoring-run-2",
        run_name="monitoring-run-2",
    )

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    result = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-2",
            recipe_id="system_default",
            recipe_version="v0",
        )
    )

    assert result.monitoring_run_id == "monitoring-run-2"
    assert result.source_run_id == "train-run-2"
    assert result.timeline_id == "experiment-1"
    assert result.sequence_index == 1
    assert result.allocated is True
    stub_client.create_monitoring_run.assert_called_once_with(
        "experiment-1",
        tags={
            "training.source_run_id": "train-run-2",
            "monitoring.sequence_index": "1",
            "monitoring.lifecycle_status": "created",
            "monitoring.recipe_id": "system_default",
            "monitoring.recipe_version": "v0",
        },
        source_run_name=stub_client.get_run_name.return_value,
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
        call("experiment-1", "monitoring.run.1", "monitoring-run-2"),
        call("experiment-1", "monitoring.latest_run_id", "monitoring-run-2"),
        call("experiment-1", "monitoring.next_sequence_index", "2"),
        call(
            "experiment-1",
            "training.train-run-2.monitoring_run_id",
            "monitoring-run-2",
        ),
    ]


def test_get_timeline_state_distinguishes_no_allocation_from_allocated_uninitialized() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {}
    stub_client.list_monitoring_runs_with_tag.return_value = ()

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    assert gateway.get_timeline_state("churn_model") is None
    with pytest.raises(
        GatewayConsistencyViolation,
        match="Timeline state for subject_id 'churn_model' not found.",
    ):
        gateway.pin_timeline_baseline("churn_model", "train-run-1")
    stub_client.set_monitoring_experiment_tag.assert_not_called()

    stub_client.list_monitoring_runs_with_tag.return_value = (
        _allocation_snapshot(
            run_id="monitoring-run-1",
            source_run_id="train-run-1",
            sequence_index=0,
        ),
    )

    timeline_state = gateway.get_timeline_state("churn_model")

    assert timeline_state is not None
    assert timeline_state.timeline_id == "experiment-1"
    assert timeline_state.baseline_source_run_id is None


def test_create_or_reuse_monitoring_run_reuses_older_recipe_without_rewriting_cache() -> None:
    v0 = _allocation_snapshot(
        run_id="monitoring-run-v0",
        source_run_id="train-run-1",
        sequence_index=0,
        recipe_version="v0",
    )
    v1 = _allocation_snapshot(
        run_id="monitoring-run-v1",
        source_run_id="train-run-1",
        sequence_index=1,
        recipe_version="v1",
    )
    stub_client = MagicMock()
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.0": "monitoring-run-v0",
        "monitoring.run.1": "monitoring-run-v1",
        "monitoring.latest_run_id": "monitoring-run-v1",
        "monitoring.next_sequence_index": "2",
        "training.train-run-1.monitoring_run_id": "monitoring-run-v1",
    }
    stub_client.list_monitoring_runs_with_tag.return_value = (v1, v0)
    stub_client.get_run_tags.return_value = dict(v0.tags)

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    result = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-1",
            recipe_id="system_default",
            recipe_version="v0",
        )
    )

    assert result.monitoring_run_id == "monitoring-run-v0"
    assert result.sequence_index == 0
    assert result.allocated is False
    stub_client.set_monitoring_experiment_tag.assert_not_called()
    stub_client.create_monitoring_run.assert_not_called()


def test_create_or_reuse_monitoring_run_repairs_stale_latest_and_source_binding() -> None:
    v0 = _allocation_snapshot(
        run_id="monitoring-run-v0",
        source_run_id="train-run-1",
        sequence_index=0,
        recipe_version="v0",
    )
    v1 = _allocation_snapshot(
        run_id="monitoring-run-v1",
        source_run_id="train-run-1",
        sequence_index=1,
        recipe_version="v1",
    )
    stub_client = MagicMock()
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.0": "monitoring-run-v0",
        "monitoring.run.1": "monitoring-run-v1",
        "monitoring.latest_run_id": "monitoring-run-v0",
        "monitoring.next_sequence_index": "2",
        "training.train-run-1.monitoring_run_id": "monitoring-run-v0",
    }
    stub_client.list_monitoring_runs_with_tag.return_value = (v0, v1)
    stub_client.get_run_tags.return_value = dict(v0.tags)

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    result = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            subject_id="churn_model",
            source_run_id="train-run-1",
            recipe_id="system_default",
            recipe_version="v0",
        )
    )

    assert result.monitoring_run_id == "monitoring-run-v0"
    assert stub_client.set_monitoring_experiment_tag.call_args_list == [
        call("experiment-1", "monitoring.latest_run_id", "monitoring-run-v1"),
        call(
            "experiment-1",
            "training.train-run-1.monitoring_run_id",
            "monitoring-run-v1",
        ),
    ]
    stub_client.create_monitoring_run.assert_not_called()


@pytest.mark.parametrize(
    ("snapshots", "experiment_tags", "expected_reason"),
    [
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=0,
                ),
                _allocation_snapshot(
                    run_id="monitoring-run-2",
                    source_run_id="train-run-1",
                    sequence_index=1,
                ),
            ),
            {},
            "duplicate_identity",
            id="duplicate-identity",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=0,
                ),
                _allocation_snapshot(
                    run_id="monitoring-run-2",
                    source_run_id="train-run-2",
                    sequence_index=0,
                ),
            ),
            {},
            "duplicate_sequence",
            id="duplicate-sequence",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=-1,
                ),
            ),
            {},
            "invalid_allocation",
            id="negative-sequence",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index="not-an-int",
                ),
            ),
            {},
            "invalid_allocation",
            id="malformed-sequence",
        ),
        pytest.param(
            (
                MonitoringRunTagSnapshot(
                    run_id="monitoring-run-1",
                    tags={
                        "training.source_run_id": "train-run-1",
                        "monitoring.sequence_index": "0",
                        "monitoring.recipe_version": "v0",
                    },
                ),
            ),
            {},
            "invalid_allocation",
            id="missing-recipe-id",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=1,
                ),
            ),
            {},
            "sequence_gap",
            id="sequence-gap",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=0,
                ),
            ),
            {"monitoring.run.0": "unknown-run"},
            "timeline_conflict",
            id="timeline-conflict",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=0,
                ),
            ),
            {"monitoring.latest_run_id": "unknown-run"},
            "unknown_pointer",
            id="unknown-latest-pointer",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=0,
                ),
            ),
            {
                "monitoring.next_sequence_index": "0",
                "training.train-run-1.monitoring_run_id": "unknown-run",
            },
            "unknown_pointer",
            id="repairable-next-plus-unknown-pointer",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=0,
                ),
            ),
            {"training.train-run-2.monitoring_run_id": "monitoring-run-1"},
            "source_binding_conflict",
            id="wrong-source-binding",
        ),
        pytest.param(
            (
                _allocation_snapshot(
                    run_id="monitoring-run-1",
                    source_run_id="train-run-1",
                    sequence_index=0,
                ),
            ),
            {"monitoring.next_sequence_index": "2"},
            "next_sequence_ahead",
            id="next-sequence-ahead",
        ),
    ],
)
def test_create_or_reuse_monitoring_run_fails_closed_before_writes(
    snapshots: tuple[MonitoringRunTagSnapshot, ...],
    experiment_tags: dict[str, str],
    expected_reason: str,
) -> None:
    stub_client = MagicMock()
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = experiment_tags
    stub_client.list_monitoring_runs_with_tag.return_value = snapshots

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        gateway.create_or_reuse_monitoring_run(
            IdempotencyKey(
                subject_id="churn_model",
                source_run_id="train-run-new",
                recipe_id="system_default",
                recipe_version="v0",
            )
        )

    assert exc_info.value.code == _MONITORING_ALLOCATION_INCONSISTENT_FOR_TESTS
    assert ("reason", expected_reason) in exc_info.value.details
    stub_client.set_monitoring_experiment_tag.assert_not_called()
    stub_client.create_monitoring_run.assert_not_called()


def test_create_or_reuse_monitoring_run_writes_idempotency_tag_last() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_tags.return_value = {}
    stub_client.get_or_create_monitoring_experiment.return_value = "experiment-1"
    stub_client.create_monitoring_run.return_value = MonitoringRunInfo(
        run_id="monitoring-run-1", run_name="monitoring-run-1"
    )

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


def test_write_monitoring_run_json_artifact_rejects_run_outside_monitoring_namespace() -> None:
    stub_client = MagicMock()
    stub_client.get_run_experiment_name.return_value = "training/churn"

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayNamespaceViolation, match="monitoring namespace"):
        gateway.write_monitoring_run_json_artifact(
            monitoring_run_id="train-run-1",
            data={"value": 1},
            path="state/prepared_context.json",
        )

    stub_client.get_run_tags.assert_not_called()
    stub_client.read_monitoring_run_json_artifact.assert_not_called()
    stub_client.log_monitoring_run_json_artifact.assert_not_called()


@pytest.mark.parametrize(
    "missing_tag",
    [
        "training.source_run_id",
        "monitoring.sequence_index",
        "monitoring.lifecycle_status",
        "monitoring.recipe_id",
        "monitoring.recipe_version",
    ],
)
def test_write_monitoring_run_json_artifact_rejects_missing_allocation_tag(
    missing_tag: str,
) -> None:
    stub_client = MagicMock()
    stub_client.get_run_experiment_name.return_value = "mlflow_monitor/churn_model"
    stub_client.get_run_tags.return_value = {
        "training.source_run_id": "train-run-1",
        "monitoring.sequence_index": "0",
        "monitoring.lifecycle_status": "created",
        "monitoring.recipe_id": "system_default",
        "monitoring.recipe_version": "v0",
    }
    del stub_client.get_run_tags.return_value[missing_tag]

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        gateway.write_monitoring_run_json_artifact(
            monitoring_run_id="monitoring-run-1",
            data={"value": 1},
            path="state/prepared_context.json",
        )

    assert exc_info.value.code == _MONITORING_ALLOCATION_INCONSISTENT_FOR_TESTS
    assert exc_info.value.details == (
        ("reason", "invalid_allocation"),
        ("monitoring_run_id", "monitoring-run-1"),
        ("missing_tags", missing_tag),
    )
    stub_client.read_monitoring_run_json_artifact.assert_not_called()
    stub_client.log_monitoring_run_json_artifact.assert_not_called()


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
    ("lifecycle_status", "expected_mlflow_status"),
    [
        (LifecycleStatus.CHECKED, "FINISHED"),
        (LifecycleStatus.FAILED, "FAILED"),
    ],
)
@pytest.mark.parametrize(
    (
        "artifact_paths",
        "current_mlflow_status",
        "should_log_artifact",
        "should_terminate",
    ),
    [
        (["outputs/result.json"], "EXPECTED", False, False),
        ([], "EXPECTED", True, False),
        (["outputs/result.json"], "RUNNING", False, True),
        ([], "RUNNING", True, True),
    ],
)
def test_finalize_monitoring_run_result_repairs_only_missing_terminal_side_effects(
    lifecycle_status: LifecycleStatus,
    expected_mlflow_status: str,
    artifact_paths: list[str],
    current_mlflow_status: str,
    should_log_artifact: bool,
    should_terminate: bool,
) -> None:
    stub_client = MagicMock()
    stub_client.list_artifact_paths.return_value = artifact_paths
    stub_client.get_run.return_value = _make_mlflow_run(
        expected_mlflow_status if current_mlflow_status == "EXPECTED" else current_mlflow_status
    )

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

    if should_log_artifact:
        stub_client.log_monitoring_run_json_artifact.assert_called_once_with(
            "monitoring-run-1",
            result.to_dict(),
            "outputs/result.json",
        )
    else:
        stub_client.log_monitoring_run_json_artifact.assert_not_called()

    if should_terminate:
        stub_client.terminate_monitoring_run.assert_called_once_with(
            "monitoring-run-1",
            expected_mlflow_status,
        )
    else:
        stub_client.terminate_monitoring_run.assert_not_called()


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


def test_get_monitoring_run_hydrates_source_and_reference_pairs() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.0": "monitoring-run-previous",
        "monitoring.run.1": "monitoring-run-current",
    }

    def get_run_tags(monitoring_run_id: str) -> dict[str, str]:
        if monitoring_run_id == "monitoring-run-current":
            return {
                "training.source_run_id": "train-run-current",
                "monitoring.sequence_index": "1",
                "monitoring.lifecycle_status": "checked",
                "monitoring.reference.baseline": "train-run-baseline",
                "monitoring.reference.previous": "monitoring-run-previous",
            }
        return {"training.source_run_id": "train-run-previous"}

    stub_client.get_run_tags.side_effect = get_run_tags

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    record = gateway.get_monitoring_run("churn_model", "monitoring-run-current")

    assert record is not None
    assert record.source_run_id == "train-run-current"
    assert record.references == (
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id="train-run-baseline",
        ),
        MonitoringRunReference(
            kind=DiffReferenceKind.PREVIOUS,
            monitoring_run_id="monitoring-run-previous",
            source_run_id="train-run-previous",
        ),
    )


def test_upsert_monitoring_run_rejects_conflicting_reference_pair() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.1": "monitoring-run-current",
    }

    def get_run_tags(monitoring_run_id: str) -> dict[str, str]:
        if monitoring_run_id == "monitoring-run-current":
            return {"training.source_run_id": "train-run-current"}
        return {"training.source_run_id": "train-run-persisted-reference"}

    stub_client.get_run_tags.side_effect = get_run_tags

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-current",
            source_run_id="train-run-current",
            lifecycle_status=LifecycleStatus.CHECKED,
            sequence_index=1,
            references=(
                MonitoringRunReference(
                    kind=DiffReferenceKind.PREVIOUS,
                    monitoring_run_id="monitoring-run-previous",
                    source_run_id="train-run-claimed-reference",
                ),
            ),
        )

    assert exc_info.value.code == "monitoring_run_upsert_field_override"
    assert dict(exc_info.value.details) == {
        "monitoring_run_id": "monitoring-run-previous",
        "source_run_id": "train-run-claimed-reference",
        "persisted_source_run_id": "train-run-persisted-reference",
    }


def test_upsert_monitoring_run_rejects_conflicting_primary_pair() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-1"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.0": "monitoring-run-1",
    }
    stub_client.get_run_tags.return_value = {
        "training.source_run_id": "train-run-persisted",
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-1",
            source_run_id="train-run-claimed",
            lifecycle_status=LifecycleStatus.CREATED,
            sequence_index=0,
        )

    assert exc_info.value.code == "monitoring_run_upsert_field_override"
    assert dict(exc_info.value.details) == {
        "monitoring_run_id": "monitoring-run-1",
        "source_run_id": "train-run-claimed",
        "persisted_source_run_id": "train-run-persisted",
    }


def test_upsert_monitoring_run_rejects_run_outside_subject_timeline_before_run_access() -> None:
    stub_client = MagicMock()
    stub_client.get_monitoring_experiment_id_by_name.return_value = "experiment-churn"
    stub_client.get_monitoring_experiment_tags.return_value = {
        "monitoring.run.0": "monitoring-run-churn",
    }
    stub_client.get_run_tags.return_value = {
        "training.source_run_id": "train-run-foreign",
    }

    with patch("mlflow_monitor.mlflow_gateway.MonitorMLflowClient", return_value=stub_client):
        gateway = MLflowMonitoringGateway(GatewayConfig())

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        gateway.upsert_monitoring_run(
            subject_id="churn_model",
            monitoring_run_id="monitoring-run-foreign",
            source_run_id="train-run-foreign",
            lifecycle_status=LifecycleStatus.CHECKED,
            sequence_index=0,
        )

    assert exc_info.value.code == "monitoring_run_subject_inconsistent"
    assert exc_info.value.details == (
        ("subject_id", "churn_model"),
        ("monitoring_run_id", "monitoring-run-foreign"),
    )
    stub_client.get_run_tags.assert_not_called()
    stub_client.set_monitoring_run_tags.assert_not_called()


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
            "training.source_run_id": "train-run-bad",
            "monitoring.sequence_index": "not-an-int",
            "monitoring.lifecycle_status": "checked",
        },
        {
            "training.source_run_id": "train-run-good",
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
