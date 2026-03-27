"""Integration tests for the real-MLflow monitoring gateway."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mlflow import MlflowClient

from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus, MonitoringRunReference
from mlflow_monitor.gateway import GatewayConfig, IdempotencyKey
from mlflow_monitor.mlflow_gateway import MLflowMonitoringGateway
from mlflow_monitor.orchestration import run_orchestration


def test_mlflow_gateway_first_run_bootstraps_and_finalizes_result(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
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
    create_training_run: Callable[..., str],
) -> None:
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
    first_current_run_id = create_training_run(
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
    second_current_run_id = create_training_run(
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
    create_training_run: Callable[..., str],
) -> None:
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
    create_training_run: Callable[..., str],
) -> None:
    raw = MlflowClient(tracking_uri=tracking_uri)
    training_run_id = create_training_run(
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


def test_mlflow_gateway_resolve_source_run_id_rejects_monitoring_owned_runs(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
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

    assert (
        gateway.resolve_source_run_id(
            subject_id="churn_model",
            source_experiment=None,
            run_selector=result.monitoring_run_id,
        )
        is None
    )


def test_mlflow_gateway_rejects_monitoring_run_id_as_source_run_input(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
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
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    first_result = run_orchestration(
        subject_id="churn_model",
        source_run_id=current_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    invalid_result = run_orchestration(
        subject_id="churn_model",
        source_run_id=first_result.monitoring_run_id,
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    invalid_monitoring_run = raw.get_run(invalid_result.monitoring_run_id)
    payload_dir = Path(raw.download_artifacts(invalid_result.monitoring_run_id, "outputs"))
    payload = json.loads((payload_dir / "result.json").read_text())

    assert invalid_result.lifecycle_status is LifecycleStatus.FAILED
    assert invalid_result.error is not None
    assert invalid_result.error.code == "prepare_source_run_not_found"
    assert invalid_monitoring_run.info.status == "FAILED"
    assert payload["error"]["code"] == "prepare_source_run_not_found"


def test_mlflow_gateway_rejects_other_subject_monitoring_run_as_source_input(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
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
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    other_subject_result = run_orchestration(
        subject_id="fraud_model",
        source_run_id=current_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    invalid_result = run_orchestration(
        subject_id="churn_model",
        source_run_id=other_subject_result.monitoring_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    assert invalid_result.lifecycle_status is LifecycleStatus.FAILED
    assert invalid_result.error is not None
    assert invalid_result.error.code == "prepare_source_run_not_found"


def test_mlflow_gateway_create_or_reuse_allocates_new_run_for_recipe_version_change(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
    raw = MlflowClient(tracking_uri=tracking_uri)
    training_run_id = create_training_run(
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


def test_mlflow_gateway_warn_outcome_via_environment_mismatch(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
    snapshot_training_run: Callable[..., dict[str, Any]],
    assert_training_run_unchanged: Callable[..., None],
) -> None:
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
    baseline_snapshot = snapshot_training_run(raw=raw, run_id=baseline_run_id)
    current_snapshot = snapshot_training_run(raw=raw, run_id=current_run_id)
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

    monitoring_run = raw.get_run(result.monitoring_run_id)
    artifact_dir = Path(raw.download_artifacts(result.monitoring_run_id, "outputs"))
    payload = json.loads((artifact_dir / "result.json").read_text())

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert result.comparability_status is ComparabilityStatus.WARN
    assert monitoring_run.info.status == "FINISHED"
    assert monitoring_run.data.tags["monitoring.lifecycle_status"] == "checked"
    assert monitoring_run.data.tags["monitoring.comparability_status"] == "warn"
    assert payload["lifecycle_status"] == "checked"
    assert payload["comparability_status"] == "warn"

    assert_training_run_unchanged(
        raw=raw,
        run_id=baseline_run_id,
        snapshot=baseline_snapshot,
    )
    assert_training_run_unchanged(
        raw=raw,
        run_id=current_run_id,
        snapshot=current_snapshot,
    )


def test_mlflow_gateway_baseline_immutability_rejection(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
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
    competing_baseline_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="baseline-other",
        metrics={"f1": 0.86},
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
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    first = run_orchestration(
        subject_id="churn_model",
        source_run_id=current_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    experiment_before = raw.get_experiment_by_name("mlflow_monitor/churn_model")
    assert experiment_before is not None
    experiment_tags_before = dict(experiment_before.tags)
    monitoring_run_before = raw.get_run(first.monitoring_run_id)
    artifact_dir_before = Path(raw.download_artifacts(first.monitoring_run_id, "outputs"))
    payload_before = json.loads((artifact_dir_before / "result.json").read_text())

    second = run_orchestration(
        subject_id="churn_model",
        source_run_id=current_run_id,
        baseline_source_run_id=competing_baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    experiment_after = raw.get_experiment_by_name("mlflow_monitor/churn_model")
    assert experiment_after is not None
    monitoring_run_after = raw.get_run(first.monitoring_run_id)
    artifact_dir_after = Path(raw.download_artifacts(first.monitoring_run_id, "outputs"))
    payload_after = json.loads((artifact_dir_after / "result.json").read_text())

    assert second.monitoring_run_id == first.monitoring_run_id
    assert second.lifecycle_status is LifecycleStatus.FAILED
    assert second.comparability_status is None
    assert second.error is not None
    assert second.error.code == "prepare_baseline_override_existing_timeline"
    assert dict(experiment_after.tags) == experiment_tags_before
    assert experiment_after.tags["training.baseline_run_id"] == baseline_run_id
    assert monitoring_run_before.info.status == "FINISHED"
    assert monitoring_run_after.info.status == "FINISHED"
    assert monitoring_run_after.data.tags["monitoring.lifecycle_status"] == "checked"
    assert monitoring_run_after.data.tags["monitoring.comparability_status"] == "pass"
    assert payload_before == payload_after
    assert payload_after["lifecycle_status"] == "checked"
    assert payload_after["comparability_status"] == "pass"


def test_mlflow_gateway_list_timeline_monitoring_runs_ordered_traversal(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
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
    pass_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="pass",
        metrics={"f1": 0.91},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "schema.income": "float",
            "data_scope": "validation:2026-03-01",
        },
    )
    warn_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="warn",
        metrics={"f1": 0.89},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.11",
            "schema.age": "int",
            "schema.income": "float",
            "data_scope": "validation:2026-03-01",
        },
    )
    fail_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="fail",
        metrics={"f1": 0.88},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "schema.income": "float",
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
        source_run_id=pass_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    second = run_orchestration(
        subject_id="churn_model",
        source_run_id=warn_run_id,
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    third = run_orchestration(
        subject_id="churn_model",
        source_run_id=fail_run_id,
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    experiment = raw.get_experiment_by_name("mlflow_monitor/churn_model")
    assert experiment is not None
    records = gateway.list_timeline_monitoring_runs("churn_model")

    assert tuple(record.monitoring_run_id for record in records) == (
        first.monitoring_run_id,
        second.monitoring_run_id,
        third.monitoring_run_id,
    )
    assert tuple(record.sequence_index for record in records) == (0, 1, 2)
    assert experiment.tags["monitoring.run.0"] == records[0].monitoring_run_id
    assert experiment.tags["monitoring.run.1"] == records[1].monitoring_run_id
    assert experiment.tags["monitoring.run.2"] == records[2].monitoring_run_id


def test_mlflow_gateway_namespace_isolation_across_subjects(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
) -> None:
    raw = MlflowClient(tracking_uri=tracking_uri)
    alpha_baseline_run_id = create_training_run(
        raw=raw,
        experiment_name="training/alpha",
        artifact_root_uri=artifact_root_uri,
        run_name="alpha-baseline",
        metrics={"f1": 0.87},
        params={"feature_columns": "age"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "data_scope": "validation:2026-03-01",
        },
    )
    alpha_current_run_id = create_training_run(
        raw=raw,
        experiment_name="training/alpha",
        artifact_root_uri=artifact_root_uri,
        run_name="alpha-current",
        metrics={"f1": 0.91},
        params={"feature_columns": "age"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "data_scope": "validation:2026-03-01",
        },
    )
    beta_baseline_run_id = create_training_run(
        raw=raw,
        experiment_name="training/beta",
        artifact_root_uri=artifact_root_uri,
        run_name="beta-baseline",
        metrics={"f1": 0.82},
        params={"feature_columns": "income"},
        tags={
            "python_version": "3.12",
            "schema.income": "float",
            "data_scope": "validation:2026-03-01",
        },
    )
    beta_current_run_id = create_training_run(
        raw=raw,
        experiment_name="training/beta",
        artifact_root_uri=artifact_root_uri,
        run_name="beta-current",
        metrics={"f1": 0.83},
        params={"feature_columns": "income"},
        tags={
            "python_version": "3.12",
            "schema.income": "float",
            "data_scope": "validation:2026-03-01",
        },
    )
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    alpha = run_orchestration(
        subject_id="alpha_model",
        source_run_id=alpha_current_run_id,
        baseline_source_run_id=alpha_baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    beta = run_orchestration(
        subject_id="beta_model",
        source_run_id=beta_current_run_id,
        baseline_source_run_id=beta_baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )

    alpha_experiment = raw.get_experiment_by_name("mlflow_monitor/alpha_model")
    beta_experiment = raw.get_experiment_by_name("mlflow_monitor/beta_model")
    assert alpha_experiment is not None
    assert beta_experiment is not None

    alpha_records = gateway.list_timeline_monitoring_runs("alpha_model")
    beta_records = gateway.list_timeline_monitoring_runs("beta_model")

    assert alpha_experiment.experiment_id != beta_experiment.experiment_id
    assert alpha_experiment.tags["training.baseline_run_id"] == alpha_baseline_run_id
    assert beta_experiment.tags["training.baseline_run_id"] == beta_baseline_run_id
    assert alpha_experiment.tags["monitoring.latest_run_id"] == alpha.monitoring_run_id
    assert beta_experiment.tags["monitoring.latest_run_id"] == beta.monitoring_run_id
    assert alpha_experiment.tags["monitoring.next_sequence_index"] == "1"
    assert beta_experiment.tags["monitoring.next_sequence_index"] == "1"
    assert alpha_experiment.tags["monitoring.run.0"] == alpha.monitoring_run_id
    assert beta_experiment.tags["monitoring.run.0"] == beta.monitoring_run_id
    assert alpha.monitoring_run_id != beta.monitoring_run_id
    assert tuple(record.monitoring_run_id for record in alpha_records) == (alpha.monitoring_run_id,)
    assert tuple(record.monitoring_run_id for record in beta_records) == (beta.monitoring_run_id,)


def test_mlflow_gateway_training_runs_unchanged_across_pass_warn_fail(
    tracking_uri: str,
    artifact_root_uri: str,
    create_training_run: Callable[..., str],
    snapshot_training_run: Callable[..., dict[str, Any]],
    assert_training_run_unchanged: Callable[..., None],
) -> None:
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
    pass_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="pass",
        metrics={"f1": 0.91},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "schema.income": "float",
            "data_scope": "validation:2026-03-01",
        },
    )
    warn_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="warn",
        metrics={"f1": 0.89},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.11",
            "schema.age": "int",
            "schema.income": "float",
            "data_scope": "validation:2026-03-01",
        },
    )
    fail_run_id = create_training_run(
        raw=raw,
        experiment_name="training/churn",
        artifact_root_uri=artifact_root_uri,
        run_name="fail",
        metrics={"f1": 0.88},
        params={"feature_columns": "age,income"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "schema.income": "float",
            "data_scope": "validation:2026-03-02",
        },
    )
    run_snapshots = {
        run_id: snapshot_training_run(raw=raw, run_id=run_id)
        for run_id in (baseline_run_id, pass_run_id, warn_run_id, fail_run_id)
    }
    gateway = MLflowMonitoringGateway(
        GatewayConfig(),
        tracking_uri=tracking_uri,
        artifact_location=artifact_root_uri,
    )

    pass_result = run_orchestration(
        subject_id="churn_model",
        source_run_id=pass_run_id,
        baseline_source_run_id=baseline_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    assert pass_result.comparability_status is ComparabilityStatus.PASS
    for run_id, snapshot in run_snapshots.items():
        assert_training_run_unchanged(raw=raw, run_id=run_id, snapshot=snapshot)

    warn_result = run_orchestration(
        subject_id="churn_model",
        source_run_id=warn_run_id,
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    assert warn_result.comparability_status is ComparabilityStatus.WARN
    for run_id, snapshot in run_snapshots.items():
        assert_training_run_unchanged(raw=raw, run_id=run_id, snapshot=snapshot)

    fail_result = run_orchestration(
        subject_id="churn_model",
        source_run_id=fail_run_id,
        baseline_source_run_id=None,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
    assert fail_result.comparability_status is ComparabilityStatus.FAIL
    for run_id, snapshot in run_snapshots.items():
        assert_training_run_unchanged(raw=raw, run_id=run_id, snapshot=snapshot)
