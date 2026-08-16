"""Integration tests for the public monitor Python API facade."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow import MlflowClient

from mlflow_monitor import monitor
from mlflow_monitor.domain import (
    ComparabilityStatus,
    DiffReferenceKind,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.recipe import build_system_default_recipe
from mlflow_monitor.recipe_compiler import compile_recipe


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
        MonitoringRunReference(
            kind=DiffReferenceKind.BASELINE,
            monitoring_run_id=None,
            source_run_id=baseline_run_id,
        ),
    )
    assert experiment.tags["monitoring.latest_run_id"] == result.monitoring_run_id

    monitoring_run = raw.get_run(result.monitoring_run_id)
    assert monitoring_run.info.status == "FINISHED"

    artifact_dir = Path(raw.download_artifacts(result.monitoring_run_id, "outputs"))
    payload = json.loads((artifact_dir / "result.json").read_text())
    assert payload["monitoring_run_id"] == result.monitoring_run_id
    assert payload["lifecycle_status"] == "checked"

    prepared_path = Path(
        raw.download_artifacts(
            result.monitoring_run_id,
            "state/prepared_context.json",
        )
    )
    prepared_payload = json.loads(prepared_path.read_text())
    compiled_recipe = compile_recipe()
    contract = compiled_recipe.contract

    assert prepared_payload == {
        "artifact_schema_version": "v0",
        "monitoring_run_id": result.monitoring_run_id,
        "source_run_id": current_run_id,
        "subject_id": "churn_model",
        "timeline_id": experiment.experiment_id,
        "sequence_index": 0,
        "baseline_source_run_id": baseline_run_id,
        "effective_recipe": compiled_recipe.effective_plan.to_dict(),
        "contract": {
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "schema_contract_ref": contract.schema_contract_ref,
            "feature_contract_ref": contract.feature_contract_ref,
            "metric_contract_ref": contract.metric_contract_ref,
            "data_scope_contract_ref": contract.data_scope_contract_ref,
            "execution_contract_ref": contract.execution_contract_ref,
        },
        "references": [
            {
                "kind": DiffReferenceKind.BASELINE,
                "monitoring_run_id": None,
                "source_run_id": baseline_run_id,
            }
        ],
    }


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


def test_monitor_run_persists_supplied_compiled_recipe_identity(
    tracking_uri: str,
    artifact_root_uri: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    create_training_run,
) -> None:
    monkeypatch.chdir(tmp_path)
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
        run_name="current-custom-recipe",
        metrics={"f1": 0.91},
        params={"feature_columns": "age"},
        tags={
            "python_version": "3.12",
            "schema.age": "int",
            "data_scope": "validation:2026-03-01",
        },
    )
    recipe = build_system_default_recipe()
    recipe["identity"] = {
        "recipe_id": "custom-churn",
        "recipe_version": "7",
    }
    compiled_recipe = compile_recipe(recipe)

    previous_tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    try:
        result = monitor.run(
            subject_id="churn_model",
            source_run_id=current_run_id,
            baseline_source_run_id=baseline_run_id,
            recipe=compiled_recipe,
        )
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)

    monitoring_run = raw.get_run(result.monitoring_run_id)

    assert result.lifecycle_status is LifecycleStatus.CHECKED
    assert monitoring_run.data.tags["monitoring.recipe_id"] == "custom-churn"
    assert monitoring_run.data.tags["monitoring.recipe_version"] == "7"
