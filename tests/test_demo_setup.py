"""Tests for the repo-level fraud demo seeding workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from mlflow import MlflowClient


def _load_demo_setup_module() -> ModuleType:
    """Load the repo-level demo setup script as a module for tests."""
    return _load_module("demo_setup", "setup.py")


def _load_module(module_name: str, filename: str) -> ModuleType:
    """Load a repo-level demo script as a module for tests."""
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "demo" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Failed to load demo setup module from {module_path}")

    repo_root_text = str(repo_root)
    if repo_root_text not in sys.path:
        sys.path.insert(0, repo_root_text)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


DEMO_SETUP = _load_demo_setup_module()
DEMO_EXPERIMENT_NAME = DEMO_SETUP.DEMO_EXPERIMENT_NAME
SCENARIO_NAMES = DEMO_SETUP.SCENARIO_NAMES
seed_demo_training_runs = DEMO_SETUP.seed_demo_training_runs
DEMO_RUNNER = _load_module("demo_run_monitoring", "run_monitoring.py")
MONITORING_EXPERIMENT_NAME = DEMO_RUNNER.MONITORING_EXPERIMENT_NAME
run_demo_monitoring = DEMO_RUNNER.run_demo_monitoring


def _list_artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> list[str]:
    """Return all artifact paths for one run."""
    paths: list[str] = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            paths.extend(_list_artifact_paths(client, run_id, item.path))
        else:
            paths.append(item.path)
    return sorted(paths)


def test_seed_demo_training_runs_creates_four_terminal_training_runs(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    seeded = seed_demo_training_runs(tracking_uri=tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)

    assert experiment is not None
    assert len(seeded.training_runs) == 4
    assert tuple(run.scenario_name for run in seeded.training_runs) == SCENARIO_NAMES

    for seeded_run in seeded.training_runs:
        run = client.get_run(seeded_run.run_id)
        assert run.info.status == "FINISHED"


def test_seed_demo_training_runs_logs_expected_metrics_tags_and_artifacts(
    tmp_path: Path,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    seeded = seed_demo_training_runs(tracking_uri=tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    comparable_run = next(
        run for run in seeded.training_runs if run.scenario_name == "comparable_candidate"
    )
    warn_run = next(run for run in seeded.training_runs if run.scenario_name == "warning_candidate")
    fail_run = next(
        run for run in seeded.training_runs if run.scenario_name == "non_comparable_candidate"
    )

    comparable = client.get_run(comparable_run.run_id)
    warn = client.get_run(warn_run.run_id)
    fail = client.get_run(fail_run.run_id)

    assert {"accuracy", "auc", "f1", "precision", "recall"} <= set(comparable.data.metrics)
    assert comparable.data.params["feature_columns"]
    assert comparable.data.tags["data_scope"] == "transactions_2026_q1"
    assert comparable.data.tags["schema.transaction_amount"] == "float"
    assert comparable.data.tags["python_version"] == "3.12"

    assert warn.data.tags["python_version"] == "3.13"
    assert warn.data.tags["data_scope"] == comparable.data.tags["data_scope"]
    assert warn.data.params["feature_columns"] == comparable.data.params["feature_columns"]

    assert fail.data.tags["schema.transaction_amount"] == "int"
    assert fail.data.tags["data_scope"] == comparable.data.tags["data_scope"]

    artifact_paths = _list_artifact_paths(client, comparable_run.run_id)
    assert "data/sample_rows.json" in artifact_paths
    assert "data/summary.json" in artifact_paths
    assert any(path.startswith("model/") for path in artifact_paths)


def test_run_demo_monitoring_executes_pass_warn_and_fail_in_order(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    seed_demo_training_runs(tracking_uri=tracking_uri)

    results = run_demo_monitoring(tracking_uri=tracking_uri)

    assert tuple(result.scenario_name for result in results) == (
        "comparable_candidate",
        "warning_candidate",
        "non_comparable_candidate",
    )
    assert [result.lifecycle_status for result in results] == ["checked", "checked", "checked"]
    assert [result.comparability_status for result in results] == ["pass", "warn", "fail"]

    client = MlflowClient(tracking_uri=tracking_uri)
    monitoring_experiment = client.get_experiment_by_name(MONITORING_EXPERIMENT_NAME)

    assert monitoring_experiment is not None
    monitoring_runs = client.search_runs(
        experiment_ids=[monitoring_experiment.experiment_id],
        order_by=["attribute.start_time ASC"],
    )

    assert len(monitoring_runs) == 3
    assert [run.data.tags["monitoring.comparability_status"] for run in monitoring_runs] == [
        "pass",
        "warn",
        "fail",
    ]
    for run in monitoring_runs:
        artifact_paths = _list_artifact_paths(client, run.info.run_id)
        assert "outputs/result.json" in artifact_paths
