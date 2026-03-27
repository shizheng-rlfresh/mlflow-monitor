"""Tests for the repo-level fraud demo seeding workflow."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import mlflow
import pytest
from mlflow import MlflowClient


def _load_module(repo_root: Path, module_name: str, filename: str) -> ModuleType:
    """Load a repo-level demo script as a module for tests."""
    module_path = repo_root / "demo" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Failed to load demo setup module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def repo_root() -> Path:
    """Return the repository root for demo module loading."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture
def demo_setup_module(repo_root: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load `demo/setup.py` with fixture-scoped import state."""
    monkeypatch.syspath_prepend(str(repo_root))
    module = _load_module(repo_root, "demo_setup", "setup.py")
    monkeypatch.setitem(sys.modules, "demo_setup", module)
    monkeypatch.setitem(sys.modules, "setup", module)
    return module


@pytest.fixture
def demo_runner_module(
    repo_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    demo_setup_module: ModuleType,
) -> ModuleType:
    """Load `demo/run_monitoring.py` with fixture-scoped import state."""
    monkeypatch.syspath_prepend(str(repo_root))
    monkeypatch.setitem(sys.modules, "demo_setup", demo_setup_module)
    monkeypatch.setitem(sys.modules, "setup", demo_setup_module)
    module = _load_module(repo_root, "demo_run_monitoring", "run_monitoring.py")
    monkeypatch.setitem(sys.modules, "demo_run_monitoring", module)
    return module


def _list_artifact_paths(client: MlflowClient, run_id: str, path: str = "") -> list[str]:
    """Return all artifact paths for one run."""
    paths: list[str] = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            paths.extend(_list_artifact_paths(client, run_id, item.path))
        else:
            paths.append(item.path)
    return sorted(paths)


def test_seed_demo_training_runs_creates_four_terminal_training_runs(
    tmp_path: Path,
    demo_setup_module: ModuleType,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    seeded = demo_setup_module.seed_demo_training_runs(tracking_uri=tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(demo_setup_module.DEMO_EXPERIMENT_NAME)

    assert experiment is not None
    assert len(seeded.training_runs) == 4
    assert (
        tuple(run.scenario_name for run in seeded.training_runs) == demo_setup_module.SCENARIO_NAMES
    )

    for seeded_run in seeded.training_runs:
        run = client.get_run(seeded_run.run_id)
        assert run.info.status == "FINISHED"


def test_seed_demo_training_runs_restores_previous_tracking_uri(
    tmp_path: Path,
    demo_setup_module: ModuleType,
) -> None:
    previous_tracking_uri = "sqlite:///:memory:"
    mlflow.set_tracking_uri(previous_tracking_uri)

    demo_setup_module.seed_demo_training_runs(tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}")

    assert mlflow.get_tracking_uri() == previous_tracking_uri


def test_seed_demo_training_runs_logs_expected_metrics_tags_and_artifacts(
    tmp_path: Path,
    demo_setup_module: ModuleType,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    seeded = demo_setup_module.seed_demo_training_runs(tracking_uri=tracking_uri)

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
    assert "data/eval.csv" in artifact_paths
    assert "data/sample_rows.json" in artifact_paths
    assert "data/summary.json" in artifact_paths
    assert "data/train.csv" in artifact_paths
    assert any(path.startswith("model/") for path in artifact_paths)


def test_run_demo_monitoring_executes_pass_warn_and_fail_in_order(
    tmp_path: Path,
    demo_setup_module: ModuleType,
    demo_runner_module: ModuleType,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    demo_setup_module.seed_demo_training_runs(tracking_uri=tracking_uri)

    results = demo_runner_module.run_demo_monitoring(tracking_uri=tracking_uri)

    assert tuple(result.scenario_name for result in results) == (
        "comparable_candidate",
        "warning_candidate",
        "non_comparable_candidate",
    )
    assert [result.lifecycle_status for result in results] == ["checked", "checked", "checked"]
    assert [result.comparability_status for result in results] == ["pass", "warn", "fail"]

    client = MlflowClient(tracking_uri=tracking_uri)
    monitoring_experiment = client.get_experiment_by_name(
        demo_runner_module.MONITORING_EXPERIMENT_NAME
    )

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


def test_run_demo_monitoring_restores_previous_tracking_uri(
    tmp_path: Path,
    demo_setup_module: ModuleType,
    demo_runner_module: ModuleType,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    previous_tracking_uri = "sqlite:///:memory:"
    mlflow.set_tracking_uri(previous_tracking_uri)
    demo_setup_module.seed_demo_training_runs(tracking_uri=tracking_uri)

    demo_runner_module.run_demo_monitoring(tracking_uri=tracking_uri)

    assert mlflow.get_tracking_uri() == previous_tracking_uri


def test_run_demo_monitoring_uses_newest_seeded_runs_after_many_reseeds(
    tmp_path: Path,
    demo_setup_module: ModuleType,
    demo_runner_module: ModuleType,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    latest_seed = None
    for _ in range(26):
        latest_seed = demo_setup_module.seed_demo_training_runs(tracking_uri=tracking_uri)

    assert latest_seed is not None
    latest_run_id_by_scenario = {run.scenario_name: run.run_id for run in latest_seed.training_runs}

    results = demo_runner_module.run_demo_monitoring(tracking_uri=tracking_uri)

    assert [result.source_run_id for result in results] == [
        latest_run_id_by_scenario["comparable_candidate"],
        latest_run_id_by_scenario["warning_candidate"],
        latest_run_id_by_scenario["non_comparable_candidate"],
    ]


def test_run_demo_monitoring_ignores_newer_incomplete_runs_with_matching_names(
    tmp_path: Path,
    demo_setup_module: ModuleType,
    demo_runner_module: ModuleType,
) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    seeded = demo_setup_module.seed_demo_training_runs(tracking_uri=tracking_uri)
    latest_run_id_by_scenario = {run.scenario_name: run.run_id for run in seeded.training_runs}

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(demo_setup_module.DEMO_EXPERIMENT_NAME)

    assert experiment is not None
    interrupted_run = client.create_run(
        experiment_id=experiment.experiment_id,
        tags={"mlflow.runName": demo_setup_module.SCENARIO_RUN_NAMES["comparable_candidate"]},
    )
    client.set_terminated(interrupted_run.info.run_id, status="FAILED")

    results = demo_runner_module.run_demo_monitoring(tracking_uri=tracking_uri)

    assert [result.source_run_id for result in results] == [
        latest_run_id_by_scenario["comparable_candidate"],
        latest_run_id_by_scenario["warning_candidate"],
        latest_run_id_by_scenario["non_comparable_candidate"],
    ]


def test_seed_demo_training_runs_uses_effective_tracking_uri_for_artifacts(
    tmp_path: Path,
    monkeypatch,
    demo_setup_module: ModuleType,
) -> None:
    demo_root = tmp_path / ".mlflow-dev"
    tracking_uri = f"sqlite:///{demo_root / 'mlflow.db'}"
    previous_tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    try:
        demo_setup_module.seed_demo_training_runs()

        client = MlflowClient(tracking_uri=tracking_uri)
        training_experiment = client.get_experiment_by_name(demo_setup_module.DEMO_EXPERIMENT_NAME)

        assert training_experiment is not None
        assert training_experiment.artifact_location == (demo_root / "artifacts").resolve().as_uri()
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)


def test_run_demo_monitoring_uses_effective_tracking_uri_for_monitoring_store(
    tmp_path: Path,
    monkeypatch,
    demo_setup_module: ModuleType,
    demo_runner_module: ModuleType,
) -> None:
    demo_root = tmp_path / ".mlflow-dev"
    tracking_uri = f"sqlite:///{demo_root / 'mlflow.db'}"
    previous_tracking_uri = mlflow.get_tracking_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)

    try:
        demo_setup_module.seed_demo_training_runs()
        results = demo_runner_module.run_demo_monitoring()

        client = MlflowClient(tracking_uri=tracking_uri)
        monitoring_experiment = client.get_experiment_by_name(
            demo_runner_module.MONITORING_EXPERIMENT_NAME
        )

        assert len(results) == 3
        assert monitoring_experiment is not None
        assert (
            monitoring_experiment.artifact_location == (demo_root / "artifacts").resolve().as_uri()
        )
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)
