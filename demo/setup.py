"""Seed a local fraud-model demo into MLflow."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

DEMO_EXPERIMENT_NAME = "training/fraud_model"
DEMO_SUBJECT_ID = "fraud_model"
SCENARIO_NAMES = (
    "baseline",
    "comparable_candidate",
    "warning_candidate",
    "non_comparable_candidate",
)
FEATURE_COLUMNS = (
    "transaction_amount",
    "account_age_days",
    "merchant_risk_score",
    "device_velocity_24h",
    "country_match",
    "chargeback_rate_30d",
)
SCENARIO_RUN_NAMES = {
    "baseline": "fraud-model-v1",
    "comparable_candidate": "fraud-model-v2",
    "warning_candidate": "fraud-model-v3",
    "non_comparable_candidate": "fraud-model-v4",
}
ASSET_ROOT = Path(__file__).resolve().parent / "seed_assets"


@dataclass(frozen=True, slots=True)
class SeededTrainingRun:
    """One seeded training run for the fraud demo."""

    scenario_name: str
    run_name: str
    run_id: str


@dataclass(frozen=True, slots=True)
class SeededDemo:
    """Result of seeding the fraud-model training experiment."""

    experiment_name: str
    training_runs: tuple[SeededTrainingRun, ...]


def _scenario_configs() -> tuple[dict[str, Any], ...]:
    """Return the seeded run scenarios in execution order."""
    return (
        {
            "scenario_name": "baseline",
            "run_name": SCENARIO_RUN_NAMES["baseline"],
            "metrics": {
                "accuracy": 0.9521,
                "auc": 0.9784,
                "f1": 0.7619,
                "precision": 0.8,
                "recall": 0.7273,
            },
            "model_params": {
                "model_type": "pyfunc_fraud_score",
                "score_threshold": 0.55,
                "model_revision": "v1",
            },
            "environment_tags": {
                "python_version": "3.12",
                "model_runtime": "mlflow.pyfunc",
            },
            "data_scope": "transactions_2026_q1",
            "schema_overrides": {},
        },
        {
            "scenario_name": "comparable_candidate",
            "run_name": SCENARIO_RUN_NAMES["comparable_candidate"],
            "metrics": {
                "accuracy": 0.9564,
                "auc": 0.9812,
                "f1": 0.7826,
                "precision": 0.8182,
                "recall": 0.75,
            },
            "model_params": {
                "model_type": "pyfunc_fraud_score",
                "score_threshold": 0.56,
                "model_revision": "v2",
            },
            "environment_tags": {
                "python_version": "3.12",
                "model_runtime": "mlflow.pyfunc",
            },
            "data_scope": "transactions_2026_q1",
            "schema_overrides": {},
        },
        {
            "scenario_name": "warning_candidate",
            "run_name": SCENARIO_RUN_NAMES["warning_candidate"],
            "metrics": {
                "accuracy": 0.9543,
                "auc": 0.9798,
                "f1": 0.7727,
                "precision": 0.8095,
                "recall": 0.7391,
            },
            "model_params": {
                "model_type": "pyfunc_fraud_score",
                "score_threshold": 0.57,
                "model_revision": "v3",
            },
            "environment_tags": {
                "python_version": "3.13",
                "model_runtime": "mlflow.pyfunc",
            },
            "data_scope": "transactions_2026_q1",
            "schema_overrides": {},
        },
        {
            "scenario_name": "non_comparable_candidate",
            "run_name": SCENARIO_RUN_NAMES["non_comparable_candidate"],
            "metrics": {
                "accuracy": 0.9415,
                "auc": 0.9688,
                "f1": 0.72,
                "precision": 0.75,
                "recall": 0.6923,
            },
            "model_params": {
                "model_type": "pyfunc_fraud_score",
                "score_threshold": 0.54,
                "model_revision": "v4",
            },
            "environment_tags": {
                "python_version": "3.12",
                "model_runtime": "mlflow.pyfunc",
            },
            "data_scope": "transactions_2026_q1",
            "schema_overrides": {"transaction_amount": "int"},
        },
    )


def _artifact_location_for_tracking_uri(tracking_uri: str | None) -> str | None:
    """Return a local artifact root for SQLite-backed demo stores."""
    if tracking_uri is None or not tracking_uri.startswith("sqlite:///"):
        return None

    database_path = Path(tracking_uri.removeprefix("sqlite:///")).resolve()
    artifact_root = database_path.parent / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root.as_uri()


def resolve_effective_tracking_uri(tracking_uri: str | None = None) -> str | None:
    """Return the explicit or active MLflow tracking URI for the demo."""
    if tracking_uri is not None:
        return tracking_uri

    active_tracking_uri = mlflow.get_tracking_uri()
    return active_tracking_uri or None


def _asset_dir_for_scenario(scenario_name: str) -> Path:
    """Return the checked-in asset directory for one demo scenario."""
    return ASSET_ROOT / scenario_name


def _load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Load CSV rows as dictionaries."""
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _dataset_summary(
    train_rows: list[dict[str, str]],
    eval_rows: list[dict[str, str]],
) -> dict[str, Any]:
    """Return one small JSON summary artifact for the checked-in dataset."""
    all_rows = train_rows + eval_rows
    positive_labels = sum(int(row["is_fraud"]) for row in all_rows)
    return {
        "rows": len(all_rows),
        "columns": list(FEATURE_COLUMNS),
        "positive_rate": round(positive_labels / len(all_rows), 4),
        "split": {"train_rows": len(train_rows), "eval_rows": len(eval_rows)},
    }


def _sample_rows(eval_rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    """Return a few eval rows as JSON-friendly dictionaries."""
    sampled_rows: list[dict[str, float | int]] = []
    for raw_row in eval_rows[:5]:
        row: dict[str, float | int] = {}
        for feature_name in FEATURE_COLUMNS:
            value = raw_row[feature_name]
            if feature_name in {"account_age_days", "country_match", "device_velocity_24h"}:
                row[feature_name] = int(float(value))
            else:
                row[feature_name] = round(float(value), 4)
        row["is_fraud"] = int(raw_row["is_fraud"])
        sampled_rows.append(row)
    return sampled_rows


def seed_demo_training_runs(tracking_uri: str | None = None) -> SeededDemo:
    """Log the checked-in fraud-model demo runs into MLflow."""
    effective_tracking_uri = resolve_effective_tracking_uri(tracking_uri)
    previous_tracking_uri = mlflow.get_tracking_uri()

    try:
        if effective_tracking_uri is not None:
            mlflow.set_tracking_uri(effective_tracking_uri)

        client = MlflowClient(tracking_uri=effective_tracking_uri)
        experiment = client.get_experiment_by_name(DEMO_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = client.create_experiment(
                DEMO_EXPERIMENT_NAME,
                artifact_location=_artifact_location_for_tracking_uri(effective_tracking_uri),
            )
        else:
            experiment_id = experiment.experiment_id

        seeded_runs: list[SeededTrainingRun] = []
        for config in _scenario_configs():
            asset_dir = _asset_dir_for_scenario(config["scenario_name"])
            train_rows = _load_csv_rows(asset_dir / "train.csv")
            eval_rows = _load_csv_rows(asset_dir / "eval.csv")
            summary = _dataset_summary(train_rows, eval_rows)
            sample_rows = _sample_rows(eval_rows)
            schema = {feature_name: "float" for feature_name in FEATURE_COLUMNS}
            schema.update(config["schema_overrides"])

            with mlflow.start_run(experiment_id=experiment_id, run_name=config["run_name"]) as run:
                mlflow.log_params(
                    {
                        "feature_columns": ",".join(FEATURE_COLUMNS),
                        **config["model_params"],
                    }
                )
                mlflow.log_metrics(config["metrics"])
                mlflow.set_tags(
                    {
                        "data_scope": config["data_scope"],
                        **config["environment_tags"],
                    }
                )
                for key, value in schema.items():
                    mlflow.set_tag(f"schema.{key}", value)
                mlflow.log_artifact(str(asset_dir / "train.csv"), artifact_path="data")
                mlflow.log_artifact(str(asset_dir / "eval.csv"), artifact_path="data")
                mlflow.log_dict(summary, "data/summary.json")
                mlflow.log_dict(sample_rows, "data/sample_rows.json")
                mlflow.log_artifacts(str(asset_dir / "model"), artifact_path="model")

                seeded_runs.append(
                    SeededTrainingRun(
                        scenario_name=config["scenario_name"],
                        run_name=config["run_name"],
                        run_id=run.info.run_id,
                    )
                )

        return SeededDemo(
            experiment_name=DEMO_EXPERIMENT_NAME,
            training_runs=tuple(seeded_runs),
        )
    finally:
        mlflow.set_tracking_uri(previous_tracking_uri)


def _print_demo_instructions(seeded: SeededDemo) -> None:
    """Print next steps for the seeded demo."""
    print("Seeded training experiment:", seeded.experiment_name)
    for run in seeded.training_runs:
        print(f"- {run.scenario_name}: {run.run_name} -> {run.run_id}")
    print()
    print("Next step: run the monitoring step against the same tracking store.")
    print("Example:")
    print("MLFLOW_TRACKING_URI=sqlite:///./.mlflow-dev/mlflow.db uv run demo/run_monitoring.py")
    print()
    print("Expected monitoring outcomes:")
    print("- comparable_candidate -> pass")
    print("- warning_candidate -> warn")
    print("- non_comparable_candidate -> fail")


def main() -> None:
    """Seed the local MLflow fraud demo and print next steps."""
    seeded = seed_demo_training_runs()
    _print_demo_instructions(seeded)


if __name__ == "__main__":
    main()
