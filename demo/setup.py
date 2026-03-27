"""Seed a local fraud-model demo into MLflow."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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
    "baseline": "fraud-model-baseline-v1",
    "comparable_candidate": "fraud-model-candidate-v2",
    "warning_candidate": "fraud-model-env-shift-v3",
    "non_comparable_candidate": "fraud-model-schema-shift-v4",
}


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


def _build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic toy fraud-like classification data."""
    features, target = make_classification(
        n_samples=600,
        n_features=len(FEATURE_COLUMNS),
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.92, 0.08],
        class_sep=1.1,
        random_state=48,
    )
    return features, target


def _dataset_summary(
    *,
    features: np.ndarray,
    target: np.ndarray,
    split: dict[str, int],
) -> dict[str, Any]:
    """Return one small JSON summary artifact for the dataset."""
    positive_rate = float(target.mean())
    return {
        "rows": int(features.shape[0]),
        "columns": list(FEATURE_COLUMNS),
        "positive_rate": round(positive_rate, 4),
        "split": split,
    }


def _sample_rows(features: np.ndarray, target: np.ndarray) -> list[dict[str, float | int]]:
    """Return a few sample rows as JSON-friendly dictionaries."""
    rows: list[dict[str, float | int]] = []
    for feature_row, label in zip(features[:5], target[:5], strict=False):
        row = {
            feature_name: round(float(value), 4)
            for feature_name, value in zip(FEATURE_COLUMNS, feature_row, strict=True)
        }
        row["is_fraud"] = int(label)
        rows.append(row)
    return rows


def _scenario_configs() -> tuple[dict[str, Any], ...]:
    """Return the seeded run scenarios in execution order."""
    return (
        {
            "scenario_name": "baseline",
            "run_name": SCENARIO_RUN_NAMES["baseline"],
            "model_params": {"C": 1.0, "max_iter": 400, "random_state": 11},
            "python_version": "3.12",
            "sklearn_version": "1.7.1",
            "data_scope": "transactions_2026_q1",
            "schema_overrides": {},
        },
        {
            "scenario_name": "comparable_candidate",
            "run_name": SCENARIO_RUN_NAMES["comparable_candidate"],
            "model_params": {"C": 0.8, "max_iter": 400, "random_state": 22},
            "python_version": "3.12",
            "sklearn_version": "1.7.1",
            "data_scope": "transactions_2026_q1",
            "schema_overrides": {},
        },
        {
            "scenario_name": "warning_candidate",
            "run_name": SCENARIO_RUN_NAMES["warning_candidate"],
            "model_params": {"C": 1.2, "max_iter": 450, "random_state": 33},
            "python_version": "3.13",
            "sklearn_version": "1.8.0",
            "data_scope": "transactions_2026_q1",
            "schema_overrides": {},
        },
        {
            "scenario_name": "non_comparable_candidate",
            "run_name": SCENARIO_RUN_NAMES["non_comparable_candidate"],
            "model_params": {"C": 1.1, "max_iter": 450, "random_state": 44},
            "python_version": "3.12",
            "sklearn_version": "1.7.1",
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


def seed_demo_training_runs(tracking_uri: str | None = None) -> SeededDemo:
    """Train and log the fraud-model demo runs into MLflow."""
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

        features, target = _build_dataset()
        train_x, eval_x, train_y, eval_y = train_test_split(
            features,
            target,
            test_size=0.25,
            stratify=target,
            random_state=7,
        )
        summary = _dataset_summary(
            features=features,
            target=target,
            split={"train_rows": int(train_x.shape[0]), "eval_rows": int(eval_x.shape[0])},
        )
        sample_rows = _sample_rows(eval_x, eval_y)

        seeded_runs: list[SeededTrainingRun] = []
        for config in _scenario_configs():
            model = LogisticRegression(
                C=config["model_params"]["C"],
                max_iter=config["model_params"]["max_iter"],
                random_state=config["model_params"]["random_state"],
                solver="liblinear",
            )
            model.fit(train_x, train_y)
            predicted = model.predict(eval_x)
            probabilities = model.predict_proba(eval_x)[:, 1]

            metrics = {
                "accuracy": round(float(accuracy_score(eval_y, predicted)), 4),
                "auc": round(float(roc_auc_score(eval_y, probabilities)), 4),
                "f1": round(float(f1_score(eval_y, predicted, zero_division=0)), 4),
                "precision": round(float(precision_score(eval_y, predicted, zero_division=0)), 4),
                "recall": round(float(recall_score(eval_y, predicted, zero_division=0)), 4),
            }
            schema = {feature_name: "float" for feature_name in FEATURE_COLUMNS}
            schema.update(config["schema_overrides"])

            with mlflow.start_run(experiment_id=experiment_id, run_name=config["run_name"]) as run:
                mlflow.log_params(
                    {
                        "model_type": "logistic_regression",
                        "feature_columns": ",".join(FEATURE_COLUMNS),
                        "C": config["model_params"]["C"],
                        "max_iter": config["model_params"]["max_iter"],
                        "random_state": config["model_params"]["random_state"],
                    }
                )
                mlflow.log_metrics(metrics)
                mlflow.set_tags(
                    {
                        "python_version": config["python_version"],
                        "sklearn_version": config["sklearn_version"],
                        "data_scope": config["data_scope"],
                    }
                )
                for key, value in schema.items():
                    mlflow.set_tag(f"schema.{key}", value)
                mlflow.log_dict(summary, "data/summary.json")
                mlflow.log_dict(sample_rows, "data/sample_rows.json")
                with tempfile.TemporaryDirectory() as tmp_dir:
                    model_dir = Path(tmp_dir) / "model"
                    mlflow.sklearn.save_model(
                        sk_model=model,
                        path=str(model_dir),
                        input_example=eval_x[:5],
                    )
                    mlflow.log_artifacts(str(model_dir), artifact_path="model")

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
    print("Next step:")
    print("uv run demo/run_monitoring.py")
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
