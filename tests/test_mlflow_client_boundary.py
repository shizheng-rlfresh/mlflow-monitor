"""Regression tests for the runtime MLflow client import boundary."""

from __future__ import annotations

from pathlib import Path


def test_runtime_modules_import_mlflow_client_only_via_adapter() -> None:
    runtime_root = Path(__file__).resolve().parents[1] / "src" / "mlflow_monitor"
    disallowed_imports: list[str] = []

    for path in runtime_root.rglob("*.py"):
        if path.name == "mlflow_client.py":
            continue
        content = path.read_text()
        if "from mlflow import MlflowClient" in content or "import mlflow" in content:
            disallowed_imports.append(str(path.relative_to(runtime_root.parent.parent)))

    assert disallowed_imports == []
