"""Regression tests for the runtime MLflow client import boundary."""

from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_modules_import_mlflow_client_only_via_adapter() -> None:
    runtime_root = Path(__file__).resolve().parents[1] / "src" / "mlflow_monitor"
    disallowed_imports: list[str] = []

    for path in runtime_root.rglob("*.py"):
        if path.name == "mlflow_client.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "mlflow" or alias.name.startswith("mlflow."):
                        disallowed_imports.append(str(path.relative_to(runtime_root.parent.parent)))
                        break
            elif isinstance(node, ast.ImportFrom):
                if node.module == "mlflow" or (
                    node.module is not None and node.module.startswith("mlflow.")
                ):
                    disallowed_imports.append(str(path.relative_to(runtime_root.parent.parent)))
                    break

    assert not disallowed_imports, (
        "Runtime modules must import MLflow only via the adapter. "
        f"Found direct imports in: {disallowed_imports}"
    )
