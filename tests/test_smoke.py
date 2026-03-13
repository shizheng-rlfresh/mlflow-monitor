"""Smoke tests for mlflow_monitor package."""

import mlflow_monitor


def test_package_imports() -> None:
    """Test that mlflow_monitor package can be imported without errors."""
    assert mlflow_monitor is not None
