"""test mlflow client"""

from mlflow_monitor.mlflow_client import MonitorMLflowClient


def test_mlflow_client():
    client = MonitorMLflowClient()
    assert client is not None
