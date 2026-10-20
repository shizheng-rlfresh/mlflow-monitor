"""Both gateways reject stale Analyze completion after terminal advancement."""

from unittest.mock import MagicMock, patch

import pytest

from mlflow_monitor.domain import LifecycleStatus
from mlflow_monitor.errors import GatewayConsistencyViolation
from mlflow_monitor.gateway import GatewayConfig, InMemoryMonitoringGateway, MLflowMonitoringGateway


@pytest.mark.parametrize("backend", ["memory", "mlflow"])
@pytest.mark.parametrize(
    "persisted",
    [
        LifecycleStatus.CHECKED,
        LifecycleStatus.ANALYZED,
        LifecycleStatus.CLOSED,
        LifecycleStatus.FAILED,
    ],
)
def test_analyzed_upsert_cannot_regress_terminal_state(backend, persisted):
    kwargs = dict(
        subject_id="model", monitoring_run_id="monitoring", source_run_id="source", sequence_index=0
    )
    if backend == "memory":
        gateway = InMemoryMonitoringGateway(GatewayConfig())
        gateway.upsert_monitoring_run(**kwargs, lifecycle_status=persisted)
    else:
        client = MagicMock()
        client.get_run_tags.return_value = {
            "training.source_run_id": "source",
            "monitoring.lifecycle_status": persisted.value,
        }
        with patch("mlflow_monitor.gateway.mlflow.MonitorMLflowClient", return_value=client):
            gateway = MLflowMonitoringGateway(GatewayConfig())
        gateway.resolve_timeline_monitoring_run_id = MagicMock(return_value="monitoring")
    if persisted in {LifecycleStatus.CLOSED, LifecycleStatus.FAILED}:
        with pytest.raises(GatewayConsistencyViolation):
            gateway.upsert_monitoring_run(**kwargs, lifecycle_status=LifecycleStatus.ANALYZED)
        if backend == "memory":
            assert gateway.get_monitoring_run("model", "monitoring").lifecycle_status is persisted
        else:
            client.set_monitoring_run_tags.assert_not_called()
    else:
        gateway.upsert_monitoring_run(**kwargs, lifecycle_status=LifecycleStatus.ANALYZED)
