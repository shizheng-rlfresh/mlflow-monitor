"""Gateway module for MLflow-Monitor v0.

Gateway is where retrieval (read) and persistence (write) of monitoring data happens. It is the interface
between mlflow-monitor and the underlying storage system (e.g., MLflow Tracking Server, database, etc.).

1. `.protocol` defines the MonitoringGateway interface.
2. `.memory` provides an in-memory implementation of the MonitoringGateway interface,
    useful for testing and development, and keeping the development from drifting
    away from the interface.
3. `.mlflow` provides an implementation of the MonitoringGateway interface that uses MLflow Tracking Server
    as the underlying storage system. In v0, we store monitoring data as tags and logged artifacts, i.e., json
    files, in the MLflow metadata and artifact stores.
4. `.models` defines the data models used by the gateway, including MonitoringRunRecord, TimelineClaim, and TimelineState.
"""  # noqa: E501

from .memory import InMemoryMonitoringGateway
from .mlflow import MLflowMonitoringGateway
from .models import (
    CreateOrReuseMonitoringRunResult,
    GatewayConfig,
    IdempotencyKey,
    MonitoringRunRecord,
    TimelineClaim,
    TimelineState,
)
from .protocol import MonitoringGateway

__all__ = [
    "MonitoringGateway",
    "IdempotencyKey",
    "MonitoringRunRecord",
    "TimelineClaim",
    "TimelineState",
    "GatewayConfig",
    "CreateOrReuseMonitoringRunResult",
    "InMemoryMonitoringGateway",
    "MLflowMonitoringGateway",
]
