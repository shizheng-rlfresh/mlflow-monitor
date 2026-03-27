"""Public SDK facade for the current monitoring workflow."""

from __future__ import annotations

from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.gateway import GatewayConfig, MonitoringGateway
from mlflow_monitor.mlflow_gateway import MLflowMonitoringGateway
from mlflow_monitor.orchestration import run_orchestration
from mlflow_monitor.result_contract import MonitorRunResult


def run(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None = None,
    gateway: MonitoringGateway | None = None,
) -> MonitorRunResult:
    """Execute a monitoring run for one monitored subject, source run, and baseline source run.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        source_run_id: The original run ID from the training system that produced this run.
        baseline_source_run_id: baseline source run id for first run.
        gateway: The monitoring gateway to use for persistence during orchestration.

    Returns:
        The result of the monitoring run execution, including comparability status and any findings.
    """
    if gateway is None:
        gateway = MLflowMonitoringGateway(GatewayConfig())

    return run_orchestration(
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
