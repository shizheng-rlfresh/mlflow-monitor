"""Public SDK facade for the M1 monitoring execution slice."""

from __future__ import annotations

from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.gateway import GatewayConfig, InMemoryMonitoringGateway
from mlflow_monitor.orchestration import run_orchestration
from mlflow_monitor.result_contract import MonitorRunResult

_DEFAULT_GATEWAY = InMemoryMonitoringGateway(GatewayConfig())


def run(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None = None,
) -> MonitorRunResult:
    """Execute a monitoring run for one monitored subject, source run, and baseline source run.

    Args:
        subject_id: The ID of the monitored subject this run is associated with.
        source_run_id: The original run ID from the training system that produced this run.
        baseline_source_run_id: baseline source run id for first run.

    Returns:
        The result of the monitoring run execution, including comparability status and any findings.
    """
    return run_orchestration(
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        gateway=_DEFAULT_GATEWAY,
        contract_checker=DefaultContractChecker(),
    )
