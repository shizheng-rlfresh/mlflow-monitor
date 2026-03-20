"""Public SDK facade for the M1 monitoring execution slice."""

from __future__ import annotations

from uuid import uuid4

from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.gateway import GatewayConfig, InMemoryMonitoringGateway
from mlflow_monitor.orchestration import _run_monitoring
from mlflow_monitor.result_contract import MonitorRunResult

_DEFAULT_GATEWAY = InMemoryMonitoringGateway(GatewayConfig())


def run(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None = None,
) -> MonitorRunResult:
    """Run the M1 create/prepare/check monitoring slice."""
    return _run_monitoring(
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        gateway=_DEFAULT_GATEWAY,
        contract_checker=DefaultContractChecker(),
        run_id_factory=_generate_run_id,
    )


def _generate_run_id() -> str:
    """Return a new opaque monitoring run identifier."""
    return f"run-{uuid4().hex}"
