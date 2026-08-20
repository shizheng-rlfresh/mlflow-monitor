"""Public Python API facade for the current monitoring workflow."""

from __future__ import annotations

from mlflow_monitor.contract_checker import DefaultContractChecker
from mlflow_monitor.gateway import MonitoringGateway
from mlflow_monitor.gateway_models import GatewayConfig
from mlflow_monitor.mlflow_gateway import MLflowMonitoringGateway
from mlflow_monitor.orchestration import run_orchestration
from mlflow_monitor.recipe_compiler import CompiledRecipe
from mlflow_monitor.result_contract import MonitorRunResult


def run(
    *,
    subject_id: str,
    source_run_id: str,
    baseline_source_run_id: str | None = None,
    custom_reference_monitoring_run_id: str | None = None,
    recipe: CompiledRecipe | None = None,
    gateway: MonitoringGateway | None = None,
) -> MonitorRunResult:
    """Execute monitoring for one Source Training Run.

    Args:
        subject_id: Monitored subject identifier.
        source_run_id: Source Training Run identifier to monitor.
        baseline_source_run_id: Optional Baseline Source Run identifier used to
            bootstrap or confirm the subject's immutable baseline.
        custom_reference_monitoring_run_id: Optional invocation-owned Reference
            Monitoring Run identifier.
        recipe: Precompiled Recipe, or ``None`` for the system default.
        gateway: Optional persistence gateway override.

    Returns:
        The current synchronous monitoring result.

    Raises:
        TypeError: If ``recipe`` is neither a ``CompiledRecipe`` nor ``None``.
    """
    if gateway is None:
        gateway = MLflowMonitoringGateway(GatewayConfig())

    return run_orchestration(
        subject_id=subject_id,
        source_run_id=source_run_id,
        baseline_source_run_id=baseline_source_run_id,
        custom_reference_monitoring_run_id=custom_reference_monitoring_run_id,
        recipe=recipe,
        gateway=gateway,
        contract_checker=DefaultContractChecker(),
    )
