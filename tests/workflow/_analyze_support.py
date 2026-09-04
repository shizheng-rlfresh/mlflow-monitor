"""Small shared fixtures for Analyze execution and persistence specifications."""

from collections.abc import Sequence

from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckReason,
    ContractCheckResult,
    DiffReferenceKind,
    MonitoringRunReference,
)
from mlflow_monitor.gateway import GatewayConfig, InMemoryMonitoringGateway
from mlflow_monitor.recipe import build_system_default_recipe
from mlflow_monitor.recipe_compiler import CompiledRecipe, compile_recipe
from mlflow_monitor.workflow.prepared_context import PreparedContext, PreparedReferencePlanEntry


class MetricsGateway(InMemoryMonitoringGateway):
    """Return selected detached metrics and reject repeated source observations."""

    def __init__(self, metrics: dict[str, dict[str, float]]) -> None:
        super().__init__(GatewayConfig())
        self.metrics = metrics
        self.reads: list[tuple[str, tuple[str, ...] | None]] = []

    def get_source_run_metrics(
        self, source_run_id: str, metric_names: Sequence[str] | None = None
    ) -> dict[str, float] | None:
        assert source_run_id not in [source for source, _ in self.reads]
        self.reads.append((source_run_id, None if metric_names is None else tuple(metric_names)))
        values = self.metrics.get(source_run_id)
        if values is None:
            return None
        names = sorted(values) if metric_names is None else metric_names
        return {name: values[name] for name in names if name in values}


def context_and_recipe(
    metric_names: list[str] | None = None,
) -> tuple[PreparedContext, CompiledRecipe]:
    raw = build_system_default_recipe()
    if metric_names is not None:
        raw["analysis"] = {"metric_names": metric_names}
    recipe = compile_recipe(raw)
    context = PreparedContext(
        monitoring_run_id="monitoring-current",
        source_run_id="current",
        subject_id="model",
        timeline_id="timeline-model",
        sequence_index=1,
        baseline_source_run_id="baseline",
        effective_recipe=recipe.effective_plan,
        contract=recipe.contract,
        reference_plan=(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.BASELINE,
                reference=MonitoringRunReference(DiffReferenceKind.BASELINE, None, "baseline"),
                unavailable_reason=None,
            ),
            PreparedReferencePlanEntry(
                DiffReferenceKind.PREVIOUS, None, "previous_reference_missing"
            ),
            PreparedReferencePlanEntry(DiffReferenceKind.LKG, None, "lkg_not_selected"),
        ),
    )
    return context, recipe


def check_result(status: ComparabilityStatus = ComparabilityStatus.PASS) -> ContractCheckResult:
    if status is ComparabilityStatus.PASS:
        return ContractCheckResult(status, ())
    failed = status is ComparabilityStatus.FAIL
    return ContractCheckResult(
        status,
        (
            ContractCheckReason(
                "schema_mismatch" if failed else "environment_mismatch",
                "Data schema does not match the baseline."
                if failed
                else "Execution environment does not match the baseline.",
                failed,
            ),
        ),
    )
