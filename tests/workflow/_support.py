from mlflow_monitor.contract import (
    SYSTEM_DEFAULT_CONTRACT_ID,
    resolve_contract_v0,
)
from mlflow_monitor.domain import (
    Contract,
    DiffReferenceKind,
    MonitoringRunReference,
)
from mlflow_monitor.recipe_compiler import compile_recipe
from mlflow_monitor.workflow import PreparedContext, PreparedReferencePlanEntry

_CONTRACT = resolve_contract_v0(SYSTEM_DEFAULT_CONTRACT_ID)


def make_prepared_context(
    *,
    contract: Contract,
    source_run_id: str = "train-run-123",
    baseline_source_run_id: str = "train-run-baseline",
) -> PreparedContext:
    """Build a prepared context aligned with the common workflow test subject."""
    compiled_recipe = compile_recipe()
    return PreparedContext(
        monitoring_run_id="monitoring-run-1",
        source_run_id=source_run_id,
        subject_id="churn_model",
        timeline_id="timeline-churn_model",
        sequence_index=0,
        baseline_source_run_id=baseline_source_run_id,
        effective_recipe=compiled_recipe.effective_plan,
        contract=contract,
        reference_plan=(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.BASELINE,
                reference=MonitoringRunReference(
                    kind=DiffReferenceKind.BASELINE,
                    monitoring_run_id=None,
                    source_run_id=baseline_source_run_id,
                ),
                unavailable_reason=None,
            ),
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.PREVIOUS,
                reference=None,
                unavailable_reason="previous_reference_missing",
            ),
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.LKG,
                reference=None,
                unavailable_reason="lkg_not_selected",
            ),
        ),
    )
