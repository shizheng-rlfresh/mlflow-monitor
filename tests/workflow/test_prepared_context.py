import pytest

from mlflow_monitor.domain import DiffReferenceKind, MonitoringRunReference
from mlflow_monitor.errors import GatewayConsistencyViolation
from mlflow_monitor.recipe_compiler import compile_recipe
from mlflow_monitor.workflow import (
    PreparedContext,
    PreparedReferencePlanEntry,
    hydrate_prepared_context,
    prepared_context_to_dict,
)

from ._support import make_prepared_context


def test_prepared_context_exposes_resolved_references_from_fixed_plan() -> None:
    """Prepared context should retain unavailable planned reference groups."""
    compiled_recipe = compile_recipe()
    baseline = MonitoringRunReference(
        kind=DiffReferenceKind.BASELINE,
        monitoring_run_id=None,
        source_run_id="train-run-baseline",
    )
    reference_plan = (
        PreparedReferencePlanEntry(
            kind=DiffReferenceKind.BASELINE,
            reference=baseline,
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
    )
    context = PreparedContext(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-current",
        subject_id="churn_model",
        timeline_id="timeline-churn_model",
        sequence_index=0,
        baseline_source_run_id="train-run-baseline",
        effective_recipe=compiled_recipe.effective_plan,
        contract=compiled_recipe.contract,
        reference_plan=reference_plan,
    )

    assert context.references == (baseline,)
    assert prepared_context_to_dict(context)["reference_plan"] == [
        {
            "kind": "baseline",
            "monitoring_run_id": None,
            "source_run_id": "train-run-baseline",
            "unavailable_reason": None,
        },
        {
            "kind": "previous",
            "monitoring_run_id": None,
            "source_run_id": None,
            "unavailable_reason": "previous_reference_missing",
        },
        {
            "kind": "lkg",
            "monitoring_run_id": None,
            "source_run_id": None,
            "unavailable_reason": "lkg_not_selected",
        },
    ]


def test_hydrate_prepared_context_accepts_inconsistent_lkg_plan_shape() -> None:
    """Hydration should retain nonfatal LKG inconsistency for V0-030 replay."""
    compiled_recipe = compile_recipe()
    context = PreparedContext(
        monitoring_run_id="monitoring-run-1",
        source_run_id="train-run-current",
        subject_id="churn_model",
        timeline_id="timeline-churn_model",
        sequence_index=3,
        baseline_source_run_id="train-run-baseline",
        effective_recipe=compiled_recipe.effective_plan,
        contract=compiled_recipe.contract,
        reference_plan=(
            PreparedReferencePlanEntry(
                kind=DiffReferenceKind.BASELINE,
                reference=MonitoringRunReference(
                    kind=DiffReferenceKind.BASELINE,
                    monitoring_run_id=None,
                    source_run_id="train-run-baseline",
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
                unavailable_reason="lkg_selection_inconsistent",
            ),
        ),
    )

    hydrated = hydrate_prepared_context(
        prepared_context_to_dict(context),
        compiled_recipe=compiled_recipe,
        monitoring_run_id=context.monitoring_run_id,
        source_run_id=context.source_run_id,
        subject_id=context.subject_id,
        timeline_id=context.timeline_id,
        sequence_index=context.sequence_index,
    )

    assert hydrated == context


def test_hydrate_prepared_context_rejects_unpaired_unavailable_monitoring_run_id() -> None:
    """Unavailable plan entries should not retain an orphan Monitoring Run ID."""
    compiled_recipe = compile_recipe()
    context = make_prepared_context(contract=compiled_recipe.contract)
    raw = prepared_context_to_dict(context)
    references = raw["reference_plan"]
    assert isinstance(references, list)
    previous_reference = references[1]
    assert isinstance(previous_reference, dict)
    previous_reference["monitoring_run_id"] = "monitoring-run-orphan"

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_prepared_context(
            raw,
            compiled_recipe=compiled_recipe,
            monitoring_run_id=context.monitoring_run_id,
            source_run_id=context.source_run_id,
            subject_id=context.subject_id,
            timeline_id=context.timeline_id,
            sequence_index=context.sequence_index,
        )

    assert exc_info.value.code == "prepared_context_inconsistent"


def test_hydrate_prepared_context_reports_reference_plan_for_baseline_mismatch() -> None:
    """Baseline mismatches should identify the persisted reference-plan field."""
    compiled_recipe = compile_recipe()
    context = make_prepared_context(contract=compiled_recipe.contract)
    raw = prepared_context_to_dict(context)
    reference_plan = raw["reference_plan"]
    assert isinstance(reference_plan, list)
    baseline_reference = reference_plan[0]
    assert isinstance(baseline_reference, dict)
    baseline_reference["source_run_id"] = "train-run-other"

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_prepared_context(
            raw,
            compiled_recipe=compiled_recipe,
            monitoring_run_id=context.monitoring_run_id,
            source_run_id=context.source_run_id,
            subject_id=context.subject_id,
            timeline_id=context.timeline_id,
            sequence_index=context.sequence_index,
        )

    assert exc_info.value.details == (
        ("reason", "baseline_reference_mismatch"),
        ("field", "reference_plan"),
    )


def test_hydrate_prepared_context_rejects_obsolete_prepared_context_schema() -> None:
    """Resolved-only prepared artifacts should fail closed without migration."""
    compiled_recipe = compile_recipe()
    context = make_prepared_context(contract=compiled_recipe.contract)
    raw = prepared_context_to_dict(context)
    del raw["reference_plan"]
    raw["references"] = [reference.to_dict() for reference in context.references]

    with pytest.raises(GatewayConsistencyViolation) as exc_info:
        hydrate_prepared_context(
            raw,
            compiled_recipe=compiled_recipe,
            monitoring_run_id=context.monitoring_run_id,
            source_run_id=context.source_run_id,
            subject_id=context.subject_id,
            timeline_id=context.timeline_id,
            sequence_index=context.sequence_index,
        )

    assert exc_info.value.code == "prepared_context_inconsistent"
