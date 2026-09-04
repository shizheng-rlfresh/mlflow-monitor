"""Specifications for complete, backend-independent Analyze execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckReason,
    ContractCheckResult,
    DiffReferenceKind,
    MonitoringRunReference,
    ReferenceComparisonStatus,
)
from mlflow_monitor.errors import AnalyzeStageError
from mlflow_monitor.gateway import GatewayConfig, InMemoryMonitoringGateway
from mlflow_monitor.recipe import build_system_default_recipe
from mlflow_monitor.recipe_compiler import compile_recipe
from mlflow_monitor.workflow.analyze import execute_analyze
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


def context_and_recipe(metric_names: list[str] | None = None):
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


@pytest.mark.parametrize(
    ("selection", "expected_names", "expected_read"),
    [
        (None, ("a", "z"), None),
        ([], (), ()),
        (["z", "missing"], ("missing", "z"), ("missing", "z")),
    ],
)
def test_analyze_resolves_three_state_selection(selection, expected_names, expected_read) -> None:
    context, recipe = context_and_recipe(selection)
    gateway = MetricsGateway({"current": {"z": 3.0, "a": 2.0}, "baseline": {"z": 1.0, "a": 1.0}})
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=ContractCheckResult(ComparabilityStatus.PASS, ()),
        compiled_recipe=recipe,
        gateway=gateway,
    )
    group = output.reference_comparison_coverage[0]
    names = tuple(
        sorted(
            [diff.metric_name for diff in output.diffs]
            + [row.metric_name for row in group.metric_unavailability]
        )
    )
    assert names == expected_names
    assert group.status is ReferenceComparisonStatus.COMPLETED
    assert output.compatibility_evidence == output.findings == ()
    assert gateway.reads == [("current", expected_read), ("baseline", expected_names)]


@pytest.mark.parametrize("status", [ComparabilityStatus.WARN, ComparabilityStatus.FAIL])
def test_analyze_warn_reads_metrics_but_fail_skips_them_and_both_make_findings(status) -> None:
    context, recipe = context_and_recipe()
    failed = status is ComparabilityStatus.FAIL
    reason = ContractCheckReason(
        "schema_mismatch" if failed else "environment_mismatch",
        "Data schema does not match the baseline."
        if failed
        else "Execution environment does not match the baseline.",
        failed,
    )
    gateway = MetricsGateway({"current": {"a": 2.0}, "baseline": {"a": 1.0}})
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=ContractCheckResult(status, (reason,)),
        compiled_recipe=recipe,
        gateway=gateway,
    )
    assert len(output.compatibility_evidence) == len(output.findings) == 1
    assert output.compatibility_evidence[0].reason == reason
    assert output.findings[0].evidence_compatibility_ids == (
        output.compatibility_evidence[0].compatibility_evidence_id,
    )
    group = output.reference_comparison_coverage[0]
    assert group.status is (
        ReferenceComparisonStatus.SKIPPED if failed else ReferenceComparisonStatus.COMPLETED
    )
    assert group.reason == ("current_not_comparable" if failed else None)
    assert len(output.diffs) == (0 if failed else 1)
    assert bool(gateway.reads) is not failed
    assert [group.reason for group in output.reference_comparison_coverage[1:]] == [
        "previous_reference_missing",
        "lkg_not_selected",
    ]


def test_analyze_reuses_one_snapshot_for_current_and_shared_references() -> None:
    context, recipe = context_and_recipe()
    context = replace(
        context,
        baseline_source_run_id="current",
        reference_plan=tuple(
            PreparedReferencePlanEntry(
                kind,
                MonitoringRunReference(
                    kind,
                    None if kind is DiffReferenceKind.BASELINE else f"monitoring-{kind}",
                    "current",
                ),
                None,
            )
            for kind in DiffReferenceKind
        ),
    )
    gateway = MetricsGateway({"current": {"a": 2.0}})
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=ContractCheckResult(ComparabilityStatus.PASS, ()),
        compiled_recipe=recipe,
        gateway=gateway,
    )
    assert gateway.reads == [("current", None)]
    assert len(output.diffs) == 4
    assert all(diff.delta == 0 for diff in output.diffs)


def test_analyze_distinguishes_missing_current_source_from_empty_metrics() -> None:
    context, recipe = context_and_recipe()
    with pytest.raises(AnalyzeStageError) as caught:
        execute_analyze(
            prepared_context=context,
            contract_check_result=ContractCheckResult(ComparabilityStatus.PASS, ()),
            compiled_recipe=recipe,
            gateway=MetricsGateway({}),
        )
    assert caught.value.code == "analyze_missing_current_source_run"
    assert caught.value.details == (("source_run_id", "current"),)
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=ContractCheckResult(ComparabilityStatus.PASS, ()),
        compiled_recipe=recipe,
        gateway=MetricsGateway({"current": {}, "baseline": {}}),
    )
    assert output.diffs == ()
    assert output.reference_comparison_coverage[0].status is ReferenceComparisonStatus.COMPLETED


def test_analyze_retains_a_resolved_reference_when_its_source_is_missing() -> None:
    context, recipe = context_and_recipe()
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=ContractCheckResult(ComparabilityStatus.PASS, ()),
        compiled_recipe=recipe,
        gateway=MetricsGateway({"current": {"a": 1.0}}),
    )
    group = output.reference_comparison_coverage[0]
    assert group.status is ReferenceComparisonStatus.UNAVAILABLE
    assert group.reason == "reference_source_run_missing"
    assert group.reference is not None and group.reference.source_run_id == "baseline"
