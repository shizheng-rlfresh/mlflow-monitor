"""Specifications for complete, backend-independent Analyze execution."""

from __future__ import annotations

import subprocess
import sys
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
from mlflow_monitor.errors import AnalyzeStageError, PreparedContextConsistencyViolation
from mlflow_monitor.workflow.analyze import execute_analyze
from mlflow_monitor.workflow.prepared_context import PreparedReferencePlanEntry
from workflow._analyze_support import MetricsGateway, context_and_recipe


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


def test_analyze_rejects_changed_effective_recipe_before_reading_sources() -> None:
    context, _ = context_and_recipe()
    _, different_recipe = context_and_recipe([])
    gateway = MetricsGateway({})
    with pytest.raises(PreparedContextConsistencyViolation):
        execute_analyze(
            prepared_context=context,
            contract_check_result=ContractCheckResult(ComparabilityStatus.PASS, ()),
            compiled_recipe=different_recipe,
            gateway=gateway,
        )
    assert gateway.reads == []


@pytest.mark.parametrize("first", ["differ", "compatibility", "workflow.analyze", "workflow"])
def test_analyze_modules_import_without_order_dependent_cycles(first: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"import mlflow_monitor.{first}; "
            "from mlflow_monitor.workflow.analyze import execute_analyze",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
