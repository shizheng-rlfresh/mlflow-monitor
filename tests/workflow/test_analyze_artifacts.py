"""Canonical Analyze output projection and cross-artifact consistency."""

from dataclasses import replace

import pytest

from mlflow_monitor.domain import ComparabilityStatus, MetricComparisonUnavailable
from mlflow_monitor.errors import GatewayConsistencyViolation
from mlflow_monitor.workflow.analyze import execute_analyze
from mlflow_monitor.workflow.analyze_artifacts import (
    ANALYZE_ARTIFACT_PATHS,
    COMPATIBILITY_EVIDENCE_ARTIFACT_PATH,
    DIFFS_ARTIFACT_PATH,
    FINDINGS_ARTIFACT_PATH,
    analyze_output_to_artifacts,
)
from workflow._analyze_support import MetricsGateway, check_result, context_and_recipe


def make_output(status=ComparabilityStatus.WARN, selection=None):
    context, recipe = context_and_recipe(selection)
    check = check_result(status)
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=check,
        compiled_recipe=recipe,
        gateway=MetricsGateway({"current": {"a": 2.0}, "baseline": {"a": 1.0}}),
    )
    return context, check, output


def project(context, check, output):
    return analyze_output_to_artifacts(
        output,
        prepared_context=context,
        contract_check_result=check,
    )


def test_analyze_projects_three_artifacts_in_dependency_order_with_inherited_identity() -> None:
    context, check, output = make_output()
    artifacts = project(context, check, output)
    assert (
        tuple(artifacts)
        == ANALYZE_ARTIFACT_PATHS
        == (
            "outputs/compatibility_evidence.json",
            "outputs/diffs.json",
            "outputs/findings.json",
        )
    )
    for payload in artifacts.values():
        assert payload["artifact_schema_version"] == "v0"
        assert payload["monitoring_run_id"] == context.monitoring_run_id
        assert payload["source_run_id"] == context.source_run_id
    evidence = artifacts[COMPATIBILITY_EVIDENCE_ARTIFACT_PATH]
    assert evidence["baseline_source_run_id"] == "baseline"
    assert evidence["contract_id"] == context.contract.contract_id
    assert evidence["contract_version"] == context.contract.contract_version
    assert evidence["evidence"] == [
        {
            "compatibility_evidence_id": output.compatibility_evidence[0].compatibility_evidence_id,
            "reason": {
                "code": "environment_mismatch",
                "message": check.reasons[0].message,
                "blocking": False,
            },
        }
    ]
    assert artifacts[DIFFS_ARTIFACT_PATH]["reference_groups"] == [
        {
            "reference_kind": "baseline",
            "reference": {
                "kind": "baseline",
                "monitoring_run_id": None,
                "source_run_id": "baseline",
            },
            "status": "completed",
            "reason": None,
            "diffs": [
                {
                    "diff_id": output.diffs[0].diff_id,
                    "metric_name": "a",
                    "current_value": 2.0,
                    "reference_value": 1.0,
                    "delta": 1.0,
                }
            ],
            "metric_unavailability": [],
        },
        {
            "reference_kind": "previous",
            "reference": None,
            "status": "unavailable",
            "reason": "previous_reference_missing",
            "diffs": [],
            "metric_unavailability": [],
        },
        {
            "reference_kind": "lkg",
            "reference": None,
            "status": "unavailable",
            "reason": "lkg_not_selected",
            "diffs": [],
            "metric_unavailability": [],
        },
    ]
    finding = output.findings[0]
    assert artifacts[FINDINGS_ARTIFACT_PATH]["findings"] == [
        {
            "finding_id": finding.finding_id,
            "finding_policy_id": finding.finding_policy_id,
            "finding_policy_version": finding.finding_policy_version,
            "finding_rule_id": finding.finding_rule_id,
            "severity": finding.severity.value,
            "category": finding.category,
            "summary": finding.summary,
            "recommendation": finding.recommendation,
            "evidence_diff_ids": [],
            "evidence_compatibility_ids": list(finding.evidence_compatibility_ids),
        }
    ]


@pytest.mark.parametrize("status", list(ComparabilityStatus))
def test_analyze_projects_empty_selection_and_fail_without_omitting_artifacts(status) -> None:
    context, check, output = make_output(status, [])
    artifacts = project(context, check, output)
    assert len(artifacts) == 3
    assert artifacts[DIFFS_ARTIFACT_PATH]["reference_groups"][0]["diffs"] == []
    if status is ComparabilityStatus.PASS:
        assert artifacts[COMPATIBILITY_EVIDENCE_ARTIFACT_PATH]["evidence"] == []
        assert artifacts[FINDINGS_ARTIFACT_PATH]["findings"] == []
    if status is ComparabilityStatus.FAIL:
        assert artifacts[DIFFS_ARTIFACT_PATH]["reference_groups"][0]["status"] == "skipped"


@pytest.mark.parametrize(
    "corruption",
    [
        "pair",
        "reason",
        "orphan",
        "duplicate",
        "missing_group",
        "metric_overlap",
        "missing_metric",
        "finding_pair",
        "finding_duplicate",
        "finding_policy",
    ],
)
def test_projection_rejects_inconsistent_output_before_any_artifact_can_be_written(
    corruption,
) -> None:
    context, check, output = make_output(selection=["a"])
    if corruption == "pair":
        output = replace(
            output,
            compatibility_evidence=(
                replace(output.compatibility_evidence[0], source_run_id="other"),
            ),
        )
    elif corruption == "reason":
        evidence = output.compatibility_evidence[0]
        output = replace(
            output,
            compatibility_evidence=(
                replace(evidence, reason=replace(evidence.reason, message="different")),
            ),
        )
    elif corruption == "orphan":
        output = replace(
            output,
            reference_comparison_coverage=(
                replace(output.reference_comparison_coverage[0], diff_ids=()),
                *output.reference_comparison_coverage[1:],
            ),
        )
    elif corruption == "duplicate":
        output = replace(output, diffs=output.diffs * 2)
    elif corruption == "missing_group":
        output = replace(
            output, reference_comparison_coverage=output.reference_comparison_coverage[:1]
        )
    elif corruption == "metric_overlap":
        output = replace(
            output,
            reference_comparison_coverage=(
                replace(
                    output.reference_comparison_coverage[0],
                    metric_unavailability=(
                        MetricComparisonUnavailable("a", "current_metric_missing"),
                    ),
                ),
                *output.reference_comparison_coverage[1:],
            ),
        )
    elif corruption == "missing_metric":
        output = replace(
            output,
            diffs=(),
            reference_comparison_coverage=(
                replace(output.reference_comparison_coverage[0], diff_ids=()),
                *output.reference_comparison_coverage[1:],
            ),
        )
    elif corruption == "finding_pair":
        output = replace(output, findings=(replace(output.findings[0], source_run_id="other"),))
    elif corruption == "finding_duplicate":
        output = replace(output, findings=output.findings * 2)
    else:
        output = replace(
            output, findings=(replace(output.findings[0], finding_policy_version="other"),)
        )
    with pytest.raises(GatewayConsistencyViolation) as caught:
        project(context, check, output)
    assert caught.value.code == "monitoring_run_json_artifact_inconsistent"
    assert dict(caught.value.details)["path"] in ANALYZE_ARTIFACT_PATHS
