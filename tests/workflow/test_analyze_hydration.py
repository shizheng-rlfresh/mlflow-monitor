"""Analyze replay validates saved representations without live dependencies."""

from copy import deepcopy

import pytest

from mlflow_monitor.domain import ComparabilityStatus
from mlflow_monitor.errors import GatewayConsistencyViolation
from mlflow_monitor.workflow.analyze import execute_analyze
from mlflow_monitor.workflow.analyze_artifacts import (
    ANALYZE_ARTIFACT_PATHS,
    analyze_output_to_artifacts,
)
from mlflow_monitor.workflow.analyze_hydration import (
    hydrate_analyze_output,
    validate_partial_analyze_artifacts,
)
from workflow._analyze_support import MetricsGateway, check_result, context_and_recipe


def saved_output(status=ComparabilityStatus.WARN, selection=None):
    context, recipe = context_and_recipe(selection)
    check = check_result(status)
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=check,
        compiled_recipe=recipe,
        gateway=MetricsGateway({"current": {"a": 2.0}, "baseline": {"a": 1.0}}),
    )
    artifacts = analyze_output_to_artifacts(
        output, prepared_context=context, contract_check_result=check
    )
    return context, check, output, artifacts


@pytest.mark.parametrize("status", list(ComparabilityStatus))
@pytest.mark.parametrize("selection", [None, [], ["a", "absent"]])
def test_saved_analyze_output_round_trips_without_gateway_or_policies(status, selection):
    context, check, output, artifacts = saved_output(status, selection)
    assert (
        hydrate_analyze_output(artifacts, prepared_context=context, contract_check_result=check)
        == output
    )


@pytest.mark.parametrize("path", ANALYZE_ARTIFACT_PATHS)
@pytest.mark.parametrize("field", ["artifact_schema_version", "monitoring_run_id", "source_run_id"])
def test_replay_rejects_foreign_envelopes_with_bounded_diagnostics(path, field):
    context, check, _, artifacts = saved_output()
    artifacts[path][field] = "sensitive-payload"
    with pytest.raises(GatewayConsistencyViolation) as caught:
        hydrate_analyze_output(artifacts, prepared_context=context, contract_check_result=check)
    assert dict(caught.value.details)["path"] == path
    assert "sensitive-payload" not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_artifact",
        "extra_field",
        "reason",
        "duplicate_evidence",
        "wrong_baseline",
        "missing_group",
        "foreign_reference",
        "duplicate_diff",
        "forged_diff_id",
        "nan",
        "boolean",
        "wrong_delta",
        "missing_metric",
        "unknown_unavailability",
        "duplicate_metric",
        "duplicate_finding",
        "forged_finding_id",
        "unknown_evidence",
        "unbound_policy",
        "empty_evidence",
        "wrong_severity",
        "row_identity",
        "non_list",
        "non_object",
    ],
)
def test_replay_rejects_corrupt_shapes_and_cross_artifact_disagreement(corruption):
    context, check, _, artifacts = saved_output(selection=["a", "absent"])
    compatibility, diffs, findings = (artifacts[path] for path in ANALYZE_ARTIFACT_PATHS)
    group = diffs["reference_groups"][0]
    diff = group["diffs"][0]
    finding = findings["findings"][0]
    if corruption == "missing_artifact":
        del artifacts[ANALYZE_ARTIFACT_PATHS[0]]
    elif corruption == "extra_field":
        diffs["unexpected"] = 1
    elif corruption == "reason":
        compatibility["evidence"][0]["reason"]["blocking"] = True
    elif corruption == "duplicate_evidence":
        compatibility["evidence"] *= 2
    elif corruption == "wrong_baseline":
        compatibility["baseline_source_run_id"] = "foreign"
    elif corruption == "missing_group":
        diffs["reference_groups"].pop()
    elif corruption == "foreign_reference":
        group["reference"]["source_run_id"] = "foreign"
    elif corruption == "duplicate_diff":
        group["diffs"] *= 2
    elif corruption == "forged_diff_id":
        diff["diff_id"] = "forged"
    elif corruption == "nan":
        diff["current_value"] = float("nan")
    elif corruption == "boolean":
        diff["delta"] = True
    elif corruption == "wrong_delta":
        diff["delta"] = 20.0
    elif corruption == "missing_metric":
        group["metric_unavailability"] = []
    elif corruption == "unknown_unavailability":
        group["metric_unavailability"][0]["reason"] = "unknown"
    elif corruption == "duplicate_metric":
        group["metric_unavailability"] *= 2
    elif corruption == "duplicate_finding":
        findings["findings"] *= 2
    elif corruption == "forged_finding_id":
        finding["finding_id"] = "forged"
    elif corruption == "unknown_evidence":
        finding["evidence_compatibility_ids"] = ["foreign"]
    elif corruption == "unbound_policy":
        finding["finding_policy_version"] = "foreign"
    elif corruption == "empty_evidence":
        finding["evidence_compatibility_ids"] = []
    elif corruption == "wrong_severity":
        finding["severity"] = "unknown"
    elif corruption == "row_identity":
        diff["source_run_id"] = "current"
    elif corruption == "non_list":
        group["metric_unavailability"] = {}
    else:
        group["diffs"][0] = None
    with pytest.raises(GatewayConsistencyViolation):
        hydrate_analyze_output(artifacts, prepared_context=context, contract_check_result=check)


@pytest.mark.parametrize("mask", range(8))
def test_checked_partial_artifacts_are_validated_without_filling_missing_dependencies(mask):
    context, check, _, artifacts = saved_output()
    partial = {
        path: raw for index, (path, raw) in enumerate(artifacts.items()) if mask & (1 << index)
    }
    validate_partial_analyze_artifacts(
        partial, prepared_context=context, contract_check_result=check
    )
    for path in partial:
        corrupt = deepcopy(partial)
        corrupt[path]["source_run_id"] = "foreign"
        with pytest.raises(GatewayConsistencyViolation):
            validate_partial_analyze_artifacts(
                corrupt, prepared_context=context, contract_check_result=check
            )
