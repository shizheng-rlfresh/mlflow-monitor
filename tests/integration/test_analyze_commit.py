"""Analyze durable commit against real MLflow, without public workflow cutover."""

import pytest
from mlflow import MlflowClient

from mlflow_monitor.analyze_orchestration import commit_analyze_stage
from mlflow_monitor.domain import (
    ComparabilityStatus,
    ContractCheckReason,
    ContractCheckResult,
    DiffReferenceKind,
    LifecycleStatus,
    MonitoringRunReference,
)
from mlflow_monitor.gateway import GatewayConfig, IdempotencyKey, MLflowMonitoringGateway
from mlflow_monitor.orchestration import OrchestrationState
from mlflow_monitor.recipe_compiler import SYSTEM_DEFAULT_COMPILED_RECIPE
from mlflow_monitor.workflow import contract_check_result_to_dict, prepared_context_to_dict
from mlflow_monitor.workflow.analyze_artifacts import ANALYZE_ARTIFACT_PATHS
from mlflow_monitor.workflow.prepared_context import PreparedContext, PreparedReferencePlanEntry


def seed_checked(gateway, sources, status):
    """Persist a checked-but-running fixture without the legacy finalizer."""
    recipe = SYSTEM_DEFAULT_COMPILED_RECIPE
    allocation = gateway.create_or_reuse_monitoring_run(
        IdempotencyKey(
            "model", sources[0], recipe.identity.recipe_id, recipe.identity.recipe_version
        )
    )
    context = PreparedContext(
        monitoring_run_id=allocation.monitoring_run_id,
        source_run_id=sources[0],
        subject_id="model",
        timeline_id=allocation.timeline_id,
        sequence_index=allocation.sequence_index,
        baseline_source_run_id=sources[1],
        effective_recipe=recipe.effective_plan,
        contract=recipe.contract,
        reference_plan=(
            PreparedReferencePlanEntry(
                DiffReferenceKind.BASELINE,
                MonitoringRunReference(DiffReferenceKind.BASELINE, None, sources[1]),
                None,
            ),
            PreparedReferencePlanEntry(
                DiffReferenceKind.PREVIOUS, None, "previous_reference_missing"
            ),
            PreparedReferencePlanEntry(DiffReferenceKind.LKG, None, "lkg_not_selected"),
        ),
    )
    reasons = ()
    if status is not ComparabilityStatus.PASS:
        failed = status is ComparabilityStatus.FAIL
        reasons = (
            ContractCheckReason(
                "schema_mismatch" if failed else "environment_mismatch",
                "Data schema does not match the baseline."
                if failed
                else "Execution environment does not match the baseline.",
                failed,
            ),
        )
    check = ContractCheckResult(status, reasons)
    for path, data in (
        ("outputs/prepared_context.json", prepared_context_to_dict(context)),
        ("outputs/contract_check.json", contract_check_result_to_dict(context, check)),
    ):
        gateway.write_monitoring_run_json_artifact(
            monitoring_run_id=context.monitoring_run_id, data=data, path=path
        )
    gateway.upsert_monitoring_run(
        subject_id=context.subject_id,
        monitoring_run_id=context.monitoring_run_id,
        source_run_id=context.source_run_id,
        sequence_index=context.sequence_index,
        lifecycle_status=LifecycleStatus.CHECKED,
        contract_check_result=check,
        references=context.references,
    )
    return OrchestrationState(
        subject_id=context.subject_id,
        source_run_id=context.source_run_id,
        baseline_source_run_id=context.baseline_source_run_id,
        compiled_recipe=recipe,
        custom_reference_monitoring_run_id=None,
        timeline_id=context.timeline_id,
        monitoring_run_id=context.monitoring_run_id,
        existing_monitoring_run=None,
        is_new_monitoring_run=False,
        sequence_index=context.sequence_index,
    )


@pytest.mark.parametrize("status", list(ComparabilityStatus))
@pytest.mark.parametrize("interrupted", [False, True])
def test_real_mlflow_analyze_commit_recovery_and_replay_preserve_training_runs(
    tracking_uri,
    artifact_root_uri,
    create_training_run,
    snapshot_training_run,
    assert_training_run_unchanged,
    monkeypatch,
    status,
    interrupted,
):
    raw = MlflowClient(tracking_uri=tracking_uri)
    sources = [
        create_training_run(
            raw=raw,
            experiment_name="training/analyze",
            artifact_root_uri=artifact_root_uri,
            run_name=name,
            metrics={"a": value},
            params={"feature_columns": "age"},
            tags={"schema.age": "int"},
            artifact_payload={"training": name},
        )
        for name, value in (("current", 2.0), ("baseline", 1.0))
    ]
    snapshots = {source: snapshot_training_run(raw=raw, run_id=source) for source in sources}
    source_artifacts = {
        source: [(item.path, item.file_size) for item in raw.list_artifacts(source, "outputs")]
        for source in sources
    }
    gateway = MLflowMonitoringGateway(
        GatewayConfig(), tracking_uri=tracking_uri, artifact_location=artifact_root_uri
    )
    state = seed_checked(gateway, sources, status)
    assert raw.get_run(state.monitoring_run_id).info.status == "RUNNING"
    writes = []
    write = gateway.write_monitoring_run_json_artifact

    def record_write(**kwargs):
        write(**kwargs)
        writes.append(kwargs["path"])
        if interrupted and kwargs["path"] == ANALYZE_ARTIFACT_PATHS[1]:
            raise RuntimeError("interrupted real artifact write")

    monkeypatch.setattr(gateway, "write_monitoring_run_json_artifact", record_write)
    if interrupted:
        with pytest.raises(RuntimeError, match="interrupted"):
            commit_analyze_stage(state=state, gateway=gateway)
        assert (
            gateway.get_monitoring_run(state.subject_id, state.monitoring_run_id).lifecycle_status
            is LifecycleStatus.CHECKED
        )
        assert writes == list(ANALYZE_ARTIFACT_PATHS[:2])
        gateway = MLflowMonitoringGateway(
            GatewayConfig(), tracking_uri=tracking_uri, artifact_location=artifact_root_uri
        )
    output = commit_analyze_stage(state=state, gateway=gateway)
    assert (
        gateway.get_monitoring_run(state.subject_id, state.monitoring_run_id).lifecycle_status
        is LifecycleStatus.ANALYZED
    )
    assert raw.get_run(state.monitoring_run_id).info.status == "RUNNING"
    assert (
        gateway.read_monitoring_run_json_artifact(
            monitoring_run_id=state.monitoring_run_id, path="outputs/result.json"
        )
        is None
    )
    assert {item.path for item in raw.list_artifacts(state.monitoring_run_id, "outputs")} == {
        "outputs/prepared_context.json",
        "outputs/contract_check.json",
        *ANALYZE_ARTIFACT_PATHS,
    }
    replay_gateway = MLflowMonitoringGateway(
        GatewayConfig(), tracking_uri=tracking_uri, artifact_location=artifact_root_uri
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not read live metrics, execute policies, or write")

    monkeypatch.setattr(replay_gateway, "get_source_run_metrics", forbidden)
    monkeypatch.setattr(replay_gateway, "write_monitoring_run_json_artifact", forbidden)
    monkeypatch.setattr(replay_gateway, "upsert_monitoring_run", forbidden)
    monkeypatch.setattr("mlflow_monitor.analyze_orchestration.execute_analyze", forbidden)
    assert commit_analyze_stage(state=state, gateway=replay_gateway) == output
    for source in sources:
        assert_training_run_unchanged(raw=raw, run_id=source, snapshot=snapshots[source])
        assert [
            (item.path, item.file_size) for item in raw.list_artifacts(source, "outputs")
        ] == source_artifacts[source]
