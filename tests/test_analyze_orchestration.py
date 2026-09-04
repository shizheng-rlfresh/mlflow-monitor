"""Write-last Analyze stage commit, interruption recovery, and read-only replay."""

from copy import deepcopy

import pytest
from workflow._analyze_support import MetricsGateway, seed_checked

from mlflow_monitor.analyze_orchestration import commit_analyze_stage
from mlflow_monitor.domain import ComparabilityStatus, LifecycleStatus
from mlflow_monitor.errors import AnalyzeStageError, GatewayConsistencyViolation
from mlflow_monitor.workflow.analyze import execute_analyze
from mlflow_monitor.workflow.analyze_artifacts import (
    ANALYZE_ARTIFACT_PATHS,
    analyze_output_to_artifacts,
)


class CommitGateway(MetricsGateway):
    def __init__(self):
        super().__init__({"current": {"a": 2.0}, "baseline": {"a": 1.0}})
        self.events = []
        self.interrupt = None
        self.corrupt_readback = False

    def write_monitoring_run_json_artifact(self, *, monitoring_run_id, data, path):
        if path not in ANALYZE_ARTIFACT_PATHS:
            return super().write_monitoring_run_json_artifact(
                monitoring_run_id=monitoring_run_id, data=data, path=path
            )
        self.events.append(path)
        super().write_monitoring_run_json_artifact(
            monitoring_run_id=monitoring_run_id, data=data, path=path
        )
        if self.interrupt == path:
            raise RuntimeError("interrupted after artifact write")

    def read_monitoring_run_json_artifact(self, *, monitoring_run_id, path):
        raw = super().read_monitoring_run_json_artifact(
            monitoring_run_id=monitoring_run_id, path=path
        )
        if self.corrupt_readback and path in self.events and raw is not None:
            raw["source_run_id"] = "foreign"
        return raw

    def upsert_monitoring_run(self, *args, **kwargs):
        if kwargs.get("lifecycle_status") is LifecycleStatus.ANALYZED:
            self.events.append("analyzed")
            if self.interrupt == "before_marker":
                raise RuntimeError("interrupted before marker")
        super().upsert_monitoring_run(*args, **kwargs)
        if (
            kwargs.get("lifecycle_status") is LifecycleStatus.ANALYZED
            and self.interrupt == "after_marker"
        ):
            raise RuntimeError("interrupted after marker")

    def finalize_monitoring_run_result(self, **kwargs):
        raise AssertionError("Analyze must not use legacy terminalization")


def record(gateway, state):
    return gateway.get_monitoring_run(state.subject_id, state.monitoring_run_id)


@pytest.mark.parametrize("status", list(ComparabilityStatus))
def test_commit_writes_all_artifacts_before_analyzed_and_preserves_checked_metadata(status):
    gateway = CommitGateway()
    state, context, _ = seed_checked(gateway, status)
    before = record(gateway, state)
    output = commit_analyze_stage(state=state, gateway=gateway)
    after = record(gateway, state)
    assert gateway.events == [*ANALYZE_ARTIFACT_PATHS, "analyzed"]
    assert after.lifecycle_status is LifecycleStatus.ANALYZED
    assert after.contract_check_result == before.contract_check_result
    assert after.references == context.references
    assert len(output.reference_comparison_coverage) == 3
    assert bool(gateway.reads) == (status is not ComparabilityStatus.FAIL)


@pytest.mark.parametrize("interruption", [*ANALYZE_ARTIFACT_PATHS, "before_marker", "after_marker"])
def test_retry_recovers_each_commit_boundary_without_rewriting_saved_artifacts(interruption):
    gateway = CommitGateway()
    state, _, _ = seed_checked(gateway)
    gateway.interrupt = interruption
    with pytest.raises(RuntimeError, match="interrupted"):
        commit_analyze_stage(state=state, gateway=gateway)
    existing_paths = set(gateway.events) & set(ANALYZE_ARTIFACT_PATHS)
    assert record(gateway, state).lifecycle_status is (
        LifecycleStatus.ANALYZED if interruption == "after_marker" else LifecycleStatus.CHECKED
    )
    gateway.interrupt = None
    gateway.events.clear()
    gateway.reads.clear()
    output = commit_analyze_stage(state=state, gateway=gateway)
    assert output.findings
    assert not (set(gateway.events) & existing_paths)
    assert record(gateway, state).lifecycle_status is LifecycleStatus.ANALYZED
    if interruption == "after_marker":
        assert gateway.events == gateway.reads == []


def test_analyzed_replay_ignores_live_metrics_and_never_executes_policies(monkeypatch):
    gateway = CommitGateway()
    state, _, _ = seed_checked(gateway)
    output = commit_analyze_stage(state=state, gateway=gateway)
    gateway.metrics.clear()
    gateway.events.clear()
    gateway.reads.clear()

    def forbidden(**kwargs):
        raise AssertionError("committed replay must not recompute")

    monkeypatch.setattr("mlflow_monitor.analyze_orchestration.execute_analyze", forbidden)
    assert commit_analyze_stage(state=state, gateway=gateway) == output
    assert gateway.events == gateway.reads == []


def test_valid_but_conflicting_partial_is_rejected_before_filling_any_missing_artifact():
    gateway = CommitGateway()
    state, context, check = seed_checked(gateway)
    output = execute_analyze(
        prepared_context=context,
        contract_check_result=check,
        compiled_recipe=state.compiled_recipe,
        gateway=gateway,
    )
    artifacts = analyze_output_to_artifacts(
        output, prepared_context=context, contract_check_result=check
    )
    # Only a later artifact exists. A conflict must not cause earlier writes.
    path = ANALYZE_ARTIFACT_PATHS[2]
    raw = deepcopy(artifacts[path])
    raw["findings"][0]["summary"] = "different but structurally valid"
    gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=state.monitoring_run_id, data=raw, path=path
    )
    gateway.events.clear()
    gateway.reads.clear()
    with pytest.raises(GatewayConsistencyViolation):
        commit_analyze_stage(state=state, gateway=gateway)
    assert gateway.events == []
    assert record(gateway, state).lifecycle_status is LifecycleStatus.CHECKED


def test_invalid_partial_is_rejected_before_source_reads():
    gateway = CommitGateway()
    state, _, _ = seed_checked(gateway)
    gateway.write_monitoring_run_json_artifact(
        monitoring_run_id=state.monitoring_run_id, path=ANALYZE_ARTIFACT_PATHS[1], data={}
    )
    gateway.events.clear()
    with pytest.raises(GatewayConsistencyViolation):
        commit_analyze_stage(state=state, gateway=gateway)
    assert gateway.events == gateway.reads == []


def test_bad_readback_does_not_advance_lifecycle():
    gateway = CommitGateway()
    state, _, _ = seed_checked(gateway)
    gateway.corrupt_readback = True
    with pytest.raises(GatewayConsistencyViolation):
        commit_analyze_stage(state=state, gateway=gateway)
    assert gateway.events == list(ANALYZE_ARTIFACT_PATHS)
    assert record(gateway, state).lifecycle_status is LifecycleStatus.CHECKED


def test_owned_analyze_failure_leaves_checked_without_partial_artifacts_or_terminalization():
    gateway = CommitGateway()
    state, _, _ = seed_checked(gateway)
    gateway.metrics.pop("current")
    with pytest.raises(AnalyzeStageError, match="Source Training Run"):
        commit_analyze_stage(state=state, gateway=gateway)
    assert gateway.events == []
    assert record(gateway, state).lifecycle_status is LifecycleStatus.CHECKED


@pytest.mark.parametrize(
    "status",
    [
        LifecycleStatus.CREATED,
        LifecycleStatus.PREPARED,
        LifecycleStatus.CLOSED,
        LifecycleStatus.FAILED,
    ],
)
def test_internal_analyze_rejects_wrong_stage_without_mutation(status):
    gateway = CommitGateway()
    state, _, _ = seed_checked(gateway)
    gateway.upsert_monitoring_run(
        subject_id=state.subject_id,
        monitoring_run_id=state.monitoring_run_id,
        source_run_id=state.source_run_id,
        sequence_index=state.sequence_index,
        lifecycle_status=status,
    )
    with pytest.raises(GatewayConsistencyViolation):
        commit_analyze_stage(state=state, gateway=gateway)
    assert gateway.events == gateway.reads == []
    assert record(gateway, state).lifecycle_status is status


@pytest.mark.parametrize("boundary", ["before_write", "before_marker", "inside_upsert"])
@pytest.mark.parametrize("status", [LifecycleStatus.CLOSED, LifecycleStatus.FAILED])
def test_concurrent_advancement_is_not_overwritten_by_analyze(monkeypatch, boundary, status):
    gateway = CommitGateway()
    state, _, _ = seed_checked(gateway)
    upsert = gateway.upsert_monitoring_run

    def advance():
        upsert(
            subject_id=state.subject_id,
            monitoring_run_id=state.monitoring_run_id,
            source_run_id=state.source_run_id,
            sequence_index=state.sequence_index,
            lifecycle_status=status,
        )

    if boundary == "before_write":
        execute = execute_analyze

        def execute_and_advance(**kwargs):
            output = execute(**kwargs)
            advance()
            return output

        monkeypatch.setattr(
            "mlflow_monitor.analyze_orchestration.execute_analyze", execute_and_advance
        )
    elif boundary == "before_marker":
        write = gateway.write_monitoring_run_json_artifact

        def write_and_advance(**kwargs):
            write(**kwargs)
            if kwargs["path"] == ANALYZE_ARTIFACT_PATHS[-1]:
                advance()

        monkeypatch.setattr(gateway, "write_monitoring_run_json_artifact", write_and_advance)
    else:

        def advance_and_upsert(**kwargs):
            advance()
            upsert(**kwargs)

        monkeypatch.setattr(gateway, "upsert_monitoring_run", advance_and_upsert)
    with pytest.raises(GatewayConsistencyViolation):
        commit_analyze_stage(state=state, gateway=gateway)
    assert record(gateway, state).lifecycle_status is status
    if boundary == "before_write":
        assert gateway.events == []
